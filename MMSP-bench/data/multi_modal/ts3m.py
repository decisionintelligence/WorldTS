import logging
import os
from enum import Enum
from typing import Dict, List, Tuple, Union, Iterator

import gluonts.dataset
import numpy as np
import pandas as pd
from gluonts.transform import Identity
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat
from functools import cached_property, partial
from gluonts.dataset.field_names import FieldName
from gluonts.dataset import DataEntry
import math

from data.split import TestData, split

TEST_SPLIT = 0.1

logger = logging.getLogger(__name__)


class Term(Enum):
    SHORT = "short"
    SHORT_2 = "2short"
    SHORT_3 = "3short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.SHORT_2:
            return 2
        elif self == Term.SHORT_3:
            return 3
        elif self == Term.MEDIUM:
            return 5
        elif self == Term.LONG:
            return 10


def get_data_spower(data_dir, solar_power_file, num_sites=10, num_ignored_sites=0):
    data_df = pd.read_csv(os.path.join(data_dir, solar_power_file),
                          parse_dates=['datetime'])
    data_df = data_df.fillna(0)
    data_sp = []
    for i, (k, v) in enumerate(data_df.groupby('site')):
        if i >= num_sites:
            break
        if i < num_ignored_sites:
            continue
        print('dataset k,v len', k, len(v))
        lats_lons = torch.tensor([v['lat'].values[0], v['lon'].values[0]])
        values = torch.from_numpy(v['power'].values)
        if values.isnan().any():
            print('dataset contains nan len', k, len(v))
        data_sp += [{'site': k, 'values': values, 'lats_lons': lats_lons}]
    length = len(v)
    month = v['datetime'].apply(lambda x: x.month)
    day = v['datetime'].apply(lambda x: x.day)
    hour = v['datetime'].apply(lambda x: x.hour)
    time = torch.from_numpy(np.stack([month, day, hour], axis=0))
    time_dt = v['datetime']
    return data_sp, time, time_dt, length


def get_data_satellite(data_dir, satellite_dir, norm_stl=True, num_feature=4, encoder=None):
    print('load satellite from: [{}]'.format(os.path.join(data_dir, satellite_dir)))
    array_satellite = np.load(os.path.join(data_dir, satellite_dir, 'satellite.npy'))
    T, H, W, C = array_satellite.shape
    scaler = None
    if encoder is not None:
        features = []

        batch_size = 16
        with torch.no_grad():
            for i in range(0, T, batch_size):
                batch = array_satellite[i:min(i + batch_size, T)]

                if batch.max() <= 1.0:
                    batch = (batch * 255).astype(np.uint8)
                else:
                    batch = batch.astype(np.uint8)

                if C == 1:
                    batch = np.repeat(batch, 3, axis=-1)

                inputs = encoder.preprocessor(images=batch, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                outputs = encoder.model(**inputs)
                batch_features = outputs.last_hidden_state.mean(dim=1)

                if torch.cuda.is_available():
                    batch_features = batch_features.cpu()

                features.append(batch_features.numpy())

        features = np.concatenate(features, axis=0)

        data_satellite = torch.from_numpy(features)

    else:
        if norm_stl:
            scaler = StandardScaler()
            array_satellite = scaler.fit_transform(array_satellite.reshape(T, -1)).reshape(T, H, W, C)
        data_satellite = torch.from_numpy(array_satellite)
    data_satellite_coords = torch.from_numpy(
        np.load(os.path.join(data_dir, satellite_dir, 'satellite_coords.npy')))
    data_satellite_times = np.load(os.path.join(data_dir, satellite_dir, 'satellite_times.npy'))
    return data_satellite[..., :num_feature], data_satellite_times, data_satellite_coords, scaler


def get_data_nwp(data_dir, nwp_file, norm_nwp=True):
    df = pd.read_csv(os.path.join(data_dir, nwp_file),
                     parse_dates=['fcst_date']).interpolate()
    # get nwp start time
    nwp_start_time = df['fcst_date'].iloc[0]

    # process nwp dataframe
    df['lat'] = np.round(df['lat'], 1)
    df['lon'] = np.round(df['lon'], 1)
    df = df.drop(columns=['fcst_date'])
    # normalize nwp dataframe
    columns = df.columns.drop(['lat', 'lon'])
    if norm_nwp:
        scaler = StandardScaler()
        df[columns] = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
    data_nwp_grouped = df.groupby(['lat', 'lon'])
    return data_nwp_grouped, nwp_start_time, len(columns) + 2


class ImageMultimodalDataset(Dataset, gluonts.dataset.Dataset):

    def __init__(
            self,
            name: str,
            storage_env_var: str = "DATA_PATH",
            indexed_sample: bool = False,
            term: Union[Term, str] = Term.SHORT,
            remap=False,
            satellite_dir: str = 'satellite',
            num_sites: int = 10,
            num_ignored_sites: int = 0,
            wo_nwp: bool = False,
            norm_nwp: bool = True,
            norm_stl: bool = True,
            pretrain: bool = True,
            encoder_path=None,
    ) -> None:
        self.base_transform = partial(Identity(), is_train=False)
        from dotenv import load_dotenv
        load_dotenv()

        self.indexed_sample = indexed_sample
        self.name = name
        self.item_map = None
        self.wo_nwp = wo_nwp

        self.num_sites = num_sites
        self.num_ignored_sites = num_ignored_sites

        storage_path = os.getenv(storage_env_var)
        self.data_dir = storage_path + "/" + name if storage_path else name

        self.data_sp, self.data_sp_time, self.data_sp_time_dt, self.data_sp_length = (
            get_data_spower(data_dir=self.data_dir,
                            solar_power_file='solar_power/solar_power.csv',
                            num_sites=num_sites,
                            num_ignored_sites=num_ignored_sites))

        encoder = None
        if pretrain and encoder_path:
            try:
                class FeatureExtractor:
                    def __init__(self, model_name):
                        from transformers import AutoImageProcessor, AutoModelForImageClassification
                        self.preprocessor = AutoImageProcessor.from_pretrained(model_name)
                        self.model = AutoModelForImageClassification.from_pretrained(model_name)

                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")
                        self.model.eval()

                encoder = FeatureExtractor(encoder_path)
            except Exception as e:
                encoder = None

        self.data_stl, self.data_stl_times, self.data_stl_coords, self.scaler = \
            get_data_satellite(data_dir=self.data_dir, satellite_dir=satellite_dir, norm_stl=norm_stl, encoder=encoder)

        self.data_ec_grouped, self.ec_start_time, self.ec_dim = (
            get_data_nwp(data_dir=self.data_dir, nwp_file='nwp/nwp.csv', norm_nwp=norm_nwp))

        self.dim_params = {
            "dynamic_dims": [] if self.wo_nwp else [1] * self.ec_dim,
            "static_dims": [2],
        }
        self.term = term

    def __iter__(self) -> Iterator[DataEntry]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return self.num_sites

    def __getitem__(self, idx: int) -> DataEntry:
        # site_id = idx // self.data_sp_length
        site_id = idx
        # time_idx = idx % self.data_sp_length

        ts_values = self.data_sp[site_id]['values']
        ts_coords = self.data_sp[site_id]['lats_lons']

        lat = np.round(float(ts_coords[0]), 1)
        lon = np.round(float(ts_coords[1]), 1)

        stl_begin_index = (self.data_sp_time_dt.iloc[0] - pd.to_datetime(self.data_stl_times[0]))
        stl_begin_index = int(stl_begin_index.total_seconds() // 3600)

        ec_begin_index = (self.data_sp_time_dt.iloc[0] - self.ec_start_time)
        ec_begin_index = int(ec_begin_index.total_seconds() // 3600)
        satellite_data = self.data_stl[stl_begin_index:stl_begin_index + self.data_sp_length]
        T, H, W, C = satellite_data.shape
        ts_time = repeat(self.data_sp_time, 'c t -> h w c t', h=H, w=W)
        satellite_data = rearrange(satellite_data, 't h w c -> h w c t')
        satellite_coords = self.data_stl_coords

        ec_data = self.data_ec_grouped.get_group((lat, lon)).values
        ec_data = ec_data[ec_begin_index:ec_begin_index + self.data_sp_length]

        data_entry = {
            FieldName.TARGET: ts_values.numpy().astype(np.float32),
            FieldName.START: pd.Timestamp(self.data_sp_time_dt.iloc[0]).to_period("h"),
            FieldName.FEAT_STATIC_CAT: np.array([site_id], dtype=np.float32),
            FieldName.FEAT_STATIC_REAL: ts_coords.numpy().astype(np.float32),
            FieldName.ITEM_ID: f"site_{site_id}",
            "satellite_data": satellite_data.numpy().astype(np.float32),
            "satellite_coords": satellite_coords.numpy().astype(np.float32),
            "time_coords": ts_time.numpy().astype(np.float32),
            "freq": "h",
        }
        if not self.wo_nwp:
            data_entry[FieldName.FEAT_DYNAMIC_REAL] = ec_data.T.astype(np.float32)

        return data_entry

    @property
    def freq(self) -> str:
        return "H"

    @cached_property
    def split_point(self) -> int:

        w = math.ceil(TEST_SPLIT * self._min_series_length)
        return max(1, w)

    @cached_property
    def _min_series_length(self) -> int:
        return self.data_sp_length

    @cached_property
    def sum_series_length(self) -> int:
        return self.data_sp_length * self.num_sites

    @property
    def prediction_length(self) -> int:

        return 24

    @property
    def target_dim(self) -> int:
        return 1

    @property
    def past_feat_dynamic_real_dim(self) -> int:

        return 0

    @property
    def training_dataset(self) -> Dataset:
        split_point = -self.split_point * 2 + self.prediction_length - 1
        if self.indexed_sample:
            logger.info("####### Using indexed sample ########")
            return OffsetDataset(self, split_point, self.base_transform)
        else:
            logger.info("####### Using iterator sample ########")
            train_dataset, _ = split(
                self, offset=split_point
            )
            return train_dataset

    @property
    def validation_dataset(self) -> Dataset:
        split_point = -self.split_point
        if self.indexed_sample:
            logger.info("####### Using indexed sample ########")
            return OffsetDataset(self, split_point, self.base_transform)
        else:
            logger.info("####### Using iterator sample ########")
            validation_dataset, _ = split(
                self, offset=split_point
            )
            return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self, offset=-self.split_point
        )
        logger.info(f"Test prediction length: {self.prediction_length}")
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.split_point - self.prediction_length + 1,
            distance=1,
        )
        logger.info(f"Test data length: {len(test_data)}")
        return test_data

    @cached_property
    def num_series(self) -> int:
        return self.num_sites


class OffsetDataset(Dataset):
    def __init__(self, dataset, offset, transform):
        self.dataset = dataset
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):
        return self.dataset[idx + self.offset]

    def __len__(self):
        return len(self.dataset) - self.offset
