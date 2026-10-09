import functools
import json
import logging
import math
import os
from collections import defaultdict
from enum import Enum
from functools import cached_property
from functools import partial
from pathlib import Path
from typing import Iterable, Iterator, Union

import datasets
import numpy as np
import pyarrow.compute as pc
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
from toolz import compose
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
MAX_WINDOW = 200

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

M5_PRED_LENGTH_MAP = {
    "W": 4,
    "D": 28,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
}


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


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def uni_to_multi(data_entry: DataEntry, field="past_feat_dynamic_real") -> DataEntry:
    if field in data_entry:
        val_ls = data_entry[field]
        if val_ls.ndim == 1:
            val_ls = np.expand_dims(val_ls, axis=0)
            data_entry[field] = val_ls
    return data_entry


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
            self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                data_entry[self.field] = val
                data_entry["item_id"] = item_id + "_dim" + str(id)
                yield data_entry


class UnivariateToMultivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(
            self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            val_ls = data_entry[self.field]
            if val_ls.ndim == 1:
                val_ls = np.expand_dims(val_ls, axis=0)
                data_entry[self.field] = val_ls
            yield data_entry


def map_element(x, mapping_dict):
    return mapping_dict[str(x)]


def convert_string_features(example, map_dict):
    for k, maps in map_dict.items():
        features = example[k]
        example[k] = []
        for m, f in zip(maps, features):
            if isinstance(f, np.ndarray):
                vectorized_map = np.vectorize(partial(map_element, mapping_dict=m))
                example[k].append(vectorized_map(f))
            else:
                example[k].append(map_element(f, m))
    return example


class OffsetDataset(Dataset):
    def __init__(self, dataset, offset, transform):
        self.dataset = dataset
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):
        return self.dataset[idx + self.offset]

    def __len__(self):
        return len(self.dataset) - self.offset


class Dataset:
    def __init__(
            self,
            name: str,
            term: Union[Term, str] = Term.SHORT,
            storage_env_var: str = "DATA_PATH",
            remap=False,
            indexed_sample=False
    ):
        load_dotenv()
        self.indexed_sample = indexed_sample
        storage_path = Path(os.getenv(storage_env_var))
        directory = str(storage_path / name)
        self.hf_dataset = datasets.load_from_disk(directory).with_format(
            "numpy"
        )
        map_dict, self.dim_params = self.construct_map(directory, remap)
        self.item_map = self.construct_item_map()

        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )
        self.hf_dataset = self.hf_dataset.map(functools.partial(convert_string_features, map_dict=map_dict),
                                              num_proc=8)

        self.base_transform = compose(process, itemize_start, uni_to_multi)
        self.gluonts_dataset = Map(self.base_transform, self.hf_dataset)


        self.term = Term(term)
        self.name = name

    def to_univariate(self):
        self.gluonts_dataset = MultivariateToUnivariate("target").apply(
            self.gluonts_dataset
        )

    @cached_property
    def prediction_length(self) -> int:
        if "retail" in self.name:
            if self.term == Term.MEDIUM:
                return 16
            elif self.term == Term.LONG:
                return 32
            elif self.term == Term.SHORT_2:
                return 16
            elif self.term == Term.SHORT_3:
                return 8 * 3
            else:
                return 8
        freq = norm_freq_str(to_offset(self.freq).name).upper()
        if "m5" in self.name:
            pred_len = M5_PRED_LENGTH_MAP[freq]
        elif "m4" in self.name:
            pred_len = M4_PRED_LENGTH_MAP[freq]
        elif "retail" in self.name:
            pred_len = 8
        else:
            pred_len = PRED_LENGTH_MAP[freq]
        return self.term.multiplier * pred_len
        # return 256

    @cached_property
    def freq(self) -> str:
        if "freq" in self.hf_dataset[0]:
            return self.hf_dataset[0]["freq"]
        else:
            return "D"

    @cached_property
    def target_dim(self) -> int:
        return (
            target.shape[0]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif (
                len(
                    (
                            past_feat_dynamic_real := self.hf_dataset[0][
                                "past_feat_dynamic_real"
                            ]
                    ).shape
                )
                > 1
        ):
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def split_point(self) -> int:
        if "m4" in self.name or "m5" in self.name or "retail" in self.name:
            return self.prediction_length
        w = math.ceil(TEST_SPLIT * self._min_series_length)
        return max(1, w)

    @cached_property
    def num_series(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            return len(self.hf_dataset.data.column("target")[0])
        else:
            return len(self.hf_dataset.data.column("target"))

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(
                    pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)
                )
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(self.hf_dataset.data.column("target"))
            )
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> Dataset:
        split_point = -self.split_point * 2 + self.prediction_length - 1
        if self.indexed_sample:
            logger.info("####### Using indexed sample ########")
            return OffsetDataset(self.hf_dataset, split_point, self.base_transform)
        else:
            logger.info("####### Using iterator sample ########")
            train_dataset, _ = split(
                self.gluonts_dataset, offset=split_point
            )
            return train_dataset

    @property
    def validation_dataset(self) -> Dataset:
        split_point = -self.split_point
        if self.indexed_sample:
            logger.info("####### Using indexed sample ########")
            return OffsetDataset(self.hf_dataset, split_point, self.base_transform)
        else:
            logger.info("####### Using iterator sample ########")
            validation_dataset, _ = split(
                self.gluonts_dataset, offset=split_point
            )
            return validation_dataset


    @property
    def test_dataset(self) -> TrainingDataset:
        dataset = self.gluonts_dataset
        offset = -self.split_point
        return TrainingDataset(dataset=dataset, splitter=OffsetSplitter(offset))

    @property
    def test_data(self) -> TestData:
        _, test_template = split(
            self.gluonts_dataset, offset=-self.split_point
        )
        logger.info(f"Test prediction length: {self.prediction_length}")
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.split_point - self.prediction_length + 1,
            distance=1,
        )
        logger.info(f"Test data length: {len(test_data)}")
        return test_data

    def construct_item_map(self):
        dataset = self.hf_dataset
        if FieldName.ITEM_ID in dataset.features:
            cat_features = dataset[FieldName.ITEM_ID]
            print(f"convert {cat_features.dtype} type to int for {FieldName.FEAT_STATIC_CAT}")
            unique_features = np.unique(cat_features)
            if cat_features.dtype == np.int64:
                cat_features_map = {int(k): v for v, k in enumerate(unique_features)}
            else:
                cat_features_map = {k: v for v, k in enumerate(unique_features)}
            return cat_features_map
        return None

    def construct_map(self, directory, remap=False):
        dataset = self.hf_dataset
        map_file = os.path.join(directory, "map.json")
        dim_params_file = os.path.join(directory, "dim_params.json")
        if (not remap) and os.path.exists(map_file) and os.path.exists(dim_params_file):
            logger.info(f"loading map from {map_file}")
            map_dict = json.load(open(map_file, "r"))
            logger.info(f"loading dim_params from {dim_params_file}")
            dimension_params = json.load(open(dim_params_file, "r"))
            return map_dict, dimension_params
        logger.info(f"constructing map and save to {directory}")
        map_dict = {}
        dimension_params = defaultdict(list)
        if FieldName.FEAT_DYNAMIC_REAL in dataset.features:
            real_feature: np.ndarray = dataset[FieldName.FEAT_DYNAMIC_REAL][0]
            dimension_params["dynamic_dims"] = [1 if real_feature.ndim == 1 else real_feature.shape[0]]
        if FieldName.FEAT_STATIC_REAL in dataset.features:
            real_feature = dataset[FieldName.FEAT_STATIC_REAL][0]
            dimension_params["static_dims"] = [real_feature.shape[0]]
        if FieldName.PAST_FEAT_DYNAMIC_REAL in dataset.features:
            real_feature = dataset[FieldName.PAST_FEAT_DYNAMIC_REAL][0]
            dimension_params["past_dynamic_dims"] = [1 if real_feature.ndim == 1 else real_feature.shape[0]]

        if FieldName.FEAT_STATIC_CAT in dataset.features:
            cat_features = dataset[FieldName.FEAT_STATIC_CAT]
            print(f"convert {cat_features.dtype} type to int for {FieldName.FEAT_STATIC_CAT}")
            cat_features_maps = []
            for i in range(cat_features.shape[1]):
                unique_features = np.unique(cat_features[:, i])
                dimension_params["static_cardinalities"].append(len(unique_features))
                if cat_features.dtype == np.int64:
                    cat_features_map = {int(k): v for v, k in enumerate(unique_features)}
                else:
                    cat_features_map = {k: v for v, k in enumerate(unique_features)}
                cat_features_maps.append(cat_features_map)
            map_dict[FieldName.FEAT_STATIC_CAT] = cat_features_maps
        if FieldName.FEAT_DYNAMIC_CAT in dataset.features:
            cat_features = dataset[FieldName.FEAT_DYNAMIC_CAT]
            print(f"convert {cat_features.dtype} type to int for {FieldName.FEAT_DYNAMIC_CAT}")
            cat_features_maps = []
            for i in range(cat_features[0].shape[0]):
                unique_features = np.unique(np.concatenate([cat_f[i] for cat_f in cat_features]))
                dimension_params["dynamic_cardinalities"].append(len(unique_features))
                if unique_features.dtype == np.int64:
                    cat_features_map = {int(k): v for v, k in enumerate(unique_features)}
                else:
                    cat_features_map = {k: v for v, k in enumerate(unique_features)}
                cat_features_maps.append(cat_features_map)
            map_dict[FieldName.FEAT_DYNAMIC_CAT] = cat_features_maps
        if FieldName.PAST_FEAT_DYNAMIC_CAT in dataset.features:
            cat_features = dataset[FieldName.PAST_FEAT_DYNAMIC_CAT]
            print(f"convert {cat_features.dtype} type to int for {FieldName.PAST_FEAT_DYNAMIC_CAT}")
            cat_features_maps = []
            for i in range(cat_features[0].shape[0]):
                unique_features = np.unique(np.concatenate([cat_f[i] for cat_f in cat_features]))
                dimension_params["past_dynamic_cardinalities"].append(len(unique_features))
                if unique_features.dtype == np.int64:
                    cat_features_map = {int(k): v for v, k in enumerate(unique_features)}
                else:
                    cat_features_map = {k: v for v, k in enumerate(unique_features)}
                cat_features_maps.append(cat_features_map)
            map_dict[FieldName.PAST_FEAT_DYNAMIC_CAT] = cat_features_maps
        with open(map_file, "w") as f:
            json.dump(map_dict, f)
            logger.info(f"map saved to {map_file}")
        map_dict = json.load(open(map_file, "r"))
        with open(dim_params_file, "w") as f:
            json.dump(dimension_params, f)
            logger.info(f"dimension params saved to {dim_params_file}")
        dimension_params = json.load(open(dim_params_file, "r"))
        return map_dict, dimension_params
