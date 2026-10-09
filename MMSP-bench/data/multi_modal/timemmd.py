import logging
import os
from functools import cached_property, partial
from typing import Iterator

import gluonts.dataset
import math
import torch
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import Identity
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from data.split import TestData, split


from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

TEST_SPLIT = 0.2
logger = logging.getLogger(__name__)
class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


class TextMultiModalDataset(Dataset, gluonts.dataset.Dataset):

    def __init__(
            self,
            name: str,
            storage_env_var: str = "DATA_PATH",
            indexed_sample: bool = False,
            term: str = "short",
            remap=False,
            features='S',
            target='OT',
            scale=True,
            timeenc=0,
            freq='h',
            text_len=2,
            tokenizer_name="bert-base-chinese",
            max_length=128
    ) -> None:
        self.base_transform = partial(Identity(), is_train=False)
        from dotenv import load_dotenv
        load_dotenv()

        self.name = name
        self.term = term
        self.indexed_sample = indexed_sample
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.text_len = text_len
        self.item_map = None
        storage_path = os.getenv(storage_env_var)
        self.data_dir = storage_path + name if storage_path else "./data/" + name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        self.__read_data__()

        self.dim_params = {
            "dynamic_dims": [],
            "static_dims": [],
        }

    def encode_with_bert(self, cleaned_texts, batch_size=16):
        from transformers import BertModel, BertTokenizer
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        model = BertModel.from_pretrained(self.tokenizer_name).to(device)
        model.eval()

        text_embeddings = []
        for i in tqdm(range(0, len(cleaned_texts), batch_size)):
            batch_texts = cleaned_texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=self.max_length,
                               padding='max_length', truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            text_embeddings.extend(cls_embeddings)

        return np.array(text_embeddings)

    def encode_with_gpt(self, cleaned_texts, batch_size=16):
        from transformers import GPT2Model, GPT2Tokenizer
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained(self.tokenizer_name).to(device)
        model.eval()

        text_embeddings = []
        for i in tqdm(range(0, len(cleaned_texts), batch_size)):
            batch_texts = cleaned_texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=self.max_length,
                               padding='max_length', truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden = outputs.last_hidden_state
            attention_mask = inputs.get('attention_mask',
                                        torch.ones(last_hidden.shape[0], last_hidden.shape[1], device=device))
            sum_embeddings = torch.sum(last_hidden * attention_mask.unsqueeze(-1), dim=1)
            count = torch.sum(attention_mask, dim=1, keepdim=True)
            mean_embeddings = (sum_embeddings / (count + 1)).cpu().numpy()
            text_embeddings.extend(mean_embeddings)

        return np.array(text_embeddings)

    def encode_with_llama(self, cleaned_texts, batch_size=8):
        from transformers import LlamaModel, LlamaTokenizer
        import torch

        tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaModel.from_pretrained(self.tokenizer_name)
        model.eval()

        text_embeddings = []
        for i in tqdm(range(0, len(cleaned_texts), batch_size)):
            batch_texts = cleaned_texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", max_length=self.max_length,
                               padding='max_length', truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            masked_embedding = hidden_states * attention_mask.unsqueeze(-1)
            sum_embeddings = torch.sum(masked_embedding, dim=1)
            seq_len = torch.sum(attention_mask, dim=1, keepdim=True)
            mean_embeddings = (sum_embeddings / (seq_len + 1)).cpu().numpy()
            text_embeddings.extend(mean_embeddings)

        return np.array(text_embeddings)

    def __read_data__(self):
        text_name = f'Final_Search_{self.text_len}'

        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise FileNotFoundError(f"csv file not found in {self.data_dir}")
        data_path = os.path.join(self.data_dir, csv_files[0])
        df_raw = pd.read_csv(data_path)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')

        req_columns = ['date'] + cols + [self.target] + ['prior_history_avg'] + ['start_date'] + ['end_date'] + [
            text_name]
        df_raw = df_raw[req_columns]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values.reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.values[:, 1:]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data = data.squeeze()
        self.data_prior = data_prior
        self.data_stamp = data_stamp
        self.data_length = len(self.data)

        self.dates = pd.to_datetime(df_raw['date'].values)
        raw_text_data = df_raw[text_name].values[:, 0]
        with torch.no_grad():
            cleaned_texts = [text if not pd.isna(text) else "" for text in raw_text_data]
            model_type = self.tokenizer_name

            if 'bert' in model_type:
                logger.info(f"BERT tokenizer {model_type}")
                text_data = self.encode_with_bert(cleaned_texts)
            elif 'gpt' in model_type:
                logger.info(f"GPT tokenizer {model_type}")
                text_data = self.encode_with_gpt(cleaned_texts)
            elif 'llama' in model_type:
                logger.info(f"Llama tokenizer {model_type}")
                text_data = self.encode_with_llama(cleaned_texts)
            else:
                model = SentenceTransformer(self.tokenizer_name)
                text_data = model.encode(cleaned_texts,
                                         show_progress_bar=False,
                                         convert_to_numpy=True)
            self.text_data = text_data


    def __iter__(self) -> Iterator[DataEntry]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> DataEntry:
        target = self.data

        text_feat = self.text_data

        data_entry = {
            FieldName.TARGET: target.astype(np.float32),
            FieldName.START: pd.Timestamp(self.dates[idx]).to_period('h'),
            FieldName.ITEM_ID: f"item_{idx}",
            "text_data": text_feat.astype(np.float32).T,
            "freq": self.freq
        }

        return data_entry

    def to_univariate(self):
        if self.data.shape[1] == 1:
            return
        self.data = self.data[:, :1]

    @property
    def prediction_length(self) -> int:
        return 12

    @cached_property
    def split_point(self) -> int:
        w = math.ceil(TEST_SPLIT * self._min_series_length)
        return max(1, w)

    @cached_property
    def _min_series_length(self) -> int:
        return self.data_length

    @cached_property
    def sum_series_length(self) -> int:
        return self.data_length

    @property
    def target_dim(self) -> int:
        return 1

    @property
    def past_feat_dynamic_real_dim(self) -> int:
        return len(self.data_stamp[0]) if hasattr(self, 'data_stamp') else 0

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


class OffsetDataset(Dataset):
    def __init__(self, dataset, offset, transform):
        self.dataset = dataset
        self.offset = offset
        self.transform = transform

    def __getitem__(self, idx):
        return self.dataset[idx + self.offset]

    def __len__(self):
        return len(self.dataset) - self.offset
