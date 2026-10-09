import logging
import multiprocessing
from functools import partial
from typing import List, Tuple, Type

import numpy as np
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import InstanceSampler, Transformation, MapTransformation, \
    MissingValueImputation, DummyValueImputation
from gluonts.zebras._util import pad_axis
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SlicedDataset(Dataset):

    def __init__(self, entry: DataEntry,
                 indexes: np.ndarray,
                 slice_cols: List[str],
                 past_length: int,
                 future_length: int,
                 target_field: str = FieldName.TARGET,
                 is_pad_field: str = FieldName.IS_PAD,
                 start_field: str = FieldName.START,
                 forecast_start_field: str = FieldName.FORECAST_START,
                 observed_value_field: str = FieldName.OBSERVED_VALUES,
                 lead_time: int = 0,
                 output_NTC: bool = True,
                 time_series_fields: List[str] = [],
                 past_time_series_fields: List[str] = [],
                 dummy_value: float = 0.0,
                 transform=None,
                 is_train=True,
                 field_names=None):
        assert future_length > 0, "The value of `future_length` should be > 0"
        self.entry = entry
        self.indexes = indexes
        self.slice_cols = slice_cols
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value
        self.initialized = False
        self.transform = transform
        self.is_train = is_train
        self.field_names = field_names
        assert past_length > 0, "The value of `past_length` should be > 0"

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields
        self.lazy_initialize()

    def lazy_initialize(self):
        if self.transform is not None:
            entry = list(self.transform([self.entry], self.is_train))
            assert len(entry) == 1
            self.entry = entry[0]
        self.initialized = True

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        if not self.initialized:
            self.lazy_initialize()
        return self.get_instance(self.entry, self.indexes[idx])

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def _split_array(
            self, array: np.ndarray, idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if idx >= self.past_length:
            past_piece = array[..., idx - self.past_length: idx]
        else:
            past_piece = pad_axis(
                array[..., :idx],
                axis=-1,
                left=self.past_length - idx,
                value=self.dummy_value,
            )

        future_start = idx + self.lead_time
        future_slice = slice(future_start, future_start + self.future_length)
        future_piece = array[..., future_slice]

        return past_piece, future_piece

    def _split_instance(self, idx: int) -> DataEntry:
        entry = self.entry
        slice_cols = self.ts_fields + [self.target_field]
        dtype = entry[self.target_field].dtype

        entry = entry.copy()

        for ts_field in slice_cols:
            past_piece, future_piece = self._split_array(entry[ts_field], idx)

            if self.output_NTC:
                past_piece = past_piece.transpose()
                future_piece = future_piece.transpose()

            entry[self._past(ts_field)] = past_piece
            entry[self._future(ts_field)] = future_piece
            del entry[ts_field]

        pad_indicator = np.zeros(self.past_length, dtype=dtype)
        pad_length = max(self.past_length - idx, 0)
        pad_indicator[:pad_length] = 1

        entry[self._past(self.is_pad_field)] = pad_indicator
        entry[self.forecast_start_field] = (
                entry[self.start_field] + idx + self.lead_time
        )

        return entry

    def get_instance(self, data, i):
        pl = self.future_length
        lt = self.lead_time
        pad_length = max(self.past_length - i, 0)
        d = data.copy()
        for field in self.slice_cols:
            if i >= self.past_length:
                past_piece = d[field][..., i - self.past_length: i]
            else:
                pad_block = np.full(
                    shape=d[field].shape[:-1] + (pad_length,),
                    fill_value=self.dummy_value,
                    dtype=d[field].dtype,
                )
                past_piece = np.concatenate(
                    [pad_block, d[field][..., :i]], axis=-1
                )
            future_piece = d[field][..., (i + lt): (i + lt + pl)]
            if field in self.ts_fields:
                piece = np.concatenate([past_piece, future_piece], axis=-1)
                if self.output_NTC:
                    piece = piece.transpose()
                d[field] = piece
            else:
                if self.output_NTC:
                    past_piece = past_piece.transpose()
                    future_piece = future_piece.transpose()
                if field not in self.past_ts_fields:
                    d[self._past(field)] = past_piece
                    d[self._future(field)] = future_piece
                    del d[field]
                else:
                    d[field] = past_piece
        pad_indicator = np.zeros(self.past_length)
        if pad_length > 0:
            pad_indicator[:pad_length] = 1
        d[self._past(self.is_pad_field)] = pad_indicator
        d[self.forecast_start_field] = d[self.start_field] + i + lt
        if self.field_names is not None:
            d = self.select_field_names(d)
        return d

    def select_field_names(self, data):
        return {f: data[f] for f in self.field_names}


class MixedRandomInstanceSplitter:

    def __init__(
            self,
            instance_sampler: InstanceSampler,
            past_length: int,
            future_length: int,
            target_field: str = FieldName.TARGET,
            is_pad_field: str = FieldName.IS_PAD,
            start_field: str = FieldName.START,
            forecast_start_field: str = FieldName.FORECAST_START,
            observed_value_field: str = FieldName.OBSERVED_VALUES,
            lead_time: int = 0,
            output_NTC: bool = True,
            time_series_fields: List[str] = [],
            past_time_series_fields: List[str] = [],
            dummy_value: float = 0.0,
    ) -> None:
        assert future_length > 0, "The value of `future_length` should be > 0"
        self.instance_sampler = instance_sampler
        self.past_length = past_length
        self.future_length = future_length
        self.lead_time = lead_time
        self.output_NTC = output_NTC
        self.ts_fields = time_series_fields
        self.target_field = target_field
        self.is_pad_field = is_pad_field
        self.start_field = start_field
        self.forecast_start_field = forecast_start_field
        self.dummy_value = dummy_value

        assert past_length > 0, "The value of `past_length` should be > 0"

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields

    def initialize_sliced_dataset(self, data_entry: DataEntry, transform: Transformation, is_train: bool,
                                  field_names: List[str] = None):
        target = data_entry[self.target_field]
        sampled_indices = self.instance_sampler(target)
        slice_cols = (
                self.ts_fields
                + self.past_ts_fields
                + [self.target_field, self.observed_value_field]
        )
        s_dataset = SlicedDataset(data_entry, sampled_indices, slice_cols, self.past_length, self.future_length,
                                  self.target_field, self.is_pad_field, self.start_field, self.forecast_start_field,
                                  self.observed_value_field, self.lead_time, self.output_NTC, self.ts_fields,
                                  self.past_ts_fields, self.dummy_value, transform=transform,
                                  is_train=is_train, field_names=field_names)
        return s_dataset

    def __call__(
            self, data_it: List[DataEntry], transform: Transformation, is_train: bool,
            field_names: List[str] = None) -> Dataset:
        with multiprocessing.Pool(processes=16) as pool:
            dataset_list = list(
                tqdm(pool.imap(partial(self.initialize_sliced_dataset, transform=transform, is_train=is_train,
                                       field_names=field_names), data_it), total=len(data_it)))
        return ConcatDataset(dataset_list)


class AddItemIDFeature(MapTransformation):
    """
    Expands a `const` value along the time axis as a dynamic feature, where the
    T-dimension is defined as the sum of the `pred_length` parameter and the
    length of a time series specified by the `target_field`.

    If `is_train=True` the feature matrix has the same length as the `target`
    field. If `is_train=False` the feature matrix has length
    `len(target) + pred_length`.

    Parameters
    ----------
    output_field
        Field name for output.
    target_field
        Field containing the target array. The length of this array will be
        used.
    pred_length
        Prediction length (this is necessary since features have to be
        available in the future)
    const
        Constant value to use.
    dtype
        Numpy dtype to use for resulting array.
    """

    @validated()
    def __init__(
            self,
            output_field: str,
            item_map: dict = None,
            dtype: Type = np.int64,
    ) -> None:
        self.dtype = dtype
        self.output_field = output_field
        self.item_map = item_map

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if self.item_map is None or data[FieldName.ITEM_ID] not in self.item_map:
            data[self.output_field] = [0]
        else:
            data[self.output_field] = [self.item_map[data[FieldName.ITEM_ID]]]
        return data


class LastValueImputation2D(MissingValueImputation):
    """
    This class replaces each missing value with the last value that was not
    missing.

    (If the first values are missing, they are replaced by the closest non
    missing value.)
    """

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = values[np.arange(idx.shape[0])[:, None], idx]

        values = out
        mask = np.isnan(values)
        values[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), values[~mask]
        )

        return values
