import bisect
import io
import logging
import multiprocessing as mp
import pickle
import sys
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import (
    Callable,
    Iterator,
    NamedTuple,
    TypeVar,
    Optional, List,
)
from typing import Iterable

import numpy as np
import torch.utils.data
from gluonts.dataset import DataEntry
from gluonts.dataset.common import DataBatch, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.itertools import IterableSlice, PseudoShuffled, batcher, rows_to_columns, SizedIterable, Cyclic
from gluonts.transform import Identity, Transformation, Valmap, SelectFields, InstanceSampler
from pydantic import BaseModel

T = TypeVar("T")
logger = logging.getLogger(__name__)


class MixedRandomDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset,
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
            multi_modal_fields: List[str] = ["satellite_data"],
            dummy_value: float = 0.0,
            offset=None,
            is_train=True,
            field_names=None,
            transform=None,
            subsample: int = -1,
    ) -> None:
        self.dataset = dataset
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
        self.is_train = is_train
        self.field_names = field_names
        self.multi_modal_fields = multi_modal_fields

        assert past_length > 0, "The value of `past_length` should be > 0"

        self.observed_value_field = observed_value_field
        self.past_ts_fields = past_time_series_fields

        self.offset = offset
        # self.splitter = OffsetSplitter(self.offset)
        self.cumulative_sizes = self.cumsum()
        self.index_map = np.arange(self.cumulative_sizes[-1])
        if 0 < subsample < len(self.index_map):
            self.index_map = np.random.choice(self.index_map, subsample, replace=False)
        self.transform = transform
        self.cache: List[DataEntry] = [None] * len(self.dataset)
        self.slice_cols = (
                self.ts_fields
                + self.past_ts_fields
                + [self.target_field, self.observed_value_field]
        )

    def cumsum(self):
        self.indexes = []
        r, s = [], 0
        for d in self.dataset:
            target_array = d["target"][:self.offset]
            idx = self.instance_sampler(target_array)
            l = len(idx)
            r.append(l + s)
            s += l
            self.indexes.append(idx)
        return r

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        idx = self.index_map[idx]
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if self.cache[dataset_idx] is None:
            self.cache[dataset_idx] = self.transform(self.dataset[dataset_idx])
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        s_idx = self.indexes[dataset_idx][sample_idx]
        return self.get_instance(self.cache[dataset_idx], s_idx)

    def _past(self, col_name):
        return f"past_{col_name}"

    def _future(self, col_name):
        return f"future_{col_name}"

    def get_instance(self, data: DataEntry, i: int):
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


def forecast_start(entry):
    return entry[FieldName.START] + len(entry[FieldName.TARGET])


class Stack(Transformation, BaseModel):
    def __call__(self, data, is_train):
        for batch in data:
            yield rows_to_columns(batch, np.array)


class PermutedDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed

    def __iter__(self):
        for data in self.dataset:
            data = data.copy()
            permutation = np.random.permutation(len(data["target"]))
            data["target"] = data["target"][permutation]
            yield data


class MultiProcessDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self):
        if MPWorkerInfo.worker_id is not None:
            logger.info("Starting MultiProcessDataset : {}".format(MPWorkerInfo.worker_id))
        else:
            logger.info("Starting MultiProcessDataset : single process")
        for index, data in enumerate(self.dataset):
            if MPWorkerInfo.worker_id is not None:
                if index % MPWorkerInfo.num_workers != MPWorkerInfo.worker_id:
                    continue
            yield data


class MPWorkerInfo:
    """
    Contains the current worker information.
    """

    worker_process = False
    num_workers = None
    worker_id = None

    @classmethod
    def set_worker_info(cls, num_workers: int, worker_id: int):
        cls.worker_process = True
        cls.num_workers = num_workers
        cls.worker_id = worker_id
        mp.current_process().name = f"worker_{worker_id}"


class DataLoadingBounds(NamedTuple):
    lower: int
    upper: int


def get_bounds_for_mp_data_loading(dataset_len: int) -> DataLoadingBounds:
    """
    Utility function that returns the bounds for which part of the dataset
    should be loaded in this worker.
    """
    if not MPWorkerInfo.worker_process:
        return DataLoadingBounds(0, dataset_len)

    assert MPWorkerInfo.num_workers is not None
    assert MPWorkerInfo.worker_id is not None

    segment_size = int(dataset_len / MPWorkerInfo.num_workers)
    lower = MPWorkerInfo.worker_id * segment_size
    upper = (
        (MPWorkerInfo.worker_id + 1) * segment_size
        if MPWorkerInfo.worker_id + 1 != MPWorkerInfo.num_workers
        else dataset_len
    )
    return DataLoadingBounds(lower=lower, upper=upper)


DataLoader = Iterable[DataBatch]


def win32_guard(cls, num_workers):
    if num_workers is None:
        return None

    assert num_workers > 0, "num_workers can't be negative"

    if sys.platform == "win32":
        logger.warning(
            "Multiprocessing is not supported on Windows, "
            "num_workers will be set to None."
        )
        return None

    return num_workers


def _encode(value):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(value)
    return buf.getvalue()


def worker_fn(
        worker_id: int,
        dataset,
        num_workers: int,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
):
    MPWorkerInfo.set_worker_info(
        num_workers=num_workers,
        worker_id=worker_id,
    )
    logger.info(f"Worker {worker_id} started")

    while True:
        try:
            input_queue.get()
            for encoded_entry in map(_encode, dataset):
                output_queue.put(encoded_entry)
            output_queue.put(_encode(None))
        except (EOFError, BrokenPipeError):
            return


class MultiProcessLoader(DataLoader):
    def __init__(
            self,
            dataset: Dataset,
            num_workers: int,
            max_queue_size: Optional[int] = None,
            decode_fn: Callable = lambda x: x,
            queue_timeout_seconds: int = 300,
    ):
        assert num_workers >= 1

        if max_queue_size is None:
            max_queue_size = 5 * num_workers
        else:
            assert max_queue_size >= num_workers

        self.decode_fn = decode_fn
        self.queue_timeout_seconds = queue_timeout_seconds
        self.manager = mp.Manager()
        self.output_queue = self.manager.Queue(maxsize=max_queue_size)
        self.input_queues = [self.manager.Queue() for _ in range(num_workers)]
        self.num_workers = num_workers

        self.processes = [
            mp.Process(
                target=worker_fn,
                kwargs={
                    "worker_id": worker_id,
                    "dataset": dataset,
                    "num_workers": num_workers,
                    "input_queue": input_queue,
                    "output_queue": self.output_queue,
                },
            )
            for worker_id, input_queue in enumerate(self.input_queues)
        ]

        for process in self.processes:
            process.start()

    def __iter__(self):
        num_finished = 0
        for input_queue in self.input_queues:
            input_queue.put(_encode(True))
        while num_finished < self.num_workers:
            raw = self.output_queue.get(timeout=self.queue_timeout_seconds)
            data = pickle.loads(raw)
            if data is None:
                num_finished += 1
                continue
            yield self.decode_fn(data)


class Batch(Transformation, BaseModel):
    batch_size: int

    def __call__(self, data, is_train):
        yield from batcher(data, self.batch_size)


# def as_stacked_batches(
#     dataset: Dataset,
#     *,
#     batch_size: int,
#     num_batches_per_epoch: Optional[int] = None,
#     shuffle_buffer_length: Optional[int] = None,
#     output_type: Optional[Callable] = None,
#     field_names: Optional[list] = None,
# ):
#     """
#     Prepare data in batches to be passed to a network.
#
#     Input data is collected into batches of size ``batch_size`` and then
#     columns are stacked on top of each other. In addition, the result is
#     wrapped in ``output_type`` if provided.
#
#     If ``num_batches_per_epoch`` is provided, only those number of batches are
#     effectively returned. This is especially useful for training when
#     providing a cyclic dataset.
#
#     To pseudo shuffle data, ``shuffle_buffer_length`` can be set to collect
#     inputs into a buffer first, from which we then randomly sample.
#
#     Setting ``field_names`` will only consider those columns in the input data
#     and discard all other values.
#     """
#
#     if shuffle_buffer_length:
#         dataset = PseudoShuffled(dataset, shuffle_buffer_length)
#
#     transform: Transformation = Identity()
#
#     if field_names is not None:
#         transform += SelectFields(field_names)
#
#     transform += Batch(batch_size=batch_size)
#     transform += Stack()
#
#     if output_type is not None:
#         transform += Valmap(output_type)
#
#     # Note: is_train needs to be provided but does not have an effect
#     transformed_dataset = transform.apply(dataset, is_train=True)
#     return IterableSlice(transformed_dataset, num_batches_per_epoch)

class MPTransformedDataset(Dataset):
    """
    A dataset that corresponds to applying a list of transformations to each
    element in the base_dataset. This only supports SimpleTransformations,
    which do the same thing at prediction and training time.

    Parameters
    ----------
    base_dataset
        Dataset to transform
    transformations
        List of transformations to apply
    """

    def __init__(
            self,
            base_dataset: Dataset,
            transformation: Transformation,
            is_train=True,
    ) -> None:
        self.base_dataset = base_dataset
        self.transformation = transformation
        self.is_train = is_train

    def __len__(self):
        # NOTE this is unsafe when transformations are run with is_train = True
        # since some transformations may not be deterministic
        # (instance splitter)
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[DataEntry]:
        # source_name = "list_data"
        # Basic idea is to split the dataset into roughly equally sized
        # segments with lower and upper bound, where each worker is assigned
        # one segment
        # bounds = get_bounds_for_mp_data_loading(len(self))
        # for row_number, data in enumerate(self.base_dataset):
        #     if MPWorkerInfo.worker_id is not None:
        #         if row_number % MPWorkerInfo.num_workers != MPWorkerInfo.worker_id:
        #             continue
        #
        #     data = data.copy()
        #     data = self.transformation(data, is_train=self.is_train)
        #     # data["source"] = SourceContext(source=source_name, row=row_number)
        #     yield data
        yield from self.transformation(
            self.base_dataset, is_train=self.is_train
        )


def IndexedTrainDataLoader(
        dataset: torch.utils.data.Dataset,
        *,
        batch_size: int,
        # transform: Transformation = Identity(),
        num_batches_per_epoch: Optional[int] = None,
        prefetch_factor: Optional[int] = None,
        num_workers: int = 0,
        shuffle=True
):
    """
    Construct an iterator of batches for training purposes.
    This function wraps around ``DataLoader`` to offer training-specific
    behaviour and options, as follows:
        1. The provided dataset is iterated cyclically, so that one can go over
        it multiple times in a single epoch. 2. A transformation must be
        provided, that is lazily applied as the dataset is being iterated;
        this is useful e.g. to slice random instances of fixed length out of
        each time series in the dataset. 3. The resulting batches can be
        iterated in a pseudo-shuffled order.
    The returned object is a stateful iterator, whose length is either
    ``num_batches_per_epoch`` (if not ``None``) or infinite (otherwise).
    Parameters
    ----------
    dataset
        Data to iterate over.
    batch_size
        Number of entries to include in a batch.
    num_batches_per_epoch
        Length of the iterator. If ``None``, then the iterator is endless.
    num_workers
        Number of worker processes to use. Default: None.
    prefetch_factor
        Sets the length of the queue of batches being produced by worker
        processes. (Only meaningful when ``num_workers is not None``).
    Returns
    -------
    Iterator[DataBatch]
        An iterator of batches.
    """
    # if shuffle_buffer_length:
    #     dataset = PseudoShuffled(dataset, shuffle_buffer_length)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, prefetch_factor=prefetch_factor,
                        num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    # batches = iter(loader)
    if num_batches_per_epoch is None:
        return loader
    else:
        batches = Cyclic(loader)
        return IterableSlice(batches, num_batches_per_epoch)


def TrainDataLoader(
        dataset: Dataset,
        *,
        batch_size: int,
        transform: Transformation = Identity(),
        num_batches_per_epoch: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        num_workers: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
        output_type: Optional[Callable] = None,
        field_names: Optional[list] = None,
):
    """
    Construct an iterator of batches for training purposes.
    This function wraps around ``DataLoader`` to offer training-specific
    behaviour and options, as follows:
        1. The provided dataset is iterated cyclically, so that one can go over
        it multiple times in a single epoch. 2. A transformation must be
        provided, that is lazily applied as the dataset is being iterated;
        this is useful e.g. to slice random instances of fixed length out of
        each time series in the dataset. 3. The resulting batches can be
        iterated in a pseudo-shuffled order.
    The returned object is a stateful iterator, whose length is either
    ``num_batches_per_epoch`` (if not ``None``) or infinite (otherwise).
    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).
    num_batches_per_epoch
        Length of the iterator. If ``None``, then the iterator is endless.
    num_workers
        Number of worker processes to use. Default: None.
    num_prefetch
        Sets the length of the queue of batches being produced by worker
        processes. (Only meaningful when ``num_workers is not None``).
    shuffle_buffer_length
        Size of the buffer used for shuffling. Default: None, in which case no
        shuffling occurs.
    decode_fn
        A function called on each batch after it's been taken out of the queue.
        (Only meaningful when ``num_workers is not None``).
    Returns
    -------
    Iterator[DataBatch]
        An iterator of batches.
    """
    # if shuffle_buffer_length:
    #     dataset = PseudoShuffled(dataset, shuffle_buffer_length)

    if field_names is not None:
        transform += SelectFields(field_names)

    transform += Batch(batch_size=batch_size)
    transform += Stack()

    if output_type is not None:
        transform += Valmap(output_type)
    # transform += Batch(batch_size=batch_size) + AdhocTransform(stack_fn)
    transformed_dataset = transform.apply(dataset, is_train=True)
    transformed_dataset = MultiProcessDataset(transformed_dataset)
    if shuffle_buffer_length:
        transformed_dataset = PseudoShuffled(transformed_dataset, shuffle_buffer_length)
    # transformed_dataset = MPTransformedDataset(
    #     base_dataset=transformed_dataset.base_dataset,
    #     transformation=transformed_dataset.transformation,
    #     is_train=transformed_dataset.is_train
    # )

    if num_workers is not None:
        loader = MultiProcessLoader(
            transformed_dataset,
            decode_fn=decode_fn,
            num_workers=num_workers,
            max_queue_size=num_prefetch,
        )
        batches = iter(loader)
    else:
        batches = iter(transformed_dataset)

    if num_batches_per_epoch is None:
        return batches
    else:
        return IterableSlice(batches, num_batches_per_epoch)


def ValidationDataLoader(
        dataset: Dataset,
        *,
        transform: Transformation = Identity(),
        batch_size: int,
        num_prefetch: Optional[int] = None,
        num_workers: Optional[int] = None,
        decode_fn: Callable = lambda x: x,
        output_type: Optional[Callable] = None,
        field_names: Optional[list] = None,
):
    """
    Construct an iterator of batches for validation purposes.
    Parameters
    ----------
    dataset
        Data to iterate over.
    transform
        Transformation to be lazily applied as data is being iterated.
        The transformation is applied in "training mode" (``is_train=True``).
    batch_size
        Number of entries to include in a batch.
    stack_fn
        Function to use to stack data entries into batches.
        This can be used to set a specific array type or computing device
        the arrays should end up onto (CPU, GPU).
    num_workers
        Number of worker processes to use. Default: None.
    num_prefetch
        Sets the length of the queue of batches being produced by worker
        processes. (Only meaningful when ``num_workers is not None``).
    decode_fn
        A function called on each batch after it's been taken out of the queue.
        (Only meaningful when ``num_workers is not None``).
    Returns
    -------
    Iterable[DataBatch]
        An iterable sequence of batches.
        :param field_names:
        :param output_type:
    """

    if field_names is not None:
        transform += SelectFields(field_names)

    transform += Batch(batch_size=batch_size)
    transform += Stack()

    if output_type is not None:
        transform += Valmap(output_type)
    transformed_dataset = transform.apply(dataset, is_train=True)

    if num_workers is None:
        return transformed_dataset

    return MultiProcessLoader(
        transformed_dataset,
        decode_fn=decode_fn,
        num_workers=num_workers,
        max_queue_size=num_prefetch,
    )


@dataclass
class MPCyclic:
    """
    Like `itertools.cycle`, but does not store the data.
    """

    iterable: SizedIterable

    def __iter__(self):
        at_least_one = False
        while True:
            for index, el in enumerate(self.iterable):
                if MPWorkerInfo.worker_id is not None:
                    if index % MPWorkerInfo.num_workers != MPWorkerInfo.worker_id:
                        continue
                at_least_one = True
                yield el
            if not at_least_one:
                break

    def stream(self):
        """
        Return a continuous stream of self that has no fixed start.

        When re-iterating ``Cyclic`` it will yield elements from the start of
        the passed ``iterable``. However, this is not always desired; e.g. in
        training we want to treat training data as an infinite stream of
        values and not start at the beginning of the dataset for each epoch.

        >>> from toolz import take
        >>> c = Cyclic([1, 2, 3, 4])
        >>> assert list(take(5, c)) == [1, 2, 3, 4, 1]
        >>> assert list(take(5, c)) == [1, 2, 3, 4, 1]

        >>> s = Cyclic([1, 2, 3, 4]).stream()
        >>> assert list(take(5, s)) == [1, 2, 3, 4, 1]
        >>> assert list(take(5, s)) == [2, 3, 4, 1, 2]
        """
        return iter(self)

    def __len__(self) -> int:
        return len(self.iterable)
