from data.unimodal.data import *


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing_extensions import Protocol, runtime_checkable

from gluonts.itertools import roundrobin

# Dictionary used for data flowing through the transformations.
DataEntry = Dict[str, Any]
DataBatch = Dict[str, Any]


@runtime_checkable
class IndexedDataset(Protocol):

    def __getitem__(self, idx: int) -> DataEntry:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


@dataclass
class DatasetCollection:
    """
    Flattened access to a collection of datasets.
    """

    datasets: List[Dataset]
    interleave: bool = False

    def iter_sequential(self):
        for dataset in self.datasets:
            yield from dataset

    def iter_interleaved(self):
        yield from roundrobin(*self.datasets)

    def __iter__(self):
        if self.interleave:
            yield from self.iter_interleaved()
        else:
            yield from self.iter_sequential()

    def __len__(self):
        return sum(map(len, self.datasets))


class DatasetWriter:
    def write_to_file(self, dataset: Dataset, path: Path) -> None:
        raise NotImplementedError

    def write_to_folder(
            self, dataset: Dataset, folder: Path, name: Optional[str] = None
    ) -> None:
        raise NotImplementedError
