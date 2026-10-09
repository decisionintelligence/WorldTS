import logging
from typing import Any, Tuple

import numpy as np
from gluonts.transform.sampler import InstanceSampler

logger = logging.getLogger(__name__)


class RandomSampler(InstanceSampler):
    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)

        if a > b:
            return np.array([], dtype=int)

        window_size = b - a + 1
        # (indices,) = np.where(np.random.random_sample(window_size) < self.p)
        indices = np.random.permutation(window_size)
        # logger.info(f"#### random sample indices with length {len(indices)} ####")
        return indices + a


class UniformWithStartSampler(InstanceSampler):
    start: int = 0

    def _get_bounds(self, ts: np.ndarray) -> Tuple[int, int]:
        # get positive index for self.start
        start = self.start % ts.shape[self.axis]
        # real_start = start - self.min_future + 1
        real_start = start
        assert real_start >= self.min_past, f"real_start: {real_start}, min_past: {self.min_past}"
        return (
            real_start,
            ts.shape[self.axis] - self.min_future,
        )

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)

        if a > b:
            return np.array([], dtype=int)

        window_size = b - a + 1
        # indices = np.random.permutation(window_size)
        indices = np.arange(window_size)
        return indices + a
