from __future__ import annotations

import math
import random
from collections.abc import Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 42, multiplier: int = 50):
        if batch_size <= 0 or multiplier <= 0:
            raise ValueError("Batch size and bucket multiplier must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.multiplier = multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        pool_size = self.batch_size * self.multiplier
        batches = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start : start + pool_size], key=self.lengths.__getitem__)
            batches.extend(pool[i : i + self.batch_size] for i in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        yield from batches
