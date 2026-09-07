from __future__ import annotations

import math
import random
from collections.abc import Sequence

from torch.utils.data import Dataset, Sampler


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


class DistributedBatchSampler(Sampler[list[int]]):
    """Shard global batches without dropping or repeating supervised examples.

    A rank with no example in the last global batch receives a -1 placeholder.
    Its collator masks all labels so it participates in DDP with zero weight.
    """

    def __init__(self, lengths, batch_size, rank, world_size, seed=42, multiplier=50, shuffle=True, bucket=True):
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size or multiplier <= 0:
            raise ValueError("Invalid distributed batch sampler configuration")
        self.lengths, self.batch_size = lengths, int(batch_size)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed, self.multiplier = int(seed), int(multiplier)
        self.shuffle, self.bucket = bool(shuffle), bool(bucket)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.lengths) / (self.batch_size * self.world_size))

    def __iter__(self):
        global_batch_size = self.batch_size * self.world_size
        if self.shuffle and self.bucket:
            sampler = LengthBucketBatchSampler(self.lengths, global_batch_size, self.seed, self.multiplier)
            sampler.set_epoch(self.epoch)
            batches = iter(sampler)
        else:
            indices = list(range(len(self.lengths)))
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(indices)
            batches = (indices[start : start + global_batch_size] for start in range(0, len(indices), global_batch_size))
        for batch in batches:
            yield batch[self.rank :: self.world_size] or [-1]


class DistributedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[max(0, index)], index >= 0


class DistributedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, rows):
        batch = self.collator([record for record, _ in rows])
        for index, (_, real) in enumerate(rows):
            if not real:
                batch["labels"][index].fill_(-100)
        batch["example_count"] = sum(real for _, real in rows)
        return batch
