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

    A worker with no example in the final global batch receives ``-1``.  The
    distributed collator masks that placeholder's labels, so all workers still
    enter the same DDP collectives without contributing a loss term.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int = 42,
        multiplier: int = 50,
        shuffle: bool = True,
        bucket: bool = True,
    ):
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size or multiplier <= 0:
            raise ValueError("Invalid distributed batch sampler configuration")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.multiplier = int(multiplier)
        self.shuffle = bool(shuffle)
        self.bucket = bool(bucket)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
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
            batches = (
                indices[start : start + global_batch_size] for start in range(0, len(indices), global_batch_size)
            )
        for batch in batches:
            yield batch[self.rank :: self.world_size] or [-1]


class DistributedDataset(Dataset):
    """Keep the original index visible to the collator for placeholder rows."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[max(0, index)], index >= 0


class DistributedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __getattr__(self, name):
        # Runtime toggles include_targets after construction during evaluation.
        # During multiprocessing ``spawn`` unpickling, ``collator`` may not
        # have been restored yet.  Fetch it through ``object`` so a missing
        # attribute raises normally instead of recursively calling this
        # method until the worker hits the recursion limit.
        try:
            collator = object.__getattribute__(self, "collator")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(collator, name)

    def __setattr__(self, name, value):
        if name == "collator" or "collator" not in self.__dict__:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "collator"), name, value)

    def __call__(self, rows):
        batch = self.collator([record for record, _ in rows])
        for index, (_, real) in enumerate(rows):
            if not real:
                batch["labels"][index].fill_(-100)
        batch["example_count"] = sum(real for _, real in rows)
        return batch


class DistributedEvalSampler(Sampler[int]):
    """Deterministic, non-overlapping evaluation indices for every rank.

    When byte-length estimates and a batch size are supplied, examples are
    sorted inside global batches before striding across ranks. This keeps long
    examples balanced between GPUs while preserving exact source indices for
    rank-zero output ordering.
    """

    def __init__(
        self,
        dataset_size: int,
        rank: int,
        world_size: int,
        max_examples: int = 0,
        lengths: Sequence[int] | None = None,
        batch_size: int = 1,
    ):
        if dataset_size < 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed evaluation sampler configuration")
        if batch_size <= 0:
            raise ValueError("Evaluation batch size must be positive")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.limit = min(self.dataset_size, int(max_examples)) if max_examples > 0 else self.dataset_size
        self.lengths = lengths
        self.batch_size = int(batch_size)

    def __iter__(self):
        if self.lengths is None:
            yield from range(self.rank, self.limit, self.world_size)
            return
        indices = sorted(range(self.limit), key=self.lengths.__getitem__)
        global_batch_size = self.batch_size * self.world_size
        for start in range(0, len(indices), global_batch_size):
            yield from indices[start : start + global_batch_size][self.rank :: self.world_size]

    def __len__(self) -> int:
        return max(0, (self.limit - self.rank + self.world_size - 1) // self.world_size)
