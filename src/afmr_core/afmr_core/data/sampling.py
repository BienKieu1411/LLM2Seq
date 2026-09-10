"""Deterministic length bucketing and replayable global batch manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset, Sampler


def materialize_global_batches(
    lengths: Sequence[int],
    global_batch_size: int,
    *,
    seed: int = 42,
    epoch: int = 0,
    multiplier: int = 50,
    shuffle: bool = True,
    bucket: bool = True,
    tail_policy: str = "keep",
) -> list[list[int]]:
    """Build the one canonical order that every rank consumes.

    ``tail_policy=keep`` retains each real example once and lets the
    distributed wrapper insert a masked placeholder on ranks that have no
    element in the final partial batch.  ``drop`` is provided only for
    explicitly registered throughput controls.
    """

    if global_batch_size <= 0 or multiplier <= 0:
        raise ValueError("global_batch_size and multiplier must be positive")
    if tail_policy not in {"keep", "drop"}:
        raise ValueError("tail_policy must be keep or drop")
    indices = list(range(len(lengths)))
    rng = random.Random(int(seed) + int(epoch))
    if shuffle:
        rng.shuffle(indices)
    if bucket:
        pool_size = global_batch_size * int(multiplier)
        batches: list[list[int]] = []
        for start in range(0, len(indices), pool_size):
            pool = sorted(indices[start : start + pool_size], key=lengths.__getitem__)
            batches.extend(
                pool[offset : offset + global_batch_size] for offset in range(0, len(pool), global_batch_size)
            )
        if shuffle:
            rng.shuffle(batches)
    else:
        batches = [indices[offset : offset + global_batch_size] for offset in range(0, len(indices), global_batch_size)]
    if tail_policy == "drop" and batches and len(batches[-1]) < global_batch_size:
        batches.pop()
    return batches


@dataclass(frozen=True)
class CanonicalBatchManifest:
    """Serializable global batch schedule used for exact DDP replay."""

    batches: tuple[tuple[int, ...], ...]
    dataset_size: int
    global_batch_size: int
    seed: int
    epoch: int
    multiplier: int
    shuffle: bool
    bucket: bool
    tail_policy: str = "keep"

    @classmethod
    def build(cls, lengths: Sequence[int], global_batch_size: int, **kwargs) -> "CanonicalBatchManifest":
        batches = materialize_global_batches(lengths, global_batch_size, **kwargs)
        return cls(
            tuple(tuple(batch) for batch in batches),
            len(lengths),
            int(global_batch_size),
            int(kwargs.get("seed", 42)),
            int(kwargs.get("epoch", 0)),
            int(kwargs.get("multiplier", 50)),
            bool(kwargs.get("shuffle", True)),
            bool(kwargs.get("bucket", True)),
            str(kwargs.get("tail_policy", "keep")),
        )

    @property
    def manifest_hash(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "batches": [list(batch) for batch in self.batches],
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "multiplier": self.multiplier,
            "shuffle": self.shuffle,
            "bucket": self.bucket,
            "tail_policy": self.tail_policy,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["manifest_hash"] = self.manifest_hash
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CanonicalBatchManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = payload.pop("manifest_hash", None)
        manifest = cls(
            tuple(tuple(int(index) for index in batch) for batch in payload["batches"]),
            int(payload["dataset_size"]),
            int(payload["global_batch_size"]),
            int(payload["seed"]),
            int(payload["epoch"]),
            int(payload["multiplier"]),
            bool(payload["shuffle"]),
            bool(payload["bucket"]),
            str(payload.get("tail_policy", "keep")),
        )
        if expected is not None and expected != manifest.manifest_hash:
            raise ValueError("Canonical batch manifest hash mismatch")
        return manifest


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int = 42, multiplier: int = 50):
        if batch_size <= 0 or multiplier <= 0:
            raise ValueError("Batch size and bucket multiplier must be positive")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.multiplier = int(multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    @property
    def manifest_hash(self) -> str:
        return CanonicalBatchManifest.build(
            self.lengths,
            self.batch_size,
            seed=self.seed,
            epoch=self.epoch,
            multiplier=self.multiplier,
            shuffle=True,
            bucket=True,
        ).manifest_hash

    def __iter__(self):
        yield from materialize_global_batches(
            self.lengths,
            self.batch_size,
            seed=self.seed,
            epoch=self.epoch,
            multiplier=self.multiplier,
            shuffle=True,
            bucket=True,
        )


class DistributedBatchSampler(Sampler[list[int]]):
    """Shard a canonical global manifest while preserving every real example."""

    def __init__(
        self,
        lengths,
        batch_size,
        rank,
        world_size,
        seed=42,
        multiplier=50,
        shuffle=True,
        bucket=True,
        manifest: CanonicalBatchManifest | None = None,
        tail_policy: str = "keep",
    ):
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size or multiplier <= 0:
            raise ValueError("Invalid distributed batch sampler configuration")
        self.lengths, self.batch_size = lengths, int(batch_size)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed, self.multiplier = int(seed), int(multiplier)
        self.shuffle, self.bucket = bool(shuffle), bool(bucket)
        self.tail_policy = tail_policy
        self.epoch = 0
        self.manifest = manifest
        if manifest is not None and manifest.dataset_size != len(lengths):
            raise ValueError("Canonical manifest dataset size does not match sampler")
        if manifest is not None and manifest.global_batch_size != self.batch_size * self.world_size:
            raise ValueError("Canonical manifest global batch size does not match local batch/world size")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.manifest is not None and self.manifest.epoch != self.epoch:
            raise ValueError("A fixed canonical manifest cannot be reused for a different epoch")

    def _manifest(self) -> CanonicalBatchManifest:
        return self.manifest or CanonicalBatchManifest.build(
            self.lengths,
            self.batch_size * self.world_size,
            seed=self.seed,
            epoch=self.epoch,
            multiplier=self.multiplier,
            shuffle=self.shuffle,
            bucket=self.bucket,
            tail_policy=self.tail_policy,
        )

    def __len__(self):
        return len(self._manifest().batches)

    @property
    def manifest_hash(self) -> str:
        return self._manifest().manifest_hash

    def __iter__(self):
        for batch in self._manifest().batches:
            shard = list(batch[self.rank :: self.world_size])
            yield shard or [-1]


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


__all__ = [
    "CanonicalBatchManifest",
    "DistributedBatchSampler",
    "DistributedCollator",
    "DistributedDataset",
    "LengthBucketBatchSampler",
    "materialize_global_batches",
]
