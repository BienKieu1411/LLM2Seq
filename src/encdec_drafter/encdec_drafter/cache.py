from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheTransaction:
    manager: "FullHistoryCache"
    start_length: int
    closed: bool = False

    def commit_to(self, absolute_length: int) -> None:
        if self.closed:
            raise RuntimeError("Cache transaction is already closed")
        self.manager.rollback_to(absolute_length)
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        self.manager.rollback_to(self.start_length)
        self.closed = True


class FullHistoryCache:
    """Dynamic full-history target cache with rollback-safe speculative writes."""

    def __init__(self, cache: Any):
        self.cache = cache
        self._ensure_full_history()

    @classmethod
    def create(cls) -> "FullHistoryCache":
        try:
            from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        except ImportError as exc:
            raise RuntimeError("Installed Transformers lacks EncoderDecoderCache/DynamicCache") from exc
        return cls(EncoderDecoderCache(DynamicCache(), DynamicCache()))

    def _ensure_full_history(self) -> None:
        cache = self.cache.self_attention_cache
        for layer in cache.layers:
            if bool(getattr(layer, "is_sliding", False)):
                raise RuntimeError(
                    "Speculative rollback requires full-history DynamicLayer caches; sliding cache was supplied"
                )

    @property
    def length(self) -> int:
        return int(self.cache.get_seq_length())

    def begin(self) -> CacheTransaction:
        self._ensure_full_history()
        return CacheTransaction(manager=self, start_length=self.length)

    def rollback_to(self, absolute_length: int) -> None:
        if absolute_length < 0 or absolute_length > self.length:
            raise ValueError(f"Cannot crop cache length {self.length} to {absolute_length}")
        if absolute_length != self.length:
            self.cache.crop(absolute_length)
        self._ensure_full_history()
        if self.length != absolute_length:
            raise RuntimeError(f"Cache rollback requested {absolute_length}, got {self.length}")

    def abort_on_error(self, transaction: CacheTransaction, error: BaseException) -> None:
        try:
            transaction.abort()
        except Exception as rollback_error:
            raise RuntimeError("Forward failed and target cache rollback also failed") from rollback_error
        raise error
