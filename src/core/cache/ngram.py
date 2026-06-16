"""
core/cache/ngram.py

N-gram cache with pluggable eviction strategies.

Each entry tracks:
  - hit_count
  - acceptance_rate (running mean)
  - last_used (step index)
  - insert_step

Eviction strategies:
  - lru   : evict least recently used
  - lfu   : evict least frequently used
  - acc   : acceptance-weighted (score = hit_count * acceptance_rate / age)
  - hybrid: score = hit_count * acceptance_rate / age  (alias for acc)
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    token_ids: list[int]  # predicted continuation token ids
    logits: torch.Tensor | None  # drafter logits (vocab,)
    hit_count: int = 0
    acceptance_rate: float = 0.5  # initialise optimistically
    insert_step: int = 0
    last_used_step: int = 0

    def update_acceptance(self, accepted: bool, alpha: float = 0.1) -> None:
        """Exponential moving average of acceptance signal."""
        self.acceptance_rate = (1 - alpha) * self.acceptance_rate + alpha * float(accepted)

    def eviction_score(self, current_step: int) -> float:
        """Higher score → keep longer."""
        age = max(1, current_step - self.insert_step)
        return (self.hit_count * self.acceptance_rate) / age


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class NgramCache:
    """
    Keyed by (n-gram tuple of token ids).

    Parameters
    ----------
    max_size   : maximum number of entries before eviction
    n          : n-gram order used as lookup key
    eviction   : 'lru' | 'lfu' | 'acc' | 'hybrid'
    ema_alpha  : smoothing factor for acceptance rate EMA
    """

    def __init__(
        self,
        max_size: int = 65536,
        n: int = 3,
        eviction: str = "hybrid",
        ema_alpha: float = 0.1,
    ) -> None:
        self.max_size = max_size
        self.n = n
        self.eviction = eviction
        self.ema_alpha = ema_alpha

        # OrderedDict preserves insertion order → useful for LRU
        self._store: OrderedDict[tuple[int, ...], CacheEntry] = OrderedDict()
        self._step: int = 0

        # Global stats
        self.total_lookups: int = 0
        self.total_hits: int = 0
        logger.info("NgramCache initialized: max_size=%d n=%d eviction=%s", max_size, n, eviction)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, context: list[int]) -> CacheEntry | None:
        """
        Look up the n-gram suffix of *context*.

        Returns the matching CacheEntry or None.
        """
        key = self._make_key(context)
        self.total_lookups += 1
        entry = self._store.get(key)
        if entry is None:
            logger.debug("Cache MISS for key=%s", key)
            return None
        logger.debug("Cache HIT for key=%s hit_count=%d", key, entry.hit_count + 1)
        # Update recency
        entry.hit_count += 1
        entry.last_used_step = self._step
        if self.eviction == "lru":
            self._store.move_to_end(key)
        self.total_hits += 1
        return entry

    def insert(
        self,
        context: list[int],
        token_ids: list[int],
        logits: torch.Tensor | None = None,
    ) -> None:
        """
        Insert or overwrite a cache entry.

        If the cache is full, an entry is evicted according to the strategy.
        """
        key = self._make_key(context)
        if key in self._store:
            # Refresh entry, preserve acceptance stats
            entry = self._store[key]
            entry.token_ids = token_ids
            entry.logits = logits
            entry.last_used_step = self._step
            if self.eviction == "lru":
                self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_size:
                self._evict()
            self._store[key] = CacheEntry(
                token_ids=token_ids,
                logits=logits,
                insert_step=self._step,
                last_used_step=self._step,
            )
            logger.debug("Cache INSERT key=%s size=%d/%d", key, len(self._store), self.max_size)

    def remove(self, context: list[int]) -> bool:
        """
        Explicitly remove an entry.

        Returns True if the entry existed.
        """
        key = self._make_key(context)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def update_acceptance(self, context: list[int], accepted: bool) -> None:
        """Propagate accept/reject signal to the entry's EMA."""
        key = self._make_key(context)
        entry = self._store.get(key)
        if entry is not None:
            entry.update_acceptance(accepted, alpha=self.ema_alpha)

    def step(self) -> None:
        """Advance the internal step counter (call once per decoding step)."""
        self._step += 1

    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return self.total_hits / self.total_lookups

    def mean_acceptance_rate(self) -> float:
        if not self._store:
            return 0.0
        return sum(e.acceptance_rate for e in self._store.values()) / len(self._store)

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "total_lookups": self.total_lookups,
            "total_hits": self.total_hits,
            "hit_rate": self.hit_rate(),
            "mean_acceptance_rate": self.mean_acceptance_rate(),
            "step": self._step,
        }

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_key(self, context: list[int]) -> tuple[int, ...]:
        return tuple(context[-self.n :])

    def _evict(self) -> None:
        strategy = self.eviction
        if strategy == "lru":
            # OrderedDict front = oldest
            self._store.popitem(last=False)
        elif strategy == "lfu":
            victim = min(self._store, key=lambda k: self._store[k].hit_count)
            del self._store[victim]
        elif strategy in ("acc", "hybrid"):
            victim = min(
                self._store,
                key=lambda k: self._store[k].eviction_score(self._step),
            )
            del self._store[victim]
        else:
            raise ValueError(f"Unknown eviction strategy: {strategy!r}")
        logger.debug(
            "Cache EVICT strategy=%s size=%d/%d", strategy, len(self._store), self.max_size
        )
