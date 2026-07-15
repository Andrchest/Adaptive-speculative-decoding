"""GPU-event-based substep timer for bottleneck diagnosis in speculative decoding.

Uses ``torch.cuda.Event`` for real GPU kernel timing instead of
``time.perf_counter()`` which only measures CPU-side launch time.

Usage
-----
Enable globally before a run, disable after, then call ``summary()``::

    from core.profiling.substep_timer import substep_timer

    substep_timer.enable()
    # ... run speculative decoding ...
    substep_timer.disable()
    for name, stats in substep_timer.summary().items():
        print(f"{name:40s}  min={stats['min']:.2f}ms  ...")

Inside instrumented code::

    from core.profiling.substep_timer import substep_timer

    with substep_timer.track("ar.logsumexp"):
        log_norm = torch.logsumexp(t_logits, dim=-1)

The timer is a no-op when disabled (``active == False``).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class SubstepTimer:
    """Accumulator for named GPU-kernel timing measurements via CUDA events."""

    def __init__(self) -> None:
        self._active: bool = False
        self._data: dict[str, list[float]] = {}
        # Pending (name, start_event, end_event) — flushed lazily on summary()
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    # -- control --

    def enable(self) -> None:
        self._active = True
        self._data.clear()
        self._pending.clear()

    def disable(self) -> None:
        self._flush()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    # -- recording --

    def record(self, name: str, elapsed_ms: float) -> None:
        if self._active:
            self._data.setdefault(name, []).append(elapsed_ms)

    @contextmanager
    def track(self, name: str):
        """Context manager — records real GPU time via CUDA events.

        Falls back to ``time.perf_counter()`` when CUDA is unavailable.
        """
        if not self._active:
            yield
            return

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._data.setdefault(name, []).append((time.perf_counter() - t0) * 1000)

    # -- internal --

    def _flush(self) -> None:
        """Sync GPU once and compute all pending event timings."""
        if not self._pending:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending:
            self._data.setdefault(name, []).append(start.elapsed_time(end))
        self._pending.clear()

    # -- reporting --

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{name: {min, max, avg, sum, count}}`` for every recorded substep."""
        self._flush()
        result: dict[str, dict[str, float]] = {}
        for name, times in self._data.items():
            result[name] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "sum": sum(times),
                "count": len(times),
            }
        return result

    def raw(self) -> dict[str, list[float]]:
        """Return the raw per-call lists (for external analyzers)."""
        self._flush()
        return dict(self._data)


# Module-level singleton — import and use directly.
substep_timer = SubstepTimer()
