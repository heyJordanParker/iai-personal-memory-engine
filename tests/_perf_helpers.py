"""Load-robustness helpers for the wall-clock perf/latency benches.

Two primitives shared by the correctness-adjacent timing benches that STAY in
the default gate (the perf regression guards):

- ``skip_if_loaded(threshold_per_core)`` — skip-with-reason when the 1-minute
  load average per logical core exceeds ``threshold_per_core``. A heavily
  loaded host inflates wall-clock latency for reasons unrelated to a code
  regression; skipping (rather than failing) keeps "green means green" — the
  bench still runs on an idle host and at the phase gate.

- ``best_of_n(fn, n)`` — run a zero-arg timing callable ``n`` times and return
  the MINIMUM. The minimum is the run least perturbed by transient scheduler
  noise / page-cache misses, so a single unlucky GC pause or context switch
  cannot turn a passing bench red.

Chosen defaults:
- ``threshold_per_core = 1.5`` — a 1-min load of 1.5× the core count is the
  conventional "machine is busy" line; above it, wall-clock timing is noise.
- ``n = 3`` — three samples reliably reject a single transient outlier while
  keeping the in-gate cost bounded (each perf run is ~1s at N=100).

``os.getloadavg()`` is POSIX-only and used nowhere else in the tree; the
``except`` clause makes the skip a no-op on platforms without it (e.g. a
hypothetical Windows runner) so the bench simply runs unguarded there.
"""
from __future__ import annotations

import os
from typing import Callable, TypeVar

import pytest

T = TypeVar("T")


def skip_if_loaded(threshold_per_core: float = 1.5) -> None:
    """Skip the calling perf test when the host's 1-min load per core is high.

    No-op on platforms without ``os.getloadavg`` (raises nothing, the bench
    runs unguarded). On a quiet host the load-per-core is well under 1.5 and
    this returns immediately.
    """
    try:
        load1 = os.getloadavg()[0] / (os.cpu_count() or 1)
    except (OSError, AttributeError):
        return
    if load1 > threshold_per_core:
        pytest.skip(
            f"machine load {load1:.2f}/core > {threshold_per_core} — perf bench skipped"
        )


def best_of_n(fn: Callable[[], T], n: int = 3) -> T:
    """Return the minimum of ``n`` independent invocations of ``fn``.

    ``fn`` must be a zero-arg callable returning an orderable timing metric
    (e.g. a p95 latency in ms). Each call is independent — the caller is
    responsible for making each invocation use a fresh store / seed so the
    runs do not share warm-cache state.
    """
    if n < 1:
        raise ValueError(f"best_of_n needs n >= 1, got {n}")
    return min(fn() for _ in range(n))
