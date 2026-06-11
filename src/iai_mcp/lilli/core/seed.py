from __future__ import annotations

import hashlib
import logging
import math

import numpy as np

log = logging.getLogger(__name__)


def seed_from_str(prefix: str, value: str) -> int:
    digest = hashlib.sha256(f"{prefix}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def hv_from_seed(seed: int, D: int) -> bytes:
    if D <= 0:
        raise ValueError(f"D must be a positive integer, got {D}")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=D, dtype=np.uint8)
    return np.packbits(bits).tobytes()
