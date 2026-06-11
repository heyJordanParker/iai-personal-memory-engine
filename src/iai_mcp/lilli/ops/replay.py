from __future__ import annotations

import numpy as np


def replay_with_noise(
    hv: bytes,
    sigma: float = 0.05,
    seed: int | None = None,
) -> bytes:
    if not (0.0 <= sigma <= 1.0):
        raise ValueError(
            f"sigma must be in [0.0, 1.0], got {sigma!r}"
        )

    if sigma == 0.0:
        return hv

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    bits = np.unpackbits(np.frombuffer(hv, dtype=np.uint8))
    flip_mask = (rng.random(bits.shape) < sigma).astype(np.uint8)
    noisy = bits ^ flip_mask
    return np.packbits(noisy).tobytes()
