from __future__ import annotations

from iai_mcp.lilli.tiers import bsc


def empty_hv(*, D: int = bsc.LILLI_BSC_DEFAULT_DIM) -> bytes:
    return bytes(D // 8)


def add_pair(
    hv: bytes,
    role: str,
    filler_value: str,
    *,
    D: int = bsc.LILLI_BSC_DEFAULT_DIM,
) -> bytes:
    bound_pair = bsc.bind(bsc.role_hv(role, D=D), bsc.filler_hv(filler_value, D=D))
    return bsc.bind(hv, bound_pair)


def update_role(
    hv: bytes,
    old_filler_value: str,
    role: str,
    new_filler_value: str,
    *,
    D: int = bsc.LILLI_BSC_DEFAULT_DIM,
) -> bytes:
    hv_without_old = add_pair(hv, role, old_filler_value, D=D)
    return add_pair(hv_without_old, role, new_filler_value, D=D)
