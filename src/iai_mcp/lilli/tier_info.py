from __future__ import annotations

from iai_mcp.lilli.tiers import bsc, fhrr, sparse_vsa

_TIER_REGISTRY: dict[str, dict] = {
    "bsc": bsc.TIER_INFO,
    "fhrr": fhrr.TIER_INFO,
    "sparse_vsa": sparse_vsa.TIER_INFO,
}


def tier_info(tier_name: str) -> dict:
    if tier_name not in _TIER_REGISTRY:
        raise ValueError(
            f"Unknown tier {tier_name!r}; expected one of {sorted(_TIER_REGISTRY)}"
        )
    return dict(_TIER_REGISTRY[tier_name])  # return a copy to prevent mutation


def list_tiers() -> list[str]:
    return sorted(_TIER_REGISTRY)
