from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_V4


def _first_50_chars(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    return s[:50].rstrip()


class ReflectionAgent:

    def synthesize(self, store, window_hours: int) -> MemoryRecord:
        from iai_mcp.events import query_events  # noqa: F401  (kept for symmetry)

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        recs = store.all_records()
        in_window: list = []
        for r in recs:
            created = getattr(r, "created_at", None)
            if created is None:
                continue
            try:
                if getattr(created, "tzinfo", None) is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError, AttributeError):
                continue
            if created < cutoff:
                continue
            prov_list = getattr(r, "provenance", None) or []
            if any(
                (p.get("synthesized_by") == "dmn_reflection" if isinstance(p, dict) else False)
                for p in prov_list
            ):
                continue
            in_window.append(r)

        captured_count = len(in_window)

        community_to_records: dict[UUID, list] = {}
        for r in in_window:
            cid = getattr(r, "community_id", None)
            if cid is None:
                continue
            community_to_records.setdefault(cid, []).append(r)

        community_counts: Counter = Counter(
            {cid: len(rlist) for cid, rlist in community_to_records.items()}
        )

        community_labels: dict[UUID, str] = {}
        for cid, rlist in community_to_records.items():
            top = max(rlist, key=lambda r: getattr(r, "created_at", now))
            community_labels[cid] = _first_50_chars(
                getattr(top, "literal_surface", "")
            )

        top_cids = [cid for cid, _ in community_counts.most_common(5)]
        topics: list[str] = [
            community_labels[cid]
            for cid in top_cids
            if community_labels.get(cid)
        ]

        recall_events = query_events(
            store,
            kind="memory_recall",
            since=cutoff,
            limit=10000,
        )
        recalled_count = len(recall_events)

        topics_str = "[" + ", ".join(topics) + "]"
        literal_surface = (
            f"Daily reflection: top topics were {topics_str}; "
            f"captured {captured_count} turns; "
            f"recalled {recalled_count} times."
        )

        provenance_entry: dict[str, Any] = {
            "synthesized_by": "dmn_reflection",
            "window_hours": int(window_hours),
            "topics": list(topics),
            "captured_count": int(captured_count),
            "recalled_count": int(recalled_count),
            "ts": now.isoformat(),
        }

        embed_dim = int(store.embed_dim)
        embedding = [0.0] * embed_dim

        return MemoryRecord(
            id=uuid4(),
            tier="semantic",
            literal_surface=literal_surface,
            aaak_index="",
            embedding=embedding,
            community_id=None,
            centrality=0.5,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[provenance_entry],
            created_at=now,
            updated_at=now,
            language="en",
            tags=[],
            s5_trust_score=0.5,
            profile_modulation_gain={},
            schema_version=SCHEMA_VERSION_V4,
            structure_hv=b"",
        )


class MetaAnalyst:

    _QUERY_LIMIT: int = 10000

    def snapshot(self, store, window_hours: int) -> dict[str, Any]:
        from iai_mcp.events import query_events

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=window_hours)

        events_list = query_events(
            store,
            since=cutoff,
            limit=self._QUERY_LIMIT,
        )

        recall_count = 0
        capture_count = 0
        sleep_cycles_count = 0
        breach_count = 0
        erasure_count = 0

        for ev in events_list:
            kind = ev.get("kind") or ""
            if kind == "memory_recall":
                recall_count += 1
            elif kind == "memory_capture":
                capture_count += 1
            elif kind == "sleep_step_completed":
                data = ev.get("data") or {}
                step_name = data.get("step")
                if step_name == "HIPPO_CLEANUP":
                    sleep_cycles_count += 1
            elif kind == "essential_variable_breach":
                breach_count += 1
            elif kind == "erasure_agent_pass":
                erasure_count += 1

        if (capture_count + erasure_count) > 0:
            average_record_count_delta = float(
                capture_count - erasure_count
            )
        else:
            average_record_count_delta = 0.0

        return {
            "recall_count": recall_count,
            "capture_count": capture_count,
            "sleep_cycles_count": sleep_cycles_count,
            "breach_count": breach_count,
            "erasure_count": erasure_count,
            "average_record_count_delta": average_record_count_delta,
            "window_hours": int(window_hours),
            "generated_at": now.isoformat(),
        }
