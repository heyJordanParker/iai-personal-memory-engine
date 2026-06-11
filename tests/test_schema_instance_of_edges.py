from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import EDGES_TABLE, MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def _rec(*, text: str = "t", tags: list[str] | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language="en",
    )

def test_schema_instance_of_edge_created_on_persist(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["m", "n"]) for i in range(5)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:m+n",
        confidence=0.9,
        evidence_count=5,
        evidence_ids=[r.id for r in ev],
        status="auto",
    )
    schema_id = persist_schema(store, cand)
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    sio = edges[edges["edge_type"] == "schema_instance_of"]
    assert len(sio) == 5

def test_schema_instance_of_edge_never_decays(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema
    from iai_mcp.sleep import _decay_edges

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["a", "b"]) for i in range(3)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:a+b", confidence=0.9, evidence_count=3,
        evidence_ids=[r.id for r in ev], status="auto",
    )
    persist_schema(store, cand)

    edges_tbl = store.db.open_table(EDGES_TABLE)
    from datetime import timedelta
    ancient = datetime.now(timezone.utc) - timedelta(days=500)
    edges_tbl.update(
        where="edge_type = 'schema_instance_of'",
        values={"updated_at": ancient, "weight": 0.0001},
    )
    _decay_edges(store)

    df = edges_tbl.to_pandas()
    sio = df[df["edge_type"] == "schema_instance_of"]
    assert len(sio) == 3

def test_schema_record_becomes_hub(tmp_path):
    from iai_mcp.schema import SchemaCandidate, persist_schema

    store = MemoryStore(path=tmp_path)
    ev = [_rec(text=f"x{i}", tags=["p", "q"]) for i in range(5)]
    for r in ev:
        store.insert(r)

    cand = SchemaCandidate(
        pattern="tags:p+q", confidence=0.9, evidence_count=5,
        evidence_ids=[r.id for r in ev], status="auto",
    )
    schema_id = persist_schema(store, cand)

    rec = store.get(schema_id)
    assert rec is not None
    assert rec.detail_level == 3
    assert rec.never_decay is True
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    sio = edges[edges["edge_type"] == "schema_instance_of"]
    assert len(sio) == 5
