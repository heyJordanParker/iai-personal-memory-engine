from __future__ import annotations

import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from iai_mcp.community import detect_communities
from iai_mcp.graph import IGRAPH_THRESHOLD
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import _aaak_overlap, recall_for_response
from iai_mcp.richclub import rich_club_nodes
from iai_mcp.store import MemoryStore
from iai_mcp.types import MemoryRecord


class FakeEmbedder:
    DIM = 384

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 383


def _random_emb(seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.random() for _ in range(384)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n > 0 else v


def _make(vec: list[float], text: str = "rec") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[],
        language="en",
    )


def bench_pipeline_100_records() -> None:
    print("=== Pipeline stage timings: 100 records, ~200 edges ===")
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(path=Path(td))
        records = [_make(_random_emb(i), text=f"r{i}") for i in range(100)]
        t0 = time.perf_counter()
        for r in records:
            store.insert(r)
        t1 = time.perf_counter()
        print(f"  store insert (100 records):         {(t1 - t0) * 1000:8.2f} ms")

        graph = MemoryGraph()
        for r in records:
            graph.add_node(r.id, community_id=None, embedding=r.embedding)
        for i in range(100):
            graph.add_edge(records[i].id, records[(i + 1) % 100].id)
            graph.add_edge(records[i].id, records[(i + 2) % 100].id)
        t2 = time.perf_counter()
        print(f"  graph build (100 nodes, 200 edges): {(t2 - t1) * 1000:8.2f} ms")

        assignment = detect_communities(graph, prior=None)
        t3 = time.perf_counter()
        print(
            f"  community detect:                   {(t3 - t2) * 1000:8.2f} ms  "
            f"backend={assignment.backend}  Q={assignment.modularity:.3f}"
        )

        rc = rich_club_nodes(graph, percent=0.10)
        t4 = time.perf_counter()
        print(
            f"  rich_club (top 10%):                {(t4 - t3) * 1000:8.2f} ms  "
            f"selected={len(rc)} nodes"
        )

        embedder = FakeEmbedder()
        resp = recall_for_response(
            store,
            graph,
            assignment,
            rc,
            embedder,
            cue="rec query",
            session_id="bench",
        )
        t5 = time.perf_counter()
        print(
            f"  recall_for_response (full 5 stages):    {(t5 - t4) * 1000:8.2f} ms  "
            f"hits={len(resp.hits)}  trace={len(resp.activation_trace)}"
        )
        print(f"  TOTAL end-to-end:                   {(t5 - t0) * 1000:8.2f} ms")


def bench_two_cliques_modularity() -> None:
    print("\n=== Two-clique modularity (150 + 150 = 300 nodes) ===")
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    t0 = time.perf_counter()
    a = detect_communities(g, prior=None)
    t1 = time.perf_counter()
    comm_count = len(set(a.node_to_community.values()))
    print(f"  backend:     {a.backend}")
    print(f"  modularity:  Q = {a.modularity:.4f}")
    print(f"  communities: {comm_count}")
    print(f"  duration:    {(t1 - t0) * 1000:8.2f} ms")


def bench_backend_flip() -> None:
    print("\n=== Backend switch at IGRAPH_THRESHOLD ===")
    g1 = MemoryGraph()
    for _ in range(IGRAPH_THRESHOLD - 1):
        g1.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    print(f"  N = {IGRAPH_THRESHOLD - 1}:  backend = {g1.backend}")
    g2 = MemoryGraph()
    for _ in range(IGRAPH_THRESHOLD):
        g2.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    print(f"  N = {IGRAPH_THRESHOLD}:  backend = {g2.backend}")
    g3 = MemoryGraph()
    for _ in range(IGRAPH_THRESHOLD + 1):
        g3.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    print(f"  N = {IGRAPH_THRESHOLD + 1}:  backend = {g3.backend}")


def bench_aaak_jaccard_quality() -> None:
    print("\n=== AAAK overlap Jaccard quality notes ===")
    cases = [
        ("hello world", "hello world", "identical"),
        ("hello world", "hello universe", "1 token shared"),
        ("auth/login", "auth/login", "identical with slash"),
        ("auth/login", "auth/logout", "slash-split catches 'auth'"),
        ("TESTCASE", "testcase", "case-insensitive"),
        ("", "anything", "empty cue -> 0.0"),
        ("a b c d", "e f g h", "zero overlap"),
        ("alice speaks", "alice russian", "partial overlap"),
    ]
    print(f"  {'cue':<20} | {'aaak_index':<20} | {'jaccard':>8} | note")
    print(f"  {'-' * 20} | {'-' * 20} | {'-' * 8} | ----")
    for cue, idx, note in cases:
        score = _aaak_overlap(cue, idx)
        print(f"  {cue:<20} | {idx:<20} | {score:>8.3f} | {note}")


if __name__ == "__main__":
    bench_pipeline_100_records()
    bench_two_cliques_modularity()
    bench_backend_flip()
    bench_aaak_jaccard_quality()
