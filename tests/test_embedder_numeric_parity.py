from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


BASELINE = Path(__file__).parent.parent / "bench" / "embedder_baseline"


def _cosine(a: list[float], b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    return float(np.dot(a_arr, b) / (np.linalg.norm(a_arr) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def baseline_texts() -> list[str]:
    return json.loads((BASELINE / "texts.json").read_text())


@pytest.fixture(scope="module")
def baseline_vectors() -> np.ndarray:
    arr = np.load(BASELINE / "vectors.npy")
    assert arr.shape == (100, 384)
    assert arr.dtype == np.float32
    return arr


def _rust_available() -> bool:
    try:
        from iai_mcp_native import embed  # noqa: F401
        return True
    except ImportError:
        return False


_HF_CACHE = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
HAS_BGE_SMALL_CACHE = any(_HF_CACHE.rglob("*bge-small-en*")) if _HF_CACHE.exists() else False


@pytest.mark.skipif(not _rust_available() or not HAS_BGE_SMALL_CACHE, reason="iai_mcp_native not installed or bge-small-en-v1.5 not cached")
def test_rust_cosine_parity(baseline_texts, baseline_vectors):
    from iai_mcp.embed import Embedder
    e = Embedder()
    assert e._backend == "rust"
    failures: list[tuple[int, float, str]] = []
    for i, text in enumerate(baseline_texts):
        got = e.embed(text)
        assert len(got) == 384, f"text[{i}] len={len(got)}"
        cos = _cosine(got, baseline_vectors[i])
        if cos < 0.9999:
            failures.append((i, cos, text[:60]))
    assert not failures, (
        f"{len(failures)} of 100 texts failed cosine >= 0.9999:\n"
        + "\n".join(f"  text[{i}] cos={c:.6f} preview={p!r}" for i, c, p in failures[:10])
    )


@pytest.mark.skipif(not _rust_available() or not HAS_BGE_SMALL_CACHE, reason="iai_mcp_native not installed or bge-small-en-v1.5 not cached")
def test_default_backend_is_rust():
    from iai_mcp.embed import Embedder
    e = Embedder()
    assert e._backend == "rust"
    v = e.embed("hello")
    assert len(v) == 384


@pytest.mark.skipif(not _rust_available() or not HAS_BGE_SMALL_CACHE, reason="iai_mcp_native not installed or bge-small-en-v1.5 not cached")
def test_backend_routing_rust():
    from iai_mcp.embed import Embedder
    e = Embedder()
    assert e._backend == "rust"
    v = e.embed("hello")
    assert len(v) == 384


@pytest.mark.skipif(not _rust_available() or not HAS_BGE_SMALL_CACHE, reason="iai_mcp_native not installed or bge-small-en-v1.5 not cached")
def test_tokenizer_handles_oversized_text():
    from iai_mcp.embed import Embedder
    e = Embedder()
    long_text = "word " * 1000
    v = e.embed(long_text)
    assert len(v) == 384


def test_tokenizer_id_byte_parity(baseline_texts):
    try:
        from tokenizers import Tokenizer as RustTok
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("tokenizers or transformers not installed")

    revision = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    snapshot_dir = (
        Path.home() / ".cache/huggingface/hub"
        / "models--BAAI--bge-small-en-v1.5"
        / "snapshots" / revision
    )
    tok_json = snapshot_dir / "tokenizer.json"
    if not tok_json.exists():
        pytest.skip(f"HF cache missing at {tok_json}")

    rs = RustTok.from_file(str(tok_json))
    rs.enable_truncation(max_length=512)

    pt = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5",
        revision=revision,
    )

    mismatches: list[tuple[int, list[int], list[int]]] = []
    for i, text in enumerate(baseline_texts):
        rs_ids = rs.encode(text).ids
        pt_ids = pt(text, truncation=True, max_length=512)["input_ids"]
        if list(rs_ids) != list(pt_ids):
            mismatches.append((i, list(rs_ids), list(pt_ids)))

    assert not mismatches, (
        f"{len(mismatches)} of 100 texts produced divergent token IDs:\n"
        + "\n".join(
            f"  text[{i}] rs[:10]={rs_ids[:10]} pt[:10]={pt_ids[:10]} (lengths rs={len(rs_ids)} pt={len(pt_ids)})"
            for i, rs_ids, pt_ids in mismatches[:5]
        )
    )
