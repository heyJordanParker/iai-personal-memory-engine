#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


PRODUCTION_STORE = Path.home() / ".iai-mcp"

BENCH_PASSPHRASE = "iai-mcp-bench-personal-fact-drift-2026"

DEFAULT_STORE_DIR = "/tmp/iai-mcp-bench-personal-fact-drift/store"
DEFAULT_OUTPUT_DIR = "bench/results"
DEFAULT_SEEDS = [13, 42, 137]
DEFAULT_K = 10
CANDIDATE_POOL_SIZE = 200
WARMUP_PASSES = 5

SHIP_GATE_RECALL_AT_10 = 0.80
SHIP_GATE_RETENTION_LOSS_CEILING = 0.10

SCALE_PRESETS = {
    "smoke":    (5,       3,        2,                      10),
    "mvp":      (50,      20,       10,                     100),
    "honest":   (50,      50,       30,                     500),
}


FACT_SPECS: list[dict[str, str]] = [
    {"text": "My favorite color is teal.",
     "probe": "What color do I prefer?",
     "category": "preference"},
    {"text": "I drink yerba mate in the morning.",
     "probe": "What do I drink in the morning?",
     "category": "preference"},
    {"text": "My primary hobby is woodworking.",
     "probe": "What hobby do I do?",
     "category": "preference"},
    {"text": "My favorite cuisine is Vietnamese.",
     "probe": "What cuisine do I enjoy most?",
     "category": "preference"},
    {"text": "I prefer ambient music while coding.",
     "probe": "What music do I play while coding?",
     "category": "preference"},
    {"text": "My favorite sport is bouldering.",
     "probe": "What sport do I practice?",
     "category": "preference"},
    {"text": "I like longform interview podcasts.",
     "probe": "What kind of podcasts do I enjoy?",
     "category": "preference"},
    {"text": "I exercise in the early evening.",
     "probe": "When do I usually exercise?",
     "category": "preference"},
    {"text": "I read historical biographies for fun.",
     "probe": "What kind of books do I read for fun?",
     "category": "preference"},
    {"text": "My favorite dessert is dark chocolate sorbet.",
     "probe": "What dessert do I love most?",
     "category": "preference"},
    {"text": "On weekends I prefer hiking trails.",
     "probe": "How do I spend my weekends?",
     "category": "preference"},
    {"text": "I drink oolong tea in the afternoon.",
     "probe": "What hot drink do I have in the afternoon?",
     "category": "preference"},
    {"text": "I take my coffee as a single espresso.",
     "probe": "How do I take my coffee?",
     "category": "preference"},
    {"text": "I wind down by sketching in a notebook.",
     "probe": "How do I unwind in the evening?",
     "category": "preference"},
    {"text": "My favorite season is late autumn.",
     "probe": "Which season do I like best?",
     "category": "preference"},
    {"text": "I prefer commuting by bicycle.",
     "probe": "How do I prefer to commute?",
     "category": "preference"},
    {"text": "My go-to comfort food is miso soup.",
     "probe": "What comfort food do I reach for?",
     "category": "preference"},

    {"text": "My current milestone is shipping the alpha release.",
     "probe": "What milestone am I working toward?",
     "category": "project"},
    {"text": "My active workstream is retrieval tuning.",
     "probe": "What workstream am I focused on right now?",
     "category": "project"},
    {"text": "My primary branch this week is feature integration.",
     "probe": "What branch of work do I have this week?",
     "category": "project"},
    {"text": "I keep design notes in a private wiki.",
     "probe": "Where do I store my design notes?",
     "category": "project"},
    {"text": "My code review tool is a self-hosted forge.",
     "probe": "What tool do I use for code review?",
     "category": "project"},
    {"text": "My documentation format is plain markdown.",
     "probe": "What format do I write docs in?",
     "category": "project"},
    {"text": "My build system is a Makefile wrapper.",
     "probe": "What build system do I use?",
     "category": "project"},
    {"text": "My test runner of choice is pytest.",
     "probe": "Which test runner do I rely on?",
     "category": "project"},
    {"text": "My deployment target is a local staging cluster.",
     "probe": "Where do I deploy for staging?",
     "category": "project"},
    {"text": "My monitoring stack is built on Prometheus.",
     "probe": "What monitoring stack do I run?",
     "category": "project"},
    {"text": "My scheduling tool is a paper bullet journal.",
     "probe": "How do I schedule my work?",
     "category": "project"},
    {"text": "My main editor is a terminal-based editor.",
     "probe": "Which editor do I work in?",
     "category": "project"},
    {"text": "My version control workflow is trunk-based.",
     "probe": "What version control workflow do I follow?",
     "category": "project"},
    {"text": "My prototyping framework is a small Python harness.",
     "probe": "How do I prototype new ideas?",
     "category": "project"},
    {"text": "My design-system reference lives in a Figma library.",
     "probe": "Where is my design system?",
     "category": "project"},
    {"text": "My on-call rotation is one week per quarter.",
     "probe": "How often am I on call?",
     "category": "project"},
    {"text": "My documentation review cycle is every two weeks.",
     "probe": "How often do I review my documentation?",
     "category": "project"},

    {"text": "I am allergic to peanuts.",
     "probe": "What am I allergic to?",
     "category": "constraint"},
    {"text": "I avoid raw shellfish for medical reasons.",
     "probe": "What food do I avoid for medical reasons?",
     "category": "constraint"},
    {"text": "My focus block is mornings before noon.",
     "probe": "When is my focus block?",
     "category": "constraint"},
    {"text": "I cannot work on Sundays.",
     "probe": "Which day can I not work?",
     "category": "constraint"},
    {"text": "I need a mechanical keyboard for daily work.",
     "probe": "What hardware do I need for daily work?",
     "category": "constraint"},
    {"text": "I keep my monitor brightness at a low setting.",
     "probe": "How do I set my monitor brightness?",
     "category": "constraint"},
    {"text": "My maximum meeting length is forty-five minutes.",
     "probe": "How long can my meetings run?",
     "category": "constraint"},
    {"text": "I keep a two-day buffer before any deadline.",
     "probe": "How much buffer do I keep before a deadline?",
     "category": "constraint"},
    {"text": "I sleep from eleven at night to seven in the morning.",
     "probe": "What are my sleep hours?",
     "category": "constraint"},
    {"text": "My no-meetings window is Friday afternoon.",
     "probe": "When is my no-meetings window?",
     "category": "constraint"},
    {"text": "I work at most five consecutive days without a break.",
     "probe": "How many days can I work in a row?",
     "category": "constraint"},
    {"text": "I follow a low-sodium dietary restriction.",
     "probe": "What dietary restriction do I follow?",
     "category": "constraint"},
    {"text": "I avoid running on hard pavement due to a knee issue.",
     "probe": "What exercise do I avoid?",
     "category": "constraint"},
    {"text": "I require a sit-stand desk for long work sessions.",
     "probe": "What desk setup do I require?",
     "category": "constraint"},
    {"text": "I work best in quiet rooms below fifty decibels.",
     "probe": "What noise level do I work best in?",
     "category": "constraint"},
    {"text": "I rely on natural daylight for deep work.",
     "probe": "What lighting do I rely on for deep work?",
     "category": "constraint"},
]

FILLER_SENTENCES: list[str] = [
    "The team meets every Tuesday for sync.",
    "Documentation is hosted internally on the wiki.",
    "Quarterly reviews happen at the end of each quarter.",
    "The build pipeline is configured in CI.",
    "Linting runs on every pull request.",
    "Database backups happen nightly.",
    "Monitoring dashboards live in the ops portal.",
    "Onboarding takes about two weeks.",
    "Release notes are published on the changelog page.",
    "Code reviews require at least one approval.",
    "The road map is reviewed monthly.",
    "Customer feedback comes in through the support tracker.",
    "Performance regressions are caught by the perf suite.",
    "Security audits run twice a year.",
    "Vendor renewals are tracked in the procurement spreadsheet.",
    "Incident response follows the documented runbook.",
    "The on-call rotation is one week per engineer.",
    "Demos happen at the all-hands meeting.",
    "The design system is the single source for components.",
    "Translations are handled by an external agency.",
]


@dataclass
class PersonalFact:

    fact_id: str
    text: str
    probe: str
    expects: str
    category: str
    attribute: str


@dataclass
class ProbeOutcome:

    probe_id: str
    seed: int
    cue: str
    expects: str
    category: str
    attribute: str
    recall_at_10_pre: bool
    recall_at_10_post: bool
    top1_pre: str
    top1_post: str
    top1_changed: bool


def _refuse_production_store(store_path: Path) -> None:
    resolved_store = store_path.expanduser().resolve()
    resolved_prod = PRODUCTION_STORE.expanduser().resolve()
    if resolved_store == resolved_prod or resolved_prod in resolved_store.parents:
        print(
            f"[setup-gate] REFUSE: --store-dir resolves to {resolved_store}, "
            f"which is inside production store {resolved_prod}.",
            file=sys.stderr,
        )
        sys.exit(2)


def _gather_env_metadata(store_dir: Path, seed_list: list[int]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent

    def _git(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root)] + cmd, text=True
            ).strip()
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            return "unknown"

    sha = _git(["rev-parse", "--short", "HEAD"])
    dirty = _git(["status", "--porcelain"]) != ""

    def _pkg_version(pkg: str) -> str:
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:  # noqa: BLE001 -- importlib.metadata raises various errors
            return "unknown"

    cpu_brand = "unknown"
    try:
        cpu_brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass

    ram_gb = "unknown"
    try:
        bytes_ = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        ram_gb = f"{bytes_ / (1024**3):.1f}"
    except (subprocess.SubprocessError, OSError, ValueError, FileNotFoundError):
        pass

    embedder_model = os.environ.get("IAI_MCP_EMBED_MODEL", "bge-small-en-v1.5")

    return {
        "cpu_brand": cpu_brand,
        "cpu_cores_physical": os.cpu_count() or "unknown",
        "ram_gb": ram_gb,
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "iai_mcp_git_sha": sha,
        "iai_mcp_git_dirty": dirty,
        "lance_version": _pkg_version("lance"),
        "lancedb_version": _pkg_version("lancedb"),
        "pyarrow_version": _pkg_version("pyarrow"),
        "sentence_transformers_version": _pkg_version("sentence-transformers"),
        "embedder_model": embedder_model,
        "seed_list": seed_list,
        "iai_mcp_store": str(store_dir),
        "wall_clock_start_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_fact_corpus(
    seed: int,
    n_facts: int,
    n_probes: int,
) -> tuple[list[PersonalFact], list[PersonalFact]]:
    rng = random.Random(seed)

    if n_facts > len(FACT_SPECS):
        n_facts = len(FACT_SPECS)

    indices = list(range(len(FACT_SPECS)))
    rng.shuffle(indices)
    picked = [FACT_SPECS[i] for i in indices[:n_facts]]

    facts: list[PersonalFact] = []
    for i, spec in enumerate(picked):
        fact_id = f"f{i:05d}-s{seed}"
        category = spec["category"]
        facts.append(PersonalFact(
            fact_id=fact_id,
            text=spec["text"],
            probe=spec["probe"],
            expects=spec["text"],
            category=category,
            attribute=category,
        ))

    if n_probes > n_facts:
        n_probes = n_facts
    probe_indices = rng.sample(range(n_facts), n_probes)
    probes = [facts[i] for i in sorted(probe_indices)]

    return facts, probes


def _filler_chatter(rng: random.Random, n_turns: int, session_id: str) -> list[str]:
    return [rng.choice(FILLER_SENTENCES) for _ in range(n_turns)]


def run_one_seed(
    seed: int,
    n_facts: int,
    n_probes: int,
    n_intervening_sessions: int,
    n_chatter_turns: int,
    store_dir_for_seed: Path,
    embedder_key: str,
    k_hits: int,
) -> tuple[list[ProbeOutcome], dict[str, Any]]:
    from iai_mcp.embed import Embedder
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import MemoryRecord

    gate = {"insert_ok": False, "snapshot_ok": False, "intervening_ok": False,
            "reprobe_ok": False, "errors": []}

    store_dir_for_seed.mkdir(parents=True, exist_ok=True)
    _env_snapshot = {
        "IAI_MCP_STORE": os.environ.get("IAI_MCP_STORE"),
        "IAI_MCP_CRYPTO_PASSPHRASE": os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"),
    }
    os.environ["IAI_MCP_STORE"] = str(store_dir_for_seed)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE

    def _restore_env() -> None:
        for key, prior in _env_snapshot.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior

    store = MemoryStore(path=store_dir_for_seed / "lancedb")
    embedder = Embedder(model_key=embedder_key)

    _ = embedder.embed_batch(["warm-up " + str(i) for i in range(WARMUP_PASSES)])

    facts, probes = generate_fact_corpus(seed, n_facts, n_probes)
    rng = random.Random(seed * 7919 + 31)

    fact_text_to_id: dict[str, UUID] = {}
    try:
        all_texts = [f.text for f in facts]
        embeddings = embedder.embed_batch(all_texts)
        now = datetime.now(timezone.utc)
        for fact, emb in zip(facts, embeddings):
            rec_id = uuid4()
            rec = MemoryRecord(
                id=rec_id,
                tier="episodic",
                literal_surface=fact.text,
                aaak_index="",
                embedding=list(emb),
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"ts": now.isoformat(), "cue": fact.text[:60],
                             "session_id": f"fact-ingest-s{seed}"}],
                created_at=now,
                updated_at=now,
                tags=["bench-personal-fact-drift", f"category:{fact.category}",
                      f"attribute:{fact.attribute}", f"seed:{seed}"],
                language="en",
            )
            store.insert(rec)
            fact_text_to_id[fact.text] = rec_id
        flush_record_buffer(store)
        gate["insert_ok"] = True
    except Exception as exc:  # noqa: BLE001 -- record failures, don't crash bench
        gate["errors"].append(f"insert: {exc!r}")
        _restore_env()
        return [], gate

    try:
        graph, assignment, rich_club = build_runtime_graph(store)
    except Exception as exc:  # noqa: BLE001 -- ditto
        gate["errors"].append(f"build_runtime_graph_pre: {exc!r}")
        _restore_env()
        return [], gate

    snapshot_top1: dict[str, str] = {}
    snapshot_r10: dict[str, bool] = {}
    try:
        for probe in probes:
            resp = recall_for_benchmark(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=probe.probe, session_id=f"bench-probe-snap-s{seed}",
                k_hits=max(k_hits, CANDIDATE_POOL_SIZE),
                mode="concept",
            )
            hits = list(resp.hits) if hasattr(resp, "hits") else []
            top1 = _hit_surface(hits[0]) if hits else ""
            snapshot_top1[probe.fact_id] = top1
            snapshot_r10[probe.fact_id] = _expected_in_top_k(hits, probe.expects, k_hits)
        gate["snapshot_ok"] = True
    except Exception as exc:  # noqa: BLE001 -- ditto
        gate["errors"].append(f"snapshot: {exc!r}")
        _restore_env()
        return [], gate

    if n_intervening_sessions > 0:
        try:
            iso_state_path = store_dir_for_seed / "lifecycle_state.json"
            iso_log_dir = store_dir_for_seed / "logs"
            iso_log_dir.mkdir(exist_ok=True)
            event_log = LifecycleEventLog(log_dir=iso_log_dir)
            pipeline = SleepPipeline(
                store=store,
                lifecycle_state_path=iso_state_path,
                event_log=event_log,
                quarantine_ttl_hours=0.001,
            )
            per_session = max(1, (n_chatter_turns + n_intervening_sessions - 1)
                              // max(n_intervening_sessions, 1))
            now2 = datetime.now(timezone.utc)
            for sess_idx in range(n_intervening_sessions):
                session_id = f"inter-s{seed}-i{sess_idx:03d}"
                chatter = _filler_chatter(rng, per_session, session_id)
                chat_embs = embedder.embed_batch(chatter)
                for text, emb in zip(chatter, chat_embs):
                    rec = MemoryRecord(
                        id=uuid4(),
                        tier="episodic",
                        literal_surface=text,
                        aaak_index="",
                        embedding=list(emb),
                        community_id=None,
                        centrality=0.0,
                        detail_level=1,
                        pinned=False,
                        stability=0.0,
                        difficulty=0.0,
                        last_reviewed=None,
                        never_decay=False,
                        never_merge=False,
                        provenance=[{"ts": now2.isoformat(), "cue": text[:60],
                                     "session_id": session_id}],
                        created_at=now2,
                        updated_at=now2,
                        tags=["bench-chatter", f"seed:{seed}",
                              f"intervening:{sess_idx}"],
                        language="en",
                    )
                    store.insert(rec)
                result = pipeline.force_run()
                if result.get("failed_step"):
                    gate["errors"].append(
                        f"sleep_pipeline failed_step={result.get('failed_step')} "
                        f"sess={sess_idx}"
                    )
            flush_record_buffer(store)
            gate["intervening_ok"] = True
        except Exception as exc:  # noqa: BLE001 -- ditto
            gate["errors"].append(f"intervening: {exc!r}")
            _restore_env()
            return [], gate
    else:
        gate["intervening_ok"] = True

    try:
        graph, assignment, rich_club = build_runtime_graph(store)
    except Exception as exc:  # noqa: BLE001 -- ditto
        gate["errors"].append(f"build_runtime_graph_post: {exc!r}")
        _restore_env()
        return [], gate

    outcomes: list[ProbeOutcome] = []
    try:
        for probe in probes:
            resp = recall_for_benchmark(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=probe.probe, session_id=f"bench-probe-reprobe-s{seed}",
                k_hits=max(k_hits, CANDIDATE_POOL_SIZE),
                mode="concept",
            )
            hits = list(resp.hits) if hasattr(resp, "hits") else []
            top1_post = _hit_surface(hits[0]) if hits else ""
            top1_pre = snapshot_top1.get(probe.fact_id, "")
            r10_post = _expected_in_top_k(hits, probe.expects, k_hits)
            outcomes.append(ProbeOutcome(
                probe_id=probe.fact_id,
                seed=seed,
                cue=probe.probe,
                expects=probe.expects,
                category=probe.category,
                attribute=probe.attribute,
                recall_at_10_pre=snapshot_r10.get(probe.fact_id, False),
                recall_at_10_post=r10_post,
                top1_pre=top1_pre,
                top1_post=top1_post,
                top1_changed=(top1_pre != top1_post),
            ))
        gate["reprobe_ok"] = True
    except Exception as exc:  # noqa: BLE001 -- ditto
        gate["errors"].append(f"reprobe: {exc!r}")

    _restore_env()
    return outcomes, gate


def _hit_surface(hit: Any) -> str:
    if hasattr(hit, "literal_surface"):
        return str(hit.literal_surface)
    if isinstance(hit, dict):
        return str(hit.get("literal_surface", ""))
    return ""


def _expected_in_top_k(hits: list, expected_text: str, k: int) -> bool:
    for hit in hits[:k]:
        surface = _hit_surface(hit)
        if surface and surface == expected_text:
            return True
    return False


def _hit_id(hit: Any) -> str:
    if hasattr(hit, "id"):
        return str(hit.id)
    if isinstance(hit, dict):
        return str(hit.get("id", ""))
    return ""


def _expected_id_in_top_k(hits: list, expected_id: str, k: int) -> bool:
    for hit in hits[:k]:
        if _hit_id(hit) == expected_id:
            return True
    return False


def _compute_recall_at_10(probe_results: list[dict]) -> float:
    if not probe_results:
        return 0.0
    hit = sum(1 for r in probe_results if r.get("recall_at_10_post"))
    return hit / len(probe_results)


def _compute_retention_loss_at_10(probe_results: list[dict]) -> float:
    if not probe_results:
        return 0.0
    pre_hit = sum(1 for r in probe_results if r.get("recall_at_10_pre"))
    post_hit = sum(1 for r in probe_results if r.get("recall_at_10_post"))
    n = len(probe_results)
    return (pre_hit - post_hit) / n


def _outcomes_to_dicts(outcomes: list[ProbeOutcome]) -> list[dict]:
    out = []
    for o in outcomes:
        out.append({
            "probe_id": o.probe_id,
            "seed": o.seed,
            "cue": o.cue,
            "expects": o.expects,
            "category": o.category,
            "attribute": o.attribute,
            "recall_at_10_pre": o.recall_at_10_pre,
            "recall_at_10_post": o.recall_at_10_post,
            "top1_pre": o.top1_pre,
            "top1_post": o.top1_post,
            "top1_changed": o.top1_changed,
        })
    return out


def aggregate(
    per_seed_outcomes: dict[int, list[ProbeOutcome]],
) -> dict[str, Any]:
    per_seed_rows = []
    flat_probes: list[dict] = []
    r10_values: list[float] = []
    loss_values: list[float] = []

    for seed in sorted(per_seed_outcomes.keys()):
        outcomes = per_seed_outcomes[seed]
        dicts = _outcomes_to_dicts(outcomes)
        flat_probes.extend(dicts)
        r10 = _compute_recall_at_10(dicts)
        loss = _compute_retention_loss_at_10(dicts)
        per_seed_rows.append({
            "seed": seed,
            "recall_at_10": round(r10, 6),
            "retention_loss_at_10": round(loss, 6),
            "n_probes": len(outcomes),
        })
        r10_values.append(r10)
        loss_values.append(loss)

    mean_r10 = sum(r10_values) / len(r10_values) if r10_values else 0.0
    mean_loss = sum(loss_values) / len(loss_values) if loss_values else 0.0
    passed = (
        mean_r10 >= SHIP_GATE_RECALL_AT_10
        and mean_loss < SHIP_GATE_RETENTION_LOSS_CEILING
    )

    return {
        "recall_at_10": round(mean_r10, 6),
        "retention_loss_at_10": round(mean_loss, 6),
        "per_seed": per_seed_rows,
        "per_probe": flat_probes,
        "ship_gate": {
            "recall_at_10_threshold": SHIP_GATE_RECALL_AT_10,
            "retention_loss_ceiling": SHIP_GATE_RETENTION_LOSS_CEILING,
            "passed": passed,
        },
    }


def write_outputs(
    output_dir: Path,
    run_id: str,
    summary: dict[str, Any],
    env: dict[str, Any],
    duration_seconds: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"personal_fact_drift_{run_id}.json"
    env_with_duration = {**env, "wall_clock_duration_seconds": round(duration_seconds, 2)}
    blob = {"env": env_with_duration, "summary": summary}
    json_path.write_text(json.dumps(blob, indent=2, default=str))
    print(f"[bench] Wrote: {json_path}")
    return json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Personal-fact-drift bench (VAL-03)",
    )
    parser.add_argument("--scale", choices=list(SCALE_PRESETS.keys()), default="smoke")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedder", default="bge-small-en-v1.5")
    parser.add_argument("--k-hits", type=int, default=DEFAULT_K)
    args = parser.parse_args(argv)

    if len(args.seeds) < 3:
        print(
            f"[setup-gate] REFUSE: need >=3 seeds, got {len(args.seeds)}.",
            file=sys.stderr,
        )
        return 2

    store_dir = Path(args.store_dir).expanduser().resolve()
    _refuse_production_store(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).expanduser().resolve()

    n_facts, n_probes, n_intervening, n_chatter = SCALE_PRESETS[args.scale]

    env = _gather_env_metadata(store_dir, args.seeds)
    env.update({
        "scale": args.scale,
        "n_facts": n_facts,
        "n_probes": n_probes,
        "n_intervening_sessions": n_intervening,
        "n_chatter_turns": n_chatter,
        "k_hits": args.k_hits,
        "embedder_model": args.embedder,
        "ship_gate_recall_at_10": SHIP_GATE_RECALL_AT_10,
        "ship_gate_retention_loss_ceiling": SHIP_GATE_RETENTION_LOSS_CEILING,
    })

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{env['iai_mcp_git_sha']}"
        f"-seeds{'-'.join(map(str, args.seeds))}"
        f"-scale_{args.scale}"
    )

    print(f"[bench] Run ID: {run_id}")
    print(f"[bench] Scale: {args.scale}  ({n_facts} facts, {n_probes} probes, "
          f"{n_intervening} intervening sessions, {n_chatter} chatter turns)")
    print(f"[bench] Seeds: {args.seeds}")
    print(f"[bench] Store: {store_dir}")
    print(f"[bench] git: {env['iai_mcp_git_sha']} "
          f"({'dirty' if env['iai_mcp_git_dirty'] else 'clean'})")

    t0 = time.time()
    per_seed_outcomes: dict[int, list[ProbeOutcome]] = {}
    cell_errors: list[str] = []

    for seed in args.seeds:
        cell_dir = store_dir / f"seed{seed}"
        print(f"[bench]   seed={seed}  (n_facts={n_facts} n_probes={n_probes} "
              f"n_intervening={n_intervening})  ...", flush=True)
        cell_t0 = time.time()
        outcomes, gate = run_one_seed(
            seed=seed,
            n_facts=n_facts,
            n_probes=n_probes,
            n_intervening_sessions=n_intervening,
            n_chatter_turns=n_chatter,
            store_dir_for_seed=cell_dir,
            embedder_key=args.embedder,
            k_hits=args.k_hits,
        )
        cell_dur = time.time() - cell_t0
        if gate["errors"]:
            cell_errors.extend([f"seed{seed}: {e}" for e in gate["errors"]])
            print(f"[bench]     seed={seed} ERRORS: {gate['errors']}", flush=True)
        per_seed_outcomes[seed] = outcomes
        print(f"[bench]     seed={seed} done in {cell_dur:.1f}s "
              f"(outcomes={len(outcomes)})", flush=True)

    duration = time.time() - t0
    summary = aggregate(per_seed_outcomes)
    summary["cell_errors"] = cell_errors

    json_path = write_outputs(output_dir, run_id, summary, env, duration)

    print(f"[bench] DONE  recall_at_10={summary['recall_at_10']:.4f}  "
          f"retention_loss_at_10={summary['retention_loss_at_10']:.4f}  "
          f"gate_passed={summary['ship_gate']['passed']}")
    print(f"[bench] Duration: {duration:.1f}s")

    if args.scale == "smoke":
        return 0

    return 0 if summary["ship_gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
