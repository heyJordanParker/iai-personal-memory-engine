from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Callable

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

from bench.tokens import (
    _anthropic_count_tokens,
    _char4_count,
    _tiktoken_count,
)


_ADAPTER_TIMEOUT_SECONDS = 30


def _log_adapter_unavailable(tool: str, reason: str) -> None:
    line = json.dumps({
        "event": "bench_adapter_unavailable",
        "tool": tool,
        "reason": reason,
    })
    print(line, file=sys.stderr)


def _run_subprocess_adapter(
    *,
    tool_name: str,
    cli_name: str,
    argv_template: Callable[[str], list[str]],
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    exe = shutil.which(cli_name)
    if exe is None:
        _log_adapter_unavailable(tool_name, "cli_not_found")
        return None

    total = 0
    for turn in script:
        argv = [exe, *argv_template(turn["input"])[1:]]
        try:
            proc = subprocess.run(
                argv,
                timeout=_ADAPTER_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _log_adapter_unavailable(tool_name, f"timeout: {exc}")
            return None
        except (OSError, ValueError) as exc:
            _log_adapter_unavailable(tool_name, f"subprocess_error: {exc}")
            return None

        if proc.returncode != 0:
            _log_adapter_unavailable(
                tool_name,
                f"non_zero_exit={proc.returncode} stderr={proc.stderr[:200]!r}",
            )
            return None

        stdout = proc.stdout or ""
        total += int(counter(stdout))

    return total


def _run_mempalace_adapter(
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    return _run_subprocess_adapter(
        tool_name="mempalace",
        cli_name="mempalace",
        argv_template=lambda text: ["mempalace", "search", text],
        script=script,
        counter=counter,
    )


def _run_claude_mem_adapter(
    script: list[dict],
    counter: Callable[[str], int],
) -> int | None:
    return _run_subprocess_adapter(
        tool_name="claude-mem",
        cli_name="claude-mem",
        argv_template=lambda text: ["claude-mem", "recall", text],
        script=script,
        counter=counter,
    )


SCRIPT_NAME = "session-cost-v1"

_SCRIPT: list[dict] = [
    {
        "kind": "recall",
        "input": "Tell me the decisions we made about the storage architecture",
    },
    {
        "kind": "chat",
        "input": "Let me iterate on this function; no recall needed here",
    },
    {
        "kind": "recall",
        "input": "What did I say about bench discipline?",
    },
    {
        "kind": "recall_cross_community",
        "input": "What is the connection between the formality knob and the autistic kernel?",
    },
    {
        "kind": "save",
        "input": "Decision locked: use cachetools TTLCache for the session cache LRU",
    },
    {
        "kind": "introspect",
        "input": "profile_get_set operation=get knob=wake_depth",
    },
    {
        "kind": "chat",
        "input": "Continuing this refactor; still no recall",
    },
    {
        "kind": "recall",
        "input": "alice said something about pressplay cross-validation",
    },
    {
        "kind": "reinforce",
        "input": "memory_reinforce the last 3 hits",
    },
    {
        "kind": "introspect",
        "input": "events_query kind=first_turn_recall limit=5",
    },
]


_POST_TOK15_TOOL_DESCRIPTIONS = "\n".join([
    "Recall verbatim memories matching cue. Returns hits + anti_hits.",
    "Structural recall over role->filler bindings. Returns hits.",
    "Boost Hebbian edges among co-retrieved record ids.",
    "Mark a record contradicted; new fact stored as new record.",
    "Trigger memory consolidation.",
    "Read or write a profile knob (15 sealed). operation: get|set.",
    "List pending curiosity questions. Optional session_id filter.",
    "List induced schemas. Optional domain + confidence_min filters.",
    "Query user-visible events by kind, since, severity, limit.",
    "Topology snapshot: N, C, L, sigma, community_count, regime.",
    "Camouflaging detection status; window_size weekly points.",
])

_RESULT_BODIES: dict[str, str] = {
    "recall": (
        "hits=[{record_id, literal_surface, score}] "
        "anti_hits=[{record_id, reason}] "
        "activation_trace=[community_gate, spread, rank] "
        "budget_used=200"
    ),
    "save": "ok=true id=<uuid>",
    "introspect": '{"value": "minimal"}',
    "reinforce": "ok=true edges_boosted=3",
    "chat": "",
    "recall_cross_community": (
        "hits=[{record_id, literal_surface, score, community_id}] "
        "anti_hits=[] activation_trace=[cross_community_spread] "
        "budget_used=350"
    ),
}


def _select_counter(
    count_tokens_fn: Callable[[str], int] | None = None,
) -> tuple[Callable[[str], int], str]:
    if count_tokens_fn is not None:
        return count_tokens_fn, "injected"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _anthropic_count_tokens, "anthropic-count-tokens"
    try:
        import tiktoken  # noqa: F401
        return _tiktoken_count, "tiktoken-cl100k-proxy"
    except ImportError:
        return _char4_count, "heuristic-char4"


def _session_start_overhead_tokens(wake_depth: str) -> int:
    if wake_depth == "minimal":
        return 24
    if wake_depth == "standard":
        return 1388
    return 2000


def _simulate_turn(
    turn: dict,
    counter: Callable[[str], int],
) -> int:
    parts: list[str] = [
        _POST_TOK15_TOOL_DESCRIPTIONS,
        turn["input"],
        _RESULT_BODIES.get(turn["kind"], ""),
    ]
    return int(counter("\n".join(p for p in parts if p)))


def run_total_session_cost(
    *,
    wake_depth: str = "minimal",
    mempalace_ref: int | None = None,
    claude_mem_ref: int | None = None,
    measure_mempalace: bool = False,
    measure_claude_mem: bool = False,
    count_tokens_fn: Callable[[str], int] | None = None,
) -> dict:
    counter, mode = _select_counter(count_tokens_fn)

    per_turn: list[int] = []
    for i, turn in enumerate(_SCRIPT):
        t = _simulate_turn(turn, counter)
        if i == 0:
            t += _session_start_overhead_tokens(wake_depth)
        per_turn.append(int(t))

    total = int(sum(per_turn))

    refs: dict[str, int] = {}
    passed = True

    mp_measured: int | None = None
    cm_measured: int | None = None
    if measure_mempalace:
        mp_measured = _run_mempalace_adapter(_SCRIPT, counter)
        if mp_measured is not None:
            refs["mempalace_measured"] = int(mp_measured)
    if measure_claude_mem:
        cm_measured = _run_claude_mem_adapter(_SCRIPT, counter)
        if cm_measured is not None:
            refs["claude_mem_measured"] = int(cm_measured)

    if mempalace_ref is not None:
        key = "mempalace_manual" if mp_measured is not None else "mempalace"
        refs[key] = int(mempalace_ref)
    if claude_mem_ref is not None:
        key = "claude_mem_manual" if cm_measured is not None else "claude_mem"
        refs[key] = int(claude_mem_ref)

    mp_gate = refs.get(
        "mempalace_measured", refs.get("mempalace", refs.get("mempalace_manual"))
    )
    cm_gate = refs.get(
        "claude_mem_measured", refs.get("claude_mem", refs.get("claude_mem_manual"))
    )
    if mp_gate is not None and total > mp_gate:
        passed = False
    if cm_gate is not None and total > cm_gate:
        passed = False

    return {
        "adapter": "iai-mcp",
        "wake_depth": wake_depth,
        "total_tokens": total,
        "per_turn": per_turn,
        "mode": mode,
        "refs": refs,
        "passed": passed,
        "script_name": SCRIPT_NAME,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.total_session_cost",
        description=(
            "Total session cost bench. Fixed 10-turn "
            "representative script; measures token cost "
            "at wake_depth minimal|standard|deep and optionally compares "
            "to supplied mempalace / claude-mem reference totals."
        ),
    )
    parser.add_argument(
        "--wake-depth",
        choices=("minimal", "standard", "deep"),
        default="minimal",
        help="session-start payload size (default minimal)",
    )
    parser.add_argument(
        "--ref-mempalace",
        dest="mempalace_ref",
        type=int, default=None,
        help="mempalace reference total (tokens) for the comparative gate",
    )
    parser.add_argument(
        "--ref-claude-mem",
        dest="claude_mem_ref",
        type=int, default=None,
        help="claude-mem reference total (tokens) for the comparative gate",
    )
    parser.add_argument(
        "--measure-mempalace",
        action="store_true",
        help=(
            "attempt a live mempalace subprocess run to fill the "
            "reference column; on failure emits a bench_adapter_unavailable "
            "stderr event and records no measurement"
        ),
    )
    parser.add_argument(
        "--measure-claude-mem",
        action="store_true",
        help=(
            "attempt a live claude-mem subprocess run; identical fallback "
            "shape to --measure-mempalace"
        ),
    )
    args = parser.parse_args(argv)

    result = run_total_session_cost(
        wake_depth=args.wake_depth,
        mempalace_ref=args.mempalace_ref,
        claude_mem_ref=args.claude_mem_ref,
        measure_mempalace=args.measure_mempalace,
        measure_claude_mem=args.measure_claude_mem,
    )
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
