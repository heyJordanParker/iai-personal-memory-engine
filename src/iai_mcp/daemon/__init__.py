from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import json
import logging
import os
import resource
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

from iai_mcp import s4
from iai_mcp.concurrency import serve_control_socket  # noqa: F401 -- re-exported here for the test suite; the function lives in concurrency.py
from iai_mcp.daemon_state import load_state, save_state
from iai_mcp.dream import run_rem_cycle
from iai_mcp.events import (
    CRISIS_MODE_AUTO_EXPIRED,
    DAEMON_MEMORY_PRESSURE_KILL,
    DAEMON_SLEEP_CYCLE_STALE,
    DAEMON_WATCHDOG_NEEDS_OPERATOR,
    DAEMON_WEDGE_KILL,
    write_event,
)
from iai_mcp.identity_audit import continuous_audit
from iai_mcp.quiet_window import (
    BUCKET_COUNT,
    BUCKET_MINUTES,
    learn_quiet_window,
    should_bootstrap_trigger,
    should_relearn,
)
from iai_mcp.hippo import AccessMode
from iai_mcp.lock_protocol import cleanup_stale_consolidation_intent
from iai_mcp.native_guard import _require_native
from iai_mcp.sleep_wal import SleepWAL
from iai_mcp.socket_server import SocketServer
from iai_mcp.store import MemoryStore
from iai_mcp.tz import load_user_tz


STATE_WAKE: str = "WAKE"
STATE_TRANSITIONING: str = "TRANSITIONING"
STATE_SLEEP: str = "SLEEP"
STATE_DREAMING: str = "DREAMING"

VALID_TRANSITIONS: dict[str, set[str]] = {
    STATE_WAKE: {STATE_TRANSITIONING},
    STATE_TRANSITIONING: {STATE_SLEEP, STATE_WAKE},
    STATE_SLEEP: {STATE_DREAMING, STATE_WAKE},
    STATE_DREAMING: {STATE_SLEEP},
}

TICK_INTERVAL_SEC: int = 30

DEFAULT_CYCLE_COUNT: int = 4

S4_OFFLINE_INTERVAL_SEC: int = 60 * 60

S4_FIRST_ITER_GRACE_SEC: float = float(
    os.environ.get("IAI_MCP_S4_FIRST_ITER_GRACE_SEC", str(S4_OFFLINE_INTERVAL_SEC)),
)

SESSION_START_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"
from iai_mcp.session import SESSION_START_CACHE_MAX_CHARS  # noqa: E402 -- placed after PATH constant for readability

INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC: float = 30.0


def _hippo_health_check_on_boot(store) -> dict[str, int | str]:
    try:
        db = store.db
        sqlite_count_row = db._conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NULL"
        ).fetchone()
        sqlite_count = int(sqlite_count_row[0]) if sqlite_count_row else 0
    except Exception as exc:
        return {
            "sqlite_count": -1,
            "hnsw_active_count": -1,
            "hnsw_raw_count": -1,
            "action": "sqlite_count_failed",
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    try:
        active_label_count = int(len(db._label_map))
    except Exception:
        active_label_count = -1
    try:
        hnsw_raw_count = int(db._hnsw.get_current_count())
    except Exception:
        hnsw_raw_count = -1
    action = (
        "ok"
        if sqlite_count == active_label_count
        else "divergence_at_boot"
    )
    return {
        "sqlite_count": sqlite_count,
        "hnsw_active_count": active_label_count,
        "hnsw_raw_count": hnsw_raw_count,
        "action": action,
    }


_DAEMON_NOFILE_FLOOR_DEFAULT: int = 8192


def _raise_fd_limit() -> None:
    try:
        floor = int(
            os.environ.get("IAI_MCP_DAEMON_NOFILE_FLOOR", _DAEMON_NOFILE_FLOOR_DEFAULT)
        )
    except (TypeError, ValueError):
        floor = _DAEMON_NOFILE_FLOOR_DEFAULT

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return

    effective_hard = hard if hard != resource.RLIM_INFINITY else floor

    target = min(max(soft, floor), effective_hard)
    if target <= soft:
        return

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        log.debug("daemon_fd_limit_raised soft=%d->%d hard=%d", soft, target, hard)
    except (OSError, ValueError) as exc:
        log.debug("daemon_fd_limit_raise failed (non-fatal): %s", exc)


def _should_drain_on_drowsy_edge(prev, current) -> bool:
    from iai_mcp.lifecycle_state import LifecycleState as _L
    return prev is _L.WAKE and current is _L.DROWSY


def _run_drowsy_drain(store, *, drain_fn, write_event_fn) -> None:
    try:
        result = drain_fn(store)
    except Exception as e:  # noqa: BLE001 -- lifecycle_tick MUST NOT crash
        log.warning("drowsy drain failed: %s", e, exc_info=True)
        try:
            write_event_fn(
                store,
                "deferred_drain_failed",
                {"error": str(e)[:200], "phase": "drowsy"},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write deferred_drain_failed event: %s", e)
        return
    if not isinstance(result, dict):
        return
    if result.get("files_drained") or result.get("files_failed"):
        try:
            write_event_fn(
                store,
                "deferred_drain_drowsy",
                result,
                severity="info",
            )
        except Exception:  # noqa: BLE001 -- event write non-critical
            log.debug("failed to write deferred_drain_drowsy event")


def _kick_drowsy_rgc_rebuild(store) -> None:
    import threading as _threading

    def _bg() -> None:
        try:
            import iai_mcp.runtime_graph_cache as _rgc
            _rgc._rebuild_and_save_rgc(store)
        except Exception:  # noqa: BLE001 -- best-effort; cache stays cold on failure
            log.debug("drowsy-edge graph-cache rebuild failed", exc_info=True)
        finally:
            try:
                import iai_mcp.runtime_graph_cache as _rgc
                _rgc.rebuild_ready.set()
            except Exception:  # noqa: BLE001
                log.debug("rebuild_ready.set() failed", exc_info=True)

    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _rgc.rebuild_ready.clear()
    except Exception:  # noqa: BLE001
        log.debug("rebuild_ready.clear() failed", exc_info=True)

    _threading.Thread(target=_bg, daemon=True).start()


def _wake_hook_rebuild_if_cold(store) -> None:
    try:
        import iai_mcp.runtime_graph_cache as _rgc
        _, _, _, _src = _rgc.load_recall_structural(store)
        if _src in ("cold_degrade", "last_good"):
            # This site only fires when the cache is already cold, so the gate's
            # own coldness term would rebuild here anyway; force makes the intent
            # explicit at the wake-if-cold edge.
            _rgc._rebuild_and_save_rgc(store, force=True)
    except Exception:  # noqa: BLE001 -- best-effort, never crash the wake hook
        log.debug("wake-hook graph-cache rebuild failed", exc_info=True)


def transition(state: dict, new_fsm: str) -> None:
    current = state.get("fsm_state", STATE_WAKE)
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_fsm not in allowed:
        raise ValueError(
            f"Illegal transition {current} -> {new_fsm}; allowed: {sorted(allowed)}"
        )
    state["fsm_state"] = new_fsm
    state["fsm_transition_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


def _store_is_empty(store: MemoryStore) -> bool:
    try:
        return store.db.open_table("records").count_rows() == 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        log.debug("store empty check failed, assuming empty: %s", exc)
        return True


def _is_inside_window(
    window: tuple[int, int] | list | None,
    now: datetime,
    tz,
) -> bool:
    if not window:
        return False
    try:
        start, duration = int(window[0]), int(window[1])
    except (TypeError, ValueError, IndexError):
        return False
    if duration <= 0:
        return False
    now_local = now.astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // BUCKET_MINUTES
    end = (start + duration) % BUCKET_COUNT
    if start < end:
        return start <= cur_bucket < end
    return cur_bucket >= start or cur_bucket < end


def _update_pending_digest(state: dict, cycle_result: dict) -> None:
    digest = state.get("pending_digest") or {
        "rem_cycles_completed": 0,
        "episodes_processed": 0,
        "schemas_induced_tier0": 0,
        "claude_call_used": False,
        "main_insight_text": None,
        "timed_out_cycles": 0,
    }
    digest["rem_cycles_completed"] = int(digest.get("rem_cycles_completed", 0)) + 1
    digest["episodes_processed"] = int(digest.get("episodes_processed", 0)) + int(
        cycle_result.get("summaries_created", 0) or 0
    )
    digest["schemas_induced_tier0"] = int(digest.get("schemas_induced_tier0", 0)) + int(
        cycle_result.get("schema_candidates", 0) or 0
    )
    if cycle_result.get("claude_call_used"):
        digest["claude_call_used"] = True
        digest["main_insight_text"] = cycle_result.get("main_insight_text")
    if cycle_result.get("timed_out"):
        digest["timed_out_cycles"] = int(digest.get("timed_out_cycles", 0)) + 1
    state["pending_digest"] = digest


def _write_session_start_cache(store, *, cache_path: Path = SESSION_START_CACHE_PATH) -> None:
    try:
        from iai_mcp import retrieve
        from iai_mcp.session import (
            _compose_session_start_payload,
            format_payload_as_markdown,
        )

        _graph, assignment, rc = retrieve.build_runtime_graph(store)
        payload = _compose_session_start_payload(
            store,
            assignment,
            rc,
            session_id="precache",
            profile_state={"wake_depth": "standard"},
        )
        rendered = format_payload_as_markdown(payload)
        if not rendered:
            return
        if len(rendered) > SESSION_START_CACHE_MAX_CHARS:
            rendered = rendered[:SESSION_START_CACHE_MAX_CHARS]

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(rendered)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, cache_path)
    except Exception as exc:  # noqa: BLE001 -- cache write MUST NOT crash the REM loop
        log.warning("session start cache write failed: %s", exc, exc_info=True)
        try:
            write_event(
                store,
                "session_start_cache_write_failed",
                {"error": str(exc)[:200]},
                severity="warning",
            )
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("failed to write session_start_cache_write_failed event")


async def _tick_body(
    store: MemoryStore,
    state: dict,
    *,
    mcp_socket: SocketServer | None = None,
) -> None:
    try:
        from iai_mcp.daemon_state import (
            FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
            prune_first_turn_pending,
        )

        state, dropped = prune_first_turn_pending(
            state, now=datetime.now(timezone.utc),
        )
        if dropped:
            try:
                await asyncio.to_thread(save_state, state)
            except (OSError, ValueError) as exc:  # noqa: BLE001 -- state save non-critical
                log.debug("save_state after prune failed: %s", exc)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "first_turn_pending_expired",
                    {
                        "dropped_count": len(dropped),
                        "session_ids": dropped,
                        "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                        "phase": "tick",
                    },
                    severity="info",
                )
            except (OSError, RuntimeError) as exc:  # noqa: BLE001 -- event write non-critical
                log.debug("first_turn_pending_expired event write failed: %s", exc)
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.warning("tick step 0.5 (prune first_turn_pending) failed", exc_info=True)

    try:
        _s4bg_ts = state.get("_last_s4bg_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_s4bg = not _s4bg_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_s4bg_ts)
        ).total_seconds() > 3600
        if _should_s4bg:
            from iai_mcp.s4 import s4_background_scan
            await asyncio.to_thread(s4_background_scan, store, 50)
            state["_last_s4bg_ts"] = _now_iso
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.6 (s4_background_scan) failed", exc_info=True)

    try:
        _forage_ts = state.get("_last_forage_ts", "")
        _now_iso = datetime.now(timezone.utc).isoformat()
        _should_forage = not _forage_ts or (
            datetime.fromisoformat(_now_iso) - datetime.fromisoformat(_forage_ts)
        ).total_seconds() > 3600
        if _should_forage:
            _skip_foraging_in_sleep = False
            try:
                from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, LifecycleState, load_state as _load_ls
                _ls_rec = await asyncio.to_thread(_load_ls, LIFECYCLE_STATE_PATH)
                _ls_current = _ls_rec.get("current_state", "")
                if _ls_current == LifecycleState.SLEEP.value:
                    _skip_foraging_in_sleep = True
            except Exception:
                _skip_foraging_in_sleep = True
            if not _skip_foraging_in_sleep:
                from iai_mcp.foraging import forage_for_connections
                _foraged = await asyncio.to_thread(forage_for_connections, store, 3)
                state["_last_forage_ts"] = _now_iso
                if _foraged > 0:
                    await asyncio.to_thread(
                        write_event, store, "self_foraging_pass",
                        {"edges_created": _foraged}, severity="info",
                    )
            else:
                log.debug("tick step 0.7 (foraging) skipped: canonical FSM in SLEEP")
    except Exception:  # noqa: BLE001 -- tick step MUST NOT crash
        log.debug("tick step 0.7 (foraging) failed", exc_info=True)

    try:
        from iai_mcp.events import (
            _last_flush_at,
            flush_event_buffer,
            should_flush_by_time,
        )

        if should_flush_by_time(id(store), _last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_event_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("events buffer periodic flush skipped: %s", str(e)[:120])

    try:
        from iai_mcp.store import (
            _record_last_flush_at,
            flush_record_buffer,
            should_flush_record_buffer_by_time,
        )

        if should_flush_record_buffer_by_time(id(store), _record_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_record_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("records buffer periodic flush skipped: %s", str(e)[:120])

    try:
        from iai_mcp.store import (
            _edge_last_flush_at,
            flush_edge_buffer,
            should_flush_edge_buffer_by_time,
        )

        if should_flush_edge_buffer_by_time(id(store), _edge_last_flush_at.get(id(store))):
            await asyncio.to_thread(flush_edge_buffer, store)
    except Exception as e:  # noqa: BLE001 -- periodic flush MUST NOT crash tick
        log.debug("edges buffer periodic flush skipped: %s", str(e)[:120])


    if state.get("scheduler_paused") is True:
        try:
            await asyncio.to_thread(
                write_event,
                store,
                "daemon_tick_skipped",
                {"reason": "paused"},
                severity="info",
            )
        except (OSError, RuntimeError) as exc:
            log.debug("daemon_tick_skipped event write failed: %s", exc)
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "paused"
        try:
            await asyncio.to_thread(save_state, state)
        except (OSError, ValueError) as exc:
            log.debug("save_state (paused) failed: %s", exc)
        return

    if await asyncio.to_thread(_store_is_empty, store):
        state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
        state["last_tick_skipped_reason"] = "empty_store"
        await asyncio.to_thread(save_state, state)
        return

    now = datetime.now(timezone.utc)
    try:
        tz = load_user_tz()
    except (OSError, ValueError, KeyError) as exc:
        log.debug("load_user_tz failed, using UTC: %s", exc)
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("UTC")

    last_learned_raw = state.get("quiet_window_learned_at")
    last_learned_dt: datetime | None = None
    if last_learned_raw:
        try:
            last_learned_dt = datetime.fromisoformat(last_learned_raw)
        except (TypeError, ValueError):
            last_learned_dt = None
    if should_relearn(last_learned_dt, now):
        try:
            window = await asyncio.to_thread(learn_quiet_window, store, now, tz)
        except (OSError, ValueError, RuntimeError) as exc:
            log.debug("learn_quiet_window failed: %s", exc)
            window = None
        state["quiet_window"] = list(window) if window else None
        state["quiet_window_learned_at"] = now.isoformat()
        await asyncio.to_thread(save_state, state)


    state["last_tick_at"] = datetime.now(timezone.utc).isoformat()
    try:
        await asyncio.to_thread(save_state, state)
    except (OSError, ValueError) as exc:
        log.debug("save_state after tick failed: %s", exc)


async def _scheduler_tick(
    store: MemoryStore,
    state: dict,
    *,
    tick_body: Callable[..., Awaitable[None]] | None = None,
    mcp_socket: SocketServer | None = None,
) -> None:
    body = tick_body or _tick_body
    while True:
        try:
            await body(store, state, mcp_socket=mcp_socket)
        except TypeError:
            try:
                await body(store, state)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001 -- daemon tick boundary
                log.warning("tick failed (legacy body): %s", exc, exc_info=True)
                try:
                    write_event(
                        store,
                        "tick_error",
                        {"error": str(exc), "type": type(exc).__name__},
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- event write inside boundary guard
                    log.debug("tick_error event write failed")
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 -- daemon must never die mid-tick
            log.warning("tick failed: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "tick_error",
                    {"error": str(exc), "type": type(exc).__name__},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("tick_error event write failed")
        try:
            await asyncio.sleep(TICK_INTERVAL_SEC)
        except asyncio.CancelledError:
            break


async def _s4_offline_loop(store: MemoryStore, shutdown: asyncio.Event) -> None:
    if S4_FIRST_ITER_GRACE_SEC > 0:
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_FIRST_ITER_GRACE_SEC
            )
            return
        except asyncio.TimeoutError:
            pass
    while not shutdown.is_set():
        try:
            await asyncio.to_thread(s4.run_offline_pass, store)
        except Exception as exc:  # noqa: BLE001 -- never die on offline-pass failure
            log.warning("S4 offline pass failed: %s", exc, exc_info=True)
            try:
                await asyncio.to_thread(
                    write_event,
                    store,
                    "s4_offline_pass_error",
                    {"error": str(exc)[:500]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("s4_offline_pass_error event write failed")
        try:
            await asyncio.wait_for(
                shutdown.wait(), timeout=S4_OFFLINE_INTERVAL_SEC
            )
            break
        except asyncio.TimeoutError:
            continue


from iai_mcp.daemon_config import (  # noqa: E402
    ErasureConfig,
    _load_erasure_config,
    PatSepConfig,
    _load_patsep_config,
    S2Config,
    _load_s2_config,
    SleepOverhaulConfig,
    _load_sleep_overhaul_config,
    ReconsolidationConfig,
    _load_reconsolidation_config,
    StcConfig,
    _load_stc_config,
    UserModelConfig,
    _load_user_model_config,
    SpatialConfig,
    _load_spatial_config,
    DmnConfig,
    _load_dmn_config,
    PaskConfig,
    _load_pask_config,
)


_USER_SHUTDOWN_FLAG = "user_requested_shutdown"


def _clear_user_shutdown_sentinel(state: dict) -> None:
    try:
        on_disk = load_state()
        if _USER_SHUTDOWN_FLAG in on_disk:
            on_disk.pop(_USER_SHUTDOWN_FLAG, None)
            save_state(on_disk)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.debug("clear_user_shutdown_sentinel disk op failed: %s", exc)
    state.pop(_USER_SHUTDOWN_FLAG, None)


def _install_warm_embedder_override(store) -> tuple[object, bool]:
    import iai_mcp.embed as _embed_mod

    orig_efs = _embed_mod.embedder_for_store
    try:
        warm = orig_efs(store)
        warm.embed("warmup")

        def _held_embedder_for_store(_store):
            return warm

        _embed_mod.embedder_for_store = _held_embedder_for_store
        return orig_efs, True
    except Exception as exc:  # noqa: BLE001 -- prewarm/hold failure is non-fatal
        log.warning("embedder prewarm/hold failed: %s", exc, exc_info=True)
        try:
            write_event(store, "prewarm_failed", {"error": str(exc)}, severity="warning")
        except Exception:  # noqa: BLE001 -- event write inside boundary guard
            log.debug("prewarm_failed event write failed")
        return orig_efs, False


def _restore_embedder_funnel(orig_efs: object, installed: bool) -> None:
    if not installed:
        return
    try:
        import iai_mcp.embed as _embed_mod

        _embed_mod.embedder_for_store = orig_efs
    except Exception:  # noqa: BLE001 -- shutdown must never crash on restore
        log.debug("embedder funnel restore failed", exc_info=True)


def _set_process_title(title: str = "iai lilli (iai_mcp.daemon)") -> None:
    try:
        from setproctitle import setproctitle as _setproctitle
        _setproctitle(title)
    except Exception:  # noqa: BLE001
        pass


async def main() -> int:
    _set_process_title()
    _require_native()
    _raise_fd_limit()

    store = await _open_exclusive_store_with_backoff(
        lambda: MemoryStore(
            read_consistency_interval=timedelta(seconds=0),
            access_mode=AccessMode.EXCLUSIVE,
        )
    )

    try:
        hippo_lock_path = store.root / "hippo" / ".lock"
        cleanup_stale_consolidation_intent(hippo_lock_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        from iai_mcp.crypto_key_watch import check_crypto_key_file_rotation_event

        check_crypto_key_file_rotation_event(store)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        log.debug("crypto key rotation check skipped: %s", exc)

    from iai_mcp.migrate import detect_partial_migration
    _migration_state = detect_partial_migration(store.db)
    if _migration_state["state"] == "partial_swap_inconsistent":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback to restore from "
                        "records_old_<ts>, then iai-mcp daemon-start."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for partial_swap_inconsistent failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_rollback":
        try:
            sys.stderr.write(
                json.dumps({
                    "event": "daemon_boot_blocked_partial_migration",
                    "state": _migration_state,
                    "remediation": (
                        "iai-mcp migrate --rollback (discard the partial "
                        "staging) OR iai-mcp migrate --resume (continue "
                        "from migration_progress.json checkpoint)."
                    ),
                }) + "\n"
            )
        except (OSError, ValueError, TypeError) as exc:
            log.debug("stderr write for needs_rollback failed: %s", exc)
        raise SystemExit(2)
    if _migration_state["state"] == "needs_cleanup":
        for _old_name in _migration_state.get("old_tables", []):
            try:
                store.db.drop_table(_old_name)
            except (OSError, RuntimeError, KeyError) as _exc:
                log.warning("migrate cleanup drop_table(%s) failed: %s", _old_name, _exc)
                try:
                    sys.stderr.write(
                        json.dumps({
                            "event": "migrate_cleanup_failed",
                            "table": _old_name,
                            "err": str(_exc)[:120],
                        }) + "\n"
                    )
                except (OSError, ValueError, TypeError):
                    pass

    _respawn_by = os.environ.pop("IAI_DAEMON_RESPAWN_BY", None)
    if _respawn_by:
        try:
            write_event(
                store,
                "doctor_action",
                {"action": "daemon_respawned_by_doctor", "respawned_by": _respawn_by},
            )
        except Exception:  # noqa: BLE001 -- audit write must not block boot
            log.debug("failed to write respawn audit event")

    _load_erasure_config()
    _load_patsep_config()
    _load_s2_config()
    _load_sleep_overhaul_config()
    _load_reconsolidation_config()
    _load_stc_config()
    _load_dmn_config()
    _load_pask_config()

    _orig_efs: object = None
    _override_installed = False

    from iai_mcp.lifecycle_lock import LifecycleLock, LifecycleLockConflict

    lifecycle_lock = LifecycleLock()
    try:
        lifecycle_lock.acquire()
    except LifecycleLockConflict as exc:
        sys.stderr.write(f"daemon already running: {exc}\n")
        return 1

    _orig_efs, _override_installed = _install_warm_embedder_override(store)

    try:
        try:
            from iai_mcp.fsm_reconcile import reconcile_fsm_state

            _drift_report = reconcile_fsm_state(auto_correct=True)
            if _drift_report.get("drift") is True:
                log.warning(
                    "fsm_drift_detected canonical=%s legacy=%s",
                    _drift_report.get("canonical"),
                    _drift_report.get("legacy"),
                )
                try:
                    write_event(
                        store,
                        "fsm_drift_detected",
                        _drift_report,
                        severity="warning",
                        domain="ops",
                    )
                except Exception:  # noqa: BLE001 -- fail-safe
                    log.debug("fsm_drift_detected event write failed")
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("fsm_reconcile failed", exc_info=True)

        try:
            from iai_mcp.archive_backups import archive_stuck_backups

            archive_stuck_backups()
        except Exception:  # noqa: BLE001 -- fail-safe boundary
            log.debug("archive_stuck_backups failed", exc_info=True)

        state = await asyncio.to_thread(load_state)
        state.setdefault("fsm_state", STATE_WAKE)
        state["daemon_started_at"] = datetime.now(timezone.utc).isoformat()
        global _daemon_started_monotonic
        _daemon_started_monotonic = time.monotonic()
        state["daemon_pid"] = os.getpid()
        await asyncio.to_thread(save_state, state)
        write_event(store, "daemon_started", {"state": state["fsm_state"]})

        _wake_was_pending = False
        try:
            from pathlib import Path as _Path

            from iai_mcp.wake_handler import WakeHandler

            _wake_signal_path = _Path("~/.iai-mcp/wake.signal").expanduser()
            if WakeHandler(_wake_signal_path).consume_wake_signal():
                _wake_was_pending = True
                write_event(
                    store, "wake_signal_consumed", {"phase": "startup"}, severity="info"
                )
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake-handler
            log.debug("wake signal consume failed", exc_info=True)

        try:
            from iai_mcp.capture import capture_turn as _capture_turn
            from iai_mcp.capture_queue import CaptureQueue

            _capture_queue = CaptureQueue()
            def _capture_handler(record: dict) -> None:
                kwargs = {
                    "cue": record.get("cue", ""),
                    "text": record.get("text", record.get("surface", "")),
                    "tier": record.get("tier", "episodic"),
                    "session_id": record.get("session_id", "-"),
                    "role": record.get("role", "user"),
                }
                _capture_turn(store, **kwargs)

            ingested = await asyncio.to_thread(
                _capture_queue.ingest_pending, _capture_handler,
            )
            if ingested > 0:
                write_event(
                    store,
                    "capture_queue_drained",
                    {"phase": "startup", "ingested": ingested},
                    severity="info",
                )
        except Exception as exc:  # noqa: BLE001 -- never block boot on queue drain
            log.warning("capture queue drain failed at startup: %s", exc, exc_info=True)
            try:
                write_event(
                    store,
                    "capture_queue_drain_failed",
                    {"phase": "startup", "error": str(exc)[:200]},
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 -- event write inside boundary guard
                log.debug("capture_queue_drain_failed event write failed")

        try:
            from iai_mcp.daemon_state import (
                FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                prune_first_turn_pending,
            )

            state, dropped = prune_first_turn_pending(
                state, now=datetime.now(timezone.utc),
            )
            if dropped:
                await asyncio.to_thread(save_state, state)
                try:
                    write_event(
                        store,
                        "first_turn_pending_expired",
                        {
                            "dropped_count": len(dropped),
                            "session_ids": dropped,
                            "ttl_sec": FIRST_TURN_PENDING_TTL_SEC_DEFAULT,
                            "phase": "startup",
                        },
                        severity="info",
                    )
                except (OSError, RuntimeError) as exc:
                    log.debug("first_turn_pending_expired (startup) event write failed: %s", exc)
        except Exception:  # noqa: BLE001 -- boot MUST NOT block on startup prune
            log.debug("startup prune first_turn_pending failed", exc_info=True)

        try:
            _wal = SleepWAL()
            pending = _wal.pending_entries()
            if pending:
                log.warning(
                    "daemon startup: %d pending WAL entries found — prior process may have"
                    " died mid-sleep; entries logged but NOT re-executed",
                    len(pending),
                )
                write_event(
                    store,
                    "sleep_wal_pending_recovered",
                    {"count": len(pending), "phase": "startup"},
                    severity="info",
                )
        except Exception:  # noqa: BLE001 -- WAL check MUST NOT crash boot
            log.exception("daemon startup: sleep_wal pending check failed")


        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                loop.add_signal_handler(sig, shutdown.set)
            except (NotImplementedError, RuntimeError):
                pass

        try:
            health = _hippo_health_check_on_boot(store)
            await asyncio.to_thread(
                write_event,
                store,
                "hippo_boot_health",
                health,
                severity=("info" if health.get("action") == "ok" else "warning"),
            )
        except Exception:  # noqa: BLE001
            log.debug("hippo boot health check failed", exc_info=True)

        mcp_socket = SocketServer(store, state=state)
        mcp_socket_task = asyncio.create_task(mcp_socket.serve())
        await asyncio.sleep(0.05)

        try:
            from iai_mcp import runtime_graph_cache as _rgc_mod

            async def _boot_preload() -> None:
                try:
                    from iai_mcp import retrieve as _retrieve_preload
                    _g, _a, _rc = await asyncio.to_thread(
                        _retrieve_preload.build_runtime_graph, store,
                    )
                    await asyncio.to_thread(
                        _rgc_mod.save, store, _a, _rc, None, 0,
                    )
                except Exception as _exc:  # noqa: BLE001 -- preload MUST NOT crash daemon
                    log.debug("boot_preload failed: %s", _exc, exc_info=True)
                finally:
                    _rgc_mod.preload_ready.set()

            asyncio.create_task(_boot_preload())
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("boot_preload scheduling failed", exc_info=True)
            try:
                import iai_mcp.runtime_graph_cache as _rgc_fallback
                _rgc_fallback.preload_ready.set()
            except Exception:  # noqa: BLE001
                pass

        try:
            from iai_mcp.capture import drain_deferred_captures as _drain

            async def _drain_and_report() -> None:
                try:
                    drain_counts = await asyncio.to_thread(_drain, store)
                    if drain_counts.get("files_drained") or drain_counts.get(
                        "files_failed"
                    ):
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_startup",
                            drain_counts,
                            severity="info",
                        )
                except Exception as e:  # noqa: BLE001 -- drain MUST NOT crash daemon
                    log.warning("startup deferred drain failed: %s", e, exc_info=True)
                    try:
                        await asyncio.to_thread(
                            write_event,
                            store,
                            "deferred_drain_failed",
                            {"error": str(e)[:200], "phase": "startup"},
                            severity="warning",
                        )
                    except Exception:  # noqa: BLE001 -- event write inside boundary guard
                        log.debug("deferred_drain_failed (startup) event write failed")

            _drain_task = asyncio.create_task(_drain_and_report())
            try:
                mcp_socket._test_drain_task = _drain_task  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                log.debug("test drain task attach failed: %s", exc)
        except Exception:  # noqa: BLE001 -- scheduling failure must not block boot
            log.debug("startup drain scheduling failed", exc_info=True)


        from iai_mcp.heartbeat_scanner import HeartbeatScanner as _HeartbeatScanner
        from iai_mcp.idle_detector import IdleDetector as _IdleDetector
        from iai_mcp.lifecycle import (
            LifecycleEvent as _LifecycleEvent,
        )
        from iai_mcp.lifecycle import (
            LifecycleStateMachine as _LifecycleStateMachine,
        )
        from iai_mcp.lifecycle_state import LifecycleState as _LifecycleState
        from iai_mcp.s2_coordinator import (
            S2Coordinator,
            S2OscillationBlocked,
            S2OscillationConflict,
        )
        from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline as _SleepPipeline

        from pathlib import Path as _PathHere
        _store_root = os.environ.get("IAI_MCP_STORE")
        _wrappers_dir = (
            _PathHere(_store_root) if _store_root else _PathHere.home() / ".iai-mcp"
        ) / "wrappers"
        _heartbeat_scanner = _HeartbeatScanner(_wrappers_dir)
        _idle_detector = _IdleDetector()
        _sleep_pipeline = _SleepPipeline(store=store)

        from pathlib import Path as _PathS2
        _s2_config = _load_s2_config()
        _s2_coord = S2Coordinator(
            store=store,
            state_path=_PathS2.home() / ".iai-mcp" / "lifecycle_state.json",
            min_interval_sec=_s2_config.min_interval_sec,
            dry_run=_s2_config.dry_run,
        )

        from iai_mcp.peri_event_buffer import PeriEventBuffer, set_buffer
        _stc_config = _load_stc_config()
        _peri_event_buffer = PeriEventBuffer(maxlen=_stc_config.peri_event_buffer_size)
        set_buffer(_peri_event_buffer)

        _state_machine = _LifecycleStateMachine(coordinator=_s2_coord)

        if _wake_was_pending:
            try:
                await _state_machine.dispatch(
                    _LifecycleEvent.WAKE_SIGNAL,
                    reason="wake_on_signal_consumed",
                )
            except (S2OscillationConflict, S2OscillationBlocked):
                pass
            except Exception:  # noqa: BLE001 -- boot MUST NOT block on wake dispatch
                log.debug("wake signal dispatch failed", exc_info=True)

        global _cascade_executor
        _cascade_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="iai-cascade"
        )

        tick_task = asyncio.create_task(
            _scheduler_tick(store, state, mcp_socket=mcp_socket)
        )
        audit_task = asyncio.create_task(
            continuous_audit(store, shutdown)
        )
        s4_task = asyncio.create_task(
            _s4_offline_loop(store, shutdown)
        )
        cascade_task = asyncio.create_task(
            _hippea_cascade_loop(store, shutdown)
        )

        cpu_watchdog_task = asyncio.create_task(
            _cpu_watchdog_loop(store, shutdown)
        )

        _watchdog_stop = threading.Event()
        watchdog_thread = threading.Thread(
            target=_liveness_watchdog,
            args=(store, _watchdog_stop),
            name="iai-liveness-watchdog",
            daemon=True,
        )
        watchdog_thread.start()


        LIFECYCLE_TICK_INTERVAL_SEC: float = 30.0
        DROWSY_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_DROWSY_AFTER_SEC", "300")
        )
        HIBERNATE_AFTER_SEC: float = float(
            os.environ.get("LIFECYCLE_HIBERNATE_AFTER_SEC", "7200")
        )
        SLEEP_HEARTBEAT_IDLE_SEC: float = float(
            os.environ.get("LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC", "1800")
        )

        _last_active_monotonic: list[float] = [time.monotonic()]
        _prev_lifecycle_state: list = [_LifecycleState.WAKE]
        _lock_downgraded_to_shared: list[bool] = [False]

        async def lifecycle_tick() -> None:
            while not shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        shutdown.wait(),
                        timeout=LIFECYCLE_TICK_INTERVAL_SEC,
                    )
                    return
                except asyncio.TimeoutError:
                    pass

                try:
                    from iai_mcp.lifecycle_state import (
                        load_state as _load_lc,
                        save_state as _save_lc,
                    )
                    _lc_state = await asyncio.to_thread(_load_lc)
                    _now_utc = datetime.now(timezone.utc)
                    _expired, _ctx = _check_crisis_mode_expiry(_lc_state, _now_utc)
                    if _expired:
                        _lc_state["crisis_mode"] = False
                        _lc_state["crisis_mode_since_ts"] = None
                        await asyncio.to_thread(_save_lc, _lc_state)
                        try:
                            def _emit_expiry() -> None:
                                write_event(
                                    store,
                                    CRISIS_MODE_AUTO_EXPIRED,
                                    _ctx,
                                    severity="warning",
                                )
                            await asyncio.to_thread(_emit_expiry)
                        except Exception:  # noqa: BLE001 -- ledger emit failure non-fatal
                            log.debug(
                                "crisis_mode_auto_expired emit failed",
                                exc_info=True,
                            )
                    elif _ctx.get("backfilled_since_ts"):
                        _lc_state["crisis_mode_since_ts"] = _ctx["backfilled_since_ts"]
                        await asyncio.to_thread(_save_lc, _lc_state)
                except Exception:  # noqa: BLE001 -- expiry check MUST NOT crash lifecycle_tick
                    log.debug(
                        "lifecycle_tick crisis_mode expiry check failed",
                        exc_info=True,
                    )

                try:
                    scanner_active = await asyncio.to_thread(
                        _heartbeat_scanner.is_active,
                    )
                    heartbeat_idle = await asyncio.to_thread(
                        _heartbeat_scanner.heartbeat_idle_30min,
                    )
                    sleep_eligible = await asyncio.to_thread(
                        _idle_detector.sleep_eligible, heartbeat_idle,
                    )

                    now_mono = time.monotonic()
                    idle_elapsed = now_mono - _last_active_monotonic[0]

                    try:
                        from iai_mcp.daemon_state import load_state as _load_ds
                        _ds = await asyncio.to_thread(_load_ds)
                        _force_rem = bool((_ds.get("force_rem_request") or {}).get("pending"))
                        _user_sleep = bool((_ds.get("user_sleep_request") or {}).get("pending"))
                        if _force_rem or _user_sleep:
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.FORCE_SLEEP,
                                    reason="force_sleep_request",
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass
                            if _state_machine.current_state is _LifecycleState.DROWSY:
                                try:
                                    await _state_machine.dispatch(
                                        _LifecycleEvent.FORCE_SLEEP,
                                        reason="force_sleep_drowsy_to_sleep",
                                    )
                                except (S2OscillationConflict, S2OscillationBlocked):
                                    pass
                            if _state_machine.current_state is _LifecycleState.SLEEP:
                                _now_iso = __import__("datetime").datetime.now(
                                    __import__("datetime").timezone.utc,
                                ).isoformat()
                                _ds_upd = dict(_ds)
                                if _force_rem:
                                    req = dict(_ds_upd.get("force_rem_request") or {})
                                    req["pending"] = False
                                    req["honored_at"] = _now_iso
                                    _ds_upd["force_rem_request"] = req
                                if _user_sleep:
                                    req = dict(_ds_upd.get("user_sleep_request") or {})
                                    req["pending"] = False
                                    req["honored_at"] = _now_iso
                                    _ds_upd["user_sleep_request"] = req
                                from iai_mcp.daemon_state import save_state as _save_ds
                                await asyncio.to_thread(_save_ds, _ds_upd)
                    except Exception:  # noqa: BLE001 -- FORCE_SLEEP dispatch is best-effort
                        log.debug("lifecycle_tick FORCE_SLEEP dispatch failed", exc_info=True)

                    try:
                        from iai_mcp.fsm_reconcile import reconcile_fsm_state
                        reconcile_fsm_state(auto_correct=True)
                    except Exception:  # noqa: BLE001 -- reconcile is best-effort
                        pass

                    if scanner_active:
                        _last_active_monotonic[0] = now_mono
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.HEARTBEAT_REFRESH,
                                reason="heartbeat_refresh_active_wrapper",
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass
                    elif idle_elapsed >= SLEEP_HEARTBEAT_IDLE_SEC and sleep_eligible:
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.IDLE_30MIN,
                                reason="sleep_on_idle_30min",
                                sleep_eligible=True,
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass
                    elif idle_elapsed >= DROWSY_AFTER_SEC:
                        try:
                            await _state_machine.dispatch(
                                _LifecycleEvent.IDLE_5MIN,
                                reason="drowsy_on_idle_5min",
                            )
                        except (S2OscillationConflict, S2OscillationBlocked):
                            pass

                    current = _state_machine.current_state
                    if _should_drain_on_drowsy_edge(_prev_lifecycle_state[0], current):
                        try:
                            from iai_mcp.capture import drain_deferred_captures

                            await asyncio.to_thread(
                                _run_drowsy_drain,
                                store,
                                drain_fn=drain_deferred_captures,
                                write_event_fn=write_event,
                            )
                        except Exception:  # noqa: BLE001 -- drowsy drain non-fatal
                            log.debug("lifecycle_tick drowsy drain failed", exc_info=True)

                        try:
                            from iai_mcp.embed import embedder_for_store
                            from iai_mcp import runtime_graph_cache as _rgc

                            def _run_wake_sequence():
                                try:
                                    _emb = embedder_for_store(store)
                                except Exception:
                                    _emb = None
                                result = store.db.pending_embeddings_wake_sequence(embedder=_emb)
                                if result.get("action") != "skip":
                                    try:
                                        _rgc.invalidate(store)
                                    except Exception:
                                        pass
                                return result

                            _wake_seq_result = await asyncio.to_thread(_run_wake_sequence)
                            if (
                                isinstance(_wake_seq_result, dict)
                                and _wake_seq_result.get("action") != "skip"
                            ):
                                try:
                                    _kick_drowsy_rgc_rebuild(store)
                                except Exception:  # noqa: BLE001 -- best-effort
                                    log.debug("drowsy-edge kick failed", exc_info=True)
                        except Exception:  # noqa: BLE001 -- wake sequence non-fatal
                            log.debug("lifecycle_tick pending_embeddings_wake_sequence failed", exc_info=True)
                    if (
                        not _lock_downgraded_to_shared[0]
                        and current in (
                            _LifecycleState.WAKE,
                            _LifecycleState.DROWSY,
                        )
                    ):
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            _lock_downgraded_to_shared[0] = True
                            log.debug("daemon_lock_downgrade: EX→SH on first WAKE entry")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade failed", exc_info=True)

                    _prev_lifecycle_state[0] = current
                    if current is _LifecycleState.SLEEP:
                        def _interrupt_check() -> bool:
                            if mcp_socket.active_connections > 0:
                                return True
                            elapsed = (
                                time.monotonic() - mcp_socket.last_activity_ts
                            )
                            return elapsed < INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC

                        try:
                            await asyncio.to_thread(store.db.escalate_to_exclusive)
                            log.debug("daemon_lock_escalate: SH→EX for sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_escalate failed", exc_info=True)

                        result = await asyncio.to_thread(
                            _sleep_pipeline.run, _interrupt_check,
                        )

                        # --- WAKE hook (UNDER LOCK_EX, BEFORE downgrade) ---
                        try:
                            await asyncio.to_thread(_write_session_start_cache, store)
                        except Exception:  # noqa: BLE001 -- precache MUST NOT crash
                            log.debug("lifecycle_tick _write_session_start_cache failed", exc_info=True)
                        try:
                            from iai_mcp.memory_bank import write_processed_salience_top_n
                            await asyncio.to_thread(write_processed_salience_top_n, store)
                        except (ImportError, OSError, ValueError, RuntimeError) as exc:
                            log.debug("lifecycle_tick write_processed_salience_top_n failed: %s", exc)
                        try:
                            from iai_mcp.capture import drain_active_live_captures
                            _live_drain = await asyncio.to_thread(
                                drain_active_live_captures, store, exclude_session_id="-",
                            )
                            if _live_drain.get("events_inserted"):
                                await asyncio.to_thread(
                                    write_event, store, "active_live_drain_wake",
                                    _live_drain, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- drain MUST NOT crash
                            log.debug("lifecycle_tick active_live_drain failed: %s", _exc)
                        try:
                            from iai_mcp.provenance_buffer import flush_deferred_provenance
                            _prov_count = await asyncio.to_thread(
                                flush_deferred_provenance, store,
                            )
                            if _prov_count > 0:
                                await asyncio.to_thread(
                                    write_event, store, "deferred_provenance_flush_wake",
                                    {"count": _prov_count}, severity="info",
                                )
                        except Exception as _exc:  # noqa: BLE001 -- flush MUST NOT crash
                            log.debug("lifecycle_tick flush_deferred_provenance failed: %s", _exc)
                        try:
                            await asyncio.to_thread(_wake_hook_rebuild_if_cold, store)
                        except Exception as _exc:  # noqa: BLE001 -- best-effort
                            log.debug("lifecycle_tick wake-hook rebuild-if-cold failed: %s", _exc)

                        # Downgrade EX → SH after the consolidation window.
                        try:
                            await asyncio.to_thread(store.db.downgrade_to_shared)
                            log.debug("daemon_lock_downgrade: EX→SH after sleep pipeline")
                        except Exception:  # noqa: BLE001
                            log.debug("daemon_lock_downgrade_post_sleep failed", exc_info=True)
                        if (
                            not result.get("interrupted", False)
                            and result.get("failed_step") is None
                            and not result.get("quarantine_triggered", False)
                            and len(result.get("completed_steps", [])) >= 5
                        ):
                            still_idle_now = await asyncio.to_thread(
                                _heartbeat_scanner.heartbeat_idle_30min,
                            )
                            sleep_eligible_now = await asyncio.to_thread(
                                _idle_detector.sleep_eligible, still_idle_now,
                            )
                            try:
                                await _state_machine.dispatch(
                                    _LifecycleEvent.SLEEP_CYCLE_DONE,
                                    reason="hibernate_on_sleep_cycle_done",
                                    still_idle=(still_idle_now and sleep_eligible_now),
                                )
                            except (S2OscillationConflict, S2OscillationBlocked):
                                pass

                    current = _state_machine.current_state
                    if (
                        current is _LifecycleState.HIBERNATION
                        and not _state_machine.shadow_run
                    ):
                        try:
                            write_event(
                                store,
                                "lifecycle_hibernation_exit",
                                {
                                    "reason": "lifecycle_tick_hibernation",
                                    "shadow_run": False,
                                },
                                severity="info",
                            )
                        except (OSError, RuntimeError) as exc:
                            log.debug("lifecycle_hibernation_exit event write failed: %s", exc)
                        shutdown.set()
                        return
                except Exception:  # noqa: BLE001 -- lifecycle tick must NEVER crash
                    log.warning("lifecycle tick iteration failed", exc_info=True)

        lifecycle_tick_task = asyncio.create_task(lifecycle_tick())

        try:
            await shutdown.wait()
        finally:
            try:
                mcp_socket.shutdown_event.set()
            except (AttributeError, RuntimeError) as exc:
                log.debug("mcp_socket shutdown_event.set failed: %s", exc)
            try:
                _watchdog_stop.set()
            except (NameError, RuntimeError) as exc:
                log.debug("watchdog stop set failed: %s", exc)
            try:
                if _cascade_executor is not None:
                    _cascade_executor.shutdown(wait=False)
            except Exception as exc:  # noqa: BLE001
                log.debug("cascade executor shutdown failed: %s", exc)
            _cancel_targets = [
                tick_task, audit_task, s4_task, cascade_task,
                mcp_socket_task,
                cpu_watchdog_task,
                lifecycle_tick_task,
            ]
            for t in _cancel_targets:
                t.cancel()
            await asyncio.gather(*_cancel_targets, return_exceptions=True)
            try:
                from iai_mcp.events import flush_event_buffer

                events_count = flush_event_buffer(store)
                if events_count > 0:
                    log.info("events buffer flushed on shutdown: count=%d", events_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("events buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                from iai_mcp.store import flush_record_buffer

                records_count = flush_record_buffer(store)
                if records_count > 0:
                    log.info("records buffer flushed on shutdown: count=%d", records_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("records buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                from iai_mcp.store import flush_edge_buffer

                edges_count = flush_edge_buffer(store)
                if edges_count > 0:
                    log.info("edges buffer flushed on shutdown: count=%d", edges_count)
            except Exception as e:  # noqa: BLE001 -- shutdown MUST complete
                log.warning("edges buffer shutdown flush failed: %s", e, exc_info=True)
            try:
                write_event(store, "daemon_stopped", {"state": state.get("fsm_state")})
            except (OSError, RuntimeError) as exc:
                log.debug("daemon_stopped event write failed: %s", exc)
            _clear_user_shutdown_sentinel(state)
            try:
                state.pop("daemon_pid", None)
                state["daemon_stopped_at"] = datetime.now(timezone.utc).isoformat()
                await asyncio.to_thread(save_state, state)
            except (OSError, ValueError) as exc:
                log.debug("final save_state failed: %s", exc)
            try:
                lifecycle_lock.release()
            except (OSError, RuntimeError) as exc:
                log.debug("lifecycle_lock release failed: %s", exc)
    finally:
        _restore_embedder_funnel(_orig_efs, _override_installed)
    return 0


from iai_mcp.daemon._watchdog import (  # noqa: E402 -- re-exported after main() so the package namespace is the single patchable source of truth
    HIPPEA_CASCADE_POLL_SEC,
    HIPPEA_CASCADE_MIN_INTERVAL_SEC,
    _last_cascade_completed_at,
    _cascade_executor,
    WATCHDOG_POLL_SEC,
    WATCHDOG_THRESHOLD_PERCENT,
    WATCHDOG_EVENT_COOLDOWN_SEC,
    WATCHDOG_SAMPLE_WINDOW,
    WATCHDOG_LIVENESS_POLL_SEC,
    WATCHDOG_WARN_POLL_SEC,
    WATCHDOG_PROBE_TIMEOUT_SEC,
    WATCHDOG_FAILURE_DEBOUNCE_N,
    WATCHDOG_RSS_HARD_CAP_BYTES,
    WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES,
    WATCHDOG_MAX_RECOVERIES,
    WATCHDOG_RECOVERY_WINDOW_SEC,
    WATCHDOG_COLD_START_GRACE_SEC,
    WATCHDOG_SLEEP_STALE_THRESHOLD_SEC,
    WATCHDOG_CRISIS_MODE_EXPIRY_SEC,
    _WATCHDOG_LOG_FD,
    _WATCHDOG_BLACKBOX_FD,
    _WATCHDOG_BLACKBOX_EPISODE_FIRED,
    _WATCHDOG_BLACKBOX_ENABLED,
    BOOT_LOCK_RETRY_ATTEMPTS,
    BOOT_LOCK_RETRY_BACKOFF_SEC,
    _last_overload_event_at,
    _last_sleep_stale_started_at,
    _daemon_started_monotonic,
    _hippea_cascade_loop,
    _watchdog_active_task_names,
    _cpu_watchdog_loop,
    _next_poll_interval,
    _evaluate_watchdog,
    _check_sleep_cycle_staleness,
    _check_crisis_mode_expiry,
    _watchdog_state_dir,
    _watchdog_log_path,
    _watchdog_socket_path,
    _vm_pressure_level,
    _own_rss_bytes,
    _iso_now,
    _write_breadcrumb,
    _self_kill,
    _capture_blackbox,
    _open_exclusive_store_with_backoff,
    _load_recovery_timestamps,
    _probe_status_roundtrip,
    _watchdog_tick,
    _liveness_watchdog,
)

__all__ = [
    # lifecycle / main
    "main",
    "transition",
    "log",
    "serve_control_socket",
    "_hippo_health_check_on_boot",
    "_raise_fd_limit",
    "_run_drowsy_drain",
    "_should_drain_on_drowsy_edge",
    "_kick_drowsy_rgc_rebuild",
    "_wake_hook_rebuild_if_cold",
    "_store_is_empty",
    "_set_process_title",
    "_install_warm_embedder_override",
    "_restore_embedder_funnel",
    "_clear_user_shutdown_sentinel",
    "_USER_SHUTDOWN_FLAG",
    "_is_inside_window",
    "_update_pending_digest",
    "_write_session_start_cache",
    "_tick_body",
    "_scheduler_tick",
    "_s4_offline_loop",
    # FSM + tick constants
    "STATE_WAKE",
    "STATE_TRANSITIONING",
    "STATE_SLEEP",
    "STATE_DREAMING",
    "VALID_TRANSITIONS",
    "TICK_INTERVAL_SEC",
    "DEFAULT_CYCLE_COUNT",
    "S4_OFFLINE_INTERVAL_SEC",
    "S4_FIRST_ITER_GRACE_SEC",
    "SESSION_START_CACHE_PATH",
    "SESSION_START_CACHE_MAX_CHARS",
    "INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC",
    "_DAEMON_NOFILE_FLOOR_DEFAULT",
    # daemon_config
    "ErasureConfig",
    "_load_erasure_config",
    "PatSepConfig",
    "_load_patsep_config",
    "S2Config",
    "_load_s2_config",
    "SleepOverhaulConfig",
    "_load_sleep_overhaul_config",
    "ReconsolidationConfig",
    "_load_reconsolidation_config",
    "StcConfig",
    "_load_stc_config",
    "UserModelConfig",
    "_load_user_model_config",
    "SpatialConfig",
    "_load_spatial_config",
    "DmnConfig",
    "_load_dmn_config",
    "PaskConfig",
    "_load_pask_config",
    # watchdog
    "HIPPEA_CASCADE_POLL_SEC",
    "HIPPEA_CASCADE_MIN_INTERVAL_SEC",
    "_last_cascade_completed_at",
    "_cascade_executor",
    "WATCHDOG_POLL_SEC",
    "WATCHDOG_THRESHOLD_PERCENT",
    "WATCHDOG_EVENT_COOLDOWN_SEC",
    "WATCHDOG_SAMPLE_WINDOW",
    "WATCHDOG_LIVENESS_POLL_SEC",
    "WATCHDOG_WARN_POLL_SEC",
    "WATCHDOG_PROBE_TIMEOUT_SEC",
    "WATCHDOG_FAILURE_DEBOUNCE_N",
    "WATCHDOG_RSS_HARD_CAP_BYTES",
    "WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES",
    "WATCHDOG_MAX_RECOVERIES",
    "WATCHDOG_RECOVERY_WINDOW_SEC",
    "WATCHDOG_COLD_START_GRACE_SEC",
    "WATCHDOG_SLEEP_STALE_THRESHOLD_SEC",
    "WATCHDOG_CRISIS_MODE_EXPIRY_SEC",
    "_WATCHDOG_LOG_FD",
    "_WATCHDOG_BLACKBOX_FD",
    "_WATCHDOG_BLACKBOX_EPISODE_FIRED",
    "_WATCHDOG_BLACKBOX_ENABLED",
    "BOOT_LOCK_RETRY_ATTEMPTS",
    "BOOT_LOCK_RETRY_BACKOFF_SEC",
    "_last_overload_event_at",
    "_last_sleep_stale_started_at",
    "_daemon_started_monotonic",
    "_hippea_cascade_loop",
    "_watchdog_active_task_names",
    "_cpu_watchdog_loop",
    "_next_poll_interval",
    "_evaluate_watchdog",
    "_check_sleep_cycle_staleness",
    "_check_crisis_mode_expiry",
    "_watchdog_state_dir",
    "_watchdog_log_path",
    "_watchdog_socket_path",
    "_vm_pressure_level",
    "_own_rss_bytes",
    "_iso_now",
    "_write_breadcrumb",
    "_self_kill",
    "_capture_blackbox",
    "_open_exclusive_store_with_backoff",
    "_load_recovery_timestamps",
    "_probe_status_roundtrip",
    "_watchdog_tick",
    "_liveness_watchdog",
    "DAEMON_MEMORY_PRESSURE_KILL",
    "DAEMON_SLEEP_CYCLE_STALE",
    "DAEMON_WATCHDOG_NEEDS_OPERATOR",
    "DAEMON_WEDGE_KILL",
    "CRISIS_MODE_AUTO_EXPIRED",
]
