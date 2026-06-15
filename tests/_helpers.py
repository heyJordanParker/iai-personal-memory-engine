from __future__ import annotations

import tempfile
from pathlib import Path

from iai_mcp.store import MemoryStore


def short_socket_path(prefix: str = "iai-sock-") -> Path:
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d / "d.sock"


def make_tmp_store(tmp_path: Path) -> MemoryStore:
    store_root = tmp_path / "hippo"
    store_root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=store_root)


def set_tmp_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
