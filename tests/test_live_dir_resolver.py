"""live_dir() stays on the current active top-level path; ops/live/ is a documented
future target, not an auto-cutover.

Tenth-pass correction (Storage Implementation Tranche 1): the ninth pass had
live_dir() prefer ops/live/ the moment it was physically populated. That
auto-cutover is reverted -- switching the live-signal transport is an explicit
operator decision, not something a resolver may infer from a directory's mere
existence. See docs/STORAGE_CONTRACT.md §8a.
"""
from __future__ import annotations

from src.common.data_paths import live_dir


def test_live_dir_resolves_to_top_level_when_ops_live_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    assert live_dir() == tmp_path / "live"


def test_live_dir_does_not_switch_merely_because_ops_live_exists(tmp_path, monkeypatch):
    """The reverted behavior: a populated ops/live/ must NOT change the resolved
    path. Adopting it requires an explicit future cutover (a code change), not a
    filesystem probe -- see docs/STORAGE_CONTRACT.md §8a."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    ops_live = tmp_path / "ops" / "live"
    ops_live.mkdir(parents=True)
    (ops_live / "signals.jsonl").write_text('{"probe": true}\n')
    assert live_dir() == tmp_path / "live"


def test_live_dir_is_the_only_construction_site(tmp_path, monkeypatch):
    """webui/live.py and execution/config.py must resolve identically -- a
    hardcoded second construction is exactly the bug the eighth pass fixed 4x."""
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    expected = tmp_path / "live"

    from src.webui.live import _live_dir as webui_live_dir
    assert webui_live_dir() == expected

    # execution_gateway.yaml's real default leaves signal_path/heartbeat_path
    # null, so load_execution_config()'s _path() fallback resolves through
    # live_dir() -- proving the fourth call site is wired, not hardcoded.
    from src.execution.config import load_execution_config
    cfg = load_execution_config()
    assert cfg.signal_path == expected / "signals.jsonl"
    assert cfg.heartbeat_path == expected / "heartbeat.json"
