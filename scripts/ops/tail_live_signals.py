"""Tail data/live/signals.jsonl + monitor heartbeat.json.

Stub for the broker-side bridge: proves the signal exporter emits, appends
new rows chronologically, and that the heartbeat file stays fresh. The real
broker-side reader (NinjaScript / cAlgo / etc.) will do the same thing —
follow the file, parse JSON per line, and act on new signals.

Usage:
    python -m scripts.ops.tail_live_signals            # tail data/live/
    python -m scripts.ops.tail_live_signals --once     # print state and exit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.data_paths import live_dir

STALE_HEARTBEAT_S = 30.0


def _live_dir() -> Path:
    return live_dir()


def _fmt_signal(row: dict) -> str:
    d = row.get("dir", "?")
    conf = row.get("conf")
    entry = row.get("entry_px")
    tgt = row.get("tgt_pts")
    stop = row.get("stop_pts")
    return (f"[{row.get('ts')}] {d.upper():<4} conf={conf} @ {entry}  "
            f"tgt=+{tgt} stop=-{stop}  ({row.get('objective_id')} / "
            f"{row.get('model')})")


def _heartbeat_age_s(hb_path: Path) -> float | None:
    if not hb_path.exists():
        return None
    try:
        hb = json.loads(hb_path.read_text())
        ts = hb.get("heartbeat_at")
        if not ts:
            return None
        return (datetime.now(timezone.utc)
                - datetime.fromisoformat(ts)).total_seconds()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] heartbeat parse: {exc}", file=sys.stderr)
        return None


def _snapshot(sig_path: Path, hb_path: Path) -> None:
    print(f"signals: {sig_path}  (exists={sig_path.exists()}, "
          f"size={sig_path.stat().st_size if sig_path.exists() else 0} B)")
    print(f"heartbeat: {hb_path}  (age={_heartbeat_age_s(hb_path)} s)")
    if sig_path.exists() and sig_path.stat().st_size:
        with sig_path.open() as f:
            last = None
            for line in f:
                last = line
            if last:
                print("last signal:", _fmt_signal(json.loads(last)))


def _tail(sig_path: Path, hb_path: Path, poll_s: float = 1.0) -> None:
    print(f"tailing {sig_path} (Ctrl-C to stop)")
    # Seek to end so we only show *new* signals from this point on.
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.touch(exist_ok=True)
    inode = sig_path.stat().st_ino
    f = sig_path.open()
    f.seek(0, 2)  # end
    last_hb_warn = 0.0
    buf = ""
    while True:
        chunk = f.read()
        if chunk:
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    print(_fmt_signal(json.loads(line)))
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] bad line: {exc}: {line!r}", file=sys.stderr)
        age = _heartbeat_age_s(hb_path)
        if age is not None and age > STALE_HEARTBEAT_S:
            now = time.time()
            if now - last_hb_warn > 30:  # rate-limit warning to 30s
                print(f"[warn] heartbeat stale ({age:.0f}s) — snapshot builder "
                      f"may be down", file=sys.stderr)
                last_hb_warn = now
        # Handle rotation (inode change) — reopen from start.
        try:
            new_inode = sig_path.stat().st_ino
            if new_inode != inode:
                print("[info] signals.jsonl rotated — reopening", file=sys.stderr)
                f.close()
                f = sig_path.open()
                inode = new_inode
        except FileNotFoundError:
            pass
        time.sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true",
                    help="print current state and exit (no tail)")
    ap.add_argument("--poll-s", type=float, default=1.0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)  # so tail output isn't block-buffered when piped
    except Exception:  # noqa: BLE001
        pass

    live = _live_dir()
    sig_path = live / "signals.jsonl"
    hb_path = live / "heartbeat.json"

    _snapshot(sig_path, hb_path)
    if args.once:
        return
    try:
        _tail(sig_path, hb_path, poll_s=args.poll_s)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
