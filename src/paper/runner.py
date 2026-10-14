"""CLI: live Databento paper or true-path TBBO replay.

Examples::

    # Live: Databento stream for fills + periodic model signal
    python -m src.paper.runner --stream --objective nq-50pt-econ

    # Offline true-path: replay databento.tbbo for a session
    python -m src.paper.runner --replay-db 2026-08-04 --objective nq-50pt-econ

    # Signal-only dry check (no market loop)
    python -m src.paper.runner --signal-check --date 2026-08-04
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.paper.engine import PaperEngine
from src.paper.ledger import save_engine
from src.paper.policy import load_paper_policy, resolve_default_model
from src.paper.signals import (
    SignalCursor,
    latest_signal,
    load_feature_day,
    score_feature_frame,
)
from src.paper.stream import iter_tbbo_from_db, run_databento_live

logger = logging.getLogger("vtech.paper.runner")


def _resolve_model(objective_id: str | None, model: str | None) -> Path:
    if model:
        return Path(model)
    pol = load_paper_policy(objective_id=objective_id)
    m = resolve_default_model(pol.objective_id)
    if m is None:
        raise FileNotFoundError(
            f"no staged model for {pol.objective_id}; pass --model")
    return m


def run_replay_db(
    day: str,
    *,
    objective_id: str | None = None,
    model_path: str | Path | None = None,
    dir_threshold: float | None = None,
    entry_style: str = "last",
    entry_timeout_seconds: float = 60.0,
    save: bool = True,
    tick_limit: int | None = None,
) -> dict[str, Any]:
    """Replay TimescaleDB tbbo prints; place limits when model signals fire in time."""
    mpath = Path(model_path) if model_path else _resolve_model(objective_id, None)
    policy = load_paper_policy(
        objective_id=objective_id, model_path=mpath,
        dir_threshold=dir_threshold,
        entry_style=entry_style,  # type: ignore[arg-type]
        entry_timeout_seconds=entry_timeout_seconds,
    )
    # thr from bundle if not set
    if dir_threshold is None:
        from src.serving.bundle import load_serving_bundle
        b = load_serving_bundle(mpath, validate=False)
        policy = load_paper_policy(
            objective_id=policy.objective_id, model_path=mpath,
            dir_threshold=float(b.get("dir_thr", policy.dir_threshold)),
            entry_style=entry_style,  # type: ignore[arg-type]
            entry_timeout_seconds=entry_timeout_seconds,
        )

    df = load_feature_day(day)
    preds = score_feature_frame(df, mpath)
    cursor = SignalCursor(preds)
    engine = PaperEngine(policy=policy)

    n_ticks = 0
    for tick in iter_tbbo_from_db(day, limit=tick_limit):
        n_ticks += 1
        # signals whose feature bar time has arrived
        for sig_ts, direction, conf in cursor.signals_through(tick.ts):
            engine.on_signal(direction, conf, ts=sig_ts)
        engine.on_tick(tick)
        if n_ticks % 100_000 == 0:
            logger.info("replay_progress ticks=%s closed=%s", n_ticks, len(engine.closed))

    # end of day flat
    if engine.position is not None and engine.last_tick is not None:
        engine._force_close(engine.last_tick, "eod")
    # cancel dangling entry
    for od in list(engine.working):
        if od.kind == "entry_limit":
            od.status = "cancelled"

    out: dict[str, Any] = {
        "mode": "replay_db",
        "day": day,
        "model": str(mpath),
        "n_ticks": n_ticks,
        "summary": engine.summary(),
    }
    if save:
        root = save_engine(engine)
        out["saved_to"] = str(root)
    return out


def run_stream(
    *,
    objective_id: str | None = None,
    model_path: str | Path | None = None,
    dir_threshold: float | None = None,
    entry_style: str = "last",
    entry_timeout_seconds: float = 60.0,
    signal_every_s: float = 5.0,
    rebuild_features: bool = True,
    save_every_s: float = 30.0,
) -> dict[str, Any]:
    """Live loop: Databento TBBO fills + periodic model signal."""
    mpath = Path(model_path) if model_path else _resolve_model(objective_id, None)
    policy = load_paper_policy(
        objective_id=objective_id, model_path=mpath,
        dir_threshold=dir_threshold,
        entry_style=entry_style,  # type: ignore[arg-type]
        entry_timeout_seconds=entry_timeout_seconds,
    )
    engine = PaperEngine(policy=policy)
    stop = {"flag": False}
    lock = threading.Lock()
    last_sig_key: dict[str, Any] = {"ts": None}

    def _handle_sig(*_a: Any) -> None:
        stop["flag"] = True
        logger.info("paper_stream_stop_requested")

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    def on_tick(tick: Any) -> None:
        with lock:
            engine.on_tick(tick)

    def signal_loop() -> None:
        while not stop["flag"]:
            try:
                sig = latest_signal(
                    mpath, rebuild_live=rebuild_features,
                )
                if sig.get("ok"):
                    key = (sig.get("ts"), sig.get("direction"), sig.get("confidence"))
                    if key != (last_sig_key.get("ts"), last_sig_key.get("dir"), last_sig_key.get("conf")):
                        last_sig_key["ts"] = sig.get("ts")
                        last_sig_key["dir"] = sig.get("direction")
                        last_sig_key["conf"] = sig.get("confidence")
                        ts = datetime.fromisoformat(str(sig["ts"]).replace("Z", "+00:00"))
                        with lock:
                            engine.on_signal(
                                str(sig["direction"]),
                                float(sig["confidence"]),
                                ts=ts,
                            )
                        logger.info(
                            "paper_signal dir=%s conf=%.3f ts=%s",
                            sig["direction"], sig["confidence"], sig["ts"],
                        )
            except Exception as exc:
                logger.warning("paper_signal_loop_error: %s", exc)
            # sleep in small steps for responsive stop
            for _ in range(int(max(1, signal_every_s * 10))):
                if stop["flag"]:
                    break
                time.sleep(0.1)

    def save_loop() -> None:
        while not stop["flag"]:
            for _ in range(int(max(1, save_every_s * 10))):
                if stop["flag"]:
                    break
                time.sleep(0.1)
            try:
                with lock:
                    save_engine(engine)
            except Exception as exc:
                logger.warning("paper_save_error: %s", exc)

    t_sig = threading.Thread(target=signal_loop, name="paper-signals", daemon=True)
    t_save = threading.Thread(target=save_loop, name="paper-ledger", daemon=True)
    t_sig.start()
    t_save.start()

    logger.info(
        "paper_stream_start model=%s objective=%s entry_style=%s",
        mpath, policy.objective_id, policy.entry_style,
    )
    try:
        run_databento_live(on_tick, stop_flag=lambda: stop["flag"])
    finally:
        stop["flag"] = True
        with lock:
            if engine.position is not None and engine.last_tick is not None:
                engine._force_close(engine.last_tick, "shutdown")
            root = save_engine(engine)
        logger.info("paper_stream_stopped saved=%s", root)

    return {"mode": "stream", "model": str(mpath), "summary": engine.summary(), "saved_to": str(root)}


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Paper trading: Databento live ticks + limit entry + OCO bracket")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stream", action="store_true",
                   help="Databento Live tbbo stream (true live paper)")
    g.add_argument("--replay-db", metavar="DATE",
                   help="Replay databento.tbbo from Timescale for DATE")
    g.add_argument("--signal-check", action="store_true",
                   help="Print latest model signal only")
    p.add_argument("--date", default="", help="session date for --signal-check")
    p.add_argument("--objective", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--dir-threshold", type=float, default=None)
    p.add_argument("--entry-style", choices=("last", "join", "cross"), default="last",
                   help="limit price: last trade / join bid-ask / cross spread")
    p.add_argument("--entry-timeout", type=float, default=60.0,
                   help="seconds before unfilled entry limit expires")
    p.add_argument("--signal-every", type=float, default=5.0,
                   help="seconds between live signal refreshes")
    p.add_argument("--no-rebuild", action="store_true",
                   help="live signals: use on-disk parquet only (no FeatureEngine rebuild)")
    p.add_argument("--tick-limit", type=int, default=None, help="replay: max tbbo rows")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    try:
        if args.signal_check:
            mpath = _resolve_model(args.objective, args.model)
            day = args.date or datetime.now(timezone.utc).date().isoformat()
            out = latest_signal(mpath, day=day, rebuild_live=False)
            print(json.dumps(out, indent=2, default=str))
            return 0 if out.get("ok") else 1
        if args.replay_db:
            out = run_replay_db(
                args.replay_db,
                objective_id=args.objective,
                model_path=args.model,
                dir_threshold=args.dir_threshold,
                entry_style=args.entry_style,
                entry_timeout_seconds=args.entry_timeout,
                save=not args.no_save,
                tick_limit=args.tick_limit,
            )
            print(json.dumps(out, indent=2, default=str))
            return 0
        if args.stream:
            out = run_stream(
                objective_id=args.objective,
                model_path=args.model,
                dir_threshold=args.dir_threshold,
                entry_style=args.entry_style,
                entry_timeout_seconds=args.entry_timeout,
                signal_every_s=args.signal_every,
                rebuild_features=not args.no_rebuild,
            )
            print(json.dumps(out, indent=2, default=str))
            return 0
    except Exception as exc:
        logger.error("paper_failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
