#!/usr/bin/env python3
"""Live prediction display — builds the current feature parquet from live
TimescaleDB data and shows the wall-clock time + the staged model's prediction,
refreshing in place.

Pipeline each cycle:
    live DB  →  FeatureEngine.build_date(today)  →  data/sources/features/timeframe=5s/symbol=NQ/year=/month=/features_5s_partN.parquet
           →  last row aligned to the model's feature_schema.json
           →  scaler.transform → XGBoost.predict_proba → direction + confidence

The model is a frozen serving bundle (model.ubj + scaler.pkl + feature_schema.json).
Requires the live streamer to be running (python -m src.acquisition.live_streamer).

Usage:
    python -m scripts.live_predict_display                       # infinite, ~5s cadence
    python -m scripts.live_predict_display --iterations 3        # run 3 cycles then exit
    python -m scripts.live_predict_display --model data/models/model_staging/<obj>/<version>
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import xgboost as xgb

CST = ZoneInfo("America/Chicago")
DEFAULT_MODEL = "data/models/model_staging/nq-50pt-tbm/xgboost_20260711T033354Z"

# ANSI
CLR = "\033[2J\033[H"      # clear screen + cursor home
GRN, RED, GRY, BOLD, DIM, RST = "\033[92m", "\033[91m", "\033[90m", "\033[1m", "\033[2m", "\033[0m"
CYAN, YEL = "\033[96m", "\033[93m"


def load_bundle(model_dir: Path):
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    manifest = json.loads((model_dir / "finalize_manifest.json").read_text())
    booster = xgb.XGBClassifier()
    booster.load_model(str(model_dir / "model.ubj"))
    scaler = joblib.load(model_dir / "scaler.pkl") if (model_dir / "scaler.pkl").exists() else None
    dir_thr = float(manifest.get("spec_params", {}).get("dir_threshold", 0.5))
    return schema["feature_cols"], booster, scaler, dir_thr, manifest


def predict_last_row(feat_cols, booster, scaler, dir_thr, parquet_path: Path):
    """Read the latest row from the 5s hive part returned by FeatureEngine."""
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return None
    if "bar_end_ts" in df.columns:
        df = df.copy()
        df["bar_end_ts"] = pd.to_datetime(df["bar_end_ts"], utc=True)
        df = df.set_index("bar_end_ts").sort_index()
    row = df.reindex(columns=feat_cols).iloc[[-1]]
    n_nan = int(row.isna().sum(axis=1).iloc[0])
    X = row.to_numpy("float64")
    if scaler is not None:
        X = scaler.transform(X)
    p_down, p_flat, p_up = (float(x) for x in booster.predict_proba(X)[0])
    if p_up >= dir_thr and p_up >= p_down:
        direction, conf = "up", p_up
    elif p_down >= dir_thr and p_down > p_up:
        direction, conf = "down", p_down
    else:
        direction, conf = "flat", p_flat
    return {
        "bar_ts": df.index.max(), "n_rows": len(df), "n_nan": n_nan,
        "p_down": p_down, "p_flat": p_flat, "p_up": p_up,
        "direction": direction, "conf": conf,
    }


def render(res, tau, build_s, model_version, err=None):
    now = datetime.now(timezone.utc)
    now_cst = now.astimezone(CST)
    lines = []
    lines.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RST}")
    lines.append(f"{BOLD}{CYAN}║   VTech — LIVE model prediction                              ║{RST}")
    lines.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RST}")
    lines.append("")
    lines.append(f"  {BOLD}Time (CST){RST}   {YEL}{now_cst:%Y-%m-%d  %H:%M:%S}{RST}  "
                 f"{DIM}({now:%H:%M:%S} UTC){RST}")
    lines.append(f"  {BOLD}Model{RST}        {model_version}")
    if err:
        lines.append("")
        lines.append(f"  {RED}⚠ {err}{RST}")
        lines.append(f"  {DIM}(is the live streamer running? waiting for data…){RST}")
        return "\n".join(lines)

    d = res["direction"]
    color = GRN if d == "up" else RED if d == "down" else GRY
    arrow = "▲" if d == "up" else "▼" if d == "down" else "■"
    signal = d != "flat" and res["conf"] >= tau
    sig_txt = f"{GRN}● SIGNAL{RST}" if signal else f"{DIM}○ no-trade{RST}"

    lines.append(f"  {BOLD}Feature bar{RST}  {res['bar_ts']:%Y-%m-%d %H:%M:%S} UTC   {DIM}build {build_s:.1f}s{RST}")
    lines.append("")
    lines.append(f"  {BOLD}PREDICTION{RST}   {color}{BOLD}{arrow} {d.upper():5}{RST}   "
                 f"confidence {color}{BOLD}{res['conf']:.3f}{RST}   (τ={tau})   {sig_txt}")
    lines.append("")
    # probability bars
    for name, p, c in (("down", res["p_down"], RED), ("flat", res["p_flat"], GRY), ("up", res["p_up"], GRN)):
        bar = "█" * int(round(p * 40))
        lines.append(f"    p({name:4}) {p:.3f}  {c}{bar}{RST}")
    lines.append("")
    warm = res["n_nan"]
    warn = RED if warm > 60 else YEL if warm > 20 else GRN
    lines.append(f"  {DIM}rows today {res['n_rows']:>4}   features {warn}{184 - warm}/184 present{RST}"
                 f"{DIM}  ({warm} NaN — warmup){RST}")
    lines.append("")
    lines.append(f"  {DIM}Ctrl-C to stop.{RST}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--interval", type=float, default=5.0, help="min seconds between cycles")
    ap.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    model_dir = Path(args.model)
    feat_cols, booster, scaler, dir_thr, manifest = load_bundle(model_dir)
    version = manifest.get("version", model_dir.name)

    from src.features.engine import FeatureEngine
    engine = FeatureEngine()

    i = 0
    while True:
        i += 1
        t0 = time.time()
        err, res = None, None
        try:
            today = datetime.now(timezone.utc).date()
            # Suppress the engine's verbose progress output; we own the screen.
            with contextlib.redirect_stdout(io.StringIO()):
                path = engine.build_date(today)
            if not path or not Path(path).exists():
                err = "no feature parquet built yet (insufficient live data)"
            else:
                res = predict_last_row(feat_cols, booster, scaler, dir_thr, Path(path))
                if res is None:
                    err = "feature parquet is empty"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        build_s = time.time() - t0

        print(CLR + render(res, args.tau, build_s, version, err), flush=True)

        if args.iterations and i >= args.iterations:
            break
        # pace to the requested interval (build already consumed some of it)
        time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
