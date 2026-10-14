#!/usr/bin/env python3
"""MBO event-flow heatmap (price x time) behind a candlestick chart — prototype.

Three stacked panels over the same candles, each a price-x-time grid of order-event
volume aggregated server-side from `databento.mbo` (never raw rows):

    Added      action 'A'  — resting liquidity placed
    Cancelled  action 'C'  — resting liquidity pulled
    Filled     action 'F'  — resting liquidity actually executed (passive side)

Candles come from action 'T' (aggressor trade records); their per-minute volume equals
`databento.tbbo` exactly (verified 2026-09-26: 78,378 trades / 112,751 contracts for
NQZ6 2026-09-24 13:30-14:30 UTC). Snapshot rows (`flags & 32`) are book state, not
events, and are excluded. 'M' (modify) is ignored: the record carries the new size, so
the delta needs book state — this is an event-flow view, not a reconstructed book.

Usage (times are America/Chicago):
    python scripts/research/mbo_liquidity_heatmap.py --symbol NQZ6 \\
        --start "2026-09-23 17:00" --end "2026-09-24 16:00" --out session.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CT = ZoneInfo("America/Chicago")
ACTIONS = {"A": "Added", "C": "Cancelled", "F": "Filled"}

FLOW_SQL = """
SELECT time_bucket(%(bucket)s::interval, ts_recv) AS t,
       floor(price / %(step)s) * %(step)s          AS p,
       action,
       sum(size)::bigint                           AS qty
FROM databento.mbo
WHERE ts_recv >= %(t0)s AND ts_recv < %(t1)s
  AND symbol = %(symbol)s
  AND action IN ('A', 'C', 'F')
  AND (flags & 32) = 0
  AND price IS NOT NULL
GROUP BY 1, 2, 3
"""

CANDLE_SQL = """
SELECT time_bucket(%(bucket)s::interval, ts_recv) AS t,
       first(price, ts_recv) AS o, max(price) AS h, min(price) AS l,
       last(price, ts_recv)  AS c, sum(size)::bigint AS v
FROM databento.mbo
WHERE ts_recv >= %(t0)s AND ts_recv < %(t1)s
  AND symbol = %(symbol)s AND action = 'T' AND price IS NOT NULL
GROUP BY 1 ORDER BY 1
"""


def _dsn() -> str:
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    return (
        f"host={os.environ.get('PGHOST', '127.0.0.1')} "
        f"port={os.environ.get('PGPORT', '5432')} "
        f"dbname={os.environ['PGDATABASE']} "
        f"user={os.environ['PGUSER']} "
        f"password={os.environ['PGPASSWORD']}"
    )


def _bar(stage: str, frac: float) -> None:
    n = int(20 * frac)
    print(f"\r[{'█' * n}{'░' * (20 - n)}] {stage:<28}", end="", flush=True)


def load(symbol: str, t0: datetime, t1: datetime, bucket_min: int, step: float
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Server-side aggregates: (flow[t,p,action,qty], candles[t,o,h,l,c,v])."""
    import psycopg
    params = {"symbol": symbol, "t0": t0, "t1": t1, "step": step,
              "bucket": f"{bucket_min} minutes"}
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            _bar("candles (T records)", 0.1)
            cur.execute(CANDLE_SQL, params)
            candles = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            _bar("order-event flow (A/C/F)", 0.35)
            cur.execute(FLOW_SQL, params)
            flow = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    return flow, candles


def grids(flow: pd.DataFrame, t_index: pd.DatetimeIndex, p_index: np.ndarray, step: float
          ) -> dict[str, np.ndarray]:
    """Dense (price, time) matrices per action, price ascending."""
    out = {}
    ti = pd.Series(np.arange(len(t_index)), index=t_index)
    for act in ACTIONS:
        g = np.zeros((len(p_index), len(t_index)), dtype=np.float64)
        f = flow[flow["action"] == act]
        pi = np.rint((f["p"].to_numpy(dtype=float) - p_index[0]) / step).astype(int)
        ci = ti.reindex(f["t"]).to_numpy()
        ok = (pi >= 0) & (pi < len(p_index)) & ~np.isnan(ci)
        np.add.at(g, (pi[ok], ci[ok].astype(int)), f["qty"].to_numpy(dtype=float)[ok])
        out[act] = g
    return out


def draw(candles: pd.DataFrame, mats: dict[str, np.ndarray], t_index: pd.DatetimeIndex,
         p_index: np.ndarray, step: float, symbol: str, bucket_min: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    t_ct = t_index.tz_convert(CT)
    x0 = mdates.date2num(t_ct.to_pydatetime())
    w = bucket_min / 1440.0
    extent = (x0[0], x0[-1] + w, p_index[0], p_index[-1] + step)

    fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.06})
    cmaps = {"A": "viridis", "C": "magma", "F": "cividis"}
    cndl = candles.assign(x=mdates.date2num(candles["t"].dt.tz_convert(CT).dt.to_pydatetime()))
    for ax, (act, name) in zip(axes, ACTIONS.items()):
        g = mats[act]
        pos = g[g > 0]
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        vmin = max(1.0, np.percentile(pos, 5)) if pos.size else 1.0
        im = ax.imshow(np.where(g > 0, g, np.nan), origin="lower", aspect="auto",
                       extent=extent, cmap=cmaps[act], norm=LogNorm(vmin=vmin, vmax=vmax),
                       interpolation="nearest")
        for r in cndl.itertuples():
            up = r.c >= r.o
            col = "#ffffff" if up else "#ff5c5c"
            ax.plot([r.x + w / 2] * 2, [r.l, r.h], color=col, lw=0.6, alpha=0.9)
            ax.add_patch(Rectangle((r.x + w * 0.15, min(r.o, r.c)), w * 0.7,
                                   max(abs(r.c - r.o), step * 0.25),
                                   facecolor=col, edgecolor="none", alpha=0.9))
        ax.set_ylabel(f"{name}  (price, {step:g}-pt bins)")
        cb = fig.colorbar(im, ax=ax, pad=0.005, fraction=0.025)
        cb.set_label("contracts / bin (log)")
        ax.set_facecolor("#0b0b10")
        ax.grid(alpha=0.12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=CT))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=CT))
    axes[-1].set_xlabel("Chicago time")
    fig.suptitle(
        f"{symbol} MBO event flow — {t_ct[0]:%Y-%m-%d %H:%M} → {t_ct[-1]:%Y-%m-%d %H:%M} CT, "
        f"{bucket_min}-min buckets; candles from trade records", y=0.905, fontsize=14)
    fig.savefig(out, dpi=110, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="NQZ6")
    ap.add_argument("--start", required=True, help="America/Chicago, 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--end", required=True, help="America/Chicago, exclusive")
    ap.add_argument("--minutes", type=int, default=1, help="time bucket (default 1)")
    ap.add_argument("--price-step", type=float, default=1.0, help="price bin in points")
    ap.add_argument("--pad", type=float, default=25.0, help="points beyond session high/low")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    t0 = datetime.fromisoformat(a.start).replace(tzinfo=CT)
    t1 = datetime.fromisoformat(a.end).replace(tzinfo=CT)
    tic = time.time()
    flow, candles = load(a.symbol, t0, t1, a.minutes, a.price_step)
    if candles.empty:
        sys.exit(f"\nno trades for {a.symbol} in that window — is it ingested? (databento.mbo)")

    _bar("building grids", 0.7)
    lo = np.floor((candles["l"].min() - a.pad) / a.price_step) * a.price_step
    hi = np.ceil((candles["h"].max() + a.pad) / a.price_step) * a.price_step
    p_index = np.arange(lo, hi, a.price_step)
    t_index = pd.date_range(t0.astimezone(ZoneInfo("UTC")), t1.astimezone(ZoneInfo("UTC")),
                            freq=f"{a.minutes}min", inclusive="left")
    mats = grids(flow, t_index, p_index, a.price_step)

    _bar("rendering", 0.9)
    draw(candles, mats, t_index, p_index, a.price_step, a.symbol, a.minutes, a.out)
    _bar("done", 1.0)

    tot = {ACTIONS[k]: int(v.sum()) for k, v in mats.items()}
    print(f"\n✅ Done in {time.time() - tic:.0f}s -> {a.out}")
    print(f"   contracts  added={tot['Added']:,}  cancelled={tot['Cancelled']:,}  "
          f"filled={tot['Filled']:,}  traded(T)={int(candles['v'].sum()):,}")
    print(f"   cancelled/added = {tot['Cancelled'] / max(tot['Added'], 1):.1%}   "
          f"filled/added = {tot['Filled'] / max(tot['Added'], 1):.2%}")


if __name__ == "__main__":
    main()
