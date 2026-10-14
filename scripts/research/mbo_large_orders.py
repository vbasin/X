#!/usr/bin/env python3
"""Large resting-order lifelines + reload (refill-at-same-price) zones over candles — prototype.

Every order >= --min-size contracts (from `databento.mbo`, by order_id) is drawn as a horizontal
line from placement to removal at its price:

    colour      bid = green, ask = red
    thickness   grows with log2(size); numbers printed on orders >= --label-min
    end marker  ● filled   ✕ cancelled   ▶ still resting (dotted line runs to the right edge)
    ↦           a modify that changed price ends one segment and starts the next

Outcome classes per order (price path from 1-min trade candles):
    FILLED               fully executed
    PULLED_AFTER_TOUCH   price traded at/through the level while it rested, then it was cancelled
    PULLED_ON_APPROACH   cancelled with price within --approach pts of it and never touched
    PULLED_FAR           cancelled while price was further away
    RESTING              still on the book at the end of the window

Reload chains: an order of >= --clip-min lots is fully FILLED, and within --chain-dt seconds a
new order of similar size (x0.5-2) appears at the SAME side and price; repeated >= --chain-n
times. Drawn as a shaded band at that price. ASK chain = passive selling / distribution at that
price (someone keeps re-offering as buyers lift it); BID chain = passive buying / accumulation.
`pinned` = price never traded through the level while the chain ran; `after` = HELD/BROKE over
the next 5 min (through = --break-pts). Level-turnover at the touch is NOT used: re-adds there
come from many participants and would flag every level near price.

Caveats: only displayed size is visible (icebergs show as many orders); "pulled on approach" is
behaviour, not intent; orders placed before the window are unknown unless a UTC-midnight
snapshot row lists them (drawn as hollow-start, "pre-existing").

Usage (times America/Chicago):
    python scripts/research/mbo_large_orders.py --symbol NQZ6 \\
        --start "2026-09-23 17:00" --end "2026-09-24 16:00" --out session.png
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.research.mbo_liquidity_heatmap import CT, _bar, _dsn
from src.webui.mbo_orders import (CANDLE_SQL, ORDERS_SQL, build_segments, classify,
                                  find_chains)

UTC = ZoneInfo("UTC")

def _fetch(conn, sql: str, params: dict) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def draw(seg, reloads, candles, args, t0, t1, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MultipleLocator

    c5 = (candles.set_index("t").resample(f"{args.candle_min}min")
          .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
          .dropna().reset_index())
    lo = np.floor(c5["l"].min() - args.pad)
    hi = np.ceil(c5["h"].max() + args.pad)

    def num(ts):
        return mdates.date2num(pd.to_datetime(ts, utc=True).dt.tz_convert(CT).dt.to_pydatetime())

    fig, ax = plt.subplots(figsize=(34, 19))
    fig.patch.set_facecolor("#0b0b10")
    ax.set_facecolor("#0b0b10")
    x0, x1 = mdates.date2num(t0), mdates.date2num(t1)
    ax.set_xlim(x0, x1 + (x1 - x0) * 0.015)
    ax.set_ylim(lo, hi)

    # y gridlines: minor every 10 pts, strong every 50; labels every 10 on BOTH sides
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", color="#9aa4bb", alpha=0.38, lw=0.9)
    for y in np.arange(np.ceil(lo / 50) * 50, hi, 50):
        ax.axhline(y, color="#e6ebf7", alpha=0.65, lw=1.6, zorder=1)
    ax.tick_params(axis="y", labelsize=10, labelcolor="#dfe6f5", labelleft=True, labelright=True)
    span_h = (t1 - t0).total_seconds() / 3600
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=CT) if span_h > 6
                               else mdates.MinuteLocator(byminute=range(0, 60, 10), tz=CT))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=CT))
    ax.tick_params(axis="x", labelsize=11, labelcolor="#dfe6f5")
    ax.grid(axis="x", color="#8a93a6", alpha=0.12, lw=0.6)
    for s in ax.spines.values():
        s.set_color("#444a5a")

    # candles (thin, behind the order lines)
    cx = num(c5["t"])
    w = args.candle_min / 1440.0
    for x, r in zip(cx, c5.itertuples()):
        col = "#9db4d8" if r.c >= r.o else "#d8a09d"
        ax.plot([x + w / 2] * 2, [r.l, r.h], color=col, lw=0.8, alpha=0.75, zorder=2)
        ax.add_patch(Rectangle((x + w * 0.2, min(r.o, r.c)), w * 0.6, max(abs(r.c - r.o), 0.25),
                               fc=col, ec="none", alpha=0.42, zorder=2))

    # reload zones
    for r in reloads.itertuples():
        xa, xb = mdates.date2num(r.t_start.tz_convert(CT)), mdates.date2num(r.t_end.tz_convert(CT))
        xb = max(xb, xa + 3 / 1440.0)
        col = "#ff9f1c" if r.side == "A" else "#b388ff"
        ax.add_patch(Rectangle((xa, r.price - 0.9), xb - xa, 1.8, fc=col, ec=col, alpha=0.33,
                               lw=1.2, zorder=3))
        ax.text(xb + 0.0004, r.price, f"↻ {'ASK' if r.side == 'A' else 'BID'} reload ×{r.orders} "
                f"(~{r.clip} lots, Σ{r.absorbed})  {'PINNED ' if r.pinned else ''}{r.after}",
                color=col, fontsize=9, fontweight="bold",
                va="center", zorder=8,
                bbox=dict(fc="#0b0b10", ec="none", alpha=0.6, pad=0.8))

    # orders
    inside = seg[(seg["price"] >= lo) & (seg["price"] <= hi)]
    far = seg[~seg.index.isin(inside.index)]
    xs, xe = num(inside["t_start"]), num(inside["t_end"])
    xr = ax.get_xlim()[1]
    lines, lws, cols = [], [], []
    for (a, b), r in zip(zip(xs, xe), inside.itertuples()):
        lines.append([(a, r.price), (b, r.price)])
        lws.append(min(1.8 + 2.2 * np.log2(max(r.size0, args.min_size) / args.min_size), 14.0))
        cols.append("#2ecc71" if r.side == "B" else "#ff4d4d")
    ax.add_collection(LineCollection(lines, linewidths=lws, colors=cols, alpha=0.88, zorder=5,
                                     capstyle="butt"))
    mv = inside[inside["end"] == "moved"]
    nxt = inside.sort_values("t_start").groupby("order_id")
    conn = []
    for oid, g in nxt:
        if len(g) < 2:
            continue
        g = g.sort_values("t_start")
        xa = num(g["t_end"]); ya = g["price"].to_numpy()
        xb = num(g["t_start"])
        for i in range(len(g) - 1):
            conn.append([(xa[i], ya[i]), (xb[i + 1], ya[i + 1])])
    if conn:
        ax.add_collection(LineCollection(conn, colors="#cfd6e6", linewidths=0.8, alpha=0.55,
                                         linestyles=":", zorder=4))
    for (a, b), r in zip(zip(xs, xe), inside.itertuples()):
        col = "#2ecc71" if r.side == "B" else "#ff4d4d"
        if r.end == "filled":
            ax.plot(b, r.price, "o", ms=7, mfc="#ffffff", mec=col, mew=1.6, zorder=7)
        elif r.end == "cancelled":
            ax.plot(b, r.price, "x", ms=6, color=col, mew=1.6, zorder=7)
        elif r.end == "moved":
            ax.plot(b, r.price, "|", ms=8, color=col, mew=1.4, zorder=7)
        elif r.end == "resting":
            ax.plot([b, xr], [r.price] * 2, ":", color=col, lw=1.6, alpha=0.9, zorder=5)
            ax.plot(xr, r.price, ">", ms=8, color=col, zorder=7)
        if r.pre:
            ax.plot(a, r.price, "o", ms=6, mfc="none", mec="#ffffff", mew=1.0, zorder=7)
        if r.size0 >= args.label_min and (b - a) * 1440 >= args.label_min_min:
            ax.text(a, r.price + 0.55, f"{r.size0}", color="#ffffff", fontsize=8.5,
                    fontweight="bold", ha="left", va="bottom", zorder=9,
                    bbox=dict(fc=col, ec="none", alpha=0.75, pad=0.6, boxstyle="round,pad=0.15"))

    # far-away orders: edge markers + counts
    for side_up, sub in ((True, far[far["price"] > hi]), (False, far[far["price"] < lo])):
        if sub.empty:
            continue
        ax.plot(num(sub["t_start"]), [hi - 1.2 if side_up else lo + 1.2] * len(sub),
                "^" if side_up else "v", ms=7, color="#ffd166", zorder=7)
        ax.text(x0, hi - 3.5 if side_up else lo + 3.0,
                f"{len(sub)} orders {'above' if side_up else 'below'} view "
                f"(≥{args.min_size}; {sub['price'].max() if side_up else sub['price'].min():,.2f} furthest)",
                color="#ffd166", fontsize=10, zorder=9)

    handles = [
        Line2D([0], [0], color="#2ecc71", lw=4, label="BID order (thickness ∝ log size)"),
        Line2D([0], [0], color="#ff4d4d", lw=4, label="ASK order"),
        Line2D([0], [0], marker="o", mfc="white", mec="#aaa", lw=0, ms=7, label="filled"),
        Line2D([0], [0], marker="x", color="#aaa", lw=0, ms=6, label="cancelled"),
        Line2D([0], [0], marker=">", color="#aaa", lw=0, ms=8, label="resting → extends right"),
        Line2D([0], [0], marker="o", mfc="none", mec="white", lw=0, ms=6, label="pre-existing at snapshot"),
        Patch(fc="#ff9f1c", alpha=0.5, label="ASK reload chain (re-offered after each fill)"),
        Patch(fc="#b388ff", alpha=0.5, label="BID reload chain (re-bid after each fill)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=4,
              fontsize=10, facecolor="#141420", edgecolor="#444a5a", labelcolor="#dfe6f5",
              framealpha=0.95)
    ax.set_ylabel("price (10-pt grid, bold line every 50)", color="#dfe6f5")
    ax.set_title(f"{args.symbol} large resting orders (≥{args.min_size} lots; numbers on ≥{args.label_min}) "
                 f"+ reload zones — {t0.astimezone(CT):%Y-%m-%d %H:%M} → {t1.astimezone(CT):%Y-%m-%d %H:%M} CT   "
                 f"[{len(seg):,} order segments, {len(reloads)} reload zones]",
                 color="#ffffff", fontsize=16, pad=14)
    fig.savefig(out, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="NQZ6")
    ap.add_argument("--start", required=True, help="America/Chicago 'YYYY-MM-DD HH:MM'")
    ap.add_argument("--end", required=True, help="America/Chicago, exclusive")
    ap.add_argument("--min-size", type=int, default=20)
    ap.add_argument("--label-min", type=int, default=50, help="print size on orders >= this")
    ap.add_argument("--label-min-min", type=float, default=1.0,
                    help="...and alive at least this many minutes")
    ap.add_argument("--approach", type=float, default=10.0, help="pts: 'pulled on approach'")
    ap.add_argument("--clip-min", type=int, default=5, help="smallest order tracked for reload chains")
    ap.add_argument("--chain-n", type=int, default=3, help="min consecutive re-adds after a fill")
    ap.add_argument("--chain-dt", type=float, default=2.0, help="max seconds fill -> re-add")
    ap.add_argument("--chain-min-secs", type=float, default=10.0,
                    help="chain must last this long (drops sweep bursts)")
    ap.add_argument("--chain-min-abs", type=int, default=50,
                    help="...and absorb at least this many contracts")
    ap.add_argument("--break-pts", type=float, default=1.0)
    ap.add_argument("--candle-min", type=int, default=5)
    ap.add_argument("--pad", type=float, default=12.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv-prefix", type=Path, default=None, help="also write <prefix>_orders/_reloads.csv")
    a = ap.parse_args()

    import psycopg
    t0 = datetime.fromisoformat(a.start).replace(tzinfo=CT).astimezone(UTC)
    t1 = datetime.fromisoformat(a.end).replace(tzinfo=CT).astimezone(UTC)
    p = {"symbol": a.symbol, "t0": t0, "t1": t1, "min_size": min(a.clip_min, a.min_size)}
    tic = time.time()
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        _bar("candles", 0.05)
        candles = _fetch(conn, CANDLE_SQL, p)
        if candles.empty:
            sys.exit(f"\nno trades for {a.symbol} in that window")
        _bar("large-order events", 0.15)
        ev = _fetch(conn, ORDERS_SQL, p)
    _bar("building segments", 0.85)
    ev["ts_recv"] = pd.to_datetime(ev["ts_recv"], utc=True)
    candles["t"] = pd.to_datetime(candles["t"], utc=True)
    seg = build_segments(ev, pd.Timestamp(t1))
    if seg.empty:
        sys.exit("\nno large orders found — lower --min-size or check the window")
    for c in ("t_start", "t_end"):
        seg[c] = pd.to_datetime(seg[c], utc=True)
    seg = classify(seg, candles, a.approach)
    reloads = find_chains(seg, candles, a.chain_dt, a.chain_n, a.break_pts,
                          a.chain_min_secs, a.chain_min_abs)
    seg = seg[seg["size0"] >= a.min_size].copy()   # tracked >= clip-min; draw/report >= min-size
    _bar("rendering", 0.92)
    draw(seg, reloads, candles, a, t0, t1, a.out)
    _bar("done", 1.0)

    print(f"\n✅ Done in {time.time() - tic:.0f}s -> {a.out}")
    last = seg[seg["end"] != "moved"]
    print(f"orders ≥{a.min_size}: {last['order_id'].nunique():,}   segments: {len(seg):,}   "
          f"reload chains flagged: {len(reloads)} "
          f"(raw {reloads.attrs.get('raw', len(reloads))} before the "
          f"≥{a.chain_min_secs:g}s / ≥{a.chain_min_abs} persistence floor)")
    print("\noutcome by side (final segments):")
    print(pd.crosstab(last["outcome"], last["side"], margins=True).to_string())
    if not reloads.empty:
        print("\nreload zones:")
        r = reloads.assign(t=reloads["t_start"].dt.tz_convert(CT).dt.strftime("%H:%M:%S"))
        print(r[["t", "side", "price", "orders", "clip", "absorbed", "secs", "pinned", "after", "mfe5m"]]
              .to_string(index=False))
    if a.csv_prefix:
        seg.to_csv(f"{a.csv_prefix}_orders.csv", index=False)
        reloads.to_csv(f"{a.csv_prefix}_reloads.csv", index=False)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/opt/vtech/.env")
    main()
