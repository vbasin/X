"""MBO large-order lifecycles, outcomes and reload chains (DB-free core).

Pure pandas/numpy on event frames from `databento.mbo`; the SQL constants below are what
callers run to obtain those frames. Shared by `scripts/research/mbo_large_orders.py` and the
WebUI chart overlay, so the definitions live in exactly one place.

Order lifecycle (per `order_id`): `A` places it, `M` changes size and/or price, `F` is a fill of
the resting order, `C` removes it. A fully filled order ends with `F` then a `C` of the same
size; CME records can share `(ts_recv, sequence)`, so "filled" is decided from the order's own
set of fills, never from record order. Snapshot rows (`flags & 32`) re-list live orders at each
UTC midnight and are ignored if the order was already seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ts_recv prunes 1-day chunks (partition column). ts_event is compress_orderby, so the
# same bounds let Timescale skip compressed batches. instrument_id is compress_segmentby:
# without it a 5-day window decompresses every instrument in those chunks (minutes) instead
# of NQU6/NQZ6 only (~40s). The two clocks differ by ms; a 5-day window does not miss rows.
# Do not drop the ts_recv predicate — that would scan every chunk.
_MBO_WIN = ("ts_recv >= %(t0)s AND ts_recv < %(t1)s "
            "AND ts_event >= %(t0)s AND ts_event < %(t1)s "
            "AND instrument_id = %(instrument_id)s")

ORDERS_SQL = f"""
WITH big AS (
    SELECT DISTINCT order_id FROM databento.mbo
    WHERE {_MBO_WIN} AND symbol = %(symbol)s
      AND action = 'A' AND size >= %(min_size)s
    UNION
    -- order_ids the size filter above cannot reach on its own, supplied by the caller:
    --   * icebergs (ICEBERG_SQL) -- their *displayed* size is below min_size by definition;
    --   * orders seeded from the resting-book checkpoint -- their Add predates this range, so
    --     without them here their later Modify/Fill/Cancel rows are never fetched and the chart
    --     draws them resting to the window edge forever. Real case: 6882390809813, a 400-lot ask
    --     added 2026-09-21 01:14 UTC and cancelled 06:19 UTC, drawn across the whole of 09-21.
    SELECT unnest(%(extra_ids)s::bigint[])
)
SELECT m.ts_recv, m.sequence, m.order_id, m.action, m.side, m.size, m.price,
       (m.flags & 32) > 0 AS snap
FROM databento.mbo m JOIN big b USING (order_id)
WHERE m.ts_recv >= %(t0)s AND m.ts_recv < %(t1)s
  AND m.ts_event >= %(t0)s AND m.ts_event < %(t1)s
  AND m.instrument_id = %(instrument_id)s
  AND m.symbol = %(symbol)s
  AND m.action IN ('A', 'M', 'C', 'F')
-- Ordered by ts_event, NOT ts_recv: ts_event is the matching engine's own clock and is what
-- disambiguates several records for one order inside a single (ts_recv, sequence) packet. Sorting
-- by ts_recv leaves those tied, and an invented action/size tiebreaker then scrambles them --
-- which made an ordinary decrementing order (6882403938353: 33 -1-> 32 -2-> 30 -7-> 23 -12-> 11
-- -8-> 3 -3-> C) look as though its size had been restored. ts_event makes the chain exact.
ORDER BY m.order_id, m.ts_event, m.sequence,
         CASE m.action WHEN 'A' THEN 0 WHEN 'F' THEN 1 WHEN 'M' THEN 2 ELSE 3 END
"""

# ADR-149: exchange-side removals. An order can leave the book without the trader ever cancelling
# it, and neither case shows up as a `C`:
#   * a real exchange Clear (`R` without F_SNAPSHOT) wipes the whole book -- observed weekly at
#     Sunday pre-open;
#   * a Day order purged at a session close just stops appearing in the next snapshot group, with
#     no message at all. Absence is the only signal.
# The chart needs both so it stops drawing a dead order to the edge of the window.
# Native iceberg / reserve orders. A CME iceberg carries a display quantity and a hidden remainder;
# the matching engine executes against the whole order but only ever publishes the display slice,
# refreshing it with an `M` after each fill. So the order map's `big` CTE -- which filters on the
# *displayed* Add size -- cannot see one by construction, and displaying small is the entire point:
# on 2026-09-28 (NQZ6) all 872 reserve-confirmed orders had a display under 20 lots, absorbing 7,311
# contracts, while zero had a display of 20+.
#
# Detection is proof-grade, not heuristic: `absorbed > the largest size ever displayed` can only
# happen if quantity existed without being shown. Two things make that comparison honest:
#   * it is per (order_id, PRICE), not per order. Reserve sits at a level. An order that re-prices
#     gets a fresh queue position, and attributing its absorbed total to every level it ever visited
#     paints iceberg marks at prices that never traded -- 374 of 2,226 orders flagged in one
#     2026-09-21..23 window had moved, one across 94 levels.
#   * it compares against max(A, M) size at that level rather than the first Add, so a trader who
#     merely *increases* their order is not misreported as hidden reserve.
# Worked example, NQU6 6881584016268: a 10-lot trade printed at 29300 and this order supplied 9 of it
# while showing 1, then took an `M` back to 1 (refreshed) in the same packet where three exhausted
# orders took a `C`. See ADR-149.
ICEBERG_SQL = f"""
WITH f AS (
    SELECT order_id, price, sum(size) AS absorbed, max(size) AS biggest_fill
    FROM databento.mbo
    WHERE {_MBO_WIN} AND symbol = %(symbol)s AND action = 'F'
    GROUP BY order_id, price
    HAVING sum(size) >= %(min_size)s
), d AS (
    SELECT m.order_id, m.price, max(m.size) AS disp
    FROM databento.mbo m
    WHERE m.ts_recv >= %(t0)s AND m.ts_recv < %(t1)s
      AND m.ts_event >= %(t0)s AND m.ts_event < %(t1)s
      AND m.instrument_id = %(instrument_id)s
      AND m.symbol = %(symbol)s
      AND m.action IN ('A', 'M') AND m.order_id IN (SELECT order_id FROM f)
    GROUP BY m.order_id, m.price
)
SELECT f.order_id, f.price, f.absorbed, f.biggest_fill, d.disp
FROM f JOIN d USING (order_id, price)
WHERE f.absorbed > d.disp
"""

CLEARS_SQL = f"""
SELECT ts_recv FROM databento.mbo
WHERE {_MBO_WIN} AND symbol = %(symbol)s
  AND action = 'R' AND (flags & 32) = 0
ORDER BY ts_recv
"""

SNAP_BOUNDARIES_SQL = f"""
SELECT DISTINCT ts_recv FROM databento.mbo
WHERE {_MBO_WIN} AND symbol = %(symbol)s
  AND action = 'A' AND (flags & 32) > 0
ORDER BY 1
"""

CANDLE_SQL = f"""
SELECT time_bucket('1 minute', ts_recv) AS t,
       first(price, ts_recv) AS o, max(price) AS h, min(price) AS l,
       last(price, ts_recv) AS c, sum(size)::bigint AS v
FROM databento.mbo
WHERE {_MBO_WIN} AND symbol = %(symbol)s
  AND action = 'T' AND price IS NOT NULL
GROUP BY 1 ORDER BY 1
"""

# ADR-147: what large orders were already resting when a window opened, so ORDERS_SQL's own
# `big` CTE (which only ever sees orders whose Add falls inside the queried range) doesn't miss
# them. Finds the newest checkpoint at or before t0 and returns its large open orders; the
# caller widens ORDERS_SQL's own start to that checkpoint's timestamp and feeds
# seed_from_snapshot()'s output into build_segments() alongside the real rows.
#
# ADR-149: reads mbo_resting_orders_v2. The original table's checkpoints seeded phantom orders --
# ones CME purged at a weekend close without ever sending a Cancel, which the old A/M/F/C-only
# book had no way to remove (808 of them, 27% of the book, as of 2026-09-28) -- and carried
# t_start values truncated to the last UTC-midnight snapshot. `t_start_exact` is selected so a
# future display change can render "resting >= Xh" for an order we only ever saw already resting.
RESTING_SNAPSHOT_SQL = """
WITH nearest AS (
    SELECT max(snapshot_ts) AS ts FROM databento.mbo_resting_orders_v2
    WHERE symbol = %(symbol)s AND snapshot_ts <= %(t0)s
)
SELECT s.order_id, s.side, s.price, s.size, s.t_start, s.t_start_exact, n.ts AS snapshot_ts
FROM databento.mbo_resting_orders_v2 s JOIN nearest n ON s.snapshot_ts = n.ts
WHERE s.symbol = %(symbol)s AND s.size >= %(min_size)s
"""


def seed_from_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    """Resting-snapshot rows (RESTING_SNAPSHOT_SQL's output) -> synthetic Add events in the same
    shape ORDERS_SQL produces, ready to concat with real rows before build_segments(). Each
    becomes one `snap=True` Add at the order's *true* origin time (`t_start` -- when it was
    really placed or last re-priced, which can be well before the checkpoint), preserving the
    real resting duration rather than displaying it as if it started at the checkpoint. This is
    exactly the shape build_segments() already treats as "pre-existing" (see its
    `pre=bool(r.snap)` handling, covered by test_mbo_orders.py's snapshot-relisting test);
    `sequence=-1` sorts it before any real same-instant event a caller unions in."""
    if snap.empty:
        return pd.DataFrame(columns=["ts_recv", "sequence", "order_id", "action", "side",
                                     "size", "price", "snap"])
    return pd.DataFrame({
        "ts_recv": snap["t_start"], "sequence": -1, "order_id": snap["order_id"],
        "action": "A", "side": snap["side"], "size": snap["size"], "price": snap["price"],
        "snap": True,
    })


def _exchange_death(g: pd.DataFrame, clears, boundaries) -> pd.Timestamp | None:
    """When (if ever) the exchange removed this order without a Cancel: the first real Clear, or
    the first snapshot boundary at which it should have been re-listed and was not. `None` means it
    was never purged within the window (ADR-149)."""
    if len(clears) == 0 and len(boundaries) == 0:
        return None
    seen_snap = {r.ts_recv for r in g.itertuples(index=False) if r.snap}
    first_ev = g["ts_recv"].iloc[0]
    cands = [c for c in clears if c > first_ev]
    cands += [b for b in boundaries if b > first_ev and b not in seen_snap]
    return min(cands) if cands else None


def build_segments(ev: pd.DataFrame, t1: pd.Timestamp, clears=(), boundaries=()) -> pd.DataFrame:
    """Order events -> one row per (order, price segment) with outcome on the last segment.

    `clears`/`boundaries` (ADR-149) let a still-open segment be closed where the *exchange* removed
    the order rather than the trader: a real Clear, or omission from a snapshot re-listing. Without
    them a Day order purged at a weekend close is drawn as resting to the edge of the window.
    """
    rows: list[dict] = []
    for oid, g in ev.groupby("order_id", sort=False):
        started = False
        prev_a, prev_s, prev_t = None, None, None
        filled = 0          # this price segment's own fills, reset whenever the order re-prices
        death = _exchange_death(g, clears, boundaries)
        fills = {(f.ts_recv, int(f.size)) for f in g.itertuples(index=False) if f.action == "F"}
        for r in g.itertuples(index=False):
            if r.action == "A":
                if started:
                    prev_a, prev_s, prev_t = "A", r.size, r.ts_recv
                    continue                      # snapshot re-listing of a live order
                started, side, size0, cur = True, r.side, int(r.size), int(r.size)
                disp = int(r.size)      # largest size this segment ever DISPLAYED here
                seg = dict(order_id=oid, side=side, price=float(r.price), size0=size0,
                           t_start=r.ts_recv, pre=bool(r.snap), t_end=None, end="")
            elif not started:
                continue                          # placed before the window: unknown
            elif r.action == "M":
                if r.price != seg["price"] and not np.isnan(r.price):
                    # Re-price = a fresh queue position at a new level. Fill accounting restarts
                    # with it, so activity is never attributed to a price it did not happen at --
                    # an order that walked 94 levels used to report its whole absorbed total at
                    # every one of them, painting "filled"/iceberg marks at prices that never
                    # traded. `filled` is therefore per segment, not per order.
                    seg.update(t_end=r.ts_recv, end="moved", filled=filled, disp=disp,
                               left=max(cur, 0))
                    rows.append(seg)
                    filled, disp = 0, int(r.size)
                    seg = dict(order_id=oid, side=side, price=float(r.price), size0=size0,
                               t_start=r.ts_recv, pre=False, t_end=None, end="")
                else:
                    disp = max(disp, int(r.size))
                cur = int(r.size)
            elif r.action == "F":
                # Only fills at THIS segment's own price count. An order can be an aggressor and a
                # resting order in one life: a marketable limit sweeps what it can (T records, plus
                # F records at the prices it took) and rests the remainder. Those aggressive fills
                # carry their own, different price, and booking them to the resting level inflated
                # it -- order 6882409490573 read 68 absorbed at 30561.50 when 51 executed there and
                # the other 17 were taken at 30569.50/69.75/70.00 one sequence before it re-priced.
                if float(r.price) == seg["price"]:
                    filled += int(r.size)
                    cur -= int(r.size)
            elif r.action == "C":
                after_fill = (r.ts_recv, int(r.size)) in fills
                done = after_fill or filled >= size0
                seg.update(t_end=r.ts_recv, end="filled" if done else "cancelled")
                # How much was still resting when it left. Taken from the venue's own records, not
                # from the running size: CME packs several F/M rows for one order into a single
                # (ts_recv, sequence) and their order inside that packet is arbitrary, so a running
                # counter is not trustworthy at the boundary. A completed order has nothing left by
                # definition; a genuine cancel states the quantity being pulled in the C itself.
                seg["filled"], seg["disp"] = filled, disp
                seg["left"] = 0 if done else int(r.size)
                rows.append(seg)
                started = False
                break
            prev_a, prev_s, prev_t = r.action, r.size, r.ts_recv
        if started:
            if death is not None and death > seg["t_start"]:
                # Exchange took it out: stop the line there instead of running it to the window
                # edge as if it were still live.
                seg.update(t_end=death, end="cleared", filled=filled, disp=disp, left=max(cur, 0))
            else:
                seg.update(t_end=t1, end="resting", filled=filled, disp=disp, left=max(cur, 0))
            rows.append(seg)
    seg_df = pd.DataFrame(rows)
    if not seg_df.empty:
        seg_df["filled"] = seg_df["filled"].fillna(0).astype(int)
    return seg_df


def classify(seg: pd.DataFrame, candles: pd.DataFrame, approach: float) -> pd.DataFrame:
    """Outcome class per final segment using 1-min trade candles as the price path."""
    idx = candles["t"].dt.tz_localize(None).to_numpy()      # UTC wall time, datetime64[ns]
    lo, hi, cl = candles["l"].to_numpy(), candles["h"].to_numpy(), candles["c"].to_numpy()
    out = []
    for r in seg.itertuples(index=False):
        i0 = np.searchsorted(idx, np.datetime64(r.t_start.replace(tzinfo=None), "ns"), "left")
        i1 = np.searchsorted(idx, np.datetime64(r.t_end.replace(tzinfo=None), "ns"), "right")
        if i1 <= i0:
            i1 = min(i0 + 1, len(idx))
        touched = bool((lo[i0:i1] <= r.price).any()) if r.side == "B" else bool((hi[i0:i1] >= r.price).any())
        last = cl[min(max(i1 - 1, 0), len(cl) - 1)]
        dist = (last - r.price) if r.side == "B" else (r.price - last)   # >0 = market away from order
        if r.end == "filled":
            cls = "FILLED"
        elif r.end == "resting":
            cls = "RESTING"
        elif r.end == "moved":
            cls = "MOVED"
        elif r.end == "cleared":
            cls = "CLEARED"      # exchange removed it; never a trader "pull" (ADR-149)
        elif touched:
            cls = "PULLED_AFTER_TOUCH"
        elif dist <= approach:
            cls = "PULLED_ON_APPROACH"
        else:
            cls = "PULLED_FAR"
        out.append((touched, float(dist), cls))
    seg = seg.copy()
    seg[["touched", "dist_end", "outcome"]] = pd.DataFrame(out, index=seg.index)
    return seg


def find_chains(seg: pd.DataFrame, candles: pd.DataFrame, dt_s: float, n_links: int,
                break_pts: float, min_secs: float = 0.0, min_abs: int = 0,
                ratio: tuple[float, float] = (0.5, 2.0)) -> pd.DataFrame:
    """Chains: filled order -> new same-side/same-price order of similar size within dt_s."""
    s = seg[seg["end"].isin(["filled", "cancelled", "resting"])].copy()
    s["t_start"] = pd.to_datetime(s["t_start"], utc=True)
    s["t_end"] = pd.to_datetime(s["t_end"], utc=True)
    filled = s[s["end"] == "filled"][["order_id", "side", "price", "size0", "filled", "t_end"]]
    starts = s[["order_id", "side", "price", "size0", "t_start"]].rename(
        columns={"order_id": "next_id", "size0": "next_size"})
    m = filled.merge(starts, on=["side", "price"])
    dt = (m["t_start"] - m["t_end"]).dt.total_seconds()
    r = m["next_size"] / m["size0"]
    m = m[(dt >= 0) & (dt <= dt_s) & (m["next_id"] != m["order_id"])
          & (r >= ratio[0]) & (r <= ratio[1])]
    m = m.sort_values(["order_id", "t_start", "next_id"]).drop_duplicates("order_id")
    nxt = dict(zip(m["order_id"], m["next_id"]))
    info = s.set_index("order_id")
    heads = sorted(set(nxt) - set(nxt.values()))
    ct = candles.set_index("t")
    out = []
    for h in heads:
        chain, cur = [h], h
        while cur in nxt and nxt[cur] not in chain:
            cur = nxt[cur]
            chain.append(cur)
        if len(chain) - 1 < n_links:
            continue
        rows = info.loc[chain]
        side, price = rows["side"].iloc[0], float(rows["price"].iloc[0])
        t_a = rows["t_start"].min()
        fl = rows[rows["end"] == "filled"]
        t_b = fl["t_end"].max() if len(fl) else rows["t_start"].max()
        w = ct.loc[t_a.floor("min"): t_b.floor("min")]
        pinned = bool(w["h"].max() <= price) if side == "A" else bool(w["l"].min() >= price)
        w5 = ct.loc[t_b.floor("min"): t_b + pd.Timedelta(minutes=5)]
        if side == "B":
            broke, mfe = w5["l"].min() <= price - break_pts, w5["h"].max() - price
        else:
            broke, mfe = w5["h"].max() >= price + break_pts, price - w5["l"].min()
        out.append(dict(side=side, price=price, t_start=t_a, t_end=t_b, orders=len(chain),
                        clip=int(rows["size0"].median()), absorbed=int(rows["filled"].sum()),
                        pinned=pinned, after="BROKE" if broke else "HELD",
                        mfe5m=float(mfe) if len(w5) else np.nan))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["secs"] = (df["t_end"] - df["t_start"]).dt.total_seconds()
    df.attrs["raw"] = len(df)
    df = df[(df["secs"] >= min_secs) & (df["absorbed"] >= min_abs)]
    return df.sort_values("t_start").reset_index(drop=True)
