# ADR-057: 5s mean MBP-10 depth imbalance (`mbp_`)

## Status
Accepted — implemented 2026-08-22.

## Date
2026-08-22

## Context

The Border Visualizer already computes per-snapshot L0–L9 size sums
(`bid_count`, `ask_count`) and `imbalance_diff` / `imbalance_log` from
`databento.mbp_10`. Those four series need to land on the 5-second parquet so
models can use the same definition.

MBP-10 is 13–19M book-changing events per day. Pulling the wide 60-level rows
into pandas for a session is not viable. Each row already holds the full
top-10 book after the event, so the 5s feature is an **average of those
snapshots**, not a reconstruction from deltas.

## Decision

1. **Prefix `mbp_`**, module `src/features/mbp_depth.py`, four columns:
   - `mbp_bid_count` — mean of `sum(bid_sz_00..09)` (NULL level → 0)
   - `mbp_ask_count` — mean of `sum(ask_sz_00..09)`
   - `mbp_imbalance_diff` — mean of `(bid_count − ask_count)` per event
   - `mbp_imbalance_log` — mean of `ln(bid_count) − ln(ask_count)`

2. **Denominator is the number of occurrences used in that average.** Bid /
   ask / diff use every event in `[T, T+5s)`. Log uses only events with both
   sums `> 0` (SQL `AVG(...) FILTER (WHERE …)`; pandas `mean` skipna). Empty
   bars are NaN.

3. **Server-side `time_bucket` + `AVG`.** The engine does not load raw MBP-10
   events. Query window is the session (`training_start`/`end`) on a full
   build, or `t_start`/`t_end` on live/patch. Partition key is `ts_recv`.

4. **Left-labeled 5s buckets** matching pandas `resample('5s')` / the rest of
   the matrix. No ffill: a bar with no MBP updates has no occurrences.

5. **Named view `mbp_depth` = `["mbp_"]`.** Not mixed into existing champion
   views. Dates outside the 1-month MBP-10 table are all-NaN.

6. **Patch** `python scripts/features/patch_parquets.py --modules mbp`
   (default range 2026-07-21 → 2026-08-20, the ingested window).

## Consequences

- Catalog v5.12 242 → **v5.13 246**.
- Live runtime keeps a tiny 5s-aggregated buffer, not raw L2.
- `imbalance_log` is **not** `ln(mean bid) − ln(mean ask)`; it is the mean of
  per-event logs (Jensen). Diff mean equals mean(bid) − mean(ask) when every
  event is included.

## Related

- [DATABASE_SCHEMA.md §2.8](../architecture/DATABASE_SCHEMA.md) — `databento.mbp_10`
- Border Visualizer `#/border` — same four stats, optional time-bucket mean
- [ADR-041](ADR-041-closed-bar-slower-features.md) — 5s grid labeling
