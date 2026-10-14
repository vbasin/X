# Discovery-only NQ source recovery — 2026-09-24

Two incomplete sessions were repaired in a **new research preparation artifact**,
before the retry selected its final policy or consulted trading validation.
The existing path cache, PostgreSQL database and original aggregate remain intact.

Among 2,288 calendar-eligible discovery sessions in 2011–2020, two original source
files lacked the final cash minute:

| Session | Original source rows | Last source timestamp UTC | Missing cash minutes |
|---|---:|---|---:|
| 2020-02-28 | 10,582 | 15:58:16 | 301 |
| 2020-06-30 | 6,045 | 14:10:56 | 349 |

Read-only, date-bounded, parameterized database queries confirmed the same ends
in `databento.ohlcv_1s`, including all symbol aliases of the matching instrument
IDs (10204 and 14028). EXPLAIN showed compressed-chunk/index filtering. The cache
was faithful to an incomplete upstream realization. No alternate 2020 NQ raw
file was found locally; other instruments' files cannot repair NQ executions.

## Recovery and mismatch detection

`scripts/research/repair_nq_discovery_gaps.py` checked Databento's cost immediately
before each request and refused any nonzero cost. Both narrow recovery requests
and both subsequent complete-session requests returned **exactly $0.00**. Cost
evidence and the vendor responses are retained. No paid retrieval occurred.

The narrow-window splice was rejected when exact overlapping OHLCV differed.
The recovery then fetched complete 07:00–15:15 Chicago sessions for both dates,
preserving the revised vendor source as a whole instead of mixing two versions
inside one session. Both replacement sessions contain all 390 cash minutes.

The vendor explicitly flags **both dates as degraded**. Complete-session recovery
does not remove that qualification or prove a perfect underlying event history.

The full-overlap audit found:

* February: 10,581 of 10,582 overlapping seconds match all OHLCV fields exactly.
  Only the original final second differs: its volume was 19 rather than 35,
  with a smaller high and different close. Zero timestamp shift gives the best
  match, consistent with a truncated final second.
* June: only 68 of 5,075 same-timestamp overlaps match all fields. Mean absolute
  close difference is 0.777 point; prices and volume both differ. Testing shifts
  from −3 through +3 seconds gives the best alignment when the **new timestamp
  is moved forward two seconds**: 2,518 of 5,849 overlaps match all fields and
  mean close error falls to 0.143 point. This is not a simple uniform one-second
  bar-open/bar-close relabeling. The positive shift aligns earlier new values
  with later old timestamps; it does not demonstrate future-dependent old
  features. These two degraded dates cannot certify all other historical dates.

## Final artifact and provenance

Final directory:
`data/research/runs/nq-edge-retry/preparation_20260924_long_history_refreshed_free_v2/`.

The finalized `bars_1m.parquet` has **1,726,174 rows** and SHA-256
`08a9f88e03529a77e56758ffd7df5253e00b86402ac4e00dfb07dc06cb917664`.
All other 3,537 sessions are frame-identical to the original aggregate, including
column types. An intermediate assembly caught unsigned vendor volume promoting
the canonical signed integer column to float; the completed build explicitly
checks the range and casts vendor volume to signed int64 before aggregation.

`build_manifest.json` embeds original/new raw-source hashes, exact cost evidence
references, whole-session replacement flags, the degraded-data caveat, per-field
agreement, and all timestamp-offset diagnostics. `source_inventory.parquet`
identifies the two new complete-session source paths; the `repairs/` directory
contains raw vendor data, merged source frames and overlap evidence. Failed
splice/assembly attempts are preserved in separate preparation directories.

This is a new auditable source realization, not a reason to retroactively drop
losing sessions or claim trading profitability. Model selection must rerun on
this version before frozen validation is consulted.
