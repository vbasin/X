# Data Pipeline

> **Present**: Three-tier DBN → TimescaleDB → Parquet (+ greeks/surface side paths); live options are the five-chain raw_symbol set (ADR-034). **One `db.Live` session per CME week; the session date advances at 16:05 CT and the archive rotates on it, one directory per session date (ADR-155).** Vendor definitions are read/acquired by raw symbol alongside subscriptions; rotated archives replay mapping records (ADR-156). Bounded historical capture recovery uses exact-zero-cost immutable archives and full-payload append reconciliation (ADR-157). CME Sunday/Friday timers (ADR-060). `databento.mbp_10` is empty (chunks dropped 2026-09-30; DDL kept). NQ `tbbo_15s` / `tbbo_5m` / `tbbo_1h` / `tbbo_1d` caggs (ADR-067). 5s parquet `of_` includes tick OFI (`of_ofi`, ADR-068); `cs_` includes 5s OHLCV (`cs_close` = `_close`, ADR-069). 5s hive monthly parts under `sources/features/timeframe=5s/` (ADR-079); native 15s 1-bar hive under `timeframe=15s/` (ADR-075); native 1m hive under `timeframe=1m/` (ADR-085); native 5m hive under `timeframe=5m/` (ADR-090); native 1h / 1d HTF-context hive under `timeframe=1h/` and `timeframe=1d/` (ADR-091).
> **Supersedes**: Live synthetic definition fallback and header-only rotation (ADR-156).
> **Next**: Resolve vendor-source coverage and causal option feature timing identified by October recovery (ADR-157). Observe the next real session roll with vendor acquisition and mapping replay (ADR-156). Serving asof for `tf_oof_meta` (ADR-093). Storage capacity / backfill under ROADMAP P5.
> **Source of truth for**: acquisition patterns, ingestion flow, raw data layout, Parquet cache spec.  
> **See also**: [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md); [SYSTEM.md](SYSTEM.md); [WORKFLOW.md](WORKFLOW.md).

---

## 1. Three-Tier Data Architecture

```
Tier 1 — Raw DBN Archives
  data/raw/backfill/**/*.dbn.zst   ← Immutable archive (Databento binary format)
  data/raw/live/**/*.dbn.zst       ← Live streaming archives

Tier 2 — TimescaleDB (vtech_market, schema: databento)
  tbbo              ← Futures TBBO: L1 trade + quote ticks
  bbo_1s            ← Options & futures best bid/offer at 1-second intervals
  definitions       ← Instrument metadata (strike, expiry, underlier, multiplier)
  statistics        ← Daily settlement, open interest, high/low
  equity_ohlcv      ← Static equity daily bars (unused in v5 pipeline)
  tbbo_1min (cagg)  ← 1-minute bars from tbbo (auto-refresh on demand)
  tbbo_15s (cagg)   ← NQ outrights, 15s from tbbo ticks (ADR-067)
  tbbo_5m (cagg)    ← NQ outrights, clock 5-minute from tbbo_1min (ADR-067)
  tbbo_1h (cagg)    ← NQ outrights, clock hour from tbbo_1min (ADR-067)
  tbbo_1d (cagg)    ← NQ outrights, Globex session [T-1 17:00, T 17:00) CT (ADR-067)
  nq_option_greeks_1s   ← Full 0DTE C/P Greeks (two-sided), ~330 MB/day compressed
  nq_option_surface_1s  ← Compact 8-leg delta-bucket surface, 1 row/sec
  mbp_10            ← NQ lead L2 top-10 book hypertable (empty as of 2026-09-30; DDL kept)
                      (5s parquet `mbp_*` is SQL AVG of L0–L9 size — ADR-057)

Tier 3 — Parquet Cache
  sources/features/parquet_5s/features_YYYY-MM-DD.parquet  ← leftover day files (not written; ADR-081)
  sources/features/timeframe=5s/symbol=NQ/year=/month=/features_5s_partN.parquet
      ← canonical 5s hive (ADR-079); 08:00–11:00 CT; datetime `bar_end_ts`; ≤100 MB/part
  data/parquet_raw1s/features_YYYY-MM-DD.parquet  ← raw NQ 1s + VIX for LSTM
  data/parquet_options_surface/features_YYYY-MM-DD.parquet  ← 0DTE surface (57 feats + meta)
  data/parquet_options_greeks/greeks_YYYY-MM-DD.parquet     ← full 0DTE Greeks day archive
  sources/features/timeframe=15s/symbol=NQ/year=/month=/features_15s_partN.parquet
      ← native 15s 1-bar hive (ADR-075); Globex hours; datetime `bar_end_ts`
  sources/features/timeframe=1m/symbol=NQ/year=/month=/features_1m_partN.parquet
      ← native 1m hive (ADR-085); so_/cs_/ps_/vw_/rv_/vp_/ms_; Globex; `bar_end_ts`
  sources/features/timeframe={5m,1h,1d}/symbol=NQ/year=/month=/
      ← native-TF hive (ADR-070); dirs only until those builders write
```

Each tier is **fully regenerable** from the tier above it. Rebuild priority: Parquet from DB; DB from raw DBN.

---

## 2. Acquisition Patterns

### 2.1 Batch Backfill (Historical API)

Used for: initial backfill and large date ranges (>5 GB).

```python
# scripts/databento_backfill.py
client = db.Historical(api_key)
client.batch.submit_job(
    dataset="GLBX.MDP3",
    symbols=["NQ.FUT"],
    schema="tbbo",          # or "bbo-1s", "definition", "statistics"
    stype_in="continuous",
    start="2025-04-01",
    end="2026-04-07",
)
# Job produces .dbn.zst file(s) in Databento cloud; download within 30 days
```

**Cost check before submitting** (always):
```bash
python scripts/databento_cost_check.py --symbol NQ.FUT --schema tbbo --start 2025-04-01
```

Batch download caveats:
- Submitted jobs bill once; unlimited re-downloads for 30 days. See [reference/DATABENTO_PRICING.md](../reference/DATABENTO_PRICING.md).
- Recommended for options data and anything >5 GB.

### 2.2 0DTE Weekly Options (Two-Phase Download)

**Phase 1** — Symbol map (run once):
```python
# scripts/download_nq_0dte_weekly.py
# For each of 24 weekly parent symbols (Q1A–Q5D, QN1–QN4):
#   timeseries.get_range(schema="definition") → in-memory only
#   group by expiration.dt.date → {iso_date: [raw_symbol, ...]}
#   save to data/raw/backfill/nq_0dte_weekly/symbol_map.json
```

**Phase 2** — Per-date BBO-1s download:
```python
# For each of 227 expiry dates:
#   timeseries.get_range(schema="bbo-1s", stype_in="raw_symbol", symbols=[...])
#   output: data/raw/backfill/nq_0dte_weekly/YYYY-MM-DD.dbn.zst
#   checkpoint.json tracks done/failed dates for safe resume
```

| Weekly parent symbols (24 total) | Expiry day |
|----------------------------------|-----------|
| Q1A.OPT – Q5A.OPT | Monday |
| Q1B.OPT – Q5B.OPT | Tuesday |
| Q1C.OPT – Q5C.OPT | Wednesday |
| Q1D.OPT – Q5D.OPT | Thursday |
| QN1.OPT – QN4.OPT | Friday |

> **⚠️ Note**: Phase 1 was run in-memory only — weekly definitions were **not** written to `databento.definitions`. Phase 17C must ingest them before `iv_0dte_atm` and `iv_term_slope` can be reverted to true 0DTE formulas.

Additional ad hoc option backfills use the same underlying contracts:

| Backfill | Parent/symbol selection | Purpose |
|----------|-------------------------|---------|
| Quarterly | `NQ.OPT` parent | Front quarterly and other NQ quarterly quotes; required by ≤90 DTE GEX |
| 0DTE | Exact raw symbols from all 24 daily-weekly parents plus `QNE.OPT`, filtered to expiry=session | `iv_0dte_atm` + 0DTE Greeks |
| 1DTE | The same raw symbols, downloaded on the prior CME trading session | Today's `iv_1dte_atm` from contracts expiring next trading day |
| End-of-month | `QNE.OPT` parent | Next two QNE expiries for 30-day constant-maturity IV |

The 24 daily-weekly parents are `Q1A.OPT`–`Q5D.OPT` and `QN1.OPT`–`QN4.OPT`.
`QNE.OPT` is a separate end-of-month parent, not a 25th weekly parent.

### 2.3 Live Streaming (Live API)

Used for: daily real-time data accumulation.

```python
# src/acquisition/live_streamer.py
# Streams: NQ.FUT (tbbo + bbo-1s + statistics), ES.FUT (tbbo)
# Options bbo-1s + definitions: session five-chain raw_symbol
#   0DTE, 1DTE, closest QNE, 2nd closest QNE, closest quarterly
# Archives to: data/sources/raw/live/<Globex session date>/stream_<HHMMSS>.dbn
#   Session date = the trading day the open session SETTLES into, advancing at
#   16:05 CT (not Chicago midnight). One db.Live connection spans the week;
#   RotatingDbnStream opens a new file at each roll (ADR-155).
# Injects to DB on the fly via connection pool
```

Each Chicago session streams **exactly those five expiries**. Strike lists come from
Databento `symbology.resolve` and are subscribed as `raw_symbol`
([ADR-034](../decisions/ADR-034-five-chain-live-option-subscribe.md)).

The connection is **held for the week**, Sunday open to Friday close
([ADR-155](../decisions/ADR-155-weekly-live-session-and-rotating-archive.md)). A
session roll is additive — the new chains are subscribed on the live connection
rather than the client being restarted — and it happens at 16:05 CT, inside the
16:01–16:44 CT window where no quote is written. The previous Chicago-midnight
teardown cost 10–50 s of unrecoverable quotes four times a week. Expired chains
stay subscribed (Databento Live has no unsubscribe API) and write ~100
`UNDEF_PRICE` teardown rows before going silent at 16:00:01 CT.
The 26-parent tuple in `src.acquisition.options_universe` remains the historical /
repair list only. Extra weeklies, back quarterlies, and `UD:` combos are not live-subscribed.

Databento instrument IDs are session-scoped and may be reused. The streamer therefore
bootstraps only the stable set of known raw symbols for definition deduplication; live
definition and symbol-mapping records populate the current-session ID-to-symbol cache.
Historical instrument IDs are never used to label current-session BBO records.

`DefinitionAcquirer` reads usable vendor definitions from the table by raw symbol
and obtains absent, legacy synthetic or incomplete terms in parallel with quote
capture. The exact historical request must be quoted at $0.00 before downloading;
failures and unavailable terms remain pending for five-minute retries. Vendor
terms bind to current live IDs and must be published at or before the quote.
Quotes continue while calculations wait for terms; no synthetic rows are created.
Contract size uses `unit_of_measure_qty`, and missing vendor ticks are not invented.
See [ADR-156](../decisions/ADR-156-normal-option-definition-acquisition.md) and
[operational checks](../operations/NORMAL_OPTION_DEFINITION_LIFECYCLE.md).

Each rotated DBN file replays cached symbol-mapping records immediately after its
header. Their raw bytes, aliases and validity intervals preserve resolution for
subscriptions retained across the session roll, without querying definitions.


Service: `systemd/vtech-live-streamer.service` plus CME timers
(`Sun 17:00` / `Fri 16:00 America/Chicago`). A weekend boot exits 0 without
opening Databento ([ADR-060](../decisions/ADR-060-vm-relative-limits-and-cme-live-hours.md)).

---

## 3. Ingestion Pipeline

### 3.1 DBN → TimescaleDB

```python
# src/ingestion/loader.py
# 1. Read .dbn.zst with databento.DBNStore
# 2. Convert to pandas DataFrame
# 3. Normalize column names to DB schema
# 4. Batch INSERT / COPY via psycopg3
# 5. Dedup on (instrument_id, ts_event) if re-ingesting
```

**Futures (tbbo)**: `scripts/ingest_tbbo_chunked.py` — chunk size 500,000 rows.

**Futures L2 (mbp-10, NQU6)**: `scripts/data/ingest_mbp10.py` — DBN Rust CSV encoder → FIFO → `COPY`, then `compress_chunk` per UTC day. Originally 27 files (2026-07-21 → 2026-08-20) → **438,266,822 rows** in 120.9 min at 3 workers. On 2026-09-30 all 27 chunks were dropped (`drop_chunks`) to free ~16 GB for an NQU6 MBO backfill; the 27 raw `.dbn.zst` files (12.796 GB) were deleted the same day. Hypertable DDL remains. Job metadata only: `data/sources/raw/backfill/nq_fut_mbp10/GLBX-20260821-WLJAD4WJ5C/`. The Live Chart MBP-10 ladder that read this table through `src/webui/book.py` was removed (ADR-148). The module remains on disk and is not mounted on the console.

**Futures L3 (mbo, NQU6 → NQZ6)**: `scripts/data/backfill_mbo.py` (cost gate → day-split batch → download) then `scripts/data/ingest_mbo.py` (same FIFO → `COPY` → `compress_chunk` path as mbp-10). 29 day files (2026-08-23 → 2026-09-24) → **475,152,566 rows**, 6.5 GB compressed, $0.00. Extended 2026-10-01 by a paid NQU6 pull of 2026-06-14 → 2026-08-26 (**1,162,963,092 rows**, 63 day files, $113.19 of an approved $125 budget), so `databento.mbo` now spans **2026-06-14 → 2026-10-01**; every segment's row count matched Databento's own `get_record_count` exactly. A further NQM6 pull on 2026-10-01 added June 7/9/10/11 (**113,342,826 rows**, $10.64), so coverage now starts **2026-06-07** and those four days hold NQM6 while 06-14 onward holds NQU6 — the first date-varying symbol coverage in this table. **Caveat:** `ingest_mbo.py`'s per-file idempotency `DELETE` is scoped to `ts_recv` only, with no instrument filter, so re-ingesting a different symbol for an already-populated day would silently drop the day's existing rows; scope that DELETE before any multi-symbol same-day pull. 60 new day-chunks, all compressed at 11.1× (169 GB → 15 GB). Raw archive: `data/sources/raw/backfill/nq_fut_mbo/`. See [DATABASE_SCHEMA.md §2.9](DATABASE_SCHEMA.md).

`backfill_mbo.py` gates a *paid* pull on four things before any batch job is submitted (added 2026-09-30 for the NQU6 June–August backfill). Defaults are unchanged, so `daily_mbo_pull.py` and every free-window call keep the unconditional $0.00 behaviour of AGENTS.md §5:

| Flag | Default | Effect |
|---|---|---|
| `--max-cost USD` | `0.0` | Cumulative budget across all segments. At the default any non-zero quote aborts with nothing submitted; a non-zero value must have been explicitly approved by the user for that specific pull. Re-checked against the remaining budget immediately before each segment's `submit_job`. |
| `--require-key KEY` | unset | Aborts unless `DATABENTO_API_KEY` matches exactly, so a pull cannot silently bill a different account. |
| `--min-free-gb GB` | `15.0` | Preflight on `--out-dir`'s filesystem, plus a re-check before each later segment. Also refuses a plan whose projected raw + hypertable footprint would breach the floor. |
| `--ingest-each` | off | Ingest each segment right after its own download instead of downloading everything first — bounds peak raw-on-disk and surfaces an ingest failure before the remaining segments are billed. |

Space projection constants, measured over the 2026-08-23–2026-09-30 load (557.6M rows): **17.9 bytes/record** as `.dbn.zst` (3.13× zstd on the 56 B/record DBN wire format) and **~14.4 bytes/record** in the compressed hypertable (~11× columnar compression). One uncompressed day-chunk peaks at ~4.3 GB before `ingest_mbo.py` compresses it, so peak transient is roughly `--workers` × 4.3 GB — the reason `--workers 2` is preferred over the default 3 on this volume. `scripts/ops/nqu6_mbo_jun_aug_backfill.sh` pins the resulting parameter set for the NQU6 2026-06-14–2026-08-23 pull ($109.18 quoted, 1,162,963,092 records, ~20.8 GB raw + ~16.8 GB hypertable).

**MBO** (`databento.mbo`): true live streaming (`LiveStreamer` subscribing to `schema="mbo"`) remains built but disabled — the live gateway rejects this account's `mbo` subscription (`'Not authorized for mbo schema'`; historical access does not imply live access), pending a Databento entitlement fix (ADR-146). In the meantime (ADR-147, 2026-09-29), **`vtech-mbo-incremental.timer`** keeps `databento.mbo` current automatically: every 30 minutes, Sunday session open through Friday close (gated by `cme_weekend_closed()`, same check the live streamer uses), `scripts/ops/mbo_incremental_pull.py` cost-checks and appends `[checkpoint, now − 8h01m)` via `timeseries.get_range()` — append-only, never a `DELETE`, checkpointed on `databento.mbo`'s own `max(ts_recv, sequence)`. The same tick also writes a dated checkpoint of every currently-resting order into `databento.mbo_resting_orders_v2` (§2.10 in DATABASE_SCHEMA.md; "current state" is just its newest checkpoint, not a separate table), backfilled from MBO's own inception, so `mbo_orders.py`/the Live Chart order-map overlay can correctly show a large order that predates the visible window instead of missing it. Since 2026-09-30 (ADR-149) that book also honours `R` (Clear) and **reconciles against each snapshot re-listing**, which is what finally closes orders CME purges at the weekend close without ever sending a `Cancel` — the defect that had left 808 phantom orders (27% of the book) in the original `mbo_resting_orders`, now frozen as a rollback path. `vtech-mbo-daily.timer` (the older nightly batch catch-up, ADR-146) stays retired; `scripts/ops/daily_mbo_pull.py`/`backfill_mbo.py` remain the manual/occasional whole-day gap-fill tools, now used for filling any hole that predates the incremental timer's own start. Logs: `logs/mbo-incremental.log` / `.err`.

**Options (bbo-1s)**: `scripts/ingest_nq_opt.py --glob "*.dbn.zst"` — parallel workers (4), COPY for throughput.
- 0DTE weekly ingestion: 227 files × ~12M rows avg, 295 min wall time, ~50K rows/s.

**Definitions**: `scripts/ingest_nq_opt.py --schema definition` (same script, different schema).

### 3.2 Ingestion Quality Checks

```python
# src/ingestion/quality.py
# - Row count before/after vs expected
# - Date range coverage gaps
# - instrument_id completeness in definitions
# - ts_event monotonicity check
```

After ingestion, verify with:
```bash
python scripts/databento_verify.py --schema bbo-1s --date-range 2025-04-06:2026-04-24
```

### 3.3 Purged Data

Phase 17B purged `instrument_class IN ('T', 'M')` rows from `bbo_1s`:
- Removed: 500,382,858 rows (T/M class combo rows — `UD:2V:` vertical spread symbols)
- These are NOT vanilla C/P options; cannot be used for IV computation
- Script: `scripts/purge_ud_combos.py`

---

## 4. Raw Data Inventory (as of 2026-04-29)

| Path | Size | Contents | Status |
|------|------|----------|--------|
| `data/raw/backfill/nq_fut/` | 2.8 GB | NQ.FUT tbbo, bbo-1s, definitions, statistics | ✅ ingested |
| `data/raw/backfill/nq_opt/` | 53 GB | NQ.OPT quarterly BBO-1s + definitions | ✅ ingested |
| `data/raw/backfill/nq_0dte_weekly/` | ~38 GB | 0DTE weekly BBO-1s (227 files) | ✅ BBO-1s ingested; ⚠️ definitions NOT ingested |
| `data/raw/live/` | 584 MB | Live DBN archives (3 days) | ✅ ingested |

---

## 5. Parquet Cache Specification

Each day's features are materialized as a single Parquet file:

```
data/parquet/features_YYYY-MM-DD.parquet
  Rows:    1,080 (360 bars/hr × 3hr session: 08:00–11:00 CST)
  Columns: 185
    ├── 183 feature columns (all float32)
    ├── _meta_nq_contract  (string, e.g. "NQZ5")  — excluded from training
    └── _meta_es_contract  (string, e.g. "ESZ5")  — excluded from training
  Index:   DatetimeTZDtype (UTC, 5-second frequency)
```

Accompanying manifest:
```json
// data/parquet/features_YYYY-MM-DD.parquet.manifest.json
{
  "filename": "features_2025-10-15.parquet",
  "sha256": "...",
  "row_count": 1080,
  "col_count": 185,
  "size_bytes": ...,
  "columns": { "col_name": "dtype", ... }
}
```

### 5.1 Current Cache State

| Period | Trading Days | Status |
|--------|-------------|--------|
| 2025-10-01 → 2025-10-31 | 23 | ✅ Complete |
| 2025-11-03 → 2025-11-28 | 18 | ✅ (Nov 21 missing — OOM) |
| 2025-12-01 → 2025-12-31 | 19 | ✅ (Dec 15–18 missing — OOM; Dec 19 holiday) |
| 2026-04-02, 04-03, 04-05 | 3 | ✅ Live/recent builds |
| **Total** | **63 files present** | **All at 185 columns** |

> **⚠️ Rebuild pending** after Phase 17D reverts `iv_0dte_atm`/`iv_term_slope`. Current cache uses interim quarterly-based IV values.

OOM-killed dates (5 total): Nov 21, Dec 15-18, 2025. Mitigation: `VTECH_BBO_PRIMARY_ONLY=1` limits `bbo_1s` query to NQ futures + options within 400-day expiry window. All 5 now build successfully under this flag. See [operations/CACHE_REBUILD.md](../operations/CACHE_REBUILD.md).

### 5.2 Column Naming Convention

Feature columns are prefixed by module. The trainer's `_`-prefix filter excludes metadata columns automatically. No columns named `label` appear in the Parquet — labels are computed at training time from forward price moves.

---

## 6. Data freshness and bounded recovery

The October 6 recovery restored the reviewed NQ/ES parent-futures capture gaps, including later outrights and calendar spreads, and independently accepted delivered records (ADR-157). A bounded live check at 18:25 UTC found NQZ6 and ESZ6 TBBO records through 18:25:49 and 18:25:51 respectively. The live streamer is active.

October 5 22:50–October 6 00:00 remains a vendor-source coverage gap. Restoring current streaming does not fill earlier historical gaps automatically; dates outside the reviewed recovery require their own inventory. Full 5-second feature restoration and live scoring remain gated by timing and memory issues documented in the recovery results.

---

## Related Docs

- [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) — full SQL DDL for all tables
- [operations/BACKFILL.md](../operations/BACKFILL.md) — how to run backfill
- [operations/CACHE_REBUILD.md](../operations/CACHE_REBUILD.md) — how to rebuild Parquet cache
- [reference/DATA_SCHEMAS.md](../reference/DATA_SCHEMAS.md) — exact column types
- [ADR-001](../decisions/ADR-001-5s-timestep.md) — why 5-second timestep

## Bounded capture-gap recovery — October 2026 (ADR-157)

**Present:** `scripts/data/recover_databento_gaps.py` executes the exact-symbol request manifest linked in `docs/operations/DATABENTO_GAP_BACKFILL_PLAN_2026_10_06.md`. It preserves existing archives and stages each downloaded archive completely in session-local TEMP tables before reconciliation. Raw archives and SHA256 receipts live under `raw_dir()/backfill/gaps_20261002_20261006/`; operational job receipts live under `_jobs/`.

Every new vendor request uses current per-schema availability and a fresh finite exact-$0.00 quote for identical API parameters. Downloads use host-sized parallelism and retain a 20 GiB reserve. Frames preserve nanosecond source timestamps in raw DBN; the DB adapter matches existing live microsecond rounding. Generated BBO spread is excluded from INSERT. Legacy undefined prices/quantities/reference timestamps are normalized in comparison projections without rewriting old rows.

Complete-file staging and full-payload `EXCEPT ALL` preserve repeated source occurrences and same-key distinct events. A recovery-only advisory lock serializes each target's merges while independent staging runs in parallel. Target windows are read once into an indexed TEMP snapshot, avoiding repeated hourly scans of unindexed BBO chunks. Missing occurrences are appended in bounded time segments; retries reconcile already committed segments. A statistics unique-key payload conflict fails explicitly. No UNLOGGED conversion, index removal, target deletion, or explicit chunk compression/decompression is used.

Vendor definition archives retain publication history. Current definitions are keyed by raw symbol and only missing/synthetic/incomplete rows are upgraded through the existing vendor provenance guard. Downstream historical Greeks use dated publication-as-of terms, the quote's historical instrument ID, and exact as-of underlying prices. The mutable current definitions table is not a historical instrument-ID join. Offline definition availability is separate from original publication nanoseconds: ordinary records use the later event/receive timestamp; weekday-midnight snapshots retain prior publication availability. The October recovery audit confirms ordinary messages preceded their quote sessions.

**Supersedes:** blind replay of overlapping windows through non-deduplicating COPY for this recovery workflow only. Existing historical loaders and their paths remain available.

Accepted October recovery totals and downstream publication receipts are in [the recovery results](../operations/DATABENTO_GAP_RECOVERY_RESULTS_2026_10_06.md): all 39 archives passed independent acceptance, and closed futures/context plus six option products were published. Option publication preserves affected DB/file originals, merges analyzed TEMP candidates hourly and validates accepted keys/material payloads; legacy missing-number representations are classified separately without compressed rewrites.

**Next:** resolve the vendor-source interval and validate live-feature memory limits. Existing 30-second option and 5-second underlying last-price feature timestamp semantics need a separate causal fix; the 5-second recovery branch emits blocked manifests without invoking the feature engine or producing Parquet.

Independent acceptance uses `scripts/data/validate_gap_recovery.py`: SHA256/header/mapping checks, exact decoded row and symbol/hour populations, indexed or time-bounded server aggregates, archived ingestion receipts, and successful zero-insert retries. Vendor count metadata is supplementary: [Databento documents](https://databento.com/docs/api-reference-historical/metadata/metadata-get-record-count) 10-minute accuracy boundaries (24 hours for definitions), so partial-window estimates are not treated as exact delivery counts. Aligned interior comparisons are recorded separately.
