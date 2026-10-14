# Database Schema

> **Present**: Definitions are keyed by raw symbol; live vendor acquisition supplies terms alongside subscriptions (ADR-156).
> **Supersedes**: Live quote-derived synthetic insertion under ADR-151.
> **Next**: Observe coverage through the next live session roll; historical inactive synthetic rows remain.


> **Source of truth for**: hypertable DDL, indexes, compression policies, continuous aggregates, TimescaleDB config
> **Database**: `vtech_market`, schema `databento`
> **Stack**: PostgreSQL 18.3 + TimescaleDB 2.26.1

---

## 1. Row Counts (as of 2026-05-19)

| Table | Rows | Date Range | Size (compressed) |
|-------|------|-----------|------------------|
| `databento.tbbo` | 93,133,627 | 2025-04-06 → 2026-04-05 | — |
| `databento.bbo_1s` | ~6,887,000,000 | 2025-04-06 → 2026-05-05 | 53 GB |
| `databento.definitions` | ~1,067,403 | 2025-03-30 → 2030-12-20 | — |
| `databento.statistics` | 1,028,670 | 2025-04-06 → 2026-04-05 | — |
| `databento.equity_ohlcv` | 469,178 | 2025-04-01 → 2026-04-02 | — |
| `databento.ohlcv_1s` | 275,785,685 | 2010-06 → 2026-05 | 3.1 GB (23x compression) |
| `databento.mbp_10` | 0 | — (emptied 2026-09-30) | 16 kB; table/DDL kept, all 27 chunks dropped |
| `databento.mbo` | 497,471,975 | 2026-08-23 → 2026-09-25 | ~6.9 GB (14.4 B/row; NQU6 → NQZ6 L3, 30 daily chunks; 2026-09-25 added 22,319,409 rows) |

### `bbo_1s` breakdown

| Contents | Rows |
|----------|------|
| NQ futures | ~42,000,000 |
| NQ quarterly options (C/P) | ~1,012,000,000 |
| NQ 0DTE weekly options (C/P) | ~2,729,000,000 |
| NQ 1DTE weekly options (C/P) | ~3,102,000,000 (ingested Phase 17I, 2026-05-14) |
| T/M class combos (purged Phase 17B) | 0 (500,382,858 removed) |
| Far-expiry quarterly options >90 DTE (purged Phase 17H) | 0 (~527,000,000 removed) |

> **Phase 17H disk crisis recovery** (2026-05-14): `scripts/purge_far_quarterly.py` ran across 5 workers covering
> 2025-04-01 → 2026-05-01, deleting all NQ option rows with `expiry_date > session_date + 90` from `bbo_1s`.
> ~527M rows removed. Disk recovered from 95% → 34% full (373 GB free). Hypertable total size: 267 GB → ~41 GB.
>
> **Phase 17I 1DTE ingest** (2026-05-14): 226 files, 3,102M rows in 601 min (batch-decompress mode).
> `bbo_1s` total after ingest: **6,887M rows** (53 GB compressed, 336 chunks). Disk: 36% (360 GB free).

### `definitions` breakdown

| Contents | Rows |
|----------|------|
| NQ futures contracts | ~1,200 |
| NQ quarterly options (C/P) | ~75,000 |
| NQ spreads | ~63,000 |
| NQ 0DTE weekly options (added Phase 17C) | ~246,875 (total table: ~1,067,403 rows) |
| NQ QNE end-of-month options | 0 — pending Phase 17I ingest |

---

## 2. Table DDL

### 2.1 `databento.tbbo` — Trade + BBO Ticks

```sql
CREATE TABLE databento.tbbo (
    ts_event       TIMESTAMPTZ    NOT NULL,  -- Trade event timestamp (exchange)
    ts_recv        TIMESTAMPTZ    NOT NULL,  -- Receive timestamp (Databento gateway)
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    price          DOUBLE PRECISION,         -- Trade price (USD, scaled by min_tick)
    size           INTEGER,                  -- Trade quantity (contracts)
    side           CHAR(1),                  -- 'B' (buyer-initiated) or 'A' (seller-initiated)
    sequence       BIGINT,
    -- BBO snapshot at time of trade
    bid_px         DOUBLE PRECISION,
    ask_px         DOUBLE PRECISION,
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    bid_ct         INTEGER,                  -- Bid order count
    ask_ct         INTEGER,                  -- Ask order count
    ts_in_delta    INTEGER,                  -- Matching engine latency (ns)
    dataset        TEXT           DEFAULT 'GLBX.MDP3'
);

SELECT create_hypertable('databento.tbbo', 'ts_event',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_tbbo_symbol_ts ON databento.tbbo (symbol, ts_event DESC);
-- idx_tbbo_instrument_ts removed (2026-05-18): unused, saves ~5–6 GB post-ingest
-- UNIQUE(instrument_id, ts_event, sequence) constraint removed (2026-05-19): CME same-price
--   sweep fills can produce legitimately duplicate (instrument_id, ts_event, sequence) tuples
--   (multiple fills at the same price level within a single matching event). Constraint dropped
--   before full table re-ingest; table truncated and reloaded from scratch. NQ outrights only;
--   93,133,627 rows from NQ_FUT_tbbo_2025-04-06_2026-04-06_dbn_zst (588,272 calendar-spread
--   rows filtered). tbbo_1min cagg manually refreshed 2025-04-06→2026-04-07 after ingest.
```

### 2.2 `databento.bbo_1s` — Best Bid/Offer at 1-Second Intervals

```sql
CREATE TABLE databento.bbo_1s (
    ts_recv        TIMESTAMPTZ    NOT NULL,  -- Receive timestamp (partition key)
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    price          DOUBLE PRECISION,         -- Last trade price in the 1-second window
    size           INTEGER,                  -- Last trade size
    bid_px         DOUBLE PRECISION,         -- Best bid at interval close
    ask_px         DOUBLE PRECISION,         -- Best ask at interval close
    bid_sz         INTEGER,
    ask_sz         INTEGER,
    -- Computed spread
    spread         DOUBLE PRECISION GENERATED ALWAYS AS (ask_px - bid_px) STORED,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_recv)
);

SELECT create_hypertable('databento.bbo_1s', 'ts_recv',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

### 2.3 `databento.definitions` — Instrument Metadata

```sql
CREATE TABLE databento.definitions (
    ts_event             TIMESTAMPTZ    NOT NULL,
    instrument_id        INTEGER        NOT NULL,
    symbol               TEXT           NOT NULL,
    instrument_class     CHAR(1),                   -- 'C'all, 'P'ut, 'F'uture
    strike_price         DOUBLE PRECISION,
    expiration           TIMESTAMPTZ,                -- PM-settled weeklies: 15:00 CT
    underlying           TEXT,                       -- Underlying FUTURE (NQZ6), not the root
    exchange             TEXT,
    currency             TEXT           DEFAULT 'USD',
    min_price_increment  DOUBLE PRECISION,
    multiplier           DOUBLE PRECISION,
    dataset              TEXT           DEFAULT 'GLBX.MDP3',
    source               TEXT           NOT NULL DEFAULT 'databento',
    PRIMARY KEY (symbol),
    CONSTRAINT definitions_source_chk CHECK (source IN ('databento', 'synthetic'))
);

CREATE INDEX idx_def_expiration       ON databento.definitions (expiration);
CREATE INDEX idx_def_source_synthetic ON databento.definitions (symbol)
    WHERE source = 'synthetic';
```

**`underlying` is the underlying futures contract, not the product root.** NQ futures list
quarterly months only (H/M/U/Z), so a serial or weekly option settles into the first quarterly
still trading at its expiry instant — an October weekly settles into NQZ6, and a 15:00 CT weekly
expiring *on* a 3rd Friday outlives the 08:30 CT futures fixing and rolls to the next quarterly.
Never derive it from the option symbol's own month code.

**`source` records provenance.** `'databento'` is an authoritative vendor definition.
`'synthetic'` identifies a legacy quote-derived row. The live streamer no longer creates
such rows. Normal acquisition reads usable vendor rows by symbol and obtains
missing/legacy synthetic/incomplete terms alongside subscription. Vendor rows replace
synthetic rows; usable vendor rows remain protected. An incomplete core vendor row may
be corrected by an equal/newer vendor publication. Historical `instrument_id` is not a
current gateway mapping. Contract size uses vendor `unit_of_measure_qty`; missing ticks
remain null on new rows, with usable legacy ticks preserved on upgrade. See
[ADR-156](../decisions/ADR-156-normal-option-definition-acquisition.md).

### 2.4 `databento.statistics` — Daily Settlement / OI / High / Low

```sql
CREATE TABLE databento.statistics (
    ts_event       TIMESTAMPTZ    NOT NULL,
    ts_recv        TIMESTAMPTZ    NOT NULL,
    instrument_id  INTEGER        NOT NULL,
    symbol         TEXT           NOT NULL,
    stat_type      SMALLINT       NOT NULL,  -- 1=Opening, 3=Settlement, 4=Session low, 5=Session high, 6=Cleared volume, 9=Open interest
    price          DOUBLE PRECISION,
    quantity       BIGINT,
    sequence       BIGINT,
    ts_ref         TIMESTAMPTZ,              -- Reference date for the stat
    update_action  SMALLINT,                 -- 1=New, 2=Delete
    stat_flags     SMALLINT,
    dataset        TEXT           DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event, stat_type, sequence)
);

SELECT create_hypertable('databento.statistics', 'ts_event',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);
```

### 2.5 `databento.ohlcv_1s` — NQ + ES 1-Second OHLCV (Historical Backfill)

Covers NQ and ES continuous futures from 2010 through 2026. Populated by `scripts/ingest_ohlcv_1s.py` from
384 `.dbn.zst` files in `data/raw/backfill/{nq,es}_fut_ohlcv_1s/` (2.81 GB compressed). Created 2026-05-19
(Phase 17K). Chunk interval is **1 month** (not 1 day) — 15 yr × 12 mo ≈ 180 chunks is manageable.

```sql
CREATE TABLE databento.ohlcv_1s (
    ts_event       TIMESTAMPTZ      NOT NULL,
    instrument_id  INTEGER          NOT NULL,
    symbol         TEXT             NOT NULL,
    open           DOUBLE PRECISION,
    high           DOUBLE PRECISION,
    low            DOUBLE PRECISION,
    close          DOUBLE PRECISION,
    volume         BIGINT,
    dataset        TEXT             NOT NULL DEFAULT 'GLBX.MDP3',
    UNIQUE (instrument_id, ts_event)
);

SELECT create_hypertable('databento.ohlcv_1s', 'ts_event',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX idx_ohlcv_1s_instrument_ts ON databento.ohlcv_1s (instrument_id, ts_event DESC);
CREATE INDEX idx_ohlcv_1s_symbol_ts     ON databento.ohlcv_1s (symbol, ts_event DESC);

ALTER TABLE databento.ohlcv_1s SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_event ASC'
);
SELECT add_compression_policy('databento.ohlcv_1s', INTERVAL '30 days', if_not_exists => TRUE);
```

**Download scripts**: `scripts/download_nq_ohlcv_1s.py` (submitted 17 NQ jobs) +
`scripts/download_es_ohlcv_1s.py` (submitted 17 ES jobs) +
`scripts/download_ohlcv_parallel.py` (parallel file download, 16 threads, 67s for 384 files).
**Ingest script**: `scripts/ingest_ohlcv_1s.py` — ProcessPoolExecutor(12), COPY pattern,
checkpoint at `data/checkpoints/ingest_ohlcv_1s.json`.

### 2.6 `databento.nq_option_greeks_1s` — Full 0DTE Per-Option Greeks (1s)

App-computed (not DB-generated) Black–Scholes Greeks for every two-sided NQ 0DTE C/P quote.
One compute path shared by historical backfill (`scripts/backfill_options_greeks.py`) and the
live streamer. ~7–8M rows/day; **~330 MB/day compressed** (segmentby `instrument_id`). Full
~285-day history needs ~90–100 GB free — expand disk before a full backfill.

```sql
CREATE TABLE databento.nq_option_greeks_1s (
    ts              TIMESTAMPTZ       NOT NULL,
    instrument_id   INTEGER           NOT NULL,
    symbol          TEXT              NOT NULL,
    expiry          DATE              NOT NULL,
    strike          DOUBLE PRECISION  NOT NULL,
    is_call         BOOLEAN           NOT NULL,
    underlying_sym  TEXT              NOT NULL,
    nq_price        DOUBLE PRECISION,
    bid_px, ask_px, mid, spread, iv, delta, gamma, vega, theta  DOUBLE PRECISION,
    UNIQUE (instrument_id, ts)
);
SELECT create_hypertable('databento.nq_option_greeks_1s', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
-- compress_segmentby = instrument_id; policy compress_after 3 days
-- (backfill also compresses each day immediately after COPY)
```

### 2.7 `databento.nq_option_surface_1s` — Compact 0DTE Delta-Bucket Surface (1s)

One row per second: 8 legs (`c10,p10,c25,p25,c40,p40,catm,patm`) × 7 fields
(`iv,delta,gamma,vega,theta,mid,spread`) + `nq_price` + meta. LSTM-ready fixed-width
representation (keys by delta bucket, not strike). ~30–40k rows/day; trivial storage.
Mirrored to `data/parquet_options_surface/features_YYYY-MM-DD.parquet`.

```sql
CREATE TABLE databento.nq_option_surface_1s (
    ts TIMESTAMPTZ PRIMARY KEY,
    expiry DATE NOT NULL,
    underlying_sym TEXT NOT NULL,
    n_legs_filled SMALLINT NOT NULL,
    nq_price DOUBLE PRECISION,
    -- 56 leg columns: {c10,p10,c25,p25,c40,p40,catm,patm}_{iv,delta,gamma,vega,theta,mid,spread}
    ...
);
SELECT create_hypertable('databento.nq_option_surface_1s', 'ts',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);
```

**Pipeline**: `src/features/options_greeks.py` (compute) +
`src/features/options_greeks_pipeline.py` (day load/COPY/parquet) +
`scripts/backfill_options_greeks.py`. Live dual-write in `src/acquisition/live_streamer.py`
(disable with `VTECH_LIVE_GREEKS=0`). Full Greeks also mirrored to
`data/parquet_options_greeks/greeks_YYYY-MM-DD.parquet` (zstd).

> **Stored rows from 2026-07-15 onward are invalid — rebuild before use**
> ([ADR-153](../decisions/ADR-153-options-iv-underlying-and-solver.md)). `NQ.c.0` stopped
> being ingested into `ohlcv_1s` on 2026-07-14, after which the underlying price silently fell
> back to a mix of every NQ outright (1,166 points of spread on 2026-09-29), and the IV solver
> reported its 0.3 seed as a solution on up to 24% of rows. `iv_*` features, dealer-GEX and the
> surface parquet mirror share the defect. 2025-04 → 2025-12 predates it and is sound; there is
> no surface data for 2026-01 → 2026-07.

**Underlying price**: each option is priced against the contract named in its own
`databento.definitions.underlying` row (`_load_underlying_prices_1s`), never against a
continuous series — `NQ.c.0` is not a valid forward for an option, least of all across a roll.
Any price read from `bbo_1s` must be bounded **above** as well as below: the DBN `UNDEF_PRICE`
sentinel (9223372036.854776) is positive and survives a `px > 0` filter.

### 2.8 `databento.mbp_10` — Market-by-Price, Top 10 Levels (L2)

Full-depth-of-book L2 for the **NQ lead future**. Databento `mbp-10` schema (DBN `rtype` 10,
368 bytes/record). One row per book-changing event — ~13–19M rows per trading day.

MBP-10 sits in Databento's **L2 tier: 1-month rolling free window** (see
[DATABENTO_PRICING.md](../reference/DATABENTO_PRICING.md)). Anything older is pay-as-you-go,
so this table is a rolling 1-month window, not a growing archive.

```sql
CREATE TABLE databento.mbp_10 (
    ts_recv        TIMESTAMPTZ NOT NULL,  -- capture-server receive (partition key)
    ts_event       TIMESTAMPTZ NOT NULL,  -- matching-engine event time
    instrument_id  INTEGER     NOT NULL,
    symbol         TEXT        NOT NULL,
    publisher_id   SMALLINT,              -- GLBX.MDP3 = 1
    rtype          SMALLINT,              -- always 10
    action         CHAR(1),               -- A=Add C=Cancel M=Modify R=Clear T=Trade F=Fill N=None
    side           CHAR(1),               -- B=Bid A=Ask N=None
    depth          SMALLINT,              -- 0–9, level where the update occurred
    price          DOUBLE PRECISION,      -- triggering order/trade price
    size           INTEGER,
    flags          SMALLINT,              -- 128=LAST 64=TOB 32=SNAPSHOT 16=MBP_AGG
                                          --   8=BAD_TS_RECV 4=MAYBE_BAD_BOOK
    ts_in_delta    INTEGER,               -- ns; matching-engine send latency before ts_recv
    sequence       BIGINT,                -- uint32 venue sequence — BIGINT, exceeds INTEGER range
    -- 60 flat level columns: {bid,ask}_{px,sz,ct}_{00..09}
    --   *_px DOUBLE PRECISION, *_sz / *_ct INTEGER
    dataset        TEXT        NOT NULL DEFAULT 'GLBX.MDP3'
);   -- 75 columns

SELECT create_hypertable('databento.mbp_10', 'ts_recv',
    chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE);

ALTER TABLE databento.mbp_10 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_recv ASC, sequence ASC'
);
SELECT add_compression_policy('databento.mbp_10', INTERVAL '30 days', if_not_exists => TRUE);
```

Full DDL with per-column DBN type annotations: [`scripts/data/create_mbp_10_table.sql`](../../scripts/data/create_mbp_10_table.sql).

#### Design decisions

| Decision | Why |
|---|---|
| **Flat-wide levels** (60 cols), not arrays | TimescaleDB compresses each column independently (Gorilla for float8, delta-delta for int). An array column is an opaque varlena blob and defeats that entirely. Matches the 56-flat-column precedent in `nq_option_surface_1s`. |
| **`DOUBLE PRECISION` prices**, not native int64 1e-9 | Lossless on the CME grid: every NQ tick is a multiple of 0.25 and \|px_nanos\| ≈ 2.4e13 ≪ 2^53 ≈ 9.0e15. Keeps types consistent with `tbbo` / `bbo_1s` so feature modules need no cast. |
| **Partition on `ts_recv`** | The timestamp Databento sorts MBP records by, and the only one safe for book reconstruction. `ts_event` kept as a plain column. |
| **`sequence BIGINT`** | Venue sequence is `uint32` (max 4.29e9) and overflows `INTEGER`. |
| **No UNIQUE constraint** | Same lesson as `tbbo` (2026-05-19): CME same-price sweep fills produce legitimately duplicate `(instrument_id, ts_event, sequence)` tuples. A unique index over ~438M rows would also cost ~15–20 GB. Idempotency comes from the ingest script deleting each day's `ts_recv` range before COPY. |
| **No extra indexes** | `create_hypertable` already indexes `ts_recv`. At this row count every extra index costs GBs (cf. `idx_tbbo_instrument_ts`, dropped 2026-05-18), and with a single instrument a `symbol` index is dead weight. |

#### Known fidelity limit — nanosecond → microsecond

DBN timestamps are **nanosecond**; `TIMESTAMPTZ` is **microsecond**. Postgres rounds
**half-up**, so `ts_recv` / `ts_event` carry ≤500 ns of error (verified:
`'…257500Z'` → `…258`, `'…257499Z'` → `…257`). Event order is still fully recoverable
from `sequence`. **The `.dbn.zst` files are the lossless archive of record** — keep them
(or push to OCI) rather than treating this table as the primary copy.

Absent book levels arrive as SQL `NULL`: the DBN CSV encoder emits `UNDEF_PRICE` (2^63−1)
as an empty field, verified by byte-patching a real record. No sentinel `9.22e9` prices leak in.

**Console reader (Live Chart optional ladder, `#/chart`)**: `src/webui/book.py` never
`COUNT(*)`s the hypertable. Coverage comes from `timescaledb_information.chunks`.
Playback windows are capped at **30 minutes** and downsampled with
`time_bucket` + `last()` on `ts_recv` (measured ~0.8s for 5 minutes × 10 levels
on compressed chunks). Snapshots are `ORDER BY ts_recv DESC, sequence DESC LIMIT 1`
with a 5s lookback so they hit the `ts_recv` partition key. Full RTH scans are
rejected — 13–19M rows/day is not an interactive query.

**5s parquet (`mbp_*`, ADR-057)**: `src/features/engine.py::_load_mbp10_depth_bars`
uses `time_bucket` + `AVG` of L0–L9 size sums (and of per-event
`ln(bid)−ln(ask)` with `FILTER` for the log). No raw event pull.

#### Measured characteristics

| Metric | Value |
|---|---|
| Compression ratio | **12.8×** full month (218 GB → 17.0 GB). A 2.28M-row sample was 20.7×; the month's TOAST dominates. |
| Compressed size | **38.8 bytes/row** (16 GB / 438,266,822) |
| Ingest throughput | 438,266,822 rows in **120.9 min** (60,420 rows/s aggregate, 3 workers), including per-day `compress_chunk` |
| 1 month, NQ lead | **438,266,822 rows / 16 GB** compressed, 27/27 daily chunks compressed. Verified 2026-08-21: COPY rows = Databento `get_record_count` for every day (delta 0); SQL `COUNT(*)` spot-check on 2026-08-16, 2026-07-29, 2026-08-20 matches. |
| Current (2026-09-30) | **Empty.** `drop_chunks('databento.mbp_10')` removed all 27 chunks (~16 GB). The 27 raw `.dbn.zst` files (12.796 GB) under `data/sources/raw/backfill/nq_fut_mbp10/` were deleted the same day. Hypertable DDL, compression settings, and `ingest_mbp10.py` remain. |

#### Pipeline

```bash
# 1. submit batch job (verify $0.00 first — L2 free window is only 1 month)
PYTHONPATH=/opt/vtech python scripts/data/databento_cost_check.py \
    --symbols NQU6 --schema mbp-10 --stype-in raw_symbol --start 2026-07-21

# 2. wait for the job, then download its day files
python scripts/data/download_mbp10.py --job GLBX-20260821-WLJAD4WJ5C

# 3. ingest (day-split files -> one chunk each, compressed straight after COPY)
python scripts/data/ingest_mbp10.py \
    --data-dir /opt/vtech/data/sources/raw/backfill/nq_fut_mbp10 --workers 3
```

`ingest_mbp10.py` streams `.dbn.zst` → **DBN Rust CSV encoder** → FIFO → `COPY`, with no
Python per-record loop and no multi-GB temp CSV. The encoder's CSV header is a name-for-name
match with this table, so COPY needs no transform step. Measured **4.7× faster** than the
pandas `to_df()` path used by `ingest_dbn.py` (91.8k vs 19.7k rows/s on CSV generation),
which matters at 438M rows: ~80 min vs ~370 min single-worker.

> **`show_chunks()` gotcha**: it only returns chunks falling *entirely* within
> `(newer_than, older_than)`, so mid-day bounds match nothing and post-COPY compression
> silently no-ops. `ingest_mbp10.py` snaps both bounds to whole UTC days.

---

### 2.9 `databento.mbo` — Market-by-Order, Full Depth (L3)

Order-level event stream for the **NQ lead future**. Databento `mbo` schema (DBN `rtype` 160,
56 bytes/record). One row per order event (add / cancel / modify / trade / fill / clear) —
~15–25M rows per full trading day. Loaded 2026-09-25: **475,152,566 rows, 2026-08-23 → 2026-09-24**,
NQU6 through the 2026-09-13 roll, NQZ6 from 2026-09-13.

MBO sits in Databento's **L3 tier: 1-month rolling free window** (same as MBP-10; see
[DATABENTO_PRICING.md](../reference/DATABENTO_PRICING.md)). Older history is pay-as-you-go, so
this table is a rolling window unless a daily capture is added.

```sql
CREATE TABLE databento.mbo (
    ts_recv        TIMESTAMPTZ NOT NULL,  -- capture-server receive (partition key)
    ts_event       TIMESTAMPTZ NOT NULL,  -- matching-engine event time
    rtype          SMALLINT,              -- 160
    publisher_id   SMALLINT,              -- GLBX.MDP3 = 1
    instrument_id  INTEGER     NOT NULL,
    action         CHAR(1),               -- A=Add C=Cancel M=Modify R=Clear T=Trade F=Fill N=None
    side           CHAR(1),               -- B=Bid A=Ask N=None
    price          DOUBLE PRECISION,
    size           INTEGER,
    channel_id     SMALLINT,
    order_id       BIGINT,                -- uint64 in DBN; CME ids ~7e12, COPY errors if >= 2^63
    flags          SMALLINT,              -- 128=LAST 64=TOB 32=SNAPSHOT 16=MBP 8=BAD_TS_RECV 4=MAYBE_BAD_BOOK
    ts_in_delta    INTEGER,
    sequence       BIGINT,
    symbol         TEXT        NOT NULL,
    dataset        TEXT        NOT NULL DEFAULT 'GLBX.MDP3'
);   -- hypertable on ts_recv, 1-day chunks; compress segmentby instrument_id, orderby ts_event, sequence
```

Full DDL: [`scripts/data/create_mbo_table.sql`](../../scripts/data/create_mbo_table.sql). Design
decisions match §2.8 (`DOUBLE PRECISION` prices, `ts_recv` partition, no unique constraint, no
extra indexes, idempotent per-day delete-then-COPY, same ns→µs rounding — the `.dbn.zst` files
remain the lossless archive).

#### MBO-specific notes

- **Snapshots**: a request starting at UTC midnight makes Databento emit a synthetic full-book
  snapshot (`flags & 32`), so each day file is self-contained for book reconstruction. Sub-day
  starts do not get one. Use UTC-midnight boundaries.
- **Ordering**: compression `orderby` is `(ts_event, sequence)` (set 2026-09-30; Timescale
  also stores partition-column `ts_recv DESC` as a third catalog key). Partition remains
  `ts_recv`. Books are per `instrument_id`; the roll from NQU6 to NQZ6 is a separate book,
  not a continuation.
- **Sunday / holiday chunks are small** (e.g. 2026-08-30 ≈ 0.77M rows, 2026-09-07 Labor Day ≈ 4.9M).

#### Measured characteristics

| Metric | Value |
|---|---|
| Rows | **475,152,566** — equals Databento's `get_record_count` for both segments exactly; per-day SQL count = Databento per-day count for all 29 days (0 mismatches) |
| Compressed size | **6,523 MB = 14.4 B/row**, 29/29 chunks compressed |
| Raw archive | 8.0 GB `.dbn.zst`, `data/sources/raw/backfill/nq_fut_mbo/` (~17 B/record) |
| Ingest | 474.7M rows in 69.9 min (113k rows/s aggregate, 2 workers) |
| Cost | **$0.00** — both segments cost-checked before submit and re-checked immediately before each `submit_job` |

#### Pipeline

```bash
# 1. cost gate + day-split batch job + download + ingest (aborts on any non-zero cost)
PYTHONPATH=/opt/vtech python scripts/data/backfill_mbo.py --dry-run \
    NQU6:2026-08-22:2026-09-13 NQZ6:2026-09-13:2026-09-25      # check only
PYTHONPATH=/opt/vtech python scripts/data/backfill_mbo.py --workers 2 \
    NQU6:2026-08-22:2026-09-13 NQZ6:2026-09-13:2026-09-25

# or ingest existing day files only (checkpoint: data/legacy/checkpoints/ingest_mbo.json)
python scripts/data/ingest_mbo.py --data-dir /opt/vtech/data/sources/raw/backfill/nq_fut_mbo --workers 2
```

`backfill_mbo.py` reuses `src/acquisition/historical.py` (`cost_check`, `submit_batch` with
`split_duration="day"`, `poll_batch_until_done`, `download_batch`). `ingest_mbo.py` is a copy of
`ingest_mbp10.py` (FIFO/COPY path) with only the table, column list and checkpoint changed; the
CSV header was verified against a real NQZ6 record and a 300-row field-level comparison.

> **Disk incident (2026-09-25)**: the first ingest ran 2 workers on a volume with 8 GB free and
> filled it, crashing PostgreSQL (recovered cleanly via WAL replay, no data loss). An uncompressed
> MBO day is ~2 GB until `compress_chunk` runs, plus WAL. Keep **≥ 15 GB free** before ingesting;
> 39 GB of regenerable `research/search/_fold_cache` was cleared to make room.

> **Current state (2026-09-29, ADR-147)**: `databento.mbo` is kept current automatically by
> `vtech-mbo-incremental.timer` (every 30 min, append-only — see §2.10) rather than by hand.
> `backfill_mbo.py`/`ingest_mbo.py` above remain the tools for a one-off manual gap-fill.

### 2.10 `databento.mbo_resting_orders_v2` — resting-order book state (checkpoint history)

Added 2026-09-29 (ADR-147) so `src/webui/mbo_orders.py` can tell what large orders were already
resting when a query window opens, not only ones whose own `Add` falls inside it. Maintained
incrementally by `scripts/ops/mbo_incremental_pull.py`, in the same transaction as each MBO
append — never a full recompute. Full DDL:
[`scripts/data/create_mbo_resting_orders_v2.sql`](../../scripts/data/create_mbo_resting_orders_v2.sql).

> **`_v2` since 2026-09-30 (ADR-149).** The original `mbo_resting_orders` replayed only
> `A`/`M`/`F`/`C`, but **CME purges Day orders at the weekend close and sends no `Cancel`** — the
> only signal is the order's absence from the next snapshot re-listing. With no way to remove them,
> v1 accumulated 808 phantom orders by 2026-09-28 (3,758 shown vs the exchange's own 2,951 — 27% of
> the book) while missing only 1 genuinely-resting order, and re-listings applied as fresh `Add`s
> had reset `t_start` on 1,051 of 5,043 orders to exactly 19:00:00 CT. v2 reconciles against each
> snapshot group, which is authoritative. v1 is **frozen intact** as a rollback path — a corrected
> checkpoint has different rows for the same PK, so it could not be fixed by appending, and
> rewriting it in place would break the append-only rule. Retiring v1 needs explicit approval.

One table, not two: each tick writes a new dated partition of every currently-open order.
**"Current state" is not a stored table — it's simply the newest `snapshot_ts` partition for a
symbol**, found via the index below (see ADR-147 for why an earlier "current + history"
two-table design was collapsed into this one). `order_id = -1` is a reserved marker row written
when a tick closes out every open order and adds none, so an all-closed partition still
registers as the newest checkpoint rather than being invisibly empty.

```sql
CREATE TABLE databento.mbo_resting_orders_v2 (
    snapshot_ts   TIMESTAMPTZ NOT NULL,
    order_id      BIGINT NOT NULL,
    symbol        TEXT NOT NULL,
    side          CHAR(1) NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    size          INTEGER NOT NULL,                    -- remaining, post-fill
    t_start       TIMESTAMPTZ NOT NULL,                -- origin (Add, or last price-changing Modify)
    t_start_exact BOOLEAN NOT NULL DEFAULT TRUE,       -- FALSE => t_start is only a lower bound
    PRIMARY KEY (snapshot_ts, order_id)
);  -- hypertable on snapshot_ts, 7-day chunks; index (symbol, snapshot_ts DESC);
    -- compress segmentby symbol after 30 days
```

`t_start_exact = FALSE` means the age is a lower bound only: the order was first seen already
resting in a snapshot group, or appeared in one at a price we had not recorded (a Modify lost to a
gap). Lets the chart say "resting ≥ Xh" rather than asserting an origin it cannot know.

Sizing (measured, not estimated — see ADR-147): ~1,300–2,500 open orders at a time for the NQ
lead contract → ~200–250 KB per checkpoint → ~10.5 MB/day, ~220 MB/month at the 30-min cadence —
negligible next to raw MBO. State-transition logic is pure, DB-free, and unit-tested:
`src/acquisition/mbo_resting_book.py` — `Add`/`Modify`/`Fill`/`Cancel` per order, plus `Clear` and
**snapshot-group reconciliation** as whole-book operations (ADR-149). Events must reach it in
strict chronological `(ts_recv, sequence)` order, not grouped per `order_id`, since `Clear` and
snapshot groups only mean anything relative to every other order's events. Query side:
`RESTING_SNAPSHOT_SQL` + `seed_from_snapshot()` in `src/webui/mbo_orders.py`, consumed by
`mbo_orders_api.py::order_map()`.

**Snapshot groups** (every row sharing one `ts_recv` with Databento's `F_SNAPSHOT` flag, at
19:00:00 CT daily and again after the weekly Sunday exchange `Clear`) are treated as the
exchange's complete statement of the book. Verified: a snapshot `ts_recv` carries no other rows, so
a group is contiguous and never straddles a 30-min tick. Reconciling against it is what removes
exchange-purged orders — and it heals pull-window gaps for free, since each boundary re-establishes
ground truth.

**Historical coverage**: backfilled from MBO inception (2026-08-23) forward, not just from
2026-09-29 — `scripts/ops/mbo_resting_book_backfill.py` replays the already-ingested
`databento.mbo` archive through the same `update_resting_book()` the live timer uses, on the
same 30-min grid, purely locally (no Databento calls). Run once, after confirming
`databento.mbo` itself has no gaps (day-by-day count check against Databento's
`get_record_count()` — see ADR-147). `--symbol`/`--start`/`--end`/`--table` scope a run, which is
how a sample is verified against a scratch table before touching the real one.

**Invariant worth monitoring** (cheap and exact): at any snapshot boundary the book must equal the
snapshot. Any phantom — an order in our checkpoint absent from the exchange's group at the same
instant — is a hard failure signal.

---

## 3. Compression Policies

Chunks older than 30 days are compressed. Compression achieves ~10x reduction on `bbo_1s` (267 GB compressed, estimated ~2.7 TB uncompressed).

```sql
-- tbbo compression
ALTER TABLE databento.tbbo SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_event ASC'
);
SELECT add_compression_policy('databento.tbbo', INTERVAL '30 days');

-- bbo_1s compression
ALTER TABLE databento.bbo_1s SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_recv ASC'
);
SELECT add_compression_policy('databento.bbo_1s', INTERVAL '30 days');

-- ohlcv_1s compression (policy id=1003, monthly chunks)
ALTER TABLE databento.ohlcv_1s SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_event ASC'
);
SELECT add_compression_policy('databento.ohlcv_1s', INTERVAL '30 days', if_not_exists => TRUE);
```

---

## 4. Continuous Aggregate — `tbbo_1min`

1-minute OHLCV + BBO derived from `tbbo`. Auto-refreshes every minute. Used by: `wv_`, `cs_`, `ht_` feature modules (35-day lookback per build day).

```sql
CREATE MATERIALIZED VIEW databento.tbbo_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', ts_event) AS bucket,
    instrument_id,
    symbol,
    first(price, ts_event)  AS open,
    max(price)              AS high,
    min(price)              AS low,
    last(price, ts_event)   AS close,
    sum(size)               AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*)                AS trade_count,
    last(bid_px, ts_event)  AS bid_px,
    last(ask_px, ts_event)  AS ask_px,
    last(bid_sz, ts_event)  AS bid_sz,
    last(ask_sz, ts_event)  AS ask_sz
FROM databento.tbbo
GROUP BY bucket, instrument_id, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('databento.tbbo_1min',
    start_offset      => INTERVAL '3 hours',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);
```

**Manual refresh** (if stale — e.g. after ingesting historical data out of order):
```sql
CALL refresh_continuous_aggregate(
    'databento.tbbo_1min',
    '2025-04-06'::timestamptz,
    '2026-04-07'::timestamptz
);
```
> Last manual refresh: 2026-05-19, covered full tbbo range after table re-ingest (108s, maintenance_work_mem=128GB).

### 4.1 NQ native-TF caggs — `tbbo_15s`, `tbbo_5m`, `tbbo_1h`, `tbbo_1d`

Created 2026-08-27 ([ADR-067](../decisions/ADR-067-nq-tbbo-15s-1h-1d-caggs.md)). NQ outrights only (`symbol ~ '^NQ[HMUZ][0-9]+$'`). Same OHLCV + last BBO + buy/sell volume columns as `tbbo_1min`. Script: `scripts/data/create_tbbo_tf_caggs.py`.

| View | Source | Bucket |
|------|--------|--------|
| `databento.tbbo_15s` | `tbbo` ticks | `time_bucket('15 seconds', ts_event)` (left) |
| `databento.tbbo_5m` | `tbbo_1min` (nested) | `time_bucket('5 minutes', bucket)` clock 5-minute (unix `% 300 == 0`) |
| `databento.tbbo_1h` | `tbbo_1min` (nested) | `time_bucket('1 hour', bucket)` clock hours |
| `databento.tbbo_1d` | `tbbo` ticks | Globex session: `[T-1 17:00, T 17:00) America/Chicago`, left-labeled at session open (`time_bucket(..., 'America/Chicago', origin 17:00 CT)`). Halt 16:00–17:00 CT is empty. |

Backfill window: 2025-04-06 00:00 UTC → session 2026-08-25 (exclusive end 2026-08-25 17:00 CDT). Auto-refresh policies keep them current after that. `tbbo_15s` compresses chunks older than 7 days.

As of 2026-08-27: `tbbo_15s` 2,225,197 rows / 54 MB; `tbbo_5m` 158,339 rows / 42 MB; `tbbo_1h` 17,572 rows / 8 MB; `tbbo_1d` 1,003 rows / 3 MB. NQ outrights only (NQM5…NQH7). All 1d labels are 17:00 America/Chicago. All 5m labels are unix-epoch 5-minute buckets.

```sql
-- 15s (ticks)
CREATE MATERIALIZED VIEW databento.tbbo_15s
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('15 seconds', ts_event) AS bucket,
    instrument_id, symbol,
    first(price, ts_event) AS open, max(price) AS high, min(price) AS low,
    last(price, ts_event) AS close, sum(size) AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*) AS trade_count,
    last(bid_px, ts_event) AS bid_px, last(ask_px, ts_event) AS ask_px,
    last(bid_sz, ts_event) AS bid_sz, last(ask_sz, ts_event) AS ask_sz
FROM databento.tbbo
WHERE symbol ~ '^NQ[HMUZ][0-9]+$'
GROUP BY 1, 2, 3
WITH NO DATA;

-- 5m (nested on tbbo_1min)
CREATE MATERIALIZED VIEW databento.tbbo_5m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', bucket) AS bucket,
    instrument_id, symbol,
    first(open, bucket) AS open, max(high) AS high, min(low) AS low,
    last(close, bucket) AS close, sum(volume) AS volume,
    sum(buy_volume) AS buy_volume, sum(sell_volume) AS sell_volume,
    sum(trade_count) AS trade_count,
    last(bid_px, bucket) AS bid_px, last(ask_px, bucket) AS ask_px,
    last(bid_sz, bucket) AS bid_sz, last(ask_sz, bucket) AS ask_sz
FROM databento.tbbo_1min
WHERE symbol ~ '^NQ[HMUZ][0-9]+$'
GROUP BY 1, 2, 3
WITH NO DATA;

-- 1h (nested on tbbo_1min)
CREATE MATERIALIZED VIEW databento.tbbo_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', bucket) AS bucket,
    instrument_id, symbol,
    first(open, bucket) AS open, max(high) AS high, min(low) AS low,
    last(close, bucket) AS close, sum(volume) AS volume,
    sum(buy_volume) AS buy_volume, sum(sell_volume) AS sell_volume,
    sum(trade_count) AS trade_count,
    last(bid_px, bucket) AS bid_px, last(ask_px, bucket) AS ask_px,
    last(bid_sz, bucket) AS bid_sz, last(ask_sz, bucket) AS ask_sz
FROM databento.tbbo_1min
WHERE symbol ~ '^NQ[HMUZ][0-9]+$'
GROUP BY 1, 2, 3
WITH NO DATA;

-- 1d Globex session (ticks; cannot nest on tbbo_1min — different bucket origin)
CREATE MATERIALIZED VIEW databento.tbbo_1d
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(
        '1 day'::interval, ts_event, 'America/Chicago',
        TIMESTAMPTZ '2000-01-01 17:00:00-06'
    ) AS bucket,
    instrument_id, symbol,
    first(price, ts_event) AS open, max(price) AS high, min(price) AS low,
    last(price, ts_event) AS close, sum(size) AS volume,
    sum(CASE WHEN side = 'B' THEN size ELSE 0 END) AS buy_volume,
    sum(CASE WHEN side = 'A' THEN size ELSE 0 END) AS sell_volume,
    count(*) AS trade_count,
    last(bid_px, ts_event) AS bid_px, last(ask_px, ts_event) AS ask_px,
    last(bid_sz, ts_event) AS bid_sz, last(ask_sz, ts_event) AS ask_sz
FROM databento.tbbo
WHERE symbol ~ '^NQ[HMUZ][0-9]+$'
GROUP BY 1, 2, 3
WITH NO DATA;
```

---

## 5. Useful Views

### Active NQ Options with DTE

```sql
CREATE VIEW databento.active_nq_options AS
SELECT
    d.instrument_id,
    d.symbol,
    d.instrument_class,
    d.strike_price,
    d.expiration,
    EXTRACT(DAY FROM d.expiration - CURRENT_TIMESTAMP) AS dte,
    d.min_price_increment,
    d.multiplier,
    d.open_interest
FROM databento.definitions d
WHERE d.underlying = 'NQ'
  AND d.expiration > CURRENT_TIMESTAMP
  AND d.instrument_class IN ('C', 'P')
ORDER BY d.expiration, d.strike_price;
```

---

## 6. TimescaleDB Configuration

```ini
# Do not hardcode GB from a previous OCI shape. On boot,
# systemd/vtech-vm-resources.service writes
# /etc/postgresql/16/main/conf.d/z-vtech-vm-ram.conf from get_vm_config():
#   shared_buffers         ≈ 25% RAM (clamp 1–32 GB)
#   effective_cache_size   ≈ 50% RAM
#   maintenance_work_mem   ≈ 8% RAM (clamp 1–8 GB)
# Manual apply: sudo python /opt/vtech/scripts/ops/apply_vm_resource_limits.py --apply-postgres
# Changing shared_buffers requires a Postgres restart.

work_mem                  = 256MB
max_connections           = 50
wal_buffers               = 64MB

timescaledb.max_background_workers = 8
```

---

## 7. Quick Diagnostics

```bash
# Row counts per table
psql vtech_market -c "
  SELECT relname, n_live_tup
  FROM pg_stat_user_tables
  WHERE schemaname = 'databento'
  ORDER BY n_live_tup DESC;"

# Chunk info for bbo_1s
psql vtech_market -c "
  SELECT chunk_name, range_start, range_end, is_compressed
  FROM timescaledb_information.chunks
  WHERE hypertable_name = 'bbo_1s'
  ORDER BY range_start DESC LIMIT 20;"

# Check definitions: 0DTE weekly missing?
psql vtech_market -c "
  SELECT underlying, instrument_class, count(*)
  FROM databento.definitions
  GROUP BY underlying, instrument_class;"
```

---

## 8. Migration Notes

### Phase 17B — T/M Class Purge

```sql
-- Purged via scripts/purge_ud_combos.py
DELETE FROM databento.bbo_1s
WHERE instrument_class IN ('T', 'M');
-- Removed: 500,382,858 rows
```

### Phase 17C — Weekly Definitions (Pending)

```python
# TODO: Ingest ~200K–500K rows for 24 weekly parent symbols
# See scripts/download_nq_0dte_weekly.py Phase 1 (symbol map already exists)
# Use ingest_nq_opt.py --schema definition --glob "symbol_map.json"
# or write a dedicated Phase 17C script
```

---

## Related Docs

- [DATA_PIPELINE.md](DATA_PIPELINE.md) — how data flows into these tables
- [operations/BACKFILL.md](../operations/BACKFILL.md) — ingestion procedures
- [operations/TROUBLESHOOTING.md](../operations/TROUBLESHOOTING.md) — cagg refresh, stale data errors
