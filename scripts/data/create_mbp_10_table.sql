-- databento.mbp_10 — Market-by-Price, top 10 levels (DBN rtype 10, 368 B/record)
-- Source: Databento GLBX.MDP3 `mbp-10` schema (L2, 1-month rolling free window)
-- Ref: https://databento.com/docs/schemas-and-data-formats/mbp-10
--
-- Design notes:
--   * Flat-wide levels (60 cols) rather than arrays: TimescaleDB columnar compression
--     encodes each column independently (Gorilla for float8, delta-delta for int),
--     which arrays defeat by storing an opaque varlena blob. Also matches the
--     56-flat-column precedent in databento.nq_option_surface_1s.
--   * Prices as DOUBLE PRECISION, not the native int64 1e-9 fixed point. Lossless for
--     the CME price grid: every NQ tick is a multiple of 0.25, and |px_nanos| ~ 2.4e13
--     is far below float64's exact-integer limit of 2^53 ~ 9.0e15. Keeps the column
--     type consistent with tbbo / bbo_1s so downstream feature modules need no cast.
--   * Partitioned on ts_recv (capture-server receive), which is the timestamp Databento
--     sorts MBP records by and the only one safe for book reconstruction. ts_event
--     (matching-engine) is kept as a plain column.
--   * NO unique constraint. Same lesson as tbbo (2026-05-19): CME same-price sweep fills
--     yield legitimately duplicate (instrument_id, ts_event, sequence) tuples. A unique
--     index over ~438M rows would also cost ~15-20 GB. Idempotency is handled by the
--     ingest script deleting the day's ts_recv range before COPY.

CREATE SCHEMA IF NOT EXISTS databento;

CREATE TABLE IF NOT EXISTS databento.mbp_10 (
    -- ── record header ──────────────────────────────────────────────────────
    ts_recv        TIMESTAMPTZ      NOT NULL,   -- uint64 ns UTC; capture-server receive (partition key)
    ts_event       TIMESTAMPTZ      NOT NULL,   -- uint64 ns UTC; matching-engine event time
    instrument_id  INTEGER          NOT NULL,   -- uint32; numeric instrument id (per publisher/day)
    symbol         TEXT             NOT NULL,   -- resolved raw symbol (e.g. NQU6) via symbology map
    publisher_id   SMALLINT,                    -- uint16; venue/dataset/publisher (GLBX.MDP3 = 1)
    rtype          SMALLINT,                    -- uint8; always 10 (RType.MBP_10) for this schema
    -- ── event that triggered the book update ───────────────────────────────
    action         CHAR(1),                     -- char: A=Add C=Cancel M=Modify R=Clear T=Trade F=Fill N=None
    side           CHAR(1),                     -- char: B=Bid A=Ask N=None (side of the resting order)
    depth          SMALLINT,                    -- uint8; 0-9, book level where the update occurred
    price          DOUBLE PRECISION,            -- int64 1e-9 -> USD; price of the triggering order/trade
    size           INTEGER,                     -- uint32; quantity of the triggering order/trade
    flags          SMALLINT,                    -- uint8 bit field: 128=LAST 64=TOB 32=SNAPSHOT
                                                --   16=MBP_AGG 8=BAD_TS_RECV 4=MAYBE_BAD_BOOK
    ts_in_delta    INTEGER,                     -- int32 ns; matching-engine send latency before ts_recv
    sequence       BIGINT,                      -- uint32; venue message sequence number

    -- ── level 0 ──────────────────────────────────────────────────────────
    bid_px_00      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_00      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_00      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_00      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_00      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_00      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 1 ──────────────────────────────────────────────────────────
    bid_px_01      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_01      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_01      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_01      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_01      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_01      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 2 ──────────────────────────────────────────────────────────
    bid_px_02      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_02      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_02      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_02      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_02      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_02      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 3 ──────────────────────────────────────────────────────────
    bid_px_03      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_03      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_03      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_03      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_03      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_03      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 4 ──────────────────────────────────────────────────────────
    bid_px_04      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_04      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_04      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_04      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_04      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_04      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 5 ──────────────────────────────────────────────────────────
    bid_px_05      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_05      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_05      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_05      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_05      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_05      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 6 ──────────────────────────────────────────────────────────
    bid_px_06      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_06      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_06      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_06      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_06      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_06      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 7 ──────────────────────────────────────────────────────────
    bid_px_07      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_07      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_07      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_07      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_07      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_07      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 8 ──────────────────────────────────────────────────────────
    bid_px_08      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_08      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_08      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_08      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_08      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_08      INTEGER,                     -- uint32; number of ask orders at level
    -- ── level 9 ──────────────────────────────────────────────────────────
    bid_px_09      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    ask_px_09      DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if level absent)
    bid_sz_09      INTEGER,                     -- uint32; total resting bid qty at level
    ask_sz_09      INTEGER,                     -- uint32; total resting ask qty at level
    bid_ct_09      INTEGER,                     -- uint32; number of bid orders at level
    ask_ct_09      INTEGER,                     -- uint32; number of ask orders at level

    dataset        TEXT             NOT NULL DEFAULT 'GLBX.MDP3'
);

SELECT create_hypertable('databento.mbp_10', 'ts_recv',
    chunk_time_interval => INTERVAL '1 day',   -- ~13-19M rows / ~8 GB uncompressed per day
    if_not_exists       => TRUE
);

-- Columnar compression. orderby (ts_recv, sequence) gives a deterministic order within
-- same-nanosecond bursts, which matters for replaying book state.
ALTER TABLE databento.mbp_10 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_recv ASC, sequence ASC'
);

-- Deliberately NO extra indexes: create_hypertable already indexes ts_recv, and at this
-- row count each additional index costs GBs (cf. idx_tbbo_instrument_ts, dropped 2026-05-18).

-- The ingest script compresses each day immediately after COPY, so the 30-day policy is
-- only a backstop for anything that slipped through (e.g. a live tail).
SELECT add_compression_policy('databento.mbp_10', INTERVAL '30 days', if_not_exists => TRUE);

