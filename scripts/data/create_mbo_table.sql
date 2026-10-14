-- databento.mbo — Market-by-Order, full depth (DBN rtype 160, 56 B/record)
-- Source: Databento GLBX.MDP3 `mbo` schema (L3, 1-month rolling free window)
-- Ref: https://databento.com/docs/schemas-and-data-formats/mbo
--
-- Design mirrors databento.mbp_10 (see create_mbp_10_table.sql for the rationale):
--   * DOUBLE PRECISION prices (lossless on the 0.25 CME grid), partition on ts_recv,
--     NO unique constraint (same-price sweep fills yield legitimate duplicate
--     (instrument_id, ts_event, sequence) tuples), NO extra indexes.
--   * order_id is uint64 in DBN. CME ids observed ~6.9e12, far below 2^63, so BIGINT.
--     A value >= 2^63 makes COPY fail loudly rather than wrap.
--   * Idempotency: ingest_mbo.py deletes each day's ts_recv range before COPY.
--   * Databento emits a synthetic full-book snapshot (flags & 32) at a UTC-midnight
--     request start, so each day file is self-contained for book reconstruction.

CREATE SCHEMA IF NOT EXISTS databento;

CREATE TABLE IF NOT EXISTS databento.mbo (
    ts_recv        TIMESTAMPTZ      NOT NULL,   -- uint64 ns UTC; capture-server receive (partition key)
    ts_event       TIMESTAMPTZ      NOT NULL,   -- uint64 ns UTC; matching-engine event time
    rtype          SMALLINT,                    -- uint8; 160 (RType.MBO)
    publisher_id   SMALLINT,                    -- uint16; GLBX.MDP3 = 1
    instrument_id  INTEGER          NOT NULL,   -- uint32
    action         CHAR(1),                     -- A=Add C=Cancel M=Modify R=Clear T=Trade F=Fill N=None
    side           CHAR(1),                     -- B=Bid A=Ask N=None
    price          DOUBLE PRECISION,            -- int64 1e-9 -> USD (NULL if undefined)
    size           INTEGER,                     -- uint32
    channel_id     SMALLINT,                    -- uint8; CME MDP channel
    order_id       BIGINT,                      -- uint64; venue order id
    flags          SMALLINT,                    -- uint8: 128=LAST 64=TOB 32=SNAPSHOT 16=MBP 8=BAD_TS_RECV 4=MAYBE_BAD_BOOK
    ts_in_delta    INTEGER,                     -- int32 ns; matching-engine send latency before ts_recv
    sequence       BIGINT,                      -- uint32 venue sequence (BIGINT: exceeds INTEGER)
    symbol         TEXT             NOT NULL,   -- raw symbol (e.g. NQZ6) via symbology map
    dataset        TEXT             NOT NULL DEFAULT 'GLBX.MDP3'
);

SELECT create_hypertable('databento.mbo', 'ts_recv',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

ALTER TABLE databento.mbo SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby   = 'ts_event ASC, sequence ASC'
);
-- Timescale appends the partition column (`ts_recv DESC`) as a third orderby
-- key in the catalog. Primary compressed order is ts_event, sequence.

-- ingest_mbo.py compresses each day right after COPY; the policy is only a backstop.
SELECT add_compression_policy('databento.mbo', INTERVAL '30 days', if_not_exists => TRUE);
