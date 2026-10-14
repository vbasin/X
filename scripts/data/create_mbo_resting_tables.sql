-- databento.mbo_resting_orders — order-book state history: one row set per checkpoint tick.
-- "Current state" is not a separate table -- it is simply the newest snapshot_ts partition for
-- a symbol, found via the (symbol, snapshot_ts DESC) index below. See ADR-147.
--
-- This was originally two tables (a continuously-UPSERTed "current" table plus a separate
-- history table). Collapsed to one on review: the "current" table had exactly one consumer --
-- the ingestion script's own bookkeeping -- and that question ("what's open right now") is
-- answered just as well, at this data volume, by "what's in the newest snapshot_ts partition",
-- with no separate table, no UPSERT/DELETE pass, and no risk of the two ever disagreeing. This
-- also matches the rest of the schema's own convention: everywhere else, "current" is derived
-- from an append-only log by querying, not maintained as its own mutated table.
--
-- Maintained incrementally by scripts/ops/mbo_incremental_pull.py, in the same transaction as
-- each MBO append: read the newest partition, apply this tick's new A/M/F/C events to it
-- (src/acquisition/mbo_resting_book.py), write the result as a new partition dated at the
-- tick's own end boundary. A closed order (Cancel or full Fill) needs no explicit delete or
-- status flag -- it is simply absent from the next partition.

CREATE SCHEMA IF NOT EXISTS databento;

CREATE TABLE IF NOT EXISTS databento.mbo_resting_orders (
    snapshot_ts TIMESTAMPTZ NOT NULL,
    order_id    BIGINT NOT NULL,
    symbol      TEXT NOT NULL,
    side        CHAR(1) NOT NULL,
    price       DOUBLE PRECISION NOT NULL,
    size        INTEGER NOT NULL,          -- remaining (post-fill) size
    t_start     TIMESTAMPTZ NOT NULL,      -- true origin: first Add, or last price-changing Modify
    PRIMARY KEY (snapshot_ts, order_id)
);

SELECT create_hypertable('databento.mbo_resting_orders', 'snapshot_ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Fast "newest partition" (current state) and "nearest partition at or before t0" lookups.
CREATE INDEX IF NOT EXISTS mbo_resting_orders_lookup_idx
    ON databento.mbo_resting_orders (symbol, snapshot_ts DESC);

-- Small table (~1-3k open orders x ~46 checkpoints/day, see ADR-147's storage estimate) --
-- compress well after it's cold, mirroring the rest of the schema's convention.
ALTER TABLE databento.mbo_resting_orders SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'snapshot_ts ASC, order_id ASC'
);
SELECT add_compression_policy('databento.mbo_resting_orders', INTERVAL '30 days',
                              if_not_exists => TRUE);

-- ============================================================================================
-- SUPERSEDED 2026-09-30 (ADR-149). This table is FROZEN: no process writes to it any more.
-- Its checkpoints carry phantom orders -- CME purges Day orders at the weekend close and sends
-- no Cancel, and the A/M/F/C-only book below had no way to remove them (808 phantoms, 27% of the
-- book, as of 2026-09-28) -- and t_start values truncated to the last UTC-midnight snapshot.
-- Kept intact as a rollback path only. Live table and rationale:
--   scripts/data/create_mbo_resting_orders_v2.sql
-- ============================================================================================
