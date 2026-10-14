-- databento.mbo_resting_orders_v2 — order-book state history, one row set per checkpoint tick.
-- Supersedes databento.mbo_resting_orders (v1). See ADR-149.
--
-- WHY A NEW TABLE RATHER THAN A REBUILD IN PLACE
-- v1's checkpoints are contaminated and cannot be corrected by appending: a fixed checkpoint has
-- *different* rows for the same (snapshot_ts, order_id) key, so ON CONFLICT DO NOTHING would keep
-- the wrong ones and an UPDATE/DELETE pass would violate this pipeline's append-only rule. v1 is
-- therefore left untouched and frozen as a rollback path; readers move to v2.
--
-- The two v1 defects, both measured against the exchange's own snapshot re-listings:
--   1. Phantom orders. CME purges Day orders at the weekend close with NO Cancel message, so the
--      only signal is absence from the next snapshot group. v1 applied A/M/F/C only, so it never
--      removed them: 126 leaked across the 2026-09-24 -> 09-27 weekend, 808 phantoms accumulated
--      by 09-28 19:00 CT (our 3,758 vs the exchange's 2,951 — 27% of the displayed book). v1
--      missed only 1 genuinely-resting order in the same comparison, i.e. it added correctly and
--      never subtracted.
--   2. Truncated ages. Snapshot re-listings were applied as fresh Adds, resetting t_start. 1,051
--      of 5,043 orders in v1's 2026-09-28 12:00 checkpoint carried t_start of exactly 19:00:00 CT
--      (a UTC-midnight boundary) rather than their true placement time.
--
-- "Current state" is not a separate table -- it is the newest snapshot_ts partition for a symbol,
-- found via the (symbol, snapshot_ts DESC) index below (v1's design note on collapsing the old
-- two-table layout into one still applies and is unchanged here).
--
-- Maintained incrementally by scripts/ops/mbo_incremental_pull.py in the same transaction as each
-- MBO append: read the newest partition, apply this tick's new events to it
-- (src/acquisition/mbo_resting_book.py), write the result as a new partition dated at the tick's
-- own end boundary. A closed order needs no delete or status flag -- it is simply absent from the
-- next partition.

CREATE SCHEMA IF NOT EXISTS databento;

CREATE TABLE IF NOT EXISTS databento.mbo_resting_orders_v2 (
    snapshot_ts   TIMESTAMPTZ NOT NULL,
    order_id      BIGINT NOT NULL,
    symbol        TEXT NOT NULL,
    side          CHAR(1) NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    size          INTEGER NOT NULL,        -- remaining (post-fill) size
    t_start       TIMESTAMPTZ NOT NULL,    -- origin: first Add, or last price-changing Modify
    -- FALSE => t_start is only a lower bound on age: the order was first seen already resting in
    -- a snapshot group, or appeared in one at a price we had not recorded (a missed Modify). Lets
    -- the chart render "resting >= Xh" instead of asserting an origin it cannot know.
    t_start_exact BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (snapshot_ts, order_id)
);

SELECT create_hypertable('databento.mbo_resting_orders_v2', 'snapshot_ts',
    chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);

-- Fast "newest partition" (current state) and "nearest partition at or before t0" lookups.
CREATE INDEX IF NOT EXISTS mbo_resting_orders_v2_lookup_idx
    ON databento.mbo_resting_orders_v2 (symbol, snapshot_ts DESC);

-- Small table (~1-3k open orders x ~46 checkpoints/day) -- compresses well once cold, mirroring
-- the rest of the schema's convention.
ALTER TABLE databento.mbo_resting_orders_v2 SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby   = 'snapshot_ts ASC, order_id ASC'
);
SELECT add_compression_policy('databento.mbo_resting_orders_v2', INTERVAL '30 days',
                              if_not_exists => TRUE);
