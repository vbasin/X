"""Schema definitions and DBN-to-database column mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ── Databento schema constants ──────────────────────────────────

DATASET = "GLBX.MDP3"

# Backfill jobs in priority order
BACKFILL_JOBS: list[dict] = [
    {"symbols": "NQ.FUT", "schema": "tbbo",       "stype_in": "parent", "priority": 0},
    {"symbols": "NQ.FUT", "schema": "bbo-1s",     "stype_in": "parent", "priority": 0},
    {"symbols": "NQ.OPT", "schema": "bbo-1s",     "stype_in": "parent", "priority": 1},
    {"symbols": "NQ.OPT", "schema": "definition",  "stype_in": "parent", "priority": 1},
    {"symbols": "NQ.FUT", "schema": "statistics",  "stype_in": "parent", "priority": 2},
    {"symbols": "ES.FUT", "schema": "tbbo",        "stype_in": "parent", "priority": 2},
]


# ── Column mappings: DBNStore.to_df() column → database column ──

@dataclass(frozen=True)
class ColMap:
    """Maps a DataFrame column name (from to_df) to a DB column name."""
    df_col: str
    db_col: str
    dtype: str = "text"


# TBBO: trades with best bid/offer
TBBO_COLUMNS: Sequence[ColMap] = (
    ColMap("ts_event",    "ts_event",       "timestamptz"),
    ColMap("ts_recv",     "ts_recv",        "timestamptz"),
    ColMap("instrument_id", "instrument_id", "integer"),
    ColMap("symbol",      "symbol",         "text"),
    ColMap("price",       "price",          "double precision"),
    ColMap("size",        "size",           "integer"),
    ColMap("side",        "side",           "char(1)"),
    ColMap("action",      "action",         "char(1)"),
    ColMap("flags",       "flags",          "smallint"),
    ColMap("sequence",    "sequence",       "bigint"),
    ColMap("bid_px_00",   "bid_px",         "double precision"),
    ColMap("ask_px_00",   "ask_px",         "double precision"),
    ColMap("bid_sz_00",   "bid_sz",         "integer"),
    ColMap("ask_sz_00",   "ask_sz",         "integer"),
    ColMap("bid_ct_00",   "bid_ct",         "integer"),
    ColMap("ask_ct_00",   "ask_ct",         "integer"),
    ColMap("ts_in_delta", "ts_in_delta",    "integer"),
)

# BBO-1S: 1-second quote snapshots
BBO_1S_COLUMNS: Sequence[ColMap] = (
    ColMap("ts_recv",       "ts_recv",        "timestamptz"),
    ColMap("instrument_id", "instrument_id",  "integer"),
    ColMap("symbol",        "symbol",         "text"),
    ColMap("price",         "price",          "double precision"),
    ColMap("size",          "size",           "integer"),
    ColMap("bid_px_00",     "bid_px",         "double precision"),
    ColMap("ask_px_00",     "ask_px",         "double precision"),
    ColMap("bid_sz_00",     "bid_sz",         "integer"),
    ColMap("ask_sz_00",     "ask_sz",         "integer"),
)

# Definition: instrument/option contract specs
DEFINITION_COLUMNS: Sequence[ColMap] = (
    ColMap("ts_event",               "ts_event",               "timestamptz"),
    ColMap("instrument_id",          "instrument_id",          "integer"),
    ColMap("symbol",                 "symbol",                 "text"),
    ColMap("instrument_class",       "instrument_class",       "char(1)"),
    ColMap("strike_price",           "strike_price",           "double precision"),
    ColMap("expiration",             "expiration",             "timestamptz"),
    ColMap("underlying",             "underlying",             "text"),
    ColMap("exchange",               "exchange",               "text"),
    ColMap("currency",               "currency",               "text"),
    ColMap("min_price_increment",    "min_price_increment",    "double precision"),
    ColMap("multiplier",             "multiplier",             "double precision"),
)

# Statistics: daily settlement, hi/lo, volume, OI
STATISTICS_COLUMNS: Sequence[ColMap] = (
    ColMap("ts_event",      "ts_event",      "timestamptz"),
    ColMap("ts_recv",       "ts_recv",       "timestamptz"),
    ColMap("instrument_id", "instrument_id", "integer"),
    ColMap("symbol",        "symbol",        "text"),
    ColMap("stat_type",     "stat_type",     "smallint"),
    ColMap("price",         "price",         "double precision"),
    ColMap("quantity",      "quantity",      "bigint"),
    ColMap("sequence",      "sequence",      "bigint"),
    ColMap("ts_ref",        "ts_ref",        "timestamptz"),
    ColMap("update_action", "update_action", "smallint"),
    ColMap("stat_flags",    "stat_flags",    "smallint"),
)


def get_column_map(schema: str) -> Sequence[ColMap]:
    """Return the column mapping for a given Databento schema name."""
    return {
        "tbbo":       TBBO_COLUMNS,
        "bbo-1s":     BBO_1S_COLUMNS,
        "definition": DEFINITION_COLUMNS,
        "statistics": STATISTICS_COLUMNS,
    }[schema]
