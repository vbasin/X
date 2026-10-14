"""0DTE NQ options Greeks pipeline: day compute → PG hypertables + surface parquet.

One compute path, three sinks (no recompute):
  1. ``databento.nq_option_greeks_1s``  — every two-sided 0DTE C/P quote + Greeks
  2. ``databento.nq_option_surface_1s``  — compact 8-leg × 7-field + nq_price row / second
  3. ``data/parquet_options_surface/features_YYYY-MM-DD.parquet`` — same compact surface

Both the historical backfill script and the live streamer call :func:`process_day`
/ :func:`compute_second_batch` so train and serve share identical semantics.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from src.common.db import get_dsn
from src.features.options_greeks import (
    _STRIKE_GRID,
    _TARGET_DELTAS,
    SURFACE_FEATURE_COLS,
    SURFACE_FIELDS,
    SURFACE_LEGS,
    SURFACE_META_COLS,
    delta_bucket_row,
    n_legs_filled,
    option_greeks,
)

logger = logging.getLogger("vtech.features.options_greeks_pipeline")

_CT = ZoneInfo("America/Chicago")
_RISK_FREE = 0.05
from src.common.data_paths import options_greeks_dir, options_surface_dir

def _default_surface_dir() -> Path:
    return options_surface_dir()

def _default_greeks_dir() -> Path:
    return options_greeks_dir()

# ── DDL ──────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS databento.nq_option_greeks_1s (
    ts              TIMESTAMPTZ       NOT NULL,
    instrument_id   INTEGER           NOT NULL,
    symbol          TEXT              NOT NULL,
    expiry          DATE              NOT NULL,
    strike          DOUBLE PRECISION  NOT NULL,
    is_call         BOOLEAN           NOT NULL,
    underlying_sym  TEXT              NOT NULL,
    nq_price        DOUBLE PRECISION,
    bid_px          DOUBLE PRECISION,
    ask_px          DOUBLE PRECISION,
    mid             DOUBLE PRECISION,
    spread          DOUBLE PRECISION,
    iv              DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    gamma           DOUBLE PRECISION,
    vega            DOUBLE PRECISION,
    theta           DOUBLE PRECISION,
    UNIQUE (instrument_id, ts)
);

SELECT create_hypertable(
    'databento.nq_option_greeks_1s', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_greeks_1s_expiry_ts
    ON databento.nq_option_greeks_1s (expiry, ts DESC);
CREATE INDEX IF NOT EXISTS idx_greeks_1s_symbol_ts
    ON databento.nq_option_greeks_1s (symbol, ts DESC);

ALTER TABLE databento.nq_option_greeks_1s SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'instrument_id',
    timescaledb.compress_orderby = 'ts'
);
SELECT add_compression_policy('databento.nq_option_greeks_1s', INTERVAL '3 days',
                              if_not_exists => TRUE);

-- ── compact per-second surface ──────────────────────────────────
CREATE TABLE IF NOT EXISTS databento.nq_option_surface_1s (
    ts              TIMESTAMPTZ  PRIMARY KEY,
    expiry          DATE         NOT NULL,
    underlying_sym  TEXT         NOT NULL,
    n_legs_filled   SMALLINT     NOT NULL,
    nq_price        DOUBLE PRECISION,
    c10_iv DOUBLE PRECISION, c10_delta DOUBLE PRECISION, c10_gamma DOUBLE PRECISION,
    c10_vega DOUBLE PRECISION, c10_theta DOUBLE PRECISION, c10_mid DOUBLE PRECISION, c10_spread DOUBLE PRECISION,
    p10_iv DOUBLE PRECISION, p10_delta DOUBLE PRECISION, p10_gamma DOUBLE PRECISION,
    p10_vega DOUBLE PRECISION, p10_theta DOUBLE PRECISION, p10_mid DOUBLE PRECISION, p10_spread DOUBLE PRECISION,
    c25_iv DOUBLE PRECISION, c25_delta DOUBLE PRECISION, c25_gamma DOUBLE PRECISION,
    c25_vega DOUBLE PRECISION, c25_theta DOUBLE PRECISION, c25_mid DOUBLE PRECISION, c25_spread DOUBLE PRECISION,
    p25_iv DOUBLE PRECISION, p25_delta DOUBLE PRECISION, p25_gamma DOUBLE PRECISION,
    p25_vega DOUBLE PRECISION, p25_theta DOUBLE PRECISION, p25_mid DOUBLE PRECISION, p25_spread DOUBLE PRECISION,
    c40_iv DOUBLE PRECISION, c40_delta DOUBLE PRECISION, c40_gamma DOUBLE PRECISION,
    c40_vega DOUBLE PRECISION, c40_theta DOUBLE PRECISION, c40_mid DOUBLE PRECISION, c40_spread DOUBLE PRECISION,
    p40_iv DOUBLE PRECISION, p40_delta DOUBLE PRECISION, p40_gamma DOUBLE PRECISION,
    p40_vega DOUBLE PRECISION, p40_theta DOUBLE PRECISION, p40_mid DOUBLE PRECISION, p40_spread DOUBLE PRECISION,
    catm_iv DOUBLE PRECISION, catm_delta DOUBLE PRECISION, catm_gamma DOUBLE PRECISION,
    catm_vega DOUBLE PRECISION, catm_theta DOUBLE PRECISION, catm_mid DOUBLE PRECISION, catm_spread DOUBLE PRECISION,
    patm_iv DOUBLE PRECISION, patm_delta DOUBLE PRECISION, patm_gamma DOUBLE PRECISION,
    patm_vega DOUBLE PRECISION, patm_theta DOUBLE PRECISION, patm_mid DOUBLE PRECISION, patm_spread DOUBLE PRECISION
);

SELECT create_hypertable(
    'databento.nq_option_surface_1s', 'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

ALTER TABLE databento.nq_option_surface_1s SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'ts'
);
SELECT add_compression_policy('databento.nq_option_surface_1s', INTERVAL '7 days',
                              if_not_exists => TRUE);
"""

_GREEKS_COLS = [
    "ts", "instrument_id", "symbol", "expiry", "strike", "is_call",
    "underlying_sym", "nq_price", "bid_px", "ask_px", "mid", "spread",
    "iv", "delta", "gamma", "vega", "theta",
]

_SURFACE_DB_COLS = list(SURFACE_META_COLS) + SURFACE_FEATURE_COLS


def ensure_tables(conn: psycopg.Connection | None = None) -> None:
    """Create Greeks + surface hypertables (idempotent).

    Fast path: if both tables already exist, return immediately — avoids
    ``ALTER TABLE … SET (timescaledb.compress)`` taking AccessExclusiveLock
    and blocking behind live COPY/compress from the backfill.
    """
    own = conn is None
    if own:
        conn = psycopg.connect(get_dsn(), autocommit=True)
    try:
        was_ac = conn.autocommit
        conn.autocommit = True
        n = conn.execute(
            """
            SELECT count(*) FROM information_schema.tables
            WHERE table_schema = 'databento'
              AND table_name IN ('nq_option_greeks_1s', 'nq_option_surface_1s')
            """
        ).fetchone()[0]
        if n == 2:
            logger.info("options_greeks_tables_ready")
            conn.autocommit = was_ac
            return

        # Statement-by-statement: each commits independently under autocommit so a
        # non-fatal "already exists" on a policy does not poison later DDL.
        for stmt in _DDL.split(";"):
            s = stmt.strip()
            if not s:
                continue
            try:
                conn.execute(s)
            except Exception as exc:  # noqa: BLE001 — idempotent DDL
                msg = str(exc).lower()
                if any(k in msg for k in (
                    "already exists", "already a hypertable",
                    "compression", "policy", "duplicate",
                )):
                    logger.debug("ddl_skip: %s", exc)
                    continue
                raise
        conn.autocommit = was_ac
        logger.info("options_greeks_tables_ready")
    finally:
        if own:
            conn.close()


def day_already_done(d: date, conn: psycopg.Connection | None = None) -> bool:
    """True if surface table already has rows for session date ``d`` (CT)."""
    own = conn is None
    if own:
        conn = psycopg.connect(get_dsn())
    try:
        # Session window in UTC covering the CT calendar day
        start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
        end_ct = start_ct + timedelta(days=1)
        n = conn.execute(
            "SELECT 1 FROM databento.nq_option_surface_1s "
            "WHERE ts >= %s AND ts < %s LIMIT 1",
            (start_ct.astimezone(timezone.utc), end_ct.astimezone(timezone.utc)),
        ).fetchone()
        return n is not None
    finally:
        if own:
            conn.close()


def _load_0dte_defs(conn: psycopg.Connection, d: date) -> pd.DataFrame:
    """Most-recent definition per instrument expiring on CT date ``d`` (0DTE chain)."""
    # Prefer DISTINCT ON instrument_id (defs can re-appear); expiry CT date = d.
    sql_q = """
        SELECT DISTINCT ON (instrument_id)
            instrument_id, symbol, strike_price, instrument_class, expiration, underlying
        FROM databento.definitions
        WHERE instrument_class IN ('C', 'P')
          AND (expiration AT TIME ZONE 'America/Chicago')::date = %s
        ORDER BY instrument_id, ts_event DESC NULLS LAST
    """
    return pd.read_sql(sql_q, conn, params=(d,))


def _load_0dte_bbo(conn: psycopg.Connection, d: date, instrument_ids: list[int]) -> pd.DataFrame:
    """Two-sided 0DTE BBO-1s for the UTC day window covering CT session date ``d``.

    Deliberately narrow: ``symbol`` is an object column that costs ~920 MB of the
    ~1.64 GB a day's 15M quotes occupy in pandas, and it is already on the 884-row
    definitions frame this is merged with. ``bid_sz``/``ask_sz`` are only ever
    predicates, never outputs. Dropping all three cuts the per-day peak by ~71%,
    which is what lets the backfill run several days in parallel — at 8 GB/worker
    this VM otherwise allows exactly one.
    """
    if not instrument_ids:
        return pd.DataFrame()
    start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
    end_ct = start_ct + timedelta(days=1)
    start_utc = start_ct.astimezone(timezone.utc)
    end_utc = end_ct.astimezone(timezone.utc)
    sql_q = """
        SELECT ts_recv, instrument_id, bid_px, ask_px
        FROM databento.bbo_1s
        WHERE ts_recv >= %s AND ts_recv < %s
          AND instrument_id = ANY(%s)
          AND bid_sz > 0 AND ask_sz > 0
          AND bid_px > 0 AND ask_px > 0
          AND bid_px < %s AND ask_px < %s
        ORDER BY ts_recv
    """
    return pd.read_sql(
        sql_q, conn,
        params=(start_utc, end_utc, instrument_ids, _UNDEF_PX, _UNDEF_PX),
    )


def _load_nq_price_1s(conn: psycopg.Connection, d: date) -> pd.Series:
    """1s NQ continuous close (``NQ.c.0``) for CT date ``d``, indexed by floored second UTC."""
    start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
    end_ct = start_ct + timedelta(days=1)
    start_utc = start_ct.astimezone(timezone.utc)
    end_utc = end_ct.astimezone(timezone.utc)
    sql_q = """
        SELECT ts_event, close
        FROM databento.ohlcv_1s
        WHERE symbol = 'NQ.c.0'
          AND ts_event >= %s AND ts_event < %s
        ORDER BY ts_event
    """
    df = pd.read_sql(sql_q, conn, params=(start_utc, end_utc))
    if df.empty:
        # Fallback: front NQ outright mid from bbo_1s that day
        sql_b = """
            SELECT ts_recv, (bid_px + ask_px) / 2.0 AS mid
            FROM databento.bbo_1s
            WHERE ts_recv >= %s AND ts_recv < %s
              AND symbol ~ '^NQ[FGHJKMNQUVXZ][0-9]$'
              AND bid_px > 0 AND ask_px > 0
            ORDER BY ts_recv
        """
        df = pd.read_sql(sql_b, conn, params=(start_utc, end_utc))
        if df.empty:
            return pd.Series(dtype=np.float64)
        df = df.rename(columns={"ts_recv": "ts_event", "mid": "close"})
        # Keep densest symbol only — re-query with primary
        # (without symbol split this mixes contracts; prefer densest second stream)
        # For simplicity take all and last-wins on duplicate seconds.
    s = df.set_index(pd.to_datetime(df["ts_event"], utc=True))["close"].astype(np.float64)
    s.index = s.index.floor("s")
    return s[~s.index.duplicated(keep="last")].sort_index()


# DBN encodes an absent price as UNDEF_PRICE (2^63-1 nanos). It survives a
# ``px > 0`` filter because it is positive, so every price read here must bound
# it from above as well.
_UNDEF_PX = 9.0e8


def _load_underlying_prices_1s(
    conn: psycopg.Connection, d: date, symbols: list[str]
) -> pd.DataFrame:
    """1s mid per *named* underlying future for CT date ``d``.

    Returns a frame indexed by floored-second UTC with one column per symbol.

    Each option is priced against the specific quarterly it settles into
    (``databento.definitions.underlying``), never against a continuous series.
    The previous loader read ``NQ.c.0`` from ``ohlcv_1s`` and, when that returned
    nothing, fell back to *every* NQ outright at once and kept whichever contract
    quoted last in each second. ``NQ.c.0`` has not been ingested since 2026-07-14,
    so from 2026-07-15 every build silently took that fallback: the underlying
    oscillated across NQZ6/NQH7/NQM7/NQU7 — about 1,170 points on 2026-09-29 —
    which prices ATM calls as deep ITM (vega collapses, IV unsolvable) and ATM
    puts as far OTM (IV explodes). That is the call/put asymmetry in
    ``nq_option_surface_1s`` for 2026-08 onward, and it is independent of the
    ADR-151 DTE error.
    """
    if not symbols:
        return pd.DataFrame()
    start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
    end_ct = start_ct + timedelta(days=1)
    sql_q = """
        SELECT ts_recv, symbol, (bid_px + ask_px) / 2.0 AS mid
        FROM databento.bbo_1s
        WHERE ts_recv >= %s AND ts_recv < %s
          AND symbol = ANY(%s)
          AND bid_px > 0 AND ask_px > 0
          AND bid_px < %s AND ask_px < %s
        ORDER BY ts_recv
    """
    df = pd.read_sql(
        sql_q, conn,
        params=(start_ct.astimezone(timezone.utc), end_ct.astimezone(timezone.utc),
                list(symbols), _UNDEF_PX, _UNDEF_PX),
    )
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts_recv"], utc=True).dt.floor("s")
    # last quote wins within a second, per contract
    df = df.drop_duplicates(subset=["symbol", "ts"], keep="last")
    wide = df.pivot(index="ts", columns="symbol", values="mid").sort_index()
    return wide.astype(np.float64)


def compute_greeks_frame(
    bbo: pd.DataFrame,
    defs: pd.DataFrame,
    und_1s: pd.DataFrame | pd.Series,
    *,
    r: float = _RISK_FREE,
) -> pd.DataFrame:
    """Join BBO + defs + underlying price and compute full Greeks (vectorised).

    ``und_1s`` is the wide frame from :func:`_load_underlying_prices_1s` — one
    column per underlying future, indexed by second. Each option is priced against
    the contract named in its own definition row. A bare Series is still accepted
    and applied to every option, which is only correct when a single underlying
    covers the whole chain.

    Returns one row per two-sided option-second that has an underlying price and a
    finite IV. Ready for COPY into ``nq_option_greeks_1s``.
    """
    if bbo.empty or defs.empty or und_1s.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)

    ddef = defs.copy()
    ddef["is_call"] = ddef["instrument_class"].astype(str).str.upper().eq("C")
    ddef["expiry"] = pd.to_datetime(ddef["expiration"], utc=True).dt.tz_convert(_CT).dt.date
    # Prefer explicit underlying from defs; fall back to NQ continuous label.
    und_sym = (
        ddef["underlying"].dropna().astype(str).mode().iloc[0]
        if ddef["underlying"].notna().any()
        else "NQ.c.0"
    )
    ddef["underlying_sym"] = (
        ddef["underlying"].astype(str) if "underlying" in ddef.columns else und_sym
    )
    ddef.loc[ddef["underlying_sym"].isin(("", "nan", "None")), "underlying_sym"] = und_sym
    ddef = ddef[["instrument_id", "symbol", "strike_price", "is_call", "expiry",
                 "expiration", "underlying_sym"]]

    m = bbo.merge(ddef, on="instrument_id", how="inner", suffixes=("", "_def"))
    if m.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)
    if "symbol_def" in m.columns:      # a caller that still supplies bbo.symbol
        m["symbol"] = m["symbol"].fillna(m["symbol_def"])

    m["ts"] = pd.to_datetime(m["ts_recv"], utc=True).dt.floor("s")
    if isinstance(und_1s, pd.Series):
        m["nq_price"] = m["ts"].map(und_1s)
    else:
        # Per-option underlying: forward-fill each contract onto the option grid,
        # then pick the column named by that option's own definition.
        grid = und_1s.reindex(
            und_1s.index.union(pd.DatetimeIndex(m["ts"].unique()))
        ).ffill()
        known = [c for c in grid.columns]
        pos = m["underlying_sym"].map({c: i for i, c in enumerate(known)})
        vals = grid.reindex(m["ts"]).to_numpy()
        row_ix = np.arange(len(m))
        col_ix = pos.to_numpy()
        ok = np.isfinite(col_ix.astype(np.float64))
        px = np.full(len(m), np.nan)
        if ok.any():
            px[ok] = vals[row_ix[ok], col_ix[ok].astype(np.int64)]
        m["nq_price"] = px
    m = m.dropna(subset=["nq_price", "strike_price"])
    if m.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)

    m["mid_px"] = (m["bid_px"].astype(np.float64) + m["ask_px"].astype(np.float64)) / 2.0
    # DTE in years from each quote timestamp to expiration
    exp = pd.to_datetime(m["expiration"], utc=True)
    dte = (exp - m["ts"]).dt.total_seconds().to_numpy() / (365.25 * 86400.0)
    # Keep only still-alive options (positive DTE)
    alive = dte > 0
    m = m.loc[alive].copy()
    dte = dte[alive]
    if m.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)

    g = option_greeks(
        strike=m["strike_price"].to_numpy(),
        mid=m["mid_px"].to_numpy(),
        bid=m["bid_px"].to_numpy(),
        ask=m["ask_px"].to_numpy(),
        is_call=m["is_call"].to_numpy(),
        dte_years=dte,
        underlying=m["nq_price"].to_numpy(),
        r=r,
    )
    for k, v in g.items():
        m[k] = v
    # Drop unsolvable IV (deep ITM/OTM, crossed books, etc.)
    m = m[np.isfinite(m["iv"])].copy()
    if m.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)

    m["strike"] = m["strike_price"].astype(np.float64)
    out = m.rename(columns={"bid_px": "bid_px", "ask_px": "ask_px"})[
        [c for c in _GREEKS_COLS if c in m.columns or c in ("bid_px", "ask_px", "mid", "spread",
                                                            "iv", "delta", "gamma", "vega", "theta",
                                                            "ts", "instrument_id", "symbol", "expiry",
                                                            "strike", "is_call", "underlying_sym",
                                                            "nq_price")]
    ].copy()
    # Ensure column order
    out = pd.DataFrame({
        "ts": m["ts"].values,
        "instrument_id": m["instrument_id"].astype(np.int64).values,
        "symbol": m["symbol"].astype(str).values,
        "expiry": m["expiry"].values,
        "strike": m["strike"].values,
        "is_call": m["is_call"].astype(bool).values,
        "underlying_sym": m["underlying_sym"].astype(str).values,
        "nq_price": m["nq_price"].astype(np.float64).values,
        "bid_px": m["bid_px"].astype(np.float64).values,
        "ask_px": m["ask_px"].astype(np.float64).values,
        "mid": m["mid"].astype(np.float64).values,
        "spread": m["spread"].astype(np.float64).values,
        "iv": m["iv"].astype(np.float64).values,
        "delta": m["delta"].astype(np.float64).values,
        "gamma": m["gamma"].astype(np.float64).values,
        "vega": m["vega"].astype(np.float64).values,
        "theta": m["theta"].astype(np.float64).values,
    })
    return out


def build_surface_from_greeks(greeks: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-option Greeks to one delta-bucket surface row per second.

    Fully vectorised: the leg selection is "for each (second, side, target delta)
    take the row minimising |delta - target|", which is eight grouped argmins over
    the whole day rather than a Python loop over ~54,000 seconds. The loop version
    spent 87% of its time in per-second pandas indexing (``.iloc`` row extraction
    and Series arithmetic) and cost ~155 s/day; this runs in a few seconds.

    Output is row-for-row identical to the loop, including NaN legs.
    """
    if greeks.empty:
        return pd.DataFrame(columns=_SURFACE_DB_COLS)

    g = greeks
    # Pre-filter to the liquid strike lattice + finite IV (what a leg may be drawn from)
    strike = g["strike"].to_numpy()
    # Two strike filters, both of which the loop applied: the caller-level round()
    # mask and delta_bucket_row's own truncating astype("int64") mask. They agree
    # on the real lattice (NQ strikes are integers) but not on fractional strikes,
    # so both are kept to stay exactly equivalent.
    mask = (np.isfinite(g["iv"].to_numpy())
            & (np.round(strike).astype(np.int64) % _STRIKE_GRID == 0)
            & (strike.astype(np.int64) % _STRIKE_GRID == 0))
    g = g.loc[mask, [
        "ts", "strike", "is_call", "iv", "delta", "gamma", "vega", "theta",
        "mid", "spread", "nq_price", "underlying_sym", "expiry",
    ]]
    if g.empty:
        return pd.DataFrame(columns=_SURFACE_DB_COLS)

    g = g.reset_index(drop=True)
    is_call = g["is_call"].to_numpy(dtype=bool)
    sides = {"c": g.loc[is_call], "p": g.loc[~is_call]}

    # One row per distinct second; per-second scalars come from the first quote.
    first = g.groupby("ts", sort=True).first()
    out = pd.DataFrame(index=first.index)
    out["nq_price"] = first["nq_price"].astype(np.float64)
    out["expiry"] = first["expiry"]
    out["underlying_sym"] = first["underlying_sym"].astype(str)

    for tgt in _TARGET_DELTAS:
        tag = "atm" if tgt >= 0.5 else f"{int(tgt * 100):02d}"
        for side, sign in (("c", 1.0), ("p", -1.0)):
            pool = sides[side]
            cols = [f"{side}{tag}_{f}" for f in SURFACE_FIELDS]
            if pool.empty:
                for c in cols:
                    out[c] = np.nan
                continue
            dist = (pool["delta"] - sign * tgt).abs()
            # idxmin over the grouped distance == the loop's argmin, in C
            win_ix = dist.groupby(pool["ts"], sort=True).idxmin()
            win = pool.loc[win_ix.to_numpy(), list(SURFACE_FIELDS)]
            win.index = win_ix.index
            for f, c in zip(SURFACE_FIELDS, cols):
                out[c] = win[f].astype(np.float64)

    leg_cols = [f"{leg}_iv" for leg in SURFACE_LEGS]
    out["n_legs_filled"] = out[leg_cols].notna().sum(axis=1).astype(np.int16)

    out = out.reset_index().rename(columns={"index": "ts"})
    for c in _SURFACE_DB_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[_SURFACE_DB_COLS].sort_values("ts").reset_index(drop=True)


def _cell(v):
    """Normalise a cell for psycopg COPY (NaN → None, timestamps → aware datetime)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.floating, float)) and not np.isfinite(float(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        ts = v.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(v, datetime) and v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    if isinstance(v, np.ndarray):
        return v.item()
    return v


def _copy_df(conn: psycopg.Connection, table: str, df: pd.DataFrame, columns: list[str]) -> int:
    """COPY DataFrame into ``table`` (schema.table) via CSV buffer (fast path).

    Caller is responsible for deleting the day slice before COPY when re-running.
    """
    if df.empty:
        return 0
    if "." in table:
        schema, name = table.split(".", 1)
        table_sql = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(name))
    else:
        table_sql = sql.Identifier(table)
    cols_sql = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    copy_sql = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
    ).format(table_sql, cols_sql)

    sub = df[columns].copy()
    # Normalise timestamps to ISO UTC strings; bools to true/false; NaN → \\N via na_rep
    for c in sub.columns:
        s = sub[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            ts = pd.to_datetime(s, utc=True)
            sub[c] = ts.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        elif pd.api.types.is_bool_dtype(s):
            sub[c] = s.map({True: "t", False: "f", pd.NA: None})
        elif pd.api.types.is_float_dtype(s):
            # Replace non-finite with NA so na_rep emits \\N
            arr = s.to_numpy(dtype=np.float64, copy=True)
            arr[~np.isfinite(arr)] = np.nan
            sub[c] = arr

    buf = io.StringIO()
    sub.to_csv(buf, index=False, header=False, na_rep="\\N")
    data = buf.getvalue().encode("utf-8")
    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(data)
    return len(df)


def _delete_day(conn: psycopg.Connection, d: date) -> None:
    start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
    end_ct = start_ct + timedelta(days=1)
    start_utc = start_ct.astimezone(timezone.utc)
    end_utc = end_ct.astimezone(timezone.utc)
    conn.execute(
        "DELETE FROM databento.nq_option_greeks_1s WHERE ts >= %s AND ts < %s",
        (start_utc, end_utc),
    )
    conn.execute(
        "DELETE FROM databento.nq_option_surface_1s WHERE ts >= %s AND ts < %s",
        (start_utc, end_utc),
    )


def write_surface_parquet(
    surface: pd.DataFrame,
    d: date,
    out_dir: Path = _default_surface_dir(),
) -> Path | None:
    """Write compact surface to ``out_dir/features_YYYY-MM-DD.parquet``."""
    if surface.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"features_{d.isoformat()}.parquet"
    # Index by ts for fold_cache / LSTM loaders
    out = surface.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.set_index("ts").sort_index()
    out.to_parquet(path, engine="pyarrow", compression="zstd")
    return path


def write_greeks_parquet(
    greeks: pd.DataFrame,
    d: date,
    out_dir: Path = _default_greeks_dir(),
) -> Path | None:
    """Write full per-option Greeks day file (block-volume archive; zstd)."""
    if greeks.empty:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"greeks_{d.isoformat()}.parquet"
    out = greeks.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _compress_day_chunks(conn: psycopg.Connection, d: date) -> None:
    """Compress Timescale chunks covering CT date ``d`` (greeks + surface)."""
    start_ct = datetime(d.year, d.month, d.day, tzinfo=_CT)
    end_ct = start_ct + timedelta(days=1)
    start_utc = start_ct.astimezone(timezone.utc)
    end_utc = end_ct.astimezone(timezone.utc)
    for ht in ("nq_option_greeks_1s", "nq_option_surface_1s"):
        rows = conn.execute(
            """
            SELECT chunk_schema, chunk_name
            FROM timescaledb_information.chunks
            WHERE hypertable_schema = 'databento' AND hypertable_name = %s
              AND range_start < %s AND range_end > %s
              AND NOT is_compressed
            """,
            (ht, end_utc, start_utc),
        ).fetchall()
        for schema, name in rows:
            chunk_reg = f"{schema}.{name}"
            try:
                conn.execute("SELECT compress_chunk(%s::regclass)", (chunk_reg,))
                logger.info("compressed_chunk", extra={"chunk": chunk_reg, "date": str(d)})
            except Exception as exc:  # noqa: BLE001
                logger.warning("compress_chunk_failed", extra={"chunk": chunk_reg, "err": str(exc)})


def process_day(
    d: date,
    *,
    force: bool = False,
    write_db: bool = True,
    write_greeks_db: bool = True,
    write_parquet: bool = True,
    write_greeks_parquet: bool = True,
    compress: bool = True,
    parquet_dir: Path = _default_surface_dir(),
    greeks_parquet_dir: Path = _default_greeks_dir(),
    conn: psycopg.Connection | None = None,
) -> dict:
    """Full day pipeline: load → Greeks → surface → DB + parquet.

    Returns a stats dict. Skips if surface already has data for ``d`` unless
    ``force=True``.

    Storage note: full Greeks are ~300–350 MB/day compressed in PG. Full history
    (~285 0DTE days) needs ~90–100 GB free — check disk before long backfills.
    Surface is tiny (~tens of MB total). Greeks are also mirrored to zstd parquet.

    ``write_greeks_db=False`` builds and returns the Greeks but skips the
    ``nq_option_greeks_1s`` COPY. That COPY is 11M rows and ~19 min of a ~24 min
    day — 77% of the whole build — while no code in this repository reads the
    table: the feature engine derives ``iv_``/``gx_`` from ``bbo_1s`` +
    ``definitions`` directly, and the WebUI reads the surface. Use it when the
    surface (and its parquet mirror) is what you actually need.

    When skipping it, mind that ``force`` has already deleted the day's Greeks
    rows, so the table is left without that day rather than holding stale values.
    """
    t0 = pd.Timestamp.utcnow()
    own = conn is None
    if own:
        conn = psycopg.connect(get_dsn())
    try:
        ensure_tables(conn)
        if not force and day_already_done(d, conn):
            return {"date": str(d), "skipped": True, "reason": "already_done"}

        logger.info("process_day_start", extra={"date": str(d)})
        defs = _load_0dte_defs(conn, d)
        if defs.empty:
            return {"date": str(d), "skipped": True, "reason": "no_0dte_defs",
                    "n_defs": 0}
        logger.info("loaded_defs", extra={"date": str(d), "n": len(defs)})

        ids = defs["instrument_id"].astype(int).tolist()
        bbo = _load_0dte_bbo(conn, d, ids)
        n_bbo = len(bbo)
        logger.info("loaded_bbo", extra={"date": str(d), "n": n_bbo})
        und_syms = sorted(
            {str(x) for x in defs.get("underlying", pd.Series(dtype=str)).dropna()
             if str(x) not in ("", "nan", "None")}
        )
        nq = _load_underlying_prices_1s(conn, d, und_syms)
        logger.info("loaded_underlying", extra={
            "date": str(d), "symbols": ",".join(und_syms), "n": len(nq)})
        if nq.empty:
            return {"date": str(d), "skipped": True,
                    "reason": "no_underlying_prices", "underlyings": und_syms}
        greeks = compute_greeks_frame(bbo, defs, nq)
        logger.info("computed_greeks", extra={"date": str(d), "n": len(greeks)})
        # Free raw inputs before surface / COPY
        del bbo, nq, defs
        surface = build_surface_from_greeks(greeks)
        logger.info("built_surface", extra={"date": str(d), "n": len(surface)})

        n_g = n_s = 0
        pq_path = greeks_pq = None
        if write_db and (not greeks.empty or not surface.empty):
            if force:
                _delete_day(conn, d)
            if not greeks.empty and write_greeks_db:
                logger.info("copy_greeks_start", extra={"date": str(d), "n": len(greeks)})
                n_g = _copy_df(conn, "databento.nq_option_greeks_1s", greeks, _GREEKS_COLS)
            if not surface.empty:
                logger.info("copy_surface_start", extra={"date": str(d), "n": len(surface)})
                n_s = _copy_df(conn, "databento.nq_option_surface_1s", surface, _SURFACE_DB_COLS)
            conn.commit()
            logger.info("db_committed", extra={"date": str(d), "n_g": n_g, "n_s": n_s})
            if compress and (n_g or n_s):
                _compress_day_chunks(conn, d)
                conn.commit()

        if write_parquet and not surface.empty:
            pq_path = write_surface_parquet(surface, d, parquet_dir)
        if write_greeks_parquet and not greeks.empty:
            greeks_pq = write_greeks_parquet(greeks, d, greeks_parquet_dir)

        elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
        stats = {
            "date": str(d),
            "skipped": False,
            "n_defs": len(ids),
            "n_bbo": n_bbo,
            "n_greeks": int(len(greeks)),
            "n_surface": int(len(surface)),
            "n_legs_mean": float(surface["n_legs_filled"].mean()) if not surface.empty else 0.0,
            "parquet": str(pq_path) if pq_path else None,
            "greeks_parquet": str(greeks_pq) if greeks_pq else None,
            "elapsed_s": round(elapsed, 1),
        }
        logger.info("process_day_done", extra=stats)
        return stats
    except Exception:
        if own:
            conn.rollback()
        raise
    finally:
        if own:
            conn.close()


def compute_second_batch(
    quotes: pd.DataFrame,
    defs_by_id: dict[int, dict],
    nq_price: float | Mapping[str, float],
    underlying_sym: str,
    ts: datetime,
    *,
    r: float = _RISK_FREE,
) -> tuple[pd.DataFrame, dict | None]:
    """Live path: Greeks + one surface row for a single second of option quotes.

    ``quotes`` columns: instrument_id, bid_px, ask_px (already two-sided).
    ``defs_by_id`` maps instrument_id → {symbol, strike, is_call, expiry, expiration,
    underlying}.

    ``nq_price`` is normally a mapping {underlying symbol → mid}, and each option is
    priced against the contract named in its own ``defs_by_id`` entry. A bare float
    is still accepted and applied to the whole chain, which is only correct when one
    contract underlies every option in it.

    The scalar form was the live twin of the batch fallback fixed in ADR-153: the
    streamer tracked a single "NQ mid" overwritten by whichever outright quoted last,
    so a 0DTE chain on NQZ6 could be priced against NQH7 — which is what stamped
    ``underlying_sym = NQH7`` on the 2026-09-30 QNE chain and split ATM call/put IV
    0.054 against 1.316.

    Returns (greeks_df, surface_row_dict_or_None).
    """
    if quotes.empty:
        return pd.DataFrame(columns=_GREEKS_COLS), None
    if isinstance(nq_price, Mapping):
        fut_mids = {
            str(k): float(v) for k, v in nq_price.items()
            if v is not None and np.isfinite(v) and 0 < float(v) < _UNDEF_PX
        }
        if not fut_mids:
            return pd.DataFrame(columns=_GREEKS_COLS), None
    else:
        if not np.isfinite(nq_price) or not (0 < float(nq_price) < _UNDEF_PX):
            return pd.DataFrame(columns=_GREEKS_COLS), None
        fut_mids = {str(underlying_sym): float(nq_price)}

    rows = []
    for rec in quotes.itertuples(index=False):
        meta = defs_by_id.get(int(rec.instrument_id))
        if meta is None:
            continue
        rows.append({
            "instrument_id": int(rec.instrument_id),
            "symbol": meta["symbol"],
            "strike_price": meta["strike"],
            "is_call": meta["is_call"],
            "expiry": meta["expiry"],
            "expiration": meta["expiration"],
            "underlying": str(meta.get("underlying") or underlying_sym),
            "bid_px": float(rec.bid_px),
            "ask_px": float(rec.ask_px),
            "ts_recv": ts,
        })
    if not rows:
        return pd.DataFrame(columns=_GREEKS_COLS), None

    bbo = pd.DataFrame(rows)
    bbo["symbol"] = bbo["symbol"].astype(str)
    defs = pd.DataFrame([{
        "instrument_id": r["instrument_id"],
        "symbol": r["symbol"],
        "strike_price": r["strike_price"],
        "instrument_class": "C" if r["is_call"] else "P",
        "expiration": r["expiration"],
        "underlying": r["underlying"],
    } for r in rows]).drop_duplicates("instrument_id")

    sec = pd.DatetimeIndex([pd.Timestamp(ts).tz_convert("UTC").floor("s")])
    und_1s = pd.DataFrame({k: [v] for k, v in fut_mids.items()}, index=sec)
    greeks = compute_greeks_frame(bbo, defs, und_1s, r=r)
    if greeks.empty:
        return greeks, None
    surface_df = build_surface_from_greeks(greeks)
    if surface_df.empty:
        return greeks, None
    return greeks, surface_df.iloc[0].to_dict()


def insert_second(
    conn: psycopg.Connection,
    greeks: pd.DataFrame,
    surface_row: dict | None,
) -> None:
    """Insert one second of live Greeks + surface (ON CONFLICT DO NOTHING)."""
    if not greeks.empty:
        cols = _GREEKS_COLS
        values = [tuple(row[c] for c in cols) for _, row in greeks.iterrows()]
        placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"
        # executemany with ON CONFLICT
        sql_ins = f"""
            INSERT INTO databento.nq_option_greeks_1s ({",".join(cols)})
            VALUES {placeholders}
            ON CONFLICT (instrument_id, ts) DO NOTHING
        """
        with conn.cursor() as cur:
            cur.executemany(sql_ins, values)

    if surface_row is not None:
        cols = _SURFACE_DB_COLS
        vals = [surface_row.get(c) for c in cols]
        placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"
        sql_s = f"""
            INSERT INTO databento.nq_option_surface_1s ({",".join(cols)})
            VALUES {placeholders}
            ON CONFLICT (ts) DO NOTHING
        """
        conn.execute(sql_s, vals)
    conn.commit()
