#!/usr/bin/env python3
"""Download + ingest `definition` schema for 34 cross-asset futures products.

Downloads instrument metadata (multiplier, tick size, expiration, instrument_class,
underlying, exchange, etc.) for all products ingested via pipeline_ohlcv_all.py
into databento.ohlcv_1s, plus ES which was pre-populated separately.

NQ is excluded — 16 NQ futures definition rows already exist in databento.definitions.

SAFETY (AGENTS.md mandatory rule): Aborts automatically if any product's definition
download would cost > $0.00 unless --approve is passed explicitly.

Definition schema is L0 metadata — always free on GLBX.MDP3.

Checkpoint: data/legacy/checkpoints/ingest_definitions_cross_asset.json
States: pending → cost_ok → downloaded → ingested → done | error

Usage:
    # Step 1: survey costs (read-only, no downloads)
    python scripts/download_ingest_definitions.py --action cost

    # Step 2: run full pipeline (download + ingest); aborts if any cost > $0.00
    python scripts/download_ingest_definitions.py --action run

    # Run with explicit cost approval
    python scripts/download_ingest_definitions.py --action run --approve

    # Check current progress
    python scripts/download_ingest_definitions.py --action status
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import databento as db
import psycopg
from dotenv import load_dotenv

load_dotenv("/opt/vtech/.env")
sys.path.insert(0, "/opt/vtech")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET       = "GLBX.MDP3"
SCHEMA        = "definition"
STYPE         = "parent"

# 60-day window: captures current front-month + next quarterly contract for all products.
# Definition schema streams one snapshot record per listed instrument per trading day,
# so a short window is orders of magnitude faster than a multi-year range.
# The DB's PRIMARY KEY (symbol) deduplicates on re-run — safe to extend later.
_today        = date.today()
_sixty_days   = _today - __import__('datetime').timedelta(days=60)
INGEST_START  = _sixty_days.isoformat()
INGEST_END    = _today.isoformat()

DEF_OUT_DIR   = Path("/opt/vtech/data/sources/raw/backfill/definitions")
CKPT_PATH     = Path("/opt/vtech/data/legacy/checkpoints/ingest_definitions_cross_asset.json")

# DB insert columns — order must match COPY statement
DB_INSERT_COLS = [
    "ts_event", "instrument_id", "symbol", "instrument_class",
    "strike_price", "expiration", "underlying", "exchange",
    "currency", "min_price_increment", "multiplier",
]

# ---------------------------------------------------------------------------
# Products: (asset_code, FUT_parent_symbol, description)
# NQ excluded — definitions already exist in the DB.
# ES included — has ohlcv_1s data but currently no definitions rows.
# ---------------------------------------------------------------------------

PRODUCTS: list[tuple[str, str, str]] = [
    ("CL",  "CL.FUT",  "Crude Oil WTI"),
    ("NG",  "NG.FUT",  "Nat Gas Henry Hub"),
    ("ES",  "ES.FUT",  "E-Mini S&P 500"),
    ("RB",  "RB.FUT",  "RBOB Gasoline"),
    ("HO",  "HO.FUT",  "Heating Oil"),
    ("GC",  "GC.FUT",  "Gold"),
    ("YM",  "YM.FUT",  "E-Mini Dow"),
    ("ZS",  "ZS.FUT",  "Soybeans"),
    ("ZC",  "ZC.FUT",  "Corn"),
    ("6E",  "6E.FUT",  "Euro FX"),
    ("SR3", "SR3.FUT", "3-Month SOFR"),
    ("HG",  "HG.FUT",  "Copper"),
    ("6J",  "6J.FUT",  "Japanese Yen"),
    ("RTY", "RTY.FUT", "E-Mini Russell 2000"),
    ("SI",  "SI.FUT",  "Silver"),
    ("ZN",  "ZN.FUT",  "10Y T-Note"),
    ("ZW",  "ZW.FUT",  "Chicago SRW Wheat"),
    ("LE",  "LE.FUT",  "Live Cattle"),
    ("6B",  "6B.FUT",  "British Pound"),
    ("6A",  "6A.FUT",  "Australian Dollar"),
    ("ZF",  "ZF.FUT",  "5Y T-Note"),
    ("HE",  "HE.FUT",  "Lean Hog"),
    ("ZB",  "ZB.FUT",  "30Y T-Bond"),
    ("6C",  "6C.FUT",  "Canadian Dollar"),
    ("UB",  "UB.FUT",  "Ultra T-Bond"),
    ("TN",  "TN.FUT",  "Ultra 10Y T-Note"),
    ("ZT",  "ZT.FUT",  "2Y T-Note"),
    ("QM",  "QM.FUT",  "E-mini Crude Oil"),
    ("BTC", "BTC.FUT", "Bitcoin"),
    ("QG",  "QG.FUT",  "E-mini Natural Gas"),
    ("XK",  "XK.FUT",  "Mini Soybeans"),
    ("XC",  "XC.FUT",  "Mini Corn"),
    ("ETH", "ETH.FUT", "Ether"),
    ("XW",  "XW.FUT",  "Mini SRW Wheat"),
    # NQ intentionally omitted — already has 16 futures rows in the DB
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_ingest_definitions")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _dsn() -> str:
    return (
        f"postgresql://{os.environ['PGUSER']}:{os.environ['PGPASSWORD']}"
        f"@{os.environ.get('PGHOST', 'localhost')}:{os.environ.get('PGPORT', '5432')}"
        f"/{os.environ.get('PGDATABASE', 'vtech_market')}"
    )

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_ckpt() -> dict:
    if CKPT_PATH.exists():
        try:
            return json.loads(CKPT_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_ckpt(ckpt: dict) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(ckpt, indent=2))
    tmp.replace(CKPT_PATH)

# ---------------------------------------------------------------------------
# Download: timeseries.get_range() — streaming, no batch job polling needed.
# Definition files are small (metadata), so batch API overhead is unnecessary.
# ---------------------------------------------------------------------------

def _download(asset: str, fut_symbol: str) -> Path:
    """Download definition DBN for one futures parent. Returns path to saved file."""
    out_path = DEF_OUT_DIR / f"{asset}_FUT_definition.dbn.zst"
    DEF_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("  %s: already downloaded (%s, %.1f KB)",
                    asset, out_path.name, out_path.stat().st_size / 1024)
        return out_path

    logger.info("  %s: fetching %s definition %s → %s ...",
                asset, fut_symbol, INGEST_START, INGEST_END)
    client = db.Historical()
    client.timeseries.get_range(
        dataset=DATASET,
        symbols=[fut_symbol],
        schema=SCHEMA,
        stype_in=STYPE,
        start=INGEST_START,
        end=INGEST_END,
        path=out_path,
    )
    size = out_path.stat().st_size
    logger.info("  %s: saved %s (%.1f KB)", asset, out_path.name, size / 1024)
    return out_path

# ---------------------------------------------------------------------------
# Ingest: parse DBN → COPY temp table → INSERT ON CONFLICT (symbol) DO NOTHING
# ---------------------------------------------------------------------------

def _ingest(asset: str, dbn_path: Path) -> int:
    """Parse definition DBN file and populate databento.definitions.

    Uses a COPY-to-temp-table approach so ON CONFLICT handling works cleanly.
    Returns the number of rows actually inserted (conflicts are skipped silently).
    """
    import pandas as pd

    store = db.DBNStore.from_file(str(dbn_path))

    # map_symbols=False: keep raw_symbol as the contract symbol (e.g. CLN5, not CL.FUT)
    # pretty_ts=True: timestamps become tz-aware pd.Timestamps (UTC), compatible with psycopg
    df = store.to_df(map_symbols=False, pretty_ts=True, price_type="float")

    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        logger.warning("  %s: empty DataFrame from %s", asset, dbn_path.name)
        return 0

    # Index is ts_recv for definition schema; ts_event is a regular column
    df = df.reset_index()

    # Rename DBN fields → DB column names
    col_map = {
        "raw_symbol":          "symbol",
        "unit_of_measure_qty": "multiplier",
    }
    df = df.rename(columns=col_map)

    # Fill missing DB columns with None (e.g. strike_price for futures)
    for col in DB_INSERT_COLS:
        if col not in df.columns:
            df[col] = None

    df = df[DB_INSERT_COLS]

    # Drop rows without a usable symbol (spread/combo legs with empty raw_symbol)
    df = df[df["symbol"].notna() & (df["symbol"].astype(str).str.strip() != "")]

    if df.empty:
        logger.warning("  %s: no valid symbol rows after filtering", asset)
        return 0

    cols_sql = ", ".join(DB_INSERT_COLS)

    with psycopg.connect(_dsn()) as conn:
        cur = conn.cursor()

        # Temp table — dropped automatically on commit (ON COMMIT DROP)
        cur.execute("""
            CREATE TEMP TABLE tmp_defs (
                ts_event            TIMESTAMPTZ,
                instrument_id       INTEGER,
                symbol              TEXT,
                instrument_class    CHAR(1),
                strike_price        DOUBLE PRECISION,
                expiration          TIMESTAMPTZ,
                underlying          TEXT,
                exchange            TEXT,
                currency            TEXT,
                min_price_increment DOUBLE PRECISION,
                multiplier          DOUBLE PRECISION
            ) ON COMMIT DROP
        """)

        # COPY DataFrame into temp table
        buf = io.BytesIO()
        df.to_csv(buf, header=False, index=False, na_rep="",
                  quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
        buf.seek(0)

        with cur.copy(
            f"COPY tmp_defs ({cols_sql}) FROM STDIN WITH (FORMAT csv, NULL '')"
        ) as copy:
            while block := buf.read(1 << 16):
                copy.write(block)

        cur.execute("SELECT COUNT(*) FROM tmp_defs")
        n_tmp = cur.fetchone()[0]

        # Insert into real table. A vendor row overwrites one the live streamer
        # reconstructed from a quote symbol (source='synthetic'); an existing
        # vendor row is left alone, so re-running stays a no-op.
        cur.execute(f"""
            INSERT INTO databento.definitions ({cols_sql}, source)
            SELECT {cols_sql}, 'databento' FROM tmp_defs
            ON CONFLICT (symbol) DO UPDATE SET
                ts_event            = EXCLUDED.ts_event,
                instrument_id       = EXCLUDED.instrument_id,
                instrument_class    = EXCLUDED.instrument_class,
                strike_price        = EXCLUDED.strike_price,
                expiration          = EXCLUDED.expiration,
                underlying          = EXCLUDED.underlying,
                exchange            = EXCLUDED.exchange,
                currency            = EXCLUDED.currency,
                min_price_increment = CASE
                    WHEN EXCLUDED.min_price_increment IS NULL
                      OR EXCLUDED.min_price_increment <> EXCLUDED.min_price_increment
                      OR EXCLUDED.min_price_increment > 1e6
                    THEN databento.definitions.min_price_increment
                    ELSE EXCLUDED.min_price_increment END,
                multiplier          = COALESCE(EXCLUDED.multiplier,
                                               databento.definitions.multiplier),
                source              = EXCLUDED.source
            WHERE databento.definitions.source = 'synthetic'
        """)
        rows_inserted = cur.rowcount

        conn.commit()

    logger.info("  %s: %d rows from file → %d inserted (%d conflicts/dupes skipped)",
                asset, n_tmp, rows_inserted, n_tmp - rows_inserted)
    return rows_inserted

# ---------------------------------------------------------------------------
# Action: cost
# ---------------------------------------------------------------------------

def cmd_cost() -> dict[str, float]:
    """Check and print download cost for all 34 products. Returns {asset: cost}."""
    client = db.Historical()
    costs: dict[str, float] = {}

    print(f"\n{'Asset':<6} {'Symbol':<10} {'Description':<30} {'Cost':>10}")
    print("-" * 62)
    total = 0.0

    for asset, fut_symbol, desc in PRODUCTS:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET,
                symbols=[fut_symbol],
                schema=SCHEMA,
                stype_in=STYPE,
                start=INGEST_START,
                end=INGEST_END,
            )
        except Exception as exc:
            print(f"  {asset:<6} {fut_symbol:<10} {desc:<30} ERROR: {exc}")
            costs[asset] = -1.0
            continue

        costs[asset] = cost
        cost_str = "FREE" if cost == 0.0 else f"${cost:.4f}"
        print(f"  {asset:<6} {fut_symbol:<10} {desc:<30} {cost_str:>10}")
        total += cost

    print("-" * 62)
    total_str = "FREE" if total == 0.0 else f"${total:.4f}"
    print(f"  {'TOTAL':<46} {total_str:>10}\n")
    return costs

# ---------------------------------------------------------------------------
# Action: status
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    ckpt = _load_ckpt()
    print(f"\n{'Asset':<6} {'Symbol':<10} {'Description':<30} {'State':<12} {'Rows':>8}")
    print("-" * 72)
    for asset, fut_symbol, desc in PRODUCTS:
        info = ckpt.get(asset, {})
        state = info.get("state", "pending")
        rows  = str(info.get("rows_inserted", "-"))
        print(f"  {asset:<6} {fut_symbol:<10} {desc:<30} {state:<12} {rows:>8}")
    print()

# ---------------------------------------------------------------------------
# Action: run
# ---------------------------------------------------------------------------

def cmd_run(approve: bool) -> None:
    ckpt = _load_ckpt()

    # ── Phase 1: cost check ────────────────────────────────────────────────
    print("\n⏳ [Phase 1/3] Cost check...")
    costs = cmd_cost()

    errors   = {a: c for a, c in costs.items() if c < 0.0}
    non_free = {a: c for a, c in costs.items() if c > 0.0}

    if errors:
        logger.error("Cost check API errors for: %s — aborting.", list(errors))
        sys.exit(1)

    if non_free and not approve:
        print(f"\n⚠️  COST GATE: {len(non_free)} products have non-zero cost: {non_free}")
        print("   Re-run with --approve to proceed after reviewing.")
        sys.exit(1)

    if non_free:
        print(f"\n⚠️  Proceeding with paid downloads (--approve set): {non_free}")

    # Mark cost_ok for pending products
    for asset, _, _ in PRODUCTS:
        if ckpt.get(asset, {}).get("state") in ("downloaded", "ingested", "done"):
            continue
        ckpt.setdefault(asset, {})["state"] = "cost_ok"
        ckpt[asset]["cost_usd"] = costs.get(asset, 0.0)
    _save_ckpt(ckpt)

    # ── Phase 2: download ──────────────────────────────────────────────────
    print(f"\n⏳ [Phase 2/3] Downloading {len(PRODUCTS)} definition files...")
    for i, (asset, fut_symbol, desc) in enumerate(PRODUCTS):
        info = ckpt.get(asset, {})
        if info.get("state") in ("ingested", "done"):
            logger.info("  [%d/%d] %s: already done, skipping",
                        i + 1, len(PRODUCTS), asset)
            continue

        bar = "█" * int((i + 1) / len(PRODUCTS) * 20) + "░" * (20 - int((i + 1) / len(PRODUCTS) * 20))
        print(f"\r  [{bar}] {i+1}/{len(PRODUCTS)} — {asset}", end="", flush=True)

        try:
            dbn_path = _download(asset, fut_symbol)
            ckpt[asset]["state"] = "downloaded"
            ckpt[asset]["dbn_path"] = str(dbn_path)
            _save_ckpt(ckpt)
        except Exception as exc:
            print()  # newline after progress bar
            logger.error("  %s: download failed: %s", asset, exc)
            ckpt[asset]["state"] = "error"
            ckpt[asset]["error"] = str(exc)
            _save_ckpt(ckpt)
            sys.exit(1)

    print(f"\r  [{'█' * 20}] {len(PRODUCTS)}/{len(PRODUCTS)} — all downloaded\n")

    # ── Phase 3: ingest ────────────────────────────────────────────────────
    print(f"⏳ [Phase 3/3] Ingesting into databento.definitions...")
    total_inserted = 0

    for i, (asset, fut_symbol, desc) in enumerate(PRODUCTS):
        info = ckpt.get(asset, {})
        if info.get("state") == "done":
            logger.info("  [%d/%d] %s: already ingested, skipping",
                        i + 1, len(PRODUCTS), asset)
            continue

        dbn_path = Path(
            info.get("dbn_path",
                     str(DEF_OUT_DIR / f"{asset}_FUT_definition.dbn.zst"))
        )
        if not dbn_path.exists():
            logger.error("  %s: DBN file missing: %s", asset, dbn_path)
            sys.exit(1)

        try:
            rows = _ingest(asset, dbn_path)
            total_inserted += rows
            ckpt[asset]["state"] = "done"
            ckpt[asset]["rows_inserted"] = rows
            _save_ckpt(ckpt)
        except Exception as exc:
            logger.error("  %s: ingest failed: %s", asset, exc)
            ckpt[asset]["state"] = "error"
            ckpt[asset]["error"] = str(exc)
            _save_ckpt(ckpt)
            sys.exit(1)

    print(f"\n✅ Complete — {total_inserted:,} new rows inserted into databento.definitions")
    print(f"   Checkpoint: {CKPT_PATH}")
    print(f"   Raw files:  {DEF_OUT_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + ingest definition schema for 34 cross-asset futures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--action", default="cost",
        choices=["cost", "run", "status"],
        help="cost: check costs only (default); run: download+ingest; status: print progress",
    )
    parser.add_argument(
        "--approve", action="store_true",
        help="Proceed even if any download cost > $0.00 (required for non-free data)",
    )
    args = parser.parse_args()

    if args.action == "cost":
        cmd_cost()
    elif args.action == "status":
        cmd_status()
    elif args.action == "run":
        cmd_run(approve=args.approve)


if __name__ == "__main__":
    main()
