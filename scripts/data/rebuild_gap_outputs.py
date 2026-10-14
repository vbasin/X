#!/usr/bin/env python3
"""Rebuild accepted gap inputs into isolated, reviewable downstream candidates.

This runner never downloads, deletes database rows, or compresses chunks. Raw
definition snapshots, not the current definitions table's historical IDs, own
option identity. Run only after raw ingestion has passed acceptance. Publication
is a separate operation which preserves existing files before replacing them.
"""
from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.causal import as_of_index, assert_known_at_or_before

CT = ZoneInfo("America/Chicago")
TERMS = ("strike_price", "expiration", "instrument_class", "underlying", "multiplier")


def utc(value: Any) -> pd.Timestamp:
    """Normalize a timestamp; callers must provide a timezone."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError("asof and availability timestamps must include timezone")
    return ts.tz_convert("UTC")


def causal_option_inputs(
    bbo: pd.DataFrame, definitions: pd.DataFrame, underlying: pd.DataFrame,
    *, source_gaps: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach symbol terms and contract prices available at each quote timestamp.

    Definition IDs are deliberately ignored. Quote IDs are retained. Definitions
    need source=databento and complete terms; future rows never fill past holes.
    Underlying timestamps remain exact, rather than flooring later prices back.
    """
    counts = {"quotes": len(bbo), "missing_definition": 0, "missing_underlying": 0,
              "source_gap_price_exclusions": 0,"source_gap_live_price_exclusions":0}
    if bbo.empty:
        return bbo.copy(), counts
    quotes = bbo.copy().reset_index(drop=True)
    quotes["ts_recv"] = pd.to_datetime(quotes["ts_recv"], utc=True).dt.as_unit("ns")
    defs = definitions.copy()
    if defs.empty:
        counts["missing_definition"] = len(quotes)
        return pd.DataFrame(), counts
    defs["ts_event"] = pd.to_datetime(defs["ts_event"], utc=True).dt.as_unit("ns")
    defs["available_at"] = pd.to_datetime(defs.get("available_at",defs["ts_event"]),utc=True).dt.as_unit("ns")
    defs["expiration"] = pd.to_datetime(defs["expiration"], utc=True)
    good = (defs["source"].eq("databento") & defs["available_at"].notna()
            & defs["instrument_class"].isin(("C", "P"))
            & np.isfinite(pd.to_numeric(defs["strike_price"], errors="coerce"))
            & defs["strike_price"].gt(0) & defs["expiration"].notna()
            & defs["underlying"].notna() & defs["underlying"].astype(str).ne(""))
    defs = defs.loc[good].sort_values("available_at").drop_duplicates(
        ["symbol", "available_at"], keep="last")
    joined = []
    for symbol, q in quotes.groupby("symbol", sort=False,observed=True):
        q = q.sort_values("ts_recv").copy()
        ref = defs.loc[defs["symbol"].eq(symbol)].reset_index(drop=True)
        if ref.empty:
            counts["missing_definition"] += len(q)
            continue
        pos = as_of_index(ref["available_at"].astype("int64").to_numpy(),
                          q["ts_recv"].astype("int64").to_numpy())
        seen = pos >= 0
        counts["missing_definition"] += int((~seen).sum())
        q = q.loc[seen].copy()
        attached = ref.iloc[pos[seen]]
        for column in TERMS:
            q[column] = attached[column].to_numpy()
        q["definition_available_ts"] = attached["available_at"].to_numpy()
        joined.append(q)
    if not joined:
        return pd.DataFrame(), counts
    priced = []
    und = underlying.copy()
    if not und.empty:
        und["ts_recv"] = pd.to_datetime(und["ts_recv"], utc=True).dt.as_unit("ns")
        und["mid"] = (pd.to_numeric(und["bid_px"], errors="coerce")
                      + pd.to_numeric(und["ask_px"], errors="coerce")) / 2
        und = und.loc[np.isfinite(und["mid"]) & und["mid"].gt(0)]
    for symbol, q in pd.concat(joined).groupby("underlying", sort=False):
        q = q.sort_values("ts_recv").copy()
        ref = und.loc[und["symbol"].eq(symbol)].sort_values("ts_recv") if not und.empty else und
        ref = ref.drop_duplicates("ts_recv", keep="last").reset_index(drop=True)
        if ref.empty:
            counts["missing_underlying"] += len(q)
            continue
        pos = as_of_index(ref["ts_recv"].astype("int64").to_numpy(),
                          q["ts_recv"].astype("int64").to_numpy())
        seen = pos >= 0
        counts["missing_underlying"] += int((~seen).sum())
        q = q.loc[seen].copy()
        q["nq_price"] = ref.iloc[pos[seen]]["mid"].to_numpy()
        q["underlying_available_ts"] = ref.iloc[pos[seen]]["ts_recv"].to_numpy()
        for column in ("definition_available_ts", "underlying_available_ts"):
            q[column] = pd.to_datetime(q[column], utc=True)
            assert_known_at_or_before(q[column].astype("int64").to_numpy(),
                                      q["ts_recv"].astype("int64").to_numpy())
        blocked = pd.Series(False,index=q.index)
        for gap_start,gap_end in source_gaps or []:
            lo,hi = utc(gap_start),utc(gap_end)
            # During a known source outage no quote is priced. After it ends,
            # only a fresh underlying quote after the outage can resume pricing.
            blocked |= q["ts_recv"].ge(lo) & (
                q["ts_recv"].lt(hi) | q["underlying_available_ts"].lt(hi))
        counts["source_gap_price_exclusions"] += int(blocked.sum())
        alive = q["ts_recv"].dt.ceil("s") < pd.to_datetime(q["expiration"],utc=True)
        counts["source_gap_live_price_exclusions"] += int((blocked & alive).sum())
        q = q.loc[~blocked].copy()
        priced.append(q)
    out = pd.concat(priced, ignore_index=True) if priced else pd.DataFrame()
    return out, counts


def compute_causal_greeks(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute Greeks, labeling non-grid quotes when they become knowable."""
    from src.features.options_greeks import option_greeks
    from src.features.options_greeks_pipeline import _GREEKS_COLS
    if inputs.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)
    m = inputs.copy()
    m["ts"] = pd.to_datetime(m["ts_recv"], utc=True).dt.ceil("s")
    collisions = m.groupby(["instrument_id", "ts"])["symbol"].nunique()
    if (collisions > 1).any():
        raise ValueError("quote ID maps to multiple raw symbols at the same second")
    m = m.sort_values("ts_recv").drop_duplicates(["instrument_id", "ts"], keep="last")
    bid = pd.to_numeric(m["bid_px"],errors="coerce")
    ask = pd.to_numeric(m["ask_px"],errors="coerce")
    m = m.loc[np.isfinite(bid) & np.isfinite(ask) & bid.gt(0) & ask.ge(bid) & ask.lt(1e6)].copy()
    expiry = pd.to_datetime(m["expiration"], utc=True)
    dte = (expiry - m["ts"]).dt.total_seconds() / (365.25 * 86400)
    alive = dte.gt(0)
    m, dte, expiry = m.loc[alive].copy(), dte.loc[alive], expiry.loc[alive]
    if m.empty:
        return pd.DataFrame(columns=_GREEKS_COLS)
    m["is_call"] = m["instrument_class"].eq("C")
    mid = (m["bid_px"].astype(float) + m["ask_px"].astype(float)) / 2
    values = option_greeks(
        strike=m["strike_price"].to_numpy(dtype=float), mid=mid.to_numpy(),
        bid=m["bid_px"].to_numpy(dtype=float), ask=m["ask_px"].to_numpy(dtype=float),
        is_call=m["is_call"].to_numpy(), dte_years=dte.to_numpy(),
        underlying=m["nq_price"].to_numpy(dtype=float), r=0.05)
    for column, value in values.items():
        m[column] = value
    m["strike"] = m["strike_price"]
    m["underlying_sym"] = m["underlying"]
    m["expiry"] = expiry.dt.tz_convert(CT).dt.date
    return m.loc[np.isfinite(m["iv"]), _GREEKS_COLS].reset_index(drop=True)


def definition_record_availability(record: Any) -> tuple[pd.Timestamp,bool]:
    """Keep snapshot publication time; ordinary updates await vendor receipt.

    InstrumentDefMsg has no snapshot flags. Databento emits its documented
    definition snapshots at UTC midnight Monday-Friday, retaining the earlier
    publication ts_event. Ordinary records use the later event/receive time.
    """
    event = pd.Timestamp(int(record.ts_event),unit="ns",tz="UTC")
    raw_recv = getattr(record,"ts_recv",None)
    received = pd.Timestamp(int(raw_recv),unit="ns",tz="UTC") if raw_recv is not None else event
    snapshot = received == received.floor("D") and received.dayofweek < 5
    return (event if snapshot else max(event,received)),snapshot


def load_vendor_snapshots(paths: list[Path]) -> pd.DataFrame:
    """Read dated vendor DBN records offline, retaining all publication times."""
    import databento as db
    from src.acquisition.definition_acquisition import DEFINITION_COLUMNS, VendorDefinition
    rows = []
    for path in paths:
        for record in db.DBNStore.from_file(path):
            if not hasattr(record, "raw_symbol") or not hasattr(record, "strike_price"):
                continue
            definition = VendorDefinition.from_record(record)
            if definition.option_ready:
                row = list(definition.row())
                # SQL normalization uses Python datetimes (microseconds). The
                # offline as-of join must retain the vendor's full nanoseconds.
                row[0] = pd.Timestamp(int(record.ts_event),unit="ns",tz="UTC")
                row[5] = pd.Timestamp(int(record.expiration),unit="ns",tz="UTC")
                available,snapshot = definition_record_availability(record)
                rows.append((*row,available,snapshot))
    columns = (*DEFINITION_COLUMNS,"available_at","weekday_snapshot")
    return pd.DataFrame(rows, columns=columns).drop_duplicates(
        ["symbol", "ts_event","available_at"], keep="last") if rows else pd.DataFrame(columns=columns)


def closed_rows(frame: pd.DataFrame, asof: Any) -> pd.DataFrame:
    """Only already-closed native bars may be published."""
    if frame.empty:
        return frame.copy()
    return frame.loc[pd.to_datetime(frame["bar_end_ts"], utc=True) <= utc(asof)].copy()


def source_quality(start: Any, end: Any, source_gaps: list[tuple[str,str]]) -> dict[str,Any]:
    """Separate a completed rebuild from completeness of its upstream source."""
    lo,hi = utc(start),utc(end)
    overlaps = [{"start":max(lo,utc(a)).isoformat(),"end":min(hi,utc(b)).isoformat()}
                for a,b in source_gaps if utc(a) < hi and utc(b) > lo]
    return {"source_complete":not overlaps,"unrecoverable_source_gaps":overlaps}


def exclude_source_gap_bars(frame: pd.DataFrame, timeframe: str,
                            source_gaps: list[tuple[str,str]]) -> tuple[pd.DataFrame,int]:
    """Drop any native bar whose print interval overlaps a known source gap."""
    if frame.empty:
        return frame.copy(),0
    from src.features.bar_features import timestep_delta
    ends = pd.to_datetime(frame["bar_end_ts"],utc=True)
    starts = ends - timestep_delta(timeframe)
    blocked = pd.Series(False,index=frame.index)
    for a,b in source_gaps:
        blocked |= starts.lt(utc(b)) & ends.gt(utc(a))
    return frame.loc[~blocked].copy(),int(blocked.sum())


def closed_context_sources(raw: pd.DataFrame, timeframe: str, asof: Any,
                            source_gaps: list[tuple[str,str]]) -> tuple[pd.DataFrame,int]:
    """Filter partial/gapped source candles before HTF rolling calculations."""
    if raw.empty:
        return raw.copy(),0
    from src.features.native_tf_htf import bar_end_from_left
    starts = pd.to_datetime(raw["bucket"],utc=True)
    ends = pd.Series(bar_end_from_left(starts,timeframe),index=raw.index)
    blocked = ends.gt(utc(asof))
    for a,b in source_gaps:
        blocked |= starts.lt(utc(b)) & ends.gt(utc(a))
    return raw.loc[~blocked].copy(),int(blocked.sum())


def definition_receipt_paths(directory: Path) -> list[Path]:
    """Resolve saved vendor snapshot paths from acquisition JSON receipts."""
    paths = []
    for file in sorted(directory.glob("*.json")):
        receipt = json.loads(file.read_text())
        if receipt.get("params",{}).get("schema") == "definition":
            path = Path(receipt["receipt"]["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    return list(dict.fromkeys(paths))


def numeric_tape_side(side: pd.Series) -> pd.Series:
    """Normalize compact DB categoricals without adding a category via fillna."""
    return side.astype(object).map({"B":1,"A":-1,"N":0}).fillna(0).astype("int8")


def option_product_window(day0: Any, day1: Any, expiration: Any, asof: Any,
                          source_gaps: list[tuple[str,str]]) -> dict[str,Any]:
    """0DTE availability ends at vendor expiry; raw-chain coverage is separate."""
    start,expiry,cutoff = utc(day0),utc(expiration),utc(asof)
    end = min(utc(day1),expiry,cutoff)
    if end <= start:
        raise ValueError("no positive 0DTE product availability window")
    exclusive = end+pd.Timedelta(microseconds=1) if end == cutoff and end < expiry else end
    return {"scope_start":start.isoformat(),"scope_end":exclusive.isoformat(),
            "vendor_expiration":expiry.isoformat(),"availability_end":end.isoformat(),
            **source_quality(start,end,source_gaps)}


def evaluate_option_release(run: Path, definitions: pd.DataFrame,
                            source_gaps: list[tuple[str,str]]) -> dict[str,Any]:
    """Retain original run evidence and create an expiry-scoped release manifest."""
    source = run / "manifest.json"
    target = run / "release_manifest.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads(source.read_text())
    for day in manifest["days"]:
        d = date.fromisoformat(day["date"])
        if "error" in day:
            raise ValueError(f"failed option candidate day {d}")
        day0 = pd.Timestamp(datetime.combine(d,datetime.min.time(),CT))
        day1 = min(day0+pd.Timedelta(days=1),utc(manifest["asof"]))
        expiries = pd.to_datetime(definitions["expiration"],utc=True)
        dated = expiries.loc[expiries.dt.tz_convert(CT).dt.date.eq(d)]
        if dated.empty:
            raise ValueError(f"no vendor 0DTE expiration for {d}")
        product_scope = option_product_window(day0,day1,dated.max(),manifest["asof"],source_gaps)
        coverage = day["option_coverage"]
        coverage["raw_calendar_day_source_complete"] = coverage["source_complete"]
        coverage["zero_dte_availability_source_complete"] = product_scope["source_complete"]
        for name in ("greeks","surface"):
            product = day["products"][name]
            frame = pd.read_parquet(product["path"],columns=["ts"])
            times = pd.to_datetime(frame["ts"],utc=True)
            if (times >= dated.max()).any() or (times > utc(manifest["asof"])).any():
                raise ValueError("candidate labels extend to/past expiry or after cutoff")
            product.update(product_scope)
            product["raw_calendar_day_source_complete"] = coverage["source_complete"]
            complete = (product_scope["source_complete"] and not coverage["missing_definition"]
                        and not coverage["missing_underlying"]
                        and not coverage.get("source_gap_live_price_exclusions",
                            coverage["source_gap_price_exclusions"] if any(utc(a) < dated.max() for a,_ in source_gaps) else 0))
            product["publishable"] = bool(product["rows"] and complete)
            product["reason"] = "" if complete else "missing terms/prices or source gap within 0DTE availability"
    manifest["release_scope"] = "normal pipeline 0DTE availability ending at vendor expiry; full raw-chain source quality remains separate"
    manifest["original_manifest"] = str(source)
    target.write_text(json.dumps(manifest,indent=2,default=str)+"\n")
    return manifest


def _run_manifest(run: Path) -> dict[str,Any]:
    release = run / "release_manifest.json"
    return json.loads((release if release.exists() else run / "manifest.json").read_text())


def replace_window(existing: pd.DataFrame, rebuilt: pd.DataFrame,
                   start: Any, end: Any, *, timestamp: str,
                   instrument_ids: list[int] | None = None) -> pd.DataFrame:
    """Replace an accepted scope while preserving all rows outside it."""
    lo, hi = utc(start), utc(end)
    new_ts = pd.to_datetime(rebuilt[timestamp], utc=True)
    if ((new_ts < lo) | (new_ts >= hi)).any():
        raise ValueError("rebuilt rows extend outside accepted replacement window")
    old_ts = pd.to_datetime(existing[timestamp], utc=True)
    affected = (old_ts >= lo) & (old_ts < hi)
    if instrument_ids is not None:
        affected &= existing["instrument_id"].isin(instrument_ids)
    kept = existing.loc[~affected]
    return pd.concat([kept, rebuilt], ignore_index=True).sort_values(timestamp).reset_index(drop=True)


def preserve_then_write(frame: pd.DataFrame, destination: Path, preserved: Path,
                        *, index: bool = False) -> dict[str, str]:
    """Preserve old bytes before an authorized, atomic parquet replacement."""
    old = preserved / destination.name
    if destination.exists():
        old.parent.mkdir(parents=True, exist_ok=True)
        if old.exists():
            raise FileExistsError(f"preserved copy already exists: {old}")
        shutil.copy2(destination, old)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".gap_rebuild.tmp")
    if tmp.exists():
        raise FileExistsError(tmp)
    frame.to_parquet(tmp, index=index, compression="zstd")
    tmp.replace(destination)
    return {"path": str(destination), "preserved": str(old) if old.exists() else ""}


def publish_native_candidate(
    frame: pd.DataFrame, directory: Path, preserved: Path, *, timeframe: str,
    start: Any, end: Any,
) -> list[dict[str, str]]:
    """Publish accepted native rows without deleting existing monthly parts.

    Overlapping old rows are removed from their original part only after its
    exact bytes are preserved. Non-overlapping rows stay in that same file.
    New rows use new part numbers. This avoids the canonical writer's unlink
    behavior and protects adjacent sessions in the same month.
    """
    import re
    from src.features.native_tf import _rows_per_part
    from src.common.data_paths import NATIVE_TF_PART_MAX_BYTES
    lo, hi = utc(start), utc(end)
    times = pd.to_datetime(frame["bar_end_ts"], utc=True)
    if frame.empty or times.duplicated().any() or ((times < lo) | (times >= hi)).any():
        raise ValueError("candidate must contain unique bars inside accepted scope")
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob(f"features_{timeframe}_part*.parquet"))
    changes = []
    highest = 0
    affected = []
    for path in existing:
        match = re.fullmatch(rf"features_{re.escape(timeframe)}_part(\d+)\.parquet",path.name)
        if match:
            highest = max(highest,int(match.group(1)))
        old = pd.read_parquet(path)
        ts = pd.to_datetime(old["bar_end_ts"],utc=True)
        mask = (ts >= lo) & (ts < hi)
        if mask.any():
            if (preserved / path.name).exists():
                raise FileExistsError(preserved / path.name)
            affected.append((path,old.loc[~mask].copy()))
    # All reads and preservation conflicts are checked before first mutation.
    for path,kept in affected:
        changes.append(preserve_then_write(kept,path,preserved,index=False))
    rows_per = _rows_per_part(frame,NATIVE_TF_PART_MAX_BYTES)
    for begin in range(0,len(frame),rows_per):
        highest += 1
        path = directory / f"features_{timeframe}_part{highest}.parquet"
        changes.append(preserve_then_write(frame.iloc[begin:begin+rows_per],path,preserved,index=False))
    return changes


def publish_candidates(run: Path, selected: list[str] | dict[str,list[str]],
                       *, allow_partial_options: bool = False) -> list[dict[str, str]]:
    """Publish explicitly selected, accepted file products; retain DB outputs.

    options are parquet mirrors here: SQL replacement needs its own retained
    row snapshot and identity validation, rather than a forced daily DELETE.
    """
    from src.common.data_paths import native_tf_dir, price_tape_dir, parquet_path1s_dir
    from src.features.session_profile import profile_path, session_levels_path
    manifest = _run_manifest(run)
    record = run / "publication.json"
    if record.exists():
        raise FileExistsError(record)
    allowed = {"native_15s","native_1m","native_5m","native_1h","native_1d","tape","path1s","session_levels","session_profile","greeks","surface"}
    all_selected = set(selected) if isinstance(selected,list) else {name for names in selected.values() for name in names}
    if not all_selected <= allowed:
        raise ValueError("selected product is not enabled for file publication")
    changes = []
    # Reject the entire selection before performing any publication.
    for day in manifest["days"]:
        if "error" in day:
            raise ValueError(f"failed candidate day: {day['date']}")
        names = selected if isinstance(selected,list) else selected.get(day["date"],[])
        for name in names:
            product = day["products"].get(name)
            partial_options = bool(allow_partial_options and name in ("greeks","surface")
                                   and product and product.get("rows"))
            if product is None or (not product.get("publishable") and not partial_options):
                raise ValueError(f"{day['date']} {name}: candidate has not passed release gate")
    for day in manifest["days"]:
        iso = day["date"]
        d = date.fromisoformat(iso)
        names = selected if isinstance(selected,list) else selected.get(iso,[])
        for name in names:
            product = day["products"][name]
            frame = pd.read_parquet(product["path"])
            preserved = run / "preserved" / iso / name
            if name.startswith("native_"):
                from src.features.native_tf import split_by_utc_month
                tf = name.removeprefix("native_")
                for (year,month),part in split_by_utc_month(frame).items():
                    changes.extend(publish_native_candidate(part,native_tf_dir(tf,year=year,month=month),
                        preserved / f"{year}-{month:02d}",timeframe=tf,start=product["scope_start"],end=product["scope_end"]))
            elif name in ("tape","path1s"):
                destination = (price_tape_dir("NQ") / f"tape_{iso}.parquet" if name == "tape"
                               else parquet_path1s_dir() / f"path_{iso}.parquet")
                if destination.exists():
                    old = pd.read_parquet(destination)
                    # Keep any prior rows beyond the accepted current cutoff.
                    beyond = old.loc[pd.to_datetime(old.index,utc=True) > frame.index.max()]
                    frame = pd.concat([frame,beyond]).sort_index(kind="stable")
                changes.append(preserve_then_write(frame,destination,preserved,index=True))
            elif name == "session_levels":
                destination = session_levels_path()
                if destination.exists():
                    old = pd.read_parquet(destination)
                    old_dates = pd.to_datetime(old["session"]).dt.date
                    frame = pd.concat([old.loc[old_dates.ne(d)],frame],ignore_index=True).sort_values("session")
                changes.append(preserve_then_write(frame,destination,preserved))
            elif name == "session_profile":
                changes.append(preserve_then_write(frame,profile_path(d),preserved))
            elif name in ("greeks","surface"):
                from src.common.data_paths import options_greeks_dir, options_surface_dir
                destination = (options_greeks_dir() / f"greeks_{iso}.parquet" if name == "greeks"
                               else options_surface_dir() / f"features_{iso}.parquet")
                if destination.exists():
                    old = pd.read_parquet(destination)
                    if "ts" not in old.columns:
                        old = old.reset_index().rename(columns={old.index.name or "index":"ts"})
                    ids = frame["instrument_id"].astype("int64").unique().tolist() if name == "greeks" else None
                    frame = replace_window(old,frame,product["scope_start"],product["scope_end"],timestamp="ts",instrument_ids=ids)
                if name == "surface":
                    frame = frame.set_index("ts")
                change = preserve_then_write(frame,destination,preserved,index=name == "surface")
                change["source_complete"] = str(product.get("source_complete",True)).lower()
                change["release_mode"] = "complete" if product.get("publishable") else "partial_causal_rows"
                changes.append(change)
            else:
                raise ValueError(f"file publication is not enabled for {name}")
    record.write_text(json.dumps(changes,indent=2)+"\n")
    return changes


def option_upsert_statement(table: str, columns: list[str], keys: list[str],
                            *, append_only: bool) -> Any:
    """Build a bounded staging merge; compressed existing keys are not rewritten."""
    from psycopg import sql
    if table not in ("nq_option_greeks_1s","nq_option_surface_1s"):
        raise ValueError("unsupported option output table")
    target = sql.Identifier("databento",table)
    stage = sql.Identifier("gap_output_stage")
    cols = sql.SQL(",").join(sql.Identifier(c) for c in columns)
    selected = sql.SQL(",").join(sql.Identifier("s",c) for c in columns)
    keycols = sql.SQL(",").join(sql.Identifier(c) for c in keys)
    equality = sql.SQL(" AND ").join(sql.SQL("t.{}=s.{}").format(sql.Identifier(k),sql.Identifier(k)) for k in keys)
    prefix = sql.SQL("INSERT INTO {} AS kept ({}) SELECT {} FROM {} s WHERE s.ts >= %s AND s.ts < %s ").format(target,cols,selected,stage)
    if append_only:
        return prefix + sql.SQL("AND NOT EXISTS (SELECT 1 FROM {} t WHERE t.ts >= %s AND t.ts < %s AND {}) ON CONFLICT ({}) DO NOTHING").format(target,equality,keycols)
    updates = sql.SQL(",").join(sql.SQL("{}=EXCLUDED.{}").format(sql.Identifier(c),sql.Identifier(c)) for c in columns if c not in keys)
    kept = sql.SQL(",").join(sql.Identifier("kept",c) for c in columns)
    excluded = sql.SQL(",").join(sql.Identifier("excluded",c) for c in columns)
    return prefix + sql.SQL("ON CONFLICT ({}) DO UPDATE SET {} WHERE kept.ts >= %s AND kept.ts < %s AND ROW({}) IS DISTINCT FROM ROW({})").format(keycols,updates,kept,excluded)


def file_sha256(path: Path) -> str:
    """Fingerprint retained source files without loading them into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_option_publication_attempt(source: Path, destination: Path) -> dict[str,Any]:
    """Resume an interrupted publication in a new directory, retaining originals.

    Hourly merges are idempotent. The new attempt snapshots its starting DB state;
    every prior snapshot remains in its original attempt directory. Candidate
    files are referenced without rewriting them and checked against retained
    provenance when available. Stop the prior invocation before calling this.
    """
    if destination.exists():
        raise FileExistsError(destination)
    manifest = _run_manifest(source)
    prior_path = source / "publication_provenance.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    fingerprints = {}
    for day in manifest["days"]:
        for name in ("greeks","surface"):
            product = day["products"][name]
            path = Path(product["path"]).resolve()
            value = file_sha256(path)
            if str(path) in prior.get("candidates",{}) and prior["candidates"][str(path)] != value:
                raise ValueError("candidate changed since retained publication provenance")
            fingerprints[str(path)] = value
    if prior.get("candidates") and set(prior["candidates"]) != set(fingerprints):
        raise ValueError("candidate inventory changed since retained publication provenance")
    manifests = {name:file_sha256(source / name) for name in ("manifest.json","release_manifest.json") if (source / name).exists()}
    if prior.get("manifests") and prior["manifests"] != manifests:
        raise ValueError("manifest changed since retained publication provenance")
    for path,value in prior.get("prior_snapshots",{}).items():
        if file_sha256(Path(path)) != value:
            raise ValueError("retained original DB snapshot has changed")
    provenance = {"source_attempt":str(source.resolve()),"original_attempt":prior.get("original_attempt",str(source.resolve())),
                  "candidates":fingerprints,"manifests":manifests,"prior_snapshots":dict(prior.get("prior_snapshots",{})),
                  "recovery":"repeat idempotent hourly merges in a fresh attempt; retain all previous attempts"}
    for path in (source / "preserved_db").rglob("*.parquet"):
        provenance["prior_snapshots"][str(path.resolve())] = file_sha256(path)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ("manifest.json","release_manifest.json"):
        path = source / name
        if path.exists():
            shutil.copy2(path,destination / name)
    (destination / "publication_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return provenance


def option_candidate_scope(frame: pd.DataFrame, keys: list[str], start: Any, end: Any) -> list[int]:
    """Validate exact 1s canonical keys before the normal pipeline CSV COPY."""
    ts = pd.to_datetime(frame["ts"],utc=True)
    if (frame[keys].isna().any().any() or frame.duplicated(keys).any()
            or ts.isna().any() or ts.ne(ts.dt.floor("s")).any()
            or ((ts < utc(start)) | (ts >= utc(end))).any()):
        raise ValueError("candidate keys must be non-null, unique whole seconds inside accepted scope")
    return frame["instrument_id"].astype("int64").unique().tolist() if "instrument_id" in keys else []


def compressed_append_only(chunks: list[tuple]) -> bool:
    """A compressed overlap forbids rewriting even an initially empty ID scope."""
    return any(bool(chunk[-1]) for chunk in chunks)


def classify_sql_payload_differences(rows: list[dict[str,Any]], numeric_columns: set[str]) -> dict[str,int]:
    """Distinguish legacy SQL missing-number representations from changed values."""
    material = representation = 0
    for row in rows:
        changed = False
        for column,values in row["changes"].items():
            def missing(value: Any) -> bool:
                return value is None or (column in numeric_columns and
                    ((isinstance(value,str) and value in {"NaN","Infinity","-Infinity"})
                     or (isinstance(value,(float,np.floating)) and not np.isfinite(value))))
            if not (missing(values["old"]) and missing(values["new"])):
                changed = True
        material += int(changed)
        representation += int(not changed)
    return {"exact_sql_differing_rows":len(rows),"material_differing_rows":material,
            "legacy_missing_representation_only_rows":representation}


def validate_option_parquets(run: Path) -> dict[str,Any]:
    """Independently compare all candidate payloads and retained outside scopes.

    Row-group reads stay bounded. No canonical files are modified. Exact ordered
    comparisons include every key and payload column, with nulls retained.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    from itertools import zip_longest,groupby
    from src.common.vm_config import get_vm_config
    record = run / "parquet_publication_validation.json"
    if record.exists():
        raise FileExistsError(record)
    manifest = _run_manifest(run)
    publications = {row["path"]:row for row in json.loads((run / "publication.json").read_text())}
    from src.common.data_paths import options_greeks_dir,options_surface_dir

    def validate(day: dict[str,Any], name: str) -> dict[str,Any]:
        product = day["products"][name]
        source = Path(product["path"])
        target = (options_greeks_dir() / f"greeks_{day['date']}.parquet" if name == "greeks"
                  else options_surface_dir() / f"features_{day['date']}.parquet")
        publication = publications[str(target)]
        canonical_path = target
        if name == "surface":
            # The live streamer keeps appending after our accepted cutoff.
            # Validate one retained, coherent version instead of reopening the
            # changing canonical path for its inside/outside checks and hash.
            target = run / "validation_inputs" / day["date"] / "surface.parquet"
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(canonical_path,target)
        columns = pq.ParquetFile(source).schema_arrow.names
        reference_schema = pa.unify_schemas([pq.ParquetFile(source).schema_arrow,
            pa.schema([pq.ParquetFile(target).schema_arrow.field(c) for c in columns])],promote_options="permissive")
        ids = set()
        if name == "greeks":
            for batch in pq.ParquetFile(source).iter_batches(batch_size=100_000,columns=["instrument_id"],use_threads=False):
                ids.update(pc.unique(batch.column(0)).to_pylist())
        start,end = utc(product["scope_start"]),utc(product["scope_end"])

        def stream(path: Path, inside: bool | None):
            carry = None
            for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000,columns=columns,use_threads=False):
                # Retained rows can widen an integer dtype during Pandas concat;
                # compare at the common wider type, never narrowing payloads.
                table = pa.Table.from_batches([batch]).select(columns).cast(reference_schema,safe=True)
                if inside is not None:
                    field = table["ts"]
                    mask = pc.and_(pc.greater_equal(field,pa.scalar(start,type=field.type)),
                                 pc.less(field,pa.scalar(end,type=field.type)))
                    if ids:
                        mask = pc.and_(mask,pc.is_in(table["instrument_id"],value_set=pa.array(sorted(ids),type=table["instrument_id"].type)))
                    table = table.filter(mask if inside else pc.invert(mask))
                if not table.num_rows:
                    continue
                carry = table if carry is None else pa.concat_tables([carry,table])
                while carry.num_rows >= 100_000:
                    yield carry.slice(0,100_000)
                    carry = carry.slice(100_000)
            if carry is not None and carry.num_rows:
                yield carry

        def equal(left, right, *, check_keys: bool = False) -> int:
            count = 0
            previous_ts,previous_ids = None,set()
            keys = ["instrument_id","ts"] if name == "greeks" else ["ts"]
            for a,b in zip_longest(left,right):
                if a is None or b is None or not a.equals(b,check_metadata=False):
                    raise RuntimeError(f"{day['date']} {name}: canonical payload differs from retained source")
                if check_keys:
                    frame = a.select(keys).to_pandas()
                    if frame.duplicated(keys).any() or not frame["ts"].is_monotonic_increasing:
                        raise RuntimeError("candidate keys duplicate or unordered")
                    first,last = frame["ts"].iloc[0],frame["ts"].iloc[-1]
                    head_ids = set(frame.loc[frame["ts"].eq(first),"instrument_id"]) if ids else {None}
                    if previous_ts is not None and (first < previous_ts or (first == previous_ts and previous_ids & head_ids)):
                        raise RuntimeError("candidate keys duplicate across stream boundaries")
                    previous_ts = last
                    previous_ids = set(frame.loc[frame["ts"].eq(last),"instrument_id"]) if ids else {None}
                count += a.num_rows
            return count

        matched = equal(stream(source,None),stream(target,True),check_keys=True)
        if matched != product["rows"]:
            raise RuntimeError("canonical candidate row count differs from accepted manifest")
        saved = Path(publication["preserved"]) if publication.get("preserved") else None
        def outside_subset(left,right) -> tuple[int,int]:
            def rows(tables):
                for table in tables:
                    yield from table.to_pylist()
            original = groupby(rows(left),key=lambda row:row["ts"])
            current = iter((ts,list(group)) for ts,group in groupby(rows(right),key=lambda row:row["ts"]))
            observed = next(current,None)
            preserved_rows = extra_rows = 0
            for ts,group in original:
                wanted = list(group)
                while observed is not None and observed[0] < ts:
                    extra_rows += len(observed[1])
                    observed = next(current,None)
                if observed is None or observed[0] != ts:
                    raise RuntimeError("outside-scope original timestamp was lost")
                index = {row.get("instrument_id"):row for row in observed[1]}
                if len(index) != len(observed[1]):
                    raise RuntimeError("outside-scope canonical keys duplicate")
                if any(index.get(row.get("instrument_id")) != row for row in wanted):
                    raise RuntimeError("outside-scope original payload was changed")
                preserved_rows += len(wanted)
                extra_rows += len(observed[1])-len(wanted)
                observed = next(current,None)
            if observed is not None:
                extra_rows += len(observed[1])
            extra_rows += sum(len(group) for _,group in current)
            return preserved_rows,extra_rows
        outside,new_outside = outside_subset(stream(saved,False) if saved else iter(()),stream(target,False))
        result = {"date":day["date"],"product":name,"canonical":str(canonical_path),"candidate":str(source),
                  "candidate_keys":matched,"duplicate_keys":0,"payload_differences":0,
                  "outside_scope_rows_preserved":outside,"new_live_outside_scope_rows":new_outside,
                  "canonical_rows":pq.ParquetFile(target).metadata.num_rows,"validated_snapshot":str(target) if target != canonical_path else "",
                  "candidate_sha256":file_sha256(source),"canonical_sha256":file_sha256(target),
                  "preserved":str(saved) if saved else "","preserved_sha256":file_sha256(saved) if saved else ""}
        print(f"[{'█'*20}] validate parquet {day['date']} {name}: {matched:,} exact payloads; {outside:,} outside-scope rows retained",flush=True)
        return result

    tasks = [(day,name) for day in manifest["days"] for name in ("greeks","surface")]
    workers = min(len(tasks),get_vm_config().workers_for(mem_per_worker_gb=1.0))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda task:validate(*task),tasks))
    receipt = {"validation":"independent bounded Arrow stream key/payload equality and retained outside scopes", "workers":workers,"results":results}
    record.write_text(json.dumps(receipt,indent=2)+"\n")
    return receipt


def publish_option_database(run: Path, dates: list[str], *, allow_partial: bool = False) -> dict[str,Any]:
    """Preserve affected DB rows and merge causal 0DTE output candidates.

    Compressed chunks use missing-key append only. Their older differing values
    remain explicitly pending, and never cause bulk decompression/recompression.
    No DDL on permanent tables or indexes, deletes, or truncation is performed.
    """
    import psycopg
    from psycopg import sql
    from src.common.db import get_dsn
    from src.features.options_greeks_pipeline import _copy_df,_GREEKS_COLS,_SURFACE_DB_COLS
    manifest = _run_manifest(run)
    selected = [day for day in manifest["days"] if day["date"] in dates]
    if len(selected) != len(set(dates)):
        raise ValueError("requested date is absent from candidate manifest")
    record = run / "database_publication.json"
    if record.exists():
        raise FileExistsError(record)
    provenance_path = run / "publication_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for path,value in provenance["candidates"].items():
            if file_sha256(Path(path)) != value:
                raise ValueError("candidate changed since publication recovery was prepared")
        for name,value in provenance.get("manifests",{}).items():
            if file_sha256(run / name) != value:
                raise ValueError("manifest changed since publication recovery was prepared")
        for path,value in provenance.get("prior_snapshots",{}).items():
            if file_sha256(Path(path)) != value:
                raise ValueError("retained original DB snapshot has changed")
    for day in selected:
        for name in ("greeks","surface"):
            product = day.get("products",{}).get(name,{})
            if not product.get("rows") or (not product.get("publishable") and not allow_partial):
                raise ValueError(f"{day['date']} {name}: candidate has not passed release gate")
            if (run / "preserved_db" / day["date"] / f"{name}.parquet").exists():
                raise FileExistsError("DB rows already preserved for this candidate")
    results = []
    # Check both canonical schemas before committing any of the six products.
    with psycopg.connect(get_dsn()) as conn:
        for table,columns,keys in (("nq_option_greeks_1s",_GREEKS_COLS,["instrument_id","ts"]),
                                   ("nq_option_surface_1s",_SURFACE_DB_COLS,["ts"])):
            actual = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",("databento",table)).fetchall()
            if not set(columns) <= {row[0] for row in actual}:
                raise ValueError("option output schema differs from current pipeline")
            constraints = conn.execute("SELECT array_agg(a.attname ORDER BY x.ord) FROM pg_constraint c CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY x(attnum,ord) JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=x.attnum WHERE c.conrelid=to_regclass(%s) AND c.contype IN ('p','u') GROUP BY c.oid",("databento."+table,)).fetchall()
            if keys not in [list(row[0]) for row in constraints]:
                raise ValueError("option output conflict key differs from current pipeline")
    for day in selected:
        for name,table,columns,keys in (
            ("greeks","nq_option_greeks_1s",_GREEKS_COLS,["instrument_id","ts"]),
            ("surface","nq_option_surface_1s",_SURFACE_DB_COLS,["ts"]),
        ):
            product = day["products"][name]
            start,end = utc(product["scope_start"]),utc(product["scope_end"])
            frame = pd.read_parquet(product["path"])
            ts = pd.to_datetime(frame["ts"],utc=True)
            candidate_ids = option_candidate_scope(frame,keys,start,end)
            scope_params = (start,end,candidate_ids) if candidate_ids else (start,end)
            with psycopg.connect(get_dsn(),options="-c statement_timeout=300000 -c lock_timeout=5000") as conn:
                actual = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",("databento",table)).fetchall()
                if not set(columns) <= {row[0] for row in actual}:
                    raise ValueError("option output schema differs from current pipeline")
                constraints = conn.execute("SELECT array_agg(a.attname ORDER BY x.ord) FROM pg_constraint c CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY x(attnum,ord) JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=x.attnum WHERE c.conrelid=to_regclass(%s) AND c.contype IN ('p','u') GROUP BY c.oid",("databento."+table,)).fetchall()
                if keys not in [list(row[0]) for row in constraints]:
                    raise ValueError("option output conflict key differs from current pipeline")
                chunks = conn.execute("SELECT chunk_schema,chunk_name,range_start,range_end,is_compressed FROM timescaledb_information.chunks WHERE hypertable_schema=%s AND hypertable_name=%s AND range_start < %s AND range_end > %s ORDER BY range_start",("databento",table,end,start)).fetchall()
                target = sql.Identifier("databento",table)
                colsql = sql.SQL(",").join(sql.Identifier(c) for c in columns)
                scope = sql.SQL("ts >= %s AND ts < %s")
                target_scope = sql.SQL("t.ts >= %s AND t.ts < %s")
                if candidate_ids:
                    scope += sql.SQL(" AND instrument_id=ANY(%s)")
                    target_scope += sql.SQL(" AND t.instrument_id=ANY(%s)")
                existing = _read(conn,sql.SQL("SELECT {} FROM {} WHERE {} ORDER BY ts").format(colsql,target,scope),scope_params)
                saved = run / "preserved_db" / day["date"] / f"{name}.parquet"
                saved.parent.mkdir(parents=True,exist_ok=True)
                existing.to_parquet(saved,index=False,compression="zstd")
                compressed_existing = compressed_append_only(chunks)
                conn.execute(sql.SQL("CREATE TEMP TABLE gap_output_stage (LIKE {} INCLUDING DEFAULTS) ON COMMIT PRESERVE ROWS").format(target))
                for offset in range(0,len(frame),50_000):
                    _copy_df(conn,"gap_output_stage",frame.iloc[offset:offset+50_000],columns)
                    copied = min(offset+50_000,len(frame))
                    filled = 20*copied//len(frame)
                    print(f"[{'█'*filled}{'░'*(20-filled)}] COPY {day['date']} {name} {copied:,}/{len(frame):,}",flush=True)
                conn.execute("CREATE INDEX gap_output_stage_ts_idx ON gap_output_stage (ts)")
                conn.execute("ANALYZE gap_output_stage")
                equality = sql.SQL(" AND ").join(sql.SQL("t.{}=s.{}").format(sql.Identifier(k),sql.Identifier(k)) for k in keys)
                row_t = sql.SQL(",").join(sql.Identifier("t",c) for c in columns)
                row_s = sql.SQL(",").join(sql.Identifier("s",c) for c in columns)
                comparison = sql.SQL("SELECT count(*),count(*) FILTER(WHERE ROW({}) IS DISTINCT FROM ROW({})) FROM {} t JOIN gap_output_stage s ON {} WHERE {}").format(row_t,row_s,target,equality,target_scope)
                matched,differing = conn.execute(comparison,scope_params).fetchone()
                merge = option_upsert_statement(table,columns,keys,append_only=compressed_existing)
                busiest = ts.dt.floor("h").value_counts().idxmax()
                plan_start,plan_end = max(start,busiest),min(busiest+pd.Timedelta(hours=1),end)
                plan = conn.execute(sql.SQL("EXPLAIN (FORMAT JSON) ")+merge,(plan_start,plan_end,plan_start,plan_end)).fetchone()[0]
                plan_path = run / "preserved_db" / day["date"] / f"{name}_merge_plan.json"
                plan_path.write_text(json.dumps(plan,indent=2,default=str)+"\n")
                affected = 0
                hour_start = start
                while hour_start < end:
                    hour_end = min(hour_start+pd.Timedelta(hours=1),end)
                    if shutil.disk_usage(run).free < 20*1024**3:
                        raise RuntimeError("option publication stopped at 20GiB disk reserve")
                    cursor = conn.execute(merge,(hour_start,hour_end,hour_start,hour_end))
                    affected += cursor.rowcount
                    conn.commit()
                    checkpoint = {"date":day["date"],"product":name,"completed_through":hour_end.isoformat(),"affected_rows":affected,"preserved":str(saved)}
                    progress_path = run / "database_publication_progress.jsonl"
                    with progress_path.open("a") as stream:
                        stream.write(json.dumps(checkpoint,default=str)+"\n")
                    print(f"[{'█'*20}] option DB {day['date']} {name}: merged through {hour_end.isoformat()}",flush=True)
                    hour_start = hour_end
                matched_after,differing_after = conn.execute(comparison,scope_params).fetchone()
                material_after = differing_after
                if differing_after:
                    floating = {row[0] for row in conn.execute("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s AND data_type=ANY(%s)",("databento",table,["real","double precision"])).fetchall()}
                    def normalized(alias: str):
                        parts = []
                        for column in columns:
                            field = sql.Identifier(alias,column)
                            parts.append(sql.SQL("CASE WHEN {} IN ('NaN'::double precision,'Infinity'::double precision,'-Infinity'::double precision) THEN NULL ELSE {} END").format(field,field) if column in floating else field)
                        return sql.SQL(",").join(parts)
                    material_query = sql.SQL("SELECT count(*) FROM {} t JOIN gap_output_stage s ON {} WHERE {} AND ROW({}) IS DISTINCT FROM ROW({}) AND ROW({}) IS DISTINCT FROM ROW({})").format(target,equality,target_scope,row_t,row_s,normalized("t"),normalized("s"))
                    material_after = conn.execute(material_query,scope_params).fetchone()[0]
                if matched_after != len(frame) or (not compressed_existing and differing_after):
                    raise RuntimeError("canonical option output does not match accepted staging keys")
                total = conn.execute(sql.SQL("SELECT count(*) FROM {} WHERE {}").format(target,scope),scope_params).fetchone()[0]
                conn.commit()
            result = {"date":day["date"],"product":name,"scope_start":start.isoformat(),"scope_end":end.isoformat(),
                "candidate_rows":len(frame),"existing_rows_before":len(existing),"canonical_rows_after":total,
                "affected_rows":affected,"matched_existing_before":matched,"differing_existing_before":differing,
                "retained_unmatched_existing_rows":total-len(frame),"candidate_keys_persisted":matched_after,
                "candidate_payload_differences_after":differing_after,"append_only_compressed_existing":compressed_existing,
                "material_payload_differences_after":material_after,
                "legacy_missing_representation_differences_after":differing_after-material_after,
                "pending_compression_boundary":bool(compressed_existing and material_after),
                "source_complete":product.get("source_complete",True),"preserved":str(saved),
                "actual_quote_instrument_ids":candidate_ids,"merge_plan":str(plan_path),
                "compressed_chunks":[f"{schema}.{chunk}" for schema,chunk,_,_,compressed in chunks if compressed]}
            results.append(result)
            print(f"[{'█'*20}] option DB {day['date']} {name}: {affected} affected; {material_after} material differences; {differing_after-material_after} missing-representation differences",flush=True)
    report = {"scope":"normal pipeline 0DTE Greeks/surface; all raw option chains recovered separately", "results":results}
    record.write_text(json.dumps(report,indent=2,default=str)+"\n")
    return report


def _read(conn: Any, query: str, params: tuple) -> pd.DataFrame:
    from src.features.engine import frame_from_cursor
    with conn.cursor(name=f"gap_read_{time.monotonic_ns()}") as cur:
        cur.execute(query, params)
        return frame_from_cursor(cur, [column.name for column in cur.description])


def blocked_feature_product() -> dict[str, Any]:
    """Record unresolved causal/runtime requirements without constructing an engine."""
    return {"status":"blocked","rows":0,"publishable":False,"reasons":[
        "legacy IV/GEX 30s underlying-price lookahead requires a causal fix",
        "legacy underlying 5s last-price labels can expose later prices within the interval",
        "default training hours are 08:00-11:00 America/Chicago; full-day scope is unvalidated",
        "data_dir does not isolate canonical paths and private compute can refresh CAGGs",
        "live feature memory/service cap must be validated before restoration",
    ]}


def rebuild_day(payload: tuple[str, str, str, list[str], list[str], list[tuple[str,str]]]) -> dict[str, Any]:
    """Build one day in an isolated run directory using read-only DB connections."""
    iso, asof_iso, run_path, definition_paths, products, source_gaps = payload
    if set(products) == {"features"}:
        # Stop before vendor-file parsing, database imports or engine creation.
        out = Path(run_path) / "candidates" / iso
        out.mkdir(parents=True,exist_ok=False)
        result = {"date":iso,"asof":utc(asof_iso).isoformat(),"status":"blocked",
                  "known_source_gaps":source_gaps,"products":{"features_5s":blocked_feature_product()}}
        (out / "manifest.json").write_text(json.dumps(result,indent=2)+"\n")
        return result
    import psycopg
    from src.common.contract_calendar import active_contract
    from src.common.db import get_dsn
    from src.features.native_tf import globex_session_window, load_tbbo
    d, asof, run = date.fromisoformat(iso), utc(asof_iso), Path(run_path)
    out = run / "candidates" / iso
    out.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {"date": iso, "asof": asof.isoformat(), "products": {}}
    defs = load_vendor_snapshots([Path(path) for path in definition_paths])
    start, end = globex_session_window(d)
    end = min(pd.Timestamp(end), asof)
    symbol = active_contract("NQ", d)
    result["globex_source_quality"] = source_quality(start,end,source_gaps)
    with psycopg.connect(get_dsn(), options="-c default_transaction_read_only=on -c statement_timeout=300000") as conn:
        if "context" in products:
            from src.features.native_tf_htf import (
                clip_bar_end,compute_htf_features,globex_bounds,load_cagg,select_lead_month,
            )
            warm0,_ = globex_bounds(d,d)
            for tf in ("1h","1d"):
                raw = select_lead_month(load_cagg(conn,tf,warm0,end))
                ready,removed = closed_context_sources(raw,tf,asof,source_gaps)
                candles = ready.set_index("bucket")[["open","high","low","close","volume"]]
                frame = compute_htf_features(candles,timeframe=tf,symbol=ready.set_index("bucket")["symbol"])
                frame = clip_bar_end(closed_rows(frame,asof),d,d)
                path = out / f"native_{tf}.parquet"
                frame.to_parquet(path,index=False,compression="zstd")
                result["products"][f"native_{tf}"] = {"path":str(path),"rows":len(frame),"publishable":not frame.empty,
                    **source_quality(start,end,source_gaps),"excluded_partial_or_gap_source_candles":removed,
                    "warmup_start":pd.Timestamp(warm0).isoformat(),"reason":"" if not frame.empty else "no already-closed context bar for session",
                    "scope_start":(pd.Timestamp(start)+pd.Timedelta(nanoseconds=1)).isoformat(),"scope_end":(end+pd.Timedelta(nanoseconds=1)).isoformat()}
        if "futures" in products:
            ticks = load_tbbo(conn, start, end, symbol)
            from src.features.native_tf_1m import prev_session_date
            prior = prev_session_date(d)
            p0, p1 = globex_session_window(prior)
            prev = load_tbbo(conn, p0, p1, active_contract("NQ", prior))
            for tf, module in (("15s", "bar_features"), ("1m", "native_tf_1m"), ("5m", "native_tf_5m")):
                mod = importlib.import_module(f"src.features.{module}")
                if tf == "15s":
                    from src.features.target_geometry import attach_target_geometry
                    frame = attach_target_geometry(mod.compute_bar_features(ticks, tf))
                else:
                    frame = getattr(mod, f"compute_{tf}_features")(ticks, prev_tbbo=prev)
                frame = closed_rows(frame, asof)
                frame,excluded = exclude_source_gap_bars(frame,tf,source_gaps)
                frame["_meta_nq_contract"] = symbol
                path = out / f"native_{tf}.parquet"
                frame.to_parquet(path, index=False, compression="zstd")
                result["products"][f"native_{tf}"] = {"path": str(path), "rows": len(frame), "publishable": not frame.empty,
                    **source_quality(start,end,source_gaps),"excluded_source_gap_bars":excluded,
                    "scope_start": (pd.Timestamp(start) + pd.Timedelta(nanoseconds=1)).isoformat(), "scope_end": (end + pd.Timedelta(nanoseconds=1)).isoformat()}
            cash0 = pd.Timestamp(datetime.combine(d, datetime.min.time(), CT)) + pd.Timedelta(hours=7)
            cash1 = min(cash0 + pd.Timedelta(hours=8, minutes=15), asof)
            tape = _read(conn, "SELECT ts_event,price,size,side,bid_px,ask_px,bid_sz,ask_sz FROM databento.tbbo WHERE symbol=%s AND ts_event >= %s AND ts_event < %s ORDER BY ts_event,sequence", (symbol,cash0,cash1))
            if not tape.empty:
                from src.features.tape import tape_to_path
                tape = tape.rename(columns={"ts_event": "ts"}).set_index("ts")
                tape.index = pd.to_datetime(tape.index, utc=True)
                tape["side"] = numeric_tape_side(tape["side"])
                tape_path = out / "tape.parquet"
                tape.to_parquet(tape_path, compression="zstd")
                path = tape_to_path(tape)
                path_file = out / "path1s.parquet"
                path.to_parquet(path_file, compression="zstd")
                for name,file,frame in (("tape",tape_path,tape),("path1s",path_file,path)):
                    result["products"][name] = {"path":str(file),"rows":len(frame),"publishable":True}
                close = pd.Timestamp(datetime(d.year,d.month,d.day,15,tzinfo=CT))
                if asof >= close:
                    from src.features.session_profile import build_one
                    built = build_one(tape,d)
                    if built:
                        row,hist = built
                        for name,frame in (("session_levels",pd.DataFrame([row])),("session_profile",hist)):
                            file = out / f"{name}.parquet"
                            frame.to_parquet(file,index=False,compression="zstd")
                            result["products"][name] = {"path":str(file),"rows":len(frame),"publishable":True}
                else:
                    result["products"]["session_profile"] = {"publishable":False,"reason":"cash session not closed"}
        if "options" in products:
            from src.acquisition.options_universe import root_for
            from src.features.options_greeks_pipeline import build_surface_from_greeks
            day0 = pd.Timestamp(datetime.combine(d,datetime.min.time(),CT))
            day1 = min(day0 + pd.Timedelta(days=1), asof)
            expiration = pd.to_datetime(defs["expiration"],utc=True).dt.tz_convert(CT).dt.date
            zero = defs.loc[expiration.eq(d)]
            symbols = zero["symbol"].unique().tolist()
            und_symbols = zero["underlying"].dropna().unique().tolist()
            chain_root = root_for(d)
            # Inventory comes from the raw-symbol quote population, including
            # symbols absent from vendor snapshots. Never filter by dated IDs.
            bbo = _read(conn,"SELECT ts_recv,instrument_id,symbol,bid_px,ask_px FROM databento.bbo_1s WHERE ts_recv >= %s AND ts_recv < %s AND split_part(symbol,' ',1)=%s ORDER BY ts_recv",(day0,day1,chain_root))
            # Warm the day's first quotes with already-known preceding prices.
            # Known source gaps below still reject any cross-outage carry.
            underlying = _read(conn,"SELECT ts_recv,symbol,bid_px,ask_px FROM databento.bbo_1s WHERE ts_recv >= %s AND ts_recv < %s AND symbol=ANY(%s) ORDER BY ts_recv",(day0-pd.Timedelta(minutes=5),day1,und_symbols))
            inputs, counts = causal_option_inputs(bbo,zero,underlying,source_gaps=source_gaps)
            greeks = compute_causal_greeks(inputs)
            greeks = greeks.loc[pd.to_datetime(greeks["ts"],utc=True) <= asof]
            surface = build_surface_from_greeks(greeks)
            quality = source_quality(day0,day1,source_gaps)
            product_quality = option_product_window(day0,day1,pd.to_datetime(zero["expiration"],utc=True).max(),asof,source_gaps) if not zero.empty else quality
            complete = bool(len(bbo) and not counts["missing_definition"] and not counts["missing_underlying"]
                            and not counts["source_gap_live_price_exclusions"] and product_quality["source_complete"])
            result["option_coverage"] = {**counts,**quality,"raw_calendar_day_source_complete":quality["source_complete"],
                "zero_dte_availability_source_complete":product_quality["source_complete"],
                "vendor_0dte_symbols":len(symbols),"quote_symbols":int(bbo["symbol"].nunique()),"chain_root":chain_root}
            for name,frame in (("greeks",greeks),("surface",surface)):
                file = out / f"{name}.parquet"
                frame.to_parquet(file,index=False,compression="zstd")
                result["products"][name] = {"path":str(file),"rows":len(frame),"publishable":complete and not frame.empty,
                    **product_quality,"raw_calendar_day_source_complete":quality["source_complete"],
                    "reason":"" if complete else "missing causal vendor terms/prices or source gap within 0DTE availability"}
    if "features" in products:
        result["products"]["features_5s"] = blocked_feature_product()
        result["status"] = "blocked"
    (out / "manifest.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates",required=True,help="comma-separated session dates")
    parser.add_argument("--asof",required=True,help="UTC closed-bar cutoff")
    parser.add_argument("--run-dir",type=Path,required=True,help="new isolated output directory")
    parser.add_argument("--definition-files",nargs="*",type=Path,default=[])
    parser.add_argument("--acquisition-receipts",type=Path,help="saved acquisition receipt directory with vendor definition files")
    parser.add_argument("--source-gap",nargs=2,action="append",default=[],metavar=("START","END"),
                        help="known unrecoverable UTC source gap; may be repeated")
    parser.add_argument("--products",default="futures,options,features")
    parser.add_argument("--raw-accepted",action="store_true",help="raw inputs have passed the operator acceptance gate")
    args = parser.parse_args()
    if not args.raw_accepted:
        parser.error("candidate rebuild requires --raw-accepted after ingestion validation")
    if args.run_dir.exists():
        parser.error("run-dir must be a new directory")
    asof = utc(args.asof)
    dates = [date.fromisoformat(item).isoformat() for item in args.dates.split(",")]
    if len(set(dates)) != len(dates):
        parser.error("dates must be distinct")
    products = args.products.split(",")
    if not set(products) <= {"futures","options","features","context"}:
        parser.error("unknown product")
    source_gaps = []
    for a,b in args.source_gap:
        if utc(b) <= utc(a):
            parser.error("source-gap END must follow START")
        source_gaps.append((utc(a).isoformat(),utc(b).isoformat()))
    definitions = [*args.definition_files]
    if args.acquisition_receipts:
        definitions.extend(definition_receipt_paths(args.acquisition_receipts))
    from src.common.vm_config import get_vm_config
    vm = get_vm_config()
    # Full-chain matrices and today's 15M+ row 0DTE quote days can peak near
    # 20 GB. Futures-only products use the normal 7 GB feature-worker budget.
    budget = 20.0 if set(products) & {"features","options"} else 7.0
    workers = min(len(dates),vm.workers_for(mem_per_worker_gb=budget))
    args.run_dir.mkdir(parents=True)
    started = time.monotonic()
    results = []
    payloads = [(iso,asof.isoformat(),str(args.run_dir),[str(p) for p in definitions],products,source_gaps) for iso in dates]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(rebuild_day,payload):payload[0] for payload in payloads}
        for done,future in enumerate(as_completed(futures),1):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"date":futures[future],"error":f"{type(exc).__name__}: {exc}"})
            filled = 20*done//len(futures)
            print(f"[{'█'*filled}{'░'*(20-filled)}] rebuild {done}/{len(futures)}",flush=True)
            if results[-1].get("status") == "blocked":
                print(f"{results[-1]['date']}: BLOCKED features_5s; see candidate manifest for causal/runtime requirements",flush=True)
    (args.run_dir / "manifest.json").write_text(json.dumps({"asof":asof.isoformat(),"workers":workers,
        "requested_products":products,"known_source_gaps":source_gaps,"days":results},indent=2,default=str)+"\n")
    print(f"✅ Done in {time.monotonic()-started:.1f}s",flush=True)
    return int(any("error" in result or result.get("status") == "blocked" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
