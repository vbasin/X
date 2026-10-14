"""Always-on warm live feature runtime — per-bar full-schema appends.

Design
------
* **One long-lived process** (systemd ``vtech-live-features``): imports, DB pool
  state, definitions, and sliding raw-data windows stay warm.
* **Per closed 5s bar**: fetch only *new* ticks since the last watermark, append
  to in-memory lookback buffers, recompute the full ~183-col feature set on those
  warm frames (no multi-hour DB reload), append **only new rows** into the 5s
  hive, then score via ``src.webui.live.build_snapshot``.
* **Heartbeat** at ``ops/webui/live_features_heartbeat.json`` so WebUI can skip
  its own heavy ``extend_date`` when this runtime is healthy.

This is the live critical path. Research still uses ``FeatureEngine.build_date``.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.contract_calendar import active_contract
from src.common.data_paths import webui_dir
from src.features.engine import (
    FeatureEngine,
    last_closed_bar_ts,
    merge_feature_rows,
    timestep_seconds,
)

logger = logging.getLogger("vtech.features.live_runtime")

_HEARTBEAT_PATH = Path(os.environ.get(
    "VTECH_LIVE_FEATURES_HEARTBEAT",
    str(webui_dir() / "live_features_heartbeat.json"),
))
# How long WebUI trusts the external runtime without falling back to extend_date.
_HEARTBEAT_FRESH_S = float(os.environ.get("VTECH_LIVE_FEATURES_HEARTBEAT_FRESH_S", "45"))
# Sliding raw-data window kept in RAM for rolling modules (cs_ma_120 needs ~2h).
_LOOKBACK = pd.Timedelta(hours=float(os.environ.get("VTECH_LIVE_FEATURES_LOOKBACK_H", "2")))
# Options BBO is heavy — keep a shorter warm window; IV/GEX still full schema.
# Options BBO volume is huge even with primary_only; keep this tight for live.
_OPTIONS_LOOKBACK = pd.Timedelta(
    minutes=float(os.environ.get("VTECH_LIVE_FEATURES_OPTIONS_LOOKBACK_MIN", "3"))
)
# How often to reload/recompute IV+GEX. Between refreshes, iv_/gx_ are ffilled
# so the 5s hot path stays under the parquet-lag budget.
_OPTIONS_EVERY_S = float(os.environ.get("VTECH_LIVE_FEATURES_OPTIONS_EVERY_S", "20"))
# Only the last N minutes of options BBO are passed into options_surface/GEX
# on a refresh tick (surface is snapshot-like; rolling IV lives on the parquet).
_OPTIONS_COMPUTE_LOOKBACK = pd.Timedelta(
    seconds=float(os.environ.get("VTECH_LIVE_FEATURES_OPTIONS_COMPUTE_S", "90"))
)
# Pad after bar boundary before reading ticks (allow late prints).
_BAR_PAD_S = float(os.environ.get("VTECH_LIVE_FEATURES_BAR_PAD_S", "0.25"))
# When more than this many bars behind, catch up in one warm compute pass.
_MAX_BARS_PER_TICK = int(os.environ.get("VTECH_LIVE_FEATURES_MAX_BARS", "60"))


def heartbeat_path() -> Path:
    return _HEARTBEAT_PATH


def is_runtime_fresh(*, max_age_s: float | None = None) -> bool:
    """True when the always-on runtime heartbeat is recent and ok."""
    path = _HEARTBEAT_PATH
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    if not data.get("ok"):
        return False
    ts = data.get("ts")
    if not ts:
        return False
    try:
        hb = pd.Timestamp(ts)
        if hb.tzinfo is None:
            hb = hb.tz_localize("UTC")
        else:
            hb = hb.tz_convert("UTC")
    except Exception:  # noqa: BLE001
        return False
    age = (pd.Timestamp.now(tz="UTC") - hb).total_seconds()
    limit = _HEARTBEAT_FRESH_S if max_age_s is None else max_age_s
    return age <= limit


def read_heartbeat() -> dict[str, Any]:
    path = _HEARTBEAT_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def tbbo_to_1m_candles(tbbo: pd.DataFrame) -> pd.DataFrame:
    """Build 1-minute OHLCV from TBBO trades — always as fresh as last print.

    The ``tbbo_1min`` continuous aggregate can lag several minutes on the live
    path (and we skip cagg refresh on hot deltas). Candle-structure features
    (``cs_*``) require 1m bars; synthesizing them from warm TBBO keeps the
    trailing 5s edge populated.
    """
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return pd.DataFrame()
    t = tbbo.copy()
    ts_col = "ts_event" if "ts_event" in t.columns else "ts_recv"
    if ts_col not in t.columns:
        return pd.DataFrame()
    t[ts_col] = pd.to_datetime(t[ts_col], utc=True)
    t = t.set_index(ts_col).sort_index()
    ohlc = t["price"].resample("1min").ohlc()
    if "size" in t.columns:
        ohlc["volume"] = t["size"].resample("1min").sum()
    else:
        ohlc["volume"] = t["price"].resample("1min").count()
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    if ohlc.empty:
        return ohlc
    as_of = ohlc.index.max()
    if not t.empty:
        as_of = t.index.max()
    from src.features.closed_bars import drop_incomplete_1m
    return drop_incomplete_1m(ohlc, as_of=as_of)


def _write_heartbeat(payload: dict[str, Any]) -> None:
    path = _HEARTBEAT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        **payload,
        "ts": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str))
    tmp.replace(path)


def _tz_utc(ts: pd.Timestamp | datetime | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _concat_trim(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    *,
    time_col: str | None,
    keep_from: pd.Timestamp,
) -> pd.DataFrame:
    """Append *new* rows and drop rows older than *keep_from*."""
    if new is None or new.empty:
        base = existing
    elif existing is None or existing.empty:
        base = new.copy()
    else:
        base = pd.concat([existing, new], ignore_index=time_col is not None)
        if time_col is None and isinstance(base.index, pd.DatetimeIndex):
            base = base.loc[~base.index.duplicated(keep="last")].sort_index()
        elif time_col is not None and time_col in base.columns:
            base = base.drop_duplicates(subset=[time_col], keep="last")
            base = base.sort_values(time_col)

    if base is None or base.empty:
        return pd.DataFrame() if existing is None else existing

    if time_col is None and isinstance(base.index, pd.DatetimeIndex):
        idx = base.index
        if idx.tz is None:
            base = base.copy()
            base.index = idx.tz_localize("UTC")
        return base.loc[base.index >= keep_from]

    if time_col is not None and time_col in base.columns:
        col = pd.to_datetime(base[time_col], utc=True)
        return base.loc[col >= keep_from].copy()

    return base


class LiveFeatureRuntime:
    """Warm always-on feature materializer + score trigger."""

    def __init__(self) -> None:
        # Align feature day filter with active objective session (not the
        # research default 08:00–11:00) so live continues through RTH close.
        train_start, train_end, tz_name = self._objective_hours()
        self.engine = FeatureEngine(
            training_start=train_start,
            training_end=train_end,
            timezone=tz_name,
        )
        self.engine._keep_conn = True
        self.step_s = timestep_seconds(self.engine.timestep)
        self.session_date: date | None = None
        self.features: pd.DataFrame = pd.DataFrame()
        self.tbbo: pd.DataFrame = pd.DataFrame()
        self.bbo_fut: pd.DataFrame = pd.DataFrame()
        self.bbo_opt: pd.DataFrame = pd.DataFrame()
        self.candles: pd.DataFrame = pd.DataFrame()
        self.defs: pd.DataFrame = pd.DataFrame()
        self.stats: pd.DataFrame = pd.DataFrame()
        self.mbp_depth: pd.DataFrame = pd.DataFrame()
        self.watermark: pd.Timestamp | None = None  # last raw tick fetch end
        self._opt_watermark: pd.Timestamp | None = None
        self._last_options_at: float = 0.0
        self._opt_symbols: list[str] = []
        self._stop = False
        self._last_score_bar: pd.Timestamp | None = None

    @staticmethod
    def _objective_hours() -> tuple[str, str, str]:
        """Return (training_start, training_end, timezone) for the live objective."""
        try:
            from src.webui.live import active_objective_id, objective_trade_params
            trade = objective_trade_params(active_objective_id())
            start = (
                trade.get("training_start")
                or trade.get("trading_start")
                or "08:00"
            )
            end = (
                trade.get("training_end")
                or trade.get("trading_end")
                or "14:45"
            )
            tz_name = trade.get("timezone") or os.environ.get(
                "VTECH_TIMEZONE", "America/Chicago"
            )
            return str(start), str(end), str(tz_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_runtime_objective_hours_fallback: %s", exc)
            return (
                os.environ.get("VTECH_TRAINING_HOURS_START", "08:00"),
                os.environ.get("VTECH_TRAINING_HOURS_END", "14:45"),
                os.environ.get("VTECH_TIMEZONE", "America/Chicago"),
            )

    def request_stop(self, *_args) -> None:
        self._stop = True

    # ── session / disk ────────────────────────────────────────────

    def _session_date_today(self) -> date:
        """Calendar date for the feature parquet (training session, 08:00 CT)."""
        from src.webui.live import session_context
        try:
            return session_context(roll_at="training_start")["session_date"]
        except Exception:  # noqa: BLE001
            now = pd.Timestamp.now(tz="UTC").tz_convert(self.engine.timezone)
            return now.date()

    def _restore_disk_if_memory_ahead(self, d: date) -> None:
        """Rewrite hive month parts if in-memory features extend past disk."""
        if self.features.empty:
            return
        disk = self._load_disk_features(d)
        mem_last = _tz_utc(self.features.index.max())
        disk_last = _tz_utc(disk.index.max()) if not disk.empty else None
        if disk_last is not None and mem_last is not None and mem_last <= disk_last:
            if len(self.features) <= len(disk):
                return
        if mem_last is None:
            return
        try:
            self.engine._persist_day_features(self.features)
            logger.info(
                "live_runtime_restored_disk date=%s mem_rows=%d disk_rows=%d last=%s",
                d, len(self.features), len(disk), mem_last.isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_runtime_restore_disk_failed: %s", exc)

    def _load_disk_features(self, d: date) -> pd.DataFrame:
        return self.engine._load_day_features(d)

    # ── warm buffer bootstrap / delta fetch ───────────────────────

    def _ensure_session(self, d: date) -> None:
        if self.session_date == d:
            return
        logger.info("live_runtime_session_switch old=%s new=%s", self.session_date, d)
        self.session_date = d
        self.engine._primary_nq = active_contract("NQ", d)
        self.engine._primary_es = active_contract("ES", d)
        self.features = self._load_disk_features(d)
        self.tbbo = pd.DataFrame()
        self.bbo_fut = pd.DataFrame()
        self.bbo_opt = pd.DataFrame()
        self.candles = pd.DataFrame()
        self.defs = pd.DataFrame()
        self.stats = pd.DataFrame()
        self.mbp_depth = pd.DataFrame()
        self.watermark = None
        self._opt_watermark = None
        self._last_options_at = 0.0
        self._opt_symbols = []
        self._last_score_bar = None

    def _bootstrap_buffers(self, d: date, end: pd.Timestamp) -> None:
        """One-shot load of lookback raw data into warm buffers.

        TBBO / futures BBO / candles: full rolling lookback (cheap, NQ-only).
        Options BBO: shorter window (dominant cost) — still full IV/GEX schema.
        """
        start = end - _LOOKBACK
        opt_start = end - _OPTIONS_LOOKBACK
        eng = self.engine
        eng._primary_nq = active_contract("NQ", d)
        eng._primary_es = active_contract("ES", d)
        t0 = time.time()
        logger.info(
            "live_runtime_bootstrap_buffers date=%s start=%s opt_start=%s end=%s",
            d, start.isoformat(), opt_start.isoformat(), end.isoformat(),
        )
        self.defs = eng._load_definitions(d)
        self.stats = eng._load_statistics(d)
        self.tbbo = eng._load_tbbo(d, t_start=start, t_end=end)
        self.candles = eng._load_candles_1min(
            d, t_start=start, t_end=end, refresh_cagg=True,
        )
        # Futures BBO only (lead contract) — small.
        self.bbo_fut = eng._load_bbo_fut_primary(
            d, eng._primary_nq, t_start=start, t_end=end,
        )
        self.mbp_depth = eng._load_mbp10_depth_bars(d, t_start=start, t_end=end)
        # Options: primary-filtered chain, shorter lookback.
        bbo_opt_raw = eng._load_bbo_1s(
            d, t_start=opt_start, t_end=end, primary_only=True,
        )
        if not self.defs.empty and "instrument_class" in self.defs.columns:
            self._opt_symbols = (
                self.defs.loc[self.defs["instrument_class"].isin(("C", "P")), "symbol"]
                .astype(str).drop_duplicates().tolist()
            )
        if not bbo_opt_raw.empty and self._opt_symbols:
            opt_symbols = set(self._opt_symbols)
            self.bbo_opt = bbo_opt_raw[bbo_opt_raw["symbol"].isin(opt_symbols)].copy()
            # Any futures that slipped into the primary filter stay in bbo_fut
            fut_extra = bbo_opt_raw[~bbo_opt_raw["symbol"].isin(opt_symbols)]
            if not fut_extra.empty:
                nq = eng._primary_nq
                fut_extra = fut_extra[fut_extra["symbol"] == nq] if "symbol" in fut_extra.columns else fut_extra
                if not fut_extra.empty:
                    self.bbo_fut = pd.concat([self.bbo_fut, fut_extra], ignore_index=True)
                    self.bbo_fut = self.bbo_fut.drop_duplicates(
                        subset=["ts_recv", "symbol"], keep="last",
                    )
        else:
            self.bbo_opt = pd.DataFrame()
        # Prefer TBBO-derived 1m candles for recency (cagg often lags live).
        self._refresh_candles_from_tbbo()
        self._trim_buffers(end)
        self.watermark = end
        self._opt_watermark = end
        self._last_options_at = time.time()
        logger.info(
            "live_runtime_bootstrap_done elapsed_s=%.2f tbbo=%d bbo_fut=%d bbo_opt=%d candles=%d",
            time.time() - t0,
            len(self.tbbo), len(self.bbo_fut), len(self.bbo_opt), len(self.candles),
        )

    def _refresh_candles_from_tbbo(self) -> None:
        """Merge TBBO-synthesized 1m OHLCV into the warm candle buffer."""
        synth = tbbo_to_1m_candles(self.tbbo)
        if synth.empty:
            return
        # Keep only OHLC+volume columns candle_structure needs.
        keep_cols = [c for c in ("open", "high", "low", "close", "volume") if c in synth.columns]
        synth = synth[keep_cols]
        if self.candles.empty:
            self.candles = synth
            return
        # Align existing cagg candles to OHLC set where possible.
        base = self.candles
        if not isinstance(base.index, pd.DatetimeIndex):
            self.candles = synth
            return
        if base.index.tz is None:
            base = base.copy()
            base.index = base.index.tz_localize("UTC")
        else:
            base = base.copy()
            base.index = base.index.tz_convert("UTC")
        cols = [c for c in keep_cols if c in base.columns]
        if not cols:
            self.candles = synth
            return
        base = base[cols]
        merged = pd.concat([base, synth])
        merged = merged.loc[~merged.index.duplicated(keep="last")].sort_index()
        self.candles = merged

    def _split_bbo(self, bbo_all: pd.DataFrame) -> None:
        if bbo_all is None or bbo_all.empty:
            self.bbo_fut = pd.DataFrame()
            self.bbo_opt = pd.DataFrame()
            return
        defs = self.defs
        if not defs.empty and "instrument_class" in defs.columns:
            opt_symbols = set(defs.loc[defs["instrument_class"].isin(("C", "P")), "symbol"])
        else:
            opt_symbols = set()
        if opt_symbols:
            self.bbo_fut = bbo_all[~bbo_all["symbol"].isin(opt_symbols)].copy()
            self.bbo_opt = bbo_all[bbo_all["symbol"].isin(opt_symbols)].copy()
        else:
            self.bbo_fut = bbo_all.copy()
            self.bbo_opt = pd.DataFrame()
        if not self.bbo_fut.empty and "symbol" in self.bbo_fut.columns:
            nq = self.engine._primary_nq
            self.bbo_fut = self.bbo_fut[self.bbo_fut["symbol"] == nq].copy()

    def _trim_buffers(self, end: pd.Timestamp, *, trim_options: bool = True) -> None:
        keep = end - _LOOKBACK
        opt_keep = end - _OPTIONS_LOOKBACK
        if not self.tbbo.empty and "ts_event" in self.tbbo.columns:
            col = pd.to_datetime(self.tbbo["ts_event"], utc=True)
            self.tbbo = self.tbbo.loc[col >= keep]
        if not self.bbo_fut.empty and "ts_recv" in self.bbo_fut.columns:
            col = pd.to_datetime(self.bbo_fut["ts_recv"], utc=True)
            self.bbo_fut = self.bbo_fut.loc[col >= keep]
        if trim_options and not self.bbo_opt.empty and "ts_recv" in self.bbo_opt.columns:
            col = pd.to_datetime(self.bbo_opt["ts_recv"], utc=True)
            self.bbo_opt = self.bbo_opt.loc[col >= opt_keep]
        if not self.candles.empty:
            idx = self.candles.index
            if isinstance(idx, pd.DatetimeIndex):
                if idx.tz is None:
                    self.candles = self.candles.copy()
                    self.candles.index = idx.tz_localize("UTC")
                self.candles = self.candles.loc[self.candles.index >= keep]
        if not self.mbp_depth.empty:
            if "ts" in self.mbp_depth.columns:
                col = pd.to_datetime(self.mbp_depth["ts"], utc=True)
                self.mbp_depth = self.mbp_depth.loc[col >= keep]
            elif isinstance(self.mbp_depth.index, pd.DatetimeIndex):
                idx = self.mbp_depth.index
                if idx.tz is None:
                    mbp = self.mbp_depth.copy()
                    mbp.index = idx.tz_localize("UTC")
                    self.mbp_depth = mbp
                self.mbp_depth = self.mbp_depth.loc[self.mbp_depth.index >= keep]

    def _fetch_delta(self, d: date, end: pd.Timestamp, *, load_options: bool = False) -> None:
        """Pull only ticks after watermark into warm buffers.

        *load_options*: if False (5s hot path), skip options BBO — that query is
        ~10s and dominates parquet lag. IV/GEX are refreshed on a slower cadence.
        """
        eng = self.engine
        if self.watermark is None:
            self._bootstrap_buffers(d, end)
            return
        start = self.watermark
        # Overlap slightly to avoid boundary misses
        start = start - pd.Timedelta(seconds=1)
        if start >= end:
            return
        t0 = time.time()
        tbbo_new = eng._load_tbbo(d, t_start=start, t_end=end)
        mbp_new = eng._load_mbp10_depth_bars(d, t_start=start, t_end=end)
        # Fast path: TBBO only. Mid/1m candles are derived from TBBO; skip
        # extra BBO/cagg round-trips (~5s). Options load is cadence-gated.
        fut_new = pd.DataFrame()
        candles_new = pd.DataFrame()
        opt_raw = pd.DataFrame()
        if load_options:
            fut_new = eng._load_bbo_fut_primary(
                d, eng._primary_nq, t_start=start, t_end=end,
            )
            opt_start = self._opt_watermark or start
            opt_start = opt_start - pd.Timedelta(seconds=1)
            opt_raw = eng._load_bbo_window(d, t_start=opt_start, t_end=end)

        # defs/stats: refresh at most when empty (day-static enough for live)
        if self.defs.empty:
            self.defs = eng._load_definitions(d)
        if self.stats.empty:
            self.stats = eng._load_statistics(d)

        if not mbp_new.empty:
            if self.mbp_depth.empty:
                self.mbp_depth = mbp_new.copy()
            else:
                self.mbp_depth = pd.concat([self.mbp_depth, mbp_new], ignore_index=True)
                if "ts" in self.mbp_depth.columns:
                    self.mbp_depth = self.mbp_depth.drop_duplicates(subset=["ts"], keep="last")

        # Merge TBBO
        if not tbbo_new.empty:
            if self.tbbo.empty:
                self.tbbo = tbbo_new
            else:
                self.tbbo = pd.concat([self.tbbo, tbbo_new], ignore_index=True)
                self.tbbo = self.tbbo.drop_duplicates(
                    subset=["ts_event", "symbol"] if "symbol" in self.tbbo.columns else ["ts_event"],
                    keep="last",
                )

        if not fut_new.empty:
            if self.bbo_fut.empty:
                self.bbo_fut = fut_new.copy()
            else:
                self.bbo_fut = pd.concat([self.bbo_fut, fut_new], ignore_index=True)
                self.bbo_fut = self.bbo_fut.drop_duplicates(
                    subset=["ts_recv", "symbol"], keep="last",
                )

        if not opt_raw.empty:
            defs = self.defs
            if not defs.empty and "instrument_class" in defs.columns:
                opt_symbols = set(defs.loc[defs["instrument_class"].isin(("C", "P")), "symbol"])
                opt_new = opt_raw[opt_raw["symbol"].isin(opt_symbols)]
            else:
                opt_new = opt_raw
            if not opt_new.empty:
                if self.bbo_opt.empty:
                    self.bbo_opt = opt_new.copy()
                else:
                    self.bbo_opt = pd.concat([self.bbo_opt, opt_new], ignore_index=True)
                    self.bbo_opt = self.bbo_opt.drop_duplicates(
                        subset=["ts_recv", "symbol"], keep="last",
                    )

        if not candles_new.empty:
            if self.candles.empty:
                self.candles = candles_new
            else:
                self.candles = pd.concat([self.candles, candles_new])
                self.candles = self.candles.loc[~self.candles.index.duplicated(keep="last")].sort_index()

        self._refresh_candles_from_tbbo()
        self._trim_buffers(end, trim_options=load_options)
        self.watermark = end
        if load_options:
            self._opt_watermark = end
            self._last_options_at = time.time()
        logger.info(
            "live_runtime_delta elapsed_s=%.3f tbbo=%d fut=%d opt=%d candles=%d options=%s",
            time.time() - t0, len(self.tbbo), len(self.bbo_fut), len(self.bbo_opt),
            len(self.candles), load_options,
        )

    # ── compute + append ──────────────────────────────────────────

    def _loaded_dict(self, *, include_options: bool = True) -> dict[str, pd.DataFrame]:
        opt = pd.DataFrame()
        if include_options and not self.bbo_opt.empty and "ts_recv" in self.bbo_opt.columns:
            col = pd.to_datetime(self.bbo_opt["ts_recv"], utc=True)
            cutoff = col.max() - _OPTIONS_COMPUTE_LOOKBACK
            opt = self.bbo_opt.loc[col >= cutoff]
        tbbo = self.tbbo
        if not include_options and not tbbo.empty and "ts_event" in tbbo.columns:
            # 45m covers rv_360; empty fut BBO forces mid from TBBO (always fresh).
            cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=45)
            col = pd.to_datetime(tbbo["ts_event"], utc=True)
            tbbo = tbbo.loc[col >= cut]
        return {
            "tbbo": tbbo,
            "bbo_fut": self.bbo_fut if include_options else pd.DataFrame(),
            "bbo_opt": opt,
            "definitions": self.defs,
            "statistics": self.stats,
            "candles_1m": self.candles,
            "mbp_depth": self.mbp_depth,
            "_live_fast": not include_options,
        }

    def _append_new_bars(
        self, d: date, last_closed: pd.Timestamp, *, include_options: bool = False,
    ) -> tuple[int, pd.Timestamp | None, float | None]:
        """Compute full features on warm buffers; append new rows.

        Returns ``(n_appended, parquet_ready_at_utc, compute_s)``.
        """
        if self.features.empty:
            last_ex = last_closed - pd.Timedelta(seconds=self.step_s)
        else:
            last_ex = _tz_utc(self.features.index.max())
            assert last_ex is not None
            if last_ex >= last_closed:
                return 0, None, None

        # Cap catch-up size so one tick stays bounded
        max_span = pd.Timedelta(seconds=self.step_s * _MAX_BARS_PER_TICK)
        if last_closed - last_ex > max_span:
            # advance last_ex so we only materialize the most recent window;
            # older gaps remain until post-market build_date if needed
            last_ex = last_closed - max_span
            logger.warning(
                "live_runtime_catchup_capped to_last=%s bars=%d",
                last_ex.isoformat(), _MAX_BARS_PER_TICK,
            )

        t0 = time.time()
        frame = self.engine._compute_feature_frame(
            d,
            data_start=last_closed - _LOOKBACK,
            data_end=last_closed + pd.Timedelta(seconds=self.step_s),
            quiet=True,
            loaded=self._loaded_dict(include_options=include_options),
        )
        if frame is None or frame.empty:
            logger.warning("live_runtime_compute_empty date=%s", d)
            return 0, None, None

        idx = frame.index
        if idx.tz is None:
            frame = frame.copy()
            frame.index = idx.tz_localize("UTC")
        else:
            frame = frame.copy()
            frame.index = idx.tz_convert("UTC")

        # Only bars after what we already have on disk/memory
        disk_last = None
        if not self.features.empty:
            disk_last = _tz_utc(self.features.index.max())
        cut = disk_last if disk_last is not None else last_ex
        new_rows = frame.loc[(frame.index > cut) & (frame.index <= last_closed)]
        if new_rows.empty:
            return 0, None, None

        # Align columns with existing full schema when possible
        if not self.features.empty:
            combined = merge_feature_rows(self.features, new_rows)
        else:
            combined = new_rows

        # Repair trailing-edge slow features (cs_*/wv_*) from earlier rows in
        # the same day file — handles cagg lag and partial prior appends.
        from src.features.candle_structure import CS_5S_OHLC_COLS
        slow_cols = [
            c for c in combined.columns
            if isinstance(c, str)
            and c.startswith(("cs_", "wv_", "iv_", "gx_", "ca_", "ht_", "hk_"))
            and c not in CS_5S_OHLC_COLS
        ]
        if slow_cols:
            combined[slow_cols] = combined[slow_cols].ffill()

        self.engine._persist_day_features(combined)
        self.features = combined
        n = len(new_rows)
        compute_s = time.time() - t0
        parquet_ready_at = pd.Timestamp.now(tz="UTC")
        logger.info(
            "live_runtime_appended date=%s n=%d total=%d last=%s compute_s=%.2f",
            d, n, len(combined), new_rows.index.max().isoformat(), compute_s,
        )
        return n, parquet_ready_at, compute_s

    def _maybe_score(
        self, *, lags: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Score latest bar into live snapshot cache (cheap if features warm).

        Optional *lags* are merged into the snapshot for the Live UI tracker.
        """
        try:
            from src.webui import live as live_mod
            # Ensure resolve_feature_day does not re-extend while we own the path:
            # heartbeat is written before score with ok=True.
            t_score = time.time()
            snap = live_mod.build_snapshot()
            score_s = time.time() - t_score
            if lags is not None:
                lags = dict(lags)
                lags.setdefault("prediction_score_s", round(score_s, 3))
                # Prefer wall-clock from parquet_ready → now if not already set.
                if lags.get("prediction_lag_s") is None and lags.get("parquet_ready_at"):
                    try:
                        pr = pd.Timestamp(lags["parquet_ready_at"])
                        if pr.tzinfo is None:
                            pr = pr.tz_localize("UTC")
                        lags["prediction_lag_s"] = round(
                            (pd.Timestamp.now(tz="UTC") - pr).total_seconds(), 3
                        )
                    except Exception:  # noqa: BLE001
                        lags["prediction_lag_s"] = round(score_s, 3)
                if lags.get("prediction_lag_s") is None:
                    lags["prediction_lag_s"] = round(score_s, 3)
                lags["predicted_at"] = datetime.now(timezone.utc).isoformat()
                if (
                    lags.get("parquet_lag_s") is not None
                    and lags.get("prediction_lag_s") is not None
                ):
                    lags["total_lag_s"] = round(
                        float(lags["parquet_lag_s"]) + float(lags["prediction_lag_s"]), 3
                    )
                snap = dict(snap)
                snap["lags"] = lags
                live_mod._commit_snapshot(snap)  # noqa: SLF001 — refresh cache w/ lags
            bar = None
            pred = snap.get("prediction") or {}
            bar = pred.get("bar_ts")
            logger.info(
                "live_runtime_scored status=%s bar=%s build_s=%s parquet_lag=%s pred_lag=%s",
                snap.get("status"), bar, snap.get("build_s"),
                (lags or {}).get("parquet_lag_s"),
                (lags or {}).get("prediction_lag_s"),
            )
            return lags
        except Exception as exc:  # noqa: BLE001
            logger.warning("live_runtime_score_failed: %s", exc)
            return lags

    # ── main tick ─────────────────────────────────────────────────

    def tick_once(self) -> dict[str, Any]:
        """One bar-aligned iteration. Returns status dict for heartbeat."""
        t_loop = time.time()
        d = self._session_date_today()
        self._ensure_session(d)
        last_closed = last_closed_bar_ts(timestep=self.engine.timestep)
        # End bound for raw loads: just past the closed bar
        fetch_end = last_closed + pd.Timedelta(seconds=self.step_s)

        # If a post-market research rebuild truncated the day file (08:00–11:00)
        # while this process still holds the full live session in RAM, write it
        # back before scoring from disk.
        self._restore_disk_if_memory_ahead(d)

        # Bootstrap features file if empty
        if self.features.empty:
            self.features = self._load_disk_features(d)
        if self.features.empty:
            logger.info("live_runtime_bootstrap_build_date date=%s", d)
            path = self.engine.build_date(d)
            self.features = self._load_disk_features(d)
            self._bootstrap_buffers(d, fetch_end)
            _write_heartbeat({
                "ok": True,
                "session_date": str(d),
                "last_bar": (
                    _tz_utc(self.features.index.max()).isoformat()
                    if not self.features.empty else None
                ),
                "n_rows": len(self.features),
                "mode": "bootstrap_build_date",
                "path": str(path) if path else None,
                "elapsed_s": round(time.time() - t_loop, 3),
            })
            self._maybe_score()
            return read_heartbeat()

        # Warm buffers. Options BBO is loaded only on the slower IV cadence.
        need_opt = (time.time() - self._last_options_at) >= _OPTIONS_EVERY_S
        if self.watermark is None:
            self._bootstrap_buffers(d, fetch_end)
        else:
            self._fetch_delta(d, fetch_end, load_options=need_opt)

        n_new, parquet_ready_at, compute_s = self._append_new_bars(
            d, last_closed, include_options=need_opt,
        )
        last_bar = _tz_utc(self.features.index.max()) if not self.features.empty else None
        now_utc = pd.Timestamp.now(tz="UTC")
        age_s = (
            (now_utc - last_bar).total_seconds()
            if last_bar is not None else None
        )
        # Left-labeled 5s bar T covers [T, T+5s) and is complete at T+5s —
        # that is when the bar's market data is available for feature build.
        bar_complete_at = (
            last_bar + pd.Timedelta(seconds=self.step_s)
            if last_bar is not None else None
        )
        if parquet_ready_at is None:
            # No new rows this tick — parquet was already ready; lag is how far
            # the latest bar still trails wall clock (observed data→feature lag).
            parquet_ready_at = now_utc
            parquet_lag_s = (
                round((now_utc - bar_complete_at).total_seconds(), 3)
                if bar_complete_at is not None else None
            )
        else:
            parquet_lag_s = (
                round((parquet_ready_at - bar_complete_at).total_seconds(), 3)
                if bar_complete_at is not None else None
            )

        lags: dict[str, Any] = {
            "bar_ts": last_bar.isoformat() if last_bar is not None else None,
            "bar_complete_at": (
                bar_complete_at.isoformat() if bar_complete_at is not None else None
            ),
            "parquet_ready_at": parquet_ready_at.isoformat(),
            "parquet_lag_s": parquet_lag_s,
            "feature_compute_s": round(compute_s, 3) if compute_s is not None else None,
            "n_appended": n_new,
            "options_refresh": need_opt,
            "definitions": {
                "parquet_lag": (
                    "Seconds from bar data availability (bar close = label+5s) "
                    "to feature parquet row write."
                ),
                "prediction_lag": (
                    "Seconds from feature parquet ready to model prediction commit."
                ),
            },
        }

        _write_heartbeat({
            "ok": True,
            "session_date": str(d),
            "last_bar": last_bar.isoformat() if last_bar is not None else None,
            "last_closed": last_closed.isoformat(),
            "n_rows": len(self.features),
            "n_appended": n_new,
            "parquet_age_s": age_s,
            "mode": "warm_per_bar",
            "elapsed_s": round(time.time() - t_loop, 3),
            "lags": lags,
            "buffers": {
                "tbbo": len(self.tbbo),
                "bbo_fut": len(self.bbo_fut),
                "bbo_opt": len(self.bbo_opt),
                "candles": len(self.candles),
            },
        })
        # Always score so WebUI can adopt a fresh cache after restart even when
        # no new bars were appended (e.g. brief idle between bars).
        lags = self._maybe_score(lags=lags) or lags
        # Refresh heartbeat with final prediction lag after score.
        _write_heartbeat({
            "ok": True,
            "session_date": str(d),
            "last_bar": last_bar.isoformat() if last_bar is not None else None,
            "last_closed": last_closed.isoformat(),
            "n_rows": len(self.features),
            "n_appended": n_new,
            "parquet_age_s": age_s,
            "mode": "warm_per_bar",
            "elapsed_s": round(time.time() - t_loop, 3),
            "lags": lags,
            "buffers": {
                "tbbo": len(self.tbbo),
                "bbo_fut": len(self.bbo_fut),
                "bbo_opt": len(self.bbo_opt),
                "candles": len(self.candles),
            },
        })
        self._last_score_bar = last_bar
        return read_heartbeat()

    def run_forever(self) -> None:
        logger.info(
            "live_runtime_start lookback=%s options_lookback=%s step_s=%s heartbeat=%s",
            _LOOKBACK, _OPTIONS_LOOKBACK, self.step_s, _HEARTBEAT_PATH,
        )
        while not self._stop:
            try:
                hb = self.tick_once()
                elapsed = float(hb.get("elapsed_s") or 0)
                # Align to next bar boundary + pad
                now = time.time()
                step = self.step_s
                next_b = (int(now) // step + 1) * step + _BAR_PAD_S
                sleep_s = max(0.05, next_b - time.time())
                # If tick was very slow, don't sleep past another bar unnecessarily
                if elapsed > step:
                    sleep_s = min(sleep_s, 0.5)
                time.sleep(sleep_s)
            except Exception as exc:  # noqa: BLE001
                logger.exception("live_runtime_tick_error: %s", exc)
                _write_heartbeat({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                time.sleep(self.step_s)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Load .env if present (systemd also injects EnvironmentFile)
    try:
        from dotenv import load_dotenv
        load_dotenv("/opt/vtech/.env", override=False)
    except Exception:  # noqa: BLE001
        pass

    rt = LiveFeatureRuntime()
    signal.signal(signal.SIGTERM, rt.request_stop)
    signal.signal(signal.SIGINT, rt.request_stop)
    rt.run_forever()
    logger.info("live_runtime_stop")


if __name__ == "__main__":
    main()
