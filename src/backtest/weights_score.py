"""Path-backtest using staged serving weights (shared ``src.serving`` inference).

Scoring is **not** implemented here — load + predict go through
:mod:`src.serving` so Backtest cannot drift from Live. This module only:

  1. loads historical feature parquets for a date window
  2. calls ``predict_canonical`` (same as Live)
  3. runs the path backtest engine **or** (TBBO DATA) signal-first
     triple-barrier resolution on Databento TBBO trade prints

Provenance note: scores use **current** staged weights on past feature bars —
not as-of-date reconstruction of fold models.
"""

from __future__ import annotations

from src.features.stores import default_parquet_dir

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.backtest.weights_score")

_STORE_TOKENS = ("15s", "5s", "5m", "1m", "1h", "1d")


def infer_bundle_feature_store(bundle: dict[str, Any]) -> str | None:
    """Native-TF hive for a last-fold bundle (``xgb_1m`` → ``1m``), or None."""
    from src.features.stores import require_store

    if bundle.get("tf_oof_meta") or str(bundle.get("family") or "") == "tf_oof_meta":
        meta = bundle.get("meta") or {}
        tf = str(meta.get("decision_timestep") or bundle.get("native_tf") or "15s")
        try:
            return require_store(tf)
        except ValueError:
            return "15s"
    man = bundle.get("manifest") or {}
    meta = bundle.get("meta") or {}
    for raw in (
        bundle.get("native_tf"),
        meta.get("decision_timestep"),
        man.get("native_tf"),
        man.get("decision_timestep"),
        man.get("feature_store"),
        man.get("store"),
    ):
        raw = str(raw or "").strip()
        if not raw:
            continue
        alias = {"1min": "1m", "5min": "5m", "15min": "15s"}.get(raw, raw)
        try:
            return require_store(alias)
        except ValueError:
            continue
    parts = str(bundle.get("candidate_id") or "").replace("-", "_").split("_")
    for tok in _STORE_TOKENS:
        if tok in parts:
            return tok
    cols = [str(c) for c in (bundle.get("feature_cols") or [])]
    if any(c.startswith("svp_") or c.startswith("ps_5m") for c in cols):
        return "5m"
    if any(c.startswith("so_") or c.startswith("ps_1m") for c in cols):
        return "1m"
    if any(c.startswith(("lg_", "st_")) or c == "of_ofi" for c in cols):
        return "15s"
    return None


def infer_source_feature_store(
    source: str | Path,
    bundle: dict[str, Any] | None = None,
) -> str | None:
    """Store for an OOF parquet or last-fold folder (``oof_xgb_1m`` → ``1m``)."""
    if bundle:
        hit = infer_bundle_feature_store(bundle)
        if hit:
            return hit
    p = Path(source)
    if p.is_dir():
        schema_p = p / "feature_schema.json"
        cols: list[str] = []
        if schema_p.is_file():
            try:
                import json
                cols = list((json.loads(schema_p.read_text()) or {}).get("feature_cols") or [])
            except Exception:
                cols = []
        return infer_bundle_feature_store({
            "candidate_id": p.name,
            "feature_cols": cols,
            "manifest": _read_manifest(p),
        })
    if p.suffix == ".parquet":
        parent = p.parent
        if p.name == "oof.parquet":
            return infer_source_feature_store(parent)
        stem = p.stem
        if stem.startswith("oof_"):
            return infer_bundle_feature_store({"candidate_id": stem[4:]})
        if parent.name.startswith("oof_") is False:
            return infer_bundle_feature_store({"candidate_id": parent.name})
    return None


def _read_manifest(model_dir: Path) -> dict[str, Any]:
    import json
    man = model_dir / "finalize_manifest.json"
    if not man.is_file():
        return {}
    try:
        raw = json.loads(man.read_text())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _attach_bundle_side_features(df, bundle: dict[str, Any]):
    """HTF pack / session VP when the schema asked for them (fold-cache only)."""
    cols = [str(c) for c in (bundle.get("feature_cols") or [])]
    if any(c.startswith("svp_") for c in cols):
        try:
            from src.features.session_vp import attach_session_vp
            df = attach_session_vp(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bundle feature load: session VP attach failed: %s", exc)
    if any(c.startswith(("h1_", "d1_")) for c in cols):
        try:
            from src.features.htf_context import attach_htf_to_frame
            df = attach_htf_to_frame(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bundle feature load: HTF attach failed: %s", exc)
    return df


def _with_dual_binary_fire(
    bundle: dict[str, Any],
    objective: dict[str, Any] | None,
    *,
    confidence_gate: bool,
    tau: float | None = None,
) -> dict[str, Any]:
    """Event cut for last-fold predict (ADR-104).

    UI τ is the dual-binary fire when the gate is on. Gate off uses library
    0.1, not book ``labels.fire`` 0.40 (that cut is empty on the 15s LSTM).
    τ≤0 is treated as 0.1 so every bar is not a trade.
    """
    from src.models.dual_binary import DUAL_BINARY_FIRE, dual_binary_fire

    out = dict(bundle)
    if not confidence_gate:
        fire = float(DUAL_BINARY_FIRE)
    elif tau is not None:
        fire = float(tau)
        if fire <= 0:
            fire = float(DUAL_BINARY_FIRE)
    else:
        fire = float(dual_binary_fire(objective))
    out["dual_binary_fire"] = fire
    return out


def resolve_bundle_feature_dir(
    bundle: dict[str, Any],
    objective: dict[str, Any] | None,
    parquet_dir: Path | str | None = None,
) -> tuple[Path, str | None]:
    """Hive directory to score *bundle* on, plus inferred store id."""
    from src.features.stores import parquet_dir_for_store

    if parquet_dir is not None:
        return Path(parquet_dir), infer_bundle_feature_store(bundle)
    store = infer_bundle_feature_store(bundle)
    if store:
        return parquet_dir_for_store(store), store
    return default_parquet_dir(objective), store


def load_bundle_feature_frame(
    bundle: dict[str, Any],
    *,
    start: str,
    end: str,
    store: str | None = None,
    pdir: Path | None = None,
):
    """Load the tape this bundle was trained on (not the 15s decision grid)."""
    from src.features.stores import load_feature_frame
    from src.training.data_pipeline import DataPipeline

    sid = store or infer_bundle_feature_store(bundle)
    if sid:
        df = load_feature_frame(sid, start, end)
    else:
        df = DataPipeline(data_dir=str(pdir or default_parquet_dir(None))).load_features(
            start, end)
    if df is None or getattr(df, "empty", True):
        return df
    df = _attach_bundle_side_features(df, bundle)
    need = list(bundle.get("feature_cols") or [])
    if need:
        present = sum(1 for c in need if c in df.columns)
        logger.info(
            "bundle_features store=%s rows=%d schema=%d present=%d missing=%d",
            sid, len(df), len(need), present, len(need) - present,
        )
    return df


def _hhmm_to_minutes(s: str) -> int:
    parts = str(s or "00:00").strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h * 60 + m


def overlay_session(
    objective: dict[str, Any] | None, policy: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Copy of *objective* with trading window from the bundle inference policy."""
    if not objective or not policy:
        return objective
    obj = dict(objective)
    sess = dict(obj.get("session") or {})
    if policy.get("trading_start"):
        sess["trading_start"] = str(policy["trading_start"])
    if policy.get("trading_end"):
        sess["trading_end"] = str(policy["trading_end"])
    if policy.get("timezone"):
        sess["timezone"] = str(policy["timezone"])
    obj["session"] = sess
    return obj


def apply_inference_policy_to_preds(
    pred_frame: Any,
    *,
    tau: float,
    policy: dict[str, Any] | None,
    confidence_gate: bool = True,
) -> Any:
    """Flatten predicted fires that violate the bundle's open / side / cooldown book.

    Live scoring is unchanged (every bar). Backtest applies this so the sequential
    book matches ADR-047: 08:30–10:00 CT, long-only, 10-minute uniqueness.
    """
    import numpy as np
    import pandas as pd
    from zoneinfo import ZoneInfo

    from src.search.uniqueness import apply_trade_cooldown

    if not policy or pred_frame is None or getattr(pred_frame, "empty", True):
        return pred_frame
    out = pred_frame.copy()
    if "ts" not in out.columns or "direction" not in out.columns:
        return out
    ts = pd.to_datetime(out["ts"], utc=True)
    tz_name = str(policy.get("timezone") or "America/Chicago")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("America/Chicago")
    local = ts.dt.tz_convert(tz)
    minutes = local.dt.hour * 60 + local.dt.minute
    start_m = _hhmm_to_minutes(str(policy.get("trading_start") or "08:30"))
    end_m = _hhmm_to_minutes(str(policy.get("trading_end") or "14:45"))
    in_win = (minutes >= start_m) & (minutes < end_m)

    sides = {str(s) for s in (policy.get("sides") or ("up", "down"))}
    dir_ok = out["direction"].astype(str).isin(sides)
    conf = pd.to_numeric(out.get("confidence"), errors="coerce").fillna(0.0)
    fire = (
        np.asarray(in_win, dtype=bool)
        & np.asarray(dir_ok, dtype=bool)
    )
    if confidence_gate:
        fire = fire & (np.asarray(conf, dtype="float64") >= float(tau))

    cooldown = int(policy.get("cooldown_bars") or 0)
    if cooldown > 0 and fire.any():
        idx = np.flatnonzero(fire)
        order = np.argsort(ts.to_numpy()[idx])
        idx_sorted = idx[order]
        keep = apply_trade_cooldown(ts.to_numpy()[idx_sorted], cooldown)
        admitted = np.zeros(len(out), dtype=bool)
        admitted[idx_sorted[keep]] = True
    else:
        admitted = fire

    deny = ~admitted
    if deny.any():
        out.loc[deny, "direction"] = "flat"
    return out


def _bundle_policy_and_objective(
    bundle: dict[str, Any],
    *,
    objective_id: str | None,
    sides: list[str] | None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, list[str] | None]:
    """Resolve objective + inference policy + default sides from a loaded bundle."""
    obj_id = objective_id or bundle.get("objective_id")
    objective = None
    if obj_id:
        from src.objectives.loader import load_objective
        objective = load_objective(obj_id)
    policy = bundle.get("inference_policy") or (
        (bundle.get("manifest") or {}).get("inference_policy")
    )
    if not isinstance(policy, dict) or not policy:
        policy = None
    else:
        policy = dict(policy)
    if policy:
        objective = overlay_session(objective, policy)
    side_list = list(sides) if sides else None
    if side_list is None and policy and policy.get("sides"):
        side_list = [str(s) for s in policy["sides"] if s in ("up", "down")] or None
    elif policy and side_list:
        policy["sides"] = side_list
    return policy, obj_id, objective, side_list

# TBBO DATA loads ticks per session day (signal windows only), not the full
# multi-month range in one query. Cap matches default path (webui BACKTEST_MAX_SESSIONS).
BACKTEST_TBBO_DATA_MAX_SESSIONS = 400


def _attach_archetypes(
    payload: dict[str, Any],
    bundle_path: str | Path,
    target_pts: float,
    stop_pts: float,
) -> None:
    """Enrich the backtest payload with per-archetype attribution (in place).

    Adds two keys:
      - ``payload["archetypes"]``: aggregated list, sorted by live PnL desc
      - ``payload["trades"][i]["archetype"]`` / ``archetype_label`` /
        ``archetype_train_hit_rate`` / ``archetype_train_n``

    Fails soft (empty archetypes list) if the training OOF frame is missing.
    """
    try:
        from src.backtest.archetypes import (
            attribute_trades,
            training_archetype_stats,
        )
        stats = training_archetype_stats(
            bundle_path,
            tau=0.8,
            target_pts=float(target_pts),
            stop_pts=float(stop_pts),
        )
        tagged, agg = attribute_trades(payload.get("trades") or [], stats)
        payload["trades"] = tagged
        payload["archetypes"] = agg
    except Exception as exc:  # noqa: BLE001
        logger.warning("archetype attribution failed: %s", exc)
        payload.setdefault("archetypes", [])


def run_weights_backtest_inmemory(
    *,
    model_path: str | Path,
    tau: float = 0.8,
    confidence_gate: bool = True,
    max_positions: int = 1,
    start: str | None = None,
    end: str | None = None,
    objective_id: str | None = None,
    sides: list[str] | None = None,
    parquet_dir: str | Path | None = None,
    max_sessions: int = 400,
    live_data: bool = False,
    tbbo_data: bool | None = None,
    tbbo_dev: bool | None = None,  # legacy alias
    oco: bool | None = None,       # alias for tbbo_data (UI: OCO)
    mae_adjust: float = 0.0,
    target_adjust: float = 0.0,
    price_mode: str = "mid",       # "mid" | "mkt"
    entry_latency_ms: float = 0.0,
    exit_latency_ms: float = 0.0,
) -> dict[str, Any]:
    """Score feature parquets with staged weights, then execute fills.

    * ``on_close`` + ``mid`` + 0 latency — classic path: first-touch on 5s
      ``_close`` (same as original console backtest); **actual** PnL, not ±50/6.
    * ``on_close`` + MKT and/or latency — still only act at 5s closes; fill prices
      from TBBO mid/mkt at close+latency; **actual** entry/exit PnL.
    * ``oco`` — continuous TBBO bracket (stop/target vs trade/bid/ask).

    Signals always use the current staged/registry bundle (not as-of-date weights).
    """
    if oco is not None:
        use_oco = bool(oco)
    elif tbbo_data is not None:
        use_oco = bool(tbbo_data)
    elif tbbo_dev is not None:
        use_oco = bool(tbbo_dev)
    else:
        use_oco = bool(live_data)

    px_mode = str(price_mode or "mid").lower().strip()
    e_lat = float(entry_latency_ms or 0)
    x_lat = float(exit_latency_ms or 0)

    # Classic path: on close + mid + no latency → original path engine on _close
    if not use_oco and px_mode == "mid" and e_lat <= 0 and x_lat <= 0:
        return _run_weights_backtest_classic_close(
            model_path=model_path,
            tau=tau,
            confidence_gate=bool(confidence_gate),
            max_positions=max_positions,
            start=start,
            end=end,
            objective_id=objective_id,
            sides=sides,
            parquet_dir=parquet_dir,
            max_sessions=max_sessions,
            mae_adjust=float(mae_adjust),
            target_adjust=float(target_adjust),
        )

    return _run_weights_backtest_signals(
        model_path=model_path,
        tau=tau,
        confidence_gate=bool(confidence_gate),
        max_positions=max_positions,
        start=start,
        end=end,
        objective_id=objective_id,
        sides=sides,
        parquet_dir=parquet_dir,
        max_sessions=min(int(max_sessions), BACKTEST_TBBO_DATA_MAX_SESSIONS),
        mae_adjust=float(mae_adjust),
        target_adjust=float(target_adjust),
        price_mode=px_mode,
        entry_latency_ms=e_lat,
        exit_latency_ms=x_lat,
        execution="oco" if use_oco else "on_close",
    )


def _run_weights_backtest_classic_close(
    *,
    model_path: str | Path,
    tau: float = 0.8,
    confidence_gate: bool = True,
    max_positions: int = 1,
    start: str | None = None,
    end: str | None = None,
    objective_id: str | None = None,
    sides: list[str] | None = None,
    parquet_dir: str | Path | None = None,
    max_sessions: int = 400,
    mae_adjust: float = 0.0,
    target_adjust: float = 0.0,
) -> dict[str, Any]:
    """Original on-close path: first-touch barriers on 5s ``_close`` (actual PnL)."""
    from src.backtest.engine import run_backtest
    from src.backtest.runner import (
        _apply_prediction_exit_defaults,
        _attach_predictions,
        _available_dates,
        _backtest_payload,
    )
    from src.backtest.spec import load_strategy_spec
    from src.serving import load_serving_bundle, predict_canonical

    bundle = load_serving_bundle(model_path)
    policy, obj_id, objective, sides = _bundle_policy_and_objective(
        bundle, objective_id=objective_id, sides=sides)
    bundle = _with_dual_binary_fire(
        bundle, objective, confidence_gate=bool(confidence_gate),
        tau=float(tau))

    pdir, store = resolve_bundle_feature_dir(bundle, objective, parquet_dir)
    avail = _available_dates(pdir)
    if not avail:
        raise FileNotFoundError(f"No feature parquets found in {pdir}")
    lo = str(start or avail[0])
    hi = str(end or avail[-1])
    n_sessions = len([d for d in avail if lo <= d <= hi])
    if n_sessions == 0:
        raise ValueError(f"no feature parquets in {lo} → {hi}")
    if n_sessions > max_sessions:
        raise ValueError(
            f"date range spans {n_sessions} sessions (> {max_sessions}) — narrow the window")

    logger.info(
        "weights_backtest classic on_close model=%s store=%s window=%s→%s sessions=%d",
        bundle["candidate_id"], store, lo, hi, n_sessions)
    df = load_bundle_feature_frame(
        bundle, start=lo, end=hi, store=store, pdir=pdir)
    if df is None or df.empty:
        raise ValueError(f"feature load returned 0 rows for {lo}→{hi}")
    from src.backtest.engine import ensure_price_close
    df = ensure_price_close(df)

    pred_frame = predict_canonical(df, bundle)
    gate_on = bool(confidence_gate)
    if policy:
        pred_frame = apply_inference_policy_to_preds(
            pred_frame, tau=float(tau), policy=policy,
            confidence_gate=gate_on)
    n_dir = int(pred_frame["direction"].isin(["up", "down"]).sum())
    dir_mask = pred_frame["direction"].isin(["up", "down"])
    n_gate = int(dir_mask.sum() if not gate_on else
                 (dir_mask & (pred_frame["confidence"] >= float(tau))).sum())

    pred_block: dict[str, Any] = {
        "source": str(bundle["path"]),
        "tau": float(tau),
        "confidence_gate": gate_on,
    }
    if sides:
        pred_block["sides"] = list(sides)

    sid = f"live_weights_{bundle['candidate_id']}"
    sid = "".join(c if c.isalnum() or c in "._-" else "_" for c in sid)[:80]

    # Optional barrier adjustments via objective override on the strategy exit
    labels = (objective or {}).get("labels") or {}
    base_tgt = float(labels.get("move_threshold", 50))
    base_stop = float(labels.get("mae_tolerance", 6))
    tgt = max(0.25, base_tgt + float(target_adjust))
    stp = max(0.25, base_stop + float(mae_adjust))

    spec = load_strategy_spec({
        "schema_version": "v1",
        "strategy_id": sid,
        **({"objective_id": obj_id} if obj_id else {}),
        "predictions": pred_block,
        "exit": {
            "target_points": tgt,
            "stop_points": stp,
        },
        "backtest": {
            "max_concurrent_positions": max(1, int(max_positions)),
            **({"start": start} if start else {}),
            **({"end": end} if end else {}),
        },
    }, objective_id=obj_id)
    _apply_prediction_exit_defaults(spec, objective)
    # re-apply adjustments if defaults overwrote
    if spec.exit is None:
        spec.exit = {}
    spec.exit["target_points"] = tgt
    spec.exit["stop_points"] = stp

    pred_attach = pred_frame[["ts", "direction", "confidence"]].copy()
    df = _attach_predictions(df, pred_attach)
    result = run_backtest(df, spec, objective=objective)
    payload = _backtest_payload(
        result, spec, df, lo, hi, n_sessions,
        feature_cols=list(bundle["feature_cols"]))

    payload["config"]["prediction_mode"] = "weights"
    payload["config"]["fill_path"] = "on_close"
    payload["config"]["execution"] = "on_close"
    payload["config"]["live_data"] = False
    payload["config"]["tbbo_data"] = False
    payload["config"]["oco"] = False
    payload["config"]["price_mode"] = "mid"
    payload["config"]["entry_latency_ms"] = 0.0
    payload["config"]["exit_latency_ms"] = 0.0
    payload["config"]["target_points"] = tgt
    payload["config"]["stop_points"] = stp
    payload["config"]["target_adjust"] = float(target_adjust)
    payload["config"]["mae_adjust"] = float(mae_adjust)
    payload["config"]["prediction_mode_label"] = (
        "on close · MID · 5s _close first-touch (actual PnL; classic path)"
    )
    payload["config"]["model_path"] = bundle["path"]
    payload["config"]["candidate_id"] = bundle["candidate_id"]
    payload["config"]["model_family"] = bundle["family"]
    payload["config"]["model_version"] = bundle["version"]
    payload["config"]["dir_threshold"] = bundle["dir_thr"]
    payload["config"]["n_features"] = len(bundle["feature_cols"])
    payload["config"]["n_pred_directional"] = n_dir
    payload["config"]["n_pred_gated"] = n_gate
    payload["config"]["confidence_gate"] = gate_on
    payload["config"]["weights_note"] = (
        f"on close · MID · 0 latency — classic first-touch on feature _close "
        f"(actual entry/exit close PnL, not fixed ±target/stop). "
        f"Bundle {bundle['candidate_id']} on {lo}→{hi}. Barriers for touch levels "
        f"+{tgt:g}/−{stp:g}."
    )
    if policy:
        payload["config"]["inference_policy"] = policy
        note = str(policy.get("note") or "").strip()
        if note:
            payload["config"]["weights_note"] = (
                payload["config"]["weights_note"] + " " + note
            )
    _attach_archetypes(payload, bundle["path"], tgt, stp)
    return payload


# ── TBBO DATA: signal-first triple barrier on prints + book ───────────────────

def _horizon_seconds(objective: dict[str, Any] | None) -> float:
    raw = None
    if objective:
        raw = (objective.get("horizon") or {}).get("max_trade_horizon")
    if not raw:
        return 600.0  # 10m default
    s = str(raw).strip().lower()
    if s[-1:].isalpha():
        n, u = float(s[:-1]), s[-1]
        mult = {"s": 1.0, "m": 60.0, "h": 3600.0}.get(u, 60.0)
        return max(1.0, n * mult)
    return max(1.0, float(s))


def _load_tbbo_window(
    *,
    symbol: str,
    t0,
    t1,
) -> tuple[Any, Any, Any, Any]:
    """Return (ts_ns, trade_px, bid_px, ask_px) for TBBO rows in [t0, t1].

    Bid/ask may be NaN when the book is missing on a row.
    """
    import numpy as np
    import psycopg
    from src.common.db import get_dsn

    sql = """
        SELECT ts_event, price, bid_px, ask_px
        FROM databento.tbbo
        WHERE ts_event >= %s
          AND ts_event <= %s
          AND symbol = %s
          AND price IS NOT NULL
          AND price > 0
        ORDER BY ts_event ASC
    """
    with psycopg.connect(get_dsn()) as conn:
        rows = conn.execute(sql, [t0, t1, symbol]).fetchall()
    if not rows:
        z = np.empty(0, dtype="float64")
        return np.empty(0, dtype="int64"), z, z.copy(), z.copy()
    n = len(rows)
    ts_ns = np.empty(n, dtype="int64")
    px = np.empty(n, dtype="float64")
    bid = np.empty(n, dtype="float64")
    ask = np.empty(n, dtype="float64")
    for i, (ts_event, price, bid_px, ask_px) in enumerate(rows):
        if hasattr(ts_event, "timestamp"):
            ts_ns[i] = int(ts_event.timestamp() * 1e9)
        else:
            ts_ns[i] = int(pd_timestamp_ns(ts_event))
        px[i] = float(price)
        bid[i] = float(bid_px) if bid_px is not None else np.nan
        ask[i] = float(ask_px) if ask_px is not None else np.nan
    return ts_ns, px, bid, ask


def pd_timestamp_ns(ts_event: Any) -> int:
    import pandas as pd
    return int(pd.Timestamp(ts_event).value)


def _finite(x: float) -> bool:
    import math
    return x is not None and math.isfinite(float(x))


def _mid_px(bid: float, ask: float, trade: float) -> float:
    import math
    b, a, t = float(bid), float(ask), float(trade)
    if math.isfinite(b) and math.isfinite(a):
        return 0.5 * (b + a)
    if math.isfinite(t):
        return t
    if math.isfinite(b):
        return b
    if math.isfinite(a):
        return a
    return float("nan")


def _quote_px(
    *,
    is_buy: bool,
    price_mode: str,
    bid: float,
    ask: float,
    trade: float,
) -> float:
    """Fill price for a buy or sell order under mid / mkt mode."""
    import math
    mode = (price_mode or "mid").lower()
    if mode == "mkt":
        # buy pays ask; sell hits bid
        px = float(ask) if is_buy else float(bid)
        if math.isfinite(px):
            return px
        t = float(trade)
        return t if math.isfinite(t) else float("nan")
    return _mid_px(bid, ask, trade)


def _idx_at_or_after(tbbo_ts: Any, t_ns: int) -> int:
    import numpy as np
    n = len(tbbo_ts)
    if n == 0:
        return -1
    i = int(np.searchsorted(tbbo_ts, t_ns, side="left"))
    if i >= n:
        return n - 1
    return i


def _lookup_fill(
    *,
    t_ns: int,
    is_buy: bool,
    price_mode: str,
    tbbo_ts: Any,
    tbbo_px: Any,
    tbbo_bid: Any,
    tbbo_ask: Any,
) -> tuple[int, float]:
    """TBBO price at or after *t_ns* under mid/mkt."""
    import math
    i = _idx_at_or_after(tbbo_ts, t_ns)
    if i < 0:
        return int(t_ns), float("nan")
    px = _quote_px(
        is_buy=is_buy,
        price_mode=price_mode,
        bid=float(tbbo_bid[i]),
        ask=float(tbbo_ask[i]),
        trade=float(tbbo_px[i]),
    )
    return int(tbbo_ts[i]), float(px) if math.isfinite(px) else float("nan")


def _resolve_signal_oco(
    *,
    is_long: bool,
    signal_ts_ns: int,
    horizon_ns: int,
    target_pts: float,
    stop_pts: float,
    tbbo_ts: Any,
    tbbo_px: Any,
    tbbo_bid: Any,
    tbbo_ask: Any,
    price_mode: str = "mid",
    entry_latency_ns: int = 0,
    exit_latency_ns: int = 0,
) -> dict[str, Any]:
    """OCO bracket: continuous TBBO first-touch after entry (+ latencies)."""
    import math
    import numpy as np

    n = len(tbbo_ts)
    if n == 0:
        return {
            "outcome": "timeout", "pnl_points": 0.0,
            "entry_ts_ns": signal_ts_ns, "exit_ts_ns": signal_ts_ns,
            "entry_px": float("nan"), "exit_px": float("nan"), "holding_ns": 0,
        }

    entry_order_ns = int(signal_ts_ns) + max(0, int(entry_latency_ns))
    entry_ts_ns, entry_px = _lookup_fill(
        t_ns=entry_order_ns, is_buy=is_long, price_mode=price_mode,
        tbbo_ts=tbbo_ts, tbbo_px=tbbo_px, tbbo_bid=tbbo_bid, tbbo_ask=tbbo_ask)
    if not math.isfinite(entry_px):
        return {
            "outcome": "timeout", "pnl_points": 0.0,
            "entry_ts_ns": entry_ts_ns, "exit_ts_ns": entry_ts_ns,
            "entry_px": entry_px, "exit_px": entry_px, "holding_ns": 0,
        }

    t_end = entry_order_ns + horizon_ns
    lo = _idx_at_or_after(tbbo_ts, entry_order_ns)
    hi = int(np.searchsorted(tbbo_ts, t_end, side="right"))
    if lo >= hi:
        hi = min(n, lo + 1)

    mode = (price_mode or "mid").lower()
    if is_long:
        stop_lvl, tgt_lvl = entry_px - stop_pts, entry_px + target_pts
    else:
        stop_lvl, tgt_lvl = entry_px + stop_pts, entry_px - target_pts

    hit_j, hit_stop = None, False
    for j in range(lo, hi):
        bid, ask, tr = float(tbbo_bid[j]), float(tbbo_ask[j]), float(tbbo_px[j])
        if mode == "mid":
            mon = _mid_px(bid, ask, tr)
        else:
            # OCO: trade print or exit-side quote can trigger
            mon_trade = tr if math.isfinite(tr) else float("nan")
            mon_book = _quote_px(
                is_buy=not is_long, price_mode="mkt", bid=bid, ask=ask, trade=tr)
            # check both
            for mon in (mon_trade, mon_book):
                if not math.isfinite(mon):
                    continue
                if is_long:
                    s_hit, t_hit = mon <= stop_lvl, mon >= tgt_lvl
                else:
                    s_hit, t_hit = mon >= stop_lvl, mon <= tgt_lvl
                if s_hit or t_hit:
                    hit_j, hit_stop = j, bool(s_hit)
                    break
            if hit_j is not None:
                break
            continue
        if not math.isfinite(mon):
            continue
        if is_long:
            s_hit, t_hit = mon <= stop_lvl, mon >= tgt_lvl
        else:
            s_hit, t_hit = mon >= stop_lvl, mon <= tgt_lvl
        if s_hit or t_hit:
            hit_j, hit_stop = j, bool(s_hit)
            break

    def _exit_at(j: int) -> tuple[int, float]:
        order_ns = int(tbbo_ts[j]) + max(0, int(exit_latency_ns))
        return _lookup_fill(
            t_ns=order_ns, is_buy=not is_long, price_mode=price_mode,
            tbbo_ts=tbbo_ts, tbbo_px=tbbo_px, tbbo_bid=tbbo_bid, tbbo_ask=tbbo_ask)

    if hit_j is not None:
        exit_ts_ns, exit_px = _exit_at(hit_j)
        if hit_stop:
            return {
                "outcome": "stop", "pnl_points": round(-float(stop_pts), 2),
                "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
                "entry_px": float(entry_px),
                "exit_px": float(exit_px) if math.isfinite(exit_px) else float(stop_lvl),
                "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
            }
        return {
            "outcome": "target", "pnl_points": round(float(target_pts), 2),
            "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
            "entry_px": float(entry_px),
            "exit_px": float(exit_px) if math.isfinite(exit_px) else float(tgt_lvl),
            "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
        }

    j_last = max(lo, hi - 1)
    exit_ts_ns, exit_px = _exit_at(j_last)
    if not math.isfinite(exit_px):
        exit_px = entry_px
    sign = 1.0 if is_long else -1.0
    return {
        "outcome": "timeout",
        "pnl_points": round((float(exit_px) - float(entry_px)) * sign, 2),
        "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
        "entry_px": float(entry_px), "exit_px": float(exit_px),
        "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
    }


def _resolve_signal_on_close(
    *,
    is_long: bool,
    signal_ts_ns: int,
    signal_close_px: float,
    bar_ts_ns: Any,
    bar_close_px: Any,
    horizon_ns: int,
    target_pts: float,
    stop_pts: float,
    tbbo_ts: Any,
    tbbo_px: Any,
    tbbo_bid: Any,
    tbbo_ask: Any,
    price_mode: str = "mid",
    entry_latency_ns: int = 0,
    exit_latency_ns: int = 0,
) -> dict[str, Any]:
    """On-close: act only at 5s candle closes.

    * MID + zero latency → use the candle ``_close`` literally (no TBBO).
    * MID + latency > 0  → TBBO mid at (close_ts + latency).
    * MKT (any latency)  → TBBO bid/ask at (close_ts + latency); buy=ask, sell=bid.

    The candle close decides whether a barrier fired. The selected price mode
    then determines the submitted exit price; MKT takes the executable bid/ask,
    never the theoretical target/stop level.
    """
    import math
    import numpy as np

    mode = (price_mode or "mid").lower()
    e_lat = max(0, int(entry_latency_ns))
    x_lat = max(0, int(exit_latency_ns))
    has_tbbo = len(tbbo_ts) > 0

    def _px_at_close(close_ts_ns: int, candle_close: float, *, is_buy: bool,
                     latency_ns: int) -> tuple[int, float]:
        """Price for an action tied to a 5s close (+ optional latency)."""
        if mode == "mid" and latency_ns <= 0:
            # literal candle close — no TBBO
            return int(close_ts_ns), float(candle_close)
        if not has_tbbo:
            # no TBBO available: degrade to candle close
            return int(close_ts_ns) + latency_ns, float(candle_close)
        t_ns = int(close_ts_ns) + latency_ns
        ts_out, px = _lookup_fill(
            t_ns=t_ns, is_buy=is_buy, price_mode=mode,
            tbbo_ts=tbbo_ts, tbbo_px=tbbo_px, tbbo_bid=tbbo_bid, tbbo_ask=tbbo_ask)
        if not math.isfinite(px):
            return t_ns, float(candle_close)
        return ts_out, float(px)

    entry_ts_ns, entry_px = _px_at_close(
        int(signal_ts_ns), float(signal_close_px), is_buy=is_long, latency_ns=e_lat)
    if not math.isfinite(entry_px):
        return {
            "outcome": "timeout", "pnl_points": 0.0,
            "entry_ts_ns": entry_ts_ns, "exit_ts_ns": entry_ts_ns,
            "entry_px": entry_px, "exit_px": entry_px, "holding_ns": 0,
        }

    if is_long:
        stop_lvl, tgt_lvl = entry_px - stop_pts, entry_px + target_pts
    else:
        stop_lvl, tgt_lvl = entry_px + stop_pts, entry_px - target_pts

    t_end = int(signal_ts_ns) + horizon_ns
    # later bars: (ts, close) pairs
    later: list[tuple[int, float]] = []
    for t, c in zip(bar_ts_ns, bar_close_px):
        ti = int(t)
        if ti > int(signal_ts_ns) and ti <= t_end and math.isfinite(float(c)):
            later.append((ti, float(c)))

    if not later:
        return {
            "outcome": "timeout", "pnl_points": 0.0,
            "entry_ts_ns": entry_ts_ns, "exit_ts_ns": entry_ts_ns,
            "entry_px": float(entry_px), "exit_px": float(entry_px), "holding_ns": 0,
        }

    for bar_ts, bar_c in later:
        exit_ts_ns, mon = _px_at_close(
            bar_ts, bar_c, is_buy=not is_long, latency_ns=x_lat)
        if not math.isfinite(mon):
            continue
        if is_long:
            s_hit, t_hit = bar_c <= stop_lvl, bar_c >= tgt_lvl
        else:
            s_hit, t_hit = bar_c >= stop_lvl, bar_c <= tgt_lvl
        sign = 1.0 if is_long else -1.0
        # The 5s close triggers the exit; MKT then fills at the executable quote.
        act = round((float(mon) - float(entry_px)) * sign, 2)
        if s_hit:
            return {
                "outcome": "stop", "pnl_points": act,
                "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
                "entry_px": float(entry_px), "exit_px": float(mon),
                "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
            }
        if t_hit:
            return {
                "outcome": "target", "pnl_points": act,
                "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
                "entry_px": float(entry_px), "exit_px": float(mon),
                "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
            }

    bar_ts, bar_c = later[-1]
    exit_ts_ns, exit_px = _px_at_close(
        bar_ts, bar_c, is_buy=not is_long, latency_ns=x_lat)
    if not math.isfinite(exit_px):
        exit_px = entry_px
    sign = 1.0 if is_long else -1.0
    return {
        "outcome": "timeout",
        "pnl_points": round((float(exit_px) - float(entry_px)) * sign, 2),
        "entry_ts_ns": entry_ts_ns, "exit_ts_ns": exit_ts_ns,
        "entry_px": float(entry_px), "exit_px": float(exit_px),
        "holding_ns": max(0, exit_ts_ns - entry_ts_ns),
    }


# back-compat name
_resolve_signal_tbbo = _resolve_signal_oco


def _run_weights_backtest_signals(
    *,
    model_path: str | Path,
    tau: float = 0.8,
    confidence_gate: bool = True,
    max_positions: int = 1,
    start: str | None = None,
    end: str | None = None,
    objective_id: str | None = None,
    sides: list[str] | None = None,
    parquet_dir: str | Path | None = None,
    max_sessions: int = BACKTEST_TBBO_DATA_MAX_SESSIONS,
    mae_adjust: float = 0.0,
    target_adjust: float = 0.0,
    price_mode: str = "mid",
    entry_latency_ms: float = 0.0,
    exit_latency_ms: float = 0.0,
    execution: str = "on_close",
) -> dict[str, Any]:
    """Signal-gated backtest: *execution* ``oco`` or ``on_close``.

    1. Score feature bars with staged weights.
    2. Keep long/short rows with conf ≥ τ.
    3. Enforce *max_positions*.
    4. Resolve fills via TBBO mid/mkt + latencies under the chosen execution mode.
    """
    from collections import Counter

    import numpy as np
    import pandas as pd

    from src.backtest.runner import _available_dates
    from src.common.contract_calendar import active_contract
    from src.common.instruments import point_value_from_objective, primary_symbol
    from src.serving import load_serving_bundle, predict_canonical

    bundle = load_serving_bundle(model_path)
    policy, obj_id, objective, sides = _bundle_policy_and_objective(
        bundle, objective_id=objective_id, sides=sides)
    bundle = _with_dual_binary_fire(
        bundle, objective, confidence_gate=bool(confidence_gate),
        tau=float(tau))

    pdir, store = resolve_bundle_feature_dir(bundle, objective, parquet_dir)
    avail = _available_dates(pdir)
    if not avail:
        raise FileNotFoundError(f"No feature parquets found in {pdir}")
    lo = str(start or avail[0])
    hi = str(end or avail[-1])
    days = [d for d in avail if lo <= d <= hi]
    n_sessions = len(days)
    if n_sessions == 0:
        raise ValueError(f"no feature parquets in {lo} → {hi}")
    if n_sessions > max_sessions:
        raise ValueError(
            f"backtest spans {n_sessions} sessions (>{max_sessions}) — narrow the window")

    labels = (objective or {}).get("labels") or {}
    base_tgt = float(labels.get("move_threshold", 50))
    base_stop = float(labels.get("mae_tolerance", 6))
    target_pts = max(0.25, base_tgt + float(target_adjust))
    stop_pts = max(0.25, base_stop + float(mae_adjust))
    horizon_s = _horizon_seconds(objective)
    horizon_ns = int(horizon_s * 1e9)
    max_hold_label = (
        (objective or {}).get("horizon") or {}
    ).get("max_trade_horizon") or f"{int(horizon_s)}s"
    open_slots = max(1, min(50, int(max_positions)))
    px_mode = (price_mode or "mid").lower().strip()
    if px_mode not in ("mid", "mkt"):
        px_mode = "mid"
    exec_mode = (execution or "on_close").lower().strip()
    if exec_mode not in ("oco", "on_close"):
        exec_mode = "on_close"
    entry_lat_ns = int(max(0.0, float(entry_latency_ms)) * 1_000_000)
    exit_lat_ns = int(max(0.0, float(exit_latency_ms)) * 1_000_000)
    pad_s = max(0.0, float(entry_latency_ms), float(exit_latency_ms)) / 1000.0 + 1.0

    pv = point_value_from_objective(objective, default=20.0) if objective else 20.0
    instrument = primary_symbol(objective) if objective else "NQ"
    thr = float(tau)
    gate_on = bool(confidence_gate)
    side_set = set(sides) if sides else None

    n_dir = 0
    n_gate = 0
    closed_trades: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    days_with_signals = 0
    days_failed: list[str] = []
    open_until_ns: list[int] = []

    logger.info(
        "weights_backtest exec=%s model=%s window=%s→%s sessions=%d "
        "target=%.2f stop=%.2f horizon=%.0fs max_pos=%d "
        "price_mode=%s entry_lat_ms=%.0f exit_lat_ms=%.0f",
        exec_mode, bundle["candidate_id"], lo, hi, n_sessions,
        target_pts, stop_pts, horizon_s, open_slots, px_mode,
        float(entry_latency_ms), float(exit_latency_ms),
    )

    for day in days:
        fpath = pdir / f"features_{day}.parquet"
        try:
            if store:
                df = load_bundle_feature_frame(
                    bundle, start=day, end=day, store=store, pdir=pdir)
            elif fpath.exists():
                df = pd.read_parquet(fpath)
            else:
                from src.features.hive_5s import load_session_day
                df = load_session_day(day)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "bar_end_ts" in df.columns:
                    df = df.set_index("bar_end_ts")
                elif "ts" in df.columns:
                    df = df.set_index("ts")
                df.index = pd.to_datetime(df.index, utc=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.sort_index()
            if df.empty:
                skipped["no_close"] += 1
                continue
            from src.backtest.engine import ensure_price_close
            try:
                df = ensure_price_close(df)
            except ValueError:
                skipped["no_close"] += 1
                continue
            preds = predict_canonical(df, bundle)
            if policy:
                preds = apply_inference_policy_to_preds(
                    preds, tau=thr, policy=policy,
                    confidence_gate=gate_on)
            n_total_feat = len(bundle["feature_cols"])
            _feat_present = (
                n_total_feat
                - df.reindex(columns=bundle["feature_cols"]).isna().sum(axis=1)
            ).astype("int64")
        except Exception as exc:  # noqa: BLE001
            logger.warning("tbbo_data_score_failed day=%s: %s", day, exc)
            days_failed.append(day)
            continue

        n_dir += int(preds["direction"].isin(["up", "down"]).sum())
        gate = preds["direction"].isin(["up", "down"])
        if gate_on:
            gate = gate & (preds["confidence"] >= thr)
        if side_set:
            gate = gate & preds["direction"].isin(side_set)
        sig = preds.loc[gate].copy()
        n_gate += len(sig)
        if sig.empty:
            skipped["no_signals"] += 1
            continue

        # align entry price from feature _close
        close = df["_close"].astype("float64")
        # preds ts may equal index
        entry_pxs: list[float] = []
        entry_ts_list: list[pd.Timestamp] = []
        dirs: list[str] = []
        confs: list[float] = []
        n_present_list: list[int] = []
        for _, row in sig.iterrows():
            ts = pd.Timestamp(row["ts"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            # nearest prior/equal close
            if ts in close.index:
                px = float(close.loc[ts])
            else:
                loc = close.index.searchsorted(ts, side="right") - 1
                if loc < 0:
                    skipped["no_entry_px"] += 1
                    continue
                px = float(close.iloc[loc])
                ts = close.index[loc]
            if not np.isfinite(px):
                skipped["bad_entry_px"] += 1
                continue
            entry_pxs.append(px)
            entry_ts_list.append(ts)
            dirs.append(str(row["direction"]))
            confs.append(float(row["confidence"]))
            try:
                n_present_list.append(int(_feat_present.loc[ts]))
            except KeyError:
                n_present_list.append(int(n_total_feat))

        if not entry_ts_list:
            continue

        days_with_signals += 1
        bar_ts_ns = df.index.asi8
        close_arr = close.to_numpy(dtype="float64", copy=True)
        # TBBO only when needed: OCO always; on_close when MKT or any latency > 0
        need_tbbo = (
            exec_mode == "oco"
            or px_mode == "mkt"
            or entry_lat_ns > 0
            or exit_lat_ns > 0
        )
        if need_tbbo:
            t_min = min(entry_ts_list) - pd.Timedelta(seconds=pad_s)
            t_max = max(entry_ts_list) + pd.Timedelta(seconds=horizon_s + pad_s)
            try:
                d = __import__("datetime").date.fromisoformat(day)
                symbol = active_contract("NQ", d)
                tbbo_ts, tbbo_px, tbbo_bid, tbbo_ask = _load_tbbo_window(
                    symbol=symbol, t0=t_min.to_pydatetime(), t1=t_max.to_pydatetime())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "tbbo_load_failed day=%s: %s — mid/mkt with latency may fall back to close",
                    day, exc)
                skipped["tbbo_load_error"] += 1
                tbbo_ts = np.empty(0, dtype="int64")
                tbbo_px = np.empty(0, dtype="float64")
                tbbo_bid = np.empty(0, dtype="float64")
                tbbo_ask = np.empty(0, dtype="float64")
            if len(tbbo_ts) == 0:
                skipped["no_tbbo"] += 1
        else:
            tbbo_ts = np.empty(0, dtype="int64")
            tbbo_px = np.empty(0, dtype="float64")
            tbbo_bid = np.empty(0, dtype="float64")
            tbbo_ask = np.empty(0, dtype="float64")

        def _try_admit(entry_ts_ns: int) -> bool:
            """Drop finished positions; return True if a slot is free for *entry_ts_ns*."""
            nonlocal open_until_ns
            open_until_ns = [ex for ex in open_until_ns if ex > entry_ts_ns]
            if len(open_until_ns) >= open_slots:
                skipped["in_position"] += 1
                return False
            return True

        for ts, px, direction, conf, n_present in zip(
                entry_ts_list, entry_pxs, dirs, confs, n_present_list):
            is_long = direction == "up"
            signal_ts_ns = int(ts.value)
            admit_ns = signal_ts_ns + entry_lat_ns
            if not _try_admit(admit_ns):
                continue
            if exec_mode == "oco":
                res = _resolve_signal_oco(
                    is_long=is_long,
                    signal_ts_ns=signal_ts_ns,
                    horizon_ns=horizon_ns,
                    target_pts=target_pts,
                    stop_pts=stop_pts,
                    tbbo_ts=tbbo_ts,
                    tbbo_px=tbbo_px,
                    tbbo_bid=tbbo_bid,
                    tbbo_ask=tbbo_ask,
                    price_mode=px_mode,
                    entry_latency_ns=entry_lat_ns,
                    exit_latency_ns=exit_lat_ns,
                )
            else:
                res = _resolve_signal_on_close(
                    is_long=is_long,
                    signal_ts_ns=signal_ts_ns,
                    signal_close_px=float(px),
                    bar_ts_ns=bar_ts_ns,
                    bar_close_px=close_arr,
                    horizon_ns=horizon_ns,
                    target_pts=target_pts,
                    stop_pts=stop_pts,
                    tbbo_ts=tbbo_ts,
                    tbbo_px=tbbo_px,
                    tbbo_bid=tbbo_bid,
                    tbbo_ask=tbbo_ask,
                    price_mode=px_mode,
                    entry_latency_ns=entry_lat_ns,
                    exit_latency_ns=exit_lat_ns,
                )
            exit_ts_ns = int(res["exit_ts_ns"])
            fill_entry_ns = int(res.get("entry_ts_ns") or admit_ns)
            open_until_ns.append(max(exit_ts_ns, fill_entry_ns + 1))
            entry_ts_out = pd.Timestamp(fill_entry_ns, unit="ns", tz="UTC")
            exit_ts = pd.Timestamp(exit_ts_ns, unit="ns", tz="UTC")
            hold_s = float(res["holding_ns"]) / 1e9
            ep = float(res["entry_px"])
            xp = float(res["exit_px"])
            closed_trades.append({
                "entry_ts": entry_ts_out.isoformat(),
                "exit_ts": exit_ts.isoformat(),
                "side": "long" if is_long else "short",
                "entry_px": round(ep, 2) if ep == ep else round(float(px), 2),
                "exit_px": round(xp, 2) if xp == xp else round(float(px), 2),
                "pnl_points": float(res["pnl_points"]),
                "holding_seconds": round(hold_s, 1),
                "exit_reason": res["outcome"],
                "confidence": conf,
                "session_date": day,
                "mae_points": 0.0,
                "mfe_points": 0.0,
                "holding_bars": max(1, int(round(hold_s / 5.0))),
                "n_present": int(n_present),
                "n_total": int(n_total_feat),
            })

    closed_trades.sort(key=lambda t: str(t["entry_ts"]))

    is_oco = exec_mode == "oco"
    config: dict[str, Any] = {
        "strategy_id": f"{exec_mode}_{bundle['candidate_id']}"[:80],
        "objective_id": obj_id,
        "tau": thr,
        "confidence_gate": gate_on,
        "sides": sorted(side_set) if side_set else ["up", "down"],
        "max_concurrent_positions": open_slots,
        "target_points": target_pts,
        "stop_points": stop_pts,
        "target_base": base_tgt,
        "stop_base": base_stop,
        "target_adjust": float(target_adjust),
        "mae_adjust": float(mae_adjust),
        "max_holding": max_hold_label,
        "point_value": float(pv),
        "instrument": instrument or "NQ",
        "window": {
            "start": lo, "end": hi, "n_sessions": n_sessions,
            "n_bars": None, "days_with_signals": days_with_signals,
        },
        "prediction_mode": "weights",
        "fill_path": "oco" if is_oco else "on_close",
        "execution": exec_mode,
        "live_data": is_oco,
        "tbbo_data": is_oco,
        "oco": is_oco,
        "price_mode": px_mode,
        "entry_latency_ms": float(entry_latency_ms or 0),
        "exit_latency_ms": float(exit_latency_ms or 0),
        "prediction_mode_label": (
            f"{'OCO' if is_oco else 'on close'} · price={px_mode.upper()} "
            f"· entry_lat={float(entry_latency_ms or 0):.0f}ms "
            f"· exit_lat={float(exit_latency_ms or 0):.0f}ms"
        ),
        "model_path": bundle["path"],
        "candidate_id": bundle["candidate_id"],
        "model_family": bundle["family"],
        "model_version": bundle["version"],
        "dir_threshold": bundle["dir_thr"],
        "n_features": len(bundle["feature_cols"]),
        "n_pred_directional": n_dir,
        "n_pred_gated": n_gate,
        "weights_note": (
            f"{'OCO' if is_oco else 'on close'} · bundle {bundle['candidate_id']} "
            f"on {lo}→{hi}. "
            f"{'Signals ungated (confidence gate off)' if not gate_on else f'Signals conf≥{thr:.2f}'} "
            f"· max open {open_slots} · "
            f"price {px_mode.upper()} · lat entry {float(entry_latency_ms or 0):.0f}ms / "
            f"exit {float(exit_latency_ms or 0):.0f}ms. "
            f"Barriers +{target_pts:g}/−{stop_pts:g} · horizon {max_hold_label}. "
            + ("Continuous TBBO bracket (fixed ±target/stop on hit). " if is_oco else
               "On close: MID+0ms=_close; latency/MKT=TBBO; actual PnL. ")
            + f"{days_with_signals}/{n_sessions} sessions with signals"
            + (f"; skipped in_position={skipped.get('in_position', 0)}"
               if skipped.get("in_position") else "")
            + (f"; issues: {', '.join(days_failed[:6])}"
               + ("…" if len(days_failed) > 6 else "") if days_failed else "")
            + "."
        ),
    }
    if policy:
        config["inference_policy"] = policy
        note = str(policy.get("note") or "").strip()
        if note:
            config["weights_note"] = str(config["weights_note"]) + " " + note

    labels_cm = ["down", "flat", "up"]
    if not closed_trades:
        empty_keys = ("gross_profit", "gross_loss", "net_pnl", "win_rate", "expectancy",
                      "profit_factor", "max_drawdown", "sharpe_daily", "avg_win", "avg_loss")
        return {
            "config": config,
            "summary": {
                "n_trades": 0, "n_wins": 0, "n_losses": 0, "point_value": float(pv),
                "net_value": 0.0, "by_exit_reason": {},
                "n_sessions_traded": 0,
                **{k: None for k in empty_keys},
            },
            "equity_curve": [],
            "confusion_matrix": {
                "labels": labels_cm,
                "rows_true_cols_pred": [[0] * 3 for _ in range(3)],
            },
            "trades": [],
            "skipped": dict(skipped),
        }

    pnls = np.array([t["pnl_points"] for t in closed_trades], dtype="float64")
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    cum = np.cumsum(pnls)
    equity = cum
    hwm = np.maximum.accumulate(equity)
    hwm = np.maximum(hwm, 0.0)
    dd = equity - hwm

    by_day: dict[str, float] = {}
    for t, p in zip(closed_trades, pnls):
        by_day[t["session_date"]] = by_day.get(t["session_date"], 0.0) + float(p)
    daily = np.array(list(by_day.values()), dtype="float64") if by_day else np.array([])

    def _sharpe(x: np.ndarray) -> float | None:
        if len(x) < 2:
            return None
        s = float(x.std(ddof=1))
        if s == 0.0:
            return None
        return float(x.mean() / s)

    by_exit: Counter[str] = Counter(t["exit_reason"] for t in closed_trades)
    n = len(closed_trades)
    summary = {
        "n_trades": n,
        "n_wins": int((pnls > 0).sum()),
        "n_losses": int((pnls <= 0).sum()),
        "gross_profit": float(wins.sum()) if len(wins) else 0.0,
        "gross_loss": float(losses.sum()) if len(losses) else 0.0,
        "net_pnl": float(pnls.sum()),
        "net_value": float(pnls.sum() * pv),
        "win_rate": float((pnls > 0).mean()),
        "expectancy": float(pnls.mean()),
        "profit_factor": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and float(losses.sum()) != 0.0 else None
        ),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "sharpe_daily": _sharpe(daily),
        "sharpe_trade": _sharpe(pnls),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "avg_holding_bars": float(np.mean([t["holding_bars"] for t in closed_trades])),
        "avg_mae": None,
        "avg_mfe": None,
        "exposure_frac": None,
        "n_sessions_traded": len(by_day),
        "by_exit_reason": dict(by_exit),
        "point_value": float(pv),
    }

    equity_curve = [{"t": str(closed_trades[0]["entry_ts"]), "equity": 0.0}]
    equity_curve += [
        {"t": str(closed_trades[j]["exit_ts"]), "equity": float(cum[j])}
        for j in range(n)
    ]

    confusion = {
        "labels": labels_cm,
        "rows_true_cols_pred": [[0] * 3 for _ in range(3)],
    }

    trades_out = [{
        "entry_ts": str(t["entry_ts"]),
        "exit_ts": str(t["exit_ts"]),
        "side": t["side"],
        "entry_px": t["entry_px"],
        "exit_px": t["exit_px"],
        "pnl_points": t["pnl_points"],
        "cum_pnl": float(cum[j]),
        "mae_points": t["mae_points"],
        "mfe_points": t["mfe_points"],
        "holding_bars": t["holding_bars"],
        "exit_reason": t["exit_reason"],
        "n_present": t.get("n_present"),
        "n_total": t.get("n_total"),
        "confidence": t.get("confidence"),
    } for j, t in enumerate(closed_trades)]

    payload = {
        "config": config,
        "summary": summary,
        "equity_curve": equity_curve,
        "confusion_matrix": confusion,
        "trades": trades_out,
        "skipped": dict(skipped),
    }
    _attach_archetypes(payload, bundle["path"], float(target_pts), float(stop_pts))
    return payload


# legacy name
_run_weights_backtest_tbbo_data = _run_weights_backtest_signals

