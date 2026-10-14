"""Curated per-family feature views — Phase 23.

The parquet feature matrix is the **superset** of engineered features (~21 prefix groups,
see docs/reference/FEATURE_CATALOG.md). Feeding all of it to every family is wrong for two
reasons established in Phase 23 testing:

  1. **Statistical fit** — distance/density/covariance models (KMeans, DBSCAN, GMM, HMM)
     degrade or fail outright in ~175 float32 dimensions (distance concentration; GMM
     ill-conditioned covariance was observed crashing on the full matrix). Regime models
     want a *small, curated descriptor set*: volatility, trend strength, range compression,
     volume behaviour.
  2. **Methodology** — the case for every feature group must be made per trade objective
     (Stage 0 data-requirements analysis), not inherited by default from whatever previous
     objectives happened to compute.

A *view* is a named list of feature-prefix groups. Resolution order in the controller:

  ``objective["search"]["feature_views"][family]``  (per-objective override, from Stage 0)
      → ``CatalogEntry.feature_view``               (catalog default for the family)
      → full matrix                                  (no view)

Views select by **prefix**, so they survive column additions within a group and tolerate
parquet-vintage drift (missing groups simply select fewer columns).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.models.feature_views")

# ── Named views (prefix groups; see FEATURE_CATALOG.md for the group meanings) ──────────
FEATURE_VIEWS: dict[str, list[str]] = {
    # Regime descriptors: realized volatility (rv_), daily trend/ATR-regime (ht_),
    # daily context levels (dc_), VWAP relation (vw_), candle structure / range
    # compression (cs_). ~40 dims — the scale regime/clustering models are built for.
    "regime_core": ["rv_", "ht_", "dc_", "vw_", "cs_"],

    # Minimalist regime baseline — deliberately ISOLATED exact columns (NOT whole
    # prefix groups, which reintroduce collinear duplicates), one representative per
    # axis. The achievable stand-in for a "returns-only" descriptor (parquet has no NQ
    # return column; _close is label-only):
    #   vw_vwap_dist       — price location / return sign  (first moment)
    #   vw_vwap_slope_120  — 10-min trend direction
    #   rv_realized_vol_60 — realized-vol level            (second moment)
    #   rv_vol_ratio       — vol compression/expansion regime
    # A Gaussian HMM/GMM further recovers the vol axis via each state's emission
    # variance. NB: view matching is prefix-based, but each full name resolves to
    # exactly itself (no unintended siblings share these full prefixes).
    "regime_min": ["vw_vwap_dist", "vw_vwap_slope_120",
                   "rv_realized_vol_60", "rv_vol_ratio"],

    # A/B partner of regime_min: the same isolated baseline + an ISOLATED implied-vol
    # trio (level / skew / curvature) — the one orthogonal channel a price/return
    # series structurally cannot contain. Only 3 of the 14 iv_ columns, chosen to be
    # non-redundant and reliably populated (term-structure / 0DTE / 1DTE IV are
    # NaN-until-1DTE and excluded). Cross-asset (ca_) and microstructure/liquidity
    # (ms_, bp_) are deliberately excluded from this comparison.
    #   iv_atm_mean  — 30d constant-maturity ATM IV level
    #   iv_skew      — put/call skew (fear asymmetry)
    #   iv_butterfly — smile curvature / tail premium
    # ⚠ iv_ is populated only ~2025-06 → 2026-03 (options streamer stale after
    #   2026-04-02); the A/B eval window MUST sit inside that range or IV is all-NaN.
    "regime_min_iv": ["vw_vwap_dist", "vw_vwap_slope_120",
                      "rv_realized_vol_60", "rv_vol_ratio",
                      "iv_atm_mean", "iv_skew", "iv_butterfly"],

    # Standard-indicator baseline for sequence models: the universal trader set —
    # price/VWAP relation (vw_), multi-day trend + levels (ht_, dc_), candle structure
    # (cs_, including 5s OHLCV `cs_open`…`cs_return`, ADR-069), realized volatility
    # bands (rv_), time-of-day context (tc_). This is the engineered-feature
    # equivalent of "price, volume, VWAP, EMAs, bands, HTF levels".
    "sequence_baseline": ["vw_", "ht_", "dc_", "cs_", "rv_", "tc_"],

    # Microstructure-only view (order-flow-driven families / experiments).
    # `of_` includes tick OFI (`of_ofi` / `of_ofi_zscore_60`, ADR-068).
    "microstructure": ["bp_", "of_", "tl_", "ss_", "lt_", "ta_", "ms_", "sd_", "vp_"],

    # Economics / momentum edge (ADR-027, objective nq-50pt-econ default):
    # order-flow + book pressure + VWAP location + realized vol + HTF context +
    # sub-bar dynamics + time-of-day. Intentionally omits full options surface,
    # wavelets, macro/economic calendar noise — sparse enough for trees, rich
    # enough for L1 momentum. ~60–80 dims depending on parquet vintage.
    "momentum_edge": ["of_", "bp_", "vw_", "rv_", "ht_", "dc_", "sd_", "tc_", "tl_", "ss_"],

    # Causal champion-prefix replica (ADR-046). Same groups as registry
    # `xgb_tbm_topgroups` (`cs_/rv_/vp_/iv_`) after the ADR-041 closed-bar
    # parquet patch. 15m/30m CLV is the *previous closed* bucket, not the
    # forming one. Used by `nq-50pt-xgb-unique` — `momentum_edge` omitted CS
    # and the first unique wave had no side skill.
    "tbm_topgroups": ["cs_", "rv_", "vp_", "iv_"],

    # Closed-candle volume footprint (ADR-056): 1m/5m wick/body volume + 1m POC.
    # Not mixed into existing views so champions keep their column set.
    "candle_footprint": ["vf_"],

    # Parabolic SAR (ADR-055): 1m trail + 5s pierce/flip. Not mixed into
    # existing views so champions keep their column set.
    "psar": ["ps_"],

    # 5s mean MBP-10 L0–L9 depth imbalance (ADR-057). Isolated so champions
    # keep their column set. Only populated on dates with databento.mbp_10.
    "mbp_depth": ["mbp_"],

    # ATR-pop auction preliminary: tape flow + footprint + value + VWAP/daily
    # levels + causal au_* (IB / overnight / flow-at-level).
    "auction_pop": ["au_", "of_", "vf_", "vp_", "vw_", "dc_", "cs_", "tl_", "ss_"],
}


def view_column_indices(view: str, feature_cols: list[str]) -> list[int]:
    """Indices of *feature_cols* selected by *view* (prefix match). Raises on unknown view."""
    prefixes = FEATURE_VIEWS[view]
    return [i for i, c in enumerate(feature_cols) if any(c.startswith(p) for p in prefixes)]


def apply_feature_view(folds: list[dict[str, Any]], view: str) -> list[dict[str, Any]]:
    """Return folds with X arrays masked to the view's columns.

    Requires ``feature_cols`` on the folds (attached by the fold builder). Masks the
    feature axis of every ``X_*`` array (last axis, so both 2-D tabular and 3-D sequence
    shapes work) and rewrites ``feature_cols`` to the selected subset. Falls back to the
    unmasked folds if the view selects nothing (schema drift), with a warning.
    """
    if not folds:
        return folds
    feature_cols = folds[0].get("feature_cols")
    if not feature_cols:
        logger.warning("feature_view '%s' skipped: folds carry no feature_cols", view)
        return folds
    idx = view_column_indices(view, list(feature_cols))
    if not idx:
        logger.warning("feature_view '%s' selected 0 of %d columns — using full matrix",
                       view, len(feature_cols))
        return folds
    logger.info("feature_view '%s': %d → %d features", view, len(feature_cols), len(idx))
    return _mask_folds(folds, idx, list(feature_cols))


def _mask_folds(
    folds: list[dict[str, Any]], idx: list[int], feature_cols: list[str],
) -> list[dict[str, Any]]:
    """Mask the feature axis of every ``X_*`` array in *folds* to *idx*.

    Masks the last axis (so both 2-D tabular and 3-D sequence shapes work) and rewrites
    ``feature_cols`` to the selected subset. Shared by ``apply_feature_view`` (named views)
    and ``apply_feature_selection`` (explicit user picks).
    """
    sel_cols = [feature_cols[i] for i in idx]
    ix = np.asarray(idx, dtype=int)
    out: list[dict[str, Any]] = []
    for f in folds:
        g = dict(f)
        for k, v in f.items():
            if k.startswith("X_") and isinstance(v, np.ndarray):
                g[k] = np.take(v, ix, axis=-1)
            elif k.startswith("X_") and v is not None:      # memory-mapped / array-like
                g[k] = np.take(np.asarray(v), ix, axis=-1)
        g["feature_cols"] = sel_cols
        out.append(g)
    return out


def resolve_selection_indices(
    selection: dict[str, Any], feature_cols: list[str],
) -> list[int]:
    """Resolve a user-override feature *selection* to column indices into *feature_cols*.

    ``selection`` is exactly one of:

      * ``{"view": "<name>"}``     — a named view from ``FEATURE_VIEWS`` (prefix match).
      * ``{"prefixes": [...]}``    — keep columns whose name starts with any prefix.
      * ``{"columns": [...]}``     — keep columns matching these names exactly.

    Returns the (sorted, original-order) list of selected indices. Raises ``ValueError``
    for a malformed selection or an unknown view name.
    """
    if not isinstance(selection, dict) or len(selection) != 1:
        raise ValueError(
            "feature selection must be a dict with exactly one of "
            f"'view' | 'prefixes' | 'columns'; got {selection!r}")

    if "view" in selection:
        view = selection["view"]
        if view not in FEATURE_VIEWS:
            raise ValueError(f"unknown feature view '{view}'; "
                             f"known views: {sorted(FEATURE_VIEWS)}")
        return view_column_indices(view, feature_cols)

    if "prefixes" in selection:
        prefixes = list(selection["prefixes"])
        return [i for i, c in enumerate(feature_cols)
                if any(c.startswith(p) for p in prefixes)]

    if "columns" in selection:
        wanted = set(selection["columns"])
        return [i for i, c in enumerate(feature_cols) if c in wanted]

    raise ValueError(
        f"feature selection key must be 'view' | 'prefixes' | 'columns'; got {list(selection)}")


_CLOSE_NAMES = frozenset({"_close", "_close_logret", "close"})


def _is_close_only(selection: dict[str, Any]) -> bool:
    cols = selection.get("columns") if isinstance(selection, dict) else None
    return isinstance(cols, (list, tuple)) and len(cols) == 1 and str(cols[0]) in _CLOSE_NAMES


def close_logret_1d(close: np.ndarray) -> np.ndarray:
    """Causal log-return of a 1-D close series → shape ``(N, 1)`` float32."""
    c = np.asarray(close, dtype="float64").reshape(-1)
    if c.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    s = pd.Series(c)
    s = s.where(s > 0.0)
    s = s.ffill().bfill()
    c = s.to_numpy(dtype="float64")
    r = np.zeros(c.size, dtype=np.float32)
    if c.size > 1:
        prev = np.maximum(c[:-1], 1e-8)
        r[1:] = np.log(np.maximum(c[1:], 1e-8) / prev).astype(np.float32)
        r[~np.isfinite(r)] = 0.0
    return r.reshape(-1, 1)


def apply_close_logret(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace ``X_*`` with a 1-channel log-return of parquet ``_close``.

    Prefers fold extras ``close_{train,val,oof}`` (aligned to the split). Falls
    back to joining ``_close`` from the 5s parquet on ``ts_*``.
    """
    from src.search.side_head import aligned_close

    out: list[dict[str, Any]] = []
    for f in folds:
        g = dict(f)
        for split in ("train", "val", "oof"):
            xk = f"X_{split}"
            if xk not in g or g[xk] is None:
                continue
            n = int(np.asarray(g[xk]).shape[0])
            src = g.get(f"close_{split}")
            ts = g.get(f"ts_{split}")
            if src is None and ts is not None and n > 0:
                src = aligned_close(np.asarray(ts))
            if n == 0:
                g[xk] = np.zeros((0, 1), dtype=np.float32)
                continue
            if src is None or int(np.asarray(src).reshape(-1).size) != n:
                logger.warning(
                    "close_logret split=%s n=%d missing close — zeros", split, n,
                )
                g[xk] = np.zeros((n, 1), dtype=np.float32)
                continue
            g[xk] = close_logret_1d(np.asarray(src))
        g["feature_cols"] = ["_close_logret"]
        out.append(g)
    logger.info("feature_selection close-logret: 1 channel from parquet _close")
    return out


def apply_feature_selection(
    folds: list[dict[str, Any]], selection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return *folds* masked to a user-override feature *selection* (see
    ``resolve_selection_indices``).

    Falls back to the unmasked folds (with a warning) when the folds carry no
    ``feature_cols`` or the selection matches nothing — a candidate must never be
    killed by a feature-selection miss (schema/vintage drift).

    ``{"columns": ["_close"]}`` (aliases ``_close_logret`` / ``close``) is the
    5s-close TCN path: one causal log-return channel from parquet ``_close``.
    """
    if not folds or not selection:
        return folds
    if _is_close_only(selection):
        return apply_close_logret(folds)
    feature_cols = folds[0].get("feature_cols")
    if not feature_cols:
        logger.warning("feature_selection %s skipped: folds carry no feature_cols", selection)
        return folds
    idx = resolve_selection_indices(selection, list(feature_cols))
    if not idx:
        logger.warning("feature_selection %s matched 0 of %d columns — using full matrix",
                       selection, len(feature_cols))
        return folds
    logger.info("feature_selection %s: %d → %d features",
                selection, len(feature_cols), len(idx))
    return _mask_folds(folds, idx, list(feature_cols))
