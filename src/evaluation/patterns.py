"""Outcome pattern mining — deterministic success/failure analysis over episodes.

Consumes the episode frames recorded by :mod:`src.evaluation.episodes` (one row per
fired-signal win/loss or missed true move, annotated with market context) and computes
**which conditions separate the successes from the failures**:

* numeric context separation — Cohen's d + Mann-Whitney p between win vs loss episodes
  (e.g. "losses concentrate when rv_realized_vol_60 is high");
* **blind spots** — the same separation between *missed* true moves and *caught* ones
  (what the model/strategy systematically fails to see);
* time-structure lift — win-rate by hour / day-of-week vs the overall base rate;
* confidence calibration (models) — precision per confidence bucket;
* directional asymmetry — up vs down win rates;
* deterministic **suggestions** — machine-readable seeds (add feature / add filter /
  avoid window / rebalance direction) that the Research Director agents and the
  feature-refinement loop interpret.

Everything here is pure pandas/scipy — the LLM agents *interpret* these numbers, they
never compute them (LLM proposes, deterministic code disposes).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.evaluation.patterns")

_META_COLS = {"ts", "direction", "confidence", "direction_true", "outcome", "fired",
              "pnl_points", "exit_reason", "holding_bars", "mae_points", "mfe_points",
              "side", "hour", "day_of_week"}
_LOSS_OUTCOMES = ("wrong_direction", "false_positive_flat")

_MIN_GROUP = 5            # minimum episodes per side for a separation stat
_STRONG_D = 0.5           # |Cohen's d| considered actionable
_MIN_SUPPORT = 20         # minimum bucket size for a lift pattern
_LIFT_FLOOR = 0.10        # |win-rate lift| considered actionable


def mine_patterns(
    episodes: pd.DataFrame,
    *,
    min_support: int = _MIN_SUPPORT,
    top_k: int = 12,
) -> dict[str, Any]:
    """Mine success/failure patterns from an episodes frame. Never raises on shape."""
    if episodes is None or episodes.empty or "outcome" not in episodes.columns:
        return {"n_episodes": 0, "patterns": [], "suggestions": [], "summary_text": ""}

    ep = episodes
    fired = ep[ep["outcome"].isin(("win",) + _LOSS_OUTCOMES)]
    wins = ep[ep["outcome"] == "win"]
    losses = ep[ep["outcome"].isin(_LOSS_OUTCOMES)]
    missed = ep[ep["outcome"] == "missed"]
    base_rate = float(len(wins) / len(fired)) if len(fired) else float("nan")

    ctx_cols = [c for c in ep.columns
                if c not in _META_COLS and pd.api.types.is_numeric_dtype(ep[c])]

    patterns = _numeric_separations(wins, losses, ctx_cols, top_k=top_k,
                                    a_name="win", b_name="loss")
    blind = _numeric_separations(wins, missed, ctx_cols, top_k=max(6, top_k // 2),
                                 a_name="caught", b_name="missed")
    tod = _bucket_lift(fired, "hour", base_rate, min_support)
    dow = _bucket_lift(fired, "day_of_week", base_rate, min_support)
    calib = _calibration(fired)
    asym = _asymmetry(fired)

    suggestions = _suggestions(patterns, blind, tod, asym, base_rate)

    report = {
        "n_episodes":       int(len(ep)),
        "outcome_counts":   ep["outcome"].value_counts().to_dict(),
        "win_rate_fired":   base_rate,
        "patterns":         patterns,
        "missed_blind_spots": blind,
        "time_of_day":      tod,
        "day_of_week":      dow,
        "calibration":      calib,
        "direction_asymmetry": asym,
        "suggestions":      suggestions,
    }
    report["summary_text"] = _render_summary(report)
    return report


# ── stats ─────────────────────────────────────────────────────────────────────

def separation_table(
    episodes: pd.DataFrame,
    cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Full per-feature |Cohen's d| tables (no top-k truncation) for programmatic use.

    Returns ``{"win_vs_loss": {feature: d}, "caught_vs_missed": {feature: d}}`` —
    the evidence base the feature-refinement loop uses to decide adds/removals.
    """
    if episodes is None or episodes.empty or "outcome" not in episodes.columns:
        return {"win_vs_loss": {}, "caught_vs_missed": {}}
    if cols is None:
        cols = [c for c in episodes.columns
                if c not in _META_COLS and pd.api.types.is_numeric_dtype(episodes[c])]
    wins = episodes[episodes["outcome"] == "win"]
    losses = episodes[episodes["outcome"].isin(_LOSS_OUTCOMES)]
    missed = episodes[episodes["outcome"] == "missed"]
    wl = _numeric_separations(wins, losses, cols, top_k=len(cols) or 1,
                              a_name="win", b_name="loss")
    cm = _numeric_separations(wins, missed, cols, top_k=len(cols) or 1,
                              a_name="caught", b_name="missed")
    return {
        "win_vs_loss":      {r["feature"]: r["cohens_d"] for r in wl},
        "caught_vs_missed": {r["feature"]: r["cohens_d"] for r in cm},
    }


def _numeric_separations(
    a: pd.DataFrame,
    b: pd.DataFrame,
    cols: list[str],
    *,
    top_k: int,
    a_name: str,
    b_name: str,
) -> list[dict[str, Any]]:
    """Cohen's d + Mann-Whitney p for every context col between groups *a* and *b*."""
    if len(a) < _MIN_GROUP or len(b) < _MIN_GROUP:
        return []
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    out: list[dict[str, Any]] = []
    for c in cols:
        xa = pd.to_numeric(a[c], errors="coerce").dropna()
        xb = pd.to_numeric(b[c], errors="coerce").dropna()
        if len(xa) < _MIN_GROUP or len(xb) < _MIN_GROUP:
            continue
        sa, sb = float(xa.std(ddof=1)), float(xb.std(ddof=1))
        pooled = np.sqrt(((len(xa) - 1) * sa**2 + (len(xb) - 1) * sb**2)
                         / max(1, len(xa) + len(xb) - 2))
        if not np.isfinite(pooled) or pooled == 0.0:
            continue
        d = float((xa.mean() - xb.mean()) / pooled)
        p = None
        if mannwhitneyu is not None:
            try:
                p = float(mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            except Exception:  # noqa: BLE001
                p = None
        out.append({
            "kind":        f"{a_name}_vs_{b_name}_separation",
            "feature":     c,
            "cohens_d":    round(d, 4),
            "p_value":     (round(p, 6) if p is not None else None),
            f"{a_name}_mean": round(float(xa.mean()), 6),
            f"{b_name}_mean": round(float(xb.mean()), 6),
            "n": [int(len(xa)), int(len(xb))],
            "description": (
                f"{c}: {a_name} mean {xa.mean():.4g} vs {b_name} mean {xb.mean():.4g} "
                f"(d={d:+.2f}{', p=' + format(p, '.2g') if p is not None else ''}) — "
                f"{b_name} episodes sit {'higher' if d < 0 else 'lower'} on this feature"),
        })
    out.sort(key=lambda r: abs(r["cohens_d"]), reverse=True)
    return out[:top_k]


def _bucket_lift(
    fired: pd.DataFrame,
    col: str,
    base_rate: float,
    min_support: int,
) -> list[dict[str, Any]]:
    """Win-rate per bucket of *col* vs the overall fired base rate."""
    if col not in fired.columns or fired.empty or not np.isfinite(base_rate):
        return []
    rows: list[dict[str, Any]] = []
    for val, grp in fired.groupby(col):
        n = int(len(grp))
        if n < min_support:
            continue
        wr = float((grp["outcome"] == "win").mean())
        rows.append({
            "kind": "bucket_lift", "feature": col, "bucket": int(val),
            "win_rate": round(wr, 4), "baseline": round(base_rate, 4),
            "lift": round(wr - base_rate, 4), "support": n,
            "description": (f"{col}={val}: win rate {wr:.2f} vs baseline "
                            f"{base_rate:.2f} ({wr - base_rate:+.2f}, n={n})"),
        })
    rows.sort(key=lambda r: abs(r["lift"]), reverse=True)
    return rows


def _calibration(fired: pd.DataFrame) -> list[dict[str, Any]]:
    """Precision per confidence bucket (meaningful for model sources only)."""
    if fired.empty or fired["confidence"].nunique() <= 1:
        return []
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    out = []
    cut = pd.cut(fired["confidence"], bins=bins, right=False)
    for iv, grp in fired.groupby(cut, observed=True):
        if len(grp) == 0:
            continue
        out.append({"bucket": str(iv), "n": int(len(grp)),
                    "precision": round(float((grp["outcome"] == "win").mean()), 4)})
    return out


def _asymmetry(fired: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d, grp in fired.groupby("direction"):
        out[str(d)] = {"n": int(len(grp)),
                       "win_rate": round(float((grp["outcome"] == "win").mean()), 4)}
    return out


# ── suggestions (machine-readable optimization seeds) ─────────────────────────

def _suggestions(
    patterns: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    tod: list[dict[str, Any]],
    asym: dict[str, Any],
    base_rate: float,
) -> list[dict[str, Any]]:
    sug: list[dict[str, Any]] = []
    for p in patterns:
        if abs(p["cohens_d"]) >= _STRONG_D:
            sug.append({
                "action": "add_filter_or_regime_gate",
                "target": p["feature"],
                "evidence": p["description"],
                "rationale": ("win/loss episodes separate strongly on this feature — "
                              "gating or conditioning on it should cut the failure mode"),
            })
    for p in blind:
        if abs(p["cohens_d"]) >= _STRONG_D:
            sug.append({
                "action": "add_feature_or_data",
                "target": p["feature"],
                "evidence": p["description"],
                "rationale": ("missed true moves live in a different context than caught "
                              "ones — the system is blind here; add features/data that "
                              "resolve this region"),
            })
    for t in tod:
        if t["lift"] <= -_LIFT_FLOOR:
            sug.append({
                "action": "avoid_time_window",
                "target": f"hour={t['bucket']}",
                "evidence": t["description"],
                "rationale": "win rate materially below base rate in this window",
            })
    ups, downs = asym.get("up"), asym.get("down")
    if ups and downs and ups["n"] >= _MIN_GROUP and downs["n"] >= _MIN_GROUP:
        gap = ups["win_rate"] - downs["win_rate"]
        if abs(gap) >= _LIFT_FLOOR:
            weak = "down" if gap > 0 else "up"
            sug.append({
                "action": "rebalance_direction",
                "target": weak,
                "evidence": f"up wr={ups['win_rate']:.2f} vs down wr={downs['win_rate']:.2f}",
                "rationale": (f"the {weak} side underperforms materially — consider "
                              "side-specific thresholds, features, or a specialist model"),
            })
    return sug[:12]


# ── report assembly / persistence ─────────────────────────────────────────────

def _render_summary(r: dict[str, Any]) -> str:
    """Compact text block for Context-Pack injection (agents read this)."""
    lines = [
        f"episodes={r['n_episodes']} outcomes={r['outcome_counts']} "
        f"fired_win_rate={r['win_rate_fired'] if r['win_rate_fired'] == r['win_rate_fired'] else 'n/a'}",
    ]
    if r["patterns"]:
        lines.append("Top win-vs-loss separations (what distinguishes failures):")
        lines += [f"  - {p['description']}" for p in r["patterns"][:6]]
    if r["missed_blind_spots"]:
        lines.append("Blind spots (missed true moves vs caught ones):")
        lines += [f"  - {p['description']}" for p in r["missed_blind_spots"][:4]]
    hot = [t for t in r["time_of_day"] if abs(t["lift"]) >= _LIFT_FLOOR][:4]
    if hot:
        lines.append("Time-of-day effects: " + "; ".join(t["description"] for t in hot))
    if r["direction_asymmetry"]:
        lines.append(f"Direction asymmetry: {r['direction_asymmetry']}")
    if r["suggestions"]:
        lines.append("Deterministic optimization seeds:")
        lines += [f"  * [{s['action']}] {s['target']} — {s['rationale']}"
                  for s in r["suggestions"][:6]]
    return "\n".join(lines)


def analyze_run(
    objective_id: str,
    source_id: str,
    run_ts: str,
    episodes: pd.DataFrame,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mine patterns and wrap them in a persistable outcome report."""
    report = mine_patterns(episodes)
    report.update({
        "objective_id": objective_id,
        "source_id":    source_id,
        "run_ts":       run_ts,
        "kind":         (meta or {}).get("kind", "model"),
        "meta":         meta or {},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    return report


def analyze_and_write(
    objective_id: str,
    source_id: str,
    run_ts: str,
    episodes: pd.DataFrame,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``analyze_run`` + persist ``<run_ts>__<source_id>__outcome_report.json``."""
    from src.evaluation.episodes import _episodes_dir, _safe

    report = analyze_run(objective_id, source_id, run_ts, episodes, meta=meta)
    d = _episodes_dir(objective_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_ts}__{_safe(source_id)}__outcome_report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("outcome_report_written %s", path)
    return report
