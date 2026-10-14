"""Ensemble combination layer over cached OOF predictions — Phase 23 W6.

Combines the base models' **out-of-fold (OOF) predictions** — already produced by the
full walk-forward search — into ensemble candidates, **without retraining any base model**.
Every ensemble is scored on the *same frozen evaluation contract* as the base models
(fair contest); scoring/diagnostics live in the controller, the combiner *math* lives here.

Methods (all operate on the cached ``ts / direction / confidence / direction_true`` OOF
frames; base OOF carries argmax + confidence, not per-class probabilities):

* ``consensus`` — fire a direction only when ``min_agree`` members agree; confidence = the
  mean of the *agreeing* members' confidences. **Tail-preserving:** unlike mean-vote it does
  not average in the disagreeing/low-confidence members, so the high-τ signal survives — this
  is the fix for the W4 finding that mean-vote produced ~0 signals at τ≥0.9.
* ``stacking`` — a **cross-fitted** (K-fold) multinomial logistic-regression meta-model on
  confidence-weighted one-hot meta-features (per base model: ``conf·1{dir==d}`` for
  d∈{down,flat,up}). Cross-fitting yields leakage-free meta-OOF predictions with no base
  retraining. Learns to trust confident, historically-correct members rather than averaging.
* ``vote`` — legacy mean-confidence majority vote, kept as a **labeled baseline** so the
  leaderboard shows the dilution effect explicitly. Not recommended as a champion.

Not built from cached OOF (documented, not faked):
* ``bagging`` ≈ soft-voting once we only have argmax+confidence → equivalent to ``vote`` here.
* ``boosting`` is a *base-level* method (xgboost/lightgbm already ARE boosting) — a separate
  OOF-boosting ensemble would require sequential base retraining, which W6 avoids.
* ``regime_routing`` needs a per-row ``regime_id`` from the regime plugins (not in the OOF
  today) — deferred until regime plugins persist cluster assignments.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.search.ensembles")

TS, DIR, CONF, TRUE = "ts", "direction", "confidence", "direction_true"
_DIRS = ("down", "flat", "up")
_DIR_IDX = {"down": 0, "flat": 1, "up": 2}
_IDX_DIR = {0: "down", 1: "flat", 2: "up"}

ALLOWED_METHODS = ("consensus", "stacking", "vote")
DEFAULT_METHODS = ("consensus", "stacking", "vote")
DEFAULT_MAX_BASE = 4


# ── alignment ──────────────────────────────────────────────────────────────────

def _align(frames: list[pd.DataFrame]) -> tuple[pd.Series, list[pd.DataFrame]]:
    """Align OOF frames. Prefer inner join on ``ts`` when every frame has real
    timestamps (sequence models drop warmup bars). Fall back to positional
    truncate when ``ts`` is missing/NaT."""
    usable = []
    for f in frames:
        if TS not in f.columns:
            usable = []
            break
        ts = pd.to_datetime(f[TS], utc=True, errors="coerce")
        if ts.notna().sum() < max(1, int(0.5 * len(f))):
            usable = []
            break
        usable.append(f)
    if len(usable) == len(frames) and frames:
        common = None
        for f in frames:
            ts = pd.to_datetime(f[TS], utc=True, errors="coerce")
            s = set(ts.dropna().astype("datetime64[ns, UTC]"))
            common = s if common is None else (common & s)
        if common:
            aligned = []
            for f in frames:
                g = f.copy()
                g[TS] = pd.to_datetime(g[TS], utc=True, errors="coerce")
                g = g[g[TS].isin(common)].drop_duplicates(TS).sort_values(TS)
                aligned.append(g.reset_index(drop=True))
            n = min(len(f) for f in aligned)
            aligned = [f.iloc[:n] for f in aligned]
            y_true = (aligned[0][TRUE].reset_index(drop=True)
                      if TRUE in aligned[0].columns else pd.Series([None] * n))
            return y_true, aligned
    n = min(len(f) for f in frames)
    aligned = [f.reset_index(drop=True).iloc[:n] for f in frames]
    y_true = aligned[0][TRUE].reset_index(drop=True) if TRUE in aligned[0].columns else pd.Series([None] * n)
    return y_true, aligned


def _spine(frames: list[pd.DataFrame], n: int) -> pd.DataFrame:
    """ts + direction_true carrier for the combined OOF frame."""
    f0 = frames[0].reset_index(drop=True).iloc[:n]
    return pd.DataFrame({
        TS:   f0[TS].reset_index(drop=True) if TS in f0.columns else pd.NaT,
        TRUE: f0[TRUE].reset_index(drop=True) if TRUE in f0.columns else None,
    })


# ── combiners ────────────────────────────────────────────────────────────────

def consensus_combine(frames: list[pd.DataFrame], *, min_agree: int | None = None) -> pd.DataFrame:
    """Fire a direction only when ≥ ``min_agree`` members agree; confidence = mean of the
    agreeing members. Default ``min_agree`` = simple majority (unanimity for pairs)."""
    y_true, aligned = _align(frames)
    n, k = len(aligned[0]), len(aligned)
    if min_agree is None:
        min_agree = (k // 2) + 1                       # majority; k=2 → 2 (unanimous)

    dirs = np.stack([f[DIR].to_numpy() for f in aligned], axis=1)      # (n, k)
    confs = np.stack([f[CONF].to_numpy(dtype="float64") for f in aligned], axis=1)

    out_dir = np.full(n, "flat", dtype=object)
    out_conf = np.zeros(n, dtype="float64")
    for d in ("up", "down"):                            # flat is the default / no-fire
        mask_member = dirs == d                         # (n, k) which members voted d
        votes = mask_member.sum(axis=1)
        fire = votes >= min_agree
        # confidence = mean of the agreeing members' confidences (no-vote rows → NaN, discarded)
        with np.errstate(invalid="ignore"):
            mean_conf = np.nanmean(np.where(mask_member, confs, np.nan), axis=1)
        mean_conf = np.nan_to_num(mean_conf)
        take = fire & (out_dir == "flat")               # up checked first; don't overwrite
        out_dir[take] = d
        out_conf[take] = mean_conf[take]

    merged = _spine(aligned, n)
    merged[DIR] = out_dir
    merged[CONF] = out_conf
    return merged


def mean_vote_combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Legacy majority direction + mean confidence (diluting baseline — kept for comparison)."""
    y_true, aligned = _align(frames)
    n = len(aligned[0])
    dir_votes = pd.concat([f[DIR].reset_index(drop=True) for f in aligned], axis=1)
    voted = dir_votes.mode(axis=1)[0]
    avg_conf = pd.concat([f[CONF].reset_index(drop=True) for f in aligned], axis=1).mean(axis=1)
    merged = _spine(aligned, n)
    merged[DIR] = voted.to_numpy()
    merged[CONF] = avg_conf.to_numpy()
    return merged


def _meta_features(frames: list[pd.DataFrame]) -> np.ndarray:
    """Per base model, 3 confidence-weighted one-hot columns (conf·1{dir==d}). (n, 3k)."""
    blocks = []
    for f in frames:
        d = f[DIR].to_numpy()
        c = f[CONF].to_numpy(dtype="float64")
        block = np.zeros((len(f), 3), dtype="float64")
        for name, idx in _DIR_IDX.items():
            block[:, idx] = np.where(d == name, c, 0.0)
        blocks.append(block)
    return np.hstack(blocks)


def stacking_combine(frames: list[pd.DataFrame], *, n_splits: int = 5,
                     seed: int = 42) -> pd.DataFrame:
    """Cross-fitted multinomial-LR meta-model on confidence-weighted meta-features.

    Produces leakage-free meta-OOF predictions: each row is predicted by a meta-model fit
    on the *other* folds. Falls back to a single in-sample fit only if the data is too small
    to stratify (rare)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y_true, aligned = _align(frames)
    n = len(aligned[0])
    X = _meta_features(aligned)
    y = np.array([_DIR_IDX.get(str(v), 1) for v in y_true.to_numpy()], dtype=int)

    proba = np.full((n, 3), np.nan, dtype="float64")
    classes_seen = np.unique(y)
    min_class = int(min((y == c).sum() for c in classes_seen)) if len(classes_seen) else 0

    def _fit(Xtr, ytr):
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        lr.fit(Xtr, ytr)
        return lr

    if len(classes_seen) >= 2 and min_class >= 2:
        splits = max(2, min(n_splits, min_class))
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            lr = _fit(X[tr], y[tr])
            p = lr.predict_proba(X[te])
            for j, c in enumerate(lr.classes_):
                proba[te, int(c)] = p[:, j]
    else:  # degenerate: single fit (best-effort; not enough classes to cross-fit)
        lr = _fit(X, y)
        p = lr.predict_proba(X)
        for j, c in enumerate(lr.classes_):
            proba[:, int(c)] = p[:, j]

    proba = np.nan_to_num(proba)
    idx = proba.argmax(axis=1)
    merged = _spine(aligned, n)
    merged[DIR] = [_IDX_DIR[i] for i in idx]
    merged[CONF] = proba.max(axis=1)
    return merged


# ── combination planning (bounded) ─────────────────────────────────────────────

def plan_ensembles(
    names: list[str],
    primary_precisions: dict[str, float],
    *,
    max_base: int = DEFAULT_MAX_BASE,
    methods: tuple[str, ...] = DEFAULT_METHODS,
) -> list[dict[str, Any]]:
    """Bounded combination plan: pick the top-``max_base`` base learners by primary precision,
    then enumerate all pairs + triples + the all-N set, each crossed with every method.

    Avoids literal 2^N while staying thorough over the strongest base learners."""
    methods = tuple(m for m in methods if m in ALLOWED_METHODS) or DEFAULT_METHODS
    top = sorted(names, key=lambda n: primary_precisions.get(n, -1.0), reverse=True)[:max_base]

    combos: list[tuple[str, ...]] = []
    combos += list(combinations(top, 2))
    if len(top) >= 3:
        combos += list(combinations(top, 3))
    if len(top) > 3:
        combos.append(tuple(top))                       # all-N (only if not already a triple)

    # de-dup (sorted membership)
    seen: set[tuple[str, ...]] = set()
    plans: list[dict[str, Any]] = []
    for c in combos:
        key = tuple(sorted(c))
        if key in seen:
            continue
        seen.add(key)
        for m in methods:
            plans.append({"method": m, "members": list(c)})
    return plans


def combine(method: str, frames: list[pd.DataFrame], *, min_agree: int | None = None) -> pd.DataFrame:
    """Dispatch to the named combiner."""
    if method == "consensus":
        return consensus_combine(frames, min_agree=min_agree)
    if method == "stacking":
        return stacking_combine(frames)
    if method == "vote":
        return mean_vote_combine(frames)
    raise ValueError(f"unknown ensemble method '{method}'")
