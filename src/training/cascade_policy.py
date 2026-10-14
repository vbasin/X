"""Single source of truth for the cascade's risk geometry.

Stage 1 asks whether a move happens; Stage 3 asks which target that move
reaches. The two only compose if they are asked about the **same trade**, so the
move label's thresholds are derived from the target policy rather than chosen
independently::

    move_r    = min_tgt_r                 the smallest target worth taking
    adverse_r = stop_ratio * min_tgt_r    the stop that target would carry
    horizon   = shared with Stage 3

Before this alignment the three were unrelated literals — ``move_r`` 0.25,
``adverse_r`` 0.10, ``min_tgt_r`` 0.25 — so Stage 1 qualified a move against a
0.10 R adverse bound while Stage 3 traded it with a 0.125 R stop. A move Stage 1
called clean could already have been stopped out, and a move it rejected might
never have been: the two stages disagreed about what counted as the same trade.
"""

from __future__ import annotations

import numpy as np

# ── Stage 3 target policy (ADR-130) ──────────────────────────────────────────
MIN_TGT_R: float = 0.25          # smallest target worth taking after costs
STOP_RATIO: float = 0.5          # stop as a fraction of target distance
MAX_STOP_R: float = 0.25         # hard cap on risk per trade

# ── Shared cadence ───────────────────────────────────────────────────────────
HORIZON_SECONDS: int = 1800
DECISION_GRID_SECONDS: int = 5

# ── Stage 1 move label, derived from the target policy ───────────────────────
MOVE_R: float = MIN_TGT_R                  # 0.25 R
ADVERSE_R: float = STOP_RATIO * MIN_TGT_R  # 0.125 R


def stop_dist_r(tgt_dist_r, stop_ratio: float = STOP_RATIO,
                max_stop_r: float = MAX_STOP_R):
    """Target-adaptive stop distance in R (ADR-130); vectorised over arrays."""
    return np.minimum(stop_ratio * np.asarray(tgt_dist_r, dtype="float64"),
                      max_stop_r)


def describe() -> str:
    """One-line summary for run logs, so a rebuild records its own geometry."""
    return (f"move_r={MOVE_R:g} adverse_r={ADVERSE_R:g} min_tgt_r={MIN_TGT_R:g} "
            f"stop_ratio={STOP_RATIO:g} max_stop_r={MAX_STOP_R:g} "
            f"horizon={HORIZON_SECONDS}s grid={DECISION_GRID_SECONDS}s")
