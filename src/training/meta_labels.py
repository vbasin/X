"""Meta-labeling (primary/secondary dual-layer signals) — López de Prado ch. 3.

Canonical companion to triple-barrier labels:

* **Primary signal** — a directional model proposes a side: up / down / flat.
* **Secondary (meta) signal** — a binary trade/no-trade model predicts whether
  ACTING on the primary's proposed side would have won under the objective's label
  scheme. It filters the primary's signals to raise precision; it never sets direction.

Active path (ADR-027): ``src/search/meta_compose.py`` cross-fits a meta gate on
primary OOF when the objective sets ``search.topology.mode: primary_meta``
(see ``config/objectives/nq-50pt-econ.yaml``). Legacy objectives leave topology
unset and keep the single-model 3-class signal.

LEAKAGE GUARD: the meta model must be trained ONLY on out-of-fold primary
predictions (walk-forward / time-ordered CV). Training it on in-sample primary
predictions leaks the primary's memorised outcomes and inflates meta precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw label values (src/training/labels.py)
_BIG_DOWN, _NO_SIGNAL, _BIG_UP = 0, 1, 2
_WIN_LABEL = {"up": _BIG_UP, "down": _BIG_DOWN}


def generate_meta_labels(
    primary_direction: pd.Series,
    raw_labels: pd.Series,
) -> pd.Series:
    """Binary meta-labels: 1 iff acting on the primary's side would have won.

    Parameters
    ----------
    primary_direction:
        Canonical direction strings per bar: ``"up"`` / ``"down"`` / ``"flat"``
        (a primary model's OOF ``direction`` column, index-aligned to *raw_labels*).
    raw_labels:
        Ground-truth raw labels per bar: {0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP} from
        ``generate_labels`` (either scheme).

    Returns
    -------
    pd.Series (int8)
        1 where the primary proposed a side AND the label confirms that side;
        0 otherwise. ``flat`` proposals are always 0 (no trade proposed →
        nothing for the meta layer to approve).
    """
    if len(primary_direction) != len(raw_labels):
        raise ValueError(
            f"length mismatch: primary={len(primary_direction)} labels={len(raw_labels)}")
    dir_arr = primary_direction.astype(str).to_numpy()
    lbl_arr = raw_labels.to_numpy()
    win = np.zeros(len(lbl_arr), dtype=np.int8)
    for side, win_label in _WIN_LABEL.items():
        win |= ((dir_arr == side) & (lbl_arr == win_label)).astype(np.int8)
    return pd.Series(win, index=raw_labels.index, name="meta_label")


def build_meta_training_frame(
    primary_oof: pd.DataFrame,
    raw_labels: pd.Series,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the secondary model's training frame from a primary model's OOF.

    Parameters
    ----------
    primary_oof:
        Canonical OOF frame (``ts``, ``direction``, ``confidence``) as persisted by
        the search controller (``oof_<candidate>.parquet``). Must be out-of-fold —
        see the module docstring leakage guard.
    raw_labels:
        Ground-truth raw labels indexed by bar timestamp (tz-aware UTC).
    context:
        Optional per-bar context features (same timestamp index) to give the meta
        model regime/microstructure awareness beyond the primary's confidence.

    Returns
    -------
    pd.DataFrame
        One row per primary DIRECTIONAL proposal (flat rows dropped): columns
        ``direction`` (categorical side), ``confidence`` (primary), any *context*
        columns, and the target ``meta_label``.
    """
    oof = primary_oof.copy()
    oof["ts"] = pd.to_datetime(oof["ts"], utc=True)
    oof = oof.set_index("ts")
    aligned = raw_labels.reindex(oof.index)
    frame = oof[["direction", "confidence"]].copy()
    frame["meta_label"] = generate_meta_labels(frame["direction"], aligned).to_numpy()
    if context is not None:
        frame = frame.join(context, how="left")
    directional = frame[frame["direction"].isin(("up", "down"))].dropna(subset=["confidence"])
    return directional
