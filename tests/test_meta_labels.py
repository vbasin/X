"""Tests: meta-label generation (primary/secondary dual-layer prep — inactive path)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.meta_labels import build_meta_training_frame, generate_meta_labels


def _idx(n):
    return pd.date_range("2026-02-02 15:00", periods=n, freq="5s", tz="UTC")


def test_meta_label_win_mapping():
    idx = _idx(6)
    direction = pd.Series(["up", "up", "down", "down", "flat", "up"], index=idx)
    raw = pd.Series([2, 0, 0, 2, 2, 1], index=idx)  # BIG_UP, BIG_DOWN, ...
    meta = generate_meta_labels(direction, raw)
    #                 up+UP  up+DOWN down+DOWN down+UP flat  up+NS
    assert meta.tolist() == [1, 0, 1, 0, 0, 0]
    assert meta.dtype == np.int8


def test_meta_label_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        generate_meta_labels(pd.Series(["up"]), pd.Series([2, 1]))


def test_build_meta_training_frame_alignment_and_flat_drop():
    idx = _idx(5)
    oof = pd.DataFrame({
        "ts": idx,
        "direction": ["up", "flat", "down", "up", "down"],
        "confidence": [0.9, 0.5, 0.8, 0.7, 0.6],
    })
    raw = pd.Series([2, 2, 0, 0, 1], index=idx)
    ctx = pd.DataFrame({"rv": np.arange(5.0)}, index=idx)
    frame = build_meta_training_frame(oof, raw, context=ctx)
    assert len(frame) == 4                       # flat row dropped
    assert list(frame["meta_label"]) == [1, 1, 0, 0]
    assert "rv" in frame.columns and frame["rv"].notna().all()


def test_meta_labels_consumers_are_the_approved_compose_path():
    """ADR-027: meta_labels is consumed by the primary+meta compose path only.

    Allowed importers: ``src/search/meta_compose.py`` (and this module's package).
    Anything else under src/ still needs an explicit design review.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    allowed = {
        root / "search" / "meta_compose.py",
        root / "training" / "meta_labels.py",
    }
    offenders = []
    for p in root.rglob("*.py"):
        if p in allowed or p.name == "meta_labels.py":
            continue
        text = p.read_text()
        if "meta_labels" in text or "from src.training.meta_labels" in text:
            offenders.append(str(p.relative_to(root.parent)))
    assert offenders == [], f"unexpected meta_labels consumers: {offenders}"
