"""Shared serving-bundle load + inference (Live and Backtest SSOT).

Both the Live tab and the Backtest tab **must** score via this package so
direction/confidence cannot drift between consoles.

Public API::

    from src.serving import (
        is_serving_bundle,
        validate_bundle_path,
        load_serving_bundle,
        predict_canonical,
        direction_from_proba,
    )
"""

from src.serving.bundle import (
    is_serving_bundle,
    load_serving_bundle,
    validate_bundle_path,
)
from src.serving.predict import direction_from_proba, predict_canonical

__all__ = [
    "is_serving_bundle",
    "validate_bundle_path",
    "load_serving_bundle",
    "predict_canonical",
    "direction_from_proba",
]
