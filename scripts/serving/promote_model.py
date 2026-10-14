#!/usr/bin/env python3
"""Promote a staged serve bundle into model_registry/<objective_id>/LATEST.

Thin wrapper around :func:`src.serving.promote.promote`.

Usage::

    python scripts/serving/promote_model.py \\
        --staging-dir data/models/model_staging/nq-50pt-econ/xgb_tbm_topgroups_20260715T234308Z

    python -m src.serving.promote --objective nq-50pt-econ --version xgb_tbm_topgroups_20260715T234308Z
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as scripts/serving/promote_model.py without install
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.serving.promote import promote_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(promote_cli())
