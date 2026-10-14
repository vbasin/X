"""Reproducibility helpers: deterministic seeding, git SHA, experiment log.

Used by `src.training.trainer`. Designed to be imported lazily so that scripts
that don't need TensorFlow don't pay the import cost.
"""

from __future__ import annotations

from src.common.data_paths import checkpoints_dir

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


_DEFAULT_SEED = 42
_EXPERIMENT_LOG = checkpoints_dir() / "_experiments.jsonl"


def get_seed() -> int:
    """Resolve the active seed from `VTECH_SEED` (default 42)."""
    raw = os.environ.get("VTECH_SEED")
    if raw is None:
        return _DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_SEED


def set_global_seed(seed: int | None = None) -> int:
    """Seed every RNG that affects training reproducibility.

    Returns the seed actually applied so callers can record it.
    """
    seed = seed if seed is not None else get_seed()

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # local import — keep this module light
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        # Force op-level determinism (slower but exact reproducibility).
        # Skip if TF version is too old.
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass

    return seed


def get_git_sha() -> str | None:
    """Return current HEAD short SHA, or None if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "-C", "/opt/vtech", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def hash_dict(d: dict[str, Any]) -> str:
    """Stable SHA-256 hash of a JSON-serialisable dict, first 12 chars."""
    payload = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def log_experiment(record: dict[str, Any]) -> None:
    """Append an experiment record to /opt/vtech/checkpoints/_experiments.jsonl.

    The caller is responsible for assembling `record`. We add `ts` and
    `git_sha` automatically if absent.
    """
    record = dict(record)
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record.setdefault("git_sha", get_git_sha())

    _EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _EXPERIMENT_LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
