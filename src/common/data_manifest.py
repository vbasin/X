"""Data-versioning helpers: parquet manifest, schema snapshot, schema diff.

Used by:
  * `src.features.engine` — write a `_manifest.json` next to each parquet on build.
  * `src.training.trainer` — capture a feature-schema snapshot in the model
    checkpoint so serve-time can verify column-level compatibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file (full SHA-256, hex)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def manifest_for_parquet(path: Path) -> dict[str, Any]:
    """Lightweight manifest entry for a parquet file."""
    df = pd.read_parquet(path)
    return {
        "filename": path.name,
        "sha256": hash_file(path),
        "row_count": int(df.shape[0]),
        "col_count": int(df.shape[1]),
        "size_bytes": path.stat().st_size,
        "columns": list(df.columns),
    }


def schema_for_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Extract a column-name → dtype mapping (str dtypes for JSON safety)."""
    return {col: str(df[col].dtype) for col in df.columns}


def diff_schemas(a: dict[str, str], b: dict[str, str]) -> dict[str, list[str]]:
    """Return the column-set delta between two schemas: missing/extra/changed."""
    a_keys, b_keys = set(a), set(b)
    missing = sorted(a_keys - b_keys)  # in a, not in b
    extra = sorted(b_keys - a_keys)    # in b, not in a
    changed = sorted(k for k in a_keys & b_keys if a[k] != b[k])
    return {"missing": missing, "extra": extra, "changed_dtype": changed}


def assert_schema_compatible(
    train_schema: dict[str, str],
    serve_df: pd.DataFrame,
    *,
    ignore_extra_at_serve: bool = True,
) -> None:
    """Raise ValueError if a serve-time DataFrame is incompatible with the
    schema captured at train time.
    """
    serve_schema = schema_for_dataframe(serve_df)
    diff = diff_schemas(train_schema, serve_schema)

    problems = []
    if diff["missing"]:
        problems.append(f"missing columns: {diff['missing']}")
    if diff["changed_dtype"]:
        problems.append(f"dtype changed: {diff['changed_dtype']}")
    if diff["extra"] and not ignore_extra_at_serve:
        problems.append(f"unexpected extra columns: {diff['extra']}")

    if problems:
        raise ValueError("Schema incompatibility: " + "; ".join(problems))


def write_manifest(parquet_path: Path) -> Path:
    """Write `<parquet>.manifest.json` next to a parquet. Returns its path."""
    manifest = manifest_for_parquet(parquet_path)
    out = parquet_path.with_suffix(parquet_path.suffix + ".manifest.json")
    with out.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return out
