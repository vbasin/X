#!/usr/bin/env python3
"""Migrate one existing Track B run into the canonical layout.

The ``research/custom_test/`` tree has already been migrated and removed.
This script remains for a leftover directory that still has the old shape.

Thin CLI over ``src/search/canonical_run.py::emit_canonical_run`` -- the same
function ``src/search/controller.py`` calls when a search run finishes, so a
migrated historical run and a freshly written one go through one code path.

If a leftover source directory is passed, the script copies into
``research/runs/`` and does not delete the source.

Example::

    python scripts/research/migrate_custom_test.py \
        --objective nq-50pt-econ-lg-st --run-ts 20260829T182514Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data_paths import custom_test_dir  # noqa: E402
from src.search.canonical_run import emit_canonical_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--src", default=None,
                    help="source run dir; defaults to custom_test/<objective>/<run_ts>")
    ap.add_argument("--dest-run-ts", default=None,
                    help="write under a different run_ts (default: the same one)")
    ap.add_argument("--model-id", default=None,
                    help="override the model id. Only valid for a single-candidate run; "
                         "the historical candidate_id is normally the model id.")
    ap.add_argument("--target", default="direction",
                    help="label column these models fit (default: direction)")
    ap.add_argument("--link", action="store_true",
                    help="hard-link weight/bundle files instead of copying them. "
                         "One copy of the bytes, two valid paths -- only safe when "
                         "the source and research/runs/ are on one filesystem.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files already present in the destination")
    args = ap.parse_args()

    src = Path(args.src) if args.src else custom_test_dir() / args.objective / args.run_ts
    dest = emit_canonical_run(
        src,
        objective_id=args.objective,
        run_ts=args.run_ts,
        dest_run_ts=args.dest_run_ts,
        target=args.target,
        model_id_override=args.model_id,
        force=args.force,
        link_weights=args.link,
    )
    return 0 if dest is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
