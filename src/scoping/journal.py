"""T3 proven-live journal — the highest-trust evidence tier (ADR-025 §2).

One JSONL file per objective at ``data/evaluation/proven_journal/<objective_id>.jsonl``.
Entries record VALIDATED live-run results (model config, live period, realized metrics)
anchored to an ``mlflow_run_id`` for lineage.

**Write gate**: entries are added ONLY through the CLI's interactive confirmation — the
operator sees the full entry and must answer ``y``. Agents (and any code path) calling
:func:`add_entry` without ``confirmed=True`` get ``PermissionError``. This guards the
tier that outranks everything else (T3 > T2 > T1) against self-reinforcing noise and
memory poisoning. There is deliberately NO ``--yes`` flag.

CLI::

    python -m src.scoping.journal add --objective nq-50pt-tbm --from-run <mlflow_run_id> \\
        --live-start 2026-06-01 --live-end 2026-07-01 \\
        --metric precision_at_tau=0.86 --metric n_signals=41 [--notes "..."]
    python -m src.scoping.journal list --objective nq-50pt-tbm
"""

from __future__ import annotations

from src.common.data_paths import data_root, evaluation_dir

import argparse
import getpass
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vtech.scoping.journal")

from src.common.config_paths import schema_path as _schema_path
_SCHEMA_PATH = _schema_path("proven_journal.schema.json")


def _journal_dir() -> Path:
    return evaluation_dir() / "proven_journal"


def journal_path(objective_id: str) -> Path:
    return _journal_dir() / f"{objective_id}.jsonl"


def load_journal(objective_id: str) -> list[dict[str, Any]]:
    """All T3 entries for an objective (read side — ungated, agents use this freely)."""
    p = journal_path(objective_id)
    if not p.exists():
        return []
    entries: list[dict[str, Any]] = []
    for i, line in enumerate(p.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("journal_bad_line file=%s line=%d err=%s", p, i, exc)
    return entries


def load_proven_evidence(objective_id: str) -> list[dict[str, Any]]:
    """The internal evidence Pass B is allowed to integrate: user-confirmed PROVEN-SUCCESS
    entries only (ADR-025 T3), gathered from this objective's own journal PLUS sibling
    objectives that share the same base instrument key.

    Rationale (2026-07-15 directive): a proven-success model recorded for one objective is
    legitimately informative for another that trades the same instrument family — cross-
    objective base-key sharing qualifies. What does NOT qualify is un-confirmed experimental
    data (a Scoping-stage "champion" from `results_index`/leaderboard/MLflow is a starting
    base model, not a finalized success). So Pass B sources this and only this; when it is
    empty, Pass B has no legitimate internal evidence and its integration step is skipped.

    Each returned entry is tagged ``_match`` ('exact' for this objective, 'base' for a
    base-key sibling) and carries its originating ``objective_id``."""
    entries: list[dict[str, Any]] = []
    for e in load_journal(objective_id):
        e["_match"] = "exact"
        e.setdefault("objective_id", objective_id)
        entries.append(e)

    try:
        from src.objectives.loader import load_objective
        from src.search.results_index import base_key
        target_base = base_key(load_objective(objective_id))
    except Exception as exc:  # noqa: BLE001 — cannot resolve siblings → own journal only
        logger.debug("proven_evidence_sibling_scan_skipped objective=%s err=%s", objective_id, exc)
        return entries

    d = _journal_dir()
    if not d.exists():
        return entries
    for p in sorted(d.glob("*.jsonl")):
        sibling = p.stem
        if sibling == objective_id:
            continue
        try:
            if base_key(load_objective(sibling)) != target_base:
                continue
        except Exception:  # noqa: BLE001 — unknown/legacy objective id → skip
            continue
        for e in load_journal(sibling):
            e["_match"] = "base"
            e.setdefault("objective_id", sibling)
            entries.append(e)
    return entries


def add_entry(objective_id: str, entry: dict[str, Any], *, confirmed: bool = False) -> Path:
    """Append one schema-validated entry. ``confirmed=True`` is set ONLY by the CLI after
    the operator's interactive 'y' — any other caller is refused."""
    if not confirmed:
        raise PermissionError(
            "T3 proven-journal writes require interactive user confirmation — "
            "use `python -m src.scoping.journal add` (ADR-025 write gate)")
    import jsonschema
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(entry, schema)

    p = journal_path(objective_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.info("proven_journal_entry_added objective=%s run=%s", objective_id, entry.get("mlflow_run_id"))
    return p


# ── CLI ───────────────────────────────────────────────────────────────────────

def _entry_from_run(objective_id: str, run_id: str) -> dict[str, Any]:
    """Pull family/params lineage for the entry from the MLflow run (best-effort)."""
    model: dict[str, Any] = {"family": "unknown"}
    try:
        import mlflow
        from src.training.config import MLflowConfig
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLflowConfig().tracking_uri))
        run = mlflow.get_run(run_id)
        tags, params = run.data.tags, run.data.params
        model = {
            "family": tags.get("family") or params.get("family") or "unknown",
            "candidate_id": tags.get("candidate_id") or "",
            "params": {k[2:]: v for k, v in params.items() if k.startswith("p_")},
        }
        model = {k: v for k, v in model.items() if v}
    except Exception as exc:  # noqa: BLE001 — lineage enrichment is best-effort
        logger.warning("mlflow_lookup_failed run=%s err=%s — enter model manually if needed", run_id, exc)
    return {"objective_id": objective_id, "mlflow_run_id": run_id, "model": model}


def _parse_metric(kv: str) -> tuple[str, Any]:
    k, _, v = kv.partition("=")
    try:
        return k, float(v)
    except ValueError:
        return k, v


def _cli() -> None:
    p = argparse.ArgumentParser(description="T3 proven-live journal (ADR-025, user-gated writes)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Add one PROVEN entry (interactive confirmation required)")
    a.add_argument("--objective", required=True)
    a.add_argument("--from-run", required=True, dest="run_id", help="MLflow run id (lineage anchor)")
    a.add_argument("--live-start", required=True)
    a.add_argument("--live-end", required=True)
    a.add_argument("--metric", action="append", default=[], metavar="NAME=VALUE",
                   help="Realized live metric; repeatable (at least one required)")
    a.add_argument("--rationale", required=True,
                   help="WHY this model is a proven success — the criteria it met and why it "
                        "generalizes. Recorded so downstream scoping can reuse the result "
                        "without re-litigating it (unless the objective's requirements differ).")
    a.add_argument("--notes", default="")

    ls = sub.add_parser("list", help="Show the journal for an objective")
    ls.add_argument("--objective", required=True)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if args.cmd == "list":
        entries = load_journal(args.objective)
        if not entries:
            print(f"(no proven entries for {args.objective})")
            return
        for e in entries:
            m = e.get("model", {})
            lp = e.get("live_period", {})
            print(f"• {m.get('family')} ({m.get('candidate_id') or '-'})  "
                  f"live {lp.get('start')}..{lp.get('end')}  "
                  f"{json.dumps(e.get('realized_metrics'))}  run={e.get('mlflow_run_id')[:12]}")
        return

    if not args.metric:
        p.error("at least one --metric NAME=VALUE is required")
    entry = _entry_from_run(args.objective, args.run_id)
    entry["live_period"] = {"start": args.live_start, "end": args.live_end}
    entry["realized_metrics"] = dict(_parse_metric(m) for m in args.metric)
    entry["success_rationale"] = args.rationale
    if args.notes:
        entry["notes"] = args.notes
    entry["added_by"] = getpass.getuser()
    entry["user_confirmed_at"] = datetime.now(timezone.utc).isoformat()

    print("\n── PROVEN journal entry (T3 — outranks ALL other evidence) ──")
    print(json.dumps(entry, indent=2, default=str))
    if not sys.stdin.isatty():
        print("\n✗ refused: stdin is not a TTY — T3 writes require a live interactive confirmation.")
        sys.exit(2)
    ans = input("\nWrite this entry? Only validated LIVE results belong here. [y/N]: ").strip().lower()
    if ans != "y":
        print("aborted — nothing written.")
        sys.exit(1)
    path = add_entry(args.objective, entry, confirmed=True)
    print(f"✅ appended to {path}")


if __name__ == "__main__":
    _cli()
