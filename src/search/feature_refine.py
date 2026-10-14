"""Agent-driven feature add/remove refinement — Phase 23 entry path 2 closure.

The USER-OVERRIDE spec (ADR-019) lets the user pin the model(s) and an initial feature
set; this module closes the second half of that entry path: **the agent then adds/removes
features through downstream analysis**. Each iteration:

1. runs the pinned candidate at full walk-forward fidelity (:func:`run_search` — fold
   building, GPU dispatch, scoring, gate all inherited),
2. builds **outcome episodes** for the run with the *entire* feature matrix as context
   and mines the per-feature win/loss + caught/missed separations
   (:mod:`src.evaluation.patterns`),
3. proposes a bounded mutation — **add** features that separate outcomes but are invisible
   to the model, **remove** features that show no outcome separation (dead weight). The
   proposer is deterministic by default; ``--llm`` lets the provider-agnostic LLM
   (``src.common.llm_factory``) interpret the pattern report instead — clamped to the
   available columns and falling back to the deterministic proposal on any failure
   (LLM proposes, controller disposes),
4. re-runs and **keeps the mutation only if the objective KPI improves** (target-met >
   requirement-met > best per-direction precision@primary-tau with a signal floor),
5. stops on target met, ``--max-iters``, or ``--patience`` consecutive non-improvements.

The winner is written back as a ready-to-run manual spec.

Usage::

    python -m src.search.feature_refine --objective nq-50pt \
        --manual-spec config/my_models_run.yaml [--candidate xgb_custom] \
        [--max-iters 6] [--max-add 4] [--max-remove 4] [--llm] [--dry-run]
"""

from __future__ import annotations

from src.common.data_paths import data_root, search_dir

import argparse
import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger("vtech.search.feature_refine")

_REPO_ROOT = Path(__file__).resolve().parents[2]

_MIN_FEATURES = 5          # never shrink below this
_ADD_D = 0.20              # |Cohen's d| a feature must show to be worth adding
_REMOVE_D = 0.05           # |Cohen's d| below which a feature is dead weight


def _data_root() -> Path:
    return Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))


# ── ranking (which run is better) ─────────────────────────────────────────────

def _rank_key(res: dict[str, Any], objective: dict[str, Any]) -> tuple:
    """Comparable goodness tuple for one candidate result dict.

    Order: target met > requirement met > best per-direction precision@primary-tau
    (with the objective's min-signal floor) > directional f1. NaNs rank lowest.
    """
    from src.evaluation.kpi_criteria import (
        DEFAULT_MIN_SIGNALS, meets_requirement, meets_target, primary_tau)

    diag = res.get("diagnostics")
    score = res.get("score") or {}
    tgt, _ = meets_target(objective, diag, score)
    req, _ = meets_requirement(objective, diag, score)

    tau = primary_tau(objective)
    best_prec = float("-inf")
    for e in (diag or {}).get("precision_at_tau_sweep", []):
        if tau is not None and abs(float(e.get("tau", -1)) - tau) < 1e-9 and \
                int(e.get("n_signals", 0)) >= DEFAULT_MIN_SIGNALS:
            p = e.get("precision")
            if p is not None and not (isinstance(p, float) and math.isnan(p)):
                best_prec = max(best_prec, float(p))
    if best_prec == float("-inf"):
        p = score.get("precision_at_tau")
        best_prec = float(p) if isinstance(p, (int, float)) and not math.isnan(p) else -1.0

    f1 = score.get("directional_f1") or 0.0
    return (bool(tgt), bool(req), best_prec, float(f1))


# ── proposers ─────────────────────────────────────────────────────────────────

def propose_mutation(
    current: list[str],
    sep: dict[str, dict[str, float]],
    available: list[str],
    *,
    max_add: int = 4,
    max_remove: int = 4,
    min_features: int = _MIN_FEATURES,
) -> dict[str, Any]:
    """Deterministic add/remove proposal from the separation tables.

    * ADD — features outside the current set whose win-vs-loss OR caught-vs-missed
      |Cohen's d| is strongest (>= ``_ADD_D``): the outcomes separate on them but the
      model cannot see them.
    * REMOVE — current features whose |d| is below ``_REMOVE_D`` in BOTH tables:
      no evidence they relate to success, failure, or blindness.
    """
    cur = set(current)
    wl, cm = sep.get("win_vs_loss", {}), sep.get("caught_vs_missed", {})

    strength: dict[str, float] = {}
    for f in set(wl) | set(cm):
        strength[f] = max(abs(wl.get(f, 0.0)), abs(cm.get(f, 0.0)))

    adds = sorted(
        (f for f in strength if f not in cur and f in available and strength[f] >= _ADD_D),
        key=lambda f: strength[f], reverse=True)[:max_add]

    # Only features we actually have evidence for can be judged dead weight.
    judged = [f for f in current if f in strength]
    removable = sorted(
        (f for f in judged if strength[f] <= _REMOVE_D),
        key=lambda f: strength[f])
    room = max(0, len(cur) + len(adds) - min_features)
    removes = removable[:min(max_remove, room)]

    return {
        "add": adds, "remove": removes,
        "rationale": (
            f"deterministic: +{len(adds)} outcome-separating features the model can't see "
            f"(|d|>={_ADD_D}), -{len(removes)} dead-weight features (|d|<={_REMOVE_D} in "
            "both win/loss and caught/missed tables)"),
        "proposer": "deterministic",
    }


def propose_mutation_llm(
    current: list[str],
    report: dict[str, Any],
    available: list[str],
    *,
    max_add: int,
    max_remove: int,
    min_features: int = _MIN_FEATURES,
) -> dict[str, Any] | None:
    """LLM-interpreted proposal (provider-agnostic factory). Returns None on failure —
    the caller falls back to the deterministic proposer. Output is clamped: only real
    columns, caps enforced, floor respected (LLM proposes, controller disposes)."""
    try:
        from src.common.llm import make_llm
        llm = make_llm()
        schema = {
            "type": "object",
            "required": ["add", "remove", "rationale"],
            "properties": {
                "add":       {"type": "array", "items": {"type": "string"}},
                "remove":    {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
            },
        }
        prefixes: dict[str, int] = {}
        for c in available:
            p = c.split("_")[0] + "_" if "_" in c else c
            prefixes[p] = prefixes.get(p, 0) + 1
        user = (
            "Latest outcome-pattern report (deterministic win/loss/missed analysis):\n"
            + (report.get("summary_text") or "")[:4000]
            + "\n\nCurrent model feature set (" + str(len(current)) + "):\n"
            + json.dumps(sorted(current))
            + "\n\nAvailable feature columns by prefix group: " + json.dumps(prefixes)
            + "\n\nPropose feature mutations to fix the observed failure modes: "
            + f"add at most {max_add} available columns the model lacks that the report "
              "shows separate outcomes; remove at most "
            + f"{max_remove} current columns with no outcome evidence. Only exact column "
              "names that exist. Return JSON {add, remove, rationale}."
        )
        out = llm.chat_json(
            system=("You are a feature-selection analyst for a trading-model research "
                    "loop. You only propose; deterministic code validates and evaluates."),
            user=user, agent="feature_refine", schema=schema)
        cur = set(current)
        avail = set(available)
        adds = [f for f in out.get("add", []) if f in avail and f not in cur][:max_add]
        removes = [f for f in out.get("remove", []) if f in cur]
        room = max(0, len(cur) + len(adds) - min_features)
        removes = removes[:min(max_remove, room)]
        return {"add": adds, "remove": removes,
                "rationale": str(out.get("rationale", ""))[:500], "proposer": "llm"}
    except Exception as exc:  # noqa: BLE001 — any LLM failure → deterministic fallback
        logger.warning("llm_proposer_failed (%s) — falling back to deterministic", exc)
        return None


# ── main loop ─────────────────────────────────────────────────────────────────

def refine_features(
    objective_id: str,
    manual_spec: str | Path | dict[str, Any],
    *,
    candidate_id: str | None = None,
    max_iters: int = 6,
    max_add: int = 4,
    max_remove: int = 4,
    patience: int = 2,
    use_llm: bool = False,
    parquet_dir: str | Path | None = None,
    dry_run: bool = False,
    run_search_fn: Any = None,
) -> dict[str, Any]:
    """Iteratively add/remove features on a pinned manual candidate until the objective
    KPI stops improving (or the target is met). Returns the refine report dict."""
    from src.objectives.loader import load_objective
    from src.search.manual_spec import load_manual_spec

    if run_search_fn is None:
        from src.search.controller import run_search as run_search_fn

    objective = load_objective(objective_id)
    ms = load_manual_spec(manual_spec, objective_id=objective_id)
    raw = copy.deepcopy(ms.raw)
    cands = raw.get("candidates") or []
    pos = 0
    if candidate_id:
        ids = [c.get("id") or c["family"] for c in cands]
        if candidate_id not in ids:
            raise ValueError(f"--candidate '{candidate_id}' not in spec (have: {ids})")
        pos = ids.index(candidate_id)
    cand = cands[pos]
    cid = cand.get("id") or cand["family"]

    from src.features.stores import default_parquet_dir

    pdir = Path(parquet_dir) if parquet_dir else default_parquet_dir(objective)
    available = _available_columns(objective, pdir)
    current = _resolve_columns(cand.get("features"), available)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    from src.common.run_layout import new_run_dir
    out_dir = new_run_dir(objective_id, f"refine_{run_ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🧬 FEATURE REFINE — candidate={cid!r} objective={objective_id!r} "
          f"start_features={len(current)} avail={len(available)} "
          f"proposer={'llm+fallback' if use_llm else 'deterministic'}")

    history: list[dict[str, Any]] = []
    best_key: tuple | None = None
    best_features = list(current)
    best_result: dict[str, Any] | None = None
    no_improve = 0
    tested: set[frozenset] = set()
    t0 = time.time()

    for it in range(max_iters):
        tested.add(frozenset(current))
        spec_iter = _spec_with_columns(raw, pos, current)
        print(f"\n⏳ Iteration {it + 1}/{max_iters} — {len(current)} features…")
        summary = run_search_fn(
            objective_id, manual_spec=spec_iter, ensembles=False,
            parquet_dir=str(pdir), dry_run=dry_run)
        res = _extract_candidate(summary, cid)
        if res is None or res.get("status") != "ok":
            history.append({"iter": it, "n_features": len(current), "status": "failed",
                            "error": (res or {}).get("error")})
            print(f"❌ run failed — stopping ({(res or {}).get('error', 'no result')})")
            break

        res = _attach_diagnostics(res, summary)
        key = _rank_key(res, objective) if not dry_run else (False, False, -1.0, 0.0)
        entry = {
            "iter": it, "n_features": len(current), "features": sorted(current),
            "rank_key": list(key), "score": _slim(res.get("score")),
            "status": "ok",
        }

        improved = best_key is None or key > best_key
        if improved:
            best_key, best_features, best_result = key, list(current), res
            no_improve = 0
        else:
            no_improve += 1
            current = list(best_features)          # revert to the best set
        entry["kept"] = improved
        history.append(entry)
        print(f"{'✅ kept' if improved else '↩️  reverted'} — rank={list(key)} "
              f"(no_improve={no_improve})")

        if key[0]:                                  # target met — stop immediately
            print("🎯 objective target met — stopping")
            break
        if no_improve >= patience:
            print(f"🛑 no improvement for {patience} iterations — stopping")
            break
        if it == max_iters - 1:
            break

        # ── Analysis → mutation for the next iteration ─────────────────────────
        mutation = _analyze_and_propose(
            res, objective, pdir, best_features, available,
            max_add=max_add, max_remove=max_remove, use_llm=use_llm, dry_run=dry_run)
        entry["mutation"] = mutation
        if not mutation or not (mutation["add"] or mutation["remove"]):
            print("🛑 no further mutation proposed — stopping")
            break
        nxt = sorted((set(best_features) - set(mutation["remove"]))
                     | set(mutation["add"]))
        if frozenset(nxt) in tested:
            print("🛑 proposed feature set already tested — stopping")
            break
        current = nxt
        print(f"🧪 next: +{mutation['add']} -{mutation['remove']} "
              f"[{mutation['proposer']}]")

    target_met = bool(best_key and best_key[0])
    winner_path = None
    if best_result is not None:
        winner = _spec_with_columns(raw, pos, best_features)
        winner_path = out_dir / "winner_manual_spec.yaml"
        winner_path.write_text(yaml.safe_dump(winner, sort_keys=False))

    report = {
        "objective_id": objective_id, "candidate_id": cid, "run_ts": run_ts,
        "target_met": target_met, "n_iterations": len(history),
        "best_rank_key": list(best_key) if best_key else None,
        "best_n_features": len(best_features), "best_features": sorted(best_features),
        "elapsed_s": round(time.time() - t0, 1),
        "history": history,
        "winner_spec_path": str(winner_path) if winner_path else None,
    }
    (out_dir / "refine_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'🎯' if target_met else '🏁'} refine complete — best {len(best_features)} "
          f"features, report: {out_dir / 'refine_report.json'}")
    return report


# ── helpers ───────────────────────────────────────────────────────────────────

def _available_columns(objective: dict[str, Any], pdir: Path) -> list[str]:
    """All feature columns in the current matrix (fold cache first, schema peek fallback)."""
    try:
        from src.search.fold_cache import build_or_load_fold_arrays
        arrays = build_or_load_fold_arrays(objective, pdir)
        if arrays is not None:
            return list(arrays[3])
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow.parquet as pq
        from src.common.data_paths import list_feature_parquet_files

        sample = next(iter(list_feature_parquet_files(pdir)), None)
        skip = {"bar_end_ts", "__index_level_0__", "index"}
        if sample:
            return [c for c in pq.read_schema(sample).names
                    if not c.startswith("_") and not c.startswith("label") and c not in skip]
    except Exception:  # noqa: BLE001
        pass
    return []


def _resolve_columns(features: dict[str, Any] | None, available: list[str]) -> list[str]:
    """Resolve a manual-spec feature selection (view/prefixes/columns/None) to columns."""
    if not features:
        return list(available)
    if "columns" in features:
        return [c for c in features["columns"] if c in available] or list(features["columns"])
    if "prefixes" in features:
        return [c for c in available
                if any(c.startswith(p) for p in features["prefixes"])]
    if "view" in features:
        from src.models.feature_views import FEATURE_VIEWS
        prefixes = FEATURE_VIEWS.get(features["view"], [])
        return [c for c in available if any(c.startswith(p) for p in prefixes)]
    return list(available)


def _spec_with_columns(raw: dict[str, Any], pos: int, columns: list[str]) -> dict[str, Any]:
    spec = copy.deepcopy(raw)
    spec["candidates"] = [spec["candidates"][pos]]
    spec["candidates"][0]["features"] = {"columns": sorted(columns)}
    spec["ensembles"] = False
    return spec


def _extract_candidate(summary: dict[str, Any], cid: str) -> dict[str, Any] | None:
    for c in summary.get("candidates", []):
        if c.get("candidate_id") == cid:
            return dict(c)
    return None


def _attach_diagnostics(res: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Candidates in search_result are diagnostics-stripped — reload from disk."""
    if res.get("diagnostics"):
        return res
    oof_path = res.get("oof_path")
    if oof_path:
        p = Path(oof_path).parent / f"diagnostics_{res['candidate_id']}.json"
        if p.exists():
            try:
                res["diagnostics"] = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                pass
    return res


def _analyze_and_propose(
    res: dict[str, Any],
    objective: dict[str, Any],
    pdir: Path,
    current: list[str],
    available: list[str],
    *,
    max_add: int,
    max_remove: int,
    use_llm: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Episodes over ALL features → separation tables → mutation proposal."""
    if dry_run:
        return {"add": [], "remove": [], "rationale": "dry_run", "proposer": "none"}
    oof_path = res.get("oof_path")
    if not oof_path or not Path(oof_path).exists():
        return None
    from src.evaluation.episodes import build_episodes
    from src.evaluation.patterns import analyze_run, separation_table
    from src.search.fold_cache import build_or_load_fold_arrays

    arrays = build_or_load_fold_arrays(objective, pdir)
    if arrays is None:
        return None
    X, _, ts_arr, feature_cols = arrays
    oof = pd.read_parquet(oof_path)
    episodes = build_episodes(
        oof, X=X, ts=ts_arr, feature_cols=list(feature_cols),
        objective=objective, context_cols=list(feature_cols))   # ALL features as context
    if episodes is None or episodes.empty:
        return None
    sep = separation_table(episodes)

    if use_llm:
        report = analyze_run(objective["objective_id"], res["candidate_id"],
                             "refine", episodes)
        prop = propose_mutation_llm(current, report, available,
                                    max_add=max_add, max_remove=max_remove)
        if prop is not None:
            return prop
    return propose_mutation(current, sep, available,
                            max_add=max_add, max_remove=max_remove)


def _slim(score: dict[str, Any] | None) -> dict[str, Any]:
    if not score:
        return {}
    return {k: v for k, v in score.items()
            if k in ("precision_at_tau", "precision_up", "precision_down",
                     "directional_f1", "n_signals", "n_up", "n_down")}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Agent-driven feature add/remove refinement of a manual candidate")
    p.add_argument("--objective", required=True)
    p.add_argument("--manual-spec", required=True,
                   help="Manual candidates spec (config/manual_candidates.schema.json)")
    p.add_argument("--candidate", default="", help="Candidate id to refine (default: first)")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--max-add", type=int, default=4)
    p.add_argument("--max-remove", type=int, default=4)
    p.add_argument("--patience", type=int, default=2,
                   help="Stop after N consecutive non-improving iterations")
    p.add_argument("--llm", action="store_true",
                   help="Let the LLM interpret the pattern report (falls back to "
                        "deterministic on any failure)")
    p.add_argument("--parquet-dir", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    refine_features(
        args.objective, args.manual_spec,
        candidate_id=args.candidate or None,
        max_iters=args.max_iters, max_add=args.max_add, max_remove=args.max_remove,
        patience=args.patience, use_llm=args.llm,
        parquet_dir=args.parquet_dir or None, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
