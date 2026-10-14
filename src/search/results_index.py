"""Search results index — Phase 23 W4.

Records, per **objective trait key**, which model family/ensemble won (and which
merely passed the requirement), so future searches over *similar* objectives can
**warm-start** their candidate order instead of re-discovering it. Stored in the
shared object store (OCI Object Storage in prod, a local dir in tests) so it
persists across runs and is readable from both the CPU orchestrator and the GPU
runner.

Trait key (deterministic, human-browsable + collision-safe):

    ``<asset_class>.<symbol>.<security_type>.<hash10>``

where ``hash10`` is a short SHA-256 over the *full* trait tuple (all traded
products + sorted strategy keywords). The leading ``asset_class.symbol.security_type``
lets you list every run for e.g. ``equity_futures.NQ.futures`` (``match="base"``),
while the hash distinguishes exact trait sets (``match="exact"``).

Object layout:
    ``results_index/<trait_key>/<run_ts>.json``   one record per search run
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from src.common.object_store import ObjectStore, make_object_store

logger = logging.getLogger("vtech.search.results_index")

_PREFIX = "results_index"


# ── Trait key ─────────────────────────────────────────────────────────────────

def _primary_product(objective: dict[str, Any]) -> dict[str, Any]:
    prods = objective.get("traded products") or objective.get("traded_products") or []
    if not prods:
        return {}
    for p in prods:
        if str(p.get("role", "")).lower() == "primary":
            return p
    return prods[0]


def traits_of(objective: dict[str, Any]) -> dict[str, Any]:
    """Extract the trait fields the index keys on (products + strategy keywords)."""
    prods = objective.get("traded products") or objective.get("traded_products") or []
    return {
        "products": [
            {"symbol": p.get("symbol"), "security_type": p.get("security_type"),
             "asset_class": p.get("asset_class"), "role": p.get("role")}
            for p in prods
        ],
        "strategy_keywords": sorted(objective.get("strategy_keywords") or []),
    }


def trait_key(objective: dict[str, Any]) -> str:
    """Deterministic key grouping alike objectives; see module docstring."""
    prim = _primary_product(objective)
    base = ".".join(str(prim.get(k, "na")) for k in ("asset_class", "symbol", "security_type"))
    traits = traits_of(objective)
    full = json.dumps({"products": sorted(
        (json.dumps(p, sort_keys=True) for p in traits["products"])),
        "kw": traits["strategy_keywords"]}, sort_keys=True)
    h = hashlib.sha256(full.encode()).hexdigest()[:10]
    return f"{base}.{h}"


def base_key(objective: dict[str, Any]) -> str:
    """The leading ``asset_class.symbol.security_type`` (drops the trait hash)."""
    return trait_key(objective).rsplit(".", 1)[0]


# ── Index ─────────────────────────────────────────────────────────────────────

class ResultsIndex:
    """Reader/writer for the trait-keyed results index."""

    def __init__(self, *, local_dir: str | None = None, bucket: str | None = None,
                 prefix: str = _PREFIX) -> None:
        self._prefix = prefix.strip("/")
        self._store: ObjectStore
        self._store, self.backend_kind = make_object_store(
            local_dir=local_dir, local_env="VTECH_RESULTS_INDEX_LOCAL",
            bucket=bucket, bucket_env="OCI_BUCKET",
        )

    # ── write ──────────────────────────────────────────────────────────────
    def record_run(
        self,
        objective: dict[str, Any],
        gate_result: Any,           # src.search.gate.GateResult (duck-typed)
        results: list[Any],         # list[CandidateResult] (duck-typed)
        *,
        run_ts: str,
        git_sha: str | None = None,
    ) -> str:
        """Persist one search-run record under the objective's trait key. Returns the key."""
        from src.models.catalog import get_entry

        def _cost(fam: str) -> str:
            try:
                return get_entry(fam).cost_class
            except Exception:
                return "unknown"          # ensembles / non-catalogued names

        ok = [r for r in results if getattr(r, "status", None) == "ok" and getattr(r, "score", None)]
        passers = set(getattr(gate_result, "passed_ids", []) or [])

        def _entry(r: Any) -> dict[str, Any]:
            return {
                "candidate_id": getattr(r, "candidate_id", None),
                "family": getattr(r, "family", None),
                "cost_class": _cost(getattr(r, "family", "")),
                "score": getattr(r, "score", {}) or {},
                "meets_target": bool(getattr(r, "meets_target", False)),
                "passed": getattr(r, "candidate_id", None) in passers,
            }

        champ_id = getattr(gate_result, "champion_id", None)
        champ = next((r for r in ok if getattr(r, "candidate_id", None) == champ_id), None)
        record = {
            "objective_id": objective.get("objective_id"),
            "trait_key": trait_key(objective),
            "traits": traits_of(objective),
            "run_ts": run_ts,
            "git_sha": git_sha,
            "recorded_at": _utc(),
            "status": getattr(gate_result, "status", "unknown"),
            "champion": _entry(champ) if champ is not None else None,
            "champion_meets_target": bool(getattr(gate_result, "champion_meets_target", False)),
            "families": [_entry(r) for r in ok],
        }
        key = f"{self._prefix}/{trait_key(objective)}/{run_ts}.json"
        self._store.put(key, json.dumps(record, indent=2, default=str).encode())
        logger.info("results_index_recorded key=%s champion=%s status=%s",
                    key, champ_id, record["status"])
        return key

    # ── read ───────────────────────────────────────────────────────────────
    def list_records(self, objective: dict[str, Any], *, match: str = "base") -> list[dict[str, Any]]:
        """Return prior run records for this objective's traits.

        match="base"  → all runs sharing asset_class.symbol.security_type (broad warm-start)
        match="exact" → only runs with the identical trait set (products + keywords)
        """
        prefix = (f"{self._prefix}/{base_key(objective)}"
                  if match == "base" else f"{self._prefix}/{trait_key(objective)}")
        records: list[dict[str, Any]] = []
        for k in self._store.list(prefix):
            raw = self._store.get(k)
            if raw:
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        return sorted(records, key=lambda r: r.get("run_ts", ""))

    def query_priors(self, objective: dict[str, Any], *, match: str = "base") -> dict[str, Any]:
        """Aggregate prior records into warm-start priors for the Model Scout (W5).

        Returns win/pass counts per family, the best precision seen per family, a
        recommended try-order (by wins, then best precision), and whether the target
        was ever met for this trait group.
        """
        recs = self.list_records(objective, match=match)
        wins: dict[str, int] = {}
        passes: dict[str, int] = {}
        best: dict[str, dict[str, Any]] = {}
        ever_target = False

        for rec in recs:
            ever_target = ever_target or bool(rec.get("champion_meets_target"))
            champ = rec.get("champion") or {}
            cfam = champ.get("family")
            if cfam and rec.get("status") == "promoted":
                wins[cfam] = wins.get(cfam, 0) + 1
            for fam_entry in rec.get("families", []):
                fam = fam_entry.get("family")
                if not fam:
                    continue
                if fam_entry.get("passed"):
                    passes[fam] = passes.get(fam, 0) + 1
                prec = float((fam_entry.get("score") or {}).get("precision_at_tau") or 0.0)
                if fam not in best or prec > best[fam]["precision_at_tau"]:
                    best[fam] = {"precision_at_tau": prec, "run_ts": rec.get("run_ts"),
                                 "cost_class": fam_entry.get("cost_class")}

        order = sorted(
            best.keys(),
            key=lambda f: (wins.get(f, 0), passes.get(f, 0), best[f]["precision_at_tau"]),
            reverse=True,
        )
        return {
            "trait_key": trait_key(objective),
            "base_key": base_key(objective),
            "match": match,
            "n_runs": len(recs),
            "family_win_counts": wins,
            "family_pass_counts": passes,
            "best_by_family": best,
            "recommended_order": order,
            "ever_met_target": ever_target,
        }


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
