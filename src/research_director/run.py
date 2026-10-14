"""Production entrypoint for the Research Director loop.

Wires config → the cycle driver and runs ONE model-improvement cycle (systemd-timer driven,
mirroring vtech-trainer). The cognitive work is the Scoping-v2-style loop in
``src.research_director.orchestrator`` (diagnose → cross-review → synthesize → auto-validate); this
module only assembles the Governor, Journal, MLflow tracker, and the loop config, then hands off
to :class:`src.research_director.director.ResearchDirector`.

Usage:
  python -m src.research_director.run                 # one live research cycle (spends LLM $)
  python -m src.research_director.run --dry-run       # validate config + print wiring plan
  python -m src.research_director.run --dry-run-loop  # offline full loop (no LLM/GPU spend)
"""

from __future__ import annotations

import argparse
import json
import logging
import os

# systemd's EnvironmentFile does NOT expand ${VAR} references — lines like
# ``ANTHROPIC_API_KEY=${EXAMPLE_API_KEY}`` in /opt/vtech/.env arrive at this process
# as the literal 17-char string ``${EXAMPLE_API_KEY}`` and Anthropic returns 401
# ("invalid x-api-key"). python-dotenv DOES expand ${VAR} references (in file order),
# so we re-load .env here with override=True to resolve those placeholders before any
# LLM client reads os.environ. Safe for manual/CLI invocations too.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.environ.get("VTECH_ENV_FILE", "/opt/vtech/.env"), override=True)
except Exception:
    pass

from src.research_director.config import RDConfig

log = logging.getLogger("research_director.run")


def _loop_cfg(cfg: RDConfig) -> dict:
    """The orchestrator's rd_loop config, with the driver's program context injected so the
    per-lens research/critic registries build correctly."""
    from src.research_director.orchestrator import load_rd_cfg
    lc = load_rd_cfg()
    lc.setdefault("registry_dir", cfg.registry_dir)
    lc.setdefault("model_name", cfg.model_name)
    lc.setdefault("external_search_enabled", cfg.external_search_enabled)
    lc.setdefault("objective_id", cfg.objective_id)
    return lc


def build_director(cfg: RDConfig, *, dry_run: bool = False):
    """Wire the cycle driver from config. Connects to the DB (Journal/Governor) + ensures the
    pgvector agent-memory schema exists (skipped for pure dry-run-loop)."""
    from src.research_director.director import ResearchDirector
    from src.research_director.mlflow_tracking import MLflowTracker

    journal = None
    governor = None
    if not dry_run:
        from src.research_director import rag_memory
        from src.research_director.governor import Governor
        from src.research_director.journal import Journal

        # Per-lens memory is a shared pgvector store (research_director.agent_memory).
        rag_memory.ensure_schema()
        journal = Journal()
        governor = Governor(
            min_gap_s=cfg.gov_min_gap_s,
            act_cap_24h_s=cfg.gov_act_cap_24h_s,
            usd_cap_day=cfg.gov_usd_cap_day,
            usd_cap_week=cfg.gov_usd_cap_week,
        )
    else:
        # dry-run-loop: no journal writes; stub journal for type surface
        class _NoJournal:
            def recent_signatures(self, *_a, **_k):
                return []

            def create_experiment(self, **_k):
                return 0

            def transition(self, *_a, **_k):
                return None

        journal = _NoJournal()

    tracker = MLflowTracker(
        enabled=cfg.mlflow_enabled and not dry_run,
        tracking_uri=cfg.mlflow_tracking_uri,
        experiment_name=cfg.mlflow_experiment_name,
    )
    return ResearchDirector(
        program_id=cfg.program_id,
        objective_id=cfg.objective_id,
        program_objectives=cfg.program_objectives,
        registry_dir=cfg.registry_dir,
        model_name=cfg.model_name,
        journal=journal,
        governor=governor,
        tracker=tracker,
        loop_cfg=_loop_cfg(cfg),
        dry_run=dry_run,
    )


def _cfg_for_objective(base: RDConfig, objective_id: str) -> RDConfig:
    """Clone driver config with a different objective (same caps/registry)."""
    from dataclasses import replace
    return replace(
        base,
        objective_id=objective_id,
        program_id=objective_id,
        program_objectives={
            **(base.program_objectives or {}),
            "objectives": [objective_id],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Research Director — one or more model-improvement cycles")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate config + list RD targets; no LLM/DB/GPU contacted",
    )
    parser.add_argument(
        "--dry-run-loop", action="store_true",
        help="run full orchestrator stages offline for each enabled target",
    )
    parser.add_argument(
        "--objective", default=None,
        help="run only this objective_id (must be RD-enabled + have registry LATEST)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from src.research_director.rd_enable import list_rd_targets, status_for_objective

    cfg = RDConfig.load()
    targets = list_rd_targets()
    if args.objective:
        st = status_for_objective(args.objective)
        if not st["has_registry_latest"]:
            log.error("objective %s has no registry LATEST — promote first", args.objective)
            return 1
        if not st["rd_enabled"]:
            log.error("objective %s RD disabled — enable in WebUI Objectives tab", args.objective)
            return 1
        targets = [t for t in targets if t["objective_id"] == args.objective] or [{
            "objective_id": args.objective,
            "version": st.get("registry_version"),
            "path": st.get("registry_path"),
        }]

    if args.dry_run and not args.dry_run_loop:
        lc = _loop_cfg(cfg)
        print(json.dumps(
            {
                "rd_targets": targets,
                "n_targets": len(targets),
                "config_objective_id_pin": cfg.objective_id,
                "note": (
                    "Each timer fire runs one cycle per RD-enabled objective that has "
                    "registry LATEST. Enable/disable in WebUI Objectives tab."
                ),
                "workflow": "diagnose → cross_review → synthesis → validate → handoff",
                "fields": lc.get("fields"),
                "caps": lc.get("caps"),
                "providers": lc.get("providers"),
                "cost": lc.get("cost"),
                "registry_dir": cfg.registry_dir,
                "governor": {
                    "min_gap_s": cfg.gov_min_gap_s,
                    "act_cap_24h_s": cfg.gov_act_cap_24h_s,
                    "usd_cap_day": cfg.gov_usd_cap_day,
                    "usd_cap_week": cfg.gov_usd_cap_week,
                },
            },
            indent=2,
            default=str,
        ))
        print("dry-run OK — config valid; no GPU/LLM/DB contacted")
        return 0

    if not targets:
        log.warning(
            "no RD targets (need registry LATEST + rd_enabled); nothing to do")
        print(json.dumps({"ran": False, "reason": "no_rd_targets", "reports": []}, indent=2))
        return 0

    reports: list[dict] = []
    dry = bool(args.dry_run_loop)
    for t in targets:
        oid = t["objective_id"]
        ocfg = _cfg_for_objective(cfg, oid)
        log.info("rd_cycle_start objective=%s version=%s dry=%s", oid, t.get("version"), dry)
        director = build_director(ocfg, dry_run=dry)
        report = director.run_cycle(dry_run=dry)
        rec = {
            "objective_id": oid,
            "version": t.get("version"),
            "ran": report.ran,
            "reason": report.reason,
            "verdict": report.verdict,
            "n_changes": report.n_changes,
            "experiment_id": report.experiment_id,
        }
        reports.append(rec)
        log.info("rd_cycle_done %s", rec)

    print(json.dumps({"n": len(reports), "reports": reports}, indent=2, default=str))
    if dry:
        print("dry-run-loop OK — offline stages complete; no LLM/GPU spend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
