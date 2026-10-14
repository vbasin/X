"""Reusable agent skills — parameterized procedures (component C6, in-codebase).

A *skill* is a named playbook an agent fetches (via the `get_skill` tool) and follows when
designing an experiment. Skills encode the repo's hard-won procedure so the two agents do
not reinvent — or mis-apply — it each cycle. They are content, not code paths: cheap,
versionable, and surfaced through the tool registry like any other tool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Skill:
    name: str
    when_to_use: str
    steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SKILLS: dict[str, Skill] = {
    "design_walkforward": Skill(
        name="design_walkforward",
        when_to_use="Designing any experiment whose result must generalize across regimes.",
        steps=[
            "Use expanding-window walk-forward folds; never a single random split (leakage + regime luck).",
            "Keep the SAME held-out test composition as the baseline so metric deltas are apples-to-apples.",
            "Require improvement on >=2 of 3 folds before claiming success (one lucky fold is not signal — cf. v010 fold-1).",
            "Report precision@tau and BU/BD F1 on the test set, not just val f1_macro.",
            "Fit the scaler on train only; never refit on the test window.",
        ],
    ),
    "feature_group_ablation": Skill(
        name="feature_group_ablation",
        when_to_use="Deciding whether a feature group adds signal, or should be toggled off.",
        steps=[
            "Read read_attribution: low-importance groups are removal candidates; groups absent vs FEATURE_CATALOG are addition candidates.",
            "Toggle exactly ONE group via features.enabled_groups in config_diff (scope=feature) so the effect is attributable.",
            "Hold every other knob at baseline; same folds, same seed.",
            "Success = primary metric improves OR holds while the model gets simpler (simplicity policy favours removal).",
        ],
    ),
    "audit_calibration": Skill(
        name="audit_calibration",
        when_to_use="precision@tau is the objective and you suspect mis-calibration at the tau thresholds.",
        steps=[
            "Inspect precision@tau at tau in {0.5, 0.8, 0.9} for BIG_UP and BIG_DOWN from training_result.json.",
            "If precision collapses at high tau with few signals, propose temperature scaling / isotonic calibration (scope=training) — not a new architecture.",
            "Confirm n_signals per direction is large enough (>50) to trust the precision estimate before acting.",
        ],
    ),
    "estimate_cost": Skill(
        name="estimate_cost",
        when_to_use="Filling est_cost_usd on an experiment spec.",
        steps=[
            "OCI A10 GPU ~= $0.25-0.30/hr; one walk-forward fold ~= 1-3 hr depending on epochs.",
            "est_cost_usd ~= folds * (est_epochs / 30) * 2.0 USD as a rough floor; keep under the per-spec cost cap.",
            "Prefer CONFIG_DIFF (cheap) over NEW_COMPONENT (needs implementation + longer runs) when both test the hypothesis.",
        ],
    ),
}


def get_skill(name: str) -> dict[str, Any]:
    """Return a named skill playbook, or the list of available skills if unknown."""
    skill = SKILLS.get(name)
    if skill is None:
        return {"error": f"unknown skill '{name}'", "available": sorted(SKILLS)}
    return skill.to_dict()
