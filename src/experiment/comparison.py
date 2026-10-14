"""Compare two RunManifests conservatively.

The question a comparison must answer is not "what differs?" but "**may I
attribute the difference in results to the difference in setup?**" Those are
not the same question, and answering the first while implying the second is
how a research programme talks itself into a false causal claim.

Why this module is deliberately pessimistic
-------------------------------------------
An earlier version set ``causally_attributable = (exactly one
responsibility_versions entry differs)``. That is unsound: two runs can differ
in exactly one responsibility while *also* sitting on different datasets,
different fold schedules, different seeds, different cost assumptions and
different code. It would have certified such a pair as causally attributable.

This version:

  * compares **every** scientific dimension, not just responsibilities;
  * treats an *absent* record as ``missing_evidence``, never as "same" --
    the repo's runs frequently omit seeds, dataset hashes and cost
    assumptions entirely (``docs/FUNCTIONAL_COVERAGE.md`` §11), and silence
    is not agreement;
  * reports ``controlled_comparison`` only when exactly one dimension varies
    **and** nothing material is unrecorded.

``controlled_comparison=False`` is the safe default and the common case. That
is the intended behaviour, not a limitation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.experiment.manifest import RunManifest

#: Dimensions that must be *proven equal* for a comparison to be controlled.
#: Order is presentation order.
COMPARED_DIMENSIONS = (
    "objective_id",
    "responsibility_versions",
    "config_snapshot",
    "datasets",
    "validation",
    "seeds",
    "code",
    "execution_assumptions",
    "decision_policy",
    "calibration",
)

#: Dimensions whose absence in BOTH manifests is unrecorded evidence rather
#: than agreement. Comparing two runs that both forgot to record their cost
#: assumptions does not establish that their cost assumptions matched.
#:
#: This must cover **every** compared dimension that can legitimately be
#: empty, or the "silence is not agreement" rule is enforced for some
#: dimensions and silently violated for the rest. An adversarial review found
#: exactly that hole: with only the first four entries present, two runs using
#: different decision-policy thresholds but both leaving ``decision_policy={}``
#: produced no leaf diff, so the dimension was reported as NOT varying, and a
#: single unrelated difference elsewhere was certified as an attributable
#: cause. The remaining three entries close it.
#:
#: The other two compared dimensions (``objective_id``,
#: ``responsibility_versions``) are absent from this tuple because
#: ``RunManifest.__post_init__`` already refuses to construct a manifest in
#: which they are empty.
#:
#: A run that genuinely performed no calibration records that fact --
#: ``calibration={"method": "none"}`` -- rather than leaving the field empty.
#: That is a meaningful record, not an empty placeholder artifact: "we did not
#: calibrate" and "we did not write down whether we calibrated" are different
#: claims, and only the first can support a controlled comparison.
EVIDENCE_REQUIRED = (
    "datasets", "validation", "seeds", "execution_assumptions",
    "config_snapshot", "decision_policy", "calibration",
    # ``code`` may now be None: an unavailable git identity is recorded as
    # absent rather than fabricated as "unknown". It must therefore be
    # evidence-required, or two runs that both lack code identity would read
    # as agreeing on it -- the exact hole closed above for decision_policy.
    "code",
)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively flatten mappings/sequences to leaf paths.

    Top-level-only comparison (the previous behaviour) reports
    ``config_snapshot`` as "different" without saying which key moved, which
    is useless for attribution and hides single-key changes inside large
    nested blocks.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _dimension_value(manifest: RunManifest, dimension: str) -> Any:
    from dataclasses import asdict, is_dataclass

    value = getattr(manifest, dimension, None)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [asdict(v) if is_dataclass(v) and not isinstance(v, type) else v for v in value]
    return value


def _is_empty(value: Any) -> bool:
    return value is None or value == {} or value == [] or value == ()


@dataclass(frozen=True)
class ComparisonReport:
    """Result of comparing two runs' manifests.

    Attributes:
        run_id_a, run_id_b: The two runs compared.
        differences: dimension -> flattened leaf path -> ``(value_in_a,
            value_in_b)`` for every leaf that differs.
        uncontrolled_dimensions: Dimensions that differ. When more than one
            is present, no single-cause attribution is supportable.
        missing_evidence: Dimensions not recorded well enough to judge.
            These are neither "same" nor "different" -- they are unknown.
        controlled_comparison: True only when exactly one dimension differs
            and ``missing_evidence`` is empty.
        warning: Plain-language explanation of what the comparison does and
            does not license. Empty only for a controlled comparison.
    """

    run_id_a: str
    run_id_b: str
    differences: Mapping[str, Mapping[str, tuple[Any, Any]]] = field(default_factory=dict)
    uncontrolled_dimensions: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    controlled_comparison: bool = False
    warning: str = ""

    @property
    def attributable_dimension(self) -> str | None:
        """The single varying dimension, when the comparison is controlled."""
        if self.controlled_comparison and len(self.uncontrolled_dimensions) == 1:
            return self.uncontrolled_dimensions[0]
        return None


def compare_runs(a: RunManifest, b: RunManifest) -> ComparisonReport:
    """Compare two manifests across every scientific dimension."""
    differences: dict[str, dict[str, tuple[Any, Any]]] = {}
    varying: list[str] = []
    missing: list[str] = []

    for dim in COMPARED_DIMENSIONS:
        va, vb = _dimension_value(a, dim), _dimension_value(b, dim)

        if dim in EVIDENCE_REQUIRED and _is_empty(va) and _is_empty(vb):
            missing.append(dim)
            continue

        fa, fb = _flatten(va), _flatten(vb)
        leaves = sorted(set(fa) | set(fb))
        dim_diff = {
            leaf: (fa.get(leaf), fb.get(leaf))
            for leaf in leaves
            if fa.get(leaf) != fb.get(leaf)
        }
        if dim_diff:
            differences[dim] = dim_diff
            varying.append(dim)

    controlled = len(varying) == 1 and not missing

    if controlled:
        warning = ""
    elif missing and len(varying) <= 1:
        warning = (
            f"NOT a controlled comparison: {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not recorded in either run, "
            f"so {'it' if len(missing) == 1 else 'they'} cannot be shown to have "
            f"been held fixed. An unrecorded dimension is unknown, not identical. "
            f"Do not attribute any metric difference to "
            f"{varying[0] if varying else 'these runs'} until "
            f"{'it is' if len(missing) == 1 else 'they are'} recorded."
        )
    elif len(varying) > 1:
        warning = (
            f"NOT a controlled comparison: {len(varying)} dimensions differ "
            f"simultaneously ({', '.join(varying)})"
            + (f", and {', '.join(missing)} are unrecorded" if missing else "")
            + ". No single-cause claim (\"X improved Y\") is supportable. "
              "Re-run varying exactly one dimension."
        )
    else:
        warning = (
            "No recorded dimension differs between these runs. Any metric "
            "difference is therefore unexplained by the manifests and should be "
            "investigated as nondeterminism or as an unrecorded input -- note "
            "that seeds are not currently recorded by any existing run path."
        )

    return ComparisonReport(
        run_id_a=a.run_id,
        run_id_b=b.run_id,
        differences=differences,
        uncontrolled_dimensions=tuple(varying),
        missing_evidence=tuple(missing),
        controlled_comparison=controlled,
        warning=warning,
    )


# --- result comparison -------------------------------------------------------
#
# Setup comparison (above) answers "may I attribute a difference to a change?".
# Result comparison answers "what actually differed?". They are reported
# separately and deliberately never merged: a result difference between two
# runs whose setup comparison is uncontrolled is an observation, not a finding.

#: Which result artifacts are compared, and when each is expected, now come
#: from the single canonical registry (``artifacts.ARTIFACTS``) instead of a
#: parallel list here. Only small JSON artifacts are diffed leaf-by-leaf.
def _result_specs():
    from src.experiment.artifacts import ARTIFACTS
    return tuple(a for a in ARTIFACTS if a.comparable)


@dataclass(frozen=True)
class ResultComparison:
    """Differences between two runs' persisted result artifacts.

    Attributes:
        differences: artifact -> leaf path -> ``(value_in_a, value_in_b)``.
        one_sided: artifacts present in exactly one run (artifact -> "a"/"b").
        missing_expected: artifacts a run's own manifest implies it should
            have produced but did not, as ``(artifact, "a"|"b")``.
        invalid_evidence: ``(artifact, side, error)`` for artifacts that
            exist but could not be parsed. Reported, never skipped.
        not_applicable: artifacts neither run was expected to produce. Listed
            so their absence is visibly a non-finding.
    """

    differences: Mapping[str, Mapping[str, tuple[Any, Any]]] = field(default_factory=dict)
    one_sided: Mapping[str, str] = field(default_factory=dict)
    missing_expected: tuple[tuple[str, str], ...] = ()
    invalid_evidence: tuple[tuple[str, str, str], ...] = ()
    not_applicable: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        """True only when nothing differs AND every artifact was readable.

        A comparison with unreadable evidence is never clean, however few
        differences it managed to compute.
        """
        return not (self.differences or self.one_sided
                    or self.missing_expected or self.invalid_evidence)


def compare_results(
    dir_a, dir_b, manifest_a: RunManifest, manifest_b: RunManifest
) -> ResultComparison:
    """Compare two runs' result artifacts. Read-only; never writes.

    Every artifact that EXISTS is read and content-validated first, on each
    side independently. Only then is presence compared. The earlier version
    returned at "present in only one run" before reading the file that did
    exist, so a corrupt artifact paired with an absent one was invisible.
    """
    import json
    from pathlib import Path

    from src.experiment.artifacts import validate_content

    dirs = {"a": Path(dir_a), "b": Path(dir_b)}
    manifests = {"a": manifest_a, "b": manifest_b}
    differences: dict[str, dict[str, tuple[Any, Any]]] = {}
    one_sided: dict[str, str] = {}
    missing: list[tuple[str, str]] = []
    invalid: list[tuple[str, str, str]] = []
    n_a: list[str] = []

    for spec in _result_specs():
        name = spec.filename
        expected = {k: bool(spec.applies(m)) for k, m in manifests.items()}
        present = {k: (d / name).is_file() for k, d in dirs.items()}

        if not any(expected.values()) and not any(present.values()):
            n_a.append(name)
            continue

        parsed: dict[str, Any] = {}
        for side in ("a", "b"):
            if not present[side]:
                if expected[side]:
                    missing.append((name, side))
                continue
            try:
                payload = json.loads((dirs[side] / name).read_text())
            except (OSError, ValueError) as exc:
                invalid.append((name, side, f"{type(exc).__name__}: {exc}"))
                continue
            gaps = validate_content(spec, payload)
            if gaps:
                invalid.append((name, side, f"missing required content {gaps}"))
                continue
            parsed[side] = payload

        if present["a"] != present["b"]:
            one_sided[name] = "a" if present["a"] else "b"
            continue
        if len(parsed) != 2:
            continue
        fa, fb = _flatten(parsed["a"]), _flatten(parsed["b"])
        diff = {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb))
                if fa.get(k) != fb.get(k)}
        if diff:
            differences[name] = diff

    return ResultComparison(
        differences=differences,
        one_sided=one_sided,
        missing_expected=tuple(missing),
        invalid_evidence=tuple(invalid),
        not_applicable=tuple(n_a),
    )
