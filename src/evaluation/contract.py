"""Frozen Evaluation Contract — Phase 23 W2.

The contract is the single most important comparability guarantee of Phase 23:
every candidate model — GBM, LSTM, ensemble, regime model — is scored by the
**same** metric suite on the **same** pinned forward hold-out.  The contract
version is stamped on every leaderboard row; comparisons across contract
versions are forbidden.

Key guarantees
--------------
* A contract is created ONCE per objective and pinned by a SHA-256 hash of the
  hold-out parquet files.  It is written to
  ``data/evaluation/<objective_id>/contract_v<N>.json``.
* ``score()`` REFUSES to evaluate a prediction set whose data version hash
  differs from the contract's pinned hash.
* Adding a new contract version opens a new leaderboard generation automatically.

Usage::

    from src.evaluation.contract import EvalContract, build_contract

    # Create once (W2 setup step):
    contract = build_contract(
        objective_id="nq-50pt",
        hold_out_parquet_paths=[Path("sources/features/timeframe=5s/symbol=NQ/year=2026/month=02/features_5s_part1.parquet"), ...],
    )

    # Score a candidate:
    from src.evaluation._kpi import canonical_trade_kpi
    predictions = plugin.predict(fitted, X_test)
    result = contract.score(predictions, ground_truth, data_hash=current_hash)
"""

from __future__ import annotations

from src.common.data_paths import data_root, evaluation_dir, native_tf_dir

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation._kpi import canonical_trade_kpi

logger = logging.getLogger("vtech.evaluation.contract")

_REPO_ROOT   = Path(__file__).resolve().parents[2]
_DATA_ROOT   = _REPO_ROOT / "data"
_EVAL_ROOT   = evaluation_dir()


# ── Contract definition ───────────────────────────────────────────────────────

@dataclass
class EvalContract:
    """Immutable frozen evaluation contract.

    Attributes
    ----------
    objective_id:       Objective this contract belongs to.
    contract_version:   Monotonic integer string, e.g. ``"1"``.
    hold_out_hash:      SHA-256 of the concatenated hold-out parquet bytes.
    hold_out_dates:     ISO date strings covered by the hold-out window.
    created_at:         ISO-8601 UTC timestamp.
    tau_thresholds:     List of confidence thresholds to score at.
    objective_snapshot: Snapshot of the objective at contract creation time.
    """

    objective_id:       str
    contract_version:   str
    hold_out_hash:      str
    hold_out_dates:     list[str]
    created_at:         str
    tau_thresholds:     list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9])
    objective_snapshot: dict[str, Any] = field(default_factory=dict)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Persist the contract to disk. Returns the written path."""
        if path is None:
            dir_ = _EVAL_ROOT / self.objective_id
            dir_.mkdir(parents=True, exist_ok=True)
            path = dir_ / f"contract_v{self.contract_version}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        logger.info("contract_saved", extra={"path": str(path), "hash": self.hold_out_hash[:12]})
        return path

    @classmethod
    def load(cls, path: Path) -> "EvalContract":
        d = json.loads(path.read_text())
        return cls(**d)

    @classmethod
    def from_manifest(cls, run_dir: Path) -> "EvalContract | None":
        """The hold-out pin recorded on one run's ``manifest.json``, or None.

        This is the forward path for the handover described in
        ``docs/new_file_map.md``: a migrated run records ``hold_out_dates`` and
        ``hold_out_hash`` on its manifest, so the pin no longer has to live in
        a separate ``evaluation/<objective>/contract_v<N>.json``. Returns None
        whenever the manifest does not actually carry a pin -- an absent pin is
        reported as absent, never reconstructed from the fold design.

        ``contract_version`` is the producing ``run_ts``, not an integer: a
        manifest-sourced pin is versioned by the run that established it. Do
        not ``save()`` one of these -- that would write a
        ``contract_v<run_ts>.json`` the file loader cannot order.
        """
        from src.experiment.manifest import read_run_record

        try:
            doc = read_run_record(Path(run_dir))
        except (FileNotFoundError, OSError, ValueError):
            return None
        validation = doc.get("validation") or {}
        snapshot = doc.get("config_snapshot") or {}
        dates = list(validation.get("hold_out_dates") or [])
        digest = snapshot.get("hold_out_hash") or validation.get("hold_out_hash")
        if not dates or not digest:
            return None
        taus = snapshot.get("tau_thresholds")
        return cls(
            objective_id=str(doc.get("objective_id") or ""),
            contract_version=str(doc.get("run_ts") or Path(run_dir).name),
            hold_out_hash=str(digest),
            hold_out_dates=dates,
            created_at=str(doc.get("created_at") or ""),
            **({"tau_thresholds": [float(t) for t in taus]} if taus else {}),
            objective_snapshot=dict(snapshot),
        )

    @classmethod
    def load_latest(
        cls, objective_id: str, *, run_dir: Path | None = None,
    ) -> "EvalContract | None":
        """Load the highest-versioned contract for *objective_id*, or None.

        ``run_dir`` opts into the manifest as the source of the hold-out pin.
        When it is given and that run carries a pin:

          * no file contract exists -> the manifest pin IS the contract;
          * a file contract exists and agrees -> the manifest pin is returned
            (same science, now read from the run that established it);
          * a file contract exists and DISAGREES -> ``ContractHashMismatch``.

        The disagreement case is the whole point of sequencing this handover:
        two live sources of truth for a hold-out window is precisely how a
        model gets scored against the wrong pin, so a divergence is raised
        rather than resolved by precedence. Callers that pass no ``run_dir``
        get exactly the previous behaviour.
        """
        dir_ = _EVAL_ROOT / objective_id
        from_file: "EvalContract | None" = None
        if dir_.exists():
            files = sorted(dir_.glob("contract_v*.json"),
                           key=lambda p: int(p.stem.split("_v")[1]))
            from_file = cls.load(files[-1]) if files else None
        if run_dir is None:
            return from_file
        from_run = cls.from_manifest(run_dir)
        if from_run is None:
            return from_file
        if from_file is not None and from_file.hold_out_hash != from_run.hold_out_hash:
            raise ContractHashMismatch(
                f"hold-out pin for {objective_id!r} differs between "
                f"{dir_}/contract_v{from_file.contract_version}.json "
                f"({from_file.hold_out_hash[:12]}…, {len(from_file.hold_out_dates)} dates) "
                f"and {run_dir}/manifest.json "
                f"({from_run.hold_out_hash[:12]}…, {len(from_run.hold_out_dates)} dates). "
                f"Two live sources of truth for one hold-out window: reconcile "
                f"them before scoring, do not let precedence pick one."
            )
        return from_run

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        *,
        data_hash: str | None = None,
        objective: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score *predictions* under this contract.

        Refuses if *data_hash* is provided and doesn't match the pinned hash.
        Returns a result dict ready for the leaderboard writer.

        Parameters
        ----------
        predictions:
            Canonical prediction frame (ts, direction, confidence).
        ground_truth:
            Frame with column ``direction_true`` ("up"/"down"/"flat").
        data_hash:
            SHA-256 of the parquet files used for inference.  If supplied and
            mismatched, raises ``ContractHashMismatch``.
        objective:
            Optional objective dict for cost-aware KPIs.
        """
        if data_hash is not None and data_hash != self.hold_out_hash:
            raise ContractHashMismatch(
                f"Data hash mismatch: got {data_hash[:12]}… "
                f"expected {self.hold_out_hash[:12]}… "
                f"(contract v{self.contract_version}). "
                "Comparisons across different hold-out versions are forbidden."
            )

        per_tau: list[dict[str, Any]] = []
        for tau in self.tau_thresholds:
            kpi = canonical_trade_kpi(
                predictions, ground_truth,
                objective=objective or self.objective_snapshot,
                tau=tau,
            )
            per_tau.append({"tau": tau, **kpi})

        # Primary KPI at the YAML-declared fire cut (None = argmax book).
        primary_tau = _primary_tau(self.objective_snapshot)
        if primary_tau is None:
            kpi0 = canonical_trade_kpi(
                predictions, ground_truth,
                objective=objective or self.objective_snapshot, tau=None,
            )
            primary = {"tau": None, **kpi0}
        else:
            primary = next((r for r in per_tau if r["tau"] == primary_tau), per_tau[0])

        return {
            "objective_id":      self.objective_id,
            "contract_version":  self.contract_version,
            "hold_out_hash":     self.hold_out_hash,
            "scored_at":         datetime.now(timezone.utc).isoformat(),
            "n_predictions":     len(predictions),
            "primary_tau":       primary_tau,
            "primary_precision": primary.get("precision_at_tau"),
            "primary_dir_f1":    primary.get("directional_f1"),
            "per_tau":           per_tau,
        }


class ContractHashMismatch(ValueError):
    """Raised when the inference data hash doesn't match the pinned hold-out."""


# ── Factory ───────────────────────────────────────────────────────────────────

def build_contract(
    objective_id: str,
    hold_out_parquet_paths: list[Path],
    *,
    tau_thresholds: list[float] | None = None,
    save: bool = True,
) -> EvalContract:
    """Hash the hold-out parquet files and create a new contract.

    Parameters
    ----------
    objective_id:
        Must match an existing ``config/objectives/<id>.yaml``.
    hold_out_parquet_paths:
        The parquet files constituting the frozen forward hold-out.
        Order matters for the hash — sort by date before calling.
    tau_thresholds:
        Confidence thresholds to sweep. Defaults from the objective's kpi block.
    save:
        If True (default), persist the contract to disk.
    """
    from src.objectives.loader import load_objective

    objective = load_objective(objective_id)

    # Compute hash over sorted parquet bytes
    h = hashlib.sha256()
    for p in sorted(hold_out_parquet_paths):
        h.update(p.read_bytes())
    hold_out_hash = h.hexdigest()

    dates = sorted(
        p.stem.replace("features_", "") for p in hold_out_parquet_paths
        if p.stem.startswith("features_")
    )

    # Next contract version
    dir_ = _EVAL_ROOT / objective_id
    dir_.mkdir(parents=True, exist_ok=True)
    existing = list(dir_.glob("contract_v*.json"))
    next_version = str(max((int(p.stem.split("_v")[1]) for p in existing), default=0) + 1)

    taus = tau_thresholds or _taus_from_objective(objective)

    contract = EvalContract(
        objective_id=objective_id,
        contract_version=next_version,
        hold_out_hash=hold_out_hash,
        hold_out_dates=dates,
        created_at=datetime.now(timezone.utc).isoformat(),
        tau_thresholds=taus,
        objective_snapshot=objective,
    )

    if save:
        contract.save()

    logger.info(
        "contract_built",
        extra={"objective_id": objective_id, "version": next_version,
               "n_files": len(hold_out_parquet_paths), "hash": hold_out_hash[:12]},
    )
    return contract


def pin_hold_out_from_parquet_dir(
    objective_id: str,
    parquet_dir: Path | str,
    start_date: str,
    end_date: str,
) -> EvalContract:
    """Convenience wrapper: find parquet files in *parquet_dir* for the date range
    ``[start_date, end_date]`` (inclusive, ISO format) and build a contract.

    Example::

        contract = pin_hold_out_from_parquet_dir(
            "nq-50pt",
            str(native_tf_dir("5s")),
            start_date="2026-02-03",
            end_date="2026-04-30",
        )
    """
    from src.common.data_paths import list_feature_parquet_files

    parquet_dir = Path(parquet_dir)
    files = list_feature_parquet_files(parquet_dir)
    dated = [
        p for p in files
        if len(p.stem) == len("features_YYYY-MM-DD")
        and start_date <= p.stem.replace("features_", "") <= end_date
    ]
    if dated:
        paths = dated
    else:
        y0, m0 = int(start_date[:4]), int(start_date[5:7])
        y1, m1 = int(end_date[:4]), int(end_date[5:7])
        paths = []
        for p in files:
            month_s = p.parent.name
            year_s = p.parent.parent.name
            if month_s.startswith("month=") and year_s.startswith("year="):
                y, m = int(year_s.split("=", 1)[1]), int(month_s.split("=", 1)[1])
                if (y, m) < (y0, m0) or (y, m) > (y1, m1):
                    continue
            paths.append(p)
    if not paths:
        raise FileNotFoundError(
            f"No parquet files found in {parquet_dir} for {start_date}..{end_date}"
        )
    logger.info("hold_out_files", extra={"n": len(paths), "first": str(paths[0].name),
                                          "last": str(paths[-1].name)})
    return build_contract(objective_id, paths)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _taus_from_objective(objective: dict[str, Any]) -> list[float]:
    from src.evaluation.kpi_criteria import objective_taus
    return objective_taus(objective)


def _primary_tau(objective: dict[str, Any]) -> float | None:
    """YAML-declared fire cut, or None when the objective has no confidence cut."""
    from src.evaluation.kpi_criteria import primary_tau
    return primary_tau(objective)
