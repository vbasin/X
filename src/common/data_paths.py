"""Runtime data-root path SSOT after the 2026-08 data reorg.

All application code should resolve durable paths through this module (or thin
wrappers that call it). Layout under ``$VTECH_DATA_DIR`` (default ``/opt/vtech/data``)::

    sources/raw/
    sources/features/parquet_5s/          # leftover live day files (not GPU/OCI)
    sources/features/parquet_raw1s/
    sources/features/parquet_options_surface/
    sources/features/parquet_options_greeks/
    sources/features/timeframe={5s,15s,1m,5m,1h,1d}/symbol=NQ/year=/month=/
                                          # hive (ADR-070/075/079/081); 5s+15s = monthly parts

    models/model_staging/
    models/model_registry/
    models/training_runs/
    models/leaders/                       # per-objective search leaderboards
    research/search/
    research/custom_test/
    research/scoping/
    research/research_director/
    research/episodes/                    # was data/evaluation/episodes
    evaluation/                           # frozen contracts (leaderboards: models/leaders/)
    ops/catalog|gpu_health|gpu_jobs|webui|tmp|mlflow_artifacts/
    legacy/checkpoints|hpo|research_runs/

Relative paths like ``data/parquet`` in docs/CLI still resolve via :func:`resolve_data_path`
for operator convenience.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path

NATIVE_TF_TIMEFRAMES: tuple[str, ...] = ("15s", "1m", "5m", "1h", "1d")
# Hive keys on disk. 5s research matrix uses the same tree (ADR-079);
# 1-bar native product is still NATIVE_TF_TIMEFRAMES (ADR-070 / ADR-075).
HIVE_TIMEFRAMES: tuple[str, ...] = ("5s",) + NATIVE_TF_TIMEFRAMES
_NATIVE_TF_SET = frozenset(HIVE_TIMEFRAMES)
NATIVE_TF_SYMBOL_DEFAULT = "NQ"

# Logical name → path under data root (posix relative).
_LAYOUT: dict[str, str] = {
    "raw": "sources/raw",
    "parquet": "sources/features/parquet_5s",
    "parquet_5s": "sources/features/parquet_5s",
    "parquet_raw1s": "sources/features/parquet_raw1s",
    "parquet_path1s": "sources/features/parquet_path1s",
    "parquet_tape": "sources/features/parquet_tape",
    # Print tape for one instrument root. New writes land here
    # (sources/price_tape/NQ/); the legacy parquet_tape key remains the
    # existing per-session files until they are compacted.
    "price_tape": "sources/price_tape",
    "runs": "research/runs",
    "parquet_pathlabels": "sources/features/parquet_pathlabels",
    "parquet_labels_15s": "sources/features/parquet_labels_15s",
    "parquet_labels_5s": "sources/features/parquet_labels_5s",
    "parquet_options_surface": "sources/features/parquet_options_surface",
    "parquet_options_greeks": "sources/features/parquet_options_greeks",
    # Outcome/label datasets: contract-addressed sets (label_store.py) and
    # Track C's hand-built datasets both live here. Previously had no key
    # and was hand-constructed in 14+ sites (docs/STORAGE_CONTRACT.md).
    "labels": "sources/labels",
    "model_staging": "models/model_staging",
    "model_registry": "models/model_registry",
    "training_runs": "models/training_runs",
    "leaders": "models/leaders",
    "search": "research/search",
    "custom_test": "research/custom_test",
    "scoping": "research/scoping",
    "research_director": "research/research_director",
    "episodes": "research/episodes",
    "evaluation": "evaluation",
    "catalog": "ops/catalog",
    "gpu_health": "ops/gpu_health",
    "gpu_jobs": "ops/gpu_jobs",
    "webui": "ops/webui",
    "tmp": "ops/tmp",
    "mlflow_artifacts": "ops/mlflow_artifacts",
    "paper": "ops/paper",
    "execution": "ops/execution",
    # Live-signal file bridge (webui/live.py, regmodel.py, execution/config.py all
    # constructed this independently before it had a _LAYOUT key -- H13,
    # docs/STORAGE_CONTRACT.md §2.5).
    "live": "live",
    "session_profile": "sources/features/session_profile",
    "checkpoints": "legacy/checkpoints",
    "hpo": "legacy/hpo",
    "research_runs": "legacy/research_runs",
    # Hive snapshot (removed from disk; path kept for tools that still look)
    "features_hive": "legacy/features_hive",
    "features": "legacy/features_hive",
}

# Old flat relative segments operators may still type on the CLI.
_LEGACY_ALIASES: dict[str, str] = {
    "data/parquet": "parquet_5s",
    "data/parquet_raw1s": "parquet_raw1s",
    "data/parquet_path1s": "parquet_path1s",
    "data/parquet_pathlabels": "parquet_pathlabels",
    "data/parquet_labels_15s": "parquet_labels_15s",
    "data/parquet_labels_5s": "parquet_labels_5s",
    "data/parquet_options_surface": "parquet_options_surface",
    "data/parquet_options_greeks": "parquet_options_greeks",
    "data/model_staging": "model_staging",
    "data/model_registry": "model_registry",
    "data/training_runs": "training_runs",
    "data/models/leaders": "leaders",
    "data/search": "search",
    "data/custom_test": "custom_test",
    "data/scoping": "scoping",
    "data/research_director": "research_director",
    "data/evaluation/episodes": "episodes",
    "data/evaluation": "evaluation",
    "data/catalog": "catalog",
    "data/gpu_health": "gpu_health",
    "data/gpu_jobs": "gpu_jobs",
    "data/webui": "webui",
    "data/tmp": "tmp",
    "data/mlflow_artifacts": "mlflow_artifacts",
    "data/paper": "paper",
    "data/execution": "execution",
    "data/checkpoints": "checkpoints",
    "data/raw": "raw",
    "data/features": "features_hive",
    "data/research_runs": "research_runs",
    "data/hpo": "hpo",
}


def data_root() -> Path:
    """Live runtime data root (honours ``VTECH_DATA_DIR`` every call)."""
    return Path(os.environ.get("VTECH_DATA_DIR", "/opt/vtech/data"))


def data_path(*parts: str) -> Path:
    """Join under data root (no logical remapping)."""
    return data_root().joinpath(*parts)


def layout(key: str) -> Path:
    """Resolve a logical layout key (e.g. ``\"parquet_5s\"``, ``\"model_staging\"``)."""
    rel = _LAYOUT.get(key)
    if rel is None:
        raise KeyError(f"unknown data layout key: {key!r}; known={sorted(_LAYOUT)}")
    return data_root() / rel



def labels_dir() -> Path:
    """Root of outcome/label datasets (``sources/labels``)."""
    return layout("labels")


def outcome_dataset_dir(outcome_family: str, resolution: str, contract_id: str) -> Path:
    """Canonical NEW-write directory for any outcome/label dataset.

    ``sources/labels/<outcome_family>/<resolution>/<contract_id>/`` --
    docs/STORAGE_CONTRACT.md §17. One pattern for Track B's existing TBM
    outcomes, Track C's move/target outcomes and any future outcome
    definition; ``outcome_family`` and ``resolution`` are kept as two
    distinct dimensions (what the outcome MEANS at a category level, vs. the
    decision-time cadence it was computed at) rather than overloaded into
    one directory slot, which is the exact ambiguity this function replaces
    -- Track B's ``5s``/``15s`` are resolutions of ONE family; Track C's
    move/target outcomes are genuinely DIFFERENT families, not resolutions
    of each other. ``contract_id`` is the semantic-contract hash (see
    ``src.training.label_store.outcome_contract_id`` / ``label_contract_id``)
    -- this function does no hashing itself, it is pure path construction.

    This is a NEW-write target only. It does not resolve, search or fall
    back to any existing Track B/C compatibility path -- see
    docs/STORAGE_CONTRACT.md §17.2 for the mapping from those paths to this
    pattern. No existing data is moved by this function's existence.
    """
    for field_name, value in (("outcome_family", outcome_family),
                              ("resolution", resolution),
                              ("contract_id", contract_id)):
        if not str(value).strip():
            raise ValueError(f"outcome_dataset_dir: {field_name} must be non-empty")
    return labels_dir() / str(outcome_family) / str(resolution) / str(contract_id)


def logical_path(path: Path | str) -> str | None:
    """A data-root-relative POSIX path, or ``None`` if outside the data root.

    The machine-independent form in which a physical location is RECORDED
    (e.g. ``DatasetRef.location``). ``/opt/vtech/data`` is one machine's
    mount; ``sources/labels/15s/<id>`` means the same thing on every machine
    that resolves ``data_root()``. Returns ``None`` rather than guessing when
    the path is not under the data root.
    """
    try:
        return Path(path).resolve().relative_to(data_root().resolve()).as_posix()
    except (ValueError, OSError):
        return None


# ── Convenience accessors ─────────────────────────────────────────────────────

def raw_dir() -> Path:
    return layout("raw")


def _prefer(*candidates: Path) -> Path:
    """Return the first existing path, else the first (canonical write target)."""
    for c in candidates:
        try:
            if c.exists():
                return c
        except OSError:
            continue
    return candidates[0]


def parquet_5s_dir() -> Path:
    """Leftover 5s *day-file* dir (live ``extend_date``). Research/GPU use the hive.

    Canonical 5s research cache is :func:`native_tf_dir` ``("5s")`` (ADR-079 / ADR-081).
    """
    root = data_root()
    return _prefer(layout("parquet_5s"), root / "parquet", root / "parquet_5s")


def parquet_raw1s_dir() -> Path:
    root = data_root()
    return _prefer(layout("parquet_raw1s"), root / "parquet_raw1s")


def parquet_path1s_dir() -> Path:
    """1-second OHLC fill-resolution path cache (``sources/features/parquet_path1s``)."""
    root = data_root()
    return _prefer(layout("parquet_path1s"), root / "parquet_path1s")


def price_tape_dir(root: str = "NQ") -> Path:
    """Print-level tape for one instrument root (``sources/price_tape/NQ``).

    Built by ``scripts/data/build_tape.py`` from Timescale ``databento.tbbo``
    for that session's lead outright. ``root`` is the instrument whose prices
    are in the file (NQ), not the outright month.
    """
    sym = str(root).strip() or "NQ"
    return layout("price_tape") / sym


def runs_dir() -> Path:
    """Objective runs: ``research/runs/<objective_id>/<run_ts>/``."""
    return layout("runs")


def parquet_tape_dir() -> Path:
    """Print-level TBBO tape cache (``sources/features/parquet_tape``).

    One file per session, lead-contract only. Supersedes ``parquet_path1s`` for the
    label pipeline: prints carry exact price, size and aggressor side, so
    volume-at-price, delta-at-price and first-touch ordering are observed rather
    than reconstructed from 1-second bars.
    """
    root = data_root()
    return _prefer(layout("parquet_tape"), root / "parquet_tape")


def parquet_labels_15s_dir() -> Path:
    """15s dual-binary labels subsampled from 5s TBM (ADR-066)."""
    root = data_root()
    return _prefer(
        layout("parquet_labels_15s"),
        root / "sources" / "labels" / "parquet_labels_15s",
    )


def parquet_labels_5s_dir() -> Path:
    """5s 3-class TBM labels persisted from ``_close`` (ADR-076)."""
    return layout("parquet_labels_5s")


def native_tf_features_root() -> Path:
    """Hive parent: ``sources/features`` (``timeframe=`` keys sit beside leftover ``parquet_5s/``).

    Dataset readers must start at ``timeframe={tf}/``, not this parent — leftover
    live day files and other flat caches are siblings. See ADR-070 / ADR-079 / ADR-081.
    """
    return data_root() / "sources" / "features"


def hive_timeframe_from_path(path: Path | str) -> str | None:
    """Return the ``timeframe=`` key if *path* is under a hive tree, else None."""
    p = Path(path)
    for anc in (p, *p.parents):
        if anc.name.startswith("timeframe="):
            tf = anc.name.split("=", 1)[1]
            return tf if tf in _NATIVE_TF_SET else None
    return None


def hive_root_from_path(path: Path | str) -> Path | None:
    """``timeframe={tf}`` directory for a hive path, else None."""
    p = Path(path)
    for anc in (p, *p.parents):
        if anc.name.startswith("timeframe="):
            return anc
    return None


def list_feature_parquet_files(pdir: Path | str) -> list[Path]:
    """Dated day files or hive monthly parts under *pdir*.

    Prefers ``features_YYYY-MM-DD.parquet`` when those exist (live/tmp day dirs).
    Else hive ``year=*/month=*/features_*.parquet`` (and ``symbol=*/…`` if *pdir*
    is a ``timeframe=`` root).
    """
    p = Path(pdir)
    if not p.is_dir():
        return []
    dated = sorted(
        x for x in p.glob("features_*.parquet")
        if x.is_file() and len(x.stem) == len("features_YYYY-MM-DD")
    )
    if dated:
        return dated
    hive = sorted(p.glob("year=*/month=*/features_*.parquet"))
    if hive:
        return hive
    return sorted(p.glob("symbol=*/year=*/month=*/features_*.parquet"))


def _require_native_tf(timeframe: str) -> str:
    tf = str(timeframe).strip()
    if tf not in _NATIVE_TF_SET:
        raise ValueError(
            f"unknown native timeframe {timeframe!r}; "
            f"expected one of {HIVE_TIMEFRAMES} "
            f"(5s uses the same timeframe=/symbol=/year=/month= tree)"
        )
    return tf


def _month_part(month: int | str) -> str:
    m = int(month)
    if m < 1 or m > 12:
        raise ValueError(f"month must be 1..12, got {month!r}")
    return f"{m:02d}"


def _as_date(day: date | datetime | str) -> date:
    if isinstance(day, datetime):
        return day.date()
    if isinstance(day, date):
        return day
    return date.fromisoformat(str(day)[:10])


def native_tf_dir(
    timeframe: str,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    year: int | str | None = None,
    month: int | str | None = None,
) -> Path:
    """Hive partition dir: ``timeframe=/symbol=/year=/month=``.

    Omitting ``year`` stops at ``symbol=``. ``month`` requires ``year``.
    """
    tf = _require_native_tf(timeframe)
    sym = str(symbol).strip() or NATIVE_TF_SYMBOL_DEFAULT
    p = native_tf_features_root() / f"timeframe={tf}" / f"symbol={sym}"
    if year is None:
        if month is not None:
            raise ValueError("month requires year")
        return p
    p = p / f"year={int(year):04d}"
    if month is None:
        return p
    return p / f"month={_month_part(month)}"


def native_tf_parquet_path(
    timeframe: str,
    day: date | datetime | str,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
) -> Path:
    """Day file inside the month partition: ``features_YYYY-MM-DD.parquet``.

    5s and 15s files are monthly size-split parts (ADR-075 / ADR-079), not
    day files. Prefer :func:`native_tf_part_path` / :func:`native_tf_part_files`
    for those timeframes.
    """
    d = _as_date(day)
    return native_tf_dir(
        timeframe, symbol=symbol, year=d.year, month=d.month,
    ) / f"features_{d.isoformat()}.parquet"


NATIVE_TF_PART_MAX_BYTES: int = 100 * 1024 * 1024
_PART_STEM = "features_{tf}_part{n}.parquet"


def native_tf_part_name(timeframe: str, part: int) -> str:
    """``features_15s_part1.parquet`` (1-based part index)."""
    tf = _require_native_tf(timeframe)
    n = int(part)
    if n < 1:
        raise ValueError(f"part must be >= 1, got {part!r}")
    return _PART_STEM.format(tf=tf, n=n)


def native_tf_part_path(
    timeframe: str,
    year: int | str,
    month: int | str,
    part: int,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
) -> Path:
    """Size-split monthly part: ``features_{tf}_partN.parquet`` (ADR-075)."""
    return native_tf_dir(
        timeframe, symbol=symbol, year=year, month=month,
    ) / native_tf_part_name(timeframe, part)


def native_tf_part_files(
    timeframe: str,
    year: int | str,
    month: int | str,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
) -> list[Path]:
    """Existing ``features_{tf}_part*.parquet`` in a month dir, ordered by part N."""
    tf = _require_native_tf(timeframe)
    d = native_tf_dir(timeframe, symbol=symbol, year=year, month=month)
    if not d.is_dir():
        return []
    rx = re.compile(rf"^features_{re.escape(tf)}_part(\d+)\.parquet$")
    found: list[tuple[int, Path]] = []
    for p in d.iterdir():
        m = rx.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _hive_ym_from_path(path: Path) -> tuple[int, int] | None:
    month_s = path.parent.name
    year_s = path.parent.parent.name
    if not month_s.startswith("month=") or not year_s.startswith("year="):
        return None
    try:
        return int(year_s.split("=", 1)[1]), int(month_s.split("=", 1)[1])
    except ValueError:
        return None


def clip_hive_part_files(
    files: list[Path],
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
) -> list[Path]:
    """Keep hive monthly parts whose ``year=/month=`` partition intersects [*start*, *end*].

    Parts that are not under a ``year=*/month=*/`` path are dropped when a
    range is set (dated day files use a different loader). With both bounds
    omitted the input list is returned unchanged.
    """
    if start is None and end is None:
        return list(files)
    y0, m0 = (0, 0)
    y1, m1 = (9999, 12)
    if start is not None:
        d0 = _as_date(start)
        y0, m0 = d0.year, d0.month
    if end is not None:
        d1 = _as_date(end)
        y1, m1 = d1.year, d1.month
    out: list[Path] = []
    for p in files:
        ym = _hive_ym_from_path(p)
        if ym is None:
            continue
        y, m = ym
        if (y, m) >= (y0, m0) and (y, m) <= (y1, m1):
            out.append(p)
    return out


def iter_hive_part_files(
    timeframe: str,
    *,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
) -> list[Path]:
    """All ``features_{tf}_part*.parquet`` under a hive, optionally clipped by UTC month."""
    tf = _require_native_tf(timeframe)
    root = native_tf_dir(tf, symbol=symbol)
    if not root.is_dir():
        return []
    files = sorted(root.glob(f"year=*/month=*/features_{tf}_part*.parquet"))
    return clip_hive_part_files(files, start, end)


def ensure_native_tf_dirs(
    *,
    timeframes: tuple[str, ...] | list[str] = NATIVE_TF_TIMEFRAMES,
    symbol: str = NATIVE_TF_SYMBOL_DEFAULT,
    start_year: int = 2025,
    start_month: int = 1,
    end_year: int = 2026,
    end_month: int = 12,
) -> list[Path]:
    """``mkdir -p`` month partitions for each native TF. Idempotent."""
    tfs = [_require_native_tf(tf) for tf in timeframes]
    out: list[Path] = []
    y, m = int(start_year), int(start_month)
    ey, em = int(end_year), int(end_month)
    _month_part(m)
    _month_part(em)
    while (y, m) <= (ey, em):
        for tf in tfs:
            d = native_tf_dir(tf, symbol=symbol, year=y, month=m)
            d.mkdir(parents=True, exist_ok=True)
            out.append(d)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out


def parquet_pathlabels_dir() -> Path:
    """Honest first-touch labels + dense targets from the 1s path.

    Built by :mod:`scripts.features.build_path_labels` off
    :func:`parquet_path1s_dir`. One parquet per session; ``label`` follows the
    ``src.training.labels`` encoding (``BIG_DOWN=0``, ``NO_SIGNAL=1``, ``BIG_UP=2``),
    plus dense competing-risks + MFE/MAE targets. See ADR-052 (path-realised) and the
    label-geography ADR for the vol-normalised barrier contract.
    """
    root = data_root()
    return _prefer(layout("parquet_pathlabels"), root / "parquet_pathlabels")


def options_surface_dir() -> Path:
    root = data_root()
    return _prefer(layout("parquet_options_surface"), root / "parquet_options_surface")


def options_greeks_dir() -> Path:
    root = data_root()
    return _prefer(layout("parquet_options_greeks"), root / "parquet_options_greeks")


def model_staging_dir() -> Path:
    root = data_root()
    return _prefer(layout("model_staging"), root / "model_staging")


def model_registry_dir() -> Path:
    root = data_root()
    return _prefer(layout("model_registry"), root / "model_registry")


def training_runs_dir() -> Path:
    root = data_root()
    return _prefer(layout("training_runs"), root / "training_runs")


def leaders_dir() -> Path:
    """Per-objective search leaderboards (``models/leaders/<objective_id>/``).

    Holds ``leaderboard_v*.jsonl`` and ``latest_leaders.json``. Frozen eval
    contracts stay under :func:`evaluation_dir`.
    """
    return layout("leaders")


def search_dir() -> Path:
    root = data_root()
    return _prefer(layout("search"), root / "search")


def custom_test_dir() -> Path:
    """Custom Test lab runs (``research/custom_test/<objective>/<run_ts>/``)."""
    root = data_root()
    return _prefer(layout("custom_test"), root / "research" / "custom_test",
                   root / "custom_test")


def scoping_dir() -> Path:
    root = data_root()
    return _prefer(layout("scoping"), root / "scoping")


def research_director_dir() -> Path:
    root = data_root()
    return _prefer(layout("research_director"), root / "research_director")


def evaluation_dir() -> Path:
    """Frozen eval contracts (``evaluation/<objective_id>/contract_v*.json``).

    Search leaderboards live under :func:`leaders_dir`.
    """
    return layout("evaluation")


def episodes_dir() -> Path:
    """Outcome episodes root (was ``evaluation/episodes``)."""
    root = data_root()
    return _prefer(layout("episodes"), root / "evaluation" / "episodes")


def catalog_dir() -> Path:
    root = data_root()
    return _prefer(layout("catalog"), root / "catalog")


def gpu_health_dir() -> Path:
    root = data_root()
    return _prefer(layout("gpu_health"), root / "gpu_health")


def gpu_jobs_dir() -> Path:
    """Live GPU job heartbeats + per-fold checkpoints (ADR-073)."""
    root = data_root()
    return _prefer(layout("gpu_jobs"), root / "gpu_jobs")


def webui_dir() -> Path:
    root = data_root()
    return _prefer(layout("webui"), root / "webui")


def tmp_dir() -> Path:
    root = data_root()
    return _prefer(layout("tmp"), root / "tmp")


def paper_dir() -> Path:
    """Paper blotter + open positions (P2 mock execution)."""
    root = data_root()
    return _prefer(layout("paper"), root / "paper", root / "ops" / "paper")


def execution_dir() -> Path:
    """Broker execution state, audit ledger, and default file bridge."""
    root = data_root()
    return _prefer(layout("execution"), root / "execution", root / "ops" / "execution")


def live_dir() -> Path:
    """Live-signal file bridge (``signals.jsonl``, ``heartbeat.json``).

    Written by ``src/webui/live.py``, read by ``src/execution/config.py``'s
    default ``signal_path``/``heartbeat_path``. Distinct from
    :func:`execution_dir` (``ops/execution``, unpopulated today) -- this is
    the directory actually carrying the live-signal contract.

    Semantic ownership is ``ops/`` (mutable operational state, same category
    as :func:`execution_dir`/:func:`paper_dir`) -- but the CURRENT, actively
    written physical location is top-level ``live/``, which predates that
    convention (docs/STORAGE_CONTRACT.md §8a). ``ops/live/`` is the
    documented FUTURE canonical target, not the current one.

    Tenth-pass correction: a prior pass (ninth) had this resolver
    auto-prefer ``ops/live`` the instant it existed on disk, reasoning that
    would make a future migration require no caller change. That auto-
    cutover is reverted here -- switching the live-signal transport to a new
    directory is an operational decision (verifying nothing external still
    watches the old path, coordinating the switch with whatever is running)
    that this resolver must not make unilaterally just because a directory
    happens to exist. This always resolves to the current active top-level
    path; adopting ``ops/live`` requires an explicit future cutover (a
    dedicated code change), not a filesystem probe.
    """
    return layout("live")


def mlflow_artifacts_dir() -> Path:
    root = data_root()
    return _prefer(layout("mlflow_artifacts"), root / "mlflow_artifacts")


def checkpoints_dir() -> Path:
    """Legacy trainer + ingest job JSON (``legacy/checkpoints``)."""
    root = data_root()
    return _prefer(layout("checkpoints"), root / "checkpoints")


def is_default_5s_feature_dir(p: Path | str) -> bool:
    """True if *p* is the leftover 5s day-file dir (not the hive)."""
    path = Path(p)
    try:
        if path.resolve() == parquet_5s_dir().resolve():
            return True
    except OSError:
        pass
    return path.name in ("parquet", "parquet_5s")


def model_artifact_roots() -> list[Path]:
    """Dirs Live/Backtest may load serving bundles from.

    Accepts the post-reorg layout (``models/model_staging``) and the pre-reorg
    flat layout (``model_staging`` under data root) so tests and partial
    migrations keep working.
    """
    roots: list[Path] = []
    seen: set[Path] = set()
    candidates = [
        model_staging_dir(),
        model_registry_dir(),
        # research/runs/ is where every new run's bundles live. The two legacy
        # research roots stay allowed because staged/registered bundles still
        # point into research/search/ (see project_manifest open findings).
        runs_dir(),
        search_dir(),
        custom_test_dir(),
        data_root() / "model_staging",
        data_root() / "model_registry",
        data_root() / "research" / "search",
        data_root() / "research" / "custom_test",
        data_root() / "search",
    ]
    for root in candidates:
        try:
            if root.exists():
                r = root.resolve()
                if r not in seen:
                    seen.add(r)
                    roots.append(r)
        except OSError:
            pass
    return roots


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a user/CLI path that may use pre-reorg names.

    Order: as given if it exists → logical/legacy alias under data root →
    join under data root as-is.
    """
    p = Path(path)
    if p.is_file() or p.is_dir():
        return p
    s = str(path).replace("\\", "/").lstrip("./")
    # Longest alias prefix so nested keys like data/models/leaders/<obj>/… map.
    best_alias = ""
    for alias in _LEGACY_ALIASES:
        if s == alias or s.startswith(alias + "/"):
            if len(alias) > len(best_alias):
                best_alias = alias
    if best_alias:
        key = _LEGACY_ALIASES[best_alias]
        extra = s[len(best_alias):].lstrip("/")
        return layout(key) / extra if extra else layout(key)
    # First path segment as layout key
    parts = Path(s).parts
    if parts and parts[0] in _LAYOUT:
        return layout(parts[0]).joinpath(*parts[1:])
    if parts and parts[0] == "data" and len(parts) > 1:
        if parts[1] in _LAYOUT:
            return layout(parts[1]).joinpath(*parts[2:])
    # Absolute under old /opt/vtech/data/X → remap
    try:
        root = data_root().resolve()
        rp = p if p.is_absolute() else (Path.cwd() / p)
        # common absolute old paths
        old_root = Path("/opt/vtech/data")
        if str(rp).startswith(str(old_root) + "/"):
            rel = rp.relative_to(old_root)
            segs = rel.parts
            if segs and segs[0] in _LAYOUT:
                return layout(segs[0]).joinpath(*segs[1:])
            # already new layout under same root
            return root / rel
    except (ValueError, OSError):
        pass
    if not p.is_absolute():
        return data_root() / p
    return p
