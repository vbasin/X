"""Canonical paths under ``config/`` after ADR-028 layout consolidation.

Layout::

    config/
      objectives/     trade objective YAMLs + objective.schema.json
      agents/         personas + duties
      schemas/        JSON schemas (all other *.schema.json)
      specs/          manual/iterate/strategy/run candidate YAMLs
      runtime/        research_director.yaml, scoping.yaml, meta_ab, …

Resolvers prefer the new path and fall back to the legacy flat ``config/`` location
so partially-migrated checkouts and operator overrides keep working.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = _REPO_ROOT / "config"

OBJECTIVES_DIR = CONFIG_ROOT / "objectives"
AGENTS_DIR = CONFIG_ROOT / "agents"
SCHEMAS_DIR = CONFIG_ROOT / "schemas"
SPECS_DIR = CONFIG_ROOT / "specs"
RUNTIME_DIR = CONFIG_ROOT / "runtime"


def config_file(*parts: str, prefer: str | None = None) -> Path:
    """Resolve a config-relative file.

    *prefer*: ``"schemas" | "specs" | "runtime" | "agents" | "objectives" | None``
    searches that subdir first, then ``config/`` root (legacy flat layout).
    """
    name = Path(*parts)
    search: list[Path] = []
    if prefer == "schemas":
        search.append(SCHEMAS_DIR / name)
    elif prefer == "specs":
        search.append(SPECS_DIR / name)
    elif prefer == "runtime":
        search.append(RUNTIME_DIR / name)
    elif prefer == "agents":
        search.append(AGENTS_DIR / name)
    elif prefer == "objectives":
        search.append(OBJECTIVES_DIR / name)
    elif prefer is None and len(parts) == 1:
        # Heuristic: extension-based default homes
        p = parts[0]
        if p.endswith(".schema.json"):
            search.append(SCHEMAS_DIR / p)
        elif p.endswith((".yaml", ".yml")) and p not in (
            # never auto-route objectives here
        ):
            # runtime yaml names vs specs — check both later
            search.extend([RUNTIME_DIR / p, SPECS_DIR / p])
    search.append(CONFIG_ROOT / name)
    for cand in search:
        if cand.is_file():
            return cand
    # Preferred location even if missing (callers create/error on read)
    return search[0] if search else CONFIG_ROOT / name


def schema_path(name: str) -> Path:
    """``critic_objection.schema.json`` etc. — schemas/ then flat config/."""
    if not name.endswith(".schema.json") and not name.endswith(".json"):
        name = f"{name}.schema.json" if not name.endswith(".json") else name
    for cand in (SCHEMAS_DIR / name, CONFIG_ROOT / name, OBJECTIVES_DIR / name):
        if cand.is_file():
            return cand
    return SCHEMAS_DIR / name


def spec_path(name: str) -> Path:
    for cand in (SPECS_DIR / name, CONFIG_ROOT / name):
        if cand.is_file():
            return cand
    return SPECS_DIR / name


def runtime_path(name: str) -> Path:
    for cand in (RUNTIME_DIR / name, CONFIG_ROOT / name):
        if cand.is_file():
            return cand
    return RUNTIME_DIR / name


def resolve_user_path(path: str | Path, *, kind: str = "specs") -> Path:
    """Resolve an operator-supplied path that may still use pre-P0 flat locations.

    Order: as given (absolute or relative to CWD) → basename under the kind dir
    → basename under config/ root. Relative paths under the repo root are also tried.
    """
    p = Path(path)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(_REPO_ROOT / p)
    name = p.name
    if kind == "specs":
        candidates.extend([SPECS_DIR / name, CONFIG_ROOT / name])
    elif kind == "runtime":
        candidates.extend([RUNTIME_DIR / name, CONFIG_ROOT / name])
    elif kind == "schemas":
        candidates.extend([SCHEMAS_DIR / name, CONFIG_ROOT / name])
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[0]
