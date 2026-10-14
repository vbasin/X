"""Runtime configuration: YAML primary source, env overrides for secrets/legacy.

Loading order (highest precedence wins):
  1. Environment variables (legacy / secrets).
  2. /opt/vtech/config.yaml (single source of truth for runtime knobs).
  3. Hardcoded defaults in this dataclass.

Secrets — `DATABENTO_API_KEY`, `PG*` — must come from environment / .env;
they are never read from YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from pathlib import Path
from typing import Any


_DEFAULT_CONFIG_PATH = Path("/opt/vtech/config.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    databento_api_key: str = field(repr=False)
    pg_dsn: str = field(repr=False)
    data_dir: str = "/opt/vtech/data"
    log_level: str = "INFO"
    features_timestep: str = "5s"
    label_threshold: int = 50
    sequence_length: int = 128
    forward_windows: tuple[int, ...] = (60, 120, 180, 360)
    training_hours_start: str = "08:00"
    training_hours_end: str = "11:00"
    timezone: str = "America/Chicago"
    seed: int = 42

    @classmethod
    def from_env(cls, config_path: Path | str | None = None) -> "AppConfig":
        """Build config from YAML + env-var overrides + secrets.

        config_path defaults to /opt/vtech/config.yaml.
        """
        yaml_data = _load_yaml(Path(config_path) if config_path else _DEFAULT_CONFIG_PATH)
        paths = yaml_data.get("paths", {}) or {}
        data_cfg = yaml_data.get("data", {}) or {}
        labels = yaml_data.get("labels", {}) or {}
        model = yaml_data.get("model", {}) or {}
        training = yaml_data.get("training", {}) or {}
        logging_cfg = yaml_data.get("logging", {}) or {}

        # Helper: env > yaml > default
        def pick(env_key: str, yaml_val, default):
            v = environ.get(env_key)
            return v if v is not None else (yaml_val if yaml_val is not None else default)

        fwd_default = labels.get("forward_windows", [60, 120, 180, 360])
        fwd_env = environ.get("VTECH_FORWARD_WINDOWS")
        if fwd_env is not None:
            fwd_tuple = tuple(int(x) for x in fwd_env.split(","))
        else:
            fwd_tuple = tuple(int(x) for x in fwd_default)

        pg_dsn = (
            f"postgresql://{environ['PGUSER']}:{environ['PGPASSWORD']}"
            f"@{environ.get('PGHOST', 'localhost')}:{environ.get('PGPORT', '5432')}"
            f"/{environ.get('PGDATABASE', 'vtech_market')}"
        )

        return cls(
            databento_api_key=environ["DATABENTO_API_KEY"],
            pg_dsn=pg_dsn,
            data_dir=str(pick("VTECH_DATA_DIR", paths.get("data_dir"), "/opt/vtech/data")),
            log_level=str(pick("VTECH_LOG_LEVEL", logging_cfg.get("level"), "INFO")),
            features_timestep=str(pick("VTECH_FEATURES_TIMESTEP", data_cfg.get("feature_timestep"), "5s")),
            label_threshold=int(pick("VTECH_LABEL_THRESHOLD", labels.get("big_move_threshold"), 50)),
            sequence_length=int(pick("VTECH_SEQUENCE_LENGTH", model.get("sequence_length"), 128)),
            forward_windows=fwd_tuple,
            training_hours_start=str(pick("VTECH_TRAINING_HOURS_START", data_cfg.get("training_start"), "08:00")),
            training_hours_end=str(pick("VTECH_TRAINING_HOURS_END", data_cfg.get("training_end"), "11:00")),
            timezone=str(pick("VTECH_TIMEZONE", data_cfg.get("timezone"), "America/Chicago")),
            seed=int(pick("VTECH_SEED", training.get("seed"), 42)),
        )


def load_yaml_section(section: str, config_path: Path | str | None = None) -> dict[str, Any]:
    """Read a single top-level section from /opt/vtech/config.yaml.

    Useful for callers that need richer YAML structure than `AppConfig`
    exposes (e.g. the trainer's nested model/training sub-trees).
    """
    yaml_data = _load_yaml(Path(config_path) if config_path else _DEFAULT_CONFIG_PATH)
    return yaml_data.get(section, {}) or {}
