# Configuration Reference

> **Config file**: `/opt/vtech/config.yaml` — non-secret runtime knobs
> **Secrets file**: `/opt/vtech/.env` — API keys and DB credentials only
> **Objectives**: `/opt/vtech/config/objectives/<id>.yaml` — labels, session, KPI, costs, search topology (SSOT; see ADR-018 / ADR-027)
> **External models**: `/opt/vtech/config/external_models/<id>.yaml` — factory cards for the External Models tab (`psar`, `ema20`, `zigzag`, `timed_psar`; ADR-058 / ADR-062 / ADR-063 / ADR-096). Shared **Backtest Parameters** (session, ATR gate, PT/SL, sides, costs) are edited on `#/backtest` when type is EXT Model; PT units are points, × ATR, or × stop. Overlays live under `$VTECH_DATA_DIR/ops/webui/external_models/_book/`. Per-model defaults and hardened snapshots live under `$VTECH_DATA_DIR/ops/webui/external_models/<id>/` and are not in git. `exit_at_sar` is a `psar` knob (ADR-061). Timed PSAR knobs: entry clock or bars-from-open, 1h/1d SAR direction.
> **Loaders**: `src/common/config.py` — `AppConfig.load()`; `src/training/config.py` — `MLConfig.from_env()`; `src/objectives/loader.py` — objectives.

---

## Loading Order

1. `config.yaml` is read as the base
2. Environment variables matching `VTECH_*` override YAML values (backwards compatibility)
3. `.env` is loaded by `python-dotenv` before any imports (secrets only)
4. Active trade objective is named by `objective_id` in `config.yaml` (required when multiple objective files exist)

## Execution gateway (`config/runtime/execution_gateway.yaml`)

This file is loaded independently by `src.execution.gateway`. It is fail-closed:
`enabled: false`, `mode: SIM`, `account: Sim101`, and blank `ninja_instrument`.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Master command-publication gate |
| `mode` | `SIM` | Command mode; `SIM` or `LIVE` |
| `account` | `Sim101` | Exact NinjaTrader account name |
| `objectives` | `[nq-50pt-econ]` | Objective allowlist |
| `ninja_instrument` | blank | Exact NinjaTrader contract, e.g. `NQ 09-26`; required when enabled |
| `quantity` | `1` | Contracts per approved signal |
| `entry_order_type` | `MARKET` | `MARKET` or `LIMIT`; MARKET avoids stale signal prices |
| `max_positions` | `1` | Bridge v1 requires one active/pending strategy position |
| `max_orders_per_session` | `10` | Emitted-command cap per objective/session |
| `max_daily_loss_usd` | `500` | Blocks new commands at or below the negative threshold |
| `signal_ttl_s` / `heartbeat_stale_s` | `45` / `30` | Signal and VM-live-loop freshness gates |
| `command_ttl_s` | `20` | NinjaTrader must receive a command before expiry |
| `poll_s` | `0.25` | Linux signal-ledger poll interval |
| `bridge_dir` / `state_dir` | under `ops/execution` | Atomic transport and durable gateway state |

`VTECH_EXECUTION_CONFIG` selects another config file.
`VTECH_EXECUTION_BRIDGE_DIR` overrides only the bridge path, normally to a private
SMB mount. See [NINJATRADER_EXECUTION.md](../operations/NINJATRADER_EXECUTION.md).

---

## `objective_id` (config.yaml)

| Key | Type | Description |
|-----|------|-------------|
| `objective_id` | string | Active objective file stem under `config/objectives/`. Examples: `nq-50pt`, `nq-50pt-tbm`, `nq-50pt-econ`. |

### Objective document (schema: `config/objectives/objective.schema.json`)

| Block | Purpose |
|-------|---------|
| `labels` | Ground truth: `scheme` (`detection_2` \| `triple_barrier`), `move_threshold`, `mae_tolerance`, `target_observation` |
| `session` | Trading window (drives eval); timezone |
| `kpi.primary` | Champion rank key: `precision_at_tau` (legacy) or `net_expectancy` / `hit_rate` / `sharpe` (ADR-027) |
| `kpi.requirement` / `kpi.target` | Gate floor vs program goal (ADR-014); economics metrics allowed |
| `costs` / `risk_return` | Friction + soft risk targets for expectancy |
| `search.feature_views` / `default_feature_view` | Sparse per-family views (W4) |
| `search.topology` | `none` / `primary_meta` / `output_meta` / `meta_stack`. Custom Test may overlay this per run (ADR-035) without editing the YAML. |
| `eligible_model_families` | Optional catalog name/group filter |

### Custom Test board (`config/schemas/custom_test.schema.json`)

WebUI **Custom test** (`#/objectives/<id>/test`) compiles a board into a
manual-candidates spec + topology overlay and launches the search controller.

| Field | Purpose |
|-------|---------|
| `models[]` | Catalog `family` + dragged `items` (`view` / `group` / `feature`) |
| `models[].items[].ablation` | If true, that unit is included and excluded (`2^k` per slot) |
| `models[].params` | Optional hyperparameter overrides merged over catalog `params_default` (ADR-040) |
| `models[].hpo` / `hpo_params` | Opt-in HPO: subset of that family's catalog `search_space`. Empty `hpo_params` with `hpo: true` = all catalog knobs. Families with no space stay fixed. |
| `tune_trials` | Optuna trials per HPO-enabled candidate (`--tune-trials`). 0 = 20 when any slot has HPO, else off. Max 200. |
| `meta` | Optional gate. Default from the objective; `nq-50pt-xgb-tcn` = `meta_stack` logistic, `walk_forward_OOF`, `linear_weights`, τ=0.55, 5 folds, 120-bar embargo |
| `walk_forward` | Expanding calendar designer (ADR-042). `enabled` + `start`/`stop` + train/test `n`+unit (`days`/`weeks`/`months`). Optional `val` (`cut_from`: train\|test). Default: 2025-05-01→2026-06-30, 2 months/2 months. Disable = objective YAML equal-count folds. |
| Artifacts | Launch scratch: `ops/webui/custom_tests/<objective>/<run_id>.{board,manual,topology,walk_forward}.yaml`. Run output: `research/runs/<objective>/<run_ts>/models/<cand>/`. `research/custom_test/` is gone |
| CLI | `python -m src.search.controller --objective … --manual-spec … --topology-overlay … --walk-forward-overlay … --no-ensemble --experiment-type custom_test` (`--tune-trials N` when a slot enables HPO). WebUI adds the flags; specs under `ops/webui/custom_tests/` also infer `custom_test` |

Feature builder (compose a new column from existing names) is **not** in this
schema — parked.

---

## `paths`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `paths.data_dir` | string | `/opt/vtech/data` | Root data directory |
| `paths.checkpoint_dir` | string | `/opt/vtech/checkpoints` | Model checkpoint root |
| `paths.parquet_dir` | string | `/opt/vtech/data/parquet` | Parquet feature cache |

---

## `data`

Data loading, session filtering, and preprocessing parameters.

| Key | Type | Default | Env var override | Description |
|-----|------|---------|-----------------|-------------|
| `data.feature_timestep` | string | `5s` | `VTECH_FEATURES_TIMESTEP` | Bar aggregation interval |
| `data.training_start` | string | `"08:00"` | `VTECH_TRAINING_HOURS_START` | Training-data window open (CST) — pre-market start, NOT the market open reference |
| `data.training_end` | string | `"11:00"` | `VTECH_TRAINING_HOURS_END` | Training-data window close (CST) |
| *(engine only)* | string | `"08:30"` | `VTECH_MARKET_OPEN` | CME RTH / NYSE open (CST). Zero-point for `tc_minutes_into_session`, `tc_block_open_*`, `tc_or{5,15,30}_*`, `tc_session_progress`. Not in `config.yaml`; set via env var or `FeatureEngine(market_open=...)`. |
| `data.timezone` | string | `America/Chicago` | `VTECH_TIMEZONE` | Timezone for session hours |
| `data.scaler_type` | string | `robust` | — | Feature scaler: `robust`, `standard`, or `minmax` |
| `data.bbo_primary_only` | bool | `true` | `VTECH_BBO_PRIMARY_ONLY` | Restrict `bbo_1s` to lead contract (OOM mitigation for high-volume dates) |

---

## `labels` (legacy config.yaml / env overrides)

> **Prefer the objective doc** (`config/objectives/<id>.yaml` `labels` block). Trainer and search
> load labels from the active objective (ADR-018). Keys below remain as env/YAML fallbacks for
> older entry points.

Detection-2 / TBM labelling: directional when the forward path meets the objective barriers;
everything else is NO_SIGNAL (masked or scored as flat depending on the path).

| Key | Type | Default | Env var override | Description |
|-----|------|---------|-----------------|-------------|
| `labels.big_move_threshold` | int | `50` | `VTECH_LABEL_THRESHOLD` | Profit barrier / min move (NQ points) — objective `move_threshold` |
| `labels.mae_tolerance` | float | `5` | `VTECH_MAE_TOLERANCE` | Stop barrier / max adverse excursion — objective `mae_tolerance` |
| `labels.target_observation` | string | `candle_close` | — | Triple-barrier target source: `candle_close`, `candle_hilo`, or `tbbo`. Stops remain 5s closes. `tbbo` uses lossless 5s `cs_high` / `cs_low` extrema sourced from trade prints, so it is equivalent to `candle_hilo` on the canonical 5s store. Changing it requires rebuilding label sidecars and fold caches (ADR-106). |

15s dual-binary labels are selected from the normalized objective parameters,
not a manual directory name. The resolver hashes the complete label contract
and loads `sources/labels/15s/<contract_id>/`; the builder emits compressed
`labels_15s_partN.parquet` parts bounded near 100 MB. Each native-TF run writes
`label_contract.json` with the chosen contract, path, digest, and parts
([ADR-107](../decisions/ADR-107-contract-addressed-label-sets.md)).
| `labels.forward_windows` | list[int] | `[60, 120, 180, 360]` | `VTECH_FORWARD_WINDOWS` | Forward horizons in bars (modeling; horizon max still on objective) |

---

## `features`

Feature-group toggle: controls which feature groups are included in the training feature matrix. The Parquet cache always stores all features — disabling a group here excludes it at training time only, with no cache rebuild required.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `features.enabled_groups` | dict[str, bool] | all `true` | Map of group name → enabled flag. Any group absent from the YAML defaults to `true`. See group list below. |

### `features.enabled_groups` — group reference

| Group key | Column prefix | Features | Description |
|-----------|--------------|---------|-------------|
| `book_pressure` | `bp_` | 24 | Bid/ask imbalance, depth, fleeting-quote CV, spread |
| `order_flow` | `of_` | 15 | Cumulative volume delta, trade-side divergence |
| `trade_location` | `tl_` | 16 | Size-weighted 5s location/sweep + 30s/5m/30m MAs and pair z-score / short−long |
| `same_side` | `ss_` | 3 | Consecutive same-direction trade runs |
| `large_trade_cvd` | `lt_` | 4 | CVD restricted to prints ≥10 contracts |
| `trade_arrival` | `ta_` | 3 | Trade arrival rate and acceleration |
| `microstructure` | `ms_` | 7 | VPIN, Kyle's λ, Amihud illiquidity |
| `realized_vol` | `rv_` | 6 | Realised volatility + vol-of-vol |
| `sub_bar_dynamics` | `sd_` | 8 | Intra-5s bid/ask range, trade density |
| `volume_profile` | `vp_` | 5 | POC, value area, volume-at-price |
| `iv_surface` | `iv_` | 14 | ATM IV, term slope, skew (options chain) |
| `dealer_gex` | `gx_` | 4 | Gamma exposure per strike (options chain) |
| `daily_context` | `dc_` | 4 | Prior-day range, gap, session stats |
| `wavelets` | `wv_` | 14 | Multi-scale wavelet decomposition (1-min candles) |
| `candle_structure` | `cs_` | 18 | Candle body/wick ratios, patterns (1-min + multi-TF) |
| `time_context` | `tc_` | 24 | Time-of-day, session blocks (premkt/hourly), opening range |
| `vwap` | `vw_` | 5 | VWAP distance and band position |
| `cross_asset` | `ca_` | 12 | ES futures correlation and lead-lag divergence |
| `economic_calendar` | `ec_` | 4 | Scheduled macro event proximity flags |
| `hawkes_clustering` | `hp_` | 3 | Self-exciting event clustering intensity |
| `higher_timeframe` | `ht_` | 5 | Back-adjusted daily OHLC context |

**Usage**: to run an ablation without options-derived features, set `iv_surface: false` and `dealer_gex: false` — no Parquet rebuild needed.

---

## `model`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model.sequence_length` | int | `128` | LSTM lookback window in bars (10.7 min at 5s) |
| `model.num_classes` | int | `3` | Softmax output classes: BIG_DOWN=0, NO_SIGNAL=1, BIG_UP=2 (3-class since Phase 17G; [ADR-008](../decisions/ADR-008-3class-softmax-class-weight.md)) |
| `model.class_weight_directional` | float | `15.0` | Inverse-frequency class weight applied to BIG_DOWN and BIG_UP during training. NO_SIGNAL weight is fixed at 1.0. Approx `1 / signal_rate`. |
| `model.lstm_units` | list[int] | `[128]` | Units per LSTM layer (first layer; second layer = first ÷ 2) |
| `model.attention_heads` | int | `4` | Number of multi-head self-attention heads |
| `model.dense_units` | list[int] | `[64]` | Dense layer units after attention |
| `model.dense_dropout` | float | `0.30` | Dropout rate on dense layers |
| `model.use_attention` | bool | `true` | Enable multi-head attention after LSTM stack |

---

## `training`

| Key | Type | Default | Env var override | Description |
|-----|------|---------|-----------------|-------------|
| `training.seed` | int | `42` | `VTECH_SEED` | Global random seed for reproducibility |
| `training.batch_size` | int | `64` | — | Training batch size |
| `training.epochs` | int | `30` | — | Max training epochs (early stopping may cut short) |
| `training.learning_rate` | float | `0.001` | — | Initial Adam learning rate |
| `training.early_stop_patience` | int | `5` | — | Epochs without val improvement before stopping |
| `training.early_stop_min_delta` | float | `0.001` | — | Minimum change to reset patience counter |
| `training.lr_reduce_patience` | int | `3` | — | Epochs before reducing learning rate |
| `training.lr_reduce_factor` | float | `0.5` | — | Learning rate reduction factor |
| `training.min_lr` | float | `0.00001` | — | Minimum learning rate floor |
| `training.scaler_type` | string | `robust` | — | Feature scaler: `robust` (RobustScaler) or `standard` |

---

## `logging`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `logging.level` | string | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## `.env` — Secrets

The `.env` file is never committed to version control (permissions: 600). Required variables:

| Variable | Description |
|----------|-------------|
| `DATABENTO_API_KEY` | Databento API key |
| `PGHOST` | PostgreSQL host (default: `localhost`) |
| `PGPORT` | PostgreSQL port (default: `5432`) |
| `PGDATABASE` | Database name (default: `vtech_market`) |
| `PGUSER` | Database user |
| `PGPASSWORD` | Database password |

---

## Runtime-Only Environment Variables

These variables are set at invocation time (not in `.env`) for specific operational scenarios:

| Variable | Purpose |
|----------|---------|
| `VTECH_BBO_PRIMARY_ONLY=1` | Restrict `bbo_1s` loader to lead contract only — use for OOM-prone dates (Nov 21, Dec 15-18, 2025) |
| `VTECH_SEED=N` | Override global seed |
| `VTECH_MAE_TOLERANCE=N` | Override `AppConfig` MAE tolerance only (`MLConfig` reads directly from YAML) |

---

## Python API

```python
from src.common.config import AppConfig

cfg = AppConfig.load()          # reads config.yaml, applies env-var overrides
print(cfg.features.timestep)   # "5s"
print(cfg.training.seed)       # 42

# Training pipeline — reads config.yaml only (no env-var overrides)
from src.training.config import MLConfig

ml = MLConfig.from_env()               # reads /opt/vtech/config.yaml by default
print(ml.model.lstm_units)             # [128] (from config.yaml)
print(ml.training.early_stop_patience) # 5 (from config.yaml)

# Override YAML path (e.g. for testing):
from pathlib import Path
ml = MLConfig.from_env(config_path=Path("/opt/vtech/config_test.yaml"))
```

---

## Related Docs

- [ADR-006](../decisions/ADR-006-config-yaml-single-source.md) — rationale for centralizing config
- [SYSTEM.md](../architecture/SYSTEM.md) — core parameter reference table
