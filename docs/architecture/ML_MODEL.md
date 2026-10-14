# ML Model

> **Present**: Many catalog families (trees, TCN, LSTM, patch Transformer, …) share one
> plugin contract `{ts, direction, confidence}`. `make shap` TreeSHAPs 3-class
> boosters and dual-binary `DualBinaryPair` / OvR wrappers (per-head unwrap). The **objective YAML** picks labels, KPI,
> session, and eligible families — there is no single “main” architecture. Live pin is
> `nq-50pt-econ` (TBM 5s-close + economics meta). Research tracks: `nq-50pt-xgb-unique`
> (causal trees + ADR-047 inference policy, staging only), `nq-50pt-path-hazard`
> (ADR-053 honest 1s-path labels + TCN/`patch_hazard` heads, not trained yet), and
> `nq-50pt-econ-lg-st` (ADR-066 dual `P_final_long`/`P_final_short`, native 15s/1m/5m
> bases + 1h/1d `regime_v1` pack on bases **and** meta [ADR-095](../decisions/ADR-095-regime-context-fire-aux.md);
> OOF composer ADR-092; joint search
> ADR-093 / [ADR-094](../decisions/ADR-094-15s-lstm-temporal-base.md) trains
> YAML bases `lstm_15s` (Attention-LSTM, seq_len=60) / `xgb_1m` / `xgb_5m` then meta;
> fire `labels.fire: 0.40`;
> hive dirs ADR-070, 15s files ADR-075;
> Custom Test dual-binary sequence heads ADR-080; aux path labels + per-head
> target geometry ADR-082, YAML-opt-in ADR-084, side-specific aux y
> [ADR-086](../decisions/ADR-086-side-specific-aux-targets.md), decision
> rank [ADR-088](../decisions/ADR-088-aux-side-rank-decision.md);
> Time_to_TGT NaN on timeout [ADR-098](../decisions/ADR-098-time-to-tgt-nan-on-timeout.md);
> 08:30–11:00 CT + onset uniqueness + session VP [ADR-099](../decisions/ADR-099-lg-st-morning-onset-session-vp.md)), and
> `nq-50pt-tbm-opt` (ADR-072 3×3 expected-cost train loss, research only). WebUI
> Custom Test (ADR-035/036/040/042/043/044/071/074). GPU jobs emit a live heartbeat
> + per-fold OOF ([ADR-073](../decisions/ADR-073-gpu-job-progress-heartbeat.md)).
> Keras serving load: first-party
> `.keras` via `load_keras_model` ([ADR-059](../decisions/ADR-059-keras-serializable-last-timestep.md);
> Keras 3.15 GPU LSTM files: [ADR-077](../decisions/ADR-077-keras-15-initializer-forward-compat.md);
> dual-binary LSTM with `lg_*`/`st_*` packs `long_ctx`/`short_ctx` at serve).
> **External Models** (`src/external_models`,
> [ADR-058](../decisions/ADR-058-external-models-pine-sar-ema.md),
> [ADR-061](../decisions/ADR-061-psar-ema-exit-at-sar.md),
> [ADR-062](../decisions/ADR-062-ext-model-split-and-stack.md),
> [ADR-096](../decisions/ADR-096-ext-timed-psar.md)) are a separate
> category: Parabolic SAR, 20 EMA, ZigZag, and Timed PSAR engines scored on
> `databento.tbbo_1min`, stackable with shared book knobs — not catalog families.  
> **Supersedes**: LSTM-as-only-path narrative; unrolled LSTM on the A10 (ADR-043). TBM
> 5s-close remains the live-pin labeller; vol-normalised 1s-path labels (ADR-053) are a
> **parallel** dataset for new objectives, not a silent replacement of every YAML.  
> **Next**: Serving asof for `tf_oof_meta` (not a Live/Backtest bundle yet).
> Train/score `path_tcn` on `nq-50pt-path-hazard` (GPU). SSL pretrain;
> Encoder B only if the TCN is not enough. No VIX. Do not promote.  
> **Source of truth for**: label *schemes*, training/search machinery, evaluation
> contract, model registry — not a single network diagram.  
> **Evaluation contract note**: leaderboard economics are label-implied and close-only; the
> path-realised account simulator (§4.6, [ADR-052](../decisions/ADR-052-path-realised-account-simulator.md))
> is the honest counterpart and reports materially lower numbers.

---

## 1. Problem Formulation

Every supervised family in the catalog consumes a **sparse integer label** on a
timestamped feature row. Encoding is shared so plugins and KPI stay interchangeable:

| Class | ID | Meaning |
|-------|----|---------|
| `BIG_DOWN` | 0 | Qualifying downward first-touch (or Detection-2 down) |
| `NO_SIGNAL` | 1 | Neither side qualified (flat / censored / stop-first) |
| `BIG_UP` | 2 | Qualifying upward first-touch (or Detection-2 up) |

**What “qualifying” means is not global** — it is the objective’s `labels` block
([ADR-018](../decisions/ADR-018-objective-doc-single-source-labels.md)):

| `labels.scheme` | Path | Horizon (typical) | Used by |
|-----------------|------|-------------------|---------|
| `detection_2` | 5s close, order-agnostic peak (§2) | 5–30 min windows | `nq-50pt` |
| `triple_barrier` | 5s-close stop; configurable close or TBBO-extrema target (§2.1) | 10 min | live pin `nq-50pt-econ`, `nq-50pt-xgb-unique` |
| `path_vol_normalized` | 1s high/low path, vol-normalised barriers (§2.2) | 600 s | `nq-50pt-path-hazard` |
| `triple_barrier` + `encoding: dual_binary` | same 5s TBM, subsampled to 15s; `y_long`/`y_short` plus 10 path-quality aux columns (ADR-082) | 10 min | `nq-50pt-econ-lg-st` |

NO_SIGNAL is still a labeller output. How a *family* treats it (mask, 3-class CE,
hazard joint class, …) is a plugin choice. Session window, costs, and KPI gates
also live on the objective, not in this section.

Canonical serving output for **every** family: `{ts, direction, confidence}` with
`direction ∈ {up, down, flat}`.

---

## 2. Label Generation (Detection-2 + MAE filter)

```python
max_future = max(price[t+1 .. t+60])
min_future = min(price[t+1 .. t+60])

up_move   = max_future - price[t]   # peak excursion upward in next 60 bars
down_move = price[t] - min_future   # peak excursion downward in next 60 bars

if up_move   >= big_move_threshold AND down_move < mae_tolerance: → BIG_UP   (raw 2)
if down_move >= big_move_threshold AND up_move   < mae_tolerance: → BIG_DOWN (raw 0)
else                                                              : → NO_SIGNAL (raw 1, masked)
```

**Source of truth** — the trade objective doc [`config/objectives/nq-50pt.yaml`](../../config/objectives/nq-50pt.yaml) `labels` block (config.yaml only references which objective is active via `objective_id`). Both the trainer and the model-selection search read label params from here, so they cannot drift:
- `move_threshold` → `big_move_threshold` = 50 NQ points
- `mae_tolerance`  = 5 NQ points

The MAE (Maximum Adverse Excursion) condition filters spike-and-retrace and choppy moves: a 50-point rise that's preceded by a 10-point dip becomes NO_SIGNAL because the trader would have been stopped out before the move materialised. Only clean moves where the price moves toward the target without first going against you are labeled directional.

Implementation: `src/training/labels.py` (`generate_labels`, `_classify_detection_2`).
Config: `src/training/config.py` (`LabelConfig`, populated from the objective via `_labels_from_objective`).

> See [ADR-007](../decisions/ADR-007-detection-2-mae-filter.md) for the detection-2 + MAE design rationale; [ADR-004](../decisions/ADR-004-adaptive-labels.md) (superseded) covers the prior 3-class adaptive-threshold scheme.

### 2.1 Alternative scheme: Triple Barrier Model (`labels.scheme: triple_barrier`)

The objective doc may select a **path-dependent first-touch** scheme instead ([ADR-023](../decisions/ADR-023-triple-barrier-label-scheme.md)); Detection-2 stays the default. Used by [`config/objectives/nq-50pt-tbm.yaml`](../../config/objectives/nq-50pt-tbm.yaml) (`move_threshold=50`, `mae_tolerance=6`):

```python
# per bar t, first offset k in [1, window] touching each barrier:
BIG_UP   iff first_touch(+move_threshold) < first_touch(-mae_tolerance) and ≤ window
BIG_DOWN iff first_touch(-move_threshold) < first_touch(+mae_tolerance) and ≤ window
else NO_SIGNAL   # stop barrier first, or time barrier (window end) reached
```

`labels.target_observation` controls only the target source: `candle_close`
(default), `candle_hilo`, or `tbbo`. Stops remain 5s closes. Canonical 5s
`cs_high` / `cs_low` are lossless extrema of TBBO trade prints, so `tbbo` and
`candle_hilo` are identical at the 5s label grid. The setting is part of the
persisted label contract and cache key; it requires sidecar rebuild before a
new run ([ADR-106](../decisions/ADR-106-configurable-tbm-target-observation.md)).

15s dual-binary sidecars are contract-addressed: the normalized labels block
selects `sources/labels/15s/<contract_id>/`, whose compact parquet parts are
bounded near 100 MB. Every native-TF research run writes a `label_contract.json`
provenance snapshot referring to that immutable contract set ([ADR-107](../decisions/ADR-107-contract-addressed-label-sets.md)).

Unlike Detection-2's order-agnostic window filter, first-touch credits a captured winner even if price later reverses, and rejects a target printed only after the stop was breached. TBM admits a superset of Detection-2's directional bars at equal params (~21.5% vs ~17.6% directional on a sample day at 50/6 vs 50/5). Same `{0,1,2}` encoding; the fold builders (`src/search/fold_cache.py`) dispatch on the objective's `labels.scheme` and probe the parquet bar interval, so 5s feature parquets and 1s raw parquets (`data/parquet_raw1s/`, built by `scripts/build_raw1s_dataset.py`) both label correctly. Implementation: `generate_labels_tbm` / `_classify_triple_barrier` in `src/training/labels.py`; tests: `tests/test_tbm_labels.py`.

### 2.2 Vol-normalised 1s-path labels (`src/training/path_labels.py`)

[ADR-053](../decisions/ADR-053-vol-normalized-path-labels-and-encoder-ablation.md).
For any 50/6-equivalent geometry, labelling on 5s closes is optimistic: a 5s NQ bar that
traded 9 pt against the entry and closed +1 pt is scored as a clean winner even though
the 6-pt stop was struck intrabar. `src/training/path_labels.py` walks the **1s high/low
path** (`parquet_path1s`) and resolves first-touch honestly.

* **`mode="vol_normalized"`** (canonical): barriers are `target_r × R` and `stop_r × R`
  where `R` is a **causal** yardstick — the median of the previous 5 sessions' median
  10-minute range in the entry window. Defaults `target_r=0.743`, `stop_r=0.0891` are the
  fixed 50/6 geometry divided by 2026 realised R (67.3 pt), so the *event* is the same in
  every era while the underlying range grows from ~6 pt in 2011 to ~67 pt in 2026.
* **`min_stop_ticks = 8`** (tick=0.25 pt) is the absolute microstructure floor. Sessions
  whose normalised stop falls below the floor are labelled `usable=False` rather than
  silently training on fills inside the spread. Per-year `usable_frac` under the canonical
  geometry: 0% for 2011–2017 (stop 1.7–3.2 ticks), 23% / 9% for 2018 / 2019, and ≥79% from
  2020 on.
* **Directional rate is stationary at 17–19% every year** — the geometry is scale-invariant
  and the binding constraint is the tick floor, not label incidence.
* **Dense targets** alongside the sparse `{0, 1, 2}` label: `event_{up,down} ∈ {0 censored,
  1 target, 2 stop}`, `bucket_{up,down} ∈ [0, 20]` (30 s each × 600 s horizon), `mfe/mae`
  in points and R. The sparse label yields one supervised value per episode; the dense
  targets are defined on *every* bar and are what let 2011–2017 supervise via SSL /
  hazard pretraining even where `usable=False`.
* **Tie policy on the 1s path** (`PathLabelConfig.tie_policy`): `stop_first` (default,
  conservative) or `target_first` (optimistic bound) when one 1s bar straddles both
  barriers. This is a labelling choice. **`close_only` is not a third PathLabelConfig
  value** — it lives on the path simulator ([ADR-052](../decisions/ADR-052-path-realised-account-simulator.md))
  and means “ignore high/low; only the close can hit a barrier,” i.e. reproduce the
  leaky 5s labeller as a bound. At 1s resolution `ambiguous_rate ≤ 5×10⁻⁵` under this
  geometry, so the 1s tie policy is not a live concern (it would be for 5/5).

**Persisted dataset.** `data/sources/features/parquet_pathlabels/`
(`src/common/data_paths.py::parquet_pathlabels_dir()`), one parquet per session.
Build finished 2026-08-21: **3,906** `labels_*.parquet` (~554 MB) covering 2011-01-10 →
2026-06-30, 35.1 M entry rows. Of 3,919 `parquet_path1s` sessions, 5 are unlabellable
(causal-R warm-up) and 8 are closed/early-close days with no 08:30–11:00 CT bars.
Geography: `_geography.csv` next to the parquets. Built by
`scripts/features/build_path_labels.py --out …`.
**Supervision start = 2018-01-02** with a strict `usable=True` filter; **pretraining pool
= 2011-01-03 onward** using the dense targets only (those parquets exist even where
`usable=False`). 5s-join trainers cannot read 2011–2017 until 5s features exist for
those years; SSL / path-only training can. Label **entry** window is
**08:30–11:00 America/Chicago** (9,000 1s rows on a full day). Any walk-forward
**must** set `purge_seconds = 600` (duration of the 10-minute horizon — correct at
5s or 1s). `purge_bars` is a raw row count and is not interchangeable across
timesteps. Live inference on the fixed +50/−6 geometry gates at `R ≥ 67.3 pt`.

**Shared heads** (`src/models/heads.py`). Competing-risks hazard: 41-class joint
`{no-touch} ∪ (target × 20 buckets) ∪ (stop × 20 buckets)` per side, sparse CE on
logits; inference integrates target-bucket mass to `P(target_up first)` /
`P(target_down first)` and emits `{direction, confidence}` (tie → `flat`). Quantile
head: pinball at `{0.1, 0.5, 0.9}` on the four `*_r` excursion columns. SSL head not
built.

**Encoders (recovered spec, not yet built).** Both read **`parquet_path1s`**
(1s OHLC+volume, 07:00–15:15 CT, 2011→2026), ATR-normalised so 2011 and 2026 share
a distribution. Labels join from `parquet_pathlabels` on `ts`.

* **A — 1s TCN (`path_tcn`, wired):** 10 dilated causal layers, dilations `1…512`,
  kernel 2, receptive field 1,024 s, last-step embedding, heads 1–2. Not trained.
  Ship this if it matches B.
* **B — `patch_hazard`:** 96 tokens from the 1s path (4 resolutions × 24 patches:
  120 s@5 s, 600 s@25 s, 3,600 s@150 s, 8 h@20 m), 7 ATR-normalised features per
  patch, static context token (tod / ATR₁₀ / ADR-051 distances; no VIX), 4-layer
  Transformer ~0.9 M. Same heads.

A 5s-feature join (`tcn_hazard` / 5s `patch_hazard` on `sequence_baseline`) still
exists in the repo but is **not** this ablation. `nq-50pt-path-hazard` launches
`path_tcn` on `search.input: path1s`. Tests: `tests/test_path_labels.py`,
`tests/test_heads.py`, `tests/test_path1s_encoder.py`. Legacy `generate_labels_tbm`
stays for existing objectives.

**Meta-labeling (active under `nq-50pt-econ`, ADR-027)**: `src/training/meta_labels.py` builds binary win/lose labels for a primary side; `src/search/meta_compose.py` cross-fits a trade/no-trade gate on primary OOF when the objective sets `search.topology.mode: primary_meta`. Legacy objectives (`nq-50pt`, `nq-50pt-tbm`) keep the single-model path. Economics ranking uses `kpi.primary: net_expectancy` with precision floors — see [ADR-027](../decisions/ADR-027-economics-meta-sparse-objective.md).

**N-base output-meta (`nq-50pt-xgb-tcn`, ADR-033)**: walk-forward any `topology.bases` on the same TBM 50/6 labels and assigned feature view; `src/search/output_meta.py` aligns OOF on `ts` and fits a binary gate (default `logistic_regression`) on those prediction vectors only. Side is resolved from the bases; the gate only approves/rejects. Sequence families window 2-D folds via `src/models/sequence_windows.py` using actual `F` after the view. LSTM and TCN `fit` via `src/models/dl_memory.py` (Keras `Sequence` + val stride-cap + chunked predict) so the A10 never receives one eager fold tensor. Serve/finalize loop `bases` then `meta.pkl` — no family-name branches.

**Native-TF dual-head OOF meta (`nq-50pt-econ-lg-st`, ADR-066 / [ADR-092](../decisions/ADR-092-tf-oof-meta-composer.md))**: second-stage composer, not a joint search. Per-store Custom Test bases write dual-binary OOF; `src/search/tf_oof_meta.py` asof-aligns 1m/5m (and 1h/1d context) onto the 15s grid, then walk-forward two XGBs for `p_final_long` / `p_final_short` (block 0 warmup, last calendar block = untouched test, embargo 40 × 15s). Automated search stays fail-closed (one store per fold cache). CLI `python -m src.search.tf_oof_meta --base id=RUN_TS`. Joint-search last-fold stack is a Backtest serving bundle (`meta.pkl` + `bases/`, [ADR-103](../decisions/ADR-103-tf-oof-meta-serving-asof.md)); not Live. Live pin unchanged. First compose uses 15s LSTM `20260828T175240Z` + 1m XGB `20260828T200724Z` + 5m XGB `20260828T192637Z` (skip collided `20260828T200039Z`). Report bundle `20260828T210653Z` (`class_balance=sqrt`): OOF long AUC 0.712 / short 0.677, Prec@0.5 0.14, NE@0.5 +1.86; serving τ=0.1 over-fires (NE −1.22). No-weight `20260828T210839Z` is the τ=0.1 economics variant (NE +0.71 / holdout +1.55) but `p_max` 0.40. Morning-session joint search `20260829T182514Z` (ADR-098 / ADR-099, fire 0.40): LSTM 23-col GPU (HTF not on the A10 hive), 1m 84-col, 5m 77-col incl. `svp_*`, meta 28-col `serving: false`. Finite OOF meta long AUC 0.629 / short 0.594, n@0.40=16,095 NE +2.04; holdout n=3,862 NE +3.09. LSTM `p_max` 0.38 — 0 fires at 0.40. Gates not met; do not promote.

**WebUI Custom Test (ADR-035 / ADR-036 / ADR-040 / ADR-042 / ADR-043 / [ADR-071](../decisions/ADR-071-optional-champion-gate.md) / [ADR-074](../decisions/ADR-074-custom-test-saved-feature-views.md))**: from an objective, operators pick catalog families, click a model card to view catalog `params_default` and edit them for the run, tick knobs from that family's `search_space` and enable HPO (Optuna `--tune-trials`; omitted `search_space` still means no auto-tuning per ADR-019), drag named views / prefix groups / columns onto each model (views and groups expand to per-column chips; **save as view** stores the subset in `ops/webui/custom_feature_views/` for the feature bank), flag units for ablation (`2^k` candidates), optionally enable a meta gate, and design an expanding calendar walk-forward (start/stop, train/test period in days/weeks/months, optional val). The board compiles to an ADR-019 manual spec plus `--topology-overlay` and `--walk-forward-overlay` (YAML on disk is not edited). Custom Test does **not** apply the champion gate (`run.json` `status: report`; every fitted model is written). Search / HPO / scoping pick a champion only when `kpi.champion_gate: true`. Custom Test **max_rows** default is **off** (full train). Operators can cap last-N consecutive train bars per fold (catalog suggestion 100k); OOF is never capped. Automated search still uses catalog `search_max_rows`. Default walk-forward is 2025-05-01 → 2026-06-30 with 2-month train/test (6 folds); disable the designer to keep the objective equal-count splitter. Last-fold bundles (bases + `meta_stack`) are written to `research/custom_test/<obj>/<run_ts>/models/<cand>/` and listed on Backtest. Default board for `nq-50pt-xgb-tcn` is XGBoost + TCN on their assigned views with the yaml `meta_stack` logistic. Feature builder (compose a new column from existing names) is not implemented. TCN `pooling=last` (and `path_tcn` / `patch_hazard` gathers) use registered Keras layers, not Python-lambda `Lambda`; serving loads first-party `.keras` via `src.models.keras_layers.load_keras_model` ([ADR-059](../decisions/ADR-059-keras-serializable-last-timestep.md)). GPU Keras 3.15 LSTM files deserialize on the CPU 3.13 venv by dropping unknown initializer keys ([ADR-077](../decisions/ADR-077-keras-15-initializer-forward-compat.md)).

**Research objective `nq-50pt-xgb-unique` (ADR-045 / [ADR-046](../decisions/ADR-046-causal-tbm-topgroups-proof.md) / [ADR-047](../decisions/ADR-047-inference-uniqueness-open-policy.md))**: TBM 50/6/10m, **cost 1 pt**, calendar 6-fold WF, train uniqueness 24 bars, view `tbm_topgroups`. Causal 3-class overlapping OOF (`20260821T110058Z`) NE 6.85 / HR 0.247 / n=1035 / MDD −553. **Inference policy** (08:30–10:00 CT + 10-minute cooldown @ τ=0.65, `20260821T113200Z`): n=58 HR 0.328 NE **11.34** MDD **−49**; walk-forward τ n=49 NE 11.29 MDD −42. Event-meta XGB veto (`src/search/event_meta.py`) has OOF AUC ~0.5 and is not the policy. A binary **side head** on directional labels only (`src/search/side_head.py`, [ADR-048](../decisions/ADR-048-binary-side-head-rejected.md) `20260821T115346Z`) has OOF AUC 0.49; attaching it to the timer cuts NE 11.7 → 2.8. A **±10 TBM side** model on all 216 cols joined to the 50/6 timer ([ADR-049](../decisions/ADR-049-tbm10-side-transfer-rejected.md) `20260821T120555Z`) has 10/10 AUC 0.51 and transfer 0.53; the join stays 47/1 longs. Clears n and MDD; misses NE 20 / HR 0.55. Live pin remains `nq-50pt-econ`. Two-sided direction is not in L1 causal TBM 50/6.

**GPU Attention-LSTM plugin (ADR-043 / ADR-044 / [ADR-080](../decisions/ADR-080-dual-binary-sequence-heads.md) / [ADR-082](../decisions/ADR-082-aux-labels-and-head-target-geometry.md))**: `layers.LSTM(..., unroll=False)` so cuDNN runs on the A10; catalog `batch_size=128`, `epochs=15` (last GPU Custom Test), `class_balance=sqrt` per-sample loss weights (same helper as TCN — unweighted CE on ~87% flat is always-flat). Dual-binary (`params.head: dual_binary`) is two named sigmoid outputs `p_long` / `p_short` with independent BCE and per-head `class_balance` on LSTM, TCN, CNN, and Transformer; diagnostics lead with `labels: ["long", "short"]` and per-head AUC. When the objective lists `labels.aux` (`nq-50pt-econ-lg-st` only today, [ADR-084](../decisions/ADR-084-objective-opt-in-aux-heads.md)) and those columns are on the label parquet, the LSTM shared encoder sees only the tape columns; last-bar `long_ctx` / `short_ctx` are concatenated onto that side's trunk only; each trunk Huber-predicts its MFE / MAE-before / time-to-target plus `forward_return_*` (z-scored per fold; **short trunk trains on `−z(return)`**, [ADR-086](../decisions/ADR-086-side-specific-aux-targets.md); `loss_weight` 0.15 / 0.05). Shared `MAE_before_target_or_MFE` is parquet-only. A `side_rank` output `p_long / (p_long + p_short)` trains on which side had the larger 10m MFE (`loss_weight` 0.3, sample weight `clip(|Δ|/50, 0, 1)`, [ADR-088](../decisions/ADR-088-aux-side-rank-decision.md)) so aux regularizes **decision order**. Canonical fire is argmax of the two sigmoids **and** `max(p_long, p_short) ≥ labels.fire` (library default 0.1, [ADR-089](../decisions/ADR-089-dual-binary-argmax-fire.md); `nq-50pt-econ-lg-st` is **0.40**, [ADR-095](../decisions/ADR-095-regime-context-fire-aux.md)); per-head `confusion_at_fire` is still 0.5 as a calibration diagnostic. lg-st aux is MFE / MAE_b4 / 10m return / `Time_to_TGT_*` (seconds to first ±50; **NaN** if the 10m window never hits — [ADR-098](../decisions/ADR-098-time-to-tgt-nan-on-timeout.md)) plus binary `*_stop_first` (MAE≥6) and `side_rank`; trees use MFE near-miss weights. Other LSTMs (3-class, or dual-binary without `labels.aux`) do not grow aux heads. Serving still uses only `p_long` / `p_short`, but Live/Backtest must pack the three Keras inputs (`window_pack_for_model`) — a single `(N, T, 16)` window raises `expects 3 input(s)`. `EarlyStopping(patience=10)`; each walk-forward fold rebuilds the Keras graph from the **assigned** feature count (Custom Test does not expose `n_features`). `seq_len` is lookback in 5s bars (default 128 = 10.7 min on 5s; 32 min on 15s). `fit` stays Keras-`Sequence` fed (`src/models/dl_memory.py`). Custom Test model-card params list `class_balance` as a categorical (`None` / `balanced` / `sqrt`). Keras stays `verbose=0`; epoch/fold progress is the ADR-073 heartbeat (`ops/gpu_jobs/<run_id>/progress.json`) plus `folds/fold_KK.json` confusion — not a journal `\r` bar.

**GPU job progress ([ADR-073](../decisions/ADR-073-gpu-job-progress-heartbeat.md))**: `src/training/job_progress.py` is a process-local session started by `job_exec`. Plugins write fold/epoch into `progress.json` and checkpoint each fold’s OOF. The runner tees `job_exec` to `job.log` + journal. CPU `health_monitor` SSHes the heartbeat onto `ops/gpu_health/latest.json`. Watch: `make gpu-monitor` (snapshot), `make gpu-watch` (15s), `make gpu-follow` (journal). The probe shows the **current** job only — leftover `uploading` 100% files on the GPU disk are omitted.

---

## 3. Catalog families (not a single network)

The search catalog (`src/models/catalog.py`) is the list of *types* an objective may
enable. Trees, TCN, LSTM, `patch_hazard`, clustering, … are peers. This section
documents the **Attention-LSTM** family as it is implemented today — it is not the
program architecture. Hazard-head wiring for TCN / patch Transformer is §2.2.

### 3.1 Input (sequence families)

- Shape: **(seq_len, F)** — default `seq_len=128` (10.7 min of 5s bars) × the
  feature count assigned by the objective view.

### 3.2 Attention-LSTM (catalog `attention_lstm`)

```
Input (128, F)
  → LSTM 64 units  (return_sequences=True, dropout=0.2)
  → LSTM 32 units  (return_sequences=True, dropout=0.2)
  → Multi-Head Self-Attention (4 heads)
  → LayerNormalization
  → GlobalAveragePooling1D
  → Dense 64 + ReLU + Dropout(0.3)
  → Dense 32 + ReLU + Dropout(0.3)
  → Dense 2 + Softmax  →  [P(BIG_DOWN), P(BIG_UP)]
```

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 64 |
| Max epochs | 100 (early stop patience=10) |
| Optimizer | Adam (lr=0.001, reduce-on-plateau) |
| Loss | Sparse categorical crossentropy |
| Sample weighting | NO_SIGNAL bars → weight 0; directional bars → weight 1 |
| Scaler | `RobustScaler` fit on train split, applied to val/test |

### 3.3 Data Split (temporal, chronological)

```
Train  70%  → Model training (NO_SIGNAL masked via sample_weight)
Val    15%  → Early stopping + LR schedule (NO_SIGNAL masked)
Test   15%  → Final evaluation; NO_SIGNAL bars dropped before scoring
```

No shuffling — temporal ordering is preserved.

---

## 4. Training Pipeline

### 4.1 Sequence Generation

`src/training/data_pipeline.py`:
- Loads Parquet files for the date range
- Excludes `_`-prefixed metadata columns and the label column
- Sliding-window sequences of length `seq_len=128`
- Each sample = (128 lookback bars, label of the final bar)

### 4.2 Label Remapping & Masking (in trainer)

```python
y_raw  ∈ {0=BIG_DOWN, 1=NO_SIGNAL, 2=BIG_UP}      # from labels.py
y_mapped = where(y_raw == 2, 1, 0)                # → {0=BIG_DOWN, 1=BIG_UP}
sample_weight = (y_raw != 1).astype(float32)       # 0 for NO_SIGNAL, 1 otherwise
# Test set: drop NO_SIGNAL rows entirely before scoring
```

### 4.3 Reproducibility

`src/common/repro.py` — called at the very top of `trainer.py run()`:

```python
set_global_seed(seed=42)
# Seeds: PYTHONHASHSEED, random, numpy, tf.keras.utils.set_random_seed
# Enables tf.config.experimental.enable_op_determinism()
```

After training, saves `feature_schema.json` to the checkpoint directory for serve-time compatibility checks.

### 4.4 Experiment Logging

Every training run appends to `_experiments.jsonl` with: timestamp, git SHA, seed, config hash, n_features, n_train_seq, n_test_seq (signal-only), test_accuracy, test_f1_macro, training_seconds.

### 4.5 Walk-Forward Validation

`src/training/walk_forward.py` — expanding or sliding window:

```
Fold 1: Train Oct, Test Nov
Fold 2: Train Oct+Nov, Test Dec
```

Confirms single-split metrics are not artifacts.

> **Prefer `purge_seconds`, not `purge_bars`.** The label horizon is a **duration**
> (600 s). `purge_bars` is a raw row count: 120 is 10 minutes at 5s and 2 minutes at
> 1s. `resolve_purge_rows` converts `purge_seconds` using the probed bar interval and
> warns if a raw `purge_bars` implies the wrong duration. New objectives pass
> `purge_seconds: 600`.

### 4.6 Path-realised account simulation ([ADR-052](../decisions/ADR-052-path-realised-account-simulator.md))

The leaderboard's economics are **label-implied**: `label_implied_pnl`
(`src/evaluation/_kpi.py`) prices a trade from the confusion matrix — a correct call earns
`+move_threshold`, a wrong one is charged `-mae_tolerance` — and the TBM labels beneath it
compare **5s closes** to both barriers (§2.1). At `move_threshold = 50` that is fine; at
`mae_tolerance = 6` it is optimistic, because a 5s NQ bar at the cash open routinely
ranges more than 6 points, so an intrabar stop-out is recorded as an untouched win.

`src/backtest/portfolio_sim.py` is the second, independent evaluator: first-touch resolved
on the **1-second high/low path** (`parquet_path1s`), then run through an account —
contract sizing, margin capacity, concurrent-position cap, per-session loss halt,
commissions, adverse stop slippage. It reads no labels.

| Evaluator | Sees intrabar stops? | Account rules? | Reads labels? |
|---|---|---|---|
| `label_implied_pnl` | no | no | yes |
| `src/backtest/engine.py` (5s closes) | no | position cap / cooldown only | no |
| `src/backtest/portfolio_sim.py` (1s path) | **yes** | **yes** | **no** |

The two are **not interchangeable**: rescoring the ADR-047 policy moved hit rate
0.328 → 0.190 and net expectancy 11.34 → 3.97 points. Leaderboard numbers stay
label-implied for historical comparability; path-realised numbers are reported separately.

### 4.7 Opt-in 3×3 expected-cost train loss ([ADR-072](../decisions/ADR-072-cost-matrix-train-loss.md))

Default search / Custom Test 3-class heads still minimize **sparse categorical
CE** (LSTM/TCN/CNN/Transformer) or **`mlogloss`** (XGBoost), with optional
`class_balance` sample weights. That is **not** net expectancy.

`custom_optimization: true` plus `cost_matrix` on the objective switches those
3-class heads to **expected cost** `C[true] · softmax`. `nq-50pt-tbm-opt` holds
the table (dir 0/50/56, flat 6/0/6). The flag is currently **false** (sparse CE
+ `class_balance=sqrt`) after LSTM `C · p` collapsed to a one-hot vertex. RF /
logistic / SVM / `xgboost_ovr` / binary meta do **not** take the matrix.
`class_balance` is skipped while the matrix is on. Post-fit KPI and serving are
unchanged. The WebUI Objectives grid reads the YAML with `yaml.safe_load`; a
missing `cost_matrix:` key under `custom_optimization: false` is a ScannerError
(blank symbol/scheme/move/mae/kpi, detail used to 500).

Design rules the simulator encodes:

- **Constraints are enforced, not learned.** Position caps, margin, hold limit, session
  window, cooldown and the loss halt are environment rules. A model must not spend
  capacity rediscovering a rule that can be applied for free.
- **Trade frequency is a dial, not a training target.** `sweep_tau` reports the
  `(n_trades, expectancy, drawdown, $/month)` frontier; the operating point is chosen on
  validation folds and frozen for test. Frequency remains a *constraint* on model
  selection (`kpi.requirement` `n_signals`), never a term in the objective — a
  minimum-trade quota makes idling impossible but also makes negative-expectancy trades
  mandatory.
- **Bias is measured, not assumed.** `tie_policy` ∈ `{stop_first, target_first,
  close_only}`; `compare_tie_policies` re-prices identical signals under all three.

CLI: `scripts/search/run_portfolio_sim.py`. Path cache: `scripts/data/build_path1s.py`.

---

## 5. Inference Signal Gate (Planned — Phase 18)

At inference, the 2-class softmax produces `[P(BIG_DOWN), P(BIG_UP)]` for every bar. A directional signal fires only when `max(P_down, P_up) ≥ τ` for some confidence threshold `τ` (target: 0.6 or higher). Bars where neither class crosses the threshold are treated as NO_SIGNAL operationally — equivalent to the masked NO_SIGNAL training label, but derived from model confidence rather than a third output class.

---

## 5.1 SHAP attribution (XGBoost serving bundles)

`python -m src.evaluation.shap_explain --bundle <staging-or-registry-dir>`
(`make shap MODEL=…`) runs `shap.TreeExplainer` on the **refit booster** (not
walk-forward OOF fold models). Preprocessing matches search training: the
bundle `feature_cols`, `NaN→0` (`fold_cache` contract), then `scaler.pkl`. The
sample oversamples τ-gated directional fires so the ranking is not just “what
predicts flat”.

**Dual-binary / OvR trees** persist as `DualBinaryPair` / `OvrDirectional`
(`model.pkl`), not a single 3-class booster. TreeExplainer cannot consume the
wrapper: each binary XGB is explained on its own columns (including
head-private `long_idx` / `short_idx`), then mapped onto the CSV 3-axis
(`down` = short / down-vs-rest, `up` = long / up-vs-rest, `flat` unused).
`shap_meta.json` records `head` and `class_map`.

Writes into a `SHAP/` subdirectory of the search / Custom Test run (or of the
staging/registry bundle): `SHAP_SUMMARY.md`, `shap_importance.csv`,
`shap_groups.csv`, `shap_meta.json`, `shap_values_sample.npz`, bar + beeswarm PNGs.

TreeSHAP is XGBoost / LightGBM / RF. Output-meta / TCN / LSTM bundles are out
of scope (use Integrated Gradients / permutation scripts under
`scripts/search/` for DL).

---

## 6. HPO (Hyperparameter Optimization)

`scripts/hpo_search.py` — Optuna search over LSTM units, attention heads, dense units, dropout, learning rate, batch size, sequence length, `big_move_threshold`, and `mae_tolerance`.

---

## 7. Model Registry

| Checkpoint | Architecture | Date Range | Notes |
|-----------|---|-----------|-------|
| `model_v001_baseline` … `model_v007_0dte` | 3-class softmax (legacy) | Oct 2025 – Apr 2026 | Pre-detection-2; see [PHASE_13-16_RESULTS.md](../changelog/phases/PHASE_13-16_RESULTS.md) and [PHASE_17_RESULTS.md](../changelog/phases/PHASE_17_RESULTS.md) |
| `model_v008_detection` | Detection-2 + MAE (superseded) | Oct 1 – Dec 31, 2025 | `test_acc=0.511`, `test_f1_macro=0.504` on signal-only test set (1,308 seqs). Superseded by v009 (coin-flip diagnosis; [ADR-008](../decisions/ADR-008-3class-softmax-class-weight.md)). |
| `walk_forward_v008/` | Detection-2 + MAE | Oct–Dec 2025 (rolling 1-month) | Pending |
| `model_v009_3class` | **3-class softmax + class_weight** ([ADR-008](../decisions/ADR-008-3class-softmax-class-weight.md)) **(current)** | Oct 1 – Dec 31, 2025 | 8 epochs, 62 min. `test_acc=0.691`, `test_f1_macro=0.331` (all-bar test set; not comparable to v008 signal-only). Per-class: BIG_DOWN f1=0.026, NO_SIGNAL f1=0.822, BIG_UP f1=0.145. BIG_DOWN precision ~1.5% (collapsed). BIG_UP precision@τ=0.80: 39.7% (146 signals). Asymmetric: BIG_DOWN floods NO_SIGNAL bars (3,631 false positives). Requires further investigation. |
| `nq-50pt-xgb-tcn` search `20260815T054508Z` | output_meta: XGB + TCN + logistic gate ([ADR-033](../decisions/ADR-033-output-meta-n-bases.md)) | 5s parquet through 2026-08-04 on GPU | Champion `meta__xgboost_tcn` **requirement only** (NE=24.8, HR=0.55, 100 shorts @ τ=0.8). TCN OOF = 3 constant softmaxes (unscaled features + unweighted CE). XGB 29 trades @ τ=0.8 missed `n_signals>=30`. Superseded as TCN quality by `20260815T072046Z`. |
| `nq-50pt-xgb-tcn` search `20260815T072046Z` | same stack + RobustScaler + `class_balance=sqrt` + per-fold rebuild | GPU parquet through 2026-08-04 | TCN **non-constant** (881k softmaxes) but 0 trades @ τ=0.8 (467 `up` argmax, 0 `down`). Meta NE=6.2 HR=0.22 — gate **inconclusive**. No champion. Fold ranges in `folds.json`. |
| `nq-50pt-econ` `xgb_tbm_topgroups_20260715T234308Z` | XGBoost 3-class, prefixes `cs_/rv_/vp_/iv_` (43 cols) | 5s parquet 2025-04-07→2026-07-15 (refit) | **Registry LATEST — leak-trained (ADR-041).** Label-implied NE 26.9 / hit 0.605 @ τ=0.8 on **pre-patch** 15m/30m CS. Causal replica on patched parquet ([ADR-046](../decisions/ADR-046-causal-tbm-topgroups-proof.md) `20260821T110058Z`): NE 6.85 / HR 0.247 @ τ=0.65; side AUC 0.55. Not an honest bar. SHAP (2026-08-15, leaky features): side from `cs_*` CLV (~51% of τ-fire \|SHAP\|). |
| `nq-50pt-xgb-unique` `xgb_causal_topgroups_20260821T110058Z` | Causal 3-class XGB, prefixes `cs_/rv_/vp_/iv_` (57 cols) + ADR-047 inference policy | last expanding window (uniqueness 24; 52k train rows) | **Staging only — not promoted.** WebUI Backtest dropdown. Policy: 08:30–10:00 CT, long-only, 10-minute uniqueness. Last-fold serving τ=**0.50** (this booster rarely reaches 0.65). Stacked walk-forward OOF used τ=0.65. Live pin remains `nq-50pt-econ`. |

> **Checkpoint structure** (per directory):
> ```
> model.keras                 ← Keras 2-class softmax model
> features.json               ← Column list at training time
> feature_schema.json         ← Column → dtype for serve-time validation
> training_result.json        ← Metrics snapshot
> ```

---

## Related Docs

- [ADR-007](../decisions/ADR-007-detection-2-mae-filter.md) — detection-2 + MAE filter (current)
- [ADR-004](../decisions/ADR-004-adaptive-labels.md) — adaptive ATR labels (superseded)
- [ADR-041](../decisions/ADR-041-closed-bar-slower-features.md) — closed-bar 15m/30m CS
- [ADR-046](../decisions/ADR-046-causal-tbm-topgroups-proof.md) — causal champion-prefix replica failed the overlapping-5s gate
- [ADR-047](../decisions/ADR-047-inference-uniqueness-open-policy.md) — 10m cooldown + open session is the causal policy
- [ADR-048](../decisions/ADR-048-binary-side-head-rejected.md) — binary side add-on has no OOF skill
- [ADR-049](../decisions/ADR-049-tbm10-side-transfer-rejected.md) — ±10 side does not transfer onto 50/6 events
- [ADR-050](../decisions/ADR-050-hypothesis-side-features.md) — vol/VWAP/EMA/stochastic side model: high RV → more 50/6 ups
- [ADR-051](../decisions/ADR-051-auction-momentum-strategy.md) — auction dump/bounce shorts + timer longs
- [ADR-059](../decisions/ADR-059-keras-serializable-last-timestep.md) — serializable TCN last-timestep; first-party `.keras` load
- [reference/FEATURE_CATALOG.md](../reference/FEATURE_CATALOG.md) — feature columns
- [operations/TRAINING.md](../operations/TRAINING.md) — running training
- [changelog/phases/PHASE_13-16_RESULTS.md](../changelog/phases/PHASE_13-16_RESULTS.md) — pre-detection-2 model history
- [changelog/phases/PHASE_17_RESULTS.md](../changelog/phases/PHASE_17_RESULTS.md) — Phase 17 model history
