# ADR-053: Vol-normalised path labels + TCN/patch_hazard encoder ablation

## Status
Accepted — 2026-08-21. Adopts a **new label dataset** and **new training targets**
alongside the existing `{0, 1, 2}` sparse label. Does not change the live pin
(`nq-50pt-econ`), the leaderboard economics contract, or the serving contract
`{ts, direction, confidence}`. Extends the honest-fill programme opened by
[ADR-052](ADR-052-path-realised-account-simulator.md).

## Date
2026-08-21

## Context

The path-realised rescore in ADR-052 showed that ADR-047's τ=0.65 policy was
label-implied at 11.34 NE / 0.328 HR, and prices at **3.97 NE / 0.190 HR** on the
1-second high/low path — a 65% haircut on a metric the objective is short of. ADR-052
identified the root cause as a **labelling** problem: `generate_labels_tbm` in
`src/training/labels.py` walks a 5s close series, so a 5s bar that traded 9 points against
the entry and closed +1 is scored as an untouched stop. Every training example on the
6-point stop is a coin flip about whether the intrabar path stopped out.

Two questions had to be answered before any new model:

1. **Which years can supervise a 50/6 geometry at all?** Fixed 50/6 was measured
   under ADR-052 at 0.000% directional rate for 2011–2017 — the median 10-minute range
   in 2011–2013 was smaller than the 6-point stop, so the geometry was incoherent, not
   merely rare. That measurement only ruled out fixed 50/6; a vol-normalised equivalent
   was untested.
2. **What baseline honestly measures the value of the honest labels alone**, before
   attributing gains to a new encoder?

The `parquet_path1s` dataset was extended for this: **3,919 sessions,
2011-01-03 → 2026-06-30**, 07:00–15:15 CT, one 1s OHLC parquet per session
(`src/common/data_paths.py::parquet_path1s_dir()`; built by `scripts/data/build_path1s.py`).
That is the price path every measurement below reads.

## Decision

### 1. Label geometry: vol-normalised, causal, with a microstructure floor

`src/training/path_labels.py` labels every entry bar in the **08:30–11:00 America/Chicago**
entry window from the 1s high/low path (aligned with [ADR-002](ADR-002-08-11-session-filter.md);
the 08:30–10:00 figure that appeared in the first draft was the ADR-047 *inference*
timer, not the label window). Two barrier modes are offered; **`vol_normalized` is the
canonical geometry** and is the only one used for training. See **Amendment** below
for Encoder B without 1s path patches.

**Barriers.** `target_r = 0.743`, `stop_r = 0.0891`. Both are multiples of a causal
volatility yardstick `R` — the median of the previous **five** sessions' median 10-minute
range, evaluated inside the entry window. The multiples are the fixed 50/6 geometry
divided by the 2026 realised R (67.3 pt): the model sees the *same event* in every era
even as the underlying range grows from ~6 pt in 2011 to ~67 pt in 2026. Barriers are
**per-session constants**, published on every entry row (`target_points`, `stop_points`,
`r_value`, `stop_ticks`), so a mixed-era dataset stays auditable.

**Microstructure floor.** `min_stop_ticks = 8` (tick size 0.25 pt, absolute). Sessions
whose normalised stop falls below the floor are labelled `usable=False`. This is
non-negotiable: a normalised 2011 stop is 2.2 ticks, i.e. *inside the spread*, and a
fill at that distance is fiction.

**Causal yardstick.** `iter_session_labels` carries `R` forward from prior sessions
only. The first five sessions in any date range are unlabellable in `vol_normalized`
mode — no history, no barrier.

**Tie policy.** `stop_first` (default) is conservative; `target_first` is an optimistic
bound; `close_only` reproduces the 5s close-series blind spot. Under this geometry
`ambiguous_rate ≈ 0` at 1s resolution (see §Measurement), so the tie policy is not a
live concern for 50/6-equivalents.

**Dense targets.** Alongside the sparse `label ∈ {0, 1, 2}`, every entry row carries:

* Competing-risks `event_{up,down} ∈ {0 censored, 1 target, 2 stop}` and time-bucket
  `bucket_{up,down} ∈ [0, n_buckets]` with `n_buckets=20` at 30 s each — a discrete
  hazard target across a 600 s horizon.
* Excursions `mfe_{up,down}_points`, `mae_{up,down}_points`, and their `_r` variants
  normalised by `R`.
* A censored `bucket = n_buckets` sentinel where no barrier was struck inside the
  horizon (the hazard head must not learn a bucket).

The sparse label yields one supervised value per episode; the dense targets are defined
on *every* bar, including flat ones. That is what lets tens of millions of 1s bars
train an encoder that ~5 k directional episodes per year cannot.

### 2. Dataset: `parquet_pathlabels`

`data/sources/features/parquet_pathlabels/` is a new logical layout key
(`src/common/data_paths.py::parquet_pathlabels_dir()`), one parquet per session
`labels_YYYY-MM-DD.parquet`, materialised by
`scripts/features/build_path_labels.py --out …` for the full 2011-01-03 → 2026-06-30
range under the canonical `vol_normalized` config. All eras persist; the `usable`
column gates supervision downstream — a filter, never a rebuild.

The parquet schema is the return of `label_session` and is versioned by ADR reference.
Downstream loaders **must** join on `ts` and respect `usable`.

### 3. Supervision era: 2018-01-02 onward; pretraining pool: 2011-01-03 onward

`usable_frac` under the canonical geometry (measured 2026-08-21):

| Year | dir_rate | stop_ticks | usable_frac |
|-----:|---------:|-----------:|------------:|
| 2011 | 0.193 | 2.10 | 0.0% |
| 2012 | 0.187 | 1.83 | 0.0% |
| 2013 | 0.190 | 1.74 | 0.0% |
| 2014 | 0.180 | 2.45 | 0.0% |
| 2015 | 0.177 | 3.18 | 0.0% |
| 2016 | 0.186 | 3.00 | 0.0% |
| 2017 | 0.171 | 2.65 | 0.0% |
| **2018** | 0.183 | 6.12 | **23.3%** |
| **2019** | 0.174 | 5.10 | **9.1%** |
| **2020** | 0.171 | 11.68 | **78.8%** |
| 2021 | 0.168 | 11.50 | 85.8% |
| 2022 | 0.169 | 18.07 | 100.0% |
| 2023 | 0.174 | 11.96 | 96.8% |
| 2024 | 0.176 | 14.23 | 95.7% |
| 2025 | 0.169 | 18.93 | 100.0% |
| 2026 | 0.174 | 24.65 | 100.0% |

The directional rate is **stationary at ~17–19% every year** — the vol-normalised
geometry is scale-invariant, as intended. The binding constraint is the tick floor,
not label incidence.

* **Supervision start: 2018-01-02.** Includes 2018 and 2019 as marginal contributors
  (~82 additional usable sessions), all gated by `usable=True`. Every stop is
  ≥8 ticks by construction; those sessions are not fictional fills. The remaining
  weight naturally lands in 2020+.
* **Pretraining pool: 2011-01-03 onward.** Excursion and hazard targets are defined
  on every session in the range, `usable` or not. The encoder sees the price dynamics
  of 2011–2019 through `mfe_*_r` / `mae_*_r` / `event_*` even where no directional
  supervised label survives. This is the *only* remaining use for the sub-floor era,
  and it is a legitimate one.
* **Inference gate.** Live serving on the fixed +50/−6 geometry filters at
  `0.743 × R ≥ 50 pt`, i.e. `R ≥ 67.3 pt`. Below that, the geometry the trained model
  actually saw is smaller than the live geometry, and no bar is scored.

### 4. Encoder ablation: TCN + `patch_hazard` share folds, heads, and labels

Two encoders share **the same** walk-forward schedule, the same label parquets, the
same head module, and the same loss composition. Any economics gap between them is
attributable to the encoder alone; any gap between either and the leaky
label-implied leaderboard is attributable to the labels.

**Shared head module.** A single `heads.py` implementing three heads on an encoder's
per-bar embedding:

* **Competing-risks discrete hazard head** (`event_{up,down}`, `bucket_{up,down}`) —
  cross-entropy on the joint `(barrier, bucket) ∈ {no-touch} ∪ (target/stop × 0..19)`
  distribution per side.
* **MFE/MAE quantile head** — pinball loss at `{0.1, 0.5, 0.9}` on `mfe_up_r`,
  `mae_up_r`, `mfe_down_r`, `mae_down_r`.
* **Optional SSL head** — masked-bar reconstruction on a subset of encoder inputs;
  enabled only during the pretraining pass on 2011–2017 sessions.

At inference the hazard head is integrated over `bucket ∈ [0, horizon)` to give
`P(target_up first)`, `P(target_down first)`, `P(no touch)`. `direction` is the sign of
`P(target_up) − P(target_down)`; `confidence` is the corresponding max. **No serving
contract change.**

**Encoder A — TCN (baseline).** The existing 3-class TCN from `nq-50pt-xgb-tcn` on
the same 5s feature view, with the softmax replaced by the shared heads. Training
mechanics (`src/models/dl_memory.py`, RobustScaler, `class_balance=sqrt`, per-fold
rebuild) unchanged.

**Encoder B — `patch_hazard`.** Four parallel patch tokenisers (price / order-flow /
volume-profile / options-context if available) → 4-layer Transformer, ~0.9M params.
Fed the **same 5s feature view as the TCN**. Optional **1s path patches** (raw OHLC
from `parquet_path1s` as a second input) are a later add; without them Encoder B
does **not** encode 1s candles — only 5s engineered features, with a different
architecture. See Amendment.

**Ablation matrix.** Both encoders × same walk-forward × same three heads → four
distinct economic evaluations (label-implied and path-realised, per encoder). The path
simulator (ADR-052) is the primary judge.

### 5. Walk-forward hygiene

Any walk-forward that touches the 1s path or the new label parquets **must** set
`purge_seconds = 600`. `purge_bars` is a row count: the ADR-032-era default 120 covers
10 minutes at 5s bars and 2 minutes at 1s bars, so leaving the default in place leaks
8 of every 10 minutes of the label horizon into training. `resolve_purge_rows` in
`src/search/walk_forward_calendar.py` already accepts `purge_seconds` and warns on a
mismatch; new configs pass the seconds form.

### 6. Non-goals

* No change to the serving contract, live pin, or the label-implied leaderboard.
  Path-realised numbers report separately per ADR-052.
* No change to `generate_labels_tbm` on the 5s close series. It is preserved for
  historical objectives; the new pipeline is a **parallel dataset** with its own
  layout key.
* No new economic KPI. `net_expectancy`, `hit_rate`, `n_signals`, `mdd` retain their
  meanings; the path simulator is the honest counterpart, per ADR-052.

## Measurement — label geography, canonical geometry

* Script: `scripts/features/build_path_labels.py`, mode `vol_normalized`, config as
  above.
* Summary CSV: `/tmp/geo_vol.csv`, replayable as
  `data/sources/features/parquet_pathlabels/_geography.csv` when the persistence run
  completes.
* Directional rate is stationary at 17–19% across 16 years — the geometry is
  scale-invariant.
* `ambiguous_rate` is ≤ 5×10⁻⁵ everywhere: at 1s resolution a single bar spanning a
  56-pt round trip does not occur under this geometry. Intrabar tie-breaking is not a
  live concern here (it would be for a 5/5 geometry).
* `stop_ticks` is the binding constraint, and it moves the supervision boundary — not
  the label incidence.

## Consequences

* The honest-label story becomes measurable: the same encoder, trained on
  `parquet_pathlabels` vs `generate_labels_tbm`, quantifies how much of the
  label-implied edge was in the labels.
* The encoder-value story becomes measurable: TCN vs `patch_hazard` on the same
  labels/heads/folds isolates encoder capacity from label honesty.
* Pretraining on 2011+ becomes a legitimate use of the sub-floor era: excursion and
  hazard targets are defined every bar. The `usable=False` flag prevents those bars
  from silently entering supervised loss.
* Any 1s walk-forward that omits `purge_seconds = 600` will be flagged by
  `resolve_purge_rows` and is a leak; ADR-052 already codified the failure mode.
* Per-fold OOF is now checkpointed on disk during the wave
  ([ADR-073](ADR-073-gpu-job-progress-heartbeat.md)). The 16/24 in-process
  dump that died with no artifacts is the failure mode this closes; a
  `path_tcn` requeue is no longer blocked on “nothing persisted until job end.”

## Alternatives considered

**Widen the fixed geometry (e.g. 60/8, 40/5) to raise supervised counts pre-2020.**
Rejected. Every fixed geometry has a fixed R equivalent; changing the numbers only
changes which era hits the tick floor first. The scale invariance question is about
the label *event*, and only `vol_normalized` answers it.

**Skip 2018/2019 entirely — supervision starts 2020.** Considered. 2018/2019 add
~82 usable sessions; the majority `usable=False` rows are dropped, not mixed. Kept
because those 82 sessions are marketable market days, not fictional stops, and 2018's
early sessions include a volatility regime that 2020+ does not repeat. Reversible
without touching data — a `WHERE ts >= …` filter.

**Skip the TCN baseline; go straight to `patch_hazard`.** Rejected. Two variables
would be moving at once — labels and encoder — so any lift could not be attributed.
The TCN is cheap; running it is the cost of the ablation being interpretable.

**Build a single joint (target, stop) head instead of two competing-risks heads.**
Deferred. A joint head is more compact but harder to debug when one side underperforms;
per-side heads mirror the per-side excursion targets and make the up/down asymmetry
directly observable. Revisit if the two heads correlate at >0.95 as ADR-047's `p_up`
and `p_down` did.

**Resolve fills from `databento.tbbo` ticks.** Deferred, per ADR-052 — 1s OHLC is
sufficient at 50/6 geometry, revisit if a target under ~10 points is pursued.

## Amendment — 2026-08-21 (operator)

* **Label entry window = 08:30–11:00 America/Chicago.** Persisted
  `parquet_pathlabels` were built with `--entry-end 11:00` (9,000 1s rows on a
  full day). `PathLabelConfig.entry_end` matches. ADR-047's 08:30–10:00 remains
  an *inference* policy on the tree timer, not this labeller.
* **2011–2017 pretrain pool stays on disk.** Dense `event_*` / `mfe_*_r` targets
  exist for every labelled session including `usable=False`. SSL / path-only
  encoders read them from `parquet_path1s` + `parquet_pathlabels` directly.

### Amendment 2 — recovered encoder spec (operator)

§4's first Encoder A/B sketch (5s TCN softmax-swap; 5s feature-group patches) and
the 5s-join `tcn_hazard` / `patch_hazard` plugins wired that way are **not** this
experiment. `nq-50pt-path-hazard` now launches Encoder A (`path_tcn` on `path1s`).

**Encoder A (train first).** Dilated causal TCN **on the 1s path**
(`parquet_path1s`): 10 layers, dilations `1…512`, receptive field 1,023 s,
~200k params, ATR-normalised OHLC. Shared heads 1–2. If it matches Encoder B,
ship A.

**Encoder B (`patch_hazard`).** Tokens are built from the **1s path only**, four
resolution streams × 24 patches (last 120 s @ 5 s, 600 s @ 25 s, 3,600 s @ 150 s,
8 h @ 20 m). Seven ATR-normalised features per patch; static context token
(tod / ATR₁₀ / ADR-051 session distances; **no VIX**). 4-layer Transformer ~0.9 M.
Same heads. This is *not* a patcher of the 216-col 5s matrix.

**Heads that matter (hold the encoder loosest):** competing-risks hazard
(20 × 30 s buckets, per-side `{target-first, stop-first, neither-yet}`,
right-censored NLL) and dense MFE/MAE pinball in R units on every bar. SSL
masked-patch reconstruction is pretrain-only (2011→present). Serving contract
unchanged: `direction = argmax_side P(target first)`, `confidence = that P`.

**Stage 3** (after a 1s encoder exists): existing XGB `output_meta` / `meta_stack`
on `[hazard + MFE/MAE + 216 engineered cols + ATR regime]`.

**Why 2011 cannot supervise fixed 50/6:** measured median 10-minute range was
~5–6 pt (stop wider than the typical window). Vol-normalised 50/6-equivalent is
the same *event* across eras; the tick floor (`min_stop_ticks=8`) still excludes
2011–2017 from **supervised** barrier loss. Those years are for SSL / dense
auxiliaries only.

**Inference filter:** `0.743 × R ≥ 50` pt. Quiet tape → no trade, not a failure.

## References

* `src/training/path_labels.py` — labels + dense targets (this ADR).
* `src/models/heads.py`, `tcn_hazard_plugin.py`, `patch_hazard_plugin.py`.
* `config/objectives/nq-50pt-path-hazard.yaml` — `labels.scheme: path_vol_normalized`.
* `src/common/data_paths.py::parquet_pathlabels_dir()`, `parquet_path1s_dir()`.
* `scripts/features/build_path_labels.py`, `scripts/data/build_path1s.py`.
* `src/search/walk_forward_calendar.py::resolve_purge_rows` — `purge_seconds = 600`.
* `src/backtest/portfolio_sim.py` — path-realised simulator (the honest judge).
* [ADR-023](ADR-023-triple-barrier-label-scheme.md) — 5s close-series TBM (superseded for
  50/6-equivalent geometries by this ADR; retained for existing objectives).
* [ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md), [ADR-047](ADR-047-inference-uniqueness-open-policy.md) —
  the causal + uniqueness stack this rescores.
* [ADR-048](ADR-048-binary-side-head-rejected.md), [ADR-049](ADR-049-tbm10-side-transfer-rejected.md),
  [ADR-050](ADR-050-hypothesis-side-features.md) — side-model negative results that
  motivate a hazard, not a binary side head.
* [ADR-052](ADR-052-path-realised-account-simulator.md) — the ADR this extends.
