# DATA_CONTRACTS.md — Source-Level Data Contracts

> **2026-09-24 correction:** the later text's whole-module causal verification
> of `entry_state.py` is superseded for `_order_flow`: its full-request
> `np.nanmax` CVD/OFI normalization leaks later decisions into earlier features.
> This also affects CVD-scaled delta/divergence features. As-of alignment alone
> does not make a future-dependent normalizer causal. The new independent NQ
> experiment rebuilds features from raw tape and verifies prefix invariance;
> see [evidence](research/NQ_SELECTIVE_RESULTS.md).

**Status: CANONICAL, CURRENT.** Derived entirely from source code and existing schema documentation — **no database was queried to produce this document** (per the source-code-only restriction for this restructuring pass). Where a claim could only be confirmed by querying live data, it is marked `DEFERRED — REQUIRES DATA-SIDE VALIDATION`.

## 1. Raw data sources

| Table/hypertable | Content | Consumers |
|---|---|---|
| `tbbo` | Tick-level trade + best-bid-offer, NQ (and ES cross-asset) futures, all instruments including options chains | `src/ingestion/`, `src/features/tape.py` (Track C), `src/features/engine.py` (Track B) |
| `bbo_1s` | 1-second BBO aggregate | Track B feature engine (`src/features/engine.py`) |
| `definitions` | Instrument definitions (futures/options contract metadata, expirations) | Contract-roll logic, options surface |
| `statistics` | Settlement/high/low/OI daily statistics | `src/features/daily_context.py` (`dc_*` features) |

`DEFERRED — REQUIRES DATA-SIDE VALIDATION`: exact current row counts, ingestion freshness, and whether `ca_es_*` cross-asset features (noted stale since 2026-04-02 per ROADMAP.md) have since been backfilled. Do not assume current without checking `systemctl status vtech-live-streamer`/`vtech-live-features` and a fresh ingestion log, not this document.

## 2. Session / calendar conventions

- Track B (`nq-50pt-econ`): training session 08:02–14:45 CT, trading session 08:30–14:45 CT (`config/objectives/nq-50pt-econ.yaml`).
- Track C: session flatten at 15:00 CT (`src/training/move_label.py` caps label horizon here); the underlying print tape is deliberately extended to 15:15 CT as a "runway" (`scripts/data/build_tape.py`). **Known contract inconsistency** (audit finding, Medium severity): Track C's Outcome Builder for the downstream "target reach" outcome has no equivalent 15:00 cap, so late-session decisions can use price action past what the upstream outcome considers "in play" — see `docs/CONFIG_AUDIT.md`.
- Contract roll: `src/common/contract_calendar.py` — algorithmic CME roll calendar (no hardcoded dates), back-adjusted continuous OHLC (Panama additive method) per ADR-005.
- Options → underlying roll (`databento.definitions.underlying`): NQ futures list quarterly months
  only (H/M/U/Z). An option settles into the **first quarterly contract still trading at its expiry
  instant** — so an October weekly settles into NQZ6, not a nonexistent NQV6. The comparison is by
  instant, not date: the quarterly future settles at its 08:30 CT fixing on the 3rd Friday, so a
  15:00 CT weekly expiring that same Friday outlives it and rolls forward. Never derive the
  underlying from the option symbol's own month code — that is the defect ADR-151 corrects.
- NQ weekly/daily options are PM-settled at **15:00 CT**; store the localized instant rather than a
  fixed UTC hour, which is wrong for whichever half of the year DST does not match.
- Holidays: `src/common/cme_holidays.py` — static frozenset.

## 3. The causal ("as-of") invariant

**Repository-wide rule:** for a decision at timestamp T, no feature or context value may contain information that was not available at or before T. T itself is included. A feature for 13:01:00 may use data through 13:01:00 and must not use 13:01:01 or later. This covers raw features, rolling and aggregated features, session profiles, higher-timeframe features, normalization and scaling values, and any other context joined onto the decision.

**Labels are the exception.** An outcome may be defined from the forward price path (a barrier, a fill, a session-close mark). That forward path must not appear in a feature or context column. Known forward-looking label mechanics that are not feature lookups:

- `move_label.py:120`, `build_target_matrix.py:148` use `side="left"` for "first fill at-or-after decision" — forward-looking by design (resolving an entry fill), not a state lookup.
- `structural_transitions.py:176`'s back-window resolution uses `side="left"` for "value ~window-seconds-ago" — a minor, non-leaking measurement quirk (it can silently shorten the effective lookback on a sparse tape, but never reads forward of "now") — tracked in `docs/CONFIG_AUDIT.md`.

**Join enforcement.** `load_training_frame` and `build_oof_analysis` (`src/experiment/analysis.py`) attach features, context, and upstream predictions with `as_of_index`: within the other join keys (`side`, and `session` when it is a key), the decision receives the latest source row whose availability is at or before `ts`, and never a later row. There is no forward fill and no nearest-after match. A source that cannot state its availability is refused.

Each feature and context source declares `availability`:

| Value | When the row is knowable |
|---|---|
| `timestamp` | The time column (default `ts`) is that instant. Decision-grid features and upstream predictions use this. |
| `bar_close` | The time column is the bar open. The value is knowable at open + `bar_duration`. A left-labeled 1-minute bar stamped 13:01:00 is not visible at 13:01:00; it is visible at 13:02:00. |
| `session_close` | The row summarizes the session in `session_column` and is knowable at `session_close` in `timezone` on that session date. A decision during the session does not see that session's completed profile. It can see the latest prior session whose close is at or before T. |

If a file carries `bar_end_ts` after the join time, the resolver refuses the join unless the join time is `bar_end_ts` or the declared `bar_duration` reaches `bar_end_ts`. A feature file that omits a non-time join key such as `side` is refused, so the as-of cannot cross that key. Upstream predictions are decision-time scores (`timestamp`).

A population cache is a materialization of rows already aligned to decision timestamps. Columns already on that frame are not joined again. Columns still read from another file go through the same rule.

The reference implementation of the lookup is Track C's `_asof()` helper (`src/training/entry_state.py`), independently re-verified as clean across `entry_state.py`, `flow_state.py`, `structural_transitions.py`, and `cash_session_profile.py`'s session-history filtering: `np.searchsorted(t_ts, at, side="right") - 1`, clamped/masked for `pos < 0` (not silently clamped into a valid read).

`src/common/causal.py` extracts this pattern as `as_of_index()`. The training join calls it. New as-of lookups anywhere in the repo should use it rather than re-deriving the search (the failure mode this guards against: an unclamped or `side="left"` index silently reading a future value).

Higher-timeframe features already stored at their close and then carried forward onto a faster grid are causal. The carry-forward is staleness between the bar close and the next decision, not a look-ahead (`src/features/engine.py`, `src/features/closed_bars.py`). A left-labeled bar joined on its open is a look-ahead and is refused.

## 4. Volume profile / value-area conventions

- **In-progress session** aggregates (today's developing profile) must be truncated to the decision instant — verified in Track C via `_running_value_area` (`entry_state.py`), which truncates to `cutoff_pos[i]` per row.
- **Prior closed sessions** may be read in full once closed — Track C's `history_before()`/`r_before()` (`src/features/session_profile.py`) filter by strict session-date `<` comparison (not row-count slicing), verified to correctly exclude the in-progress session from its own lookback.
- Multi-session KDE-clustered structural bounds (`src/features/cash_session_profile.py`, ADR-126): built from up to `lookback_sessions` (default 20, extendable to `max_lookback_sessions=120`) prior **closed** sessions only — verified clean this session.

## 5. Normalization

- Track C: most distance features are scaled by `r_value`, a trailing-volatility measure computed via `r_before()` (median of prior sessions' `r_value`, strict `<` on session date — no forward window, verified clean).
- Track B: `scaler_type: robust` (`config.yaml`) — confirmed fit train-fold-only in every code path checked (`src/training/trainer.py`, `src/models/attention_lstm_plugin.py`, `src/models/svm_plugin.py`); XGBoost/LightGBM/RandomForest paths skip scaling entirely (tree-based, scale-invariant), which covers 3 of Track B's 5 eligible model families.
- A scaling dataset joined as context is as-of like any other context. A scale fitted on a sample is available only at the timestamp when that sample was known, so it does not attach to earlier decisions.

## 6. Reading further

Full per-column feature catalogs: `docs/reference/FEATURE_CATALOG*.md` (Track B), `docs/reference/STAGE1_FEATURES.md`/`STAGE3_FEATURES.md` (Track C). This document intentionally does not duplicate those — it states the *contracts* (schemas, timing, causality) those catalogs' columns must obey, derivable from code without a database connection.

## NQ retry source clarification (2026-09-24)

The `parquet_path1s_dir()` NQ cache used by ADR-139 is **unadjusted NQ.c.0**,
calendar-front, 07:00–15:15 Chicago; it is not a Panama-adjusted overnight series.
One-second OHLC timestamps are interval starts. New research minute bars are
right-labelled by close availability and never use a partial future minute.
Quote tapes may use a different quarterly lead-contract convention. This
experiment excludes the pre-known quarterly expiry week and avoids overnight
absolute-price features; it does not silently infer contract parity.

Two 2020 source days truncated before the cash close in both cache and DB.
Zero-cost fresh vendor full-session records recover their coverage in an
additive preparation artifact. June's old/fresh OHLC timing differs materially;
splicing was refused. The vendor marks those days degraded. All unaffected
3,537 sessions are frame-identical. Prior-session-only scaling does not erase
source-quality limitations. See [source audit](research/NQ_LONG_HISTORY_DATA_AUDIT.md)
and [repair evidence](research/NQ_DISCOVERY_DATA_REPAIR.md).
