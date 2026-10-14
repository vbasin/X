# ADR-046: Causal `tbm_topgroups` proof (champion replica on patched parquet)

## Status
Accepted — measured 2026-08-21. **Did not promote.** Live pin remains
`nq-50pt-econ`.

## Date
2026-08-21

## Context

Registry champion `xgb_tbm_topgroups_20260715T234308Z` (NE 26.9 / HR 60.5%
@ τ=0.8) was trained on pre-[ADR-041](ADR-041-closed-bar-slower-features.md)
parquet. Its #1 SHAP feature (`cs_15m_clv`, 25% of τ-fire |SHAP|) saw the
**forming** 15-minute bar’s final OHLC — a window that covers the 10-minute
TBM path. The operator treats that run as a false champion. Parquet is
already patched. The question is whether the tree policy still clears the
economics gate on **causal** columns.

[ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md) encoded the policy
(calendar WF, 2-minute train uniqueness, cost 1, OVR XGB, meta gate) but
pointed it at `momentum_edge`, which **omits** `cs_`. Wave
`20260821T104055Z` therefore never tested the champion prefixes:
OVR HR 15.1%, NE 1.43, `corr(p_up,p_down)=0.78`, meta approved 0.

A second bug: `compute_diagnostics` hardcoded `gate_tau=0.8`, so
`kpi.n_signals` (which prefers `trades_at_gate.taken`) counted a different
threshold than `canonical_trade_kpi` (`primary_tau` = 0.65 on this
objective).

## Decision

1. Named view `tbm_topgroups` = `cs_` / `rv_` / `vp_` / `iv_` — the champion
   prefixes, now closed-bar. `nq-50pt-xgb-unique` default + per-family view.
2. Eligible families: `xgboost` (3-class, the actual champion architecture)
   **and** `xgboost_ovr`.
3. `compute_diagnostics` `gate_tau` defaults to `primary_tau(objective)`.
4. Manual-spec `plugin_family` is the catalog plugin (`xgboost_ovr` →
   `xgboost`), not the alias name. Alias overwrite skipped the OVR candidate
   (`No plugin registered for family 'xgboost_ovr'`).
5. Retrain on current parquet only. Do **not** change `config.yaml`
   `objective_id`. Do **not** promote unless the requirement clears.

Rejected: using the leaky 26.9 NE as the comparison bar; another LSTM/TCN
softmax run (already shown to be activity-only).

## Results

Calendar expanding 6-fold, 2025-05-01→2026-06-30, uniqueness 24 (train
4.2% kept), cost 1 pt, prefixes `cs_/rv_/vp_/iv_` (57 cols on current
parquet). Live pin untouched.

| Run | Head | NE | HR | n @ τ | MDD | side AUC | activity AUC |
|-----|------|---:|---:|------:|----:|---------:|-------------:|
| leaky registry champion (not a bar) | 3-class | 26.9 | 0.605 | 4019 @ 0.8 | −259 | — | — |
| `20260821T110058Z` `xgb_causal_topgroups` | 3-class | **6.85** | **0.247** | 1035 @ 0.65 | −553 | 0.55 | 0.76 |
| `20260821T110418Z` `xgb_causal_ovr` | OVR | **3.51** | **0.188** | 3656 @ 0.65 | −1127 | 0.55 | 0.75 |
| `20260821T104055Z` OVR `momentum_edge` | OVR | 1.43 | 0.151 | 2975 @ 0.65 | −2240 | — | — |

Causal 3-class τ-sweep never exceeded HR 0.25 (peak NE 6.95 @ τ=0.70, 269
trades). @ τ=0.8: 32 trades, HR 0.094, NE −1.75. Fires are almost all
`up` (1034/1 and 3596/60). Meta approved **0** proposals both times.
Gate: **inconclusive**.

Trees without the leak have the same failure mode as LSTM/TCN: they detect
**activity** (AUC ~0.75) and not **side** (AUC ~0.55). Uniqueness and
one-vs-rest did not create a directional edge on L1 TBM +50/−6 / 10m.

## Consequences

- Registry LATEST remains the leak-trained artifact. Paper/live numbers
  from it are not an honest causal baseline.
- `nq-50pt-xgb-unique` is the research objective for further causal work;
  it is not a promoter.
- Search: manual spec
  `config/specs/manual_candidates.nq-50pt-xgb-unique.yaml`.

## Supersedes

The `momentum_edge` default on `nq-50pt-xgb-unique` in
[ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md). Uniqueness, calendar
WF, cost 1, and the OVR catalog entry stay. Complements ADR-041 (champion
weights were not refit there; this ADR measures the causal replica).
