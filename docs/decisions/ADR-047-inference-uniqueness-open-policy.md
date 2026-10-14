# ADR-047: Inference uniqueness + open session on causal trees

## Status
Accepted — measured 2026-08-21. **Did not promote.** Live pin remains
`nq-50pt-econ`.

## Date
2026-08-21

## Context

[ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md) thinned **train** rows
every 2 minutes because overlapping 5s TBM labels are not i.i.d.
**Scoring still used every OOF bar.** [ADR-046](ADR-046-causal-tbm-topgroups-proof.md)
then showed causal 3-class XGB on patched parquet:
NE 6.85 / HR 0.247 / 1035 trades / MDD −553 — activity skill, no side,
and a drawdown inflated by stacking correlated 5s fires on the same
10-minute path.

The remaining claim to prove: uniqueness belongs at **decision time**,
the clock is the 10-minute horizon (or a closed 5-minute bar), and the
session is the open — not another 3-class/OVR/LSTM on every 5s bar.

## Decision

1. **`apply_trade_cooldown`** (`src/search/uniqueness.py`) — greedy
   10-minute (120×5s) spacing on *proposed trades*, not on train rows.
2. **Open window** 08:30–10:00 America/Chicago (where TBM-directional
   rate is 16–17%, vs ~7% midday).
3. **Event-meta path** (`src/search.event_meta`): 5-minute close clock,
   primary = sign(`cs_5m_clv`) if |CLV|≥0.4, train-only RV percentile,
   binary XGB P(win | primary). τ chosen on **train** events.
4. **Policy scorer** on an existing canonical OOF: same open + τ +
   cooldown, with optional walk-forward τ (pick on prior OOF, apply to
   next). Does not change `config.yaml`.

Rejected: treating ADR-046's 1035 overlapping fires as a live book;
using leaky registry NE 26.9 / HR 60.5% as the bar.

## Results

Causal 3-class OOF `20260821T110058Z` (same weights as ADR-046), TBM
50/6, cost 1:

| Policy | n | HR | NE | MDD | total pts |
|--------|--:|---:|---:|----:|----------:|
| Overlapping 5s @ τ=0.65 (ADR-046 headline) | 1035 | 0.247 | 6.85 | −553 | 7091 |
| Open 08:30–10:00 + 10m cooldown @ τ=0.65 | **58** | **0.328** | **11.34** | **−49** | 658 |
| Walk-forward τ (prior-OOF max NE, min n=8) | **49** | **0.327** | **11.29** | **−42** | 553 |

Event-meta XGB veto (5m clock, RV p70, `20260821T112552Z`): primary
n=833 HR 0.144 NE 1.07 MDD −385; meta n=43 HR 0.209 NE 4.72 MDD −56.
Per-fold meta OOF AUC **0.46–0.54** — the veto does not rank. 1-minute
clock without RV (`20260821T112945Z`) is worse (primary NE −1.28).

Requirement from `nq-50pt-xgb-unique` (NE 20 / HR 0.55 / n 30 / MDD
−400): the open+cooldown book **clears n and MDD**, misses NE and HR.
Those two floors were calibrated to the leaky champion. Causal L1 +50/−6
/ 10m maxes around **HR 33% / NE 11** when you stop counting the same
path twelve times.

## Consequences

- Serving/search scores that emit a trade every 5s on the same TBM window
  overstate n and MDD. Cooldown is part of the model.
- Event-meta remains in the repo as the explicit primary/veto experiment;
  it is not the champion.
- Do not promote. Live pin stays `nq-50pt-econ` (still leak-trained).

## Supersedes

The scoring convention implied by ADR-045 (OOF every bar). Train
uniqueness stays. Complements ADR-046 (causal replica weights reused).
