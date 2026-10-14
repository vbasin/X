# ADR-051: Auction-momentum long/short for TBM +50/−6

## Status
Accepted as the **independent** causal direction+momentum design — measured
2026-08-21. Does **not** promote. Live pin remains `nq-50pt-econ`.

## Date
2026-08-21

## Context

Operator asked for a self-directed strategy to identify **direction and
momentum** for +50/−6, not another pass over suggested indicators.
A 50-pt first-touch with a 6-pt stop is a **session impulse**: the stop
is smaller than a normal pullback, so the only causal geometry that
matches the product is “already trending from the RTH auction, pullback
still inside the stop.”

## Decision

1. **Created features** (from `_close`, per Chicago day, causal): return
   from 08:30 open, running session high/low, pullback from high, rally
   from low, minutes since extreme, 1-minute and 5-minute return.
2. **Longs** stay the ADR-047 timer (`p_up≥0.65`, 08:30–10:00, 10m
   cooldown) — it is the best long filter we have.
3. **Shorts** are a symmetric auction dump: pullback from session high
   ≥20, bounce from the dump low ≤6, RV ≥ train-only 70th percentile,
   08:30–09:45 CT.
4. Dual one-sided XGBs on those auction features (up-vs-rest and
   down-vs-rest) as a learned alternative. `src/search/auction_momentum.py`.
   Last-fold weights: `data/research/search/nq-50pt-xgb-unique/20260821T124732Z/models/{short_head,long_head}.ubj`.
   The short **book** is the rule in `models/short_rule.json`, not the XGB
   (the XGB over-fires).

## Results (`20260821T124732Z`)

Handed rules (10m cooldown, 50/6, cost 1). Expanding 6-fold walk-forward
(`walkforward_short_and_combined.json`): each test window uses RV p70
from **that fold’s train only**.

| Book | n | HR | NE | total pts |
|------|--:|---:|---:|----------:|
| Timer longs (ADR-047, same folds) | 57 | 0.333 | **11.67** | 665 |
| Auction short (dump≥20, bounce≤6) **WF** | 362 | 0.163 | **2.13** | 770 |
| Combined **WF** | 392 | 0.184 | **3.29** | 1288 |

Short WF is lumpy: fold 1 (2025-09–10) HR 0.056 NE −3.9; fold 5
(2026-05–06) HR 0.266 NE 7.9. Dual XGB 0.55/0.55 pooled OOF: n=1341
HR 0.136 NE 0.64 L/S 695/646 (fires too often).

Two-sided learned heads: up-vs-rest AUC 0.55–0.62, down-vs-rest AUC
0.51–0.60 (fold 0 down undefined). They **do short**, but fire too often
(NE ~1). A τ grid on OOF never exceeds NE ~1.1.

`LS_impulse` / `LS_5m_cont` two-sided rules print NE ~2.5–3 on OOF with
~40/60 long/short mix — worse per-trade than the timer, more total
points.

## Consequences

- Direction for this product is **auction continuation**, not tick CLV
  or VWAP slope. Shorts exist: sell a high-vol dump that has not bounced
  more than the 6-pt stop.
- Combining shorts with the long timer makes a two-sided book (NE 4.5,
  44L/278S, MDD −119) and **lowers** NE vs longs-only (11.7). Total
  points rise (1442 vs 665).
- Still misses HR 0.55 / NE 20. Live pin unchanged.

## Supersedes

None. This is the self-directed strategy after ADR-048–050 rejected
generic side heads.
