# ADR-153: Options IV — per-option underlying, solver convergence, and EOM chains

## Status
Accepted — 2026-09-30.

## Date
2026-09-30

## Context

A review of the IV feature data ahead of exposing it in the WebUI (`#/chart`) found
`databento.nq_option_surface_1s` unusable from 2026-08 onward, and traced it to three
independent defects. ADR-151 had already corrected the `definitions` rows these builds
consume; none of what follows is fixed by that repair.

### 1. The underlying price mixed four futures contracts

`_load_nq_price_1s` read `NQ.c.0` from `databento.ohlcv_1s` and, when that returned nothing,
fell back to *every* NQ outright matching `^NQ[FGHJKMNQUVXZ][0-9]$`, keeping whichever
contract quoted last in each second. Its own comment acknowledged this ("without symbol split
this mixes contracts … for simplicity take all and last-wins").

`NQ.c.0` has not been ingested since **2026-07-14**, so from 2026-07-15 every options build
silently took that fallback. On 2026-09-29 it interleaved NQZ6 / NQH7 / NQM7 / NQU7 —
30,605 to 31,772, a **1,166-point** span — against a true NQZ6 day range of 363 points. The
series also admitted the DBN `UNDEF_PRICE` sentinel (9223372036.854776), which survives a
`px > 0` filter because it is positive.

An underlying that is too high prices ATM calls as deep ITM (vega collapses, IV unsolvable)
and ATM puts as far OTM (IV explodes). This is the call/put asymmetry in the stored surface:
September medians of `catm_iv` 0.1397 against `patm_iv` 0.9776, where a healthy month has the
two within 0.005 of each other.

### 2. The IV solver reported non-solutions as solutions

`_implied_vol_vec` (and its near-verbatim copy `_implied_vol_vec_S`) ended with only

```python
sigma = np.where((sigma > 0) & (sigma < 5.0), sigma, np.nan)
```

That is a range clamp, not a convergence test, despite a docstring promising "NaN on failure".
A row whose vega underflowed went inactive on the first iteration, kept the `sigma = 0.3`
seed, and passed the clamp; a row that exhausted `max_iter` kept its last Newton iterate and
also passed. In healthy data this was ~0.1% of rows. Once the mixed underlying collapsed vega
it reached **24% of September rows**, all of them the literal value 0.3000.

A residual test alone is insufficient: for a deep ITM or OTM leg the price barely responds to
vol, so almost any sigma reprices it to within tolerance. Identifiability must be tested too.

### 3. End-of-month option chains were invisible to the live streamer

`_WEEKLY_RESOLVED_SYMBOL_RE` accepted roots `Q[1-5][A-D]` and `QN[1-4]` but not `QNE`, the
end-of-month chain. `_register_resolved_weekly_from_bbo` returns early when a symbol does not
parse, so those instruments never entered `_opt_meta` and never entered `_0dte_ids`.
`_bootstrap_definitions` clears `_opt_meta` on startup and never rehydrates it from the
database, so the live Greeks path depends entirely on symbols it parses during the session.

On 2026-09-30 the whole 0DTE chain was `QNEU6` (376 instruments). Raw BBO flowed normally —
31,723 rows in a ten-minute window across 374 of them — while `nq_option_greeks_1s` and
`nq_option_surface_1s` received **nothing all day**. This recurs on every month-end expiry
that falls on a trading day; restarting the service does not fix it.

## Decision

1. **Price every option against the contract it settles into.** `_load_underlying_prices_1s`
   loads a 1s mid per *named* underlying from `bbo_1s`, bounded above by `_UNDEF_PX` to reject
   the DBN sentinel, and `compute_greeks_frame` selects the column named by each option's own
   `definitions.underlying` — which ADR-151 made trustworthy. A continuous or roll-adjusted
   series is never the correct forward for an option and is no longer used. `process_day`
   skips a day with `no_underlying_prices` rather than substituting a wrong price.

2. **Accept an IV only when the quote determines one.** The solver now requires both that the
   final sigma reprices the option to within `tol`, and that vega clears `tol / sigma_tol` so
   the root is pinned to better than `sigma_tol` (default 0.001). `sigma_tol` is a first-order
   budget, not a hard bound — it binds tightly when small and degrades when loosened.
   `_implied_vol_vec` now accepts scalar or per-row `S` and is the single canonical solver;
   `_implied_vol_vec_S` is an alias, so the two cannot drift again.

3. **Parse `QNE` as the last business day of the month** in the live streamer, alongside the
   existing nth-weekday roots.

4. **Apply (1) to the live path as well.** Restarting the streamer with the `QNE` fix revealed
   the same contract-mixing defect in the live per-second code, which the batch fix had not
   touched. `_on_bbo_for_greeks` kept one `self._nq_mid`, overwritten by whichever outright
   matched `_NQ_FUT_RE` last, and `compute_second_batch` stamped that single price onto every
   option in the second. The 2026-09-30 QNE chain — definitions say `NQZ6` — was written with
   `underlying_sym = NQH7` and ATM call/put IV of 0.054 against 1.316. The streamer now tracks
   `_fut_mids` per contract (bounded by `_UNDEF_PX`), `_opt_meta` carries each option's
   `underlying`, and `compute_second_batch` accepts a mapping and resolves per option. The
   scalar form is retained for a single-underlying chain.

5. **Make the `nq_option_greeks_1s` write optional** (`process_day(write_greeks_db=...)`,
   `--no-greeks-db`). Nothing in this repository reads that table: the feature engine derives
   `iv_`/`gx_` from `bbo_1s` + `definitions` directly, and the WebUI reads the surface. Its
   COPY is 11M rows and 18.7 min of a 24.3 min day.

## Consequences

Rebuilt 2026-09-29 (15.1M quotes), before and after:

| | mixed underlying | per-option underlying |
|---|---|---|
| underlying span | ~1,170 pts | 363 pts |
| quotes solved | 42.7% | 73.4% |
| put-call parity offset (median) | −369 pts | 0.00 |
| `catm_iv` finite | 70.8% | 99.6% |
| ATM call vs put IV | 0.031 vs 1.023 | 0.2309 vs 0.2358 |

The 0.005 call/put gap matches the healthy 2025 months (0.3937/0.3971 in April,
0.2292/0.2336 in December).

The live path, across the 2026-09-30 restart that deployed fix (4), in the same table:

| window | `underlying_sym` | ATM call IV | ATM put IV |
|---|---|---|---|
| 15:16–15:30 | `NQH7` | 0.0544 | 1.3166 |
| 15:30:57 onward | `NQZ6` | 0.1963 | 0.1995 |

Per-option resolution also picks the right contract either side of a quarterly roll, which a
single series structurally cannot: rebuilt 2026-09-01 resolves to `NQU6` (before the 2026-09-18
quarterly) and 2026-09-29 to `NQZ6`, each with a 0.003–0.005 call/put gap.

### Measured build cost (2026-09-29, 15.1M quotes, one worker)

| stage | before | after |
|---|---|---|
| `bbo_1s` day load | 143 s | narrowed to 4 columns; −71% frame memory |
| `build_surface_from_greeks` | 155 s | **2.9 s** (57×, output bit-identical) |
| Greeks compute | 68 s | solver 1.89× on `ndtr` + closed-form PDF |
| `nq_option_greeks_1s` COPY | 18.7 min (77% of the day) | skippable |
| whole day | 24.3 min | ~3 min surface-only |

Peak RSS is 12.08 GB, which is what caps `workers_for(mem_per_worker_gb=8.0)` at one worker on
this VM. The COPY's client half is 337 s of its 1,121 s — `to_csv` 269 s, `strftime` prep 66 s —
against a 2.79 GB payload; the rest is server-side. If the Greeks archive is ever wanted again
at volume, that serialiser is the thing to replace, not the compute.

- **All `iv_*`, Greeks, dealer-GEX and surface artifacts built from 2026-07-15 onward are
  invalid** and must be rebuilt, over and above ADR-151's expiry window. The stored
  2025-04 → 2025-12 data predates the `NQ.c.0` gap and is unaffected.
- IV coverage falls slightly on well-posed quotes (98.7% → 97.9% on synthetic round-trip) in
  exchange for bounding the worst-case error from 1.17 vol points to 0.001. Rows that used to
  read 0.3000 now read NaN; consumers must treat NaN as "no quote information", not "no data".
- `_load_nq_price_1s` is retained but unused. It should not be reintroduced without a
  per-contract split; restoring `NQ.c.0` ingestion alone would not make it correct.
- The WebUI overlay can be built against the surface table once the rebuild lands. Until the
  live streamer is restarted with the `QNE` fix, month-end sessions have no live IV.

## Alternatives considered

- **Backfill `NQ.c.0` from Databento.** Costs money (AGENTS.md §5), and a continuous series
  would still be the wrong forward across a roll. Rejected.
- **Widen the solver's `(0, 5)` clamp or lower `tol`.** Neither distinguishes a converged root
  from a stalled seed. Rejected.
- **Rehydrate `_opt_meta` from `definitions` at streamer startup.** A real robustness gap and
  still open — but orthogonal to the parse failure, which would persist regardless.

## Supersedes

Extends ADR-151. Does not supersede it: the DTE error it fixed was real, merely not the cause
of the call/put asymmetry.
