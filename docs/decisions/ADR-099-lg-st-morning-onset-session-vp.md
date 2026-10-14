# ADR-099: lg-st morning session, TBM-onset train uniqueness, 5m session VP

## Status
Accepted — implemented 2026-08-29. Amended by [ADR-104](ADR-104-lstm-session-pmax-and-ui-tau.md) (sequence families skip onset uniqueness). Live pin unchanged.

## Date
2026-08-29

## Context

`nq-50pt-econ-lg-st` trained on native-TF **Globex** bars with no clock
filter. ~5–6% of 15s rows are TBM +50/−6; most of the rest is overnight
and afternoon flat. Consecutive 15s rows that share the same 10-minute
path were treated as independent events. 1m/5m `vp_*` is a **30-minute
rolling** window, not developing **Globex session** value (17:00 CT D−1).

## Decision

1. **Clock.** `session.training_*` and `trading_*` are **08:30–11:00**
   America/Chicago. Fold cache for `search.input: native_tf` keeps only
   those Chicago minutes **after** attaching overnight-aware features
   (cache key `|sess=`). Hive files stay full Globex.
2. **Onset uniqueness.** `eval.train_onset_uniqueness: true`. After a
   train row with `y_long=1` or `y_short=1`, later train rows are skipped
   until that path’s `Time_to_TGT_*` (else 600s). Trees **drop** those
   rows. LSTM keeps the contiguous sequence and **zero-weights** them
   (`w_onset_train`). OOF is not thinned (serve cooldown remains ADR-047).
3. **Session VP on 5m only.** `search.topology.session_vp: true` attaches
   11 `svp_*` columns in the **5m** fold cache (not 15s, not 1m, not hive):
   POC / contiguous 70% VAH–VAL distances, in-VA, above-POC, broke/reclaim
   VAH and VAL, POC crosses. **Origin = Globex 17:00 CT prior day** (not
   RTH 08:30). Attach runs on the full hive **before** the 08:30–11:00
   clip so overnight volume is in the morning bars’ profile. Volume =
   `cs_volume`. Same contiguous-from-POC VA as Live Chart SVP.

Rejected: putting `svp_*` on the 15s LSTM (too static per 15s bar; same
class of leak as fat HTF on the encoder); 0.25-pt histogram; replacing
1m/5m 30m `vp_*`; LSTM row-drops.

## Consequences

- 5m XGB sees Globex session location + 30m `vp_*`. 15s LSTM stays tape +
  `lg_*`/`st_*`. 1m XGB unchanged.
- VWAP on hive already uses `market_session_date` (CME 15:00/16:00 stats
  convention). SVP uses **17:00** as specified. Live pin unchanged.
- Next lg-st run rebuilds fold cache. GPU code bundle must include
  `session_vp.py`.
- **15s row drop superseded** by [ADR-101](ADR-101-lstm-globex-tape-session-loss.md):
  the 15s LSTM tape stays Globex; session is a loss mask. 1m/5m trees
  still clip to 08:30–11:00.

## Related

- [ADR-002](ADR-002-08-11-session-filter.md)
- [ADR-047](ADR-047-inference-uniqueness-open-policy.md)
- [ADR-098](ADR-098-time-to-tgt-nan-on-timeout.md)
- `src/features/session_vp.py`, `src/search/uniqueness.py`
