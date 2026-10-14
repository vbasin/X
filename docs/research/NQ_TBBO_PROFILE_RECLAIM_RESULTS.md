# NQ TBBO confluent-profile reclaim — results

**Status: REJECTED.** This experiment did not establish an NQ trading model.
It must not be paper- or live-traded. The July–September 2026 reserve was not
read.

## Reproducible evidence

The completed immutable run is
[`nq-tbbo-profile-reclaim/20260924T051100Z`](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/).
It snapshots the policy code and protocol, hashes all 266 local tape files used
through 2026-06-30, and persists profile provenance, each candidate ledger,
the frozen policy, daily P&L, and the gate. The [manifest](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/manifest.json),
[candidate results](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/discovery_trials.json),
[frozen selection](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/frozen_policy.json),
and [validation gate](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/validation_gate.json)
are the primary evidence.

The chronologically separated sample was May 2025–March 2026 discovery (208
eligible cash sessions), then April–June 2026 frozen validation (58 sessions).
Each decision uses a right-labelled closed minute and only earlier same-contract
profiles. Entry and exit use the next valid bid/ask after 250 ms; the base case
adds 0.25 points adverse slippage per side plus the configured $3.36 one-NQ
commission. A one-point-per-side adverse-fill stress replay was also run.

## Frozen policy and discovery result

No grid member met the discovery safety gate. For disclosure, the mechanically
best mean-daily row was frozen before validation as exploratory-only:

| Rule setting | Discovery result |
|---|---:|
| Confluent bound members / sweep / print | 3 / 2 points / 20 lots |
| Executed trades | 4 |
| Net P&L / mean eligible day | $966.56 / $4.65 |
| Win rate / profit factor | 50.0% / 2.24 |
| Worst realised trade | -$398.36 |
| One-point stress net | $826.56 |
| Conservative intra-trade quote-mark drawdown bound | $1,170.00 |

It failed the discovery gate because it had only four trades, its bootstrap
mean interval included zero, and the conservative quote-mark drawdown bound
exceeded $1,000. The four [discovery positions](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/discovery/policy_6/trade_ledger.parquet)
were retained exactly as executed by the replay.

| Session | Side | Decision (UTC) | Entry / exit | Exit reason | Net P&L |
|---|---|---|---|---|---:|
| 2025-08-04 | Long | 17:49:00 | 23,256.50 / 23,294.00 | Target | $746.64 |
| 2025-10-22 | Short | 18:23:00 | 24,914.00 / 24,933.00 | Stop | -$383.36 |
| 2026-02-24 | Long | 18:40:00 | 25,043.00 / 25,023.25 | Stop | -$398.36 |
| 2026-03-05 | Short | 18:24:00 | 24,833.00 / 24,782.75 | Target | $1,001.64 |

## Frozen validation result

The pre-validation policy generated one fully resolved trade. It stopped out:

| Session | Side | Executable entry | Exit | Net P&L |
|---|---:|---:|---:|---:|
| 2026-06-08 | Long | 29,559.00 | 29,537.00 | -$443.36 |

Base validation was -$443.36 total, or -$7.64 per eligible day; its five-session
block-bootstrap 95% interval was [-$22.93, $0.00]. One-point stress was
-$473.36 total. The [base ledger](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/validation/base/trade_ledger.parquet)
and [stressed ledger](../../data/research/runs/nq-tbbo-profile-reclaim/20260924T051100Z/validation/stress_1pt/trade_ledger.parquet)
contain the quote timestamps, prices, path extrema, and exit reason.

The model therefore fails the requested $1,000-per-day target, positive
validation, positive lower confidence bound, and positive execution stress.
The $500 realised-trade constraint and the $1,000 conservative validation
quote-mark boundary happened to hold in this one-trade validation sample, but
they do not rescue the failed expectancy or establish a loss rate.

## Interpretation

The tested claim was that a confluent-profile excursion with same-direction
large-print flow and a VWAP-confirmed reclaim rotates to the next prior node.
The available May 2025–June 2026 TBBO history supplies neither enough events
nor positive holdout evidence for that claim. Selecting a new threshold after
seeing the June loss would invalidate this validation window; it was not done.
TBBO also gives a print-time top-of-book snapshot, not broker queue priority or
hidden liquidity, so even a pass would require paper validation before any
promotion.
