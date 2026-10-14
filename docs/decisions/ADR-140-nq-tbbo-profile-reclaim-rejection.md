# ADR-140 — TBBO confluent-profile reclaim is rejected

Date: 2026-09-24. Status: rejected experimental hypothesis; no promotion.

## Decision

Keep the TBBO profile-reclaim implementation and its immutable artifacts as
negative research evidence. Do not connect it to paper trading, serving, a
model registry, or live execution.

The policy combines exact prior closed-session volume-profile bounds,
right-labelled one-minute tape geometry, signed 20/50-lot aggressive prints,
and a causal session VWAP reclaim. It submits only after a 250-ms delay to a
later valid bid/ask snapshot and uses a 20-point planned stop. It was tested on
May 2025–March 2026 discovery and then a frozen April–June 2026 validation;
the July–September reserve remains unopened.

## Evidence

No discovery candidate satisfied the declared safety gate. The selected
exploratory configuration had four discovery trades, $4.65 mean net per
eligible day, and a $1,170 conservative intra-trade quote-mark drawdown bound.
Its frozen validation had one resolved trade, a -$443.36 loss, -$7.64 mean net
per eligible day, and a five-session bootstrap lower bound of -$22.93. The
one-point adverse-fill stress was also negative. These results fail positive
expectancy, confidence, stress, and the requested $1,000 daily target.

The [results record](../research/NQ_TBBO_PROFILE_RECLAIM_RESULTS.md) links the
full manifest, source hashes, candidate ledgers, frozen selection, and gate.
The policy never reads its reserve when this gate fails.

## Consequences

The new feature and quote-replay modules remain `EXPERIMENTAL` and are useful
as audited primitives for a separately pre-specified future hypothesis. That
hypothesis must receive a new date split and must not reuse the frozen
April–June validation as a tuning set. Nothing in this ADR changes Track B or
Track C status, price tape, existing models, or execution services.
