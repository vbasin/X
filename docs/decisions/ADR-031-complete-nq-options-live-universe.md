# ADR-031: Complete NQ Options Live Parent Universe

## Status
Accepted for the historical / repair parent tuple. Live BBO/definition subscribe
superseded by [ADR-034](ADR-034-five-chain-live-option-subscribe.md) (2026-08-15).

## Date
2026-08-06

---

## Context

Historical acquisition used separate ad hoc paths:

- `NQ.OPT` parent for quarterly options;
- exact raw symbols expiring on each session from 24 daily-weekly parents plus
  `QNE.OPT` for 0DTE;
- those same symbols on the prior CME trading day for 1DTE;
- `QNE.OPT` monthly acquisition for the end-of-month surface.

The live streamer subscribed only to `NQ.OPT`. That could populate quarterly-driven
GEX while leaving IV columns null because the feature engine requires quotes collected
today for both today's expiry and the next trading day's expiry, plus the next two QNE
expiries for 30-day constant-maturity IV.

## Decision

1. `src.acquisition.options_universe` is the source of truth for the live option
   parents: `NQ.OPT`, `Q1A.OPT`–`Q5D.OPT`, `QN1.OPT`–`QN4.OPT`, and `QNE.OPT`.
2. The persistent live client subscribes to both `bbo-1s` and `definition` for the
   complete parent tuple.
3. Feature modules continue to classify quotes by expiration:
   - expiry today → 0DTE;
   - expiry next CME trading day → 1DTE;
   - next two future QNE expiries → 30-day constant-maturity IV;
   - positive DTE through 90 days → dealer GEX.
4. “1DTE” is not a separate parent or delayed calculation. Its current-session quotes
   must be collected before the contract becomes 0DTE tomorrow.

## Alternatives Considered

**Daily exact-raw-symbol subscription**, mirroring historical backfill, would reduce
stream volume. It was rejected for the initial live implementation because the symbol
set must be complete before market-data subscription begins. That introduces a hard
dependency on a fresh definition download/map and can silently omit a chain when the map
is stale, a parent definition request is incomplete, or a holiday shifts the next trading
day. Parent subscriptions favor completeness and let definitions and BBO arrive together.

This decision can be revisited after production volume is measured. Any future exact-symbol
implementation must refresh definitions atomically, validate coverage for all four feature
groups, and fail closed rather than stream a partial universe.

## Consequences

**Easier**

- Live acquisition matches the complete economic universe represented by the backfill
  scripts without maintaining separate 0DTE and 1DTE subscriptions.
- Newly listed strikes arrive through the parent definition feed automatically.
- IV completeness no longer depends on `NQ.OPT` containing weekly/QNE products.

**Harder**

- Incoming BBO and definition volume increases and must be observed after deployment.
- Database growth and callback throughput become explicit P5 acceptance checks.

## Related

- [DATA_PIPELINE.md](../architecture/DATA_PIPELINE.md)
- [SYSTEM.md](../architecture/SYSTEM.md)
- [ROADMAP.md](../roadmap/ROADMAP.md)