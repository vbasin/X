# ADR-032: Risk Gateway and Atomic Broker Protocol

## Status
Accepted — implemented but not activated, 2026-08-06.

## Date
2026-08-06

---

## Context

The live signal ledger records model intent: timestamp, direction, confidence,
reference price, objective, and barriers. It does not contain a broker command ID,
exact futures contract/account, expiry, idempotency state, risk decision, broker
acknowledgement, order state, or fill reconciliation. Allowing NinjaTrader to trade
that ledger directly would merge inference, policy, transport, and broker state and
could duplicate orders after a restart or parse a partially written record.

## Decision

1. `signals.jsonl` remains an immutable audit input and is never broker truth.
2. A separate Linux process, `src.execution.gateway`, owns policy and risk checks.
3. Each approved signal maps to one deterministic command ID and one immutable JSON
   file. Files publish by same-directory temporary write + `fsync` + atomic rename.
4. NinjaTrader independently validates schema, expiry, exact account/instrument,
   SIM/LIVE authorization, kill switch, and flat strategy position.
5. NinjaTrader owns actual order submission and broker-native OCO lifecycle.
6. NinjaTrader returns immutable acknowledgements, order updates, and incremental
   execution fills through an atomic inbox. The broker blotter is replayed only from
   these events; the paper engine remains a separate simulation ledger.
7. Runtime defaults are disabled, `SIM`, `Sim101`, and no instrument. LIVE requires
   independent arming in both Linux config and the NinjaTrader strategy.

## Consequences

**Easier**

- Model intent, risk decisions, executable commands, and broker facts are auditable.
- Deterministic IDs plus NinjaTrader processed markers prevent ordinary duplicate
  submission across polling and restarts.
- Either machine can restart and replay immutable files.
- The Windows host can run anywhere with a private shared transport.

**Harder**

- A private bidirectional mount and Windows deployment must be operated securely.
- SMB/network filesystems must be validated for same-directory rename semantics.
- Broker disconnects, partial fills, protective-order rejection, and reconciliation
  require explicit Playback/Sim101 drills before production authorization.

## Rejected alternatives

- **NinjaTrader tails `signals.jsonl` directly**: no broker command envelope or
  central risk decision and unsafe replay semantics.
- **Linux submits broker orders directly**: duplicates broker-session/order-state
  ownership and bypasses NinjaTrader's native strategy/OCO lifecycle.
- **One append-only shared command file**: readers can observe an incomplete final
  line and cursor recovery is harder than immutable per-message files.

## Related

- [WORKFLOW.md](../architecture/WORKFLOW.md)
- [SYSTEM.md](../architecture/SYSTEM.md)
- [NINJATRADER_EXECUTION.md](../operations/NINJATRADER_EXECUTION.md)