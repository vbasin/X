# ADR-034: Five-Chain Live Option Subscribe (raw_symbol)

## Status
Accepted — implemented in source 2026-08-15. Requires `vtech-live-streamer` restart
to take effect on the running process.

**Decision 4 (session lifecycle) superseded by
[ADR-155](ADR-155-weekly-live-session-and-rotating-archive.md) (2026-10-06):** a
session-date roll no longer stops the client. One `db.Live` connection now spans
the CME week and the roll is additive at 16:05 CT. Everything else here — the
five-chain expiry-exact set, `raw_symbol` scope, fail-closed on an incomplete
resolve, and the 26-parent historical tuple — is unchanged and still in force.

## Date
2026-08-15

**Supersedes:** [ADR-031](ADR-031-complete-nq-options-live-universe.md) for the *live
BBO/definition subscription* only. The 26-parent tuple remains the historical /
repair parent list.

---

## Context

ADR-031 subscribed `bbo-1s` + `definition` at parent scope for all 26 NQ option
parents (`NQ.OPT`, `Q1A`–`Q5D`, `QN1`–`QN4`, `QNE.OPT`). That guaranteed 0DTE,
1DTE, both QNE surface chains, and the front quarterly would be present, but it
also streamed every other listed weekly, later QNE months, quarterlies through
2030, and `UD:` combo mappings.

On 2026-08-14 ~68% of live option BBO volume was outside the five expiries the
feature engine actually uses:

| Keep | Extra |
|---|---|
| 0DTE, 1DTE, closest QNE, 2nd closest QNE, closest quarterly | other weeklies, 3rd+ QNE, back quarterlies, `UD:2V:` |

Databento Live has no expiry filter. `stype_in=parent` always expands to the
whole product family. Historical backfill already measured the same waste
(`scripts/data/backfill_missing_option_chains.py`): one weekly parent was +52%
vs the exact strike list.

ADR-031 rejected exact-symbol subscribe because a stale local definition map
can omit a chain. That objection still holds for **local** `databento.definitions`.
It does not hold for Databento `symbology.resolve` (free metadata, complete for
the session date).

Live dealer GEX (ADR-010) previously saw every 0–90 DTE expiry present in
`bbo_1s`. After this change the live GEX window is the five-chain subset. That
is an accepted consequence of the operator rule that anything else is extra.
Historical GEX on backfilled days is unchanged.

Numbering note: this decision is ADR-034 on `/opt/vtech` because ADR-033 is
already [N-base output-meta](ADR-033-output-meta-n-bases.md).

## Decision

1. Each Chicago session streams **exactly five option expiries**, resolved by
   calendar then by vendor symbology:
   - `0dte` — session date
   - `1dte` — next CME trading day
   - `qne1` / `qne2` — next two QNE month-ends *strictly after* the session date
     (so a month-end 0DTE does not collapse onto QNE1)
   - `qtr` — closest NQ quarterly (3rd Friday of Mar/Jun/Sep/Dec) on or after
     the session date
2. Strike lists come from `Historical.symbology.resolve(parent → instrument_id)`
   for the four/five parents that own those roots. Only `raw_symbol` matches of
   `{root} [CP]{strike}` are kept. Other months under the same parent, and
   `UD:` / spread symbols, are dropped.
3. The live client subscribes those raw symbols for `definition` and `bbo-1s`.
   NQ/ES futures stay on `parent`. The client chunks lists at 500 symbols.
4. A background refresh (default 300s) re-resolves and `subscribe()`s newly
   listed strikes. A Chicago session-date roll stops the client; `LiveStreamer`
   opens a new session and archive file. There is no Live unsubscribe API.
5. Fail closed on options: never fall back to the 26-parent BBO subscribe. An
   incomplete resolve logs critical and streams only the chains that resolved,
   plus futures. Local `definitions` are never used to enumerate the live set.

`src.acquisition.options_universe` is the source of truth for both the parent
tuple (historical) and the five-chain resolver (live).

## Alternatives Considered

**Five parent symbols instead of raw strikes.** Today that would be `QN2.OPT` +
`Q3A.OPT` + `QNE.OPT` + `NQ.OPT`. Still pulls other months of those series
(e.g. `QN2U6`, `QNEV6`, `NQZ6`…`NQZ0`) and `UD:` mappings. Rejected — the
operator rule is expiry-exact.

**`stype_in=instrument_id`.** Session-scoped IDs are reused across days. Rejected.

**Local definitions table as the strike list.** Incomplete (missing expiries and
strikes). Same reason ADR-031 rejected a local map. Vendor `symbology.resolve`
is the complete, free substitute.

**Keep parent `definition` for discovery, raw_symbol only for BBO.** Extra
product defs would still land in the live `.dbn` via `add_stream`. Rejected
for archive cleanliness. New strikes are discovered by the periodic resolve.

## Consequences

**Easier**

- Live archive and `bbo_1s` ingest drop ~⅔ of option quote volume.
- Same five-chain identity as the historical gap-backfill driver.
- Newly listed strikes on a keep-chain still arrive (refresh + session restart).

**Harder**

- Live GEX no longer sees intervening weeklies (accepted).
- Startup depends on a successful metadata resolve; futures still start if
  options resolve fails.
- Requires a streamer restart (or CT date roll) to apply on a running process.

## Consumer alignment (added 2026-08-20)

The subscribe change alone left the *consumer* on the old contract.
`FeatureEngine._load_bbo_1s` scoped options by "any chain not yet expired"
(documented variously as a 100-day or 400-day window, while the SQL had no bound
at all), justified by a v3-era comment about needing back-month quarterlies for
term structure. `options_surface` v4 had already redefined `iv_term_slope` as
30d CM-IV − iv_1dte_atm, and `dealer_gex` caps at 90 DTE — so no module read
those back-months. On 2026-08-20 the filter matched 10,300 option instruments
against the 3,202 the five chains need: a 3.2x over-read on every load, and the
likely driver of the `vtech-live-features` OOM restart loop under `MemoryMax=32G`.

Changes:

- `_load_bbo_1s` now scopes options via `FeatureEngine._session_chain_roots(d)`,
  which delegates to `session_option_targets` — the same resolver the streamer
  uses, so producer and consumer cannot drift.
- Matched on chain **root**, not expiration date: other series can share an
  expiry with a keep-chain (the Sep weekly `QN3U6` expires on the same Friday as
  the `NQU6` quarterly), and `dealer_gex` selects purely on DTE, so an
  expiry-based scope would readmit co-expiring series into its gamma sum.
- `VTECH_BBO_PRIMARY_ONLY` now defaults **on**. It previously defaulted off,
  so `vtech-feature-builder` (no env var) loaded the whole unfiltered day while
  `vtech-live-features` (env var set) loaded the narrow set — the same session
  produced different features depending on which service built it. Set
  `VTECH_BBO_PRIMARY_ONLY=0` for a deliberate full-universe load.

Measured on 2026-08-13 14:30–15:00Z (69.4% fewer rows, futures unchanged):

| Feature group | Effect |
|---|---|
| `iv_skew`, `iv_25d_risk_reversal` | bit-identical (QNE chains selected from `definitions`, both retained) |
| `iv_0dte_atm`, `iv_1dte_atm` | identical to float noise |
| `iv_atm_mean` / `iv_butterfly` | 1.1% / 6.1% median — quote-density change alters ffill alignment |
| `gx_*` | 33–84% median — the accepted "GEX no longer sees intervening weeklies" |

`gx_dealer_gex` is an unweighted `nansum` over every selected instrument with
`oi = bid_sz + ask_sz` as an OI proxy, so it scales with subscription breadth.
Its history therefore spans three regimes — pre-2026-08-12 (`NQ.OPT` only),
2026-08-12→20 (26 parents), and five-chain thereafter. **Cached historical
features must be rebuilt** so the series is on one scale, and normalising this
feature (e.g. per-instrument mean) would make it regime-stable.

## Related

- [ADR-031](ADR-031-complete-nq-options-live-universe.md) (superseded for live subscribe)
- [ADR-010](ADR-010-gex-expiry-window-90d.md)
- [ADR-033](ADR-033-output-meta-n-bases.md) (unrelated; occupies the 033 slot)
- [DATA_PIPELINE.md](../architecture/DATA_PIPELINE.md)
- [src/acquisition/options_universe.py](../../src/acquisition/options_universe.py)
