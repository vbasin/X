# ADR-151: `databento.definitions` — correct `underlying` resolution, PM settle time, and row provenance

## Status
Accepted — 2026-09-30.

## Date
2026-09-30

## Context

`databento.definitions` contained options whose `underlying` was `NQU6` with expirations up to
2026-09-29 — impossible, because the NQU6 future made its final settlement on 2026-09-18. The same
column also held `NQQ6` and `NQV6`, which have never been tradeable contracts at all: NQ futures list
quarterly months only (H/M/U/Z).

The cause was **not** vendor data. `LiveStreamer._register_resolved_weekly_from_bbo` reconstructs a
definition row by regex-parsing the option symbol off a BBO quote, for strikes that are quoted before
their definition message has been seen. That reconstruction fabricated two fields wrongly:

1. **`underlying`** was built as `"NQ" + <the OPTION's own month code>`. The month code in `QN4U6` is
   the *option's* contract month, not the underlying future's. Every serial and post-roll expiry was
   therefore mislabelled, and serial months produced contracts that do not exist.
2. **`expiration`** was hardcoded to `21:00` UTC. NQ weekly/daily options are PM-settled at 15:00 CT,
   which is 20:00 UTC under CDT and 21:00 UTC under CST — so every CDT expiry was stamped an hour
   late. All 28,603 reconstructed rows fell in the CDT window and were wrong.

The damage was permanent because the table's primary key is `symbol` alone and **every** insert path
used `ON CONFLICT (symbol) DO NOTHING`. The first writer won forever, so a later correct vendor
backfill could not repair a reconstructed row. Within a single chain (`QN4U6`, expiry 2026-09-25) the
strikes were split — those first seen 2026-08-12 correctly said `NQZ6`, those first seen 2026-09-24
said `NQU6`, interleaved by strike.

This also explains the apparent absence of NQZ6 dailies before 2026-09-25: the chains existed, but
were labelled `NQU6`. Only `QNEU6` (end-of-month) was correct, because the `QNE` root does not match
the reconstruction regex and so came from the vendor untouched.

## Decision

1. **Resolve `underlying` from the quarterly roll, not the symbol.** `_underlying_future_for_expiry`
   returns the first quarterly contract still trading at the option's expiry *instant*.
2. **Compare instants, not dates.** The quarterly future settles at its 08:30 CT fixing on the 3rd
   Friday; a 15:00 CT weekly expiring that same Friday outlives it and rolls to the next quarterly.
   Comparing dates alone is wrong on exactly four days a year. Validated against every authentic
   vendor row: **264,970 / 264,970 agreement**, up from 261,473 with a date-only comparison.
3. **Localize the settle time.** `expiration` is 15:00 `America/Chicago`, converted to UTC, never a
   hardcoded UTC hour.
4. **Record provenance.** `databento.definitions.source` is `'databento'` or `'synthetic'`
   (`CHECK` constrained, default `'databento'`).
5. **Let vendor rows win — except where the vendor has nothing to give.** All insert paths now
   upsert: a vendor row overwrites a `'synthetic'` row; nothing overwrites a vendor row.
   Reconstruction remains a fallback, but a self-correcting one. The exception is `multiplier` and
   `min_price_increment`: CME's definition record supplies no usable value for either on **100%** of
   vendor option rows (264,970/264,970 — `multiplier` NULL, `min_price_increment` NaN or a leaked
   DBN `UNDEF_PRICE` sentinel). Those two columns therefore keep whatever usable value is already
   stored, so an upgrade never trades a correct reconstructed 20/0.25 for garbage.
7. **The fail-safe column default is `'synthetic'`, not `'databento'`.** The column was *added*
   defaulting to `'databento'` so the one-time backfill classified existing (overwhelmingly vendor)
   rows correctly; the ongoing default is the opposite. A writer that forgets to set `source` should
   produce a correctable row, not one that silently claims vendor authority forever — which is
   exactly the failure mode this ADR exists to fix.
6. **Track the two sources separately in memory.** `LiveStreamer._known_symbols` holds vendor-backed
   symbols only; `_synthetic_symbols` holds reconstructed ones. Previously a reconstructed symbol
   entered `_known_symbols` and permanently suppressed the vendor write that would have fixed it.

## Consequences

- 28,603 rows repaired in place by `scripts/data/repair_synthetic_definitions.py` (idempotent).
  Pre-repair copies are preserved in `databento.definitions_backup_20260930`.
- Post-repair invariants hold: zero `NQU6` rows expiring after 2026-09-18 13:30 UTC, and zero
  underlyings referencing a non-quarterly NQ month.
- **DTE changed for every reconstructed row.** Time-to-expiry had been overstated by one hour, a 2x
  error for a 0DTE option an hour before the bell, feeding IV/gamma/theta in
  `src/features/options_greeks_pipeline.py`. Any Greeks, IV surface, or dealer-GEX artifact built
  from the affected span (expiries 2026-08-06 → 2026-10-16) should be rebuilt.
- `options_greeks_pipeline.py` takes `und_sym` as the *mode* of `defs["underlying"]`, which returned
  `NQU6` for post-roll chains. That is a provenance label only — spot comes from `nq_1s` — so priced
  Greeks were not corrupted by the label itself, but they were by the expiration.
- **The suspected NQZ6 coverage gap was an artefact of the mislabelling, not a real gap.** Before
  repair NQZ6 held 1,331 rows against 42–50k for each prior quarterly; after repair it holds 9,499
  across 14 expiries, which is proportionate for a contract two weeks into its quarter. Every
  business day from the 2026-09-18 roll onward has a chain, with no missing expiries. No coverage
  backfill is required.
- What the affected chains still lack is *provenance*: six of the nine post-roll expiries are
  reconstructed rather than vendor rows, so their `ts_event` is quote-arrival time rather than CME's
  publication time. `scripts/data/refresh_option_definitions.py` upgrades them (cost-checked, $0.00
  for the definition schema); this is polish, not a correctness fix, since `underlying` and
  `expiration` are already right.
- Known residual: the reconstruction assumes a 15:00 CT settle and does not model early-close
  sessions (authentic data shows 12:00 CT expiries around Thanksgiving and July 3). Those rows are
  now self-correcting once the vendor definition arrives.
- Separately observed, not addressed here: vendor rows carry `multiplier IS NULL` and
  `min_price_increment` of `NaN` or `9223372036.854776` — a DBN `UNDEF_PRICE` sentinel leaking
  through `_px()` unmapped.
