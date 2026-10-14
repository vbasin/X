# ADR-156 — Normal option subscription and vendor definition acquisition

**Date:** 2026-10-06  
**Status:** Accepted and implemented  
**Supersedes:** ADR-151's live synthetic reconstruction; ADR-155's reporting-only definition gaps and header-only archive rotation. ADR-151's vendor precedence and usable-tick preservation remain, with the incomplete-row exception below. ADR-034's five-chain subscription selection remains.

## Context

A symbol-mapping message establishes the current gateway instrument ID and raw
symbol. It does not supply authoritative strike, expiration or underlying terms.
The definitions table has a primary key on raw symbol and contains historical
instrument IDs; those IDs cannot seed today's live mapping.

The old BBO fallback reconstructed terms and inserted synthetic rows even when
the vendor's prior-day definition was available. `Q1CV6 P25000` demonstrated
that identity and core terms matched, but the synthetic publication timestamp
and minimum tick were not vendor facts. A live definition subscription without
an explicit replay start does not guarantee an initial snapshot.

## Decision

1. On startup and each subscription refresh, submit the accumulated exact option
   symbols to `DefinitionAcquirer` alongside live subscription. It runs in the
   same service with nonblocking submission, host-sized parallel batches and
   coalesced five-minute retries. This is normal acquisition, not a repair service.
2. Read indexed, committed vendor rows by raw symbol first. Missing rows, legacy
   synthetic rows and incomplete core vendor rows need acquisition. Existing
   usable vendor rows need no historical download.
3. Query `GLBX.MDP3`, `definition`, `raw_symbol` for the latest available partial
   UTC day plus its preceding daily snapshot. Clamp to schema availability and
   the current time. Quote the exact same request before downloading. Proceed
   only for a finite, exact $0.00 quote; any other quote or API failure leaves
   symbols pending. Paid downloads require explicit operator approval.
4. Normalize actual vendor records, select the latest usable publication per
   requested symbol and upsert in a short transaction. No connection is held
   during vendor network calls. Reread committed winners before publishing the
   cache, so a concurrent live vendor write cannot be shadowed.
5. Synthetic rows can become vendor rows; usable existing vendor rows are
   protected. An incomplete vendor row can be corrected only by a vendor record
   with an equal or newer publication timestamp. Core usability requires C/P,
   positive finite strike, expiration and underlying. No quote creates a
   synthetic definition. Historical synthetic rows outside subscriptions remain.
6. Join vendor terms by raw symbol to the current live ID, checking publication
   time is at or before the quote. Capture raw quotes while metadata is pending;
   defer their option calculations. Clear stale option state on ID reassignment.
7. Normalize contract size from `unit_of_measure_qty`; do not invent a multiplier
   or tick. Missing vendor tick remains null on a new row. The existing upsert
   preserves a previously stored usable tick, whose legacy provenance is not
   independently verified by this acquisition. Retain vendor publication time.
8. Cache received DBN symbol-mapping records, including aliases and validity
   intervals. Each rotated archive begins with its original metadata header and
   those mapping records before new records. Preserve original receipt order
   and bytes; decoding does not depend on another archive or a database query.

## Consequences and validation

Vendor or database outages do not fabricate terms or interrupt quote capture.
Pending coverage and blocked-cost events are logged and retried. Metadata
availability can lag new listings, so a newly subscribed symbol may temporarily
lack Greeks. Shutdown stops further acquisition and joins its worker; endpoint
and transaction timeouts limit outstanding work.

The acquisition scope exercises provenance, cost checks, live ID binding,
causality, concurrent arrivals, retries, shutdown and standalone rotated DBN
resolution. Deployment evidence and operational checks are recorded in
[the lifecycle guide](../operations/NORMAL_OPTION_DEFINITION_LIFECYCLE.md).
