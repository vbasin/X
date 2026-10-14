# ADR-157 — bounded append-only Databento capture-gap recovery

**Status:** Accepted, 2026-10-06. **Scope:** The operator-authorized October 2–6 recovery.

The outage and restart envelopes overlap retained live records. TBBO and BBO do
not have event uniqueness constraints; timestamp or instrument-ID deduplication
would discard distinct events. Current definitions are one row per raw symbol,
so their latest historical ID cannot identify past quotes causally.

Acquire exact reviewed requests only after a fresh finite exact-$0.00 quote and
per-schema availability check. Preserve every existing archive; new attempts
have unique paths, SHA256 receipts, and a retained disk reserve.

Stage the whole decoded file in a session TEMP table, then reconcile complete
persisted payload occurrences with EXCEPT ALL. Match live DB timestamp rounding
and normalize legacy vendor sentinels only in comparison. Append missing rows
in bounded time intervals. Reject mapping, source-window, integrity and unique
key payload conflicts. Retry safely after partial commits. Preserve source
nanoseconds, schema fields and definition publication history in immutable DBN.

Rebuild downstream artifacts into an isolated run. Historical options attach
vendor terms by raw symbol and publication as-of, retaining each quote's own
instrument ID. Retain original publication nanoseconds separately from definition
availability: ordinary messages use the later of publication and receive times;
weekday-midnight snapshots retain earlier publication availability. Underlying
prices must be known at that exact quote timestamp.
Publish accepted, closed outputs only after preserving touched existing files;
retain all outside-scope rows. Preserve affected DB rows by actual quote
instrument IDs and time, stage and analyze candidates, and merge hourly with
durable checkpoints. Existing compressed-key payloads remain unchanged; missing
keys can be appended without an explicit chunk rewrite. The existing Greek force/delete path and native
writer's unlink behavior are not used for this recovery.

A vendor-source gap remains a quality finding even when all delivered records
are ingested. Existing 30-second option feature timing prevents claiming the
broad 5-second feature matrix is causal; its rebuild emits a blocked manifest
without invoking the feature engine pending a separately validated fix. Existing loaders and paths remain in place.
