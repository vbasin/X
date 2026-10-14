# ADR-146: MBO live streaming — architecture built, blocked on Databento entitlement

**Date**: 2026-09-29
**Status**: Partially implemented; live MBO capture is disabled pending a Databento entitlement fix
**Related**: [ADR-145](ADR-145-live-chart-mbo-order-map.md) (Live Chart order-map overlay, reads whatever is in `databento.mbo` regardless of how it got there)

## Context

`databento.mbo` was populated by a nightly batch job (`vtech-mbo-daily.timer` → `scripts/ops/daily_mbo_pull.py`), unlike every other live schema (`tbbo`, `bbo-1s`, `definition`, `statistics`), which are captured in real time by `src.acquisition.live_streamer`. This meant MBO was always ~8h stale (Databento's *historical* `mbo` endpoint runs ~8h behind wall clock) and depended on a recurring timer that had, in practice, been silently failing every night since install (a persistent, unexplained 810-record mismatch on one specific day caused a hard failure that blocked every later day from being pulled — see the fix in `daily_mbo_pull.py`).

Decision: capture MBO the same way as tbbo — a live subscription on the existing `LiveStreamer` connection — and retire the nightly timer, keeping `daily_mbo_pull.py`/`backfill_mbo.py` only as manual/occasional gap-fill tools (mirroring how `scripts/data/backfill_gaps_20260925.py` exists for tbbo: a one-off tool, not a standing job).

## What was built

1. **`_BufferedTableWriter`** (`src/acquisition/live_streamer.py`) — a generic background batched Postgres writer, generalized from the resilience shape already proven in production for `bbo-1s` (queue → size/time-based batch flush → retry with backoff → bounded "degraded" window that drops rows rather than raising, since the raw `.dbn` archive is the system of record for replay → escalates only if Postgres stays down past the window).
2. **tbbo migrated onto it.** Its previous synchronous, per-record `conn.execute()` inside the shared record-callback thread meant a slow/contended Postgres commit on a trade tick could stall processing of the *next* record for every schema sharing that one live connection — a real latency/robustness gap, not something specific to adding MBO. Now batched like bbo-1s. Verified live in production: `databento.tbbo` rows landing within ~2s of receipt, `tests/test_live_options_subscription.py` (27 tests, covers `LiveStreamer` construction/dispatch) green.
3. **MBO writer built the same way**, sized for its much higher event rate (COPY-based flush, not multi-row INSERT — `databento.mbo` deliberately has no unique constraint, so there's no conflict-arbiter complexity to navigate).
4. **`subscribe_live_feeds()`**: an `mbo` subscription line was added, then **commented out** — see Incident below.
5. **`vtech-mbo-daily.timer`/`.service`**: disabled, unlinked from systemd, `daemon-reload`d. Source files kept in the repo, headers marked RETIRED, not deleted.
6. **`daily_mbo_pull.py`**: hard-fail-on-verification-mismatch removed (now logs a warning and returns 0, since a human runs this by hand now rather than a timer silently failing every night); docstring rewritten to describe its new role as a manual/occasional tool.

## Incident: the live MBO subscription broke live streaming for ~70 seconds

Deploying the `mbo` subscription (`client.subscribe(dataset=DATASET, schema="mbo", symbols="NQ.FUT", stype_in="parent")`) on 2026-09-29 09:01 UTC caused the Databento live gateway to reject it — `'Not authorized for mbo schema'` — and, critically, **that rejection tore down the entire live client connection**, not just the mbo subscription. Since tbbo/bbo-1s/statistics/options-definition all share one `db.Live` connection, this crash-looped the *whole* live streamer (`client.start()` → `ValueError: cannot start a live client after it is closed` → retry after 15s → same failure) for about 70 seconds across two crash-loop iterations, during which **no live data of any kind was captured** (verified via `journalctl`).

Root cause: **historical MBO access does not imply live MBO access.** `DATABENTO_PRICING.md`'s note that "the Live API... uses the same GLBX.MDP3 dataset and subscription" was read as covering this, and was wrong for this schema — real-time full order-book (L3) distribution is evidently a separately-licensed entitlement from CME/Databento, distinct from the historical batch access this account already has (which is what `backfill_mbo.py` has used successfully since the original MBO backfill). This should have been checked with a live connectivity test *before* deploying to the production connection, not discovered via a production crash loop. That's on me.

Fix: the `mbo` subscription line was commented out immediately (with the incident recorded in-line) and the service redeployed. Confirmed recovered: `databento.tbbo` and `databento.bbo_1s` both showed fresh rows within seconds of the fix.

## Current state

- **tbbo**: migrated to the buffered writer, live, verified working. This part of the plan is done regardless of the MBO entitlement outcome.
- **mbo live writer**: fully built and wired (`_insert_mbo`, `_flush_mbo_batch`, `mbo_writer_started` logs correctly on startup) but **inert** — the subscription that would feed it is commented out, so it never receives records. No behavior change from before this ADR other than the retired timer.
- **`databento.mbo`**: static at 2026-09-27 (the last successful nightly pull before the timer was retired). Not growing until either the entitlement is resolved (live) or a manual `daily_mbo_pull.py` run is made (batch, $0-gated, unchanged mechanism).

## Next steps (not done as part of this ADR)

1. Confirm with Databento (billing page or support) what entitlement live `mbo` requires and whether/how to add it to this account's Standard plan.
2. Once confirmed, re-enable the subscription — but only after a low-risk connectivity check (a short-lived standalone `db.Live` client subscribing to `mbo` alone, outside the production connection) rather than deploying straight onto the shared production session again.
3. Until then, `databento.mbo` will drift stale; a manual `python scripts/ops/daily_mbo_pull.py` run is the $0-gated way to catch it up in the interim, at the user's discretion (not run automatically as part of this ADR).
4. The live-first-then-gap-fill backfill ordering (start live capture, then batch-pull the historical range up to the live stream's first captured timestamp) remains the right approach for whenever live MBO does go live — it just hasn't been reachable yet.
