# ADR-155: Weekly live session, 16:05 CT session roll, rotating DBN archive

## Status

Accepted — 2026-10-06. Implemented in source; **requires a `vtech-live-streamer`
restart to take effect**, and the first live week is the real validation (see
Consequences).

**Supersedes** [ADR-034](ADR-034-five-chain-live-option-subscribe.md) on the
*session lifecycle* only — specifically its decision 4, "A Chicago session-date
roll stops the client; `LiveStreamer` opens a new session and archive file."
ADR-034's five-chain expiry-exact subscribe, its `raw_symbol` scope, its
fail-closed rule and its 26-parent historical tuple are all unchanged and remain
in force. Its note that "the client chunks lists at 500 symbols" is also correct
and refers to `databento-python`'s own `protocol.py`, not to code in this
repository.

## Date

2026-10-06

---

## Context

ADR-034 left the streamer running five sessions a week: the option-refresh loop
compared the Chicago session date against its own and called `client.stop()` on
a mismatch, after which the outer loop opened a fresh `db.Live` connection and a
new archive file. Three problems with that, in order of cost.

### 1. The boundary was in the middle of live trading

The session date came from the Chicago *calendar* date, so the teardown happened
at CT midnight. Measured gaps in `databento.bbo_1s` across the 2026-09-28 week:

| boundary | last row | resumes | gap |
|---|---|---|---:|
| 2026-09-29 | 05:02:28 | 05:02:53 | 25 s |
| 2026-09-30 | 05:03:59 | 05:04:49 | 50 s |
| 2026-10-01 | 05:03:10 | 05:03:52 | 42 s |
| 2026-10-02 | 05:02:03 | 05:02:13 | 10 s |

Roughly two minutes a week, during the Asian session, and **unrecoverable**: the
streamer never passes `start=` so no intraday replay happens, and
`_handle_reconnect` logged the gap boundaries Databento hands it into `extra={}`
fields the log formatter was discarding.

Meanwhile 16:01–16:44 CT is measurably dead. Per-minute counts for 2026-09-29,
confirmed on 2026-09-28, 09-30 and 10-01:

```
15:59   5,584 rows
16:00   1,276 rows   <- closing snapshot, one final quote per instrument
16:01 - 16:44   ZERO ROWS, every session
16:45       6
16:55      18        <- pre-open trickle
17:00   9,414 rows   <- Globex reopen
```

The 16:00 hour totals 1,335 rows against 462,629 in the 15:00 hour — 0.3%.

### 2. The calendar did not match the session being traded

A CME equity-index session runs 17:00 CT to 16:00 CT the next day and is named
for the day it *ends*. Keying on the calendar date meant that from 17:00 to
24:00 CT the streamer called "0DTE" the chain that had already expired at 15:00
CT that afternoon, while the chain expiring at the end of the *open* session was
only subscribed as "1DTE".

### 3. Nothing required the teardown

Databento documents no session lifetime and
[recommends reusing a session](https://databento.com/blog/changes-to-live-connection-limits)
rather than opening new ones. The only scheduled CME Globex gateway restart is
Saturday 02:00 CT — inside the weekend halt this process is already stopped for.
And a single connection already spanned the daily 16:00–17:00 CT halt in
practice: `raw/live/2026-10-01/stream_050310.dbn.zst` ran Thu 05:03:10 UTC to
Fri 05:02:03 UTC, with 169,834 rows before the close, 1,533 during it and
318,944 after the reopen.

From 2026-04-26 Databento also stopped terminating a session on an invalid
symbol, which removes one failure mode a long-lived session would otherwise
carry.

---

## Decision

1. **One `db.Live` connection per week**, CME Sunday open to Friday close. The
   session-roll branch no longer calls `client.stop()`. `cme_weekend_closed()`
   is now the only thing that ends the connection.

2. **The session date advances at 16:05 CT**, via
   `options_universe.globex_session_date()`, which
   `live_option_session_date()` delegates to so there is one calendar
   implementation. Five minutes after the close, not at it: the 16:00 CT minute
   carries the closing snapshot (~1,300 rows, one final quote per instrument)
   and that data belongs to the session that is ending. With the refresh loop at
   300 s the roll is detected between 16:05 and 16:10 CT, inside the dead
   window.

3. **A roll is additive.** `_roll_session()` adopts the new date, flushes the
   per-session Greeks and surface-parquet buffers, and logs both dates; the
   refresh then subscribes the new chains on the live connection. Expired chains
   stay subscribed because Databento Live has no unsubscribe API — measured cost
   is **98–131 rows per expired chain**, all `UNDEF_PRICE` book-teardown
   markers, with the last row at exactly 16:00:01 CT on all four chains tested.

4. **The archive rotates inside the session.** `src/acquisition/rotating_stream.py`
   provides `RotatingDbnStream`, a writable byte stream for `Live.add_stream`
   that opens a new file whenever the session date advances, preserving the
   one-directory-per-session-date layout in
   [DATA_PIPELINE.md](../architecture/DATA_PIPELINE.md). `add_stream` has no
   removal counterpart, so without this a weekly session would write one file
   for the week.

   Two client properties make it safe, both verified against `databento-python`
   0.74.1: the DBN metadata header is written exactly once and is **not**
   re-emitted after a reconnect, so the class captures and replays it at the
   head of every rotated file; and `_dispatch_writes` calls
   `stream.write(bytes(record))` once per record, so a rotation taken between
   calls cannot split one.

   Files open **lazily, on the first record** of a session date. A roll with no
   records behind it creates nothing — which eliminates the header-only archive
   stubs reconnect churn produced under the per-session writer (three 13-byte
   files in 65 seconds on 2026-09-29).

5. **A missing definition is reported, not silently permanent.** The refresh
   previously diffed only on subscription state and returned early when nothing
   new resolved, so a strike that was subscribed and whose `definition` message
   never arrived was skipped forever. Definition presence is now a separate
   check that runs on every refresh, logging `option_definitions_missing` with
   counts and affected chain roots. It is an in-memory set difference against
   the bootstrap's view of `databento.definitions`, so it costs nothing. When
   that view is degraded (a failed table read, observed as
   `definitions_bootstrap_degraded`) the check is skipped rather than reporting
   every symbol as missing.

6. **`extra={}` log fields now reach the log.** `src/common/logging.py`
   configured the stdlib handler with `"%(message)s"`, discarding every
   structured field. `ExtraFormatter` appends them as `key=value` with the
   message still the leading token, so existing `journalctl | grep <event>`
   usage is unaffected. Without this, `live_option_universe`,
   `live_option_symbols_added`, `option_definitions_missing` and
   `reconnect_gap` are all unobservable — and a week-long session is only
   diagnosable through them.

---

## Alternatives considered

**Keep the per-session teardown and just move it to 16:05 CT.** This is a real
option and delivers most of the benefit: the gap moves into the dead window and
costs nothing. Rejected as the end state because it still re-resolves the whole
universe and reopens a connection five times a week for no reason, against
Databento's own guidance — but note that decision 2 alone achieves the
boundary win, so the two decisions can be deployed separately if the weekly
session proves troublesome.

**Parent-scope `definition` subscribe to fix the missing-definition gap.**
Rejected, and ADR-034's original rejection of it is vindicated for a better
reason than archive cleanliness: the live `definition` schema does not deliver
records for instruments listed *before* the subscribe. `symbology.resolve`
returns all 246 NQZ6 strikes, so all 246 are subscribed for `definition` at
`raw_symbol` scope, yet only 172 hold rows and no vendor definition has been
written since 2026-08-25. Parent scope would add ~35,000 out-of-scope records
without producing the 74 missing rows. The repair is a cost-gated Historical
`definition` pull — measured **$0.00**, ~520 B/strike — which is
`scripts/data/refresh_option_definitions.py`'s job, not a live callback's.

**Patch the replayed DBN header's `start` field per rotated file** via
`databento_dbn.update_encoded_metadata`. Rejected: undocumented signature, and
the field is a hint. Record timestamps are unaffected, and both the ingest path
(`src/ingestion/loader.py`, `scripts/data/ingest_dbn.py`) and `DBNStore` read
record timestamps rather than the header.

---

## Consequences

**Easier**

- No data lost at a session boundary. Four boundaries a week move from live
  trading into a window where zero rows are written.
- `0dte` is the chain the open session settles into. At Monday 17:00 CT, `0dte`
  expires Tuesday and `1dte` expires Wednesday.
- One archive file per session date still, but reconnect churn no longer
  produces empty stubs.
- A missing definition surfaces within one refresh cycle instead of never.

**Harder**

- Subscriptions accumulate across the week: **9 chains / ~8,000 symbols by
  Friday**, from ~3,900 on Sunday. Within normal bounds (the client chunks at
  500 and the pre-ADR-034 26-parent subscribe streamed far more), but a
  mid-week reconnect replays the full accumulated set.
- A mid-week reconnect gap is still unrecoverable — `start=` cannot be set after
  `Live.start()`, and Databento drops a post-start replay anchor. The gap window
  is now at least *logged* with its exact boundaries. Persisting those windows
  for later reconciliation, and any decision to spend on a historical backfill,
  are deliberately out of scope here (AGENTS.md §5 cost gate).
- `RotatingDbnStream` is the only genuinely new mechanism and is the main risk.
  It is covered by 18 tests including byte-for-byte record preservation through
  rotation on two real archive files, but it has never run against the live
  gateway.

**Unchanged, deliberately**

- `_refresh_0dte_set()` still keys on the Chicago calendar date of the quote,
  not the session date, so no live Greeks are computed in the evening session.
  That is pre-existing, self-healing per quote, and independent of this change;
  revisiting it is a question about the Greeks path, which ADR-153 established
  nothing reads.
- `src/common/cme_holidays.py` excludes early-close sessions as well as full
  closures, because truncated volume distorts features. Acquisition inherits
  that, so roughly eight sessions a year (the Friday after Thanksgiving, July 3
  and similar) resolve to the *next* full session and are not captured. Recorded
  as a known gap in `tests/test_globex_session_date.py`; splitting the
  acquisition calendar from the feature-building one is its own change.

**Validation**

`make test-acquisition` — 143 tests. New coverage:
`tests/test_globex_session_date.py` (session calendar and the roll point),
`tests/test_weekly_session_lifecycle.py` (roll is additive, buffers flush,
weekend close is the only stop, archive wiring),
`tests/test_rotating_dbn_stream.py` (rotation, header replay, concurrency, real
DBN round-trip), `tests/test_definition_gap_detection.py`,
`tests/test_logging_extras.py`.

What tests cannot establish is the thing that matters most: whether one
connection survives five 16:05 CT rolls with subscriptions accumulating
correctly. **The first live week is the validation.** Watch
`live_option_session_roll` (four per week, each between 16:05 and 16:10 CT),
`dbn_archive_opened` (one per session date), `live_option_symbols_added`
(cumulative total rising to ~8,000 by Friday), `reconnect_gap`, and
`option_definitions_missing`. All five are only visible because of decision 6.

## Related

- [ADR-034](ADR-034-five-chain-live-option-subscribe.md) — superseded on the session lifecycle only
- [ADR-031](ADR-031-complete-nq-options-live-universe.md) — the 26-parent historical tuple
- [ADR-151](ADR-151-definitions-underlying-and-settle-time.md) — `definitions.underlying` and provenance
- [ADR-153](ADR-153-options-iv-underlying-and-solver.md) — the write-only Greeks table and `--no-greeks-db`
- [ADR-060](ADR-060-vm-relative-limits-and-cme-live-hours.md) — the CME Sunday/Friday timers
- [DATA_PIPELINE.md](../architecture/DATA_PIPELINE.md) — archive layout
- [src/acquisition/rotating_stream.py](../../src/acquisition/rotating_stream.py)
