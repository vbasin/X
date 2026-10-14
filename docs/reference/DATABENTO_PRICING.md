# Databento Plan & Pricing Reference

> **Last updated**: 2026-04-06
> **Note**: This is a point-in-time snapshot. Check [app.databento.com](https://app.databento.com) for current pricing and billing status.

---

## Subscription

| Detail | Value |
|--------|-------|
| Dataset | **GLBX.MDP3** (CME Globex) |
| Plan | **Standard** — $179/month |
| Renewal date | 2026-05-01 |
| Live license | CME personal license included (up to 2 devices) |

---

## Included Data Windows

| Level | Schemas | Included History |
|:-----:|---------|:----------------:|
| L0 | OHLCV-*, Definition, Statistics, Status | **15+ years** |
| L1 | MBP-1, TBBO, BBO-1s/1m, Trades | **12 months (rolling)** |
| L2 | MBP-10 | 1 month |
| L3 | MBO, Imbalance | 1 month |

Data outside the included window is available at pay-as-you-go rates.

---

## `get_cost()` API Behavior

`client.metadata.get_cost()` **respects subscription inclusions** and returns `$0.00` for data within the rolling window.

```python
import databento as db
client = db.Historical()
cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["NQ.FUT"],
    schema="tbbo",
    start="2025-04-06",   # must be ≥ today−365d to be free
    end="2026-04-07",
)
print(f"${cost:.4f}")     # $0.00 if within window
```

### L1 Rolling Window Example

The L1 window is a **rolling 12 months from today**:
- On 2026-04-02: `start=2025-04-01` → `$0.00` (within window)
- On 2026-04-06: `start=2025-04-01` → non-zero (5 days outside window)
- On 2026-04-06: `start=2025-04-06` → `$0.00` (within window)

**Always run the cost check script before submitting a batch job**:
```bash
python scripts/databento_cost_check.py --symbol NQ.FUT --schema tbbo --start 2025-04-06
```

---

## Batch Job Behavior

| Property | Value |
|----------|-------|
| Billing | Charged once at job submission |
| Re-downloads | Free, unlimited, for **30 days** after job completion |
| Recommended for | Data >5 GB |
| Download expiry | Must download within 30 days or re-submit |

---

## Active Batch Jobs

| Job ID | Schema | Size | Expires |
|--------|--------|------|---------|
| `GLBX-20260821-WLJAD4WJ5C` | mbp-10 (NQU6, 2026-07-21 → 08-21) | 12.80 GB downloaded; **438,266,822 rows ingested** 2026-08-21 | **~2026-09-20** |
| `GLBX-20260402-S4EMNFKM8K` | bbo-1s (NQ.OPT quarterly) | 56 GB compressed | 2026-05-02 (lapsed) |

> ⚠️ Re-downloads are free for 30 days after job completion. Pull
> `GLBX-20260821-WLJAD4WJ5C` before ~**2026-09-20** or re-submit.

### L2 / MBP-10 free-window edge (measured 2026-08-21)

`mbp-10` is **L2 — 1 month rolling**, a much tighter window than L1's 12 months.
Probing `get_cost()` for NQU6 → 2026-08-21 located the edge exactly:

| `start` | Cost |
|---------|------|
| 2026-07-15 | $8.9454 |
| 2026-07-16 | $6.2339 |
| 2026-07-17 | $3.3783 |
| **2026-07-18** | **$0.0000** ← edge |
| 2026-07-21 | $0.0000 |

The edge moves forward daily, and batch jobs bill **once at submission**. Submit with a
few days of buffer inside the edge (the 2026-08-21 job used `start=2026-07-21`, 3 days in)
and always re-check `get_cost()` immediately before `submit_job` — `ingest`/`download`
scripts here abort on a non-zero cost.

> **Volume warning**: MBP-10 is ~440× the row count of `bbo-1s` for the same instrument.
> One month of the NQ lead contract alone is 438,266,822 records / 161 GB uncompressed
> (~13 GB as `.dbn.zst`). Never request `mbp-10` for an `NQ.OPT` or `NQ.FUT` parent symbol.

### L3 / MBO free-window edge (measured 2026-09-25)

`mbo` is **L3 — 1 month rolling**, the same window as MBP-10. Probing `get_cost()` for a single
NQ contract by `raw_symbol`, one day at a time:

| `start` | NQZ6 | NQU6 |
|---------|------|------|
| 2026-08-20 | $0.2874 | $2.3856 |
| 2026-08-21 | $0.1843 | $1.6544 |
| **2026-08-22** | **$0.0000** ← edge | **$0.0000** ← edge |

The edge sits ~34 days back and moves forward daily; anything older than the window is lost at
$0.00. The 2026-08-22 → 2026-09-25 window for NQU6 (→ 09-13) + NQZ6 (09-13 →) was pulled and
ingested on 2026-09-25 (475,152,566 records, 26.6 GB uncompressed, ~8 GB `.dbn.zst`, **$0.00**).
Use UTC-midnight boundaries so Databento adds a synthetic book snapshot at each day start.
Never request `mbo` for an `NQ.OPT` or `NQ.FUT` parent — pull lead contracts by `raw_symbol`.
Batch jobs: `GLBX-20260925-H5Q47G8CSX` (NQU6), `GLBX-20260925-UH3TKHYVTE` (NQZ6),
`GLBX-20260926-5E98G3Y3P7` (NQZ6 2026-09-25 → 21:00 UTC); re-downloadable free until **~2026-10-25**.

**Automation**: `vtech-mbo-daily.timer` (03:30 America/Chicago) runs `scripts/ops/daily_mbo_pull.py`, which keeps `databento.mbo`
current inside this free window (see DATA_PIPELINE.md). A run missed for more than ~1 month cannot be recovered at $0.

**Availability lag**: the `mbo` schema is served about **8 hours behind real time** (measured 2026-09-26:
`get_dataset_range` mbo end = now − 8 h), and `get_cost` returns 422 `dataset_unavailable_range` for an
`end` beyond it. A session that closes at 21:00 UTC is fully available ~05:00 UTC next day; request
`end` as a timestamp (e.g. `2026-09-25T21:00:00`), not the next midnight, until the feed catches up.

---

## Definition Schema Cost

The `definition` schema is **L0** — included in the 15+ year window at no cost regardless of date range. Downloading weekly 0DTE option definitions (Phase 17C) will incur `$0.00`.

---

## Caveats

1. **Rolling window**: The L1 start date must stay within 12 months of today or `get_cost()` will show charges for the out-of-window portion. The dynamic cost check script uses `today − 365d` → `today`.
2. **Live vs Historical**: The Live API streams over the same GLBX.MDP3 dataset, and each connected device counts against the 2-device CME license limit — but **historical access to a schema does not imply live access to it**. Measured 2026-09-29: this account can pull `mbo` historically ($0 within the rolling window, as above) but a live `client.subscribe(schema="mbo", ...)` is rejected by the gateway with `'Not authorized for mbo schema'`. Worse, that rejection tore down the *entire* live connection (tbbo/bbo-1s/statistics/options all share one), not just the mbo subscription — crash-looping the whole live streamer for ~70s until reverted (ADR-146). Before adding any new schema to a live subscription, test it on a short-lived standalone connection first, never directly on the shared production one.
3. **Schema availability**: Not all schemas are available for all datasets and time periods. Verify with `client.metadata.get_dataset_condition()` — note this only reports historical availability, not live entitlement (there is no equivalent live-entitlement-check call found so far).
