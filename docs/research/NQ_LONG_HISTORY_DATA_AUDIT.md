# NQ long-history data audit — 2026-09-24

The independent research retry aggregates existing source files with
`scripts/research/build_nq_long_bars.py`. This adds a research preparation artifact;
it does not change the source cache, existing features, or any production model.

## Scope and source identity

The canonical `src.common.data_paths.parquet_path1s_dir()` resolves the input.
The historical build selects 2011-01-01 through 2024-12-31 and records every
source file's SHA-256, size, source row count, time bounds, minute coverage, and
quality counters. Default worker count comes from `get_vm_config().workers_for`
with a 2 GB per-worker budget. The output directory must not already exist;
source files are read-only. Dates on or after 2026-07-01 are rejected.

`scripts/data/build_path1s.py` reads `databento.ohlcv_1s`, passes `ts_event`
through to the index, and selects **07:00–15:15 America/Chicago**, with a default
symbol of `NQ.c.0`. The source files have OHLCV and UTC `ts`, but no instrument ID
or quote spread. Full overnight features cannot be constructed from this cache.

## Availability and causality

Databento's [OHLCV specification](https://databento.com/docs/schemas-and-data-formats/ohlcv)
defines timestamps as the interval start and omits intervals without trades.
The one-second close is therefore not available until the following second.
The new aggregation uses left-closed, right-labelled one-minute buckets:
`ts` is the minute's **close availability**, `bar_open_ts` is its start.
A source row stamped 08:31:00 belongs to the minute ending 08:32:00.
The cash-opening minute has `minute_ct=510` (08:30) but is only available at
08:31. This is theoretical historical availability; actual publication latency
is not measured by OHLCV files.

No scaling, smoothing, backfill, interpolation, or full-session normalization is
performed. Empty minutes remain absent. `observed_seconds` counts actual source
records; it is not a missing-data flag because no-trade seconds are omitted by
the provider. Features consuming this dataset must still use causal lookups and
prior-only fitted transforms; an honest timestamp alone cannot fix a feature
whose formula uses future observations.

Tests in `tests/test_nq_long_bars.py` check minute-boundary exclusion, availability,
future-perturbation invariance of already-closed bars, missing-minute behavior,
and fail-closed handling of duplicate, unsorted, malformed, or misdated sources.

## Contract-roll caveat

The input pipeline documents original unadjusted prices. The provider's
[continuous-contract documentation](https://databento.com/docs/examples/symbology/continuous)
also specifies unadjusted prices; `c.0` selects the nearest calendar expiry.
This differs from the generic Panama-adjustment statement in
`docs/DATA_CONTRACTS.md`, which is **not the contract of these path files**.

The tape pipeline uses `src/common/contract_calendar.py`, which rolls on the
Monday before the third Friday of the expiry month. Provider calendar-front
OHLCV and the tape's lead outright can differ during expiry week. The parquet
cache omits instrument IDs, so contract parity cannot be established from its
columns alone. Avoid cross-session absolute levels/overnight returns across
rolls; exclude expiry-week observations from direct contract-parity comparisons
or verify the underlying instruments separately. Intraday price differences
within a single contract do not require back-adjustment.

## Artifact

Preparation directory:
`data/research/runs/nq-edge-retry/preparation_20260924_long_history/`.
`bars_1m.parquet` is the research input; `source_inventory.parquet` and
`build_manifest.json` retain provenance and quality evidence. This preparation
artifact is not a fitted model run or a profitability claim.

The completed build took 2m 13s on six workers: **3,539 sessions and 1,725,494
minute bars**, 2011-01-03 through 2024-12-31. All inputs passed finite/positive
OHLC, ordering, uniqueness and session-date checks, with zero prices outside
the NQ quarter-point tick grid. Eight focused tests passed.

3,362 sessions have all 390 cash-session minute buckets; 177 are incomplete.
This includes shortened holiday sessions, thin expiry-contract activity, and
2020 cash-market halts. Maximum within-file source gap is 4,615 seconds on
2020-03-18; 2020-03-09/12/16 each lack 13–14 cash minutes. Missing intervals
cannot universally be treated as unchanged tradable prices. Excluding a whole
session at its open because it will have missing minutes later is itself a
future-dependent population filter. Calendar eligibility must be predetermined,
and execution must explicitly handle a data gap or halt when it occurs.

Output bars SHA-256:
`068a557a0b21548018b0c02a5a8c07879005cbc09dc7ab7dcf883daebbc6a550`.
