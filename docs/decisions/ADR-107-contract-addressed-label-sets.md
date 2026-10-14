# ADR-107: Contract-addressed label sets and run provenance

## Status
Accepted — implemented 2026-09-02; extended to 5s and 1s 2026-09-03. Live pin unchanged.

## Date
2026-09-02

## Context

Mutable 5s and 15s label directories could each hold only one TBM contract.
Changing the target, MAE, horizon, encoding, or target observation would overwrite
labels needed to reproduce another objective or historic research run. Day-sized
label files also created hundreds of small files for one contract.

## Decision

1. A normalized label contract deterministically hashes to a 16-character
   `contract_id`. Objective label parameters select this ID; YAML does not name
   an arbitrary directory. The directory name deliberately has no creation-time
   suffix: the same semantics must always resolve to the same path.
2. Label sets live at `sources/labels/<resolution>/<contract_id>/`, where
   `<resolution>` is `1s`, `5s`, or `15s`. Each set contains `_contract.json` plus
   ordered zstd `labels_<resolution>_partN.parquet` files, capped near 100 MB.
   For `1s` pathlabels, `sources/labels/1s/<contract_id>/` contains `_contract.json`,
   `_geography.csv` metadata, and yearly subfolders (`year=YYYY/labels_1s_partN.parquet`).
3. Fold construction resolves only the set matching its objective contract.
   CPU and GPU mirror that same contract-addressed directory. A missing or
   mismatched 5s set retains the existing in-memory label-generation fallback.
4. Every native-TF search or Custom Test writes `<run>/label_contract.json` at
   the research-output root with the selected contract, source directory,
   contract digest, and part inventory. `_contract.json` records immutable
   `created_at_utc` metadata for human inspection without changing the resolver
   key.
5. Historic flat `parquet_labels_{5s,15s}/labels_YYYY-MM-DD.parquet` files are
   left intact as legacy audit artifacts and are not selected by current code.
   The initial close-only 15s set was compacted without modification into
   `sources/labels/15s/1aa87aaf223d0f7a/` on 2026-09-02; the 5s three-class set with
   auxiliary targets was compacted losslessly into `sources/labels/5s/97d8f02d91e3f1c8/`
   on 2026-09-03. The legacy close-only 5s set from `parquet_labels_5s` was converted into
   `sources/labels/5s/0504f68a26bd64a1/`, and `parquet_pathlabels` was converted into
   `sources/labels/1s/998d02439ab8f190/` with yearly subfolders and preserved geography metadata on 2026-09-03.

## Consequences

- Multiple objectives may use different 5s and 15s TBM contracts concurrently.
- A run is reproducible without copying the label parquet data into every run.
- The first run under a contract requires a build. `--force` deliberately
  regenerates that contract's compact part set when source features change.

## Related

- [ADR-076](ADR-076-custom-test-store-head-labels.md) — persisted labels
- [ADR-106](ADR-106-configurable-tbm-target-observation.md) — target source
- `src/training/label_store.py`, `scripts/features/build_labels_5s.py`,
  `scripts/features/build_dual_binary_15s.py`