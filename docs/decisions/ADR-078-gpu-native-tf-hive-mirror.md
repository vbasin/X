# ADR-078: GPU native-TF hive mirror (`timeframe={tf}/` + recursive keys)

## Status
Accepted — implemented 2026-08-27. Amended by [ADR-102](ADR-102-gpu-htf-context-hydrate.md) (1h/1d HTF sidecar on GPU jobs). Live pin unchanged.

## Date
2026-08-27

Amends the GPU hydrate half of [ADR-022](ADR-022-oci-queue-code-bundle-autonomous-gpu.md)
and the Custom Test store path in [ADR-076](ADR-076-custom-test-store-head-labels.md).

## Context

Custom Test `nq-50pt-econ-lg-st/20260827T221956Z` (15s LSTM) failed with
`No walk-forward folds built from parquet data.` Local CPU fold build on
the same topology/walk-forward overlay succeeded (~484k rows). The GPU
job never saw the hive.

Dispatch used `data_s3_prefix = {pdir.name}/`. Training `parquet_dir` for
the 15s store is `…/timeframe=15s/symbol=NQ`, so the prefix became
`symbol=NQ/`. GPU boot `sync_parquet.py` pulls `timeframe=15s/` into
`sources/features/timeframe=15s`. `push_features` was a non-recursive
`pdir.glob("features_*.parquet")` keyed by **basename**; hive files sit
at `year=/month=/features_15s_part1.parquet` (the same basename in every
month). Result: `uploaded=0 skipped=0`. The GPU VM was already RUNNING,
so ExecStartPre did not re-pull. `job_exec` hydrated the wrong prefix
into `symbol=NQ` and flattened keys to the filename.

15s dual-binary labels live at `sources/labels/parquet_labels_15s`
(`data_paths._prefer`). `oci_sync.conf` pointed at the empty canonical
`sources/features/parquet_labels_15s`, so the daily timer never uploaded
them, and `job_exec` did not pull a labels prefix.

## Decision

1. **Prefix from hive root, not `pdir.name`.**
   `gpu_feature_mirror(pdir)` maps `timeframe={tf}/symbol=…` → push root
   `timeframe={tf}` and prefix `timeframe={tf}/`. Flat 5s day files stay
   `timeframe={tf}/` for every hive store including 5s
   ([ADR-079](ADR-079-5s-hive-monthly-parts.md) /
   [ADR-081](ADR-081-5s-hive-cutover.md)). Leftover `parquet_5s` paths
   remap to `timeframe=5s/`, not `parquet/`.
2. **Recursive relative keys.** `push_features` / `pull_features` walk
   the tree and store `symbol=NQ/year=/month=/features_*.parquet` under
   the prefix. Same-named monthly parts do not collide.
3. **`data_s3_dest` is the hive root.** `job_exec` pulls into that dest
   (not into `parquet_dir` = `symbol=NQ`) so boot sync and job hydrate
   land on the same tree. `parquet_dir` is unchanged for the fold builder.
4. **Label sidecar.** Native-TF jobs push/pull `parquet_labels_15s/`
   (and `_contract.json`); 5s jobs push `parquet_labels_5s/` when present.
   GPU dest is `parquet_labels_*_dir()` (canonical
   `sources/features/parquet_labels_15s` when that path does not yet
   exist on the GPU).
5. **Fail closed on empty mirror.** If local hive files exist and a
   successful list+push mirrors 0 of N, dispatch fails on the CPU with
   that error instead of enqueueing a doomed GPU job.
6. **`oci_sync.conf` local path** for `parquet_labels_15s/` is the live
   dir (`sources/labels/parquet_labels_15s`). Prefix unchanged; GPU boot
   still hydrates into `sources/features/parquet_labels_15s`.

## Consequences

- Re-run the 15s LSTM Custom Test after this lands (code bundle + push).
  Do not reuse `20260827T221956Z`.
- 5s GPU jobs: prefix `timeframe=5s/` (ADR-081). Do not use `parquet/`.
- Live pin unchanged.

## Related

- [ADR-022](ADR-022-oci-queue-code-bundle-autonomous-gpu.md) — GPU job payload
- [ADR-070](ADR-070-native-tf-hive-layout.md) — hive keys
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s monthly parts
- [ADR-076](ADR-076-custom-test-store-head-labels.md) — store × head × labels
- [OCI_SYNC.md](../operations/OCI_SYNC.md)
