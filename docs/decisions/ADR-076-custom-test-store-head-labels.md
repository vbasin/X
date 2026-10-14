# ADR-076: Custom Test feature store × head × persisted labels

## Status
Accepted — implemented 2026-08-27. Live pin unchanged.
**Amended 2026-08-27 (ADR-078):** GPU dispatch pushes the 15s hive under
`timeframe=15s/` (recursive keys) and the live `parquet_labels_15s/`
sidecar; `job_exec` hydrates the hive root, not `symbol=NQ/`.
**Amended 2026-08-28:** GPU job spec carries `topology_overlay` (including
`feature_store` when `mode: none`). `job_exec` reloads the objective YAML
and must re-apply the overlay; without it the fold builder uses
`decision_timestep` (15s) and Custom Test 1m column lists silently
intersect to the 15s `cs_*` overlap (`20260828T130917Z`).
**Amended 2026-08-27 (ADR-080):** sequence families (LSTM, TCN, CNN,
Transformer) compile named `p_long` / `p_short` heads with per-head
`class_balance`; diagnostics lead with `labels: ["long", "short"]`.
`20260827T224538Z` trained dual-binary but reported 3-class diagnostics
and skipped sample weights.

## Date
2026-08-27

## Context

`nq-50pt-econ-lg-st` (ADR-066) needs Custom Test on native 15s features with a
dual-binary head and no meta. Custom Test was 5s-catalog-only, LSTM is 3-class
softmax, fold cache regenerates TBM labels from `_close`, and
`assert_tf_oof_meta_ready` fail-closed every search **and** Custom Test.

Approved shape (operator 2026-08-27): store (timeframe), not a parquet file;
head is a slot adapter so any family can run dual-binary the same way; persist
5s TBM labels like 15s and join on `ts`.

## Decision

1. **Feature store.** Custom Test board `feature_store` ∈ `{5s, 15s, 1m, 5m, 1h, 1d}`.
   The feature bank (`GET /api/custom-test/bank?store=`) reads that store's
   catalog. 1m catalog: [FEATURE_CATALOG_1m.md](../reference/FEATURE_CATALOG_1m.md)
   (ADR-085). 5m/1h/1d are listed but unavailable until hive files exist.
   Loader: `src.features.stores.load_feature_frame`. 15s uses hive parts,
   index = `bar_end_ts`. Compiled into topology overlay `feature_store` even
   when `mode: none`. Per-slot store is reserved (schema) but v1 is board-level.

2. **Head.** Slot `head` ∈ `{multiclass, ovr, dual_binary}`, default from
   `labels.encoding` (`dual_binary` on lg-st, else `multiclass`). Merged into
   candidate `params.head`. Dual-binary is **not** `xgboost_ovr` (up-vs-rest).
   Trees: two binary fits on `y_long` / `y_short`. LSTM: shared encoder, two
   sigmoid heads, BCE. OOF adds `p_long` / `p_short`; `direction` /
   `confidence` are derived so existing KPI still scores
   (`up` if `p_long > p_short` and `p_long ≥ 0.1`, short symmetric, else flat;
   `confidence = max(p_long, p_short)`). **Amended [ADR-089](ADR-089-dual-binary-argmax-fire.md):**
   event cut is 0.1 on the winning head, not 0.5. `direction_true` from the two y
   columns (never both 1).

3. **Labels.** Persist 5s 3-class TBM as
   `parquet_labels_5s/labels_YYYY-MM-DD.parquet` (`ts` index, `label`,
   `close`) with `_contract.json`. Fold cache **joins** when the contract
   matches; otherwise falls back to `generate_labels`. 15s dual-binary stays
   `parquet_labels_15s/` (projection of 5s TBM). Join is inner on `ts`:
   15s `bar_end_ts` equals the 15s-grid 5s timestamp (closed bar ↔ decision
   time). Overnight 15s feature bars without labels drop. Sequence lookback
   into unlabeled Globex is out of scope for v1.

4. **Fail-closed.** Automated search on `input: native_tf` still raises.
   Custom Test is allowed when topology `mode` is `none`, the selected store
   is available, and meta is off. Joint `tf_oof_meta` search is still
   unwired. OOF-only compose from existing Custom Test runs:
   [ADR-092](ADR-092-tf-oof-meta-composer.md).

## Consequences

- Live pin unchanged. 5s Custom Test / 3-class search path unchanged when
  `feature_store` is omitted (defaults to `5s`).
- GPU sync adds `parquet_labels_5s/`. Native-TF GPU hydrate: [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md).
- 1m Custom Test store is available after ADR-085 hive files exist.
  5m/1h/1d Custom Test waits on those hive files.

## Related

- [ADR-018](ADR-018-objective-doc-single-source-labels.md) — encoding SSOT
- [ADR-035](ADR-035-webui-custom-test.md) — Custom Test compiler
- [ADR-045](ADR-045-xgb-ovr-uniqueness-objective.md) — ovr ≠ dual_binary
- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — lg-st
- [ADR-075](ADR-075-native-tf-15s-bar-features.md) — 15s hive
- [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md) — GPU hive + labels push
- [ADR-082](ADR-082-aux-labels-and-head-target-geometry.md) — aux labels + per-head target geometry
- Catalog: [FEATURE_CATALOG_15s.md](../reference/FEATURE_CATALOG_15s.md)
