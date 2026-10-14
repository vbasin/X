# ADR-102: GPU jobs hydrate 1h/1d HTF context hives

## Status
Accepted — implemented 2026-08-29. Live pin unchanged.

## Date
2026-08-29

Amends [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md) (job hydrate was the
**predictive** store only). Related:
[ADR-095](ADR-095-regime-context-fire-aux.md),
[ADR-100](ADR-100-backtest-native-tf-bundle-store.md),
[ADR-101](ADR-101-lstm-globex-tape-session-loss.md).

## Context

`nq-50pt-econ-lg-st` YAML joins a slim 1h/1d regime pack onto every
predictive base (`wants_htf_on_store`). Fold cache calls
`attach_htf_to_frame`, which loads `timeframe=1h/` and `timeframe=1d/`
from the local hive. GPU dispatch pushed only `timeframe=15s/` (the
LSTM store). On an already-RUNNING A10, `job_exec` never pulled 1h/1d.
`store_available("1h")` was false, attach skipped, last-fold schema
was **23** hive cols instead of 23+13. CPU 1m/5m trees on this host
did see HTF (84 / 77 cols).

WebUI Backtest lists Custom Test dirs that pass `is_serving_bundle`
(`feature_schema.json` + `model.keras|pkl|ubj`). Joint-search meta
writes `oof.parquet` + `meta_estimator.pkl` without that pair, so
`#/backtest` could not pick the composed stack. LSTM/1m/5m last-fold
folders already had serving files; historical replay uses nested
`oof.parquet` (ADR-100).

`--families` on `tf_oof_meta` was ignored: topology.bases always ran
all three stores. Retraining LSTM after ADR-101 required a family
whitelist.

## Decision

1. **Context sidecar.** When `wants_htf_on_store`, CPU
   `push_job_training_data` also uploads `timeframe=1h/` and
   `timeframe=1d/`. The job payload carries `context_s3_prefixes`.
   GPU `job_exec` pulls those prefixes into `native_tf_features_root()`
   **before** fold cache (same dest the boot `DATASETS` list uses).
2. **WebUI lab dirs.** `#/backtest` lists a Custom Test `models/<cand>/`
   folder if it is a serving bundle **or** it has `oof.parquet`.
   `classify_backtest_source` accepts that dir (OOF replay) or nested
   `oof.parquet`. Browse marks OOF-only folders selectable. Root
   `oof_*.parquet` tapes stay unlistable. Labels include `native_tf`.
3. **`--families` on joint search** matches candidate id (`lstm_15s`),
   family key (`attention_lstm`), or store (`15s`). Meta still skips
   when a YAML base OOF is missing. Custom Test
   `--experiment-type custom_test` still writes last-fold serving
   files so the new LSTM shows in the Backtest dropdown.
4. **τ-off OOF replay.** ADR-101 fell back to fire 0.1 only when book
   fire admitted **0** rows. Globex+HTF LSTM `20260829T204718Z` has
   `p_max` 0.401 and **2** fires @ 0.40 — `#/backtest` with τ off
   would still show two trades. Fallback now triggers when directional
   book rows **< 10**. 1m/5m n@0.40 stays on 0.40 (tens–hundreds).

Live pin unchanged. Serving asof of live 15s/1m/5m bases is still not
wired.

## Consequences

- Next GPU `lstm_15s` rebuilds the Globex 15s cache (ADR-101) **and**
  asofs HTF. Schema should be 23+13 when 1h/1d parts exist in
  `vtech_hot`. Do not reuse `20260829T182514Z` as the Globex+HTF
  LSTM.
- Cold boot already listed 1h/1d in `sync_parquet.py` DATASETS; this
  covers the RUNNING-VM path that skipped ExecStartPre.
- Meta `tf_oof_meta__*` folders without keras become Backtest sources
  via nested OOF (report-only, not Live).
- `--families lstm_15s` is the LSTM-only Custom Test path.

## Related

- [ADR-078](ADR-078-gpu-native-tf-hive-mirror.md), [ADR-093](ADR-093-tf-oof-meta-joint-search.md)
- `src/gpu_runner/parquet_sync.py`, `src/gpu_runner/job_exec.py`,
  `src/webui/datasources.py`, `src/search/tf_oof_meta.py`
