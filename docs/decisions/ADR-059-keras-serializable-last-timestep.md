# ADR-059: Serializable last-timestep layer + first-party Keras load

## Status
Accepted — implemented 2026-08-26.
**Amended 2026-08-27** by [ADR-077](ADR-077-keras-15-initializer-forward-compat.md):
`load_keras_model` also drops Keras 3.15 initializer keys (`input_axes` /
`output_axes`) so GPU LSTM artifacts load on the CPU 3.13 venv.

## Date
2026-08-26

## Context

WebUI Backtest of Custom Test TCN `tcn_so_mbp_vix`
(`research/custom_test/nq-50pt-econ/20260822T050238Z/models/tcn_so_mbp_vix`,
`pooling: last`) failed at bundle load with Keras 3:

```
Requested the deserialization of a `Lambda` layer whose `function` is a
Python lambda. This carries a potential risk of arbitrary code execution
and thus it is disallowed by default.
```

`tcn_embedding` (and `path_tcn_embedding`) selected the last causal timestep
with `layers.Lambda(lambda t: t[:, -1, :], name="last_step")`. Keras 3
serializes that as Python bytecode and, by default (`safe_mode=True`),
refuses to load it. `src/serving/bundle.py` called
`keras.models.load_model` with no override, so Live/Backtest could not
score the last-fold `.keras` file even though the artifact was first-party.

`patch_hazard` used the same pattern for feature-index gather / empty-stream
zeros.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Global `keras.config.enable_unsafe_deserialization()` | One-line | Process-wide; not scoped to our artifacts |
| B. `safe_mode=False` only, keep Python lambdas | Unblocks existing files | New saves keep repeating the same trap |
| C. Serializable layers + scoped `safe_mode=False` on first-party load (chosen) | New graphs load under default safe_mode; old Custom Test files still load | Serving explicitly trusts in-house `.keras` files |

## Decision

1. **No Python-lambda `Lambda` layers in plugin graphs.** Last causal
   timestep is `LastTimestep` (`src/models/keras_layers.py`). Patch
   streams use `GatherFeatures` / `ZerosTokens`. All three are
   `@keras.saving.register_keras_serializable(package="vtech")`.

2. **First-party load helper** `load_keras_model` (same module) is the
   SSOT for `.keras` files in serving, LSTM trainer reload, stack trainer,
   and IG/permutation scripts. It passes registered custom objects
   (including `AttentionLayer`) and **`safe_mode=False`**. These paths
   only load models trained by this repo under
   `model_staging/`, `model_registry/`, `research/search/`, or
   `research/custom_test/`.

3. **Do not rewrite existing lab artifacts.** The
   `20260822T050238Z/models/tcn_so_mbp_vix/model.keras` file stays as saved
   (Lambda `last_step`). New TCN `pooling=last` fits emit `LastTimestep`.

## Consequences

- WebUI Backtest / Live can load TCN `pooling=last` Custom Test bundles
  without retraining.
- New TCN / `path_tcn` / `patch_hazard` checkpoints deserialize under
  Keras 3 `safe_mode=True`.
- Unsafe deserialization is **per load**, not a global Keras flag.
- GPU Keras 3.15 LSTM files: extra initializer kwargs are dropped at load
  ([ADR-077](ADR-077-keras-15-initializer-forward-compat.md)).

## Related

- [ADR-036](ADR-036-custom-test-lab-artifacts.md) — last-fold `.keras` under
  `research/custom_test/` listed on Backtest
- `src/models/tcn_plugin.py`, `src/models/path_tcn_plugin.py`,
  `src/models/patch_hazard_plugin.py`, `src/serving/bundle.py`
- Tests: `tests/test_keras_layers.py`
