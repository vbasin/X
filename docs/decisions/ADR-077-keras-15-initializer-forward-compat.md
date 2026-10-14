# ADR-077: Load Keras 3.15 initializer configs on 3.13 serving

## Status
Accepted — implemented 2026-08-27. Live pin unchanged.

## Date
2026-08-27

Amends [ADR-059](ADR-059-keras-serializable-last-timestep.md): first-party
`.keras` load also tolerates initializer keys written by a newer Keras than
the CPU venv.

---

## Context

WebUI Backtest of Custom Test `attention_lstm`
`nq-50pt-tbm-opt/20260827T212219Z` returned HTTP 500. Other families
(XGBoost, Keras 3.13 TCN) scored fine.

The last-fold `model.keras` was saved on the GPU runner at **Keras 3.15.0**.
The CPU control plane venv is **Keras 3.13.2**. 3.15 `GlorotUniform.get_config()`
emits `input_axes` / `output_axes`. 3.13 `Initializer.from_config` does
`cls(**config)` and raises:

```
TypeError: GlorotUniform.__init__() got an unexpected keyword argument 'input_axes'
```

That `TypeError` is not mapped by the WebUI handlers, so FastAPI returns
`500 Internal Server Error`. Load happens before feature scoring — the
failure is immediate. Nine on-disk LSTM Custom Test files are Keras 3.15;
TCN files on this host are still 3.13.2.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. Pin CPU Keras to 3.15 | Exact match today | Next GPU bump repeats the 500; train-time behavior on CPU changes |
| B. Rewrite saved `config.json` on ingest | Artifacts self-contained | Mutates lab files; every new key needs another rewrite |
| C. Drop unknown initializer kwargs at load (chosen) | Serving-only; no pin dance; no artifact rewrite | Ignores axes that 3.13 never used (LSTM GlorotUniform does not) |

## Decision

1. **`load_keras_model` (ADR-059 SSOT)** wraps load in a scoped patch of
   `Initializer.from_config`: try `cls(**config)`; on `TypeError`, retry
   with kwargs the class `__init__` actually accepts. Restored after the
   call. Unknown keys (`input_axes`, `output_axes`, future extras) are
   dropped, not invented.

2. **Do not bump the CPU Keras pin** to chase the GPU image. Training
   stays on whatever the GPU worker installed; serving stays 3.13 plus
   this shim.

3. **Remaining deserialize `TypeError`s** become `ValueError` so WebUI
   maps them to HTTP 400 with the innermost Keras line, not a bare 500.

4. **Do not rewrite** existing 3.15 lab files.

## Consequences

- Backtest / Live / IG can load GPU LSTM Custom Test bundles trained on
  Keras 3.15 without retraining.
- Weights and graph match the 3.13 LSTM implementation (axes keys are
  unused for these initializers).
- A future Keras save format that changes **layer** topology still needs
  a pin bump or a new loader — this only covers extra initializer kwargs.

## Related

- [ADR-059](ADR-059-keras-serializable-last-timestep.md) — first-party
  `.keras` load helper
- [ADR-036](ADR-036-custom-test-lab-artifacts.md) — last-fold bundles on
  Backtest
- `src/models/keras_layers.py`, `src/serving/bundle.py`
- Tests: `tests/test_keras_layers.py`
