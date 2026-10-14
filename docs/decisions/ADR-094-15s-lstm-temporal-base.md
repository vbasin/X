# ADR-094: 15s predictive base is Attention-LSTM

## Status
Accepted — implemented 2026-08-28. Live pin unchanged.

## Date
2026-08-28

## Context

`sa.md` and every lg-st Custom Test on 2026-08-28 used a **15s LSTM** as
the micro-structure encoder: a shared sequence trunk with independent
`p_long` / `p_short` heads, so temporal vision over the 15s tape is not
flattened into a single bar. ADR-066 / the first joint wave
(`20260828T212130Z`) declared `xgb_15s` instead. That is the wrong
family for the 15s layer.

1m and 5m stay trees (bar-level interactions, no sequence encoder).

## Decision

1. `nq-50pt-econ-lg-st` topology.bases 15s is
   `{id: lstm_15s, family: attention_lstm, native_tf: 15s, head: dual_binary}`.
2. 1m / 5m remain `xgboost_ovr` dual-binary.
3. Joint search (ADR-093) dispatches the LSTM to the GPU VM; 1m/5m XGB
   stay local. If `GPU_INSTANCE_OCID` is unset the LSTM runs in-process
   (slow) rather than being skipped.
4. Meta input is still `P_long/P_short` from all three bases + compact
   1h/1d + clock. Fire rule unchanged (ADR-089).

## Consequences

- `20260828T212130Z` is a three-XGB stack; do not treat it as the sa.md
  architecture. Re-run joint search after this ADR for LSTM-15s OOF.
- Live pin stays `nq-50pt-econ`.

## Related

- [ADR-066](ADR-066-native-tf-dual-head-meta.md) — 15s family amended here
- [ADR-080](ADR-080-dual-binary-sequence-heads.md)
- [ADR-093](ADR-093-tf-oof-meta-joint-search.md)
- Objective: `config/objectives/nq-50pt-econ-lg-st.yaml`
