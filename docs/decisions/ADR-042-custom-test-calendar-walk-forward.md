# ADR-042: Custom Test expanding calendar walk-forward designer

## Status
Accepted — implemented 2026-08-21.
Amends [ADR-035](ADR-035-webui-custom-test.md): Custom Test may overlay
`eval.date_range` + calendar folds without mutating the Trade Objective.

## Date
2026-08-21

---

## Context

Search `_build_folds` used equal bar-count expanding windows
(`fold_size = n // (n_folds+1)`). That splitter mid-cuts sessions, can drop
the newest remainder, has no TBM purge, and does not match a month-aligned
WFO. Operators needed a Custom Test control to set start/stop, train period,
and test period (days / weeks / months) so fold count is implied
(train1+test1 = train2 until stop), with an optional val window for DL
early stopping.

## Decision

1. **Designer on the Custom Test board** (below meta). Fields: enable,
   start, stop, train `n`+unit, test `n`+unit, optional val `n`+unit cut from
   `train` or `test`. Default: enabled, **2025-05-01 → 2026-06-30**, 2 months
   train / 2 months test (six expanding folds); val off. Dates after stop
   stay out of search for backtest.
2. **Do not mutate the objective YAML.** The compiler writes
   `*.walk_forward.yaml`; the controller applies it via
   `--walk-forward-overlay` (GPU payload `eval_overlay`). Disable the
   designer to keep YAML equal-count folds.
3. **Splitter** `src/search/walk_forward_calendar.py`: session-TZ calendar
   masks, expanding train, last **120 bars** of train purged (10m TBM
   horizon). Trees ignore `X_val`; TCN/LSTM use it when val is enabled.
4. Preview shows the fold table. `folds.json` records `expanding_calendar`.

Rejected: putting this in objective YAML as the global default (ADR-018);
equal-count with a date clip only; a separate val month in the expanding
identity.

## Consequences

- Custom Test default is no longer the objective's 3 equal-count folds.
- Sequence models without val enabled do not early-stop on OOF.
- Automated search / finalize without the overlay are unchanged.

## Related

- [ADR-018](ADR-018-objective-doc-single-source-labels.md)
- [ADR-035](ADR-035-webui-custom-test.md)
- Compiler: `src/search/custom_test.py`, `src/search/walk_forward_calendar.py`
- Schema: `config/schemas/custom_test.schema.json`
