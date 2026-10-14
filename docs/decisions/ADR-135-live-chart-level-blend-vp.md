# ADR-135: Live Chart Level Blend Volume Profile

## Status
Accepted — implemented 2026-09-17. Live pin unchanged.

## Date
2026-09-17

## Context

The cascade's structural levels are not a single-session volume profile. They are
the ADR-126 KDE merge of prior closed cash-session levels (`lo/val/poc/vah/hi`)
into buffered bounds, with lookback auto-extension when price leaves the
clustered span ([ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md),
tick-parity SSOT in [ADR-129](ADR-129-tick-level-session-profile-parity.md) /
[ADR-134](ADR-134-cascade-ssot-structural-transitions-policy.md)).

Live Chart already offered Single and Recurring volume profiles (histograms +
POC/VA), but operators could not see the **same bound geometry the model
trades** overlaid on candles — especially pinned to the chart edge so the
bounds stay visible while scrolling.

## Decision

1. **New Volume Profile kind `level_blend` ("Level Blend")** in the same Add
   dropdown / card / Apply UX as Single and Recurring (ADR-108). Cap still 8
   specs. Specs persist in `lc_vp_specs`.
2. **Geometry = cascade LevelConfig defaults.** Prior `lookback_sessions`
   (default **20**, UI-editable 1–120) closed cash sessions before the eval
   Date are loaded from `session_levels.parquet`, KDE-grouped with
   `sigma_pts=5`, bound edges = member span ± `buffer_pts=1`, clipped at gap
   midpoints. When the eval day's high/low sits outside the clustered span,
   lookback grows by `extend_step` until the day is covered or
   `max_lookback_sessions` is hit (`levels_for_range`).
3. **Eval date.** Optional `session_date` (blank = last candle's Globex
   session-end date). For date D the history starts at the prior closed
   session (D−1 or earlier trading day) and walks backward — matching
   "for 5/18, prior 20 sessions starting with 5/17 going backwards."
4. **Display — bounds + optional member levels, no histogram.** Each bound is a
   shaded band with an outline. Member levels (POC solid gold, VAH/VAL dashed
   cyan, Hi/Lo dotted) are gated by the card’s **Member lines** checkbox
   (`show_members`, default **off** / bounds-only). Histogram / LVN / HVN /
   buy-sell displays are forced off for this kind. Hovering the cursor inside a
   drawn cluster rectangle shows Upper/Lower Axis-style labels, KDE
   `strength`, and bound range (`hi − lo`) in the cursor info box.
5. **Placement.** Reuses ADR-108 Place:
   - `overlay` — strip on the eval day's cash window (scrolls with the tape);
   - `left_frame` / `right_frame` — pinned to the plot edge so bounds stay
     visible while scrolling.
6. **Extend.** Same 0 / 1 / 2 / 3 sess control. Default **All sess (3)** so
   bound bands and member lines can span the visible chart from the strip.
   Level Blend extends toward the **opposite** chart edge from the strip:
   Place/Side Left stretches right (forward); Place/Side Right stretches
   left. Single/Recurring remain forward-only.
7. **Wire.** Same `profiles=` JSON on `/api/chart/series`. Level Blend jobs
   skip the TBBO VBP query; `profiles_payloads` returns `bounds[]` with
   `members[{kind,price,session}]`, `hi_session` / `lo_session` (extreme
   member session-end dates for Axis labels), `sessions_used`, and empty
   `bins`.

Not a hive feature column.

## Consequences

- Operators can audit cascade level geometry on Live Chart without rebuilding
  labels.
- Requires `session_levels.parquet` (build via
  `scripts/features/build_session_profiles.py`). Missing SSOT surfaces as
  `error` on the profile payload rather than failing the chart.
- `_limit_recurring_jobs_per_profile` now keeps every non-`recurring` kind
  (Single **and** Level Blend); previously only `kind=="current"` was pinned
  and Level Blend would have been dropped silently.
- Hard-refresh `#/chart`. Stored specs without `lookback_sessions` default to
  20.

## Related

- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — multi VP UX
- [ADR-126](ADR-126-bounded-zone-width-and-resting-entry.md) — KDE bounds
- [ADR-129](ADR-129-tick-level-session-profile-parity.md) — tick levels
- [ADR-134](ADR-134-cascade-ssot-structural-transitions-policy.md) — SSOT
- Module: `src/features/cash_session_profile.py`, `src/webui/live_chart.py`,
  `src/webui/static/app.js`
