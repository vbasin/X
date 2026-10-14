# ADR-108: Live Chart nested overlay params + multi volume profile

## Status
Accepted — implemented 2026-09-03. Present Volume Profile UI folded in
2026-09-06 (histogram / LVN / Single VP date / Custom clocks). VP# names
+ merged extend 2026-09-06. Live pin unchanged.

Amends [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) decisions
2–4 (shared CVD/OFI origin sidebar; single SVP + single range VP).
Lookback is unchanged ([ADR-105](ADR-105-live-chart-custom-lookback.md)).
Display / ETH / VA wash: [ADR-109](ADR-109-live-chart-display-fit-and-eth.md).
Replay as-of: [ADR-114](ADR-114-live-chart-replay.md).
Level Blend (KDE cascade bounds, no histogram):
[ADR-135](ADR-135-live-chart-level-blend-vp.md).

**Folds** [ADR-110](ADR-110-live-chart-lvn-node-histogram.md),
[ADR-113](ADR-113-live-chart-binned-vp-lvn-borders.md), and
[ADR-115](ADR-115-live-chart-single-vp-custom-session.md). Cite this
file for Volume Profile; those numbers are historical stubs.

## Date
2026-09-03 (VP present-state 2026-09-06, names/extend same day)

## Context

Live Chart overlays listed every series as a flat checkbox, then dumped
CVD origin, CVD plot style, SVP origin, and custom-range VP From/To in
two global sidebar blocks. CVD and OFI_CUM shared one origin clock.
Operators could not:

1. Keep parameters next to the overlay they belong to.
2. Give CVD and OFI_CUM different reset clocks.
3. Draw more than one volume profile, or pin a profile to the candles
   it covers instead of the viewport edge.
4. Recur a profile (e.g. one cash-session VP per day in the lookback).

Later the same overlay needed a date-pinned Single VP, Custom From/To
clocks on every kind (instead of a unique Custom range VP), tick vs
equal-width binned histograms, and LVN as independent borders.

These remain **chart overlays**, not hive catalog columns.

## Decision

1. **Nested params.** Each overlay row is `checkbox + name + ▸`. The
   button opens that overlay’s parameters under the row (hidden until
   clicked). CVD owns **CVD_CUM origin** (`session` / `BOD` / `MKT`) and
   plot style (line / candles / both). OFI_CUM owns a separate **OFI
   origin**. A/D owns its own origin. Overlays with no knobs have no
   button.
2. **API origins.** `GET /api/chart/series` takes `cvd_origin`,
   `ofi_origin`, and `ad_origin`. Omitted OFI/A/D origins fall back to
   `cvd_origin` so older callers stay valid.
3. **Volume Profile kinds.** The sidebar label is **Volume Profile**
   (master checkbox). Operators add as many profiles as needed (cap 8
  specs, 32 expanded **per recurring spec**). Add-order names are **VP1**, **VP2**, … (stable
   on delete; next add is max+1). Card title, canvas markers, and
   readout use that name (`VP1 Hi`, `VP1 Lo`, `VP1 VAH`, `VP1 VAL`,
   `VP1 POC`). Recurring expansions keep the spec’s number. Hover and
   the top readout list only the expansion whose window contains the
   crosshair bar ([ADR-125](ADR-125-live-chart-hover-session-vp.md)). **VP axis**
   (default on) puts those labels on a second right-hand scale so the
   OHLC price scale stays readable.
   - **Single** (`kind=current`) — one window. Optional `session_date`
     (`YYYY-MM-DD`, CT session-end date). Blank = last candle’s session
     (Replay: playhead bar).
   - **Recurring** — `daily` (one VP per day in the lookback) or
     `weekly` (CME week Sun 17:00–Fri 16:00 for full session; Mon–Fri
     hours merged for cash/premarket; calendar week for BOD; Custom
     overnight weekly is Sunday From → Friday To).
   - Kind `custom` is a parse-time alias: Single with `session_tf=custom`
     and From/To clocks taken from the old `range_start` / `range_end`
     (session-end date from `range_end`). **Custom range VP** (one
     histogram over an arbitrary datetime span) is not a UI kind.
4. **Session timeframe** (per VP, America/Chicago):
   - `full_session` — previous 5:00pm to 4:00pm CT
   - `cash` — 8:30am–3:00pm CT
   - `premarket` — 8:00am–8:30am CT (not the ADR-097 5:00pm–8:30am
     overnight window)
   - `bod` — midnight to next midnight CT
   - `custom` — From/To CT clocks (`HH:MM`) on every VP kind. Same-day
     if From < To; overnight (previous calendar day From → this date
     To) if From ≥ To. Defaults 08:30–15:00.
5. **Placement** (per VP):
   - `overlay` — histogram on the candles that window covers; side
     `left` (start of range) or `right` (end of range). Scrolls with
     the tape.
   - `left_frame` / `right_frame` — pinned to the plot edge; stays put
     when the chart is scrolled. Multiple frame VPs stack inward.
6. **Extend.** One control per VP (`extend`: `0` / `1` / `2` / `3`; UI
   Off / **1 sess** / **2 sess** / **All sess**). On means both
   VAH/VAL/POC/Hi/Lo **lines** and the 70% value-area **wash** (the old
   `extend_lines` + `extend_va` flags are aliases). **Forward only**
   (never left of this histogram; LWC full-width price lines are axis
   labels only). Target X is the **right edge of the last visible bar**
   of a Globex session day (17:00 CT cut; Sat/Sun skipped), so it matches
   the day separators on the chart:
   - **Off (0)** — lines/wash stay on this histogram.
   - **1 sess (1)** — forward to the last bar of this VP’s session day.
   - **2 sess (2)** — forward to the last bar of the **next** session day.
   - **All sess (3)** — forward to the right edge of the visible plot.
   Legacy `extend_lines` / `extend_va` true → `1`. LVN **Extend LVN**
   uses the same forward span (at least 1 sess when on).
   [ADR-109](ADR-109-live-chart-display-fit-and-eth.md) `extend_va` as a
   distinct checkbox is superseded here.
7. **Histogram / LVN.** Per VP: `hist_style` `volume` (tick histogram)
   or `binned` (equal-width price bins, default **12**, UI 8–20).
   **Highlight LVN** is independent (≤ 30% of POC). LVN is outline only;
   `extend_lvn` extends the zone’s top and bottom borders to the next
   VP. `lvn_nodes` is a legacy alias for tick histogram + LVN on.
  **2026-09-07 display controls:** each VP independently configures
  `chart_labels` (default off; Hi / VAH / POC / VAL / Lo text on the plot),
  `axis_labels` (default off; matching text in the separate VP axis), and
  `hist_width` (default **0.25** of its session span; any positive multiplier
  typed directly, with +/- buttons that step by 0.05). A multiplier above 1
  extends beyond its session width (`2.0` = two times). With axis labels off,
  no VP labels appear on either the separate VP axis or the regular OHLC axis;
  the histogram and reference lines still render. Overlay width is calculated
  only from the actual mapped profile-session span (no fixed drawing-span
  floor or plot-edge rewrite), so the selected multiplier is not overridden
  for a short custom session in a long lookback. The canvas clips only the
  off-screen portion of an extended histogram. The input accepts any positive
  value (`step="any"`); +/- remain the explicit 0.05 increments. Histogram
  painting has only a 1px floor, so it cannot mask a selected short-session
  width with a fixed 8px width.
8. **Wire.** Specs travel as `profiles=` JSON on `/api/chart/series`.
  One TBBO scan covers all expanded windows (`load_vbp_windows`). The
  20-day tape-span cap applies independently to each recurring spec; there
  is no combined cross-profile range cap. Legacy `svp_origin` / `vp_start` / `vp_end`
   still build a left-frame Single VP and/or a Custom-clock Single VP
   from the bound times.
9. **Session-day separators.** The VP canvas draws a bright vertical
   line at each Globex session-end-date change (17:00 CT) in the loaded
   bars, through the chart height above the time axis.
10. **Borders.** Console `--line` is `#4d647c`. Overlay groups
    (`.lc-og`) get their own border.
11. **VP axis.** Volume Profile params include **VP axis** (default on,
    `lc_vp_axis`). On: LWC price-line axis labels are off; a parallel
    scale to the **right** of the OHLC scale lists `VP1 POC`, `VP1 VAH`,
    … at the matching price Y (stacked if they collide). Off: labels
    sit on the main right price scale (previous behavior). The plot
    and replay layer shrink by the strip width (108px). Not a hive
    column.

Not hive `vp_*` / `svp_*` columns.

## Consequences

- Live pin unchanged.
- **2026-09-07 VP-axis fit update.** When VP axis is on, the candle price
  autoscale range includes every active profile's Hi, Lo, POC, VAH, and VAL,
  plus 4% padding (at least one tick). The parallel VP labels and profile
  canvas therefore remain inside the price pane instead of being clipped at
  its edges.
- **2026-09-07 VP-axis label-width update.** The parallel scale measures its
  longest active label and reserves 108–180px accordingly; the chart, VP
  canvas, and replay layer use that same width. Full `VPn level price` text
  is shown without ellipsis.
- **2026-09-23 VP-axis session date.** Axis labels include the referenced
  profile’s session-end date as `MM/DD` between the level name and price
  (`VP3 POC 08/21 30256.24`). Single/Recurring use the resolved
  `session_date` on the profile payload. Level Blend Hi/Lo use the session
  of the extreme member that defines that edge (`hi_session` / `lo_session`
  on each bound). Strip width fit expands to 108–220px for the longer text.
- **2026-09-07 recurring-window coverage fix.** Windows with no
  already-visible chart candle are discarded. The 32-window and 20-day
  tape-span ceilings apply independently to each recurring VP spec; there is
  no combined budget. Weekend, holiday, and out-of-view windows therefore do
  not consume a profile's budget, and one VP cannot suppress another's dates.
- **2026-09-07 explicit apply control.** The refresh icon beside the main
  Volume Profile label commits all VP-card edits and reloads profiles in one
  request. Card edits, adds, removals, and the master enabled state stay as a
  draft until it is clicked; live polling continues to render the last applied
  configuration.
- **2026-09-07 width-input synchronization fix.** Width values synchronize to
  the VP draft on each input event, and the apply control finalizes a focused
  VP field before snapshotting the draft. A directly typed Width is therefore
  included in the same refresh action without requiring another setting change.
- **2026-09-07 display-width authority fix.** Histogram width is presentation
  state. After profile data returns, each expanded profile receives its applied
  local spec's positive `hist_width`, keyed by `spec_id`, before canvas layout.
  A stale or normalizing response therefore cannot silently reduce a selected
  multiplier such as `2.05` to `1.0`.
- **2026-09-07 session separator alignment.** The session-day line is drawn
  at the left edge of the opening candle, using half of that bar's local plot
  spacing rather than the opening candle's center coordinate.
- **2026-09-07 multi-profile default placement.** A newly added profile
  is numbered against both the applied and draft cards, then alternates its
  overlay side by that number (VP1 left, VP2 right). Previously, before the
  first Apply, VP2 was also numbered VP1 and defaulted to the same session and
  left-side rectangle. Applying also normalizes any pre-existing duplicate
  draft names. The shared VBP query now joins every requested window, instead
  of a first-match `CASE`, so duplicate or overlapping windows each receive
  their own tape frame rather than leaving an earlier profile empty.
- **2026-09-07 candle-body extreme bands.** Each profile carries the highest
  body endpoint of its high-setting candle and the lowest body endpoint of
  its low-setting candle. The chart shades each primary Hi/Lo-to-body range,
  including its forward extension, without drawing or labeling secondary
  lines.
- CVD origin is no longer the OFI_CUM / A/D origin unless those
  dropdowns are left at the same value.
- ADR-097 premarket (17:00–8:30 CT) remains on the legacy `svp_origin`
  path only. New Volume Profile premarket is 8:00–8:30 CT.
- Hard-refresh `#/chart`. Stored `lc_vp_specs` with kind `custom` or
  `hist_style: lvn_nodes` migrate in the client; missing `n` is assigned
  add-order; `extend_lines` / `extend_va` become `extend` 0 or 1.
  **VP axis** defaults on (`lc_vp_axis` ≠ `0`).
- Not hive `vp_*` / `svp_*` features.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — CVD origin
  + original SVP / range VP
- [ADR-105](ADR-105-live-chart-custom-lookback.md) — lookback
- [ADR-125](ADR-125-live-chart-hover-session-vp.md) — hover VP is the
  covering session only
- [ADR-109](ADR-109-live-chart-display-fit-and-eth.md) — viewport fit,
  VA wash extend, night-session display
- [ADR-114](ADR-114-live-chart-replay.md) — playhead as-of
- [ADR-110](ADR-110-live-chart-lvn-node-histogram.md),
  [ADR-113](ADR-113-live-chart-binned-vp-lvn-borders.md),
  [ADR-115](ADR-115-live-chart-single-vp-custom-session.md) — folded
  stubs
- Module: `src/webui/live_chart.py`
