# ADR-109: Live Chart viewport fit, axis labels, VA shade, night-session display

## Status
Accepted — implemented 2026-09-03. Live pin unchanged.

Amends [ADR-087](ADR-087-live-chart.md) (tab layout / readout) and
[ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) (VP
extend-lines only).

## Date
2026-09-03

## Context

Live Chart (`#/chart`) lived in the normal scrolling `#view` (padding
plus a 60px bottom gutter). The overlay sidebar and VP cards stretched
the page, so operators scrolled the **WebUI** to reach the lower pane
or the right-hand controls instead of using the chart’s own pan/zoom.

X-axis labels used one `fmtTick` for both ticks and the crosshair, so
intraday charts repeated `MM-DD HH:MM` on every mark.

Volume Profile could extend VAH/VAL/POC/Hi/Lo **lines** across later
candles, but the 70% value-area **wash** stayed on the histogram box.

Operators also wanted Globex night/ETH as a first-class display knob
(show bars, shade non-cash hours, include overnight tape in overlays
and VP) and a single control for chart text size.

## Decision

1. **Viewport fit.** `#/chart` / `#/border` add `#view.chart-fill` and
   `#shell.chart-mode`. The page is `100vh` with overflow hidden. The
   candle host fills the remaining space under the toolbar; the overlay
   sidebar scrolls independently. Other tabs still scroll as before.
2. **X-axis labels.** `tickMarkFormatter` uses Lightweight Charts
   `tickMarkType`: year / month / `Aug 28` / `HH:MM` / `HH:MM:SS`.
   The crosshair and readout keep a full CT timestamp
   (`YYYY-MM-DD HH:MM`, seconds on 5s/15s). Date is not repeated on
   time-of-day ticks.
3. **Extend value zone.** **Superseded by [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
   decision 6:** `extend` Off / 1 sess / 2 sess / All sess draws lines
   **and** the 70% value-area wash together, **forward only**.
   `extend_va` is a legacy alias.
4. **Display → Night session** (cash = 08:30–15:00 America/Chicago):
   - **Display** (default on) — show overnight/ETH bars. Off hides
     those bars and packs the remaining RTH bars so multi-day charts
     are not 70% empty gap. 1-day TF is unchanged.
   - **Shade** (default off) — Globex overnight band
     **5:00pm–8:30am America/Chicago**, edges on the time-axis
     instants (interpolated when the TF has no bar at 08:30). Not
     bar-center ± `barSpacing` (that drifted under zoom). Requires
     Display on. 1-day TF: no band.
   - **Aggregate** (default on) — overlays and volume profiles use
     the full Globex tape. Off (`eth_aggregate=0` on
     `GET /api/chart/series`) computes CVD/VWAP/OFI/stoch/SAR/MFI/A/D
     and VP bins on cash hours only; OHLC bars are still returned so
     Display can show overnight candles. 1-day TF ignores this flag.
5. **Text size.** Sidebar slider 10–18px (default 12) sets chart
   `layout.fontSize`, readout, hover box, and VP canvas labels.

These remain chart overlays / UI state (localStorage). Not hive
catalog columns. Live pin unchanged.

## Consequences

- Hard-refresh the console (cache-bust on `index.html`) to pick up
  the CSS/JS.
- Unchecking **Aggregate** empties a premarket-only VP (that window
  is outside cash hours).
- Densified RTH-only time is plot-time for Lightweight Charts; VP
  overlay placement maps real unix → plot time.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — SVP / VP
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  nested params + multi VP
- [ADR-111](ADR-111-live-chart-large-prints.md) — large-print bubbles
  use the same half-bar lock (`timeToCoordinate(bar) + (frac−0.5)×barSpacing`)
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
