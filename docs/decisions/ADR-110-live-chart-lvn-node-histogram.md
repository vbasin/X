# ADR-110: Live Chart volume-profile LVN node histogram

## Status
Superseded — folded into [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
(2026-09-06). `hist_style: lvn_nodes` was already replaced (tick vs
binned histogram; LVN as independent borders). Do not cite for present
Volume Profile behavior.

Amends [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md)
(histogram was tick-row only) and [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md).

## Date
2026-09-03

## Context

Live Chart Volume Profile draws one pixel row per NQ tick (0.25). That
is accurate but too fine to read **Low Volume Nodes** (LVNs) as zones.
Operators wanted:

1. A choice between the existing volume histogram and a histogram
   merged into visible rectangles.
2. LVN rectangles highlighted and extended forward to the next VP.

CME / Steidlmayer Market Profile specifies **POC** and the **70% value
area**. There is **no official LVN formula**. VP platforms (TradingView,
Sierra, ATAS, Bookmap) treat an LVN as a **relative valley**: thin
volume between (or beside) high-volume shelves, often as a fraction of
POC volume.

## Decision

1. **Histogram style** (per VP spec `hist_style`):
   - `volume` — existing buy/sell tick histogram (default).
   - `lvn_nodes` — merge contiguous ticks into rectangles.
2. **LVN rule** (`volume_nodes` in `src/webui/live_chart.py`):
   - POC = max-volume tick (unchanged).
   - A tick is an LVN candidate if `vol ≤ lvn_pct × POC volume`.
   - Default `lvn_pct` = **0.30** (30% of POC; common VP-software
     default). UI is 10–50%. Values `> 1` are treated as percent.
   - The POC tick is never LVN. Adjacent candidates merge into one
     node; adjacent above-threshold ticks merge into a body / HVN
     shelf. If every tick would be LVN, classify as body (no contrast).
3. **Draw.** LVN rectangles use a distinct amber fill (not buy/sell).
   Body nodes keep the buy/sell split. Height is the node’s price span
   (tick ± `tick_size/2`), so valleys are visible blocks rather than
   1 px rows.
4. **Extend LVN** (`extend_lvn`, default on for `lvn_nodes`): each LVN
   node’s price band is painted from this histogram through the next
   VP to its right (same x-span as extend-lines / extend-VA). Off =
   highlight only inside the histogram.
5. **Wire.** `hist_style`, `lvn_pct`, `extend_lvn` travel on
   `profiles=` JSON. `profile_payload` always includes `nodes` so the
   style switch does not change the tick bins.

Not hive `vp_*` / `svp_*` columns. Live pin unchanged.

## Consequences

- Hard-refresh the console (cache-bust on `index.html`).
- LVN is POC-relative, not a 70% value-area tail. Tails often *are*
  LVNs because they sit under 30% of POC.
- Changing LVN % reloads the series (nodes are computed server-side).

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — SVP / VP
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  present Volume Profile SSOT (this ADR is folded)
- Module: `src/webui/live_chart.py`, `src/webui/static/app.js`
