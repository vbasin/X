# ADR-120: Live Chart adaptive KDE HVN zones

## Status
Accepted — implemented 2026-09-08. Live pin unchanged.

## Context

The standard 70% Value Area is intentionally one contiguous expansion from the
POC. It cannot describe multiple high-volume distributions separated by
low-volume valleys, especially when a secondary distribution lies outside the
primary Value Area. The existing LVN/body-node logic is useful but does not
measure peak prominence or explicitly return multiple HVN zones. The first
implementation rendered broad KDE shelves but did not return or label valleys,
so adjacent distributions were not visually separated well enough.

## Decision

1. Preserve the existing raw-tick POC, VAH, and VAL calculation as the
   authoritative Value Area model.
2. Add an optional per-profile `adaptive_kde` HVN analysis mode. The selected
   grouped-tick bins are smoothed with a Gaussian kernel, local modes and
   significant valleys are detected, and valley-delimited zones are filtered by
   minimum peak prominence, minimum valley depth, minimum integrated volume,
   minimum peak separation, minimum valley width, and maximum zone count.
3. Use the original histogram to calculate each zone's volume and snap zone
   boundaries to profile prices; KDE identifies structure but does not invent
   authoritative volume.
4. Expose these controls in each Volume Profile card:
   - KDE bandwidth in bins: 1–12, default 2
   - Minimum prominence: 2–50%, default 5%
   - Minimum valley depth: 2–50%, default 5%
   - Minimum zone volume: 1–50%, default 5%
   - Minimum peak separation: 1–20 bins, default 2
   - Minimum valley width: 1–20 grouped bars, default 2
   - Maximum HVN zones: 1–10, default 5
5. Provide a **multi-section preset** (`bandwidth=1`, prominence/depth `2%`,
   separation `2`, valley width `2`, zone volume `1%`, max zones `10`) because
   the prior `separation=10` setting visibly merged nearby distributions.
6. Return explicit `hvn_peaks` and `lvn_valleys` arrays with stable labels
   (`HVN 1`, `LVN 1`, etc.). Render those labels with collision spacing when
   chart labels are enabled. Classify each zone as `inside`, `intersects`, or
   `outside` the primary 70% Value Area. The mode is off by default for
   backward-compatible rendering.
7. Always color the grouped bar nearest each detected HVN peak black and the
   grouped bar nearest each detected LVN valley white, independent of the
   Chart/Axis label toggles. The color markers remain visible when labels are
   disabled.
8. Add per-profile **Plot variable bounds**. When enabled, the backend builds
   a cumulative volume-at-price profile at each chart bar within the designated
   session and returns causal Hi, Lo, VAH, VAL, and POC points. The renderer
   draws each point forward to the next bar/session end; static final-session
   reference lines remain the default when the option is off.

The live profile sweep confirmed the practical behavior: the prior screenshot
settings with `min peak separation=10` suppress nearby structures, while the
dense multi-section preset returns ten HVN peaks and nine intervening LVN
valleys on the same 20-day profile.

## Consequences

- A profile can expose several interpretable HVN shelves and the LVN valleys
   separating them without replacing standard VAH/VAL/POC semantics.
- The selected histogram resolution affects KDE input and therefore remains
  visible and reproducible through the existing `group_ticks` setting.
- Bandwidth and prominence are intentionally bounded to prevent excessive
  one-tick noise or uncontrolled broad smoothing.
- KDE structure is calculated server-side and returned as `hvn_zones`,
   `hvn_peaks`, and `lvn_valleys`; the browser only renders the resulting zones
   and collision-spaced labels.

## Related

- [ADR-097](ADR-097-live-chart-cvd-origin-volume-profile.md) — Volume Profile foundation
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) — nested VP parameters
- [ADR-117](ADR-117-live-chart-grouped-tick-volume-profile.md) — grouped-tick histogram
- Module: `src/webui/live_chart.py`
