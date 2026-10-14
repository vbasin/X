# ADR-148: Drop the Live Chart MBP-10 ladder

## Status
Accepted — implemented 2026-09-29. Supersedes ADR-087 decision §4. Live pin unchanged.

## Date
2026-09-29

## Context

ADR-087 kept an optional MBP-10 ladder and depth-imbalance panel on Live
Chart, fed by `GET /api/book/snapshot`, and left the heatmap routes mounted
for a later panel. The ladder sat at the bottom of the overlay menu. It is
not part of the candle chart operators use, and chart meta called
`book.meta()` on every load just to enable that checkbox.

## Decision

1. **Remove the panel.** Live Chart no longer has the MBP-10 ladder checkbox,
   the ladder, or the depth-imbalance block under it.
2. **Unwire the console routes.** `GET /api/book/meta`, `/api/book/heatmap`,
   and `/api/book/snapshot` are not registered. Chart meta no longer reports
   a `book` availability field.
3. **Leave the library and the table.** `src/webui/book.py` and
   `databento.mbp_10` stay on disk. The 5s mean depth feature (`mbp_`,
   ADR-057) is unchanged. Nothing in the console calls the book module.

## Consequences

- `#/chart` and the `#/border` alias no longer draw an L2 ladder.
- A process that started before this change still serves the old routes
  until the web UI is restarted. Static files pick up the panel removal
  without a restart.
- `tests/test_book_viz.py` still covers `book.py`. The HTTP tests expect 404.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart; §4 superseded here
- [ADR-057](ADR-057-mbp10-5s-mean-depth-imbalance.md) — 5s `mbp_` feature, kept
- Module left in place, not served: `src/webui/book.py`
