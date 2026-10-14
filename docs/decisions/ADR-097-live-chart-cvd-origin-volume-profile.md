# ADR-097: Live Chart CVD origin + session / range volume profile

## Status
Accepted — implemented 2026-08-29. Live pin unchanged.

## Date
2026-08-29

## Context

Live Chart CVD (`of_cvd`) was computed from TBBO prints capped at 36 hours
and reset with `market_session_date` (15:00 CT). On a lookback longer than
the tick cap the line vanished on the left of the chart. Operators want:

1. CVD drawn for **every bar of the selected lookback**.
2. The daily reset clock to be a separate control: Globex session
   **5:00pm CT**, **BOD** (midnight CT), or **MKT open** (8:30am CT).
3. A **session volume profile** (SVP) of the current origin window: POC,
   70% value area (VAH / VAL), histogram at each NQ tick, recalculated as
   each chart bar closes.
4. A separate **range VP** from a user start time (e.g. a week ago) with
   the histogram on the right, bars going left.

These are **chart overlays**, not hive catalog columns
(`FEATURE_CATALOG*.md` is unchanged).

## Decision

1. **CVD / delta from caggs.** `of_cvd` and `of_delta` use
   `buy_volume − sell_volume` on the same bars as the candles
   (`tbbo_15s` / `tbbo_1min` / `tbbo_5m` / `tbbo_1h` / `tbbo_1d`; 5s
   `time_bucket` includes buy/sell). The 36h tick cap no longer applies
   to CVD. Warmup from the origin of the first visible bar uses
   `tbbo_1min` so a mid-session lookback does not start CVD at 0.
2. **Origin clocks** (`cvd_origin` / `svp_origin`): `session` = last
   17:00 America/Chicago; `bod` = last 00:00 CT; `mkt` = last 08:30 CT.
   Chart lookback (`session` / `4h` / `5d` / custom duration, [ADR-105](ADR-105-live-chart-custom-lookback.md)) is independent. CVD and
   **OFI_CUM** (`of_ofi_cum`) share `cvd_origin` and cumulate inside each
   origin period across the whole lookback. **ADR-112:** CVD is its
   own pane below volume (not the left axis of a shared lower band).
   Plot style is still line, candles, or both. Native pane separators
   replace the custom splitter. CVD candles are OHLC of the CVD path
   at the chart timeframe (1-minute path inside coarser bars). Cursor
   shows CVD open / close / Δ (close−open), not high/low. Line+candles
   share one last-value label on the y-axis. On 5s/15s/1m the candle
   body equals that bar’s signed volume except the first print and
   origin resets; on 5m+ the body is the 1-minute path, so it is not
   the full-bar `of_delta` overlay.
   OFI_CUM is Cont-2014 on consecutive **bar-end BBO** (cagg
   `bid_px/ask_px/sz`), not the 36h tick `of_ofi` overlay.
3. **SVP.** `GET /api/chart/series?svp_origin=session|bod|mkt|premarket` aggregates
   `databento.tbbo` size at 0.25-pt ticks from that origin through `end`.
   The origin is the last **candle** on the chart (not wall-clock now),
   so after the Globex close / weekend the last RTH session still
   profiles. If that window has no tape, fall back one origin period.
   Value area is contiguous expansion from POC until 70% of volume (CME
   style), not “highest bins until 70%”. Histogram is pinned to the
   **left** of the visible plot (not at session-open time), boxed at
   profile high/low. Hi / VAH / POC / VAL / Lo extend across the chart
   (canvas + price lines). Recomputed on each series poll. **Premarket**
   is 17:00 CT through ``min(now, next 08:30 CT)``.
4. **Range VP.** `vp_start` / `vp_end` are ISO datetimes (naive = CT).
   Same VBP query, frozen between From and To (To blank = chart end /
   now). Histogram from the right going left. Use this to clasp a prior
   session (e.g. Thu 17:00 → Fri 16:00 CT) and compare POC/VA to later
   bars. Date-only To is exclusive next CT midnight. Cap 20 days.
5. **OFI / VWAP** still load prints (≤ 36h). Options IV is not loaded.

## Consequences

- Live pin unchanged.
- **2026-09-07 origin-opening fix.** CVD accumulation already reset at each
   origin clock, but CVD candle OHLC previously derived its opening value from
   the preceding candle, carrying the prior period's final CVD into the first
   08:30 CT MKT candle. Origin-boundary CVD candles now open at `0`; this also
   applies to the 1-minute CVD path aggregated into coarser chart candles.
- CVD “session” reset is now **17:00 CT**, not `market_session_date`
  15:00 CT. VWAP overlay still uses `market_session_date`.
- SVP / VP are not `vp_*` hive features and are not in the training
  catalog.
- **ADR-108:** the sidebar no longer uses a shared CVD/OFI origin or a
  single SVP + single range VP. Nested per-overlay params, independent
  OFI/A/D origins, and N volume profiles (Single VP / recurring,
  session timeframe including Custom clocks, overlay vs frame
  placement, extend-lines) replace decisions 2–4 for the Live Chart UI.
  Legacy `svp_origin` / `vp_start` / `vp_end` query params still work.

## Related

- [ADR-087](ADR-087-live-chart.md) — Live Chart tab
- [ADR-067](ADR-067-nq-tbbo-15s-1h-1d-caggs.md) — cagg buy/sell volume
- [ADR-108](ADR-108-live-chart-nested-overlay-params-multi-vp.md) —
  nested overlay params + Volume Profile SSOT
- [ADR-112](ADR-112-live-chart-stacked-panes.md) — CVD own pane below
  volume; supersedes left-axis / shared-lower-pane layout in decision 2
- Module: `src/webui/live_chart.py`
