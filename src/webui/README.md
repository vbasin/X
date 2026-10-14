# VTech Web Console (`src/webui`)

A single-page console over the existing workflow surfaces — no build step (vanilla-JS SPA +
FastAPI). It wraps, rather than reimplements, the modules it drives:

### Live Chart Large Prints

Large Prints parameters include fill opacity, a `Display value` toggle, and `Extend level (bars)`.
Markers are centered on their owning candle at the print price with a diameter of 90% of the
displayed bar width. Extended levels use the matching aggressor-side color and selected opacity.

| Capability | UI route | Backing surface |
|---|---|---|
| 1. Create a new trade objective | `#/objectives` / `#/objectives/new` | `config/objectives/*.yaml` (ADR-018). List uses `yaml.safe_load` (not the loader). Unreadable YAML: row still listed with a YAML pill; detail returns 200 + `parse_error` + raw text (not 500). |
| 2. Run the scoping orchestrator | `#/scoping` | `python -m src.scoping.orchestrator` (ADR-025) |
| 2b. Custom Test (models + params/HPO + features + ablation + meta) | `#/objectives/<id>/test` | Compiles to `--manual-spec` + `--topology-overlay` (ADR-035/040/074); click a model for catalog defaults, per-run overrides, and opt-in HPO; views/groups expand to columns; **save as view** reuses a subset from the feature bank; default board from the objective (`nq-50pt-xgb-tcn` = XGB+TCN + `meta_stack`) |
| 3. Enable/schedule "model optimization" | `#/optimize` | `vtech-research-director.timer` + one-off `research_director.run` |
| 4. Live progress of running processes | `#/jobs`, dashboard | `JobManager` (detached subprocesses, log tail, reattach) |
| 5. View research results | `#/scoping/…`, `#/optimize/search/…` | scoping artifacts + Phase-23 `search_result.json` |
| 6. Backtest staged model weights | `#/backtest` | Initial dropdown **ML model** \| **EXT Model**. ML: staged/registry **and Custom Test** last-fold bundles (`research/custom_test/<obj>/<run>/models/<cand>/`). Browse also walks search runs. Serving bundles use `src.serving`. Bundles with `inference_policy.json` preselect τ and apply open-window / side / uniqueness. Confidence % (τ) dropdown is **5%–95%** in 5% steps (default 80%); **enable** checkbox (default on) applies `conf ≥ τ` — uncheck to admit all directional predictions. **TBBO DATA** / OCO needs a serving bundle. Summary tiles include **Avg PnL**. Summary **P/L over time** chart: Day (EOD) or Trade granularity; cumulative line and/or gain/loss bars (overlay when both on; bar style **Net G/L** or **Net G / Net L**); weekends/CME holidays omitted (`GET /api/calendar/cme`); y-axis nice ticks aligned to bars; **Points** or **Dollars**. EXT: dropdown of engines + per-engine candle timeframe + DEFAULT/hardened snapshot, ＋ to stack (same model allowed at another TF), then **Backtest Parameters** (session vs trade-entry window, ATR, PT/SL points or × ATR or PT × stop, sides, costs); matching signals enter, opposing/flat exit. No τ/OCO. **Chart** tab: book-TF candlesticks from `tbbo_1min` (CME holiday dates omitted; overlays still computed on the full series, HTF engines aligned without look-ahead), SAR / nextBarSAR / EMA / ZigZag arrows+line, trade markers; hover readout includes Wilder ATR vs min-ATR gate and ZZ state (`POST /api/backtest/chart`). |
| 6b. External Models (Pine replicas) | `#/external` | Engines ([ADR-062](../../docs/decisions/ADR-062-ext-model-split-and-stack.md), [ADR-063](../../docs/decisions/ADR-063-zigzag-high-low.md), [ADR-064](../../docs/decisions/ADR-064-ext-trade-window-tf-units.md), [ADR-065](../../docs/decisions/ADR-065-ext-hour-day-timeframes.md), [ADR-096](../../docs/decisions/ADR-096-ext-timed-psar.md)): **Parabolic SAR**, **20 EMA**, **ZigZag High/Low**, **Timed PSAR**, each with own long/short signals. Shared session/trade-window/ATR/PT-SL/cost knobs are **Backtest Parameters** on `#/backtest` (EXT Model). Candle TFs: `1min` / `5min` / `15min` / `1hour` / `1day`. PT units: points, × ATR, or × stop. Optional **Exit at SAR on intersection** on the SAR card ([ADR-061](../../docs/decisions/ADR-061-psar-ema-exit-at-sar.md)). Tab load is factory YAML + saved overlays only — it does **not** query `tbbo_1min`. Save defaults / hardened snapshots under `ops/webui/external_models/`, send to Backtest (candles load on run). |
| 7. Live streamer + model score | `#/live` | Compact stream status (overall health + streamer/tbbo/disk tiles + ingest table; connections/raw archive under details) + model score via `src.serving`. **TBBO DATA** toggle on the day blotter. Overall health tracks process/sockets/tbbo/bbo; sparse `statistics` lag is table-level only unless stale |
| 8. Live Chart (/NQ candlesticks) | `#/chart` (`#/border` alias) | Timeframe dropdown (5s / 15s / 1m / 5m / 15m / 1h / 1d) over Databento TBBO caggs. The page **fills the browser viewport** (no WebUI scroll to reach the chart; overlay sidebar scrolls on its own). Lookback presets through 1 year plus **Custom…** (duration `6h` / `3d` / `2w` / `90d` / … through now, or From/To Chicago calendar dates; no per-TF lookback cap — responses keep at most 50,000 newest bars, ADR-105 / ADR-124 / ADR-128). X-axis ticks are compact CT labels (time-of-day without repeating the date; crosshair keeps the full timestamp). Optional TBBO overlays (CVD, OFI, OFI_CUM, VWAP, CLV, MFI 14, A/D, stochastic, SAR as stars, **Large prints**, multi-EMA). Overlay **▸** opens that series’ parameters: CVD has **CVD_CUM origin** + plot style; OFI_CUM has its own origin (ADR-108); **Large prints** has min size (default 10; TBBO fills ≥ that size in the chart lookback as price bubbles locked to the candle via `timeToCoordinate(bar) + (frac−0.5)×barSpacing`, black outline + price rails, hover price / qty / `ts_event`). **Parabolic SAR** uses the same AF card pattern as EMA/VP (refresh + `start` / `increment` / `maximum`, ADR-141). **VWAP** ▸ params include **Bollinger bands (±2σ)** on the price pane (same origin as VWAP; ADR-142). Volume is a dedicated pane **directly under price** with its own y-axis labels (not mixed with CVD). CVD and every other non-price overlay (`pane: osc` in the catalog) each get their **own pane** below volume, separated by a divider, with an independent y-axis (drag the scale to zoom that pane only; drag the divider to resize). Signed panes (CVD, delta, OFI, A/D, CLV, VWAP distance) draw a **zero floor line**. CVD plot style is line, candles, or both. **Display**: Night session **Display** / **Shade** (Globex 5:00pm–8:30am CT, locked to the time axis) / **Aggregate** (include overnight in overlays and VP); **Text size** 10–18px. **Volume Profile** (master checkbox) adds **VP1**, **VP2**, … (**Single** = calendar date, blank = last candle’s session; **Recurring** daily/weekly). Markers are `VP1 POC`, `VP1 VAH`, … **VP axis** (default on) puts those on a second right-hand scale so the OHLC axis stays readable. Session: full / cash / premarket / BOD / **Custom** From/To CT. Place: Overlay / Left / Right. **Extend** (forward only): Off / 1 sess (last bar of this session day) / 2 sess (last bar of the next session day) / All sess (visible right edge). **Hist**: Tick vs Binned (8–20, default 12). **LVN** independent (≤ 30% of POC), outline only, optional Extend LVN. Crosshair: color-coded OHLC + overlay values in a strip above the chart and a hover box at the cursor (drag the handle to snap the box to a chart corner; double-click to follow again). Volume Profile rows in that popup (VHi, VLo, VAH, VAL, POC, plus volume-at-price / LVN) are only the profile whose session window contains the hovered bar, not earlier recurring sessions ([ADR-125](../../docs/decisions/ADR-125-live-chart-hover-session-vp.md)). Optional **MBP-10 ladder** at the crosshair. Toolbar **Select** brushes a time range; **Create Note** saves symbol+range name, text, and every visible overlay/parameter to `ops/webui/chart_notes/` (browse at `#/notes`, ADR-141). Instrument is always /NQ (lead-month). [ADR-087](../../docs/decisions/ADR-087-live-chart.md), [ADR-097](../../docs/decisions/ADR-097-live-chart-cvd-origin-volume-profile.md), [ADR-105](../../docs/decisions/ADR-105-live-chart-custom-lookback.md), [ADR-124](../../docs/decisions/ADR-124-live-chart-lookback-date-range.md), [ADR-108](../../docs/decisions/ADR-108-live-chart-nested-overlay-params-multi-vp.md), [ADR-109](../../docs/decisions/ADR-109-live-chart-display-fit-and-eth.md), [ADR-111](../../docs/decisions/ADR-111-live-chart-large-prints.md), [ADR-112](../../docs/decisions/ADR-112-live-chart-stacked-panes.md), [ADR-114](../../docs/decisions/ADR-114-live-chart-replay.md), [ADR-125](../../docs/decisions/ADR-125-live-chart-hover-session-vp.md), [ADR-141](../../docs/decisions/ADR-141-live-chart-sar-params-and-notes.md). **Replay** (Display, collapsible): scrubber + Play/Pause (one bar/tick) + Step + speeds ¼–4×; candles/overlays/VP are as-of the playhead. Level Blend cards: **Member lines** toggle (default bounds-only), per-card collapse, cluster hover (Upper/Lower/strength/range). Display help copy lives behind right-aligned `i` icons. Optional **Order map (MBO)** overlay (`lt_mbo_orders`, [ADR-145](../../docs/decisions/ADR-145-live-chart-mbo-order-map.md)): large resting orders from `databento.mbo` as size-scaled lines with end markers, dotted extension for resting orders, reload-chain bands, key bar, hover details, 10/50 grid; wheel scrolls the price axis (Ctrl+wheel zooms), *Fit candles* / *Fit all orders* / `▲ n` `▼ n` jump buttons; MBO is ~8 h behind real time, kept current automatically every 30 min by `vtech-mbo-incremental.timer` (ADR-147), which also seeds the view with large orders that were already resting before the visible window opened rather than only ones added inside it. |
| 8b. Notes (chart scenarios) | `#/notes` | Server-side Live Chart snapshots from Select + Create Note (`ops/webui/chart_notes/`). Name = contract + selected CT range; detail shows note text plus retained overlays/params ([ADR-141](../../docs/decisions/ADR-141-live-chart-sar-params-and-notes.md)). |

Model-selection search (`python -m src.search.controller`) is also launchable from `#/optimize`.
Custom Test on an objective is the same controller in user-override mode (`--experiment-type
custom_test`): pick families, click a model card (or **params**) to view catalog defaults,
edit them for this run, tick knobs for HPO, and enable HPO (passes `--tune-trials`; ranges
come from `CatalogEntry.search_space`). Drag catalog views/groups/columns onto each model
(views and groups expand to their columns on the card). **save as view** stores that
subset under `ops/webui/custom_feature_views/` and lists it on the Views tab for reuse
([ADR-074](../../docs/decisions/ADR-074-custom-test-saved-feature-views.md)).
Flag units for ablation (`2^k` candidates per model), and optionally enable a meta gate.
Artifacts land in `research/custom_test/<objective>/<run_ts>/models/<cand>/` (last-fold
weights + OOF + `meta_stack` bundle). **＋ all features** on a selected model card assigns
the full catalog in one click. Feature builder (compose a new column from existing names)
is parked.

**Live model resolution** (`src/webui/live.py`), in order:

1. `VTECH_LIVE_MODEL` env override (ops)
2. **Manual pin** from the Live-tab dropdown (`POST /api/live/model` →
   `data/webui/live_model.json`)
3. **Auto**: newest `model_registry/<active objective>` → newest
   `model_staging/<active objective>` → any-objective fallback (with warning)

The Live UI lists every staged/registry bundle (grouped by objective) with an
**Apply to streamer** control. Dropdown and “Scoring now” names use
``<objective>/<bundle_dir>`` (one folder up from the model files — that dir
carries the search timestamp). `POST /api/live/reload` clears the in-process cache
and rescores immediately.

## Launch

### Dev / interactive (Makefile)

```bash
cd /opt/vtech       # required — do not start the console from a Grok worktree
make webui          # start on http://127.0.0.1:8700 (cwd=/opt/vtech; logs → logs/webui.log)
make webui-status   # is it running?
make webui-stop     # stop it (waits until :8700 is free; SIGKILL leftover python)
make webui-restart  # stop + start; success only after GET / returns 200
make webui-bust     # rewrite app.js/style.css ?v= cache-bust in index.html
make webui-refresh  # bust static assets + restart (after UI or API changes)
```

`make webui-restart` used to wait ≤2s for the port and then print "started"
even if the new uvicorn hit `EADDRINUSE` and exited. The browser tab then
spun forever on the SSH/VS Code keep-alive to the killed process. Stop now
waits until `:8700` is free; start waits for `GET /` 200.

After `make webui-refresh`, reload the browser tab. If the shell still looks
stale, hard-reload: **Ctrl+Shift+R** (Linux/Windows) or **Cmd+Shift+R** (macOS).

If the tab itself never finishes loading, closing the tab is **not enough**
when VS Code has reconnected the SSH session (remoteagent logs
`The client has reconnected`). The Ports forward is then a dead multiplex:
uvicorn never sees `GET /`, so a new browser tab still spins. On the VM,
`curl -sf -D - http://127.0.0.1:8700/` returns 200 with `X-VTech-Rev` in
milliseconds.

Operator runbook (diagnose on the VM, then the four Ports steps):
[docs/operations/TROUBLESHOOTING.md](../../docs/operations/TROUBLESHOOTING.md) §14.

Equivalent by hand:

```bash
cd /opt/vtech && source venv/bin/activate && set -a && source .env && set +a
python -m src.webui.app --host 127.0.0.1 --port 8700
```

### Production (systemd)

```bash
sudo cp systemd/vtech-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vtech-webui.service
journalctl -u vtech-webui -f          # or tail logs/webui.log
```

Bind is `127.0.0.1` (not a Host allow-list). VS Code Ports / tunnels send
non-localhost `Host` values; a 400 `Invalid host header` makes the forwarded
tab spin forever. Responses use `Connection: close` so a restart cannot leave
the Ports-tab browser on a dead keep-alive. Mutations still require
`X-VTech-Console`.

`KillMode=process` is deliberate: restarting the console kills only uvicorn — long-running
workflow subprocesses keep running (they own their log fds) and the `JobManager` reattaches to
them by pid + `/proc` starttime on startup.

## Access — it binds 127.0.0.1 and has NO authentication

Same trust model as the MLflow UI: reach it over an SSH tunnel, never bind it to a public
interface.

```bash
ssh -N -L 8700:127.0.0.1:8700 <user>@<vm>     # then open http://localhost:8700
```

Defense-in-depth on top of the loopback bind: a Host-header allow-list (blocks DNS-rebinding)
and a custom `X-VTech-Console` header required on every mutating request (blocks cross-site
POSTs). These are *not* a substitute for the tunnel — anyone who can reach the port has full
control (create objectives, launch LLM/GPU-spending jobs, toggle the schedule).

## Prerequisites for full functionality

The console degrades gracefully when a dependency is missing (panels show "unavailable" rather
than erroring), but for every feature to work:

- **MLflow links** — `make mlflow` (the console links out to `VTECH_MLFLOW_URL`, default
  `http://127.0.0.1:5000`).
- **Enable/disable the Research Director schedule** (the `#/optimize` toggle) — requires the
  timer unit to be installed, plus passwordless `systemctl` for the console user:
  ```bash
  sudo cp systemd/vtech-research-director.{timer,service} /etc/systemd/system/
  sudo install -m 0440 deploy/webui/vtech-webui-sudoers /etc/sudoers.d/vtech-webui
  sudo visudo -c                         # verify the drop-in parses
  ```
  Without the sudoers drop-in the toggle falls back to plain `systemctl` (works only if polkit
  allows it) and otherwise returns a clear hint.
- **Research funnel & spend panel** — the `research_director` Postgres schema must exist
  (same queries as `scripts/rd_dashboard.py`). Absent → the panel shows "DB unavailable".

## Environment

- `VTECH_DATA_DIR` (default `/opt/vtech/data`) — roots for `scoping/`, `search/`, and
  `webui/jobs/` (job metadata + logs).
- `VTECH_MLFLOW_URL` (default `http://127.0.0.1:5000`).
- Secrets (`.env`) are loaded by the Makefile target and the systemd `EnvironmentFile`, so
  console-launched jobs inherit API keys / DB credentials.

## Tests

```bash
python -m pytest tests/test_webui.py tests/test_book_viz.py tests/test_live_chart.py -q
```

Uses FastAPI's `TestClient` with a `JobManager` rooted in `tmp_path`, so no real workflow
subprocesses are launched (jobs run `/bin/echo` etc.).
