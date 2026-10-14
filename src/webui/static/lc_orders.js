/* Live Chart "Order map (MBO)" overlay — classic script, loaded before app.js.

   Draws every large resting order (from databento.mbo, via /api/chart/mbo-orders) as a line from
   placement to removal at its price, over the candles: bid green / ask red, thickness by size,
   size printed on the big ones, end markers (filled / cancelled / resting), dotted connectors for
   orders that chased price, shaded "reload" bands where a filled order was re-added at the same
   price, a key bar above the chart, hover details, 10/50-point grid, and wheel scrolling of the
   price axis so far-away orders can be reached. Double-clicking a drawn order opens its MBO rows.
   Iceberg (hidden-reserve) limit orders can be shown or hidden without another query.

   All state and drawing live here. app.js only calls paramsHtml / load / draw / clear with a
   `bridge` of its chart helpers, so this file never touches app.js internals. */
(function () {
  "use strict";

  const KEY = "lt_mbo_orders";
  const LS = "lc_mbo_";
  const BID = "#2ecc71", ASK = "#ff4d4d", BID_BAND = "179,136,255", ASK_BAND = "255,159,28";
  const CLR = "#7fd4ff";   // exchange Clear / session purge marker ("R"), ADR-149
  const ICE = "#c792ea";   // native iceberg / reserve order marker, ADR-149
  // server-side parameters (change => refetch); everything else is client-side (change => redraw)
  const SERVER_KEYS = ["min_size", "clip_min", "approach", "chain_n", "chain_dt", "chain_secs", "chain_abs", "break_pts"];
  const DEF = {
    min_size: 20, label_min: 50, label_alive: 1, opacity: 88, thick: 1,
    bids: true, asks: true, filled: true, cancelled: true, resting: true, icebergs: true,
    extend: true, connectors: true, pre: true, edge: true, chains: true, grid: true, wheel: true,
    clip_min: 5, chain_n: 3, chain_dt: 2, chain_secs: 10, chain_abs: 50, approach: 10, break_pts: 1,
  };
  const S = {};
  let data = null;          // last payload
  let dataQ = "";           // query string of `data`
  let loading = "";         // query string in flight
  let err = "";
  let bridge = null;
  let hits = [];            // hover targets from the last draw
  let farUp = null, farDown = null;   // nearest orders outside the visible price range
  let tipEl = null, keyEl = null, hostBound = null, timer = null;
  let detailEl = null, detailSeq = 0;

  function lsGet(k, d) {
    try { const v = localStorage.getItem(LS + k); return v == null ? d : v; } catch { return d; }
  }
  function lsSet(k, v) { try { localStorage.setItem(LS + k, String(v)); } catch {} }
  for (const k of Object.keys(DEF)) {
    const raw = lsGet(k, null);
    if (raw == null) { S[k] = DEF[k]; continue; }
    S[k] = typeof DEF[k] === "boolean" ? raw === "1" : (Number.isFinite(Number(raw)) ? Number(raw) : DEF[k]);
  }

  function css() {
    if (document.getElementById("lcmbo-css")) return;
    const st = document.createElement("style");
    st.id = "lcmbo-css";
    st.textContent = `
.lcmbo-key{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:6px 10px;margin:0 0 4px;
  background:#0f1620;border:1px solid #243041;border-radius:6px;font:11px/1.3 SF Mono,ui-monospace,Menlo,Consolas,monospace;color:#c7d2e0}
.lcmbo-key b{color:#fff;font-weight:700}
.lcmbo-key .it{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.lcmbo-key .ln{display:inline-block;width:26px;background:currentColor;border-radius:1px}
.lcmbo-key .bd{display:inline-block;width:22px;height:9px;border:1px solid currentColor;border-radius:2px;opacity:.85}
.lcmbo-key .st{flex:1 0 100%;order:-1;margin:0;text-align:left;color:#8fa0b5}
.lcmbo-key button{font:inherit;padding:2px 8px;border:1px solid #33465e;background:#16212f;color:#c7d2e0;border-radius:4px;cursor:pointer}
.lcmbo-key button:hover{background:#1f2f44}
.lcmbo-tip{position:fixed;z-index:9999;pointer-events:none;max-width:340px;padding:7px 9px;border-radius:6px;
  background:rgba(8,12,18,.96);border:1px solid #33465e;color:#dfe6f5;font:11px/1.45 SF Mono,ui-monospace,Menlo,Consolas,monospace}
.lcmbo-tip .h{font-weight:700;margin-bottom:2px}
.lcmbo-card{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;font-size:11px}
.lcmbo-card label{display:flex;flex-direction:column;gap:2px}
#lc-ov .lcmbo-card label.checkbox{flex-direction:row !important;align-items:center !important;justify-content:flex-start !important;
  gap:6px !important;text-align:left !important;white-space:nowrap;margin:0 !important}
#lc-ov .lcmbo-card label.checkbox input{width:auto !important;margin:0 !important;flex:0 0 auto}
.lcmbo-card .full{grid-column:1/-1}
.lcmbo-card h5{grid-column:1/-1;margin:6px 0 0;font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:#7f91a8}
.lcmbo-card input[type=number]{width:100%}`;
    document.head.appendChild(st);
  }

  // ---------- parameter card ----------
  function num(id, label, min, max, step, tip) {
    return `<label title="${tip || ""}">${label}<input type="number" id="lc-mbo-${id}" min="${min}" max="${max}" step="${step}" value="${S[id]}"></label>`;
  }
  function chk(id, label, tip) {
    return `<label class="checkbox" title="${tip || ""}"><input type="checkbox" id="lc-mbo-${id}"${S[id] ? " checked" : ""}> ${label}</label>`;
  }
  function paramsHtml() {
    css();
    return `<div class="lcmbo-card">
      <h5>Orders</h5>
      ${num("min_size", "Min size (lots)", 1, 100000, 1, "Only orders of at least this many contracts are drawn (NQ is mostly 1-lots: ~700 orders >=20 per session, ~110 >=50).")}
      ${num("label_min", "Label size ≥", 1, 100000, 1, "Print the size on orders at least this large.")}
      ${num("label_alive", "…alive ≥ (min)", 0, 600, 0.5, "…and alive at least this many minutes (hides instantly-filled prints).")}
      ${num("opacity", "Line opacity %", 10, 100, 1, "")}
      ${num("thick", "Thickness ×", 0.3, 4, 0.1, "Scale of the size→thickness mapping.")}
      ${chk("bids", "Bids")}${chk("asks", "Asks")}
      ${chk("filled", "Filled")}${chk("cancelled", "Cancelled")}
      ${chk("resting", "Resting")}${chk("icebergs", "Icebergs", "Limit orders with a hidden reserve (purple diamond). On draws them even when the shown size is under Min size. Off hides those orders.")}
      ${chk("extend", "Extend resting →", "Resting orders run to the right edge with a dotted line.")}
      ${chk("connectors", "Connectors", "Dotted path for orders that moved price (quote-chasing).")}
      ${chk("pre", "Pre-existing ◌", "Hollow start = already on the book at a UTC-midnight snapshot; true placement time is unknown.")}
      ${chk("edge", "Far-order edge ▲▼", "Triangles at the chart edge for orders outside the visible price range; the key-bar buttons jump to them.")}
      ${num("approach", "Approach (pts)", 0, 500, 0.5, "A cancelled order counts as pulled in when price was within this many points of it.")}
      <h5>Reload chains</h5>
      ${chk("chains", "Show chains", "A filled order re-added at the SAME price within seconds, repeatedly (ask = passive selling, bid = passive buying).")}
      ${num("clip_min", "Clip ≥ (lots)", 1, 1000, 1, "Smallest order tracked for chains (independent of Min size).")}
      ${num("chain_n", "Re-adds ≥", 1, 50, 1, "Consecutive fill→re-add links required.")}
      ${num("chain_dt", "Within (s)", 0.1, 60, 0.1, "Max seconds between a fill and its re-add.")}
      ${num("chain_secs", "Lasts ≥ (s)", 0, 3600, 1, "Drops sub-10s bursts (price sweeping through levels).")}
      ${num("chain_abs", "Absorbs ≥", 0, 100000, 1, "Total contracts filled across the chain.")}
      ${num("break_pts", "Break (pts)", 0.25, 100, 0.25, "HELD/BROKE: price traded this far through the level within 5 min.")}
      <h5>View</h5>
      ${chk("grid", "Price grid 10/50", "Faint line every 10 points, bold every 50, with labels.")}
      ${chk("wheel", "Wheel scrolls prices", "Mouse wheel over the price pane moves the price axis; Ctrl+wheel zooms it; Shift+wheel is left to the chart (time).")}
      <div class="full" style="color:#7f91a8">Source: databento.mbo (rolling ~1 month, served ~8 h behind). Window = chart lookback, newest 5 days.</div>
    </div>`;
  }

  // ---------- events on our own controls ----------
  function readControl(el) {
    const k = el.id.slice("lc-mbo-".length);
    if (!(k in DEF)) return null;
    if (el.type === "checkbox") S[k] = el.checked;
    else {
      const v = Number(el.value);
      if (!Number.isFinite(v)) return null;
      S[k] = v;
    }
    lsSet(k, typeof S[k] === "boolean" ? (S[k] ? "1" : "0") : S[k]);
    return k;
  }
  function onControl(ev) {
    const el = ev.target;
    if (!el || !el.id || el.id.indexOf("lc-mbo-") !== 0) return;
    ev.stopPropagation();                        // keep app.js's #lc-ov handlers out of it
    if (ev.type === "input" && el.type === "checkbox") return;
    const k = readControl(el);
    if (!k) return;
    if (SERVER_KEYS.includes(k)) {
      clearTimeout(timer);
      timer = setTimeout(() => { if (bridge) load(bridge, bridge.payload); }, 450);
    } else if (k === "icebergs") {
      renderKey();
      if (bridge) bridge.scheduleDraw();
    } else if (bridge) bridge.scheduleDraw();
  }
  document.addEventListener("input", onControl, true);
  document.addEventListener("change", onControl, true);

  // ---------- data ----------
  function selected(b) { return !!(b && b.selectedKeys && b.selectedKeys().includes(KEY)); }
  function windowOf(b, p) {
    let a = p && p.start ? Date.parse(p.start) / 1000 : NaN;
    let z = p && p.end ? Date.parse(p.end) / 1000 : NaN;
    const m = b.lastMaps && b.lastMaps();
    if (!Number.isFinite(a) && m && m.real && m.real.length) a = m.real[0];
    if (!Number.isFinite(z) && m && m.real && m.real.length) z = m.real[m.real.length - 1] + 60;
    return [a, z];
  }
  function query(b, p) {
    const [a, z] = windowOf(b, p);
    if (!Number.isFinite(a) || !Number.isFinite(z)) return "";
    // The chart window is "now"-relative and refreshes every few seconds: quantize it (start down to
    // 30 min, end up to 5 min) so the query identity is stable and in-flight/cached results are reused.
    const a0 = Math.floor(a / 1800) * 1800, z0 = Math.ceil(z / 300) * 300;
    const params = {
      start: new Date(a0 * 1000).toISOString(), end: new Date(z0 * 1000).toISOString(),
      min_size: S.min_size, clip_min: S.clip_min, approach: S.approach, chain_n: S.chain_n,
      chain_dt: S.chain_dt, chain_min_secs: S.chain_secs, chain_min_abs: S.chain_abs, break_pts: S.break_pts,
    };
    const sym = b.payload && b.payload.contract;
    if (sym && /^NQ[FGHJKMNQUVXZ][0-9]$/.test(sym)) params.symbol = sym;
    return new URLSearchParams(params).toString();
  }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
  function isBusy(msg) {
    return /still computing|retry shortly/i.test(String(msg || ""));
  }
  async function load(b, p) {
    bridge = b;
    css();
    if (!selected(b)) { clear(); return; }
    const q = query(b, p);
    if (!q) return;
    if (q === dataQ && data) { b.scheduleDraw(); return; }
    if (q === loading) return;
    loading = q; err = "";
    renderKey();
    if (b.scheduleDraw) b.scheduleDraw();
    const t0 = Date.now();
    try {
      // 180s per attempt: 5-day NQU6 with instrument_id is ~40s cold. 503 "still computing"
      // retries up to 360s so a competing window does not leave the overlay silently blank.
      // A superseded request still toasts if nothing has been drawn yet.
      let out;
      while (true) {
        try {
          out = await b.api(`/api/chart/mbo-orders?${q}`, { timeoutMs: 180000 });
          break;
        } catch (e) {
          const msg = (e && e.message) ? e.message : "request failed";
          if (loading !== q) throw e;
          if (isBusy(msg) && (Date.now() - t0) < 360000) {
            err = "busy — retrying…";
            renderKey();
            if (b.scheduleDraw) b.scheduleDraw();
            await sleep(2500);
            continue;
          }
          throw e;
        }
      }
      if (loading !== q) return;
      data = prep(out); dataQ = q; err = "";
    } catch (e) {
      const msg = (e && e.message) ? e.message : "request failed";
      if (loading === q) {
        data = null; dataQ = ""; err = msg;
        if (typeof toast === "function") toast("MBO order map: " + err, "err", 12000);
      } else if (!data) {
        err = err || msg;
        if (typeof toast === "function") toast("MBO order map: " + msg, "err", 12000);
      }
    }
    if (loading === q) loading = "";
    renderKey();
    if (bridge) bridge.scheduleDraw();
  }
  function prep(out) {
    const byId = new Map();
    for (const o of out.orders || []) {
      if (!byId.has(o.i)) byId.set(o.i, []);
      byId.get(o.i).push(o);
    }
    const fin = new Map();
    for (const [id, segs] of byId) {
      segs.sort((x, y) => x.a - y.a);
      fin.set(id, segs[segs.length - 1].e);
    }
    out._byId = byId; out._fin = fin;
    return out;
  }
  function clear() {
    data = null; dataQ = ""; loading = ""; err = ""; hits = []; farUp = farDown = null;
    keyEl = null;
    const head = document.getElementById("lc-mbo-head");
    if (head) { head.hidden = true; head.replaceChildren(); }
    if (tipEl) tipEl.style.display = "none";
    closeOrder();
  }

  // ---------- key bar ----------
  function ctTime(sec) {
    if (!Number.isFinite(sec)) return "";
    return new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago", hourCycle: "h23",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    }).format(new Date(sec * 1000));
  }
  function ctStamp(sec) {
    if (!Number.isFinite(sec)) return "";
    return new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago", hourCycle: "h23",
      year: "2-digit", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    }).format(new Date(sec * 1000)).replace(",", "");
  }
  function fmtP(p) { return Number(p).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }
  function statusText() {
    if (loading) return "loading order map…";
    if (err) return `unavailable: ${err}`;
    if (!data) return "";
    const sm = data.summary || {}, oc = sm.outcomes || {};
    const pulledIn = (oc.PULLED_AFTER_TOUCH || 0) + (oc.PULLED_ON_APPROACH || 0);
    const days = Math.max(1, Math.round((Number(data.max_span_h) || 120) / 24));
    const range = (Number.isFinite(data.start) && Number.isFinite(data.end))
      ? `${ctStamp(data.start)} - ${ctStamp(data.end)} CT` : "";
    return `${sm.n_orders || 0} orders ≥${S.min_size}`
      + ` (${oc.FILLED || 0} filled · ${oc.RESTING || 0} resting · ${pulledIn} pulled in · ${oc.PULLED_FAR || 0} pulled away${oc.CLEARED ? ` · ${oc.CLEARED} exchange-cleared` : ""})`
      + ` · ${(data.chains || []).length} reload chain${(data.chains || []).length === 1 ? "" : "s"}`
      + (sm.n_icebergs
        ? (S.icebergs
          ? ` · ${sm.n_icebergs} iceberg${sm.n_icebergs === 1 ? "" : "s"} absorbing ${sm.ice_absorbed}`
          : ` · ${sm.n_icebergs} iceberg${sm.n_icebergs === 1 ? "" : "s"} hidden`)
        : "")
      + (range ? ` · ${range}` : "")
      + (data.truncated ? ` · Limit: ${days} day max` : "")
      + (data.clamped ? " · clamped to MBO coverage" : "")
      + (data.note ? ` · ${data.note}` : "");
  }
  function legendBody() {
    const sizes = [20, 50, 100, 250].map((z) =>
      `<span class="it"><i class="ln" style="height:${Math.round(thickness(z))}px;color:#8fa0b5"></i>${z}${z === 250 ? "+" : ""}</span>`).join("");
    return `<span class="it" style="color:${BID}"><i class="ln" style="height:4px"></i>BID</span>`
      + `<span class="it" style="color:${ASK}"><i class="ln" style="height:4px"></i>ASK</span>`
      + `<span class="it">size:</span>${sizes}`
      + `<span class="it">● filled</span><span class="it">✕ cancelled</span><span class="it">▶ resting →</span><span class="it">◌ pre-existing</span>`
      + `<span class="it" style="color:rgb(${ASK_BAND})"><i class="bd"></i>ASK reload</span>`
      + `<span class="it" style="color:rgb(${BID_BAND})"><i class="bd"></i>BID reload</span>`
      + `<span class="it" style="color:${CLR}"><b>R</b>&nbsp;exchange cleared</span>`
      + `<span class="it"><b style="color:${ICE}">◆</b>&nbsp;iceberg (hidden reserve, rim = side, size = absorbed)</span>`
      + `<button type="button" data-act="fit" title="Back to auto price range">Fit candles</button>`
      + `<button type="button" data-act="all" title="Show every drawn order, including far-away ones">Fit all orders</button>`
      + `<button type="button" data-act="up" title="Scroll to the nearest order above the view">▲ ${farUp ? farUp.n : 0}</button>`
      + `<button type="button" data-act="down" title="Scroll to the nearest order below the view">▼ ${farDown ? farDown.n : 0}</button>`;
  }
  function renderKey() {
    const head = document.getElementById("lc-mbo-head");
    if (!bridge || !selected(bridge)) {
      keyEl = null;
      if (head) { head.hidden = true; head.replaceChildren(); }
      return;
    }
    if (!head) return;
    if (!head.dataset.bound) {
      head.dataset.bound = "1";
      head.addEventListener("click", (ev) => {
        const btn = ev.target.closest("[data-act]");
        if (!btn) return;
        ev.preventDefault();
        keyAction(btn.dataset.act);
      });
    }
    const open = lsGet("legend_open", "0") === "1";
    head.hidden = false;
    head.innerHTML = `<div class="lc-mbo-stats">`
      + `<b class="k">MBO stats</b><span class="v"></span>`
      + `<button type="button" class="lc-mbo-legend-btn" data-act="legend" aria-pressed="${open ? "true" : "false"}">Order Map</button>`
      + `</div><div class="lcmbo-key"${open ? "" : " hidden"}>${legendBody()}</div>`;
    const v = head.querySelector(".v");
    if (v) {
      v.textContent = statusText();
      v.classList.toggle("err", !!(err || (data && data.note)));
    }
    keyEl = head.querySelector(".lcmbo-key");
  }
  function keyAction(act) {
    if (act === "legend") {
      lsSet("legend_open", lsGet("legend_open", "0") === "1" ? "0" : "1");
      renderKey();
      return;
    }
    if (!bridge || !bridge.candleSeries) return;
    const ps = bridge.candleSeries.priceScale();
    if (act === "fit") { ps.setAutoScale(true); bridge.scheduleDraw(); return; }
    if (!data) return;
    const ps_ = (data.orders || []).filter(passesFilter).map((o) => o.p);
    if (act === "all" && ps_.length) {
      const lo = Math.min(...ps_), hi = Math.max(...ps_), pad = Math.max(2, (hi - lo) * 0.03);
      ps.setVisibleRange({ from: lo - pad, to: hi + pad }); bridge.scheduleDraw(); return;
    }
    const r = ps.getVisibleRange();
    if (!r) return;
    const span = r.to - r.from;
    const tgt = act === "up" ? farUp : farDown;
    if (tgt) {
      const c = tgt.price;
      ps.setVisibleRange(act === "up" ? { from: c - span * 0.85, to: c + span * 0.15 } : { from: c - span * 0.15, to: c + span * 0.85 });
      bridge.scheduleDraw();
    }
  }

  // ---------- geometry helpers ----------
  function thickness(z) {
    const t = (1.8 + 2.2 * Math.log2(Math.max(z, S.min_size) / Math.max(1, S.min_size))) * S.thick;
    return Math.max(1, Math.min(14, t));
  }
  function labelPx(b) {
    const n = b && typeof b.chartLabelSize === "function" ? Number(b.chartLabelSize()) : 20;
    return Number.isFinite(n) ? Math.min(48, Math.max(10, Math.round(n))) : 20;
  }
  function labelFont(px, weight) {
    return `${weight ? weight + " " : ""}${px}px SF Mono, ui-monospace, Menlo, Consolas, monospace`;
  }
  function passesFilter(o) {
    if (o.ice && !S.icebergs) return false;
    if (o.s === "B" ? !S.bids : !S.asks) return false;
    const end = (data && data._fin.get(o.i)) || o.e;
    if (end === "filled") return S.filled;
    if (end === "cancelled") return S.cancelled;
    if (end === "resting") return S.resting;
    return true;
  }
  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // ---------- drawing ----------
  function draw(ctx, plotW, b) {
    bridge = b;
    hits = [];
    if (!selected(b)) { clear(); return; }
    renderKey();
    bindHost(b);
    if (!data || !b.chart || !b.candleSeries) {
      if ((loading || err) && ctx && b.candleSeries) {
        const box = b.pricePaneBox();
        const leftPad = b.plotLeftPad();
        ctx.save();
        ctx.font = "700 13px SF Mono, ui-monospace, Menlo, Consolas, monospace";
        ctx.fillStyle = (err && !loading) ? "#ff6b6b" : "#dfe6f5";
        ctx.textAlign = "left"; ctx.textBaseline = "top";
        const msg = loading ? (err || "loading order map…") : ("order map: " + err);
        ctx.fillText(msg, leftPad + 10, (box && box.top != null ? box.top : 8) + 8);
        ctx.restore();
      }
      return;
    }
    const box = b.pricePaneBox();
    const leftPad = b.plotLeftPad();
    const maps = b.lastMaps && b.lastMaps();
    if (!maps || !maps.real || !maps.real.length) return;
    const tFirst = maps.real[0], tLast = maps.real[maps.real.length - 1];
    const xL = leftPad, xR = leftPad + plotW;
    const xOf = (t) => {
      const c = Math.min(Math.max(t, tFirst), tLast);
      const x = b.timeToX(c);
      return x == null || !Number.isFinite(x) ? null : x + leftPad;
    };
    const showNight = b.ethDisplay();
    const alpha = Math.max(0.1, Math.min(1, S.opacity / 100));
    const pTop = b.candleYToPrice(box.top), pBot = b.candleYToPrice(box.top + box.height);
    if (pTop == null || pBot == null) return;
    const lo = Math.min(pTop, pBot), hi = Math.max(pTop, pBot);
    const fs = labelPx(b);

    ctx.save();
    ctx.beginPath();
    ctx.rect(xL, box.top, plotW, box.height);
    ctx.clip();

    if (S.grid) drawGrid(ctx, b, box, xL, xR, lo, hi, fs);

    // reload chains (bands under the lines)
    if (S.chains) {
      for (const c of data.chains || []) {
        if (b.pastReplay(c.a)) continue;
        const xa = xOf(c.a), xb0 = xOf(Math.min(c.b, tLast));
        const y0 = b.candlePriceToY(c.p + 0.9), y1 = b.candlePriceToY(c.p - 0.9);
        if (xa == null || xb0 == null || y0 == null || y1 == null) continue;
        const xb = Math.max(xb0, xa + 6), yy = Math.min(y0, y1), hh = Math.max(6, Math.abs(y1 - y0));
        const rgb = c.s === "A" ? ASK_BAND : BID_BAND;
        ctx.fillStyle = `rgba(${rgb},.33)`; ctx.strokeStyle = `rgba(${rgb},.9)`; ctx.lineWidth = 1;
        ctx.fillRect(xa, yy, xb - xa, hh); ctx.strokeRect(xa, yy, xb - xa, hh);
        ctx.font = labelFont(fs, 700);
        ctx.fillStyle = `rgb(${rgb})`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        ctx.fillText(`↻ ${c.s === "A" ? "ASK" : "BID"} reload ×${c.n} (~${c.clip} lots, Σ${c.abs}) ${c.pinned ? "PINNED " : ""}${c.after}`, xb + 4, yy + hh / 2);
        hits.push({ kind: "chain", x0: xa, x1: xb, y0: yy, y1: yy + hh, c });
      }
    }

    // connectors for orders that changed price
    if (S.connectors) {
      ctx.setLineDash([2, 3]); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(207,214,230,.55)";
      for (const segs of data._byId.values()) {
        if (segs.length < 2 || !passesFilter(segs[0]) || segs[0].z < S.min_size) continue;
        for (let i = 0; i + 1 < segs.length; i++) {
          if (!passesFilter(segs[i]) || !passesFilter(segs[i + 1])) continue;
          const xa = xOf(segs[i].b), xb = xOf(segs[i + 1].a);
          const ya = b.candlePriceToY(segs[i].p), yb = b.candlePriceToY(segs[i + 1].p);
          if (xa == null || xb == null || ya == null || yb == null) continue;
          ctx.beginPath(); ctx.moveTo(xa, ya); ctx.lineTo(xb, yb); ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // order lines
    let nUp = 0, nDown = 0, upP = null, downP = null;
    const marks = [], labels = [];
    for (const o of data.orders || []) {
      if (!passesFilter(o)) continue;
      if (!showNight && !b.isRthUnix(o.a) && !b.isRthUnix(o.b)) continue;
      if (b.pastReplay(o.a)) continue;
      if (o.p > hi) { nUp++; if (upP == null || o.p < upP) upP = o.p; continue; }
      if (o.p < lo) { nDown++; if (downP == null || o.p > downP) downP = o.p; continue; }
      const y = b.candlePriceToY(o.p);
      if (y == null || !Number.isFinite(y)) continue;
      const resting = o.e === "resting" || b.pastReplay(o.b);
      const xa = xOf(o.a);
      let xb = resting && S.extend ? xR : xOf(o.b);
      if (xa == null || xb == null || xb < xL || xa > xR) continue;
      // An iceberg's displayed size is tiny by design, so thickness comes from what it actually
      // absorbed -- otherwise the order that soaked up the most liquidity draws the thinnest line.
      const w = thickness(o.ice ? o.ia : o.z), col = o.s === "B" ? BID : ASK;
      ctx.strokeStyle = rgba(col, alpha); ctx.lineWidth = w; ctx.lineCap = "butt";
      ctx.beginPath(); ctx.moveTo(xa, y); ctx.lineTo(Math.max(xb, xa + 1), y); ctx.stroke();
      hits.push({ kind: "order", x0: xa, x1: Math.max(xb, xa + 1), y, w: Math.max(w, 6), o });
      marks.push({ o, xa, xb: resting ? xR : xb, y, col, resting });
      if (o.z >= S.label_min && (o.b - o.a) / 60 >= S.label_alive) labels.push({ o, xa, y, col });
    }
    for (const m of marks) {
      const { o, y, col } = m;
      if (o.e === "filled") {
        ctx.beginPath(); ctx.arc(m.xb, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff"; ctx.fill(); ctx.lineWidth = 1.6; ctx.strokeStyle = col; ctx.stroke();
      } else if (o.e === "cancelled") {
        ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.beginPath();
        ctx.moveTo(m.xb - 3.5, y - 3.5); ctx.lineTo(m.xb + 3.5, y + 3.5);
        ctx.moveTo(m.xb - 3.5, y + 3.5); ctx.lineTo(m.xb + 3.5, y - 3.5); ctx.stroke();
      } else if (o.e === "cleared") {
        // Exchange took it out (session/weekend purge or a book Clear) -- not the trader's doing,
        // so deliberately NOT the cancel cross. ADR-149.
        ctx.save();
        ctx.font = labelFont(fs, 700);
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillStyle = CLR; ctx.strokeStyle = "rgba(8,12,18,.85)"; ctx.lineWidth = Math.max(2.5, fs * 0.2);
        ctx.strokeText("R", m.xb + Math.round(fs * 0.55), y); ctx.fillText("R", m.xb + Math.round(fs * 0.55), y);
        ctx.restore();
      } else if (m.resting) {
        if (S.extend) {
          ctx.setLineDash([1, 3]); ctx.strokeStyle = rgba(col, 0.9); ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(m.xa, y); ctx.lineTo(xR, y); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.fillStyle = col; ctx.beginPath();
        ctx.moveTo(xR - 1, y); ctx.lineTo(xR - 9, y - 4.5); ctx.lineTo(xR - 9, y + 4.5); ctx.closePath(); ctx.fill();
      }
      if (S.pre && o.pre) {
        ctx.beginPath(); ctx.arc(m.xa, y, 4, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2; ctx.stroke();
      }
      if (o.ice) {
        // Hidden reserve. Purple CENTRE (the hidden part) inside a side-coloured rim, so BID and
        // ASK icebergs stay visually distinct; radius steps with what it actually absorbed.
        const n = o.ia || 0;
        const r = n >= 100 ? 8 : n >= 50 ? 6.5 : n >= 20 ? 5 : 3.5;
        ctx.beginPath();
        ctx.moveTo(m.xa, y - r); ctx.lineTo(m.xa + r, y);
        ctx.lineTo(m.xa, y + r); ctx.lineTo(m.xa - r, y); ctx.closePath();
        ctx.fillStyle = ICE; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      }
    }
    ctx.font = labelFont(fs, 700);
    ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    const labelH = fs + Math.max(2, Math.round(fs * 0.2));
    const labelPad = Math.max(3, Math.round(fs * 0.28));
    for (const l of labels) {
      const t = String(l.o.z), tw = ctx.measureText(t).width + labelPad * 2;
      const bx = Math.max(xL + 1, l.xa), by = l.y - thickness(l.o.z) / 2 - 1;
      ctx.fillStyle = rgba(l.col, 0.85);
      ctx.fillRect(bx, by - labelH, tw, labelH);
      ctx.fillStyle = "#ffffff"; ctx.fillText(t, bx + labelPad, by - Math.max(1, Math.round(fs * 0.08)));
    }

    // far-order edge markers
    if (S.edge) {
      ctx.fillStyle = "#ffd166";
      for (const o of data.orders || []) {
        if (!passesFilter(o) || (o.p <= hi && o.p >= lo)) continue;
        const x = xOf(o.a);
        if (x == null || x < xL || x > xR) continue;
        const up = o.p > hi, y = up ? box.top + 6 : box.top + box.height - 6;
        ctx.beginPath();
        if (up) { ctx.moveTo(x, y - 4); ctx.lineTo(x - 4, y + 3); ctx.lineTo(x + 4, y + 3); }
        else { ctx.moveTo(x, y + 4); ctx.lineTo(x - 4, y - 3); ctx.lineTo(x + 4, y - 3); }
        ctx.closePath(); ctx.fill();
      }
      ctx.font = labelFont(fs, 600);
      const edgePad = Math.max(6, Math.round(fs * 0.7));
      const edgeX = xL + Math.round(fs * 3);
      ctx.textBaseline = "top";
      if (nUp) ctx.fillText(`▲ ${nUp} above (nearest ${fmtP(upP)})`, edgeX, box.top + edgePad);
      ctx.textBaseline = "bottom";
      if (nDown) ctx.fillText(`▼ ${nDown} below (nearest ${fmtP(downP)})`, edgeX, box.top + box.height - edgePad);
    }
    ctx.restore();

    farUp = nUp ? { n: nUp, price: upP } : null;
    farDown = nDown ? { n: nDown, price: downP } : null;
    updateFarButtons();
  }
  function updateFarButtons() {
    if (!keyEl) return;
    const up = keyEl.querySelector('[data-act="up"]'), dn = keyEl.querySelector('[data-act="down"]');
    if (up) up.textContent = `▲ ${farUp ? farUp.n : 0}`;
    if (dn) dn.textContent = `▼ ${farDown ? farDown.n : 0}`;
  }
  function drawGrid(ctx, b, box, xL, xR, lo, hi, fs) {
    const px = Number.isFinite(fs) ? fs : 20;
    const pxPer = box.height / Math.max(1e-9, hi - lo);
    const step = [10, 50, 100, 250, 500, 1000].find((s) => s * pxPer >= px + 6) || 1000;
    const major = step >= 50 ? step * 2 : 50;
    ctx.font = labelFont(px);
    ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    for (let p = Math.ceil(lo / step) * step; p <= hi; p += step) {
      const y = b.candlePriceToY(p);
      if (y == null) continue;
      const isMajor = Math.abs(p % (step === 10 ? 50 : major)) < 1e-6;
      ctx.strokeStyle = isMajor ? "rgba(214,222,240,.34)" : "rgba(150,164,190,.16)";
      ctx.lineWidth = isMajor ? 1.2 : 0.8;
      ctx.beginPath(); ctx.moveTo(xL, Math.round(y) + 0.5); ctx.lineTo(xR, Math.round(y) + 0.5); ctx.stroke();
      ctx.fillStyle = isMajor ? "rgba(214,222,240,.75)" : "rgba(150,164,190,.55)";
      ctx.fillText(p.toLocaleString("en-US"), xL + 4, y - 1);
    }
  }

  // ---------- hover + wheel ----------
  function bindHost(b) {
    const host = b.host();
    if (!host || hostBound === host) return;
    hostBound = host;
    host.addEventListener("mousemove", onMove);
    host.addEventListener("mouseleave", () => { if (tipEl) tipEl.style.display = "none"; });
    host.addEventListener("wheel", onWheel, { capture: true, passive: false });
    host.addEventListener("dblclick", onDblClick, true);
  }
  function tip() {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "lcmbo-tip"; document.body.appendChild(tipEl); }
    return tipEl;
  }
  function pickHit(ev) {
    if (!bridge || !hits.length) return null;
    const host = bridge.host && bridge.host();
    if (!host) return null;
    const r = host.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 1e9, chain = null;
    for (const h of hits) {
      if (h.kind === "chain") {
        if (!chain && x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1) chain = h;
      } else if (x >= h.x0 - 8 && x <= h.x1 + 8) {
        const d = Math.abs(y - h.y);
        if (d <= Math.max(h.w / 2 + 5, 8) && d < bd) { best = h; bd = d; }
      }
    }
    return best || chain;
  }
  function onMove(ev) {
    if (!bridge || !selected(bridge) || !hits.length) { if (tipEl) tipEl.style.display = "none"; return; }
    const best = pickHit(ev);
    const el = tip();
    if (!best) { el.style.display = "none"; return; }
    el.innerHTML = best.kind === "chain" ? chainTip(best.c) : orderTip(best.o);
    el.style.display = "block";
    const w = el.offsetWidth, hgt = el.offsetHeight;
    el.style.left = Math.min(window.innerWidth - w - 8, ev.clientX + 14) + "px";
    el.style.top = Math.min(window.innerHeight - hgt - 8, ev.clientY + 14) + "px";
  }
  const OUT = {
    FILLED: "filled", PULLED_AFTER_TOUCH: "pulled after price touched it", PULLED_ON_APPROACH: "pulled as price approached",
    PULLED_FAR: "pulled while price was far", RESTING: "still resting", MOVED: "moved",
    CLEARED: "removed by the exchange at a session close — not cancelled by the trader",
  };
  function dur(sec) {
    if (sec < 90) return `${Math.round(sec)} s`;
    if (sec < 5400) return `${(sec / 60).toFixed(1)} min`;
    return `${(sec / 3600).toFixed(1)} h`;
  }
  function orderTip(o) {
    const col = o.s === "B" ? BID : ASK;
    const side = o.s === "B" ? "BID" : "ASK";
    const shown = o.ice ? o.idisp : o.z;                 // what it ever displayed here
    const got = o.ice ? o.ia : o.f;                      // what actually executed here
    const verb = o.ice ? "absorbed" : "filled";
    const iceHdr = o.ice ? ` <span style="color:${ICE}">◇ ICEBERG</span>` : "";
    // Where this order went next, if it re-priced: direction, distance and what it did there.
    const segs = (data && data._byId.get(o.i)) || [];
    const nxt = segs.filter(s => s.a >= o.b && s.p !== o.p).sort((x, y) => x.a - y.a)[0];
    let nextLine;
    if (o.e === "moved" && nxt) {
      const d = nxt.p - o.p;
      const did = nxt.ice ? nxt.ia : nxt.f;
      nextLine = `moved price ${d > 0 ? "up" : "down"} ${Math.abs(d).toFixed(2)} pts`
        + ` → ${did ? `${nxt.ice ? "absorbed" : "filled"} ${did}` : "nothing filled there"}`;
    } else if (o.e === "moved") {
      nextLine = "moved price";
    } else {
      nextLine = (OUT[o.o] || o.o || "")
        + (o.d != null && o.o && o.o !== "FILLED" ? ` · ${Number(o.d).toFixed(1)} pts from price at end` : "");
    }
    const L = (k, v) => `<span style="color:#7f91a8">${k}:</span> ${v}`;
    return `<div class="h" style="color:${col}">${side} ${got} / ${shown} ${verb}${iceHdr}</div>`
      + L("order_id", o.i) + "<br>"
      + L("price", fmtP(o.p)) + "<br>"
      + L("time", `${dur(o.b - o.a)} (${ctTime(o.a)} - ${ctTime(o.b)} CT)`
                  + (o.pre ? " · start at snapshot, true time unknown" : "")) + "<br>"
      + L("next", nextLine)
      + `<br><span style="color:#8fa0b5">double-click for MBO rows</span>`;
  }
  function chainTip(c) {
    return `<div class="h" style="color:rgb(${c.s === "A" ? ASK_BAND : BID_BAND})">${c.s === "A" ? "ASK" : "BID"} reload chain @ ${fmtP(c.p)}</div>`
      + `${c.n} orders of ~${c.clip} lots, each filled then re-added at the same price<br>`
      + `${c.abs} contracts absorbed over ${dur(c.b - c.a)} (${ctTime(c.a)}–${ctTime(c.b)} CT)<br>`
      + `${c.pinned ? "price never traded through the level while it ran<br>" : ""}`
      + `next 5 min: <b>${c.after}</b> (${c.after === "BROKE" ? "traded through the level" : "did not trade through"})`
      + `${c.mfe != null ? ` · best move away from the level ${Number(c.mfe).toFixed(1)} pts` : ""}`
      + `<br><span style="color:#8fa0b5">${c.s === "A" ? "passive selling / distribution" : "passive buying / accumulation"} — behaviour, not proof of intent</span>`;
  }
  function onWheel(ev) {
    if (!bridge || !selected(bridge) || !S.wheel || ev.shiftKey || !bridge.candleSeries) return;
    const box = bridge.pricePaneBox();
    const r = bridge.host().getBoundingClientRect();
    const y = ev.clientY - r.top;
    if (y < box.top || y > box.top + box.height) return;             // other panes keep native behaviour
    const ps = bridge.candleSeries.priceScale();
    const cur = ps.getVisibleRange();
    if (!cur) return;
    ev.preventDefault(); ev.stopPropagation();
    const span = cur.to - cur.from;
    if (ev.ctrlKey || ev.metaKey) {
      const f = ev.deltaY > 0 ? 1.15 : 1 / 1.15, mid = (cur.from + cur.to) / 2;
      ps.setVisibleRange({ from: mid - (span * f) / 2, to: mid + (span * f) / 2 });
    } else {
      const d = Math.max(-0.3, Math.min(0.3, ev.deltaY / 700)) * span * -1;
      ps.setVisibleRange({ from: cur.from + d, to: cur.to + d });
    }
    bridge.scheduleDraw();
  }

  // ---------- one order's MBO rows (double-click) ----------
  const ACT = { A: "Add", C: "Cancel", M: "Modify", F: "Fill", T: "Trade", R: "Clear", N: "None" };
  const SIDE = { B: "Bid", A: "Ask", N: "None" };
  const DETAIL_COLS = [
    ["ts_event", "ts_event"], ["sequence", "sequence"], ["action", "action"], ["side", "side"],
    ["price", "price"], ["size", "size"], ["ts_recv", "ts_recv"], ["flags", "flags"],
    ["order_id", "order_id"], ["symbol", "symbol"], ["channel_id", "channel_id"],
    ["instrument_id", "instrument_id"], ["ts_in_delta", "ts_in_delta"], ["rtype", "rtype"],
    ["publisher_id", "publisher_id"], ["dataset", "dataset"],
  ];
  function ctMicro(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?/.exec(String(iso || ""));
    if (!m) return iso == null ? "" : String(iso);
    const frac = (m[7] || "0").slice(0, 6).padEnd(6, "0");
    const ms = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago", hourCycle: "h23",
      year: "2-digit", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    }).formatToParts(new Date(ms));
    const g = {};
    for (const p of parts) if (p.type !== "literal") g[p.type] = p.value;
    return `${g.month}/${g.day}/${g.year} ${g.hour}:${g.minute}:${g.second}.${frac}`;
  }
  function flagText(n) {
    if (n == null || n === "") return "";
    const v = Number(n);
    if (!Number.isFinite(v)) return String(n);
    const bits = [];
    if (v & 128) bits.push("LAST");
    if (v & 64) bits.push("TOB");
    if (v & 32) bits.push("SNAPSHOT");
    if (v & 16) bits.push("MBP");
    if (v & 8) bits.push("BAD_TS_RECV");
    if (v & 4) bits.push("MAYBE_BAD_BOOK");
    return bits.length ? `${v} ${bits.join(" ")}` : String(v);
  }
  function cellText(key, row) {
    const v = row[key];
    if (v == null) return "";
    if (key === "ts_event" || key === "ts_recv") return ctMicro(v);
    if (key === "action") return `${v}${ACT[v] ? " " + ACT[v] : ""}`;
    if (key === "side") return `${v}${SIDE[v] ? " " + SIDE[v] : ""}`;
    if (key === "flags") return flagText(v);
    if (key === "price") return fmtP(v);
    return String(v);
  }
  function ensureDialog() {
    if (detailEl) return detailEl;
    const root = document.createElement("div");
    root.id = "lc-mbo-order";
    root.className = "lc-mbo-order";
    root.hidden = true;
    root.innerHTML = `<div class="lc-mbo-order-back" data-close="1"></div>`
      + `<div class="lc-mbo-order-panel" role="dialog" aria-modal="true" aria-labelledby="lc-mbo-order-title">`
      + `<div class="lc-mbo-order-bar"><h2 id="lc-mbo-order-title">MBO order</h2>`
      + `<button type="button" data-close="1">Close</button></div>`
      + `<div class="lc-mbo-order-meta"></div>`
      + `<div class="lc-mbo-order-scroll"></div></div>`;
    root.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-close]")) closeOrder();
    });
    document.body.appendChild(root);
    if (!window._lcMboOrderEsc) {
      window._lcMboOrderEsc = (ev) => {
        const dlg = document.getElementById("lc-mbo-order");
        if (!dlg || dlg.hidden || ev.key !== "Escape") return;
        ev.preventDefault();
        ev.stopPropagation();
        closeOrder();
      };
      window.addEventListener("keydown", window._lcMboOrderEsc, true);
    }
    detailEl = root;
    return root;
  }
  function closeOrder() {
    detailSeq += 1;
    if (!detailEl) return;
    detailEl.hidden = true;
  }
  function orderLife(o) {
    const segs = (data && data._byId && data._byId.get(o.i)) || [o];
    let a = o.a, b = o.b;
    for (const s of segs) {
      if (Number.isFinite(s.a) && s.a < a) a = s.a;
      if (Number.isFinite(s.b) && s.b > b) b = s.b;
    }
    return { a, b };
  }
  function fillDetail(root, payload) {
    const meta = root.querySelector(".lc-mbo-order-meta");
    const sc = root.querySelector(".lc-mbo-order-scroll");
    const days = Math.max(1, Math.round((Number(payload.max_span_h) || 120) / 24));
    const range = (Number.isFinite(payload.start) && Number.isFinite(payload.end))
      ? `${ctStamp(payload.start)} - ${ctStamp(payload.end)} CT` : "";
    const bits = [`${payload.n || 0} row${payload.n === 1 ? "" : "s"}`, "sorted by ts_event, sequence"];
    if (range) bits.push(range);
    if (payload.truncated) bits.push(`Limit: ${days} day max`);
    if (payload.row_cap) bits.push("first 20000 rows");
    if (payload.note) bits.push(payload.note);
    meta.textContent = bits.join(" · ");
    sc.replaceChildren();
    const rows = payload.rows || [];
    if (!rows.length) {
      const p = document.createElement("p");
      p.textContent = payload.note || "No MBO rows for this order in that window.";
      sc.appendChild(p);
      return;
    }
    const table = document.createElement("table");
    const thead = document.createElement("thead");
    const hr = document.createElement("tr");
    for (const [key, label] of DETAIL_COLS) {
      const th = document.createElement("th");
      th.textContent = label;
      th.scope = "col";
      hr.appendChild(th);
    }
    thead.appendChild(hr);
    const tb = document.createElement("tbody");
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const [key] of DETAIL_COLS) {
        const td = document.createElement("td");
        td.textContent = cellText(key, row);
        if (key === "action" && row.action) td.className = "act-" + row.action;
        if (key === "side" && row.side) td.className = "side-" + row.side;
        if ((key === "ts_event" || key === "ts_recv") && row[key]) td.title = String(row[key]);
        tr.appendChild(td);
      }
      tb.appendChild(tr);
    }
    table.appendChild(thead);
    table.appendChild(tb);
    sc.appendChild(table);
    sc.scrollTop = 0;
  }
  function onDblClick(ev) {
    const h = pickHit(ev);
    if (!h || h.kind !== "order") return;
    ev.preventDefault();
    ev.stopPropagation();
    openOrder(h.o);
  }
  async function openOrder(o) {
    if (!o || !bridge) return;
    if (tipEl) tipEl.style.display = "none";
    const root = ensureDialog();
    const seq = ++detailSeq;
    const title = root.querySelector("#lc-mbo-order-title");
    const meta = root.querySelector(".lc-mbo-order-meta");
    const sc = root.querySelector(".lc-mbo-order-scroll");
    const side = o.s === "B" ? "BID" : "ASK";
    title.textContent = `MBO order ${o.i}`;
    meta.textContent = `${side} ${fmtP(o.p)} · loading rows…`;
    sc.replaceChildren();
    const loading = document.createElement("p");
    loading.textContent = "Loading MBO rows for this order…";
    sc.appendChild(loading);
    root.hidden = false;
    const closeBtn = root.querySelector(".lc-mbo-order-bar button");
    if (closeBtn) closeBtn.focus();
    const life = orderLife(o);
    const sym = data && data.symbol;
    if (!sym || !Number.isFinite(life.a) || !Number.isFinite(life.b)) {
      meta.textContent = "This order has no symbol or time range to query.";
      sc.replaceChildren();
      return;
    }
    const q = new URLSearchParams({
      order_id: String(o.i),
      symbol: sym,
      start: new Date(life.a * 1000).toISOString(),
      end: new Date(life.b * 1000).toISOString(),
    });
    try {
      const payload = await bridge.api(`/api/chart/mbo-order?${q}`, { timeoutMs: 120000 });
      if (seq !== detailSeq) return;
      if (payload && payload.order_id) title.textContent = `MBO order ${payload.order_id}`;
      fillDetail(root, payload || {});
    } catch (e) {
      if (seq !== detailSeq) return;
      meta.textContent = "";
      sc.replaceChildren();
      const p = document.createElement("p");
      p.textContent = (e && e.message) ? e.message : "request failed";
      sc.appendChild(p);
    }
  }

  // test/debug hook: hover targets from the last draw, in host-relative CSS pixels
  function debugHits() {
    return hits.map((h) => h.kind === "chain"
      ? { kind: "chain", x: (h.x0 + h.x1) / 2, y: (h.y0 + h.y1) / 2 }
      : { kind: "order", x: (h.x0 + h.x1) / 2, y: h.y, z: h.o.z, p: h.o.p, w: h.w, ice: !!h.o.ice });
  }
  window.LCOrders = {
    KEY, paramsHtml, load, draw, clear, debugHits, closeDetail: closeOrder,
    state: () => Object.assign({}, S),
  };
})();
