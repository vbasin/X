/* VTech Console SPA — hash router + fetch API + polling. No dependencies. */
"use strict";

/* ── utilities ─────────────────────────────────────────────────────────── */
const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const esc = s => String(s ?? "").replace(/[&<>"']/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

/** Small right-aligned info icon; tip HTML shows on hover/focus of the icon only. */
function lcInfoTip(html) {
  return `<span class="lc-info" tabindex="0" aria-label="More info">`
    + `<span class="lc-info-i" aria-hidden="true">i</span>`
    + `<span class="lc-info-pop" role="tooltip">${html}</span>`
    + `</span>`;
}

async function api(path, opts = {}) {
  const {timeoutMs: timeoutMsOpt, headers: extraHeaders, ...fetchOpts} = opts;
  const method = (fetchOpts.method || "GET").toUpperCase();
  const headers = {...(extraHeaders || {})};
  if (method !== "GET" && method !== "HEAD") {
    if (headers["Content-Type"] == null) headers["Content-Type"] = "application/json";
    headers["X-VTech-Console"] = "1";
  }
  const timeoutMs = timeoutMsOpt != null ? timeoutMsOpt : 12000;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(path, {...fetchOpts, headers, signal: ctrl.signal});
  } catch (e) {
    if (e && e.name === "AbortError") throw new Error(`timeout after ${timeoutMs / 1000}s: ${path}`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok) throw new Error((body && body.detail) ? JSON.stringify(body.detail).replace(/^"|"$/g, "") : `${res.status} ${res.statusText}`);
  return body;
}
const post = (path, data, extra = {}) => api(path, {method: "POST", body: JSON.stringify(data), ...extra});
const put = (path, data) => api(path, {method: "PUT", body: JSON.stringify(data)});
const del = (path) => api(path, {method: "DELETE"});

function setNavCollapsed() {
  // Page links live in the top bar. A stored left-wall collapse must not hide them.
  const shellEl = document.getElementById("shell");
  if (shellEl) shellEl.classList.remove("nav-collapsed");
  localStorage.setItem("lc_nav_collapsed", "0");
  requestAnimationFrame(() => {
    try {
      if (typeof window._lcResizeChart === "function") window._lcResizeChart();
    } catch {}
  });
}
function armNavCollapse() {
  if (document.documentElement.dataset.navCollapseArmed) return;
  document.documentElement.dataset.navCollapseArmed = "1";
  setNavCollapsed();
}
armNavCollapse();

function toast(msg, kind = "info", ms = 5000) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toast-stack").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

/* All timestamps render in Chicago time (CST/CDT). API timestamps are UTC ISO
   strings; naive strings are treated as UTC. Non-parseable inputs (e.g. run ids
   like 20260711T033354Z) fall back to the old string surgery unchanged. */
const _CT_FMT = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"});
const TZ_ABBR = (() => {                              // "CDT" or "CST" for today
  try {
    return new Intl.DateTimeFormat("en-US", {timeZone: "America/Chicago", timeZoneName: "short"})
      .formatToParts(new Date()).find(p => p.type === "timeZoneName").value;
  } catch { return "CT"; }
})();
function toCT(ts) {
  let s = String(ts);
  if (!/[Zz]$|[+-]\d\d:?\d\d$/.test(s)) s += "Z";     // naive ⇒ assume UTC
  const d = new Date(s);
  if (isNaN(d)) return null;
  const p = Object.fromEntries(_CT_FMT.formatToParts(d).map(x => [x.type, x.value]));
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
}
const fmtTs = ts => {
  if (!ts) return "—";
  return toCT(ts) ?? String(ts).replace("T", " ").replace(/\.\d+Z?$/, "Z");
};
/* Like fmtTs but keeps milliseconds — for tick-exact times (signal/exit). */
const fmtTsMs = ts => {
  if (!ts) return "—";
  const base = toCT(ts);
  if (base == null) return fmtTs(ts);
  let s = String(ts);
  if (!/[Zz]$|[+-]\d\d:?\d\d$/.test(s)) s += "Z";
  const ms = new Date(s).getMilliseconds();
  return `${base}.${String(ms).padStart(3, "0")}`;
};
/* Tick-exact Databento ts_event: keep microseconds from the ISO string. */
function fmtTsUs(ts) {
  if (!ts) return "—";
  const base = toCT(ts);
  if (base == null) return fmtTs(ts);
  const m = String(ts).match(/\.(\d+)/);
  const frac = m ? m[1].padEnd(6, "0").slice(0, 6) : "000000";
  return `${base}.${frac}`;
}
const fmtDur = s => s == null ? "—" : s < 60 ? `${Math.round(s)}s` : s < 3600 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m`;
/** One folder up from the model files: ``<objective>/<run_dir_with_timestamp>``. */
function bundleRunLabel(path) {
  if (!path) return "";
  const parts = String(path).replace(/\\/g, "/").replace(/\/+$/, "").split("/");
  const name = parts.pop() || "";
  const parent = parts.pop() || "";
  if (!parent || /^(model_staging|model_registry|models)$/.test(parent)) return name;
  return `${parent}/${name}`;
}
/* Number formatting: standard en-US grouping (commas every 3 digits) when |n| ≥ 1,000.
   User-facing counts and large point totals stay readable (e.g. 12,345). */
const _numFinite = v => {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};
const fmtNum = (v, d = 3) => {
  const n = _numFinite(v);
  if (n == null) return "—";
  const abs = Math.abs(n);
  return n.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
    useGrouping: abs >= 1000,
  });
};
/** Integer-ish counts (signals, trades, bars) — commas at 1,000+. */
const fmtInt = v => {
  const n = _numFinite(v);
  if (n == null) return "—";
  return Math.round(n).toLocaleString("en-US", {useGrouping: Math.abs(n) >= 1000});
};
/** Signed points (e.g. +1,234.50). */
const fmtPts = (v, d = 2) => {
  const n = _numFinite(v);
  if (n == null) return "—";
  const body = fmtNum(Math.abs(n), d);
  if (n > 0) return "+" + body;
  if (n < 0) return "−" + body;   // U+2212 minus for display
  return body;
};
/** Dollar amount with $ and commas (e.g. $12,375 or −$1,200). */
const fmtUsd = (v, d = 0) => {
  const n = _numFinite(v);
  if (n == null) return "—";
  const body = Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
    useGrouping: Math.abs(n) >= 1000,
  });
  if (n < 0) return "−$" + body;
  return "$" + body;
};
/** Default CME $ / point when API omits point_value (NQ E-mini). */
const DEFAULT_POINT_VALUE = 20;

/* ── shared OCO / TBBO DATA prefs (Backtest + Live) ─────────────────────────
 * OCO off → on-close path (5s _close); OCO on → TBBO triple-barrier fills.
 * Pref keys: bt_oco / oco; migrate bt_tbbo_data / live_data / … */
function readBoolPref(key, legacyKeys) {
  const v = localStorage.getItem(key);
  if (v === "1" || v === "0") return v === "1";
  for (const ok of (legacyKeys || [])) {
    const ov = localStorage.getItem(ok);
    if (ov === "1" || ov === "0") {
      localStorage.setItem(key, ov);
      localStorage.removeItem(ok);
      return ov === "1";
    }
  }
  return false;
}
function writeBoolPref(key, on) {
  localStorage.setItem(key, on ? "1" : "0");
}
/** Bare track switch (labels supplied by caller left/right). checked ⇒ right side. */
function bareSwitchHtml(id, checked, onChangeExpr, title) {
  return `<label class="switch" title="${esc(title || "")}" for="${esc(id)}">
    <input type="checkbox" id="${esc(id)}" ${checked ? "checked" : ""}
      onchange="${onChangeExpr}">
    <span class="switch-track" aria-hidden="true"></span>
  </label>`;
}
// Live tab still uses TBBO DATA naming helpers
function readTbboDataPref(key) {
  return readBoolPref(key, key === "bt_tbbo_data"
    ? ["bt_oco", "bt_tbbo_dev", "bt_live_data"]
    : ["oco", "tbbo_data", "tbbo_dev", "live_data"]);
}
function writeTbboDataPref(key, on) { writeBoolPref(key, on); }
function tbboDataSwitchHtml(id, on, onChangeExpr, title) {
  return bareSwitchHtml(id, on, onChangeExpr, title);
}
const resolvePointValue = (ctx) => {
  if (!ctx) return DEFAULT_POINT_VALUE;
  const raw = ctx.point_value ?? (ctx.config && ctx.config.point_value)
    ?? (ctx.summary && ctx.summary.point_value);
  const n = _numFinite(raw);
  return (n != null && n > 0) ? n : DEFAULT_POINT_VALUE;
};
/** Points + optional $ (when point_value is meaningful). e.g. "+618.75 pt ($12,375)" */
const fmtPtsUsd = (pts, pointValue, d = 2) => {
  const p = fmtPts(pts, d);
  if (p === "—") return "—";
  const pv = _numFinite(pointValue);
  if (pv == null || pv <= 0) return `${p} pt`;
  const n = _numFinite(pts);
  if (n == null) return `${p} pt`;
  return `${p} pt (${fmtUsd(n * pv, 0)})`;
};
const since = iso => { if (!iso) return "—"; const s = (Date.now() - new Date(iso)) / 1000; return s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)}m ago` : `${(s / 3600).toFixed(1)}h ago`; };

const statusPill = st => {
  const map = {running: "info", complete: "ok", succeeded: "ok", failed: "err", cancelled: "warn",
               lost: "warn", aborted: "warn", empty: "dim", report: "dim",
               PASSED: "ok", NO_CHAMPION: "warn"};
  return `<span class="pill ${map[st] || "dim"}">${esc(st)}</span>`;
};

/* Minimal markdown renderer (reports are pipeline-generated; predictable subset). */
function renderMd(src) {
  const inline = t => t
    .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const lines = esc(src).split("\n");
  let html = "", i = 0;
  const flushP = buf => buf.length ? `<p>${inline(buf.join(" "))}</p>` : "";
  let para = [];
  while (i < lines.length) {
    const l = lines[i];
    if (/^```/.test(l)) {                                  // fence
      html += flushP(para); para = [];
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++; html += `<pre><code>${buf.join("\n")}</code></pre>`; continue;
    }
    if (/^#{1,6}\s/.test(l)) { html += flushP(para); para = [];
      const n = l.match(/^#+/)[0].length;
      html += `<h${n}>${inline(l.slice(n + 1))}</h${n}>`; i++; continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(l)) { html += flushP(para); para = []; html += "<hr>"; i++; continue; }
    if (/^\s*&gt;\s?/.test(l)) { html += flushP(para); para = [];
      const buf = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) buf.push(lines[i++].replace(/^\s*&gt;\s?/, ""));
      html += `<blockquote>${inline(buf.join(" "))}</blockquote>`; continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      html += flushP(para); para = [];
      const cells = r => r.trim().replace(/^\||\|$/g, "").split("|").map(c => inline(c.trim()));
      html += `<table><thead><tr>${cells(l).map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i]))
        html += `<tr>${cells(lines[i++]).map(c => `<td>${c}</td>`).join("")}</tr>`;
      html += "</tbody></table>"; continue;
    }
    if (/^\s*([-*+]|\d+\.)\s+/.test(l)) {
      html += flushP(para); para = [];
      const ordered = /^\s*\d+\./.test(l);
      html += ordered ? "<ol>" : "<ul>";
      while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i]))
        html += `<li>${inline(lines[i++].replace(/^\s*([-*+]|\d+\.)\s+/, ""))}</li>`;
      html += ordered ? "</ol>" : "</ul>"; continue;
    }
    if (!l.trim()) { html += flushP(para); para = []; i++; continue; }
    para.push(l.trim()); i++;
  }
  return html + flushP(para);
}

/* ── polling manager: timers for the active view (cleared on navigate) ── */
let _pollTimers = [];
function clearPoll() {
  for (const t of _pollTimers) { clearInterval(t); clearTimeout(t); }
  _pollTimers = [];
}
function addPoll(fn, ms) { if (fn) _pollTimers.push(setInterval(fn, ms)); }
function setPoll(fn, ms) { clearPoll(); addPoll(fn, ms); }

/* ── shared components ─────────────────────────────────────────────────── */
function stageBar(detail) {
  const stages = detail.stage_order || ["pass_a", "pass_b", "synthesis", "gap_check", "final_report"];
  const st = (detail.state && detail.state.stages) || {};
  const running = detail.status === "running";
  let firstPending = stages.findIndex(s => !(st[s] && st[s].status === "done"));
  const seg = stages.map((s, idx) => {
    const info = st[s] || {};
    let cls = "";
    if (info.status === "done") cls = "done";
    else if (info.status === "failed") cls = "failed";
    else if (running && idx === firstPending) cls = "running";
    return `<div class="stage ${cls}" title="${esc(s)}: ${esc(info.status || "pending")}${info.elapsed_s != null ? ` (${fmtDur(info.elapsed_s)})` : ""}"></div>`;
  }).join("");
  return `<div class="stages">${seg}</div>
          <div class="stage-labels">${stages.map(s => `<span>${esc(s.replace("_", " "))}</span>`).join("")}</div>`;
}

function jobRow(j) {
  return `<tr class="clickable" onclick="location.hash='#/jobs/${esc(j.job_id)}'">
    <td class="mono">${esc(j.job_id)}</td><td>${esc(j.kind)}</td>
    <td class="mono">${esc(j.objective_id)}</td><td>${statusPill(j.status)}</td>
    <td class="mono">${since(j.created_at)}</td>
    <td class="mono">${j.exit_code ?? ""}</td></tr>`;
}

function serviceDot(s) {
  if (!s.available) return `<span class="pill err" title="${esc(s.error || "")}">${esc(s.unit)} ?</span>`;
  const on = s.active_state === "active";
  const en = (s.enabled || "").startsWith("enabled");
  const cls = on ? "ok" : en ? "warn" : "dim";
  return `<span class="pill ${cls}" title="enabled: ${esc(s.enabled || "—")}, next: ${esc(s.next_run || "—")}">
    ${esc(s.unit.replace(".service", "").replace("vtech-", ""))}: ${esc(s.active_state)}${s.unit.endsWith(".timer") ? "/" + esc(s.enabled || "") : ""}</span>`;
}

/* ── views ─────────────────────────────────────────────────────────────── */

async function viewDashboard() {
  const o = await api("/api/overview");
  const pin = $("#active-objective");
  if (pin) pin.textContent = `config: ${o.active_objective || "—"}`;
  if ($("#mlflow-link")) $("#mlflow-link").href = o.mlflow_url;
  const vm = o.vm || {};
  view.innerHTML = `
  <div class="grid cols-3">
    <div class="card"><h3>Active objective</h3>
      <div class="big">${esc(o.active_objective || "—")}</div>
      <div class="muted">${o.objectives.length} objective(s) defined · pinned in config.yaml</div></div>
    <div class="card"><h3>Running processes</h3>
      <div class="big">${o.running_jobs.length}</div>
      <div class="muted">${o.running_jobs.map(j => esc(j.label || j.kind)).join(", ") || "console-launched jobs idle"}</div></div>
    <div class="card"><h3>VM</h3>
      <div class="muted mono">${esc(vm.summary || vm.error || "—")}</div></div>
  </div>
  <div class="section-title">Services</div>
  <div class="card">${o.services.map(serviceDot).join(" ")}</div>
  <div class="grid cols-2" style="margin-top:16px">
    <div class="card"><h3>Recent scoping runs <a href="#/scoping" style="float:right;font-size:12px">all →</a></h3>
      <table><thead><tr><th>objective</th><th>run</th><th>status</th><th>stages</th></tr></thead><tbody>
      ${o.scoping_runs.map(r => `<tr class="clickable" onclick="location.hash='#/scoping/${esc(r.objective_id)}/${esc(r.run_ts)}'">
        <td class="mono">${esc(r.objective_id)}</td><td class="mono">${fmtTs(r.run_ts)}</td>
        <td>${statusPill(r.status)}${r.dry_run ? ' <span class="pill dim">dry</span>' : ""}</td>
        <td class="mono">${r.stages_done}/${r.stages_total}</td></tr>`).join("") || '<tr><td colspan="4" class="empty">none yet</td></tr>'}
      </tbody></table></div>
    <div class="card"><h3>Recent model-selection runs <a href="#/optimize" style="float:right;font-size:12px">all →</a></h3>
      <table><thead><tr><th>objective</th><th>run</th><th>gate</th><th>champion</th></tr></thead><tbody>
      ${o.search_runs.map(r => `<tr class="clickable" onclick="location.hash='#/optimize/search/${esc(r.objective_id)}/${esc(r.run_ts)}'">
        <td class="mono">${esc(r.objective_id)}</td><td class="mono">${esc(r.run_ts)}</td>
        <td>${statusPill(r.gate_status || "—")}</td><td class="mono">${esc(r.champion_id || "—")}</td></tr>`).join("") || '<tr><td colspan="4" class="empty">none yet</td></tr>'}
      </tbody></table></div>
  </div>
  <div class="section-title">Recent jobs</div>
  <div class="card"><table><thead><tr><th>job</th><th>kind</th><th>objective</th><th>status</th><th>started</th><th>exit</th></tr></thead>
    <tbody>${o.recent_jobs.map(jobRow).join("") || '<tr><td colspan="6" class="empty">no jobs launched yet</td></tr>'}</tbody></table></div>`;
  setPoll(() => viewDashboard().catch(() => {}), 5000);
}

/* ── objectives ────────────────────────────────────────────────────────── */

window._rdToggle = async (oid, on) => {
  try {
    const r = await post(`/api/objectives/${encodeURIComponent(oid)}/rd-enable`, {enabled: !!on});
    toast(r.rd_enabled
      ? `RD enabled for ${oid}` + (r.has_registry_latest ? "" : " (needs registry LATEST to run)")
      : `RD disabled for ${oid}`, "ok");
    // re-render current view if still on objectives
    if (location.hash.startsWith("#/objectives")) {
      if (location.hash === "#/objectives" || location.hash === "#/objectives/") viewObjectives();
      else viewObjectiveDetail(oid);
    }
  } catch (e) { toast(e.message || String(e), "err"); }
};

async function viewObjectives() {
  const objs = await api("/api/objectives");
  view.innerHTML = `
  <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:14px">
    <div class="muted">Trade objectives are the declarative root of every workflow (ADR-018):
      they define labels + KPI and never encode a model choice. Immutable once created.
      <b>RD</b> = Research Director — only registry LATEST models with RD on are cycled.</div>
    <div class="fixed"><button class="primary" onclick="location.hash='#/objectives/new'">＋ New objective</button></div>
  </div>
  <div class="card"><table>
    <thead><tr><th>id</th><th>symbol</th><th>scheme</th><th class="num">move</th><th class="num">mae</th><th>kpi</th>
      <th>registry</th><th>RD</th><th></th></tr></thead>
    <tbody>${objs.map(o => `<tr class="clickable" onclick="location.hash='#/objectives/${esc(o.objective_id)}'">
      <td class="mono">${esc(o.objective_id)} ${o.active ? '<span class="pill ok">active</span>' : ""}
        ${o.error ? `<span class="pill err" title="${esc(o.error)}">YAML</span>` : ""}</td>
      <td class="mono">${esc(o.symbol || "—")}</td><td class="mono">${esc(o.scheme || "—")}</td>
      <td class="num">${esc(o.move_threshold ?? "—")}</td><td class="num">${esc(o.mae_tolerance ?? "—")}</td>
      <td class="mono">${esc(o.kpi_primary || "—")}</td>
      <td>${o.has_registry_latest
        ? `<span class="pill ok" title="${esc(o.registry_version || "")}">LATEST</span>`
        : `<span class="pill dim">none</span>`}</td>
      <td onclick="event.stopPropagation()">
        <label class="muted" style="display:flex;align-items:center;gap:6px;cursor:pointer">
          <input type="checkbox" ${o.rd_enabled ? "checked" : ""}
            onchange="window._rdToggle('${esc(o.objective_id)}', this.checked)"
            ${o.has_registry_latest ? "" : "title=\"Promote a model first (toggle still saved)\""} />
          ${o.rd_enabled ? "on" : "off"}
        </label>
      </td>
      <td onclick="event.stopPropagation()">
        <button class="small" onclick="location.hash='#/scoping?objective=${esc(o.objective_id)}'">scope →</button>
        <button class="small" onclick="location.hash='#/objectives/${esc(o.objective_id)}/test'">test →</button>
      </td>
    </tr>`).join("")}</tbody></table></div>`;
}

const objSubTabs = (id, active) => `
  <div class="tabs" style="margin-top:10px">
    <button class="${active === "home" ? "active" : ""}" onclick="location.hash='#/objectives/${esc(id)}'">Home</button>
    <button class="${active === "model" ? "active" : ""}" onclick="location.hash='#/objectives/${esc(id)}/model'">Registered Model</button>
    <button class="${active === "test" ? "active" : ""}" onclick="location.hash='#/objectives/${esc(id)}/test'">Custom Test</button>
  </div>`;

async function viewObjectiveDetail(id) {
  const o = await api(`/api/objectives/${encodeURIComponent(id)}`);
  const regPill = o.has_registry_latest
    ? `<span class="pill ok">registry ${esc(o.registry_version || "LATEST")}</span>`
    : `<span class="pill dim">no registry LATEST</span>`;
  const rdPill = o.rd_enabled ? `<span class="pill ok">RD on</span>` : `<span class="pill dim">RD off</span>`;
  const yamlErr = o.parse_error
    ? `<div class="empty">⚠ YAML parse error — grid fields are blank until this file is valid.
      <pre class="code" style="margin-top:8px;white-space:pre-wrap">${esc(o.parse_error)}</pre></div>`
    : "";
  view.innerHTML = `
  <a class="back" href="#/objectives">← objectives</a>
  <h2 style="margin:4px 0 2px">${esc(id)} ${o.active ? '<span class="pill ok">active</span>' : ""} ${regPill} ${rdPill}</h2>
  <div class="muted">${esc((o.parsed && o.parsed.description || "").trim())}</div>
  ${yamlErr}
  ${objSubTabs(id, "home")}
  <div class="card" style="margin-bottom:12px">
    <h3 style="margin-top:0">Research Director</h3>
    <div class="muted" style="margin-bottom:8px">When the RD timer fires, each <b>enabled</b> objective that has a
      <b>registry LATEST</b> model gets one improvement cycle. Staged-only models are not used.</div>
    <label style="display:flex;align-items:center;gap:10px;cursor:pointer">
      <input type="checkbox" ${o.rd_enabled ? "checked" : ""}
        onchange="window._rdToggle('${esc(id)}', this.checked)" />
      <span>Enable RD experimentation for this objective</span>
    </label>
    ${o.has_registry_latest
      ? `<div class="muted" style="margin-top:8px">Baseline: <span class="mono">${esc(o.registry_path || o.registry_version || "")}</span></div>`
      : `<div class="muted" style="margin-top:8px">Promote a staged model first:
          <span class="mono">python -m src.serving.promote --objective ${esc(id)} --version …</span></div>`}
  </div>
  <div class="row" style="margin-bottom:16px">
    <div class="fixed"><button class="primary" onclick="location.hash='#/scoping?objective=${esc(id)}'">⌕ Run scoping</button></div>
    <div class="fixed"><button onclick="location.hash='#/optimize?objective=${esc(id)}'">⚙ Model search</button></div>
    <div class="fixed"><button onclick="location.hash='#/objectives/${esc(id)}/test'">⚗ Custom test</button></div>
  </div>
  <div class="card"><h3>Definition (config/objectives/${esc(id)}.yaml)</h3>
  <pre class="code">${esc(o.yaml)}</pre></div>`;
}

/* ── Registered Model sub-page ─────────────────────────────────────────── */

async function viewObjectiveModel(id) {
  const BLOTTER_CAP = 500;
  view.classList.add("wide");       // blotter is wide — use the full viewport
  let lastB = null;                 // blotter payload {rows, stats, meta, status}
  let saving = false;
  let pollMs = 0;

  const apiBase = `/api/objectives/${encodeURIComponent(id)}/model`;

  window._rmSave = async () => {
    if (saving) return;
    const g = i => document.getElementById(i);
    const body = {
      order_type: g("rm-order").value,
      execution: g("rm-exec").value,
      trading_start: g("rm-start").value,
      trading_end: g("rm-end").value,
    };
    const tau = parseFloat(g("rm-tau").value);
    if (Number.isFinite(tau)) body.tau = tau;
    const mp = parseInt(g("rm-maxpos").value, 10);
    if (Number.isFinite(mp)) body.max_positions = mp;
    saving = true;
    document.activeElement?.blur?.();
    paint();
    try {
      await post(`${apiBase}/settings`, body);
      toast("Settings saved — rebuilding blotter…", "ok", 4000);
      await loadAll(true);
    } catch (e) { toast(e.message, "err"); }
    finally { saving = false; paint(); }
  };

  window._rmResetSession = async () => {
    const eff = lastB && lastB.status && lastB.status.effective;
    if (!eff) return;
    document.activeElement?.blur?.();
    try {
      await post(`${apiBase}/settings`, {
        trading_start: eff.default_trading_start,
        trading_end: eff.default_trading_end,
      });
      toast("Session times back to model defaults", "ok", 4000);
      await loadAll(true);
    } catch (e) { toast(e.message, "err"); }
  };

  const checkChip = (ok, label) =>
    `<span class="pill ${ok ? "ok" : "dim"}" title="${esc(label)}">${ok ? "✓" : "○"} ${esc(label)}</span>`;

  const statsHtml = () => {
    const s = (lastB && lastB.stats) || {};
    const r = (lastB && lastB.replay) || {};
    const summ = r.summary || {};
    const meta = r.meta || {};
    const pv = Number(meta.point_value || 20);
    const sign = v => v == null ? "" : v >= 0 ? "pos" : "neg";
    const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
    const ptsUsd = (pts, d = 2) => fmtPtsUsd(pts, pv, d);
    const tile = (lbl, val, cls, sub) =>
      `<div class="tile"><div class="lbl">${lbl}</div><div class="val ${cls || ""}">${val}</div>${
        sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
    const capNote = meta.max_positions
      ? `cap ${meta.max_positions} · ${fmtInt(meta.n_skipped_slots || 0)} blocked`
      : "";
    return `
    <div class="tiles" style="margin-bottom:12px">
      <div class="tile"><div class="lbl">emitted signals today</div>
        <div class="val">${fmtInt(s.n_signals ?? 0)}
          <span class="val2"><span class="pos">▲ ${fmtInt(s.n_up ?? 0)}</span> · <span class="neg">▼ ${fmtInt(s.n_down ?? 0)}</span></span></div>
        <div class="sub">${capNote}</div></div>
      ${summ.n_trades != null ? `
      ${tile("Replay Net P/L", ptsUsd(summ.net_pnl), sign(summ.net_pnl), `${fmtInt(summ.n_trades)} trades`)}
      ${tile("Win rate", pct(summ.win_rate), "",
        `${fmtInt(summ.n_wins)} W · ${fmtInt(summ.n_losses)} L`)}
      ${tile("Expectancy", ptsUsd(summ.expectancy, 2), sign(summ.expectancy), "per trade")}
      ${tile("Profit factor", fmtNum(summ.profit_factor, 2), summ.profit_factor >= 1 ? "pos" : "neg",
        `avg ${fmtPtsUsd(summ.avg_win, pv, 1)} / ${fmtPtsUsd(summ.avg_loss, pv, 1)}`)}
      ${tile("Max drawdown", ptsUsd(summ.max_drawdown), "neg", "peak-to-trough")}
      ${tile("Avg holding", `${fmtInt(summ.avg_holding_bars)} bars`, "",
        `~${fmtInt(summ.avg_holding_seconds)}s`)}
      ` : ""}
    </div>`;
  };

  const panelHtml = st => {
    const eff = st.effective || {};
    const ovr = eff.session_override_active
      ? `<span class="pill warn">today-only override — resets next session</span>
         <button type="button" class="small" onclick="window._rmResetSession()">reset to default</button>`
      : `<span class="pill dim">model default</span>`;
    return `
    <div class="card" id="rm-panel" style="margin-bottom:12px">
      <h3 style="margin-top:0">Execution controls</h3>
      <div class="row" style="gap:14px;align-items:flex-end;flex-wrap:wrap">
        <div><label for="rm-order">stop/target order type</label>
          <select id="rm-order">
            ${["MKT", "LIMIT"].map(v => `<option${eff.order_type === v ? " selected" : ""}>${v}</option>`).join("")}
          </select></div>
        <div><label for="rm-exec">execution</label>
          <select id="rm-exec" title="OCO: first-touch on live TBBO prints. OnClose: barriers checked at 5s closes only, order sent at the triggering close.">
            <option value="OCO"${eff.execution === "OCO" ? " selected" : ""}>OCO (tick first-touch)</option>
            <option value="ONCLOSE"${eff.execution === "ONCLOSE" ? " selected" : ""}>OnClose (5s close trigger)</option>
          </select></div>
        <div><label for="rm-tau">confidence gate τ${eff.tau_is_default ? " (objective default)" : ""}</label>
          <input id="rm-tau" type="number" step="0.01" min="0" max="1" value="${esc(fmtNum(eff.tau, 2))}" style="width:90px"></div>
        <div><label for="rm-maxpos">max positions</label>
          <input id="rm-maxpos" type="number" step="1" min="1" max="50" value="${esc(String(eff.max_positions ?? 1))}" style="width:80px"></div>
        <div><label for="rm-start">session start</label>
          <input id="rm-start" type="time" value="${esc(eff.trading_start || "")}" style="width:110px"></div>
        <div><label for="rm-end">session end</label>
          <input id="rm-end" type="time" value="${esc(eff.trading_end || "")}" style="width:110px"></div>
        <div class="fixed"><button type="button" class="primary" ${saving ? "disabled" : ""}
          onclick="window._rmSave()">${saving ? "Saving…" : "Save"}</button></div>
      </div>
      <div class="muted" style="margin-top:8px">
        Session window ${ovr}
        · default ${esc(eff.default_trading_start || "—")}–${esc(eff.default_trading_end || "—")} ${esc((st.session && st.session.timezone) || "")}
        ${eff.order_type === "MKT" && eff.execution === "ONCLOSE"
          ? `<br>MKT + OnClose: a 5s candle close triggers the exit; the order fills at the executable market bid/ask, never at an exact target or stop price.`
          : ""}
      </div>
    </div>`;
  };

  const blotterHtml = () => {
    const rows = (lastB && lastB.rows) || [];
    const meta = (lastB && lastB.meta) || {};
    const replay = (lastB && lastB.replay) || {};
    const rmeta = replay.meta || {};
    const trades = replay.trades || [];
    const pv = Number(rmeta.point_value || 20);
    const byTs = new Map();
    for (const t of trades) if (t.signal_ts) byTs.set(String(t.signal_ts), t);
    const shown = rows.slice(0, BLOTTER_CAP);
    const body = shown.map(r => {
      const dcls = r.dir === "up" ? "pos" : "neg";
      const arrow = r.dir === "up" ? "▲" : "▼";
      const t = byTs.get(String(r.ts));
      const pnlCls = t && t.pnl_points != null ? (t.pnl_points >= 0 ? "pos" : "neg") : "muted";
      const cumCls = t && t.cum_pnl != null ? (t.cum_pnl >= 0 ? "pos" : "neg") : "muted";
      const reason = t ? esc(t.exit_reason || "—") : `<span class="muted">blocked (cap)</span>`;
      return `<tr>
        <td class="mono">${fmtTsMs(r.ts)}</td>
        <td class="${dcls}">${arrow} ${esc(String(r.dir || "").toUpperCase())}</td>
        <td class="mono">${r.conf != null ? fmtNum(r.conf, 3) : "—"}</td>
        <td class="mono">${t && t.entry_px != null ? fmtNum(t.entry_px, 2)
          : (r.entry_px != null ? fmtNum(r.entry_px, 2) : "—")}</td>
        <td class="mono">${t && t.exit_px != null ? fmtNum(t.exit_px, 2) : "—"}</td>
        <td class="mono">${t ? fmtTsMs(t.exit_ts) : "—"}</td>
        <td class="mono">${reason}</td>
        <td class="num">${t && t.holding_bars ? fmtInt(t.holding_bars) : "—"}</td>
        <td class="num">${r.n_present != null ? fmtInt(r.n_present) : "—"}</td>
        <td class="num">${r.n_total != null ? fmtInt(r.n_total) : "—"}</td>
        <td class="num ${pnlCls}">${t && t.pnl_points != null ? fmtPtsUsd(t.pnl_points, pv) : "—"}</td>
        <td class="num ${cumCls}">${t && t.cum_pnl != null ? fmtPtsUsd(t.cum_pnl, pv) : "—"}</td>
      </tr>`;
    }).join("");
    const barrierNote = rmeta.target_points != null
      ? ` · target +${rmeta.target_points} / stop −${rmeta.stop_points} · ${rmeta.horizon_s}s horizon · MID close first-touch · max ${rmeta.max_positions}`
      : "";
    return `
    <div class="card">
      <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap">
        <h3 style="margin:0">Emitted signals + replay · ${esc(String(meta.session_date || "—"))} (${TZ_ABBR})</h3>
        <div class="muted mono">source: ${esc(meta.source || "signals.jsonl")} · ${fmtInt(rows.length)} records${barrierNote}</div>
      </div>
      ${meta.note ? `<div class="muted" style="margin-top:6px">⚠ ${esc(meta.note)}</div>` : ""}
      ${rmeta.note ? `<div class="muted" style="margin-top:6px">${esc(rmeta.note)}</div>` : ""}
      <div id="rm-scroll" style="max-height:60vh;overflow:auto;margin-top:8px">
        <table>
          <thead><tr>
            <th>emitted (${TZ_ABBR})</th><th>dir</th><th>conf</th>
            <th class="num">entry px</th><th class="num">exit px</th>
            <th>exit (${TZ_ABBR})</th><th>reason</th>
            <th class="num">hold (bars)</th>
            <th class="num">feat avail</th><th class="num">feat total</th>
            <th class="num">P/L (pt · $)</th>
            <th class="num">cum P/L (pt · $)</th>
          </tr></thead>
          <tbody>${body || `<tr><td colspan="12" class="empty">No emitted signals recorded for this session day.</td></tr>`}</tbody>
        </table>
      </div>
      ${rows.length > BLOTTER_CAP ? `<div class="muted" style="margin-top:6px">showing ${BLOTTER_CAP} of ${rows.length} rows</div>` : ""}
      <div class="muted" style="margin-top:6px">Replay uses real emitted signals + 5s bar-close first-touch at MID. Positions are capped by the max_positions setting above; blocked signals show as "blocked (cap)" and contribute nothing to P/L.</div>
    </div>`;
  };

  const executionHtml = () => {
    const payload = (lastB && lastB.executions) || {};
    const rows = payload.rows || [];
    const stats = payload.stats || {};
    const meta = payload.meta || {};
    const body = rows.slice(0, BLOTTER_CAP).map(r => {
      const side = r.side === "long" ? "▲ LONG" : r.side === "short" ? "▼ SHORT" : "—";
      const sideCls = r.side === "long" ? "pos" : r.side === "short" ? "neg" : "muted";
      const pnlCls = (r.pnl_usd ?? 0) >= 0 ? "pos" : "neg";
      return `<tr>
        <td class="mono">${fmtTsMs(r.entry_ts || r.last_event_at)}</td>
        <td class="${sideCls}">${side}</td>
        <td class="mono">${esc(r.instrument || "—")}</td>
        <td class="mono">${fmtInt(r.quantity || 0)}</td>
        <td>${statusPill(r.state || "—")}</td>
        <td class="mono">${r.entry_px != null ? fmtNum(r.entry_px, 2) : "—"}</td>
        <td class="mono">${r.exit_px != null ? fmtNum(r.exit_px, 2) : "—"}</td>
        <td class="mono">${esc(r.exit_reason || r.reason || "—")}</td>
        <td class="mono ${r.pnl_usd != null ? pnlCls : "muted"}">${r.pnl_usd != null ? fmtUsd(r.pnl_usd, 2) : "—"}</td>
        <td class="mono muted">${esc(r.command_id || "—")}</td>
      </tr>`;
    }).join("");
    return `<div class="card" style="margin-top:12px">
      <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap">
        <h3 style="margin:0">Broker executions · ${esc(String(meta.session_date || "—"))}</h3>
        <div class="muted mono">NinjaTrader events · ${fmtInt(stats.n_orders || 0)} orders ·
          ${fmtInt(stats.n_open || 0)} open · ${fmtInt(stats.n_rejected || 0)} rejected ·
          net ${fmtUsd(stats.net_pnl_usd || 0, 2)}</div>
      </div>
      ${meta.bridge_heartbeat_at ? `<div class="muted" style="margin-top:6px">bridge heartbeat ${since(meta.bridge_heartbeat_at)}</div>` : ""}
      ${meta.note ? `<div class="muted" style="margin-top:6px">${esc(meta.note)}</div>` : ""}
      <div style="max-height:45vh;overflow:auto;margin-top:8px"><table>
        <thead><tr><th>broker time (${TZ_ABBR})</th><th>side</th><th>instrument</th><th>qty</th><th>state</th>
          <th>entry</th><th>exit</th><th>reason</th><th>P/L</th><th>command</th></tr></thead>
        <tbody>${body || `<tr><td colspan="10" class="empty">No NinjaTrader execution events recorded for this session day.</td></tr>`}</tbody>
      </table></div>
      <div class="muted" style="margin-top:6px">This table is reconstructed only from NinjaTrader acknowledgements and fills; model signals never masquerade as trades.</div>
    </div>`;
  };

  const paint = () => {
    const panel = document.getElementById("rm-panel");
    if (panel && panel.contains(document.activeElement)) return;    // don't clobber edits
    const st = (lastB && lastB.status) || {};
    const model = st.model;
    const checks = st.active_checks || {};
    const activePill = st.active
      ? `<span class="pill ok">● ACTIVE — generating signals</span>`
      : `<span class="pill dim">○ inactive</span>`;
    const banner = st.market_closed_today
      ? `<div class="banner warn">Market is closed today (${esc(st.market_closed_reason || "non-trading day")}) —
          no live signals; blotter shows the last session day.</div>`
      : "";
    const modelLine = model
      ? `<span class="mono">${esc(model.candidate_id || model.name || "")}</span>
         <span class="pill ${model.registered === "registry_latest" ? "ok" : "info"}">${
           model.registered === "registry_latest" ? "registry LATEST" : esc(model.registered || model.source || "")}</span>
         <span class="muted mono" style="font-size:11px">${esc(model.path || "")}</span>`
      : `<span class="pill err">no registered model</span>
         <span class="muted">promote or stage a serving bundle for this objective</span>`;
    const keep = document.getElementById("rm-scroll")?.scrollTop || 0;

    view.innerHTML = `
    <a class="back" href="#/objectives">← objectives</a>
    <h2 style="margin:4px 0 2px">${esc(id)} ${activePill}</h2>
    <div class="row" style="gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0">${modelLine}</div>
    <div class="row" style="gap:6px;flex-wrap:wrap;margin:4px 0">
      ${checkChip(checks.registered_model, "model")}
      ${checkChip(checks.trading_day, "trading day")}
      ${checkChip(checks.in_session, `in session ${(st.session && st.session.trading_start) || ""}–${(st.session && st.session.trading_end) || ""}`)}
      ${checkChip(checks.feed_live, "feed live")}
    </div>
    ${objSubTabs(id, "model")}
    ${banner}
    ${statsHtml()}
    ${panelHtml(st)}
    ${blotterHtml()}
    ${executionHtml()}`;

    const cur = document.getElementById("rm-scroll");
    if (cur) cur.scrollTop = keep;
  };

  const loadAll = async (force = false) => {
    try {
      lastB = await api(`${apiBase}/blotter${force ? "?force=1" : ""}`);
    } catch (e) {
      lastB = {rows: [], stats: {}, meta: {note: String(e.message || e)}, status: lastB && lastB.status};
    }
    paint();
    const want = lastB.status && lastB.status.active ? 3000 : 15000;
    if (want !== pollMs) { pollMs = want; setPoll(() => loadAll(false), pollMs); }
  };

  return loadAll(false);
}

function kpiRowHtml(block, idx) {
  return `<div class="row kpi-row" data-block="${block}">
    <div><label>metric</label><select data-f="metric">
      ${["precision_at_tau", "recall", "directional_f1", "f1"].map(m => `<option>${m}</option>`).join("")}
    </select></div>
    <div><label>direction</label><select data-f="direction"><option value="">—</option><option>up</option><option>down</option></select></div>
    <div><label>tau</label><input data-f="tau" type="number" step="0.05" min="0" max="1" placeholder="e.g. 0.8"></div>
    <div><label>min</label><input data-f="min" type="number" step="0.01" placeholder="e.g. 0.85"></div>
    <div class="fixed"><button class="small danger" onclick="this.closest('.kpi-row').remove()">✕</button></div>
  </div>`;
}

function collectObjectiveForm() {
  const val = id => { const el = $("#" + id); return el ? el.value.trim() : ""; };
  const num = id => { const v = val(id); return v === "" ? undefined : Number(v); };
  const rows = block => [...document.querySelectorAll(`.kpi-row[data-block="${block}"]`)].map(r => {
    const g = f => r.querySelector(`[data-f="${f}"]`).value.trim();
    const c = {metric: g("metric")};
    if (g("direction")) c.direction = g("direction");
    if (g("tau") !== "") c.tau = Number(g("tau"));
    if (g("min") !== "") c.min = Number(g("min"));
    return c;
  });
  const obj = {
    objective_id: val("f-id"),
    schema_version: "v2",
    description: val("f-desc"),
    "traded products": [{symbol: val("f-symbol").toUpperCase(), security_type: val("f-sectype"),
                         asset_class: val("f-assetclass"), role: "primary"}],
    strategy_keywords: val("f-keywords") ? val("f-keywords").split(",").map(s => s.trim()).filter(Boolean) : [],
    horizon: {max_trade_horizon: val("f-horizon"), holding_max: val("f-holding")},
    session: {training_start: val("f-trainstart"), training_end: val("f-trainend"),
              trading_start: val("f-tstart"), trading_end: val("f-tend"),
              timezone: val("f-tz")},
    labels: {scheme: val("f-scheme"), move_threshold: num("f-move"), mae_tolerance: num("f-mae"),
             unit: val("f-unit")},
    kpi: {primary: val("f-kpi-primary"), requirement: {any: rows("req")}, target: {all: rows("tgt")}},
    eligible_model_families: val("f-families") ? val("f-families").split(",").map(s => s.trim()).filter(Boolean) : [],
    eval: {hold_out: null, walk_forward_folds: num("f-folds") ?? 3},
  };
  return obj;
}

async function viewObjectiveNew() {
  view.innerHTML = `
  <a class="back" href="#/objectives">← objectives</a>
  <h2 style="margin:4px 0 14px">New trade objective</h2>
  <div class="grid" id="obj-form" style="grid-template-columns: minmax(420px,1fr) minmax(320px,480px)">
  <div>
    <fieldset><legend>Identity</legend>
      <label>objective id (stable, lowercase: instrument + threshold)</label>
      <input id="f-id" placeholder="e.g. nq-30pt-tbm" pattern="[a-z0-9][a-z0-9_-]*">
      <label>description (plain-English trading goal)</label>
      <textarea id="f-desc" rows="3" placeholder="Detect intraday ±30 point NQ futures moves within a 15-minute forward window…"></textarea>
    </fieldset>
    <fieldset><legend>Primary product</legend>
      <div class="row">
        <div><label>symbol</label><input id="f-symbol" value="NQ"></div>
        <div><label>security type</label><input id="f-sectype" value="futures"></div>
        <div><label>asset class</label><input id="f-assetclass" value="equity_futures"></div>
      </div>
      <label>strategy keywords (comma-separated)</label>
      <input id="f-keywords" value="momentum, intraday, directional">
    </fieldset>
    <fieldset><legend>Horizon &amp; session</legend>
      <div class="row">
        <div><label>max trade horizon</label><input id="f-horizon" value="10m"></div>
        <div><label>holding max</label><input id="f-holding" value="intraday"></div>
      </div>
      <div class="row">
        <div><label>training start</label><input id="f-trainstart" value="08:00"></div>
        <div><label>training end</label><input id="f-trainend" value="14:45"></div>
      </div>
      <div class="row">
        <div><label>trading start</label><input id="f-tstart" value="08:30"></div>
        <div><label>trading end</label><input id="f-tend" value="14:45"></div>
        <div><label>timezone</label><input id="f-tz" value="America/Chicago"></div>
      </div>
    </fieldset>
    <fieldset><legend>Labels — what counts as a signal</legend>
      <label>scheme</label>
      <select id="f-scheme">
        <option value="detection_2">detection_2 — order-agnostic window filter (peak move ≥ threshold AND adverse &lt; MAE)</option>
        <option value="triple_barrier">triple_barrier — first-touch: profit barrier before stop barrier before time barrier</option>
      </select>
      <div class="row">
        <div><label>move threshold</label><input id="f-move" type="number" value="50"></div>
        <div><label>MAE tolerance (stop)</label><input id="f-mae" type="number" value="6"></div>
        <div><label>unit</label><input id="f-unit" value="points"></div>
      </div>
    </fieldset>
    <fieldset><legend>KPI</legend>
      <label>primary metric</label>
      <select id="f-kpi-primary"><option>precision_at_tau</option><option>directional_f1</option><option>net_expectancy</option></select>
      <label style="margin-top:14px">requirement — champion-gate bar (passes if <b>ANY</b> condition holds)</label>
      <div id="req-rows"></div>
      <button class="small" onclick="$('#req-rows').insertAdjacentHTML('beforeend', kpiRowHtml('req'))">＋ condition</button>
      <label style="margin-top:14px">target — program goal (achieved when <b>ALL</b> hold)</label>
      <div id="tgt-rows"></div>
      <button class="small" onclick="$('#tgt-rows').insertAdjacentHTML('beforeend', kpiRowHtml('tgt'))">＋ condition</button>
    </fieldset>
    <fieldset><legend>Search &amp; evaluation</legend>
      <label>eligible model families (comma-separated; empty = all catalogued)</label>
      <input id="f-families" placeholder="e.g. xgboost, lightgbm, attention_lstm">
      <label>walk-forward folds</label><input id="f-folds" type="number" value="3" min="1">
    </fieldset>
    <div class="row" style="margin-top:14px">
      <div class="fixed"><button onclick="objValidate()">Validate</button></div>
      <div class="fixed"><button class="primary" onclick="objCreate()">Create objective</button></div>
    </div>
    <div id="obj-errors" style="margin-top:12px"></div>
  </div>
  <div>
    <div class="card" style="position:sticky;top:70px"><h3>Live preview</h3>
      <pre class="code" id="obj-preview" style="max-height:70vh"></pre></div>
  </div>
  </div>`;
  $("#req-rows").insertAdjacentHTML("beforeend", kpiRowHtml("req"));
  $("#tgt-rows").insertAdjacentHTML("beforeend", kpiRowHtml("tgt"));
  const refresh = () => { const el = $("#obj-preview"); if (el) el.textContent = JSON.stringify(collectObjectiveForm(), null, 2); };
  const form = $("#obj-form");                           // listeners live on this node (not the persistent #view) so they're GC'd when the next route replaces it
  form.addEventListener("input", refresh);
  form.addEventListener("click", () => setTimeout(refresh, 0));
  refresh();
}

window.objValidate = async () => {
  try {
    const r = await post("/api/objectives/validate", {objective: collectObjectiveForm()});
    $("#obj-errors").innerHTML = r.valid
      ? '<span class="pill ok">schema valid ✓</span>'
      : r.errors.map(e => `<div class="pill err" style="display:block;margin:4px 0">${esc(e)}</div>`).join("");
  } catch (e) { toast(e.message, "err"); }
};

window.objCreate = async () => {
  try {
    const obj = collectObjectiveForm();
    const v = await post("/api/objectives/validate", {objective: obj});
    if (!v.valid) { window.objValidate(); toast("fix schema errors first", "err"); return; }
    const r = await post("/api/objectives", {objective: obj});
    toast(`objective '${r.objective_id}' created`, "ok");
    location.hash = `#/objectives/${r.objective_id}`;
  } catch (e) { toast(e.message, "err", 9000); }
};

/* ── scoping ───────────────────────────────────────────────────────────── */

async function viewScoping(params) {
  const [objs, runs] = await Promise.all([api("/api/objectives"), api("/api/scoping/runs")]);
  const presel = params.get("objective") || "";
  view.innerHTML = `
  <div class="card"><h3>Launch scoping run (ADR-025 — dual-perspective research loop)</h3>
    <div class="muted">Digests the objective through external research (T1), internal evidence (T2/T3),
      adversarial critique, synthesis and a deterministic gap check → final research report + draft candidate spec.
      A live (non-dry) run spends LLM budget (typically $8–15, cap below).</div>
    <div class="row" style="margin-top:10px">
      <div><label>objective</label><select id="sc-obj">
        ${objs.map(o => `<option ${o.objective_id === presel ? "selected" : ""}>${esc(o.objective_id)}</option>`).join("")}
      </select></div>
      <div class="fixed"><label>budget cap (USD)</label><input id="sc-budget" type="number" value="30" min="0" max="500" style="width:110px"></div>
      <div class="fixed"><label class="checkbox" style="margin-top:26px"><input type="checkbox" id="sc-dry" checked> dry-run (no LLM spend)</label></div>
      <div class="fixed"><button class="primary" style="margin-top:18px" onclick="scopingLaunch()">▶ Launch</button></div>
    </div></div>
  <div class="section-title">Runs</div>
  <div class="card" id="sc-runs">${scopingRunsTable(runs)}</div>`;
  setPoll(async () => {
    try { $("#sc-runs").innerHTML = scopingRunsTable(await api("/api/scoping/runs")); } catch {}
  }, 5000);
}

function scopingRunsTable(runs) {
  return `<table><thead><tr><th>objective</th><th>run</th><th>status</th><th style="width:220px">stages</th><th>elapsed</th><th>report</th></tr></thead>
  <tbody>${runs.map(r => `<tr class="clickable" onclick="location.hash='#/scoping/${esc(r.objective_id)}/${esc(r.run_ts)}'">
    <td class="mono">${esc(r.objective_id)}</td><td class="mono">${fmtTs(r.run_ts)}</td>
    <td>${statusPill(r.status)}${r.dry_run ? ' <span class="pill dim">dry</span>' : ""}</td>
    <td><div class="stages">${Array.from({length: r.stages_total}, (_, i) =>
      `<div class="stage ${i < r.stages_done ? "done" : (r.status === "failed" && i === r.stages_done ? "failed" : r.status === "running" && i === r.stages_done ? "running" : "")}"></div>`).join("")}</div></td>
    <td class="mono">${fmtDur(r.elapsed_s)}</td>
    <td>${r.has_report ? "📄" : ""}</td></tr>`).join("") || '<tr><td colspan="6" class="empty">no scoping runs yet</td></tr>'}
  </tbody></table>`;
}

window.scopingLaunch = async () => {
  try {
    const r = await post("/api/scoping/launch", {
      objective_id: $("#sc-obj").value, dry_run: $("#sc-dry").checked,
      budget: $("#sc-budget").value === "" ? null : Number($("#sc-budget").value)});
    toast(`scoping launched: ${r.job.job_id}`, "ok");
    location.hash = `#/jobs/${r.job.job_id}`;
  } catch (e) { toast(e.message, "err", 9000); }
};

const _ART_TABS = [
  ["final_research_report.md", "Report", "md"],
  ["scoping_plan.json", "Plan", "json"],
  ["data_gap_report.json", "Gap report", "json"],
  ["manual_candidates.draft.yaml", "Candidates draft", "code"],
  ["objective_patch.suggested.yaml", "Objective patch", "code"],
];

async function viewScopingRun(obj, ts, params) {
  const d = await api(`/api/scoping/runs/${encodeURIComponent(obj)}/${encodeURIComponent(ts)}`);
  const artNames = new Set(d.artifacts.map(a => a.name));
  const tabs = _ART_TABS.filter(([n]) => artNames.has(n));
  const active = params.get("tab") || (tabs[0] ? tabs[0][0] : "");
  view.innerHTML = `
  <a class="back" href="#/scoping">← scoping runs</a>
  <h2 style="margin:4px 0 4px" class="mono">${esc(obj)} / ${fmtTs(ts)}</h2>
  <div style="margin:6px 0 12px">${statusPill(d.status)} ${d.state && d.state.dry_run ? '<span class="pill dim">dry-run</span>' : ""}
    <span class="muted">· ${d.stages_done}/${d.stages_total} stages · ${fmtDur(d.state && d.state.elapsed_s)}</span></div>
  <div class="card" style="max-width:760px">${stageBar(d)}</div>
  <div class="tabs">${tabs.map(([n, label]) =>
    `<button class="${n === active ? "active" : ""}" onclick="location.hash='#/scoping/${esc(obj)}/${esc(ts)}?tab=${encodeURIComponent(n)}'">${esc(label)}</button>`).join("")}
    <button class="${active === "_all" ? "active" : ""}" onclick="location.hash='#/scoping/${esc(obj)}/${esc(ts)}?tab=_all'">All artifacts (${d.artifacts.length})</button>
  </div>
  <div id="art-body"><div class="empty">loading…</div></div>`;
  if (active === "_all") {
    $("#art-body").innerHTML = `<div class="card"><table>
      <thead><tr><th>artifact</th><th class="num">size</th></tr></thead><tbody>
      ${d.artifacts.map(a => `<tr class="clickable" onclick="location.hash='#/scoping/${esc(obj)}/${esc(ts)}?tab=${encodeURIComponent(a.name)}'">
        <td class="mono">${esc(a.name)}</td><td class="num">${(a.size / 1024).toFixed(1)} KB</td></tr>`).join("")}
      </tbody></table></div>`;
  } else if (active) {
    try {
      const art = await api(`/api/scoping/runs/${encodeURIComponent(obj)}/${encodeURIComponent(ts)}/artifact/${encodeURIComponent(active)}`);
      const kind = (_ART_TABS.find(([n]) => n === active) || [])[2]
        || (active.endsWith(".md") ? "md" : active.endsWith(".json") ? "json" : "code");
      let body;
      if (kind === "md") body = `<div class="md card">${renderMd(art.content)}</div>`;
      else if (kind === "json") { let t = art.content; try { t = JSON.stringify(JSON.parse(art.content), null, 2); } catch {} body = `<pre class="code">${esc(t)}</pre>`; }
      else body = `<pre class="code">${esc(art.content)}</pre>`;
      $("#art-body").innerHTML = body + (art.truncated ? '<div class="muted">⚠ truncated</div>' : "");
    } catch (e) { $("#art-body").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
  } else {
    $("#art-body").innerHTML = '<div class="empty">no artifacts yet — stage outputs appear here as the run progresses</div>';
  }
  if (d.status === "running") setPoll(() => viewScopingRun(obj, ts, params).catch(() => {}), 6000);
}

/* ── optimization: research director + model search ────────────────────── */

async function viewOptimize(params) {
  const [dir, objs, runs, families] = await Promise.all([
    api("/api/director/status"), api("/api/objectives"),
    api("/api/search/runs"), api("/api/search/families")]);
  const preselObj = params.get("objective") || "";
  const t = dir.timer, enabled = (t.enabled || "").startsWith("enabled");
  const db = dir.db || {};
  view.innerHTML = `
  <div class="grid cols-2">
    <div class="card"><h3>Model optimization schedule — Research Director</h3>
      <div class="muted">Automated research loop: two independent expert agents assess the latest model,
        design + run experiments on a timer, and stage improvements behind a human promotion gate.</div>
      <div style="margin:12px 0">
        ${t.available ? `${statusPill(enabled ? "succeeded" : "warn").replace(/>[^<]+</, `>${enabled ? "scheduled" : "disabled"}<`)}
          <span class="pill dim">${esc(t.active_state)}/${esc(t.sub_state)}</span>` : `<span class="pill err">timer unavailable: ${esc(t.error || "")}</span>`}
      </div>
      <dl class="kv">
        <dt>cadence</dt><dd>${esc(dir.schedule_hint)}</dd>
        <dt>next run</dt><dd>${esc(t.next_run || "—")}</dd>
        <dt>last run</dt><dd>${esc(t.last_run || "—")}</dd>
        ${dir.config && dir.config.available ? `
        <dt>LLM cap / cycle</dt><dd>$${esc(dir.config.llm_usd_cap_cycle ?? "—")}</dd>
        <dt>per-agent cap</dt><dd>$${esc(dir.config.per_agent_usd_cap ?? "—")}</dd>` : ""}
      </dl>
      <div class="row" style="margin-top:14px">
        <div class="fixed"><button class="${enabled ? "" : "primary"}" onclick="dirTimer('${enabled ? "disable" : "enable"}')">
          ${enabled ? "⏸ Disable schedule" : "▶ Enable schedule"}</button></div>
        <div class="fixed"><button onclick="dirRunCycle(false)">⚡ Run one cycle now</button></div>
        <div class="fixed"><button onclick="dirRunCycle(true)">Dry-run cycle</button></div>
      </div></div>
    <div class="card"><h3>Research funnel &amp; spend</h3>
      ${db.available ? `
        <table><thead><tr><th>state</th><th class="num">n</th><th class="num">7d</th></tr></thead><tbody>
          ${(db.funnel || []).slice(0, 8).map(f => `<tr><td class="mono">${esc(f.state)}</td>
            <td class="num">${f.n}</td><td class="num">${f.n_7d}</td></tr>`).join("") || '<tr><td colspan="3" class="empty">journal empty</td></tr>'}
        </tbody></table>
        <div class="section-title">LLM spend (7d)</div>
        ${(db.llm_calls_7d || []).map(l => `<span class="pill dim">${esc(l.agent)}: ${l.calls} calls / $${l.usd.toFixed(2)}</span>`).join(" ") || '<span class="muted">none</span>'}
        ${(db.awaiting_promotion || []).length ? `<div class="section-title">⚠ Awaiting promotion ack</div>
          ${db.awaiting_promotion.map(p => `<div class="mono" style="font-size:12px">${esc(p.experiment_id)} — ${esc(p.title)}</div>`).join("")}` : ""}
      ` : `<div class="empty">research journal DB unavailable<br><span class="mono" style="font-size:11px">${esc(db.error || "")}</span></div>`}
    </div>
  </div>

  <div class="section-title">Model-selection search (Phase 23 — full-fidelity walk-forward)</div>
  <div class="card">
    <div class="row">
      <div><label>objective</label><select id="se-obj">
        ${objs.map(o => `<option ${o.objective_id === preselObj ? "selected" : ""}>${esc(o.objective_id)}</option>`).join("")}</select></div>
      <div class="fixed"><label>tune trials / candidate</label><input id="se-trials" type="number" value="0" min="0" max="200" style="width:110px"></div>
      <div class="fixed"><label class="checkbox" style="margin-top:26px"><input type="checkbox" id="se-dry"> dry-run</label></div>
      <div class="fixed"><label class="checkbox" style="margin-top:26px"><input type="checkbox" id="se-noens"> skip ensembles</label></div>
      <div class="fixed"><button class="primary" style="margin-top:18px" onclick="searchLaunch()">▶ Launch search</button></div>
    </div>
    <label style="margin-top:12px">families (none selected = all catalogued)</label>
    <div id="se-fams">${families.map(f =>
      `<span class="chip" data-fam="${esc(f.name)}" onclick="this.classList.toggle('sel')">${esc(f.name)}${f.compute_class === "gpu" ? " ⚡gpu" : ""}</span>`).join("")}</div>
  </div>
  <div class="section-title">Search runs</div>
  <div class="card"><table>
    <thead><tr><th>objective</th><th>run</th><th class="num">candidates</th><th class="num">ok</th><th>gate</th><th>champion</th><th class="num">precision</th></tr></thead>
    <tbody>${runs.map(r => `<tr class="clickable" onclick="location.hash='#/optimize/search/${esc(r.objective_id)}/${esc(r.run_ts)}'">
      <td class="mono">${esc(r.objective_id)}</td><td class="mono">${esc(r.run_ts)}</td>
      <td class="num">${r.n_candidates ?? "—"}</td><td class="num">${r.n_ok ?? "—"}</td>
      <td>${statusPill(r.gate_status || "—")}</td><td class="mono">${esc(r.champion_id || "—")}</td>
      <td class="num">${fmtNum(r.champion_precision)}</td></tr>`).join("") || '<tr><td colspan="7" class="empty">no search runs yet</td></tr>'}
    </tbody></table></div>`;
}

window.dirTimer = async action => {
  try {
    await post("/api/director/timer", {action});
    toast(`research-director schedule ${action}d`, "ok");
    route();
  } catch (e) { toast(e.message, "err", 12000); }
};

window.dirRunCycle = async dry => {
  try {
    const r = await post("/api/director/run-cycle", {dry_run: dry});
    toast(`director cycle launched: ${r.job.job_id}`, "ok");
    location.hash = `#/jobs/${r.job.job_id}`;
  } catch (e) { toast(e.message, "err", 9000); }
};

window.searchLaunch = async () => {
  try {
    const fams = [...document.querySelectorAll("#se-fams .chip.sel")].map(c => c.dataset.fam);
    const r = await post("/api/search/launch", {
      objective_id: $("#se-obj").value, families: fams,
      dry_run: $("#se-dry").checked, tune_trials: Number($("#se-trials").value) || 0,
      no_ensemble: $("#se-noens").checked});
    toast(`model search launched: ${r.job.job_id}`, "ok");
    location.hash = `#/jobs/${r.job.job_id}`;
  } catch (e) { toast(e.message, "err", 9000); }
};

let _lbSort = {key: "precision_at_tau", dir: -1};
async function viewSearchRun(obj, ts) {
  const d = await api(`/api/search/runs/${encodeURIComponent(obj)}/${encodeURIComponent(ts)}`);
  const gate = d.gate_result || {};
  const score = c => (c.score && c.score[_lbSort.key]) ?? (c[_lbSort.key] ?? null);
  const cands = [...(d.candidates || [])].sort((a, b) => ((score(b) ?? -1e9) - (score(a) ?? -1e9)) * -_lbSort.dir);
  const cols = [["precision_at_tau", "P@τ"], ["directional_f1", "dir F1"], ["f1_up", "F1↑"], ["f1_down", "F1↓"], ["n_signals", "signals"]];
  view.innerHTML = `
  <a class="back" href="#/optimize">← optimization</a>
  <h2 style="margin:4px 0 4px" class="mono">${esc(obj)} / ${esc(ts)}</h2>
  <div style="margin:6px 0 14px">${statusPill(gate.status || "—")}
    ${gate.champion_id ? `<span class="pill ok">champion: ${esc(gate.champion_id)}</span>` : ""}
    ${gate.champion_meets_target ? '<span class="pill ok">meets target 🎯</span>' : ""}
    <span class="muted">· ${d.n_ok}/${d.n_candidates} candidates ok · <a href="${esc(d.mlflow_url)}" target="_blank" rel="noopener">open in MLflow ↗</a></span></div>
  ${(gate.reasons || []).length ? `<div class="card" style="margin-bottom:14px"><h3>Gate reasons</h3>
    ${gate.reasons.map(r => `<div class="muted">• ${esc(typeof r === "string" ? r : JSON.stringify(r))}</div>`).join("")}</div>` : ""}
  <div class="card"><h3>Leaderboard</h3><table>
    <thead><tr><th>candidate</th><th>family</th><th>status</th>
      ${cols.map(([k, label]) => `<th class="num sortable" onclick="lbSort('${k}','${esc(obj)}','${esc(ts)}')">${label}${_lbSort.key === k ? (_lbSort.dir < 0 ? " ▾" : " ▴") : ""}</th>`).join("")}
      <th class="num">elapsed</th><th>target</th></tr></thead>
    <tbody>${cands.map(c => `<tr>
      <td class="mono">${esc(c.candidate_id)}${gate.champion_id === c.candidate_id ? " 🏆" : ""}</td>
      <td class="mono">${esc(c.family)}</td><td>${statusPill(c.status === "ok" ? "succeeded" : c.status)}</td>
      ${cols.map(([k]) => `<td class="num">${k === "n_signals" ? fmtInt((c.score || {})[k]) : fmtNum((c.score || {})[k])}</td>`).join("")}
      <td class="num">${fmtDur(c.elapsed_s)}</td>
      <td>${c.meets_target ? "🎯" : ""}</td></tr>`).join("")}
    </tbody></table></div>`;
}
window.lbSort = (key, obj, ts) => {
  _lbSort = {key, dir: _lbSort.key === key ? -_lbSort.dir : -1};
  viewSearchRun(obj, ts).catch(e => toast(e.message, "err"));
};

/* ── jobs ──────────────────────────────────────────────────────────────── */

async function viewJobs() {
  const jobs = await api("/api/jobs");
  view.innerHTML = `
  <div class="muted" style="margin-bottom:12px">Processes launched from this console. Jobs survive a console
    restart (log + PID reattach); cancel sends SIGTERM to the process group.</div>
  <div class="card"><table>
    <thead><tr><th>job</th><th>kind</th><th>objective</th><th>status</th><th>started</th><th>exit</th></tr></thead>
    <tbody>${jobs.map(jobRow).join("") || '<tr><td colspan="6" class="empty">no jobs yet — launch scoping, a model search, a custom test, or a director cycle</td></tr>'}</tbody>
  </table></div>`;
  setPoll(() => viewJobs().catch(() => {}), 4000);
}

let _logOffset = 0, _logJob = null;
async function viewJobDetail(id) {
  const j = await api(`/api/jobs/${encodeURIComponent(id)}`);
  if (_logJob !== id) { _logOffset = 0; _logJob = id; }
  view.innerHTML = `
  <a class="back" href="#/jobs">← jobs</a>
  <h2 style="margin:4px 0 4px" class="mono">${esc(j.job_id)}</h2>
  <div style="margin:6px 0 12px">${statusPill(j.status)}
    <span class="pill dim">${esc(j.kind)}</span> <span class="pill dim">${esc(j.objective_id)}</span>
    ${j.reattached ? '<span class="pill warn">reattached</span>' : ""}
    <span class="muted">· pid ${j.pid ?? "—"} · started ${since(j.created_at)}${j.ended_at ? ` · ended ${since(j.ended_at)}` : ""}${j.exit_code != null ? ` · exit ${j.exit_code}` : ""}</span></div>
  <div class="mono muted" style="margin-bottom:10px;font-size:12px">$ ${esc((j.argv || []).join(" "))}</div>
  ${j.status === "running" ? '<button class="danger" onclick="jobCancel(\'' + esc(j.job_id) + '\')">✕ Cancel job</button>' : ""}
  ${j.kind === "scoping" && j.status !== "running" ? `<button onclick="location.hash='#/scoping'">view scoping runs →</button>` : ""}
  ${j.kind === "search" && j.status !== "running" ? `<button onclick="location.hash='#/optimize'">view search runs →</button>` : ""}
  ${j.kind === "custom_test" && j.status !== "running" ? `<button onclick="location.hash='#/optimize'">view search runs →</button>` : ""}
  <div class="section-title">Log</div>
  <pre class="log" id="job-log"></pre>`;
  const logEl = $("#job-log");
  _logOffset = 0;
  const pull = async () => {
    try {
      const chunk = await api(`/api/jobs/${encodeURIComponent(id)}/log?offset=${_logOffset}`);
      if (chunk.content) {
        const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 60;
        logEl.textContent += chunk.content;
        _logOffset = chunk.offset;
        if (atBottom) logEl.scrollTop = logEl.scrollHeight;
      }
    } catch {}
  };
  await pull();
  if (j.status === "running") {
    setPoll(async () => {
      await pull();
      try {
        const cur = await api(`/api/jobs/${encodeURIComponent(id)}`);
        if (cur.status !== "running") {
          clearPoll();                                   // stop THIS poller: the terminal re-render below re-arms nothing, so the interval would otherwise re-toast + re-render forever
          toast(`job ${cur.status}`, cur.status === "succeeded" ? "ok" : "err");
          viewJobDetail(id);
        }
      } catch {}
    }, 2500);
  }
}
window.jobCancel = async id => {
  try { await post(`/api/jobs/${encodeURIComponent(id)}/cancel`, {}); toast("cancel requested (SIGTERM)", "ok"); }
  catch (e) { toast(e.message, "err"); }
};

/* ── backtest (staged Live weights via src.serving) ──────────────────────── */

const _btTs = t => t ? (toCT(t) ?? String(t).replace("T", " ").replace("+00:00", "").replace(/\.\d+/, "")) : "—";
const _pts = (v, d = 2) => fmtPts(v, d);
const extCard = (id) => (window._btExtCatalog || []).find(m => m.id === id) || null;

function chicagoToday() {
  const p = Object.fromEntries(_CT_FMT.formatToParts(new Date()).map(x => [x.type, x.value]));
  return `${p.year}-${p.month}-${p.day}`;
}

/** Resolve backtest start/end for a range preset. Dates are ISO YYYY-MM-DD. */
function backtestRangeBounds(rangeMode, spanStart, spanEnd, today) {
  const asOfToday = today || chicagoToday();
  const asOf = (spanEnd && spanEnd < asOfToday) ? spanEnd : asOfToday;
  const clamp = iso => {
    let v = iso;
    if (spanStart && v < spanStart) v = spanStart;
    if (spanEnd && v > spanEnd) v = spanEnd;
    return v;
  };
  if (rangeMode === "all") {
    if (!spanStart && !spanEnd) return null;
    return {start: spanStart || asOf, end: spanEnd || asOf};
  }
  if (rangeMode === "mtd" || rangeMode === "ytd") {
    const [y, mo] = asOf.split("-");
    const start = rangeMode === "mtd" ? `${y}-${mo}-01` : `${y}-01-01`;
    return {start: clamp(start), end: clamp(asOf)};
  }
  const days = Number(rangeMode);
  if (!Number.isFinite(days) || days <= 0) return null;
  const end = clamp(asOf);
  const d = new Date(`${end}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() - days);
  return {start: clamp(d.toISOString().slice(0, 10)), end};
}

async function viewBacktest(params) {
  view.classList.add("wide-bt");    // 1.25× the default max-width so the blotter fits
  const taus = [];
  for (let pct = 5; pct <= 95; pct += 5) taus.push(pct / 100);
  const posOpts = [1, 2, 3, 5, 10, 25];
  /* Switch: left OCO, right on close. checked = on close. */
  let btOnClose = true;
  if (localStorage.getItem("bt_on_close") != null) {
    btOnClose = readBoolPref("bt_on_close", []);
  } else if (localStorage.getItem("bt_oco") != null) {
    // legacy: bt_oco true meant OCO path
    btOnClose = !readBoolPref("bt_oco", []);
  }
  /* price mode: true = MID, false = MKT — applies to BOTH OCO and on close */
  let btMid = readBoolPref("bt_price_mid", []);
  if (localStorage.getItem("bt_price_mid") == null) btMid = true;

  const paintHelp = () => {
    const el = $("#bt-mode-help");
    if (!el) return;
    const px = btMid ? "MID" : "MKT";
    if (!btOnClose) {
      el.innerHTML = `
        <span class="pill ok">OCO</span>
        Bracket on continuous <b>TBBO</b> (trade / bid / ask). Entry at signal + entry latency
        using <b>${px}</b>; stop/target first-touch; exit fill at hit + exit latency (${px}).
        MKT = buy ask / sell bid. Open positions caps concurrency.`;
    } else {
      el.innerHTML = `
        <span class="pill ok">on close</span>
        Same as the original path: signal at a 5s close; stop/target checked only on
        later 5s closes. <b>MID + 0 latency</b> = candle <span class="mono">_close</span>
        (classic; PnL is actual close-to-close, not fixed ±50/±6).
        <b>Latency &gt; 0</b> or <b>MKT</b> = TBBO mid/bid/ask at close + latency.
        Open positions caps concurrency.`;
    }
    const adjRow = $("#bt-adj-row");
    if (adjRow) adjRow.style.display = "flex"; // MAE/target for both modes
    const mktL = $("#bt-mkt-lab"), midL = $("#bt-mid-lab");
    if (mktL) mktL.classList.toggle("on", !btMid);
    if (midL) midL.classList.toggle("on", btMid);
    const ocoL = $("#bt-oco-lab"), closeL = $("#bt-close-lab");
    if (ocoL) ocoL.classList.toggle("on", !btOnClose);
    if (closeL) closeL.classList.toggle("on", btOnClose);
  };

  view.innerHTML = `
  <div class="card"><h3>Backtest</h3>
    <div id="bt-mode-help" class="muted" style="margin-bottom:12px;line-height:1.45"></div>
    <div class="row" style="margin-top:4px">
      <div class="fixed">
        <label>backtest type</label>
        <select id="bt-plane" style="width:150px">
          <option value="ml">ML model</option>
          <option value="ext">EXT Model</option>
        </select>
      </div>
      <div id="bt-ml-row" style="flex:2.5;min-width:220px;display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end">
      <div style="flex:2.5;min-width:220px"><label>staged / registry model</label>
        <select id="bt-model" style="width:100%"></select></div>
      <div class="fixed" style="align-self:flex-end">
        <button type="button" id="bt-browse-btn" onclick="window._btToggleBrowse()">Browse…</button>
      </div>
      <div class="fixed" id="bt-tau-wrap"><label>minimum p_reach</label>
        <div class="bt-tau-ctrl">
          <label class="checkbox" style="margin:0" title="When checked, enter only if p_reach is at least this value. Unchecked = no Stage 3 gate.">
            <input type="checkbox" id="bt-tau-on" checked> enable</label>
          <input id="bt-tau" type="number" min="0" max="1" step="0.01" value="0.75" style="width:90px">
        </div></div>
      <div class="fixed" id="bt-mv-wrap" title="Cascade Stage 1 gate (target_scores.parquet): p_move ≥ this literal score. Ignored for non-cascade models."><label>minimum p_move — cascade only</label>
        <div class="bt-tau-ctrl">
          <label class="checkbox" style="margin:0" title="When checked, enter only if p_move is at least this value.">
            <input type="checkbox" id="bt-mv-on" checked> enable</label>
          <input id="bt-mv" type="number" min="0" max="1" step="0.01" value="0.50" style="width:90px">
        </div></div>
      <div class="fixed" id="bt-mintgt-wrap" title="Cascade target-size gate: skip trades whose target distance in R units is below this floor. Ignored for non-cascade models."><label>min tgt distance (R) — cascade only</label>
        <input id="bt-mintgt" type="number" min="0" max="10" step="0.05" value="0.25" style="width:90px">
      </div>
      <div class="fixed" id="bt-pos-wrap"><label>open positions</label>
        <select id="bt-pos" style="width:120px">${posOpts.map(p => `<option>${p}</option>`).join("")}</select></div>
      </div>
    </div>
    <div id="bt-ext-panel" hidden style="margin-top:8px">
      <div id="bt-ext-legs"></div>
      <button type="button" class="bt-ext-add" id="bt-ext-add" title="Add external model">＋</button>
      <div id="bt-ext-book" class="bt-ext-book">
        <h3>Backtest Parameters <span id="bt-book-dirty" class="ext-dirty" hidden>unsaved</span></h3>
        <p class="muted">Session is the trading window (session end = last exit). Trade start/end gate new entries.
          Book candle timeframe is display + ATR / PT / SL bars; each stacked engine has its own candle timeframe.
          Profit target may be points, × ATR, or × stop.</p>
        <div class="bt-book-fields" id="bt-ext-book-grid"></div>
        <div class="muted" id="bt-book-saved" style="margin-top:10px"></div>
        <div class="ext-actions">
          <button type="button" class="primary" id="bt-book-save">Save as defaults</button>
          <button type="button" id="bt-book-reset">Reset to factory</button>
        </div>
      </div>
    </div>
    <div class="row" id="bt-adj-row" style="margin-top:6px;display:none;justify-content:flex-end">
      <div class="fixed"><label>MAE adjustment (pts) — + widens stop</label>
        <input id="bt-mae-adj" type="number" step="0.25" value="0" style="width:130px"
          title="Added to objective mae_tolerance (e.g. 6 + 2 → stop 8 pts)"></div>
      <div class="fixed"><label>profit target adjustment (pts) — + raises target</label>
        <input id="bt-tgt-adj" type="number" step="0.25" value="0" style="width:130px"
          title="Added to objective move_threshold (e.g. 50 + 5 → target 55 pts)"></div>
    </div>
    <div class="bt-row-dates">
      <div class="bt-dates-left">
        <div class="fixed"><label>date range</label>
          <select id="bt-range" style="width:200px">
            <option value="mtd" selected>MTD (month to date)</option>
            <option value="ytd">YTD (year to date)</option>
            <option value="all">Full available span</option>
            <option value="30">Last ~1 month</option>
            <option value="90">Last ~3 months</option>
            <option value="custom">Custom…</option></select></div>
        <div class="fixed"><label>start</label><input id="bt-start" type="date" style="width:150px"></div>
        <div class="fixed"><label>end</label><input id="bt-end" type="date" style="width:150px"></div>
      </div>
      <div class="bt-run-panel" id="bt-run-panel">
        <div class="bt-toggle-row">
          <button type="button" class="primary" onclick="btRun()">▶ Run backtest</button>
          <span id="bt-ml-exec" style="display:flex;align-items:center;gap:10px">
          <span id="bt-oco-lab" class="bt-side-label" title="Continuous TBBO stop/target bracket">OCO</span>
          ${bareSwitchHtml("bt-on-close", btOnClose, "window._btToggleOnClose(this.checked)",
            "Left/off = OCO (continuous TBBO bracket). Right/on = on close (5s candle close + latency)")}
          <span id="bt-close-lab" class="bt-side-label" title="5s candle close + latency">on close</span>
          </span>
        </div>
        <div id="bt-exec-opts">
          <div class="bt-toggle-row" style="margin-top:2px">
            <span id="bt-mkt-lab" class="bt-side-label" title="Buy on ask / sell on bid">MKT</span>
            ${bareSwitchHtml("bt-mid", btMid, "window._btToggleMid(this.checked)",
              "Applies to OCO and on close. OFF = MKT; ON = MID")}
            <span id="bt-mid-lab" class="bt-side-label" title="Mid price (bid+ask)/2">MID</span>
          </div>
          <div class="bt-toggle-row" style="margin-top:8px;flex-wrap:wrap">
            <div class="bt-lat-field">
              <input id="bt-entry-lat" type="number" min="0" step="1" value="0"
                title="ms after signal/close before entry price is taken from TBBO">
              <span class="bt-lat-lab">entry latency (ms)</span>
            </div>
            <div class="bt-lat-field">
              <input id="bt-exit-lat" type="number" min="0" step="1" value="0"
                title="ms after exit trigger/close before exit price is taken from TBBO">
              <span class="bt-lat-lab">exit latency (ms)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div id="bt-modelmeta" class="muted" style="margin-top:10px;line-height:1.5"></div>
    <div id="bt-browse" class="bt-browse" hidden></div>
  </div>
  <div id="bt-results"><div class="empty">configure and run a backtest</div></div>`;

  window._btModels = [];
  window._btPicked = null;   // browse selection overrides the dropdown
  window._btBrowseOpen = false;
  window._btBrowsePath = null;
  window._btOnClose = btOnClose;
  window._btMid = btMid;
  window._btPlane = "ml";
  window._btExtCatalog = [];
  window._btExtBook = null;
  window._btExtStack = [{ model: "", hardened: "", timeframe: "" }];
  window._btExtBookOverride = null;

  const paintOcoControls = () => {
    const sw = $("#bt-on-close");
    if (sw) sw.checked = btOnClose;
    const mid = $("#bt-mid");
    if (mid) mid.checked = btMid;
    paintHelp();
  };
  window._btToggleOnClose = (on) => {
    btOnClose = !!on;
    window._btOnClose = btOnClose;
    writeBoolPref("bt_on_close", btOnClose);
    writeBoolPref("bt_oco", !btOnClose); // keep legacy in sync
    paintOcoControls();
  };
  window._btToggleMid = (on) => {
    btMid = !!on;
    window._btMid = btMid;
    writeBoolPref("bt_price_mid", btMid);
    paintOcoControls();
  };

  const bookTimeframe = () => {
    try {
      const fromForm = extReadParams(extBookSpec(), "bt-b-");
      if (fromForm.timeframe) return fromForm.timeframe;
    } catch { /* ignore */ }
    const book = window._btExtBookOverride
      || (window._btExtBook && (window._btExtBook.defaults || window._btExtBook.factory_params))
      || {};
    return book.timeframe || "1min";
  };
  const candleTfs = () => {
    const fromBook = (window._btExtBook && window._btExtBook.timeframes) || [];
    return fromBook.length ? fromBook : ["1min", "5min", "15min", "1hour", "1day"];
  };

  const paintExtLegs = () => {
    const host = $("#bt-ext-legs");
    if (!host) return;
    const catalog = window._btExtCatalog || [];
    if (!catalog.length) {
      host.innerHTML = `<div class="muted">No external models in <span class="mono">config/external_models/</span></div>`;
      return;
    }
    const stack = window._btExtStack || [];
    const bookTf = bookTimeframe();
    const tfs = candleTfs();
    host.innerHTML = stack.map((leg, i) => {
      const modelOpts = catalog.map(m =>
        `<option value="${esc(m.id)}"${m.id === leg.model ? " selected" : ""}>${esc(m.title || m.id)}</option>`).join("");
      const card = extCard(leg.model);
      const snaps = (card && card.hardened) || [];
      const snapOpts = [`<option value=""${!leg.hardened ? " selected" : ""}>DEFAULT</option>`]
        .concat(snaps.map(h =>
          `<option value="${esc(h.name)}"${leg.hardened === h.name ? " selected" : ""}>${esc(h.name)}</option>`))
        .join("");
      const snapHidden = !leg.model;
      const tfVal = leg.timeframe || bookTf;
      const tfOpts = tfs.map(tf =>
        `<option value="${esc(tf)}"${tf === tfVal ? " selected" : ""}>${esc(tf)}</option>`).join("");
      const remove = stack.length > 1
        ? `<button type="button" class="small bt-ext-remove" data-rm="${i}" title="Remove">−</button>`
        : "";
      return `<div class="bt-ext-leg" data-leg="${i}">
        <div><label>${i === 0 ? "EXT model" : "stacked model"}</label>
          <select class="bt-ext-model" data-leg="${i}">
            <option value="">(pick a model)</option>${modelOpts}
          </select></div>
        <div><label>candle timeframe</label>
          <select class="bt-ext-tf" data-leg="${i}">${tfOpts}</select></div>
        <div${snapHidden ? " hidden" : ""}><label>snapshot</label>
          <select class="bt-ext-snap" data-leg="${i}">${snapOpts}</select></div>
        ${remove}
      </div>`;
    }).join("");
    host.querySelectorAll(".bt-ext-model").forEach(sel => {
      sel.addEventListener("change", () => {
        const i = Number(sel.dataset.leg);
        const picked = extCard(sel.value);
        if (picked && picked.standalone) {
          window._btExtStack = [{ model: sel.value, hardened: "", timeframe: "1min" }];
          paintExtLegs();
          paintExtBook();
          syncMeta();
          return;
        }
        const prev = window._btExtStack[i] || {};
        window._btExtStack[i] = {
          model: sel.value, hardened: "",
          timeframe: prev.timeframe || bookTimeframe(),
        };
        paintExtLegs();
        paintExtBook();
        syncMeta();
      });
    });
    host.querySelectorAll(".bt-ext-tf").forEach(sel => {
      sel.addEventListener("change", () => {
        const i = Number(sel.dataset.leg);
        if (!window._btExtStack[i]) return;
        window._btExtStack[i].timeframe = sel.value || "";
        syncMeta();
      });
    });
    host.querySelectorAll(".bt-ext-snap").forEach(sel => {
      sel.addEventListener("change", () => {
        const i = Number(sel.dataset.leg);
        if (!window._btExtStack[i]) return;
        window._btExtStack[i].hardened = sel.value || "";
        syncMeta();
      });
    });
    host.querySelectorAll("[data-rm]").forEach(btn => {
      btn.addEventListener("click", () => {
        const i = Number(btn.dataset.rm);
        window._btExtStack.splice(i, 1);
        if (!window._btExtStack.length) {
          window._btExtStack = [{ model: "", hardened: "", timeframe: bookTimeframe() }];
        }
        paintExtLegs();
        syncMeta();
      });
    });
  };

  const extBookSpec = () => (window._btExtBook && window._btExtBook.param_spec) || [];
  const extBookBase = () => (window._btExtBook && (window._btExtBook.defaults || window._btExtBook.factory_params)) || {};

  const extStandaloneCard = () => {
    const stack = (window._btExtStack || []).filter(leg => leg.model);
    if (stack.length !== 1) return null;
    const card = extCard(stack[0].model);
    return card && card.standalone ? card : null;
  };
  const extActiveSpec = () => {
    const card = extStandaloneCard();
    return card ? (card.param_spec || []) : extBookSpec();
  };
  const extActiveBase = () => {
    const card = extStandaloneCard();
    return card ? (card.defaults || card.factory_params || {}) : extBookBase();
  };
  const extActivePrefix = () => extStandaloneCard() ? "bt-m-" : "bt-b-";

  const markBtBookDirty = () => {
    const el = $("#bt-book-dirty");
    if (!el) return;
    try {
      el.hidden = JSON.stringify(extReadParams(extActiveSpec(), extActivePrefix()))
        === JSON.stringify(extActiveBase());
    } catch { el.hidden = true; }
  };

  const paintExtBook = (vals) => {
    const grid = $("#bt-ext-book-grid");
    if (!grid) return;
    const standalone = extStandaloneCard();
    const spec = extActiveSpec();
    const prefix = extActivePrefix();
    const src = vals || (standalone ? extActiveBase()
      : (window._btExtBookOverride || extBookBase()));
    grid.innerHTML = btBookFieldsHtml(spec, src, prefix);
    const heading = document.querySelector("#bt-ext-book h3");
    if (heading) heading.childNodes[0].nodeValue = standalone
      ? `${standalone.title || standalone.id} Parameters `
      : "Backtest Parameters ";
    const intro = document.querySelector("#bt-ext-book p.muted");
    if (intro) intro.textContent = standalone
      ? "This standalone engine owns its session, entries, exits and risk controls below. Stacking is disabled."
      : "Session is the trading window (session end = last exit). Trade start/end gate new entries. Book candle timeframe is display + ATR / PT / SL bars; each stacked engine has its own candle timeframe. Profit target may be points, × ATR, or × stop.";
    const saved = standalone ? standalone.defaults_saved_at
      : (window._btExtBook && window._btExtBook.saved_at);
    const note = $("#bt-book-saved");
    if (note) {
      note.textContent = saved
        ? `Defaults saved ${saved}.`
        : (standalone ? `Factory ${standalone.title || standalone.id} values. Persist model defaults on External Models.`
          : "Defaults are factory Pine values.");
    }
    spec.forEach(s => {
      const el = document.getElementById(`${prefix}${s.key}`);
      if (!el) return;
      el.addEventListener("input", markBtBookDirty);
      el.addEventListener("change", markBtBookDirty);
    });
    grid.querySelectorAll(".bt-unit-toggle button").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.unitFor;
        const unit = btn.dataset.unit;
        const hidden = document.getElementById(`${prefix}${key}`);
        if (hidden) hidden.value = unit;
        grid.querySelectorAll(`[data-unit-for="${key}"]`).forEach(b => {
          b.classList.toggle("on", b.dataset.unit === unit);
        });
        markBtBookDirty();
      });
    });
    markBtBookDirty();
  };

  const paintPlane = () => {
    const extOn = window._btPlane === "ext";
    const mlRow = $("#bt-ml-row");
    const extPanel = $("#bt-ext-panel");
    const plane = $("#bt-plane");
    if (plane) plane.value = window._btPlane;
    if (mlRow) mlRow.style.display = extOn ? "none" : "flex";
    if (extPanel) extPanel.hidden = !extOn;
    const browse = $("#bt-browse-btn");
    if (browse) browse.style.display = extOn ? "none" : "";
    if (extOn) {
      window._btBrowseOpen = false;
      const box = $("#bt-browse");
      if (box) { box.hidden = true; box.innerHTML = ""; }
      paintExtLegs();
      const grid = $("#bt-ext-book-grid");
      if (grid && !grid.childElementCount) paintExtBook();
    }
    syncMeta();
  };

  const paintModelSelect = (selectedIdx) => {
    const models = window._btModels || [];
    const sel = $("#bt-model");
    if (!sel) return;
    if (!models.length) {
      sel.innerHTML = `<option value="">(none)</option>`;
      return;
    }
    sel.innerHTML = models.map((m, i) =>
      `<option value="${i}"${m.backtestable === false ? " disabled" : ""}>${esc(
        m.label || m.run_label || m.model)}${m.backtestable === false ? "  ⚠" : ""}</option>`).join("");
    const ix = Number.isInteger(selectedIdx) && selectedIdx >= 0 ? selectedIdx : 0;
    sel.value = String(Math.min(ix, models.length - 1));
  };

  const selectedModel = () => {
    const v = $("#bt-model")?.value;
    if (v !== "" && v !== "__browse__" && Number.isInteger(+v)) {
      return (window._btModels || [])[+v] || window._btPicked || null;
    }
    return window._btPicked || null;
  };

  const applyRange = () => {
    const m = selectedModel();
    const rangeMode = $("#bt-range").value, s = $("#bt-start"), e = $("#bt-end");
    s.disabled = e.disabled = rangeMode !== "custom";
    if (m && m.span_start) { s.min = e.min = m.span_start; s.max = e.max = m.span_end; }
    if (rangeMode === "custom") return;
    const bounds = backtestRangeBounds(rangeMode, m && m.span_start, m && m.span_end);
    if (bounds) { s.value = bounds.start; e.value = bounds.end; }
  };

  const syncMeta = () => {
    const el = $("#bt-modelmeta");
    const extOn = window._btPlane === "ext";
    const tauWrap = $("#bt-tau-wrap"), posWrap = $("#bt-pos-wrap");
    if (tauWrap) tauWrap.style.display = extOn ? "none" : "";
    if (posWrap) posWrap.style.display = extOn ? "none" : "";
    const mlExec = $("#bt-ml-exec"), execOpts = $("#bt-exec-opts"), adj = $("#bt-adj-row");
    if (mlExec) mlExec.style.display = extOn ? "none" : "flex";
    if (execOpts) execOpts.style.display = extOn ? "none" : "";
    if (adj) adj.style.display = extOn ? "none" : "flex";
    const help = $("#bt-mode-help");
    if (extOn) {
      const stack = (window._btExtStack || []).filter(l => l.model);
      const names = stack.map(l => {
        const card = extCard(l.model);
        const title = (card && card.title) || l.model;
        const tf = l.timeframe ? ` ${l.timeframe}` : "";
        return l.hardened ? `${title}${tf} / ${l.hardened}` : `${title}${tf} (DEFAULT)`;
      });
      if (help) {
        help.innerHTML = `<span class="pill info">EXT</span>
          External engines on PostgreSQL <span class="mono">databento.tbbo_1min</span>
          (lead-month NQ). Each stacked model votes long / short / flat; matching
          sides enter, opposing or flat signals stand down or exit.
          Per-engine <b>candle timeframe</b> can differ (same model twice is allowed).
          <b>Backtest Parameters</b> below apply to the whole stack
          (session vs trade window, ATR/PT/SL units).
          Engine knobs (SAR AF, EMA length) are on
          <a href="#/external">External Models</a>.
          ML τ / TBBO OCO / 50-6 labels do not apply.`;
      }
      if (el) {
        el.innerHTML = stack.length
          ? `<b>Stack:</b> ${names.map(n => `<span class="pill info">${esc(n)}</span>`).join(" ")}
             ${stack.length > 1 ? `<br><span class="muted">Consensus AND: all engines must agree to enter.</span>` : ""}
             <br><span class="muted">Tune SAR / EMA knobs on
             <a href="#/external">External Models</a>. Session, trade window, ATR, PT/SL, and costs are
             <b>Backtest Parameters</b> on this tab.</span>`
          : `<span class="muted">Pick an EXT model, then DEFAULT or a hardened snapshot.
             Use ＋ to stack another engine.</span>`;
      }
      applyRange();
      return;
    }
    const m = selectedModel();
    if (!m) { if (el) el.textContent = ""; return; }
    const kind = m.source_kind === "cascade_oof" ? "cascade OOF"
      : (m.source_kind === "oof" ? "search OOF"
      : (m.source_kind === "custom_test" ? "custom test"
        : (m.source_kind || "bundle")));
    el.innerHTML = `<b>Source:</b>
      <span class="pill info">${esc(kind)}</span>
      <span class="mono">${esc(m.run_label || bundleRunLabel(m.source) || m.model)}</span>
      · objective <span class="mono">${esc(m.objective_id || "—")}</span><br>
      <b>Scores feature days:</b>
      <span class="mono">${esc(m.span_start || "?")} → ${esc(m.span_end || "?")}</span>
      (${m.n_rows != null ? m.n_rows + " sessions on disk" : "—"})<br>
      <span class="mono" style="font-size:11px">${esc(m.source)}</span><br>
      <span class="muted">${esc(m.detail || "")}</span>`;
    paintHelp();
    const pol = m.inference_policy;
    if (pol && typeof pol === "object") {
      const sides = (pol.sides || []).join("/") || "up/down";
      const win = `${pol.trading_start || "?"}–${pol.trading_end || "?"}`;
      const cd = pol.cooldown_bars != null ? ` · cooldown ${pol.cooldown_bars} bars` : "";
      el.innerHTML += `<br><b>Inference policy:</b> τ=${esc(pol.tau ?? "—")} · ${esc(win)}`
        + ` ${esc(pol.timezone || "")} · ${esc(sides)}${esc(cd)}`;
    }
    if (m.recommended_tau != null && m.source_kind !== "cascade_oof") {
      const tauSel = $("#bt-tau");
      if (tauSel) tauSel.value = Number(m.recommended_tau).toFixed(2);
    }
    applyRange();
  };

  const renderBrowse = listing => {
    const box = $("#bt-browse");
    if (!box) return;
    if (!window._btBrowseOpen) { box.hidden = true; box.innerHTML = ""; return; }
    box.hidden = false;
    const crumb = listing.path
      ? `<span class="mono">${esc(listing.path)}</span>`
      : `<span class="muted">pick a tree</span>`;
    const up = listing.parent
      ? `<button type="button" class="small" onclick="window._btBrowseTo(${JSON.stringify(listing.parent)})">← up</button>`
      : (listing.path
        ? `<button type="button" class="small" onclick="window._btBrowseTo(null)">← roots</button>`
        : "");
    const rows = (listing.entries || []).map(e => {
      const icon = e.kind === "oof" ? "▤" : e.kind === "bundle" ? "▣" : "📁";
      const extra = e.kind === "oof" ? "OOF"
        : e.kind === "bundle" ? "bundle"
        : (e.n_oof ? `${e.n_oof} OOF` : (e.n_children != null ? `${e.n_children} items` : ""));
      const payload = encodeURIComponent(JSON.stringify(e.selectable ? e : e.path));
      const fn = e.selectable ? "_btPickBrowse" : "_btBrowseTo";
      return `<button type="button" class="bt-browse-row"
        onclick="window.${fn}(JSON.parse(decodeURIComponent('${payload}')))">
        <span>${icon} <span class="mono">${esc(e.name)}</span></span>
        <span class="muted">${esc(extra)}</span>
      </button>`;
    }).join("") || `<div class="empty">empty</div>`;
    box.innerHTML = `
      <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:8px">
        <div><b>Browse</b> ${crumb}</div>
        <div class="fixed">${up}
          <button type="button" class="small" onclick="window._btToggleBrowse(false)">close</button></div>
      </div>
      <div class="muted" style="margin-bottom:8px">Pick a serving folder or a Custom Test model folder (<span class="mono">models/lstm_15s</span> with <span class="mono">oof.parquet</span>). Walk-forward OOF replay is as-of. Root <span class="mono">oof_*.parquet</span> tapes are not listed.</div>
      <div class="bt-browse-list">${rows}</div>`;
  };

  window._btToggleBrowse = async (force) => {
    window._btBrowseOpen = force === false ? false : !window._btBrowseOpen;
    const btn = $("#bt-browse-btn");
    if (btn) btn.classList.toggle("primary", window._btBrowseOpen);
    if (!window._btBrowseOpen) { renderBrowse({}); return; }
    await window._btBrowseTo(window._btBrowsePath);
  };
  window._btBrowseTo = async path => {
    window._btBrowsePath = path;
    try {
      const q = path ? `?path=${encodeURIComponent(path)}` : "";
      const listing = await api(`/api/backtest/browse${q}`);
      renderBrowse(listing);
    } catch (e) {
      toast(e.message, "err");
    }
  };
  window._btPickBrowse = entry => {
    const first = (window._btModels || [])[0] || {};
    const idOk = s => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(s || "");
    const fromLabel = (entry.run_label || "").split("/")[0];
    const oid = idOk(entry.objective_id) ? entry.objective_id
      : (idOk(fromLabel) ? fromLabel : (first.objective_id || ""));
    const picked = {
      source: entry.path,
      source_kind: entry.kind === "oof" ? "oof" : "weights",
      model: entry.candidate_id || entry.name,
      objective_id: oid,
      run_label: entry.run_label || entry.name,
      label: entry.run_label || entry.name,
      span_start: first.span_start,
      span_end: first.span_end,
      n_rows: first.n_rows,
      backtestable: true,
      recommended_tau: first.recommended_tau,
      detail: entry.kind === "oof"
        ? "Walk-forward OOF replay from a search run (not staged serving weights)."
        : "Serving bundle from the search-run folder.",
    };
    const models = window._btModels || [];
    let ix = models.findIndex(m => m.source === picked.source);
    if (ix < 0) {
      models.unshift(picked);
      ix = 0;
    } else {
      models[ix] = { ...models[ix], ...picked };
    }
    window._btModels = models;
    window._btPicked = picked;
    window._btBrowseOpen = false;
    renderBrowse({});
    const btn = $("#bt-browse-btn");
    if (btn) btn.classList.remove("primary");
    paintModelSelect(ix);
    toast(`selected ${picked.run_label}`, "ok", 3500);
    syncMeta();
  };

  const loadModels = async () => {
    $("#bt-model").innerHTML = `<option>loading…</option>`;
    try {
      const [models, extPayload] = await Promise.all([
        api("/api/backtest/models"),
        api("/api/external-models").catch(() => ({ models: [], book: {} })),
      ]);
      window._btModels = models;
      window._btExtCatalog = Array.isArray(extPayload) ? extPayload : (extPayload.models || []);
      window._btExtBook = (extPayload && extPayload.book) || null;
      if (!window._btExtStack[0].model && window._btExtCatalog[0]) {
        window._btExtStack = [{
          model: window._btExtCatalog[0].id, hardened: "", timeframe: "",
        }];
      }
      const firstOk = models.findIndex(m => m.backtestable !== false);
      let sel = firstOk >= 0 ? firstOk : 0;
      const pre = params.get("model");
      if (pre) {
        const ix = models.findIndex(m => m.source === pre && m.backtestable !== false);
        if (ix >= 0) sel = ix;
      }
      let handoff = null;
      try { handoff = JSON.parse(sessionStorage.getItem(EXT_HANDOFF) || "null"); } catch { handoff = null; }
      const extId = params.get("external");
      const planeQ = params.get("plane");
      if (handoff && (handoff.plane === "ext" || handoff.stack || String(handoff.source || "").startsWith("external:"))) {
        window._btPlane = "ext";
        if (Array.isArray(handoff.stack) && handoff.stack.length) {
          window._btExtStack = handoff.stack.map(leg => ({
            model: leg.model,
            hardened: leg.hardened || "",
            params: leg.params || null,
            timeframe: leg.timeframe || "",
          }));
        } else if (handoff.source) {
          const src = String(handoff.source).replace(/^external:/, "");
          const bits = src.split(":");
          window._btExtStack = [{
            model: bits[0],
            hardened: bits[1] === "hardened" ? bits.slice(2).join(":") : (bits[1] || ""),
            params: handoff.params || null,
            timeframe: handoff.timeframe || "",
          }];
        }
        window._btExtBookOverride = handoff.book || null;
        sessionStorage.removeItem(EXT_HANDOFF);
      } else if (planeQ === "ext" || extId) {
        window._btPlane = "ext";
        if (extId) {
          window._btExtStack = [{ model: extId, hardened: "", timeframe: "" }];
        }
      } else if (pre) {
        window._btPlane = "ml";
      }
      if (!models.length && window._btPlane === "ml") {
        paintModelSelect(-1);
        $("#bt-modelmeta").innerHTML =
          `<div class="empty">No staged models — finalize a champion into model_staging first, or switch to EXT Model.</div>`;
      } else {
        paintModelSelect(sel);
      }
      paintPlane();
    } catch (e) {
      $("#bt-modelmeta").innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
    }
  };

  $("#bt-model").addEventListener("change", () => {
    const v = $("#bt-model").value;
    window._btPicked = (v !== "" && v !== "__browse__")
      ? ((window._btModels || [])[+v] || null)
      : window._btPicked;
    syncMeta();
  });
  $("#bt-plane").addEventListener("change", () => {
    window._btPlane = $("#bt-plane").value === "ext" ? "ext" : "ml";
    paintPlane();
  });
  $("#bt-book-save")?.addEventListener("click", async () => {
    const spec = (window._btExtBook && window._btExtBook.param_spec) || [];
    try {
      const rec = await put("/api/external-models/book/defaults", {
        params: extReadParams(spec, "bt-b-"),
      });
      window._btExtBook = {
        ...(window._btExtBook || {}),
        defaults: rec.params,
        saved_at: rec.saved_at,
        note: rec.note,
      };
      window._btExtBookOverride = rec.params;
      toast("backtest parameter defaults saved", "ok");
      paintExtBook(rec.params);
    } catch (e) { toast(e.message, "err"); }
  });
  $("#bt-book-reset")?.addEventListener("click", async () => {
    try {
      const rec = await post("/api/external-models/book/defaults/reset", {});
      window._btExtBook = {
        ...(window._btExtBook || {}),
        defaults: rec.params,
        saved_at: null,
        note: rec.note,
      };
      window._btExtBookOverride = rec.params;
      toast("backtest parameters restored to factory", "ok");
      paintExtBook(rec.params);
    } catch (e) { toast(e.message, "err"); }
  });
  $("#bt-ext-add").addEventListener("click", () => {
    const catalog = window._btExtCatalog || [];
    const stack = window._btExtStack || [];
    const standCard = extStandaloneCard();
    if (standCard) {
      toast(`${standCard.title || standCard.id} is standalone and cannot be stacked`, "warn");
      return;
    }
    const last = stack[stack.length - 1] || {};
    const next = catalog.find(m => m.id === last.model) || catalog[0];
    window._btExtStack = stack;
    window._btExtStack.push({
      model: next ? next.id : "",
      hardened: "",
      timeframe: last.timeframe || bookTimeframe(),
    });
    paintExtLegs();
    syncMeta();
  });
  $("#bt-range").addEventListener("change", applyRange);
  const tauOn = $("#bt-tau-on");
  if (tauOn) {
    if (localStorage.getItem("bt_tau_gate") === "0") tauOn.checked = false;
    const syncTauGate = () => {
      writeBoolPref("bt_tau_gate", tauOn.checked);
      const sel = $("#bt-tau");
      if (sel) sel.disabled = !tauOn.checked;
    };
    tauOn.addEventListener("change", syncTauGate);
    syncTauGate();
  }
  const mvOn = $("#bt-mv-on");
  if (mvOn) {
    if (localStorage.getItem("bt_mv_gate") === "0") mvOn.checked = false;
    const syncMvGate = () => {
      writeBoolPref("bt_mv_gate", mvOn.checked);
      const sel = $("#bt-mv");
      if (sel) sel.disabled = !mvOn.checked;
    };
    mvOn.addEventListener("change", syncMvGate);
    syncMvGate();
  }
  paintOcoControls();
  await loadModels();
}

window.btRun = async () => {
  const extOn = window._btPlane === "ext";
  const range = $("#bt-range").value;
  const onClose = window._btOnClose !== false;
  const oco = extOn ? false : !onClose;
  const mid = window._btMid !== false;
  const maeAdj = Number($("#bt-mae-adj")?.value || 0);
  const tgtAdj = Number($("#bt-tgt-adj")?.value || 0);
  const entryLat = Number($("#bt-entry-lat")?.value || 0);
  const exitLat = Number($("#bt-exit-lat")?.value || 0);
  const body = {
    source: "",
    objective_id: null,
    tau: extOn ? 0 : Number($("#bt-tau").value),
    confidence_gate: extOn ? false : ($("#bt-tau-on")?.checked !== false),
    mv: extOn ? 0 : Number($("#bt-mv")?.value ?? 0.5),
    mv_gate: extOn ? false : ($("#bt-mv-on")?.checked !== false),
    p_reach_min: extOn ? 0 : Number($("#bt-tau").value),
    p_move_min: extOn ? 0 : Number($("#bt-mv")?.value ?? 0.5),
    min_tgt_r: extOn ? 0 : Number($("#bt-mintgt")?.value ?? 0.25),
    max_positions: extOn ? 1 : Number($("#bt-pos").value),
    start: range === "all" ? null : ($("#bt-start").value || null),
    end: range === "all" ? null : ($("#bt-end").value || null),
    oco: oco,
    tbbo_data: oco,
    mae_adjust: Number.isFinite(maeAdj) ? maeAdj : 0,
    target_adjust: Number.isFinite(tgtAdj) ? tgtAdj : 0,
    price_mode: mid ? "mid" : "mkt",
    entry_latency_ms: Number.isFinite(entryLat) && entryLat > 0 ? entryLat : 0,
    exit_latency_ms: Number.isFinite(exitLat) && exitLat > 0 ? exitLat : 0,
  };
  if (extOn) {
    const selected = (window._btExtStack || []).filter(leg => leg.model);
    if (!selected.length) { toast("pick an EXT model", "err"); return; }
    const selectedCard = selected.length === 1 ? extCard(selected[0].model) : null;
    const standalone = selectedCard && selectedCard.standalone;
    const formSpec = standalone
      ? (selectedCard.param_spec || [])
      : ((window._btExtBook && window._btExtBook.param_spec) || []);
    const fromForm = extReadParams(formSpec, standalone ? "bt-m-" : "bt-b-");
    const bookTf = fromForm.timeframe
      || (window._btExtBookOverride && window._btExtBookOverride.timeframe)
      || (window._btExtBook && window._btExtBook.defaults && window._btExtBook.defaults.timeframe)
      || "1min";
    const stack = selected.map(l => ({
      model: l.model,
      hardened: l.hardened || null,
      params: standalone ? fromForm : (l.params || null),
      timeframe: l.timeframe || bookTf,
    }));
    body.stack = stack;
    body.book = standalone ? null : (Object.keys(fromForm).length
      ? fromForm
      : (window._btExtBookOverride
        || (window._btExtBook && window._btExtBook.defaults)
        || null));
    body.source = stack.length === 1
      ? (`external:${stack[0].model}` + (stack[0].hardened ? `:hardened:${stack[0].hardened}` : ""))
      : ("external:stack:" + stack.map(l => l.model + (l.hardened ? `:hardened:${l.hardened}` : "")).join("+"));
  } else {
    const v = $("#bt-model")?.value;
    let m = null;
    if (v !== undefined && v !== "" && v !== "__browse__" && Number.isInteger(+v)) {
      m = (window._btModels || [])[+v] || null;
    }
    if (!m) m = window._btPicked;
    if (!m || !m.source) { toast("pick a staged model or browse a search folder", "err"); return; }
    body.source = m.source;
    body.objective_id = m.objective_id || null;
  }
  const res = $("#bt-results");
  res.innerHTML = extOn
    ? `<div class="empty"><span class="spin">◐</span> <b>EXT</b> on <span class="mono">tbbo_1min</span>…</div>`
    : (oco
    ? `<div class="empty"><span class="spin">◐</span> <b>OCO</b> (${mid ? "MID" : "MKT"}): continuous TBBO bracket…</div>`
    : `<div class="empty"><span class="spin">◐</span> <b>on close</b> (${mid ? "MID" : "MKT"}): 5s closes + TBBO fill prices…</div>`);
  try {
    const t0 = Date.now();
    const p = await post("/api/backtest/run", body, {timeoutMs: extOn ? 300000 : 180000});
    p._elapsed = (Date.now() - t0) / 1000;
    window._btPayload = p;
    renderBacktestResults(p, "summary");
  } catch (e) {
    res.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
    toast(e.message, "err", 9000);
  }
};

window.btTab = t => { if (window._btPayload) renderBacktestResults(window._btPayload, t); };

function renderBacktestResults(p, tab) {
  const c = p.config, s = p.summary;
  const pv = resolvePointValue(p.config || p.summary);
  const tabBtn = (id, label) =>
    `<button class="${tab === id ? "active" : ""}" onclick="btTab('${id}')">${label}</button>`;
  const title = c.run_label || bundleRunLabel(c.source || c.weights_path || c.model_path)
    || c.candidate_id || c.strategy_id;
  const exec = c.execution || c.fill_path || (c.oco ? "oco" : "on_close");
  const ocoOn = exec === "oco" || exec === "tbbo_data" || exec === "tbbo_ticks";
  const extOn = c.source_kind === "external" || exec === "pine_candle" || exec === "pine_1m";
  const pxMode = (c.price_mode || "mid").toUpperCase();
  const fillPill = extOn
    ? `<span class="pill info">EXT · ${esc(c.timeframe || "1m")}</span>`
    : (ocoOn
    ? `<span class="pill ok">OCO · ${esc(pxMode)}</span>`
    : `<span class="pill ok">on close · ${esc(pxMode)}</span>`);
  const adjNote = (c.mae_adjust || c.target_adjust)
    ? ` · adj MAE ${fmtNum(c.mae_adjust || 0, 2)} / tgt ${fmtNum(c.target_adjust || 0, 2)}`
    : "";
  const latNote = (c.entry_latency_ms || c.exit_latency_ms)
    ? ` · lat entry ${fmtNum(c.entry_latency_ms || 0, 0)}ms / exit ${fmtNum(c.exit_latency_ms || 0, 0)}ms`
    : "";
  const posNote = c.max_concurrent_positions != null
    ? ` · ${fmtInt(c.max_concurrent_positions)} open position(s)` : "";
  const mvNote = c.source_kind === "cascade_oof"
    ? " · Stage 1 " + (c.mv_gate === false
      ? "p_move gate off"
      : "p_move >= " + (c.p_move_min ?? c.mv))
    : "";
  const tgtNote = c.source_kind === "cascade_oof" && (c.min_tgt_r ?? 0) > 0
    ? ` · tgt_dist_r >= ${c.min_tgt_r}` : "";
  const scoreNote = c.source_kind === "cascade_oof"
    ? (c.confidence_gate === false
      ? "p_reach gate off"
      : "p_reach >= " + (c.p_reach_min ?? c.tau))
    : (c.confidence_gate === false ? "confidence gate off" : `conf ≥ τ=${c.tau}`);
  const inst = c.instrument ? ` · ${esc(c.instrument)} @ ${fmtUsd(pv, 0)}/pt` : ` · ${fmtUsd(pv, 0)}/pt`;
  $("#bt-results").innerHTML = `
    <div style="margin:6px 0 8px">
      ${fillPill}
      <span class="pill info">${esc(title)}</span>
      <span class="muted"> ${extOn
        ? `Pine replica${c.params && c.params.min_atr != null ? " · ATR&gt;" + esc(c.params.min_atr) : ""}`
        : scoreNote}${mvNote}${posNote}
        · target ${fmtNum(c.target_points, 0)} / stop ${fmtNum(c.stop_points, 0)} / ${esc(c.max_holding || "—")}${adjNote}${latNote}${tgtNote}
        · ${esc(c.window.start)}→${esc(c.window.end)} (${fmtInt(c.window.n_sessions)} sessions)${
          p._elapsed ? " · " + p._elapsed.toFixed(1) + "s" : ""}${inst}</span>
    </div>
    ${c.weights_note ? `<div class="empty" style="margin:8px 0;text-align:left">${esc(c.weights_note)}</div>` : ""}
    ${c.n_pred_gated != null
      ? `<div class="muted" style="margin-bottom:8px">Scored:
          ${fmtInt(c.n_pred_directional)} directional ·
          <b>${fmtInt(c.n_pred_gated)}</b> ${c.confidence_gate === false
            ? "admitted (confidence gate off)"
            : `with conf ≥ ${c.tau} (before position limits)`}
          · ${fmtInt(c.n_features)} features · dir_threshold ${c.dir_threshold ?? "—"}</div>`
      : ""}
    <div class="tabs">${tabBtn("summary", "Summary")}${tabBtn("blotter", `Trade blotter (${fmtInt(s.n_trades)})`)}${tabBtn("chart", "Chart")}${tabBtn("archetypes", `Archetypes (${fmtInt((p.archetypes || []).length)})`)}</div>
    <div id="bt-tabbody"></div>`;
  btDestroyPriceChart();
  if (tab === "blotter") { $("#bt-tabbody").innerHTML = btBlotter(p); }
  else if (tab === "archetypes") { $("#bt-tabbody").innerHTML = btArchetypes(p); }
  else if (tab === "chart") {
    $("#bt-tabbody").innerHTML = btChartShell(p);
    btMountPriceChart(p);
  }
  else { $("#bt-tabbody").innerHTML = btSummary(p); btWirePnlChart(p, pv); }
}

function btSummary(p) {
  const s = p.summary;
  const pv = resolvePointValue(p.config || s);
  if (!s.n_trades) {
    const emptyHint = (p.config && p.config.source_kind === "external")
      ? "no trades — session / ATR / stacked-signal filters admitted no entries in this date range."
      : (p.config && p.config.confidence_gate === false
        ? "no trades — no entries after other filters. Try a range with feature parquets."
        : `no trades — conf ≥ τ=${p.config.tau} admitted no entries. Try lower τ or a range with feature parquets.`);
    return `<div class="empty">${emptyHint}</div>`;
  }
  const sign = v => v == null ? "" : v >= 0 ? "pos" : "neg";
  const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
  const tile = (lbl, val, cls, sub) =>
    `<div class="tile"><div class="lbl">${lbl}</div><div class="val ${cls || ""}">${val}</div>${
      sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
  const ptsUsd = (pts, d = 2) => fmtPtsUsd(pts, pv, d);
  const tiles = [
    tile("Net P/L", ptsUsd(s.net_pnl), sign(s.net_pnl), `${fmtInt(s.n_trades)} trades`),
    tile("Avg PnL", ptsUsd(s.expectancy, 2), sign(s.expectancy), "per trade (NE)"),
    tile("Total revenue", ptsUsd(s.gross_profit), "pos", `${fmtInt(s.n_wins)} winning`),
    tile("Total loss", ptsUsd(s.gross_loss), "neg", `${fmtInt(s.n_losses)} losing`),
    tile("Win rate", pct(s.win_rate), "",
      `avg ${fmtPtsUsd(s.avg_win, pv, 1)} / ${fmtPtsUsd(s.avg_loss, pv, 1)}`),
    tile("Max drawdown", ptsUsd(s.max_drawdown), "neg", "peak-to-trough net loss"),
    tile("Profit factor", fmtNum(s.profit_factor, 2), s.profit_factor >= 1 ? "pos" : "neg"),
    tile("Sharpe (daily)", fmtNum(s.sharpe_daily, 2), "", `${fmtInt(s.n_sessions_traded)} sessions traded`),
    tile("Exposure", pct(s.exposure_frac), "", `avg hold ${fmtInt(s.avg_holding_bars)} bars`),
  ].join("");
  const exits = Object.entries(s.by_exit_reason || {})
    .map(([k, v]) => `<span class="pill dim">${esc(k)}: ${fmtInt(v)}</span>`).join(" ") || "—";
  return `
    <div class="tiles">${tiles}</div>
    <div class="card" style="margin-top:14px"><h3 id="bt-pnl-title">P/L over time</h3>
      <div class="bt-pnl-toolbar" id="bt-pnl-toolbar">
        <div class="grp">
          <span class="grp-label">granularity</span>
          <label class="checkbox"><input type="radio" name="bt-pnl-level" value="day">
            <span>Day (EOD)</span></label>
          <label class="checkbox"><input type="radio" name="bt-pnl-level" value="trade">
            <span>Trade</span></label>
        </div>
        <div class="grp">
          <span class="grp-label">series</span>
          <label class="checkbox"><input type="checkbox" id="bt-pnl-line">
            <span>Cumulative line</span></label>
          <label class="checkbox"><input type="checkbox" id="bt-pnl-bars">
            <span>Gain/loss bars</span></label>
        </div>
        <div class="grp" id="bt-pnl-barmode-grp">
          <span class="grp-label">bar style</span>
          <label class="checkbox"><input type="radio" name="bt-pnl-barmode" value="net">
            <span>Net G/L</span></label>
          <label class="checkbox"><input type="radio" name="bt-pnl-barmode" value="split">
            <span>Net G / Net L</span></label>
        </div>
        <div class="grp">
          <span class="grp-label">units</span>
          <label class="checkbox"><input type="radio" name="bt-pnl-unit" value="pt">
            <span>Points</span></label>
          <label class="checkbox"><input type="radio" name="bt-pnl-unit" value="usd">
            <span>Dollars</span></label>
        </div>
      </div>
      <div class="chart-wrap" id="bt-chart"></div>
      <div class="muted" id="bt-pnl-note" style="margin-top:8px"></div>
    </div>
    <div class="grid cols-2" style="margin-top:14px">
      <div class="card"><h3>Confusion matrix — entered trades</h3>${btConfusion(p.confusion_matrix)}
        <div class="muted" style="margin-top:8px">Rows = true move; columns = predicted side. Diagonal = correct.</div></div>
      <div class="card"><h3>Trade breakdown</h3>
        <div style="margin-bottom:12px">${exits}</div>
        <dl class="kv">
          <dt>avg MAE / MFE</dt><dd>${fmtNum(s.avg_mae, 1)} / ${fmtNum(s.avg_mfe, 1)} pt</dd>
          <dt>avg holding</dt><dd>${fmtInt(s.avg_holding_bars)} bars (~${fmtInt(s.avg_holding_bars * (p.config.bar_seconds || 5))}s)</dd>
          <dt>trades / session</dt><dd>${fmtNum(s.n_trades / (s.n_sessions_traded || 1), 1)}</dd>
          <dt>signals skipped (slots full)</dt><dd>${fmtInt((p.skipped && p.skipped.in_position) || 0)}</dd>
          <dt>$ / point</dt><dd>${fmtUsd(pv, 0)}${p.config.instrument ? ` (${esc(p.config.instrument)})` : ""}</dd>
        </dl></div>
    </div>`;
}

function btConfusion(cm) {
  const L = cm.labels, M = cm.rows_true_cols_pred;
  const max = Math.max(1, ...M.flat());
  const cell = (v, diag) => {
    const a = v / max;
    const bg = v ? (diag ? `rgba(126,224,163,${0.1 + a * 0.5})` : `rgba(255,107,107,${0.1 + a * 0.5})`) : "transparent";
    return `<td class="num ${diag ? "diag" : ""}" style="background:${bg}">${v ? fmtInt(v) : ""}</td>`;
  };
  return `<table class="cm"><thead>
    <tr><th></th><th colspan="${L.length}" style="color:var(--dim);font-size:11px">predicted →</th></tr>
    <tr><th class="rowlab">true ↓</th>${L.map(l => `<th>${esc(l)}</th>`).join("")}</tr></thead>
    <tbody>${M.map((row, ri) => `<tr><td class="rowlab">${esc(L[ri])}</td>${row.map((v, ci) => cell(v, ri === ci)).join("")}</tr>`).join("")}</tbody></table>`;
}

function btBlotter(p) {
  if (!p.trades.length) return `<div class="empty">no trades taken</div>`;
  const pv = resolvePointValue(p.config || p.summary);
  return `<div class="card" style="overflow:auto"><table>
    <thead><tr><th>#</th><th>side</th><th>entry</th><th class="num">entry px</th><th>exit</th><th class="num">exit px</th>
      <th>reason</th><th class="num">hold</th><th class="num">MAE</th>
      <th class="num">feat avail</th><th class="num">feat total</th>
      <th>archetype</th><th class="num">train WR</th>
      <th class="num">P/L (pt · $)</th><th class="num">cum P/L</th></tr></thead>
    <tbody>${p.trades.map((t, i) => `<tr>
      <td class="mono">${fmtInt(i + 1)}</td>
      <td><span class="pill ${t.side === "long" ? "ok" : "err"}">${t.side === "long" ? "▲ long" : "▼ short"}</span></td>
      <td class="mono">${_btTs(t.entry_ts)}</td><td class="num">${fmtNum(t.entry_px, 2)}</td>
      <td class="mono">${_btTs(t.exit_ts)}</td><td class="num">${fmtNum(t.exit_px, 2)}</td>
      <td class="mono">${esc(t.exit_reason)}</td><td class="num">${fmtInt(t.holding_bars)}</td>
      <td class="num">${fmtNum(t.mae_points, 1)}</td>
      <td class="num">${t.n_present != null ? fmtInt(t.n_present) : "—"}</td>
      <td class="num">${t.n_total != null ? fmtInt(t.n_total) : "—"}</td>
      <td class="mono" style="font-size:11px">${esc(t.archetype_label || t.archetype || "—")}</td>
      <td class="num">${t.archetype_train_hit_rate != null
        ? (t.archetype_train_hit_rate * 100).toFixed(1) + "%"
        : "<span class='muted'>unseen</span>"}</td>
      <td class="num ${t.pnl_points >= 0 ? "pos" : "neg"}">${fmtPtsUsd(t.pnl_points, pv)}</td>
      <td class="num ${t.cum_pnl >= 0 ? "pos" : "neg"}">${fmtPtsUsd(t.cum_pnl, pv)}</td></tr>`).join("")}</tbody>
  </table></div>`;
}

function btArchetypes(p) {
  const rows = p.archetypes || [];
  if (!rows.length) {
    return `<div class="empty">archetype attribution unavailable — training OOF frame not found for this bundle</div>`;
  }
  const pv = resolvePointValue(p.config || p.summary);
  const pct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";
  const deltaCell = d => {
    if (d == null) return `<td class="num muted">—</td>`;
    const cls = d >= 0 ? "pos" : "neg";
    const sign = d >= 0 ? "+" : "";
    return `<td class="num ${cls}">${sign}${(d * 100).toFixed(1)} pp</td>`;
  };
  return `<div class="muted" style="margin:0 0 8px">
    Each row groups trades by <b>(direction · confidence bin · hour CT)</b>. Compares live
    win-rate to what the same archetype scored on training-time OOF. <b>Δ WR</b> ≪ 0 with
    n_live ≥ 20 = early decay signal for that pattern.</div>
    <div class="card" style="overflow:auto"><table>
    <thead><tr><th>archetype</th>
      <th class="num">n live</th><th class="num">WR live</th>
      <th class="num">n train</th><th class="num">WR train</th>
      <th class="num">Δ WR</th>
      <th class="num">share live</th><th class="num">share train</th>
      <th class="num">avg P/L (pt · $)</th>
      <th class="num">total P/L (pt · $)</th></tr></thead>
    <tbody>${rows.map(r => `<tr>
      <td class="mono" style="font-size:12px">${esc(r.label || r.archetype)}</td>
      <td class="num">${fmtInt(r.n_live)}</td>
      <td class="num">${pct(r.hit_rate_live)}</td>
      <td class="num ${r.n_train ? "" : "muted"}">${r.n_train ? fmtInt(r.n_train) : "unseen"}</td>
      <td class="num">${pct(r.hit_rate_train)}</td>
      ${deltaCell(r.hit_rate_delta)}
      <td class="num">${pct(r.share_live)}</td>
      <td class="num">${pct(r.share_train)}</td>
      <td class="num ${r.expectancy_live >= 0 ? "pos" : "neg"}">${fmtPtsUsd(r.expectancy_live, pv, 2)}</td>
      <td class="num ${r.pnl_pts_live >= 0 ? "pos" : "neg"}">${fmtPtsUsd(r.pnl_pts_live, pv)}</td>
    </tr>`).join("")}</tbody>
  </table></div>`;
}

/* ── Backtest price chart (Lightweight Charts + SAR overlay) ──────────── */
function btDestroyPriceChart() {
  if (window._btLwState && window._btLwState.ro) {
    try { window._btLwState.ro.disconnect(); } catch { /* */ }
  }
  if (window._btLwChart) {
    try { window._btLwChart.remove(); } catch { /* already gone */ }
    window._btLwChart = null;
  }
  document.querySelectorAll(".bt-sar-overlay").forEach(el => el.remove());
  window._btLwState = null;
}

function btChartShell() {
  return `<div class="card bt-price-card">
    <div class="bt-chart-toolbar">
      <label class="checkbox"><input type="checkbox" id="bt-ch-sess" checked>
        <span>session hours only</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-sar" checked>
        <span>SAR</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-nsar">
        <span>nextBarSAR</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-ema" checked>
        <span>EMA</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-vwap" checked>
        <span>VWAP</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-zz" checked>
        <span>ZZ arrows</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-zzline" checked>
        <span>ZZ line</span></label>
      <label class="checkbox"><input type="checkbox" id="bt-ch-vol" checked>
        <span>volume</span></label>
      <button type="button" class="small" id="bt-ch-fit">Fit</button>
      <button type="button" class="small" id="bt-ch-first">First trade</button>
      <button type="button" class="small" id="bt-ch-prev" title="Previous trade">◀</button>
      <button type="button" class="small" id="bt-ch-next" title="Next trade">▶</button>
      <button type="button" class="small" id="bt-ch-last">Last trade</button>
      <span class="muted" id="bt-ch-status"></span>
    </div>
    <div id="bt-ohlc-readout" class="bt-ohlc-readout mono"></div>
    <div id="bt-price-chart-host" class="bt-price-chart-host" tabindex="0"></div>
    <div class="bt-chart-legend muted">
      <span>Drag to pan · wheel to zoom · ← → to nudge · compact session axis (no overnight gaps)</span>
      <span style="color:#7ee0a3">▲ long entry</span>
      <span style="color:#ff6b6b">▼ short entry</span>
      <span style="color:#ffc857">● exit</span>
      <span style="color:#ff9f43">＋ SAR</span>
      <span style="color:#4ecdc4">＋ nextBarSAR</span>
      <span style="color:#7aa2f7">— EMA</span>
      <span style="color:#f7b955">— VWAP</span>
      <span class="bt-leg-zz" style="color:#7ee0a3">▲ ZZ up</span>
      <span class="bt-leg-zz" style="color:#ff6b6b">▼ ZZ down</span>
      <span class="bt-leg-zz" style="color:#c4b5fd">— ZZ pivots</span>
    </div>
  </div>`;
}

function lwVersion(LW) {
  try { return String((LW && LW.version && LW.version()) || ""); } catch { return ""; }
}

function loadLwCharts() {
  const cur = window.LightweightCharts;
  if (cur && cur.CandlestickSeries && lwVersion(cur).startsWith("5")) {
    return Promise.resolve(cur);
  }
  if (window._lwChartsP) return window._lwChartsP;
  window._lwChartsP = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/vendor/lightweight-charts.standalone.production.js?v=5.2.1";
    s.onload = () => {
      const LW = window.LightweightCharts;
      if (LW && LW.CandlestickSeries && lwVersion(LW).startsWith("5")) resolve(LW);
      else reject(new Error("Live Chart needs Lightweight Charts v5 (got " + lwVersion(LW) + ")"));
    };
    s.onerror = () => reject(new Error("failed to load /vendor/lightweight-charts.standalone.production.js"));
    document.head.appendChild(s);
  });
  return window._lwChartsP;
}

function lwEnsurePane(chart, index) {
  const idx = Math.max(0, index == null ? 0 : index);
  if (!chart || typeof chart.panes !== "function") {
    throw new Error("Lightweight Charts v5 panes required");
  }
  while (chart.panes().length <= idx) {
    if (typeof chart.addPane !== "function") throw new Error("chart.addPane missing");
    chart.addPane(true);
  }
  const pane = chart.panes()[idx];
  if (pane && typeof pane.setPreserveEmptyPane === "function") {
    try { pane.setPreserveEmptyPane(true); } catch {}
  }
  return pane;
}

function lwAddSeries(chart, LW, kind, opts, paneIndex) {
  const lib = LW || window.LightweightCharts;
  const def = lib && lib[kind];
  if (!def || !chart || typeof chart.addSeries !== "function") {
    throw new Error("Lightweight Charts v5 required (" + kind + ", " + lwVersion(lib) + ")");
  }
  const idx = paneIndex == null ? 0 : paneIndex;
  const pane = lwEnsurePane(chart, idx);
  let series = null;
  if (pane && typeof pane.addSeries === "function") {
    series = pane.addSeries(def, opts || {});
  } else {
    series = chart.addSeries(def, opts || {}, idx);
  }
  if (series && typeof series.moveToPane === "function") {
    try { series.moveToPane(idx); } catch {}
  }
  return series;
}

function lwForcePane(series, index) {
  if (!series || typeof series.moveToPane !== "function") return;
  try { series.moveToPane(index); } catch {}
}

function lwSetMarkers(LW, series, markers) {
  const lib = LW || window.LightweightCharts;
  if (lib && typeof lib.createSeriesMarkers === "function") {
    return lib.createSeriesMarkers(series, markers || []);
  }
  if (series && typeof series.setMarkers === "function") series.setMarkers(markers || []);
  return null;
}

function lwZeroLine(series) {
  if (!series || typeof series.createPriceLine !== "function") return null;
  try {
    return series.createPriceLine({
      price: 0,
      color: "rgba(125, 141, 160, 0.78)",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: "0",
    });
  } catch {
    return null;
  }
}

function ensureZeroInScale(series, floorOnly) {
  if (!series || typeof series.applyOptions !== "function") return;
  try {
    series.applyOptions({
      autoscaleInfoProvider: (base) => {
        const def = (typeof base === "function" ? base() : null) || {};
        const pr = def.priceRange || { minValue: 0, maxValue: 1 };
        const min = Number(pr.minValue);
        const max = Number(pr.maxValue);
        if (floorOnly) {
          const hi = Number.isFinite(max) && max > 0 ? max : 1;
          return Object.assign({}, def, { priceRange: { minValue: 0, maxValue: hi } });
        }
        const lo = Number.isFinite(min) ? min : 0;
        const hi = Number.isFinite(max) ? max : 0;
        return Object.assign({}, def, {
          priceRange: {
            minValue: Math.min(0, lo),
            maxValue: Math.max(0, hi === lo ? hi + 1 : hi),
          },
        });
      },
    });
  } catch {}
}

function _btBarIndex(tArr, unix) {
  let lo = 0, hi = tArr.length - 1, ans = 0;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (tArr[m] <= unix) { ans = m; lo = m + 1; }
    else hi = m - 1;
  }
  return ans;
}

async function btMountPriceChart(p) {
  const host = $("#bt-price-chart-host");
  const status = $("#bt-ch-status");
  if (!host) return;
  const cfg = p.config || {};
  const start = (cfg.window && cfg.window.start) || null;
  const end = (cfg.window && cfg.window.end) || null;
  if (!cfg.source || !start || !end) {
    host.innerHTML = `<div class="empty">chart needs a backtest window — run a backtest first</div>`;
    return;
  }
  const sessionOnly = $("#bt-ch-sess") ? $("#bt-ch-sess").checked : true;
  const showSar = () => !$("#bt-ch-sar") || $("#bt-ch-sar").checked;
  const showNsar = () => $("#bt-ch-nsar") && $("#bt-ch-nsar").checked;
  const showEma = () => !$("#bt-ch-ema") || $("#bt-ch-ema").checked;
  const showVwap = () => !$("#bt-ch-vwap") || $("#bt-ch-vwap").checked;
  const showZz = () => !$("#bt-ch-zz") || $("#bt-ch-zz").checked;
  const showZzLine = () => !$("#bt-ch-zzline") || $("#bt-ch-zzline").checked;
  const showVol = () => !$("#bt-ch-vol") || $("#bt-ch-vol").checked;
  const cacheKey = JSON.stringify([cfg.source, start, end, sessionOnly, cfg.params || null, cfg.stack || null, cfg.book || null]);
  status.textContent = "loading candles…";
  try {
    await loadLwCharts();
  } catch (e) {
    host.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
    return;
  }
  let data;
  if (window._btChartCache && window._btChartCache.key === cacheKey) {
    data = window._btChartCache.data;
  } else {
    try {
      data = await post("/api/backtest/chart", {
        source: cfg.source,
        start, end,
        params: cfg.params || null,
        book: cfg.book || null,
        stack: cfg.stack || null,
        session_only: sessionOnly,
      }, {timeoutMs: 180000});
      window._btChartCache = { key: cacheKey, data };
    } catch (e) {
      host.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
      status.textContent = "";
      return;
    }
  }
  if (!data || !data.n) {
    host.innerHTML = `<div class="empty">no candles in this window</div>`;
    status.textContent = "";
    return;
  }
  status.textContent = `${fmtInt(data.n)} ${esc(data.timeframe || "1m")} bars`
    + (data.note ? " · " + data.note : "");

  btDestroyPriceChart();
  const LW = window.LightweightCharts;
  const t0 = data.t[0];
  const fake = i => t0 + i * 60;
  const candles = [];
  const vol = [];
  const sarPts = [];
  const nsarPts = [];
  const emaPts = [];
  const vwapPts = [];
  const zzPts = [];
  const hasEma = !!(data.has_ema || (data.ema && data.ema.some(v => v != null)));
  const hasVwap = !!(data.has_vwap || (data.vwap && data.vwap.some(v => v != null)));
  const hasSar = data.has_sar !== false;
  const hasZz = !!(data.has_zz || (data.zz_trend && data.zz_trend.some(v => v === 1 || v === -1)));
  const emaBox = $("#bt-ch-ema"), sarBox = $("#bt-ch-sar"), nsarBox = $("#bt-ch-nsar");
  const vwapBox = $("#bt-ch-vwap");
  const zzBox = $("#bt-ch-zz"), zzLineBox = $("#bt-ch-zzline");
  if (emaBox) emaBox.closest("label").style.display = hasEma ? "" : "none";
  if (vwapBox) vwapBox.closest("label").style.display = hasVwap ? "" : "none";
  if (sarBox) sarBox.closest("label").style.display = hasSar ? "" : "none";
  if (nsarBox) nsarBox.closest("label").style.display = hasSar ? "" : "none";
  if (zzBox) zzBox.closest("label").style.display = hasZz ? "" : "none";
  if (zzLineBox) zzLineBox.closest("label").style.display = hasZz ? "" : "none";
  document.querySelectorAll(".bt-leg-zz").forEach(el => {
    el.style.display = hasZz ? "" : "none";
  });
  for (let i = 0; i < data.n; i++) {
    const tm = fake(i);
    candles.push({ time: tm, open: data.o[i], high: data.h[i], low: data.l[i], close: data.c[i] });
    vol.push({ time: tm, value: data.v[i] || 0,
      color: (data.c[i] || 0) >= (data.o[i] || 0) ? "rgba(126,224,163,.35)" : "rgba(255,107,107,.35)" });
    if (data.sar && data.sar[i] != null) sarPts.push({ time: tm, value: data.sar[i] });
    if (data.next_sar && data.next_sar[i] != null) nsarPts.push({ time: tm, value: data.next_sar[i] });
    if (data.ema && data.ema[i] != null) emaPts.push({ time: tm, value: data.ema[i] });
    if (data.vwap && data.vwap[i] != null) vwapPts.push({ time: tm, value: data.vwap[i] });
    if (data.zz && data.zz[i] != null) zzPts.push({ time: tm, value: data.zz[i] });
  }
  const fmtTick = t => {
    const i = Math.max(0, Math.min(data.n - 1, Math.round((t - t0) / 60)));
    const iso = new Date(data.t[i] * 1000).toISOString();
    const s = toCT(iso);
    return s ? s.slice(5, 16) : "";
  };
  const chart = LW.createChart(host, {
    layout: {
      background: { color: "#0b0f14" },
      textColor: "#7d8da0",
      fontFamily: "SF Mono, ui-monospace, Menlo, Consolas, monospace",
    },
    grid: {
      vertLines: { color: "#1f2a36" },
      horzLines: { color: "#1f2a36" },
    },
    crosshair: { mode: LW.CrosshairMode ? LW.CrosshairMode.Normal : 0 },
    rightPriceScale: { borderColor: "#1f2a36" },
    timeScale: {
      borderColor: "#1f2a36",
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 6,
      barSpacing: 7,
      minBarSpacing: 0.4,
      tickMarkFormatter: fmtTick,
    },
    localization: { timeFormatter: fmtTick },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    width: host.clientWidth,
    height: host.clientHeight,
  });
  const candleSeries = lwAddSeries(chart, LW, "CandlestickSeries", {
    upColor: "#7ee0a3", downColor: "#ff6b6b",
    wickUpColor: "#7ee0a3", wickDownColor: "#ff6b6b",
    borderUpColor: "#7ee0a3", borderDownColor: "#ff6b6b",
    priceLineVisible: false,
  }, 0);
  candleSeries.setData(candles);
  const volSeries = lwAddSeries(chart, LW, "HistogramSeries", {
    priceFormat: { type: "volume" },
    priceScaleId: "vol",
    lastValueVisible: false, priceLineVisible: false,
  }, 0);
  volSeries.setData(vol);
  try {
    volSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
      drawTicks: false, borderVisible: false,
    });
  } catch {
    try {
      chart.priceScale("vol").applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 },
        drawTicks: false, borderVisible: false,
      });
    } catch {}
  }
  candleSeries.priceScale().applyOptions({ scaleMargins: { top: 0.04, bottom: 0.22 } });
  const sarSeries = lwAddSeries(chart, LW, "LineSeries", {
    color: "#ff9f43", lineWidth: 1, lineVisible: true,
    pointMarkersVisible: false, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: "SAR",
  }, 0);
  sarSeries.setData(sarPts);
  const nsarSeries = lwAddSeries(chart, LW, "LineSeries", {
    color: "#4ecdc4", lineWidth: 1, lineVisible: false,
    pointMarkersVisible: false, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: "nextBarSAR",
  }, 0);
  nsarSeries.setData(nsarPts);
  const emaSeries = lwAddSeries(chart, LW, "LineSeries", {
    color: "#7aa2f7", lineWidth: 2, lineVisible: hasEma && showEma(),
    pointMarkersVisible: false, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: "EMA",
  }, 0);
  emaSeries.setData(emaPts);
  const vwapSeries = lwAddSeries(chart, LW, "LineSeries", {
    color: "#f7b955", lineWidth: 2, lineVisible: hasVwap && showVwap(),
    pointMarkersVisible: false, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: "VWAP",
  }, 0);
  vwapSeries.setData(vwapPts);
  const zzSeries = lwAddSeries(chart, LW, "LineSeries", {
    color: "#c4b5fd", lineWidth: 2, lineVisible: hasZz && showZzLine(),
    pointMarkersVisible: true, lastValueVisible: false, priceLineVisible: false,
    crosshairMarkerVisible: false, title: "ZZ",
  }, 0);
  zzSeries.setData(zzPts);

  const trades = p.trades || [];
  const tradeIdx = [];
  const markers = [];
  const tf = String(data.timeframe || "1min");
  const barSec = tf.startsWith("15") ? 900 : (tf.startsWith("5") ? 300 : 60);
  for (const tr of trades) {
    const eu = Date.parse(tr.entry_ts) / 1000;
    const xu = Date.parse(tr.exit_ts) / 1000;
    if (!Number.isFinite(eu)) continue;
    const ei = _btBarIndex(data.t, eu);
    if (!Number.isFinite(data.t[ei]) || Math.abs(data.t[ei] - eu) > barSec) continue;
    const xi = Number.isFinite(xu) ? _btBarIndex(data.t, xu) : ei;
    tradeIdx.push({ ei, xi, tr });
    const long = tr.side === "long";
    markers.push({
      time: fake(ei),
      position: long ? "belowBar" : "aboveBar",
      color: long ? "#7ee0a3" : "#ff6b6b",
      shape: long ? "arrowUp" : "arrowDown",
      text: long ? "LG" : "ST",
    });
    if (xi !== ei || tr.exit_reason) {
      markers.push({
        time: fake(xi),
        position: long ? "aboveBar" : "belowBar",
        color: "#ffc857",
        shape: "circle",
        text: String(tr.exit_reason || "x").slice(0, 14),
      });
    }
  }
  markers.sort((a, b) => a.time - b.time || a.position.localeCompare(b.position));
  lwSetMarkers(LW, candleSeries, markers);

  const overlay = document.createElement("canvas");
  overlay.className = "bt-sar-overlay";
  host.appendChild(overlay);
  const fitOverlay = () => {
    const r = host.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.max(1, Math.floor(r.width * dpr));
    overlay.height = Math.max(1, Math.floor(r.height * dpr));
    overlay.style.width = r.width + "px";
    overlay.style.height = r.height + "px";
  };
  const drawCrosses = () => {
    fitOverlay();
    const ctx = overlay.getContext("2d");
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return;
    const from = Math.max(0, Math.floor(range.from));
    const to = Math.min(candles.length - 1, Math.ceil(range.to));
    const nVis = to - from + 1;
    const zoomed = nVis <= 420;
    sarSeries.applyOptions({ lineVisible: hasSar && showSar() && !zoomed });
    nsarSeries.applyOptions({ lineVisible: hasSar && showNsar() && !zoomed });
    emaSeries.applyOptions({ lineVisible: hasEma && showEma() });
    vwapSeries.applyOptions({ lineVisible: hasVwap && showVwap() });
    zzSeries.applyOptions({ lineVisible: hasZz && showZzLine() });
    volSeries.applyOptions({ visible: showVol() });
    if (hasZz && showZz()) {
      const w = host.clientWidth || 1;
      const px = w / Math.max(1, nVis);
      const step = px >= 4 ? 1 : Math.max(1, Math.ceil(4 / px));
      const hArrow = Math.max(5, Math.min(8, px));
      const wArrow = Math.max(3, hArrow * 0.7);
      for (let i = from; i <= to; i += step) {
        const tr = data.zz_trend ? data.zz_trend[i] : 0;
        if (tr !== 1 && tr !== -1) continue;
        const x = chart.timeScale().timeToCoordinate(fake(i));
        if (x == null) continue;
        if (tr === 1) {
          const y = candleSeries.priceToCoordinate(data.l[i]);
          if (y == null) continue;
          ctx.fillStyle = "#7ee0a3";
          ctx.beginPath();
          ctx.moveTo(x, y + 3);
          ctx.lineTo(x - wArrow, y + 3 + hArrow);
          ctx.lineTo(x + wArrow, y + 3 + hArrow);
          ctx.closePath();
          ctx.fill();
        } else {
          const y = candleSeries.priceToCoordinate(data.h[i]);
          if (y == null) continue;
          ctx.fillStyle = "#ff6b6b";
          ctx.beginPath();
          ctx.moveTo(x, y - 3);
          ctx.lineTo(x - wArrow, y - 3 - hArrow);
          ctx.lineTo(x + wArrow, y - 3 - hArrow);
          ctx.closePath();
          ctx.fill();
        }
      }
    }
    if (!zoomed) return;
    const arm = 4;
    const paint = (series, ptsSet, color, enabled) => {
      if (!enabled) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      for (let i = from; i <= to; i++) {
        const val = ptsSet[i];
        if (val == null) continue;
        const x = chart.timeScale().timeToCoordinate(fake(i));
        const y = series.priceToCoordinate(val);
        if (x == null || y == null) continue;
        ctx.moveTo(x - arm, y); ctx.lineTo(x + arm, y);
        ctx.moveTo(x, y - arm); ctx.lineTo(x, y + arm);
      }
      ctx.stroke();
    };
    // Align overlay values to candle index via fake time == t0+i*60
    const sarByI = data.sar;
    const nsarByI = data.next_sar;
    paint(sarSeries, sarByI, "#ff9f43", showSar());
    paint(nsarSeries, nsarByI, "#4ecdc4", showNsar());
  };

  const readout = $("#bt-ohlc-readout");
  chart.subscribeCrosshairMove(param => {
    if (!readout) return;
    if (!param || param.time == null) { readout.textContent = ""; return; }
    const i = Math.max(0, Math.min(data.n - 1, Math.round((param.time - t0) / 60)));
    const iso = new Date(data.t[i] * 1000).toISOString();
    const tlab = toCT(iso) || "";
    const o = data.o[i], h = data.h[i], l = data.l[i], c = data.c[i];
    const sar = data.sar ? data.sar[i] : null, ns = data.next_sar ? data.next_sar[i] : null;
    const ema = data.ema ? data.ema[i] : null;
    const vwap = data.vwap ? data.vwap[i] : null;
    const atr = data.atr ? data.atr[i] : null;
    const atrP = data.atr_period || 8;
    const minAtr = data.min_atr;
    const trn = data.trend && data.trend[i] === 1 ? "up" : (data.trend && data.trend[i] === -1 ? "dn" : "—");
    let atrLab = atr == null ? "—" : fmtNum(atr, 2);
    if (atr != null && minAtr != null && Number.isFinite(Number(minAtr))) {
      const minUnit = data.min_atr_unit === "atr" ? "×ATR" : "pt";
      atrLab += ` min ${fmtNum(minAtr, 2)}${minUnit}`;
    }
    const zt = data.zz_trend ? data.zz_trend[i] : 0;
    const zzLab = zt === 1 ? "up" : (zt === -1 ? "dn" : "—");
    const zzNeed = data.zz_need ? data.zz_need[i] : null;
    const zzMax = data.zz_max ? data.zz_max[i] : null;
    const zzMin = data.zz_min ? data.zz_min[i] : null;
    readout.textContent =
      `${tlab}  O ${fmtNum(o, 2)}  H ${fmtNum(h, 2)}  L ${fmtNum(l, 2)}  C ${fmtNum(c, 2)}`
      + `  ATR(${atrP}) ${atrLab}`
      + (hasSar ? `  · SAR ${sar == null ? "—" : fmtNum(sar, 2)}  next ${ns == null ? "—" : fmtNum(ns, 2)}  ${trn}` : "")
      + (hasEma ? `  · EMA ${ema == null ? "—" : fmtNum(ema, 2)}` : "")
      + (hasVwap ? `  · VWAP ${vwap == null ? "—" : fmtNum(vwap, 2)}` : "")
      + (hasZz ? `  · ZZ ${zzLab}  need ${zzNeed == null ? "—" : fmtNum(zzNeed, 2)}  max ${zzMax == null ? "—" : fmtNum(zzMax, 2)}  min ${zzMin == null ? "—" : fmtNum(zzMin, 2)}` : "");
  });
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => drawCrosses());
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: host.clientWidth, height: host.clientHeight });
    drawCrosses();
  });
  ro.observe(host);

  const reveal = (i, bars = 80) => {
    if (i == null || i < 0) return;
    const from = fake(Math.max(0, i - bars));
    const to = fake(Math.min(data.n - 1, i + Math.floor(bars / 3)));
    chart.timeScale().setVisibleRange({ from, to });
  };
  let curTrade = 0;
  const jumpTrade = (j) => {
    if (!tradeIdx.length) return;
    curTrade = (j + tradeIdx.length) % tradeIdx.length;
    reveal(tradeIdx[curTrade].ei);
    const tr = tradeIdx[curTrade].tr;
    status.textContent = `trade ${curTrade + 1}/${tradeIdx.length}  ${tr.side}  ${tr.exit_reason || ""}`;
  };

  chart.timeScale().fitContent();
  if (tradeIdx.length) reveal(tradeIdx[0].ei, 120);

  $("#bt-ch-fit")?.addEventListener("click", () => chart.timeScale().fitContent());
  $("#bt-ch-first")?.addEventListener("click", () => jumpTrade(0));
  $("#bt-ch-last")?.addEventListener("click", () => jumpTrade(tradeIdx.length - 1));
  $("#bt-ch-prev")?.addEventListener("click", () => jumpTrade(curTrade - 1));
  $("#bt-ch-next")?.addEventListener("click", () => jumpTrade(curTrade + 1));
  $("#bt-ch-sar")?.addEventListener("change", drawCrosses);
  $("#bt-ch-nsar")?.addEventListener("change", drawCrosses);
  $("#bt-ch-ema")?.addEventListener("change", () => {
    emaSeries.applyOptions({ lineVisible: hasEma && showEma() });
  });
  $("#bt-ch-vwap")?.addEventListener("change", () => {
    vwapSeries.applyOptions({ lineVisible: hasVwap && showVwap() });
  });
  $("#bt-ch-zz")?.addEventListener("change", drawCrosses);
  $("#bt-ch-zzline")?.addEventListener("change", () => {
    zzSeries.applyOptions({ lineVisible: hasZz && showZzLine() });
  });
  $("#bt-ch-vol")?.addEventListener("change", () => {
    volSeries.applyOptions({ visible: showVol() });
  });
  $("#bt-ch-sess")?.addEventListener("change", () => {
    window._btChartCache = null;
    btMountPriceChart(p);
  });
  host.addEventListener("keydown", ev => {
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      ev.preventDefault();
      const pos = chart.timeScale().scrollPosition();
      chart.timeScale().scrollToPosition(pos + (ev.key === "ArrowLeft" ? 12 : -12), false);
    } else if (ev.key === "[" || ev.key === "p") jumpTrade(curTrade - 1);
    else if (ev.key === "]" || ev.key === "n") jumpTrade(curTrade + 1);
  });

  window._btLwChart = chart;
  window._btLwState = { data, ro };
  requestAnimationFrame(drawCrosses);
}

/* Backtest summary P/L chart: trading-day axis, day vs trade, line and/or bars. */
let _cmeHolidays = null;
async function loadCmeHolidays() {
  if (_cmeHolidays) return _cmeHolidays;
  try {
    const d = await api("/api/calendar/cme");
    _cmeHolidays = new Set(d.holidays || []);
  } catch {
    _cmeHolidays = new Set();
  }
  return _cmeHolidays;
}
function ctYmd(ts) {
  const s = toCT(ts);
  return s ? s.slice(0, 10) : null;
}
function civilWeekday(ymd) {
  const [y, m, d] = String(ymd).split("-").map(Number);
  if (!y || !m || !d) return 0;
  return new Date(Date.UTC(y, m - 1, d)).getUTCDay();
}
function shiftCivil(ymd, days) {
  const [y, m, d] = String(ymd).split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  dt.setUTCDate(dt.getUTCDate() + days);
  return dt.toISOString().slice(0, 10);
}
function toTradingYmd(ymd, holidays) {
  if (!ymd) return ymd;
  const closed = d => civilWeekday(d) === 0 || civilWeekday(d) === 6 || (holidays && holidays.has(d));
  let d = ymd;
  for (let i = 0; i < 14 && closed(d); i++) d = shiftCivil(d, -1);
  if (!closed(d)) return d;
  d = ymd;
  for (let i = 0; i < 14 && closed(d); i++) d = shiftCivil(d, 1);
  return d;
}
function btPnlSeries(p, level, holidays) {
  const trades = (p.trades || []).filter(t => t && t.exit_ts != null && Number.isFinite(Number(t.pnl_points)));
  if (!trades.length) return [];
  if (level === "trade") {
    let cum = 0;
    return trades.map((t, i) => {
      const delta = Number(t.pnl_points) || 0;
      cum += delta;
      const ymd = ctYmd(t.exit_ts) || ctYmd(t.entry_ts) || "";
      return {
        key: `t${i}`,
        ymd,
        t: t.exit_ts,
        label: (_btTs(t.exit_ts) || ymd).slice(5, 16),
        delta,
        gain: delta > 0 ? delta : 0,
        loss: delta < 0 ? delta : 0,
        n: 1,
        n_win: delta > 0 ? 1 : 0,
        n_loss: delta < 0 ? 1 : 0,
        cum,
      };
    });
  }
  const buckets = new Map();
  for (const t of trades) {
    const raw = ctYmd(t.exit_ts) || t.session_date || ctYmd(t.entry_ts);
    const ymd = toTradingYmd(raw, holidays);
    if (!ymd) continue;
    const rec = buckets.get(ymd) || {
      delta: 0, gain: 0, loss: 0, n: 0, n_win: 0, n_loss: 0, t: t.exit_ts,
    };
    const pnl = Number(t.pnl_points) || 0;
    rec.delta += pnl;
    if (pnl > 0) { rec.gain += pnl; rec.n_win += 1; }
    else if (pnl < 0) { rec.loss += pnl; rec.n_loss += 1; }
    rec.n += 1;
    rec.t = t.exit_ts;
    buckets.set(ymd, rec);
  }
  const days = [...buckets.keys()].sort();
  let cum = 0;
  return days.map(ymd => {
    const rec = buckets.get(ymd);
    cum += rec.delta;
    return {
      key: ymd,
      ymd,
      t: rec.t,
      label: ymd.slice(5),
      delta: rec.delta,
      gain: rec.gain,
      loss: rec.loss,
      n: rec.n,
      n_win: rec.n_win,
      n_loss: rec.n_loss,
      cum,
    };
  });
}
function _niceNum(range, round) {
  const exp = Math.floor(Math.log10(Math.max(range, 1e-9)));
  const f = range / Math.pow(10, exp);
  let nf;
  if (round) {
    if (f < 1.5) nf = 1;
    else if (f < 3) nf = 2;
    else if (f < 7) nf = 5;
    else nf = 10;
  } else {
    if (f <= 1) nf = 1;
    else if (f <= 2) nf = 2;
    else if (f <= 5) nf = 5;
    else nf = 10;
  }
  return nf * Math.pow(10, exp);
}
/** Inclusive nice ticks through 0; `min`/`max` snapped to the outer ticks. */
function niceAxis(min, max, count = 5) {
  let lo = Number(min), hi = Number(max);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  if (lo > 0) lo = 0;
  if (hi < 0) hi = 0;
  if (lo === hi) { lo -= 1; hi += 1; }
  const range = _niceNum(hi - lo, false);
  const step = _niceNum(range / Math.max(1, count - 1), true);
  const tMin = Math.floor(lo / step) * step;
  const tMax = Math.ceil(hi / step) * step;
  const ticks = [];
  const n = Math.max(1, Math.round((tMax - tMin) / step));
  for (let i = 0; i <= n; i++) ticks.push(tMin + i * step);
  return { min: tMin, max: tMax, step, ticks };
}
function fmtPnlTick(v, unit, step) {
  const n = _numFinite(v);
  if (n == null) return "—";
  const z = Math.abs(n) < (step || 1) * 1e-9 ? 0 : n;
  if (unit === "usd") return fmtUsd(z, (step || 1) >= 1 ? 0 : 2);
  const d = (step || 1) >= 1 ? 0 : ((step || 1) >= 0.1 ? 1 : 2);
  return fmtNum(z, d);
}
function btWirePnlChart(p, pointValue) {
  const host = $("#bt-chart");
  if (!host) return;
  const pv = pointValue != null ? pointValue : DEFAULT_POINT_VALUE;
  const levelDay = localStorage.getItem("bt_pnl_level") !== "trade";
  const wantLine0 = localStorage.getItem("bt_pnl_line") !== "0";
  const wantBars0 = localStorage.getItem("bt_pnl_bars") !== "0";
  const unitUsd = localStorage.getItem("bt_pnl_unit") === "usd";
  const split0 = localStorage.getItem("bt_pnl_barmode") === "split";
  const dayEl = document.querySelector('input[name="bt-pnl-level"][value="day"]');
  const tradeEl = document.querySelector('input[name="bt-pnl-level"][value="trade"]');
  const ptEl = document.querySelector('input[name="bt-pnl-unit"][value="pt"]');
  const usdEl = document.querySelector('input[name="bt-pnl-unit"][value="usd"]');
  const netEl = document.querySelector('input[name="bt-pnl-barmode"][value="net"]');
  const splitEl = document.querySelector('input[name="bt-pnl-barmode"][value="split"]');
  const lineEl = $("#bt-pnl-line");
  const barsEl = $("#bt-pnl-bars");
  const barModeGrp = $("#bt-pnl-barmode-grp");
  if (dayEl) dayEl.checked = levelDay;
  if (tradeEl) tradeEl.checked = !levelDay;
  if (ptEl) ptEl.checked = !unitUsd;
  if (usdEl) usdEl.checked = unitUsd;
  if (netEl) netEl.checked = !split0;
  if (splitEl) splitEl.checked = split0;
  if (lineEl) lineEl.checked = wantLine0;
  if (barsEl) barsEl.checked = wantBars0;

  const draw = (holidays) => {
    let level = (document.querySelector('input[name="bt-pnl-level"]:checked') || {}).value || "day";
    let unit = (document.querySelector('input[name="bt-pnl-unit"]:checked') || {}).value || "pt";
    let barMode = (document.querySelector('input[name="bt-pnl-barmode"]:checked') || {}).value || "net";
    let showLine = !lineEl || lineEl.checked;
    let showBars = !barsEl || barsEl.checked;
    if (!showLine && !showBars) {
      showLine = true;
      if (lineEl) lineEl.checked = true;
    }
    if (barModeGrp) barModeGrp.style.display = showBars ? "flex" : "none";
    localStorage.setItem("bt_pnl_level", level);
    localStorage.setItem("bt_pnl_unit", unit);
    localStorage.setItem("bt_pnl_barmode", barMode);
    localStorage.setItem("bt_pnl_line", showLine ? "1" : "0");
    localStorage.setItem("bt_pnl_bars", showBars ? "1" : "0");
    const series = btPnlSeries(p, level, holidays);
    const splitBars = showBars && barMode === "split";
    const note = $("#bt-pnl-note");
    if (note) {
      const axis = level === "day"
        ? "One slot per CME session (weekends and CME holidays omitted)."
        : "One slot per closed trade (equal spacing — no weekend or holiday gaps).";
      const bars = !showBars ? ""
        : (splitBars
          ? " Bars: green = sum of winning trades, red = sum of losing trades in that slot."
          : " Bars: net P/L of the slot.");
      note.textContent = axis + bars;
    }
    const title = $("#bt-pnl-title");
    if (title) title.textContent = unit === "usd" ? "P/L over time ($)" : "P/L over time (points)";
    if (!series.length) {
      host.innerHTML = '<div class="muted">not enough trades to chart</div>';
      return;
    }
    const scale = v => unit === "usd" ? v * pv : v;
    const pts = series.map(pt => ({
      ...pt,
      delta: scale(pt.delta),
      gain: scale(pt.gain || 0),
      loss: scale(pt.loss || 0),
      cum: scale(pt.cum),
    }));
    const W = 920, H = 336, mL = unit === "usd" ? 78 : 58, mR = 16, mT = 18, mB = 40;
    const n = pts.length;
    const plotW = W - mL - mR;
    const plotH = H - mT - mB;
    const slot = plotW / Math.max(n, 1);
    const barW = Math.max(1.5, Math.min(18, slot * 0.72));
    const xAt = i => mL + slot * (i + 0.5);
    const yVals = [];
    for (const pt of pts) {
      if (showLine) yVals.push(pt.cum);
      if (showBars && splitBars) {
        yVals.push(pt.gain);
        yVals.push(pt.loss);
      } else if (showBars) {
        yVals.push(pt.delta);
      }
    }
    yVals.push(0);
    const axis = niceAxis(Math.min(...yVals), Math.max(...yVals), 5);
    const py = v => mT + (1 - (v - axis.min) / ((axis.max - axis.min) || 1)) * plotH;
    const y0 = py(0);
    const plotBottom = mT + plotH;
    const net = pts[pts.length - 1].cum;
    const hex = net >= 0 ? "#7ee0a3" : "#ff6b6b";
    let grid = `<line class="axis" x1="${mL}" y1="${mT}" x2="${mL}" y2="${plotBottom}"/>`
      + `<line class="axis" x1="${mL}" y1="${plotBottom}" x2="${W - mR}" y2="${plotBottom}"/>`;
    for (const v of axis.ticks) {
      const y = py(v);
      grid += `<line class="grid" x1="${mL}" y1="${y.toFixed(1)}" x2="${W - mR}" y2="${y.toFixed(1)}"/>`
        + `<text class="tick" x="${mL - 8}" y="${y.toFixed(1)}" text-anchor="end" dominant-baseline="middle">${esc(fmtPnlTick(v, unit, axis.step))}</text>`;
    }
    const unitLab = unit === "usd" ? "$" : "pt";
    grid += `<text class="tick" x="${mL - 8}" y="${mT - 8}" text-anchor="end">${unitLab}</text>`;
    const zero = `<line class="zero" x1="${mL}" y1="${y0.toFixed(1)}" x2="${W - mR}" y2="${y0.toFixed(1)}"/>`;
    const barRect = (x, value, cls) => {
      if (!value) return "";
      const y1 = py(value);
      const top = Math.min(y0, y1);
      const h = Math.max(1.2, Math.abs(y1 - y0));
      return `<rect class="${cls}" x="${(x - barW / 2).toFixed(1)}" y="${top.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}"/>`;
    };
    let bars = "";
    if (showBars) {
      for (let i = 0; i < n; i++) {
        const pt = pts[i];
        const x = xAt(i);
        if (splitBars) {
          bars += barRect(x, pt.gain, "bar-gain");
          bars += barRect(x, pt.loss, "bar-loss");
        } else {
          bars += barRect(x, pt.delta, pt.delta >= 0 ? "bar-gain" : "bar-loss");
        }
      }
    }
    let line = "", area = "";
    if (showLine) {
      line = pts.map((pt, i) => `${i ? "L" : "M"}${xAt(i).toFixed(1)},${py(pt.cum).toFixed(1)}`).join(" ");
      area = `M${xAt(0).toFixed(1)},${y0.toFixed(1)} `
        + pts.map((pt, i) => `L${xAt(i).toFixed(1)},${py(pt.cum).toFixed(1)}`).join(" ")
        + ` L${xAt(n - 1).toFixed(1)},${y0.toFixed(1)} Z`;
    }
    const minLabelPx = level === "day" ? 40 : 48;
    const stride = Math.max(1, Math.ceil(minLabelPx / Math.max(slot, 1)));
    const tickIdx = [];
    for (let i = 0; i < n; i += stride) tickIdx.push(i);
    if (tickIdx[tickIdx.length - 1] !== n - 1) tickIdx.push(n - 1);
    const xlab = tickIdx.map(i => {
      const pt = pts[i];
      const lab = level === "day" ? pt.ymd.slice(5) : (pt.ymd ? pt.ymd.slice(5) : pt.label);
      if (!lab) return "";
      const x = xAt(i);
      return `<line class="axis" x1="${x.toFixed(1)}" y1="${plotBottom}" x2="${x.toFixed(1)}" y2="${(plotBottom + 5).toFixed(1)}"/>`
        + `<text class="tick" x="${x.toFixed(1)}" y="${(plotBottom + 18).toFixed(1)}" text-anchor="middle">${esc(lab)}</text>`;
    }).join("");
    const aria = [level === "day" ? "daily" : "per-trade",
      unit === "usd" ? "dollars" : "points",
      showLine ? "cumulative line" : "",
      showBars ? (splitBars ? "split gain and loss bars" : "net gain/loss bars") : ""]
      .filter(Boolean).join(" ");
    host.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="P/L ${esc(aria)}">
        <defs><linearGradient id="btgrad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="${hex}" stop-opacity="0.28"/>
          <stop offset="1" stop-color="${hex}" stop-opacity="0.02"/></linearGradient></defs>
        ${grid}${zero}
        ${showLine ? `<path d="${area}" fill="url(#btgrad)" stroke="none"/>` : ""}
        ${bars}
        ${showLine ? `<path d="${line}" fill="none" stroke="${hex}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` : ""}
        ${xlab}
        <line class="cross" id="bt-cross" y1="${mT}" y2="${plotBottom}"/>
        <circle id="bt-dot" r="3.5" fill="${hex}" opacity="0"/>
        <rect id="bt-hit" x="${mL}" y="${mT}" width="${plotW}" height="${plotH}" fill="transparent" style="cursor:crosshair"/>
      </svg>
      <div class="chart-tip" id="bt-tip"></div>`;
    const svg = host.querySelector("svg"), hit = host.querySelector("#bt-hit"),
      cross = host.querySelector("#bt-cross"), dot = host.querySelector("#bt-dot"), tip = host.querySelector("#bt-tip");
    hit.addEventListener("mousemove", ev => {
      const r = svg.getBoundingClientRect();
      const plotFrac = ((ev.clientX - r.left) / r.width * W - mL) / (plotW || 1);
      let bi = Math.floor(plotFrac * n);
      if (bi < 0) bi = 0;
      if (bi > n - 1) bi = n - 1;
      const pt = pts[bi];
      const X = xAt(bi);
      const Y = py(showLine ? pt.cum : (splitBars
        ? (Math.abs(pt.gain) >= Math.abs(pt.loss) ? pt.gain : pt.loss)
        : pt.delta));
      cross.setAttribute("x1", X); cross.setAttribute("x2", X); cross.style.opacity = 1;
      dot.setAttribute("cx", X); dot.setAttribute("cy", Y); dot.style.opacity = 1;
      tip.style.left = (X / W * r.width) + "px"; tip.style.top = (Y / H * r.height) + "px"; tip.style.opacity = 1;
      const when = level === "day" ? pt.ymd : _btTs(pt.t);
      const extra = ` · ${pt.n} trade${pt.n === 1 ? "" : "s"}`;
      const fmtV = v => unit === "usd" ? fmtUsd(v, 0) : `${fmtPts(v, 1)} pt`;
      const dlt = fmtV(pt.delta);
      const cum = fmtV(pt.cum);
      let body = `${esc(when)}${extra}<br>`;
      if (splitBars) {
        body += `<span class="pos">${fmtV(pt.gain)}</span> gain (${fmtInt(pt.n_win || 0)})<br>`
          + `<span class="neg">${fmtV(pt.loss)}</span> loss (${fmtInt(pt.n_loss || 0)})<br>`
          + `<span class="${pt.delta >= 0 ? "pos" : "neg"}">${dlt}</span> net<br>`;
      } else {
        body += `<span class="${pt.delta >= 0 ? "pos" : "neg"}">${pt.delta >= 0 ? "+" : ""}${dlt}</span> period<br>`;
      }
      body += `<b class="${pt.cum >= 0 ? "pos" : "neg"}">${cum}</b> cumulative`;
      tip.innerHTML = body;
    });
    hit.addEventListener("mouseleave", () => { cross.style.opacity = 0; dot.style.opacity = 0; tip.style.opacity = 0; });
  };

  const bind = (holidays) => {
    const redraw = () => draw(holidays);
    document.querySelectorAll('input[name="bt-pnl-level"]').forEach(el => {
      el.onchange = redraw;
    });
    document.querySelectorAll('input[name="bt-pnl-unit"]').forEach(el => {
      el.onchange = redraw;
    });
    document.querySelectorAll('input[name="bt-pnl-barmode"]').forEach(el => {
      el.onchange = redraw;
    });
    if (lineEl) lineEl.onchange = redraw;
    if (barsEl) barsEl.onchange = redraw;
    draw(holidays);
  };
  loadCmeHolidays().then(h => {
    if (!$("#bt-chart")) return;
    bind(h);
  });
}

/* ── live streamer + model prediction (5s poll) ─────────────────────────── */
function viewLiveStreamer() {
  /* Clean Live UI: one model control, one stats strip, one blotter that rebinds on Apply. */
  const LEDGER_UI_CAP = 500;
  const FILTERS = ["signals", "directional", "all"];
  const num = (x, d = 3) => fmtNum(x, d);
  const sgn = (v, d = 2) => fmtPts(v, d);
  const livePv = () => {
    const st = lastStats || {};
    const s = lastS || {};
    return resolvePointValue({
      point_value: st.point_value ?? s.point_value
        ?? (s.trade && s.trade.point_value),
    });
  };
  const sgnUsd = (v, d = 2) => fmtPtsUsd(v, livePv(), d);

  let tau = parseFloat(localStorage.getItem("live_tau"));
  if (!Number.isFinite(tau)) tau = 0.8;
  let filter = localStorage.getItem("live_blotter_filter") || "directional";
  if (!FILTERS.includes(filter)) filter = "directional";
  /* TBBO DATA on → ticks path; off → bar (dataset default, currently 5s close) */
  let liveData = readTbboDataPref("tbbo_data");
  let fillMode = liveData ? "ticks" : "bar";

  let lastS = null, lastL = null, lastModel = null, lastStats = null;
  let applying = false;
  let loadingTicks = false;
  let draftPath = null;   // dropdown value before Apply
  let refreshInFlight = false;
  let streamInFlight = false;
  let statusLine = "";

  const gated = (dir, conf) => dir !== "flat" && conf >= tau;
  const setLiveData = (on) => {
    liveData = !!on;
    fillMode = liveData ? "ticks" : "bar";
    writeTbboDataPref("tbbo_data", liveData);
  };

  const statsFromLedger = (L, s) => {
    const rows = (L && L.rows) || [];
    const meta = (L && L.meta) || {};
    const tickMode = (meta.mode === "ticks") || fillMode === "ticks";
    let n_up = 0, n_down = 0, n_flat = 0, n_sig = 0, n_win = 0, n_stop = 0, n_to = 0, net = 0;
    for (const r of rows) {
      if (r.dir === "up") n_up++;
      else if (r.dir === "down") n_down++;
      else n_flat++;
      const conf = typeof r.conf === "number" ? r.conf : 0;
      const isTrade = tickMode
        ? (r.sig && !r.pending && r.outcome !== "working_limit")
        : (gated(r.dir, conf) && !r.pending);
      if (isTrade) {
        n_sig++;
        if (r.outcome === "target") n_win++;
        else if (r.outcome === "stop") n_stop++;
        else if (r.outcome === "timeout" || r.outcome === "time" || r.outcome === "eod") n_to++;
        if (typeof r.pnl === "number") net += r.pnl;
      }
    }
    return {
      candidate_id: (s && s.candidate_id) || meta.candidate_id || "—",
      model_path: (s && s.model_path) || meta.model_path || "",
      session_date: (s && s.session && s.session.date) || meta.session && meta.session.date || "—",
      built_at: (s && s.built_at) || meta.built_at,
      tau,
      n_bars: rows.length,
      n_up, n_down, n_flat,
      n_directional: n_up + n_down,
      n_signals: n_sig, n_win, n_stop, n_timeout: n_to,
      net_pnl: Math.round(net * 100) / 100,
      move_threshold: (s && s.barriers && s.barriers.move_threshold) || 50,
      mae_tolerance: (s && s.barriers && s.barriers.mae_tolerance) || 6,
      point_value: (s && s.point_value) || (meta && meta.point_value) || DEFAULT_POINT_VALUE,
      instrument: (s && s.instrument) || "NQ",
    };
  };

  window._liveTau = async (v) => {
    tau = parseFloat(v);
    localStorage.setItem("live_tau", String(tau));
    document.activeElement?.blur?.();
    if (fillMode === "ticks") {
      await loadLedger(true);
    } else {
      lastStats = statsFromLedger(lastL, lastS);
      paint();
    }
  };
  window._liveFilter = (v) => {
    filter = v;
    localStorage.setItem("live_blotter_filter", v);
    document.activeElement?.blur?.();
    paint();
  };
  window._liveToggleLiveData = async (on) => {
    const next = !!on;
    const nextMode = next ? "ticks" : "bar";
    if (next === liveData && lastL && lastL.meta && lastL.meta.mode === nextMode) {
      paint();
      return;
    }
    setLiveData(next);
    document.activeElement?.blur?.();
    // Live TBBO mode is trade-only — default filter to signals
    if (liveData && filter === "all") {
      filter = "signals";
      localStorage.setItem("live_blotter_filter", filter);
    }
    await loadLedger(true);
    paint();
  };
  window._liveApplyModel = async () => {
    if (applying) return;
    const sel = $("#live-model-sel");
    if (!sel) return;
    applying = true;
    statusLine = "Rescoring session day with selected model…";
    paint();
    try {
      const v = sel.value;
      draftPath = v;
      const body = v === "__auto__" ? {mode: "auto"} : {mode: "manual", path: v};
      const res = await post("/api/live/model", body);
      lastModel = res;
      if (res.streamer) lastS = res.streamer;
      if (res.ledger) lastL = res.ledger;
      // Keep TBBO view in sync when operator is on tick fills.
      if (fillMode === "ticks") await loadLedger(true);
      lastStats = statsFromLedger(lastL, lastS);
      if (!Number.isFinite(tau) && lastS && lastS.tau != null) tau = lastS.tau;
      draftPath = null;
      if (res.snapshot_refresh === "scheduled") {
        statusLine = "Model selection saved. Snapshot refresh scheduled…";
      } else {
        const id = lastStats.candidate_id || "model";
        statusLine = `Applied ${id} · ${fmtInt(lastStats.n_bars)} bars · `
          + `${fmtInt(lastStats.n_up)}↑ ${fmtInt(lastStats.n_down)}↓ ${fmtInt(lastStats.n_flat)} flat · `
          + `${fmtInt(lastStats.n_signals)} signals @ τ=${tau.toFixed(2)} · net ${sgnUsd(lastStats.net_pnl)}`
          + (liveData ? " · TBBO DATA" : "");
      }
      toast(statusLine, "ok", 8000);
      paint();
    } catch (e) {
      statusLine = `Apply failed: ${e.message}`;
      toast(e.message, "err");
      paint();
    } finally {
      applying = false;
      paint();
    }
  };

  const modelOptions = () => {
    const m = lastModel || {};
    const avail = m.available || [];
    const sel = m.selection || {mode: "auto"};
    const resolved = m.resolved || {};
    const pin = m.active_objective || (lastS && lastS.active_objective) || "—";
    const cur = resolved.path || (lastS && lastS.model_path) || "";
    const mode = sel.mode || "auto";
    const want = draftPath || (mode === "manual" ? (sel.path || cur) : "__auto__");

    const byObj = {};
    for (const b of avail) {
      const k = b.objective_id || "other";
      (byObj[k] = byObj[k] || []).push(b);
    }
    let opts = `<option value="__auto__"${want === "__auto__" ? " selected" : ""}>`
      + `Auto (newest for ${esc(pin)})</option>`;
    for (const g of Object.keys(byObj).sort()) {
      opts += `<optgroup label="${esc(g)}">`;
      for (const b of byObj[g]) {
        const lab = b.run_label || bundleRunLabel(b.path) || b.candidate_id || b.name;
        const mark = b.path === cur ? " ●" : "";
        opts += `<option value="${String(b.path).replace(/"/g, "&quot;")}"${want === b.path ? " selected" : ""}>`
          + `${esc(lab)} · ${esc(b.objective_id || "")}${mark}</option>`;
      }
      opts += `</optgroup>`;
    }
    if (!avail.length) {
      opts += `<option disabled>${lastModel ? "no models in staging" : "loading…"}</option>`;
    }
    return opts;
  };

  const filterBar = () => {
    const labels = fillMode === "ticks"
      ? {signals: "Filled trades", directional: "Trades + open", all: "All rows"}
      : {signals: "Signals only", directional: "Up/Down only", all: "All bars"};
    return FILTERS.map(f => {
      const on = filter === f;
      return `<button type="button" class="small${on ? " primary" : ""}" onclick="window._liveFilter('${f}')">${labels[f]}</button>`;
    }).join(" ");
  };

  const liveDataToggle = () => tbboDataSwitchHtml(
    "live-tbbo-data-toggle",
    liveData,
    "window._liveToggleLiveData(this.checked)",
    liveData
      ? "ON: TBBO DATA (day blotter tick path)"
      : "OFF: default path for this dataset (currently 5s feature close first-touch)",
  );

  const tauSelect = () => {
    let opts = "";
    for (let v = 0.50; v <= 0.951; v += 0.05) {
      const t = v.toFixed(2);
      opts += `<option value="${t}"${Math.abs(v - tau) < 0.001 ? " selected" : ""}>${t}</option>`;
    }
    return `<select id="live-tau-sel" onchange="window._liveTau(this.value)"
      style="width:auto;padding:2px 8px;font-family:var(--mono)">${opts}</select>`;
  };

  const renderBlotter = () => {
    const rows = (lastL && lastL.rows) || [];
    const st = lastStats || statsFromLedger(lastL, lastS);
    const tgt = st.move_threshold ?? 50, stp = st.mae_tolerance ?? 6;
    const sess = (lastS && lastS.session) || (lastL && lastL.meta && lastL.meta.session) || {};
    const meta = (lastL && lastL.meta) || {};
    const pathLabel = liveData
      ? "TBBO DATA · tick path"
      : "TBBO DATA off · default path (5s close)";

    let filtered = rows;
    if (filter === "signals") {
      filtered = rows.filter(r => {
        if (fillMode === "ticks") return r.sig && !r.pending && r.outcome !== "working_limit";
        return gated(r.dir, r.conf) && !r.pending;
      });
    } else if (filter === "directional") {
      filtered = rows.filter(r => r.dir === "up" || r.dir === "down");
    }
    const shown = filtered.slice(0, LEDGER_UI_CAP);

    const body = shown.map(r => {
      const d = r.dir || "flat";
      const dcls = d === "up" ? "pos" : d === "down" ? "neg" : "muted";
      const arrow = d === "up" ? "▲" : d === "down" ? "▼" : "■";
      const conf = typeof r.conf === "number" ? r.conf : 0;
      const isSig = fillMode === "ticks"
        ? !!r.sig
        : (gated(d, conf) && !r.pending);
      const sig = isSig && fillMode !== "ticks" ? ` <span class="pill ok">SIG</span>` : "";
      let result = `<span class="muted">—</span>`;
      if (r.pending || r.outcome === "working_limit") {
        result = r.outcome === "working_limit"
          ? `<span class="muted">working limit</span>`
          : `<span class="muted">open / pending</span>`;
      } else if (r.outcome === "target")
        result = `<span class="pos">WIN ${sgnUsd(r.pnl)}</span>`;
      else if (r.outcome === "stop")
        result = `<span class="neg">STOP ${sgnUsd(r.pnl)}</span>`;
      else if (r.outcome === "timeout" || r.outcome === "time")
        result = `<span class="muted">T/O ${sgnUsd(r.pnl)}</span>`;
      else if (r.outcome === "eod")
        result = `<span class="muted">EOD ${sgnUsd(r.pnl)}</span>`;
      const entryPx = r.entry_px != null ? r.entry_px : r.entry;
      const exitPx = r.exit_px;
      const hold = r.hold_s != null ? `${fmtNum(r.hold_s, 0)}s` : "—";
      return `<tr>
        <td class="mono">${r.ts ? fmtTs(r.ts) : "—"}</td>
        <td class="${dcls}">${arrow} ${d.toUpperCase()} <span class="mono muted">${conf ? num(conf, 3) : "—"}</span>${sig}</td>
        <td class="mono">${entryPx != null ? fmtNum(entryPx, 2) : "—"}</td>
        <td class="mono">${r.exit_ts ? fmtTs(r.exit_ts) : "—"}</td>
        <td class="mono">${exitPx != null ? fmtNum(exitPx, 2) : "—"}</td>
        <td class="mono muted">${hold}</td>
        <td class="num">${r.n_present != null ? fmtInt(r.n_present) : "—"}</td>
        <td class="num">${r.n_total != null ? fmtInt(r.n_total) : "—"}</td>
        <td>${result}</td></tr>`;
    }).join("");

    const emptyMsg = loadingTicks
      ? "Building TBBO DATA blotter…"
      : filter === "signals"
        ? (liveData
          ? `No filled trades at τ=${tau.toFixed(2)} (TBBO DATA).`
          : `No signals at τ=${tau.toFixed(2)}. Try lower τ or "Up/Down only".`)
        : filter === "directional"
          ? "No directional rows for this model/day."
          : "No rows loaded.";

    return `
      <div class="card" style="margin-top:12px">
        <div class="row" style="justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap">
          <h3 id="live-blotter-title" style="margin:0">Day blotter · ${esc(bundleRunLabel(st.model_path || s.model_path) || st.candidate_id || "—")} · ${esc(String(st.session_date || "—"))}</h3>
          <div class="row" style="gap:10px;align-items:center;flex-wrap:wrap">
            ${liveDataToggle()}
            <span class="muted">|</span>
            ${filterBar()}
            <span class="muted" style="margin-left:4px">τ ${tauSelect()}</span>
          </div>
        </div>
        <div id="live-blotter-summary" class="muted" style="margin:8px 0">
          <span class="pill ${liveData ? "ok" : "dim"}">${esc(pathLabel)}</span>
          · Barriers +${fmtNum(tgt, 0)}/−${fmtNum(stp, 0)} · ${fmtInt(st.n_signals)} signals
          (${fmtInt(st.n_win)}W / ${fmtInt(st.n_stop)}S / ${fmtInt(st.n_timeout)}T) · net
          <span class="${st.net_pnl >= 0 ? "pos" : "neg"}">${sgnUsd(st.net_pnl)}</span>
          · showing ${fmtInt(shown.length)}${filtered.length > LEDGER_UI_CAP ? ` of ${fmtInt(filtered.length)}` : ""}
          ${sess.session_closed ? " · after hours" : ""}
          · ${esc(st.instrument || "NQ")} @ ${fmtUsd(livePv(), 0)}/pt
          ${meta.note && fillMode === "ticks" ? ` · ${esc(meta.note)}` : ""}
          ${loadingTicks ? " · <b>loading…</b>" : ""}
        </div>
        <div id="ledger-scroll" style="max-height:55vh;overflow:auto">
          <table>
            <thead><tr>
              <th>entry time (${TZ_ABBR})</th>
              <th>prediction</th>
              <th class="num">entry px</th>
              <th>exit time (${TZ_ABBR})</th>
              <th class="num">exit px</th>
              <th>hold</th>
              <th class="num">feat avail</th>
              <th class="num">feat total</th>
              <th>result</th>
            </tr></thead>
            <tbody id="live-blotter-body">${body || `<tr><td colspan="9" class="empty">${emptyMsg}</td></tr>`}</tbody>
          </table>
        </div>
      </div>`;
  };

  const lagCls = (sec) => {
    if (sec == null || !Number.isFinite(sec)) return "muted";
    if (sec <= 5) return "pos";
    if (sec <= 30) return "muted";
    if (sec <= 90) return "warnc";
    return "neg";
  };
  const fmtLag = (sec) => {
    if (sec == null || !Number.isFinite(Number(sec))) return "—";
    const v = Number(sec);
    if (Math.abs(v) < 10) return `${v.toFixed(1)}s`;
    return `${Math.round(v)}s`;
  };

  const renderPred = () => {
    const s = lastS || {};
    const pred = s.prediction;
    const lags = s.lags || {};
    if (!pred) {
      return `<div class="card"><h3>Latest bar</h3>
        <div class="empty">${esc(s.note || s.status || "waiting for features…")}</div></div>`;
    }
    const d = pred.direction || "flat";
    const cls = d === "up" ? "pos" : d === "down" ? "neg" : "muted";
    const arrow = d === "up" ? "▲" : d === "down" ? "▼" : "■";
    const conf = typeof pred.confidence === "number" ? pred.confidence : 0;
    const sig = gated(d, conf) ? `<span class="pill ok">SIGNAL</span>` : `<span class="pill dim">no trade</span>`;
    const pLag = lags.parquet_lag_s;
    const prLag = lags.prediction_lag_s;
    const tot = lags.total_lag_s != null
      ? lags.total_lag_s
      : (pLag != null && prLag != null ? Number(pLag) + Number(prLag) : null);
    return `<div class="card"><h3>Latest bar</h3>
      <div class="tiles">
        <div class="tile"><div class="lbl">prediction</div>
          <div class="val ${cls}">${arrow} ${d.toUpperCase()}</div>
          <div class="sub">conf ${num(conf, 3)} ${sig}</div></div>
        <div class="tile"><div class="lbl">features</div>
          <div class="val">${pred.n_present ?? "—"}/${s.n_total ?? "—"}</div>
          <div class="sub">${pred.n_nan ?? 0} NaN</div></div>
        <div class="tile" title="Bar data available (close) → feature parquet row written">
          <div class="lbl">parquet lag</div>
          <div class="val ${lagCls(pLag)}">${fmtLag(pLag)}</div>
          <div class="sub">data → features</div></div>
        <div class="tile" title="Feature parquet ready → model prediction committed">
          <div class="lbl">prediction lag</div>
          <div class="val ${lagCls(prLag)}">${fmtLag(prLag)}</div>
          <div class="sub">features → model</div></div>
        <div class="tile" title="Bar data available → model prediction">
          <div class="lbl">total lag</div>
          <div class="val ${lagCls(tot)}">${fmtLag(tot)}</div>
          <div class="sub">data → model</div></div>
      </div>
      <div class="muted" style="margin-top:8px">${fmtTs(pred.bar_ts)} · ${pred.n_rows ?? "—"} bars scored
        ${lags.feature_compute_s != null ? ` · compute ${fmtLag(lags.feature_compute_s)}` : ""}
        ${lags.n_appended != null ? ` · +${fmtInt(lags.n_appended)} rows` : ""}
      </div>
    </div>`;
  };

  const healthPill = (h) => {
    const v = (h || "unknown").toLowerCase();
    if (v === "healthy" || v === "ok") return `<span class="pill ok">● healthy</span>`;
    if (v === "degraded" || v === "warming") return `<span class="pill warn">● ${esc(v)}</span>`;
    if (v === "error" || v === "stale") return `<span class="pill err">● ${esc(v)}</span>`;
    return `<span class="pill dim">○ ${esc(v)}</span>`;
  };

  const tableStatusPill = (st) => {
    const v = (st || "unknown").toLowerCase();
    if (v === "healthy") return `<span class="pill ok">${esc(v)}</span>`;
    if (v === "degraded") return `<span class="pill warn">${esc(v)}</span>`;
    if (v === "stale" || v === "error") return `<span class="pill err">${esc(v)}</span>`;
    return `<span class="pill dim">${esc(v)}</span>`;
  };

  const renderStreamStatus = () => {
    // Compact stream diagnostics: overall health lives in the page header.
    // Keep process/feed/disk tiles + ingest table; park ops noise under <details>.
    const sh = (lastS && lastS.streamer) || {};
    const proc = sh.process || (sh.processes && sh.processes[0]) || null;
    const sd = sh.systemd || {};
    const conns = sh.connections || [];
    const raw = sh.raw || {};
    const disk = sh.disk || {};
    const detail = sh.table_detail || {};
    const tableKeys = ["tbbo", "bbo_1s", "statistics"];
    const notes = Array.isArray(sh.notes) ? sh.notes : [];
    const noteLine = sh.error || (notes.length ? notes.slice(0, 3).join("; ") : "");

    const connSummary = (() => {
      const roles = conns.map(c => c.role || "other");
      const nDb = roles.filter(r => r === "databento_live").length;
      const nPg = roles.filter(r => r === "postgres").length;
      if (!conns.length) return "no sockets";
      return `dbento ${nDb} · pg ${nPg}`;
    })();

    const tableRows = tableKeys.map(k => {
      const d = detail[k] || {};
      const latest = d.latest || (sh.tables && sh.tables[k]) || null;
      const age = d.age_s != null ? d.age_s : (k === "tbbo" ? sh.tbbo_age_s : null);
      const r10 = d.rows_10s != null ? d.rows_10s : (k === "tbbo" ? sh.tbbo_rows_10s : null);
      const r2m = d.rows_2m;
      return `<tr>
        <td class="mono">${esc(d.name || ("databento." + k))}</td>
        <td>${tableStatusPill(d.status)}</td>
        <td class="mono">${latest ? fmtTs(latest) : "—"}</td>
        <td class="mono num">${age != null ? age + "s" : "—"}</td>
        <td class="mono num">${r10 != null ? fmtInt(r10) : "—"}</td>
        <td class="mono num">${r2m != null ? fmtInt(r2m) : "—"}</td>
      </tr>`;
    }).join("");

    const connRows = conns.length
      ? conns.map(c => {
          const role = c.role || "other";
          const rcls = role === "databento_live" ? "ok" : role === "postgres" ? "ok" : "dim";
          return `<tr>
            <td><span class="pill ${rcls}">${esc(role)}</span></td>
            <td class="mono">${esc(c.local || "—")}</td>
            <td class="mono">${esc(c.remote || "—")}</td>
            <td class="mono">${esc(c.state || "—")}</td>
          </tr>`;
        }).join("")
      : `<tr><td colspan="4" class="empty">No established sockets</td></tr>`;

    const rawFiles = (raw.files || []);
    const rawRows = rawFiles.length
      ? rawFiles.map(f => `<tr>
          <td class="mono">${esc(f.name)}</td>
          <td class="mono num">${f.mb != null ? f.mb + " MB" : "—"}</td>
          <td>${f.compressed ? `<span class="pill ok">zst</span>` : `<span class="pill warn">raw</span>`}</td>
          <td class="mono muted">${f.mtime ? fmtTs(f.mtime) : "—"}</td>
        </tr>`).join("")
      : `<tr><td colspan="4" class="empty">${raw.exists === false ? "today dir missing" : "no files today"}</td></tr>`;

    const openRaw = (proc && proc.open_raw_files) || [];
    const sdActive = sd.active_state || "—";
    const sdPill = sdActive === "active"
      ? `<span class="pill ok">${esc(sdActive)}</span>`
      : sdActive === "—"
        ? `<span class="pill dim">—</span>`
        : `<span class="pill err">${esc(sdActive)}</span>`;
    const streamerVal = proc
      ? `pid ${proc.pid}`
      : (sd.main_pid ? `pid ${sd.main_pid}` : "—");
    const streamerSub = proc
      ? `up ${esc(proc.etime || "—")} · rss ${proc.rss_mb != null ? proc.rss_mb + " MB" : "—"}`
      : (sdActive === "active" ? "process not observed" : "not running");

    return `
      <div class="card" id="live-stream-status" style="margin-bottom:12px">
        <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <h3 style="margin:0">Stream feed</h3>
          <div class="muted mono" style="font-size:12px">
            checked ${sh.checked_at ? fmtTs(sh.checked_at) : "—"}
            ${sh.archive_age_s != null ? ` · archive ${sh.archive_age_s}s` : ""}
            ${sh.dataset ? ` · ${esc(sh.dataset)}` : ""}
          </div>
        </div>
        ${noteLine ? `<div class="muted" style="margin-top:8px;color:var(--warn)">${esc(noteLine)}</div>` : ""}

        <div class="tiles" style="margin-top:12px">
          <div class="tile"><div class="lbl">streamer</div>
            <div class="val" style="font-size:14px">${sdPill} <span class="mono" style="font-weight:500">${streamerVal}</span></div>
            <div class="sub mono">${streamerSub}
              ${sd.memory_mb != null ? ` · unit ${sd.memory_mb} MB` : ""}</div></div>
          <div class="tile"><div class="lbl">tbbo lag</div>
            <div class="val" style="font-size:14px">${sh.tbbo_age_s != null ? sh.tbbo_age_s + "s" : "—"}</div>
            <div class="sub mono">${fmtInt(sh.tbbo_rows_10s || 0)} rows / 10s · ${esc(connSummary)}</div></div>
          <div class="tile"><div class="lbl">disk /opt/vtech/data</div>
            <div class="val" style="font-size:14px">${disk.free_gb != null ? disk.free_gb + " GB free" : "—"}</div>
            <div class="sub mono">${disk.used_pct != null ? disk.used_pct + "% used" : ""}
              ${disk.total_gb != null ? ` · ${disk.total_gb} GB` : ""}</div></div>
        </div>

        <div style="margin-top:14px">
          <h4 style="margin:0 0 6px">Ingest tables</h4>
          <table><thead><tr>
            <th>table</th><th>status</th><th>latest</th><th class="num">age</th>
            <th class="num">rows/10s</th><th class="num">rows/2m</th>
          </tr></thead>
          <tbody id="live-ingest-body">${tableRows}</tbody></table>
        </div>

        <details style="margin-top:12px">
          <summary class="muted" style="cursor:pointer;font-size:13px">Connections &amp; raw archive</summary>
          <div class="grid cols-2" style="margin-top:10px;gap:12px">
            <div>
              <h4 style="margin:0 0 6px">Connections</h4>
              <table><thead><tr><th>role</th><th>local</th><th>remote</th><th>state</th></tr></thead>
              <tbody id="live-conn-body">${connRows}</tbody></table>
              ${openRaw.length ? `<div class="muted" style="margin-top:6px;font-size:12px">open raw:
                ${openRaw.map(f => `<span class="mono">${esc(f)}</span>`).join("<br>")}</div>` : ""}
            </div>
            <div>
              <h4 style="margin:0 0 6px">Raw live archive
                <span class="muted mono" style="font-weight:400;font-size:12px">
                  ${raw.total_gb != null ? raw.total_gb + " GB today" : ""}
                </span>
              </h4>
              <table><thead><tr><th>file</th><th class="num">size</th><th>fmt</th><th>mtime</th></tr></thead>
              <tbody id="live-raw-body">${rawRows}</tbody></table>
            </div>
          </div>
        </details>
      </div>`;
  };

  const paint = () => {
    try {
      const ae = document.activeElement;
      if (ae && ae.id === "live-model-sel") draftPath = ae.value;
      if (ae && (ae.id === "live-tau-sel" || ae.id === "live-tbbo-data-toggle")) return;
      // don't clobber mid-toggle paint

      const s = lastS || {};
      const st = lastStats || statsFromLedger(lastL, lastS);
      const sess = s.session || {};
      const sh = s.streamer || {};
      const health = sh.health || sh.status || "warming";
      const livePill = sh.live
        ? `<span class="pill ok">● feed live</span>` : `<span class="pill dim">○ feed idle</span>`;
      const sessPill = sess.in_session ? `<span class="pill ok">in session</span>`
        : sess.session_closed ? `<span class="pill dim">session closed</span>`
        : `<span class="pill dim">outside hours</span>`;

      const next = document.createElement("template");
      next.innerHTML = `
        <div id="live-header" class="row" style="justify-content:space-between;align-items:center;margin-bottom:10px">
          <div>
            <h2 style="margin:0">Live streamer ${healthPill(health)} ${livePill} ${sessPill}</h2>
            <div class="muted">Session <span class="mono">${esc(String(st.session_date || sess.date || "—"))}</span>
              ${sess.trading_start ? `· ${esc(sess.trading_start)}–${esc(sess.trading_end)} ${esc(sess.timezone || "")}` : ""}</div>
          </div>
        </div>

        <div id="live-stream-status-slot">${renderStreamStatus()}</div>

        <div class="card" id="live-model-card">
          <h3 style="margin:0 0 8px">Model for this blotter</h3>
          <div class="row" style="gap:10px;align-items:flex-end;flex-wrap:wrap">
            <div style="flex:1;min-width:260px">
              <label for="live-model-sel">Select model</label>
              <select id="live-model-sel" style="width:100%">${modelOptions()}</select>
            </div>
            <button type="button" class="primary" id="live-apply-btn"
              ${applying ? "disabled" : ""} onclick="window._liveApplyModel()">
              ${applying ? "Scoring…" : "Apply & rescore day"}</button>
          </div>
          <div id="live-model-summary" style="margin-top:12px;padding:10px 12px;background:var(--bg3);border-radius:8px">
            <div style="font-size:15px;font-weight:600">Scoring now:
              <span class="mono">${esc(bundleRunLabel(st.model_path || s.model_path) || st.candidate_id || "—")}</span></div>
            <div class="muted" style="margin-top:4px">
              ${fmtInt(st.n_bars || 0)} bars ·
              <span class="pos">${fmtInt(st.n_up || 0)} up</span> ·
              <span class="neg">${fmtInt(st.n_down || 0)} down</span> ·
              ${fmtInt(st.n_flat || 0)} flat ·
              <b>${fmtInt(st.n_signals || 0)} signals</b> @ τ=${tau.toFixed(2)} ·
              net <span class="${(st.net_pnl || 0) >= 0 ? "pos" : "neg"}">${sgnUsd(st.net_pnl)}</span>
            </div>
            <div class="muted mono" style="margin-top:6px;font-size:11px;word-break:break-all"
              title="model path">${esc(st.model_path || s.model_path || "—")}</div>
            ${statusLine ? `<div class="muted" style="margin-top:6px">${esc(statusLine)}</div>` : ""}
          </div>
        </div>

        <div id="live-pred-slot" style="margin-top:12px">${renderPred()}</div>

        ${renderBlotter()}`;

      const setModelHandler = () => {
        const sel = $("#live-model-sel");
        if (!sel) return;
        if (draftPath && [...sel.options].some(o => o.value === draftPath)) sel.value = draftPath;
        sel.onchange = () => { draftPath = sel.value; };
      };
      if (!$("#live-model-card")) {
        view.replaceChildren(next.content.cloneNode(true));
        setModelHandler();
      } else {
        for (const id of ["live-header", "live-stream-status-slot", "live-model-summary",
          "live-pred-slot", "live-blotter-title", "live-blotter-summary", "live-blotter-body"]) {
          const current = document.getElementById(id);
          const fresh = $(`#${id}`, next.content);
          if (current && fresh && current.innerHTML !== fresh.innerHTML) current.innerHTML = fresh.innerHTML;
        }
        const currentSel = $("#live-model-sel");
        const freshSel = $("#live-model-sel", next.content);
        if (currentSel && freshSel && document.activeElement !== currentSel) {
          const value = draftPath || currentSel.value;
          currentSel.innerHTML = freshSel.innerHTML;
          if ([...currentSel.options].some(o => o.value === value)) currentSel.value = value;
        }
        const currentApply = $("#live-apply-btn");
        const freshApply = $("#live-apply-btn", next.content);
        if (currentApply && freshApply) {
          currentApply.disabled = freshApply.disabled;
          currentApply.textContent = freshApply.textContent;
        }
      }
    } catch (e) {
      console.error(e);
      view.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
    }
  };

  const loadModels = async () => {
    try { lastModel = await api("/api/live/model"); }
    catch (e) { lastModel = {available: [], selection: {mode: "auto"}, error: String(e.message || e)}; }
  };
  const loadLedger = async (force = false) => {
    try {
      const q = new URLSearchParams({
        mode: fillMode,
        tau: String(tau),
      });
      if (force) q.set("force", "1");
      lastL = await api(`/api/live/ledger?${q.toString()}`);
    } catch (e) {
      lastL = {rows: [], meta: {note: String(e.message || e), mode: fillMode}};
    }
    loadingTicks = fillMode === "ticks" && !!(lastL && lastL.meta && lastL.meta.building);
    lastStats = statsFromLedger(lastL, lastS);
  };

  const loadStreamer = async () => {
    try {
      lastS = await api("/api/live/streamer");
      if (lastS.tau != null && !localStorage.getItem("live_tau")) tau = lastS.tau;
      const pin = lastS.active_objective;
      if (pin) {
        const el = $("#active-objective");
        if (el) el.textContent = `config: ${pin}`;
      }
    } catch (e) { lastS = {status: "error", note: String(e.message || e), streamer: {}}; }
  };

  const loadSnap = async () => {
    await loadStreamer();
    await loadLedger(false);
  };

  const pollStreamer = async () => {
    if (streamInFlight) return;
    streamInFlight = true;
    try {
      await loadStreamer();
      paint();
    } finally {
      streamInFlight = false;
    }
  };

  const loadAll = async (includeModels = false) => {
    if (refreshInFlight) return;
    refreshInFlight = true;
    try {
      // Bundle discovery can block behind a feature build; it is not live data.
      // Load it only on entry, while stream/ledger polling remains responsive.
      if (includeModels) {
        loadModels().then(() => paint());
      }
      await loadSnap();
      if (!statusLine && lastStats && lastStats.candidate_id) {
        statusLine = `Loaded ${lastStats.candidate_id} · ${lastStats.n_bars} bars`;
      }
      paint();
      // Tick blotter is built off-request; poll every 2s until the cache lands.
      if (fillMode === "ticks" && lastL && lastL.meta && lastL.meta.building) {
        const t = setTimeout(() => loadAll(false).catch(() => {}), 2000);
        _pollTimers.push(t);
      }
    } finally {
      refreshInFlight = false;
    }
  };

  // Immediate shell so the model control is never blank
  lastStats = {candidate_id: "—", n_bars: 0, n_up: 0, n_down: 0, n_flat: 0, n_signals: 0, net_pnl: 0};
  paint();
  // Streamer health is independent of snapshot/ledger work.
  setPoll(() => pollStreamer().catch(() => {}), 5000);
  addPoll(() => loadAll(false).catch(() => {}), 15000);
  loadAll(true).catch(() => {});
  return Promise.resolve();
}

/* ── custom test ───────────────────────────────────────────────────────── */

const _CT_META_DEFAULT = {
  enabled: false, mode: "meta_stack", family: "logistic_regression",
  scheme: "walk_forward_OOF", feature: "linear_weights",
  approve_threshold: 0.55, n_folds: 5, embargo_bars: 120, context_view: null,
};
const _CT_WF_DEFAULT = {
  enabled: true, start: "2025-05-01", stop: "2026-06-30",
  train_n: 2, train_unit: "months", test_n: 2, test_unit: "months",
  val: {enabled: false, n: 10, unit: "days", cut_from: "train"},
};
const _CT_WF_UNITS = ["days", "weeks", "months"];

function _ctAddPeriod(iso, n, unit) {
  const [y, m, d] = String(iso || "").split("-").map(Number);
  if (!y || !m || !d) return "";
  const dt = new Date(Date.UTC(y, m - 1, d));
  n = Number(n) || 0;
  if (unit === "days") dt.setUTCDate(dt.getUTCDate() + n);
  else if (unit === "weeks") dt.setUTCDate(dt.getUTCDate() + 7 * n);
  else dt.setUTCMonth(dt.getUTCMonth() + n);
  return dt.toISOString().slice(0, 10);
}
function _ctDayBefore(iso) {
  return _ctAddPeriod(iso, -1, "days");
}
function _ctWfSchedule(wf) {
  if (!wf || !wf.enabled) return [];
  const start = wf.start, stop = wf.stop;
  const tn = Number(wf.train_n), tu = wf.train_unit || "months";
  const en = Number(wf.test_n), eu = wf.test_unit || "months";
  if (!start || !stop || tn < 1 || en < 1) return [];
  const stopExcl = _ctAddPeriod(stop, 1, "days");
  let trainEndExcl = _ctAddPeriod(start, tn, tu);
  const out = [];
  const valOn = !!(wf.val && wf.val.enabled);
  const vn = Number((wf.val || {}).n) || 10;
  const vu = (wf.val || {}).unit || "days";
  const cut = (wf.val || {}).cut_from || "train";
  while (out.length < 24) {
    let testEndExcl = _ctAddPeriod(trainEndExcl, en, eu);
    if (!trainEndExcl || trainEndExcl >= stopExcl) break;
    let clipped = false;
    if (testEndExcl > stopExcl) { testEndExcl = stopExcl; clipped = true; }
    if (testEndExcl <= trainEndExcl) break;
    let trainEnd = _ctDayBefore(trainEndExcl);
    let oofStart = trainEndExcl;
    let oofEnd = _ctDayBefore(testEndExcl);
    let val = null;
    if (valOn) {
      if (cut === "test") {
        const valEndExcl = _ctAddPeriod(oofStart, vn, vu);
        const valEnd = _ctDayBefore(valEndExcl);
        val = {start: oofStart, end: valEnd};
        oofStart = _ctAddPeriod(valEnd, 1, "days");
      } else {
        const trainEndEx = _ctAddPeriod(trainEnd, 1, "days");
        const valStart = _ctAddPeriod(trainEndEx, -vn, vu);
        val = {start: valStart, end: trainEnd};
        trainEnd = _ctDayBefore(valStart);
      }
    }
    out.push({fold: out.length, train: {start, end: trainEnd},
              oof: {start: oofStart, end: oofEnd}, val, clipped});
    trainEndExcl = testEndExcl;
    if (clipped) break;
  }
  return out;
}

async function viewCustomTest(id) {
  view.classList.add("wide");
  const board = await api(`/api/custom-test/default/${encodeURIComponent(id)}`);
  let storeId = board.feature_store || "5s";
  let bank = await api(`/api/custom-test/bank?store=${encodeURIComponent(storeId)}`);
  const byFam = Object.fromEntries((bank.models || []).map(m => [m.name, m]));
  const famDefaults = fam => ({...((byFam[fam] || {}).params_default || {})});
  const famSpace = fam => (byFam[fam] || {}).search_space || {};
  const famHelp = fam => (byFam[fam] || {}).params_help || {};
  const famCap = fam => {
    const n = Number((byFam[fam] || {}).search_max_rows);
    return Number.isFinite(n) && n > 0 ? n : null;
  };
  const famShowsTrainCap = fam => {
    const info = byFam[fam] || {};
    return info.compute_class === "gpu" && !!info.needs_sequence;
  };
  const capDefaults = fam => {
    const cap = famCap(fam) || 100000;
    return {maxRowsOn: false, maxRows: cap};
  };
  const hydrateCap = m => {
    const d = capDefaults(m.family);
    if (!Object.prototype.hasOwnProperty.call(m, "max_rows")) return d;
    if (m.max_rows == null || m.max_rows === "")
      return {maxRowsOn: false, maxRows: d.maxRows};
    return {maxRowsOn: true, maxRows: Math.max(1, Number(m.max_rows) || d.maxRows)};
  };
  const famParamKeys = fam => {
    const keys = [];
    for (const k of Object.keys(famSpace(fam))) keys.push(k);
    for (const k of Object.keys(famDefaults(fam)))
      if (!keys.includes(k)) keys.push(k);
    return keys.filter(k => k !== "n_features");
  };
  const columnsFor = (kind, key) => {
    if (kind === "feature") return [key];
    if (kind === "group") {
      const g = (bank.groups || []).find(x => x.prefix === key);
      if (g && Array.isArray(g.features) && g.features.length) return g.features.slice();
      return (bank.features || []).filter(f => String(f.name).startsWith(key)).map(f => f.name);
    }
    if (kind === "view") {
      const v = (bank.views || []).find(x => x.name === key);
      if (v && Array.isArray(v.features) && v.features.length) return v.features.slice();
      const prefixes = (v && v.prefixes) || [];
      return (bank.features || [])
        .filter(f => prefixes.some(p => String(f.name).startsWith(p)))
        .map(f => f.name);
    }
    return [];
  };
  const expandItems = items => {
    const out = [];
    const seen = new Set();
    for (const it of items || []) {
      if (it.kind === "feature") {
        if (!it.key || seen.has(it.key)) continue;
        seen.add(it.key);
        out.push({kind: "feature", key: it.key, label: it.label || it.key,
                  ablation: !!it.ablation});
        continue;
      }
      for (const col of columnsFor(it.kind, it.key)) {
        if (!col || seen.has(col)) continue;
        seen.add(col);
        out.push({kind: "feature", key: col, label: col, ablation: false});
      }
    }
    return out;
  };
  const hydrateSlot = m => ({
    id: m.id || m.family,
    family: m.family,
    head: m.head || board.head || "multiclass",
    items: expandItems(m.items || []),
    params: {...(m.params || {})},
    hpo: !!m.hpo,
    hpoParams: Array.isArray(m.hpo_params) ? m.hpo_params.map(String) : [],
    ...hydrateCap(m),
  });
  const st = {
    oid: id,
    store: storeId,
    defaultHead: board.head || "multiclass",
    bank,
    models: Array.isArray(board.models) ? board.models.map(hydrateSlot) : [],
    meta: {..._CT_META_DEFAULT, ...(board.meta || {})},
    walkForward: {
      ..._CT_WF_DEFAULT,
      ...(board.walk_forward || {}),
      val: {..._CT_WF_DEFAULT.val, ...((board.walk_forward || {}).val || {})},
    },
    dry: false,
    tuneTrials: Number(board.tune_trials) || 0,
    filter: "",
    bankTab: "views",
    focus: null,
    preview: null,
    previewErr: "",
    launching: false,
    saveAs: null,
    saveName: "",
  };
  if (st.models.length) st.focus = st.models[0].id;
  let previewTimer = 0;

  const slotParamsPayload = m => {
    const def = famDefaults(m.family);
    const out = {};
    for (const [k, v] of Object.entries(m.params || {})) {
      if (JSON.stringify(v) !== JSON.stringify(def[k])) out[k] = v;
    }
    return out;
  };
  const payload = () => ({
    schema_version: "v1",
    objective_id: st.oid,
    feature_store: st.store || "5s",
    models: st.models.map(m => {
      const row = {
        id: m.id, family: m.family, head: m.head || "multiclass",
        items: m.items.map(it => ({
          kind: it.kind, key: it.key,
          label: it.label || it.key, ablation: !!it.ablation,
        })),
      };
      const params = slotParamsPayload(m);
      if (Object.keys(params).length) row.params = params;
      if (m.hpo) {
        row.hpo = true;
        const keys = (m.hpoParams && m.hpoParams.length)
          ? m.hpoParams
          : Object.keys(famSpace(m.family));
        if (keys.length) row.hpo_params = keys;
      }
      if (famShowsTrainCap(m.family))
        row.max_rows = m.maxRowsOn ? Math.max(1, Number(m.maxRows) || capDefaults(m.family).maxRows) : null;
      return row;
    }),
    meta: st.meta,
    walk_forward: st.walkForward,
    dry_run: st.dry,
    tune_trials: st.tuneTrials || 0,
  });

  const comboHint = () => {
    if (!st.models.length) return "select a model to start";
    let n = 0;
    const parts = [];
    for (const m of st.models) {
      const k = m.items.filter(it => it.ablation).length;
      const c = 1 << k;
      n += c;
      parts.push(k ? `${m.id} ×${c}` : m.id);
    }
    const meta = st.meta.enabled && st.meta.mode !== "none"
      ? ` + ${st.meta.mode}` : "";
    const wfN = (st.preview && st.preview.walk_forward && st.preview.walk_forward.walk_forward_folds)
      || (st.walkForward.enabled ? _ctWfSchedule(st.walkForward).length : 0);
    const wf = st.walkForward.enabled && wfN
      ? ` · ${wfN} calendar fold${wfN === 1 ? "" : "s"}` : "";
    const hpoSlots = st.models.filter(m => m.hpo && Object.keys(famSpace(m.family)).length);
    const hpo = hpoSlots.length
      ? ` + HPO ${st.tuneTrials || 20} trial${(st.tuneTrials || 20) === 1 ? "" : "s"}`
      : "";
    return `${n} candidate${n === 1 ? "" : "s"} (${parts.join(", ")})${meta}${wf}${hpo}`;
  };

  const refreshPreview = () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      if (!st.models.length) {
        st.preview = null; st.previewErr = "";
        paintStats();
        return;
      }
      try {
        st.preview = await post("/api/custom-test/preview", payload());
        st.previewErr = "";
      } catch (e) {
        st.preview = null;
        st.previewErr = e.message || String(e);
      }
      paintStats();
      paintWfTable();
    }, 280);
  };

  const addModel = fam => {
    if (!byFam[fam]) return;
    let nid = fam, n = 2;
    while (st.models.some(m => m.id === nid)) nid = `${fam}_${n++}`;
    const viewName = (board.models || []).find(m => m.family === fam)?.items?.[0]?.key
      || byFam[fam].feature_view;
    const items = expandItems(viewName
      ? [{kind: "view", key: viewName, label: viewName, ablation: false}] : []);
    st.models.push({id: nid, family: fam, head: st.defaultHead || "multiclass",
      items, params: {}, hpo: false, hpoParams: [],
      ...capDefaults(fam)});
    st.focus = nid;
    paint();
    refreshPreview();
  };

  const toggleModel = fam => {
    const existing = st.models.filter(m => m.family === fam);
    if (existing.length) {
      // already on the bench — click-into params (remove lives on the card)
      st.focus = existing[0].id;
      paint();
      return;
    }
    addModel(fam);
  };

  const addItem = (slotId, item) => {
    const slot = st.models.find(m => m.id === slotId);
    if (!slot) return;
    const incoming = expandItems([item]);
    if (!incoming.length) {
      toast(`no features in ${item.kind} '${item.key}'`, "err");
      return;
    }
    const prior = new Set(slot.items.filter(it => it.kind === "feature").map(it => it.key));
    const added = [];
    for (const it of incoming) {
      if (prior.has(it.key)) continue;
      slot.items.push(it);
      prior.add(it.key);
      added.push(it.key);
    }
    if (!added.length) {
      toast(`${item.key} already on ${slotId}`, "info", 2500);
      return;
    }
    st.focus = slotId;
    if (item.kind !== "feature")
      toast(`${fmtInt(added.length)} feature${added.length === 1 ? "" : "s"} from ${item.key} on ${slotId}`, "ok", 3500);
    paint();
    refreshPreview();
  };

  const addAllFeatures = slotId => {
    const sid = slotId || st.focus;
    const slot = st.models.find(m => m.id === sid);
    if (!slot) { toast("click a model in the bench first", "err"); return; }
    const feats = bank.features || [];
    if (!feats.length) { toast("feature bank is empty", "err"); return; }
    const prior = new Map();
    for (const it of slot.items) {
      if (it.kind === "feature") prior.set(it.key, it);
    }
    // Full catalog replaces views/groups; keep ablation flags already set on columns.
    slot.items = feats.map(f => ({
      kind: "feature",
      key: f.name,
      label: f.name,
      ablation: !!(prior.get(f.name) && prior.get(f.name).ablation),
    }));
    st.focus = slot.id;
    const nNew = feats.filter(f => !prior.has(f.name)).length;
    toast(`all ${feats.length} features on ${slot.id}`
      + (nNew < feats.length ? ` (${nNew} new)` : ""), "ok", 4000);
    paint();
    refreshPreview();
  };

  window._ctToggle = fam => toggleModel(fam);
  window._ctHead = (sid, v) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    slot.head = v;
    paint();
    refreshPreview();
  };
  const ctStoreLabel = s => {
    if (!s) return "";
    if (s.available === false) return `${s.id} (empty)`;
    if (s.id === "1m") return "1m (so/cs/ps/vw/rv/vp/ms)";
    if (s.id === "15s") return "15s (tape + geometry)";
    if (s.id === "5s") return "5s (research matrix)";
    return s.id;
  };
  window._ctStore = async (sid) => {
    const rec = (bank.stores || []).find(s => s.id === sid);
    if (rec && rec.available === false) {
      toast(`${sid} feature files are not built yet`, "err");
      return;
    }
    st.store = sid;
    try {
      bank = await api(`/api/custom-test/bank?store=${encodeURIComponent(sid)}`);
      const known = new Set((bank.features || []).map(f => f.name));
      const views = new Set((bank.views || []).map(v => v.name));
      const groups = new Set((bank.groups || []).map(g => g.prefix));
      for (const m of st.models) {
        m.items = m.items.filter(it => {
          if (it.kind === "feature") return known.has(it.key);
          if (it.kind === "view") return views.has(it.key);
          if (it.kind === "group") return groups.has(it.key);
          return false;
        });
      }
    } catch (e) {
      toast(e.message, "err");
    }
    paint();
    refreshPreview();
  };
  window._ctFocus = sid => { st.focus = sid; paint(); };
  window._ctRemove = sid => {
    st.models = st.models.filter(m => m.id !== sid);
    if (st.focus === sid) st.focus = st.models[0] ? st.models[0].id : null;
    paint();
    refreshPreview();
  };
  window._ctDropItem = (sid, kind, key, label) => addItem(sid, {kind, key, label});
  window._ctPick = (kind, key, label) => {
    if (!st.focus) { toast("click a model in the bench first", "err"); return; }
    addItem(st.focus, {kind, key, label});
  };
  window._ctAddAll = sid => addAllFeatures(sid);
  window._ctDrag = (ev, kind, key, label) => {
    ev.dataTransfer.setData("application/json", JSON.stringify({kind, key, label}));
    ev.dataTransfer.effectAllowed = "copy";
  };
  window._ctAllow = ev => { ev.preventDefault(); ev.currentTarget.classList.add("over"); };
  window._ctLeave = ev => { ev.currentTarget.classList.remove("over"); };
  window._ctDrop = (ev, sid) => {
    ev.preventDefault();
    ev.currentTarget.classList.remove("over");
    try {
      const item = JSON.parse(ev.dataTransfer.getData("application/json") || "{}");
      if (item.kind && item.key) addItem(sid, item);
    } catch { /* ignore bad payloads */ }
  };
  window._ctAblate = (sid, idx) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot || !slot.items[idx]) return;
    slot.items[idx].ablation = !slot.items[idx].ablation;
    paint();
    refreshPreview();
  };
  window._ctUnitem = (sid, idx) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    slot.items.splice(idx, 1);
    paint();
    refreshPreview();
  };
  window._ctTab = tab => { st.bankTab = tab; paint(); };
  window._ctFilter = el => { st.filter = el.value; paintBank(); };
  window._ctSaveAs = sid => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    const n = slot.items.filter(it => it.kind === "feature").length;
    if (!n) { toast("add features to this model first", "err"); return; }
    if (st.saveAs !== sid) st.saveName = "";
    st.saveAs = sid;
    st.focus = sid;
    paint();
    const inp = document.getElementById("ct-view-name");
    if (inp) { inp.focus(); inp.select(); }
  };
  window._ctSaveAsCancel = () => { st.saveAs = null; st.saveName = ""; paint(); };
  window._ctSaveName = el => { st.saveName = el.value; };
  window._ctSaveView = async sid => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    const features = slot.items.filter(it => it.kind === "feature").map(it => it.key);
    if (!features.length) { toast("add features to this model first", "err"); return; }
    const name = String(st.saveName || "").trim().toLowerCase().replace(/\s+/g, "_");
    if (!name) { toast("name the view first", "err"); return; }
    const body = {name, features, overwrite: false};
    try {
      let r;
      try {
        r = await post("/api/custom-test/views", body);
      } catch (e) {
        const msg = e.message || String(e);
        if (/already exists/i.test(msg)) {
          if (!confirm(`view "${name}" already exists — overwrite?`)) return;
          r = await post("/api/custom-test/views", {...body, overwrite: true});
        } else {
          throw e;
        }
      }
      if (r && r.views) bank.views = r.views;
      st.saveAs = null;
      st.saveName = "";
      st.bankTab = "views";
      toast(`saved view ${name} (${fmtInt(features.length)} features)`, "ok");
      paint();
    } catch (e) {
      toast(e.message || String(e), "err", 8000);
    }
  };
  window._ctDelView = async name => {
    if (!name) return;
    if (!confirm(`delete saved view "${name}"?`)) return;
    try {
      const r = await del(`/api/custom-test/views/${encodeURIComponent(name)}`);
      if (r && r.views) bank.views = r.views;
      toast(`deleted view ${name}`, "ok");
      paint();
    } catch (e) {
      toast(e.message || String(e), "err", 8000);
    }
  };
  window._ctMeta = (key, val) => {
    if (key === "enabled") st.meta.enabled = !!val;
    else if (key === "approve_threshold" || key === "n_folds" || key === "embargo_bars")
      st.meta[key] = Number(val);
    else st.meta[key] = val === "" ? null : val;
    paintStats();
    refreshPreview();
  };
  window._ctWf = (key, val) => {
    if (key === "enabled") st.walkForward.enabled = !!val;
    else if (key === "train_n" || key === "test_n") st.walkForward[key] = Math.max(1, Number(val) || 1);
    else if (key === "train_unit" || key === "test_unit" || key === "start" || key === "stop")
      st.walkForward[key] = val;
    paintStats();
    paintWfTable();
    refreshPreview();
  };
  window._ctWfVal = (key, val) => {
    if (!st.walkForward.val) st.walkForward.val = {..._CT_WF_DEFAULT.val};
    if (key === "enabled") st.walkForward.val.enabled = !!val;
    else if (key === "n") st.walkForward.val.n = Math.max(1, Number(val) || 1);
    else st.walkForward.val[key] = val;
    paintStats();
    paintWfTable();
    refreshPreview();
  };
  window._ctDry = on => { st.dry = !!on; };
  window._ctTrials = v => {
    const n = Math.max(0, Math.min(200, Number(v) || 0));
    st.tuneTrials = n;
    refreshPreview();
  };
  const coerceParam = (fam, key, raw) => {
    const space = famSpace(fam)[key];
    const def = famDefaults(fam)[key];
    if (raw === "" && (def == null || (space && space.type === "categorical")))
      return null;
    if (space && space.type === "int") return Number(raw);
    if (space && space.type === "float") return Number(raw);
    if (typeof def === "number") return Number(raw);
    if (typeof def === "boolean") return !!raw;
    return raw;
  };
  window._ctParam = (sid, key, raw) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    slot.params[key] = coerceParam(slot.family, key, raw);
    refreshPreview();
  };
  window._ctParamJson = (sid, key, raw) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    try { slot.params[key] = JSON.parse(raw); }
    catch { toast(`invalid JSON for ${key}`, "err", 4000); return; }
    refreshPreview();
  };
  window._ctResetParams = sid => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    slot.params = {};
    Object.assign(slot, capDefaults(slot.family));
    paint();
    refreshPreview();
  };
  window._ctCapOn = (sid, on) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    slot.maxRowsOn = !!on;
    if (slot.maxRowsOn && !(Number(slot.maxRows) > 0))
      slot.maxRows = capDefaults(slot.family).maxRows;
    paint();
    refreshPreview();
  };
  window._ctCapN = (sid, raw) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    const n = Math.max(1, parseInt(raw, 10) || capDefaults(slot.family).maxRows);
    slot.maxRows = n;
    slot.maxRowsOn = true;
    refreshPreview();
  };
  window._ctHpo = (sid, on) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    const spaceKeys = Object.keys(famSpace(slot.family));
    slot.hpo = !!on && spaceKeys.length > 0;
    if (slot.hpo && !slot.hpoParams.length) slot.hpoParams = spaceKeys.slice();
    if (slot.hpo && !st.tuneTrials) st.tuneTrials = 20;
    paint();
    refreshPreview();
  };
  window._ctHpoParam = (sid, key, on) => {
    const slot = st.models.find(m => m.id === sid);
    if (!slot) return;
    const set = new Set(slot.hpoParams && slot.hpoParams.length
      ? slot.hpoParams
      : Object.keys(famSpace(slot.family)));
    if (on) set.add(key); else set.delete(key);
    slot.hpoParams = [...set];
    if (!slot.hpoParams.length) slot.hpo = false;
    else slot.hpo = true;
    if (slot.hpo && !st.tuneTrials) st.tuneTrials = 20;
    paint();
    refreshPreview();
  };
  window._ctReset = async () => {
    const fresh = await api(`/api/custom-test/default/${encodeURIComponent(id)}`);
    st.store = fresh.feature_store || "5s";
    st.defaultHead = fresh.head || "multiclass";
    bank = await api(`/api/custom-test/bank?store=${encodeURIComponent(st.store)}`);
    st.models = (fresh.models || []).map(hydrateSlot);
    st.meta = {..._CT_META_DEFAULT, ...(fresh.meta || {})};
    st.walkForward = {
      ..._CT_WF_DEFAULT,
      ...(fresh.walk_forward || {}),
      val: {..._CT_WF_DEFAULT.val, ...((fresh.walk_forward || {}).val || {})},
    };
    st.tuneTrials = Number(fresh.tune_trials) || 0;
    st.focus = st.models[0] ? st.models[0].id : null;
    paint();
    refreshPreview();
  };
  window._ctLaunch = async () => {
    if (st.launching) return;
    if (!st.models.length) { toast("add at least one model", "err"); return; }
    st.launching = true;
    paintStats();
    try {
      const r = await post("/api/custom-test/launch", payload());
      const n = (r.compiled && r.compiled.n_candidates) || "?";
      toast(`custom test launched: ${r.job.job_id} · ${n} candidate(s)`, "ok");
      location.hash = `#/jobs/${r.job.job_id}`;
    } catch (e) {
      toast(e.message, "err", 10000);
      st.launching = false;
      paintStats();
    }
  };

  const q = () => (st.filter || "").trim().toLowerCase();
  const match = (...parts) => {
    const s = q();
    if (!s) return true;
    return parts.some(p => String(p || "").toLowerCase().includes(s));
  };

  const bankItemsHtml = () => {
    const s = q();
    if (st.bankTab === "views") {
      const views = (bank.views || []).filter(v => match(
        v.name, v.description, (v.prefixes || []).join(" "), (v.features || []).join(" "), v.source));
      const card = v => {
        const n = v.n_features != null ? v.n_features : (v.features || []).length;
        const saved = v.source === "saved";
        const detail = saved
          ? (v.description || "saved column snapshot")
          : (v.description || (v.prefixes || []).join(" "));
        const meta = saved
          ? `${fmtInt(n)} column${n === 1 ? "" : "s"}`
          : `${fmtInt(n)} column${n === 1 ? "" : "s"} · ${(v.prefixes || []).length} prefix group(s)`;
        return `
        <div class="ct-item ${saved ? "saved" : ""}" draggable="true"
          ondragstart="window._ctDrag(event,'view','${esc(v.name)}','${esc(v.name)}')"
          onclick="window._ctPick('view','${esc(v.name)}','${esc(v.name)}')">
          <div class="ct-item-k mono">
            ${esc(v.name)}
            ${saved ? '<span class="pill info">saved</span>' : ""}
            ${saved ? `<button type="button" class="ct-x" title="delete saved view"
              onclick="event.stopPropagation();window._ctDelView('${esc(v.name)}')">✕</button>` : ""}
          </div>
          <div class="ct-item-d">${esc(detail)}</div>
          <div class="muted mono">${esc(meta)}</div>
        </div>`;
      };
      const saved = views.filter(v => v.source === "saved");
      const catalog = views.filter(v => v.source !== "saved");
      if (!views.length) return '<div class="empty">no views match</div>';
      return [
        saved.length ? `<div class="muted" style="padding:4px 0">saved views</div>${saved.map(card).join("")}` : "",
        `<div class="muted" style="padding:4px 0">catalog views</div>${catalog.map(card).join("") || '<div class="empty">no catalog views match</div>'}`,
      ].join("");
    }
    if (st.bankTab === "groups") {
      const groups = (bank.groups || []).filter(g => match(g.prefix, g.name));
      return groups.map(g => `
        <div class="ct-item" draggable="true"
          ondragstart="window._ctDrag(event,'group','${esc(g.prefix)}','${esc(g.name)}')"
          onclick="window._ctPick('group','${esc(g.prefix)}','${esc(g.name)}')">
          <div class="ct-item-k mono">${esc(g.prefix)}</div>
          <div class="ct-item-d">${esc(g.name)} · ${fmtInt(g.count)} columns</div>
        </div>`).join("") || '<div class="empty">no groups match</div>';
    }
    const feats = (bank.features || []).filter(f => match(f.name, f.description, f.prefix));
    const shown = s ? feats.slice(0, 120) : feats.slice(0, 40);
    const more = feats.length - shown.length;
    return (shown.map(f => `
      <div class="ct-item" draggable="true"
        ondragstart="window._ctDrag(event,'feature','${esc(f.name)}','${esc(f.name)}')"
        onclick="window._ctPick('feature','${esc(f.name)}','${esc(f.name)}')">
        <div class="ct-item-k mono">${esc(f.name)}</div>
        <div class="ct-item-d">${esc(f.description || "")}</div>
      </div>`).join("") || '<div class="empty">no features match</div>')
      + (more > 0 ? `<div class="muted" style="padding:8px">+${more} more — type to filter</div>` : "");
  };

  const fmtParam = v => {
    if (v === null || v === undefined) return "none";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  };
  const spaceHint = space => {
    if (!space) return "fixed — not in catalog HPO space";
    if (space.type === "int")
      return `int ${space.low}–${space.high}` + (space.step ? ` step ${space.step}` : "");
    if (space.type === "float")
      return `float ${space.low}–${space.high}` + (space.log ? " log" : "");
    if (space.type === "categorical")
      return "choices: " + (space.choices || []).map(c => c == null ? "none" : String(c)).join(", ");
    return space.type || "";
  };
  const paramControl = (m, key) => {
    const space = famSpace(m.family)[key];
    const def = famDefaults(m.family)[key];
    const val = Object.prototype.hasOwnProperty.call(m.params, key) ? m.params[key] : def;
    const sid = m.id;
    if (space && space.type === "categorical") {
      const choices = space.choices || [];
      const opts = choices.map(c => {
        const v = c == null ? "" : String(c);
        const sel = (c == null && (val === null || val === undefined))
          || (c !== null && c !== undefined && String(c) === String(val));
        return `<option value="${esc(v)}" ${sel ? "selected" : ""}>${c == null ? "(none)" : esc(String(c))}</option>`;
      }).join("");
      return `<select class="ct-param-in" onchange="window._ctParam('${esc(sid)}','${esc(key)}',this.value)">${opts}</select>`;
    }
    if (Array.isArray(def) || Array.isArray(val)) {
      return `<input class="ct-param-in" value="${esc(JSON.stringify(val === undefined ? def : val))}"
        onchange="window._ctParamJson('${esc(sid)}','${esc(key)}',this.value)">`;
    }
    if (typeof def === "boolean" || typeof val === "boolean") {
      return `<input type="checkbox" ${val ? "checked" : ""}
        onchange="window._ctParam('${esc(sid)}','${esc(key)}',this.checked)">`;
    }
    if ((space && space.type === "int") || (typeof def === "number" && Number.isInteger(def))) {
      const step = space && space.step ? space.step : 1;
      return `<input class="ct-param-in" type="number" step="${step}" value="${esc(val)}"
        onchange="window._ctParam('${esc(sid)}','${esc(key)}',this.value)">`;
    }
    if ((space && space.type === "float") || typeof def === "number") {
      return `<input class="ct-param-in" type="number" step="any" value="${esc(val)}"
        onchange="window._ctParam('${esc(sid)}','${esc(key)}',this.value)">`;
    }
    return `<input class="ct-param-in" type="text" value="${esc(val == null ? "" : val)}"
      onchange="window._ctParam('${esc(sid)}','${esc(key)}',this.value)">`;
  };
  const paramsPanelHtml = m => {
    const space = famSpace(m.family);
    const keys = famParamKeys(m.family);
    const canHpo = Object.keys(space).length > 0;
    const hpoSet = new Set(m.hpo
      ? (m.hpoParams.length ? m.hpoParams : Object.keys(space))
      : []);
    const nEdit = Object.keys(slotParamsPayload(m)).length;
    const showCap = famShowsTrainCap(m.family);
    if (!keys.length && !showCap) {
      return `<div class="muted" style="margin-top:8px">this family has no catalog parameters</div>`;
    }
    const capHelp = "Last N consecutive train bars per fold (OOF/test is never capped). Off = full train window.";
    const capRow = showCap ? `<tr class="${m.maxRowsOn ? "ct-param-edit" : ""}">
        <td class="mono" title="${esc(capHelp)}">max_rows</td>
        <td>
          <label class="checkbox" style="margin:0 8px 0 0" title="Cap training rows">
            <input type="checkbox" ${m.maxRowsOn ? "checked" : ""}
              onchange="window._ctCapOn('${esc(m.id)}', this.checked)"> on</label>
          <input class="ct-param-in" type="number" min="1" step="1000"
            value="${esc(m.maxRows)}" ${m.maxRowsOn ? "" : "disabled"}
            onchange="window._ctCapN('${esc(m.id)}', this.value)"
            title="${esc(capHelp)}" style="width:7.5rem">
        </td>
        <td class="muted mono">off</td>
        <td class="muted" title="${esc(capHelp)}">${esc(capHelp)}</td>
        <td><span class="muted">—</span></td>
      </tr>` : "";
    const rows = keys.map(key => {
      const sp = space[key];
      const def = famDefaults(m.family)[key];
      const help = famHelp(m.family)[key] || "";
      const edited = Object.prototype.hasOwnProperty.call(m.params, key)
        && JSON.stringify(m.params[key]) !== JSON.stringify(def);
      const hpoBox = sp
        ? `<label class="ct-hpo-inc" title="include in HPO">
            <input type="checkbox" ${hpoSet.has(key) ? "checked" : ""} ${m.hpo ? "" : "disabled"}
              onclick="event.stopPropagation()"
              onchange="window._ctHpoParam('${esc(m.id)}','${esc(key)}',this.checked)"> HPO</label>`
        : `<span class="muted">—</span>`;
      return `<tr class="${edited ? "ct-param-edit" : ""}">
        <td class="mono" ${help ? `title="${esc(help)}"` : ""}>${esc(key)}</td>
        <td>${paramControl(m, key)}</td>
        <td class="muted mono">${esc(fmtParam(def))}</td>
        <td class="muted" ${help ? `title="${esc(help)}"` : ""}>${esc(help || spaceHint(sp))}</td>
        <td>${hpoBox}</td>
      </tr>`;
    }).join("");
    return `
      <div class="ct-params" id="${st.focus === m.id ? "ct-params" : ""}" onclick="event.stopPropagation()">
        <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px">
          <label class="checkbox" style="margin:0" title="${canHpo ? "Optuna varies checked parameters over catalog ranges" : "this family has no catalog search_space"}">
            <input type="checkbox" ${m.hpo ? "checked" : ""} ${canHpo ? "" : "disabled"}
              onchange="window._ctHpo('${esc(m.id)}', this.checked)"> enable HPO</label>
          <div class="muted">${nEdit ? `${nEdit} override(s)` : "catalog defaults"}
            ${canHpo ? "" : " · no HPO space"}</div>
          <button type="button" class="small" onclick="window._ctResetParams('${esc(m.id)}')">restore defaults</button>
        </div>
        ${m.hpo ? `<div class="muted" style="margin-bottom:8px">Checked knobs are searched by Optuna (${st.tuneTrials || 20} trials).
          Unchecked knobs stay at the values on the left.</div>` : ""}
        <div class="ct-params-wrap"><table class="ct-params-tbl">
          <thead><tr><th>parameter</th><th>this run</th><th>default</th><th>HPO range</th><th>in HPO</th></tr></thead>
          <tbody>${capRow}${rows}</tbody>
        </table></div>
      </div>`;
  };
  const slotHtml = m => {
    const info = byFam[m.family] || {};
    const focused = st.focus === m.id;
    const chips = m.items.map((it, idx) => `
      <span class="ct-chip ${it.ablation ? "ab" : ""}" title="${esc(it.kind)} ${esc(it.key)}">
        <span class="mono">${esc(it.label || it.key)}</span>
        <button type="button" class="ct-ab ${it.ablation ? "on" : ""}"
          title="ablation: test with and without this item"
          onclick="event.stopPropagation();window._ctAblate('${esc(m.id)}',${idx})">${it.ablation ? "AB" : "ab"}</button>
        <button type="button" class="ct-x" onclick="event.stopPropagation();window._ctUnitem('${esc(m.id)}',${idx})">✕</button>
      </span>`).join("");
    const k = m.items.filter(it => it.ablation).length;
    const nEdit = Object.keys(slotParamsPayload(m)).length;
    const spaceN = Object.keys(famSpace(m.family)).length;
    const hpoN = m.hpo
      ? (m.hpoParams.length ? m.hpoParams.length : spaceN)
      : 0;
    const capSum = famShowsTrainCap(m.family)
      ? (m.maxRowsOn ? `cap ${fmtInt(m.maxRows)} train` : "full train")
      : null;
    const paramSum = [
      nEdit ? `${nEdit} override(s)` : "defaults",
      capSum,
      m.hpo ? `HPO ${hpoN}` : null,
    ].filter(Boolean).join(" · ");
    return `
    <div class="ct-slot ${focused ? "focus" : ""}"
      onclick="window._ctFocus('${esc(m.id)}')"
      ondragover="window._ctAllow(event)" ondragleave="window._ctLeave(event)"
      ondrop="window._ctDrop(event,'${esc(m.id)}')">
      <div class="row" style="justify-content:space-between;align-items:center;margin-bottom:6px">
        <div>
          <span class="mono" style="font-weight:600">${esc(m.id)}</span>
          <span class="pill dim">${esc(info.compute_class || "")}</span>
          ${info.needs_sequence ? '<span class="pill info">sequence</span>' : ""}
          <select class="small" title="Output head" style="margin-left:6px"
            onclick="event.stopPropagation()"
            onchange="window._ctHead('${esc(m.id)}', this.value)">
            ${["multiclass", "ovr", "dual_binary"].map(h =>
              `<option value="${h}" ${ (m.head || "multiclass") === h ? "selected" : ""}>${h}</option>`).join("")}
          </select>
          ${k ? `<span class="pill warn">${1 << k} combos</span>` : ""}
          ${m.hpo ? '<span class="pill warn">HPO</span>' : ""}
        </div>
        <div class="fixed ct-slot-actions" style="display:flex;gap:6px;align-items:center;flex-wrap:wrap" onclick="event.stopPropagation()">
          ${st.saveAs === m.id ? `
            <form class="ct-save-as" onsubmit="event.preventDefault();window._ctSaveView('${esc(m.id)}')">
              <input id="ct-view-name" placeholder="view name" maxlength="64"
                value="${esc(st.saveName)}"
                oninput="window._ctSaveName(this)"
                onclick="event.stopPropagation()">
              <button type="submit" class="small primary">save</button>
              <button type="button" class="small" onclick="window._ctSaveAsCancel()">cancel</button>
            </form>` : `
            <button type="button" class="small" title="Save this model's current features as a reusable view in the feature bank"
              onclick="window._ctSaveAs('${esc(m.id)}')">save as view</button>`}
          <button type="button" class="small" title="View and edit catalog parameters / HPO"
            onclick="window._ctFocus('${esc(m.id)}')">params</button>
          <button type="button" class="small" title="Assign every catalog feature column to this model"
            onclick="window._ctAddAll('${esc(m.id)}')">＋ all features</button>
          <button type="button" class="small danger" onclick="window._ctRemove('${esc(m.id)}')">remove</button>
        </div>
      </div>
      <div class="muted" style="margin-bottom:8px">${esc(info.display_name || m.family)}
        ${m.items.length ? `· ${fmtInt(m.items.length)} assigned` : ""}
        · ${esc(paramSum)}</div>
      <div class="ct-chips">${chips || '<span class="muted">drop a view, group, or feature here — empty uses the family default / full matrix</span>'}</div>
      ${focused ? paramsPanelHtml(m) : ""}
    </div>`;
  };

  const groupedModels = () => {
    const groups = {};
    for (const m of bank.models || []) {
      (groups[m.family_group || "other"] ||= []).push(m);
    }
    return groups;
  };

  const paintBank = () => {
    const el = document.getElementById("ct-bank-list");
    if (el) el.innerHTML = bankItemsHtml();
  };

  const paintWfTable = () => {
    const el = document.getElementById("ct-wf-table");
    if (!el) return;
    const wf = st.walkForward;
    if (!wf.enabled) {
      el.innerHTML = '<div class="muted">Disabled — this run uses the objective YAML equal-count folds.</div>';
      return;
    }
    const sched = (st.preview && st.preview.walk_forward
      && st.preview.walk_forward.walk_forward
      && st.preview.walk_forward.walk_forward.schedule)
      || _ctWfSchedule(wf);
    const n = sched.length;
    const showVal = !!(wf.val && wf.val.enabled);
    const rows = sched.map(s => {
      const tr = s.train || {};
      const oof = s.oof || {};
      const val = s.val || {};
      return `<tr>
        <td class="mono">${(s.fold ?? 0) + 1}</td>
        <td class="mono">${esc(tr.start || "—")} → ${esc(tr.end || "—")}</td>
        ${showVal ? `<td class="mono">${val.start ? `${esc(val.start)} → ${esc(val.end)}` : "—"}</td>` : ""}
        <td class="mono">${esc(oof.start || "—")} → ${esc(oof.end || "—")}</td>
      </tr>`;
    }).join("");
    el.innerHTML = n
      ? `<div class="muted" style="margin-bottom:6px">${n} expanding fold${n === 1 ? "" : "s"}
          · train ${esc(wf.train_n)} ${esc(wf.train_unit)} then test ${esc(wf.test_n)} ${esc(wf.test_unit)}
          · dates after ${esc(wf.stop)} are held out for backtest</div>
        <table class="ct-wf-tbl"><thead><tr>
          <th>fold</th><th>train</th>${showVal ? "<th>val</th>" : ""}<th>OOF / test</th>
        </tr></thead><tbody>${rows}</tbody></table>`
      : '<div class="muted">Window does not fit one train+test block — widen stop or shrink periods.</div>';
  };

  const paintStats = () => {
    const el = document.getElementById("ct-stats");
    if (!el) return;
    if (document.getElementById("ct-hpo-trials") === document.activeElement) return;
    const p = st.preview;
    const warn = (p && p.warnings || []).map(w => `<div class="muted">⚠ ${esc(w)}</div>`).join("");
    const err = st.previewErr ? `<div class="muted" style="color:var(--err)">⚠ ${esc(st.previewErr)}</div>` : "";
    const rows = ((p && p.candidates) || []).map(c => {
      const ss = c.search_space ? Object.keys(c.search_space) : [];
      const pkeys = c.params ? Object.keys(c.params) : [];
      return `
      <tr>
        <td class="mono">${esc(c.id)}</td>
        <td class="mono">${esc(c.family)}</td>
        <td class="mono">${c.features ? esc(JSON.stringify(c.features)) : "default / full"}</td>
        <td class="mono">${esc(c.ablation_tag || "")}</td>
        <td class="mono">${pkeys.length ? esc(pkeys.join(", ")) : "defaults"}</td>
        <td class="mono">${c.max_rows ? `cap ${fmtInt(c.max_rows)}` : "full train"}</td>
        <td class="mono">${ss.length ? `HPO ${ss.length}: ${esc(ss.join(", "))}` : "off"}</td>
      </tr>`;
    }).join("");
    const anyHpo = st.models.some(m => m.hpo && Object.keys(famSpace(m.family)).length);
    el.innerHTML = `
      <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
        <div><span class="pill info">${esc(comboHint())}</span>
          ${p && p.n_candidates != null ? `<span class="muted">compiled ${fmtInt(p.n_candidates)}</span>` : ""}
          ${p && p.tune_trials ? `<span class="pill warn">tune ${fmtInt(p.tune_trials)}</span>` : ""}</div>
        <div class="row fixed" style="gap:8px">
          <label class="checkbox" style="margin:0"><input type="checkbox" ${st.dry ? "checked" : ""}
            onchange="window._ctDry(this.checked)"> dry-run</label>
          ${anyHpo ? `<label class="fixed" style="margin:0">HPO trials
            <input id="ct-hpo-trials" type="number" min="1" max="200" style="width:70px"
              value="${st.tuneTrials || 20}"
              onchange="window._ctTrials(this.value)"></label>` : ""}
          <button type="button" class="small" onclick="window._ctReset()">reset to objective default</button>
          <button type="button" class="primary" ${st.launching || !st.models.length ? "disabled" : ""}
            onclick="window._ctLaunch()">${st.launching ? "Launching…" : "▶ Launch test"}</button>
        </div>
      </div>
      ${warn}${err}
      ${rows ? `<div style="max-height:160px;overflow:auto;margin-top:8px"><table>
        <thead><tr><th>candidate</th><th>family</th><th>features</th><th>ablation</th><th>params</th><th>train</th><th>HPO</th></tr></thead>
        <tbody>${rows}</tbody></table></div>` : ""}`;
  };

  const paint = () => {
    if (document.getElementById("ct-filter") === document.activeElement) {
      paintBank();
      paintStats();
      return;
    }
    if (document.getElementById("ct-meta")?.contains(document.activeElement)) {
      paintStats();
      return;
    }
    if (document.getElementById("ct-wf")?.contains(document.activeElement)) {
      paintStats();
      paintWfTable();
      return;
    }
    if (document.getElementById("ct-hpo-trials") === document.activeElement) {
      return;
    }
    if (document.getElementById("ct-params")?.contains(document.activeElement)) {
      paintStats();
      return;
    }
    if (document.getElementById("ct-view-name") === document.activeElement) {
      paintStats();
      return;
    }
    const selected = new Set(st.models.map(m => m.family));
    const groups = groupedModels();
    const groupHtml = Object.entries(groups).map(([g, ms]) => `
      <div class="ct-g">${esc(g.replace(/_/g, " "))}</div>
      ${ms.map(m => `
        <button type="button" class="ct-model ${selected.has(m.name) ? "on" : ""}"
          onclick="window._ctToggle('${esc(m.name)}')"
          title="${esc(m.notes || m.display_name)}">
          <span class="mono">${esc(m.name)}</span>
          <span class="ct-model-m">${esc(m.compute_class)}${m.needs_sequence ? " · seq" : ""}</span>
        </button>`).join("")}
    `).join("");

    const meta = st.meta;
    view.innerHTML = `
    <a class="back" href="#/objectives/${esc(id)}">← ${esc(id)}</a>
    <div class="row" style="justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:8px">
      <div>
        <h2 style="margin:4px 0 2px">Custom test</h2>
        <div class="muted">Click a model to add it. Click a card (or <b>params</b>) to view catalog
          defaults, edit them for this run, and tick knobs for HPO.
          Drag a view, group, or feature onto a model card — views and groups expand to
          their columns on the card. <b>save as view</b> stores that subset in the feature bank.
          <b>＋ all features</b> assigns the full catalog to that model.
          Mark <b>ab</b> to ablate (with and without; multiple flags multiply).
          Meta defaults come from this objective’s topology (latest stack: meta_stack / logistic).</div>
      </div>
    </div>
    ${objSubTabs(id, "test")}
    <div id="ct-stats" class="card" style="margin-bottom:12px"></div>
    <div class="ct-board">
      <aside class="ct-col">
        <div class="ct-col-h">Models</div>
        <div class="ct-col-b">${groupHtml}</div>
      </aside>
      <section class="ct-col ct-mid">
        <div class="ct-col-h">Testing window
          <span class="muted" style="font-weight:400;text-transform:none;letter-spacing:0">
            — drop features onto a card · click a card to focus it</span></div>
        <div class="ct-col-b">
          ${st.models.length ? st.models.map(slotHtml).join("")
            : '<div class="empty">No models selected. Click a family on the left — the objective default (if any) is preloaded.</div>'}
          <div class="ct-builder">
            <h3 style="margin:12px 0 6px">Feature builder</h3>
            <div class="muted">${esc((bank.feature_builder && bank.feature_builder.note)
              || "Parked — compose new features from existing names in a later increment.")}</div>
          </div>
        </div>
      </section>
      <aside class="ct-col">
        <div class="ct-col-h">Feature bank</div>
        <div class="ct-col-b">
          <label class="muted" style="display:block;margin-bottom:6px">store
            <select onchange="window._ctStore(this.value)" style="margin-left:6px">
              ${(bank.stores || [{id:"5s", available:true}]).map(s =>
                `<option value="${esc(s.id)}" ${st.store === s.id ? "selected" : ""}
                  ${s.available === false ? "disabled" : ""}
                  title="${esc(s.note || "")}">${esc(ctStoreLabel(s))}</option>`).join("")}
            </select>
          </label>
          <input id="ct-filter" placeholder="filter views / groups / features"
            value="${esc(st.filter)}" oninput="window._ctFilter(this)">
          <div class="ct-tabs">
            ${["views", "groups", "features"].map(t =>
              `<button type="button" class="${st.bankTab === t ? "active" : ""}"
                onclick="window._ctTab('${t}')">${t}</button>`).join("")}
            <button type="button" class="small" style="margin-left:auto"
              title="Assign every catalog feature to the focused model"
              onclick="window._ctAddAll()">＋ all → selected</button>
          </div>
          <div id="ct-bank-list">${bankItemsHtml()}</div>
        </div>
      </aside>
    </div>
    <div class="card" id="ct-meta" style="margin-top:12px">
      <h3 style="margin-top:0">Meta on selected models</h3>
      <div class="muted" style="margin-bottom:8px">Trains on the <b>full</b> (no-omission) variant of each
        selected family. Ablation variants are scored on their own. Disable to skip the objective’s default gate.</div>
      <div class="row" style="align-items:flex-end">
        <div class="fixed"><label class="checkbox" style="margin-top:22px">
          <input type="checkbox" ${meta.enabled ? "checked" : ""}
            onchange="window._ctMeta('enabled', this.checked)"> enable meta</label></div>
        <div><label>mode</label>
          <select onchange="window._ctMeta('mode', this.value)">
            ${["meta_stack", "output_meta", "primary_meta"].map(m =>
              `<option ${meta.mode === m ? "selected" : ""}>${m}</option>`).join("")}
          </select></div>
        <div><label>family</label>
          <select onchange="window._ctMeta('family', this.value)">
            ${["logistic_regression", "xgboost"].map(m =>
              `<option ${meta.family === m ? "selected" : ""}>${m}</option>`).join("")}
          </select></div>
        <div><label>scheme</label>
          <select onchange="window._ctMeta('scheme', this.value)">
            ${["walk_forward_OOF", "cross_fit"].map(m =>
              `<option ${meta.scheme === m ? "selected" : ""}>${m}</option>`).join("")}
          </select></div>
        <div><label>feature</label>
          <select onchange="window._ctMeta('feature', this.value)">
            ${["linear_weights", "log_weights"].map(m =>
              `<option ${meta.feature === m ? "selected" : ""}>${m}</option>`).join("")}
          </select></div>
        <div class="fixed"><label>approve τ</label>
          <input type="number" step="0.01" min="0.5" max="1" style="width:80px"
            value="${esc(meta.approve_threshold)}"
            onchange="window._ctMeta('approve_threshold', this.value)"></div>
        <div class="fixed"><label>folds</label>
          <input type="number" step="1" min="2" max="20" style="width:70px"
            value="${esc(meta.n_folds)}"
            onchange="window._ctMeta('n_folds', this.value)"></div>
        <div class="fixed"><label>embargo bars</label>
          <input type="number" step="1" min="0" style="width:90px"
            value="${esc(meta.embargo_bars)}"
            onchange="window._ctMeta('embargo_bars', this.value)"></div>
      </div>
    </div>
    <div class="card" id="ct-wf" style="margin-top:12px">
      <h3 style="margin-top:0">Walk-forward</h3>
      <div class="muted" style="margin-bottom:8px">Expanding calendar windows: <b>train 1 + test 1 = train 2</b>
        until the stop date. Not equal bar-counts. Disable to use the objective YAML folds.
        Val is only used by sequence/DL models (TCN/LSTM) for early stopping; trees ignore it.</div>
      <div class="row" style="align-items:flex-end">
        <div class="fixed"><label class="checkbox" style="margin-top:22px">
          <input type="checkbox" ${st.walkForward.enabled ? "checked" : ""}
            onchange="window._ctWf('enabled', this.checked)"> enable designer</label></div>
        <div class="fixed"><label>start</label>
          <input type="date" value="${esc(st.walkForward.start)}"
            onchange="window._ctWf('start', this.value)"></div>
        <div class="fixed"><label>stop</label>
          <input type="date" value="${esc(st.walkForward.stop)}"
            onchange="window._ctWf('stop', this.value)"></div>
        <div class="fixed"><label>train period</label>
          <div class="ct-wf-period">
            <input type="number" min="1" max="365" value="${esc(st.walkForward.train_n)}"
              onchange="window._ctWf('train_n', this.value)">
            <select onchange="window._ctWf('train_unit', this.value)">
              ${_CT_WF_UNITS.map(u =>
                `<option ${st.walkForward.train_unit === u ? "selected" : ""}>${u}</option>`).join("")}
            </select>
          </div></div>
        <div class="fixed"><label>test period</label>
          <div class="ct-wf-period">
            <input type="number" min="1" max="365" value="${esc(st.walkForward.test_n)}"
              onchange="window._ctWf('test_n', this.value)">
            <select onchange="window._ctWf('test_unit', this.value)">
              ${_CT_WF_UNITS.map(u =>
                `<option ${st.walkForward.test_unit === u ? "selected" : ""}>${u}</option>`).join("")}
            </select>
          </div></div>
      </div>
      <div class="row" style="align-items:flex-end;margin-top:8px">
        <div class="fixed"><label class="checkbox" style="margin-top:22px">
          <input type="checkbox" ${(st.walkForward.val && st.walkForward.val.enabled) ? "checked" : ""}
            onchange="window._ctWfVal('enabled', this.checked)"> enable val</label></div>
        <div class="fixed"><label>val length</label>
          <div class="ct-wf-period">
            <input type="number" min="1" max="365" value="${esc((st.walkForward.val || {}).n)}"
              onchange="window._ctWfVal('n', this.value)"
              ${(st.walkForward.val && st.walkForward.val.enabled) ? "" : "disabled"}>
            <select onchange="window._ctWfVal('unit', this.value)"
              ${(st.walkForward.val && st.walkForward.val.enabled) ? "" : "disabled"}>
              ${_CT_WF_UNITS.map(u =>
                `<option ${(st.walkForward.val || {}).unit === u ? "selected" : ""}>${u}</option>`).join("")}
            </select>
          </div></div>
        <div><label>cut val from</label>
          <select onchange="window._ctWfVal('cut_from', this.value)"
            ${(st.walkForward.val && st.walkForward.val.enabled) ? "" : "disabled"}>
            ${["train", "test"].map(c =>
              `<option ${(st.walkForward.val || {}).cut_from === c ? "selected" : ""}>${c}</option>`).join("")}
          </select></div>
      </div>
      <div id="ct-wf-table" style="margin-top:10px"></div>
    </div>`;
    paintStats();
    paintWfTable();
  };

  paint();
  refreshPreview();
}

/* ── Live Chart (TBBO candles) ─────────────────────────────────────────── */

function ctWallToUtcMs(dateStr, timeStr) {
  const [Y, Mo, D] = (dateStr || "").split("-").map(Number);
  const p = (timeStr || "00:00:00").split(":");
  const h = Number(p[0] || 0), mi = Number(p[1] || 0), se = Number(p[2] || 0);
  if (!Y || !Mo || !D) return NaN;
  let utc = Date.UTC(Y, Mo - 1, D, h, mi, se);
  for (let i = 0; i < 4; i++) {
    const parts = Object.fromEntries(_CT_FMT.formatToParts(new Date(utc)).map(x => [x.type, x.value]));
    const got = Date.UTC(+parts.year, +parts.month - 1, +parts.day, +parts.hour, +parts.minute, +parts.second);
    const want = Date.UTC(Y, Mo - 1, D, h, mi, se);
    utc += (want - got);
    if (want === got) break;
  }
  return utc;
}
function utcMsToCtParts(ms) {
  const p = Object.fromEntries(_CT_FMT.formatToParts(new Date(ms)).map(x => [x.type, x.value]));
  return {date: `${p.year}-${p.month}-${p.day}`, time: `${p.hour}:${p.minute}:${p.second}`};
}
function addIsoDate(iso, days) {
  const parts = String(iso || "").split("-").map(Number);
  if (parts.length < 3 || !parts[0]) return iso;
  const dt = new Date(Date.UTC(parts[0], parts[1] - 1, parts[2] + days));
  const mm = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(dt.getUTCDate()).padStart(2, "0");
  return `${dt.getUTCFullYear()}-${mm}-${dd}`;
}
function globexSessionEndDate(unix) {
  const p = utcMsToCtParts(unix * 1000);
  const t = (p.time || "00:00:00").split(":");
  const h = Number(t[0] || 0);
  if (h >= 17) return addIsoDate(p.date, 1);
  return p.date;
}
function isoWeekdayUtc(iso) {
  const parts = String(iso || "").split("-").map(Number);
  if (parts.length < 3 || !parts[0]) return 0;
  return new Date(Date.UTC(parts[0], parts[1] - 1, parts[2])).getUTCDay();
}
function addSessionEndDates(sessionDate, extra) {
  let d = sessionDate;
  const n = Number(extra) || 0;
  if (!n || !d) return d;
  const step = n > 0 ? 1 : -1;
  const target = Math.abs(n);
  let k = 0, guard = 0;
  while (k < target && guard < 16) {
    guard += 1;
    d = addIsoDate(d, step);
    const wd = isoWeekdayUtc(d);
    if (wd === 0 || wd === 6) continue;
    k += 1;
  }
  return d;
}
function vpSessionDay(profile) {
  if (!profile) return null;
  const a = profile.start_unix, b = profile.end_unix;
  let ref = null;
  if (a != null && Number.isFinite(a)) ref = a + 1;
  else if (b != null && Number.isFinite(b)) ref = b - 1;
  return ref != null ? globexSessionEndDate(ref) : null;
}
function profileCoversUnix(profile, unix) {
  if (!profile || unix == null || !Number.isFinite(Number(unix))) return false;
  const t = Number(unix);
  const a = Number(profile.start_unix);
  const b = Number(profile.end_unix);
  if (Number.isFinite(a) && Number.isFinite(b) && b !== a) {
    const lo = Math.min(a, b), hi = Math.max(a, b);
    return t >= lo && t < hi;
  }
  const day = vpSessionDay(profile);
  return day != null && globexSessionEndDate(t) === day;
}
function profilesCoveringUnix(profiles, unix) {
  const list = Array.isArray(profiles) ? profiles : [];
  return list.filter(p => profileCoversUnix(p, unix)).sort((a, b) => {
    const na = Number(a && a.n) || 0, nb = Number(b && b.n) || 0;
    if (na !== nb) return na - nb;
    return (Number(a && a.start_unix) || 0) - (Number(b && b.start_unix) || 0);
  });
}
function parseVpExtend(v) {
  const n = Number(v);
  if (n === 0 || n === 1 || n === 2 || n === 3) return n;
  return 1;
}
function isRthCtUnix(unix) {
  const p = utcMsToCtParts(unix * 1000);
  const t = (p.time || "00:00:00").split(":");
  const m = Number(t[0] || 0) * 60 + Number(t[1] || 0);
  return m >= 8 * 60 + 30 && m < 15 * 60;
}
function nonRthIndexRuns(realUnix) {
  const runs = [];
  let start = null;
  const n = (realUnix && realUnix.length) || 0;
  for (let j = 0; j <= n; j++) {
    const night = j < n && !isRthCtUnix(realUnix[j]);
    if (night) {
      if (start == null) start = j;
    } else if (start != null) {
      runs.push([start, j]);
      start = null;
    }
  }
  return runs;
}
function addCivilDays(dateStr, n) {
  const p = (dateStr || "").split("-").map(Number);
  if (p.length < 3 || !p[0] || !p[1] || !p[2]) return dateStr;
  const dt = new Date(Date.UTC(p[0], p[1] - 1, p[2] + n));
  const y = dt.getUTCFullYear();
  const m = String(dt.getUTCMonth() + 1).padStart(2, "0");
  const d = String(dt.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}
function globexOvernightWindows(fromUnix, toUnix) {
  const out = [];
  if (fromUnix == null || toUnix == null || toUnix <= fromUnix) return out;
  let dateStr = addCivilDays(utcMsToCtParts(fromUnix * 1000).date, -1);
  const lastDate = utcMsToCtParts(toUnix * 1000).date;
  let guard = 0;
  while (dateStr && guard < 800) {
    guard += 1;
    const a = ctWallToUtcMs(dateStr, "17:00:00") / 1000;
    const b = ctWallToUtcMs(addCivilDays(dateStr, 1), "08:30:00") / 1000;
    if (Number.isFinite(a) && Number.isFinite(b) && b > fromUnix && a < toUnix) {
      out.push([a, b]);
    }
    if (dateStr === lastDate) break;
    dateStr = addCivilDays(dateStr, 1);
  }
  return out;
}

async function viewLiveChart() {
  view.classList.add("wide");
  view.classList.add("chart-fill");
  const shell = document.getElementById("shell");
  if (shell) shell.classList.add("chart-mode");
  window._lcGen = (window._lcGen || 0) + 1;
  const gen = window._lcGen;
  const still = () => window._lcGen === gen && $("#lc-chart");

  let meta = null, payload = null, chart = null, candleSeries = null, volSeries = null;
  const overlaySeries = {};
  let pollTimer = 0, loading = false, fitted = false, lastSnap = null;
  let reloadQueued = false, drawTries = 0;
  let lastLb = "session";
  let lastMaps = null;
  let printCache = null;
  let printHits = [];
  let lastLevelBlendHits = [];
  let hoverPrint = null;
  const profilePriceLines = [];
  let vpAxisPriceRange = null;
  let replayPlayTimer = 0;
  let replayPlaying = false;
  let replaySpeed = (() => {
    const n = Number(localStorage.getItem("lc_replay_speed"));
    return [0.25, 0.5, 1, 2, 3, 4].includes(n) ? n : 1;
  })();
  const REPLAY_BASE_MS = 400;
  const TF_SEC = { "5s": 5, "15s": 15, "1min": 60, "5min": 300, "15min": 900, "1hour": 3600, "1day": 86400 };
  const _MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  let hoverCorner = localStorage.getItem("lc_hover_corner") || "follow";
  let hoverPos = (() => {
    try {
      const raw = JSON.parse(localStorage.getItem("lc_hover_pos") || "null");
      if (raw && Number.isFinite(Number(raw.left)) && Number.isFinite(Number(raw.top))) {
        return { left: Number(raw.left), top: Number(raw.top) };
      }
    } catch {}
    return null;
  })();
  let hoverDragging = false;
  let hoverResizing = false;
  let hoverSize = (() => {
    try {
      const raw = JSON.parse(localStorage.getItem("lc_hover_size") || "null");
      if (raw && Number(raw.w) >= 140 && Number(raw.h) >= 72) {
        return { w: Number(raw.w), h: Number(raw.h) };
      }
    } catch {}
    return null;
  })();
  let cvdCandleSeries = null;
  const boundLineRefs = new WeakMap();
  let lwLib = null;
  let oscLayoutKey = "";
  let cvdPaneIndex = null;
  const PRICE_FRAC = 0.58;
  const VOL_FRAC = 0.18;
  const OSC_FRAC = 0.24;

  const TF_LABEL = {
    "5s": "5 seconds", "15s": "15 seconds", "1min": "1 minute",
    "5min": "5 minutes", "15min": "15 minutes", "1hour": "1 hour", "1day": "1 day",
  };
  const LB_LABEL = {
    "30m": "30 min", "1h": "1 hour", "2h": "2 hours", "4h": "4 hours",
    "8h": "8 hours", "12h": "12 hours",
    "session": "session", "1d": "1 day", "2d": "2 days", "3d": "3 days",
    "5d": "5 days", "7d": "7 days", "10d": "10 days", "20d": "20 days",
    "30d": "30 days", "60d": "60 days", "90d": "90 days", "1y": "1 year",
  };
  const LB_CUSTOM = "__custom__";
  function parseLookbackRange(raw) {
    const s = String(raw || "").trim().toLowerCase().replace(/\s+/g, " ");
    const m = s.match(/^(\d{4}-\d{2}-\d{2})(?:\s*\.\.\s*|\s+to\s+)(\d{4}-\d{2}-\d{2})$/);
    if (!m) return null;
    if (m[1] > m[2]) return null;
    return `${m[1]}..${m[2]}`;
  }
  function parseLookbackToken(raw) {
    const range = parseLookbackRange(raw);
    if (range) return range;
    const s = String(raw || "").trim().toLowerCase().replace(/\s+/g, " ");
    if (!s) return null;
    if (s === "session" || s === "globex" || s === "eth") return "session";
    const compact = s.replace(/\s+/g, "");
    const m = compact.match(/^(\d+(?:\.\d+)?)(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks|y|yr|yrs|year|years)$/);
    if (!m) return null;
    const n = Number(m[1]);
    if (!Number.isFinite(n) || n <= 0) return null;
    const u = m[2];
    let fam = "d";
    if (u === "m" || u.startsWith("min")) fam = "m";
    else if (u[0] === "h") fam = "h";
    else if (u[0] === "d") fam = "d";
    else if (u[0] === "w") fam = "w";
    else fam = "y";
    return `${n}${fam}`;
  }
  function lookbackLabel(v) {
    const range = parseLookbackRange(v);
    if (range) {
      const [a, b] = range.split("..");
      return a === b ? "Custom: " + a : "Custom: " + a + " → " + b;
    }
    const fromMeta = meta && meta.lookback_labels && meta.lookback_labels[v];
    if (fromMeta) return fromMeta;
    if (LB_LABEL[v]) return LB_LABEL[v];
    return "Custom: " + v;
  }
  function currentLookback() {
    const sel = $("#lc-lb");
    const v = sel && sel.value;
    if (v && v !== LB_CUSTOM) return v;
    return lastLb || "session";
  }
  function ctTodayYmd() {
    return utcMsToCtParts(Date.now()).date;
  }
  function lbPromptMode() {
    const el = document.querySelector('input[name="lc-lb-mode"]:checked');
    return (el && el.value) || "duration";
  }
  function setLbPromptMode(mode) {
    for (const el of document.querySelectorAll('input[name="lc-lb-mode"]')) {
      el.checked = el.value === mode;
    }
    const dur = $("#lc-lb-duration-fields");
    const rng = $("#lc-lb-range-fields");
    if (dur) dur.hidden = mode !== "duration";
    if (rng) rng.hidden = mode !== "range";
    const title = $("#lc-lb-prompt-title");
    if (title) title.textContent = mode === "range" ? "Chart date range" : "Look back how far?";
  }
  const OV_COLOR = {
    of_delta: "#7ee0a3", of_cvd: "#f0c674", of_ofi: "#c4b5fd", of_ofi_cum: "#22d3ee",
    lt_prints: "#f59e0b",
    vw_vwap: "#e6c07b", vw_vwap_dist: "#56b6c2",
    vw_bb_upper: "#d4b36a", vw_bb_lower: "#d4b36a",
    cs_clv: "#abb2bf",
    mf_mfi: "#fb7185", ad_line: "#a3e635",
    so_fast_d_9_3: "#7aa2f7",
    so_sync_score_all: "#d19a66", so_sync_score_9_60: "#e5c07b",
    ps_sar: "#ff9f43",
  };
  const ORIGIN_LABEL = {
    session: "Session (5:00pm CT)",
    bod: "BOD (midnight CT)",
    mkt: "MKT open (8:30am CT)",
    premarket: "Premarket (5:00pm–8:30am CT)",
  };

  const selectedKeys = () =>
    [...document.querySelectorAll("#lc-ov input[data-key]:checked")].map(el => el.dataset.key);
  const liveOn = () => !!( $("#lc-live") && $("#lc-live").checked );
  const vpOn = () => vpEnabled;
  const emaOn = () => emaEnabled;
  const stochOn = () => stochEnabled;
  function emaColor(n) {
    const i = Math.max(0, (Number(n) || 1) - 1);
    return EMA_COLORS[i % EMA_COLORS.length];
  }
  function emaMark(n) {
    const i = Number(n);
    return "EMA" + (Number.isInteger(i) && i > 0 ? i : 1);
  }
  function stochColor(n) {
    return emaColor(n);
  }
  function stochMark(n) {
    const i = Number(n);
    return "Stoch" + (Number.isInteger(i) && i > 0 ? i : 1);
  }
  function stochKey(n) {
    return "stoch_" + (Number(n) > 0 ? Number(n) : 1);
  }
  const ethDisplay = () => !$("#lc-eth-display") || $("#lc-eth-display").checked;
  const ethShade = () => !!( $("#lc-eth-shade") && $("#lc-eth-shade").checked );
  const ethAgg = () => !$("#lc-eth-agg") || $("#lc-eth-agg").checked;
  const replayOn = () => !!( $("#lc-replay") && $("#lc-replay").checked );
  let replayUnix = null;
  let replayVpTimer = 0;
  let replayDragging = false;
  function printMin() {
    const n = Number($("#lc-print-min") && $("#lc-print-min").value);
    if (!Number.isFinite(n) || n < 1) return 10;
    return Math.min(10000, Math.floor(n));
  }
  function printOpacity() {
    const n = Number($("#lc-print-opacity") && $("#lc-print-opacity").value);
    return Number.isFinite(n) ? Math.min(1, Math.max(0, n / 100)) : .88;
  }
  function printDisplayValue() {
    return !$("#lc-print-display-value") || $("#lc-print-display-value").checked;
  }
  function printExtendBars() {
    const n = Number($("#lc-print-extend-bars") && $("#lc-print-extend-bars").value);
    return Number.isFinite(n) ? Math.min(5000, Math.max(0, Math.floor(n))) : 0;
  }
  function syncPrintDisplayControls() {
    const opacityValue = $("#lc-print-opacity-val");
    if (opacityValue) opacityValue.textContent = `${Math.round(printOpacity() * 100)}%`;
  }
  function textSize() {
    const n = Number($("#lc-text-size") && $("#lc-text-size").value);
    return Number.isFinite(n) ? Math.min(18, Math.max(10, n)) : 12;
  }
  // Text painted on the plot (order sizes, nearest-order notes, grid, profile labels).
  // Separate from Text size, which sets the axis and the readout. Default is 2× the old 10px.
  function chartLabelSize() {
    const n = Number($("#lc-chart-label") && $("#lc-chart-label").value);
    return Number.isFinite(n) ? Math.min(48, Math.max(10, Math.round(n))) : 20;
  }
  function chartLabelFont(weight) {
    const w = weight ? `${weight} ` : "";
    return `${w}${chartLabelSize()}px SF Mono, ui-monospace, Menlo, Consolas, monospace`;
  }
  function applyChartLabelSize() {
    const n = chartLabelSize();
    const lab = $("#lc-chart-label-val");
    if (lab) lab.textContent = n + "px";
    localStorage.setItem("lc_chart_label", String(n));
    scheduleDrawProfiles();
  }
  function isRthUnix(unix) {
    return isRthCtUnix(unix);
  }
  function barMaps(p) {
    const empty = { real: [], plot: [], idx: [], densify: false };
    if (!p || !p.n) return empty;
    const tf = p.timeframe || ($("#lc-tf") && $("#lc-tf").value) || "1min";
    const showNight = ethDisplay() || tf === "1day";
    const sec = TF_SEC[tf] || 60;
    const real = [], plot = [], idx = [];
    let t0 = null, n = 0;
    for (let i = 0; i < p.n; i++) {
      if (p.t[i] == null) continue;
      if (!showNight && !isRthUnix(p.t[i])) continue;
      if (t0 == null) t0 = p.t[i];
      real.push(p.t[i]);
      plot.push(showNight ? p.t[i] : (t0 + n * sec));
      idx.push(i);
      n += 1;
    }
    return { real, plot, idx, densify: !showNight };
  }
  function plotToReal(t) {
    if (t == null || !Number.isFinite(t)) return t;
    if (!lastMaps || !lastMaps.densify || !lastMaps.plot.length) return t;
    const arr = lastMaps.plot;
    let lo = 0, hi = arr.length - 1, ans = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (arr[m] <= t) { ans = m; lo = m + 1; }
      else hi = m - 1;
    }
    return lastMaps.real[ans];
  }
  function realToPlot(unix) {
    if (unix == null || !Number.isFinite(unix)) return unix;
    if (!lastMaps || !lastMaps.densify || !lastMaps.real.length) return unix;
    const r = lastMaps.real;
    if (unix <= r[0]) return lastMaps.plot[0];
    if (unix >= r[r.length - 1]) return lastMaps.plot[r.length - 1];
    let lo = 0, hi = r.length - 1, ans = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (r[m] <= unix) { ans = m; lo = m + 1; }
      else hi = m - 1;
    }
    if (ans >= r.length - 1) return lastMaps.plot[r.length - 1];
    const span = r[ans + 1] - r[ans];
    const frac = span > 0 ? (unix - r[ans]) / span : 0;
    return lastMaps.plot[ans] + frac * (lastMaps.plot[ans + 1] - lastMaps.plot[ans]);
  }
  function applyTextSize() {
    const n = textSize();
    const page = document.querySelector(".lc-page");
    if (page) page.style.setProperty("--lc-font", n + "px");
    const lab = $("#lc-text-size-val");
    if (lab) lab.textContent = n + "px";
    localStorage.setItem("lc_text_size", String(n));
    if (chart) {
      try {
        chart.applyOptions({
          layout: {
            background: { color: "#0b0f14" },
            textColor: "#7d8da0",
            fontSize: n,
            fontFamily: "SF Mono, ui-monospace, Menlo, Consolas, monospace",
          },
        });
      } catch {}
    }
    scheduleDrawProfiles();
  }
  const cvdOrigin = () => ($("#lc-cvd-origin") && $("#lc-cvd-origin").value) || "session";
  const vwapOrigin = () => ($("#lc-vwap-origin") && $("#lc-vwap-origin").value) || "session";
  const vwapBandsOn = () => !!(
    selectedKeys().includes("vw_vwap")
    && $("#lc-vwap-bb")
    && $("#lc-vwap-bb").checked
  );
  const ofiOrigin = () => ($("#lc-ofi-origin") && $("#lc-ofi-origin").value) || "session";
  const adOrigin = () => ($("#lc-ad-origin") && $("#lc-ad-origin").value) || "session";
  function overlayMetaBy() {
    const metaBy = {};
    for (const o of (meta && meta.overlays) || []) metaBy[o.key] = o;
    return metaBy;
  }
  function overlayIsOsc(spec, key) {
    if (key === "lt_prints" || key === "lt_mbo_orders") return false;
    if (key && String(key).startsWith("stoch_")) return true;
    if (spec && spec.needs === "prints") return false;
    const pane = (spec && spec.pane) || "osc";
    return pane !== "price" && pane !== "canvas" && pane !== "vol";
  }
  function overlayZeroLine(spec) {
    if (!spec) return false;
    const z = spec.zero_line;
    return z === "1" || z === "true" || z === true || z === 1;
  }
  function oscPaneIndex(key) {
    const i = oscKeysOn().indexOf(key);
    return i >= 0 ? 2 + i : null;
  }
  function storedMfiBounds() {
    const b = lcNormalizeOscBounds(
      localStorage.getItem("lc_mfi_upper"),
      localStorage.getItem("lc_mfi_lower"),
    );
    return { upper: b.upper, lower: b.lower, arrows: localStorage.getItem("lc_mfi_arrows") === "1" };
  }
  function oscKeysOn() {
    const metaBy = overlayMetaBy();
    const keys = selectedKeys().filter(k => overlayIsOsc(metaBy[k], k));
    if (stochOn()) {
      for (const s of stochSpecs) {
        const k = stochKey(s.n);
        if (!keys.includes(k)) keys.push(k);
      }
    }
    return keys;
  }
  function loadPaneStretch() {
    try { return JSON.parse(localStorage.getItem("lc_pane_stretch") || "{}") || {}; }
    catch { return {}; }
  }
  function paneHeights() {
    if (!chart || typeof chart.panes !== "function") return [];
    return chart.panes().map(p => (p && typeof p.getHeight === "function" ? p.getHeight() : 0));
  }
  function savePaneStretch() {
    if (!chart || typeof chart.panes !== "function") return;
    try {
      const hs = paneHeights();
      const sum = hs.reduce((a, b) => a + b, 0) || 1;
      const out = loadPaneStretch();
      if (hs[0]) out.price = hs[0] / sum;
      if (hs[1]) out.vol = hs[1] / sum;
      oscKeysOn().forEach((k, i) => {
        if (hs[2 + i]) out[k] = hs[2 + i] / sum;
      });
      localStorage.setItem("lc_pane_stretch", JSON.stringify(out));
    } catch {}
  }
  function showPaneScale(paneIndex) {
    if (!chart || typeof chart.priceScale !== "function") return;
    try {
      chart.priceScale("right", paneIndex).applyOptions({
        visible: true,
        borderVisible: true,
        borderColor: "#3d5166",
        ticksVisible: true,
        entireTextOnly: false,
      });
    } catch {}
  }
  function applyPaneHeights(useSaved) {
    if (!chart || typeof chart.panes !== "function") return;
    const host = $("#lc-chart");
    const H = host ? host.clientHeight : 0;
    const panes = chart.panes();
    const n = panes.length;
    if (n < 1 || H < 80) return;
    const oscN = Math.max(0, n - 2);
    const saved = useSaved ? loadPaneStretch() : {};
    const live = paneHeights();
    const liveSum = live.reduce((a, b) => a + b, 0);
    const frac = (key, idx, fallback) => {
      const s = Number(saved[key]);
      if (useSaved && Number.isFinite(s) && s > 0) return s;
      if (!useSaved && liveSum > 0 && live[idx] > 0) return live[idx] / liveSum;
      return fallback;
    };
    let priceF = frac("price", 0, oscN ? PRICE_FRAC : 0.78);
    let volF = n > 1 ? frac("vol", 1, oscN ? VOL_FRAC : 0.22) : 0;
    const oscKeys = oscKeysOn();
    const oscF = oscKeys.map((k, i) => frac(k, 2 + i, OSC_FRAC / Math.max(1, oscN)));
    let sum = priceF + volF + oscF.reduce((a, b) => a + b, 0);
    if (sum <= 0) sum = 1;
    priceF /= sum; volF /= sum;
    const oscFn = oscF.map(v => v / sum);
    const minP = 90, minV = 64, minO = 64;
    const inner = H;
    let priceH = Math.max(minP, Math.round(inner * priceF));
    let volH = n > 1 ? Math.max(minV, Math.round(inner * volF)) : 0;
    const oscH = oscFn.map(v => Math.max(minO, Math.round(inner * v)));
    let used = priceH + volH + oscH.reduce((a, b) => a + b, 0);
    if (used > inner && priceH > minP) {
      priceH = Math.max(minP, priceH - (used - inner));
    }
    try {
      if (panes[0] && typeof panes[0].setHeight === "function") panes[0].setHeight(priceH);
      if (panes[1] && typeof panes[1].setHeight === "function") panes[1].setHeight(volH);
      oscH.forEach((h, i) => {
        if (panes[2 + i] && typeof panes[2 + i].setHeight === "function") panes[2 + i].setHeight(h);
      });
    } catch {}
    for (let i = 0; i < n; i++) showPaneScale(i);
  }
  function applyStretch() {
    applyPaneHeights(true);
  }
  function pruneExtraPanes(keep) {
    if (!chart || typeof chart.removePane !== "function" || typeof chart.panes !== "function") return;
    const panes = chart.panes();
    for (let i = panes.length - 1; i >= keep; i--) {
      try {
        if (panes[i] && typeof panes[i].setPreserveEmptyPane === "function") {
          panes[i].setPreserveEmptyPane(false);
        }
        chart.removePane(i);
      } catch {}
    }
  }
  function paneBox(index) {
    const host = $("#lc-chart");
    const fallback = { top: 0, height: host ? host.clientHeight : 0 };
    if (!chart || typeof chart.panes !== "function") return fallback;
    try {
      const pane = chart.panes()[index];
      if (!pane) return fallback;
      const height = typeof pane.getHeight === "function" ? pane.getHeight() : fallback.height;
      let top = 0;
      const el = typeof pane.getHTMLElement === "function" ? pane.getHTMLElement() : null;
      if (el && host) top = el.getBoundingClientRect().top - host.getBoundingClientRect().top;
      else {
        const panes = chart.panes();
        for (let i = 0; i < index; i++) {
          if (panes[i] && typeof panes[i].getHeight === "function") top += panes[i].getHeight();
        }
      }
      return { top, height };
    } catch {
      return fallback;
    }
  }
  function pricePaneBox() { return paneBox(0); }
  function sessionPaneBox() {
    const p0 = paneBox(0);
    const p1 = paneBox(1);
    const height = p0.height + (p1 && p1.height ? (p1.top > p0.top ? (p1.top - p0.top - p0.height) + p1.height : p1.height) : 0);
    return { top: p0.top, height: Math.max(p0.height, height) };
  }
  const expandedParams = new Set();
  let vpSpecs = [];
  let vpDraftSpecs = [];
  let vpEnabled = false;
  let emaSpecs = [];
  let emaDraftSpecs = [];
  let emaEnabled = false;
  let stochSpecs = [];
  let stochDraftSpecs = [];
  let stochEnabled = false;
  let sarParams = { start: 0.02, increment: 0.02, maximum: 0.2 };
  let sarDraftParams = Object.assign({}, sarParams);
  let cursorMode = "crosshair"; // "crosshair" | "select"
  let noteSelection = null; // { startUnix, endUnix, startPlot, endPlot, paneIndex, priceLo, priceHi }
  let noteSelectDrag = null;
  const EMA_COLORS = [
    "#f472b6", "#38bdf8", "#a78bfa", "#34d399", "#fb923c",
    "#e879f9", "#22d3ee", "#fbbf24",
  ];

  view.innerHTML = `
    <div class="lc-page">
      <div class="lc-toolbar">
        <div class="lc-inst"><span class="lc-root">/NQ</span>
          <span class="muted" id="lc-contract">lead</span></div>
        <div class="bv-field"><label>Timeframe</label>
          <select id="lc-tf"></select></div>
        <div class="bv-field lc-lb-field"><label>Lookback</label>
          <select id="lc-lb"></select></div>
        <div class="fixed"><label class="checkbox" style="margin-top:22px">
          <input type="checkbox" id="lc-live" checked> Live</label></div>
        <div class="fixed"><button class="primary" id="lc-load">Load</button></div>
        <div class="lc-cursor-tools" role="group" aria-label="Chart cursor">
          <button type="button" id="lc-cursor-cross" class="lc-cursor-btn is-active" title="Crosshair and pan">Crosshair</button>
          <button type="button" id="lc-cursor-select" class="lc-cursor-btn" title="Drag on a pane to select a box in time and price">Select</button>
          <button type="button" id="lc-zoom-in" class="lc-cursor-btn lc-icon-btn" title="Zoom in. With a selection, zoom into that box." aria-label="Zoom in">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="6.6" cy="6.6" r="4.1" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M9.7 9.7 L13 13 M6.6 4.7 V8.5 M4.7 6.6 H8.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
          </button>
          <button type="button" id="lc-zoom-out" class="lc-cursor-btn lc-icon-btn" title="Zoom out. With a selection, zoom out from that box." aria-label="Zoom out">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="6.6" cy="6.6" r="4.1" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M9.7 9.7 L13 13 M4.7 6.6 H8.5" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
          </button>
          <button type="button" id="lc-note-create" class="primary" disabled title="Save the selected box as a Notes entry">Create Note</button>
        </div>
        <div class="lc-layout-actions" role="group" aria-label="Chart layout">
          <button type="button" id="lc-fit" title="Fit all candles in view">Fit</button>
          <button type="button" id="lc-expand" title="Expand chart to the whole screen">Expand</button>
          <label class="checkbox lc-hover-toggle" title="Show or hide the info box that follows the cursor">
            <input type="checkbox" id="lc-hover-show" checked> Hover box
          </label>
        </div>
        <div class="muted" id="lc-status" style="margin-left:auto;font-family:var(--mono);font-size:12px"></div>
      </div>
      <div id="lc-lb-prompt" class="lc-lb-prompt" hidden>
        <div class="lc-lb-prompt-card" role="dialog" aria-labelledby="lc-lb-prompt-title">
          <h3 id="lc-lb-prompt-title">Look back how far?</h3>
          <div class="lc-lb-modes" role="tablist" aria-label="Custom lookback mode">
            <label class="checkbox"><input type="radio" name="lc-lb-mode" value="duration" checked> Duration</label>
            <label class="checkbox"><input type="radio" name="lc-lb-mode" value="range"> Date range</label>
          </div>
          <div id="lc-lb-duration-fields">
            <p class="muted" id="lc-lb-prompt-help">Examples: 6h, 3d, 2w, 90d, 1y. Session = Globex from 5:00pm CT. 7d is from 7 days ago through now.</p>
            <input id="lc-lb-custom" type="text" placeholder="e.g. 3d" autocomplete="off" spellcheck="false">
          </div>
          <div id="lc-lb-range-fields" hidden>
            <p class="muted" id="lc-lb-range-help">Chicago calendar dates, inclusive. From 00:00 CT on From through 24:00 CT on To (or now if To is today).</p>
            <div class="lc-lb-range-row">
              <label>From <input id="lc-lb-from" type="date"></label>
              <label>To <input id="lc-lb-to" type="date"></label>
            </div>
          </div>
          <div class="lc-lb-prompt-actions">
            <button type="button" id="lc-lb-cancel">Cancel</button>
            <button type="button" class="primary" id="lc-lb-apply">Apply</button>
          </div>
        </div>
      </div>
      <div id="lc-note-prompt" class="lc-lb-prompt" hidden>
        <div class="lc-lb-prompt-card" role="dialog" aria-labelledby="lc-note-prompt-title">
          <h3 id="lc-note-prompt-title">Create Note</h3>
          <p class="muted" id="lc-note-prompt-meta">—</p>
          <label>Brief note
            <textarea id="lc-note-text" rows="4" maxlength="4000" placeholder="What stands out in this selection?"></textarea>
          </label>
          <div class="lc-lb-prompt-actions">
            <button type="button" id="lc-note-cancel">Cancel</button>
            <button type="button" class="primary" id="lc-note-save">Save note</button>
          </div>
        </div>
      </div>
      <div class="lc-stage">
        <div class="lc-chart-card">
          <div id="lc-readout" class="lc-readout" aria-live="polite"></div>
          <div id="lc-mbo-head" class="lc-mbo-head" hidden></div>
          <div class="lc-chart-host">
            <div id="lc-chart"></div>
            <div id="lc-vp-axis" class="lc-vp-axis" hidden></div>
            <div id="lc-replay-layer" class="lc-replay-layer" hidden>
              <div id="lc-replay-curtain" class="lc-replay-curtain"></div>
              <div id="lc-replay-line" class="lc-replay-line" title="Drag playhead — time axis still pans/zooms">
                <div class="lc-replay-handle"></div>
              </div>
            </div>
            <canvas id="lc-vp-canvas" class="lc-vp-canvas"></canvas>
            <div id="lc-select-layer" class="lc-select-layer" hidden>
              <div id="lc-select-rect" class="lc-select-rect" hidden></div>
            </div>
            <div id="lc-scale-tags" class="lc-scale-tags"></div>
            <div id="lc-hover" class="lc-hover" hidden>
              <div class="lc-hover-grip" title="Pin or right-click keeps this box still. Follow cursor tracks the pointer. Drag the bar to move it. Drag an edge to resize. Double-click the bar to follow again.">
                <button type="button" class="lc-hover-pin" id="lc-hover-pin">Pin</button>
                <button type="button" class="lc-hover-follow" id="lc-hover-follow" hidden>Follow cursor</button>
              </div>
              <div id="lc-hover-body"></div>
              <div class="lc-hover-rz" data-edge="n"></div>
              <div class="lc-hover-rz" data-edge="s"></div>
              <div class="lc-hover-rz" data-edge="e"></div>
              <div class="lc-hover-rz" data-edge="w"></div>
              <div class="lc-hover-rz" data-edge="ne"></div>
              <div class="lc-hover-rz" data-edge="nw"></div>
              <div class="lc-hover-rz" data-edge="se"></div>
              <div class="lc-hover-rz" data-edge="sw"></div>
            </div>
            <div class="lc-empty" id="lc-empty">loading coverage…</div>
          </div>
        </div>
        <aside class="lc-side" id="lc-side">
          <div class="lc-side-h">
            <h3>Display</h3>
            <button type="button" class="lc-side-collapse" id="lc-side-collapse"
              title="Collapse overlays into the right wall" aria-label="Collapse overlays menu">▸</button>
          </div>
          <button type="button" class="lc-side-rail wall-tab" id="lc-side-expand"
            title="Expand overlays menu" aria-label="Expand overlays menu">◂ Overlays</button>
          <div id="lc-display">
            <div class="lc-og">
              <div class="lc-og-h">Night session</div>
              <label class="checkbox"><input type="checkbox" id="lc-eth-display" checked> Display</label>
              <div class="lc-info-row">
                <label class="checkbox"><input type="checkbox" id="lc-eth-shade"> Shade</label>
                ${lcInfoTip("Shade is Globex overnight: 5:00pm–8:30am CT, locked to the time axis.")}
              </div>
              <div class="lc-info-row">
                <label class="checkbox"><input type="checkbox" id="lc-eth-agg" checked> Aggregate</label>
                ${lcInfoTip("When off, overlays and volume profiles ignore overnight tape. 1-day bars are unchanged.")}
              </div>
            </div>
            <div id="lc-replay-box" class="lc-og lc-replay-box">
              <div class="lc-replay-box-h">
                <div class="lc-og-h">Replay</div>
                <div class="lc-replay-box-h-actions">
                  <button type="button" class="lc-section-collapse" id="lc-replay-collapse"
                    aria-expanded="true" title="Collapse Replay" aria-label="Collapse Replay">▾</button>
                </div>
              </div>
              <div class="lc-replay-box-body">
                <div class="lc-info-row">
                  <label class="checkbox"><input type="checkbox" id="lc-replay"> Replay</label>
                  ${lcInfoTip("Drag the gold playhead (or the slider) to scrub. Play advances one candle per tick; Step advances one bar; speeds change the tick rate. The time axis still pans and zooms. Candles and overlays only include data up to the playhead. Live polling pauses.")}
                </div>
                <div id="lc-replay-controls" class="lc-replay-controls" hidden>
                  <input type="range" id="lc-replay-slider" min="0" max="1000" value="1000">
                  <span id="lc-replay-time" class="muted">—</span>
                  <div class="lc-replay-transport">
                    <button type="button" id="lc-replay-play" title="Play / Pause" aria-label="Play or pause replay">▶</button>
                    <button type="button" id="lc-replay-step" title="Step one bar" aria-label="Step one bar">Step ›</button>
                  </div>
                  <div class="lc-replay-speeds" id="lc-replay-speeds" role="group" aria-label="Replay speed">
                    <button type="button" data-replay-speed="0.25" title="Quarter speed">¼</button>
                    <button type="button" data-replay-speed="0.5" title="Half speed">½</button>
                    <button type="button" data-replay-speed="1" title="1× speed">1×</button>
                    <button type="button" data-replay-speed="2" title="2× speed">2×</button>
                    <button type="button" data-replay-speed="3" title="3× speed">3×</button>
                    <button type="button" data-replay-speed="4" title="4× speed">4×</button>
                  </div>
                </div>
              </div>
            </div>
            <div class="lc-og">
              <div class="lc-og-h">Chart formatting</div>
              <label class="lc-text-size">Text size
                <span class="lc-text-size-row">
                  <input type="range" id="lc-text-size" min="10" max="18" step="1" value="12">
                  <span id="lc-text-size-val">12px</span>
                </span>
              </label>
              <label class="lc-text-size" title="Size of text drawn on the plot: order sizes, nearest-order notes, the price grid, reload labels, and volume-profile labels.">In-chart text
                <span class="lc-text-size-row">
                  <input type="range" id="lc-chart-label" min="10" max="48" step="1" value="20">
                  <span id="lc-chart-label-val">20px</span>
                </span>
              </label>
            </div>
          </div>
          <div class="lc-info-row" style="margin:0 0 8px">
            <h3 style="margin:0">Overlays</h3>
            ${lcInfoTip("TBBO series at the chart timeframe. Volume sits under price with its own y-axis. Click ▸ next to a name for that overlay’s parameters. CVD, stoch, and other non-price series each get their own pane below volume. CVD and OFI_CUM each have their own origin. OFI / VWAP still load prints.")}
          </div>
          <div id="lc-ov"></div>
        </aside>
      </div>
    </div>`;

  const setEmpty = (msg) => {
    const el = $("#lc-empty");
    if (el) { el.textContent = msg || ""; el.style.display = msg ? "" : "none"; }
  };
  const setStatus = (msg) => { const el = $("#lc-status"); if (el) el.textContent = msg || ""; };

  function paintLookbacks(selected) {
    const sel = $("#lc-lb");
    if (!sel) return;
    const lbs = (meta && meta.lookbacks) || Object.keys(LB_LABEL);
    const preset = new Set(lbs);
    const extra = selected && !preset.has(selected) && selected !== LB_CUSTOM
      ? `<option value="${esc(selected)}">${esc(lookbackLabel(selected))}</option>`
      : "";
    sel.innerHTML = lbs.map(t =>
      `<option value="${esc(t)}">${esc(lookbackLabel(t))}</option>`).join("")
      + extra
      + `<option value="${LB_CUSTOM}">Custom…</option>`;
    sel.value = (selected && selected !== LB_CUSTOM) ? selected : "session";
    lastLb = sel.value;
  }
  function tfCapHint() {
    const maxBars = (meta && meta.max_bars) || 50000;
    return ` Responses keep at most ${fmtInt(maxBars)} newest bars.`;
  }
  function seedRangeDates(fromVal, toVal) {
    const today = ctTodayYmd();
    const min = addCivilDays(today, -1825);
    const fromEl = $("#lc-lb-from");
    const toEl = $("#lc-lb-to");
    if (!fromEl || !toEl) return;
    fromEl.min = min;
    toEl.min = min;
    fromEl.max = today;
    toEl.max = today;
    fromEl.value = fromVal || addCivilDays(today, -7);
    toEl.value = toVal || today;
  }
  function openLookbackPrompt(seed) {
    const box = $("#lc-lb-prompt");
    const inp = $("#lc-lb-custom");
    const help = $("#lc-lb-prompt-help");
    if (!box || !inp) return;
    if (help) {
      help.textContent = "Examples: 6h, 3d, 2w, 90d, 1y. Session = Globex from 5:00pm CT. 7d is from 7 days ago through now."
        + tfCapHint();
    }
    const rangeHelp = $("#lc-lb-range-help");
    if (rangeHelp) rangeHelp.textContent = "Chicago calendar dates, inclusive. From 00:00 CT on From through 24:00 CT on To (or now if To is today)."
      + tfCapHint();
    const range = parseLookbackRange(seed);
    if (range) {
      const [a, b] = range.split("..");
      setLbPromptMode("range");
      seedRangeDates(a, b);
      inp.value = "";
    } else {
      setLbPromptMode("duration");
      inp.value = (seed && seed !== "session") ? seed : "";
      seedRangeDates(null, null);
    }
    box.hidden = false;
    setTimeout(() => {
      if (lbPromptMode() === "range") {
        const fromEl = $("#lc-lb-from");
        if (fromEl) fromEl.focus();
      } else {
        inp.focus();
        inp.select();
      }
    }, 0);
  }
  function closeLookbackPrompt() {
    const box = $("#lc-lb-prompt");
    if (box) box.hidden = true;
  }
  function pauseLiveForHistoricalRange(toYmd) {
    const today = ctTodayYmd();
    if (!toYmd || toYmd >= today) return;
    const live = $("#lc-live");
    if (live && live.checked) {
      live.checked = false;
      localStorage.setItem("lc_live", "0");
      armPoll();
    }
  }
  function applyLookbackPrompt() {
    let token = null;
    if (lbPromptMode() === "range") {
      const fromEl = $("#lc-lb-from");
      const toEl = $("#lc-lb-to");
      token = parseLookbackRange(`${(fromEl && fromEl.value) || ""}..${(toEl && toEl.value) || ""}`);
      if (!token) {
        toast("Pick From and To dates (To on or after From)", "err");
        if (fromEl) fromEl.focus();
        return;
      }
      pauseLiveForHistoricalRange(token.split("..")[1]);
    } else {
      const inp = $("#lc-lb-custom");
      token = parseLookbackToken(inp && inp.value);
      if (!token) {
        toast("Use a duration like 6h, 3d, 2w, 90d, 1y, or session", "err");
        if (inp) inp.focus();
        return;
      }
    }
    closeLookbackPrompt();
    paintLookbacks(token);
    lastLb = token;
    fitted = false;
    loadSeries();
  }

  function destroyChart() {
    clearProfilePriceLines();
    if (chart) { try { chart.remove(); } catch {} }
    chart = null; candleSeries = null; volSeries = null;
    lastSnap = null;
    cvdCandleSeries = null;
    oscLayoutKey = "";
    cvdPaneIndex = null;
    Object.keys(overlaySeries).forEach(k => delete overlaySeries[k]);
    const cv = $("#lc-vp-canvas");
    if (cv) {
      const ctx = cv.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, cv.width, cv.height);
    }
    const hover = $("#lc-hover");
    if (hover) hover.hidden = true;
    const axis = $("#lc-vp-axis");
    if (axis) { axis.hidden = true; axis.innerHTML = ""; }
    const host = document.querySelector(".lc-chart-host");
    if (host) host.classList.remove("lc-vp-axis-on");
  }

  function originOptions(keys) {
    const lab = (o) => (meta && meta.origin_labels && meta.origin_labels[o]) || ORIGIN_LABEL[o] || o;
    return (keys || []).map(o =>
      `<option value="${esc(o)}">${esc(lab(o))}</option>`).join("");
  }
  function sessionTfOptions(selected) {
    const keys = (meta && meta.session_tfs) || ["full_session", "cash", "premarket", "bod", "custom"];
    const labs = (meta && meta.session_tf_labels) || {
      full_session: "Full session (5:00pm–4:00pm CT)",
      cash: "Cash session (8:30am–3:00pm CT)",
      premarket: "Premarket (8:00am–8:30am CT)",
      bod: "BOD (midnight–midnight CT)",
      custom: "Custom",
    };
    return keys.map(k =>
      `<option value="${esc(k)}"${k === selected ? " selected" : ""}>${esc(labs[k] || k)}</option>`).join("");
  }
  function overlayParamsHtml(o) {
    if (o.params === "mbo") return window.LCOrders ? LCOrders.paramsHtml() : "";
    const origins = (meta && meta.origins && meta.origins.length)
      ? meta.origins : ["session", "bod", "mkt"];
    if (o.params === "cvd") {
      return `
        <div class="lc-info-row">
          <label>CVD_CUM origin
            <select id="lc-cvd-origin">${originOptions(origins)}</select></label>
          ${lcInfoTip("Line is running CVD (resets at Session 5:00pm CT / BOD / MKT 8:30am CT). Candle body (close−open) is that bar’s signed volume on 5s/15s/1m, except the first bar and origin resets. On 5m+ the body is the 1-minute CVD path inside the bar — not the full-bar Delta overlay. Delta is the per-bar increment as its own pane.")}
        </div>
        <label>CVD plot
          <select id="lc-cvd-style">
            <option value="line">Line</option>
            <option value="candle">Candles</option>
            <option value="both">Line + candles</option>
          </select></label>`;
    }
    if (o.key === "vw_vwap") {
      const bbOn = localStorage.getItem("lc_vwap_bb") === "1";
      return `
        <div class="lc-info-row">
          <label>VWAP reset
            <select id="lc-vwap-origin">${originOptions(origins)}</select></label>
          ${lcInfoTip("VWAP resets at the selected clock: Session 5:00pm CT, BOD midnight CT, or MKT open 8:30am CT.")}
        </div>
        <div class="lc-info-row">
          <label class="checkbox"><input type="checkbox" id="lc-vwap-bb"${bbOn ? " checked" : ""}> Bollinger bands (±2σ)</label>
          ${lcInfoTip("Upper / lower = VWAP ± 2·σ on the price pane. σ is the volume-weighted std of price around VWAP over the same reset window (same math as the native 5m hive). No separate length — the window is the VWAP origin.")}
        </div>
        <div class="lc-info-row">
          <label class="checkbox"><input type="checkbox" id="lc-vwap-bb-arrows"${localStorage.getItem("lc_vwap_bb_arrows") === "1" ? " checked" : ""}> Band arrows</label>
          ${lcInfoTip("When bands are on: a red arrow when the close crosses down through the upper band, a green arrow when it crosses up through the lower band. A thin vertical line in that color spans the price pane.")}
        </div>`;
    }
    if (o.params === "ofi") {
      return `
        <div class="lc-info-row">
          <label>OFI origin
            <select id="lc-ofi-origin">${originOptions(origins)}</select></label>
          ${lcInfoTip("OFI_CUM resets at this clock, independent of CVD.")}
        </div>`;
    }
    if (o.params === "ad") {
      return `
        <label>A/D origin
          <select id="lc-ad-origin">${originOptions(origins)}</select></label>`;
    }
    if (o.key === "mf_mfi") {
      const saved = lcNormalizeOscBounds(
        localStorage.getItem("lc_mfi_upper"),
        localStorage.getItem("lc_mfi_lower"),
      );
      const arrowsOn = localStorage.getItem("lc_mfi_arrows") === "1";
      return `
        <div class="lc-info-row">
          <label>Upper bound
            <input type="number" id="lc-mfi-upper" min="0" max="100" step="0.1" value="${esc(String(saved.upper))}"></label>
          ${lcInfoTip("Dashed line on the MFI pane. Default 80. Bound arrows use this level.")}
        </div>
        <div class="lc-info-row">
          <label>Lower bound
            <input type="number" id="lc-mfi-lower" min="0" max="100" step="0.1" value="${esc(String(saved.lower))}"></label>
          ${lcInfoTip("Dashed line on the MFI pane. Default 20. Bound arrows use this level.")}
        </div>
        <div class="lc-info-row">
          <label class="checkbox"><input type="checkbox" id="lc-mfi-arrows"${arrowsOn ? " checked" : ""}> Bound arrows</label>
          ${lcInfoTip("Red arrow down when MFI crosses down through the upper bound. Green arrow up when it crosses up through the lower bound. A thin vertical line spans the MFI pane.")}
        </div>`;
    }
    if (o.params === "prints") {
      const defMin = (meta && meta.defaults && meta.defaults.print_min) || 10;
      return `
        <div class="lc-info-row">
          <label>Min size (contracts)
            <input type="number" id="lc-print-min" min="1" max="10000" step="1" value="${esc(String(defMin))}"></label>
          ${lcInfoTip("Large prints: TBBO fills in this chart’s lookback with size ≥ min. Each bubble is centered on its candle at the fill price, with a diameter of 90% of the displayed bar width. Hover a bubble for price, size, and Databento ts_event. Cached while this overlay stays on.")}
        </div>
        <label>Opacity <output id="lc-print-opacity-val">88%</output>
          <input type="range" id="lc-print-opacity" min="0" max="100" step="1" value="88"></label>
        <label class="checkbox"><input type="checkbox" id="lc-print-display-value" checked> Display value</label>
        <label>Extend level (bars)
          <input type="number" id="lc-print-extend-bars" min="0" max="5000" step="1" value="0"></label>`;
    }
    return "";
  }
  function vpKindLabel(kind) {
    const labs = (meta && meta.vp_kind_labels) || {
      current: "Single", recurring: "Recurring", level_blend: "Level Blend",
    };
    if (kind === "custom") return labs.current || "Single";
    return labs[kind] || kind;
  }
  function isLevelBlend(kind) {
    return kind === "level_blend";
  }
  function newVpId() {
    return "vp" + Math.random().toString(36).slice(2, 8);
  }
  function hmFromDatetimeLocal(raw) {
    const text = String(raw || "").trim();
    if (!text) return "";
    if (text.includes("T")) {
      const clock = text.split("T")[1] || "";
      return clock.slice(0, 5);
    }
    if (/^\d{1,2}:\d{2}/.test(text)) return text.slice(0, 5);
    return "";
  }
  function dateFromDatetimeLocal(raw) {
    const text = String(raw || "").trim();
    if (!text) return "";
    if (/^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
    return "";
  }
  function lastChartSessionDate() {
    if (payload && payload.t && payload.t.length) {
      return globexSessionEndDate(payload.t[payload.t.length - 1]);
    }
    return utcMsToCtParts(Date.now()).date;
  }
  function migrateVpSpec(s) {
    if (!s || typeof s !== "object") return s;
    const out = Object.assign({}, s);
    if (out.hist_style === "lvn_nodes") {
      out.hist_style = "binned";
      if (out.group_ticks == null) out.group_ticks = 4;
      out.show_lvn = !!out.show_lvn;
    }
    if (out.kind === "custom") {
      out.kind = "current";
      if (!out.session_tf || out.session_tf === "full_session") out.session_tf = "custom";
      out.session_from = out.session_from || hmFromDatetimeLocal(out.range_start) || "08:30";
      out.session_to = out.session_to || hmFromDatetimeLocal(out.range_end) || "15:00";
      out.session_date = out.session_date
        || dateFromDatetimeLocal(out.range_end)
        || dateFromDatetimeLocal(out.range_start)
        || "";
    }
    if (!out.session_from) out.session_from = "08:30";
    if (!out.session_to) out.session_to = "15:00";
    if (out.group_ticks == null) out.group_ticks = 4;
    const inwardLeft = out.placement === "right_frame"
      || (out.placement === "overlay" && out.overlay_side === "right");
    if (!out.left_display && !out.right_display) {
      out.left_display = inwardLeft ? "total_split" : "none";
      out.right_display = inwardLeft ? "none" : "total_split";
    }
    if (out.left_opacity == null) out.left_opacity = 100;
    if (out.right_opacity == null) out.right_opacity = 100;
    if (out.chart_labels == null) out.chart_labels = false;
    if (out.axis_labels == null) out.axis_labels = false;
    if (out.hvn_method == null) out.hvn_method = "off";
    if (out.hvn_method === "adaptive_kde" && Number(out.kde_separation) >= 10) {
      out.kde_bandwidth = 1;
      out.kde_prominence = .02;
      out.kde_valley_depth = .02;
      out.kde_volume = .01;
      out.kde_separation = 2;
      out.kde_valley_width = 2;
      out.kde_max_zones = 10;
    }
    if (out.kde_bandwidth == null) out.kde_bandwidth = 2;
    if (out.kde_prominence == null) out.kde_prominence = .05;
    if (out.kde_valley_depth == null) out.kde_valley_depth = .05;
    if (out.kde_volume == null) out.kde_volume = .05;
    if (out.kde_separation == null) out.kde_separation = 2;
    if (out.kde_valley_width == null) out.kde_valley_width = 2;
    if (out.kde_max_zones == null) out.kde_max_zones = 5;
    const histWidth = Number(out.hist_width);
    out.hist_width = Number.isFinite(histWidth)
      && histWidth > 0 ? histWidth : .25;
    if ((out.kind === "current" || out.kind === "level_blend")
        && out.session_date == null) out.session_date = "";
    if (out.lookback_sessions == null || out.lookback_sessions === "") {
      out.lookback_sessions = (meta && meta.level_blend_lookback_default) || 20;
    }
    if (out.extend == null || out.extend === "") {
      if (out.kind === "level_blend") out.extend = 3;
      else if (out.extend_lines === false && !out.extend_va) out.extend = 0;
      else if (out.extend_lines || out.extend_va || out.extend_lines == null) out.extend = 1;
      else out.extend = 0;
    } else {
      out.extend = parseVpExtend(out.extend);
    }
    out.extend_lines = out.extend >= 1;
    out.extend_va = out.extend >= 1;
    if (out.kind === "level_blend") {
      out.session_tf = "cash";
      out.hist_style = "volume";
      out.show_lvn = false;
      out.extend_lvn = false;
      out.hvn_method = "off";
      out.right_display = "none";
      out.left_display = "none";
      out.plot_variable_bounds = false;
      // Bounds-only default; missing key on older stored specs → hide members.
      out.show_members = out.show_members === true;
    } else if (out.show_members == null) {
      out.show_members = true;
    }
    // Missing key on older stored specs → keep drawing (on by default).
    out.enabled = out.enabled !== false;
    return out;
  }
  function nextVpN() {
    let max = 0;
    for (const s of [...vpSpecs, ...vpDraftSpecs]) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > max) max = n;
    }
    return max + 1;
  }
  function assignVpNs(specs) {
    const used = new Set();
    for (const s of specs) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > 0 && !used.has(n)) {
        s.n = n;
        used.add(n);
      } else if (s) s.n = 0;
    }
    let next = 1;
    for (const s of specs) {
      if (!s || s.n) continue;
      while (used.has(next)) next += 1;
      s.n = next;
      used.add(next);
    }
    return specs;
  }
  function vpMark(n) {
    if (typeof n === "string") {
      const m = n.trim().match(/^VP\s*(\d+)$/i);
      if (m) return "VP" + m[1];
    }
    const i = Number(n);
    return "VP" + (Number.isInteger(i) && i > 0 ? i : 1);
  }
  function defaultVpOverlaySide(n) {
    return n % 2 === 0 ? "right" : "left";
  }
  function defaultVpSpec(kind) {
    const k = kind === "custom" ? "current" : (kind || "current");
    const n = nextVpN();
    const blend = k === "level_blend";
    return {
      id: newVpId(),
      n,
      kind: k,
      enabled: true,
      session_tf: blend ? "cash" : "full_session",
      placement: "overlay",
      overlay_side: defaultVpOverlaySide(n),
      extend: blend ? 3 : 1,
      extend_lines: true,
      extend_va: true,
      hist_style: "volume",
      group_ticks: 4,
      right_display: blend ? "none" : (n % 2 === 0 ? "none" : "total_split"),
      left_display: blend ? "none" : (n % 2 === 0 ? "total_split" : "none"),
      right_opacity: 100,
      left_opacity: 100,
      hist_width: blend ? 0.35 : .25,
      chart_labels: false,
      axis_labels: false,
      show_lvn: false,
      lvn_pct: 0.30,
      extend_lvn: false,
      hvn_method: "off",
      kde_bandwidth: 2,
      kde_prominence: .05,
      kde_valley_depth: .05,
      kde_volume: .05,
      kde_separation: 2,
      kde_valley_width: 2,
      kde_max_zones: 5,
      lookback_sessions: (meta && meta.level_blend_lookback_default) || 20,
      show_members: blend ? false : true,
      recur: "daily",
      session_from: "08:30",
      session_to: "15:00",
      session_date: (k === "current" || blend) ? lastChartSessionDate() : "",
    };
  }
  function loadVpSpecs() {
    const raw = localStorage.getItem("lc_vp_specs");
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return assignVpNs(parsed.map(migrateVpSpec));
      } catch {}
    }
    const specs = [];
    const origMap = { session: "full_session", bod: "bod", mkt: "cash", premarket: "premarket" };
    if (localStorage.getItem("lc_svp") === "1") {
      const orig = localStorage.getItem("lc_svp_origin") || "session";
      specs.push(migrateVpSpec({
        id: "legacy-svp", n: 1, kind: "current",
        session_tf: origMap[orig] || "full_session",
        placement: "left_frame", overlay_side: "left", extend: 1, extend_lines: true, extend_va: true,
        hist_style: "volume", group_ticks: 4, right_display: "total_split", left_display: "none", right_opacity: 100, left_opacity: 100, show_lvn: false, lvn_pct: 0.30, extend_lvn: false,
        recur: "daily", session_from: "08:30", session_to: "15:00", session_date: "",
      }));
    }
    if (localStorage.getItem("lc_vp") === "1") {
      specs.push(migrateVpSpec({
        id: "legacy-vp", n: 1, kind: "custom", session_tf: "full_session",
        placement: "right_frame", overlay_side: "right", extend: 1, extend_lines: true, extend_va: true,
        hist_style: "volume", group_ticks: 4, right_display: "total_split", left_display: "none", right_opacity: 100, left_opacity: 100, show_lvn: false, lvn_pct: 0.30, extend_lvn: false,
        recur: "daily",
        range_start: localStorage.getItem("lc_vp_start") || "",
        range_end: localStorage.getItem("lc_vp_end") || "",
      }));
    }
    return assignVpNs(specs);
  }
  function persistVpSpecs() {
    localStorage.setItem("lc_vp_specs", JSON.stringify(vpSpecs));
    localStorage.setItem("lc_vp", vpOn() ? "1" : "0");
  }
  function readVpSpecsFromDom() {
    return [...document.querySelectorAll(".lc-vp-card")].map(card => {
      const val = (f) => {
        const el = card.querySelector(`[data-f="${f}"]`);
        if (!el) return "";
        if (el.type === "checkbox") return el.checked;
        return el.value;
      };
      const kind = card.dataset.kind === "custom" ? "current" : card.dataset.kind;
      const blend = isLevelBlend(kind);
      const opacityOr = (f, fallback) => {
        const raw = val(f);
        if (raw === "" || raw == null) return fallback;
        const n = Number(raw);
        return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : fallback;
      };
      // Number("") === 0 — treat blank missing controls as the default, not zero.
      const numOr = (f, fallback, { pct = false } = {}) => {
        const raw = val(f);
        if (raw === "" || raw == null) return fallback;
        const n = Number(raw);
        if (!Number.isFinite(n)) return fallback;
        return pct && n > 1 ? n / 100 : n;
      };
      return {
        id: card.dataset.vpId,
        n: Number(card.dataset.vpN) || 1,
        kind,
        enabled: card.querySelector('[data-f="enabled"]') ? !!val("enabled") : true,
        session_tf: blend ? "cash" : (val("session_tf") || "full_session"),
        placement: val("placement") || "overlay",
        overlay_side: val("overlay_side") || "left",
        extend: parseVpExtend(val("extend")),
        extend_lines: parseVpExtend(val("extend")) >= 1,
        extend_va: parseVpExtend(val("extend")) >= 1,
        hist_style: blend ? "volume" : (val("hist_style") === "lvn_nodes" ? "binned" : (val("hist_style") || "volume")),
        group_ticks: numOr("group_ticks", 4),
        right_display: blend ? "none" : (val("right_display") || "total_split"),
        left_display: blend ? "none" : (val("left_display") || "none"),
        right_opacity: opacityOr("right_opacity", 100),
        left_opacity: opacityOr("left_opacity", 100),
        hist_width: (() => {
          const n = numOr("hist_width", .25);
          return n > 0 ? n : .25;
        })(),
        chart_labels: !!val("chart_labels"),
        axis_labels: !!val("axis_labels"),
        plot_variable_bounds: blend ? false : !!val("plot_variable_bounds"),
        show_lvn: blend ? false : !!val("show_lvn"),
        lvn_min_groups: Math.max(1, Math.min(20, numOr("lvn_min_groups", 1))),
        lvn_pct: numOr("lvn_pct", 0.30, { pct: true }),
        extend_lvn: blend ? false : !!val("extend_lvn"),
        hvn_method: blend ? "off" : (val("hvn_method") || "off"),
        kde_bandwidth: numOr("kde_bandwidth", 2),
        kde_prominence: numOr("kde_prominence", .05, { pct: true }),
        kde_valley_depth: numOr("kde_valley_depth", .05, { pct: true }),
        kde_volume: numOr("kde_volume", .05, { pct: true }),
        kde_separation: numOr("kde_separation", 2),
        kde_valley_width: numOr("kde_valley_width", 2),
        kde_max_zones: numOr("kde_max_zones", 5),
        lookback_sessions: Math.max(1, Math.min(120, numOr("lookback_sessions", 20))),
        show_members: blend ? !!val("show_members") : true,
        recur: val("recur") || "daily",
        session_from: val("session_from") || "08:30",
        session_to: val("session_to") || "15:00",
        session_date: val("session_date") || "",
      };
    });
  }
  function loadVpCollapsed() {
    try {
      const raw = localStorage.getItem("lc_vp_collapsed");
      const parsed = raw ? JSON.parse(raw) : {};
      return (parsed && typeof parsed === "object") ? parsed : {};
    } catch {
      return {};
    }
  }
  function persistVpCollapsed(map) {
    localStorage.setItem("lc_vp_collapsed", JSON.stringify(map || {}));
  }
  function isVpCardCollapsed(id) {
    return !!loadVpCollapsed()[String(id)];
  }
  function setVpCardCollapsed(id, collapsed) {
    const map = loadVpCollapsed();
    if (collapsed) map[String(id)] = true;
    else delete map[String(id)];
    persistVpCollapsed(map);
  }
  function syncVpDraftFromDom() {
    vpDraftSpecs = assignVpNs(readVpSpecsFromDom());
  }
  function vpCardHtml(spec) {
    const placeLabs = (meta && meta.vp_placement_labels) || {
      overlay: "Overlay", left_frame: "Left", right_frame: "Right",
    };
    const recurLabs = (meta && meta.vp_recur_labels) || { daily: "Daily", weekly: "Weekly" };
    const places = (meta && meta.vp_placements) || ["overlay", "left_frame", "right_frame"];
    const recs = (meta && meta.vp_recur) || ["daily", "weekly"];
    const kind = spec.kind === "custom" ? "current" : spec.kind;
    const blend = isLevelBlend(kind);
    const sessionTf = blend ? "cash" : (spec.session_tf || "full_session");
    const n = Number(spec.n) > 0 ? Number(spec.n) : 1;
    const extend = parseVpExtend(spec.extend);
    const extendLabs = (meta && meta.vp_extend_labels) || {
      0: "Off", 1: "1 sess", 2: "2 sess", 3: "All sess",
    };
    const lookbackDefault = (meta && meta.level_blend_lookback_default) || 20;
    const lookbackMin = (meta && meta.level_blend_lookback_min) || 1;
    const lookbackMax = (meta && meta.level_blend_lookback_max) || 120;
    const lookback = Math.max(lookbackMin, Math.min(lookbackMax,
      Number(spec.lookback_sessions) || lookbackDefault));
    const dateRow = (kind === "current" || blend) ? `
      <label class="lc-vp-date">Date
        <input type="date" data-f="session_date" value="${esc(spec.session_date || "")}"></label>` : "";
    const customTimes = blend ? "" : `
      <div class="lc-vp-custom-hm">
        <label>From
          <input type="time" data-f="session_from" step="60" value="${esc(spec.session_from || "08:30")}"></label>
        <label>To
          <input type="time" data-f="session_to" step="60" value="${esc(spec.session_to || "15:00")}"></label>
      </div>`;
    const recur = kind === "recurring" ? `
      <label>Type
        <select data-f="recur">${recs.map(k =>
          `<option value="${esc(k)}"${k === spec.recur ? " selected" : ""}>${esc(recurLabs[k] || k)}</option>`).join("")}</select></label>` : "";
    const histLabs = {
      volume: "Tick histogram",
      binned: "Grouped ticks",
    };
    const groupTicks = (meta && meta.group_ticks) || [1, 2, 4, 8, 16, 32, 64];
    const displayModes = (meta && meta.vp_display_modes) || ["total", "total_split", "total_delta_split", "buy", "sell", "delta", "none"];
    const displayLabels = (meta && meta.vp_display_mode_labels) || {
      total: "Total volume", total_split: "Total volume split", buy: "Buy volume",
      total_delta_split: "Total Volume delta split", sell: "Sell volume",
      delta: "Delta volume", none: "None",
    };
    const hists = ["volume", "binned"];
    let histStyle = spec.hist_style || "volume";
    const showLvn = !!spec.show_lvn;
    if (histStyle === "lvn_nodes") histStyle = "binned";
    const groupTicksValue = groupTicks.includes(Number(spec.group_ticks)) ? Number(spec.group_ticks) : 4;
    const rightDisplay = spec.right_display || "total_split";
    const leftDisplay = spec.left_display || "none";
    const rightOpacity = Number.isFinite(Number(spec.right_opacity)) ? Math.min(100, Math.max(0, Number(spec.right_opacity))) : 100;
    const leftOpacity = Number.isFinite(Number(spec.left_opacity)) ? Math.min(100, Math.max(0, Number(spec.left_opacity))) : 100;
    const histWidth = Number.isFinite(Number(spec.hist_width)) ? Number(spec.hist_width) : .25;
    const lvnPct = spec.lvn_pct != null ? spec.lvn_pct : 0.30;
    const lvnPctUi = lvnPct <= 1 ? Math.round(lvnPct * 100) : Math.round(lvnPct);
    const kdeDefaults = (meta && meta.kde_defaults) || { bandwidth: 2, prominence: .05, valley_depth: .05, volume: .05, separation: 2, valley_width: 2, max_zones: 5 };
    const hvnMethods = (meta && meta.hvn_methods) || ["off", "adaptive_kde"];
    const hvnMethod = spec.hvn_method || "off";
    const showMembers = blend ? !!spec.show_members : true;
    const sessionRow = blend ? `
      <div class="lc-info-row">
        <label class="lc-vp-blend-lookback">Prior sessions
          <input type="number" data-f="lookback_sessions" min="${lookbackMin}" max="${lookbackMax}" step="1" value="${esc(String(lookback))}"></label>
        ${lcInfoTip(`Cash session levels only (08:30–15:00 CT). Prior ${lookbackDefault} sessions are KDE-merged into bounds (+1 pt buffer); lookback auto-extends when the eval day's range sits outside. Extend stretches toward the opposite chart edge (Place/Side Left → right; Right → left). Default All sess spans the visible plot.`)}
      </div>
      <label class="checkbox">
        <input type="checkbox" data-f="show_members"${showMembers ? " checked" : ""}>
        Member lines</label>` : `
      <label>Session
        <select data-f="session_tf">${sessionTfOptions(sessionTf)}</select></label>
      ${customTimes}`;
    const histBlock = blend ? "" : `
      <label>Histogram
        <select data-f="hist_style">${hists.map(k =>
          `<option value="${esc(k)}"${k === histStyle ? " selected" : ""}>${esc(histLabs[k] || k)}</option>`).join("")}</select></label>
      <label class="lc-vp-bins">Group ticks
        <select data-f="group_ticks">${groupTicks.map(k =>
          `<option value="${k}"${k === groupTicksValue ? " selected" : ""}>${k}</option>`).join("")}</select></label>
      <label>Right display
        <select data-f="right_display">${displayModes.map(k =>
          `<option value="${esc(k)}"${k === rightDisplay ? " selected" : ""}>${esc(displayLabels[k] || k)}</option>`).join("")}</select></label>
      <label>Right opacity
        <input type="range" data-f="right_opacity" min="0" max="100" step="1" value="${esc(String(rightOpacity))}"></label>
      <label>Left display
        <select data-f="left_display">${displayModes.map(k =>
          `<option value="${esc(k)}"${k === leftDisplay ? " selected" : ""}>${esc(displayLabels[k] || k)}</option>`).join("")}</select></label>
      <label>Left opacity
        <input type="range" data-f="left_opacity" min="0" max="100" step="1" value="${esc(String(leftOpacity))}"></label>`;
    const lvnBlock = blend ? "" : `
      <label class="checkbox">
        <input type="checkbox" data-f="plot_variable_bounds"${spec.plot_variable_bounds ? " checked" : ""}>
        Plot variable bounds</label>
      <label class="checkbox">
        <input type="checkbox" data-f="show_lvn"${showLvn ? " checked" : ""}>
        LVN</label>
      <label class="lc-vp-lvn">LVN %
        <input type="number" data-f="lvn_pct" min="10" max="50" step="5" value="${esc(String(lvnPctUi))}"></label>
      <label class="lc-vp-lvn">LVN min grouped bars
        <input type="number" data-f="lvn_min_groups" min="1" max="20" step="1" value="${esc(String(spec.lvn_min_groups || 1))}"></label>
      <label>HVN zones
        <select data-f="hvn_method">${hvnMethods.map(k =>
          `<option value="${esc(k)}"${k === hvnMethod ? " selected" : ""}>${k === "adaptive_kde" ? "Adaptive KDE" : "Off"}</option>`).join("")}</select></label>
      <div class="lc-vp-kde"${hvnMethod === "adaptive_kde" ? "" : " hidden"}>
        <label>KDE bandwidth (bins)
          <input type="number" data-f="kde_bandwidth" min="1" max="12" step="0.5" value="${esc(String(spec.kde_bandwidth ?? kdeDefaults.bandwidth))}"></label>
        <label>Min prominence (%)
          <input type="number" data-f="kde_prominence" min="2" max="50" step="1" value="${esc(String(Math.round((spec.kde_prominence ?? kdeDefaults.prominence) * 100)))}"></label>
        <label>Min valley depth (%)
          <input type="number" data-f="kde_valley_depth" min="2" max="50" step="1" value="${esc(String(Math.round((spec.kde_valley_depth ?? kdeDefaults.valley_depth) * 100)))}"></label>
        <label>Min zone volume (%)
          <input type="number" data-f="kde_volume" min="1" max="50" step="1" value="${esc(String(Math.round((spec.kde_volume ?? kdeDefaults.volume) * 100)))}"></label>
        <label>Min peak separation (bins)
          <input type="number" data-f="kde_separation" min="1" max="20" step="1" value="${esc(String(spec.kde_separation ?? kdeDefaults.separation))}"></label>
        <label>Min valley width (grouped bars)
          <input type="number" data-f="kde_valley_width" min="1" max="20" step="1" value="${esc(String(spec.kde_valley_width ?? kdeDefaults.valley_width))}"></label>
        <label>Max HVN zones
          <input type="number" data-f="kde_max_zones" min="1" max="10" step="1" value="${esc(String(spec.kde_max_zones ?? kdeDefaults.max_zones))}"></label>
        <div class="lc-info-row">
          <button type="button" class="small lc-kde-preset" data-kde-preset="multi">Use multi-section preset</button>
          ${lcInfoTip("For dense sectioning: bandwidth 1, prominence/depth 2%, zone volume 1%, separation 1–2 grouped bars, max 10 zones. Separation 10+ merges nearby sections.")}
        </div>
      </div>`;
    const extendLvn = blend ? "" : `
      <label class="checkbox lc-vp-lvn">
        <input type="checkbox" data-f="extend_lvn"${spec.extend_lvn ? " checked" : ""}>
        Extend LVN</label>`;
    const collapsed = isVpCardCollapsed(spec.id);
    const enabled = spec.enabled !== false;
    return `<div class="lc-vp-card${collapsed ? " is-collapsed" : ""}${enabled ? "" : " is-disabled"}" data-vp-id="${esc(spec.id)}" data-vp-n="${esc(String(n))}" data-kind="${esc(kind)}" data-session-tf="${esc(sessionTf)}" data-placement="${esc(spec.placement || "overlay")}" data-hist="${esc(histStyle)}" data-lvn="${showLvn ? "1" : "0"}" data-enabled="${enabled ? "1" : "0"}">
      <div class="lc-vp-card-h"><span>${esc(vpMark(n))}<span class="lc-vp-kind">${esc(vpKindLabel(kind))}</span></span>
        <div class="lc-vp-card-h-actions">
          <label class="checkbox lc-vp-enabled" title="Show this volume profile on the chart (Apply to update)">
            <input type="checkbox" data-f="enabled"${enabled ? " checked" : ""}> On</label>
          <button type="button" class="lc-vp-collapse" data-vp-collapse="${esc(spec.id)}"
            aria-expanded="${collapsed ? "false" : "true"}"
            title="${collapsed ? "Expand" : "Collapse"}"
            aria-label="${collapsed ? "Expand" : "Collapse"} volume profile card">${collapsed ? "▸" : "▾"}</button>
          <button type="button" data-vp-remove="${esc(spec.id)}" title="Remove">Remove</button>
        </div></div>
      <div class="lc-vp-card-body">
      ${recur}
      ${dateRow}
      ${sessionRow}
      <label>Place
        <select data-f="placement">${places.map(k =>
          `<option value="${esc(k)}"${k === spec.placement ? " selected" : ""}>${esc(placeLabs[k] || k)}</option>`).join("")}</select></label>
      <label class="lc-vp-side">Side
        <select data-f="overlay_side">
          <option value="left"${spec.overlay_side === "left" ? " selected" : ""}>Left</option>
          <option value="right"${spec.overlay_side === "right" ? " selected" : ""}>Right</option>
        </select></label>
      ${histBlock}
      <label>Width
        <span class="lc-vp-width-stepper">
          <input type="number" data-f="hist_width" min="0.01" step="any" value="${esc(String(histWidth))}">
          <span class="lc-vp-width-buttons">
            <button type="button" data-vp-width-step="1" title="Increase width by 0.05" aria-label="Increase width by 0.05">+</button>
            <button type="button" data-vp-width-step="-1" title="Decrease width by 0.05" aria-label="Decrease width by 0.05">-</button>
          </span>
        </span></label>
      <div class="lc-vp-labels"><span>Labels:</span>
        <label class="checkbox"><input type="checkbox" data-f="chart_labels"${spec.chart_labels ? " checked" : ""}> Chart</label>
        <label class="checkbox"><input type="checkbox" data-f="axis_labels"${spec.axis_labels ? " checked" : ""}> Axis</label>
      </div>
      ${lvnBlock}
      <label>Extend
        <select data-f="extend">${[0, 1, 2, 3].map(k =>
          `<option value="${k}"${extend === k ? " selected" : ""}>${esc(extendLabs[k] || extendLabs[String(k)] || String(k))}</option>`).join("")}</select></label>
      ${extendLvn}
      </div>
    </div>`;
  }
  function paintVpCards() {
    const host = $("#lc-vp-list");
    if (!host) return;
    host.innerHTML = vpDraftSpecs.map(vpCardHtml).join("")
      || `<p class="muted">Add VP1, VP2, …</p>`;
  }
  function profilesForApi() {
    // enabled === false keeps the card/settings but skips the chart request.
    return vpSpecs.filter(s => s && s.enabled !== false).map(s => {
      const kind = s.kind === "custom" ? "current" : s.kind;
      const blend = isLevelBlend(kind);
      const extend = parseVpExtend(s.extend);
      const row = {
        id: s.id, n: Number(s.n) > 0 ? Number(s.n) : 1, kind, enabled: true,
        session_tf: blend ? "cash" : s.session_tf,
        placement: s.placement, overlay_side: s.overlay_side,
        extend,
        extend_lines: extend >= 1,
        extend_va: extend >= 1,
        hist_style: blend ? "volume" : (s.hist_style === "lvn_nodes" ? "binned" : (s.hist_style || "volume")),
        group_ticks: s.group_ticks != null ? s.group_ticks : 4,
        right_display: blend ? "none" : (s.right_display || "total_split"),
        left_display: blend ? "none" : (s.left_display || "none"),
        right_opacity: s.right_opacity != null ? s.right_opacity : 100,
        left_opacity: s.left_opacity != null ? s.left_opacity : 100,
        hist_width: s.hist_width != null ? s.hist_width : .25,
        chart_labels: !!s.chart_labels,
        axis_labels: !!s.axis_labels,
        plot_variable_bounds: blend ? false : !!s.plot_variable_bounds,
        show_lvn: blend ? false : !!s.show_lvn,
        lvn_min_groups: Math.max(1, Math.min(20, Number(s.lvn_min_groups) || 1)),
        lvn_pct: s.lvn_pct != null ? s.lvn_pct : 0.30,
        extend_lvn: blend ? false : !!s.extend_lvn,
        hvn_method: blend ? "off" : (s.hvn_method || "off"),
        kde_bandwidth: s.kde_bandwidth != null ? s.kde_bandwidth : 4,
        kde_prominence: s.kde_prominence != null ? (s.kde_prominence > 1 ? s.kde_prominence / 100 : s.kde_prominence) : .05,
        kde_valley_depth: s.kde_valley_depth != null ? (s.kde_valley_depth > 1 ? s.kde_valley_depth / 100 : s.kde_valley_depth) : .05,
        kde_volume: s.kde_volume != null ? (s.kde_volume > 1 ? s.kde_volume / 100 : s.kde_volume) : .05,
        kde_separation: s.kde_separation != null ? s.kde_separation : 2,
        kde_valley_width: s.kde_valley_width != null ? s.kde_valley_width : 2,
        kde_max_zones: s.kde_max_zones != null ? s.kde_max_zones : 5,
        lookback_sessions: Math.max(1, Math.min(120, Number(s.lookback_sessions) || 20)),
      };
      if (kind === "recurring") row.recur = s.recur || "daily";
      if (!blend && s.session_tf === "custom") {
        row.session_from = s.session_from || "08:30";
        row.session_to = s.session_to || "15:00";
      }
      if ((kind === "current" || blend) && s.session_date) row.session_date = s.session_date;
      return row;
    });
  }
  function applyVpDisplaySettings(response) {
    if (!response || !Array.isArray(response.profiles)) return response;
    const byId = new Map(vpSpecs.map(spec => [String(spec.id), spec]));
    for (const profile of response.profiles) {
      const spec = byId.get(String(profile && (profile.spec_id || profile.id)));
      if (!profile || !spec) continue;
      const width = Number(spec.hist_width);
      if (Number.isFinite(width) && width > 0) profile.hist_width = width;
      if (isLevelBlend(profile.kind) || profile.level_blend || isLevelBlend(spec.kind)) {
        profile.show_members = !!spec.show_members;
      }
    }
    return response;
  }

  function emaLookbackDefault() {
    const n = Number(meta && meta.ema_lookback_default);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 21;
  }
  function emaLookbackMin() {
    const n = Number(meta && meta.ema_lookback_min);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 1;
  }
  function emaLookbackMax() {
    const n = Number(meta && meta.ema_lookback_max);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 500;
  }
  function emaMaxSpecs() {
    const n = Number(meta && meta.ema_max);
    return Number.isFinite(n) && n >= 1 ? Math.floor(n) : 8;
  }
  function clampEmaLookback(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n)) return emaLookbackDefault();
    return Math.min(emaLookbackMax(), Math.max(emaLookbackMin(), Math.floor(n)));
  }
  function newEmaId() {
    return "ema" + Math.random().toString(36).slice(2, 8);
  }
  function nextEmaN() {
    let max = 0;
    for (const s of [...emaSpecs, ...emaDraftSpecs]) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > max) max = n;
    }
    return max + 1;
  }
  function assignEmaNs(specs) {
    const used = new Set();
    for (const s of specs) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > 0 && !used.has(n)) {
        s.n = n;
        used.add(n);
      } else if (s) s.n = 0;
    }
    let next = 1;
    for (const s of specs) {
      if (!s || s.n) continue;
      while (used.has(next)) next += 1;
      s.n = next;
      used.add(next);
    }
    return specs;
  }
  function defaultEmaSpec(lookback) {
    const n = nextEmaN();
    return {
      id: newEmaId(),
      n,
      lookback: clampEmaLookback(lookback == null ? emaLookbackDefault() : lookback),
    };
  }
  function loadEmaSpecs() {
    const raw = localStorage.getItem("lc_ema_specs");
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return assignEmaNs(parsed.map(s => ({
            id: (s && s.id) || newEmaId(),
            n: Number(s && s.n) || 0,
            lookback: clampEmaLookback(s && s.lookback),
          })));
        }
      } catch {}
    }
    return [];
  }
  function persistEmaSpecs() {
    localStorage.setItem("lc_ema_specs", JSON.stringify(emaSpecs));
    localStorage.setItem("lc_ema", emaOn() ? "1" : "0");
  }
  function readEmaSpecsFromDom() {
    return [...document.querySelectorAll(".lc-ema-card")].map(card => {
      const input = card.querySelector('[data-f="lookback"]');
      return {
        id: card.dataset.emaId,
        n: Number(card.dataset.emaN) || 1,
        lookback: clampEmaLookback(input && input.value),
      };
    });
  }
  function syncEmaDraftFromDom() {
    emaDraftSpecs = assignEmaNs(readEmaSpecsFromDom());
  }
  function emasForApi() {
    return emaSpecs.map(s => ({
      id: s.id,
      n: s.n,
      lookback: clampEmaLookback(s.lookback),
    }));
  }
  function emaCardHtml(spec) {
    const n = Number(spec.n) > 0 ? Number(spec.n) : 1;
    const lookback = clampEmaLookback(spec.lookback);
    const color = emaColor(n);
    return `<div class="lc-ema-card" data-ema-id="${esc(spec.id)}" data-ema-n="${esc(String(n))}">
      <div class="lc-ema-card-h">
        <span><span class="lc-ema-swatch" style="background:${esc(color)}"></span>${esc(emaMark(n))}</span>
        <button type="button" data-ema-remove="${esc(spec.id)}">Remove</button>
      </div>
      <div class="lc-info-row">
        <label>Lookback bars
          <input type="number" data-f="lookback" min="${esc(String(emaLookbackMin()))}" max="${esc(String(emaLookbackMax()))}" step="1" value="${esc(String(lookback))}"></label>
        ${lcInfoTip(`EMA uses the last ${esc(String(lookback))} closes ending on the bar being plotted (current bar included).`)}
      </div>
    </div>`;
  }
  function paintEmaCards() {
    const host = $("#lc-ema-list");
    if (!host) return;
    const specs = emaDraftSpecs.length ? emaDraftSpecs : emaSpecs;
    host.innerHTML = specs.map(emaCardHtml).join("")
      || `<p class="muted">Add EMA1, EMA2, …</p>`;
    const box = $("#lc-ema");
    if (box) box.checked = emaEnabled;
  }
  function emaAddOptionsHtml() {
    const presets = (meta && meta.ema_presets && meta.ema_presets.length)
      ? meta.ema_presets : [9, 21, 50, 100, 200];
    const def = emaLookbackDefault();
    const opts = presets.map(p =>
      `<option value="${esc(String(p))}"${Number(p) === def ? " selected" : ""}>EMA ${esc(String(p))}</option>`
    ).join("");
    return opts + `<option value="custom">Custom…</option>`;
  }

  function stochKDefault() {
    return (meta && meta.stoch_k_default) || (meta && meta.defaults && meta.defaults.stoch_k_period) || 9;
  }
  function stochDDefault() {
    return (meta && meta.stoch_d_default) || (meta && meta.defaults && meta.defaults.stoch_d_period) || 3;
  }
  function stochKMin() { return (meta && meta.stoch_k_min) || 1; }
  function stochKMax() { return (meta && meta.stoch_k_max) || 200; }
  function stochDMin() { return (meta && meta.stoch_d_min) || 1; }
  function stochDMax() { return (meta && meta.stoch_d_max) || 50; }
  function clampStochK(v) {
    let n = Number(v);
    if (!Number.isFinite(n)) n = stochKDefault();
    return Math.min(stochKMax(), Math.max(stochKMin(), Math.floor(n)));
  }
  function clampStochD(v) {
    let n = Number(v);
    if (!Number.isFinite(n)) n = stochDDefault();
    return Math.min(stochDMax(), Math.max(stochDMin(), Math.floor(n)));
  }
  function newStochId() {
    return "stoch-" + Math.random().toString(36).slice(2, 10);
  }
  function nextStochN() {
    let max = 0;
    for (const s of [...stochSpecs, ...stochDraftSpecs]) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > max) max = n;
    }
    return max + 1;
  }
  function assignStochNs(specs) {
    const used = new Set();
    for (const s of specs) {
      const n = Number(s && s.n);
      if (Number.isInteger(n) && n > 0 && !used.has(n)) {
        s.n = n;
        used.add(n);
      } else if (s) s.n = 0;
    }
    let next = 1;
    for (const s of specs) {
      if (!s || s.n) continue;
      while (used.has(next)) next += 1;
      s.n = next;
      used.add(next);
    }
    return specs;
  }
  function defaultStochSpec(kPeriod, dPeriod) {
    const bounds = lcNormalizeOscBounds(80, 20);
    return {
      id: newStochId(),
      n: nextStochN(),
      k_period: clampStochK(kPeriod == null ? stochKDefault() : kPeriod),
      d_period: clampStochD(dPeriod == null ? stochDDefault() : dPeriod),
      upper: bounds.upper,
      lower: bounds.lower,
      arrows: false,
    };
  }
  function loadStochSpecs() {
    const raw = localStorage.getItem("lc_stoch_specs");
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          return assignStochNs(parsed.map(s => {
            const bounds = lcNormalizeOscBounds(s && s.upper, s && s.lower);
            return {
              id: (s && s.id) || newStochId(),
              n: Number(s && s.n) || 0,
              k_period: clampStochK(s && s.k_period),
              d_period: clampStochD(s && s.d_period),
              upper: bounds.upper,
              lower: bounds.lower,
              arrows: !!(s && s.arrows),
            };
          }));
        }
      } catch {}
    }
    // Migrate legacy single catalog overlay so_fast_d_9_3 → one Stoch card.
    const legacy = (localStorage.getItem("lc_overlays") || "").split(",").filter(Boolean);
    if (legacy.includes("so_fast_d_9_3")) {
      return [defaultStochSpec(9, 3)];
    }
    return [];
  }
  function persistStochSpecs() {
    localStorage.setItem("lc_stoch_specs", JSON.stringify(stochSpecs));
    localStorage.setItem("lc_stoch", stochOn() ? "1" : "0");
  }
  function readStochSpecsFromDom() {
    return [...document.querySelectorAll(".lc-stoch-card")].map(card => {
      const kEl = card.querySelector('[data-f="k_period"]');
      const dEl = card.querySelector('[data-f="d_period"]');
      const uEl = card.querySelector('[data-f="upper"]');
      const lEl = card.querySelector('[data-f="lower"]');
      const aEl = card.querySelector('[data-f="arrows"]');
      const bounds = lcNormalizeOscBounds(uEl && uEl.value, lEl && lEl.value);
      return {
        id: card.dataset.stochId,
        n: Number(card.dataset.stochN) || 1,
        k_period: clampStochK(kEl && kEl.value),
        d_period: clampStochD(dEl && dEl.value),
        upper: bounds.upper,
        lower: bounds.lower,
        arrows: !!(aEl && aEl.checked),
      };
    });
  }
  function syncStochDraftFromDom() {
    stochDraftSpecs = assignStochNs(readStochSpecsFromDom());
  }
  function stochsForApi() {
    return stochSpecs.map(s => ({
      id: s.id,
      n: s.n,
      k_period: clampStochK(s.k_period),
      d_period: clampStochD(s.d_period),
    }));
  }
  function stochCardHtml(spec) {
    const n = Number(spec.n) > 0 ? Number(spec.n) : 1;
    const k = clampStochK(spec.k_period);
    const d = clampStochD(spec.d_period);
    const bounds = lcNormalizeOscBounds(spec.upper, spec.lower);
    const color = stochColor(n);
    return `<div class="lc-stoch-card lc-ema-card" data-stoch-id="${esc(spec.id)}" data-stoch-n="${esc(String(n))}">
      <div class="lc-ema-card-h">
        <span><span class="lc-ema-swatch" style="background:${esc(color)}"></span>${esc(stochMark(n))} Fast %D (${esc(String(k))},${esc(String(d))})</span>
        <button type="button" data-stoch-remove="${esc(spec.id)}">Remove</button>
      </div>
      <div class="lc-info-row">
        <label>K period
          <input type="number" data-f="k_period" min="${esc(String(stochKMin()))}" max="${esc(String(stochKMax()))}" step="1" value="${esc(String(k))}"></label>
        ${lcInfoTip("Lane %K lookback in bars (HighestHigh / LowestLow).")}
      </div>
      <div class="lc-info-row">
        <label>D smooth
          <input type="number" data-f="d_period" min="${esc(String(stochDMin()))}" max="${esc(String(stochDMax()))}" step="1" value="${esc(String(d))}"></label>
        ${lcInfoTip("Wilder smooth of Fast %K → Fast %D (TOS StochasticFast).")}
      </div>
      <div class="lc-info-row">
        <label>Upper bound
          <input type="number" data-f="upper" min="0" max="100" step="0.1" value="${esc(String(bounds.upper))}"></label>
        ${lcInfoTip("Dashed line on this Stoch pane. With Bound arrows on, a cross down through it draws a red arrow.")}
      </div>
      <div class="lc-info-row">
        <label>Lower bound
          <input type="number" data-f="lower" min="0" max="100" step="0.1" value="${esc(String(bounds.lower))}"></label>
        ${lcInfoTip("Dashed line on this Stoch pane. With Bound arrows on, a cross up through it draws a green arrow.")}
      </div>
      <div class="lc-info-row">
        <label class="checkbox"><input type="checkbox" data-f="arrows"${spec.arrows ? " checked" : ""}> Bound arrows</label>
        ${lcInfoTip("Red arrow down when Fast %D crosses down through the upper bound. Green arrow up when it crosses up through the lower bound. A thin vertical line in the arrow color spans this pane. Apply (↻) commits.")}
      </div>
    </div>`;
  }
  function paintStochCards() {
    const host = $("#lc-stoch-list");
    if (!host) return;
    const specs = stochDraftSpecs.length ? stochDraftSpecs : stochSpecs;
    host.innerHTML = specs.map(stochCardHtml).join("")
      || `<p class="muted">Add Stoch1, Stoch2, …</p>`;
    const box = $("#lc-stoch");
    if (box) box.checked = stochEnabled;
  }
  function stochAddOptionsHtml() {
    const presets = (meta && meta.stoch_presets && meta.stoch_presets.length)
      ? meta.stoch_presets
      : [{ k_period: 9, d_period: 3 }, { k_period: 14, d_period: 3 }, { k_period: 40, d_period: 4 }];
    const defK = stochKDefault(), defD = stochDDefault();
    const opts = presets.map(p => {
      const k = Number(p.k_period), d = Number(p.d_period);
      const sel = k === defK && d === defD ? " selected" : "";
      return `<option value="${esc(String(k))},${esc(String(d))}"${sel}>Fast %D ${esc(String(k))},${esc(String(d))}</option>`;
    }).join("");
    return opts + `<option value="custom">Custom…</option>`;
  }

  function sarDefaults() {
    const d = (meta && meta.defaults && meta.defaults.sar)
      || (meta && meta.sar_defaults)
      || { start: 0.02, increment: 0.02, maximum: 0.2 };
    return {
      start: Number(d.start),
      increment: Number(d.increment),
      maximum: Number(d.maximum),
    };
  }
  function sarBounds() {
    return (meta && meta.sar_bounds) || {
      start: { min: 0.001, max: 1, step: 0.01 },
      increment: { min: 0.001, max: 1, step: 0.01 },
      maximum: { min: 0.01, max: 1, step: 0.01 },
    };
  }
  function clampSarAf(value, field) {
    const b = sarBounds()[field] || { min: 0.001, max: 1 };
    const def = sarDefaults()[field];
    let n = Number(value);
    if (!Number.isFinite(n)) n = def;
    return Math.min(b.max, Math.max(b.min, n));
  }
  function normalizeSarParams(raw) {
    const src = raw && typeof raw === "object" ? raw : {};
    return {
      start: clampSarAf(src.start, "start"),
      increment: clampSarAf(src.increment, "increment"),
      maximum: clampSarAf(src.maximum, "maximum"),
    };
  }
  function loadSarParams() {
    const raw = localStorage.getItem("lc_sar_params");
    if (raw) {
      try { return normalizeSarParams(JSON.parse(raw)); } catch {}
    }
    return normalizeSarParams(sarDefaults());
  }
  function persistSarParams() {
    localStorage.setItem("lc_sar_params", JSON.stringify(sarParams));
    const box = document.querySelector('#lc-ov input[data-key="ps_sar"]');
    localStorage.setItem("lc_sar", box && box.checked ? "1" : "0");
  }
  function readSarParamsFromDom() {
    const card = document.querySelector(".lc-sar-card");
    if (!card) return normalizeSarParams(sarDraftParams);
    const val = (f) => {
      const el = card.querySelector(`[data-f="${f}"]`);
      return el ? el.value : null;
    };
    return normalizeSarParams({
      start: val("start"),
      increment: val("increment"),
      maximum: val("maximum"),
    });
  }
  function syncSarDraftFromDom() {
    sarDraftParams = readSarParamsFromDom();
  }
  function sarForApi() {
    return normalizeSarParams(sarParams);
  }
  function sarCardHtml(spec) {
    const p = normalizeSarParams(spec);
    const b = sarBounds();
    return `<div class="lc-sar-card">
      <div class="lc-ema-card-h">
        <span><span class="lc-ema-swatch" style="background:#ff9f43"></span>Parabolic SAR</span>
      </div>
      <div class="lc-info-row">
        <label>SAR start (AF)
          <input type="number" data-f="start" min="${esc(String(b.start.min))}" max="${esc(String(b.start.max))}" step="${esc(String(b.start.step || 0.01))}" value="${esc(String(p.start))}"></label>
        ${lcInfoTip("Initial acceleration factor. Same knob as the Parabolic SAR external model.")}
      </div>
      <div class="lc-info-row">
        <label>SAR increment
          <input type="number" data-f="increment" min="${esc(String(b.increment.min))}" max="${esc(String(b.increment.max))}" step="${esc(String(b.increment.step || 0.01))}" value="${esc(String(p.increment))}"></label>
        ${lcInfoTip("AF step on a new extreme.")}
      </div>
      <div class="lc-info-row">
        <label>SAR maximum AF
          <input type="number" data-f="maximum" min="${esc(String(b.maximum.min))}" max="${esc(String(b.maximum.max))}" step="${esc(String(b.maximum.step || 0.01))}" value="${esc(String(p.maximum))}"></label>
        ${lcInfoTip("AF cap (Wilder default 0.2).")}
      </div>
    </div>`;
  }
  function paintSarCard() {
    const host = $("#lc-sar-list");
    if (!host) return;
    host.innerHTML = sarCardHtml(sarDraftParams);
    const box = document.querySelector('#lc-ov input[data-key="ps_sar"]');
    if (box && localStorage.getItem("lc_sar") === "1") box.checked = true;
  }

  function paintOverlaysCatalog(m) {
    const host = $("#lc-ov");
    const groups = {};
    for (const o of (m.overlays || [])) {
      if (o.key === "ps_sar") continue; // dedicated SAR block (EMA/VP pattern)
      (groups[o.group] || (groups[o.group] = [])).push(o);
    }
    const stored = new Set((localStorage.getItem("lc_overlays") || "of_cvd").split(",").filter(Boolean));
    if (localStorage.getItem("lc_sar") === "1") stored.add("ps_sar");
    const row = (o) => {
      const params = overlayParamsHtml(o);
      const open = expandedParams.has(o.key);
      const gear = params ? `<button type="button" class="lc-ov-gear" data-panel="lc-params-${esc(o.key)}"
        aria-expanded="${open ? "true" : "false"}" title="Parameters">${open ? "▾" : "▸"}</button>` : `<span></span>`;
      return `<div class="lc-ov-row" data-key="${esc(o.key)}">
        <label class="checkbox">
          <input type="checkbox" data-key="${esc(o.key)}" ${stored.has(o.key) ? "checked" : ""}>
          ${esc(o.label)}</label>
        ${gear}
        ${params ? `<div class="lc-ov-params" id="lc-params-${esc(o.key)}"${open ? "" : " hidden"}>${params}</div>` : ""}
      </div>`;
    };
    const vpOpen = expandedParams.has("volume_profile");
    const vpHtml = `
      <div class="lc-og">
        <div class="lc-og-h">Volume profile</div>
        <div class="lc-ov-row" data-key="volume_profile">
          <label class="checkbox">
            <input type="checkbox" id="lc-vp"> Volume Profile</label>
          <div class="lc-vp-actions">
            <button type="button" class="lc-vp-refresh" id="lc-vp-refresh" title="Apply Volume Profile settings" aria-label="Apply Volume Profile settings">&#x21bb;</button>
            <button type="button" class="lc-ov-gear" data-panel="lc-params-vp"
              aria-expanded="${vpOpen ? "true" : "false"}" title="Parameters">${vpOpen ? "▾" : "▸"}</button>
          </div>
          <div class="lc-ov-params" id="lc-params-vp"${vpOpen ? "" : " hidden"}>
            <div class="lc-info-row">
              <div class="lc-vp-add" style="flex:1;min-width:0">
                <select id="lc-vp-add-kind">
                  <option value="current">Single</option>
                  <option value="recurring">Recurring</option>
                  <option value="level_blend">Level Blend</option>
                </select>
                <button type="button" id="lc-vp-add">Add</button>
              </div>
              ${lcInfoTip("Cards are VP1, VP2, … Uncheck <b>On</b> on a card to hide that profile without removing it (↻ Apply). Chart and Axis labels are per profile and off by default. Axis text is <span class=\"mono\">VPn LEVEL MM/DD price</span> using the profile’s session-end date (Level Blend Hi/Lo use the extreme member’s session). Date blank = last candle’s session. Custom session = From/To clocks. Width is the profile's proportion of its session span. Extend for Single/Recurring is forward only: 1 sess = last bar of this session day, 2 sess = last bar of the next session day, All sess = visible right edge. <b>Level Blend</b> Extend stretches toward the opposite chart edge (Left strip → right; Right strip → left). Overlay sits on its range; Left/Right frames stay pinned. Hist: Tick or Grouped ticks. LVN is separate outlines; LVN min grouped bars filters isolated grouped-bar valleys. Level Blend draws ADR-126 KDE bounds (+ member levels as lines, no histogram) from prior closed cash sessions; Place Left/Right pins the strip to the chart edge so it stays visible while scrolling. Member lines are optional on Level Blend (bounds-only by default).")}
            </div>
            <div id="lc-vp-list" class="lc-vp-list"></div>
          </div>
        </div>
      </div>`;
    const emaOpen = expandedParams.has("ema");
    const emaRow = `
        <div class="lc-ov-row" data-key="ema">
          <label class="checkbox">
            <input type="checkbox" id="lc-ema"> EMA</label>
          <div class="lc-ema-actions">
            <button type="button" class="lc-ema-refresh" id="lc-ema-refresh" title="Apply EMA settings" aria-label="Apply EMA settings">&#x21bb;</button>
            <button type="button" class="lc-ov-gear" data-panel="lc-params-ema"
              aria-expanded="${emaOpen ? "true" : "false"}" title="Parameters">${emaOpen ? "▾" : "▸"}</button>
          </div>
          <div class="lc-ov-params" id="lc-params-ema"${emaOpen ? "" : " hidden"}>
            <div class="lc-info-row">
              <div class="lc-ema-add" style="flex:1;min-width:0">
                <select id="lc-ema-add-kind">${emaAddOptionsHtml()}</select>
                <button type="button" id="lc-ema-add">Add</button>
              </div>
              ${lcInfoTip("Cards are EMA1, EMA2, … Each has its own lookback in bars. The EMA at a bar uses that bar’s close as one of the lookback closes (span-N, current bar included). Pick a preset from the dropdown or Custom, then Add.")}
            </div>
            <div id="lc-ema-list" class="lc-ema-list"></div>
          </div>
        </div>`;
    const stochOpen = expandedParams.has("stoch") || expandedParams.has("stochastic");
    const stochRow = `
        <div class="lc-ov-row" data-key="stoch">
          <label class="checkbox">
            <input type="checkbox" id="lc-stoch"> Stoch Fast %D</label>
          <div class="lc-ema-actions">
            <button type="button" class="lc-ema-refresh" id="lc-stoch-refresh" title="Apply Stochastic settings" aria-label="Apply Stochastic settings">&#x21bb;</button>
            <button type="button" class="lc-ov-gear" data-panel="lc-params-stoch"
              aria-expanded="${stochOpen ? "true" : "false"}" title="Parameters">${stochOpen ? "▾" : "▸"}</button>
          </div>
          <div class="lc-ov-params" id="lc-params-stoch"${stochOpen ? "" : " hidden"}>
            <div class="lc-info-row">
              <div class="lc-ema-add" style="flex:1;min-width:0">
                <select id="lc-stoch-add-kind">${stochAddOptionsHtml()}</select>
                <button type="button" id="lc-stoch-add">Add</button>
              </div>
              ${lcInfoTip("Cards are Stoch1, Stoch2, … Each is a TOS StochasticFast %D (Wilder smooth of Lane %K). Presets 9,3 / 14,3 / 40,4 match the hive. Upper / lower default to 80 / 20. Bound arrows mark crosses of those lines. Sync scores are the other checkboxes in this Stochastic group: <b>all</b> (9+14+40+Full60) and <b>fast+slow</b> (9+Full60).")}
            </div>
            <div id="lc-stoch-list" class="lc-ema-list"></div>
          </div>
        </div>`;
    const sarOpen = expandedParams.has("ps_sar") || expandedParams.has("sar");
    const sarHtml = `
      <div class="lc-og">
        <div class="lc-og-h">Models</div>
        <div class="lc-ov-row" data-key="ps_sar">
          <label class="checkbox">
            <input type="checkbox" data-key="ps_sar" ${stored.has("ps_sar") ? "checked" : ""}> Parabolic SAR</label>
          <div class="lc-sar-actions">
            <button type="button" class="lc-sar-refresh" id="lc-sar-refresh" title="Apply SAR settings" aria-label="Apply SAR settings">&#x21bb;</button>
            <button type="button" class="lc-ov-gear" data-panel="lc-params-sar"
              aria-expanded="${sarOpen ? "true" : "false"}" title="Parameters">${sarOpen ? "▾" : "▸"}</button>
          </div>
          <div class="lc-ov-params" id="lc-params-sar"${sarOpen ? "" : " hidden"}>
            <div class="lc-info-row">
              <span class="muted" style="flex:1;min-width:0">AF knobs match the Parabolic SAR external model.</span>
              ${lcInfoTip("start / increment / maximum feed <span class=\"mono\">sar_on_closed</span> — the same engine as Backtest EXT Parabolic SAR. Refresh applies the card and redraws the stars.")}
            </div>
            <div id="lc-sar-list" class="lc-ema-list"></div>
          </div>
        </div>
        <div class="lc-og-h" style="margin-top:10px">Model design</div>
        <div class="lc-info-row">
          <button type="button" id="lc-md-open" class="lc-md-open-btn" title="Open Model design in a new browser window">Open Model design ⇱</button>
          ${lcInfoTip("Rule tree, blotter, and PnL live in a separate window. Trade markers appear on this chart when you Run.")}
        </div>
      </div>`;
    let stochInGroup = false;
    let priceInGroup = false;
    const groupHtml = Object.entries(groups).map(([g, items]) => {
      let lead = "";
      if (g === "Stochastic") {
        lead = stochRow;
        stochInGroup = true;
      } else if (g === "Price Action") {
        lead = emaRow;
        priceInGroup = true;
      }
      return `
      <div class="lc-og">
        <div class="lc-og-h">${esc(g)}</div>
        ${lead}
        ${items.map(row).join("")}
      </div>`;
    }).join("");
    const stochFallback = stochInGroup ? "" : `
      <div class="lc-og">
        <div class="lc-og-h">Stochastic</div>
        ${stochRow}
      </div>`;
    const priceFallback = priceInGroup ? "" : `
      <div class="lc-og">
        <div class="lc-og-h">Price Action</div>
        ${emaRow}
      </div>`;
    host.innerHTML = groupHtml + vpHtml + priceFallback + stochFallback + sarHtml;
    paintVpCards();
    paintEmaCards();
    paintStochCards();
    paintSarCard();
  }

  function replayCutUnix() {
    if (!replayOn() || replayUnix == null || !Number.isFinite(replayUnix)) return null;
    return replayUnix;
  }
  function pastReplay(real) {
    const cut = replayCutUnix();
    return cut != null && real != null && real > cut;
  }
  function candleData(p) {
    const maps = lastMaps || barMaps(p);
    const out = [];
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      if (pastReplay(maps.real[j])) { out.push({ time: maps.plot[j] }); continue; }
      if (p.o[i] == null || p.c[i] == null) continue;
      out.push({ time: maps.plot[j], open: p.o[i], high: p.h[i], low: p.l[i], close: p.c[i] });
    }
    return out;
  }
  function volData(p) {
    const maps = lastMaps || barMaps(p);
    const out = [];
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      if (pastReplay(maps.real[j])) { out.push({ time: maps.plot[j] }); continue; }
      if (p.c[i] == null) continue;
      out.push({
        time: maps.plot[j], value: p.v[i] || 0,
        color: (p.c[i] || 0) >= (p.o[i] || 0) ? "#7ee0a3" : "#ff6b6b",
      });
    }
    return out;
  }
  function overlaySessionKey(unix, origin) {
    // Returns a string session-bucket key that resets at the origin clock in CT.
    // session = 17:00 CT rollover, bod = 00:00 CT, mkt = 08:30 CT.
    const p = utcMsToCtParts(unix * 1000);
    const t = (p.time || "00:00:00").split(":");
    const h = Number(t[0] || 0), m = Number(t[1] || 0);
    if (origin === "bod") return p.date;
    if (origin === "mkt") {
      const before = (h < 8) || (h === 8 && m < 30);
      return before ? addIsoDate(p.date, -1) : p.date;
    }
    // session (17:00 CT rollover) — same shape as globexSessionEndDate
    return h >= 17 ? addIsoDate(p.date, 1) : p.date;
  }
  function lineData(p, key) {
    const maps = lastMaps || barMaps(p);
    const arr = (p.overlays && p.overlays[key]) || [];
    const isVwap = (key === "vw_vwap" || key === "vw_vwap_dist"
      || key === "vw_bb_upper" || key === "vw_bb_lower");
    const origin = isVwap ? (p.vwap_origin || vwapOrigin()) : null;
    const out = [];
    let prevSess = null, prevPlot = null;
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      if (pastReplay(maps.real[j])) continue;
      if (arr[i] == null || !Number.isFinite(arr[i])) continue;
      if (isVwap) {
        const sess = overlaySessionKey(maps.real[j], origin);
        if (prevSess != null && sess !== prevSess && prevPlot != null) {
          const gap = Math.max(1, Math.floor((maps.plot[j] - prevPlot) / 2));
          out.push({ time: prevPlot + gap });
        }
        prevSess = sess;
        prevPlot = maps.plot[j];
      }
      out.push({ time: maps.plot[j], value: arr[i] });
    }
    return out;
  }
  function deltaHistData(p) {
    const maps = lastMaps || barMaps(p);
    const arr = (p.overlays && p.overlays.of_delta) || [];
    const out = [];
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      if (pastReplay(maps.real[j])) { out.push({ time: maps.plot[j] }); continue; }
      if (arr[i] == null || !Number.isFinite(arr[i])) continue;
      const v = arr[i];
      out.push({
        time: maps.plot[j],
        value: v,
        color: v < 0 ? "#ff6b6b" : "#7ee0a3",
      });
    }
    return out;
  }

  function clearProfilePriceLines() {
    if (!candleSeries) { profilePriceLines.length = 0; return; }
    while (profilePriceLines.length) {
      const line = profilePriceLines.pop();
      try { candleSeries.removePriceLine(line); } catch {}
    }
  }

  function vpAxisActive() {
    return !!(vpOn() && payload && payload.profiles
      && payload.profiles.some(profile => profile && profile.axis_labels));
  }
  function vpAxisAutoscale(base) {
    const def = (typeof base === "function" ? base() : null) || {};
    if (!vpAxisPriceRange) return def;
    const pr = def.priceRange || {};
    const baseMin = Number(pr.minValue);
    const baseMax = Number(pr.maxValue);
    const lo = Math.min(
      vpAxisPriceRange.minValue,
      Number.isFinite(baseMin) ? baseMin : vpAxisPriceRange.minValue,
    );
    const hi = Math.max(
      vpAxisPriceRange.maxValue,
      Number.isFinite(baseMax) ? baseMax : vpAxisPriceRange.maxValue,
    );
    const pad = Math.max(0.25, (hi - lo) * 0.04);
    return Object.assign({}, def, {
      priceRange: { minValue: lo - pad, maxValue: hi + pad },
    });
  }
  function updateVpAxisPriceRange() {
    const values = [];
    if (vpAxisActive()) {
      for (const profile of payload.profiles) {
        if (!profile || !profile.axis_labels) continue;
        if (profile.kind === "level_blend" || profile.level_blend) {
          for (const bound of (profile.bounds || [])) {
            if (Number.isFinite(bound.hi)) values.push(bound.hi);
            if (Number.isFinite(bound.lo)) values.push(bound.lo);
          }
          continue;
        }
        for (const value of [
          profile.high, profile.low, profile.poc, profile.va_high, profile.va_low,
        ]) {
          if (Number.isFinite(value)) values.push(value);
        }
      }
    }
    vpAxisPriceRange = values.length ? {
      minValue: Math.min(...values), maxValue: Math.max(...values),
    } : null;
    if (candleSeries) {
      try { candleSeries.applyOptions({ autoscaleInfoProvider: vpAxisAutoscale }); } catch {}
    }
  }
  function applyVpAxisLayout() {
    const host = document.querySelector(".lc-chart-host");
    const axis = $("#lc-vp-axis");
    const on = vpAxisActive();
    updateVpAxisPriceRange();
    if (host) host.classList.toggle("lc-vp-axis-on", on);
    if (axis) {
      axis.hidden = !on;
      if (!on) axis.innerHTML = "";
    }
    const el = $("#lc-chart");
    if (chart && el) {
      const w = el.clientWidth, h = el.clientHeight;
      if (w > 8 && h > 8) chart.applyOptions({ width: w, height: h });
    }
  }
  function fitVpAxisWidth(items, fs) {
    const host = document.querySelector(".lc-chart-host");
    if (!host) return;
    const longest = items.reduce((n, item) => Math.max(n, String(item.text || "").length), 0);
    // Room for "VP3 POC 08/21 30256.24"-style Axis labels.
    const width = Math.max(108, Math.min(220, Math.ceil(longest * fs * 0.72 + 20)));
    host.style.setProperty("--lc-vp-axis-width", `${width}px`);
  }
  function vpSessionMd(raw) {
    if (raw == null || raw === "") return "";
    const s = String(raw).trim();
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) return `${m[2]}/${m[3]}`;
    const slash = s.match(/^(\d{1,2})\/(\d{1,2})(?:\/\d{2,4})?$/);
    if (slash) {
      return `${String(slash[1]).padStart(2, "0")}/${String(slash[2]).padStart(2, "0")}`;
    }
    return "";
  }
  function vpAxisLabelText(tag, name, price, sessionRaw) {
    const md = vpSessionMd(sessionRaw);
    const p = Number(price).toFixed(2);
    return md ? `${tag} ${name} ${md} ${p}` : `${tag} ${name} ${p}`;
  }
  function boundEdgeSession(bound, edge) {
    if (!bound) return null;
    const direct = edge === "hi" ? bound.hi_session : bound.lo_session;
    if (direct) return direct;
    const members = Array.isArray(bound.members) ? bound.members : [];
    let best = null;
    for (const member of members) {
      const price = Number(member && member.price);
      if (!Number.isFinite(price)) continue;
      if (!best) {
        best = member;
        continue;
      }
      const bestPrice = Number(best.price);
      const sess = String((member && member.session) || "");
      const bestSess = String(best.session || "");
      if (edge === "hi") {
        if (price > bestPrice || (price === bestPrice && sess > bestSess)) best = member;
      } else if (price < bestPrice || (price === bestPrice && sess < bestSess)) {
        best = member;
      }
    }
    return best && best.session ? best.session : null;
  }
  function paintVpAxis(rows) {
    const axis = $("#lc-vp-axis");
    if (!axis) return;
    applyVpAxisLayout();
    if (axis.hidden) return;
    const box = pricePaneBox();
    const fs = Math.max(9, textSize() - 2);
    const lineH = fs + 4;
    const items = (rows || []).filter(r => r && r.y != null && Number.isFinite(r.y))
      .slice()
      .sort((a, b) => a.y - b.y);
    fitVpAxisWidth(items, fs);
    let lastY = -999;
    for (const it of items) {
      if (it.y - lastY < lineH) it.y = lastY + lineH;
      lastY = it.y;
    }
    const yMin = box.top + 14;
    const yMax = box.top + Math.max(lineH, box.height - 4);
    for (const it of items) {
      if (it.y < yMin) it.y = yMin;
      if (it.y > yMax) it.y = yMax;
    }
    axis.innerHTML = `<div class="lc-vp-axis-h">VP</div>` + items.map(it =>
      `<div class="lc-vp-axis-lab" style="top:${it.y}px;color:${esc(it.color)};font-size:${fs}px">${esc(it.text)}</div>`
    ).join("");
  }
  function addProfilePriceLine(price, color, title, dashed) {
    if (!candleSeries || price == null || !Number.isFinite(price)) return;
    if (typeof candleSeries.createPriceLine !== "function") return;
    try {
      const line = candleSeries.createPriceLine({
        price,
        color,
        lineWidth: dashed ? 1 : 2,
        lineStyle: dashed ? 2 : 0,
        lineVisible: false,
        axisLabelVisible: true,
        title,
      });
      profilePriceLines.push(line);
    } catch {}
  }

  function scheduleDrawProfiles() {
    requestAnimationFrame(() => {
      try { drawProfiles(); } catch (err) { console.error(err); }
      requestAnimationFrame(() => {
        try { drawProfiles(); } catch (err) { console.error(err); }
      });
    });
  }
  function vpHistogramWidth(profile, sessionSpan) {
    const configured = Number(profile && profile.hist_width);
    const fraction = Number.isFinite(configured) && configured > 0 ? configured : .25;
    return Math.max(1, Math.abs(sessionSpan) * fraction);
  }

  function drawStar(ctx, cx, cy, r, color) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 === 0 ? r : r * 0.42;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      const x = cx + Math.cos(a) * rad;
      const y = cy + Math.sin(a) * rad;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  function drawSarStars(ctx, plotW) {
    if (!selectedKeys().includes("ps_sar") || !payload || !payload.overlays) return;
    const arr = payload.overlays.ps_sar;
    if (!arr || !arr.length) return;
    let spacing = 6;
    try { spacing = chart.timeScale().options().barSpacing || 6; } catch {}
    const r = Math.max(3.5, Math.min(7.5, spacing * 0.45));
    const maps = lastMaps || barMaps(payload);
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      const v = arr[i];
      if (v == null || !Number.isFinite(v)) continue;
      if (pastReplay(maps.real[j])) continue;
      const x = chart.timeScale().timeToCoordinate(maps.plot[j]);
      const y = candlePriceToY(v);
      if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < -10 || x > plotW + 10) continue;
      const bull = payload.c[i] != null && payload.c[i] > v;
      drawStar(ctx, x, y, r, bull ? "#7ee0a3" : "#ff6b6b");
    }
  }

  function printFillColor(side, opacity) {
    const alpha = Number.isFinite(opacity) ? opacity : .88;
    if (side === "A") return `rgba(255,107,107,${alpha})`;
    if (side === "N") return `rgba(215,227,239,${alpha})`;
    return `rgba(126,224,163,${alpha})`;
  }
  function printSideLabel(side) {
    if (side === "A") return "Sell aggressor (hit bid)";
    if (side === "N") return "Unspecified (N)";
    return "Buy aggressor (lifted ask)";
  }
  function drawLargePrints(ctx, plotW) {
    printHits = [];
    if (!selectedKeys().includes("lt_prints") || !printCache || !printCache.rows) return;
    if (!chart || !candleSeries) return;
    const rows = printCache.rows;
    if (!rows.length) return;
    const leftPad = plotLeftPad();
    ctx.font = chartLabelFont(600);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const showNight = ethDisplay();
    const opacity = printOpacity();
    const extendBars = printExtendBars();
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r == null || r.p == null || r.t == null) continue;
      if (!showNight && !isRthUnix(r.t)) continue;
      if (pastReplay(r.t)) continue;
      const bar = printBarGeometry(r.t);
      const x = bar && bar.x;
      const y = candlePriceToY(r.p);
      if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) continue;
      const cx = x + leftPad;
      const box = pricePaneBox();
      if (cx < -20 || cx > plotW + leftPad + 20 || y < box.top - 20 || y > box.top + box.height + 20) continue;
      const sz = Math.max(1, r.sz || 1);
      const rad = Math.max(1, bar.spacing * .45);
      const fill = printFillColor(r.side, opacity);
      if (extendBars > 0) {
        ctx.beginPath();
        ctx.strokeStyle = fill;
        ctx.lineWidth = Math.max(1, rad * .14);
        ctx.moveTo(cx, y);
        ctx.lineTo(cx + extendBars * bar.spacing, y);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(cx, y, rad, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = "#000000";
      ctx.stroke();
      const rail = Math.max(7, Math.min(14, rad * 0.85));
      ctx.beginPath();
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.4;
      ctx.moveTo(cx - rad - rail, y);
      ctx.lineTo(cx - rad, y);
      ctx.moveTo(cx + rad, y);
      ctx.lineTo(cx + rad + rail, y);
      ctx.stroke();
      if (printDisplayValue()) {
        ctx.fillStyle = "#0b0f14";
        ctx.fillText(String(sz), cx, y);
      }
      printHits.push({ cx, cy: y, rad, row: r });
    }
  }

  // Helpers handed to lc_orders.js (Order map / MBO overlay); it never reaches into this closure.
  function lcOrdersBridge(p) {
    return {
      chart, candleSeries, payload: p || payload, timeToX, candlePriceToY, candleYToPrice,
      pricePaneBox, plotLeftPad, ethDisplay, isRthUnix, pastReplay, selectedKeys, api, still,
      chartLabelSize, lastMaps: () => lastMaps, scheduleDraw: scheduleDrawProfiles,
      host: () => document.querySelector(".lc-chart-host"),
    };
  }
  function plotLeftPad() {
    try {
      const left = chart.priceScale("left");
      if (left && typeof left.width === "function") return left.width() || 0;
    } catch {}
    return 0;
  }
  function candlePriceToY(px) {
    if (px == null || !Number.isFinite(px) || !candleSeries || !candleSeries.priceToCoordinate) return null;
    const y = candleSeries.priceToCoordinate(px);
    if (y == null || !Number.isFinite(y)) return null;
    return y + pricePaneBox().top;
  }
  function candleYToPrice(y) {
    if (y == null || !Number.isFinite(y) || !candleSeries || !candleSeries.coordinateToPrice) return null;
    return candleSeries.coordinateToPrice(y - pricePaneBox().top);
  }
  function barDurationSec() {
    const tf = (payload && payload.timeframe) || ($("#lc-tf") && $("#lc-tf").value) || "1min";
    return TF_SEC[tf] || 60;
  }
  function unixToBarFrac(unix) {
    const maps = lastMaps;
    if (!maps || !maps.real || !maps.real.length) return null;
    const r = maps.real;
    const n = r.length;
    const dur = barDurationSec();
    if (!(dur > 0)) return null;
    let j = 0;
    if (unix > r[0]) {
      let lo = 0, hi = n - 1;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (r[m] <= unix) { j = m; lo = m + 1; }
        else hi = m - 1;
      }
    }
    const frac = Math.max(0, Math.min(1, (unix - r[j]) / dur));
    return { j, frac };
  }
  function timeToX(unix) {
    if (unix == null || !Number.isFinite(unix) || !chart || !chart.timeScale) return null;
    const scale = chart.timeScale();
    const maps = lastMaps;
    if (!maps || !maps.real || !maps.real.length) return null;
    const loc = unixToBarFrac(unix);
    if (!loc) return null;
    const xOf = t => {
      const x = scale.timeToCoordinate(t);
      return (x == null || !Number.isFinite(x)) ? null : x;
    };
    const xCenter = xOf(maps.plot[loc.j]);
    if (xCenter == null) return null;
    let spacing = (scale.options && scale.options().barSpacing) || 6;
    if (loc.j + 1 < maps.plot.length) {
      const xNext = xOf(maps.plot[loc.j + 1]);
      if (xNext != null) spacing = xNext - xCenter;
    } else if (loc.j > 0) {
      const xPrev = xOf(maps.plot[loc.j - 1]);
      if (xPrev != null) spacing = xCenter - xPrev;
    }
    return xCenter + (loc.frac - 0.5) * spacing;
  }
  function printBarGeometry(unix) {
    if (unix == null || !Number.isFinite(unix) || !chart || !chart.timeScale) return null;
    const maps = lastMaps;
    const loc = unixToBarFrac(unix);
    if (!maps || !loc || maps.plot[loc.j] == null) return null;
    const scale = chart.timeScale();
    const x = scale.timeToCoordinate(maps.plot[loc.j]);
    if (x == null || !Number.isFinite(x)) return null;
    let spacing = (scale.options && scale.options().barSpacing) || 6;
    if (loc.j + 1 < maps.plot.length) {
      const next = scale.timeToCoordinate(maps.plot[loc.j + 1]);
      if (next != null && Number.isFinite(next)) spacing = next - x;
    } else if (loc.j > 0) {
      const prev = scale.timeToCoordinate(maps.plot[loc.j - 1]);
      if (prev != null && Number.isFinite(prev)) spacing = x - prev;
    }
    return { x, spacing: Math.max(1, Math.abs(spacing)) };
  }
  function drawSessionDayLines(ctx, plotW, hostH) {
    if (!payload || !chart) return;
    const maps = lastMaps || barMaps(payload);
    if (!maps.real || !maps.real.length) return;
    const leftPad = plotLeftPad();
    const y0 = 0;
    const y1 = Math.max(8, hostH - 28);
    ctx.save();
    ctx.strokeStyle = "#1f2a36";
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    let prevKey = null;
    for (let j = 0; j < maps.real.length; j++) {
      if (maps.plot[j] == null) continue;
      const key = globexSessionEndDate(maps.real[j]);
      if (prevKey != null && key !== prevKey) {
        const x = chart.timeScale().timeToCoordinate(maps.plot[j]);
        if (x != null && Number.isFinite(x)) {
          let spacing = (chart.timeScale().options().barSpacing) || 6;
          if (j + 1 < maps.plot.length) {
            const next = chart.timeScale().timeToCoordinate(maps.plot[j + 1]);
            if (next != null && Number.isFinite(next)) spacing = next - x;
          } else if (j > 0) {
            const prior = chart.timeScale().timeToCoordinate(maps.plot[j - 1]);
            if (prior != null && Number.isFinite(prior)) spacing = x - prior;
          }
          const px = x - Math.max(0, spacing) * 0.5 + leftPad;
          if (px >= leftPad - 1 && px <= leftPad + plotW + 1) {
            ctx.beginPath();
            ctx.moveTo(Math.round(px) + 0.5, y0);
            ctx.lineTo(Math.round(px) + 0.5, y1);
            ctx.stroke();
          }
        }
      }
      prevKey = key;
    }
    ctx.restore();
  }
  function drawEthBands(ctx, plotW, hostH) {
    if (!ethShade() || !ethDisplay() || !payload || !chart) return;
    const tf = payload.timeframe || ($("#lc-tf") && $("#lc-tf").value);
    if (tf === "1day") return;
    const maps = lastMaps || barMaps(payload);
    if (!maps.real || !maps.real.length) return;
    const t0 = maps.real[0], t1 = maps.real[maps.real.length - 1];
    const windows = globexOvernightWindows(t0, t1);
    if (!windows.length) return;
    const leftPad = plotLeftPad();
    let plotRight = leftPad + plotW;
    const cut = replayCutUnix();
    if (cut != null) {
      const cx = timeToX(cut);
      if (cx != null && Number.isFinite(cx)) plotRight = Math.min(plotRight, cx + leftPad);
    }
    ctx.fillStyle = "rgba(18, 32, 52, 0.38)";
    const sess = sessionPaneBox();
    const y0 = sess.top;
    const yH = sess.height || hostH;
    for (const pair of windows) {
      const x0 = timeToX(pair[0]);
      const x1 = timeToX(pair[1]);
      if (x0 == null && x1 == null) continue;
      const a = (x0 == null ? leftPad : x0 + leftPad);
      const b = (x1 == null ? plotRight : x1 + leftPad);
      const left = Math.max(leftPad, Math.min(a, b));
      const right = Math.min(plotRight, Math.max(a, b));
      if (right - left < 1) continue;
      ctx.fillRect(left, y0, right - left, yH);
    }
  }

  function groupTicks(ticks, tickSize, groupSize) {
    const rows = Array.isArray(ticks) ? ticks : [];
    const ts = Number(tickSize) || .25;
    const size = [1, 2, 4, 8, 16, 32, 64].includes(Number(groupSize)) ? Number(groupSize) : 4;
    const grouped = new Map();
    for (const b of rows) {
      const p = Number(b && b.p);
      if (!Number.isFinite(p)) continue;
      const group = Math.floor(Math.round(p / ts) / size);
      const rec = grouped.get(group) || { lo: group * size * ts, v: 0, b: 0, s: 0 };
      rec.v += b.v || 0; rec.b += b.b || 0; rec.s += b.s || 0;
      grouped.set(group, rec);
    }
    return [...grouped.values()].sort((a, b) => a.lo - b.lo).map(rec => ({
      ...rec, hi: rec.lo + size * ts, p: rec.lo + size * ts / 2,
    }));
  }
  function seriesLastNumber(series) {
    if (!series || typeof series.data !== "function") return null;
    let data = null;
    try { data = series.data(); } catch { return null; }
    if (!data || !data.length) return null;
    for (let i = data.length - 1; i >= 0; i--) {
      const d = data[i];
      if (!d) continue;
      if (d.close != null && Number.isFinite(d.close)) return d;
      if (d.value != null && Number.isFinite(d.value)) return d.value;
    }
    return null;
  }
  function seriesPaneY(series, value, paneIndex) {
    if (!series || value == null || typeof series.priceToCoordinate !== "function") return null;
    const y = series.priceToCoordinate(value);
    if (y == null || !Number.isFinite(y)) return null;
    return y + paneBox(paneIndex).top;
  }
  function seriesTagPane(key) {
    if (!key || key === "ps_sar" || key === "lt_prints" || key === "lt_mbo_orders" || String(key).startsWith("vw_bb_")) return null;
    if (String(key).startsWith("ema_") || key === "vw_vwap") return 0;
    const spec = overlayMetaBy()[key] || {};
    if (spec.pane === "price") return 0;
    if (spec.pane === "canvas" || spec.needs === "prints") return null;
    return oscPaneIndex(key);
  }
  function tagColor(key) {
    if (String(key).startsWith("ema_")) return emaColor(Number(String(key).slice(4)));
    if (String(key).startsWith("stoch_")) return stochColor(Number(String(key).slice(6)));
    return OV_COLOR[key] || "#d7e3ef";
  }
  function rightScaleWidth(paneIndex) {
    try {
      const scale = chart.priceScale("right", paneIndex);
      if (scale && typeof scale.width === "function") {
        const w = scale.width();
        if (w > 8) return w;
      }
    } catch {}
    return 64;
  }
  function paintScaleTags() {
    const layer = $("#lc-scale-tags");
    if (!layer || !chart || !candleSeries) return;
    const fs = Math.max(9, textSize() - 1);
    const lineH = fs;
    const byPane = new Map();
    const add = (pane, item) => {
      if (pane == null || pane < 0 || item.y == null || !Number.isFinite(item.y)) return;
      if (!byPane.has(pane)) byPane.set(pane, []);
      byPane.get(pane).push(item);
    };
    const lastPx = seriesLastNumber(candleSeries);
    if (lastPx && typeof lastPx === "object") {
      const up = lastPx.open == null || lastPx.close >= lastPx.open;
      add(0, {
        key: "px", text: fmtNum(lastPx.close, 2),
        color: up ? "#7ee0a3" : "#ff6b6b",
        y: seriesPaneY(candleSeries, lastPx.close, 0), priority: 0,
      });
    }
    if (volSeries) {
      const volV = lcFiniteSeriesValue(seriesLastNumber(volSeries));
      if (volV != null && Number.isFinite(volV)) {
        add(1, {
          key: "vol", text: fmtInt(volV), color: "#c5d0dc",
          y: seriesPaneY(volSeries, volV, 1), priority: 1,
        });
      }
    }
    for (const k of Object.keys(overlaySeries)) {
      const pane = seriesTagPane(k);
      const series = overlaySeries[k];
      const v = lcFiniteSeriesValue(seriesLastNumber(series));
      if (pane == null || v == null || !Number.isFinite(v)) continue;
      add(pane, {
        key: k, text: fmtOverlay(k, v),
        color: k === "of_delta" ? (v < 0 ? "#ff6b6b" : "#7ee0a3") : tagColor(k),
        y: seriesPaneY(series, v, pane), priority: 1,
      });
    }
    if (cvdCandleSeries && !overlaySeries.of_cvd) {
      const pane = oscPaneIndex("of_cvd");
      const raw = seriesLastNumber(cvdCandleSeries);
      if (pane != null && raw && typeof raw === "object" && Number.isFinite(raw.close)) {
        add(pane, {
          key: "of_cvd", text: fmtInt(raw.close), color: "#f0c674",
          y: seriesPaneY(cvdCandleSeries, raw.close, pane), priority: 1,
        });
      }
    }
    const html = [];
    byPane.forEach((items, pane) => {
      const box = paneBox(pane);
      const packed = lcPackScaleTags(items, box.top, box.top + box.height, lineH);
      const scaleW = rightScaleWidth(pane);
      const width = layer.clientWidth || 0;
      for (const tag of packed) {
        html.push(
          `<div class="lc-scale-tag" style="top:${tag.top.toFixed(1)}px;height:${lineH}px;line-height:${lineH}px;font-size:${fs}px;color:${tag.color};border-color:${tag.color}">${esc(tag.text)}</div>`
        );
        if (tag.nudged && width > 0) {
          const x = Math.max(0, width - scaleW);
          html.push(
            `<div class="lc-scale-tick" style="top:${tag.y.toFixed(1)}px;left:${x.toFixed(1)}px;background:${tag.color}"></div>`
          );
        }
      }
    });
    layer.innerHTML = html.join("");
  }
  function drawCrossArrow(ctx, x, y, dir, color, pane) {
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 1;
    ctx.moveTo(x + 0.5, pane.top);
    ctx.lineTo(x + 0.5, pane.top + pane.height);
    ctx.stroke();
    ctx.globalAlpha = 1;
    const hw = 4.5;
    const hh = 4;
    ctx.beginPath();
    if (dir === "down") {
      ctx.moveTo(x, y + hh);
      ctx.lineTo(x - hw, y - hh);
      ctx.lineTo(x + hw, y - hh);
    } else {
      ctx.moveTo(x, y - hh);
      ctx.lineTo(x - hw, y + hh);
      ctx.lineTo(x + hw, y + hh);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = "rgba(8,12,16,.9)";
    ctx.stroke();
    ctx.restore();
  }
  function drawBoundArrows(ctx, plotW) {
    if (!payload || !chart || !ctx) return;
    const jobs = [];
    if (stochOn()) {
      for (const spec of stochSpecs) {
        if (!spec || !spec.arrows) continue;
        const key = stochKey(spec.n);
        const series = overlaySeries[key];
        const pane = oscPaneIndex(key);
        if (!series || pane == null) continue;
        const b = stochBoundsForKey(key);
        jobs.push({
          series, pane,
          marks: lcSeriesBoundCrosses(lineData(payload, key), b.upper, b.lower),
        });
      }
    }
    if (selectedKeys().includes("mf_mfi") && storedMfiBounds().arrows && overlaySeries.mf_mfi) {
      const pane = oscPaneIndex("mf_mfi");
      if (pane != null) {
        const b = storedMfiBounds();
        jobs.push({
          series: overlaySeries.mf_mfi, pane,
          marks: lcSeriesBoundCrosses(lineData(payload, "mf_mfi"), b.upper, b.lower),
        });
      }
    }
    if (vwapBandsOn() && localStorage.getItem("lc_vwap_bb_arrows") === "1" && candleSeries) {
      jobs.push({
        series: candleSeries, pane: 0,
        marks: lcPriceBandCrosses(
          candleData(payload),
          lineData(payload, "vw_bb_upper"),
          lineData(payload, "vw_bb_lower"),
        ),
      });
    }
    for (const job of jobs) {
      const pane = paneBox(job.pane);
      const scale = chart.timeScale();
      for (const m of job.marks) {
        const x = scale.timeToCoordinate(m.time);
        if (x == null || !Number.isFinite(x) || x < -12 || x > plotW + 12) continue;
        let yLocal = null;
        try { yLocal = job.series.priceToCoordinate(m.value); } catch { yLocal = null; }
        if (yLocal == null || !Number.isFinite(yLocal)) continue;
        const y = yLocal + pane.top;
        if (y < pane.top - 2 || y > pane.top + pane.height + 2) continue;
        const color = m.dir === "down" ? "#ff6b6b" : "#7ee0a3";
        drawCrossArrow(ctx, x, y, m.dir, color, pane);
      }
    }
  }
  function drawProfiles() {
    const canvas = $("#lc-vp-canvas");
    if (!canvas || !chart || !candleSeries) return;
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    if (!host) return;
    const dpr = window.devicePixelRatio || 1;
    const w = host.clientWidth, h = host.clientHeight;
    if (w < 8 || h < 8) return;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    lastLevelBlendHits = [];
    const plotW = (chart.timeScale && chart.timeScale().width)
      ? chart.timeScale().width() : w;
    if (!plotW) { paintScaleTags(); return; }
    applyVpAxisLayout();
    if (payload && (!lastMaps || !lastMaps.real || !lastMaps.real.length)) {
      lastMaps = barMaps(payload);
    }
    drawSessionDayLines(ctx, plotW, h);
    const palettes = [
      { buy: "rgba(126,224,163,.55)", sell: "rgba(255,107,107,.50)",
        va: "rgba(230,192,123,.18)", box: "rgba(230,192,123,.95)",
        poc: "#e6c07b", vaLine: "#56b6c2", hi: "#e8eef4", hiBand: "rgba(232,238,244,.13)",
        lvn: "rgba(245,158,11,.70)", lvnExt: "rgba(245,158,11,.22)" },
      { buy: "rgba(167,139,250,.50)", sell: "rgba(244,114,182,.45)",
        va: "rgba(196,181,253,.16)", box: "rgba(196,181,253,.95)",
        poc: "#c4b5fd", vaLine: "#a78bfa", hi: "#d4c4f7", hiBand: "rgba(212,196,247,.13)",
        lvn: "rgba(251,191,36,.70)", lvnExt: "rgba(251,191,36,.22)" },
      { buy: "rgba(52,211,153,.50)", sell: "rgba(251,113,133,.45)",
        va: "rgba(126,224,163,.14)", box: "rgba(126,224,163,.95)",
        poc: "#7ee0a3", vaLine: "#34d399", hi: "#bbf7d0", hiBand: "rgba(187,247,208,.13)",
        lvn: "rgba(251,146,60,.70)", lvnExt: "rgba(251,146,60,.22)" },
      { buy: "rgba(34,211,238,.45)", sell: "rgba(251,146,60,.45)",
        va: "rgba(86,182,194,.16)", box: "rgba(86,182,194,.95)",
        poc: "#22d3ee", vaLine: "#56b6c2", hi: "#a5f3fc", hiBand: "rgba(165,243,252,.13)",
        lvn: "rgba(250,204,21,.70)", lvnExt: "rgba(250,204,21,.20)" },
    ];
    const yOf = px => candlePriceToY(px);
    const tOf = unix => timeToX(unix);
    function sessionDayRightX(sessionDate) {
      const maps = lastMaps || barMaps(payload);
      if (!sessionDate || !maps.real || !maps.real.length || !chart) return null;
      let lastJ = -1;
      for (let j = 0; j < maps.real.length; j++) {
        if (globexSessionEndDate(maps.real[j]) === sessionDate) lastJ = j;
      }
      if (lastJ < 0) {
        const lastDay = globexSessionEndDate(maps.real[maps.real.length - 1]);
        return sessionDate > lastDay ? plotW : null;
      }
      const scale = chart.timeScale();
      const x = scale.timeToCoordinate(maps.plot[lastJ]);
      if (x == null || !Number.isFinite(x)) return null;
      let spacing = (scale.options && scale.options().barSpacing) || 6;
      if (lastJ + 1 < maps.plot.length) {
        const xNext = scale.timeToCoordinate(maps.plot[lastJ + 1]);
        if (xNext != null && Number.isFinite(xNext)) spacing = xNext - x;
      } else if (lastJ > 0) {
        const xPrev = scale.timeToCoordinate(maps.plot[lastJ - 1]);
        if (xPrev != null && Number.isFinite(xPrev)) spacing = x - xPrev;
      }
      return x + Math.max(0, spacing) * 0.5;
    }
    function sessionDayLeftX(sessionDate) {
      const maps = lastMaps || barMaps(payload);
      if (!sessionDate || !maps.real || !maps.real.length || !chart) return null;
      let firstJ = -1;
      for (let j = 0; j < maps.real.length; j++) {
        if (globexSessionEndDate(maps.real[j]) === sessionDate) {
          firstJ = j;
          break;
        }
      }
      if (firstJ < 0) {
        const firstDay = globexSessionEndDate(maps.real[0]);
        return sessionDate < firstDay ? 0 : null;
      }
      const scale = chart.timeScale();
      const x = scale.timeToCoordinate(maps.plot[firstJ]);
      if (x == null || !Number.isFinite(x)) return null;
      let spacing = (scale.options && scale.options().barSpacing) || 6;
      if (firstJ > 0) {
        const xPrev = scale.timeToCoordinate(maps.plot[firstJ - 1]);
        if (xPrev != null && Number.isFinite(xPrev)) spacing = x - xPrev;
      } else if (firstJ + 1 < maps.plot.length) {
        const xNext = scale.timeToCoordinate(maps.plot[firstJ + 1]);
        if (xNext != null && Number.isFinite(xNext)) spacing = xNext - x;
      }
      return x - Math.max(0, spacing) * 0.5;
    }
    function vpForwardX(profile, span, histRight) {
      if (!(span > 0)) return histRight;
      if (span >= 3) return plotW;
      const day0 = vpSessionDay(profile);
      if (!day0) return histRight;
      const day = addSessionEndDates(day0, span - 1);
      const x = sessionDayRightX(day);
      if (x == null || !Number.isFinite(x)) return histRight;
      return Math.max(histRight, Math.min(plotW, x));
    }
    /** Level Blend: extend toward the opposite chart edge from the strip. */
    function levelBlendExtendRange(profile, span, stripLeft, stripRight, fromRight) {
      if (!(span > 0)) return { bandLeft: stripLeft, bandRight: stripRight };
      if (!fromRight) {
        const xTo = vpForwardX(profile, span, stripRight);
        return { bandLeft: stripLeft, bandRight: Math.max(stripRight, xTo) };
      }
      let bandLeft;
      if (span >= 3) {
        bandLeft = 0;
      } else {
        const day0 = vpSessionDay(profile);
        if (!day0) {
          bandLeft = 0;
        } else {
          const day = addSessionEndDates(day0, -(span - 1));
          const x = sessionDayLeftX(day);
          bandLeft = (x == null || !Number.isFinite(x)) ? 0 : Math.min(stripLeft, Math.max(0, x));
        }
      }
      return { bandLeft, bandRight: stripRight };
    }
    drawEthBands(ctx, plotW, h);
    const priceBox = pricePaneBox();
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, priceBox.top, w, Math.max(1, priceBox.height));
    ctx.clip();
    let mapped = 0;
    let list = (vpOn() && payload && Array.isArray(payload.profiles) && payload.profiles.length)
      ? payload.profiles
      : [];
    const replayCut = replayCutUnix();
    if (replayCut != null) {
      list = list.filter(pr => pr && (pr.start_unix == null || pr.start_unix <= replayCut));
    }
    const maxBar = Math.max(48, Math.min(160, plotW * (list.length > 2 ? 0.16 : 0.28)));
    let leftStack = 0, rightStack = 0;
    const laid = [];
    function profileHasGeometry(profile) {
      if (!profile) return false;
      if (profile.kind === "level_blend" || profile.level_blend) {
        return Array.isArray(profile.bounds) && profile.bounds.length > 0;
      }
      return Array.isArray(profile.bins) && profile.bins.length > 0;
    }
    list.forEach((profile, idx) => {
      if (!profileHasGeometry(profile)) return;
      const pal = palettes[idx % palettes.length];
      const placement = profile.placement || "overlay";
      const side = profile.overlay_side || "left";
      let x0, x1, fromRight;
      const histWidth = vpHistogramWidth(profile, maxBar);
      if (placement === "left_frame") {
        x0 = 2 + leftStack;
        x1 = x0 + histWidth;
        leftStack += histWidth + 6;
        fromRight = false;
      } else if (placement === "right_frame") {
        x1 = plotW - 2 - rightStack;
        x0 = x1 - histWidth;
        rightStack += histWidth + 6;
        fromRight = true;
      } else {
        const t0 = tOf(profile.start_unix);
        const t1 = tOf(profile.end_unix);
        if (t0 == null && t1 == null) return;
        const a = t0 == null ? (t1 - maxBar) : t0;
        const b = t1 == null ? (t0 + maxBar) : t1;
        const span = Math.abs(b - a);
        const bw = vpHistogramWidth(profile, span);
        if (side === "right") {
          x1 = Math.max(a, b);
          x0 = x1 - bw;
          fromRight = true;
        } else {
          x0 = Math.min(a, b);
          x1 = x0 + bw;
          fromRight = false;
        }
        if (x1 < 0 || x0 > plotW) return;
      }
      laid.push({ profile, pal, x0, x1, fromRight, idx });
    });
    function memberColor(kind, pal) {
      if (kind === "poc") return pal.poc;
      if (kind === "vah" || kind === "val") return pal.vaLine;
      return pal.hi;
    }
    function histBox(item) {
      const profile = item.profile;
      const pal = item.pal;
      const x0 = item.x0, x1 = item.x1;
      const fromRight = item.fromRight;
      const width = Math.max(1, Math.abs(x1 - x0));
      const left = Math.min(x0, x1);
      if (profile.kind === "level_blend" || profile.level_blend) {
        const bounds = Array.isArray(profile.bounds) ? profile.bounds : [];
        const span = parseVpExtend(
          profile.extend != null ? profile.extend : ((profile.extend_lines || profile.extend_va) ? 1 : 0),
        );
        const stripLeft = left;
        const stripRight = left + width;
        const { bandLeft, bandRight } = levelBlendExtendRange(
          profile, span, stripLeft, stripRight, fromRight,
        );
        const bandWidth = Math.max(1, bandRight - bandLeft);
        const drawMembers = profile.show_members === true;
        for (const bound of bounds) {
          const yHi = yOf(bound.hi);
          const yLo = yOf(bound.lo);
          if (yHi == null && yLo == null) continue;
          mapped++;
          const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
          const hgt = Math.max(2, Math.abs((yLo == null ? yHi : yLo) - (yHi == null ? yLo : yHi)));
          ctx.fillStyle = pal.va;
          ctx.fillRect(bandLeft, top, bandWidth, hgt);
          ctx.strokeStyle = pal.box;
          ctx.lineWidth = 1.4;
          ctx.setLineDash([]);
          ctx.strokeRect(bandLeft + 0.5, top + 0.5, Math.max(1, bandWidth - 1), hgt);
          const loPx = Number(bound.lo);
          const hiPx = Number(bound.hi);
          if (Number.isFinite(loPx) && Number.isFinite(hiPx)) {
            lastLevelBlendHits.push({
              profileId: profile.spec_id || profile.id,
              n: profile.n != null ? profile.n : profile.label,
              lo: Math.min(loPx, hiPx),
              hi: Math.max(loPx, hiPx),
              strength: Number(bound.strength),
              hi_session: bound.hi_session,
              lo_session: bound.lo_session,
              members: bound.members,
              bandLeft,
              bandRight,
              color: pal.box,
            });
          }
          if (!drawMembers) continue;
          const members = Array.isArray(bound.members) ? bound.members : [];
          for (const m of members) {
            const price = Number(m && m.price);
            if (!Number.isFinite(price)) continue;
            const y = yOf(price);
            if (y == null) continue;
            ctx.beginPath();
            ctx.strokeStyle = memberColor(m.kind, pal);
            ctx.lineWidth = m.kind === "poc" ? 1.8 : 1.1;
            ctx.setLineDash(m.kind === "poc" ? [] : (m.kind === "hi" || m.kind === "lo" ? [2, 3] : [5, 4]));
            ctx.moveTo(bandLeft, y);
            ctx.lineTo(bandRight, y);
            ctx.stroke();
            ctx.setLineDash([]);
            if (profile.chart_labels) {
              const tag = vpMark(profile.n != null ? profile.n : profile.label);
              const label = `${tag} ${(m.kind || "lvl").toUpperCase()} ${price.toFixed(2)}`;
              ctx.save();
              ctx.fillStyle = memberColor(m.kind, pal);
              ctx.font = chartLabelFont();
              ctx.textBaseline = "bottom";
              const tw = ctx.measureText(label).width;
              let lx = bandLeft + 4;
              if (lx + tw > plotW - 4) lx = Math.max(4, bandRight - tw - 4);
              ctx.fillText(label, lx, y - 4);
              ctx.restore();
            }
          }
        }
        item.ph = profile.high;
        item.pl = profile.low;
        item.levelBlendDrawn = true;
        return;
      }
      const wantBinned = (profile.hist_style || "volume") === "binned"
        || (profile.hist_style || "") === "lvn_nodes";
      const coarse = (Array.isArray(profile.coarse_bins) && profile.coarse_bins.length)
        ? profile.coarse_bins
        : (wantBinned ? groupTicks(profile.bins, profile.tick_size, profile.group_ticks) : []);
      const binned = wantBinned && coarse.length > 0;
      const rightDisplay = profile.right_display || (fromRight ? "none" : "total_split");
      const leftDisplay = profile.left_display || (fromRight ? "total_split" : "none");
      const dualDisplay = leftDisplay !== "none" && rightDisplay !== "none";
      const midpoint = left + width / 2;
      const displays = [
        { mode: leftDisplay, opacity: Number(profile.left_opacity), edge: dualDisplay ? midpoint : left + width, direction: -1 },
        { mode: rightDisplay, opacity: Number(profile.right_opacity), edge: dualDisplay ? midpoint : left, direction: 1 },
      ].filter(display => display.mode !== "none");
      function magnitude(rec, mode) {
        if (mode === "buy") return rec.b || 0;
        if (mode === "sell") return rec.s || 0;
        if (mode === "delta") return Math.abs((rec.b || 0) - (rec.s || 0));
        return rec.v || 0;
      }
      function paintDisplay(top, hgt, rec, maxV, display) {
        const maxWidth = width / displays.length;
        const bw = Math.max(1, maxWidth * magnitude(rec, display.mode) / maxV);
        const start = display.direction > 0 ? display.edge : display.edge - bw;
        ctx.save();
        ctx.globalAlpha = Number.isFinite(display.opacity)
          ? Math.min(1, Math.max(0, display.opacity / 100)) : 1;
        if (display.mode === "total_split") {
          const buyW = rec.v ? bw * ((rec.b || 0) / rec.v) : 0;
          const sellW = bw - buyW;
          ctx.fillStyle = display.direction > 0 ? pal.sell : pal.buy;
          ctx.fillRect(start, top, display.direction > 0 ? sellW : buyW, hgt);
          ctx.fillStyle = display.direction > 0 ? pal.buy : pal.sell;
          ctx.fillRect(start + (display.direction > 0 ? sellW : buyW), top,
            display.direction > 0 ? buyW : sellW, hgt);
          ctx.restore();
          return;
        }
        if (display.mode === "total_delta_split") {
          ctx.fillStyle = pal.box;
          ctx.fillRect(start, top, bw, hgt);
          const deltaW = rec.v ? bw * Math.abs((rec.b || 0) - (rec.s || 0)) / rec.v : 0;
          ctx.fillStyle = (rec.b || 0) >= (rec.s || 0) ? pal.buy : pal.sell;
          const deltaStart = display.direction > 0 ? start : start + bw - deltaW;
          ctx.fillRect(deltaStart, top, deltaW, hgt);
          ctx.restore();
          return;
        }
        if (display.mode === "buy") ctx.fillStyle = pal.buy;
        else if (display.mode === "sell") ctx.fillStyle = pal.sell;
        else if (display.mode === "delta") ctx.fillStyle = (rec.b || 0) >= (rec.s || 0) ? pal.buy : pal.sell;
        else ctx.fillStyle = pal.box;
        ctx.fillRect(start, top, bw, hgt);
        ctx.restore();
      }
      function paintDisplays(top, hgt, rec, rows) {
        for (const display of displays) {
          let maxV = 1;
          for (const row of rows) maxV = Math.max(maxV, magnitude(row, display.mode));
          paintDisplay(top, hgt, rec, maxV, display);
        }
      }
      if (binned) {
        for (const n of coarse) {
          const yHi = yOf(n.hi);
          const yLo = yOf(n.lo);
          if (yHi == null && yLo == null) continue;
          mapped++;
          const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
          const hgt = Math.max(2, Math.abs((yLo == null ? yHi : yLo) - (yHi == null ? yLo : yHi)));
          paintDisplays(top, hgt, n, coarse);
        }
      } else {
        const rows = new Map();
        for (const b of profile.bins) {
          const y = yOf(b.p);
          if (y == null) continue;
          mapped++;
          const yi = Math.round(y);
          const rec = rows.get(yi) || { v: 0, b: 0, s: 0 };
          rec.v += b.v || 0;
          rec.b += b.b || 0;
          rec.s += b.s || 0;
          rows.set(yi, rec);
        }
        const values = [...rows.values()];
        for (const [yi, rec] of rows) paintDisplays(yi, 1, rec, values);
      }
      if (binned && profile.hvn_method === "adaptive_kde") {
        const structureBars = new Map();
        const groupWidth = Math.max(
          Number(profile.tick_size || 0.25) * Number(profile.group_ticks || 1),
          Number(profile.tick_size || 0.25),
        );
        const nearestBar = (price) => {
          if (price == null || !Number.isFinite(Number(price))) return null;
          let best = null, bestD = Infinity;
          for (const bar of coarse) {
            const d = Math.abs(Number(bar.p) - Number(price));
            if (d < bestD) { best = bar; bestD = d; }
          }
          return best && bestD <= groupWidth * 0.51 ? best : null;
        };
        for (const peak of (profile.hvn_peaks || [])) {
          const bar = nearestBar(peak.price);
          if (bar) structureBars.set(`${bar.lo}:${bar.hi}`, { bar, color: "#050505" });
        }
        for (const valley of (profile.lvn_valleys || [])) {
          const bar = nearestBar(valley.price);
          if (bar && !structureBars.has(`${bar.lo}:${bar.hi}`)) {
            structureBars.set(`${bar.lo}:${bar.hi}`, { bar, color: "#ffffff" });
          }
        }
        for (const { bar, color } of structureBars.values()) {
          const yHi = yOf(bar.hi);
          const yLo = yOf(bar.lo);
          if (yHi == null && yLo == null) continue;
          const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
          const hgt = Math.max(3, Math.abs((yLo == null ? yHi : yLo) - (yHi == null ? yLo : yHi)));
          ctx.save();
          ctx.globalAlpha = 1;
          ctx.fillStyle = color;
          ctx.fillRect(left, top, width, hgt);
          ctx.restore();
        }
      }
      if (profile.show_lvn) {
        const zones = ((binned ? profile.coarse_nodes : profile.nodes) || []).filter(n => n && n.lvn);
        const half = binned ? 0 : ((profile.tick_size || 0.25) / 2);
        ctx.save();
        ctx.strokeStyle = pal.lvn;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([]);
        for (const n of zones) {
          const yHi = yOf((n.hi != null ? n.hi : n.lo) + half);
          const yLo = yOf((n.lo != null ? n.lo : n.hi) - half);
          if (yHi == null && yLo == null) continue;
          const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
          const hgt = Math.max(2, Math.abs((yLo == null ? yHi : yLo) - (yHi == null ? yLo : yHi)));
          ctx.strokeRect(left + 0.5, top + 0.5, Math.max(1, width - 1), hgt);
        }
        ctx.restore();
      }
      if (profile.hvn_method === "adaptive_kde" && Array.isArray(profile.hvn_zones)) {
        ctx.save();
        for (const zone of profile.hvn_zones) {
          const yHi = yOf(zone.hi);
          const yLo = yOf(zone.lo);
          if (yHi == null && yLo == null) continue;
          const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
          const hgt = Math.max(3, Math.abs((yLo == null ? yHi : yLo) - (yHi == null ? yLo : yHi)));
          const color = zone.va_relation === "outside" ? pal.lvn : pal.vaLine;
          ctx.fillStyle = zone.va_relation === "outside"
            ? "rgba(245, 158, 11, .12)" : "rgba(86, 182, 194, .12)";
          ctx.fillRect(left, top, width, hgt);
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.8;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(left + 0.5, top + 0.5, Math.max(1, width - 1), hgt);
        }
        ctx.restore();
      }
      if (profile.hvn_method === "adaptive_kde" && profile.chart_labels) {
        const marks = [];
        for (const peak of (profile.hvn_peaks || [])) {
          if (peak && Number.isFinite(Number(peak.price))) {
            marks.push({ ...peak, kind: "HVN", color: pal.vaLine });
          }
        }
        for (const valley of (profile.lvn_valleys || [])) {
          if (valley && Number.isFinite(Number(valley.price))) {
            marks.push({ ...valley, kind: "LVN", color: pal.lvn });
          }
        }
        marks.sort((a, b) => Number(a.price) - Number(b.price));
        const placed = [];
        ctx.save();
        const labelFs = chartLabelSize();
        const lineH = labelFs + 4;
        ctx.font = chartLabelFont();
        ctx.textBaseline = "bottom";
        for (const mark of marks) {
          const y = yOf(Number(mark.price));
          if (y == null) continue;
          let ly = y;
          for (const prior of placed) {
            if (Math.abs(ly - prior) < lineH) ly = prior + lineH;
          }
          if (ly < lineH || ly > priceBox.height - 2) continue;
          const text = `${mark.label || mark.kind} ${Number(mark.price).toFixed(2)}`;
          const tw = ctx.measureText(text).width;
          const lx = Math.min(Math.max(3, left + 4), Math.max(3, plotW - tw - 4));
          ctx.fillStyle = "rgba(11, 15, 20, .84)";
          ctx.fillRect(lx - 2, ly - lineH, tw + 6, lineH);
          ctx.fillStyle = mark.color;
          ctx.fillText(text, lx, ly - 1);
          placed.push(ly);
        }
        ctx.restore();
      }
      const ph = profile.high != null ? profile.high
        : Math.max(...profile.bins.map(b => b.p));
      const pl = profile.low != null ? profile.low
        : Math.min(...profile.bins.map(b => b.p));
      const yHi = yOf(ph), yLo = yOf(pl);
      const vaH = yOf(profile.va_high), vaL = yOf(profile.va_low);
      if (vaH != null && vaL != null) {
        const top = Math.min(vaH, vaL);
        const band = Math.max(2, Math.abs(vaL - vaH));
        ctx.fillStyle = pal.va;
        ctx.fillRect(left, top, width, band);
        ctx.strokeStyle = pal.vaLine;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([]);
        ctx.strokeRect(left + 0.5, top + 0.5, width - 1, band);
      }
      if (yHi != null && yLo != null) {
        const top = Math.min(yHi, yLo);
        const boxH = Math.max(2, Math.abs(yLo - yHi));
        ctx.strokeStyle = pal.box;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([]);
        ctx.strokeRect(left + 0.5, top + 0.5, width - 1, boxH);
      }
      item.ph = ph; item.pl = pl;
    }
    laid.forEach(histBox);
    const ordered = laid.slice().sort((a, b) =>
      (a.profile.start_unix || 0) - (b.profile.start_unix || 0));
    function hseg(item, price, color, dash, width, title, xFrom, xTo) {
      if (price == null || !Number.isFinite(price)) return;
      const y = yOf(price);
      if (y == null) return;
      if (xTo < xFrom) return;
      const xa = xFrom, xb = xTo;
      if (xb - xa < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.moveTo(xa, y);
      ctx.lineTo(xb, y);
      ctx.stroke();
      ctx.setLineDash([]);
      if (item.profile.chart_labels) {
        ctx.save();
        ctx.fillStyle = color;
        ctx.font = chartLabelFont();
        ctx.textBaseline = "bottom";
        const label = `${title} ${price.toFixed(2)}`;
        const tw = ctx.measureText(label).width;
        let lx = xa + 4;
        if (lx + tw > plotW - 4) lx = Math.max(4, xb - tw - 4);
        ctx.fillText(label, lx, y - 4);
        ctx.restore();
      }
    }
    function variableSegs(item, key, color, dash, width, title) {
      const points = (item.profile.variable_bounds || []).filter(p =>
        p && Number.isFinite(Number(p.t)) && Number.isFinite(Number(p[key]))
      );
      if (!points.length) return;
      for (let i = 0; i < points.length; i++) {
        const x0 = i === 0
          ? (tOf(Number(item.profile.start_unix)) ?? tOf(Number(points[i].t)))
          : tOf(Number(points[i].t));
        if (x0 == null) continue;
        const nextT = i + 1 < points.length
          ? Number(points[i + 1].t) : Number(item.profile.end_unix);
        const x1 = tOf(nextT);
        if (x1 == null || x1 <= x0) continue;
        hseg(item, Number(points[i][key]), color, dash, width, title, x0, x1);
      }
    }
    ordered.forEach((item, i) => {
      const p = item.profile;
      const pal = item.pal;
      if (item.levelBlendDrawn || p.kind === "level_blend" || p.level_blend) {
        // Bound washes + member lines already painted in histBox.
        return;
      }
      const histLeft = Math.min(item.x0, item.x1);
      const histRight = Math.max(item.x0, item.x1);
      const span = parseVpExtend(
        p.extend != null ? p.extend : ((p.extend_lines || p.extend_va) ? 1 : 0),
      );
      const lvnSpan = p.show_lvn && span > 0 ? span : 0;
      const drawSpan = Math.max(span, lvnSpan);
      const xFrom = histLeft;
      const xTo = vpForwardX(p, drawSpan, histRight);
      const lineFrom = xFrom;
      const lineTo = span > 0 ? xTo : histRight;
      const tag = vpMark(p.n != null ? p.n : p.label) + " ";
      if (span > 0 && xTo > histRight) {
        const vaH = yOf(p.va_high), vaL = yOf(p.va_low);
        if (vaH != null && vaL != null) {
          const top = Math.min(vaH, vaL);
          const band = Math.max(2, Math.abs(vaL - vaH));
          ctx.fillStyle = pal.va;
          ctx.fillRect(histRight, top, Math.max(2, xTo - histRight), band);
        }
      }
      if (lvnSpan > 0 && xTo > histRight) {
        const binned = (p.hist_style || "volume") === "binned";
        const zones = ((binned ? p.coarse_nodes : p.nodes) || []).filter(n => n && n.lvn);
        const half = binned ? 0 : ((p.tick_size || 0.25) / 2);
        const xa = histRight, xb = xTo;
        if (xb - xa >= 2) {
          ctx.save();
          ctx.strokeStyle = pal.lvn;
          ctx.lineWidth = 1.2;
          ctx.setLineDash([5, 4]);
          for (const n of zones) {
            const yHi = yOf((n.hi != null ? n.hi : n.lo) + half);
            const yLo = yOf((n.lo != null ? n.lo : n.hi) - half);
            if (yHi == null && yLo == null) continue;
            const top = Math.min(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
            const bot = Math.max(yHi == null ? yLo : yHi, yLo == null ? yHi : yLo);
            ctx.beginPath();
            ctx.moveTo(xa, top);
            ctx.lineTo(xb, top);
            ctx.moveTo(xa, bot);
            ctx.lineTo(xb, bot);
            ctx.stroke();
          }
          ctx.restore();
        }
      }
      for (const [primary, secondary] of [[item.ph, p.hi_body], [item.pl, p.lo_body]]) {
        const yPrimary = yOf(primary), ySecondary = yOf(secondary);
        if (yPrimary == null || ySecondary == null || yPrimary === ySecondary) continue;
        ctx.fillStyle = pal.hiBand;
        ctx.fillRect(lineFrom, Math.min(yPrimary, ySecondary), lineTo - lineFrom,
          Math.abs(ySecondary - yPrimary));
      }
      if (p.plot_variable_bounds && (p.variable_bounds || []).length) {
        variableSegs(item, "high", "#ff4d4d", [2, 3], 2.2, `${tag}Hi`);
        variableSegs(item, "low", "#ff4d4d", [2, 3], 2.2, `${tag}Lo`);
        variableSegs(item, "poc", "#ffd166", [], 2.8, `${tag}POC`);
        variableSegs(item, "va_high", "#ff4d4d", [6, 4], 2.2, `${tag}VAH`);
        variableSegs(item, "va_low", "#ff4d4d", [6, 4], 2.2, `${tag}VAL`);
      } else {
        hseg(item, item.ph, pal.hi, [2, 3], 1.2, `${tag}Hi`, lineFrom, lineTo);
        hseg(item, item.pl, pal.hi, [2, 3], 1.2, `${tag}Lo`, lineFrom, lineTo);
        hseg(item, p.poc, pal.poc, [], 2, `${tag}POC`, lineFrom, lineTo);
        hseg(item, p.va_high, pal.vaLine, [5, 4], 1.2, `${tag}VAH`, lineFrom, lineTo);
        hseg(item, p.va_low, pal.vaLine, [5, 4], 1.2, `${tag}VAL`, lineFrom, lineTo);
      }
    });
    clearProfilePriceLines();
    const axisRows = [];
    ordered.forEach((item) => {
      const p = item.profile;
      const pal = item.pal;
      const tag = vpMark(p.n != null ? p.n : p.label);
      if (!p.axis_labels) return;
      if (p.kind === "level_blend" || p.level_blend) {
        const bounds = Array.isArray(p.bounds) ? p.bounds : [];
        for (let bi = 0; bi < bounds.length; bi++) {
          const bound = bounds[bi];
          for (const [price, name, color, edge] of [
            [bound.hi, `B${bi + 1} Hi`, pal.box, "hi"],
            [bound.lo, `B${bi + 1} Lo`, pal.box, "lo"],
          ]) {
            if (price == null || !Number.isFinite(price)) continue;
            const y = yOf(price);
            if (y == null || !Number.isFinite(y)) continue;
            axisRows.push({
              y,
              text: vpAxisLabelText(tag, name, price, boundEdgeSession(bound, edge)),
              color,
            });
          }
        }
        return;
      }
      const labs = [
        { price: item.ph != null ? item.ph : p.high, name: "Hi", color: pal.hi },
        { price: item.pl != null ? item.pl : p.low, name: "Lo", color: pal.hi },
        { price: p.poc, name: "POC", color: pal.poc },
        { price: p.va_high, name: "VAH", color: pal.vaLine },
        { price: p.va_low, name: "VAL", color: pal.vaLine },
      ];
      labs.forEach(lab => {
        if (lab.price == null || !Number.isFinite(lab.price)) return;
        const y = yOf(lab.price);
        const text = vpAxisLabelText(tag, lab.name, lab.price, p.session_date);
        if (y != null && Number.isFinite(y)) {
          axisRows.push({ y, text, color: lab.color });
        }
      });
    });
    paintVpAxis(axisRows);
    drawSarStars(ctx, plotW);
    drawLargePrints(ctx, plotW);
    if (window.LCOrders) LCOrders.draw(ctx, plotW, lcOrdersBridge());
    drawBoundArrows(ctx, plotW);
    ctx.restore();
    paintScaleTags();
    const want = list.some(profileHasGeometry);
    if (want && mapped === 0 && drawTries < 12) {
      drawTries += 1;
      requestAnimationFrame(drawProfiles);
      return;
    }
    drawTries = 0;
  }

  function monthDay(s) {
    const mon = _MON[Number(s.slice(5, 7)) - 1] || s.slice(5, 7);
    return `${mon} ${Number(s.slice(8, 10))}`;
  }
  function fmtTick(t, tickMarkType) {
    const real = plotToReal(t);
    const iso = new Date(real * 1000).toISOString();
    const s = toCT(iso);
    if (!s) return "";
    const tf = ($("#lc-tf") && $("#lc-tf").value) || (payload && payload.timeframe);
    const type = tickMarkType == null ? 3 : tickMarkType;
    if (tf === "1day") {
      if (type <= 0) return s.slice(0, 4);
      if (type === 1) return _MON[Number(s.slice(5, 7)) - 1] || s.slice(5, 7);
      return monthDay(s);
    }
    if (lastMaps && lastMaps.densify) {
      const arr = lastMaps.plot;
      let lo = 0, hi = arr.length - 1, ans = 0;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (arr[m] <= t) { ans = m; lo = m + 1; }
        else hi = m - 1;
      }
      if (ans === 0) return monthDay(s);
      const prev = toCT(new Date(lastMaps.real[ans - 1] * 1000).toISOString());
      if (prev && prev.slice(0, 10) !== s.slice(0, 10)) return monthDay(s);
      if (tf === "5s" || tf === "15s") return s.slice(11, 19);
      return s.slice(11, 16);
    }
    if (type <= 0) return s.slice(0, 4);
    if (type === 1) return _MON[Number(s.slice(5, 7)) - 1] || s.slice(5, 7);
    if (type === 2) return monthDay(s);
    if (type === 4 || tf === "5s" || tf === "15s") return s.slice(11, 19);
    return s.slice(11, 16);
  }
  function fmtCrosshairTime(t) {
    const real = plotToReal(t);
    const s = toCT(new Date(real * 1000).toISOString());
    if (!s) return "";
    const tf = ($("#lc-tf") && $("#lc-tf").value) || (payload && payload.timeframe);
    if (tf === "1day") return s.slice(0, 10);
    if (tf === "5s" || tf === "15s") return s;
    return s.slice(0, 16);
  }

  const OV_SHORT = {
    of_delta: "Delta Volume", of_cvd: "CVD", of_ofi: "OFI", of_ofi_cum: "OFI_CUM",
    lt_prints: "Prints",
    mf_mfi: "MFI", ad_line: "A/D",
    vw_vwap: "VWAP", vw_vwap_dist: "dVWAP",
    vw_bb_upper: "VWAP+2σ", vw_bb_lower: "VWAP−2σ",
    cs_clv: "CLV",
    so_fast_d_9_3: "Stoch", so_sync_score_all: "Sync all",
    so_sync_score_9_60: "Sync F+S", ps_sar: "SAR",
  };

  function fmtOverlay(key, v) {
    if (v == null || !Number.isFinite(v)) return "—";
    if (key === "of_cvd" || key === "of_delta" || key === "of_ofi_cum" || key === "of_ofi" || key === "ad_line") return fmtInt(v);
    if (key === "mf_mfi") return fmtNum(v, 1);
    if (key === "vw_vwap_dist" || key === "cs_clv") return fmtNum(v, 3);
    if (key === "so_fast_d_9_3" || (key && String(key).startsWith("stoch_"))) return fmtNum(v, 1);
    if (key === "so_sync_score_all" || key === "so_sync_score_9_60") return fmtNum(v, 3);
    return fmtNum(v, 2);
  }

  function barIndexAt(unix) {
    if (unix == null) return -1;
    if (lastMaps && lastMaps.plot && lastMaps.plot.length) {
      const arr = lastMaps.plot;
      let lo = 0, hi = arr.length - 1, ans = 0;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        if (arr[m] <= unix) { ans = m; lo = m + 1; }
        else hi = m - 1;
      }
      return lastMaps.idx[ans];
    }
    const t = payload && payload.t;
    if (!t || !t.length) return -1;
    let lo = 0, hi = t.length - 1, ans = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (t[m] <= unix) { ans = m; lo = m + 1; }
      else hi = m - 1;
    }
    return ans;
  }

  function seriesVal(param, series, fallback) {
    if (param && param.seriesData && series && typeof param.seriesData.get === "function") {
      const d = param.seriesData.get(series);
      if (d) {
        if (d.value != null && Number.isFinite(d.value)) return d.value;
        if (d.close != null && Number.isFinite(d.close)) return d;
      }
    }
    return fallback;
  }

  function vbpAt(profile, px) {
    if (!profile || !profile.bins || !profile.bins.length || px == null || !Number.isFinite(px)) return null;
    const tick = profile.tick_size || 0.25;
    let best = null, bestD = Infinity;
    for (const b of profile.bins) {
      const d = Math.abs((b.p || 0) - px);
      if (d < bestD) { bestD = d; best = b; }
    }
    return (best && bestD <= tick * 1.5) ? best : null;
  }

  function chip(label, value, color, cls) {
    const style = color ? ` style="color:${color}"` : "";
    return `<span class="lc-chip${cls ? " " + cls : ""}"><span class="k">${esc(label)}</span><span class="v"${style}>${esc(value)}</span></span>`;
  }

  function hoverRow(label, value, color) {
    const style = color ? ` style="color:${color}"` : "";
    return `<div class="lc-hr"><span class="k">${esc(label)}</span><span class="v"${style}>${esc(value)}</span></div>`;
  }

  function cursorSnapshot(param) {
    if (!payload || !payload.n) return null;
    const unix = param && param.time != null ? param.time : payload.t[payload.n - 1];
    const i = barIndexAt(unix);
    if (i < 0) return null;
    const candleRaw = seriesVal(param, candleSeries, null);
    const o = (candleRaw && candleRaw.open != null) ? candleRaw.open : payload.o[i];
    const h = (candleRaw && candleRaw.high != null) ? candleRaw.high : payload.h[i];
    const l = (candleRaw && candleRaw.low != null) ? candleRaw.low : payload.l[i];
    const c = (candleRaw && candleRaw.close != null) ? candleRaw.close : payload.c[i];
    const volRaw = seriesVal(param, volSeries, payload.v[i]);
    const vol = (volRaw && volRaw.value != null) ? volRaw.value : volRaw;
    let px = null;
    if (param && param.point && candleSeries && typeof candleSeries.coordinateToPrice === "function") {
      const box = pricePaneBox();
      const y = param.point.y;
      const inPrice = (param.paneIndex != null)
        ? param.paneIndex === 0
        : (y >= box.top - 2 && y <= box.top + box.height + 2);
      if (inPrice) px = candleYToPrice(y);
    }
    const overlays = {};
    for (const k of selectedKeys()) {
      if (k === "lt_prints" || k === "lt_mbo_orders") continue;
      const fb = payload.overlays && payload.overlays[k] ? payload.overlays[k][i] : null;
      const raw = seriesVal(param, overlaySeries[k], fb);
      overlays[k] = (raw && raw.value != null) ? raw.value : raw;
    }
    if (emaOn() && payload.emas) {
      for (const spec of payload.emas) {
        if (!spec || !spec.key) continue;
        const fb = payload.overlays && payload.overlays[spec.key]
          ? payload.overlays[spec.key][i] : null;
        const raw = seriesVal(param, overlaySeries[spec.key], fb);
        overlays[spec.key] = (raw && raw.value != null) ? raw.value : raw;
      }
    }
    if (stochOn() && payload.stochs) {
      for (const spec of payload.stochs) {
        if (!spec || !spec.key) continue;
        const fb = payload.overlays && payload.overlays[spec.key]
          ? payload.overlays[spec.key][i] : null;
        const raw = seriesVal(param, overlaySeries[spec.key], fb);
        overlays[spec.key] = (raw && raw.value != null) ? raw.value : raw;
      }
    }
    if (vwapBandsOn()) {
      for (const k of ["vw_bb_upper", "vw_bb_lower"]) {
        const fb = payload.overlays && payload.overlays[k] ? payload.overlays[k][i] : null;
        const raw = seriesVal(param, overlaySeries[k], fb);
        overlays[k] = (raw && raw.value != null) ? raw.value : raw;
      }
    }
    return { unix: payload.t[i], i, o, h, l, c, vol, px, overlays };
  }

  function renderReadout(snap, hoverEl) {
    const bar = $("#lc-readout");
    if (!snap) {
      if (bar) bar.innerHTML = "";
      if (hoverEl && hoverCorner === "follow" && !hoverDragging) hoverEl.hidden = true;
      return;
    }
    const iso = new Date(snap.unix * 1000).toISOString();
    const tlab = toCT(iso) || fmtCrosshairTime(snap.unix);
    const up = (snap.c != null && snap.o != null) ? snap.c >= snap.o : true;
    const pxColor = up ? "#7ee0a3" : "#ff6b6b";
    const chips = [
      chip("T", tlab, "#d7e3ef"),
      chip("O", snap.o == null ? "—" : fmtNum(snap.o, 2), "#e8eef4"),
      chip("H", snap.h == null ? "—" : fmtNum(snap.h, 2), "#7ee0a3"),
      chip("L", snap.l == null ? "—" : fmtNum(snap.l, 2), "#ff6b6b"),
      chip("C", snap.c == null ? "—" : fmtNum(snap.c, 2), pxColor, up ? "up" : "dn"),
      chip("Vol", snap.vol == null ? "—" : fmtInt(snap.vol), "#c5d0dc"),
    ];
    if (snap.px != null && Number.isFinite(snap.px)) {
      chips.push(chip("Px", fmtNum(snap.px, 2), "#f0c674"));
    }
    const hover = [ `<div class="lc-ht">${esc(tlab)}</div>` ];
    const pxTxt = (v) => v == null ? "—" : fmtNum(v, 2);
    hover.push(hoverRow("H", pxTxt(snap.h), "#7ee0a3"));
    hover.push(hoverRow("O", pxTxt(snap.o), "#e8eef4"));
    hover.push(hoverRow("C", pxTxt(snap.c), pxColor));
    hover.push(hoverRow("L", pxTxt(snap.l), "#ff6b6b"));
    hover.push(hoverRow("Vol", snap.vol == null ? "—" : fmtInt(snap.vol), "#c5d0dc"));
    if (snap.px != null && Number.isFinite(snap.px)) {
      hover.push(hoverRow("Price", fmtNum(snap.px, 2), "#f0c674"));
    }
    for (const k of selectedKeys()) {
      if (k === "lt_prints" || k === "lt_mbo_orders" || k === "of_cvd") continue;
      const label = (k === "of_ofi_cum" || k === "ad_line")
        ? `${k === "of_ofi_cum" ? "OFI_CUM" : "A/D"} (${
            k === "of_ofi_cum" ? ofiOrigin() : adOrigin()
          })`
        : (OV_SHORT[k] || k);
      const color = OV_COLOR[k] || "#e8eef4";
      const val = fmtOverlay(k, snap.overlays[k]);
      chips.push(chip(label, val, color));
      hover.push(hoverRow(label, val, color));
    }
    if (emaOn() && payload && payload.emas) {
      for (const spec of payload.emas) {
        if (!spec || !spec.key) continue;
        const label = spec.label || `${emaMark(spec.n)} (${spec.lookback})`;
        const color = emaColor(spec.n);
        const val = fmtOverlay(spec.key, snap.overlays[spec.key]);
        chips.push(chip(label, val, color));
        hover.push(hoverRow(label, val, color));
      }
    }
    if (stochOn() && payload && payload.stochs) {
      for (const spec of payload.stochs) {
        if (!spec || !spec.key) continue;
        const label = spec.label || `${stochMark(spec.n)} (${spec.k_period},${spec.d_period})`;
        const color = stochColor(spec.n);
        const val = fmtOverlay(spec.key, snap.overlays[spec.key]);
        chips.push(chip(label, val, color));
        hover.push(hoverRow(label, val, color));
      }
    }
    if (vwapBandsOn()) {
      for (const k of ["vw_bb_upper", "vw_bb_lower"]) {
        const label = OV_SHORT[k] || k;
        const color = OV_COLOR[k] || "#d4b36a";
        const val = fmtOverlay(k, snap.overlays[k]);
        chips.push(chip(label, val, color));
        hover.push(hoverRow(label, val, color));
      }
    }
    if (selectedKeys().includes("of_cvd")) {
      const origin = (payload && payload.cvd_origin) || cvdOrigin();
      const cvdLab = `CVD (${origin})`;
      const style = cvdStyle();
      const pack = payload && payload.cvd_ohlc;
      const i = snap.i;
      const close = (pack && pack.c && pack.c[i] != null) ? pack.c[i] : snap.overlays.of_cvd;
      const wantBody = style === "candle" || style === "both";
      const open = (wantBody && pack && pack.o && pack.o[i] != null) ? pack.o[i] : null;
      if (wantBody && close != null && Number.isFinite(close)) {
        const dlt = (open != null && Number.isFinite(open)) ? close - open : null;
        const dltStr = dlt == null || !Number.isFinite(dlt) ? "—"
          : (dlt > 0 ? "+" + fmtInt(dlt) : fmtInt(dlt));
        const dltColor = dlt == null ? "#f0c674" : (dlt >= 0 ? "#7ee0a3" : "#ff6b6b");
        chips.push(chip("CVD O", open == null ? "—" : fmtInt(open), "#f0c674"));
        chips.push(chip("CVD C", fmtInt(close), "#f0c674"));
        chips.push(chip("CVD Δ", dltStr, dltColor));
        hover.push(hoverRow("CVD O", open == null ? "—" : fmtInt(open), "#f0c674"));
        hover.push(hoverRow("CVD C", fmtInt(close), "#f0c674"));
        hover.push(hoverRow("CVD Δ", dltStr, dltColor));
      } else {
        chips.push(chip(cvdLab, fmtOverlay("of_cvd", close), "#f0c674"));
        hover.push(hoverRow(cvdLab, fmtOverlay("of_cvd", close), "#f0c674"));
      }
    }
    const profiles = (vpOn() && payload && payload.profiles) || [];
    profilesCoveringUnix(profiles, snap.unix).slice(0, 8).forEach((s) => {
      if (s && (s.kind === "level_blend" || s.level_blend)) return;
      const tag = vpMark(s.n || s.label);
      const lvl = (v) => (v == null || !Number.isFinite(Number(v))) ? "—" : fmtNum(v, 2);
      const levels = [
        [`${tag} VHi`, lvl(s.high), "#e8eef4"],
        [`${tag} VLo`, lvl(s.low), "#e8eef4"],
        [`${tag} VAH`, lvl(s.va_high), "#56b6c2"],
        [`${tag} VAL`, lvl(s.va_low), "#56b6c2"],
        [`${tag} POC`, lvl(s.poc), "#e6c07b"],
      ];
      chips.push(chip(`${tag} POC`, lvl(s.poc), "#e6c07b"));
      levels.forEach(([lab, val, color]) => hover.push(hoverRow(lab, val, color)));
      const bin = vbpAt(s, snap.px != null ? snap.px : snap.c);
      if (bin) hover.push(hoverRow(`${tag} @ px`, fmtInt(bin.v), "#e6c07b"));
      const px = snap.px != null ? snap.px : snap.c;
      if (s.show_lvn && px != null) {
        const binned = (s.hist_style || "volume") === "binned";
        const zones = (binned ? s.coarse_nodes : s.nodes) || [];
        const hit = zones.find(n => n && n.lvn && n.lo <= px && px <= n.hi);
        if (hit) {
          hover.push(hoverRow(`${tag} LVN`, `${fmtNum(hit.lo, 2)}–${fmtNum(hit.hi, 2)}`, "#f59e0b"));
        }
      }
    });
    const blendHit = snap.blendCluster;
    if (blendHit) {
      const tag = vpMark(blendHit.n);
      const hiLab = vpAxisLabelText(tag, "HI", blendHit.hi, boundEdgeSession(blendHit, "hi"));
      const loLab = vpAxisLabelText(tag, "LO", blendHit.lo, boundEdgeSession(blendHit, "lo"));
      const strength = Number.isFinite(blendHit.strength) ? fmtNum(blendHit.strength, 2) : "—";
      const rangePts = (Number.isFinite(blendHit.hi) && Number.isFinite(blendHit.lo))
        ? fmtNum(blendHit.hi - blendHit.lo, 2)
        : "—";
      const color = blendHit.color || "#e6c07b";
      chips.push(chip(`${tag} str`, strength, color));
      chips.push(chip(`${tag} rng`, rangePts, color));
      hover.push(hoverRow("Upper", hiLab, color));
      hover.push(hoverRow("Lower", loLab, color));
      hover.push(hoverRow("Strength", strength, color));
      hover.push(hoverRow("Bound range", rangePts, "#c5d0dc"));
    }
    const pr = snap.print;
    if (pr) {
      const sideClr = pr.side === "A" ? "#ff6b6b" : (pr.side === "N" ? "#d7e3ef" : "#7ee0a3");
      const tlabP = pr.ts ? fmtTsUs(pr.ts) : (pr.t != null ? fmtTsUs(new Date(pr.t * 1000).toISOString()) : "—");
      hover.push(hoverRow("Print price", pr.p == null ? "—" : fmtNum(pr.p, 2), "#f0c674"));
      hover.push(hoverRow("Print qty", pr.sz == null ? "—" : fmtInt(pr.sz), "#e8eef4"));
      hover.push(hoverRow("Print time", tlabP + " CT", "#d7e3ef"));
      hover.push(hoverRow("Print side", printSideLabel(pr.side), sideClr));
    }
    if (bar) bar.innerHTML = chips.join("");
    const body = hoverEl && (hoverEl.querySelector("#lc-hover-body") || hoverEl);
    if (body) body.innerHTML = hover.join("");
  }

  function hoverCornerPos(host, el) {
    const pad = 8;
    const w = el.offsetWidth || 180, h = el.offsetHeight || 120;
    const W = host.clientWidth, H = host.clientHeight;
    return {
      tl: { left: pad, top: pad },
      tr: { left: Math.max(pad, W - w - pad), top: pad },
      bl: { left: pad, top: Math.max(pad, H - h - pad) },
      br: { left: Math.max(pad, W - w - pad), top: Math.max(pad, H - h - pad) },
    };
  }

  function nearestHoverCorner(left, top, host, el) {
    const pos = hoverCornerPos(host, el);
    let best = "tl", bestD = Infinity;
    for (const [k, p] of Object.entries(pos)) {
      const d = (p.left - left) * (p.left - left) + (p.top - top) * (p.top - top);
      if (d < bestD) { bestD = d; best = k; }
    }
    return best;
  }

  function setHoverGuides(on, active) {
    const host = document.querySelector(".lc-chart-host");
    if (!host) return;
    let g = host.querySelector(".lc-hover-guides");
    if (!on) {
      if (g) g.hidden = true;
      return;
    }
    if (!g) {
      g = document.createElement("div");
      g.className = "lc-hover-guides";
      g.innerHTML = ["tl", "tr", "bl", "br"].map(c =>
        `<div class="lc-hover-guide" data-c="${c}"></div>`).join("");
      host.appendChild(g);
    }
    g.hidden = false;
    g.querySelectorAll(".lc-hover-guide").forEach(el => {
      el.classList.toggle("on", el.dataset.c === active);
    });
  }

  function applyHoverSize() {
    const el = $("#lc-hover");
    if (!el) return;
    if (!hoverSize) {
      el.classList.remove("is-sized");
      el.style.width = "";
      el.style.height = "";
      return;
    }
    el.classList.add("is-sized");
    el.style.width = hoverSize.w + "px";
    el.style.height = hoverSize.h + "px";
  }
  function syncHoverFollowButton() {
    const follow = hoverCorner === "follow";
    const followBtn = $("#lc-hover-follow");
    const pinBtn = $("#lc-hover-pin");
    if (followBtn) followBtn.hidden = follow;
    if (pinBtn) pinBtn.hidden = !follow;
  }
  function clampHoverPos(left, top, host, el) {
    const w = (el && el.offsetWidth) || 180;
    const h = (el && el.offsetHeight) || 100;
    const maxL = Math.max(4, host.clientWidth - Math.max(40, w) - 4);
    const maxT = Math.max(4, host.clientHeight - Math.max(24, h) - 4);
    return {
      left: Math.max(4, Math.min(left, maxL)),
      top: Math.max(4, Math.min(top, maxT)),
    };
  }
  function hoverBoxOn() {
    const box = $("#lc-hover-show");
    if (box) return !!box.checked;
    return localStorage.getItem("lc_hover_visible") !== "0";
  }
  function pinHoverAt(clientX, clientY) {
    if (!hoverBoxOn()) return;
    const el = $("#lc-hover");
    const host = document.querySelector(".lc-chart-host");
    if (!el || !host) return;
    const hr = host.getBoundingClientRect();
    let left = parseFloat(el.style.left);
    let top = parseFloat(el.style.top);
    const placed = !el.hidden && Number.isFinite(left) && Number.isFinite(top);
    if (!placed) {
      const w = el.offsetWidth || 180;
      const h = el.offsetHeight || 100;
      left = clientX - hr.left + 14;
      top = clientY - hr.top + 14;
      if (left + w > host.clientWidth - 8) left = clientX - hr.left - w - 14;
      if (top + h > host.clientHeight - 8) top = clientY - hr.top - h - 14;
    }
    const pos = clampHoverPos(left, top, host, el);
    hoverPos = pos;
    hoverCorner = "pin";
    hoverDragging = false;
    localStorage.setItem("lc_hover_corner", "pin");
    localStorage.setItem("lc_hover_pos", JSON.stringify(pos));
    el.classList.remove("lc-hover-dragging");
    el.classList.add("lc-hover-docked");
    el.style.left = pos.left + "px";
    el.style.top = pos.top + "px";
    el.hidden = false;
    setHoverGuides(false);
    syncHoverFollowButton();
  }
  function setHoverFollow() {
    hoverCorner = "follow";
    hoverDragging = false;
    localStorage.setItem("lc_hover_corner", "follow");
    const el = $("#lc-hover");
    if (el) el.classList.remove("lc-hover-docked", "lc-hover-dragging");
    setHoverGuides(false);
    syncHoverFollowButton();
  }
  function applyHoverCorner() {
    if (hoverResizing) return;
    const el = $("#lc-hover");
    const host = document.querySelector(".lc-chart-host");
    syncHoverFollowButton();
    if (!el || !host || hoverCorner === "follow") {
      if (el) el.classList.remove("lc-hover-docked");
      return;
    }
    if (hoverCorner === "pin") {
      const raw = hoverPos || { left: parseFloat(el.style.left) || 8, top: parseFloat(el.style.top) || 8 };
      const pos = clampHoverPos(raw.left, raw.top, host, el);
      hoverPos = pos;
      el.hidden = false;
      el.classList.add("lc-hover-docked");
      el.style.left = pos.left + "px";
      el.style.top = pos.top + "px";
      return;
    }
    const pos = hoverCornerPos(host, el)[hoverCorner];
    if (!pos) return;
    el.hidden = false;
    el.classList.add("lc-hover-docked");
    el.style.left = pos.left + "px";
    el.style.top = pos.top + "px";
  }

  function placeHover(point) {
    const el = $("#lc-hover");
    const host = document.querySelector(".lc-chart-host");
    if (!el || !host) return;
    el.hidden = false;
    if (hoverDragging || hoverResizing) return;
    if (hoverCorner !== "follow") {
      applyHoverCorner();
      return;
    }
    if (!point) return;
    const pad = 14;
    let x = point.x + pad, y = point.y + pad;
    const w = el.offsetWidth || 180, h = el.offsetHeight || 120;
    if (x + w > host.clientWidth - 8) x = point.x - w - pad;
    if (y + h > host.clientHeight - 8) y = point.y - h - pad;
    el.style.left = Math.max(4, x) + "px";
    el.style.top = Math.max(4, y) + "px";
  }

  function hitPrint(point) {
    if (!point || !printHits.length) return null;
    const pad = plotLeftPad();
    let best = null, bestD = Infinity;
    for (let i = 0; i < printHits.length; i++) {
      const h = printHits[i];
      const xs = pad ? [h.cx, h.cx - pad] : [h.cx];
      for (let k = 0; k < xs.length; k++) {
        const dx = point.x - xs[k], dy = point.y - h.cy;
        const d = dx * dx + dy * dy;
        const lim = h.rad + 6;
        if (d <= lim * lim && d < bestD) { bestD = d; best = h; }
      }
    }
    return best;
  }
  function hitLevelBlendCluster(point, px) {
    if (!point || px == null || !Number.isFinite(px) || !lastLevelBlendHits.length) return null;
    const pad = plotLeftPad();
    const xCands = pad ? [point.x, point.x - pad] : [point.x];
    let best = null;
    for (let i = lastLevelBlendHits.length - 1; i >= 0; i--) {
      const h = lastLevelBlendHits[i];
      if (!(px >= h.lo && px <= h.hi)) continue;
      let insideX = false;
      for (let k = 0; k < xCands.length; k++) {
        const x = xCands[k];
        if (x >= h.bandLeft && x <= h.bandRight) { insideX = true; break; }
      }
      if (insideX) { best = h; break; }
    }
    return best;
  }
  function paintCursor(param) {
    const hoverEl = $("#lc-hover");
    const onChart = !!(param && param.time != null && param.point);
    const hit = (param && param.point) ? hitPrint(param.point) : null;
    hoverPrint = hit;
    const snap = cursorSnapshot(onChart ? param : (lastSnap ? { time: lastSnap.unix } : null));
    if (snap) {
      snap.print = hit && hit.row ? hit.row : null;
      snap.point = onChart ? param.point : null;
      snap.blendCluster = (onChart && snap.px != null)
        ? hitLevelBlendCluster(param.point, snap.px) : null;
      lastSnap = snap;
    }
    renderReadout(snap, hoverEl);
    if (!hoverBoxOn()) {
      if (hoverEl) hoverEl.hidden = true;
      return;
    }
    if (hoverResizing) return;
    const docked = hoverCorner !== "follow" || hoverDragging;
    if (docked) {
      if (hoverEl) hoverEl.hidden = false;
      if (!hoverDragging) applyHoverCorner();
      return;
    }
    if (onChart && snap) placeHover(param.point);
    else if (hoverEl) hoverEl.hidden = true;
  }

  function armHoverDrag() {
    const el = $("#lc-hover");
    const grip = el && el.querySelector(".lc-hover-grip");
    const host = document.querySelector(".lc-chart-host");
    if (!el || !grip || !host || grip.dataset.armed) return;
    grip.dataset.armed = "1";
    applyHoverSize();
    syncHoverFollowButton();
    let ox = 0, oy = 0, startLeft = 0, startTop = 0, moved = false;
    const onMove = (e) => {
      if (!hoverDragging) return;
      const hr = host.getBoundingClientRect();
      let left = e.clientX - hr.left - ox;
      let top = e.clientY - hr.top - oy;
      left = Math.max(0, Math.min(left, host.clientWidth - el.offsetWidth));
      top = Math.max(0, Math.min(top, host.clientHeight - el.offsetHeight));
      if (Math.hypot(left - startLeft, top - startTop) > 5) moved = true;
      el.style.left = left + "px";
      el.style.top = top + "px";
    };
    const onUp = (e) => {
      if (!hoverDragging) return;
      hoverDragging = false;
      el.classList.remove("lc-hover-dragging");
      try { grip.releasePointerCapture(e.pointerId); } catch {}
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      setHoverGuides(false);
      if (!moved) {
        if (hoverCorner !== "follow") applyHoverCorner();
        return;
      }
      const left = parseFloat(el.style.left) || 0;
      const top = parseFloat(el.style.top) || 0;
      const pos = clampHoverPos(left, top, host, el);
      hoverPos = pos;
      hoverCorner = "pin";
      localStorage.setItem("lc_hover_corner", "pin");
      localStorage.setItem("lc_hover_pos", JSON.stringify(pos));
      el.style.left = pos.left + "px";
      el.style.top = pos.top + "px";
      el.classList.add("lc-hover-docked");
      syncHoverFollowButton();
    };
    grip.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      if (e.target && e.target.closest && e.target.closest(".lc-hover-follow, .lc-hover-pin")) return;
      e.stopPropagation();
      el.hidden = false;
      const r = el.getBoundingClientRect();
      const hr = host.getBoundingClientRect();
      ox = e.clientX - r.left;
      oy = e.clientY - r.top;
      startLeft = r.left - hr.left;
      startTop = r.top - hr.top;
      moved = false;
      hoverDragging = true;
      el.classList.add("lc-hover-dragging");
      try { grip.setPointerCapture(e.pointerId); } catch {}
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
    grip.addEventListener("dblclick", (e) => {
      if (e.target && e.target.closest && e.target.closest(".lc-hover-follow, .lc-hover-pin")) return;
      e.preventDefault();
      e.stopPropagation();
      setHoverFollow();
    });
    const followBtn = $("#lc-hover-follow");
    if (followBtn) {
      followBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        setHoverFollow();
      });
    }
    const pinBtn = $("#lc-hover-pin");
    if (pinBtn) {
      pinBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        pinHoverAt(e.clientX, e.clientY);
      });
    }
    if (!host.dataset.hoverPinArmed) {
      host.dataset.hoverPinArmed = "1";
      host.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        if (e.target && e.target.closest && e.target.closest("#lc-hover-follow")) return;
        pinHoverAt(e.clientX, e.clientY);
      });
    }
    el.querySelectorAll(".lc-hover-rz").forEach(handle => {
      handle.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const edge = handle.dataset.edge || "";
        const hr = host.getBoundingClientRect();
        const r = el.getBoundingClientRect();
        const start = {
          x: e.clientX, y: e.clientY,
          l: r.left - hr.left, t: r.top - hr.top,
          w: r.width, h: r.height,
        };
        hoverResizing = true;
        el.hidden = false;
        const onRz = (ev) => {
          const dx = ev.clientX - start.x;
          const dy = ev.clientY - start.y;
          let w = start.w;
          let hgt = start.h;
          if (edge.includes("e")) w = start.w + dx;
          if (edge.includes("s")) hgt = start.h + dy;
          if (edge.includes("w")) w = start.w - dx;
          if (edge.includes("n")) hgt = start.h - dy;
          const maxW = Math.max(140, host.clientWidth - 8);
          const maxH = Math.max(72, host.clientHeight - 8);
          w = Math.max(140, Math.min(maxW, w));
          hgt = Math.max(72, Math.min(maxH, hgt));
          let left = start.l;
          let top = start.t;
          if (edge.includes("w")) left = start.l + (start.w - w);
          if (edge.includes("n")) top = start.t + (start.h - hgt);
          left = Math.max(0, Math.min(left, host.clientWidth - w));
          top = Math.max(0, Math.min(top, host.clientHeight - hgt));
          el.classList.add("is-sized");
          el.style.left = left + "px";
          el.style.top = top + "px";
          el.style.width = w + "px";
          el.style.height = hgt + "px";
        };
        const onRzUp = () => {
          hoverResizing = false;
          hoverSize = { w: el.offsetWidth, h: el.offsetHeight };
          localStorage.setItem("lc_hover_size", JSON.stringify(hoverSize));
          window.removeEventListener("pointermove", onRz);
          window.removeEventListener("pointerup", onRzUp);
          if (hoverCorner === "pin") {
            const pos = clampHoverPos(parseFloat(el.style.left) || 0, parseFloat(el.style.top) || 0, host, el);
            hoverPos = pos;
            localStorage.setItem("lc_hover_pos", JSON.stringify(pos));
            el.style.left = pos.left + "px";
            el.style.top = pos.top + "px";
          } else if (hoverCorner !== "follow") applyHoverCorner();
        };
        window.addEventListener("pointermove", onRz);
        window.addEventListener("pointerup", onRzUp);
      });
    });
  }

  function replayBarIndex() {
    const maps = lastMaps;
    if (!maps || !maps.real.length) return -1;
    const cut = replayUnix;
    if (cut == null || !Number.isFinite(cut)) return maps.real.length - 1;
    let lo = 0, hi = maps.real.length - 1, ans = 0;
    while (lo <= hi) {
      const m = (lo + hi) >> 1;
      if (maps.real[m] <= cut) { ans = m; lo = m + 1; }
      else hi = m - 1;
    }
    return ans;
  }
  function setReplayUnix(unix, fromSlider, fromDrag) {
    const maps = lastMaps;
    if (!maps || !maps.real.length) return;
    let u = unix;
    if (u == null || !Number.isFinite(u)) u = maps.real[maps.real.length - 1];
    if (u < maps.real[0]) u = maps.real[0];
    if (u > maps.real[maps.real.length - 1]) u = maps.real[maps.real.length - 1];
    replayUnix = u;
    const j = replayBarIndex();
    const slider = $("#lc-replay-slider");
    if (slider && !fromSlider) {
      slider.max = String(Math.max(0, maps.real.length - 1));
      slider.value = String(Math.max(0, j));
    }
    const lab = $("#lc-replay-time");
    if (lab) {
      const iso = new Date(u * 1000).toISOString();
      lab.textContent = toCT(iso) || fmtCrosshairTime(u);
    }
    applyReplayView();
    if (!fromDrag) positionReplayLayer();
    scheduleReplayProfiles();
  }
  function withFrozenTimeScale(fn) {
    if (!chart || !chart.timeScale) { fn(); return; }
    const ts = chart.timeScale();
    let logical = null, timeRange = null;
    try { logical = ts.getVisibleLogicalRange(); } catch {}
    try { timeRange = ts.getVisibleRange(); } catch {}
    fn();
    try {
      if (logical && typeof ts.setVisibleLogicalRange === "function") {
        ts.setVisibleLogicalRange(logical);
      } else if (timeRange && typeof ts.setVisibleRange === "function") {
        ts.setVisibleRange(timeRange);
      }
    } catch {}
  }
  function applyReplayView() {
    if (!payload || !chart || !candleSeries) return;
    lastMaps = barMaps(payload);
    withFrozenTimeScale(() => {
      candleSeries.setData(candleData(payload));
      volSeries.setData(volData(payload));
      syncOverlaySeries(payload);
    });
    scheduleDrawProfiles();
  }
  function positionReplayLayer() {
    const layer = $("#lc-replay-layer");
    const line = $("#lc-replay-line");
    const curtain = $("#lc-replay-curtain");
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    if (!layer || !host) return;
    const on = replayOn() && payload && payload.n;
    layer.hidden = !on;
    const controls = $("#lc-replay-controls");
    if (controls) controls.hidden = !replayOn();
    if (!on || !line || !curtain) return;
    const cut = replayCutUnix();
    const x = cut != null ? timeToX(cut) : null;
    const leftPad = plotLeftPad();
    const plotW = (chart && chart.timeScale && chart.timeScale().width) ? chart.timeScale().width() : host.clientWidth;
    const px = (x != null && Number.isFinite(x)) ? (x + leftPad) : (leftPad + plotW);
    line.style.left = Math.max(0, px - 6) + "px";
    curtain.style.left = px + "px";
    curtain.style.width = Math.max(0, host.clientWidth - px) + "px";
  }
  function unixFromClientX(clientX) {
    const host = $("#lc-chart");
    if (!host || !chart || !chart.timeScale) return null;
    const r = host.getBoundingClientRect();
    const x = clientX - r.left - plotLeftPad();
    let t = null;
    try {
      if (typeof chart.timeScale().coordinateToTime === "function") {
        t = chart.timeScale().coordinateToTime(x);
      }
    } catch {}
    if (t == null || !Number.isFinite(t)) {
      const maps = lastMaps;
      if (!maps || !maps.plot.length) return null;
      let lo = 0, hi = maps.plot.length - 1, ans = 0;
      while (lo <= hi) {
        const m = (lo + hi) >> 1;
        const xm = chart.timeScale().timeToCoordinate(maps.plot[m]);
        if (xm != null && xm <= x) { ans = m; lo = m + 1; }
        else hi = m - 1;
      }
      return maps.real[ans];
    }
    return plotToReal(t);
  }
  function scheduleReplayProfiles() {
    clearTimeout(replayVpTimer);
    if (!replayOn() || !vpOn() || !payload) return;
    if (replayPlaying) return; // skip refetch spam while autoplaying
    replayVpTimer = setTimeout(loadReplayProfiles, 420);
  }
  function replayIntervalMs() {
    const speed = replaySpeed > 0 ? replaySpeed : 1;
    return Math.max(50, Math.round(REPLAY_BASE_MS / speed));
  }
  function syncReplaySpeedButtons() {
    const host = $("#lc-replay-speeds");
    if (!host) return;
    host.querySelectorAll("[data-replay-speed]").forEach(btn => {
      const v = Number(btn.getAttribute("data-replay-speed"));
      btn.classList.toggle("is-active", v === replaySpeed);
    });
  }
  function syncReplayPlayButton() {
    const btn = $("#lc-replay-play");
    if (!btn) return;
    btn.textContent = replayPlaying ? "❚❚" : "▶";
    btn.setAttribute("aria-label", replayPlaying ? "Pause replay" : "Play replay");
    btn.title = replayPlaying ? "Pause" : "Play";
  }
  function stopReplayPlay() {
    replayPlaying = false;
    if (replayPlayTimer) {
      clearInterval(replayPlayTimer);
      replayPlayTimer = 0;
    }
    syncReplayPlayButton();
  }
  function stepReplay(deltaBars) {
    if (!replayOn()) return false;
    const maps = lastMaps || (payload ? barMaps(payload) : null);
    if (!maps || !maps.real.length) return false;
    const cur = replayBarIndex();
    const next = Math.max(0, Math.min(maps.real.length - 1, cur + (deltaBars || 1)));
    if (next === cur && next >= maps.real.length - 1) return false;
    setReplayUnix(maps.real[next]);
    return next < maps.real.length - 1;
  }
  function startReplayPlay() {
    if (!replayOn()) return;
    stopReplayPlay();
    replayPlaying = true;
    syncReplayPlayButton();
    const tick = () => {
      if (!still() || !replayOn() || !replayPlaying) {
        stopReplayPlay();
        return;
      }
      const more = stepReplay(1);
      if (!more) {
        stopReplayPlay();
        scheduleReplayProfiles();
      }
    };
    replayPlayTimer = setInterval(tick, replayIntervalMs());
  }
  function setReplaySpeed(speed) {
    const n = Number(speed);
    replaySpeed = [0.25, 0.5, 1, 2, 3, 4].includes(n) ? n : 1;
    localStorage.setItem("lc_replay_speed", String(replaySpeed));
    syncReplaySpeedButtons();
    if (replayPlaying) startReplayPlay();
  }
  function applyReplayBoxCollapsed() {
    const box = $("#lc-replay-box");
    const btn = $("#lc-replay-collapse");
    if (!box || !btn) return;
    const collapsed = localStorage.getItem("lc_replay_box_collapsed") === "1";
    box.classList.toggle("is-collapsed", collapsed);
    btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    btn.textContent = collapsed ? "▸" : "▾";
    btn.title = collapsed ? "Expand Replay" : "Collapse Replay";
    btn.setAttribute("aria-label", collapsed ? "Expand Replay" : "Collapse Replay");
  }
  async function loadReplayProfiles() {
    if (!still() || !replayOn() || !vpOn() || !payload) return;
    const cut = replayCutUnix();
    if (cut == null) return;
    const tf = $("#lc-tf").value || "1min";
    const lb = currentLookback();
    const keys = selectedKeys();
    const q = new URLSearchParams({ timeframe: tf, lookback: lb });
    if (keys.length) q.set("overlays", keys.join(","));
    q.set("cvd_origin", cvdOrigin());
    if (keys.includes("vw_vwap") || keys.includes("vw_vwap_dist") || vwapBandsOn()) q.set("vwap_origin", vwapOrigin());
    if (keys.includes("of_ofi_cum")) q.set("ofi_origin", ofiOrigin());
    if (keys.includes("ad_line")) q.set("ad_origin", adOrigin());
    q.set("profiles", JSON.stringify(profilesForApi()));
    if (emaOn() && emaSpecs.length) q.set("emas", JSON.stringify(emasForApi()));
    if (stochOn() && stochSpecs.length) q.set("stochs", JSON.stringify(stochsForApi()));
    if (keys.includes("ps_sar")) q.set("sar", JSON.stringify(sarForApi()));
    if (vwapBandsOn()) q.set("vwap_bands", "1");
    if (!ethAgg()) q.set("eth_aggregate", "0");
    q.set("asof", new Date(cut * 1000).toISOString());
    try {
      const p = await api(`/api/chart/series?${q.toString()}`, { timeoutMs: 180000 });
      if (!still() || !replayOn()) return;
      if (p && Array.isArray(p.profiles)) payload.profiles = p.profiles;
      if (p && Array.isArray(p.emas) && payload) {
        payload.emas = p.emas;
        if (p.overlays) {
          for (const spec of p.emas) {
            if (spec && spec.key && p.overlays[spec.key]) {
              payload.overlays[spec.key] = p.overlays[spec.key];
            }
          }
        }
        if (chart) syncEmaSeries(payload);
      }
      if (p && Array.isArray(p.stochs) && payload) {
        payload.stochs = p.stochs;
        if (p.overlays) {
          for (const spec of p.stochs) {
            if (spec && spec.key && p.overlays[spec.key]) {
              payload.overlays[spec.key] = p.overlays[spec.key];
            }
          }
        }
        if (chart) syncOverlaySeries(payload);
      }
      if (p && p.overlays && p.overlays.ps_sar && payload) {
        payload.overlays.ps_sar = p.overlays.ps_sar;
        payload.sar = p.sar || payload.sar;
        scheduleDrawProfiles();
      }
      scheduleDrawProfiles();
    } catch {}
  }
  function setChartPan(enabled) {
    if (!chart) return;
    try {
      chart.applyOptions({
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: !!enabled,
          horzTouchDrag: !!enabled,
          vertTouchDrag: false,
        },
      });
    } catch {}
  }
  function positionReplayLineAtClientX(clientX) {
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    const line = $("#lc-replay-line");
    const curtain = $("#lc-replay-curtain");
    if (!host || !line || !curtain) return;
    const r = host.getBoundingClientRect();
    const px = Math.max(0, Math.min(host.clientWidth, clientX - r.left));
    line.style.left = (px - 6) + "px";
    curtain.style.left = px + "px";
    curtain.style.width = Math.max(0, host.clientWidth - px) + "px";
  }
  function armReplayDrag() {
    const slider = $("#lc-replay-slider");
    if (slider && !slider.dataset.armed) {
      slider.dataset.armed = "1";
      slider.addEventListener("pointerdown", (e) => e.stopPropagation());
    }
    const line = $("#lc-replay-line");
    if (!line || line.dataset.armed) return;
    line.dataset.armed = "1";
    const onMove = (e) => {
      if (!replayDragging) return;
      e.preventDefault();
      positionReplayLineAtClientX(e.clientX);
      const u = unixFromClientX(e.clientX);
      if (u != null) setReplayUnix(u, false, true);
    };
    const onUp = () => {
      replayDragging = false;
      setChartPan(true);
      positionReplayLayer();
      scheduleReplayProfiles();
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    line.addEventListener("pointerdown", (e) => {
      if (!replayOn()) return;
      e.preventDefault();
      e.stopPropagation();
      stopReplayPlay();
      replayDragging = true;
      setChartPan(false);
      try { line.setPointerCapture(e.pointerId); } catch {}
      positionReplayLineAtClientX(e.clientX);
      const u = unixFromClientX(e.clientX);
      if (u != null) setReplayUnix(u, false, true);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
  }
  function enableReplay(on) {
    const box = $("#lc-replay");
    if (box) box.checked = !!on;
    localStorage.setItem("lc_replay", on ? "1" : "0");
    if (on) {
      if (liveOn() && $("#lc-live")) {
        $("#lc-live").checked = false;
        localStorage.setItem("lc_live", "0");
        armPoll();
      }
      const maps = lastMaps || (payload ? barMaps(payload) : null);
      if (maps && maps.real.length) {
        if (replayUnix == null) replayUnix = maps.real[maps.real.length - 1];
        setReplayUnix(replayUnix);
      } else {
        positionReplayLayer();
      }
      syncReplaySpeedButtons();
      syncReplayPlayButton();
    } else {
      stopReplayPlay();
      replayUnix = null;
      clearTimeout(replayVpTimer);
      positionReplayLayer();
      if (payload && chart) {
        lastMaps = barMaps(payload);
        withFrozenTimeScale(() => {
          candleSeries.setData(candleData(payload));
          volSeries.setData(volData(payload));
          syncOverlaySeries(payload);
        });
        if (vpOn()) loadSeries();
        else scheduleDrawProfiles();
      }
    }
  }

  function armPaneStretchSave() {
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    if (!host || host.dataset.stretchArmed) return;
    host.dataset.stretchArmed = "1";
    const persist = () => {
      savePaneStretch();
      scheduleDrawProfiles();
    };
    host.addEventListener("pointerup", persist);
    host.addEventListener("pointercancel", persist);
  }

  async function ensureChart() {
    const LW = await loadLwCharts();
    lwLib = LW;
    const host = $("#lc-chart");
    if (!host || chart) return LW;
    const tfNow = ($("#lc-tf") && $("#lc-tf").value) || "1min";
    chart = LW.createChart(host, {
      layout: {
        background: { color: "#0b0f14" },
        textColor: "#7d8da0",
        fontSize: textSize(),
        fontFamily: "SF Mono, ui-monospace, Menlo, Consolas, monospace",
        panes: {
          enableResize: true,
          separatorColor: "#9eb0c2",
          separatorHoverColor: "rgba(126, 224, 163, 0.55)",
        },
      },
      grid: { vertLines: { color: "#1f2a36" }, horzLines: { color: "#1f2a36" } },
      crosshair: { mode: LW.CrosshairMode ? LW.CrosshairMode.Normal : 0 },
      leftPriceScale: { visible: false, borderColor: "#1f2a36" },
      rightPriceScale: { borderColor: "#1f2a36", visible: true },
      timeScale: {
        borderColor: "#1f2a36", timeVisible: true,
        secondsVisible: tfNow === "5s" || tfNow === "15s",
        rightOffset: 4, barSpacing: 6, minBarSpacing: 0.3,
        shiftVisibleRangeOnNewBar: false,
        tickMarkFormatter: fmtTick,
      },
      localization: { timeFormatter: fmtCrosshairTime },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: {
        axisPressedMouseMove: { time: true, price: true },
        mouseWheel: true,
        pinch: true,
        axisDoubleClickReset: true,
      },
      width: host.clientWidth,
      height: host.clientHeight,
    });
    lwEnsurePane(chart, 1);
    candleSeries = lwAddSeries(chart, LW, "CandlestickSeries", {
      upColor: "#7ee0a3", downColor: "#ff6b6b",
      wickUpColor: "#7ee0a3", wickDownColor: "#ff6b6b",
      borderUpColor: "#7ee0a3", borderDownColor: "#ff6b6b",
      priceLineVisible: true,
      lastValueVisible: false,
      autoscaleInfoProvider: vpAxisAutoscale,
    }, 0);
    volSeries = lwAddSeries(chart, LW, "HistogramSeries", {
      color: "#7ee0a3",
      priceFormat: { type: "volume" },
      lastValueVisible: false,
      priceLineVisible: false,
      title: "Vol",
    }, 1);
    try {
      candleSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.04, bottom: 0.06 },
        borderVisible: true,
        visible: true,
      });
    } catch {}
    try {
      volSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.08, bottom: 0.04 },
        borderVisible: true,
        visible: true,
      });
    } catch {}
    lwZeroLine(volSeries);
    ensureZeroInScale(volSeries, true);
    showPaneScale(0);
    showPaneScale(1);
    if (!chart.panes || chart.panes().length < 2) {
      throw new Error("volume pane failed to create — Lightweight Charts v5 panes required");
    }
    applyPaneHeights(true);
    chart.subscribeCrosshairMove(param => {
      paintCursor(param);
    });
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      try {
        drawProfiles();
        if (!replayDragging) positionReplayLayer();
        if (typeof paintNoteSelection === "function") paintNoteSelection();
      } catch (err) {
        console.error(err);
      }
    });
    return LW;
  }

  function cvdStyle() {
    return ($("#lc-cvd-style") && $("#lc-cvd-style").value) || "line";
  }

  function cvdCandleData(p) {
    const maps = lastMaps || barMaps(p);
    const pack = p && p.cvd_ohlc;
    const out = [];
    if (pack && pack.c && p.n) {
      for (let j = 0; j < maps.idx.length; j++) {
        const i = maps.idx[j];
        if (pastReplay(maps.real[j])) { out.push({ time: maps.plot[j] }); continue; }
        if (pack.c[i] == null || !Number.isFinite(pack.c[i])) continue;
        const c = pack.c[i];
        const o = pack.o && pack.o[i] != null ? pack.o[i] : c;
        const h = pack.h && pack.h[i] != null ? pack.h[i] : Math.max(o, c);
        const l = pack.l && pack.l[i] != null ? pack.l[i] : Math.min(o, c);
        out.push({ time: maps.plot[j], open: o, high: h, low: l, close: c });
      }
      return out;
    }
    const arr = (p.overlays && p.overlays.of_cvd) || [];
    let prev = null;
    for (let j = 0; j < maps.idx.length; j++) {
      const i = maps.idx[j];
      if (pastReplay(maps.real[j])) continue;
      if (arr[i] == null || !Number.isFinite(arr[i])) continue;
      const c = arr[i];
      const o = prev == null ? c : prev;
      out.push({ time: maps.plot[j], open: o, high: Math.max(o, c), low: Math.min(o, c), close: c });
      prev = c;
    }
    return out;
  }

  function applyPaneLayout() {
    applyStretch();
  }

  function attachOscScale(series, spec, key, paneIndex) {
    if (!series) return;
    try {
      series.priceScale().applyOptions({
        scaleMargins: { top: 0.12, bottom: 0.12 },
        borderVisible: true,
        visible: true,
      });
    } catch {}
    if (paneIndex != null) showPaneScale(paneIndex);
    if (key === "mf_mfi" || key === "so_fast_d_9_3" || (key && String(key).startsWith("stoch_"))) {
      // 0–100 oscillators (MFI, Stoch Fast %D). Sync scores are ±1 — not this range.
      try {
        series.applyOptions({
          autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 100 } }),
          color: (key && String(key).startsWith("stoch_"))
            ? stochColor(Number(String(key).slice(6)))
            : (OV_COLOR[key] || "#7d8da0"),
          title: (key && String(key).startsWith("stoch_") && payload && Array.isArray(payload.stochs))
            ? ((payload.stochs.find(s => s && s.key === key) || {}).label || key)
            : (spec.label || key),
        });
      } catch {}
      return;
    }
    if (key === "so_sync_score_all" || key === "so_sync_score_9_60") {
      // Hive sync scores ∈ [−1, +1].
      try {
        series.applyOptions({
          autoscaleInfoProvider: () => ({ priceRange: { minValue: -1, maxValue: 1 } }),
        });
        series.createPriceLine({
          price: 0, color: "#7d8da0", lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: "0",
        });
      } catch {}
      return;
    }
    if (overlayZeroLine(spec) || key === "of_cvd") {
      lwZeroLine(series);
      ensureZeroInScale(series, false);
    }
  }

  function boundLevelTitle(n) {
    const r = Math.round(Number(n) * 10) / 10;
    if (!Number.isFinite(r)) return "";
    return Math.abs(r - Math.round(r)) < 0.05 ? String(Math.round(r)) : r.toFixed(1);
  }
  function applyBoundLines(series, upper, lower) {
    if (!series || typeof series.createPriceLine !== "function") return;
    const prev = boundLineRefs.get(series) || [];
    for (const line of prev) {
      try { series.removePriceLine(line); } catch {}
    }
    const lines = [];
    const add = (price, color) => {
      try {
        lines.push(series.createPriceLine({
          price, color, lineWidth: 1, lineStyle: 2,
          axisLabelVisible: true, title: boundLevelTitle(price),
        }));
      } catch {}
    };
    if (upper != null && Number.isFinite(upper)) add(upper, "#ff6b6b");
    if (lower != null && Number.isFinite(lower)) add(lower, "#7ee0a3");
    boundLineRefs.set(series, lines);
  }
  function stochBoundsForKey(key) {
    const n = Number(String(key).slice(6));
    const spec = stochSpecs.find(s => Number(s && s.n) === n);
    return lcNormalizeOscBounds(spec && spec.upper, spec && spec.lower);
  }
  function refreshBoundLines() {
    for (const k of Object.keys(overlaySeries)) {
      if (String(k).startsWith("stoch_")) {
        const b = stochBoundsForKey(k);
        applyBoundLines(overlaySeries[k], b.upper, b.lower);
      } else if (k === "mf_mfi") {
        const b = storedMfiBounds();
        applyBoundLines(overlaySeries[k], b.upper, b.lower);
      }
    }
  }

  function wipeOscSeries() {
    const metaBy = overlayMetaBy();
    for (const k of Object.keys(overlaySeries)) {
      const spec = metaBy[k] || { pane: "osc" };
      if (k !== "of_cvd" && !overlayIsOsc(spec, k)) continue;
      try { chart.removeSeries(overlaySeries[k]); } catch {}
      delete overlaySeries[k];
    }
    if (cvdCandleSeries) {
      try { chart.removeSeries(cvdCandleSeries); } catch {}
      cvdCandleSeries = null;
    }
  }

  function syncCvdSeries(p, LW, paneIndex) {
    const lib = LW || lwLib || window.LightweightCharts;
    const on = selectedKeys().includes("of_cvd") && paneIndex != null;
    if (on) lwEnsurePane(chart, paneIndex);
    const style = cvdStyle();
    const wantLine = on && (style === "line" || style === "both");
    const wantCandle = on && (style === "candle" || style === "both");
    const ttl = `CVD (${(p && p.cvd_origin) || cvdOrigin()})`;
    const spec = overlayMetaBy().of_cvd || { pane: "osc", zero_line: "1" };
    if (!wantLine && overlaySeries.of_cvd) {
      try { chart.removeSeries(overlaySeries.of_cvd); } catch {}
      delete overlaySeries.of_cvd;
    }
    if (wantLine && !overlaySeries.of_cvd) {
      overlaySeries.of_cvd = lwAddSeries(chart, lib, "LineSeries", {
        color: "#f0c674",
        lineWidth: style === "both" ? 1 : 2,
        lastValueVisible: false, priceLineVisible: false,
        priceScaleId: "right",
        priceFormat: { type: "price", precision: 0, minMove: 1 },
        title: ttl,
      }, paneIndex);
      attachOscScale(overlaySeries.of_cvd, spec, "of_cvd", paneIndex);
    }
    if (wantLine) {
      try { overlaySeries.of_cvd.applyOptions({ title: ttl, lineWidth: style === "both" ? 1 : 2 }); } catch {}
      overlaySeries.of_cvd.setData(lineData(p, "of_cvd"));
      lwForcePane(overlaySeries.of_cvd, paneIndex);
    }
    if (!wantCandle && cvdCandleSeries) {
      try { chart.removeSeries(cvdCandleSeries); } catch {}
      cvdCandleSeries = null;
    }
    if (wantCandle && !cvdCandleSeries) {
      cvdCandleSeries = lwAddSeries(chart, lib, "CandlestickSeries", {
        upColor: "#f0c674",
        downColor: "#56b6c2",
        wickUpColor: "#f0c674",
        wickDownColor: "#56b6c2",
        borderUpColor: "#f0c674",
        borderDownColor: "#56b6c2",
        priceScaleId: "right",
        lastValueVisible: false,
        priceLineVisible: false,
        priceFormat: { type: "price", precision: 0, minMove: 1 },
        title: wantLine ? "" : ttl,
      }, paneIndex);
      if (!wantLine) attachOscScale(cvdCandleSeries, spec, "of_cvd", paneIndex);
    }
    if (wantCandle) {
      try {
        cvdCandleSeries.applyOptions({
          title: wantLine ? "" : ttl,
          lastValueVisible: false,
        });
      } catch {}
      cvdCandleSeries.setData(cvdCandleData(p));
      lwForcePane(cvdCandleSeries, paneIndex);
    }
  }

  function createOscSeries(k, spec, paneIndex, LW) {
    const lib = LW || lwLib || window.LightweightCharts;
    if (k === "of_delta") {
      overlaySeries[k] = lwAddSeries(chart, lib, "HistogramSeries", {
        color: "#7ee0a3",
        baseLineVisible: false,
        lastValueVisible: false,
        priceLineVisible: false,
        priceScaleId: "right",
        priceFormat: { type: "price", precision: 0, minMove: 1 },
        title: spec.label || "Delta Volume",
      }, paneIndex);
      attachOscScale(overlaySeries[k], spec, k, paneIndex);
      return;
    }
    overlaySeries[k] = lwAddSeries(chart, lib, "LineSeries", {
      color: OV_COLOR[k] || "#7d8da0",
      lineWidth: 1,
      lineVisible: true,
      pointMarkersVisible: false,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      priceScaleId: "right",
      priceFormat: k === "mf_mfi"
        ? { type: "price", precision: 1, minMove: 0.1 }
        : undefined,
      title: k === "of_ofi_cum" ? `OFI_CUM (${ofiOrigin()})`
        : (k === "ad_line" ? `A/D (${adOrigin()})` : (spec.label || k)),
    }, paneIndex);
    attachOscScale(overlaySeries[k], spec, k, paneIndex);
  }

  function syncEmaSeries(p, LW) {
    const lib = LW || lwLib || window.LightweightCharts;
    const active = new Map();
    if (emaOn() && p && Array.isArray(p.emas)) {
      for (const spec of p.emas) {
        if (!spec || !spec.key) continue;
        active.set(String(spec.key), spec);
      }
    }
    for (const k of Object.keys(overlaySeries)) {
      if (!k.startsWith("ema_")) continue;
      if (!active.has(k)) {
        try { chart.removeSeries(overlaySeries[k]); } catch {}
        delete overlaySeries[k];
      }
    }
    for (const [k, spec] of active) {
      const color = emaColor(spec.n);
      const title = spec.label || `${emaMark(spec.n)} (${spec.lookback})`;
      if (!overlaySeries[k]) {
        overlaySeries[k] = lwAddSeries(chart, lib, "LineSeries", {
          color,
          lineWidth: 2,
          lineVisible: true,
          pointMarkersVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: true,
          priceScaleId: "right",
          title,
        }, 0);
      } else {
        try { overlaySeries[k].applyOptions({ color, title }); } catch {}
      }
      overlaySeries[k].setData(lineData(p, k));
    }
  }

  function syncVwapBandSeries(p, LW) {
    const lib = LW || lwLib || window.LightweightCharts;
    const want = !!(vwapBandsOn() && p && p.overlays
      && (p.overlays.vw_bb_upper || p.overlays.vw_bb_lower));
    const origin = (p && p.vwap_origin) || vwapOrigin();
    const specs = want ? [
      { key: "vw_bb_upper", title: `VWAP +2σ (${origin})`, color: OV_COLOR.vw_bb_upper || "#d4b36a" },
      { key: "vw_bb_lower", title: `VWAP −2σ (${origin})`, color: OV_COLOR.vw_bb_lower || "#d4b36a" },
    ] : [];
    const active = new Set(specs.map(s => s.key));
    for (const k of Object.keys(overlaySeries)) {
      if (!k.startsWith("vw_bb_")) continue;
      if (!active.has(k)) {
        try { chart.removeSeries(overlaySeries[k]); } catch {}
        delete overlaySeries[k];
      }
    }
    for (const spec of specs) {
      if (!(p.overlays && p.overlays[spec.key])) continue;
      if (!overlaySeries[spec.key]) {
        overlaySeries[spec.key] = lwAddSeries(chart, lib, "LineSeries", {
          color: spec.color,
          lineWidth: 1,
          lineStyle: 2,
          lineVisible: true,
          pointMarkersVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
          priceScaleId: "right",
          title: spec.title,
        }, 0);
      } else {
        try { overlaySeries[spec.key].applyOptions({ color: spec.color, title: spec.title }); } catch {}
      }
      overlaySeries[spec.key].setData(lineData(p, spec.key));
    }
  }

  function syncOverlaySeries(p, LW) {
    const lib = LW || lwLib || window.LightweightCharts;
    const keys = selectedKeys();
    const metaBy = overlayMetaBy();
    const oscKeys = oscKeysOn();
    const layout = oscKeys.join(",");
    const layoutChanged = layout !== oscLayoutKey;

    for (const k of Object.keys(overlaySeries)) {
      if (k === "of_cvd" || k.startsWith("ema_") || k.startsWith("vw_bb_")) continue;
      const spec = metaBy[k] || { pane: "osc" };
      if (overlayIsOsc(spec, k)) continue;
      if (!keys.includes(k) || spec.pane === "canvas" || spec.needs === "prints") {
        try { chart.removeSeries(overlaySeries[k]); } catch {}
        delete overlaySeries[k];
      }
    }

    if (layoutChanged) {
      wipeOscSeries();
      pruneExtraPanes(2);
      oscLayoutKey = layout;
      oscKeys.forEach((k, i) => {
        const paneIndex = 2 + i;
        const spec = metaBy[k] || { pane: "osc" };
        if (k === "of_cvd") {
          cvdPaneIndex = paneIndex;
          return;
        }
        createOscSeries(k, spec, paneIndex, lib);
      });
      if (!oscKeys.includes("of_cvd")) cvdPaneIndex = null;
    }

    for (const k of keys) {
      const spec = metaBy[k] || { pane: "osc" };
      if (k === "lt_prints" || spec.pane === "canvas" || spec.needs === "prints") continue;
      if (spec.pane !== "price") continue;
      if (!overlaySeries[k]) {
        overlaySeries[k] = lwAddSeries(chart, lib, "LineSeries", {
          color: OV_COLOR[k] || "#7d8da0",
          lineWidth: 2,
          lineVisible: k !== "ps_sar",
          pointMarkersVisible: false,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: k !== "ps_sar",
          priceScaleId: "right",
          title: (k === "vw_vwap" || k === "vw_vwap_dist")
            ? `${spec.label || k} (${p.vwap_origin || vwapOrigin()})`
            : (spec.label || k),
        }, 0);
      }
      if (k === "vw_vwap" || k === "vw_vwap_dist") {
        try { overlaySeries[k].applyOptions({ title: `${spec.label || k} (${p.vwap_origin || vwapOrigin()})` }); } catch {}
      }
      overlaySeries[k].setData(lineData(p, k));
    }

    syncEmaSeries(p, lib);
    syncVwapBandSeries(p, lib);

    const cvdIdx = oscKeys.indexOf("of_cvd");
    syncCvdSeries(p, lib, cvdIdx >= 0 ? 2 + cvdIdx : null);
    if (layoutChanged) applyStretch();
    for (const k of oscKeys) {
      if (k === "of_cvd") continue;
      if (k === "of_ofi_cum" && overlaySeries[k]) {
        try { overlaySeries[k].applyOptions({ title: `OFI_CUM (${p.ofi_origin || ofiOrigin()})` }); } catch {}
      }
      if (k === "ad_line" && overlaySeries[k]) {
        try { overlaySeries[k].applyOptions({ title: `A/D (${p.ad_origin || adOrigin()})` }); } catch {}
      }
      if (overlaySeries[k]) overlaySeries[k].setData(k === "of_delta" ? deltaHistData(p) : lineData(p, k));
    }
    pruneExtraPanes(2 + oscKeys.length);
    refreshBoundLines();
    scheduleDrawProfiles();
  }

  function printCacheKey() {
    const tf = ($("#lc-tf") && $("#lc-tf").value) || "1min";
    const lb = currentLookback();
    return [tf, lb, String(printMin()), ethAgg() ? "1" : "0"].join("|");
  }
  async function loadPrints(p) {
    if (!still() || !selectedKeys().includes("lt_prints")) {
      printCache = null;
      scheduleDrawProfiles();
      return;
    }
    const minSize = printMin();
    localStorage.setItem("lc_print_min", String(minSize));
    const key = printCacheKey();
    const tf = ($("#lc-tf") && $("#lc-tf").value) || "1min";
    const lb = currentLookback();
    const q = new URLSearchParams({
      timeframe: tf, lookback: lb, min_size: String(minSize),
    });
    if (!ethAgg()) q.set("eth_aggregate", "0");
    const reuse = printCache && printCache.key === key && printCache.rows;
    if (reuse && printCache.end) q.set("since", printCache.end);
    try {
      const out = await api(`/api/chart/prints?${q.toString()}`, { timeoutMs: 180000 });
      if (!still()) return;
      const incoming = Array.isArray(out.prints) ? out.prints : [];
      let rows = incoming;
      if (reuse && q.get("since")) {
        const seen = new Set(printCache.rows.map(r => `${r.t}|${r.p}|${r.sz}`));
        rows = printCache.rows.concat(incoming.filter(r => !seen.has(`${r.t}|${r.p}|${r.sz}`)));
      }
      const t0 = p && p.start ? Date.parse(p.start) / 1000 : NaN;
      if (Number.isFinite(t0)) rows = rows.filter(r => r.t >= t0);
      printCache = {
        key, rows, end: out.end || (reuse ? printCache.end : null),
        start: (p && p.start) || out.start,
        minSize, capped: !!(out.capped || (reuse && printCache.capped)),
        n: rows.length,
      };
      scheduleDrawProfiles();
    } catch (e) {
      if (!reuse) printCache = { key, rows: [], end: null, minSize, capped: false, n: 0 };
    }
  }

  async function loadSeries() {
    if (!still()) return;
    if (loading) { reloadQueued = true; return; }
    loading = true;
    setEmpty("");
    setStatus("loading…");
    const tf = $("#lc-tf").value || "1min";
    const lb = currentLookback();
    const keys = selectedKeys();
    localStorage.setItem("lc_tf", tf);
    localStorage.setItem("lc_lb", lb);
    localStorage.setItem("lc_overlays", keys.join(","));
    const q = new URLSearchParams({ timeframe: tf, lookback: lb });
    if (keys.length) q.set("overlays", keys.join(","));
    q.set("cvd_origin", cvdOrigin());
    if (keys.includes("vw_vwap") || keys.includes("vw_vwap_dist") || vwapBandsOn()) q.set("vwap_origin", vwapOrigin());
    if (keys.includes("of_ofi_cum")) q.set("ofi_origin", ofiOrigin());
    if (keys.includes("ad_line")) q.set("ad_origin", adOrigin());
    if (vpOn() && vpSpecs.length) q.set("profiles", JSON.stringify(profilesForApi()));
    if (emaOn() && emaSpecs.length) q.set("emas", JSON.stringify(emasForApi()));
    if (stochOn() && stochSpecs.length) q.set("stochs", JSON.stringify(stochsForApi()));
    if (keys.includes("ps_sar")) q.set("sar", JSON.stringify(sarForApi()));
    if (vwapBandsOn()) q.set("vwap_bands", "1");
    if (!ethAgg()) q.set("eth_aggregate", "0");
    localStorage.setItem("lc_eth_display", ethDisplay() ? "1" : "0");
    localStorage.setItem("lc_eth_shade", ethShade() ? "1" : "0");
    localStorage.setItem("lc_eth_agg", ethAgg() ? "1" : "0");
    localStorage.setItem("lc_cvd_origin", cvdOrigin());
    localStorage.setItem("lc_ofi_origin", ofiOrigin());
    localStorage.setItem("lc_ad_origin", adOrigin());
    localStorage.setItem("lc_vwap_bb", vwapBandsOn() ? "1" : "0");
    persistVpSpecs();
    persistEmaSpecs();
    persistStochSpecs();
    persistSarParams();
    try {
      const p = await api(`/api/chart/series?${q.toString()}`, {
        timeoutMs: vpOn() ? 180000 : 90000,
      });
      if (!still()) return;
      payload = applyVpDisplaySettings(p);
      lastSnap = null;
      $("#lc-contract").textContent = p.contract || "lead";
      if (!p.n) {
        setEmpty("no candles in this window");
        setStatus("");
        renderReadout(null, $("#lc-hover"));
        return;
      }
      const LW = await ensureChart();
      if (!still()) return;
      lastMaps = barMaps(p);
      const tfNow = p.timeframe || ($("#lc-tf") && $("#lc-tf").value) || "1min";
      try {
        chart.applyOptions({
          timeScale: { secondsVisible: tfNow === "5s" || tfNow === "15s" },
        });
      } catch {}
      applyTextSize();
      if (replayOn()) {
        const maps = lastMaps;
        if (maps.real.length && (replayUnix == null || replayUnix > maps.real[maps.real.length - 1])) {
          replayUnix = maps.real[maps.real.length - 1];
        }
        setReplayUnix(replayUnix);
      } else {
        candleSeries.setData(candleData(p));
        volSeries.setData(volData(p));
        lwForcePane(candleSeries, 0);
        lwForcePane(volSeries, 1);
        showPaneScale(0);
        showPaneScale(1);
        syncOverlaySeries(p, LW);
        scheduleDrawProfiles();
      }
      if (!fitted) { chart.timeScale().fitContent(); fitted = true; }
      setEmpty("");
      const bits = [`${fmtInt(p.n)} ${p.timeframe} bars · ${p.contract}`];
      if (p.bars_capped || p.lookback_capped) {
        const maxBars = (p.max_bars != null ? p.max_bars : (meta && meta.max_bars)) || 50000;
        bits.push(`showing newest ${fmtInt(maxBars)} bars`);
      }
      if (keys.includes("of_cvd")) {
        bits.push(`CVD ${ORIGIN_LABEL[p.cvd_origin] || p.cvd_origin}`);
      }
      if (keys.includes("of_ofi_cum")) {
        bits.push(`OFI_CUM ${ORIGIN_LABEL[p.ofi_origin] || p.ofi_origin || ofiOrigin()}`);
      }
      if (vpOn() && p.profiles && p.profiles.length) {
        const withPoc = p.profiles.filter(pr => pr && pr.poc != null);
        if (withPoc.length) {
          bits.push(`${withPoc.length} VP` + (withPoc.length > 1 ? "s" : ""));
          bits.push(`POC ${Number(withPoc[0].poc).toFixed(2)}`);
        } else {
          bits.push("VP no tape in window");
        }
      }
      if (emaOn() && p.emas && p.emas.length) {
        bits.push(p.emas.map(e => `EMA${e.n}(${e.lookback})`).join(", "));
      }
      if (stochOn() && p.stochs && p.stochs.length) {
        bits.push(p.stochs.map(s => `Stoch${s.n}(${s.k_period},${s.d_period})`).join(", "));
      }
      if (keys.includes("ps_sar") && p.sar) {
        bits.push(`SAR ${p.sar.start}/${p.sar.increment}/${p.sar.maximum}`);
      }
      if (vwapBandsOn() || p.vwap_bands) bits.push("VWAP ±2σ");
      if (keys.length) bits.push(keys.join(", "));
      if (keys.includes("lt_prints")) {
        setStatus(bits.join(" · ") + " · prints…");
        await loadPrints(p);
        if (!still()) return;
        if (printCache) {
          bits.push(`${fmtInt(printCache.n)} prints ≥${printMin()}`);
          if (printCache.capped) bits.push("prints capped");
        }
      } else {
        printCache = null;
      }
      if (window.LCOrders && keys.includes(LCOrders.KEY)) {
        bits.push("order map…");
        LCOrders.load(lcOrdersBridge(p), p);   // async; redraws itself when the data lands
      } else if (window.LCOrders) {
        LCOrders.clear();
      }
      setStatus(bits.join(" · "));
      paintCursor(null);
    } catch (e) {
      if (!still()) return;
      setEmpty(e.message || String(e));
      setStatus("");
    } finally {
      loading = false;
      if (reloadQueued && still()) {
        reloadQueued = false;
        loadSeries();
      }
    }
  }

  function armPoll() {
    clearInterval(pollTimer);
    if (!liveOn()) return;
    pollTimer = setInterval(() => { if (still() && liveOn() && !replayOn() && !loading) loadSeries(); }, 8000);
  }

  function chartExpanded() {
    const page = document.querySelector(".lc-page");
    return !!(page && page.classList.contains("is-expanded"));
  }
  function resizeLiveChart() {
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    if (!chart || !host) return;
    try {
      chart.applyOptions({ width: host.clientWidth, height: host.clientHeight });
    } catch {}
    scheduleDrawProfiles();
    if (typeof paintNoteSelection === "function") paintNoteSelection();
  }
  window._lcResizeChart = resizeLiveChart;
  function fitLiveChart() {
    if (!chart) return;
    try { chart.timeScale().fitContent(); fitted = true; } catch {}
    resizeLiveChart();
  }
  function setSideCollapsed(on) {
    const stage = document.querySelector(".lc-stage");
    const btn = $("#lc-side-collapse");
    const want = !!on;
    if (stage) stage.classList.toggle("is-side-collapsed", want);
    localStorage.setItem("lc_side_collapsed", want ? "1" : "0");
    if (btn) {
      btn.setAttribute("aria-expanded", want ? "false" : "true");
      btn.title = want ? "Expand overlays menu" : "Collapse overlays into the right wall";
    }
    requestAnimationFrame(() => resizeLiveChart());
  }
  function setChartExpanded(on) {
    const page = document.querySelector(".lc-page");
    const shellEl = document.getElementById("shell");
    const btn = $("#lc-expand");
    const want = !!on;
    if (page) page.classList.toggle("is-expanded", want);
    if (shellEl) shellEl.classList.toggle("lc-chart-expanded", want);
    localStorage.setItem("lc_expanded", want ? "1" : "0");
    if (btn) {
      btn.textContent = want ? "Exit full" : "Expand";
      btn.title = want ? "Exit full-screen chart (Esc)" : "Expand chart to the whole screen";
      btn.setAttribute("aria-pressed", want ? "true" : "false");
    }
    // Let flex/grid settle, then resize + fit so the full series is visible.
    requestAnimationFrame(() => {
      if (!want) setSideCollapsed(localStorage.getItem("lc_side_collapsed") === "1");
      resizeLiveChart();
      if (want) fitLiveChart();
      else resizeLiveChart();
    });
  }
  $("#lc-load").addEventListener("click", () => loadSeries());
  $("#lc-fit")?.addEventListener("click", () => fitLiveChart());
  $("#lc-expand")?.addEventListener("click", () => setChartExpanded(!chartExpanded()));
  $("#lc-hover-show")?.addEventListener("change", () => {
    localStorage.setItem("lc_hover_visible", hoverBoxOn() ? "1" : "0");
    const el = $("#lc-hover");
    if (!hoverBoxOn()) {
      if (el) el.hidden = true;
      return;
    }
    if (hoverCorner !== "follow") applyHoverCorner();
    else if (lastSnap && lastSnap.point) placeHover(lastSnap.point);
  });
  $("#lc-side-collapse")?.addEventListener("click", () => setSideCollapsed(true));
  $("#lc-side-expand")?.addEventListener("click", () => setSideCollapsed(false));
  setSideCollapsed(localStorage.getItem("lc_side_collapsed") === "1");
  if (window._lcExpandEsc) window.removeEventListener("keydown", window._lcExpandEsc);
  window._lcExpandEsc = (ev) => {
    if (ev.key !== "Escape") return;
    if (!$("#lc-expand") || !chartExpanded()) return;
    if ($("#lc-note-prompt") && !$("#lc-note-prompt").hidden) return;
    if ($("#lc-lb-prompt") && !$("#lc-lb-prompt").hidden) return;
    if ($("#lc-mbo-order") && !$("#lc-mbo-order").hidden) return;
    ev.preventDefault();
    setChartExpanded(false);
  };
  window.addEventListener("keydown", window._lcExpandEsc);
  $("#lc-tf").addEventListener("change", () => { fitted = false; loadSeries(); });
  $("#lc-lb").addEventListener("change", () => {
    const v = $("#lc-lb").value;
    if (v === LB_CUSTOM) {
      $("#lc-lb").value = lastLb;
      openLookbackPrompt(lastLb && lastLb !== "session" ? lastLb : "");
      return;
    }
    lastLb = v;
    fitted = false;
    loadSeries();
  });
  $("#lc-lb-apply").addEventListener("click", () => applyLookbackPrompt());
  $("#lc-lb-cancel").addEventListener("click", () => closeLookbackPrompt());
  $("#lc-lb-prompt").addEventListener("click", (ev) => {
    if (ev.target === $("#lc-lb-prompt")) closeLookbackPrompt();
  });
  $("#lc-lb-prompt").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); applyLookbackPrompt(); }
    if (ev.key === "Escape") { ev.preventDefault(); closeLookbackPrompt(); }
  });
  for (const el of document.querySelectorAll('input[name="lc-lb-mode"]')) {
    el.addEventListener("change", () => setLbPromptMode(lbPromptMode()));
  }
  $("#lc-live").addEventListener("change", () => {
    localStorage.setItem("lc_live", liveOn() ? "1" : "0");
    armPoll();
    if (liveOn()) loadSeries();
  });

  function syncCursorButtons() {
    const cross = $("#lc-cursor-cross");
    const sel = $("#lc-cursor-select");
    if (cross) cross.classList.toggle("is-active", cursorMode === "crosshair");
    if (sel) sel.classList.toggle("is-active", cursorMode === "select");
    const layer = $("#lc-select-layer");
    if (layer) layer.hidden = cursorMode !== "select" && !noteSelection;
    const host = document.querySelector(".lc-chart-host");
    if (host) host.classList.toggle("lc-select-mode", cursorMode === "select");
    setChartPan(cursorMode !== "select");
    syncNoteCreateBtn();
  }
  function syncNoteCreateBtn() {
    const btn = $("#lc-note-create");
    if (!btn) return;
    btn.disabled = !(noteSelection && noteSelection.startUnix != null && noteSelection.endUnix != null
      && noteSelection.endUnix > noteSelection.startUnix);
  }
  function clearNoteSelection() {
    noteSelection = null;
    noteSelectDrag = null;
    const rect = $("#lc-select-rect");
    if (rect) rect.hidden = true;
    syncNoteCreateBtn();
    if (cursorMode !== "select") {
      const layer = $("#lc-select-layer");
      if (layer) layer.hidden = true;
    }
  }
  function paneSeries(index) {
    if (index === 0) return candleSeries;
    if (index === 1) return volSeries;
    for (const k of Object.keys(overlaySeries)) {
      if (seriesTagPane(k) === index && overlaySeries[k]) return overlaySeries[k];
    }
    if (cvdCandleSeries && oscPaneIndex("of_cvd") === index) return cvdCandleSeries;
    return null;
  }
  function chartHostY(clientY) {
    const host = $("#lc-chart");
    if (!host) return null;
    return clientY - host.getBoundingClientRect().top;
  }
  function paneIndexAtY(y) {
    if (y == null || !chart || typeof chart.panes !== "function") return null;
    const panes = chart.panes();
    for (let i = 0; i < panes.length; i++) {
      const b = paneBox(i);
      if (b.height > 0 && y >= b.top && y < b.top + b.height) return i;
    }
    return null;
  }
  function priceOnPane(paneIndex, yHost) {
    const series = paneSeries(paneIndex);
    const box = paneBox(paneIndex);
    if (!series || !box.height || typeof series.coordinateToPrice !== "function") return null;
    const y = Math.max(box.top, Math.min(box.top + box.height - 1, yHost));
    let px = null;
    try { px = series.coordinateToPrice(y - box.top); } catch { return null; }
    return (px != null && Number.isFinite(px)) ? px : null;
  }
  function paintNoteSelection() {
    const rect = $("#lc-select-rect");
    const layer = $("#lc-select-layer");
    if (!rect || !layer || !chart || !noteSelection) {
      if (rect) rect.hidden = true;
      return;
    }
    layer.hidden = false;
    const ts = chart.timeScale();
    let x0 = ts.timeToCoordinate(noteSelection.startPlot);
    let x1 = ts.timeToCoordinate(noteSelection.endPlot);
    if (x0 == null || x1 == null || !Number.isFinite(x0) || !Number.isFinite(x1)) {
      rect.hidden = true;
      return;
    }
    if (x1 < x0) { const tmp = x0; x0 = x1; x1 = tmp; }
    const leftPad = (typeof plotLeftPad === "function") ? plotLeftPad() : 0;
    const paneIndex = noteSelection.paneIndex == null ? 0 : noteSelection.paneIndex;
    const box = paneBox(paneIndex);
    const paneTop = box.top;
    const paneBot = box.top + box.height;
    let y0 = paneTop;
    let y1 = paneBot;
    const series = paneSeries(paneIndex);
    if (series && noteSelection.priceLo != null && noteSelection.priceHi != null
        && typeof series.priceToCoordinate === "function") {
      const cHi = series.priceToCoordinate(noteSelection.priceHi);
      const cLo = series.priceToCoordinate(noteSelection.priceLo);
      if (cHi != null && cLo != null && Number.isFinite(cHi) && Number.isFinite(cLo)) {
        y0 = paneTop + Math.min(cHi, cLo);
        y1 = paneTop + Math.max(cHi, cLo);
      }
    }
    y0 = Math.max(paneTop, Math.min(paneBot, y0));
    y1 = Math.max(paneTop, Math.min(paneBot, y1));
    rect.hidden = false;
    rect.style.left = (leftPad + x0) + "px";
    rect.style.width = Math.max(2, x1 - x0) + "px";
    rect.style.top = y0 + "px";
    rect.style.height = Math.max(2, y1 - y0) + "px";
    rect.style.bottom = "auto";
  }
  function clientXToPlotTime(clientX) {
    if (!chart) return null;
    const host = document.querySelector(".lc-chart-host") || $("#lc-chart");
    if (!host) return null;
    const r = host.getBoundingClientRect();
    const leftPad = (typeof plotLeftPad === "function") ? plotLeftPad() : 0;
    const x = clientX - r.left - leftPad;
    try {
      const t = chart.timeScale().coordinateToTime(x);
      return (t == null || !Number.isFinite(t)) ? null : t;
    } catch { return null; }
  }
  function setNoteSelectionFromPoints(startPt, endPt) {
    if (!startPt || !endPt || startPt.plot == null || endPt.plot == null) return;
    let a = startPt.plot, b = endPt.plot;
    if (b < a) { const tmp = a; a = b; b = tmp; }
    let startUnix = plotToReal(a);
    let endUnix = plotToReal(b);
    if (startUnix == null || endUnix == null) return;
    const paneIndex = startPt.paneIndex == null ? 0 : startPt.paneIndex;
    let priceLo = null, priceHi = null;
    if (Number.isFinite(startPt.price) && Number.isFinite(endPt.price)) {
      priceLo = Math.min(startPt.price, endPt.price);
      priceHi = Math.max(startPt.price, endPt.price);
    }
    const priceSpan = (priceLo != null && priceHi != null) ? priceHi - priceLo : 0;
    // A vertical drag inside one bar still has a price box. Widen it to that bar.
    // A click (no time span and no price span) does not select.
    if (endUnix <= startUnix) {
      if (!(priceSpan > 0)) return;
      const step = Math.max(barDurationSec(), 1);
      b = a + step;
      endUnix = startUnix + step;
    }
    noteSelection = {
      startUnix, endUnix, startPlot: a, endPlot: b,
      paneIndex, priceLo, priceHi,
    };
    paintNoteSelection();
    syncNoteCreateBtn();
  }
  function boxValueText(paneIndex, v) {
    if (!Number.isFinite(v)) return "";
    if (paneIndex === 0) return fmtNum(v, 2);
    const abs = Math.abs(v);
    if (abs >= 1000) return fmtNum(v, 0);
    if (abs >= 10) return fmtNum(v, 1);
    return fmtNum(v, 2);
  }
  function noteRangeLabel(sel) {
    if (!sel) return "—";
    const a = toCT(new Date(sel.startUnix * 1000).toISOString());
    const b = toCT(new Date(sel.endUnix * 1000).toISOString());
    const sym = (payload && payload.contract) || (meta && meta.contract) || "NQ";
    let label;
    if (!a || !b) label = `${sym} selection`;
    else {
      const ad = a.slice(0, 10), bd = b.slice(0, 10);
      const at = a.slice(11, 16), bt = b.slice(11, 16);
      label = ad === bd
        ? `${sym} ${ad} ${at}–${bt} ${TZ_ABBR}`
        : `${sym} ${ad} ${at}–${bd} ${bt} ${TZ_ABBR}`;
    }
    if (sel.priceLo != null && sel.priceHi != null && sel.priceHi > sel.priceLo) {
      const lo = boxValueText(sel.paneIndex, sel.priceLo);
      const hi = boxValueText(sel.paneIndex, sel.priceHi);
      const where = sel.paneIndex === 1 ? " vol" : "";
      if (lo && hi) label += ` ·${where} ${lo}–${hi}`;
    }
    return label;
  }
  function collectChartSnapshot() {
    const keys = selectedKeys();
    const origins = {};
    if (keys.includes("of_cvd")) origins.cvd = cvdOrigin();
    if (keys.includes("vw_vwap") || keys.includes("vw_vwap_dist") || vwapBandsOn()) origins.vwap = vwapOrigin();
    if (keys.includes("of_ofi_cum")) origins.ofi = ofiOrigin();
    if (keys.includes("ad_line")) origins.ad = adOrigin();
    const overlayParams = {};
    if (keys.includes("of_cvd")) overlayParams.cvd_style = cvdStyle();
    if (keys.includes("vw_vwap")) overlayParams.vwap_bands = vwapBandsOn();
    if (keys.includes("lt_prints")) {
      overlayParams.prints = {
        min_size: printMin(),
        opacity: printOpacity(),
        display_value: printDisplayValue(),
        extend_bars: printExtendBars(),
      };
    }
    return {
      timeframe: ($("#lc-tf") && $("#lc-tf").value) || (payload && payload.timeframe) || "1min",
      lookback: currentLookback(),
      overlays: keys.slice(),
      overlay_params: overlayParams,
      origins,
      emas: emaOn() ? emasForApi() : [],
      ema_enabled: emaOn(),
      stochs: stochOn() ? stochsForApi() : [],
      stoch_enabled: stochOn(),
      profiles: vpOn() ? profilesForApi() : [],
      vp_enabled: vpOn(),
      sar: keys.includes("ps_sar") ? sarForApi() : null,
      vwap_bands: vwapBandsOn(),
      display: {
        eth_display: ethDisplay(),
        eth_shade: ethShade(),
        eth_aggregate: ethAgg(),
        text_size: textSize(),
        replay: replayOn(),
      },
      chart_window: payload ? { start: payload.start, end: payload.end } : null,
      contract: (payload && payload.contract) || (meta && meta.contract) || null,
      instrument: (payload && payload.instrument) || "/NQ",
    };
  }
  function openNotePrompt() {
    if (!noteSelection) { toast("Select a range first", "err"); return; }
    const box = $("#lc-note-prompt");
    const metaEl = $("#lc-note-prompt-meta");
    const text = $("#lc-note-text");
    if (metaEl) metaEl.textContent = noteRangeLabel(noteSelection);
    if (text) text.value = "";
    if (box) box.hidden = false;
    if (text) setTimeout(() => text.focus(), 0);
  }
  function closeNotePrompt() {
    const box = $("#lc-note-prompt");
    if (box) box.hidden = true;
  }
  async function saveNoteFromPrompt() {
    if (!noteSelection) { toast("Select a range first", "err"); return; }
    const text = (($("#lc-note-text") && $("#lc-note-text").value) || "").trim();
    const name = noteRangeLabel(noteSelection);
    const startIso = new Date(noteSelection.startUnix * 1000).toISOString();
    const endIso = new Date(noteSelection.endUnix * 1000).toISOString();
    const body = {
      name,
      text,
      symbol: (payload && payload.contract) || (meta && meta.contract) || "NQ",
      instrument: (payload && payload.instrument) || "/NQ",
      range: {
        start: startIso,
        end: endIso,
        start_ct: toCT(startIso),
        end_ct: toCT(endIso),
        label: name,
        price_lo: noteSelection.priceLo,
        price_hi: noteSelection.priceHi,
        pane: noteSelection.paneIndex == null ? 0 : noteSelection.paneIndex,
      },
      chart: collectChartSnapshot(),
    };
    try {
      const rec = await post("/api/chart/notes", body);
      closeNotePrompt();
      toast(`Saved note · ${rec.name || name}`);
      setCursorMode("crosshair");
    } catch (e) {
      toast(e.message || String(e), "err");
    }
  }
  function setCursorMode(mode) {
    cursorMode = mode === "select" ? "select" : "crosshair";
    noteSelectDrag = null;
    if (cursorMode !== "select") clearNoteSelection();
    syncCursorButtons();
  }
  function zoomSelectionPrice(dir, sel) {
    if (!sel || sel.priceLo == null || sel.priceHi == null) return;
    const span = sel.priceHi - sel.priceLo;
    if (!(span > 0)) return;
    const series = paneSeries(sel.paneIndex == null ? 0 : sel.paneIndex);
    if (!series || typeof series.priceScale !== "function") return;
    const scale = series.priceScale();
    if (!scale || typeof scale.setVisibleRange !== "function") return;
    let vis = null;
    try { vis = typeof scale.getVisibleRange === "function" ? scale.getVisibleRange() : null; } catch {}
    const v0 = vis && Number.isFinite(Number(vis.from)) ? Number(vis.from) : null;
    const v1 = vis && Number.isFinite(Number(vis.to)) ? Number(vis.to) : null;
    const mid = (sel.priceLo + sel.priceHi) / 2;
    let from, to;
    if (dir > 0 && v0 != null && v1 != null && (v1 - v0) > span * 1.08) {
      const pad = span * 0.04;
      from = sel.priceLo - pad;
      to = sel.priceHi + pad;
    } else {
      const base = (v0 != null && v1 != null)
        ? (dir > 0 ? Math.min(v1 - v0, span) : Math.max(v1 - v0, span))
        : span;
      const next = Math.max(span * 0.02, base * (dir > 0 ? 0.5 : 2));
      from = mid - next / 2;
      to = mid + next / 2;
    }
    if (!(to > from)) return;
    try { scale.setVisibleRange({ from, to }); } catch {}
  }
  function zoomChart(dir) {
    if (!chart) return;
    const ts = chart.timeScale();
    const sel = (noteSelection && noteSelection.endPlot > noteSelection.startPlot) ? noteSelection : null;
    const minSpan = Math.max(barDurationSec() * 3, 1);
    if (sel && typeof ts.setVisibleRange === "function") {
      const s0 = sel.startPlot, s1 = sel.endPlot;
      let vis = null;
      try { vis = ts.getVisibleRange(); } catch {}
      const v0 = vis && Number.isFinite(Number(vis.from)) ? Number(vis.from) : null;
      const v1 = vis && Number.isFinite(Number(vis.to)) ? Number(vis.to) : null;
      const selSpan = s1 - s0;
      let from, to;
      if (dir > 0 && v0 != null && v1 != null && (v1 - v0) > selSpan * 1.08) {
        const pad = Math.max(selSpan * 0.04, barDurationSec());
        from = s0 - pad;
        to = s1 + pad;
      } else {
        const mid = (s0 + s1) / 2;
        const base = (v0 != null && v1 != null)
          ? (dir > 0 ? Math.min(v1 - v0, selSpan) : Math.max(v1 - v0, selSpan))
          : selSpan;
        const span = Math.max(minSpan, base * (dir > 0 ? 0.5 : 2));
        from = mid - span / 2;
        to = mid + span / 2;
      }
      try {
        ts.setVisibleRange({ from, to });
        zoomSelectionPrice(dir, sel);
        paintNoteSelection();
        scheduleDrawProfiles();
        return;
      } catch {}
    }
    if (sel) zoomSelectionPrice(dir, sel);
    let lr = null;
    try { lr = ts.getVisibleLogicalRange(); } catch {}
    if (!lr || !(lr.to > lr.from)) return;
    const mid = (lr.from + lr.to) / 2;
    const span = Math.max(4, (lr.to - lr.from) * (dir > 0 ? 0.5 : 2));
    try { ts.setVisibleLogicalRange({ from: mid - span / 2, to: mid + span / 2 }); } catch {}
    scheduleDrawProfiles();
  }
  function armSelectGestures() {
    const host = document.querySelector(".lc-chart-host");
    if (!host || host.dataset.selectArmed) return;
    host.dataset.selectArmed = "1";
    const pointAt = (ev, paneIndex) => {
      const plot = clientXToPlotTime(ev.clientX);
      if (plot == null) return null;
      const y = chartHostY(ev.clientY);
      const pane = paneIndex == null ? paneIndexAtY(y) : paneIndex;
      if (pane == null) return null;
      const price = priceOnPane(pane, y);
      if (price == null) return null;
      return { plot, price, paneIndex: pane };
    };
    const onDown = (ev) => {
      if (cursorMode !== "select") return;
      if (ev.button != null && ev.button !== 0) return;
      const pt = pointAt(ev, null);
      if (!pt) return;
      ev.preventDefault();
      noteSelectDrag = { ...pt, pointerId: ev.pointerId };
      try { host.setPointerCapture(ev.pointerId); } catch {}
      setNoteSelectionFromPoints(pt, pt);
    };
    const onMove = (ev) => {
      if (noteSelection && !noteSelectDrag) paintNoteSelection();
      if (!noteSelectDrag || cursorMode !== "select") return;
      const pt = pointAt(ev, noteSelectDrag.paneIndex);
      if (!pt) return;
      setNoteSelectionFromPoints(noteSelectDrag, pt);
    };
    const onUp = (ev) => {
      if (!noteSelectDrag) return;
      const pt = pointAt(ev, noteSelectDrag.paneIndex);
      if (pt) setNoteSelectionFromPoints(noteSelectDrag, pt);
      noteSelectDrag = null;
    };
    host.addEventListener("pointerdown", onDown);
    host.addEventListener("pointermove", onMove);
    host.addEventListener("pointerup", onUp);
    host.addEventListener("pointercancel", onUp);
    host.addEventListener("wheel", () => {
      if (noteSelection) paintNoteSelection();
    }, { passive: true });
  }
  $("#lc-cursor-cross")?.addEventListener("click", () => setCursorMode("crosshair"));
  $("#lc-cursor-select")?.addEventListener("click", () => setCursorMode("select"));
  $("#lc-zoom-in")?.addEventListener("click", () => zoomChart(1));
  $("#lc-zoom-out")?.addEventListener("click", () => zoomChart(-1));
  $("#lc-note-create")?.addEventListener("click", () => openNotePrompt());
  $("#lc-note-cancel")?.addEventListener("click", () => closeNotePrompt());
  $("#lc-note-save")?.addEventListener("click", () => saveNoteFromPrompt());
  $("#lc-note-prompt")?.addEventListener("click", (ev) => {
    if (ev.target === $("#lc-note-prompt")) closeNotePrompt();
  });
  $("#lc-note-prompt")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { ev.preventDefault(); closeNotePrompt(); }
    if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey)) {
      ev.preventDefault();
      saveNoteFromPrompt();
    }
  });
  syncCursorButtons();
  armSelectGestures();

  $("#lc-ov").addEventListener("click", (ev) => {
    const kdePreset = ev.target.closest("[data-kde-preset]");
    if (kdePreset) {
      ev.preventDefault();
      const card = kdePreset.closest(".lc-vp-card");
      if (card) {
        const set = (field, value) => {
          const input = card.querySelector(`[data-f="${field}"]`);
          if (input) input.value = String(value);
        };
        set("hvn_method", "adaptive_kde");
        set("kde_bandwidth", 1);
        set("kde_prominence", 2);
        set("kde_valley_depth", 2);
        set("kde_volume", 1);
        set("kde_separation", 2);
        set("kde_valley_width", 2);
        set("kde_max_zones", 10);
        const kde = card.querySelector(".lc-vp-kde");
        if (kde) kde.hidden = false;
        syncVpDraftFromDom();
      }
      return;
    }
    const widthStep = ev.target.closest("[data-vp-width-step]");
    if (widthStep) {
      ev.preventDefault();
      const card = widthStep.closest(".lc-vp-card");
      const input = card && card.querySelector('[data-f="hist_width"]');
      const step = Number(widthStep.dataset.vpWidthStep) || 0;
      const current = Number(input && input.value);
      const next = Math.max(0.01, Math.round(((Number.isFinite(current) ? current : .25) + step * .05) * 100) / 100);
      if (input) input.value = String(next);
      vpDraftSpecs = readVpSpecsFromDom();
      return;
    }
    const refreshVp = ev.target.closest("#lc-vp-refresh");
    if (refreshVp) {
      ev.preventDefault();
      const active = document.activeElement;
      if (active && active.closest && active.closest(".lc-vp-card")) active.blur();
      syncVpDraftFromDom();
      if ($("#lc-vp") && $("#lc-vp").checked && !vpDraftSpecs.length) {
        vpDraftSpecs = [defaultVpSpec("current")];
      }
      vpSpecs = vpDraftSpecs.map(spec => Object.assign({}, spec));
      vpEnabled = !!($("#lc-vp") && $("#lc-vp").checked);
      persistVpSpecs();
      paintVpCards();
      loadSeries();
      return;
    }
    const gear = ev.target.closest(".lc-ov-gear");
    if (gear) {
      ev.preventDefault();
      ev.stopPropagation();
      const panel = document.getElementById(gear.dataset.panel);
      const open = panel && !panel.hidden;
      if (panel) panel.hidden = !!open;
      gear.setAttribute("aria-expanded", open ? "false" : "true");
      gear.textContent = open ? "▸" : "▾";
      const key = gear.closest(".lc-ov-row") && gear.closest(".lc-ov-row").dataset.key;
      if (key) {
        if (open) expandedParams.delete(key);
        else expandedParams.add(key);
      }
      return;
    }
    const add = ev.target.closest("#lc-vp-add");
    if (add) {
      ev.preventDefault();
      syncVpDraftFromDom();
      if (vpDraftSpecs.length >= 8) {
        toast("At most 8 volume profiles", "err");
        return;
      }
      const kindEl = $("#lc-vp-add-kind");
      const kind = (kindEl && kindEl.value) || "current";
      vpDraftSpecs.push(defaultVpSpec(kind));
      if ($("#lc-vp")) $("#lc-vp").checked = true;
      paintVpCards();
      return;
    }
    const collapseVp = ev.target.closest("[data-vp-collapse]");
    if (collapseVp) {
      ev.preventDefault();
      ev.stopPropagation();
      const id = collapseVp.getAttribute("data-vp-collapse");
      const card = collapseVp.closest(".lc-vp-card");
      if (!id || !card) return;
      const next = !card.classList.contains("is-collapsed");
      setVpCardCollapsed(id, next);
      card.classList.toggle("is-collapsed", next);
      collapseVp.setAttribute("aria-expanded", next ? "false" : "true");
      collapseVp.textContent = next ? "▸" : "▾";
      collapseVp.title = next ? "Expand" : "Collapse";
      collapseVp.setAttribute("aria-label", (next ? "Expand" : "Collapse") + " volume profile card");
      return;
    }
    const rm = ev.target.closest("[data-vp-remove]");
    if (rm) {
      ev.preventDefault();
      const id = rm.getAttribute("data-vp-remove");
      vpDraftSpecs = assignVpNs(readVpSpecsFromDom().filter(s => s.id !== id));
      paintVpCards();
      return;
    }
    const refreshEma = ev.target.closest("#lc-ema-refresh");
    if (refreshEma) {
      ev.preventDefault();
      const active = document.activeElement;
      if (active && active.closest && active.closest(".lc-ema-card")) active.blur();
      syncEmaDraftFromDom();
      if ($("#lc-ema") && $("#lc-ema").checked && !emaDraftSpecs.length) {
        emaDraftSpecs = [defaultEmaSpec(emaLookbackDefault())];
      }
      emaSpecs = emaDraftSpecs.map(spec => Object.assign({}, spec));
      emaEnabled = !!($("#lc-ema") && $("#lc-ema").checked);
      persistEmaSpecs();
      paintEmaCards();
      loadSeries();
      return;
    }
    const addEma = ev.target.closest("#lc-ema-add");
    if (addEma) {
      ev.preventDefault();
      syncEmaDraftFromDom();
      if (emaDraftSpecs.length >= emaMaxSpecs()) {
        toast(`At most ${emaMaxSpecs()} EMAs`, "err");
        return;
      }
      const kindEl = $("#lc-ema-add-kind");
      let lookback = emaLookbackDefault();
      const raw = kindEl && kindEl.value;
      if (raw === "custom") {
        const typed = window.prompt(
          `EMA lookback bars (${emaLookbackMin()}–${emaLookbackMax()})`,
          String(emaLookbackDefault()),
        );
        if (typed == null) return;
        lookback = clampEmaLookback(typed);
      } else if (raw != null && raw !== "") {
        lookback = clampEmaLookback(raw);
      }
      emaDraftSpecs.push(defaultEmaSpec(lookback));
      if ($("#lc-ema")) $("#lc-ema").checked = true;
      emaEnabled = true;
      paintEmaCards();
      return;
    }
    const rmEma = ev.target.closest("[data-ema-remove]");
    if (rmEma) {
      ev.preventDefault();
      const id = rmEma.getAttribute("data-ema-remove");
      emaDraftSpecs = assignEmaNs(readEmaSpecsFromDom().filter(s => s.id !== id));
      paintEmaCards();
      return;
    }
    const refreshStoch = ev.target.closest("#lc-stoch-refresh");
    if (refreshStoch) {
      ev.preventDefault();
      const active = document.activeElement;
      if (active && active.closest && active.closest(".lc-stoch-card")) active.blur();
      syncStochDraftFromDom();
      if ($("#lc-stoch") && $("#lc-stoch").checked && !stochDraftSpecs.length) {
        stochDraftSpecs = [defaultStochSpec()];
      }
      stochSpecs = stochDraftSpecs.map(spec => Object.assign({}, spec));
      stochEnabled = !!($("#lc-stoch") && $("#lc-stoch").checked);
      persistStochSpecs();
      paintStochCards();
      loadSeries();
      return;
    }
    const addStoch = ev.target.closest("#lc-stoch-add");
    if (addStoch) {
      ev.preventDefault();
      syncStochDraftFromDom();
      const maxN = (meta && meta.stoch_max) || 8;
      if (stochDraftSpecs.length >= maxN) {
        toast(`At most ${maxN} stochastics`, "err");
        return;
      }
      const kindEl = $("#lc-stoch-add-kind");
      let k = stochKDefault(), d = stochDDefault();
      const raw = kindEl && kindEl.value;
      if (raw === "custom") {
        const typedK = window.prompt(
          `Stoch K period (${stochKMin()}–${stochKMax()})`,
          String(stochKDefault()),
        );
        if (typedK == null) return;
        const typedD = window.prompt(
          `Stoch D smooth (${stochDMin()}–${stochDMax()})`,
          String(stochDDefault()),
        );
        if (typedD == null) return;
        k = clampStochK(typedK);
        d = clampStochD(typedD);
      } else if (raw && raw.includes(",")) {
        const parts = raw.split(",");
        k = clampStochK(parts[0]);
        d = clampStochD(parts[1]);
      }
      if (stochDraftSpecs.some(s => clampStochK(s.k_period) === k && clampStochD(s.d_period) === d)) {
        toast(`Fast %D ${k},${d} already added`, "err");
        return;
      }
      stochDraftSpecs.push(defaultStochSpec(k, d));
      if ($("#lc-stoch")) $("#lc-stoch").checked = true;
      stochEnabled = true;
      paintStochCards();
      return;
    }
    const rmStoch = ev.target.closest("[data-stoch-remove]");
    if (rmStoch) {
      ev.preventDefault();
      const id = rmStoch.getAttribute("data-stoch-remove");
      stochDraftSpecs = assignStochNs(readStochSpecsFromDom().filter(s => s.id !== id));
      paintStochCards();
      return;
    }
    const refreshSar = ev.target.closest("#lc-sar-refresh");
    if (refreshSar) {
      ev.preventDefault();
      const active = document.activeElement;
      if (active && active.closest && active.closest(".lc-sar-card")) active.blur();
      syncSarDraftFromDom();
      sarParams = normalizeSarParams(sarDraftParams);
      const box = document.querySelector('#lc-ov input[data-key="ps_sar"]');
      if (box && !box.checked) box.checked = true;
      persistSarParams();
      paintSarCard();
      loadSeries();
    }
  });
  $("#lc-ov").addEventListener("input", (ev) => {
    const input = ev.target;
    if (input && (input.id === "lc-print-opacity" || input.id === "lc-print-extend-bars")) {
      syncPrintDisplayControls();
      localStorage.setItem("lc_print_opacity", String(Math.round(printOpacity() * 100)));
      localStorage.setItem("lc_print_extend_bars", String(printExtendBars()));
      scheduleDrawProfiles();
      return;
    }
    if (input && input.dataset && input.dataset.f === "hist_width"
      && input.closest && input.closest(".lc-vp-card")) {
      syncVpDraftFromDom();
    }
  });
  $("#lc-ov").addEventListener("change", (ev) => {
    const t = ev.target;
    if (t && (t.id === "lc-print-opacity" || t.id === "lc-print-display-value" || t.id === "lc-print-extend-bars")) {
      syncPrintDisplayControls();
      localStorage.setItem("lc_print_opacity", String(Math.round(printOpacity() * 100)));
      localStorage.setItem("lc_print_display_value", printDisplayValue() ? "1" : "0");
      localStorage.setItem("lc_print_extend_bars", String(printExtendBars()));
      scheduleDrawProfiles();
      return;
    }
    if (t && t.id === "lc-cvd-style") {
      localStorage.setItem("lc_cvd_style", cvdStyle());
      if (payload && chart) syncOverlaySeries(payload);
      return;
    }
    if (t && t.closest && t.closest(".lc-vp-card")) {
      const card = t.closest(".lc-vp-card");
      if (t.dataset && t.dataset.f === "placement") {
        card.dataset.placement = t.value;
      }
      if (t.dataset && t.dataset.f === "session_tf") {
        card.dataset.sessionTf = t.value;
      }
      if (t.dataset && t.dataset.f === "hist_style") {
        card.dataset.hist = t.value === "lvn_nodes" ? "binned" : t.value;
      }
      if (t.dataset && t.dataset.f === "show_lvn") {
        card.dataset.lvn = t.checked ? "1" : "0";
      }
      if (t.dataset && t.dataset.f === "enabled") {
        card.classList.toggle("is-disabled", !t.checked);
        card.dataset.enabled = t.checked ? "1" : "0";
      }
      if (t.dataset && t.dataset.f === "hvn_method") {
        const kde = card.querySelector(".lc-vp-kde");
        if (kde) kde.hidden = t.value !== "adaptive_kde";
      }
      syncVpDraftFromDom();
      return;
    }
    if (t && t.closest && t.closest(".lc-ema-card")) {
      syncEmaDraftFromDom();
      return;
    }
    if (t && t.closest && t.closest(".lc-stoch-card")) {
      syncStochDraftFromDom();
      return;
    }
    if (t && t.closest && t.closest(".lc-sar-card")) {
      syncSarDraftFromDom();
      return;
    }
    if (t && t.id === "lc-vp") {
      if (t.checked && !vpDraftSpecs.length) {
        vpDraftSpecs = [defaultVpSpec("current")];
        paintVpCards();
      }
      return;
    }
    if (t && t.id === "lc-ema") {
      if (t.checked && !emaDraftSpecs.length) {
        emaDraftSpecs = [defaultEmaSpec(emaLookbackDefault())];
        paintEmaCards();
      }
      return;
    }
    if (t && t.id === "lc-stoch") {
      if (t.checked && !stochDraftSpecs.length) {
        stochDraftSpecs = [defaultStochSpec()];
        paintStochCards();
      }
      return;
    }
    if (t && (t.id === "lc-cvd-origin" || t.id === "lc-vwap-origin" || t.id === "lc-ofi-origin" || t.id === "lc-ad-origin")) {
      localStorage.setItem("lc_cvd_origin", cvdOrigin());
      localStorage.setItem("lc_vwap_origin", vwapOrigin());
      localStorage.setItem("lc_ofi_origin", ofiOrigin());
      localStorage.setItem("lc_ad_origin", adOrigin());
    }
    if (t && t.id === "lc-vwap-bb") {
      localStorage.setItem("lc_vwap_bb", t.checked ? "1" : "0");
    }
    if (t && (t.id === "lc-mfi-upper" || t.id === "lc-mfi-lower" || t.id === "lc-mfi-arrows")) {
      const b = lcNormalizeOscBounds(
        $("#lc-mfi-upper") && $("#lc-mfi-upper").value,
        $("#lc-mfi-lower") && $("#lc-mfi-lower").value,
      );
      localStorage.setItem("lc_mfi_upper", String(b.upper));
      localStorage.setItem("lc_mfi_lower", String(b.lower));
      const arrowsOn = !!($("#lc-mfi-arrows") && $("#lc-mfi-arrows").checked);
      localStorage.setItem("lc_mfi_arrows", arrowsOn ? "1" : "0");
      if ($("#lc-mfi-upper")) $("#lc-mfi-upper").value = String(b.upper);
      if ($("#lc-mfi-lower")) $("#lc-mfi-lower").value = String(b.lower);
      refreshBoundLines();
      scheduleDrawProfiles();
      return;
    }
    if (t && t.id === "lc-vwap-bb-arrows") {
      localStorage.setItem("lc_vwap_bb_arrows", t.checked ? "1" : "0");
      scheduleDrawProfiles();
      return;
    }
    loadSeries();
  });
  function applyVisibleBars(fit) {
    if (!payload || !chart || !candleSeries) return;
    lastMaps = barMaps(payload);
    if (replayOn()) {
      setReplayUnix(replayUnix == null ? lastMaps.real[lastMaps.real.length - 1] : replayUnix);
    } else {
      candleSeries.setData(candleData(payload));
      volSeries.setData(volData(payload));
      lwForcePane(candleSeries, 0);
      lwForcePane(volSeries, 1);
      syncOverlaySeries(payload);
      scheduleDrawProfiles();
    }
    if (fit) {
      try { chart.timeScale().fitContent(); } catch {}
      fitted = true;
    }
  }
  $("#lc-display").addEventListener("change", (ev) => {
    const t = ev.target;
    if (!t) return;
    if (t.id === "lc-eth-display") {
      localStorage.setItem("lc_eth_display", ethDisplay() ? "1" : "0");
      applyVisibleBars(true);
      return;
    }
    if (t.id === "lc-eth-shade") {
      localStorage.setItem("lc_eth_shade", ethShade() ? "1" : "0");
      scheduleDrawProfiles();
      return;
    }
    if (t.id === "lc-eth-agg") {
      localStorage.setItem("lc_eth_agg", ethAgg() ? "1" : "0");
      loadSeries();
      return;
    }
    if (t.id === "lc-text-size") applyTextSize();
    if (t.id === "lc-chart-label") applyChartLabelSize();
    if (t.id === "lc-replay") enableReplay(replayOn());
  });
  $("#lc-display").addEventListener("click", (ev) => {
    const collapse = ev.target.closest("#lc-replay-collapse");
    if (collapse) {
      ev.preventDefault();
      const next = localStorage.getItem("lc_replay_box_collapsed") !== "1";
      localStorage.setItem("lc_replay_box_collapsed", next ? "1" : "0");
      applyReplayBoxCollapsed();
      return;
    }
    if (ev.target.closest("#lc-replay-play")) {
      ev.preventDefault();
      if (!replayOn()) {
        enableReplay(true);
      }
      if (replayPlaying) {
        stopReplayPlay();
        scheduleReplayProfiles();
      } else {
        startReplayPlay();
      }
      return;
    }
    if (ev.target.closest("#lc-replay-step")) {
      ev.preventDefault();
      if (!replayOn()) enableReplay(true);
      stopReplayPlay();
      stepReplay(1);
      scheduleReplayProfiles();
      return;
    }
    const speedBtn = ev.target.closest("[data-replay-speed]");
    if (speedBtn) {
      ev.preventDefault();
      setReplaySpeed(speedBtn.getAttribute("data-replay-speed"));
    }
  });
  $("#lc-display").addEventListener("input", (ev) => {
    if (ev.target && ev.target.id === "lc-text-size") applyTextSize();
    if (ev.target && ev.target.id === "lc-chart-label") applyChartLabelSize();
    if (ev.target && ev.target.id === "lc-replay-slider" && replayOn()) {
      ev.stopPropagation();
      stopReplayPlay();
      const maps = lastMaps;
      if (!maps || !maps.real.length) return;
      const j = Math.max(0, Math.min(maps.real.length - 1, Number(ev.target.value) || 0));
      setReplayUnix(maps.real[j], true);
    }
  });
  applyReplayBoxCollapsed();
  syncReplaySpeedButtons();
  syncReplayPlayButton();

  const ro = new ResizeObserver(() => {
    if (!still() || !chart) return;
    const host = $("#lc-chart");
    if (host) chart.applyOptions({ width: host.clientWidth, height: host.clientHeight });
    applyPaneHeights(true);
    drawProfiles();
    if (!replayDragging) positionReplayLayer();
    if (hoverCorner !== "follow") applyHoverCorner();
  });
  ro.observe(document.querySelector(".lc-chart-host") || $("#lc-chart"));
  armHoverDrag();
  armPaneStretchSave();
  armReplayDrag();

  /* ── Model design bridge (ADR-117) ─────────────────────────────────
     The editor lives in a separate window (#/model-design). This tab
     just listens on a BroadcastChannel and paints the markers +
     entry/target/stop levels the popup emits after each Run. */
  const md = {
    markersCtrl: null,
    tradeLines: [],
    lastResult: null,
    channel: null,
  };

  function mdClearTradeLines() {
    if (!candleSeries) { md.tradeLines = []; return; }
    while (md.tradeLines.length) {
      const line = md.tradeLines.pop();
      try { candleSeries.removePriceLine(line); } catch {}
    }
  }
  function mdClearMarkers() {
    if (candleSeries && lwLib) {
      if (md.markersCtrl && typeof md.markersCtrl.setMarkers === "function") {
        md.markersCtrl.setMarkers([]);
      } else {
        try { lwSetMarkers(lwLib, candleSeries, []); } catch {}
      }
    }
    mdClearTradeLines();
  }
  function mdApplyMarkersFromResult(res) {
    if (!res || !candleSeries || !lwLib) return;
    const markers = (res.markers || []).map(m => {
      const isEntry = m.type.startsWith("entry_");
      const long = m.side === "long";
      const color = isEntry ? (long ? "#22d3ee" : "#f472b6")
                            : (m.type === "exit_target" ? "#a3e635"
                              : m.type === "exit_stop"   ? "#f87171" : "#eab308");
      const shape = isEntry ? (long ? "arrowUp" : "arrowDown")
                            : (m.type === "exit_target" ? "arrowUp"
                              : m.type === "exit_stop"   ? "arrowDown" : "square");
      return {
        time: m.time,
        position: isEntry ? (long ? "belowBar" : "aboveBar")
                          : (long ? "aboveBar" : "belowBar"),
        color, shape,
        text: `${m.text || ""} ${Number(m.price).toFixed(2)}`,
      };
    });
    md.markersCtrl = lwSetMarkers(lwLib, candleSeries, markers);
  }
  function mdShowTradeLines(idx) {
    mdClearTradeLines();
    if (!candleSeries || !md.lastResult || !md.lastResult.trades[idx]) return;
    const t = md.lastResult.trades[idx];
    const add = (price, color, title, dashed) => {
      if (price == null || !Number.isFinite(price)) return;
      try {
        const line = candleSeries.createPriceLine({
          price: Number(price), color, lineWidth: 1,
          lineStyle: dashed ? 2 : 0, axisLabelVisible: true, title,
        });
        md.tradeLines.push(line);
      } catch {}
    };
    add(t.entry_price, "#e2e8f0", `${t.side.toUpperCase()} @ ${Number(t.entry_price).toFixed(2)}`, false);
    add(t.target_level, "#a3e635", "target", true);
    add(t.stop_level, "#f87171", "stop", true);
  }

  try {
    md.channel = new BroadcastChannel("vtech-md");
    md.channel.onmessage = (ev) => {
      const msg = ev.data || {};
      if (msg.type === "run-result") {
        md.lastResult = msg.result || null;
        mdApplyMarkersFromResult(md.lastResult);
        setStatus(md.lastResult
          ? `Model: ${md.lastResult.trades.length} trades · ${(md.lastResult.pnl.gross_points || 0).toFixed(2)} pts`
          : "");
      } else if (msg.type === "clear-markers") {
        md.lastResult = null;
        mdClearMarkers();
      } else if (msg.type === "select-trade" && Number.isFinite(msg.idx)) {
        mdShowTradeLines(msg.idx);
      } else if (msg.type === "request-chart-req") {
        md.channel.postMessage({ type: "chart-req", req: mdBuildChartReq() });
      }
    };
  } catch {}

  function mdBuildChartReq() {
    const tf = $("#lc-tf").value || "1min";
    const lb = currentLookback();
    const keys = selectedKeys();
    return {
      timeframe: tf, lookback: lb,
      overlays: keys.length ? keys.join(",") : null,
      cvd_origin: cvdOrigin(),
      vwap_origin: keys.includes("vw_vwap") || keys.includes("vw_vwap_dist") || vwapBandsOn() ? vwapOrigin() : "session",
      ofi_origin: keys.includes("of_ofi_cum") ? ofiOrigin() : null,
      ad_origin: keys.includes("ad_line") ? adOrigin() : null,
      profiles: vpOn() && vpSpecs.length ? JSON.stringify(profilesForApi()) : null,
      emas: emaOn() && emaSpecs.length ? JSON.stringify(emasForApi()) : null,
      stochs: stochOn() && stochSpecs.length ? JSON.stringify(stochsForApi()) : null,
      sar: keys.includes("ps_sar") ? JSON.stringify(sarForApi()) : null,
      vwap_bands: vwapBandsOn(),
      eth_aggregate: ethAgg(),
      print_min: printMin(),
    };
  }

  function mdOpenPopup() {
    try {
      const req = mdBuildChartReq();
      localStorage.setItem("lc_md_chart_req", JSON.stringify(req));
    } catch {}
    const url = "#/model-design";
    window.open(url, "vtech-model-design", "popup=1,width=1200,height=820,resizable=1");
  }

  function mdOnPayload(_p) {
    // Live-Chart-side stays passive; the popup rebuilds its own catalog.
  }
  window._lc_md = { onPayload: mdOnPayload, openPopup: mdOpenPopup };

  try {
    meta = await api("/api/chart/meta", { timeoutMs: 15000 });
    if (!still()) return;
    const tfs = meta.timeframes || ["1min"];
    $("#lc-tf").innerHTML = tfs.map(t =>
      `<option value="${esc(t)}">${esc(TF_LABEL[t] || t)}</option>`).join("");
    $("#lc-tf").value = localStorage.getItem("lc_tf") || (meta.defaults && meta.defaults.timeframe) || "1min";
    lastLb = parseLookbackToken(localStorage.getItem("lc_lb") || "")
      || (meta.defaults && meta.defaults.lookback) || "session";
    paintLookbacks(lastLb);
    const liveStored = localStorage.getItem("lc_live");
    $("#lc-live").checked = liveStored == null ? true : liveStored === "1";
    const hoverShow = $("#lc-hover-show");
    if (hoverShow) hoverShow.checked = localStorage.getItem("lc_hover_visible") !== "0";
    $("#lc-contract").textContent = meta.contract || "lead";
    vpSpecs = loadVpSpecs();
    vpDraftSpecs = vpSpecs.map(spec => Object.assign({}, spec));
    emaSpecs = loadEmaSpecs();
    emaDraftSpecs = emaSpecs.map(spec => Object.assign({}, spec));
    stochSpecs = loadStochSpecs();
    stochDraftSpecs = stochSpecs.map(spec => Object.assign({}, spec));
    if (stochSpecs.length) {
      // Drop retired catalog key from overlay persistence after migration.
      const ov = (localStorage.getItem("lc_overlays") || "").split(",").filter(k => k && k !== "so_fast_d_9_3");
      localStorage.setItem("lc_overlays", ov.join(","));
      if (localStorage.getItem("lc_stoch") == null) localStorage.setItem("lc_stoch", "1");
    }
    sarParams = loadSarParams();
    sarDraftParams = Object.assign({}, sarParams);
    paintOverlaysCatalog(meta);
    if ($("#lc-cvd-origin")) {
      $("#lc-cvd-origin").value = localStorage.getItem("lc_cvd_origin")
        || (meta.defaults && meta.defaults.cvd_origin) || "session";
    }
    if ($("#lc-vwap-origin")) {
      $("#lc-vwap-origin").value = localStorage.getItem("lc_vwap_origin")
        || (meta.defaults && meta.defaults.vwap_origin) || "session";
    }
    if ($("#lc-ofi-origin")) {
      $("#lc-ofi-origin").value = localStorage.getItem("lc_ofi_origin")
        || localStorage.getItem("lc_cvd_origin")
        || (meta.defaults && meta.defaults.ofi_origin) || "session";
    }
    if ($("#lc-ad-origin")) {
      $("#lc-ad-origin").value = localStorage.getItem("lc_ad_origin")
        || localStorage.getItem("lc_cvd_origin")
        || (meta.defaults && meta.defaults.ad_origin) || "session";
    }
    if ($("#lc-cvd-style")) {
      $("#lc-cvd-style").value = localStorage.getItem("lc_cvd_style")
        || (meta.defaults && meta.defaults.cvd_style) || "line";
    }
    if ($("#lc-print-min")) {
      const n = Number(localStorage.getItem("lc_print_min"));
      const def = (meta.defaults && meta.defaults.print_min) || 10;
      $("#lc-print-min").value = String(Number.isFinite(n) && n >= 1 ? Math.floor(n) : def);
    }
    if ($("#lc-print-opacity")) {
      const storedOpacity = localStorage.getItem("lc_print_opacity");
      const opacity = Number(storedOpacity);
      $("#lc-print-opacity").value = String(storedOpacity != null && Number.isFinite(opacity)
        ? Math.min(100, Math.max(0, Math.round(opacity))) : 88);
      $("#lc-print-display-value").checked = localStorage.getItem("lc_print_display_value") !== "0";
      const extendBars = Number(localStorage.getItem("lc_print_extend_bars"));
      $("#lc-print-extend-bars").value = String(Number.isFinite(extendBars) ? Math.min(5000, Math.max(0, Math.floor(extendBars))) : 0);
      syncPrintDisplayControls();
    }
    if ($("#lc-vp")) {
      const storedOn = localStorage.getItem("lc_vp");
      $("#lc-vp").checked = storedOn == null ? vpSpecs.length > 0 : storedOn === "1";
      vpEnabled = $("#lc-vp").checked;
    }
    if ($("#lc-ema")) {
      const storedOn = localStorage.getItem("lc_ema");
      $("#lc-ema").checked = storedOn == null ? emaSpecs.length > 0 : storedOn === "1";
      emaEnabled = $("#lc-ema").checked;
    }
    if ($("#lc-stoch")) {
      const storedOn = localStorage.getItem("lc_stoch");
      $("#lc-stoch").checked = storedOn == null ? stochSpecs.length > 0 : storedOn === "1";
      stochEnabled = $("#lc-stoch").checked;
    }
    if ($("#lc-eth-display")) {
      const v = localStorage.getItem("lc_eth_display");
      $("#lc-eth-display").checked = v == null ? true : v === "1";
    }
    if ($("#lc-eth-shade")) {
      $("#lc-eth-shade").checked = localStorage.getItem("lc_eth_shade") === "1";
    }
    if ($("#lc-eth-agg")) {
      const v = localStorage.getItem("lc_eth_agg");
      $("#lc-eth-agg").checked = v == null ? true : v === "1";
    }
    if ($("#lc-text-size")) {
      const ts = Number(localStorage.getItem("lc_text_size"));
      if (Number.isFinite(ts)) $("#lc-text-size").value = String(Math.min(18, Math.max(10, ts)));
    }
    if ($("#lc-chart-label")) {
      const raw = localStorage.getItem("lc_chart_label");
      const cl = Number(raw);
      if (raw != null && Number.isFinite(cl)) {
        $("#lc-chart-label").value = String(Math.min(48, Math.max(10, Math.round(cl))));
      }
    }
    if ($("#lc-replay") && localStorage.getItem("lc_replay") === "1") {
      $("#lc-replay").checked = true;
      if ($("#lc-live")) $("#lc-live").checked = false;
    }
    applyTextSize();
    applyChartLabelSize();
    await loadSeries();
    armPoll();
    if (localStorage.getItem("lc_expanded") === "1") setChartExpanded(true);
    else setChartExpanded(false);
    const btnOpen = $("#lc-md-open");
    if (btnOpen) btnOpen.addEventListener("click", mdOpenPopup);
  } catch (e) {
    if (!still()) return;
    setEmpty(e.message || String(e));
  }
}

async function viewBorderVisualizer() {
  return viewLiveChart();
}


/* ── Model design (chart-scoped rule tree — ADR-117) ──────────────────── */
async function viewModelDesign() {
  view.classList.add("wide");
  const shell = document.getElementById("shell");
  if (shell) shell.classList.remove("chart-mode");

  const MD_DRAFT_KEY = "lc_md_draft";
  const MD_SEL_KEY = "lc_md_selected";
  const MD_CHART_REQ_KEY = "lc_md_chart_req";
  const md = {
    spec: null, models: [], catalog: {}, dirty: false,
    lastResult: null, channel: null,
    chartReq: null,
  };

  function blank() {
    return {
      id: "", name: "New model", notes: "",
      rules: [{
        when: [], if: { all: [{ gt: ["C.c", "C.o"] }] },
        then: [{ entry_long: "C.c" }, { set_target: "C.c + 5" }, { set_stop: "C.c - 2" }],
      }],
    };
  }
  function example() {
    return {
      id: "vp_break_long", name: "VP1 breakout long",
      notes: "Buy above VP1 VAH between 08:46 and 11:00 CT.",
      rules: [{
        when: [{ time_between: ["08:46", "11:00"] }],
        if: { all: [
          { eq: ["C.color", "green"] },
          { gt: ["Co", "VP1.vah"] },
          { gt: ["Ch", "VP1.vah"] },
        ]},
        then: [
          { entry_long: "Cc + 1" },
          { set_target: "next_up_of(entry, VP1.vah, VP1.high, VP2.vah, VP2.high)" },
          { set_stop: "Cl" },
        ],
      }],
    };
  }

  const LEAF_OPS = ["gt", "lt", "ge", "le", "eq", "ne", "between", "cross_above", "cross_below"];
  const LEAF_LABEL = {
    gt: ">", lt: "<", ge: "≥", le: "≤", eq: "=", ne: "≠",
    between: "between", cross_above: "crosses above", cross_below: "crosses below",
  };
  const ACTIONS = ["entry_long", "entry_short", "set_target", "set_stop", "exit"];
  const ACTION_LABEL = {
    entry_long: "Entry long =", entry_short: "Entry short =",
    set_target: "Set target =", set_stop: "Set stop =", exit: "Exit now",
  };

  function get(root, path) {
    if (!path) return root;
    return path.split(".").reduce((c, p) =>
      c == null ? null : (/^\d+$/.test(p) ? c[Number(p)] : c[p]), root);
  }
  function set(root, path, val) {
    const parts = path.split(".");
    let cur = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      cur = /^\d+$/.test(p) ? cur[Number(p)] : cur[p];
    }
    const last = parts[parts.length - 1];
    if (/^\d+$/.test(last)) cur[Number(last)] = val; else cur[last] = val;
  }
  function del(root, path) {
    const parts = path.split(".");
    let cur = root;
    for (let i = 0; i < parts.length - 1; i++) {
      const p = parts[i];
      cur = /^\d+$/.test(p) ? cur[Number(p)] : cur[p];
    }
    const last = parts[parts.length - 1];
    if (Array.isArray(cur) && /^\d+$/.test(last)) cur.splice(Number(last), 1);
    else delete cur[last];
  }

  function renderIf(node, path, depth) {
    if (!node || typeof node !== "object") return "";
    const key = Object.keys(node)[0];
    const val = node[key];
    const indent = `padding-left:${depth * 14}px`;
    if (key === "all" || key === "any") {
      const kids = val.map((c, i) => renderIf(c, `${path}.${key}.${i}`, depth + 1)).join("");
      return `<div class="lc-md-node" style="${indent}" data-path="${esc(path)}">
        <select class="lc-md-if-type" data-act="if-type">
          <option value="all"${key === "all" ? " selected" : ""}>all of</option>
          <option value="any"${key === "any" ? " selected" : ""}>any of</option>
          <option value="not">not</option>
          <option value="__leaf__">condition</option>
        </select>
        <button type="button" class="lc-md-add-child" data-act="if-add-child" title="Add child">+</button>
        ${path.endsWith(".if") ? "" : `<button type="button" class="lc-md-del" data-act="node-del">−</button>`}
      </div>${kids}`;
    }
    if (key === "not") {
      return `<div class="lc-md-node" style="${indent}" data-path="${esc(path)}">
        <select class="lc-md-if-type" data-act="if-type">
          <option value="all">all of</option>
          <option value="any">any of</option>
          <option value="not" selected>not</option>
          <option value="__leaf__">condition</option>
        </select>
        ${path.endsWith(".if") ? "" : `<button type="button" class="lc-md-del" data-act="node-del">−</button>`}
      </div>${renderIf(val, `${path}.not`, depth + 1)}`;
    }
    const args = Array.isArray(val) ? val.slice() : [];
    const isBetween = key === "between";
    const argsHtml = isBetween ? `
      <input class="lc-md-expr" data-act="leaf-arg" data-idx="0" value="${esc(String(args[0] ?? ""))}" placeholder="expr">
      <input class="lc-md-expr" data-act="leaf-arg" data-idx="1" value="${esc(String(args[1] ?? ""))}" placeholder="low">
      <span class="muted">and</span>
      <input class="lc-md-expr" data-act="leaf-arg" data-idx="2" value="${esc(String(args[2] ?? ""))}" placeholder="high">
    ` : `
      <input class="lc-md-expr" data-act="leaf-arg" data-idx="0" value="${esc(String(args[0] ?? ""))}" placeholder="expr">
      <select class="lc-md-op" data-act="leaf-op">
        ${LEAF_OPS.filter(o => o !== "between").map(o =>
          `<option value="${o}"${o === key ? " selected" : ""}>${LEAF_LABEL[o]}</option>`).join("")}
      </select>
      <input class="lc-md-expr" data-act="leaf-arg" data-idx="1" value="${esc(String(args[1] ?? ""))}" placeholder="expr">
    `;
    return `<div class="lc-md-node" style="${indent}" data-path="${esc(path)}">
      <select class="lc-md-if-type" data-act="if-type">
        <option value="all">all of</option>
        <option value="any">any of</option>
        <option value="not">not</option>
        <option value="__leaf__" selected>condition</option>
      </select>
      ${argsHtml}
      ${path.endsWith(".if") ? "" : `<button type="button" class="lc-md-del" data-act="node-del">−</button>`}
    </div>`;
  }

  function renderThen(actions, path, depth) {
    return (actions || []).map((a, i) => {
      const p = `${path}.${i}`;
      const indent = `padding-left:${depth * 14}px`;
      if (a && typeof a === "object" && "if" in a && "then" in a) {
        return `<div class="lc-md-node lc-md-nested-hdr" style="${indent}" data-path="${esc(p)}">
          <span class="lc-md-kw">if …</span>
          <button type="button" class="lc-md-del" data-act="node-del">−</button>
        </div>
        ${renderIf(a.if, `${p}.if`, depth + 1)}
        <div class="lc-md-node lc-md-nested-then" style="${indent}"><span class="lc-md-kw">then</span></div>
        ${renderThen(a.then, `${p}.then`, depth + 1)}
        <div class="lc-md-then-add" style="${indent}">
          <button type="button" data-act="then-add-action" data-path="${esc(p + ".then")}">+ action</button>
          <button type="button" data-act="then-add-nested" data-path="${esc(p + ".then")}">+ nested if</button>
        </div>`;
      }
      const key = Object.keys(a)[0];
      const val = a[key];
      const input = key === "exit" ? "" :
        `<input class="lc-md-expr" data-act="action-expr" value="${esc(String(val ?? ""))}" placeholder="expression">`;
      return `<div class="lc-md-node lc-md-action" style="${indent}" data-path="${esc(p)}">
        <select class="lc-md-action-type" data-act="action-type">
          ${ACTIONS.map(k => `<option value="${k}"${k === key ? " selected" : ""}>${ACTION_LABEL[k]}</option>`).join("")}
        </select>
        ${input}
        <button type="button" class="lc-md-del" data-act="node-del">−</button>
      </div>`;
    }).join("");
  }

  function renderWhen(gates, path) {
    const rows = (gates || []).map((g, i) => {
      const p = `${path}.${i}`;
      const kind = Object.keys(g)[0];
      const val = g[kind];
      if (kind === "time_between") {
        return `<div class="lc-md-node lc-md-when" data-path="${esc(p)}">
          <select class="lc-md-when-kind" data-act="when-kind">
            <option value="time_between" selected>time between</option>
            <option value="bars_from_open_ge">bars from open ≥</option>
            <option value="bars_from_open_le">bars from open ≤</option>
          </select>
          <input class="lc-md-hhmm" data-act="when-arg" data-idx="0" value="${esc(val[0] || "")}" placeholder="HH:MM">
          <span class="muted">and</span>
          <input class="lc-md-hhmm" data-act="when-arg" data-idx="1" value="${esc(val[1] || "")}" placeholder="HH:MM">
          <button type="button" class="lc-md-del" data-act="node-del">−</button>
        </div>`;
      }
      return `<div class="lc-md-node lc-md-when" data-path="${esc(p)}">
        <select class="lc-md-when-kind" data-act="when-kind">
          <option value="time_between">time between</option>
          <option value="bars_from_open_ge"${kind === "bars_from_open_ge" ? " selected" : ""}>bars from open ≥</option>
          <option value="bars_from_open_le"${kind === "bars_from_open_le" ? " selected" : ""}>bars from open ≤</option>
        </select>
        <input class="lc-md-hhmm" type="number" min="0" step="1" data-act="when-arg" data-idx="0" value="${Number(val) || 0}">
        <button type="button" class="lc-md-del" data-act="node-del">−</button>
      </div>`;
    }).join("");
    return `<div class="lc-md-when-list">${rows}
      <button type="button" class="lc-md-add-when" data-act="when-add" data-path="${esc(path)}">+ when</button>
    </div>`;
  }

  function renderRule(rule, ri) {
    const base = `rules.${ri}`;
    return `<div class="lc-md-rule">
      <header class="lc-md-rule-hdr">
        <span class="lc-md-kw">Rule ${ri + 1}</span>
        <button type="button" class="lc-md-rule-del" data-act="rule-del" data-path="${esc(base)}">×</button>
      </header>
      <div class="lc-md-section"><span class="lc-md-kw">When</span>
        ${renderWhen(rule.when, `${base}.when`)}</div>
      <div class="lc-md-section"><span class="lc-md-kw">If</span>
        ${renderIf(rule.if, `${base}.if`, 0)}</div>
      <div class="lc-md-section"><span class="lc-md-kw">Then</span>
        ${renderThen(rule.then, `${base}.then`, 0)}
        <div class="lc-md-then-add">
          <button type="button" data-act="then-add-action" data-path="${esc(base + ".then")}">+ action</button>
          <button type="button" data-act="then-add-nested" data-path="${esc(base + ".then")}">+ nested if</button>
        </div></div>
    </div>`;
  }

  function renderTree() {
    const host = $("#lc-md-tree");
    if (!host || !md.spec) return;
    host.innerHTML = md.spec.rules.map((r, i) => renderRule(r, i)).join("");
    $("#lc-md-name").value = md.spec.name || "";
    $("#lc-md-id").value = md.spec.id || "";
    $("#lc-md-notes").value = md.spec.notes || "";
    $("#lc-md-draft").hidden = !md.dirty;
  }

  function renderVars() {
    const host = $("#lc-md-vars-body");
    if (!host) return;
    const groups = {};
    for (const [name, info] of Object.entries(md.catalog || {})) {
      const g = info.group || "other";
      (groups[g] = groups[g] || []).push({ name, ...info });
    }
    const order = ["candle", "constant", "VP1", "VP2", "VP3", "VP4", "VP5",
                   "overlay", "large_prints", "math", "other"];
    const gnames = [...order, ...Object.keys(groups).filter(g => !order.includes(g))];
    host.innerHTML = gnames.filter(g => groups[g]).map(g => `
      <div class="lc-md-var-group">
        <div class="lc-md-var-group-h">${esc(g)}</div>
        ${groups[g].map(v => `
          <div class="lc-md-var" title="${esc(v.help || "")}">
            <code>${esc(v.name)}</code>
            <span class="muted">${esc(v.help || "")}</span>
          </div>`).join("")}
      </div>`).join("");
  }

  function markDirty() { md.dirty = true; saveDraft(); $("#lc-md-draft").hidden = false; }
  function saveDraft() { try { localStorage.setItem(MD_DRAFT_KEY, JSON.stringify(md.spec)); } catch {} }
  function loadDraft() {
    try {
      const s = localStorage.getItem(MD_DRAFT_KEY);
      if (!s) return null;
      const o = JSON.parse(s);
      return (o && typeof o === "object" && Array.isArray(o.rules)) ? o : null;
    } catch { return null; }
  }

  function readChartReq() {
    if (md.chartReq) return md.chartReq;
    try {
      const raw = localStorage.getItem(MD_CHART_REQ_KEY);
      if (raw) md.chartReq = JSON.parse(raw);
    } catch {}
    if (!md.chartReq) {
      // Reconstruct from Live-Chart persisted settings.
      const tf = localStorage.getItem("lc_tf") || "1min";
      const lb = localStorage.getItem("lc_lb") || "session";
      const keys = (localStorage.getItem("lc_overlays") || "").split(",").filter(Boolean);
      let profiles = null;
      try {
        const specs = JSON.parse(localStorage.getItem("lc_vp_specs") || "[]");
        if (specs.length && localStorage.getItem("lc_vp") !== "0") profiles = JSON.stringify(specs);
      } catch {}
      let emas = null;
      try {
        const specs = JSON.parse(localStorage.getItem("lc_ema_specs") || "[]");
        if (specs.length && localStorage.getItem("lc_ema") !== "0") emas = JSON.stringify(specs);
      } catch {}
      md.chartReq = {
        timeframe: tf, lookback: lb,
        overlays: keys.length ? keys.join(",") : null,
        cvd_origin: localStorage.getItem("lc_cvd_origin") || "session",
        vwap_origin: keys.includes("vw_vwap") || keys.includes("vw_vwap_dist")
          ? (localStorage.getItem("lc_vwap_origin") || "session") : "session",
        ofi_origin: keys.includes("of_ofi_cum") ? (localStorage.getItem("lc_ofi_origin") || "session") : null,
        ad_origin:  keys.includes("ad_line")    ? (localStorage.getItem("lc_ad_origin")  || "session") : null,
        profiles,
        emas,
        eth_aggregate: localStorage.getItem("lc_eth_agg") !== "0",
        print_min: Math.max(1, Math.floor(Number(localStorage.getItem("lc_print_min")) || 10)),
      };
    }
    return md.chartReq;
  }

  function chartReqBody(includePrint) {
    const r = readChartReq();
    const b = {
      timeframe: r.timeframe, lookback: r.lookback,
      overlays: r.overlays, cvd_origin: r.cvd_origin,
      vwap_origin: r.vwap_origin,
      ofi_origin: r.ofi_origin, ad_origin: r.ad_origin,
      profiles: r.profiles, emas: r.emas,
      eth_aggregate: !!r.eth_aggregate,
    };
    if (includePrint) return { chart: b, print_min: r.print_min, model: null, session_end: null };
    return b;
  }

  async function refreshCatalog() {
    try {
      const out = await post("/api/chart/model-vars", chartReqBody(false));
      md.catalog = out.catalog || {};
      renderVars();
      $("#lc-md-status").textContent =
        `variables: ${Object.keys(md.catalog).length} · ` +
        `chart=${readChartReq().timeframe}/${readChartReq().lookback}`;
    } catch (e) {
      $("#lc-md-status").textContent = "catalog: " + (e.message || e);
    }
  }

  async function run() {
    if (!md.spec) return;
    $("#lc-md-status").textContent = "running…";
    try {
      const body = chartReqBody(true);
      body.model = md.spec;
      const out = await post("/api/chart/simulate-model", body);
      md.lastResult = out;
      md.catalog = out.catalog || md.catalog;
      renderVars();
      renderResults(out);
      const p = out.pnl || {};
      $("#lc-md-status").textContent =
        `${out.trades.length} trades · net ${(p.gross_points || 0).toFixed(2)} pts ($${(p.gross_usd || 0).toFixed(0)})`;
      try { md.channel && md.channel.postMessage({ type: "run-result", result: out }); } catch {}
    } catch (e) {
      $("#lc-md-status").textContent = "error: " + (e.message || e);
      toast(e.message || String(e), "err");
    }
  }

  function renderResults(res) {
    $("#lc-md-results").hidden = false;
    const p = res.pnl || {};
    const pf = (p.profit_factor == null) ? "∞" : p.profit_factor;
    $("#lc-md-pnl").innerHTML = `
      <div class="lc-md-pnl-grid">
        <div><label>Trades</label><span>${p.n_trades}</span></div>
        <div><label>Win rate</label><span>${((p.win_rate || 0) * 100).toFixed(1)}%</span></div>
        <div><label>Wins / Losses</label><span>${p.n_wins} / ${p.n_losses}</span></div>
        <div><label>Gross (pts)</label><span class="${p.gross_points >= 0 ? "pos" : "neg"}">${(p.gross_points || 0).toFixed(2)}</span></div>
        <div><label>Gross ($)</label><span class="${p.gross_usd >= 0 ? "pos" : "neg"}">$${(p.gross_usd || 0).toFixed(0)}</span></div>
        <div><label>Avg win / loss</label><span>${(p.avg_win_points || 0).toFixed(2)} / ${(p.avg_loss_points || 0).toFixed(2)}</span></div>
        <div><label>Profit factor</label><span>${pf}</span></div>
        <div><label>Max drawdown</label><span class="neg">${(p.max_dd_points || 0).toFixed(2)}</span></div>
      </div>
      <canvas id="lc-md-equity" class="lc-md-equity"></canvas>`;
    drawEquity(p.equity_points || []);
    $("#lc-md-blotter-count").textContent = `(${(res.trades || []).length})`;
    const rows = (res.trades || []).map((t, i) => `
      <tr data-idx="${i}" class="${t.pnl_points >= 0 ? "pos" : "neg"}">
        <td>${i + 1}</td>
        <td class="${t.side}">${t.side.toUpperCase()}</td>
        <td>${esc(fmtTs(t.entry_ts))}</td>
        <td>${Number(t.entry_price).toFixed(2)}</td>
        <td>${esc(fmtTs(t.exit_ts))}</td>
        <td>${Number(t.exit_price).toFixed(2)}</td>
        <td>${Number(t.pnl_points).toFixed(2)}</td>
        <td>${esc(t.exit_reason)}</td>
        <td>${t.duration_bars}</td>
      </tr>`).join("");
    $("#lc-md-blotter").innerHTML = `
      <thead><tr>
        <th>#</th><th>Side</th><th>Entry ts</th><th>Entry</th>
        <th>Exit ts</th><th>Exit</th><th>PnL (pts)</th><th>Reason</th><th>Bars</th>
      </tr></thead><tbody>${rows}</tbody>`;
  }

  function drawEquity(equity) {
    const cv = $("#lc-md-equity");
    if (!cv) return;
    const w = cv.clientWidth || 400, h = cv.clientHeight || 60;
    const dpr = window.devicePixelRatio || 1;
    cv.width = w * dpr; cv.height = h * dpr;
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    if (!equity.length) return;
    const min = Math.min(0, ...equity), max = Math.max(0, ...equity);
    const range = (max - min) || 1;
    const zeroY = h - ((0 - min) / range) * h;
    ctx.strokeStyle = "#4b5563"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
    ctx.strokeStyle = equity[equity.length - 1] >= 0 ? "#22c55e" : "#ef4444";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    equity.forEach((v, i) => {
      const x = (i / Math.max(1, equity.length - 1)) * w;
      const y = h - ((v - min) / range) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  async function reloadModels() {
    try {
      md.models = await api("/api/chart-models");
      const sel = $("#lc-md-select");
      const cur = md.spec ? md.spec.id : (localStorage.getItem(MD_SEL_KEY) || "");
      sel.innerHTML = `<option value="">(draft)</option>` +
        md.models.map(m => `<option value="${esc(m.id)}">${esc(m.name || m.id)}</option>`).join("");
      sel.value = md.models.some(m => m.id === cur) ? cur : "";
    } catch { md.models = []; }
  }

  async function saveTo(newId) {
    if (!md.spec) return;
    let id = (newId || md.spec.id || "").trim().toLowerCase().replace(/[^a-z0-9_-]/g, "-");
    if (!id) { toast("id is required (lowercase, dashes/underscores)", "err"); return; }
    md.spec.id = id;
    md.spec.name = ($("#lc-md-name").value || id).trim();
    md.spec.notes = $("#lc-md-notes").value.trim();
    try {
      await put(`/api/chart-models/${encodeURIComponent(id)}`, md.spec);
      md.dirty = false;
      localStorage.setItem(MD_SEL_KEY, id);
      $("#lc-md-draft").hidden = true;
      await reloadModels();
      $("#lc-md-select").value = id;
      toast(`Saved "${md.spec.name}"`, "info", 2500);
    } catch (e) { toast(e.message || String(e), "err"); }
  }

  async function loadFromServer(id) {
    if (!id) {
      md.spec = loadDraft() || blank();
      md.dirty = !!loadDraft();
      renderTree();
      return;
    }
    try {
      const rec = await api(`/api/chart-models/${encodeURIComponent(id)}`);
      md.spec = { id: rec.id, name: rec.name, notes: rec.notes || "", rules: rec.rules || [] };
      md.dirty = false;
      localStorage.setItem(MD_SEL_KEY, id);
      $("#lc-md-draft").hidden = true;
      renderTree();
    } catch (e) { toast(e.message || String(e), "err"); }
  }

  function onTreeChange(ev) {
    const t = ev.target;
    const row = t.closest("[data-path]");
    if (!row) return;
    const act = t.dataset.act;
    const path = row.dataset.path;
    if (!act || !path || !md.spec) return;
    if (act === "if-type") {
      const cur = get(md.spec, path);
      const curKey = Object.keys(cur)[0];
      const curVal = cur[curKey];
      const nv = t.value;
      let repl;
      if (nv === "all" || nv === "any") {
        repl = { [nv]: Array.isArray(curVal) ? curVal : [{ gt: ["C.c", "C.o"] }] };
      } else if (nv === "not") {
        repl = { not: (typeof curVal === "object" && !Array.isArray(curVal)) ? curVal : { gt: ["C.c", "C.o"] } };
      } else {
        repl = { gt: ["C.c", "C.o"] };
      }
      set(md.spec, path, repl);
      markDirty(); renderTree();
      return;
    }
    if (act === "leaf-op") {
      const cur = get(md.spec, path);
      const oldKey = Object.keys(cur)[0];
      const oldArgs = cur[oldKey];
      const nk = t.value;
      let newArgs = oldArgs;
      if (nk === "between" && (!Array.isArray(oldArgs) || oldArgs.length !== 3))
        newArgs = [oldArgs[0] ?? "C.c", oldArgs[1] ?? "0", "100"];
      else if (nk !== "between" && Array.isArray(oldArgs) && oldArgs.length !== 2)
        newArgs = [oldArgs[0] ?? "C.c", oldArgs[1] ?? "0"];
      set(md.spec, path, { [nk]: newArgs });
      markDirty(); renderTree();
      return;
    }
    if (act === "leaf-arg") {
      const cur = get(md.spec, path);
      const key = Object.keys(cur)[0];
      const idx = Number(t.dataset.idx);
      const args = cur[key].slice();
      const raw = t.value;
      args[idx] = raw === "" ? "" : (isNaN(Number(raw)) || /[a-zA-Z_]/.test(raw) ? raw : Number(raw));
      set(md.spec, path, { [key]: args });
      markDirty();
      return;
    }
    if (act === "action-type") {
      const cur = get(md.spec, path);
      const oldKey = Object.keys(cur)[0];
      const oldVal = cur[oldKey];
      const nk = t.value;
      if (nk === "exit") set(md.spec, path, { exit: true });
      else set(md.spec, path, { [nk]: (oldKey === "exit" ? "C.c" : oldVal) });
      markDirty(); renderTree();
      return;
    }
    if (act === "action-expr") {
      const cur = get(md.spec, path);
      const key = Object.keys(cur)[0];
      set(md.spec, path, { [key]: t.value });
      markDirty();
      return;
    }
    if (act === "when-kind") {
      const nk = t.value;
      const val = nk === "time_between" ? ["08:30", "15:00"] : 0;
      set(md.spec, path, { [nk]: val });
      markDirty(); renderTree();
      return;
    }
    if (act === "when-arg") {
      const cur = get(md.spec, path);
      const k = Object.keys(cur)[0];
      if (k === "time_between") {
        const args = cur[k].slice();
        args[Number(t.dataset.idx)] = t.value;
        set(md.spec, path, { [k]: args });
      } else {
        set(md.spec, path, { [k]: Number(t.value) || 0 });
      }
      markDirty();
      return;
    }
  }

  function onTreeClick(ev) {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const act = btn.dataset.act;
    if (act === "if-add-child") {
      const path = btn.closest("[data-path]").dataset.path;
      const cur = get(md.spec, path);
      const key = Object.keys(cur)[0];
      if (key === "all" || key === "any") {
        cur[key].push({ gt: ["C.c", "C.o"] });
        markDirty(); renderTree();
      }
      return;
    }
    if (act === "node-del") {
      del(md.spec, btn.closest("[data-path]").dataset.path);
      markDirty(); renderTree();
      return;
    }
    if (act === "when-add") {
      const cur = get(md.spec, btn.dataset.path);
      cur.push({ time_between: ["08:30", "15:00"] });
      markDirty(); renderTree();
      return;
    }
    if (act === "then-add-action") {
      const cur = get(md.spec, btn.dataset.path);
      cur.push({ set_target: "C.c + 1" });
      markDirty(); renderTree();
      return;
    }
    if (act === "then-add-nested") {
      const cur = get(md.spec, btn.dataset.path);
      cur.push({
        if: { gt: ["C.c", "C.o"] },
        then: [{ entry_long: "C.c" }, { set_target: "C.c + 5" }, { set_stop: "C.c - 2" }],
      });
      markDirty(); renderTree();
      return;
    }
    if (act === "rule-del") {
      if (md.spec.rules.length <= 1) { toast("at least one rule is required", "warn"); return; }
      const idx = Number(btn.dataset.path.split(".")[1]);
      md.spec.rules.splice(idx, 1);
      markDirty(); renderTree();
      return;
    }
  }

  view.innerHTML = `
    <div class="lc-md-page">
      <header class="lc-md-hdr">
        <span class="lc-md-title">Model design</span>
        <select id="lc-md-select" title="Saved models"></select>
        <button type="button" id="lc-md-new" title="New model">+ New</button>
        <button type="button" id="lc-md-save" title="Save changes">Save</button>
        <button type="button" id="lc-md-saveas" title="Save as new id">Save as…</button>
        <button type="button" id="lc-md-delete" title="Delete this model">Delete</button>
        <span class="lc-md-sep">·</span>
        <button type="button" class="primary" id="lc-md-run" title="Simulate over the current Live Chart window">▶ Run</button>
        <button type="button" id="lc-md-reset" title="Clear markers on the Live Chart tab">↺ Reset chart</button>
        <span id="lc-md-draft" class="lc-md-draft muted" title="Draft auto-saved locally" hidden>Draft ●</span>
        <span id="lc-md-status" class="muted lc-md-status" style="margin-left:auto"></span>
      </header>
      <div class="lc-md-body">
        <div class="lc-md-meta">
          <label>Name<input type="text" id="lc-md-name" maxlength="128" placeholder="My rule set"></label>
          <label>Id<input type="text" id="lc-md-id" maxlength="64" placeholder="lowercase-id-only"></label>
          <label class="lc-md-notes">Notes<input type="text" id="lc-md-notes" placeholder="short description"></label>
        </div>
        <div class="lc-md-split">
          <div class="lc-md-tree-wrap">
            <div class="lc-md-tree-hdr">
              <span>Rules</span>
              <button type="button" id="lc-md-add-rule" title="Add another rule">+ Add rule</button>
            </div>
            <div id="lc-md-tree" class="lc-md-tree" aria-live="polite"></div>
          </div>
          <aside class="lc-md-vars">
            <div class="lc-md-vars-hdr">Variables
              <button type="button" id="lc-md-vars-refresh" title="Rebuild catalog from current chart">↻</button>
            </div>
            <div id="lc-md-vars-body" class="lc-md-vars-body"></div>
          </aside>
        </div>
        <div class="lc-md-results" id="lc-md-results" hidden>
          <div class="lc-md-pnl" id="lc-md-pnl"></div>
          <div class="lc-md-blotter-wrap">
            <div class="lc-md-blotter-hdr">Trade blotter <span class="muted" id="lc-md-blotter-count"></span></div>
            <div class="lc-md-blotter-scroll">
              <table class="lc-md-blotter" id="lc-md-blotter"></table>
            </div>
          </div>
        </div>
      </div>
    </div>`;

  try { md.channel = new BroadcastChannel("vtech-md"); } catch {}
  const draft = loadDraft();
  md.spec = draft || example();
  md.dirty = !!draft;

  $("#lc-md-new").addEventListener("click", () => {
    md.spec = blank();
    md.dirty = true;
    $("#lc-md-select").value = "";
    markDirty(); renderTree();
  });
  $("#lc-md-save").addEventListener("click", () => saveTo(md.spec ? md.spec.id : ""));
  $("#lc-md-saveas").addEventListener("click", async () => {
    const id = prompt("Save as (id — lowercase / dashes / underscores):", md.spec.id || "");
    if (id) await saveTo(id);
  });
  $("#lc-md-delete").addEventListener("click", async () => {
    if (!md.spec.id) { toast("nothing to delete (unsaved draft)", "warn"); return; }
    if (!confirm(`Delete "${md.spec.name}" from server?`)) return;
    try {
      await api(`/api/chart-models/${encodeURIComponent(md.spec.id)}`, { method: "DELETE" });
      localStorage.removeItem(MD_SEL_KEY);
      await reloadModels();
      md.spec = blank();
      md.dirty = false;
      renderTree();
    } catch (e) { toast(e.message || String(e), "err"); }
  });
  $("#lc-md-run").addEventListener("click", run);
  $("#lc-md-reset").addEventListener("click", () => {
    md.lastResult = null;
    $("#lc-md-results").hidden = true;
    try { md.channel && md.channel.postMessage({ type: "clear-markers" }); } catch {}
    $("#lc-md-status").textContent = "";
  });
  $("#lc-md-select").addEventListener("change", (e) => loadFromServer(e.target.value));
  $("#lc-md-vars-refresh").addEventListener("click", refreshCatalog);
  $("#lc-md-add-rule").addEventListener("click", () => {
    md.spec.rules.push(blank().rules[0]);
    markDirty(); renderTree();
  });
  ["#lc-md-name", "#lc-md-id", "#lc-md-notes"].forEach(sel => {
    $(sel).addEventListener("input", (e) => {
      if (!md.spec) return;
      md.spec[sel.replace("#lc-md-", "")] = e.target.value;
      markDirty();
    });
  });
  $("#lc-md-tree").addEventListener("change", onTreeChange);
  $("#lc-md-tree").addEventListener("input",  onTreeChange);
  $("#lc-md-tree").addEventListener("click",  onTreeClick);
  $("#lc-md-blotter").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-idx]");
    if (!tr) return;
    const idx = Number(tr.dataset.idx);
    const was = tr.classList.contains("active");
    $("#lc-md-blotter").querySelectorAll("tr.active").forEach(el => el.classList.remove("active"));
    if (was) {
      try { md.channel && md.channel.postMessage({ type: "select-trade", idx: -1 }); } catch {}
      return;
    }
    tr.classList.add("active");
    try { md.channel && md.channel.postMessage({ type: "select-trade", idx }); } catch {}
  });

  renderTree();
  await reloadModels();
  const sel = localStorage.getItem(MD_SEL_KEY);
  if (sel && md.models.some(m => m.id === sel) && !draft) {
    $("#lc-md-select").value = sel;
    await loadFromServer(sel);
  }
  await refreshCatalog();
}


/* ── External Models (Pine replicas on PostgreSQL candles) ─────────────── */
const EXT_HANDOFF = "vtech_ext_bt";

function extParamInput(spec, value, prefix = "ext-p-", vals = {}) {
  const key = spec.key;
  const id = `${prefix}${key}`;
  const help = spec.help ? ` title="${esc(spec.help)}"` : "";
  if (spec.type === "bool") {
    const on = value === true || value === "true" || value === 1 || value === "1";
    return `<label class="checkbox ext-bool"${help}>
      <input id="${esc(id)}" type="checkbox"${on ? " checked" : ""}>
      <span>${esc(spec.label || key)}</span></label>`;
  }
  if (spec.type === "enum") {
    const opts = (spec.choices || []).map(c =>
      `<option value="${esc(c)}"${String(value) === String(c) ? " selected" : ""}>${esc(c)}</option>`).join("");
    return `<select id="${esc(id)}"${help}>${opts}</select>`;
  }
  if (spec.type === "qty_unit") {
    const unitKey = spec.unit_key;
    const choices = (spec.unit_choices && spec.unit_choices.length)
      ? spec.unit_choices : ["points", "atr"];
    const unitLabels = { points: "points", atr: "× ATR", stop: "× stop" };
    const unit = String((vals && unitKey && vals[unitKey]) || choices[0] || "points");
    const step = spec.step != null ? ` step="${spec.step}"` : " step=\"0.25\"";
    const min = spec.min != null ? ` min="${spec.min}"` : "";
    const max = spec.max != null ? ` max="${spec.max}"` : "";
    const val = value == null ? "" : esc(value);
    const btns = choices.map(c =>
      `<button type="button" data-unit="${esc(c)}" data-unit-for="${esc(unitKey)}"
        class="${unit === c ? "on" : ""}">${esc(unitLabels[c] || c)}</button>`
    ).join("");
    return `<div class="bt-qty-row">
      <input id="${esc(id)}" type="number" value="${val}"${step}${min}${max}${help}>
      <input type="hidden" id="${esc(prefix)}${esc(unitKey)}" value="${esc(unit)}">
      <div class="bt-unit-toggle" role="group" aria-label="${esc(spec.label || key)} unit">
        ${btns}
      </div>
    </div>`;
  }
  const typ = spec.type === "int" || spec.type === "float" ? "number" : (spec.type === "time" ? "time" : "text");
  const step = spec.step != null ? ` step="${spec.step}"` : (spec.type === "int" ? " step=\"1\"" : "");
  const min = spec.min != null ? ` min="${spec.min}"` : "";
  const max = spec.max != null ? ` max="${spec.max}"` : "";
  const val = value == null ? "" : esc(value);
  return `<input id="${esc(id)}" type="${typ}" value="${val}"${step}${min}${max}${help}>`;
}

function extReadParams(spec, prefix = "ext-p-") {
  const out = {};
  for (const s of spec || []) {
    const el = document.getElementById(`${prefix}${s.key}`);
    if (!el) continue;
    if (s.type === "int") out[s.key] = Number.parseInt(el.value, 10);
    else if (s.type === "float" || s.type === "qty_unit") out[s.key] = Number.parseFloat(el.value);
    else if (s.type === "bool") out[s.key] = !!el.checked;
    else out[s.key] = el.value;
    if (s.unit_key) {
      const u = document.getElementById(`${prefix}${s.unit_key}`);
      if (u) out[s.unit_key] = u.value || "points";
    }
  }
  return out;
}

function extFieldsHtml(spec, vals, prefix = "ext-p-") {
  const src = vals || {};
  return (spec || []).map(s => {
    if (s.type === "bool") {
      return `<div>${extParamInput(s, src[s.key], prefix, src)}
        ${s.pine ? `<span class="pine">${esc(s.pine)}</span>` : ""}</div>`;
    }
    return `<div>
      <label>${esc(s.label)}${s.pine ? ` <span class="pine">${esc(s.pine)}</span>` : ""}</label>
      ${extParamInput(s, src[s.key], prefix, src)}
    </div>`;
  }).join("");
}

function btBookFieldsHtml(spec, vals, prefix = "bt-b-") {
  const labels = { session: "Session", entries: "Entries", exits: "Exits", size: "Size & costs" };
  const groups = [];
  const seen = new Set();
  for (const s of spec || []) {
    const g = s.group || "";
    if (!seen.has(g)) { seen.add(g); groups.push(g); }
  }
  if (!groups.length || (groups.length === 1 && !groups[0])) {
    return `<div class="ext-param-grid">${extFieldsHtml(spec, vals, prefix)}</div>`;
  }
  return groups.map(g => {
    const rows = (spec || []).filter(s => (s.group || "") === g);
    const title = labels[g] || g;
    return `<section class="bt-book-section">
      ${title ? `<h4>${esc(title)}</h4>` : ""}
      <div class="ext-param-grid">${extFieldsHtml(rows, vals, prefix)}</div>
    </section>`;
  }).join("");
}

async function viewExternalModels(presetId) {
  view.classList.remove("wide");
  view.innerHTML = `<div class="empty"><span class="spin">◐</span> loading external models…</div>`;
  let payload;
  try {
    payload = await api("/api/external-models");
  } catch (e) {
    view.innerHTML = `<div class="empty">⚠ ${esc(e.message)}</div>`;
    return;
  }
  const models = Array.isArray(payload) ? payload : (payload.models || []);
  if (!models.length) {
    view.innerHTML = `<div class="empty">No external models in <span class="mono">config/external_models/</span></div>`;
    return;
  }
  let selected = models.find(m => m.id === presetId) || models[0];

  const handoff = (hardenedName, hardParams) => {
    const modelParams = extReadParams(selected.param_spec || [], "ext-p-");
    sessionStorage.setItem(EXT_HANDOFF, JSON.stringify({
      plane: "ext",
      stack: [{
        model: selected.id,
        hardened: hardenedName || null,
        params: hardParams || (hardenedName ? null : modelParams),
      }],
      label: selected.title || selected.id,
    }));
    location.hash = `#/backtest?plane=ext&external=${encodeURIComponent(selected.id)}`;
  };

  const paint = () => {
    const spec = selected.param_spec || [];
    const vals = selected.defaults || selected.factory_params || {};
    const span = selected.span || {};
    const savedAt = selected.defaults_saved_at;
    const hardened = selected.hardened || [];
    const hardRows = hardened.map(h => `
      <div class="ext-hardened-row">
        <div>
          <div class="name">${esc(h.name)}</div>
          <div class="muted">${esc(h.saved_at || "")}${h.note ? " · " + esc(h.note) : ""}</div>
        </div>
        <div class="fixed">
          <button type="button" class="small" data-load="${esc(h.name)}">Load</button>
          <button type="button" class="small primary" data-bt="${esc(h.name)}">Backtest</button>
          <button type="button" class="small danger" data-del="${esc(h.name)}">Delete</button>
        </div>
      </div>`).join("") || `<div class="muted">No hardened snapshots yet. Tune this model's parameters, then save a named copy.</div>`;
    const table = span.table || selected.candle_table || "databento.tbbo_1min";
    const spanLine = span.available
      ? `<span class="ext-span">${esc(span.start)} → ${esc(span.end)}</span>${span.n_rows != null ? " · " + fmtInt(span.n_rows) + " 1m rows" : ""} in <span class="mono">${esc(table)}</span>`
      : `<span class="muted">Knobs only — <span class="mono">${esc(table)}</span> candles load when you run Backtest.</span>`;
    const opts = models.map(m =>
      `<option value="${esc(m.id)}"${m.id === selected.id ? " selected" : ""}>${esc(m.title || m.id)}</option>`).join("");
    view.innerHTML = `
      <div class="card">
        <h3>External Models</h3>
        <div class="ext-intro muted">
          Rule replicas that are <b>not</b> ML catalog families. Each engine
          (Parabolic SAR, 20 EMA, ZigZag High/Low, Timed PSAR) has its own long/short signals. Stack them on
          the Backtest tab (EXT Model): matching sides enter, opposing or flat
          signals stand down or exit. Session, ATR gate, profit/stop, and costs
          are <b>Backtest Parameters</b> on that tab (session vs trade window,
          per-engine candle timeframe, PT/SL/min ATR in points or × ATR; PT also × stop). Candles come from PostgreSQL
          <span class="mono">databento.tbbo_1min</span>. Save working knobs as
          <b>defaults</b>, or snapshot a <b>hardened</b> copy of the selected engine.
        </div>
        <div style="margin-top:8px">${spanLine}</div>
      </div>
      <div class="card" style="margin-top:16px">
        <div class="row">
          <div><label>model</label><select id="ext-model">${opts}</select></div>
        </div>
        <div class="muted" style="margin-top:10px">${esc(selected.description || "")}</div>
      </div>
      <div class="grid cols-2" style="margin-top:16px">
        <div class="card">
          <h3>Parameters · ${esc(selected.title || selected.id)}
            <span id="ext-dirty" class="ext-dirty" hidden>unsaved</span></h3>
          <div class="ext-param-grid">${extFieldsHtml(spec, vals, "ext-p-")}</div>
          <div class="muted" style="margin-top:10px">
            Defaults ${savedAt ? "saved " + esc(savedAt) : "are factory values"}.
            Reset restores <span class="mono">config/external_models/${esc(selected.id)}.yaml</span>.
          </div>
          <div class="ext-actions">
            <button type="button" class="primary" id="ext-save">Save as defaults</button>
            <button type="button" id="ext-reset">Reset to factory</button>
            <button type="button" class="primary" id="ext-bt">▶ Backtest this configuration</button>
          </div>
        </div>
        <div class="card">
          <h3>Hardened snapshots</h3>
          <p class="muted">Named, frozen parameter sets for <b>${esc(selected.title || selected.id)}</b>.</p>
          <div class="row">
            <div><label>name</label><input id="ext-hard-name" placeholder="tight-trail" maxlength="64"></div>
            <div><label>note</label><input id="ext-hard-note" placeholder="optional"></div>
          </div>
          <div class="checkbox"><input type="checkbox" id="ext-hard-defaults">
            <span>also write these params as the model defaults</span></div>
          <div class="ext-actions">
            <button type="button" class="primary" id="ext-hard-save">Save hardened copy</button>
          </div>
          <div id="ext-hard-list" style="margin-top:12px">${hardRows}</div>
        </div>
      </div>`;

    const markDirty = () => {
      try {
        const cur = JSON.stringify(extReadParams(spec, "ext-p-"));
        const base = JSON.stringify(vals);
        const el = $("#ext-dirty");
        if (el) el.hidden = cur === base;
      } catch { /* ignore */ }
    };
    spec.forEach(s => {
      const el = document.getElementById(`ext-p-${s.key}`);
      if (!el) return;
      el.addEventListener("input", markDirty);
      el.addEventListener("change", markDirty);
    });
    markDirty();

    $("#ext-model").onchange = () => {
      const id = $("#ext-model").value;
      if (!id || id === selected.id) return;
      selected = models.find(m => m.id === id) || selected;
      paint();
    };
    $("#ext-save").addEventListener("click", async () => {
      try {
        const rec = await put(`/api/external-models/${selected.id}/defaults`, {
          params: extReadParams(spec, "ext-p-"),
        });
        selected.defaults = rec.params;
        selected.defaults_saved_at = rec.saved_at;
        toast("defaults saved", "ok");
        paint();
      } catch (e) { toast(e.message, "err"); }
    });
    $("#ext-reset").addEventListener("click", async () => {
      try {
        const rec = await post(`/api/external-models/${selected.id}/defaults/reset`, {});
        selected.defaults = rec.params;
        selected.defaults_saved_at = null;
        toast("factory defaults restored", "ok");
        paint();
      } catch (e) { toast(e.message, "err"); }
    });
    $("#ext-bt").addEventListener("click", () => handoff(null));
    $("#ext-hard-save").addEventListener("click", async () => {
      const name = ($("#ext-hard-name").value || "").trim().toLowerCase();
      if (!name) { toast("name the hardened copy", "err"); return; }
      try {
        const rec = await post(`/api/external-models/${selected.id}/hardened`, {
          name,
          params: extReadParams(spec, "ext-p-"),
          note: ($("#ext-hard-note").value || "").trim() || null,
          set_as_defaults: $("#ext-hard-defaults").checked,
        });
        toast(`saved hardened ${rec.name}`, "ok");
        const fresh = await api(`/api/external-models/${selected.id}`);
        const ix = models.findIndex(m => m.id === selected.id);
        if (ix >= 0) models[ix] = fresh;
        selected = fresh;
        paint();
      } catch (e) { toast(e.message, "err"); }
    });
    $("#ext-hard-list").addEventListener("click", async (ev) => {
      const t = ev.target;
      if (!(t instanceof HTMLElement)) return;
      if (t.dataset.load) {
        const h = hardened.find(x => x.name === t.dataset.load);
        if (!h) return;
        selected.defaults = { ...(selected.factory_params || {}), ...(h.params || {}) };
        paint();
        toast(`loaded ${h.name} into the form (not saved as defaults)`, "ok");
      } else if (t.dataset.bt) {
        const h = hardened.find(x => x.name === t.dataset.bt);
        if (!h) return;
        handoff(h.name, h.params);
      } else if (t.dataset.del) {
        if (!confirm(`Delete hardened snapshot “${t.dataset.del}”?`)) return;
        try {
          await del(`/api/external-models/${selected.id}/hardened/${encodeURIComponent(t.dataset.del)}`);
          const fresh = await api(`/api/external-models/${selected.id}`);
          const ix = models.findIndex(m => m.id === selected.id);
          if (ix >= 0) models[ix] = fresh;
          selected = fresh;
          toast("deleted", "ok");
          paint();
        } catch (e) { toast(e.message, "err"); }
      }
    });
  };
  paint();
}

/* ── router ────────────────────────────────────────────────────────────── */

/* ── Chart Notes (Live Chart scenario snapshots) ───────────────────────── */
function noteFeatureSummary(chart) {
  if (!chart || typeof chart !== "object") return "no chart snapshot";
  const bits = [];
  if (chart.timeframe) bits.push(String(chart.timeframe));
  if (chart.lookback) bits.push(`lb ${chart.lookback}`);
  const ovs = Array.isArray(chart.overlays) ? chart.overlays.slice() : [];
  if (chart.ema_enabled && Array.isArray(chart.emas) && chart.emas.length) {
    bits.push(chart.emas.map(e => `EMA${e.n}(${e.lookback})`).join(", "));
  }
  if (chart.stoch_enabled && Array.isArray(chart.stochs) && chart.stochs.length) {
    bits.push(chart.stochs.map(s => `Stoch${s.n}(${s.k_period},${s.d_period})`).join(", "));
  }
  if (chart.vp_enabled && Array.isArray(chart.profiles) && chart.profiles.length) {
    bits.push(`${chart.profiles.length} VP`);
  }
  if (chart.sar) {
    bits.push(`SAR ${chart.sar.start}/${chart.sar.increment}/${chart.sar.maximum}`);
  }
  if (ovs.length) bits.push(ovs.join(", "));
  return bits.join(" · ") || "candles only";
}

async function viewChartNotes() {
  view.classList.remove("chart-fill");
  const shell = document.getElementById("shell");
  if (shell) shell.classList.remove("chart-mode");
  view.innerHTML = `<div class="empty"><span class="spin">◐</span> loading notes…</div>`;
  let rows = [];
  try {
    rows = await api("/api/chart/notes");
  } catch (e) {
    view.innerHTML = `<div class="empty">⚠ ${esc(e.message || e)}</div>`;
    return;
  }
  if (!rows.length) {
    view.innerHTML = `
      <div class="card">
        <h2 style="margin-top:0">Notes</h2>
        <p class="muted">No notes yet. On <a href="#/chart">Live Chart</a>, choose <b>Select</b>,
        drag a range on the candles, then <b>Create Note</b>.</p>
      </div>`;
    return;
  }
  const body = rows.map(r => `
    <tr>
      <td><a href="#/notes/${esc(r.id)}">${esc(r.name || r.id)}</a></td>
      <td class="mono muted">${esc(r.symbol || "—")}</td>
      <td class="muted">${esc((r.text || "").slice(0, 120))}${(r.text || "").length > 120 ? "…" : ""}</td>
      <td class="mono muted">${esc(fmtTs(r.created_at))}</td>
      <td><button type="button" data-note-del="${esc(r.id)}">Delete</button></td>
    </tr>`).join("");
  view.innerHTML = `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px">
        <h2 style="margin:0">Notes</h2>
        <a href="#/chart">← Live Chart</a>
      </div>
      <p class="muted">Scenario snapshots from Live Chart Select + Create Note. Name is symbol + selected range.</p>
      <div style="overflow:auto">
        <table>
          <thead><tr><th>Name</th><th>Symbol</th><th>Note</th><th>Saved</th><th></th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>`;
  view.querySelectorAll("[data-note-del]").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-note-del");
      if (!id || !window.confirm(`Delete note ${id}?`)) return;
      try {
        await api(`/api/chart/notes/${encodeURIComponent(id)}`, { method: "DELETE" });
        toast("Note deleted");
        viewChartNotes();
      } catch (e) {
        toast(e.message || String(e), "err");
      }
    });
  });
}

async function viewChartNoteDetail(nid) {
  view.classList.remove("chart-fill");
  const shell = document.getElementById("shell");
  if (shell) shell.classList.remove("chart-mode");
  view.innerHTML = `<div class="empty"><span class="spin">◐</span> loading…</div>`;
  let rec;
  try {
    rec = await api(`/api/chart/notes/${encodeURIComponent(nid)}`);
  } catch (e) {
    view.innerHTML = `<div class="empty">⚠ ${esc(e.message || e)} · <a href="#/notes">all notes</a></div>`;
    return;
  }
  const chart = rec.chart || {};
  const rng = rec.range || {};
  const emas = Array.isArray(chart.emas) ? chart.emas : [];
  const profiles = Array.isArray(chart.profiles) ? chart.profiles : [];
  const overlays = Array.isArray(chart.overlays) ? chart.overlays : [];
  const sar = chart.sar;
  const origins = chart.origins || {};
  const display = chart.display || {};
  const ovParams = chart.overlay_params || {};
  view.innerHTML = `
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">
        <div>
          <h2 style="margin:0 0 4px">${esc(rec.name || rec.id)}</h2>
          <div class="muted mono">${esc(fmtTs(rec.created_at))} · ${esc(rec.symbol || "—")} · ${esc(rec.instrument || "/NQ")}</div>
        </div>
        <div style="display:flex;gap:8px">
          <a href="#/notes">All notes</a>
          <a href="#/chart">Live Chart</a>
        </div>
      </div>
      <div class="section-title" style="margin-top:16px">Note</div>
      <p style="white-space:pre-wrap">${esc(rec.text || "(empty)")}</p>
      <div class="section-title">Selected range</div>
      <dl class="kv">
        <dt>From (CT)</dt><dd class="mono">${esc(rng.start_ct || fmtTs(rng.start) || "—")}</dd>
        <dt>To (CT)</dt><dd class="mono">${esc(rng.end_ct || fmtTs(rng.end) || "—")}</dd>
        <dt>UTC</dt><dd class="mono">${esc(rng.start || "—")} → ${esc(rng.end || "—")}</dd>
        ${rng.price_lo != null && rng.price_hi != null
          ? `<dt>Box</dt><dd class="mono">${esc(String(rng.price_lo))} – ${esc(String(rng.price_hi))}${rng.pane != null ? ` · pane ${esc(String(rng.pane))}` : ""}</dd>`
          : ""}
      </dl>
      <div class="section-title">Visible features &amp; parameters</div>
      <p class="muted">${esc(noteFeatureSummary(chart))}</p>
      <dl class="kv">
        <dt>Timeframe</dt><dd class="mono">${esc(chart.timeframe || "—")}</dd>
        <dt>Lookback</dt><dd class="mono">${esc(chart.lookback || "—")}</dd>
        <dt>Overlays</dt><dd class="mono">${esc(overlays.join(", ") || "—")}</dd>
        <dt>Origins</dt><dd class="mono">${esc(Object.keys(origins).length ? JSON.stringify(origins) : "—")}</dd>
        <dt>EMA</dt><dd class="mono">${esc(emas.length ? emas.map(e => `EMA${e.n}(${e.lookback})`).join(", ") : (chart.ema_enabled ? "on" : "off"))}</dd>
        <dt>Volume Profile</dt><dd class="mono">${esc(profiles.length ? profiles.map(p => p.label || `VP${p.n}`).join(", ") : (chart.vp_enabled ? "on" : "off"))}</dd>
        <dt>SAR</dt><dd class="mono">${esc(sar ? `start=${sar.start} increment=${sar.increment} maximum=${sar.maximum}` : "off")}</dd>
        <dt>Display</dt><dd class="mono">${esc(JSON.stringify(display))}</dd>
        <dt>Overlay params</dt><dd class="mono">${esc(Object.keys(ovParams).length ? JSON.stringify(ovParams) : "—")}</dd>
      </dl>
      ${profiles.length ? `<div class="section-title">VP specs</div><pre class="mono" style="white-space:pre-wrap;font-size:11px">${esc(JSON.stringify(profiles, null, 2))}</pre>` : ""}
      <div style="margin-top:16px">
        <button type="button" id="note-del-one">Delete note</button>
      </div>
    </div>`;
  $("#note-del-one")?.addEventListener("click", async () => {
    if (!window.confirm(`Delete note ${rec.id}?`)) return;
    try {
      await api(`/api/chart/notes/${encodeURIComponent(rec.id)}`, { method: "DELETE" });
      toast("Note deleted");
      location.hash = "#/notes";
    } catch (e) {
      toast(e.message || String(e), "err");
    }
  });
}

const ROUTES = [
  [/^#\/dashboard$/, () => viewDashboard(), "Dashboard", "dashboard"],
  [/^#\/chart$/, () => viewLiveChart(), "Live Chart", "chart"],
  [/^#\/border$/, () => viewLiveChart(), "Live Chart", "chart"],
  [/^#\/notes$/, () => viewChartNotes(), "Notes", "notes"],
  [/^#\/notes\/([^/?]+)$/, m => viewChartNoteDetail(m[1]), "Note", "notes"],
  [/^#\/model-design$/, () => viewModelDesign(), "Model design", "chart"],
  [/^#\/objectives$/, () => viewObjectives(), "Objectives", "objectives"],
  [/^#\/objectives\/new$/, () => viewObjectiveNew(), "New objective", "objectives"],
  [/^#\/objectives\/([^/?]+)\/model$/, m => viewObjectiveModel(m[1]), "Registered model", "objectives"],
  [/^#\/objectives\/([^/?]+)\/test$/, m => viewCustomTest(m[1]), "Custom test", "objectives"],
  [/^#\/objectives\/([^/?]+)$/, m => viewObjectiveDetail(m[1]), "Objective", "objectives"],
  [/^#\/scoping$/, (m, q) => viewScoping(q), "Scoping", "scoping"],
  [/^#\/scoping\/([^/?]+)\/([^/?]+)$/, (m, q) => viewScopingRun(m[1], m[2], q), "Scoping run", "scoping"],
  [/^#\/optimize$/, (m, q) => viewOptimize(q), "Optimization", "optimize"],
  [/^#\/optimize\/search\/([^/?]+)\/([^/?]+)$/, m => viewSearchRun(m[1], m[2]), "Search run", "optimize"],
  [/^#\/external$/, () => viewExternalModels(), "External Models", "external"],
  [/^#\/external\/([^/?]+)$/, m => viewExternalModels(m[1]), "External Models", "external"],
  [/^#\/backtest$/, (m, q) => viewBacktest(q), "Backtest", "backtest"],
  [/^#\/live$/, () => viewLiveStreamer(), "Live streamer", "live"],
  [/^#\/jobs$/, () => viewJobs(), "Jobs", "jobs"],
  [/^#\/jobs\/([^/?]+)$/, m => viewJobDetail(m[1]), "Job", "jobs"],
];

let _routeGen = 0;
async function route() {
  const gen = ++_routeGen;
  clearPoll();
  if (window.LCOrders && typeof LCOrders.closeDetail === "function") LCOrders.closeDetail();
  const hash = location.hash || "#/dashboard";
  const [path, qs] = hash.split("?");
  const params = new URLSearchParams(qs || "");
  const still = () => gen === _routeGen && view;
  const stuck = () => still() && view.querySelector && view.querySelector(".spin");
  let spinnerTimer = null;
  try {
    for (const [re, fn, crumb, nav] of ROUTES) {
      const m = path.match(re);
      if (!m) continue;
      const crumbEl = $("#crumb");
      if (crumbEl) crumbEl.textContent = crumb;
      if (!view) return;
      view.classList.remove("wide");
      view.classList.remove("wide-bt");
      view.classList.remove("chart-fill");
      const shellEl = document.getElementById("shell");
      if (shellEl) {
        shellEl.classList.remove("chart-mode");
        shellEl.classList.remove("lc-chart-expanded");
      }
      document.querySelectorAll("#sidebar a[data-nav]").forEach(a =>
        a.classList.toggle("active", a.dataset.nav === nav));
      view.innerHTML = '<div class="empty"><span class="spin">◐</span> loading…</div>';
      spinnerTimer = setTimeout(() => {
        if (!stuck()) return;
        view.innerHTML = `<div class="empty">⚠ still waiting on the server — hard-refresh (Ctrl+Shift+R). ${esc(path)}</div>`;
      }, 8000);
      await Promise.resolve(fn(m, params));
      return;
    }
    if (location.hash !== "#/dashboard") location.hash = "#/dashboard";
  } catch (e) {
    if (!still()) return;
    const msg = (e && e.message) ? e.message : String(e);
    view.innerHTML = `<div class="empty">⚠ ${esc(msg)}</div>`;
  } finally {
    if (spinnerTimer) clearTimeout(spinnerTimer);
  }
}

window.addEventListener("hashchange", route);
setInterval(() => { $("#clock").textContent = toCT(new Date().toISOString()) + " " + TZ_ABBR; }, 1000);
route();
(async () => {
  try {
    const o = await api("/api/overview");
    const pin = $("#active-objective");
    if (pin) pin.textContent = `config: ${o.active_objective || "—"}`;
    if ($("#mlflow-link")) $("#mlflow-link").href = o.mlflow_url;
  } catch {}
})();
