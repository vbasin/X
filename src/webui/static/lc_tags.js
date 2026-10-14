/* Live Chart helpers for compact scale tags and bound-cross marks.
   Classic script, loaded before app.js. No DOM. */

function lcNormalizeOscBounds(upper, lower) {
  function clamp(v, fallback) {
    if (v == null || v === "") return fallback;
    let n = Number(v);
    if (!Number.isFinite(n)) n = fallback;
    n = Math.round(n * 10) / 10;
    if (n < 0) n = 0;
    if (n > 100) n = 100;
    return n;
  }
  let u = clamp(upper, 80);
  let l = clamp(lower, 20);
  if (!(l < u)) {
    if (u <= 0) {
      u = 1;
      l = 0;
    } else if (l >= 100) {
      u = 100;
      l = 99;
    } else {
      l = Math.round((u - 0.1) * 10) / 10;
    }
  }
  return { upper: u, lower: l };
}

/** Down through upper, or up through lower. A gap in the series resets the cross. */
function lcSeriesBoundCrosses(points, upper, lower) {
  const out = [];
  if (!points || upper == null || lower == null) return out;
  let prev = null;
  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    const v = pt && pt.value;
    if (v == null || !Number.isFinite(v)) {
      prev = null;
      continue;
    }
    if (prev != null) {
      if (prev > upper && v <= upper) out.push({ time: pt.time, value: v, dir: "down" });
      else if (prev < lower && v >= lower) out.push({ time: pt.time, value: v, dir: "up" });
    }
    prev = v;
  }
  return out;
}

/** Price close crossing a pair of band series (VWAP ±2σ). */
function lcPriceBandCrosses(candles, uppers, lowers) {
  const out = [];
  const u = new Map();
  const l = new Map();
  const ups = uppers || [];
  const los = lowers || [];
  for (let i = 0; i < ups.length; i++) {
    const p = ups[i];
    if (p && p.value != null && Number.isFinite(p.value)) u.set(p.time, p.value);
  }
  for (let i = 0; i < los.length; i++) {
    const p = los[i];
    if (p && p.value != null && Number.isFinite(p.value)) l.set(p.time, p.value);
  }
  let prev = null;
  const rows = candles || [];
  for (let i = 0; i < rows.length; i++) {
    const c = rows[i];
    if (!c || c.close == null || !Number.isFinite(c.close)) {
      prev = null;
      continue;
    }
    const hi = u.get(c.time);
    const lo = l.get(c.time);
    if (prev != null) {
      if (hi != null && prev > hi && c.close <= hi) {
        out.push({ time: c.time, value: c.close, dir: "down" });
      } else if (lo != null && prev < lo && c.close >= lo) {
        out.push({ time: c.time, value: c.close, dir: "up" });
      }
    }
    prev = c.close;
  }
  return out;
}

/** Last plotted number from a Lightweight Charts bar, or null. null is not an object here. */
function lcFiniteSeriesValue(raw) {
  if (raw == null || raw === "") return null;
  if (typeof raw === "number") return Number.isFinite(raw) ? raw : null;
  if (typeof raw === "object") {
    if (raw.value != null && Number.isFinite(Number(raw.value))) return Number(raw.value);
    if (raw.close != null && Number.isFinite(Number(raw.close))) return Number(raw.close);
  }
  return null;
}

/**
 * Place one-line scale tags inside a pane without overlapping.
 * Lower priority is placed first and keeps the anchor (the price tag).
 */
function lcPackScaleTags(items, paneTop, paneBottom, lineH) {
  const h = Math.max(8, lineH || 12);
  const top = Number(paneTop) || 0;
  const bottom = Math.max(top + h, Number(paneBottom) || 0);
  const placed = [];
  function overlaps(y) {
    const y2 = y + h;
    for (let i = 0; i < placed.length; i++) {
      const p = placed[i];
      if (!(y2 <= p.top || p.top + h <= y)) return true;
    }
    return false;
  }
  const sorted = (items || []).slice().sort((a, b) => {
    const pa = a && a.priority != null ? a.priority : 1;
    const pb = b && b.priority != null ? b.priority : 1;
    if (pa !== pb) return pa - pb;
    return (a.y || 0) - (b.y || 0);
  });
  for (let i = 0; i < sorted.length; i++) {
    const it = sorted[i];
    if (!it || it.y == null || !Number.isFinite(it.y)) continue;
    const anchor = Math.max(top, Math.min(bottom - h, it.y - h / 2));
    let best = null;
    let bestD = Infinity;
    for (let y = top; y <= bottom - h + 0.01; y += 1) {
      if (overlaps(y)) continue;
      const d = Math.abs(y - anchor);
      if (d < bestD) {
        bestD = d;
        best = y;
      }
      if (d === 0) break;
    }
    if (best == null) best = anchor;
    placed.push({
      key: it.key,
      text: it.text,
      color: it.color,
      y: it.y,
      top: best,
      nudged: Math.abs((best + h / 2) - it.y) > 2,
    });
  }
  return placed;
}
