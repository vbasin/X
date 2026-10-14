"""Pure helpers for Live Chart scale tags and bound-cross arrows."""

import json
from pathlib import Path

import quickjs

ROOT = Path(__file__).resolve().parents[1]
JS = r"""
const crosses = lcSeriesBoundCrosses([
  {time:1, value:70},
  {time:2, value:85},
  {time:3, value:80},
  {time:4, value:10},
  {time:5, value:25},
], 80, 20);
if (crosses.length !== 2) throw new Error("cross count " + crosses.length);
if (crosses[0].dir !== "down" || crosses[0].time !== 3) throw new Error("down");
if (crosses[1].dir !== "up" || crosses[1].time !== 5) throw new Error("up");

const jumpDn = lcSeriesBoundCrosses([{time:1, value:90}, {time:2, value:10}], 80, 20);
if (jumpDn.length !== 1 || jumpDn[0].dir !== "down") throw new Error("jump down");
const jumpUp = lcSeriesBoundCrosses([{time:1, value:10}, {time:2, value:90}], 80, 20);
if (jumpUp.length !== 1 || jumpUp[0].dir !== "up") throw new Error("jump up");

const gap = lcSeriesBoundCrosses([
  {time:1, value:90}, {time:2, value:null}, {time:3, value:10},
], 80, 20);
if (gap.length !== 0) throw new Error("gap should reset");

const bands = lcPriceBandCrosses(
  [{time:1, close:90}, {time:2, close:100}, {time:3, close:110}, {time:4, close:100}],
  [{time:1, value:105}, {time:2, value:105}, {time:3, value:105}, {time:4, value:105}],
  [{time:1, value:95}, {time:2, value:95}, {time:3, value:95}, {time:4, value:95}],
);
if (bands.length !== 2 || bands[0].dir !== "up" || bands[1].dir !== "down") {
  throw new Error("bands " + JSON.stringify(bands));
}

const packed = lcPackScaleTags([
  {key:"a", text:"1", color:"#fff", y:100, priority:1},
  {key:"b", text:"2", color:"#fff", y:100, priority:1},
  {key:"px", text:"3", color:"#fff", y:100, priority:0},
], 0, 400, 14);
if (packed.length !== 3) throw new Error("pack len");
const tops = packed.map(p => p.top).sort((a, b) => a - b);
for (let i = 1; i < tops.length; i++) {
  if (tops[i] < tops[i - 1] + 14 - 0.01) throw new Error("overlap");
}
const px = packed.find(p => p.key === "px");
if (Math.abs((px.top + 7) - 100) > 1) throw new Error("px anchor");

const swapped = lcNormalizeOscBounds(20, 80);
if (!(swapped.lower < swapped.upper)) throw new Error("order");
const defaults = lcNormalizeOscBounds("nope", null);
if (defaults.upper !== 80 || defaults.lower !== 20) throw new Error(JSON.stringify(defaults));

if (lcFiniteSeriesValue(null) !== null) throw new Error("null is not a bar");
if (lcFiniteSeriesValue({value: 12.5}) !== 12.5) throw new Error("line value");
if (lcFiniteSeriesValue({open: 1, close: 9}) !== 9) throw new Error("candle close");
if (lcFiniteSeriesValue({time: 1}) !== null) throw new Error("whitespace");
"ok";
"""


def test_lc_tags_helpers():
    ctx = quickjs.Context()
    ctx.eval((ROOT / "src/webui/static/lc_tags.js").read_text())
    assert ctx.eval(JS) == "ok"


def test_app_js_parses():
    ctx = quickjs.Context()
    try:
        ctx.eval((ROOT / "src/webui/static/app.js").read_text())
    except quickjs.JSException as exc:
        assert "document" in str(exc)
    else:
        raise AssertionError("app.js ran without a DOM, expected a document reference")
