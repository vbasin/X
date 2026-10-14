"""Hump test: raw feature densities on target vs stop at decision time.

SHAP answers "what does the booster use?"
The hump test answers "do the raw X values actually separate target from stop?"

It is *not* a replacement for XGB. It is a univariate filter that runs
alongside Stage-3 SHAP so you can see when model credit and raw separation
disagree (n_bounds-style overlap is the common case).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.evaluation.hump_test")

DEFAULT_CONTEXT = Path("/opt/vtech/data/evaluation/context_analysis")
LEAK_SUBSTR = (
    "tgt_dist", "mfe", "mae", "pnl", "r_mult", "hold_sec", "exit_",
    "label", "y_true", "outcome", "realized", "future_",
)
PASS_AUROC = 0.58
PASS_D = 0.35
PASS_KS = 0.12


def _is_leak(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in LEAK_SUBSTR)


def infer_binary_y(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return y in {0,1} and a short description of how it was inferred.

    1 = target / favorable tape, 0 = stop / adverse tape.
    """
    cols = {c.lower(): c for c in frame.columns}

    for key in ("y_move", "y_true", "label", "target", "hit_target"):
        if key in cols:
            s = frame[cols[key]]
            if s.dtype == bool or set(pd.unique(s.dropna())) <= {0, 1, 0.0, 1.0, True, False}:
                y = s.astype(float).fillna(0).to_numpy()
                return (y >= 0.5).astype(int), f"column:{cols[key]}"

    for key in ("reason", "exit_reason", "outcome", "result"):
        if key in cols:
            s = frame[cols[key]].astype(str).str.lower()
            tgt = s.str.contains("target|tp|win|hit", regex=True)
            stp = s.str.contains("stop|sl|loss|fail", regex=True)
            mask = tgt | stp
            if mask.sum() >= 50:
                y = np.full(len(frame), -1, dtype=int)
                y[tgt.to_numpy()] = 1
                y[stp.to_numpy()] = 0
                return y, f"reason:{cols[key]}"

    if "direction" in cols and "confidence" in cols:
        raise ValueError(
            "no tape-outcome column found (reason/outcome/label). "
            "Refusing to use model predictions as y — that is SHAP, not a hump test."
        )

    raise ValueError(
        f"cannot infer binary target vs stop from columns: {list(frame.columns)[:40]}"
    )


def _auroc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    y_s = y[order]
    n1 = float(y_s.sum())
    n0 = float(len(y_s) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    pos = ranks[y == 1].sum()
    u = pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    xa = np.sort(a)
    xb = np.sort(b)
    grid = np.concatenate([xa, xb])
    fa = np.searchsorted(xa, grid, side="right") / len(xa)
    fb = np.searchsorted(xb, grid, side="right") / len(xb)
    return float(np.max(np.abs(fa - fb)))


def score_feature(x: np.ndarray, y: np.ndarray, name: str) -> dict[str, Any]:
    mask = np.isfinite(x) & (y >= 0)
    xx, yy = x[mask], y[mask]
    pos, neg = xx[yy == 1], xx[yy == 0]
    auroc = _auroc(xx, yy) if len(xx) else 0.5
    auroc = max(auroc, 1.0 - auroc)
    d = abs(_cohen_d(pos, neg))
    ks = _ks(pos, neg)
    leak = _is_leak(name)
    passed = (not leak) and (auroc >= PASS_AUROC or d >= PASS_D or ks >= PASS_KS)
    return {
        "feature": name,
        "n": int(len(xx)),
        "n_target": int(len(pos)),
        "n_stop": int(len(neg)),
        "mean_target": float(pos.mean()) if len(pos) else float("nan"),
        "mean_stop": float(neg.mean()) if len(neg) else float("nan"),
        "auroc": round(auroc, 4),
        "cohen_d": round(d, 4),
        "ks": round(ks, 4),
        "leak": leak,
        "pass": passed,
    }


def hump_table(frame: pd.DataFrame, y: np.ndarray, cols: Iterable[str] | None = None) -> pd.DataFrame:
    use = list(cols) if cols is not None else [
        c for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c])
    ]
    rows = []
    for c in use:
        try:
            x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        rows.append(score_feature(x, y, c))
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    return tbl.sort_values(["pass", "auroc", "ks"], ascending=[False, False, False]).reset_index(drop=True)


def plot_hump(x: np.ndarray, y: np.ndarray, name: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = np.isfinite(x) & (y >= 0)
    xx, yy = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 40
    ax.hist(xx[yy == 1], bins=bins, density=True, alpha=0.55, label="target", color="#2ecc71")
    ax.hist(xx[yy == 0], bins=bins, density=True, alpha=0.55, label="stop", color="#e74c3c")
    ax.set_title(name)
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_grid(frame: pd.DataFrame, y: np.ndarray, features: list[str], out: Path, max_n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = features[:max_n]
    if not feats:
        return
    n_cols = 3
    rows = int(math.ceil(len(feats) / n_cols))
    fig, axes = plt.subplots(rows, n_cols, figsize=(4 * n_cols, 3.2 * rows), squeeze=False)
    for i, name in enumerate(feats):
        ax = axes[i // n_cols][i % n_cols]
        x = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & (y >= 0)
        xx, yy = x[mask], y[mask]
        ax.hist(xx[yy == 1], bins=30, density=True, alpha=0.55, color="#2ecc71", label="target")
        ax.hist(xx[yy == 0], bins=30, density=True, alpha=0.55, color="#e74c3c", label="stop")
        ax.set_title(name, fontsize=9)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(feats), rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def write_report(tbl: pd.DataFrame, dest: Path, source: str, y_desc: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(dest / "hump_metrics.csv", index=False)
    passed = tbl[tbl["pass"] & ~tbl["leak"]] if not tbl.empty else tbl
    leaked = tbl[tbl["leak"]] if not tbl.empty else tbl
    lines = [
        "# Hump test — target vs stop (raw X)",
        "",
        f"Source: `{source}`",
        f"Label rule: `{y_desc}`",
        "",
        "This is **not** SHAP. SHAP is model credit. These densities are the raw "
        "feature on the frozen tape outcome at decision time.",
        "",
        f"Pass rule: AUROC ≥ {PASS_AUROC} **or** |d| ≥ {PASS_D} **or** KS ≥ {PASS_KS}, and not a leak column.",
        "",
        f"Features scored: **{len(tbl)}**. Passed: **{len(passed)}**. Leak-flagged: **{len(leaked)}**.",
        "",
        "## Top by AUROC",
        "",
        "| feature | auroc | cohen_d | ks | n_target | n_stop | pass | leak |",
        "|---------|-------|---------|----|----------|--------|------|------|",
    ]
    for _, r in tbl.head(25).iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['auroc']:.3f} | {r['cohen_d']:.3f} | {r['ks']:.3f} | "
            f"{int(r['n_target'])} | {int(r['n_stop'])} | {bool(r['pass'])} | {bool(r['leak'])} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- Overlapping humps (example: `n_bounds`) → the raw feature does not separate target from stop.",
        "- A feature can have high SHAP and fail the hump test: the tree uses it as a splitter, the tape does not care.",
        "- Leak columns (`tgt_dist`, `mfe`, …) are scored then flagged; do not promote them.",
        "",
    ]
    md = dest / "HUMP_REPORT.md"
    md.write_text("\n".join(lines))
    return md


def _load_events(path: Path) -> pd.DataFrame:
    if path.is_file():
        if path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix in {".csv", ".tsv"}:
            return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        raise ValueError(f"unsupported file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    cands = []
    for pat in ("*.parquet", "*.pq", "*.csv"):
        cands.extend(sorted(path.rglob(pat)))
    prefer = [p for p in cands if any(s in p.name.lower() for s in ("event", "context", "stage3", "analysis"))]
    pool = prefer or cands
    if not pool:
        raise FileNotFoundError(f"no parquet/csv under {path}")
    frames = []
    for p in pool[:20]:
        try:
            frames.append(pd.read_parquet(p) if p.suffix != ".csv" else pd.read_csv(p))
        except Exception as exc:
            logger.warning("skip %s: %s", p, exc)
    if not frames:
        raise FileNotFoundError(f"unreadable tables under {path}")
    return pd.concat(frames, ignore_index=True, sort=False)


def run_hump_on_frame(
    frame: pd.DataFrame,
    *,
    dest: Path,
    source: str = "frame",
    cols: Iterable[str] | None = None,
    y: np.ndarray | None = None,
) -> dict[str, Any]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if y is None:
        y, y_desc = infer_binary_y(frame)
    else:
        y_desc = "provided"
    keep = y >= 0
    frame_k = frame.loc[keep].reset_index(drop=True)
    y_k = y[keep]
    tbl = hump_table(frame_k, y_k, cols=cols)
    write_report(tbl, dest, source, y_desc)
    top = tbl.loc[~tbl["leak"], "feature"].head(12).tolist() if not tbl.empty else []
    if top:
        plot_grid(frame_k, y_k, top, dest / "hump_grid.png")
        feat_dir = dest / "features"
        feat_dir.mkdir(exist_ok=True)
        for name in top:
            x = pd.to_numeric(frame_k[name], errors="coerce").to_numpy(dtype=float)
            plot_hump(x, y_k, name, feat_dir / f"{name}.png")
    meta = {
        "source": source,
        "y_desc": y_desc,
        "n_rows": int(len(frame_k)),
        "n_target": int((y_k == 1).sum()),
        "n_stop": int((y_k == 0).sum()),
        "n_pass": int(tbl["pass"].sum()) if not tbl.empty else 0,
        "dest": str(dest),
    }
    (dest / "hump_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("hump_wrote dest=%s pass=%s", dest, meta["n_pass"])
    return meta


def run_alongside_shap(
    X_raw: pd.DataFrame,
    preds: pd.DataFrame | None,
    cols: list[str],
    idx,
    *,
    dest: Path | None = None,
    events: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Called from shap_explain after `_write_plots`.

    Prefers an events/context frame with a real tape outcome. If only
    predictions are available, refuse to invent labels — write a stub note.
    """
    out = Path(dest) / "HUMP" if dest is not None else Path("HUMP")
    try:
        if events is not None:
            return run_hump_on_frame(events, dest=out, source="shap_events", cols=cols)
        if preds is not None:
            extra = [c for c in preds.columns if c.lower() in {
                "reason", "exit_reason", "outcome", "result", "y", "y_true", "label", "hit_target",
            }]
            if extra:
                joined = X_raw.copy()
                for c in extra:
                    joined[c] = preds[c].to_numpy()
                return run_hump_on_frame(joined, dest=out, source="shap_preds_joined", cols=cols)
        ctx = DEFAULT_CONTEXT
        if ctx.exists():
            ev = _load_events(ctx)
            return run_hump_on_frame(ev, dest=out, source=str(ctx), cols=None)
        out.mkdir(parents=True, exist_ok=True)
        (out / "HUMP_REPORT.md").write_text(
            "# Hump test skipped\n\nNo tape-outcome column on the SHAP matrix and "
            f"`{DEFAULT_CONTEXT}` is missing. Run `python -m scripts.run_hump_test --events <events.parquet>`.\n"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("hump_alongside_skip: %s", exc)
        out.mkdir(parents=True, exist_ok=True)
        (out / "HUMP_REPORT.md").write_text(f"# Hump test failed\n\n`{exc}`\n")
        return None


def _cli() -> None:
    p = argparse.ArgumentParser(description="Retrospective hump test (target vs stop densities)")
    p.add_argument("--events", default=str(DEFAULT_CONTEXT),
                   help="Events parquet/csv or a directory (default: context_analysis)")
    p.add_argument("--out", default="",
                   help="Artifact dir (default: <events>/HUMP or ./HUMP)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    src = Path(args.events)
    frame = _load_events(src)
    dest = Path(args.out) if args.out else (src / "HUMP" if src.is_dir() else src.parent / "HUMP")
    meta = run_hump_on_frame(frame, dest=dest, source=str(src))
    print(f"hump test wrote {meta['n_rows']} rows -> {dest}")
    print(f"   report: {dest / 'HUMP_REPORT.md'}")


if __name__ == "__main__":
    _cli()
