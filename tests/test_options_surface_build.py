"""Contract for the vectorised delta-bucket surface builder.

``build_surface_from_greeks`` used to loop over every second and call
``delta_bucket_row``, which cost ~155 s/day and 87% of its time in per-second
pandas indexing. The vectorised form must stay exactly equivalent to that loop,
so the loop is reproduced here as the reference oracle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.options_greeks import delta_bucket_row, n_legs_filled
from src.features.options_greeks_pipeline import (
    _SURFACE_DB_COLS,
    build_surface_from_greeks,
)


def _reference(greeks: pd.DataFrame) -> pd.DataFrame:
    """The original per-second loop, kept as the oracle."""
    g = greeks
    strike = g["strike"].to_numpy()
    mask = (np.isfinite(g["iv"].to_numpy())
            & (np.round(strike).astype(np.int64) % 25 == 0))
    g = g.loc[mask, [
        "ts", "strike", "is_call", "iv", "delta", "gamma", "vega", "theta",
        "mid", "spread", "nq_price", "underlying_sym", "expiry",
    ]]
    if g.empty:
        return pd.DataFrame(columns=_SURFACE_DB_COLS)
    rows = []
    for ts, grp in g.groupby("ts", sort=False):
        snap = grp.rename(columns={"strike": "strike_price"})
        row = delta_bucket_row(snap, nq_price=float(snap["nq_price"].iloc[0]))
        row["ts"] = ts
        row["expiry"] = snap["expiry"].iloc[0]
        row["underlying_sym"] = str(snap["underlying_sym"].iloc[0])
        row["n_legs_filled"] = n_legs_filled(row)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("ts")
    for c in _SURFACE_DB_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[_SURFACE_DB_COLS].reset_index(drop=True)


def _synthetic(n_sec=120, seed=3, strike_grid=25):
    """A chain that exercises off-grid strikes, NaN IV, and sparse sides."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-09-29 14:00:00", tz="UTC")
    recs = []
    for i in range(n_sec):
        ts = base + pd.Timedelta(seconds=i)
        spot = 30_000 + rng.normal(0, 20)
        # deliberately mix on-grid and off-grid strikes
        strikes = np.concatenate([
            np.arange(spot - 300, spot + 300, strike_grid).round(-2) // 25 * 25,
            np.array([spot + 7, spot - 13]),
        ])
        for k in strikes:
            for is_call in (True, False):
                d = rng.uniform(0.02, 0.98) * (1 if is_call else -1)
                iv = rng.uniform(0.05, 1.2)
                if rng.random() < 0.05:
                    iv = np.nan           # unsolvable leg
                recs.append({
                    "ts": ts, "strike": float(k), "is_call": is_call, "iv": iv,
                    "delta": d, "gamma": rng.random(), "vega": rng.random(),
                    "theta": -rng.random(), "mid": rng.random() * 50,
                    "spread": rng.random(), "nq_price": spot,
                    "underlying_sym": "NQZ6",
                    "expiry": pd.Timestamp("2026-09-29").date(),
                })
    return pd.DataFrame(recs)


def test_matches_the_per_second_loop_exactly():
    g = _synthetic()
    got, want = build_surface_from_greeks(g), _reference(g)
    assert list(got.columns) == list(want.columns)
    assert len(got) == len(want)
    for c in _SURFACE_DB_COLS:
        a, b = got[c].to_numpy(), want[c].to_numpy()
        if a.dtype.kind == "f":
            assert np.allclose(a, b, equal_nan=True), f"{c} differs"
        else:
            assert (a == b).all(), f"{c} differs"


def test_empty_input_returns_empty_with_schema():
    out = build_surface_from_greeks(pd.DataFrame(columns=["ts", "strike", "iv"]))
    assert out.empty and list(out.columns) == _SURFACE_DB_COLS


def test_all_strikes_off_grid_yields_no_rows():
    g = _synthetic(n_sec=5)
    g["strike"] = g["strike"] + 7          # nothing on the 25-point lattice
    assert build_surface_from_greeks(g).empty


def test_n_legs_filled_counts_non_nan_iv_legs():
    g = _synthetic(n_sec=30)
    out = build_surface_from_greeks(g)
    leg_iv = [c for c in out.columns if c.endswith("_iv")]
    assert len(leg_iv) == 8
    assert (out["n_legs_filled"] == out[leg_iv].notna().sum(axis=1)).all()
    assert out["n_legs_filled"].between(0, 8).all()


def test_one_sided_chain_leaves_the_other_side_nan():
    g = _synthetic(n_sec=20)
    g = g[g["is_call"]]                    # calls only
    out = build_surface_from_greeks(g)
    assert out[[c for c in out.columns if c.startswith("p")and c.endswith("_iv")]].isna().all().all()
    assert out["n_legs_filled"].max() <= 4


# ── live per-second path ────────────────────────────────────────────────────

def _live_meta(underlying="NQZ6"):
    """Two ATM-ish legs on one 0DTE chain, as _opt_meta would hold them."""
    import datetime as dt
    exp = dt.datetime(2026, 9, 30, 20, 0, tzinfo=dt.timezone.utc)
    meta = {}
    for i, (strike, is_call) in enumerate(
        [(k, c) for k in (30_500.0, 30_525.0, 30_550.0) for c in (True, False)]
    ):
        meta[100 + i] = {
            "symbol": f"QNEU6 {'C' if is_call else 'P'}{int(strike)}",
            "strike": strike, "is_call": is_call,
            "expiry": dt.date(2026, 9, 30), "expiration": exp,
            "underlying": underlying,
        }
    return meta


def test_live_second_prices_each_option_against_its_own_underlying():
    """A mapping of futures mids must be resolved per option, not collapsed.

    The streamer used to track one "NQ mid" overwritten by whichever outright
    quoted last, so a chain on NQZ6 got priced against NQH7 — hundreds of points
    away. Passing both contracts must still select NQZ6.
    """
    import datetime as dt
    from src.features.options_greeks_pipeline import compute_second_batch

    meta = _live_meta("NQZ6")
    quotes = pd.DataFrame([
        {"instrument_id": iid, "bid_px": 20.0, "ask_px": 21.0} for iid in meta
    ])
    ts = pd.Timestamp("2026-09-30 15:30:00", tz="UTC")
    greeks, row = compute_second_batch(
        quotes, meta,
        {"NQZ6": 30_520.0, "NQH7": 30_890.0},   # 370 points apart
        "NQH7",                                  # the stale "last seen" symbol
        ts,
    )
    assert not greeks.empty
    assert set(greeks["underlying_sym"]) == {"NQZ6"}
    assert (greeks["nq_price"] == 30_520.0).all()
    if row is not None:
        assert row["underlying_sym"] == "NQZ6"


def test_live_second_still_accepts_a_scalar_price():
    """Legacy scalar form keeps working for a single-underlying chain."""
    import datetime as dt
    from src.features.options_greeks_pipeline import compute_second_batch

    meta = _live_meta("NQZ6")
    quotes = pd.DataFrame([
        {"instrument_id": iid, "bid_px": 20.0, "ask_px": 21.0} for iid in meta
    ])
    greeks, _ = compute_second_batch(
        quotes, meta, 30_520.0, "NQZ6", pd.Timestamp("2026-09-30 15:30:00", tz="UTC")
    )
    assert not greeks.empty and (greeks["nq_price"] == 30_520.0).all()


def test_live_second_rejects_the_undef_price_sentinel():
    """UNDEF_PRICE is positive and must not be accepted as an underlying mid."""
    from src.features.options_greeks_pipeline import compute_second_batch

    meta = _live_meta("NQZ6")
    quotes = pd.DataFrame([
        {"instrument_id": iid, "bid_px": 20.0, "ask_px": 21.0} for iid in meta
    ])
    greeks, row = compute_second_batch(
        quotes, meta, {"NQZ6": 9223372036.854776}, "NQZ6",
        pd.Timestamp("2026-09-30 15:30:00", tz="UTC"),
    )
    assert greeks.empty and row is None
