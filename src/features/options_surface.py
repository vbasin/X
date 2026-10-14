"""Options IV surface features computed from BBO-1s options + definitions.

Includes: ATM IV, put-call skew, term structure slope, butterfly spread,
IV velocity, GEX proxy (gamma exposure), 25-delta risk reversal.

v3: Uses bar_windows(). Added iv_25d_risk_reversal, iv_term_slope,
    iv_25d_butterfly. Doubled all rolling windows.
v4: NQE 30-day constant-maturity interpolation. iv_atm_mean and all
    surface features (iv_skew, iv_butterfly, iv_25d_risk_reversal,
    iv_25d_butterfly) are computed from the next two QNE (end-of-month)
    option chains and interpolated to 30-day constant maturity using
    VIX-style variance weighting. iv_term_slope redefined as 30d CM-IV
    minus iv_1dte_atm. New feature: iv_1dte_atm (ATM IV for options
    expiring on the next trading day). iv_0dte_atm unchanged.
    Activates fully once 1DTE BBO data is ingested; iv_1dte_atm and
    iv_term_slope are NaN until then.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from src.common.bar_windows import bars, bars_m, get_timestep

logger = logging.getLogger("vtech.features.options_surface")

# ``ndtr`` is the C routine ``scipy.stats.norm.cdf`` itself delegates to, minus the
# rv_continuous argument-validation wrapper: bit-identical output, ~2.1x faster.
# The PDF closed form is ~2.7x faster than ``norm.pdf`` and agrees to ~1e-17. Both
# are evaluated 3x per Newton iteration over millions of rows, so this is the hot
# loop of every options build.
_INV_SQRT_2PI = 0.3989422804014327


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard-normal PDF. Drop-in for ``scipy.stats.norm.pdf``, much faster."""
    return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


# ── Black-Scholes IV inversion ──────────────────────────────────


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Compute Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if is_call else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1)


def _implied_vol(
    mid: float, S: float, K: float, T: float, r: float, is_call: bool,
    tol: float = 1e-6, max_iter: int = 50,
) -> float | None:
    """Newton-Raphson IV solver. Returns None on failure."""
    if T <= 0 or mid <= 0:
        return None
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        price = _bs_price(S, K, T, r, sigma, is_call)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            return None
        sigma -= (price - mid) / vega
        if sigma <= 0:
            return None
        if abs(price - mid) < tol:
            return sigma
    return None


def _implied_vol_vec(
    mid: np.ndarray, S: float | np.ndarray, K: np.ndarray, T: np.ndarray,
    r: float, is_call: np.ndarray,
    tol: float = 1e-5, max_iter: int = 30, sigma_tol: float = 1e-3,
) -> np.ndarray:
    """Vectorized Newton-Raphson IV solver — NaN for every row that did not solve.

    ``S`` may be a scalar (one snapshot, one underlying price) or a per-row array
    (a batch of seconds). This is the single canonical IV inversion for the
    repository; :mod:`src.features.options_greeks` delegates to it so that the
    historical backfill and the live streamer cannot drift apart.

    A row survives two tests. Its final sigma must reprice the option to within
    ``tol``, and the quote must actually *identify* a vol. Newton pins sigma down
    only to roughly ``tol / vega``, so a row is rejected unless its vega clears
    ``tol / sigma_tol``. ``sigma_tol`` (default 0.001, a tenth of a vol point) is
    therefore a first-order error budget, not a hard bound — it holds tightly when
    small and degrades as it is loosened, because the price-vol relation is only
    locally linear. Without this second test deep ITM/OTM legs pass the residual
    check trivially — their price barely moves with vol, so almost any sigma
    reprices correctly — and the solver reports the 0.3 seed as though it were an IV.

    Rows that exhaust ``max_iter`` return NaN, never a half-converged iterate.
    """
    mid = np.asarray(mid, dtype=np.float64)
    n = mid.size
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if S.ndim == 0:
        S = np.full(n, float(S))

    sigma = np.full(n, 0.3)
    active = (T > 0) & (mid > 0) & (K > 0) & (S > 0) & np.isfinite(S)
    sqrt_T = np.sqrt(np.where(T > 0, T, 1.0))
    sign = np.where(is_call, 1.0, -1.0)
    discount_K = K * np.exp(-r * T)
    log_SK = np.log(np.where((S > 0) & (K > 0), S / np.where(K > 0, K, 1.0), 1.0))

    for _ in range(max_iter):
        if not active.any():
            break
        d1 = (log_SK + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        # Unified call/put price: sign*(S*N(sign*d1) - K*e^-rT*N(sign*d2))
        price = sign * (S * ndtr(sign * d1) - discount_K * ndtr(sign * d2))
        vega = S * sqrt_T * _norm_pdf(d1)
        diff = price - mid
        # Converged rows become inactive
        converged = np.abs(diff) < tol
        active = active & ~converged & (vega > 1e-12)
        step = np.where(active, diff / np.where(vega > 1e-12, vega, 1.0), 0.0)
        sigma = sigma - step
        # Negative sigma → mark invalid
        invalid = sigma <= 0
        sigma = np.where(invalid, np.nan, sigma)
        active = active & ~invalid

    # Accept only rows that genuinely solved. The previous (0, 5) clamp was not a
    # convergence test: a row that went inactive because vega underflowed kept the
    # 0.3 seed, and a row that exhausted max_iter kept whatever its last Newton
    # step produced — both passed the clamp and were written as if they were IVs.
    # On 0DTE days where ADR-151's one-hour DTE error collapsed vega this reached
    # 24% of rows, all of them the literal value 0.3000.
    ok = np.isfinite(sigma) & (sigma > 0) & (sigma < 5.0)
    probe = np.where(ok, sigma, 1.0)          # placeholder keeps the math finite
    d1 = (log_SK + (r + 0.5 * probe**2) * T) / (probe * sqrt_T)
    d2 = d1 - probe * sqrt_T
    repriced = sign * (S * ndtr(sign * d1) - discount_K * ndtr(sign * d2))
    vega = S * sqrt_T * _norm_pdf(d1)
    ok &= np.abs(repriced - mid) < tol
    ok &= vega >= (tol / sigma_tol)
    return np.where(ok, sigma, np.nan)


def _bs_delta_vec(
    S: float, K: np.ndarray, T: np.ndarray, r: float,
    sigma: np.ndarray, is_call: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes delta."""
    sqrt_T = np.sqrt(np.where(T > 0, T, 1.0))
    d1 = (np.log(S / np.where(K > 0, K, 1.0)) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)


# ── Chain selection and interpolation helpers ──────────────────


def _next_trading_day(d, holidays: frozenset):
    """Return the nearest CME trading day strictly after d (date → date | None)."""
    candidate = d + timedelta(days=1)
    for _ in range(14):
        if candidate.weekday() < 5 and candidate not in holidays:
            return candidate
        candidate += timedelta(days=1)
    return None


def _chain_stats(surf: pd.DataFrame) -> dict:
    """Extract iv_atm, iv_skew, iv_butterfly, iv_25d_rr, iv_25d_bf, dte_days
    from a single-expiry surface slice.
    """
    if surf.empty:
        return {
            "iv_atm": np.nan, "iv_skew": np.nan, "iv_butterfly": np.nan,
            "iv_25d_rr": np.nan, "iv_25d_bf": np.nan, "dte_days": np.nan,
        }

    calls = surf[surf["instrument_class"] == "C"]
    puts  = surf[surf["instrument_class"] == "P"]

    # ATM IV: 5 nearest-to-money options
    atm = surf.iloc[(surf["moneyness"].abs()).argsort()[:5]]
    iv_atm = float(atm["iv"].mean())

    # Put-call skew (OTM put IV − OTM call IV)
    otm_puts  = puts[puts["moneyness"] < -0.02]
    otm_calls = calls[calls["moneyness"] > 0.02]
    iv_skew = (
        float(otm_puts["iv"].mean()) - float(otm_calls["iv"].mean())
        if len(otm_puts) and len(otm_calls) else np.nan
    )

    # Butterfly: wings vs ATM
    wings_iv_raw = (
        pd.concat([otm_puts, otm_calls])["iv"].mean()
        if len(otm_puts) + len(otm_calls) else np.nan
    )
    iv_butterfly = float(wings_iv_raw) - iv_atm if not np.isnan(wings_iv_raw) else np.nan

    # 25-delta risk reversal and butterfly
    puts_25d  = puts[(puts["delta"] > -0.35)  & (puts["delta"] < -0.15)]
    calls_25d = calls[(calls["delta"] > 0.15) & (calls["delta"] < 0.35)]
    if len(puts_25d) and len(calls_25d):
        iv_25d_rr = float(calls_25d["iv"].mean()) - float(puts_25d["iv"].mean())
        iv_25d_bf = (
            (float(calls_25d["iv"].mean()) + float(puts_25d["iv"].mean())) / 2 - iv_atm
        )
    else:
        iv_25d_rr = np.nan
        iv_25d_bf = np.nan

    dte_days = float(surf["dte"].median()) * 365.25

    return {
        "iv_atm": iv_atm, "iv_skew": iv_skew, "iv_butterfly": iv_butterfly,
        "iv_25d_rr": iv_25d_rr, "iv_25d_bf": iv_25d_bf, "dte_days": dte_days,
    }


def _interp_iv_30d(
    iv1: float, iv2: float, t1: float, t2: float, target: float = 30.0
) -> float:
    """Variance-weighted constant-maturity IV interpolation (VIX-style).

    Interpolates/extrapolates to a constant 'target' days-to-expiry using
    variance weighting: σ_T² ∝ total variance = σ² × T.
    Both interpolation (T1 < target < T2) and extrapolation are handled;
    falls back to linear IV if the blended variance would be negative.
    """
    if np.isnan(iv1) and np.isnan(iv2):
        return np.nan
    if np.isnan(iv2):
        return float(iv1)
    if np.isnan(iv1):
        return float(iv2)
    if abs(t2 - t1) < 0.5:          # essentially same expiry
        return float(iv1)
    w1 = (t2 - target) / (t2 - t1)
    w2 = (target - t1) / (t2 - t1)
    var_30 = w1 * iv1 ** 2 + w2 * iv2 ** 2
    if var_30 > 0:
        return float(np.sqrt(var_30))
    # Negative only when extrapolating far past both chains — fall back to linear
    return float(max(w1 * iv1 + w2 * iv2, 0.0))


def _interp_linear(
    v1: float, v2: float, t1: float, t2: float, target: float = 30.0
) -> float:
    """Linear interpolation/extrapolation to constant 'target' days DTE.

    Used for skew, butterfly, and 25-delta features where linear blending
    is more appropriate than variance weighting.
    """
    if np.isnan(v1) and np.isnan(v2):
        return np.nan
    if np.isnan(v2):
        return float(v1)
    if np.isnan(v1):
        return float(v2)
    if abs(t2 - t1) < 0.5:
        return float(v1)
    w1 = (t2 - target) / (t2 - t1)
    w2 = (target - t1) / (t2 - t1)
    return float(w1 * v1 + w2 * v2)


# ── Surface builder ─────────────────────────────────────────────


def _build_surface(
    bbo_opts: pd.DataFrame,
    defs: pd.DataFrame,
    underlying_price: float,
    risk_free: float = 0.05,
    max_spread_pct: float = 0.25,
    min_price: float = 0.25,
    max_abs_moneyness: float = 0.15,
) -> pd.DataFrame:
    """Compute IV + delta for each option instrument at a single timestamp (vectorized)."""
    if bbo_opts.empty:
        return pd.DataFrame()

    # If defs already merged (columns present), skip the merge
    if "strike_price" not in bbo_opts.columns:
        merged = bbo_opts.merge(
            defs[["symbol", "strike_price", "expiration", "instrument_class", "multiplier"]],
            on="symbol",
            how="inner",
        )
    else:
        merged = bbo_opts
    if merged.empty:
        return pd.DataFrame()

    merged = merged.copy()
    merged["mid"] = (merged["bid_px"] + merged["ask_px"]) / 2
    merged["spread_pct"] = (merged["ask_px"] - merged["bid_px"]) / merged["mid"].replace(0, np.nan)
    merged["moneyness"] = np.log(merged["strike_price"].replace(0, np.nan) / underlying_price)

    # Quality + moneyness prefilter
    mask = (
        (merged["spread_pct"] <= max_spread_pct)
        & (merged["mid"] >= min_price)
        & (merged["strike_price"] > 0)
        & (merged["moneyness"].abs() <= max_abs_moneyness)
    )
    merged = merged.loc[mask]
    if merged.empty:
        return pd.DataFrame()

    # Time to expiry
    dte = (merged["expiration"] - merged["ts_recv"]).dt.total_seconds().values / (365.25 * 86400)
    merged = merged[dte > 0]
    dte = dte[dte > 0]
    if merged.empty:
        return pd.DataFrame()

    is_call = (merged["instrument_class"].values == "C")
    K = merged["strike_price"].values.astype(np.float64)
    mid = merged["mid"].values.astype(np.float64)

    # Vectorized IV
    iv = _implied_vol_vec(mid, float(underlying_price), K, dte, risk_free, is_call)
    merged = merged.assign(dte=dte, iv=iv)
    merged = merged.dropna(subset=["iv"])
    if merged.empty:
        return pd.DataFrame()

    # Vectorized delta
    merged["delta"] = _bs_delta_vec(
        float(underlying_price),
        merged["strike_price"].values.astype(np.float64),
        merged["dte"].values,
        risk_free,
        merged["iv"].values,
        (merged["instrument_class"].values == "C"),
    )

    return merged


# ── Aggregated features ─────────────────────────────────────────


def compute(
    bbo_opts: pd.DataFrame,
    definitions: pd.DataFrame,
    underlying_prices: pd.Series,
    timestep: str | None = None,
    risk_free: float = 0.05,
) -> pd.DataFrame:
    """Compute options IV surface features resampled to *timestep*.

    Surface features (iv_atm_mean, iv_skew, iv_butterfly, iv_25d_risk_reversal,
    iv_25d_butterfly) are computed from the next two NQE end-of-month option
    chains (``QNE`` symbol prefix) and interpolated to 30-day constant maturity
    using VIX-style variance weighting.

    iv_0dte_atm uses same-day expiry options (unchanged from v3).
    iv_1dte_atm uses options expiring on the next CME trading day (new in v4).
    iv_term_slope = 30d CM-IV − iv_1dte_atm (v4; was quarterly − 0DTE in v3).

    Parameters
    ----------
    bbo_opts : BBO-1s DataFrame for NQ options.
    definitions : Definitions DataFrame (must include ``symbol`` column).
    underlying_prices : Series with NQ futures mid prices.
    timestep : Output resample interval.
    """
    timestep = timestep or get_timestep()
    w30s = bars(30)
    w1m  = bars_m(1)
    w2m  = bars_m(2)

    # Resample underlying to per-timestep (5s output grid)
    und = underlying_prices.resample(timestep).last().ffill()
    ts_index = und.index

    if bbo_opts.empty or definitions.empty:
        return pd.DataFrame(index=ts_index)

    # Pre-merge defs once (big perf win — avoids per-timestep merge)
    defs_slim = definitions[
        ["symbol", "strike_price", "expiration", "instrument_class", "multiplier"]
    ].drop_duplicates("symbol")
    bbo_merged = bbo_opts.merge(defs_slim, on="symbol", how="inner")
    if bbo_merged.empty:
        return pd.DataFrame(index=ts_index)

    # Internal cadence: compute options features at 30s, forward-fill to 5s output.
    # Options IV surfaces change slowly relative to 5s bars; this cuts cost ~6x.
    internal_step = "30s"
    # Anchor the internal grid on a floored 30s boundary, and look the underlying up
    # with method="ffill".
    #
    # Both matter. `resample(internal_step)` labels every bucket on a 30s boundary
    # (:00/:30), so an internal_index seeded from a raw ts_index.min() that is not
    # itself 30s-aligned lands entirely between labels: the reindex matches nothing,
    # ffill on an all-NaN series stays all-NaN, and the loop's `if np.isnan(price):
    # continue` skips every timestep — 14/14 iv_ columns NaN for the whole session
    # despite complete quote data. 2026-08-12 hit this because NQU6's first bbo_1s
    # quote was 00:06:45 (405 s ≡ 15 mod 30); days whose underlying starts at
    # 00:00:00 are aligned by luck and were unaffected.
    #
    # Flooring also makes each window (ts-30s, ts] coincide with the aligned 30s
    # buckets that scripts/features/patch_iv_fast.py reduces to in SQL, so the
    # historical patch path and the live path see the same quote per bucket.
    internal_index = pd.date_range(
        ts_index.min().floor(internal_step), ts_index.max().ceil(internal_step),
        freq=internal_step, tz=ts_index.tz
    )
    und_internal = (
        underlying_prices.resample(internal_step).last().ffill()
        .reindex(internal_index, method="ffill")
    )

    # Pre-index bbo by ts_recv for fast window slicing
    bbo_sorted = bbo_merged.sort_values("ts_recv")
    ts_values = bbo_sorted["ts_recv"].dt.tz_convert("UTC").dt.tz_localize(None).values

    internal_td = pd.Timedelta(internal_step)

    # Session date in UTC — identifies 0DTE and 1DTE option chains by expiry date,
    # not by a floating-point DTE threshold (which is fragile at session open).
    session_date_utc = ts_index.min().tz_convert("UTC").date()

    # ── Pre-compute chain membership sets (once, not per timestep) ───────────

    # Normalize expiration timezone in defs_slim for date comparisons
    _def_exp_raw = defs_slim["expiration"].copy()
    if _def_exp_raw.dt.tz is None:
        _def_exp_raw = _def_exp_raw.dt.tz_localize("UTC")
    _def_exp_dates = _def_exp_raw.dt.tz_convert("UTC").dt.date

    # 0DTE: options expiring on session_date_utc (same-day, expiry = today)
    zero_dte_syms = frozenset(
        sym for sym, d in zip(defs_slim["symbol"], _def_exp_dates)
        if d == session_date_utc
    )

    # 1DTE: options expiring on the next CME trading day.
    # cme_holidays exports _HOLIDAYS, not CME_HOLIDAYS. Importing the wrong name
    # raised ImportError, the except swallowed it, and _holidays became empty — so
    # _next_trading_day skipped weekends but never holidays. On the session before a
    # holiday it therefore pointed 1DTE at a date with no listed chain, leaving
    # iv_1dte_atm and iv_term_slope NaN for that whole day.
    from src.common.cme_holidays import _HOLIDAYS as _CME_HOLIDAYS
    _holidays: frozenset = frozenset(
        d.date() if hasattr(d, "date") else d for d in _CME_HOLIDAYS
    )
    _next_td = _next_trading_day(session_date_utc, _holidays)
    dte1_syms = frozenset(
        sym for sym, d in zip(defs_slim["symbol"], _def_exp_dates)
        if _next_td is not None and d == _next_td
    )

    # NQE end-of-month chains: find next two QNE expiry dates after session_date.
    chain1_syms: frozenset = frozenset()
    chain2_syms: frozenset = frozenset()
    if "symbol" in definitions.columns:
        _defs_sym = (
            definitions[["symbol", "expiration", "instrument_class"]]
            .drop_duplicates("symbol")
        )
        _eom_mask = (
            _defs_sym["symbol"].str.startswith("QNE")
            & _defs_sym["instrument_class"].isin(["C", "P"])
        )
        _eom_defs = _defs_sym[_eom_mask]
        if not _eom_defs.empty:
            _eom_exp_raw = _eom_defs["expiration"].copy()
            if _eom_exp_raw.dt.tz is None:
                _eom_exp_raw = _eom_exp_raw.dt.tz_localize("UTC")
            _eom_exp_dates = _eom_exp_raw.dt.tz_convert("UTC").dt.date
            _future_eom = sorted({d for d in _eom_exp_dates if d > session_date_utc})
            if len(_future_eom) >= 1:
                _d1 = _future_eom[0]
                chain1_syms = frozenset(
                    sym for sym, d in zip(_eom_defs["symbol"], _eom_exp_dates)
                    if d == _d1
                )
            if len(_future_eom) >= 2:
                _d2 = _future_eom[1]
                chain2_syms = frozenset(
                    sym for sym, d in zip(_eom_defs["symbol"], _eom_exp_dates)
                    if d == _d2
                )

    if not chain1_syms:
        logger.warning(
            "options_surface: no NQE end-of-month chains found for %s — "
            "iv_atm_mean and related features will be NaN",
            session_date_utc,
        )

    records: list[dict] = []

    for ts in internal_index:
        window_start = ts - internal_td
        ws_np = np.datetime64(window_start.tz_convert("UTC").tz_localize(None))
        ts_np = np.datetime64(ts.tz_convert("UTC").tz_localize(None))
        # Binary search slice
        lo = ts_values.searchsorted(ws_np, side="right")
        hi = ts_values.searchsorted(ts_np, side="right")
        if lo >= hi:
            records.append({"ts": ts})
            continue
        snap = bbo_sorted.iloc[lo:hi]

        # Use last quote per instrument in this window
        snap_last = snap.groupby("symbol", sort=False, as_index=False).last()
        price = und_internal.get(ts, np.nan)
        if np.isnan(price):
            records.append({"ts": ts})
            continue

        # Build full surface once; slice by chain membership (avoids N separate builds)
        surface = _build_surface(snap_last, defs_slim, float(price), risk_free)
        if surface.empty:
            records.append({"ts": ts})
            continue

        # Slice surface into chain-specific subsets by symbol
        eom1_surf    = surface[surface["symbol"].isin(chain1_syms)]
        eom2_surf    = surface[surface["symbol"].isin(chain2_syms)]
        zero_dte_surf = surface[surface["symbol"].isin(zero_dte_syms)]
        dte1_surf    = surface[surface["symbol"].isin(dte1_syms)]

        row: dict = {"ts": ts}

        # ── NQE 30-day constant-maturity surface features ─────────────────
        f1 = _chain_stats(eom1_surf)
        f2 = _chain_stats(eom2_surf)
        t1 = f1["dte_days"]   # DTE in calendar days for chain 1
        t2 = f2["dte_days"]   # DTE in calendar days for chain 2

        have_two  = not np.isnan(f1["iv_atm"]) and not np.isnan(f2["iv_atm"])
        have_one  = not np.isnan(f1["iv_atm"])
        have_two_only = not have_one and not np.isnan(f2["iv_atm"])

        if have_two:
            row["iv_atm_mean"]           = _interp_iv_30d(f1["iv_atm"], f2["iv_atm"], t1, t2)
            row["iv_skew"]               = _interp_linear(f1["iv_skew"], f2["iv_skew"], t1, t2)
            row["iv_butterfly"]          = _interp_linear(f1["iv_butterfly"], f2["iv_butterfly"], t1, t2)
            row["iv_25d_risk_reversal"]  = _interp_linear(f1["iv_25d_rr"], f2["iv_25d_rr"], t1, t2)
            row["iv_25d_butterfly"]      = _interp_linear(f1["iv_25d_bf"], f2["iv_25d_bf"], t1, t2)
        elif have_one:
            # Only chain1 available — use directly without interpolation
            row["iv_atm_mean"]          = f1["iv_atm"]
            row["iv_skew"]              = f1["iv_skew"]
            row["iv_butterfly"]         = f1["iv_butterfly"]
            row["iv_25d_risk_reversal"] = f1["iv_25d_rr"]
            row["iv_25d_butterfly"]     = f1["iv_25d_bf"]
        elif have_two_only:
            # chain1 data absent from bbo_1s (not yet ingested or not available) —
            # fall back to chain2 directly without interpolation
            row["iv_atm_mean"]          = f2["iv_atm"]
            row["iv_skew"]              = f2["iv_skew"]
            row["iv_butterfly"]         = f2["iv_butterfly"]
            row["iv_25d_risk_reversal"] = f2["iv_25d_rr"]
            row["iv_25d_butterfly"]     = f2["iv_25d_bf"]
        else:
            row["iv_atm_mean"]          = np.nan
            row["iv_skew"]              = np.nan
            row["iv_butterfly"]         = np.nan
            row["iv_25d_risk_reversal"] = np.nan
            row["iv_25d_butterfly"]     = np.nan

        # ── 0DTE ATM IV (same-day expiry — unchanged from v3) ────────────
        if not zero_dte_surf.empty:
            atm_0dte = zero_dte_surf.iloc[(zero_dte_surf["moneyness"].abs()).argsort()[:3]]
            row["iv_0dte_atm"] = float(atm_0dte["iv"].mean())
        else:
            row["iv_0dte_atm"] = np.nan

        # ── 1DTE ATM IV (options expiring next trading day — new in v4) ──
        # NaN until 1DTE BBO data is ingested (download_1dte_options.py + ingest).
        if not dte1_surf.empty:
            atm_1dte = dte1_surf.iloc[(dte1_surf["moneyness"].abs()).argsort()[:3]]
            row["iv_1dte_atm"] = float(atm_1dte["iv"].mean())
        else:
            row["iv_1dte_atm"] = np.nan

        # ── Term slope: 30d CM-IV − 1DTE ATM IV ──────────────────────────
        # v4 definition: front = 1DTE (not 0DTE); back = 30d NQE interpolated.
        # Positive = term structure in contango (near-term IV < 30d IV).
        _iv_30d = row["iv_atm_mean"]
        _iv_1d  = row["iv_1dte_atm"]
        row["iv_term_slope"] = (
            float(_iv_30d) - float(_iv_1d)
            if not np.isnan(_iv_30d) and not np.isnan(_iv_1d) else np.nan
        )

        records.append(row)

    features = pd.DataFrame(records).set_index("ts")

    # Guarantee all expected columns exist (guards against empty-surface days)
    for col in (
        "iv_atm_mean",
        "iv_skew",
        "iv_butterfly",
        "iv_0dte_atm",
        "iv_1dte_atm",
        "iv_25d_risk_reversal",
        "iv_25d_butterfly",
        "iv_term_slope",
    ):
        if col not in features.columns:
            features[col] = np.nan

    # Upsample 30s internal → 5s output via forward-fill
    features = features.reindex(ts_index, method="ffill")

    # IV velocity (rate of change) — based on 30d CM-IV
    for w in (w30s, w1m, w2m):
        features[f"iv_atm_chg_{w}"] = features["iv_atm_mean"].diff(w)
        # Use clip(lower=1e-8) instead of replace(0, NaN): iv_atm_mean is upsampled
        # from 30s → 5s via ffill, so rolling(6).std() == 0.0 for any window that
        # falls entirely within a 30s block (identical values) — not missing data.
        # Treating std=0 as NaN would make ~16.7% of bars NaN; zscore=0 is correct.
        _std = features["iv_atm_mean"].rolling(w).std().clip(lower=1e-8)
        features[f"iv_atm_zscore_{w}"] = (
            (features["iv_atm_mean"] - features["iv_atm_mean"].rolling(w).mean())
            / _std
        )

    return features.astype(np.float32)
