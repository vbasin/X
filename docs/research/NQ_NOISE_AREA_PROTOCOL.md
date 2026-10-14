# NQ noise-area continuation retry — frozen before outcome inspection

Date: 2026-09-24. Status: research, no live trading. This is a new hypothesis,
not a relabeling of the rejected short-horizon regression experiment.

## Mechanism and corrections

Persistent directional demand can move prices outside the normal move from the
cash open. Estimate that normal movement at each minute using only the previous
14 sessions, enter confirmed departures, and let a trailing boundary/VWAP stop
retain unusually large continuation wins. This deliberately removes the prior
experiment's arbitrary fixed profit caps and overlapping candidate labels.

Research inspiration: Zarattini, Aziz and Barbon,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172.
This is an NQ adaptation, not a claimed reproduction of their SPY results.
Unidentified, unadjusted continuous-contract files cannot support reliable
cross-roll overnight returns. Bands therefore center on today's cash open only;
no prior-close anchor or overnight signal is used.

## Chronology and execution

Existing 2011–2024 1-second OHLCV is aggregated to one-minute OHLCV, timestamped
at its close. Training/selection: 2011–2020. Frozen validation: 2021–2024. The
already exposed April2025–June2026 quote period is a compatibility/stress check,
not a fresh holdout. July1–September22,2026 remains reserved for one final frozen
quote evaluation only if development and validation qualify.

Use exchange_calendars XNYS pre-known regular full-length cash sessions. Missing
bars/outcomes are disclosed, never excluded on the basis of profitable results.
Historical prices have no quotes; base price-cost proxy is 1 point round trip
plus the repository's NQ commission, stress2 and4 points round trip. Entry after
closed-bar signal at next minute open; gap-through stops fill at the worse of
open/stop; no same-bar signal/fill. Live-equivalent confirmation must replay
raw quote events with bid/ask,250ms latency and0.25point additional slippage per
side. Quote stresses:0.5/1point per side,1000ms. Historical proxy and quote
results must be shown separately.

## Candidate family (24 configurations)

14-session same-clock mean absolute return from cash open, shifted one whole
session; upper/lower=current open×(1±multiplier×historical move).
Multipliers .75/1/1.25; decisions each30 or60min after08:30; directions both or
long-only; trailing stop boundary-only or tighter boundary/VWAP. Enter09:00–14:00,
flatten15:00 Chicago. Stop ratchets each closed minute and never loosens. One NQ
contract, one position, maximum3entries/day, cease after2losses. No profit cap.
No account sizing or leverage optimization. A parallel cash benchmark always runs.

Select once on2011–2020 by worst-half(2011–15,2016–20) daily Sharpe, subject to
both halves positive after2point roundtrip cost, >=6positive years, PF>=1.10,
>=300trades. All24 candidate results are retained. If none passes, report the
strongest discovery candidate as exploratory and continue hypothesis analysis;
never claim it passed or unlock the final reserve. Apply the chosen configuration
unchanged on2021–2024. Report individual years, month PnL, both directions,
turnover, loss count/amount, trade/day drawdown and block-bootstrap uncertainty.

Validation qualification: positive net each two-year half, net PF>=1.10,
95% five-session block-bootstrap mean daily-PnL lower bound>0,
positive2point-cost stress; no unresolved selected trades. Recent quote check
must be positive net, PF>=1.10 and stress-positive before final reserve evaluation.
The final reserve is evaluated once after policy/config freeze; no threshold
adjustment after seeing it. Final results may be statistically inconclusive even
when positive. No backtest can establish globally optimal or guaranteed profit.

A close-momentum comparator and event-level learned filter may be researched
using discovery data only; each additional design must be recorded before its
validation is read. Prior failed experiment outcomes are explicitly reused only
for diagnosis and the labeled compatibility check.

Pre-outcome data clarification: exclude the Monday–Friday of quarterly third-
Friday expiration week using its known calendar. NQ.c.0 rolls at expiry while
recent tape rolls Monday; this avoids modeling their liquidity/contract mismatch
as one series. No session is removed merely for later-observed missing minutes;
halts/gaps are preserved, and unresolved positions invalidate acceptance.
