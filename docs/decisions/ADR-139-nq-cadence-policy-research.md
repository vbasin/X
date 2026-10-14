# ADR-139 — NQ policy research with causal empirical bands and cadence exits

Date: 2026-09-24. Status: accepted as an experimental research contract only.
Supersedes: no production implementation. Revises the next research approach
following ADR-138's rejected learned expected-return experiment.

## Problem and decision

The requested economic objective was not established by ADR-138. Its narrow
sample and overlap-heavy candidate labels did not justify treating the tested
regressors or fixed horizon as the best possible design. Diagnose actual policy
cash flows and expand independent discovery history before adding model layers.

Use existing local NQ one-second OHLC history to create causal close-stamped
minute research inputs. Empirical same-clock movement bands estimate the price
region expected from prior completed sessions. Their deviation is a directional
continuation hypothesis, not a calibrated probability. No supervised target is
needed: the policy is selected on chronological discovery cash flows.

The first experimental variation checked and ratcheted stops every minute. Its
2011–2020 discovery failed economic gates; primary-source review revealed that
the motivating authors' execution changes exposure only at cadence times.
Preserve that variation, its full trial records and failure diagnosis. A separate
cadence variation checks current band/VWAP conditions at 30/60-minute closes,
executes next observable prices, and optionally keeps a fixed opposite-band
emergency stop. Neither is an exact replication of the SPY paper.

Outcome semantics are realized one-contract NQ policy net dollars after explicit
execution friction and fees, with no fixed profit target. Missing exits are NaN
and acceptance-blocking. They must never become zero-dollar labels or lead to
future-conditioned day removal. Independent market-data recovery can repair an
input artifact only through additive, hashed raw records with overlap validation.

## Isolation and evidence

Discovery: 2011–2020. Frozen historical validation: 2021–2024. Already exposed
2025–June 2026: quote execution compatibility, not fresh OOF. July–September 2026
reserve is consulted once only after all declared economic/data gates pass.
Chronological bands use prior sessions only; rolling state may update with
completed evaluation sessions as it would online, never their future outcomes.
The published research itself postdates parts of historical validation; external
literature knowledge means these dates are not universally untouched evidence.

The policies use separate experimental modules, existing path resolution,
canonical economics for persisted quote fills, and canonical run manifests.
Their exact protocol is recorded as a resolved run snapshot. The current
objective-label schema requires first-passage geometry and cannot express this
policy's outcome honestly; these runners therefore do not masquerade as a
production objective-loader integration. A future integration requires an
explicit schema extension and its own compatibility tests.

Source code snapshots, raw hashes, all discovery trials, frozen policy, stress
replays, independent liquidation-mark audit, and printable results are required.
Passing unit tests establishes tested timing/accounting behavior, not trading
profitability. No model is promoted by this ADR and no live orders are authorized.

## Consequences

The new strategy can be evaluated without a database; source preparation may
use existing files and authorized read-only diagnostics. Legacy modules and
artifacts are retained. Any result must disclose differences between minute
OHLC proxies, pre-trade TBBO quotes, and actual broker execution. Report dollar
risk at one NQ; no invented account capital or percentage-return claim.
