# ADR-039: Stochastic sync exponential envelope + 9/60 pair score

## Status
Accepted — implemented 2026-08-20.

## Date
2026-08-20

## Context

v5.7 `so_sync_score` treated 20-up / 80-down as a **3-bar pulse** plus a
same-bar soft term that died once %D left 0–25 / 75–100. A Fast(9,3) that
had already crossed 80 and was at 60 no longer voted, so it could not
confirm a Full(60,10) that was still approaching 80.

## Decision

1. **Keep the soft pre-cross term** (oversold+rising in 0–25, overbought+falling
   in 75–100) so a slow %D can vote **before** it tags 20/80.
2. **Replace the 3-bar pulse** with a post-cross **armed envelope**:
   - Arm on 20-up / 80-down.
   - Vote \(= \exp_{\text{affine}}(p)\) with \(p\) = fraction of the way from
     the print to midpoint 50 (span 30). Affine exponential \(\lambda = 1\):
     1 at the print, 0 at 50.
   - Clear the arm on recross of 20/80 against the signal, or once %D is
     strictly through 50.
   - **1-bar pause**: two consecutive contrary \(\Delta d\) bars zero the
     envelope that bar (arm stays until recross / mid).
3. Score is still
   \((\text{bull}_m - \text{bear}_m)\,(0.5 + 0.5\,\text{agree})\), clipped
   to **[−1, +1]**. −1 = strong sell (80-down family); +1 = strong buy
   (20-up family).
4. Two columns:
   - `so_sync_score` — all four %Ds
   - `so_sync_score_9_60` — same model on Fast(9,3) + Full(60,10) only

## Consequences

- Fast %Ds that have crossed 80 (or 20) still contribute while they grind
  toward 50, so they can line up with a slower study that has not crossed yet.
- At 50 the envelope is 0; past 50 that oscillator is done on this episode.
- `so_` 5 → 6; catalog v5.8 (218 features).
- Existing parquets hold the old 3-bar pulse until `patch_parquets --modules ext`
  or a rebuild.

## Related

- [ADR-038](ADR-038-so-d-only-full-60.md) — %D-only names / Full(60)
- `src/features/stochastic.py`, `docs/reference/FEATURE_CATALOG.md`
