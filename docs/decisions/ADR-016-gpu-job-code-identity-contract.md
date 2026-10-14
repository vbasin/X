# ADR-016 — GPU Job Code-Identity Contract (git-sha pinning + push/dirty guard)

> **Superseded by [ADR-022](./ADR-022-oci-queue-code-bundle-autonomous-gpu.md) (2026-07-10):**
> git-sha checkout is replaced by content-addressed code bundles (tar→zstd→sha256, verified on
> download). This removes GitHub from the hot path, the push/dirty guard, detached-HEAD healing,
> and the one-sha-per-runner limit, while giving a *stronger* reproducibility guarantee.

**Date**: 2026-07-02
**Status**: Accepted
**Relates to**: [ADR-015](./ADR-015-gpu-standard-host-job-queue.md) (GPU job-queue execution plane —
hardens its "one image + `git checkout <sha>` gives code reproducibility" assumption),
[RESEARCH_DIRECTOR.md §3.1](../architecture/RESEARCH_DIRECTOR.md) (GPU execution plane),
[PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md)
**Supersedes**: none (extends, does not replace, ADR-015)

---

## Context

ADR-015 assumed "one image + `git checkout <sha>` gives code reproducibility," and
RESEARCH_DIRECTOR.md §3.1 documented that the GPU Runner "runs `git fetch && git checkout
<git_sha>` to the spec's pinned commit." The first live dispatch proved that assumption was
not actually enforced on either side of the queue:

1. **Producer had no guard.** `controller._git_sha()` enqueued a bare `git rev-parse HEAD`
   with no verification. During the first dispatch a fix was **committed but not pushed**;
   the enqueued sha existed only on the CPU VM. The GPU VM — which only ever pulls from
   `origin` — could not obtain that commit, so it would either fail to check it out or
   silently run stale code. The run consumed A10 GPU time with no usable result.
2. **Consumer ignored the pinned sha.** `gpu_runner.runner` never referenced `git_sha`. It
   ran whatever `git_update.sh`'s fast-forward pull happened to leave in the working tree, so
   the executed code identity depended on branch/pull **timing**, not on the dispatched
   commit — the exact divergence ADR-015 assumed away.
3. **No audit trail.** Nothing recorded which commit actually ran, so a divergence could not
   be detected after the fact.

A first cut of the producer guard also over-reached by blocking on **any** `git status`
output, which would derail every dispatch over incidental **untracked** files (logs, stray
config copies) that are never committed or pushed and therefore cannot affect the checkout.

## Decision

Enforce a two-sided **code-identity contract** so the GPU VM provably runs exactly the commit
the CPU VM tested and pushed:

1. **Producer guard** — `controller._git_sha(strict=True)` on the dispatch path refuses to
   enqueue unless **both** hold: (a) no *tracked* files are modified
   (`git status --porcelain --untracked-files=no` is empty) and (b) the HEAD sha is contained
   by an `origin` branch (`git branch -r --contains <sha>`). Untracked files are ignored — they
   are never committed/pushed and are irrelevant to what `git checkout` reproduces. This closes
   the "committed but not pushed" divergence class at the source, failing fast on the CPU VM
   with a clear message **before** any GPU minutes are spent.

2. **Consumer pin** — `gpu_runner.runner._checkout_git_sha()` runs, **before importing any
   job-execution module**, `git fetch origin <sha>` (falling back to a full `git fetch origin`
   if the server disallows fetch-by-sha) followed by `git checkout --detach <sha>`. Detached
   HEAD pins the working tree to precisely the dispatched commit, immune to branch state or
   pull timing. Running it before the lazy imports guarantees the pinned code is what loads.

3. **Echo-back verification** — the runner writes both `git_sha` (requested) and `ran_sha`
   (post-checkout HEAD) into the done marker; the collector logs a prominent `SHA MISMATCH`
   warning if they diverge, providing an audit trail that executed code matched dispatched code.

## Consequences

- Deterministic code identity: the GPU VM runs the exact pushed commit the CPU VM tested; the
  divergence bug is structurally impossible rather than avoided by convention.
- Dispatch fails fast on the CPU side (actionable error) when code is unpushed or the tracked
  tree is dirty, instead of silently wasting GPU minutes.
- Adds one `git fetch` + `checkout` per wave on the GPU VM (seconds).
- **One sha per runner lifetime.** A live Python process cannot reload already-imported
  modules, so a *second* job in the same runner requesting a *different* sha cannot be honoured
  — the runner logs a warning. This is safe under the current wave model (fresh VM per wave, a
  single sha per wave). Lifting it (concurrent multi-sha waves) would require a
  subprocess-per-job executor; deferred until needed.
- No model / label / metric semantics change — leaderboards remain comparable.
