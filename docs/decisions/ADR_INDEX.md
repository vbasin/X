# ADR_INDEX.md — ADR Status/Topic Index

**Status: CANONICAL, CURRENT.** 145 ADR files (144 numbered decisions; ADR-118
has two historical files) exist under `docs/decisions/`. This index exists so a
fresh agent doesn't read all 145 to find the ones relevant to a given area — per
`AGENTS.md` §1's explicit instruction not to treat the full ADR history as
required reading.

**Methodology, stated honestly**: most individual ADRs were **not** re-read to produce this index — that would be a large task disproportionate to a structural-cleanup pass (137 files). Instead: (1) ADRs directly read and verified during this session's audit/restructuring work are individually called out below with confidence; (2) the rest are bucketed by filename/topic pattern, which is a real, mechanically-checkable signal (this repo's own ADR-naming convention is topic-descriptive), not a guess; (3) none of the unbucketed-in-detail ADRs are asserted to be Superseded/Deprecated without evidence — per ADR discipline (append-only), the safe default is **Accepted-until-shown-otherwise, read directly before relying on it for a decision**.

A structural observation this index surfaces: **most ADRs in this repository do not carry an explicit `Status:` header** (unlike this restructuring's own `ADR-136`, which does). This is itself a small process gap — worth fixing prospectively (new ADRs should state Status explicitly, as ADR-136 does) rather than retroactively (rewriting 136 historical files' headers is out of scope and would violate the "don't rewrite history to look current" principle).

## 1. Individually verified this session (highest confidence)

| ADR | Status | Note |
|---|---|---|
| ADR-006 | Accepted, current | `config.yaml` single source of truth — see `docs/CONFIG_AUDIT.md` for where actual runtime behavior diverges from this ADR's intent |
| ADR-018 | Accepted, current | Objective doc = single source of labels/KPI |
| ADR-023 | Accepted, current | Triple-barrier label scheme (Track B Outcome Builder) |
| ADR-025 | Accepted, current | Scoping v2 — Track B Orchestrator |
| ADR-026 | Accepted, current, but see gap | PnL economics/backtest design — `src/backtest/engine.py`'s implementation has a confirmed slippage-translation bug relative to this ADR's intent (`docs/CONFIG_AUDIT.md`) |
| ADR-027 | Accepted, current, but see gap | Economics/meta/sparse objective (`nq-50pt-econ`) — its own design text states `cost_per_trade: 1.0`, contradicted by the live objective YAML's `0` (`docs/CONFIG_AUDIT.md`) |
| ADR-028 | Accepted, current | Structure-first layout + docs workflow — the convention `AGENTS.md` §6 and this ADR index both follow |
| ADR-036 | Accepted, current | Custom Test lab last-fold bundles |
| ADR-071 | Accepted, current | Optional champion gate — one model ever promoted under it (`project_manifest.yaml`) |
| ADR-126 | Accepted, current | Bounded-zone width + resting entry (Track C) |
| ADR-127 | Accepted, current | TBBO tape + cascade architecture (Track C's founding ADR) |
| ADR-130 | Accepted, current | Target-adaptive stops (Track C) |
| ADR-134 | Accepted, current | Cascade SSOT structural transitions policy (Track C) |
| ADR-135 | Accepted, current | Live Chart Level Blend VP |
| ADR-136 | Accepted, current | This restructuring's founding ADR |
| ADR-140 | Rejected experimental hypothesis | TBBO profile reclaim failed discovery and frozen validation; no promotion |
| ADR-141 | Accepted, current | Live Chart SAR AF params + scenario Notes |
| ADR-142 | Accepted, current | Live Chart VWAP Bollinger bands (±2σ) |
| ADR-143 | Accepted, current | Live Chart multi StochasticFast + sync pair |
| ADR-144 | Accepted, current | Live Chart compact scale tags, hover resize, bound arrows |
| ADR-148 | Accepted, current | Live Chart MBP-10 ladder and `/api/book/*` removed; supersedes ADR-087 §4 |
| ADR-150 | Accepted, current | Live Chart MBO stats row, 5-day window, selection zoom; supersedes ADR-145 §3–§4 cap |
| ADR-152 | Accepted, current | Live Chart double-click opens one order's MBO rows (`ts_event`, `sequence`); extends ADR-145 |
| ADR-154 | Accepted, current | Live Chart Select draws a time × price box; extends ADR-141 |

## 2. Topic buckets (by filename pattern — mechanically verified via `ls docs/decisions/`, not individually read)

| Bucket | Range/count | Relevant canonical responsibility / doc |
|---|---|---|
| Live Chart UI (`live-chart-*`) | 26 ADRs (ADR-087, 097, 105, 108-121, 124-125, 128, 132, 135, 141–144) | WebUI-only; not part of the Stage/Outcome/Model responsibilities in `docs/ARCHITECTURE.md`. Read only if touching `src/webui/live_chart.py` or its static assets. |
| WebUI Custom Test framework | 6 ADRs (035, 036, 040, 042, 074, 076) | Track B Orchestrator/Model Strategy — `docs/ARCHITECTURE.md` §1.4/1.12 |
| Cascade (Track C) | 2 core (127, 134) + 130, 126, 129 (session-profile parity), 133 (EMA/VWAP trend features), 131 (large-print taxonomy) | Track C — see `docs/ARCHITECTURE.md` §1.2-1.4, `docs/reference/STAGE1_FEATURES.md`/`STAGE3_FEATURES.md` |
| Objectives/economics/search/promotion | 018, 025, 026, 027, 040, 045, 046, 049, 071, 084, 106 | Track B — `docs/ARCHITECTURE.md` §1.3/1.8/1.10, `docs/CONFIG_AUDIT.md` |
| Auction/session/volume-profile | 051, 097, 099, 117, 125, 126 | Feature Builder / Decision Policy, spans both tracks |
| Data/GPU/infra plumbing | 005, 015, 016, 022, 025, 028, 029, 030, 043, 060, 073, 078, 102 | Shared infra — cross-reference `docs/architecture/SYSTEM.md`/`DATA_PIPELINE.md`/`DATA_LAYOUT.md`, which are current per `docs/ARCHITECTURE.md` §4 |
| Everything else (~75 ADRs, mostly ADR-001–ADR-104 numeric range) | not individually bucketed | Largely Track B feature-engineering/model-architecture detail (native-TF hive columns, LSTM/dual-binary head design, IV/GEX, cost-matrix training loss, etc.). **Default status: Accepted-until-shown-otherwise.** Several early low-numbered ADRs (e.g. ADR-007 `detection-2-mae-filter`, ADR-008 `3class-softmax-class-weight`) use Track A's own terminology ("Detection-2") and are very likely describing the now-archived legacy track specifically — **not individually confirmed superseded**, flagged only as a pattern worth checking before relying on them for current work. |

## 3. How to use this index

1. Identify which canonical responsibility (`docs/ARCHITECTURE.md` §1) or which track your work touches.
2. Find the matching bucket above.
3. Read only the ADRs in that bucket, starting with anything in §1 (individually verified) if it overlaps.
4. If an ADR you need isn't in either section, read it directly — this index doesn't claim completeness at the individual-ADR level, only at the bucket level.

## 4. What this index does not do

It does not reclassify, supersede, or edit the historical ADR files. It does not assert Superseded/Deprecated status for anything not already stated as such within that ADR's own text or a later ADR's explicit cross-reference (e.g. `docs/architecture/RESEARCH_DIRECTOR.md`'s own note: "Supersedes in code: deleted Phase-19/20 modules"). A full per-ADR status audit remains a real, undone piece of technical debt — tracked in `docs/MIGRATION_MANIFEST.md` §5 and the requirements audit in that same document.

- [ADR-138 — Independent selective NQ causal research](ADR-138-nq-selective-causal-research.md): accepted experiment substrate, rejected trading hypothesis; nested validation, quote execution, physical evidence and legacy normalization warning.

- [ADR-140 — TBBO confluent-profile reclaim rejection](ADR-140-nq-tbbo-profile-reclaim-rejection.md): exact-profile, large-print and quote-event policy rejected with the reserve locked.

- [ADR-141 — Live Chart SAR AF params + scenario Notes](ADR-141-live-chart-sar-params-and-notes.md): configurable SAR start/increment/maximum on `#/chart`; Select cursor + `#/notes` snapshots under `ops/webui/chart_notes/`.

- [ADR-142 — Live Chart VWAP Bollinger bands](ADR-142-live-chart-vwap-bollinger-bands.md): VWAP ± 2σ volume-weighted bands on the price pane; same origin as VWAP.

- [ADR-143 — Live Chart multi StochasticFast + sync pair](ADR-143-live-chart-multi-stoch.md): EMA-style Fast %D cards plus sync (all) and sync (fast+slow).

- [ADR-144 — Live Chart compact scale tags, hover resize, bound arrows](ADR-144-live-chart-scale-tags-bounds.md): one-line last-value tags, cursor readout pinned in place (Pin or right-click) with Follow cursor, Stoch/MFI bounds, and pane-height cross arrows.

- [ADR-151 — `databento.definitions` underlying resolution, PM settle time, row provenance](ADR-151-definitions-underlying-and-settle-time.md): options settle into the first quarterly still trading at their expiry *instant* (never the option's own month code); weeklies are PM-settled 15:00 CT, not a fixed UTC hour; new `source` column with vendor-wins upsert so reconstructed rows self-correct.

- [ADR-153 — Options IV: per-option underlying, solver convergence, EOM chains](ADR-153-options-iv-underlying-and-solver.md): the options underlying silently mixed four futures contracts after `NQ.c.0` ingestion stopped 2026-07-14; the IV solver returned its 0.3 seed as a solution; `QNE` month-end chains never parsed, blacking out live Greeks on every month-end.

- [ADR-155 — Weekly live session, 16:05 CT session roll, rotating DBN archive](ADR-155-weekly-live-session-and-rotating-archive.md): one `db.Live` connection from the CME Sunday open to the Friday close; the session date advances at 16:05 CT inside the 16:01–16:44 CT window where no quote is written, instead of tearing the client down at Chicago midnight for 10–50 s of unrecoverable quotes four times a week; `RotatingDbnStream` keeps one archive directory per session date. Supersedes ADR-034 decision 4 only.

- [ADR-156 — Normal option subscription and vendor definition acquisition](ADR-156-normal-option-definition-acquisition.md): vendor terms by symbol alongside subscription, exact-zero-cost historical acquisition, no live synthetic creation, causal current-ID binding and mapping-preserving DBN rotation. Supersedes ADR-151's synthetic fallback and ADR-155's reporting-only/header-only behavior.

- [ADR-157 — bounded append-only Databento capture-gap recovery](ADR-157-bounded-databento-gap-recovery.md) — accepted 2026-10-06; exact-cost acquisitions, immutable archives, occurrence reconciliation, historical as-of definitions.
