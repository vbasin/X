# Publication disposition

The platform is a source-only copy of a dirty operational working tree, not
a history-preserving mirror of the private development repository. Existing
destination content is retained unchanged.

| Paths | Decision |
|---|---|
| `src/` | Include Python application code, console assets, and module documentation. |
| `tests/` | Include source tests and isolation fixtures; gate only explicit portable paths. |
| `config/` | Include YAML/JSON contracts and examples, no runtime credentials. |
| `scripts/` | Include selected maintained data/build/evaluate/search/serving helpers. |
| `docs/architecture/`, `docs/reference/`, `docs/decisions/` | Retain design context; historical deployment claims are not public setup guarantees. |
| Canonical root documents | Retain architecture, experiments, contracts, and known limitations. |
| `deploy/`, `systemd/` | Optional deployment reference; private push helper omitted. |
| `notebooks/` | Keep generic OOF analysis source; users supply their own run/model IDs. |
| `research_cases/` | Do not copy raw market data, generated trades, paper corpus, or unverified summaries. |
| `archive/`, `data/`, `logs/`, `mlruns/`, backups | Omit entirely. |
| `.env`, local editor/agent state, source `.git/`, caches, venv | Omit entirely. |
| Root CSV/Parquet/PDF/scratch artifacts | Omit entirely. |

No source data or private trained model is redistributed. The RAG modules remain
available, but a user supplies an authorized corpus. Missing historical documents
linked by retained ADRs are not silently presented as included artifacts.

## Export-only corrections

- Added a synthetic offline demo and a portable explicit-path test gate.
- Corrected package discovery to preserve the `src.*` import namespace.
- Moved heavy optional integrations out of the minimal demo dependency set.
- Removed original-code MIT claims at the owner's direction; third-party
  notices remain independently applicable.
- Replaced the public README with tested setup and honest capability boundaries.
- Fixed the OOF notebook's removed namespace and private run/model identifiers.
- Left operational source intact except explicitly recorded release fixes.
- Updated the bootstrap repository URL to X. The export's update helper refuses
  dirty/detached checkouts and no longer auto-pushes or force-resets local work.

The runtime checkout is the preserved original for this copy-first release.
There is no relocation, deletion, database migration, data rebuild, or change to
the running platform. The destination's pre-existing file is outside the scope
of the platform's copyright statement.


## Narrow synthetic-fixture exception — 2026-10-15

The public showcase additionally includes `examples/transformer-regime/`: the
exact 3,000-row synthetic dataset, small native Transformer weights/scalers,
saved predictions/ledgers, reviewed evidence projections and portable MLflow
metadata. This exception does not include real market data, private trading
models, operational SQLite databases, full operational code bundles or runtime
stores. Original source evidence hashes and projection metadata are retained.
The reviewed X source supplies code for fresh inference/training. Generated
imports and reruns are ignored beneath `.demo-output/`. The original retained
operator run remains unchanged. See [the walkthrough](examples/TRANSFORMER_WORKFLOW.md).
