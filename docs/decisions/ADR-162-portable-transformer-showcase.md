# ADR-162 — Portable recorded Transformer showcase in X

Date: 2026-10-15. Status: Accepted for the public synthetic example.

## Context

X's lightweight logistic-regression demonstration does not exercise Transformer
model promotion or the operational console. The retained 3,000-row synthetic
Transformer test provides authentic evidence, but its MLflow database and model
package reference an operator filesystem and source tree. Copying the full
runtime tree would violate the curated publication scope and would not create
a portable demonstration.

## Decision

Keep the lightweight demo and add a distinct file-backed console profile. Publish
an explicit allowlist of exact synthetic data/native model files and documented
evidence projections, with original hashes. The operational source bundle and
SQLite backend remain outside the public fixture. Reuse canonical run/storage
paths for visitor-generated state.

Import historical tracking through the MLflow API into a new local store, keeping
source IDs/times and parent-child lineage. Mark imported evidence as recorded;
do not claim a new gate execution or recreate a serving promotion silently.
Relocation creates another preserved tracking generation instead of rewriting
source evidence or backend SQL.

An optional CPU rerun uses the unchanged frozen synthetic classification objective,
independently passes validation/holdout and negative-control gates, verifies
serving, and promotes only in its synthetic-service namespace. This does not
change trading objectives, waive their economic/paper gates or activate trading.
The supervisor caps the complete test at 900 seconds. Separate dependency locks
keep the viewer free of TensorFlow and preserve the retained scaler contract.

## Consequences

Visitors can inspect real small-scale evidence immediately, then run the actual
pipeline. Imported run IDs differ intentionally. The read-only console refuses
mutations and does not start operational workers. Live feeds and order books are
unavailable because the fixture contains no corresponding data. Public source
weights reproduce retained predictions, but the rebuilt package has its own
identity. See [walkthrough](../examples/TRANSFORMER_WORKFLOW.md) and
[validation](../examples/SHOWCASE_VALIDATION.md).
