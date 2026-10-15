# Transformer showcase validation — 2026-10-15

Implemented in an isolated X checkout based on `4e498b14` on local branch
`showcase/transformer-robin`. Validation used a fresh Python 3.12 environment
on Linux ARM64/OCI. GitHub-hosted x86 CI is configured but has not run remotely.
The original operational checkout and accepted dataset/run were not modified.

| Check | Result |
|---|---|
| Recorded WebUI/provider/CLI and scope gate | **40 passed**, 10.10 seconds |
| Nine explicit Transformer workflow test modules | **164 passed**, 74.04 seconds |
| Existing publication demo/shared-contract scope | **126 passed**, 3.64 seconds |
| Browser: all seven tabs, feature/split/model controls | Passed, zero console JavaScript errors or external requests |
| MLflow browser | Parent plus both design children visible; direct experiment link |
| Real launcher shutdown | SIGTERM stops console and MLflow; both ports closed |
| Artifact integrity and source immutability | Hash/tamper, traversal, symlink and nested-output tests passed |
| Relocation | Fresh MLflow import after output move; prior stores preserved; artifact download passed |
| Public package vs retained model | All 289 holdout probabilities identical, maximum absolute error **0.0** |
| Clean dependency validation | `pip check` and pinned requirement resolution passed |
| Scoped Ruff and whitespace | Passed |

Third-party Keras/NumPy and Starlette deprecation warnings remain; no validation
failure was ignored. The full operational suite was not run. These scope counts
overlap on shared scope-consistency checks; they are not a unique-test total.
See [machine-readable evidence](SHOWCASE_VALIDATION.json).

## Actual new training and promotion

Fresh run **`20261015T042710Z_f34f44d9`** finished in **71.6 seconds** under the
900-second supervisor; worker completion including promotion was **66.82 seconds**.
It ran from the public source checkout with the original frozen recipe/gates.

- Exact original 3,000-row dataset SHA reproduced.
- Depth 2 selected: model `5efbd8530cd2`.
- Validation accuracy 0.919540, macro F1 0.919771, log loss 0.304553.
- Holdout accuracy **0.934256**, macro F1 **0.934081**, log loss **0.242030**.
- All eight validation and all eight holdout checks passed.
- Prior-only and shuffled negative controls rejected.
- All 289 fresh holdout probabilities identical to the retained original run.
- MLflow parent `6f47318f304e421786baf43f815c0c21` and both children `FINISHED`.
- Version 1 of `workflow-test-transformer-20261015T042710Z_f34f44d9` resolves
  through `test-serving`; canonical registry pointer and post-promotion HTTP
  probability/direction/timestamp parity verified. Invalid input returns 400.
- Lifecycle `PRODUCTION` applies to the synthetic service only. Temporary
  inference servers stopped. No real-market data or trading service activated.

Fresh-run artifacts stay in ignored `.demo-output/transformer-rerun-01/`.
Exporting the new run through the documented `prepare --run-dir` command passed;
it preserves new model IDs and independently imports its three MLflow runs.
A second real-browser pass exercised all seven tabs and MLflow against this new
run, with zero console JavaScript errors or external requests. Both services
stopped cleanly afterward. The bundled public fixture continues to represent the
original accepted run.

## Issues found and resolved

1. **Fresh dependency conflict:** MLflow 3.10.1 requires PyArrow below 24; X's
   lightweight demo pins 26. Separate locked profiles preserve the existing demo
   and use the original scientific stack for the Transformer.
2. **CPU wheel portability:** TensorFlow's `tensorflow-cpu` wheel is unavailable
   on the tested ARM64 machine. Platform markers select `tensorflow` without CUDA
   extras on ARM and `tensorflow-cpu` on Linux x86; package metadata handles both.
3. **Nonportable MLflow records:** source SQLite URIs reference the operator's
   filesystem. API export/import creates local IDs and artifact paths with
   source lineage, and relocation creates a preserved new tracking generation.
4. **Evidence import safety:** prevent outputs nested inside source fixtures/runs,
   reject symlinks and all unlisted files, and require every consumed file to
   appear in the verified manifest. No private deployment source is exported.
5. **MLflow browser readiness:** `networkidle` does not finish while MLflow polls.
   Browser tests wait for rendered experiment/run content; links open the model
   training view directly rather than the default GenAI overview.
6. **Shutdown ownership:** explicit SIGINT/SIGTERM handling covers startup and
   console shutdown so the owned MLflow child is stopped. Real process/port
   checks confirmed cleanup.

No threshold, label definition or model recipe was changed to obtain acceptance.
The retained result and new run demonstrate the small synthetic task and the
complete orchestration path, not statistical superiority or real trading edge.
