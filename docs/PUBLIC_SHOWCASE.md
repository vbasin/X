# Public showcase

## Supported offline surface

`vtech-demo --output <new-directory>` runs without external services. It reuses
the existing causal lookup, calendar fold builder, logistic-regression estimator
factory, prediction validator, result schemas, economic evaluator, and manifest.
Only synthetic state generation, orchestration, and HTML presentation are new.

The full plugin's job-state/checkpoint hooks are intentionally not called:
the demo fits its estimator inside a fold-local scikit-learn pipeline. It does
not run the platform's production search controller or imply that all optional
integrations are exercised.

The one-step outcome uses the following minute's price, while all feature and
context columns depend only on current/past synthetic information. Each fold
purges one training row. Overnight outcomes are excluded. Scaling is fitted
only on training rows. A prefix-invariance test checks the synthetic features.

The fixed always-flat baseline is descriptive. No parameter tuning, locked
holdout claim, calibration claim, significance claim, or promotion follows.
Manifest evidence completeness describes provenance, not scientific validity.
Git-less installations honestly report missing code identity.

## Optional platform capabilities

| Capability | Included | Offline demo |
|---|---|---|
| Acquisition/ingestion and DB schema helpers | Source | Not connected |
| Feature and label engines | Source | Small synthetic feature example only |
| Model catalog | Source | Logistic-regression estimator only |
| Shared validation, causality, economics, manifest | Source | Exercised |
| Full search and model promotion | Source | Not invoked |
| Research automation/RAG | Source | Not invoked; private corpus omitted |
| GPU/cloud queue and object storage | Source | Not invoked |
| Operational charts and console | Source/assets | Separate database-backed application |
| Paper/broker bridge | Source | Not invoked or enabled |
| RL plugin | Source | Label-reward offline experiment, not P&L-trained validation |

## Optional configuration

Copy `.env.example` to your own ignored `.env` only when configuring integrations.
Do not load an operator's environment for the offline demo. Set `VTECH_DATA_DIR`
and `MLFLOW_TRACKING_URI` to your own isolated locations before using legacy
platform commands. Some historical helpers still assume `/opt/vtech`; inspect
them and the configuration audit rather than executing them as setup steps.

The shipped `config.yaml` is a non-secret historical example, not the demo's
configuration. The demo records its actual parameters and execution assumptions
in each manifest. Dead legacy model/training settings do not govern the demo.

## Verification

Use `make test-publication`: explicit synthetic-demo and shared-contract test
paths, no live database or private artifacts. The larger test collection is
preserved as engineering reference and is not represented as a green,
service-free suite. No full-suite validation is claimed.

`requirements-demo.lock` records the base dependencies verified in a fresh
Python 3.12 environment. Install it before the editable package to reproduce
that dependency set. Other Python/platform combinations may need a compatible
resolution of `pyproject.toml` rather than this lock.

`requirements.lock` is retained as the historical full training/serving lock;
it is not the lightweight public-demo dependency set. The public installation
is defined by `pyproject.toml`; CI verifies Python 3.12. Optional cloud, deep
learning, broker, and database deployment environments are not certified by
this release gate.


## Recorded Transformer console and optional rerun

The [Robin walkthrough](examples/TRANSFORMER_WORKFLOW.md) adds a real file-backed
console profile with data/features, model comparison, saved synthetic backtest,
evaluation/promotion evidence, artifacts and local MLflow. It starts through
`python -m src.showcase serve`, without operational DB/live-worker imports.
This does not make the operational `src.webui.app` entry point database-free.
The exact synthetic fixture and reviewed model files are the narrow publication
exception; the existing logistic-regression demo remains available separately.
Use `requirements-showcase.lock` in a fresh Python 3.12 environment; the optional
CPU training profile is `requirements-transformer.lock`. Do not mix the viewer
profile with the newer scientific pins in `requirements-demo.lock`.
