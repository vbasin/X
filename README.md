# X — ML Research Platform

A source showcase of the VTech quantitative research platform: market-data
acquisition, causal feature engineering, configurable model experiments,
walk-forward validation, execution-aware economics, and an operational WebUI.

**This is research infrastructure, not a proven profitable trading system.**
No vendor market data, private trained models, credentials, or operating database
is distributed. Historical research paths have known limitations; see
[capabilities and limitations](docs/PUBLIC_SHOWCASE.md) and
[the configuration audit](docs/CONFIG_AUDIT.md).

## Run the offline demonstration

Python 3.10+; Python 3.12 is the release-validation target. No GPU, database,
Databento key, LLM key, cloud account, or environment file is needed.

```bash
git clone https://github.com/vbasin/X.git
cd X
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.lock  # Tested Python 3.12 dependency set
python -m pip install -e ".[dev]"
vtech-demo --output .demo-output/first-run
make test-publication
```

Open `.demo-output/first-run/report.html` in a browser. Each invocation requires
a new output directory; existing outputs are never overwritten.

The example composes **real platform modules**:

1. Generate reproducible synthetic prices and causal trailing features.
2. Join context with the shared strict as-of primitive.
3. Build five expanding calendar folds with a one-minute purge.
4. Train the existing logistic-regression plugin's estimator using per-fold scaling.
5. Validate and persist canonical out-of-fold predictions.
6. Price illustrative one-minute trades using executable synthetic bid/ask,
   explicit adverse slippage, and one round-trip commission.
7. Persist the shared experiment manifest, input hash, fold metrics, model
   specification, trade ledger, economics, and an offline HTML report.

**All results are synthetic.** Accuracy and dollar figures demonstrate plumbing,
not a trading edge. No checkpoint is exported, no model is promoted, and no
broker connection or order submission occurs. The report is a standalone offline
walkthrough, not a mock claiming to be the full operating console.

## Platform source map

| Area | Source |
|---|---|
| Acquisition and ingestion | `src/acquisition/`, `src/ingestion/` |
| Shared storage, causality, code identity, compute | `src/common/` |
| Features, outcomes, training, model plugins | `src/features/`, `src/training/`, `src/models/` |
| Experiment identity and validation | `src/experiment/`, `src/validation/` |
| Search, replay, economic evaluation | `src/search/`, `src/backtest/`, `src/evaluation/` |
| Serving, paper simulation, execution integration | `src/serving/`, `src/paper/`, `src/execution/` |
| Optional cloud execution | `src/gpu_runner/` |
| Optional research automation | `src/agentic/`, `src/scoping/`, `src/research_director/` |
| Operational console and rule models | `src/webui/`, `src/external_models/` |

The [architecture](docs/ARCHITECTURE.md) distinguishes implemented components,
experimental research, and incomplete lifecycle wiring. Not every implementation
uses the same economic or validation path.

## Optional integrations

Install only what you intend to configure:

```bash
python -m pip install -e ".[platform,training,serving,agents]"
# Add .[deep-learning] separately for TensorFlow model families.
```

The full console is `python -m src.webui.app --host 127.0.0.1 --port 8700`,
but its useful data-backed pages require your own PostgreSQL/TimescaleDB schema,
authorized data, and local configuration. It is a local operator console, **not
a public Internet service**. Do not expose it as a hosted demo.

Historical service units, deployment recipes, and operator Makefile targets
retain deployment assumptions and are not turnkey public setup instructions.
Cloud/LLM/vendor integrations can incur charges. Paper/execution integration is
not enabled by the offline demo. The complete platform currently requires the
source checkout; the installed wheel supports the synthetic demo independently.

## Documentation

- [Public scope, setup boundaries, and known limitations](docs/PUBLIC_SHOWCASE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Experiment artifacts](docs/EXPERIMENTS.md)
- [Causality and data contracts](docs/DATA_CONTRACTS.md)
- [Publication disposition](docs/PUBLICATION_SCOPE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Copyright and reuse

No open-source license is granted for the platform's original source.
Copyright remains with the respective copyright holders; public visibility does
not grant general reuse or redistribution permission beyond applicable law and
GitHub's terms. Third-party components retain their own licenses and notices.
