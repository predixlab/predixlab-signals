# PredixLab Signals

[Contributing](CONTRIBUTING.md) | [Security](SECURITY.md) | [Apache-2.0](LICENSE)

PredixLab Signals is PredixLab's equity turning-signal research and publishing component. It consumes trading calendars, constituent history, and normalized per-symbol CSV files, builds training data, performs walk-forward validation, trains LightGBM models, and converts daily scores into sparse events. It neither collects prices nor provides a public API.

The signal, backtest, API and dashboard code is open source (Apache-2.0).

Data collection runs as part of the hosted service and isn't in the public repositories.

The publishing layer is model-extensible: each model declares its own scored
panel contract, event extractor, event vocabulary, and signal filters. The
current implementation is the `pivot` model; its buy reversals, sell reversals,
and trend breakouts are isolated under `signals.models.pivot`.

Portfolio selection is separate from event publishing. `AlphaRankEnsemble`
combines 36/60/96-month `AlphaRankModel` submodels for the Top12Drop1
portfolio; it does not consume Pivot labels, events, or outputs.

> **Disclaimer:** This project is for research and engineering validation. Its output is not investment advice.

## Role in PredixLab

```text
Data collection (hosted service)
  └─ normalized CSV, trading calendar, constituent history
       └─ PredixLab Signals
            ├─ research: dataset → walk-forward training → evaluation/backtest/report
            └─ production: approved model → daily scoring → CSV/MongoDB/optional email
```

`train` writes **candidate models** only. Production jobs read **production models** manually installed in the shared data directory; they never train, promote, or fall back automatically.

## Quick start

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
git clone https://github.com/predixlab/predixlab-signals.git
cd predixlab-signals
uv sync --frozen --extra dev --extra job
uv run predixlab-signals --help
uv run pytest -q
```

The following commands require a full historical [input data tree](docs/data-contract.md),
supplied separately under `data/` (or `PREDIXLAB_TRAINING_DATA`):

```bash
export PREDIXLAB_TRAINING_DATA=/path/to/full-history
export PREDIXLAB_DATA=~/.predixlab

uv run predixlab-signals build-dataset --market us --horizon 10 --tag default --start 2015-01-01
uv run predixlab-signals train --market us --horizon 10 --tag default
uv run predixlab-signals evaluate --market us --horizon 10 --tag default
uv run predixlab-signals backtest --market us --horizon 10 --tag default
uv run predixlab-signals report --market us --horizon 10 --tag default
```

Research artifacts default to `artifacts/<market>/` in the repository. Evaluate a candidate and install a production model before running daily jobs; see [model training and promotion](docs/model-training.md) and [production operations](docs/operations.md).

## Documentation

Start at the [documentation index](docs/index.md), or use:

- [Setup, configuration, and commands](docs/setup-and-configuration.md)
- [Input data contract](docs/data-contract.md)
- [Dataset, features, and labels](docs/dataset-features-labels.md)
- [Model training and promotion](docs/model-training.md)
- [Scores, states, and events](docs/signals-gates-picks.md)
- [Evaluation, backtesting, and reports](docs/evaluation-backtesting.md)
- [Artifacts and output locations](docs/artifacts-and-reports.md)
- [Production operations](docs/operations.md)
- [Testing, security, and limitations](docs/testing-and-limitations.md)

The package version is `0.1.0`; interfaces and storage contracts remain pre-1.0.

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing features, labels, event
rules, or publishing contracts. Use
[GitHub Issues](https://github.com/predixlab/predixlab-signals/issues) for bugs
and proposals and [SECURITY.md](SECURITY.md) for private vulnerability reports.
Install model bundles only from trusted sources; integrity checks do not make
an untrusted pickle safe.

## License

Copyright 2026 PredixLab contributors.

Project source is licensed under the [Apache License 2.0](LICENSE).
Third-party datasets, models, and dependencies may have separate terms.
The source license is not a grant of market-data rights or hosted-service access.
