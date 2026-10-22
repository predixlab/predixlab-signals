# PredixLab Signals documentation

For installation, start with [README](../README.md). Community policies are in
[CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md);
project licensing is in [LICENSE](../LICENSE).

## Role

Data collection runs as part of the hosted service and isn't in the public repositories.

PredixLab Signals connects PredixLab's market-data collection and signal-consumption layers:

- Upstream: trading calendars, constituent history, and normalized per-symbol CSV files. See the [input data contract](data-contract.md) for the required layout.
- This project: datasets, training, out-of-sample evaluation, signal generation, and publishing.
- Downstream: daily CSV files or the MongoDB `pivot_picks` collection.
- Excluded: price collection, collection scheduling, APIs, and trade execution.

Research and production are separated by manual model promotion:

```text
Research
Input data → features/labels → walk-forward training → out-of-sample predictions
           → evaluation/backtest/accuracy report → candidate model

Production
Approved production model + input data → daily score → turning event
                                      → CSV + MongoDB + optional email
```

## Shared terms

| Term | Meaning |
| --- | --- |
| Directional state | Persistent daily directional classification for every eligible symbol: `STRONG_BUY`, `BUY`, `HOLD`, `SELL`, or `STRONG_SELL`; see [state meanings](signals-gates-picks.md#a-state-is-not-an-event) |
| Turning event | Sparse event after model, price-context, and temporal rules |
| Published event (pick) | Event passing publication rules and entering CSV/MongoDB; not a daily Top-N stock list |
| Candidate model | Model written by `train` to `PIVOT_ARTIFACTS`, pending evaluation |
| Production model | Operator-approved model installed at `PREDIXLAB_DATA/models/<market>/` and read by `daily` |
| Walk-forward prediction | Out-of-sample prediction for each test date, made by a fold model that did not see that date |

A valid trading day can have no events. Zero picks are normal and do not mean the job failed.

## Read by task

### First use
1. [Setup, configuration, and commands](setup-and-configuration.md)
2. [Input data contract](data-contract.md)
3. [Dataset, features, and labels](dataset-features-labels.md)

### Training and validation
1. [Model training and promotion](model-training.md)
2. [Evaluation, backtesting, and reports](evaluation-backtesting.md)
3. [Scores, states, and events](signals-gates-picks.md)

### Production and troubleshooting
1. [Artifacts and output locations](artifacts-and-reports.md)
2. [Production operations](operations.md)
3. [Testing, security, and limitations](testing-and-limitations.md)

## Key boundaries

1. Features use only present and past data; future data is limited to training labels and retrospective evaluation.
2. Constituent history defines training eligibility; pre-membership prices are used only to warm rolling windows.
3. Evaluation, event replay, and backtests use saved walk-forward predictions, not a final model recomputing history.
4. `train` never writes a production model; `daily` never trains one.
5. Research artifacts use a separate root from production models and daily output.
6. Event extraction requires continuous history, not a one-day cross section.

> **Disclaimer:** Output is for research and engineering validation, not investment advice. Version `0.1.0` is pre-1.0.
