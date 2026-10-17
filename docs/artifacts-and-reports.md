# Artifacts and output locations

## Storage roots

| Root | Nature | Contents |
| --- | --- | --- |
| `PREDIXLAB_TRAINING_DATA` | Local, historical, research input | Full Collector history used to build training datasets |
| `PREDIXLAB_DATA` | Clean, persistent, production | Inference input, production models, daily predictions, and events |
| `PIVOT_ARTIFACTS` | Rebuildable, research | Datasets, candidate models, walk-forward predictions, evaluation, backtests, and reports |

Production jobs cannot depend on `PIVOT_ARTIFACTS` surviving across runs; a container directory may start empty each time.

## Paths

```text
PREDIXLAB_DATA/
├── models/<market>/
│   ├── pivot_h<horizon>_<tag>.pkl
│   └── pivot_h<horizon>_<tag>.meta.json       # Recommended, not required to load
├── predictions/<date>/<market>/<market>_pivot.csv
└── picks/<date>/<market>_pivot.csv

PIVOT_ARTIFACTS/<market>/
├── datasets/pivot_h<horizon>_<tag>.{parquet,meta.json}
├── models/pivot_h<horizon>_<tag>.{pkl,meta.json}
└── reports/
    ├── walkforward_h<horizon>_<tag>.parquet
    ├── walkforward_h<horizon>_<tag>.folds.json
    ├── evaluation_h<horizon>_<tag>.json
    ├── backtest_h<horizon>_<tag>.json
    ├── backtest_h<horizon>_<tag>_trades.parquet
    ├── backtest_h<horizon>_<tag>_turn.json
    ├── backtest_h<horizon>_<tag>_turn_trades.parquet
    ├── picks_report_h<horizon>_<tag>.html
    └── picks_report_h<horizon>_<tag>_picks.parquet
```

Backtests write only requested state or turn files. `report --output` places the final two files in its custom directory and derives `_picks.parquet` from the specified HTML filename.

## Daily CSV

The prediction file stores the full target-date score cross section, including symbols that did not trigger. The event file stores only published events; see [scores, states, and events](signals-gates-picks.md) for fields. On a successful zero-event day, the event CSV still contains headers, distinguishing normal zero events from a job that did not run.

The suffix is the registered `signal_model` name. Current files use `_pivot`;
future model runtimes write their own suffix without sharing or overwriting the
pivot files.

## MongoDB output

Each `pivot_picks` document is one event with unique key:

```text
(market, date, signal_model, symbol, event)
```

Reruns use upsert and do not delete same-day or historical events. Indexes also
support `market/date/signal_model/direction/rank` and `market/symbol/date`
queries. The `jobs` collection stores job status; it and email are best-effort
operational telemetry.

## Recommended archive

Archive dataset files and metadata; walk-forward predictions and folds; evaluation, trade, portfolio, and accuracy results; candidate bundles, metadata, and checksums; source revision, commands, environment, and non-default options; model approval records, production location, and production-bundle checksums.

A final bundle cannot replace walk-forward predictions because it cannot reproduce the historical out-of-sample sample. Sidecar `.meta.json` supports reading and audit; the embedded bundle manifest is authoritative for load-time integrity.

## Related documents

- [Model training and promotion](model-training.md)
- [Evaluation, backtesting, and reports](evaluation-backtesting.md)
- [Production operations](operations.md)
