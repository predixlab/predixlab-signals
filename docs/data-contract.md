# Input data contract

## Input data tree

Data collection runs as part of the hosted service and isn't in the public repositories.
It produces trading calendars, constituent history, and normalized per-symbol CSV files.
Signals requires these inputs in the layout below; it does not collect them.

Training reads the full historical tree from `PREDIXLAB_TRAINING_DATA`
(`data/` below the repository by default). Daily inference reads the clean tree
from `PREDIXLAB_DATA` (`~/.predixlab` by default):

```text
<DATA_ROOT>/
├── calendar/us.txt
├── instruments/us.txt
└── normalized_data/us_data/
    ├── AAPL.csv
    ├── MSFT.csv
    └── ^GSPC.csv
```

| Input | Format and use |
| --- | --- |
| `calendar/<market>.txt` | One `YYYY-MM-DD` trading day per line |
| `instruments/<market>.txt` | Headerless, tab-separated `symbol`, membership start, membership end |
| `normalized_data/<market>_data/<SYMBOL>.csv` | Normalized daily CSV per symbol |

The US benchmark is `^GSPC`; the China benchmark is `SH000300`. Files beginning with `^` may provide benchmark context but never enter the tradable-symbol list.

## Normalized-price fields

The minimum recognized fields are `date` and `close`. Full datasets, turning rules, and backtests also require `open`, `high`, and `low`; normal upstream output includes `symbol`, `volume`, and `factor`.

| Field | Meaning |
| --- | --- |
| `date` | Parseable trading date |
| `open/high/low/close` | Normalized OHLC |
| `volume` | Volume corresponding to normalized prices |
| `factor` | Factor used to restore trade prices and raw volume |

When `factor` exists:

```text
price      = close / factor
raw_volume = volume * factor
```

When the entire `factor` column is absent, the reader supplies `factor = 1.0`,
so `price = close` and `raw_volume = volume`. This allows files with and without
the optional column to coexist; explicit missing or invalid factors are not
replaced with that default.

Files are date-sorted, duplicate dates retain the last row, and rows with missing
dates/closes or non-positive OHLC values are dropped. Duplicate dates are recorded
before cleaning: daily inference rejects a constituent's target row when either
the target or previous trading session has duplicate records, even if their
values match. Duplicate benchmark records on either session fail the run.
Earlier historical duplicates retain the keep-last behavior.
With a calendar, non-calendar dates are also dropped.
Nonnumeric or infinite numeric fields are treated as missing, not passed into
price arithmetic. Daily inference additionally applies the
[production quality gate](operations.md#daily-data-quality-gate) before feature
construction and checks usable target-session coverage again before scoring.

## Historical-membership semantics

An instrument file may contain multiple inclusive intervals for one symbol. Missing or unreadable membership entries do not mean “valid for all history”: loading fails closed.

Dataset construction retains full price history for rolling-feature warm-up, calculates features and labels over continuous history, then keeps training rows only inside membership intervals. This prevents future membership information from entering early samples while still warming new constituents. `build-dataset --all-history` disables membership filtering and is diagnostic only; it introduces explicit historical-selection bias.

## Minimum runtime DataFrame contracts

| Object | Required columns |
| --- | --- |
| Price panel | `symbol`, `date`, `open`, `high`, `low`, `close`, `price`, `is_member` |
| Training dataset | `symbol`, `date`, `close`, `price` |
| Prediction table | `symbol`, `date`, `close`, `score`, `action` |
| Published-event table | `symbol`, `date`, `event`, `direction`, `rank` |

Boundary checks also validate non-empty data, parseable dates, and unique keys. Correct bad data during collection or normalization, not in downstream documentation or publishing.

## Date boundaries

- `build-dataset --start` trims after full-history feature warm-up; `--end` limits the input upper bound.
- `daily --end-date` requires that exact trading session and bounds input at it;
  missing data never moves the target backwards.
- Automatic `daily` targets yesterday and uses `calendar/<market>.txt` for both
  the trading-day check and the previous-session freshness comparison.

## AlphaRank publication contract

The daily job builds one labels-free inference panel, completes Pivot, then
scores the independently validated 36/60/96-month `AlphaRankEnsemble` with that
same panel. The latest session's deterministic top `ALPHARANK_TOP_N` rows
(default `15`, ordered by `alpha_score` descending then `symbol`) are written
to `ALPHARANK_COLLECTION` (default `alpharank`). Each BSON-clean row contains:

`market`, `date`, `symbol`, `rank`, `horizon`, `tag`, `generated_at`,
`signal_model="alpha_rank"`, `score` (`alpha_score`), `consensus`
(`alpha_consensus`), `predicted_excess_return` (`return_hat`),
`predicted_rank` (`rank_hat`), `top20_probability` (`p_top`), and `close`.
The score, consensus, predicted rank, and top-20 probability are finite
numbers clipped to `[0, 1]`; predicted excess return is a required finite
signed float, while close may be null.
AlphaRank uses `ALPHARANK_TAG` for its model/document version, defaulting to
the Pivot tag for compatibility; the shared horizon and inference panel remain
unchanged.

AlphaRank rows are staged under a stable `generation_id` and epoch-millisecond
`generation_version`. A compare-and-set manifest in
`ALPHARANK_GENERATIONS_COLLECTION` (default `alpharank_generations`) identifies
the active generation for each market/date/model; consumers must resolve that
manifest before reading signal rows.

The session is replaced idempotently on rerun, removing stale prior ranks.
The full latest-session AlphaRank audit CSV is stored beside the Pivot daily
prediction artifact as `<market>_alpharank.csv`. Dry runs write both CSVs but
do not connect to MongoDB.

## Related documents

- [Dataset, features, and labels](dataset-features-labels.md)
- [Production operations](operations.md)
- [Testing, security, and limitations](testing-and-limitations.md)
