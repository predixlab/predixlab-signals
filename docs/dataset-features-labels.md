# Dataset, features, and labels

## Build a dataset

```bash
uv run predixlab-signals build-dataset \
  --market us --horizon 10 --tag default --start 2015-01-01
```

Defaults are `market=us`, `horizon=10`, `tag=default`, `start=2015-01-01`, and `barrier-width=1.5`. Use `--symbols AAPL,MSFT` to limit symbols, `--end YYYY-MM-DD` to bound input, and `--all-history` only for diagnostics because it disables membership filtering and introduces selection bias.

The command reads full symbol history, membership intervals, and a market benchmark; computes backward-looking symbol and market features; builds forward labels starting on the next trading day; attaches OHLC and next-open fields for backtesting; filters membership and requested dates; makes cross-sectionally neutral targets; removes rows before slow-window warm-up; downcasts `float64` to `float32`; and writes Parquet plus matching `.meta.json`.

## Feature boundary

Features cover multi-period returns, momentum and moving-average trend; rolling range position, breakouts, drawdowns, and rebounds; RSI and mean reversion; realized volatility and ATR-scaled moves; volume, money flow, candlesticks, and gaps; and benchmark-relative return, beta, correlation, and market context.

Self-normalized features use a 252-trading-day trailing window with at least 120 observations and are clipped to `[-6, 6]`; they never use full-sample statistics.

Model selection excludes identifiers, raw price/volume, future or evaluation context such as `open_next` and `pivot`, labels/results (`y_*`, `label_*`, `fwd_*`), and existing prediction, score, and action fields. Market-level `mkt_*` fields are excluded by default. `train --include-market` explicitly enables them and changes the experiment definition and overfitting risk.

## Labels and evaluation context

Forward outcomes cover `t+1` through `t+H`. Hold labels compare the current
extreme at `t` with the following bounded window.

| Field family | Meaning |
| --- | --- |
| `y_mfe`, `y_mae` | Maximum favorable/adverse price movement |
| `y_edge` | Directional opportunity after adverse movement |
| `y_amplitude` | Forward movement magnitude |
| `y_swing_pos` | Current price location in the future swing range |
| `y_barrier`, `y_barrier_bars` | First ATR barrier reached and elapsed bars |
| `fwd_*` | Realized forward outcomes for evaluation/reporting only |
| `label_scale`, `atr_pct_label` | Label volatility scale |
| `pivot` | Zigzag turning point for retrospective evaluation |
| `y_pivot_low_hold`, `y_pivot_high_hold` | Whether the current extreme persists in the bounded future window |
| `y_pivot_low_return`, `y_pivot_high_return` | ATR-normalized directional future return, defined only when the corresponding hold succeeds |

`_cs` targets such as `y_swing_pos_cs` and `y_barrier_cs` remove the same-day market cross section and are the default final-model targets. Hold labels inspect only today and the bounded future window; prior trend is intentionally irrelevant. Conditional return labels use the configured model horizon and are clipped to three normalized units. Unresolved end-of-series rows remain missing rather than becoming negative samples. Zigzag labels remain evaluation-only.

## Leakage boundary

- Input features use present and past data only.
- Forward labels start on the next trading day.
- Unfinished terminal turning labels remain missing.
- Membership history determines training eligibility.
- Walk-forward splits purge overlapping labels and embargo adjacent periods.
- Action thresholds are calibrated on validation data, not test results.
- Event-gating aggregates use data no later than the event date.

## Output

```text
PIVOT_ARTIFACTS/<market>/datasets/
├── pivot_h<horizon>_<tag>.parquet
└── pivot_h<horizon>_<tag>.meta.json
```

Metadata records row count, symbol count, date range, horizon, and feature names. Archive Parquet and metadata together.

## Related documents

- [Input data contract](data-contract.md)
- [Model training and promotion](model-training.md)
- [Artifacts and output locations](artifacts-and-reports.md)
