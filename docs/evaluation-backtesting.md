# Evaluation, backtesting, and reports

## Shared input

`evaluate`, `backtest`, `report`, and `backfill-picks` read:

```text
PIVOT_ARTIFACTS/<market>/reports/walkforward_h<horizon>_<tag>.parquet
```

`train` produces this file with out-of-sample fold predictions. Do not recompute history with a final candidate or production model and describe it as out-of-sample.

## Prediction evaluation

```bash
uv run predixlab-signals evaluate --market us --horizon 10 --tag default --tolerance 3
```

Output covers score/forward-return rank correlations, realized outcomes by state, barrier AUC/Brier/calibration, future-hold AUC/base rate/lift, and diagnostics by state and context. `--tolerance` is the allowed trading-day difference between an event and confirmed Pivot, defaulting to `3`.

Output:

```text
PIVOT_ARTIFACTS/<market>/reports/evaluation_h<horizon>_<tag>.json
```

## Backtesting

```bash
# Persistent-state strategy
uv run predixlab-signals backtest \
  --market us --horizon 10 --tag default \
  --signal state --actions STRONG_BUY --max-positions 20

# Turning-event strategy
uv run predixlab-signals backtest \
  --market us --horizon 10 --tag default \
  --signal turn --max-hold 30

# AlphaRank Top12Drop1 portfolio
uv run predixlab-signals backtest-alpha-rank \
  --market us --horizon 10 --tag default \
  --start 2024-01-01
```

Defaults: `signal=state`, `actions=STRONG_BUY`, `max-hold=15`, `cost-bps=10`, and `max-positions=20`. Turn backtests enter only on `BUY_TURN` and exit oppositely on `SELL_TURN`; `TREND_BUY` and `BASE_BREAKOUT` do not enter this book automatically.

Simulation uses a signal from `t`-day close to enter at `t+1` open; targets and stops use model amplitude with ATR bounds; checks intraday high/low only after entry; treats simultaneous target/stop touches as stop-first; exits on opposing event, target, stop, or maximum hold; charges costs per leg; and uses fixed, equally weighted slots with unused capacity as cash.

```text
PIVOT_ARTIFACTS/<market>/reports/
├── backtest_h<horizon>_<tag>.json
├── backtest_h<horizon>_<tag>_trades.parquet
├── backtest_h<horizon>_<tag>_turn.json
└── backtest_h<horizon>_<tag>_turn_trades.parquet
```

`_turn` files appear only for `--signal turn`. No-trade backtests exit non-zero and produce no normal result.

AlphaRank uses the 96-month model's return-rank and Top20 heads for ranking and
retains names whose three-model consensus is in the day's top 45%, whose
20-session return is positive, and whose price
is above its 60-session moving average. The formal portfolio holds 12 names,
normally replaces one per session, executes at the following close, and charges
10 bps per buy or sell leg.

The trend/probability filter controls new entries. Existing holdings that fail
it move to the front of the exit queue but remain subject to Drop1; a missing
symbol is the only forced liquidation outside that limit.

Use `--portfolio-start YYYY-MM-DD` to warm indicators and market-state history
but reset holdings to cash and build a new initial TopN portfolio on that
signal date. This differs from `--start`, which only trims reported metrics.

The portfolio enters a persistent cash state after two consecutive sessions
where the benchmark has non-positive 20-session return, is below its 50-session
average, and five-session average breadth is at most 50%. It exits after two
consecutive sessions with at least 1% five-session benchmark return, at least
50% breadth, and improving breadth. A benchmark gain of at least 2% in one
session is treated as a strong reversal and restores exposure immediately.
After either reversal path, a five-session grace period prevents an immediate
return to the bear state.

The detector can enter earlier when model breadth collapses: the number of
filtered candidates must fall below 90% of its trailing 20-session median and
drop at least 15% over five sessions, while the benchmark is below its
50-session average with negative 20-session return. The early path is disabled
during high-volatility shocks (`mkt_vol_ratio > 1.5`), which are handled by the
regular trend-and-breadth confirmation because flash crashes tend to
mean-revert. Both bear entry and reversal are determined from signal-day close
data; liquidation or re-entry occurs at the next close.

Outside the bear state, portfolio exposure is volatility-scaled. At each month
start, the strategy uses only the preceding 12 months of shadow portfolio
returns to choose a target from 15%–40% annualized volatility. Daily exposure
is then `min(1, target / trailing-20-session realized volatility)`. Exposure
recovers automatically as realized volatility falls; changing exposure incurs
the same 10 bps cost assumption.

Build a self-contained portfolio report after backtesting:

```bash
uv run predixlab-signals alpha-rank-report \
  --market us --horizon 10 --tag default \
  --start 2024-01-01
```

The HTML contains interactive net-equity, drawdown, and cash-exposure charts;
calendar-year and monthly rolling returns; bear-state sessions; and a primary per-symbol holding-period
table with entry/exit dates, compounded gross and after-cost net return,
run-up, drawdown, and expandable daily return paths. A secondary portfolio log
retains searchable daily holdings, buys, and sells.

For entry-timing stability, each monthly start can be simulated as a new
account: model, bear-state, and volatility history are warmed from prior data,
but holdings are reset to cash and a fresh Top12 is built. The generated
artifact is `alpha_rank_monthly_entry_stability_h<horizon>_<tag>.html`.

## Event-accuracy report

```bash
uv run predixlab-signals report --market us --horizon 10 --tag default
```

Use `--start`/`--end` to restrict the period or `--output` for an HTML path. The command saves:

```text
PIVOT_ARTIFACTS/<market>/reports/
├── picks_report_h<horizon>_<tag>.html
└── picks_report_h<horizon>_<tag>_picks.parquet
```

With `--output /path/report.html`, the event table is
`/path/report_picks.parquet`. Directional accuracy is the share of actionable,
completed-horizon events where `signal_return > 0`; incomplete horizons remain
pending and are excluded. Pivot accuracy uses `pivot_event`; confirmed
`BUY_TURN` and `SELL_TURN` events are never relabeled by a downstream trend
filter. `TREND_BUY` and `BASE_BREAKOUT` are directional only.

## Interpretation

Ranking, probability calibration, event lift, and after-cost return answer different questions. Review stability by fold, date, and direction; compare gross and net results; and inspect turnover, exposure, capacity, and event gaps. Daily data cannot recover intraday target/stop order, so stop-first is deliberately conservative. Backtests omit market impact, capacity, borrow availability, and taxes.

## Related documents

- [Model training and promotion](model-training.md)
- [Scores, states, and events](signals-gates-picks.md)
- [Artifacts and output locations](artifacts-and-reports.md)
