"""Event-driven trade simulation and portfolio accounting."""

from predixlab_signals.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    baseline_random,
    buy_and_hold,
    max_drawdown,
    run_backtest,
    summarise,
)
from predixlab_signals.backtest.portfolio import (
    benchmark_daily,
    benchmark_performance,
    build_position_book,
    daily_returns,
    performance,
    relative_performance,
)
from predixlab_signals.backtest.topn_dropouts import (
    TopNDropoutsConfig,
    TopNDropoutsResult,
    run_topn_dropouts,
    topn_performance,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "baseline_random",
    "buy_and_hold",
    "summarise",
    "max_drawdown",
    "build_position_book",
    "daily_returns",
    "performance",
    "benchmark_performance",
    "benchmark_daily",
    "relative_performance",
    "TopNDropoutsConfig",
    "TopNDropoutsResult",
    "run_topn_dropouts",
    "topn_performance",
]
