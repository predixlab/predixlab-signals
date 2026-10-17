"""Tests for the trade simulator, portfolio accounting and the advice layer.

The rules asserted here are the ones that separate a believable backtest from a
flattering one: fills happen on the next bar, ambiguous bars resolve against the
strategy, and exposure can never exceed the capital allocated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.backtest.engine import BacktestConfig, baseline_random, run_backtest
from predixlab_signals.backtest.portfolio import (
    build_position_book,
    daily_returns,
    performance,
    relative_performance,
)


def _preds(n: int = 120, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    frames = []
    for symbol in ("AAA", "BBB"):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "close": close,
                "price": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "open_next": np.roll(close, -1),
                "atr_pct_label": np.full(n, 0.02),
                "exp_gain_pct": np.full(n, 0.04),
                "exp_loss_pct": np.full(n, -0.02),
                "score": rng.normal(size=n),
                "action": rng.choice(["HOLD", "BUY", "STRONG_BUY", "SELL"], n),
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
def test_backtest_requires_price_columns():
    with pytest.raises(KeyError, match="missing columns"):
        run_backtest(pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2020-01-01")]}))


def test_entry_is_always_the_next_bar():
    """A signal from bar t must never fill on bar t's own close."""
    preds = _preds()
    result = run_backtest(preds, BacktestConfig(horizon=10))
    bars = preds.set_index(["symbol", "date"])
    for trade in result.trades.itertuples(index=False):
        assert trade.entry_date in bars.loc[trade.symbol].index
        assert trade.exit_date >= trade.entry_date


def test_duplicate_session_rows_cannot_create_a_same_session_entry():
    preds = _preds(n=12).query("symbol == 'AAA'").reset_index(drop=True)
    preds.loc[:, "action"] = "HOLD"
    preds.loc[0, "action"] = "BUY"
    duplicated = pd.concat([preds.iloc[[0]], preds], ignore_index=True)

    result = run_backtest(duplicated, BacktestConfig(horizon=5))

    assert result.trades["entry_date"].iloc[0] == preds["date"].iloc[1]


def test_stop_wins_an_ambiguous_bar():
    """When one bar spans both barriers we cannot know the order — assume the worst."""
    n = 12
    dates = pd.bdate_range("2021-01-01", periods=n)
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    high[3] = 200.0  # both target and stop reachable on the same bar
    low[3] = 50.0
    preds = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "close": close,
            "price": close,
            "high": high,
            "low": low,
            "open_next": close,
            "atr_pct_label": np.full(n, 0.02),
            "exp_gain_pct": np.full(n, 0.05),
            "exp_loss_pct": np.full(n, -0.05),
            "score": np.ones(n),
            "action": ["STRONG_BUY"] + ["HOLD"] * (n - 1),
        }
    )
    result = run_backtest(preds, BacktestConfig(horizon=5, entry_actions=("STRONG_BUY",)))
    assert result.trades["exit_reason"].iloc[0] == "stop"
    assert result.trades["net_return"].iloc[0] < 0


def test_costs_are_charged_on_both_legs():
    preds = _preds()
    free = run_backtest(preds, BacktestConfig(horizon=10, cost_bps=0.0))
    costly = run_backtest(preds, BacktestConfig(horizon=10, cost_bps=50.0))
    diff = costly.trades["net_return"].mean() - free.trades["net_return"].mean()
    assert diff == pytest.approx(-2 * 50.0 / 10_000.0, abs=1e-9)
    np.testing.assert_allclose(free.trades["gross_return"], costly.trades["gross_return"])


def test_no_overlapping_positions_per_symbol():
    result = run_backtest(_preds(), BacktestConfig(horizon=10, max_concurrent_per_symbol=1))
    for _, group in result.trades.groupby("symbol"):
        ordered = group.sort_values("entry_date")
        assert (ordered["entry_date"].to_numpy()[1:] >= ordered["exit_date"].to_numpy()[:-1]).all()


def test_random_baseline_uses_the_same_signals():
    preds = _preds()
    cfg = BacktestConfig(horizon=10)
    shuffled = baseline_random(preds, cfg)
    assert not shuffled.trades.empty


# --------------------------------------------------------------------------
def test_position_book_respects_the_slot_limit():
    result = run_backtest(_preds(n=300), BacktestConfig(horizon=10))
    book = build_position_book(result.trades, max_positions=2)
    events = []
    for trade in book.itertuples(index=False):
        events.append((trade.entry_date, 1))
        events.append((trade.exit_date, -1))
    events.sort()
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    assert peak <= 2


def test_exposure_never_exceeds_capital():
    """The exit bar frees its slot, so it must not be double-counted as occupied."""
    preds = _preds(n=300)
    result = run_backtest(preds, BacktestConfig(horizon=10))
    book = build_position_book(result.trades, max_positions=2)
    daily = daily_returns(preds, book, max_positions=2)
    assert daily["exposure"].max() <= 1.0 + 1e-9


def test_performance_metrics_are_coherent():
    idx = pd.bdate_range("2021-01-01", periods=504)
    rng = np.random.default_rng(2)
    daily = pd.DataFrame(
        {"ret": rng.normal(0.0006, 0.01, len(idx)), "n_open": 5.0, "exposure": 0.5}, index=idx
    )
    perf = performance(daily)
    assert perf["days"] == len(idx)
    assert perf["max_drawdown"] <= 0.0
    assert perf["volatility"] > 0.0
    assert np.isfinite(perf["sharpe"])


def test_relative_performance_recovers_a_known_beta():
    idx = pd.bdate_range("2021-01-01", periods=500)
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    port = 1.5 * bench + 0.0002  # beta 1.5, positive alpha by construction
    rel = relative_performance(
        pd.DataFrame({"ret": port}), pd.DataFrame({"ret": bench})
    )
    assert rel["beta"] == pytest.approx(1.5, abs=0.05)
    assert rel["alpha_annual"] > 0


def test_trend_entries_are_excluded_from_the_default_backtest():
    """`TREND_BUY` is informational: it must not silently enter the book.

    Traded alongside the pivot events it lowers profit factor at every holding
    period (1.178 -> 1.147 at 20 sessions) while raising exposure, so the
    default entry set stays `(BUY_TURN,)`.
    """
    import inspect

    from predixlab_signals import cli

    source = inspect.getsource(cli.cmd_backtest)
    assert "entry_actions=(BUY_TURN,)" in source
    assert "TREND_BUY" not in source
