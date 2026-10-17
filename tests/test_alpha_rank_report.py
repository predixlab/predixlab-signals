from __future__ import annotations

import pandas as pd
import pytest

from predixlab_signals.alpha_rank_report import (
    _holding_episodes,
    render_alpha_rank_report,
)


def test_alpha_rank_report_contains_equity_and_holdings():
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-03", periods=3),
            "signal_date": pd.bdate_range("2024-01-02", periods=3),
            "ret": [0.01, -0.02, 0.03],
            "turnover": [1.0, 0.2, 0.2],
            "positions": [2, 0, 2],
            "target_positions": [2, 0, 2],
            "cash_weight": [0.0, 1.0, 0.0],
            "exposure_regime": ["full", "bear", "full"],
            "bear_trigger": [False, True, False],
            "bear_reversal": [False, False, True],
            "candidates": [10, 8, 9],
            "execution_price": "next_close",
        }
    )
    selections = pd.DataFrame(
        {
            "signal_date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-01-04"),
            ],
            "symbol": ["AAA", "BBB", "AAA", "CCC"],
            "rank": [1, 2, 1, 2],
            "score": [0.9, 0.8, 0.95, 0.7],
            "status": ["bought", "bought", "bought", "bought"],
            "entry_date": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-05"),
                pd.Timestamp("2024-01-05"),
            ],
            "return_date": [
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-01-08"),
                pd.Timestamp("2024-01-08"),
            ],
            "stock_return": [0.01, -0.02, 0.03, 0.04],
            "execution_price": "next_close",
        }
    )
    html, performance = render_alpha_rank_report(
        daily,
        selections,
        market="us",
        horizon=10,
        tag="test",
        cost_bps=25.0,
    )
    assert "Individual holding-period returns" in html
    assert "Daily portfolio log" in html
    assert "Risk-off sessions" in html
    assert "Monthly rolling stability" in html
    assert "rolling_months" in html
    assert "AAA" in html and "CCC" in html
    assert "Net equity" in html
    assert "Drawdown" in html
    assert "chart-overlay" in html
    assert "syncHover" in html
    assert "page-size" in html
    assert "data-key=\"return\"" in html
    assert "data-expand" in html
    assert "episode-symbol-filter" in html
    assert "data-episode-key=\"net_return\"" in html
    assert "Daily open-to-open return path" in html
    assert '"cost_bps":25.0' in html
    assert performance["max_drawdown"] < 0


def test_bear_liquidation_closes_the_last_holding_episode():
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-03", periods=2),
            "signal_date": pd.bdate_range("2024-01-02", periods=2),
            "ret": [0.01, -0.001],
            "turnover": [1.0, 1.0],
            "positions": [1, 0],
            "target_positions": [1, 0],
            "cash_weight": [0.0, 1.0],
            "exposure_regime": ["full", "bear"],
            "bear_trigger": [False, True],
            "bear_reversal": [False, False],
            "candidates": [5, 2],
            "execution_price": "next_close",
        }
    )
    selections = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02")],
            "entry_date": [pd.Timestamp("2024-01-03")],
            "return_date": [pd.Timestamp("2024-01-04")],
            "symbol": ["AAA"],
            "rank": [1],
            "score": [0.9],
            "status": ["bought"],
            "stock_return": [0.01],
            "execution_price": ["next_close"],
        }
    )
    html, _ = render_alpha_rank_report(
        daily,
        selections,
        market="us",
        horizon=10,
        tag="test",
    )
    assert '"status":"closed"' in html


def test_carry_in_episode_charges_only_the_exit_leg():
    selections = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-03")],
            "entry_date": [pd.Timestamp("2024-01-04")],
            "return_date": [pd.Timestamp("2024-01-05")],
            "symbol": ["AAA"],
            "rank": [1],
            "score": [0.9],
            "status": ["held"],
            "stock_return": [0.10],
            "execution_price": ["next_close"],
        }
    )
    episode = _holding_episodes(
        selections,
        cost_bps=100.0,
        open_symbols=set(),
    )[0]
    assert episode["net_return"] == pytest.approx(1.10 * 0.99 - 1.0)


def test_all_cash_report_serializes_undefined_sharpe_as_null():
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-03", periods=2),
            "signal_date": pd.bdate_range("2024-01-02", periods=2),
            "ret": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "positions": [0, 0],
            "target_positions": [0, 0],
            "cash_weight": [1.0, 1.0],
            "exposure_regime": ["bear", "bear"],
            "bear_trigger": [True, False],
            "bear_reversal": [False, False],
            "candidates": [2, 2],
            "execution_price": "next_close",
        }
    )
    selections = pd.DataFrame(
        columns=[
            "signal_date",
            "entry_date",
            "return_date",
            "symbol",
            "rank",
            "score",
            "status",
            "stock_return",
            "execution_price",
        ]
    )
    html, _ = render_alpha_rank_report(
        daily,
        selections,
        market="us",
        horizon=10,
        tag="test",
    )
    assert '"sharpe":null' in html
    assert 'p.sharpe==null?"N/A"' in html


def test_report_excludes_episodes_ending_before_the_daily_range():
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-02")],
            "signal_date": [pd.Timestamp("2024-02-01")],
            "ret": [0.01],
            "turnover": [1.0],
            "positions": [1],
            "target_positions": [1],
            "cash_weight": [0.0],
            "exposure_regime": ["full"],
            "bear_trigger": [False],
            "bear_reversal": [False],
            "candidates": [5],
            "execution_price": ["next_close"],
        }
    )
    selections = pd.DataFrame(
        {
            "signal_date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-02-01"),
            ],
            "entry_date": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-02-02"),
            ],
            "return_date": [
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-02-05"),
            ],
            "symbol": ["OLD", "NEW"],
            "rank": [1, 1],
            "score": [0.8, 0.9],
            "status": ["bought", "bought"],
            "stock_return": [0.02, 0.01],
            "execution_price": ["next_close", "next_close"],
        }
    )
    html, _ = render_alpha_rank_report(
        daily,
        selections,
        market="us",
        horizon=10,
        tag="test",
    )
    assert '"symbol":"OLD"' not in html
    assert '"symbol":"NEW"' in html
