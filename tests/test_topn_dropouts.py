from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.backtest import (
    TopNDropoutsConfig,
    run_topn_dropouts,
    topn_performance,
)


def _panel(periods: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for number, symbol in enumerate(
        ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"),
        start=1,
    ):
        close = 100.0 * np.cumprod(
            np.full(periods, 1.0 + number / 1000.0)
        )
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": close[index],
                    "ret_20": 0.10,
                    "px_sma60": 0.05,
                    "p_top": float(number),
                    "alpha_score": float(number),
                    "alpha_consensus": float(number),
                    "mkt_ret_1": 0.0,
                    "mkt_ret_5": 0.02,
                    "mkt_ret_20": 0.02,
                    "mkt_px_sma50": 0.01,
                    "mkt_vol_ratio": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_alpha_rank_topn_selects_the_best_eligible_names():
    result = run_topn_dropouts(
        _panel(),
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=0),
    )
    first_date = result.selections["signal_date"].min()
    first = result.selections[result.selections["signal_date"] == first_date]
    assert set(first["symbol"]) == {"EEE", "FFF"}
    assert (first["status"] == "bought").all()


def test_next_close_returns_and_costs_are_applied():
    panel = _panel()
    free = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=0),
    )
    costly = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=10),
    )
    expected = -costly.daily["turnover"] * 10.0 / 10_000.0
    np.testing.assert_allclose(costly.daily["ret"] - free.daily["ret"], expected)
    assert set(costly.daily["execution_price"]) == {"next_close"}


def test_ineligible_holdings_are_prioritized_but_respect_drop_one():
    panel = _panel()
    signal_date = sorted(panel["date"].unique())[5]
    panel.loc[
        (panel["date"] == signal_date)
        & panel["symbol"].isin(("EEE", "FFF")),
        "ret_20",
    ] = -0.10
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=0),
    )
    holdings = result.selections[
        result.selections["signal_date"] == signal_date
    ]
    assert len({"EEE", "FFF"} & set(holdings["symbol"])) == 1


def test_missing_holding_is_counted_as_a_forced_sale():
    panel = _panel()
    missing_date = sorted(panel["date"].unique())[5]
    panel = panel[
        ~((panel["symbol"] == "FFF") & (panel["date"] == missing_date))
    ]
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=10),
    )
    day = result.daily[result.daily["signal_date"] == missing_date].iloc[0]
    assert day["turnover"] >= 1.0


def test_bear_state_holds_cash_until_strong_reversal():
    panel = _panel()
    dates = sorted(panel["date"].unique())
    for index, date in enumerate(dates):
        if 22 <= index <= 27:
            panel.loc[panel["date"] == date, "mkt_ret_20"] = -0.03
            panel.loc[panel["date"] == date, "mkt_px_sma50"] = -0.02
            losing = panel["symbol"].isin(("AAA", "BBB", "CCC", "DDD"))
            panel.loc[(panel["date"] == date) & losing, "ret_20"] = -0.10
        if index == 28:
            panel.loc[panel["date"] == date, "mkt_ret_1"] = 0.03
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(
            top_n=2,
            n_drop=1,
            cost_bps=0,
            bear_entry_confirm_days=2,
        ),
    )
    bear = result.daily[result.daily["exposure_regime"] == "bear"]
    assert not bear.empty
    assert (bear["positions"] == 0).all()
    assert result.daily["bear_trigger"].sum() >= 1
    assert result.daily["bear_reversal"].sum() >= 1


def test_portfolio_start_resets_initial_holdings():
    panel = _panel()
    start = sorted(panel["date"].unique())[10]
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(top_n=2, n_drop=1, cost_bps=0),
        portfolio_start=start,
    )
    assert result.selections["signal_date"].min() == start
    first = result.selections[result.selections["signal_date"] == start]
    assert (first["status"] == "bought").all()


def test_portfolio_start_preserves_risk_history_but_not_holdings():
    panel = _panel(periods=320)
    start = sorted(panel["date"].unique())[280]
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(
            top_n=2,
            n_drop=1,
            cost_bps=0,
            volatility_targets=(0.15,),
        ),
        portfolio_start=start,
    )
    first_daily = result.daily.iloc[0]
    first_selection = result.selections[
        result.selections["signal_date"] == start
    ]
    assert pd.notna(first_daily["target_volatility"])
    assert (first_selection["status"] == "bought").all()


def test_adaptive_volatility_reduces_exposure_after_high_volatility():
    panel = _panel(periods=320)
    dates = sorted(panel["date"].unique())
    volatile_dates = dates[250:280]
    for index, date in enumerate(volatile_dates):
        panel.loc[panel["date"] == date, "close"] *= (
            1.15 if index % 2 == 0 else 0.85
        )
    result = run_topn_dropouts(
        panel,
        TopNDropoutsConfig(
            top_n=2,
            n_drop=1,
            cost_bps=0,
            volatility_targets=(0.15,),
        ),
    )
    assert result.daily["exposure_scale"].min() < 1.0
    assert result.daily["target_volatility"].notna().any()


def test_topn_performance_reports_calendar_year_returns():
    dates = pd.bdate_range("2024-01-02", "2025-12-31")
    daily = pd.DataFrame(
        {
            "ret": 0.001,
            "turnover": 0.2,
            "positions": 15,
            "target_positions": 15,
            "cash_weight": 0.0,
            "candidates": 80,
            "exposure_regime": "full",
        },
        index=dates,
    )
    perf = topn_performance(daily)
    assert set(perf["yearly_returns"]) == {"2024", "2025"}
    assert perf["yearly_returns"]["2024"] > 0.25
    assert perf["max_drawdown"] == pytest.approx(0.0)
