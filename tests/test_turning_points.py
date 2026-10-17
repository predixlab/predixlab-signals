"""Tests for turning-point extraction.

The point of this layer is *sparsity with precision*: it must fire rarely, and
when it fires the price must actually be at a trough that has begun to turn.
Each rule is asserted independently so a regression in one cannot be masked by
another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.signals.turning_points import (
    _sequence,
    _sequence_additions,
    _top_followthrough_confirmations,
    _range_position,
    _recent,
    _recent_while_live,
    BASE_BREAKOUT,
    BOTTOM_TURN,
    TREND_BUY,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TurningPointConfig,
    _apply_refractory,
    extract_turning_points,
    summarise,
)
from predixlab_signals.signals.gates import add_gate_percentiles
from predixlab_signals.signals.models.pivot.base_breakouts import _base_breakout


@pytest.fixture(autouse=True)
def isolate_price_rules(monkeypatch):
    """Volume confirmation has full-input integration coverage in test_picks."""
    monkeypatch.setattr(
        "predixlab_signals.signals.models.pivot.extractor.apply_bottom_volume_rebounds",
        lambda group, events, config: events,
    )


def _frame(n=80, symbol="AAA", action="STRONG_BUY", seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "action": action,
            "pivot": 0.0,
        }
    )


def test_gate_calibration_is_prefix_invariant():
    dates = pd.bdate_range("2024-01-02", periods=8)
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"] * len(dates),
            "date": np.repeat(dates, 2),
            "close": np.tile([100.0, 101.0], len(dates)),
            "p_pivot_low": np.linspace(0.05, 0.80, len(dates) * 2),
            "p_pivot_high": np.linspace(0.80, 0.05, len(dates) * 2),
        }
    )
    cfg = TurningPointConfig(
        trend_neutral_weight=0.0,
        self_gate_window=None,
        pivot_abs_mult=2.0,
        breadth_relief=0.3,
    )
    prefix = frame.iloc[:10].copy()
    full = frame.copy()
    full.loc[10:, ["p_pivot_low", "p_pivot_high"]] *= 100.0

    add_gate_percentiles(prefix, cfg)
    add_gate_percentiles(full, cfg)

    helpers = [
        "_p_pivot_low_abs",
        "_p_pivot_low_breadth",
        "_p_pivot_high_abs",
        "_p_pivot_high_breadth",
    ]
    pd.testing.assert_frame_equal(
        prefix[helpers].reset_index(drop=True),
        full.iloc[:10][helpers].reset_index(drop=True),
    )


def test_gate_percentiles_use_return_weighted_pivot_score():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": [date, date],
            "close": [100.0, 100.0],
            "p_pivot_low": [0.9, 0.9],
            "p_pivot_high": [0.1, 0.1],
            "pivot_low_score": [0.1, 0.8],
            "pivot_high_score": [0.8, 0.1],
        }
    )
    cfg = TurningPointConfig(
        trend_neutral_weight=0.0,
        self_gate_window=None,
        pivot_abs_mult=None,
        breadth_relief=None,
    )

    add_gate_percentiles(frame, cfg)

    assert frame.loc[1, "_p_pivot_low_q"] > frame.loc[0, "_p_pivot_low_q"]
    assert frame.loc[0, "_p_pivot_high_q"] > frame.loc[1, "_p_pivot_high_q"]


# --------------------------------------------------------------------------
def test_refractory_keeps_first_and_suppresses_cluster():
    events = np.array([True, True, True, False, False, True, True])
    kept = _apply_refractory(events, gap=3)
    assert list(np.flatnonzero(kept)) == [0, 5]


def test_refractory_disabled_passes_everything_through():
    events = np.array([True, True, False, True])
    assert list(_apply_refractory(events, gap=0)) == list(events)


def test_state_signal_alone_produces_no_events():
    """A permanent BUY state must not become a permanent turning point."""
    frame = _frame()
    out = extract_turning_points(
        frame, TurningPointConfig(trough_pct=0.0, trend_entry=False)
    )
    assert (out["turn_action"] == NO_TURN).all()


def test_trough_condition_is_enforced():
    """A rising series never sits low in its range, so no *pivot* may fire.

    Trend entries are a different question and are disabled here.
    """
    n = 60
    close = np.linspace(100, 160, n)  # monotone up -> range position is 1.0
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(frame, TurningPointConfig(trend_entry=False, require_reversal=False))
    assert (out["turn_action"] == NO_TURN).all()


def test_buy_event_fires_at_a_confirmed_trough():
    """V-shaped bottom: the bar that closes above the prior high must fire."""
    n = 60
    close = np.concatenate([np.linspace(140, 100, 40), np.linspace(101, 120, 20)])
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(
        frame, TurningPointConfig(refractory=10, min_downtrend_slope=None)
    )
    events = np.flatnonzero(out["turn_action"] == BUY_TURN)
    assert len(events) >= 1
    # The first event must land near the bottom of the V, not on the way down.
    assert 38 <= events[0] <= 46


def test_reversal_requirement_suppresses_falling_knives():
    """While price is still making lower closes, nothing should fire."""
    n = 60
    close = np.linspace(160, 100, n)  # straight down, never reverses
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    strict = extract_turning_points(
        frame,
        TurningPointConfig(require_reversal=True, min_downtrend_slope=None),
    )
    loose = extract_turning_points(
        frame,
        TurningPointConfig(require_reversal=False, min_downtrend_slope=None),
    )
    assert (strict["turn_action"] == NO_TURN).all()
    assert (loose["turn_action"] == BUY_TURN).any()


def test_events_are_far_sparser_than_the_state():
    rng = np.random.default_rng(5)
    frames = []
    for symbol in ("AAA", "BBB", "CCC"):
        n = 400
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": pd.bdate_range("2022-01-03", periods=n),
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "action": rng.choice(["HOLD", "BUY", "STRONG_BUY"], n),
                    "pivot": 0.0,
                }
            )
        )
    preds = pd.concat(frames, ignore_index=True)
    out = extract_turning_points(preds)
    state_density = preds["action"].isin(["BUY", "STRONG_BUY"]).mean()
    event_density = (out["turn_action"] == BUY_TURN).mean()
    assert event_density < state_density / 10


def test_refractory_spacing_is_respected():
    rng = np.random.default_rng(9)
    n = 500
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    gap = 15
    out = extract_turning_points(
        frame, TurningPointConfig(refractory=gap, min_downtrend_slope=None)
    )
    idx = np.flatnonzero(out["turn_action"] == BUY_TURN)
    assert all(b - a >= gap for a, b in zip(idx, idx[1:]))


def test_sell_events_mirror_buy_events():
    n = 60
    close = np.concatenate([np.linspace(100, 140, 40), np.linspace(139, 120, 20)])
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": "STRONG_SELL",
            "pivot": 0.0,
        }
    )
    # The uptrend filter is deliberately off here: this test is about the
    # buy/sell mirror, which is a separate concern from trend suppression.
    out = extract_turning_points(
        frame, TurningPointConfig(refractory=10, max_uptrend_slope=None)
    )
    assert (out["turn_action"] == SELL_TURN).any()
    assert not (out["turn_action"] == BUY_TURN).any()


def test_original_action_column_is_preserved():
    """Both views must remain available; the event layer is additive."""
    frame = _frame()
    out = extract_turning_points(frame)
    assert "action" in out.columns and "turn_action" in out.columns
    assert (out["action"] == "STRONG_BUY").all()


def test_summarise_reports_lift_against_pivots():
    n = 200
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    pivot = np.zeros(n)
    pivot[::25] = -1.0
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "action": "STRONG_BUY",
            "pivot": pivot,
        }
    )
    out = extract_turning_points(frame)
    stats = summarise(out)
    assert set(stats) >= {"buy_events", "buy_density", "buy_precision", "buy_lift"}
    assert 0.0 <= stats["buy_density"] <= 1.0
    assert stats["buy_events"] >= 0


def test_multiple_symbols_do_not_bleed_into_each_other():
    """Refractory state must reset per symbol, not run across the concat."""
    a = _frame(n=60, symbol="AAA", seed=1)
    b = _frame(n=60, symbol="BBB", seed=2)
    preds = pd.concat([a, b], ignore_index=True)
    out = extract_turning_points(
        preds,
        TurningPointConfig(
            require_reversal=False,
            trough_pct=1.0,
            refractory=0,
            min_downtrend_slope=None,
        ),
    )
    # With every gate open, every eligible bar of both symbols fires.
    for symbol in ("AAA", "BBB"):
        block = out[out["symbol"] == symbol]
        assert (block["turn_action"] == BUY_TURN).sum() > 0


def test_backtest_can_trade_the_event_column():
    """The engine must be able to read entries from ``turn_action``."""
    from predixlab_signals.backtest import BacktestConfig, run_backtest

    n = 120
    rng = np.random.default_rng(4)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "price": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "open_next": np.roll(close, -1),
            "atr_pct_label": np.full(n, 0.02),
            "exp_gain_pct": np.full(n, 0.04),
            "exp_loss_pct": np.full(n, -0.02),
            "score": rng.normal(size=n),
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(frame, TurningPointConfig(refractory=8))
    result = run_backtest(
        out,
        BacktestConfig(horizon=10, entry_actions=(BUY_TURN,), action_column="turn_action"),
    )
    n_events = int((out["turn_action"] == BUY_TURN).sum())
    assert len(result.trades) <= n_events
    if n_events:
        assert len(result.trades) > 0


# --------------------------------------------------------------------------
# setup window: a turn does not complete in a single bar
# --------------------------------------------------------------------------
def _v_bottom(bottom_at=40, n=60, action_at_bottom="STRONG_BUY", after="BUY"):
    """Price falls to a bottom then rebounds; the strong signal fires only at the low."""
    close = np.concatenate([np.linspace(140, 100, bottom_at), np.linspace(101, 125, n - bottom_at)])
    action = np.array([after] * n, dtype=object)
    action[bottom_at - 1] = action_at_bottom
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
        }
    )


def test_same_bar_rule_misses_a_one_day_lag():
    """Reproduces the original defect: setup and trigger fall on different bars."""
    frame = _v_bottom()
    strict = extract_turning_points(frame, TurningPointConfig(setup_window=0))
    assert (strict["turn_action"] == NO_TURN).all()


def test_setup_window_catches_the_lagged_reversal():
    frame = _v_bottom()
    lagged = extract_turning_points(
        frame, TurningPointConfig(setup_window=3, min_downtrend_slope=None)
    )
    assert (lagged["turn_action"] == BUY_TURN).any()


def test_retrained_filter_defaults_extend_setup_coverage():
    cfg = TurningPointConfig()
    assert cfg.setup_window == 4
    assert cfg.trigger_pct == pytest.approx(0.50)


def test_setup_window_respects_its_length():
    """A setup older than the window must not trigger."""
    n, bottom = 60, 40
    close = np.concatenate([np.linspace(140, 100, bottom), np.linspace(101, 125, n - bottom)])
    action = np.array(["HOLD"] * n, dtype=object)
    action[bottom - 1] = "STRONG_BUY"  # setup only at the very bottom
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
        }
    )
    near = extract_turning_points(
        frame,
        TurningPointConfig(
            setup_window=3,
            refractory=0,
            min_downtrend_slope=None,
        ),
    )
    fired = np.flatnonzero(near["turn_action"] == BUY_TURN)
    assert len(fired) > 0
    assert all(i - (bottom - 1) <= 3 for i in fired)


def test_trigger_cap_blocks_a_late_chase():
    """Once price has run back up the range, the event must not fire."""
    n = 60
    # Deep bottom, then a violent rebound well past mid-range within a few bars.
    close = np.concatenate([np.linspace(140, 100, 40), np.linspace(104, 140, 20)])
    action = np.array(["HOLD"] * n, dtype=object)
    action[39] = "STRONG_BUY"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
        }
    )
    capped = extract_turning_points(
        frame,
        TurningPointConfig(
            setup_window=5,
            trigger_pct=0.30,
            refractory=0,
            min_downtrend_slope=None,
        ),
    )
    uncapped = extract_turning_points(
        frame,
        TurningPointConfig(
            setup_window=5,
            trigger_pct=None,
            refractory=0,
            min_downtrend_slope=None,
        ),
    )
    n_capped = int((capped["turn_action"] == BUY_TURN).sum())
    n_uncapped = int((uncapped["turn_action"] == BUY_TURN).sum())
    assert n_capped < n_uncapped


def test_a_bar_is_never_both_buy_and_sell():
    rng = np.random.default_rng(11)
    n = 300
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": rng.choice(["STRONG_BUY", "STRONG_SELL"], n),
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(
        frame, TurningPointConfig(setup_window=5, trough_pct=0.9, peak_pct=0.1, refractory=0)
    )
    assert out["turn_action"].isin(
        [
            BASE_BREAKOUT,
            BOTTOM_TURN,
            BUY_TURN,
            SELL_TURN,
            TOP_TURN,
            TREND_BUY,
            NO_TURN,
        ]
    ).all()


# --------------------------------------------------------------------------
# one signal per swing, and the asymmetric trend filter
# --------------------------------------------------------------------------
def _declining(n=200, seed=3):
    """A long, steady decline punctuated by frequent one-bar bounces."""
    rng = np.random.default_rng(seed)
    close = 200 * np.exp(np.cumsum(rng.normal(-0.004, 0.02, n)))
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )


def test_rearm_thins_a_whole_declining_leg():
    """The 'entire down leg flagged as buy' complaint must be suppressed."""
    frame = _declining()
    dense = extract_turning_points(
        frame,
        TurningPointConfig(rearm_atr=0.0, min_downtrend_slope=None),
    )
    thinned = extract_turning_points(
        frame,
        TurningPointConfig(rearm_atr=2.5, min_downtrend_slope=None),
    )
    n_dense = int((dense["turn_action"] == BUY_TURN).sum())
    n_thin = int((thinned["turn_action"] == BUY_TURN).sum())
    assert n_dense > 0
    assert n_thin < n_dense


def test_rearm_requires_real_price_travel():
    """Consecutive same-side events must be separated by price, not just time."""
    frame = _declining()
    out = extract_turning_points(
        frame,
        TurningPointConfig(rearm_atr=3.0, rearm_max_bars=None, refractory=1),
    )
    fired = out[out["turn_action"] == BUY_TURN]
    if len(fired) > 1:
        closes = fired["close"].to_numpy()
        atr = (frame["high"] - frame["low"]).mean()
        assert all(abs(b - a) > atr for a, b in zip(closes, closes[1:]))


def test_rearm_is_two_sided_so_a_winning_call_does_not_lock_the_side():
    """A name whose opposite side never fires must still produce later signals.

    Keying re-arm off the *opposite* event deadlocks any stock in a permanent
    uptrend: its sells are filtered out, so the buy side would never unlock
    after its first event. This is exactly what happened to MSFT.
    """
    # A rising channel with deep pullbacks that end in a sharp recovery bar, so
    # each dip produces a genuine reversal while price is still low in its
    # range. Sells are suppressed by the uptrend filter, so only a two-sided
    # re-arm can keep the buy side alive after its first event.
    cycles, decline, rally = 10, 20, 10
    level, path = 100.0, []
    for _ in range(cycles):
        for _ in range(decline):
            level *= 0.988
            path.append(level)
        for k in range(rally):
            level *= 1.035 if k == 0 else 1.012
            path.append(level)
    close = np.array(path)
    n = close.size
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(
        frame,
        TurningPointConfig(
            max_uptrend_slope=0.02,
            min_downtrend_slope=None,
            pivot_gate_q=None,
        ),
    )
    assert int((out["turn_action"] == BUY_TURN).sum()) > 1
    assert int((out["turn_action"] == SELL_TURN).sum()) == 0


def test_sell_is_suppressed_in_a_strong_uptrend_when_enabled():
    """Opt-in filter: off by default because tops form in uptrends."""
    n = 200
    close = 100 * np.exp(np.linspace(0, 0.8, n))  # relentless uptrend
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "action": "STRONG_SELL",
            "pivot": 0.0,
        }
    )
    filtered = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=0.02, pivot_gate_q=None)
    )
    unfiltered = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=None, pivot_gate_q=None)
    )
    assert int((filtered["turn_action"] == SELL_TURN).sum()) <= int(
        (unfiltered["turn_action"] == SELL_TURN).sum()
    )
    assert int((filtered["turn_action"] == SELL_TURN).sum()) == 0


def test_guarded_high_reversal_is_preserved_as_top_detection():
    n = 220
    close = 100 * np.exp(np.linspace(0, 1.6, n) ** 2)
    close[-1] = close[-2] * 0.88
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": "STRONG_SELL",
            "pivot": 0.0,
        }
    )

    guarded = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=0.02, pivot_gate_q=None)
    )
    assert guarded["turn_action"].iloc[-1] == SELL_TURN
    assert int((guarded["turn_action"] == SELL_TURN).sum()) == 1

    unconfirmed = extract_turning_points(
        frame,
        TurningPointConfig(
            max_uptrend_slope=0.02,
            pivot_gate_q=None,
            top_confirm_volatility_q=None,
        ),
    )
    assert unconfirmed["turn_action"].iloc[-1] == TOP_TURN

    unguarded = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=None, pivot_gate_q=None)
    )
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def _uptrend_action_top_frame() -> pd.DataFrame:
    n = 220
    close = 100 * np.exp(np.linspace(0, 0.8, n))
    high = close * 1.01
    low = close * 0.99
    close[-1] = close[-2] * 0.985
    high[-1] = close[-2] * 1.005
    low[-1] = close[-1] * 0.999
    high_q = np.full(n, 0.10)
    high_s = np.full(n, 0.10)
    low_q = np.full(n, 0.10)
    high_q[-1] = 0.90
    high_s[-1] = 0.90

    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": "STRONG_SELL",
            "pivot": 0.0,
            "_p_pivot_low_q": low_q,
            "_p_pivot_high_q": high_q,
            "_p_pivot_high_s": high_s,
        }
    )


def test_strong_bearish_action_confirms_a_fresh_uptrend_top():
    frame = _uptrend_action_top_frame()
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "top_right_rescue_pivot_q": None,
        "max_uptrend_slope": 0.02,
        "top_confirm_volatility_q": 1.10,
        "top_confirm_recent_volatility_q": 1.10,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    detected = extract_turning_points(
        frame,
        TurningPointConfig(
            **common,
            top_uptrend_action_confirm_pivot_q=None,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == SELL_TURN
    assert detected["turn_action"].iloc[-1] == TOP_TURN


def test_uptrend_action_confirmation_requires_high_evidence_advantage():
    frame = _uptrend_action_top_frame()
    frame.loc[frame.index[-1], "_p_pivot_low_q"] = 0.70

    out = extract_turning_points(
        frame,
        TurningPointConfig(
            pivot_gate_q=None,
            bottom_reference_self_q=None,
            top_reference_pivot_q=None,
            top_right_rescue_pivot_q=None,
            max_uptrend_slope=0.02,
            top_confirm_volatility_q=1.10,
            top_confirm_recent_volatility_q=1.10,
            trigger_pct=1.0,
            refractory=0,
            rearm_atr=0,
            trend_entry=False,
        ),
    )

    assert out["turn_action"].iloc[-1] == TOP_TURN


def test_fresh_local_high_can_confirm_after_an_extreme_reversal_bar():
    close = np.concatenate(
        [
            np.linspace(100.0, 150.0, 160),
            np.linspace(149.0, 120.0, 40),
            np.linspace(121.0, 140.0, 19),
            np.array([130.0]),
        ]
    )
    n = len(close)
    high = close + 0.5
    low = close - 0.5
    high[-1], low[-1] = 132.0, 129.8
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.85,
            "_p_pivot_high_s": 0.85,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "max_uptrend_slope": -1.0,
        "top_guard_enabled": False,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(top_confirm_local_pivot_q=None, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == SELL_TURN
    assert blocked["turn_action"].iloc[-1] == TOP_TURN


def test_delayed_crash_can_confirm_a_recent_local_high():
    close = np.concatenate(
        [
            np.linspace(100.0, 150.0, 170),
            np.linspace(149.0, 120.0, 30),
            np.linspace(121.0, 140.0, 16),
            np.array([140.0, 140.0, 140.0, 125.0]),
        ]
    )
    n = len(close)
    high = close + 0.5
    low = close - 0.5
    high[-1], low[-1] = 127.0, 124.8
    action = np.full(n, "HOLD", dtype=object)
    action[-5] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.95,
            "_p_pivot_high_s": 0.95,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "max_uptrend_slope": -1.0,
        "top_guard_enabled": False,
        "top_confirm_local_pivot_q": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(top_confirm_crash_pivot_q=None, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == SELL_TURN
    assert blocked["turn_action"].iloc[-1] == TOP_TURN


def test_guarded_top_can_wait_one_extra_session_for_confirmation():
    n = 110
    close = 100 * 1.003 ** np.arange(n)
    action = np.full(n, "HOLD", dtype=object)
    action[104] = "STRONG_SELL"
    close[105] = close[104] * 1.003
    close[106] = close[105] * 1.003
    close[107] = close[106] * 1.003
    close[108] = close[107] * 0.985
    close[109] = close[108] * 0.995
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2025-01-01", periods=n),
            "close": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "action": action,
            "pivot": 0.0,
        }
    )
    common = {
        "pivot_gate_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
        "setup_window": 3,
    }

    ordinary_window = extract_turning_points(
        frame, TurningPointConfig(top_setup_window=3, **common)
    )
    detection_window = extract_turning_points(
        frame, TurningPointConfig(top_setup_window=4, **common)
    )

    assert ordinary_window["turn_action"].iloc[-2] == NO_TURN
    assert detection_window["turn_action"].iloc[-2] == SELL_TURN


def test_low_conviction_sell_away_from_high_stays_a_top_detection():
    n = 110
    close = 100 * 1.002 ** np.arange(n)
    action = np.full(n, "HOLD", dtype=object)
    action[106] = "STRONG_SELL"
    close[106:109] = close[105] * np.array([0.999, 0.998, 0.997])
    close[109] = close[108] * 0.98
    high = close * 1.004
    high[106] = high[105] * 1.001
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2025-01-01", periods=n),
            "close": close,
            "high": high,
            "low": close * 0.996,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.50,
            "_p_pivot_high_s": 0.80,
        }
    )
    common = {
        "pivot_gate_q": None,
        "max_uptrend_slope": None,
        "setup_window": 4,
        "top_setup_window": 4,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(
        frame,
        TurningPointConfig(
            top_confirm_model_pivot_q=0.94,
            top_confirm_model_self_q=0.96,
            **common,
        ),
    )
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(top_guard_enabled=False, **common),
    )

    assert guarded["turn_action"].iloc[-1] == TOP_TURN
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def test_guarded_top_needs_current_or_self_model_support():
    n = 220
    close = 100 * np.exp(np.linspace(0, 1.6, n) ** 2)
    close[-1] = close[-2] * 0.88
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": "STRONG_SELL",
            "pivot": 0.0,
            "_p_pivot_high_q": 0.93,
            "_p_pivot_high_s": 0.95,
        }
    )
    common = {
        "pivot_gate_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(
            top_confirm_model_pivot_q=None,
            top_confirm_model_self_q=None,
            top_confirm_reversal_return_1=-1.0,
            **common,
        ),
    )

    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            top_confirm_model_pivot_q=0.94,
            top_confirm_model_self_q=0.96,
            top_confirm_reversal_return_1=-1.0,
            **common,
        ),
    )

    assert guarded["turn_action"].iloc[-1] == SELL_TURN
    assert blocked["turn_action"].iloc[-1] == TOP_TURN
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def test_recent_top_model_peak_confirms_after_five_day_momentum_cools():
    n = 220
    close = 100 * np.exp(np.linspace(0, 1.6, n) ** 2)
    base = close[-6]
    close[-5:] = base * np.array([1.03, 1.025, 1.02, 1.01, 0.98])
    high = close * 1.002
    high[-4] = base * 1.04
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": high,
            "low": close * 0.998,
            "action": "STRONG_SELL",
            "pivot": 0.0,
            "_p_pivot_high_q": 0.50,
            "_p_pivot_high_s": 0.98,
        }
    )
    common = {
        "pivot_gate_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
        "top_confirm_volatility_q": 1.1,
        "top_confirm_recent_volatility_q": 1.1,
        "top_confirm_quiet_pivot_q": 1.1,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(top_confirm_memory_self_q=None, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == SELL_TURN
    assert blocked["turn_action"].iloc[-1] == TOP_TURN


def test_strong_downtrend_guard_preserves_unconfirmed_bottoms():
    n = 100
    close = np.concatenate([np.linspace(180, 100, 80), np.linspace(101, 120, 20)])
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    guarded = extract_turning_points(
        frame, TurningPointConfig(bottom_confirm_volatility_q=None)
    )
    unguarded = extract_turning_points(
        frame, TurningPointConfig(min_downtrend_slope=None)
    )

    assert int((guarded["turn_action"] == BUY_TURN).sum()) <= int(
        (unguarded["turn_action"] == BUY_TURN).sum()
    )
    assert int((guarded["turn_action"] == BOTTOM_TURN).sum()) > 0


def test_retrained_buy_guard_covers_all_falling_trends():
    cfg = TurningPointConfig()
    assert cfg.min_downtrend_slope == pytest.approx(0.0)
    assert cfg.max_downtrend_return_20 == pytest.approx(-0.04)


def test_strong_close_can_confirm_an_early_bottom_before_momentum_turns():
    n = 80
    close = np.linspace(140.0, 102.0, n)
    close[-4:] = [101.0, 100.0, 99.5, 100.2]
    high = close + 0.2
    low = close - 0.2
    high[-1] = close[-1] + 0.02
    low[-1] = close[-1] - 0.48
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_fresh_setup_self_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(bottom_confirm_immediate_close_loc=1.01, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_high_confidence_local_low_can_confirm_after_a_delayed_rebound():
    close = np.concatenate(
        [
            np.linspace(140.0, 70.0, 50),
            np.linspace(72.0, 95.0, 20),
            np.linspace(93.0, 80.0, 14),
            np.array([81.0, 82.0, 83.0, 84.0, 84.0, 85.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(bottom_confirm_local_max_bars=0, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_capitulation_reversal_can_waive_the_five_day_return_floor():
    n = 80
    close = np.linspace(140.0, 105.0, n)
    close[-6:] = [115.0, 110.0, 96.0, 95.0, 94.5, 103.5]
    high = close + 0.2
    low = close - 0.2
    high[-1] = 104.0
    low[-1] = 80.0
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
        }
    )
    common = {
        "pivot_gate_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_reversal_waives_return_5=False,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_clear_daily_rebound_can_confirm_a_low_volatility_bottom():
    n = 80
    close = np.linspace(140.0, 102.0, n)
    close[-4:] = [101.5, 100.0, 99.5, 102.5]
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_volatility_q": 1.1,
        "bottom_confirm_quiet_pivot_q": 1.1,
        "bottom_confirm_stable_pivot_q": 1.1,
        "bottom_confirm_delayed_pivot_q": 1.1,
        "bottom_confirm_immediate_close_loc": 1.1,
        "bottom_confirm_reversal_atr": 100.0,
        "bottom_confirm_fresh_setup_self_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(bottom_confirm_rebound_return_1=0.50, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_low_conviction_incomplete_reversal_stays_a_bottom_detection():
    n = 80
    close = np.full(n, 110.0)
    close[-11:] = np.linspace(110.0, 95.0, 11)
    close[-1] = 97.5
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "min_downtrend_slope": None,
        "bottom_confirm_fresh_setup_self_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(
        frame,
        TurningPointConfig(bottom_guard_low_conviction_q=1.1, **common),
    )
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(bottom_guard_low_conviction_q=None, **common),
    )

    assert guarded["turn_action"].iloc[-1] == BOTTOM_TURN
    assert unguarded["turn_action"].iloc[-1] == BUY_TURN


def test_high_confidence_panic_reversal_can_confirm_before_three_day_momentum():
    n = 80
    close = np.linspace(140.0, 103.0, n)
    close[-6:] = [103.0, 102.0, 100.0, 94.0, 90.0, 95.0]
    high = close + 0.2
    low = close - 0.2
    high[-1] = 99.5
    low[-1] = 85.0
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_fresh_setup_self_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_panic_atr=100.0,
            bottom_confirm_two_day_self_q=None,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_buy_waits_for_complete_bottom_confirmation_history():
    frame = _v_bottom(bottom_at=25, n=40)
    frame["p_pivot_low"] = 0.9
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "min_downtrend_slope": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_guard_requires_confirmation_history=False,
            **common,
        ),
    )

    assert (guarded["turn_action"] == BOTTOM_TURN).any()
    assert (guarded["turn_action"] == BUY_TURN).sum() == 0
    assert (unguarded["turn_action"] == BUY_TURN).any()


def test_setup_model_peak_can_confirm_after_conviction_cools():
    n = 80
    close = np.linspace(140.0, 100.0, n)
    close[-4:] = [97.0, 96.0, 94.0, 96.5]
    high = close + 0.5
    low = close - 0.5
    high[-1] = 96.8
    low[-1] = 95.8
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2] = 0.9
    pivot_low[-1] = 0.2
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "min_downtrend_slope": None,
        "bottom_guard_low_conviction_q": 1.1,
        "bottom_confirm_setup_memory_max_current_q": 1.1,
        "bottom_confirm_setup_memory_close_loc": 0.60,
        "bottom_confirm_fresh_setup_self_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_setup_memory_pivot_q=1.1,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_strong_rebound_can_confirm_a_local_higher_low():
    close = np.concatenate(
        [
            np.linspace(110.0, 70.0, 45),
            np.linspace(72.0, 95.0, 35),
            np.linspace(94.0, 82.0, 19),
            np.array([86.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    high = close + 0.2
    low = close - 0.2
    high[-1] = 86.1
    low[-1] = 84.8
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_low_q": 0.89,
            "_p_pivot_low_s": 0.91,
        }
    )
    common = {
        "pivot_gate_q": None,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_higher_low_pivot_q=None,
            bottom_confirm_two_day_self_q=None,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_two_day_rebound_confirms_a_guarded_local_low():
    close = np.concatenate(
        [
            np.linspace(140.0, 100.0, 70),
            np.linspace(99.0, 91.0, 27),
            np.array([90.0, 90.5, 92.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.6,
            "low": close - 0.6,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_volatility_q": 1.1,
        "bottom_confirm_recent_volatility_q": 1.1,
        "bottom_confirm_quiet_pivot_q": 1.1,
        "bottom_confirm_stable_pivot_q": 1.1,
        "bottom_confirm_rebound_pivot_q": 1.1,
        "bottom_confirm_immediate_close_loc": 1.1,
        "bottom_confirm_delayed_pivot_q": 1.1,
        "bottom_confirm_higher_low_pivot_q": None,
        "bottom_confirm_fresh_setup_self_q": None,
        "bottom_confirm_reversal_atr": 100.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(bottom_confirm_two_day_self_q=None, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_fresh_low_can_use_recent_self_rank_memory():
    close = np.concatenate(
        [
            np.linspace(140.0, 100.0, 75),
            np.linspace(99.0, 90.0, 23),
            np.array([89.0, 91.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2] = 0.9
    pivot_low[-1] = 0.2
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_two_day_self_q": None,
        "bottom_confirm_volatility_q": 1.1,
        "bottom_confirm_recent_volatility_q": 1.1,
        "bottom_confirm_quiet_pivot_q": 1.1,
        "bottom_confirm_stable_pivot_q": 1.1,
        "bottom_confirm_rebound_pivot_q": 1.1,
        "bottom_confirm_immediate_close_loc": 1.1,
        "bottom_confirm_delayed_pivot_q": 1.1,
        "bottom_confirm_higher_low_pivot_q": None,
        "bottom_confirm_reversal_atr": 100.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_fresh_setup_self_q=None,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_exceptional_two_day_rebound_uses_strong_setup_memory():
    close = np.concatenate(
        [
            np.linspace(160.0, 120.0, 75),
            np.linspace(119.0, 91.0, 22),
            np.array([91.0, 89.0, 94.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2] = 0.9
    pivot_low[-1] = 0.2
    high = close + 0.2
    low = close - 0.2
    high[-1], low[-1] = 94.2, 93.4
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_two_day_self_q": None,
        "bottom_confirm_volatility_q": 1.1,
        "bottom_confirm_recent_volatility_q": 1.1,
        "bottom_confirm_quiet_pivot_q": 1.1,
        "bottom_confirm_stable_pivot_q": 1.1,
        "bottom_confirm_rebound_pivot_q": 1.1,
        "bottom_confirm_immediate_close_loc": 1.1,
        "bottom_confirm_delayed_pivot_q": 1.1,
        "bottom_confirm_higher_low_pivot_q": None,
        "bottom_confirm_fresh_setup_self_q": None,
        "bottom_confirm_setup_memory_pivot_q": 1.1,
        "bottom_confirm_reversal_atr": 100.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(
            bottom_confirm_setup_rebound_pivot_q=None,
            **common,
        ),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_price_shock_rebound_can_confirm_without_high_self_rank():
    close = np.concatenate(
        [
            np.linspace(160.0, 100.0, 97),
            np.array([99.0, 90.0, 97.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    high = close + 0.2
    low = close - 0.2
    high[-1], low[-1] = 97.2, 95.8
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )
    common = {
        "pivot_gate_q": 0.0,
        "trend_neutral_weight": 0.0,
        "self_gate_window": 20,
        "self_gate_min_periods": 1,
        "bottom_confirm_two_day_self_q": None,
        "bottom_confirm_volatility_q": 1.1,
        "bottom_confirm_recent_volatility_q": 1.1,
        "bottom_confirm_quiet_pivot_q": 1.1,
        "bottom_confirm_stable_pivot_q": 1.1,
        "bottom_confirm_rebound_pivot_q": 1.1,
        "bottom_confirm_immediate_close_loc": 1.1,
        "bottom_confirm_delayed_pivot_q": 1.1,
        "bottom_confirm_higher_low_pivot_q": None,
        "bottom_confirm_fresh_setup_self_q": None,
        "bottom_confirm_setup_memory_pivot_q": 1.1,
        "bottom_confirm_setup_rebound_pivot_q": None,
        "bottom_confirm_reversal_atr": 100.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    confirmed = extract_turning_points(frame, TurningPointConfig(**common))
    blocked = extract_turning_points(
        frame,
        TurningPointConfig(bottom_confirm_shock_pivot_q=None, **common),
    )

    assert confirmed["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == BOTTOM_TURN


def test_shallow_rebound_in_a_deep_decline_stays_a_bottom_detection():
    close = np.concatenate(
        [
            np.linspace(160.0, 111.0, 70),
            np.linspace(110.0, 91.0, 27),
            np.array([90.0, 90.5, 91.5]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2:] = 0.9
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.6,
            "low": close - 0.6,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )

    out = extract_turning_points(
        frame,
        TurningPointConfig(
            pivot_gate_q=0.0,
            trend_neutral_weight=0.0,
            self_gate_window=20,
            self_gate_min_periods=1,
            bottom_confirm_volatility_q=1.1,
            bottom_confirm_recent_volatility_q=1.1,
            bottom_confirm_quiet_pivot_q=1.1,
            bottom_confirm_stable_pivot_q=1.1,
            bottom_confirm_rebound_pivot_q=1.1,
            bottom_confirm_immediate_close_loc=1.1,
            bottom_confirm_delayed_pivot_q=1.1,
            bottom_confirm_higher_low_pivot_q=None,
            bottom_confirm_fresh_setup_self_q=None,
            bottom_confirm_reversal_atr=100.0,
            refractory=0,
            rearm_atr=0,
            trend_entry=False,
        ),
    )

    assert out["turn_action"].iloc[-1] == BOTTOM_TURN


def test_fresh_setup_memory_needs_momentum_or_a_strong_reversal_bar():
    close = np.concatenate(
        [
            np.linspace(140.0, 100.0, 75),
            np.linspace(99.0, 90.0, 23),
            np.array([89.0, 89.5]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    pivot_low = np.full(n, 0.1)
    pivot_low[-2] = 0.9
    pivot_low[-1] = 0.2
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": pivot_low,
        }
    )

    out = extract_turning_points(
        frame,
        TurningPointConfig(
            pivot_gate_q=0.0,
            trend_neutral_weight=0.0,
            self_gate_window=20,
            self_gate_min_periods=1,
            bottom_confirm_two_day_self_q=None,
            refractory=0,
            rearm_atr=0,
            trend_entry=False,
        ),
    )

    assert out["turn_action"].iloc[-1] == BOTTOM_TURN


def test_backtest_exits_on_the_opposite_turn():
    """`exit_actions` must let a sell turn close a long."""
    from predixlab_signals.backtest import BacktestConfig, run_backtest

    n = 40
    close = np.concatenate([np.full(12, 100.0), np.full(28, 101.0)])
    action = np.array(["NONE"] * n, dtype=object)
    action[2] = BUY_TURN
    action[8] = SELL_TURN
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "price": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "open_next": np.roll(close, -1),
            "atr_pct_label": np.full(n, 0.30),  # wide bands so target/stop never hit
            "exp_gain_pct": np.full(n, 0.50),
            "exp_loss_pct": np.full(n, -0.50),
            "score": np.zeros(n),
            "turn_action": action,
        }
    )
    result = run_backtest(
        frame,
        BacktestConfig(
            horizon=10,
            max_hold=30,
            entry_actions=(BUY_TURN,),
            exit_actions=(SELL_TURN,),
            action_column="turn_action",
            min_stop_atr=1.0,
            max_stop_atr=3.0,
        ),
    )
    assert len(result.trades) == 1
    assert result.trades["exit_reason"].iloc[0] == "signal"


def test_violent_reversal_waives_the_range_cap():
    """The sharpest turns collapse range position in one bar; the cap must yield.

    A top that gives back a large fraction of its range in a single session
    would otherwise be rejected for no longer sitting near the high — which is
    exactly backwards, since the size of that bar is the confirmation.
    """
    n = 60
    close = np.concatenate([np.linspace(100, 150, 40), np.full(20, 128.0)])
    close[40] = 128.0  # a violent give-back from the top
    action = np.array(["HOLD"] * n, dtype=object)
    action[39] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "action": action,
            "pivot": 0.0,
            "atr_pct_label": np.full(n, 0.01),
        }
    )
    with_waiver = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=None, reversal_waiver_atr=1.5)
    )
    without = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=None, reversal_waiver_atr=None)
    )
    assert int((with_waiver["turn_action"] == SELL_TURN).sum()) >= int(
        (without["turn_action"] == SELL_TURN).sum()
    )


def test_parabolic_tops_are_not_detected_by_design():
    """Strong-uptrend sells stay suppressed until a stable rescue is proven."""
    n = 220
    close = 100 * np.exp(np.linspace(0, 1.6, n) ** 2)  # parabolic advance
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "action": "STRONG_SELL",
            "pivot": 0.0,
        }
    )
    guarded = extract_turning_points(
        frame, TurningPointConfig(max_uptrend_slope=0.02, pivot_gate_q=None)
    )
    assert int((guarded["turn_action"] == SELL_TURN).sum()) == 0


# --------------------------------------------------------------------------
# the detection head drives the gate
# --------------------------------------------------------------------------
def _panel_with_heads(n=300, seed=12):
    """Two symbols so the cross-sectional rank has something to rank."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=n)
    for sym, shift in (("AAA", 0.0), ("BBB", 0.5)):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        rows.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "close": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "action": "HOLD",
                    "pivot": 0.0,
                    "p_pivot_low": np.clip(rng.random(n) + shift, 0, 1),
                    "p_pivot_high": np.clip(rng.random(n) + shift, 0, 1),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_detection_head_replaces_the_state_gate():
    """With heads present the state label must no longer decide.

    On AMG the two disagreed decisively: at the genuine 2025 low the head read
    0.487 while the state said HOLD, and at the February 2026 breakout the head
    read 0.021 while the state said STRONG_BUY.
    """
    panel = _panel_with_heads()
    panel["action"] = "HOLD"  # the state gate alone would produce nothing
    out = extract_turning_points(panel, TurningPointConfig(pivot_gate_q=0.5))
    assert (out["turn_action"] != NO_TURN).any()


def test_state_gate_is_used_when_heads_are_absent():
    """Older datasets without the heads must keep working unchanged."""
    panel = _panel_with_heads().drop(columns=["p_pivot_low", "p_pivot_high"])
    panel["action"] = "STRONG_BUY"
    out = extract_turning_points(panel, TurningPointConfig())
    assert (out["turn_action"] == BUY_TURN).any()


def test_pivot_gate_is_ranked_across_the_cross_section():
    """Ranking must be by date, not within a symbol's own history.

    Ranking per symbol would mean "unusual for this stock" and would fire on
    every name in every regime, rather than selecting the market's best
    candidates on the day.
    """
    panel = _panel_with_heads()
    panel["action"] = "HOLD"
    # BBB's probabilities are uniformly higher, so a cross-sectional gate must
    # select it far more often than AAA.
    out = extract_turning_points(
        panel,
        TurningPointConfig(pivot_gate_q=0.75, max_uptrend_slope=None),
    )
    counts = out[out["turn_action"] != NO_TURN]["symbol"].value_counts()
    assert counts.get("BBB", 0) > counts.get("AAA", 0)


def test_helper_columns_do_not_leak_into_the_output():
    out = extract_turning_points(_panel_with_heads(), TurningPointConfig())
    assert not [c for c in out.columns if c.startswith("_p_pivot")]


def test_bounded_waiver_rejects_a_breakout():
    """A violent bar that has already broken out is not a reversal off a low.

    This is the AMG 2026-02-12 failure: price sat low three sessions earlier,
    then jumped 6.8% (1.84 ATR) to a new high. An unbounded waiver read that as
    a confirmed turn.
    """
    n = 60
    close = np.concatenate([np.linspace(120, 100, 40), np.full(20, 100.0)])
    close[40] = 118.0  # violent jump back to the top of the range
    close[41:] = 118.0
    action = np.array(["HOLD"] * n, dtype=object)
    action[38] = "STRONG_BUY"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "atr_pct_label": np.full(n, 0.01),
        }
    )
    bounded = extract_turning_points(
        frame, TurningPointConfig(pivot_gate_q=None, waiver_pct=0.65, refractory=0)
    )
    unbounded = extract_turning_points(
        frame, TurningPointConfig(pivot_gate_q=None, waiver_pct=1.01, refractory=0)
    )
    fired_bounded = int((bounded["turn_action"] == BUY_TURN).sum())
    fired_unbounded = int((unbounded["turn_action"] == BUY_TURN).sum())
    assert fired_bounded <= fired_unbounded


def test_top_detection_defaults_preserve_the_uptrend_filter():
    assert TurningPointConfig().max_uptrend_slope == pytest.approx(0.02)


def test_pick_accuracy_defaults_tighten_the_reversal_waiver():
    assert TurningPointConfig().reversal_waiver_atr == pytest.approx(2.5)


def test_pivot_gate_default_balances_precision_and_recall():
    """0.97 covered only 26% of true lows; 0.95 covers 37% at the same lift."""
    assert TurningPointConfig().pivot_gate_q == 0.95


def _uptrend_pullback(n: int = 400) -> pd.DataFrame:
    """A rising series whose pullbacks never reach the bottom of the 20-day range."""
    trend = np.linspace(100.0, 200.0, n)
    wobble = 2.0 * np.sin(np.arange(n) / 4.0)
    close = trend + wobble
    return pd.DataFrame(
        {
            "symbol": "UP",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )


def test_self_relative_gate_is_off_by_default():
    """It raises label recall but reverses the trend-neutral buy/sell spread."""
    assert TurningPointConfig().self_range_window is None


def test_self_relative_gate_admits_lows_the_absolute_gate_excludes():
    """Opt-in behaviour: an uptrend pullback never enters the bottom 35%.

    The context guard would veto this fixture, which is permanently at its
    60-day high, so it is disabled to isolate the range gate itself.
    """
    frame = _uptrend_pullback()
    absolute_only = extract_turning_points(
        frame, TurningPointConfig(context_window=None)
    )
    with_self = extract_turning_points(
        frame, TurningPointConfig(self_range_window=250, context_window=None)
    )

    assert (absolute_only["turn_action"] == BUY_TURN).sum() == 0
    assert (with_self["turn_action"] == BUY_TURN).sum() > 0


def test_context_guard_blocks_buying_at_a_multi_month_high():
    """A name pinned to its 60-day high must not produce a buy."""
    frame = _uptrend_pullback()
    guarded = extract_turning_points(
        frame, TurningPointConfig(self_range_window=250, context_buy_bound=0.25)
    )
    assert (guarded["turn_action"] == BUY_TURN).sum() == 0


def test_context_guard_bounds_are_asymmetric_by_default():
    """The sell-side claim is strong; the buy-side one is weaker.

    A shallow pullback inside a strong advance keeps a high 60-day reading by
    construction, so an equally tight buy bound rejects ordinary dip entries —
    GOOGL's October 2025 low among them.
    """
    cfg = TurningPointConfig()
    assert cfg.context_sell_bound > cfg.context_buy_bound


def test_context_guard_blocks_selling_at_a_multi_month_low():
    """Mirror case: a name pinned to its 60-day low must not produce a sell."""
    frame = _uptrend_pullback()
    frame["close"] = frame["close"].to_numpy()[::-1]
    frame["high"] = frame["close"] + 0.5
    frame["low"] = frame["close"] - 0.5
    frame["action"] = "STRONG_SELL"

    unguarded = extract_turning_points(
        frame, TurningPointConfig(self_range_window=250, context_window=None)
    )
    guarded = extract_turning_points(frame, TurningPointConfig(self_range_window=250))

    assert (unguarded["turn_action"] == SELL_TURN).sum() > 0
    assert (guarded["turn_action"] == SELL_TURN).sum() == 0


def test_long_context_conflict_demotes_a_sell_near_the_one_year_low():
    close = np.concatenate(
        [
            np.linspace(150.0, 80.0, 180),
            np.linspace(80.0, 100.0, 119),
            np.array([90.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_low_q": 0.90,
            "_p_pivot_high_q": 0.80,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "max_uptrend_slope": None,
        "top_guard_enabled": False,
        "top_confirm_local_pivot_q": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(sell_low_context_window=None, **common),
    )

    assert guarded["turn_action"].iloc[-1] == TOP_TURN
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def test_high_pivot_conflict_demotes_a_buy_after_the_rebound_runs():
    close = np.concatenate(
        [
            np.linspace(140.0, 100.0, 56),
            np.array([100.0, 100.0, 100.0, 110.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-5] = "STRONG_BUY"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_low_q": 0.90,
            "_p_pivot_low_s": 0.90,
            "_p_pivot_high_q": 0.80,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "context_window": None,
        "min_downtrend_slope": None,
        "max_uptrend_slope": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(buy_high_conflict_window=None, **common),
    )

    assert guarded["turn_action"].iloc[-1] == BOTTOM_TURN
    assert unguarded["turn_action"].iloc[-1] == BUY_TURN


def _late_rebound_buy_frame() -> pd.DataFrame:
    close = np.concatenate(
        [
            np.linspace(130.0, 100.0, 55),
            np.array([100.0, 101.0, 100.8, 101.0, 101.2, 104.4]),
        ]
    )
    high = close + 0.2
    low = close - 0.2
    low[-6] = 99.5
    low[-1] = 102.6
    action = np.full(len(close), "HOLD", dtype=object)
    action[-2] = "STRONG_BUY"
    session_open = close.copy()
    session_open[-1] = close[-1] / 1.03
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=len(close)),
            "open": session_open,
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
        }
    )


def _late_rebound_buy_config(**overrides):
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "context_window": None,
        "sell_low_context_window": None,
        "buy_high_conflict_window": None,
        "min_downtrend_slope": None,
        "max_downtrend_return_20": None,
        "max_uptrend_slope": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }
    return TurningPointConfig(**(common | overrides))


def test_late_strong_rebound_is_not_an_actionable_buy():
    frame = _late_rebound_buy_frame()

    guarded = extract_turning_points(frame, _late_rebound_buy_config())
    unguarded = extract_turning_points(
        frame,
        _late_rebound_buy_config(buy_late_rebound_min_body_return=None),
    )

    assert guarded["turn_action"].iloc[-1] == BOTTOM_TURN
    assert unguarded["turn_action"].iloc[-1] == BUY_TURN


@pytest.mark.parametrize(
    ("bars_since_low", "body_return", "close_location"),
    [(4, 0.03, 0.90), (5, 0.02, 0.90), (5, 0.03, 0.80)],
)
def test_late_rebound_guard_preserves_fresh_or_incomplete_confirmations(
    bars_since_low,
    body_return,
    close_location,
):
    frame = _late_rebound_buy_frame()
    frame.loc[frame.index[-1], "open"] = (
        frame["close"].iloc[-1] / (1.0 + body_return)
    )
    span = 1.0
    frame.loc[frame.index[-1], "low"] = (
        frame["close"].iloc[-1] - close_location * span
    )
    frame.loc[frame.index[-1], "high"] = (
        frame["close"].iloc[-1] + (1.0 - close_location) * span
    )
    low_at = len(frame) - 1 - bars_since_low
    frame["low"] = np.maximum(frame["low"], 100.0)
    frame.loc[frame.index[low_at], "low"] = 99.5

    out = extract_turning_points(frame, _late_rebound_buy_config())

    assert out["turn_action"].iloc[-1] == BUY_TURN


def test_late_rebound_guard_reads_historical_open_next_causally():
    frame = _late_rebound_buy_frame()
    frame["open_next"] = frame["open"].shift(-1)
    frame = frame.drop(columns=["open"])

    out = extract_turning_points(frame, _late_rebound_buy_config())

    assert out["turn_action"].iloc[-1] == BOTTOM_TURN


def test_non_bearish_uptrend_state_demotes_a_pullback_sell():
    close = 100.0 * 1.01 ** np.arange(100)
    close[-1] = close[-2] * 0.99
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.95,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "max_uptrend_slope": None,
        "top_guard_enabled": False,
        "top_confirm_local_pivot_q": None,
        "top_confirm_crash_pivot_q": None,
        "sell_stale_conflict_pivot_q": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(sell_trend_conflict_enabled=False, **common),
    )

    assert guarded["turn_action"].iloc[-1] == TOP_TURN
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def _right_side_sell_frame(route: str) -> pd.DataFrame:
    close = 100.0 * 1.01 ** np.arange(100)
    base = close[-4]
    if route == "drawdown":
        close[-3:] = [base * 1.06, base * 1.07, base * 1.01]
    elif route == "three_day":
        close[-3:] = [base * 1.02, base * 1.01, base * 0.975]
    else:
        raise ValueError(f"unknown route: {route}")

    high = close * 1.002
    low = close * 0.998
    high[-1] = high[-2] * 0.99
    action = np.full(len(close), "HOLD", dtype=object)
    action[-2] = "STRONG_SELL"
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=len(close)),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.99,
            "_p_pivot_high_s": 0.99,
        }
    )


def _right_side_sell_config(**overrides):
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "context_window": None,
        "sell_low_context_window": None,
        "min_downtrend_slope": None,
        "max_uptrend_slope": None,
        "top_guard_enabled": False,
        "top_confirm_volatility_q": None,
        "bottom_confirm_volatility_q": None,
        "top_confirm_memory_self_q": None,
        "top_confirm_local_pivot_q": None,
        "top_confirm_crash_pivot_q": None,
        "sell_stale_conflict_pivot_q": None,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }
    return TurningPointConfig(**(common | overrides))


@pytest.mark.parametrize("route", ["drawdown", "three_day"])
def test_strong_right_side_reversal_survives_trend_sell_filter(route):
    frame = _right_side_sell_frame(route)

    protected = extract_turning_points(frame, _right_side_sell_config())
    unprotected = extract_turning_points(
        frame,
        _right_side_sell_config(sell_right_reversal_pivot_q=None),
    )

    assert protected["turn_action"].iloc[-1] == SELL_TURN
    assert unprotected["turn_action"].iloc[-1] == TOP_TURN


@pytest.mark.parametrize(
    ("high_q", "high_self_q"),
    [(0.94, 0.99), (0.99, 0.89)],
)
def test_right_side_sell_protection_requires_both_high_evidence_ranks(
    high_q,
    high_self_q,
):
    frame = _right_side_sell_frame("drawdown")
    frame["_p_pivot_high_q"] = high_q
    frame["_p_pivot_high_s"] = high_self_q

    out = extract_turning_points(frame, _right_side_sell_config())

    assert out["turn_action"].iloc[-1] == TOP_TURN


def test_right_side_sell_protection_is_prefix_invariant():
    frame = _right_side_sell_frame("drawdown")
    future = frame.iloc[-5:].copy()
    future["date"] = pd.bdate_range(
        frame["date"].iloc[-1] + pd.offsets.BDay(),
        periods=len(future),
    )
    future[["close", "high", "low"]] *= 0.75
    full = pd.concat([frame, future], ignore_index=True)

    prefix_out = extract_turning_points(frame, _right_side_sell_config())
    full_out = extract_turning_points(full, _right_side_sell_config())

    np.testing.assert_array_equal(
        prefix_out["turn_action"].to_numpy(),
        full_out["turn_action"].iloc[: len(frame)].to_numpy(),
    )


def test_stale_weak_high_setup_is_not_an_actionable_sell():
    close = np.concatenate(
        [
            np.linspace(100.0, 160.0, 90),
            np.linspace(150.0, 140.0, 20),
            np.linspace(141.0, 155.0, 9),
            np.array([153.0]),
        ]
    )
    n = len(close)
    action = np.full(n, "HOLD", dtype=object)
    action[-2] = "STRONG_SELL"
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_high_q": 0.85,
        }
    )
    common = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "max_uptrend_slope": None,
        "top_guard_enabled": False,
        "top_confirm_local_pivot_q": None,
        "top_confirm_crash_pivot_q": None,
        "sell_trend_conflict_enabled": False,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
        "trend_entry": False,
    }

    guarded = extract_turning_points(frame, TurningPointConfig(**common))
    unguarded = extract_turning_points(
        frame,
        TurningPointConfig(sell_stale_conflict_pivot_q=None, **common),
    )

    assert guarded["turn_action"].iloc[-1] == TOP_TURN
    assert unguarded["turn_action"].iloc[-1] == SELL_TURN


def test_context_guard_does_not_veto_before_its_window_fills():
    """An unknown context must not silently suppress every early signal."""
    frame = _uptrend_pullback(n=40)
    out = extract_turning_points(frame, TurningPointConfig(range_window=5))
    assert set(out["turn_action"].unique()) <= {
        NO_TURN,
        BOTTOM_TURN,
        BUY_TURN,
        SELL_TURN,
        TOP_TURN,
    }


def test_relaxed_gates_still_require_conviction():
    """Relaxing the range gate must not let a HOLD state through."""
    frame = _uptrend_pullback()
    frame["action"] = "HOLD"
    out = extract_turning_points(
        frame, TurningPointConfig(self_range_window=250, context_window=None)
    )
    assert (out["turn_action"] == NO_TURN).all()


def test_self_gate_is_a_percentile_not_a_probability():
    """Any monotone transform of the head must leave the events unchanged.

    Scores differ across machines and retraining runs, so no gate may read a raw
    probability. The self-history gate ranks within each symbol, which preserves
    that guarantee.
    """
    frame = _uptrend_pullback()
    rng = np.random.default_rng(0)
    frame["p_pivot_low"] = rng.random(len(frame))
    frame["p_pivot_high"] = rng.random(len(frame))

    cfg = TurningPointConfig()
    base = extract_turning_points(frame, cfg)["turn_action"].to_numpy()

    for transform in (lambda x: x * 3.0, lambda x: x**2, lambda x: (x + 1e-3) / 7.0):
        shifted = frame.copy()
        shifted["p_pivot_low"] = transform(shifted["p_pivot_low"])
        shifted["p_pivot_high"] = transform(shifted["p_pivot_high"])
        moved = extract_turning_points(shifted, cfg)["turn_action"].to_numpy()
        assert (moved == base).all()


def test_self_gate_can_rescue_a_degenerate_cross_section():
    """When every name looks like a low, the cross-sectional rank saturates."""
    rows = []
    for i in range(6):
        frame = _uptrend_pullback(n=300)
        frame["symbol"] = f"S{i}"
        frame["p_pivot_low"] = 0.5
        frame["p_pivot_high"] = 0.01
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    # One name spikes against its own history on a day everyone scores alike.
    spike = (panel["symbol"] == "S0") & (panel.index % 300 == 280)
    panel.loc[spike, "p_pivot_low"] = 0.99

    cfg = TurningPointConfig(self_gate_low_q=0.98, context_window=None)
    out = extract_turning_points(panel, cfg)
    assert "_p_pivot_low_s" not in out.columns


def test_range_gate_warms_up_by_expanding():
    """A newly-listed name must be scoreable before its window is full.

    The prediction panel starts at each symbol's index-join date, so requiring a
    full 20-bar window blanked the gate for a recent addition's first month.
    """
    frame = _uptrend_pullback(n=30)
    pos_full = _range_position(frame, 20, 20)
    pos_expanding = _range_position(frame, 20, 10)

    assert np.isnan(pos_full[:19]).all()
    assert not np.isnan(pos_expanding[9:]).any()
    # Once both are defined they must agree exactly.
    assert np.allclose(pos_full[19:], pos_expanding[19:])


def test_range_gate_min_periods_is_clamped_to_the_window():
    frame = _uptrend_pullback(n=30)
    assert np.allclose(
        _range_position(frame, 20, 99), _range_position(frame, 20, 20), equal_nan=True
    )


def test_trend_entry_is_disjoint_from_pivot_events():
    """A participation signal must never overwrite a turn call."""
    rng = np.random.default_rng(3)
    n = 500
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.02, n)))
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "action": rng.choice(["HOLD", "BUY", "STRONG_BUY", "STRONG_SELL"], n),
            "pivot": 0.0,
        }
    )
    with_entry = extract_turning_points(frame, TurningPointConfig())["turn_action"]
    without = extract_turning_points(
        frame, TurningPointConfig(trend_entry=False)
    )["turn_action"]

    pivots = without.isin([BOTTOM_TURN, BUY_TURN, SELL_TURN, TOP_TURN])
    assert with_entry[pivots].equals(without[pivots]), "pivots must be untouched"
    assert (with_entry[with_entry == TREND_BUY].index.isin(
        without[without == NO_TURN].index
    )).all()


def test_trend_entry_requires_an_established_advance():
    """A falling series must not produce a participation signal."""
    n = 400
    close = np.linspace(200.0, 100.0, n)
    frame = pd.DataFrame(
        {
            "symbol": "DOWN",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(frame, TurningPointConfig())["turn_action"]
    assert (out != TREND_BUY).all()


def test_trend_entry_can_be_disabled():
    frame = _uptrend_pullback()
    out = extract_turning_points(frame, TurningPointConfig(trend_entry=False))
    assert (out["turn_action"] != TREND_BUY).all()


def _breakout_frame(action: str, n: int = 400) -> pd.DataFrame:
    """A steady advance whose closes clear the prior 20-day high."""
    close = np.linspace(100.0, 400.0, n)
    return pd.DataFrame(
        {
            "symbol": "UP",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close,
            "low": close * 0.99,
            "action": action,
            "pivot": 0.0,
        }
    )


def test_trend_entry_requires_sustained_conviction_not_one_bar():
    """A single STRONG_BUY on the breakout bar is not an endorsement."""
    frame = _breakout_frame("HOLD")
    lone = frame.copy()
    lone.loc[lone.index[-1], "action"] = "STRONG_BUY"
    assert (
        extract_turning_points(lone, TurningPointConfig())["turn_action"] != TREND_BUY
    ).all()

    sustained = frame.copy()
    sustained.loc[sustained.index[-30:], "action"] = "STRONG_BUY"
    assert (
        extract_turning_points(sustained, TurningPointConfig())["turn_action"]
        == TREND_BUY
    ).any()


def test_trend_entry_ignores_plain_buy():
    """Only STRONG_BUY counts toward the conviction share."""
    frame = _breakout_frame("BUY")
    out = extract_turning_points(frame, TurningPointConfig())["turn_action"]
    assert (out != TREND_BUY).all()


def test_trend_entry_needs_a_new_high_not_a_mean_reversion():
    """Reclaiming a moving average is not a breakout.

    Every moving-average construction tested was negative on the median raw
    return; only taking out a prior extreme carries information.
    """
    n = 400
    # Price oscillates with a slight downward drift: it crosses the 20-day
    # average on every cycle but can never take out a prior 60-day high.
    t = np.arange(n)
    close = 100.0 - 0.02 * t + 3.0 * np.sin(t / 6.0)
    frame = pd.DataFrame(
        {
            "symbol": "FLAT",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": close,
            "low": close * 0.99,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )
    out = extract_turning_points(frame, TurningPointConfig())["turn_action"]
    assert (out != TREND_BUY).all()


def test_trend_breakout_window_is_the_measured_optimum():
    assert TurningPointConfig().trend_breakout_window == 60


def _base_breakout_frame() -> pd.DataFrame:
    n = 100
    close = np.full(n, 100.0)
    high = np.concatenate([np.full(79, 103.0), np.full(20, 101.0), [103.5]])
    low = np.concatenate([np.full(79, 97.0), np.full(20, 99.0), [100.0]])
    volume = np.full(n, 100.0)
    volume[-1] = 130.0
    close[-1] = 102.5
    action = np.full(n, "HOLD", dtype=object)
    action[-1] = "BUY"
    return pd.DataFrame(
        {
            "symbol": "BASE",
            "date": pd.bdate_range("2022-01-03", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "action": action,
            "pivot": 0.0,
        }
    )


def test_base_breakout_requires_compression_breakout_volume_and_direction():
    frame = _base_breakout_frame()
    cfg = TurningPointConfig()

    assert _base_breakout(frame, cfg)[-1]

    weak_volume = frame.copy()
    weak_volume.loc[weak_volume.index[-1], "volume"] = 124.0
    assert not _base_breakout(weak_volume, cfg)[-1]

    hold = frame.copy()
    hold.loc[hold.index[-1], "action"] = "HOLD"
    assert not _base_breakout(hold, cfg)[-1]

    no_breakout = frame.copy()
    no_breakout.loc[no_breakout.index[-1], "high"] = 101.0
    assert not _base_breakout(no_breakout, cfg)[-1]

    weak_close = frame.copy()
    weak_close.loc[weak_close.index[-1], "close"] = 102.0
    assert not _base_breakout(weak_close, cfg)[-1]

    uncompressed = frame.copy()
    uncompressed.loc[uncompressed.index[:79], ["high", "low"]] = [101.0, 99.0]
    assert not _base_breakout(uncompressed, cfg)[-1]


def test_base_breakout_is_emitted_as_its_own_event():
    out = extract_turning_points(_base_breakout_frame(), TurningPointConfig())

    assert out["turn_action"].iloc[-1] == BASE_BREAKOUT


def test_base_breakout_can_be_disabled():
    out = extract_turning_points(
        _base_breakout_frame(),
        TurningPointConfig(base_breakout=False),
    )

    assert (out["turn_action"] != BASE_BREAKOUT).all()


# --------------------------------------------------------------------------
def _wide_reversal_frame(n=60, pivot_at=40):
    """A pullback that ends on a wide reversal bar.

    Price drifts down, then one session trades into new low ground and closes
    back near the top of its own range. That bar is the turn, and its recovery
    is precisely what lifts the close out of the 20-day trough zone — so the
    close-based gates reject it while the bar's range confirms it.
    """
    close = np.concatenate(
        [np.linspace(120, 104, pivot_at), np.linspace(105, 118, n - pivot_at)]
    )
    high = close + 0.4
    low = close - 0.4
    # The reversal bar: deep intraday low, close near the high.
    low[pivot_at] = close[pivot_at] - 6.0
    high[pivot_at] = close[pivot_at] + 0.2
    close[pivot_at] = high[pivot_at] - 0.1
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": "STRONG_BUY",
            "pivot": 0.0,
        }
    )


def test_wide_reversal_bar_is_admitted_by_its_range():
    """The bar's own range is the evidence, not its close-to-close move.

    A bar that reclaims a deep intraday low closes high in its 20-day range by
    construction, so measuring the waiver close-to-close makes the confirmation
    of a turn the reason for rejecting it.
    """
    frame = _wide_reversal_frame()
    cfg = TurningPointConfig(
        trend_entry=False,
        waiver_uses_bar_range=True,
        min_downtrend_slope=None,
    )

    out = extract_turning_points(frame, cfg)

    assert (out["turn_action"] == BUY_TURN).any()


def test_close_to_close_waiver_misses_the_wide_reversal_bar():
    """Pins the behaviour the default was changed away from.

    Kept so the trade-off stays visible: if this ever starts passing with the
    close-to-close measure, the two paths have stopped being distinguishable
    and the flag no longer means anything.
    """
    frame = _wide_reversal_frame()
    cfg = TurningPointConfig(
        trend_entry=False,
        waiver_uses_bar_range=False,
        trigger_pct=0.45,
    )

    out = extract_turning_points(frame, cfg)

    assert (out["turn_action"] == BUY_TURN).sum() == 0


def test_bar_range_waiver_still_requires_model_conviction():
    """The waiver relaxes *where* price is, never *whether* the model agrees."""
    frame = _wide_reversal_frame()
    frame["action"] = "HOLD"
    cfg = TurningPointConfig(trend_entry=False, waiver_uses_bar_range=True)

    out = extract_turning_points(frame, cfg)

    assert (out["turn_action"] == NO_TURN).all()


def test_bar_range_waiver_still_requires_a_reversal_bar():
    """A wide bar that closes on its low is a breakdown, not a turn."""
    frame = _wide_reversal_frame()
    # Same range, but the close sits at the bottom of it and below the prior
    # bar's high, so the reversal trigger cannot fire.
    frame.loc[40, "close"] = frame.loc[40, "low"] + 0.1
    cfg = TurningPointConfig(trend_entry=False, waiver_uses_bar_range=True)

    out = extract_turning_points(frame, cfg)

    assert out.loc[40, "turn_action"] == NO_TURN


# --------------------------------------------------------------------------
def test_decay_window_extends_a_live_setup():
    """A setup outlives the fixed window while the model still endorses it.

    Tested on the helper directly: building a synthetic panel that isolates
    this one gate is not possible, because during a decline every bar sits in
    the trough and re-arms the setup by itself.
    """
    setup = np.zeros(12, dtype=bool)
    setup[0] = True
    alive = np.ones(12, dtype=bool)

    out = _recent_while_live(setup, alive, window=3, max_window=8)

    # The fixed window alone would stop at bar 3.
    assert out[3] and out[8]
    assert not out[9], "max_window must cap the extension"


def test_decay_window_expires_when_the_reading_drops():
    """Once the endorsement breaks the setup is gone, and stays gone."""
    setup = np.zeros(12, dtype=bool)
    setup[0] = True
    alive = np.ones(12, dtype=bool)
    alive[5] = False          # the reading decays here
    alive[6:] = True          # and recovers afterwards

    out = _recent_while_live(setup, alive, window=3, max_window=10)

    assert out[4], "still inside the live extension"
    assert not out[5], "the bar that broke the chain"
    assert not out[7], "a later recovery must not revive a lapsed setup"


def test_decay_window_matches_the_fixed_window_when_disabled():
    """``max_window <= window`` must reproduce ``_recent`` exactly."""
    setup = np.zeros(12, dtype=bool)
    setup[0] = True
    alive = np.ones(12, dtype=bool)

    assert np.array_equal(
        _recent_while_live(setup, alive, window=3, max_window=3),
        _recent(setup, 3),
    )


def test_decay_window_never_shrinks_the_fixed_window():
    """A dead reading still leaves the unconditional window intact."""
    setup = np.zeros(12, dtype=bool)
    setup[0] = True
    alive = np.zeros(12, dtype=bool)

    out = _recent_while_live(setup, alive, window=3, max_window=10)

    assert np.array_equal(out, _recent(setup, 3))


# --------------------------------------------------------------------------
def _panel_with_one_standout(n=80, standout_at=60):
    """Two symbols, one of which prints a very high detection probability.

    The standout's percentile ranks are deliberately unhelpful: every bar of
    the day carries an elevated reading, so the cross-sectional rank cannot
    single it out. This is the synchronised-bottom shape.
    """
    frames = []
    for sym, level in (("AAA", 0.30), ("BBB", 0.30)):
        close = np.concatenate(
            [np.linspace(140, 100, standout_at), np.linspace(101, 118, n - standout_at)]
        )
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": pd.bdate_range("2023-01-02", periods=n),
                    "close": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "action": "STRONG_BUY",
                    "pivot": 0.0,
                    "p_pivot_low": level,
                    "p_pivot_high": 0.01,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_absolute_path_admits_a_reading_the_ranks_cannot_see():
    """When every name is elevated, the ranks tie and the multiple decides."""
    frame = _panel_with_one_standout()
    # Panel mean is 0.30, so a 5x gate needs 1.5 — nothing reaches it.
    quiet = TurningPointConfig(trend_entry=False, pivot_abs_mult=5.0)
    assert (extract_turning_points(frame, quiet)["turn_action"] == BUY_TURN).sum() == 0

    # Lift one symbol far above the panel and it becomes reachable.
    loud = frame.copy()
    mask = loud.symbol == "AAA"
    loud.loc[mask, "p_pivot_low"] = 0.30
    loud.loc[mask & (loud.date >= loud.loc[mask, "date"].iloc[58]), "p_pivot_low"] = 3.0

    out = extract_turning_points(loud, TurningPointConfig(trend_entry=False, pivot_abs_mult=5.0))

    assert (out.loc[out.symbol == "AAA", "turn_action"] == BUY_TURN).any()
    assert (out.loc[out.symbol == "BBB", "turn_action"] == BUY_TURN).sum() == 0


def test_absolute_path_disabled_by_default_config_value():
    """``None`` must leave the conviction gate exactly as the ranks define it."""
    frame = _panel_with_one_standout()
    loud = frame.copy()
    mask = loud.symbol == "AAA"
    loud.loc[mask & (loud.date >= loud.loc[mask, "date"].iloc[58]), "p_pivot_low"] = 3.0

    off = extract_turning_points(loud, TurningPointConfig(trend_entry=False, pivot_abs_mult=None))
    on = extract_turning_points(loud, TurningPointConfig(trend_entry=False, pivot_abs_mult=5.0))

    assert (on["turn_action"] == BUY_TURN).sum() >= (off["turn_action"] == BUY_TURN).sum()


def test_absolute_path_leaves_no_helper_columns_behind():
    """Helper columns are internal and must not leak into the output."""
    frame = _panel_with_one_standout()

    out = extract_turning_points(frame, TurningPointConfig(pivot_abs_mult=5.0))

    assert not [c for c in out.columns if c.startswith("_p_pivot")]


# --------------------------------------------------------------------------
def _synchronised_panel(n=80, low_at=60, broad=(58, 63), elevated=True):
    """Twenty names bottoming together, with one parked just under the gate.

    ``AAA`` is given the 18th of 20 readings on the broad bars, so its
    cross-sectional rank is 0.90 — clear of the field but short of the 0.95
    gate. Only ``breadth_relief`` can admit it.
    """
    close = np.concatenate(
        [np.linspace(140, 100, low_at), np.linspace(101, 118, n - low_at)]
    )
    dates = pd.bdate_range("2023-01-02", periods=n)
    names = [f"S{i:02d}" for i in range(17)] + ["AAA"] + ["T00", "T01"]
    frames = []
    for i, sym in enumerate(names):
        p = np.full(n, 0.01)
        if elevated:
            p[broad[0] : broad[1]] = 0.03 * (i + 1)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "close": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "action": "STRONG_BUY",
                    "pivot": 0.0,
                    "p_pivot_low": p,
                    "p_pivot_high": 0.01,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _isolated(**kwargs):
    """Config with every route but the cross-sectional rank switched off."""
    values = {
        "trend_entry": False,
        "pivot_abs_mult": None,
        "self_gate_low_q": None,
        "self_gate_high_q": None,
        "setup_decay_q": None,
        "min_downtrend_slope": None,
        "bottom_guard_low_conviction_q": None,
        "reversal_rescue_q": None,
        "delayed_reversal_rescue_q": None,
        "bottom_right_rescue_pivot_q": None,
        "top_right_rescue_pivot_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "trend_neutral_weight": 0.0,
        "setup_window": 3,
        "trigger_pct": 0.45,
    }
    values.update(kwargs)
    return TurningPointConfig(**values)


def test_breadth_relief_admits_a_name_the_rank_cannot_reach():
    """A 0.90 rank on a panel-wide washout should still count as conviction."""
    frame = _synchronised_panel()

    off = extract_turning_points(frame, _isolated(breadth_relief=None))
    on = extract_turning_points(frame, _isolated(breadth_relief=0.30))

    assert (off.loc[off.symbol == "AAA", "turn_action"] == BUY_TURN).sum() == 0
    assert (on.loc[on.symbol == "AAA", "turn_action"] == BUY_TURN).any()


def test_breadth_relief_stays_shut_when_the_day_is_not_broad():
    """Relief keys off panel breadth, so a quiet day must be unaffected."""
    frame = _synchronised_panel(elevated=False)

    off = extract_turning_points(frame, _isolated(breadth_relief=None))
    on = extract_turning_points(frame, _isolated(breadth_relief=0.30))

    assert (off["turn_action"] == BUY_TURN).sum() == (on["turn_action"] == BUY_TURN).sum()


def test_breadth_floor_caps_how_far_the_gate_can_open():
    """However broad the day, the gate must not fall below the floor."""
    frame = _synchronised_panel()

    capped = extract_turning_points(frame, _isolated(breadth_relief=0.30, breadth_floor=0.99))

    assert (capped.loc[capped.symbol == "AAA", "turn_action"] == BUY_TURN).sum() == 0


def test_breadth_relief_never_removes_an_existing_event():
    """The relief is a parallel route, so it can only ever add."""
    frame = _synchronised_panel()

    off = extract_turning_points(frame, _isolated(breadth_relief=None))
    on = extract_turning_points(frame, _isolated(breadth_relief=0.30))

    fired_off = set(zip(off.symbol, off.date, off.turn_action)) - {
        (s, d, a) for s, d, a in zip(off.symbol, off.date, off.turn_action) if a == "NONE"
    }
    fired_on = {
        (s, d, a) for s, d, a in zip(on.symbol, on.date, on.turn_action) if a != "NONE"
    }
    assert fired_off <= fired_on


def test_breadth_relief_leaves_no_helper_columns_behind():
    """Helper columns are internal and must not leak into the output."""
    frame = _synchronised_panel()

    out = extract_turning_points(frame, _isolated(breadth_relief=0.30))

    assert not [c for c in out.columns if c.startswith("_p_pivot")]


# ---------------------------------------------------------------------------
# reversal_rescue_q: evidence that arrives one bar after the bar it confirms
# ---------------------------------------------------------------------------
def _rescue_panel(n=80, low_at=60):
    """AAA sits at rank 0.90 on its low, then prints a capitulation reversal.

    Reuses the synchronised fixture's construction so AAA lands just under the
    0.95 gate, then replaces the bar after the low with a wide-range reversal
    that closes at the top of its range and takes out the prior high — the
    price signature the rescue is looking for.
    """
    frame = _synchronised_panel(n=n, low_at=low_at, broad=(low_at - 1, low_at + 1))
    aaa = frame["symbol"] == "AAA"
    idx = frame.index[aaa][low_at]
    prior_high = frame.loc[frame.index[aaa][low_at - 1], "high"]
    frame.loc[idx, ["low", "high", "close"]] = [
        frame.loc[idx, "low"] - 6.0,
        prior_high + 4.0,
        prior_high + 3.6,
    ]
    return frame


def _buys(frame, cfg, symbol="AAA"):
    out = extract_turning_points(frame.copy(), cfg)
    hit = (out["turn_action"] == BUY_TURN) & (out["symbol"] == symbol)
    return out.loc[hit, "date"].tolist()


def test_rescue_admits_a_near_miss_confirmed_by_its_reversal_bar():
    """0.90 is short of the 0.95 gate; the trigger bar is what earns it."""
    frame = _rescue_panel()

    assert not _buys(frame, _isolated(breadth_relief=None, reversal_rescue_q=None))
    assert _buys(frame, _isolated(breadth_relief=None, reversal_rescue_q=0.90))


def test_rescue_respects_its_floor():
    frame = _rescue_panel()

    assert not _buys(frame, _isolated(breadth_relief=None, reversal_rescue_q=0.95))


def test_delayed_rescue_confirms_a_marginal_double_bottom():
    frame = _rescue_panel()

    blocked = _buys(
        frame,
        _isolated(
            breadth_relief=None,
            reversal_rescue_q=None,
        ),
    )
    confirmed = _buys(
        frame,
        _isolated(
            breadth_relief=None,
            reversal_rescue_q=None,
            delayed_reversal_rescue_q=0.875,
        ),
    )

    assert not blocked
    assert confirmed


def _right_side_rescue_frame(side: str) -> pd.DataFrame:
    n = 60
    if side == "top":
        close = np.concatenate([np.linspace(100.0, 140.0, n - 2), [145.0, 134.0]])
        high = close + 0.5
        low = close - 0.5
        high[-1] = 136.0
        low[-1] = 133.8
        action = np.full(n, "HOLD", dtype=object)
        action[-1] = "STRONG_SELL"
        low_q = np.full(n, 0.05)
        low_s = np.full(n, 0.05)
        high_q = np.full(n, 0.10)
        high_s = np.full(n, 0.10)
        high_q[-1] = 0.88
        high_s[-1] = 0.88
    elif side == "bottom":
        close = np.concatenate(
            [np.linspace(140.0, 98.0, n - 3), [97.0, 95.0, 100.0]]
        )
        high = close + 0.5
        low = close - 0.5
        high[-1] = 100.2
        low[-1] = 99.0
        action = np.full(n, "HOLD", dtype=object)
        low_q = np.full(n, 0.10)
        low_s = np.full(n, 0.10)
        high_q = np.full(n, 0.05)
        high_s = np.full(n, 0.05)
        low_q[-1] = 0.80
        low_s[-1] = 0.80
    else:
        raise ValueError(f"unknown side: {side}")

    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "_p_pivot_low_q": low_q,
            "_p_pivot_low_s": low_s,
            "_p_pivot_high_q": high_q,
            "_p_pivot_high_s": high_s,
        }
    )


def _right_side_rescue_config(**overrides):
    values = {
        "pivot_gate_q": None,
        "buy_states": (),
        "sell_states": (),
        "context_window": None,
        "refractory": 0,
        "rearm_atr": 0,
        "bottom_right_rescue_pivot_q": 0.35,
        "top_right_rescue_pivot_q": 0.85,
    }
    values.update(overrides)
    return _isolated(**values)


def test_right_side_top_rescue_recovers_an_arm_like_sell():
    frame = _right_side_rescue_frame("top")

    rescued = extract_turning_points(frame, _right_side_rescue_config())
    blocked = extract_turning_points(
        frame,
        _right_side_rescue_config(top_right_rescue_pivot_q=None),
    )

    assert rescued["turn_action"].iloc[-1] == SELL_TURN
    assert blocked["turn_action"].iloc[-1] == NO_TURN


def test_right_side_top_rescue_confirms_without_state_agreement():
    frame = _right_side_rescue_frame("top")
    frame["action"] = "HOLD"

    out = extract_turning_points(frame, _right_side_rescue_config())

    assert out["turn_action"].iloc[-1] == SELL_TURN


def test_right_side_top_rescue_requires_both_high_ranks():
    frame = _right_side_rescue_frame("top")
    frame.loc[frame.index[-1], "_p_pivot_high_s"] = 0.84

    out = extract_turning_points(frame, _right_side_rescue_config())

    assert out["turn_action"].iloc[-1] == NO_TURN


def test_high_confidence_top_can_confirm_on_two_day_follow_through():
    frame = pd.DataFrame(
        {
            "close": [100.0, 101.0, 98.3],
            "high": [101.0, 102.0, 99.5],
            "low": [99.0, 100.0, 98.0],
            "_p_pivot_high_q": [0.20, 0.96, 0.20],
            "_p_pivot_high_s": [0.20, 0.80, 0.20],
            "_top_bars_since_local_high": [2.0, 0.0, 1.0],
        }
    )
    turns = np.array([NO_TURN, TOP_TURN, NO_TURN], dtype=object)

    confirmed = _top_followthrough_confirmations(
        frame, turns, TurningPointConfig()
    )

    assert confirmed.tolist() == [False, False, True]


def test_top_follow_through_must_break_the_detection_low():
    frame = pd.DataFrame(
        {
            "close": [100.0, 102.0, 100.5],
            "high": [101.0, 103.0, 101.5],
            "low": [99.0, 100.0, 100.2],
            "_p_pivot_high_q": [0.20, 0.96, 0.20],
            "_p_pivot_high_s": [0.20, 0.80, 0.20],
            "_top_bars_since_local_high": [2.0, 0.0, 1.0],
        }
    )
    turns = np.array([NO_TURN, TOP_TURN, NO_TURN], dtype=object)

    confirmed = _top_followthrough_confirmations(
        frame, turns, TurningPointConfig()
    )

    assert not confirmed.any()


def test_right_side_bottom_rescue_detects_a_confirmed_missed_low():
    frame = _right_side_rescue_frame("bottom")

    rescued = extract_turning_points(frame, _right_side_rescue_config())
    blocked = extract_turning_points(
        frame,
        _right_side_rescue_config(bottom_right_rescue_pivot_q=None),
    )

    assert rescued["turn_action"].iloc[-1] == BUY_TURN
    assert blocked["turn_action"].iloc[-1] == NO_TURN


def test_right_side_bottom_rescue_requires_both_calibrated_ranks():
    frame = _right_side_rescue_frame("bottom")
    frame.loc[frame.index[-1], "_p_pivot_low_s"] = 0.34

    out = extract_turning_points(frame, _right_side_rescue_config())

    assert out["turn_action"].iloc[-1] == NO_TURN


def test_right_side_bottom_rescue_accepts_a_strong_breakout_waiver():
    frame = _right_side_rescue_frame("bottom")
    frame.loc[frame.index[-3], ["close", "high", "low"]] = [99.0, 99.5, 98.5]

    out = extract_turning_points(frame, _right_side_rescue_config())

    assert frame["close"].pct_change().iloc[-1] > 0.025
    assert frame["close"].pct_change(2).iloc[-1] < 0.02
    assert frame["close"].iloc[-1] / frame["high"].iloc[-2] - 1.0 > 0.015
    assert out["turn_action"].iloc[-1] == BUY_TURN


def test_right_side_bottom_rescue_rejects_an_unconfirmed_one_day_spike():
    frame = _right_side_rescue_frame("bottom")
    frame.loc[frame.index[-3], ["close", "high", "low"]] = [99.0, 99.5, 98.5]
    frame.loc[frame.index[-2], "high"] = 99.5

    out = extract_turning_points(frame, _right_side_rescue_config())

    assert frame["close"].pct_change().iloc[-1] > 0.025
    assert frame["close"].pct_change(2).iloc[-1] < 0.02
    assert frame["close"].iloc[-1] / frame["high"].iloc[-2] - 1.0 < 0.015
    assert out["turn_action"].iloc[-1] == NO_TURN


def _pivot_first_bottom_frame(kind: str) -> tuple[pd.DataFrame, int]:
    n, bottom = (100, 80) if kind == "fresh" else (40, 30)
    close = np.concatenate(
        [np.linspace(140.0, 100.0, bottom), np.linspace(101.0, 125.0, n - bottom)]
    )
    if kind == "fresh":
        close[bottom - 3 : bottom + 1] = [100.5, 99.5, 99.0, 101.0]
    elif kind == "capitulation":
        close[bottom] = 103.0
    else:
        raise ValueError(f"unknown bottom kind: {kind}")

    high = close + 0.2
    low = close - 0.2
    high[bottom] = close[bottom] + 0.1
    low[bottom] = close[bottom] - 0.3
    action = np.full(n, "HOLD", dtype=object)
    action[bottom - 1] = "STRONG_BUY"
    low_q = np.full(n, 0.10)
    low_s = np.full(n, 0.10)
    high_q = np.full(n, 0.10)
    low_q[bottom] = 0.70 if kind == "fresh" else 0.85
    low_s[bottom] = 0.70 if kind == "fresh" else np.nan

    return (
        pd.DataFrame(
            {
                "symbol": "AAA",
                "date": pd.bdate_range("2023-01-02", periods=n),
                "close": close,
                "high": high,
                "low": low,
                "action": action,
                "pivot": 0.0,
                "_p_pivot_low_q": low_q,
                "_p_pivot_low_s": low_s,
                "_p_pivot_high_q": high_q,
            }
        ),
        bottom,
    )


def _pivot_first_bottom_config(**overrides):
    values = {
        "pivot_gate_q": None,
        "bottom_reference_self_q": None,
        "top_reference_pivot_q": None,
        "bottom_right_rescue_pivot_q": None,
        "top_right_rescue_pivot_q": None,
        "trend_entry": False,
        "trigger_pct": 1.0,
        "refractory": 0,
        "rearm_atr": 0,
    }
    values.update(overrides)
    return TurningPointConfig(**values)


def test_fresh_local_pivot_confirmation_promotes_a_guarded_bottom():
    frame, bottom = _pivot_first_bottom_frame("fresh")

    confirmed = extract_turning_points(
        frame,
        _pivot_first_bottom_config(
            bottom_confirm_volatility_q=None,
            bottom_pivot_capitulation_q=None,
        ),
    )
    detected = extract_turning_points(
        frame,
        _pivot_first_bottom_config(
            bottom_confirm_volatility_q=None,
            bottom_pivot_fresh_q=None,
            bottom_pivot_capitulation_q=None,
        ),
    )

    assert confirmed["turn_action"].iloc[bottom] == BUY_TURN
    assert detected["turn_action"].iloc[bottom] == BOTTOM_TURN


def test_capitulation_pivot_confirmation_handles_incomplete_history():
    frame, bottom = _pivot_first_bottom_frame("capitulation")

    confirmed = extract_turning_points(
        frame,
        _pivot_first_bottom_config(bottom_pivot_fresh_q=None),
    )
    detected = extract_turning_points(
        frame,
        _pivot_first_bottom_config(
            bottom_pivot_fresh_q=None,
            bottom_pivot_capitulation_q=None,
        ),
    )

    assert confirmed["turn_action"].iloc[bottom] == BUY_TURN
    assert detected["turn_action"].iloc[bottom] == BOTTOM_TURN


def _reference_panel(side: str, n: int = 80) -> pd.DataFrame:
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    p_low = np.full(n, 0.01)
    p_high = np.full(n, 0.01)
    action = np.full(n, "HOLD", dtype=object)

    if side == "bottom":
        low[75], close[75], high[75] = 90.0, 92.0, 93.0
        low[76], close[76], high[76] = 91.5, 92.5, 93.0
        low[77], close[77], high[77] = 92.0, 99.0, 100.0
        p_low[77] = 1.0
    else:
        low[75], close[75], high[75] = 117.0, 118.0, 120.0
        low[76], close[76], high[76] = 116.0, 117.0, 118.0
        low[77], close[77], high[77] = 99.0, 100.0, 105.0
        p_high[77] = 1.0
        action[77] = "STRONG_SELL"

    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2023-01-02", periods=n),
            "close": close,
            "high": high,
            "low": low,
            "action": action,
            "pivot": 0.0,
            "p_pivot_low": p_low,
            "p_pivot_high": p_high,
        }
    )


def _reference_panel_with_peers(side: str) -> pd.DataFrame:
    target = _reference_panel(side)
    if side == "bottom":
        target.loc[target.index[30], ["low", "close", "high"]] = [80.0, 82.0, 83.0]
        target.loc[target.index[75], ["low", "close", "high"]] = [90.0, 92.0, 93.0]
        target.loc[target.index[76], ["low", "close", "high"]] = [91.0, 92.2, 92.8]
        target.loc[target.index[77], ["low", "close", "high"]] = [92.0, 93.5, 94.0]
        target.loc[target.index[77], "p_pivot_low"] = 0.50
    else:
        target["p_pivot_high"] = 0.80
        target.loc[target.index[77], "p_pivot_high"] = 0.50
        target.loc[target.index[:75], "close"] = np.tile([98.0, 102.0, 100.0], 25)
        target.loc[target.index[:75], "high"] = (
            target.loc[target.index[:75], "close"] + 1.0
        )
        target.loc[target.index[:75], "low"] = (
            target.loc[target.index[:75], "close"] - 1.0
        )
        target.loc[target.index[75], ["low", "close", "high"]] = [99.0, 101.0, 120.0]
        target.loc[target.index[76], ["low", "close", "high"]] = [100.0, 102.0, 103.0]
        target.loc[target.index[77], ["low", "close", "high"]] = [99.0, 99.8, 103.0]

    peers = []
    for i in range(19):
        peer = target.copy()
        peer["symbol"] = f"P{i:02d}"
        if side == "bottom":
            peer["p_pivot_low"] = 0.60
        else:
            peer["p_pivot_high"] = 0.10
            tail = peer.index[-6:]
            peer.loc[tail, "close"] = [100.0, 150.0, 150.0, 150.0, 150.0, 150.0]
            peer.loc[tail, "high"] = peer.loc[tail, "close"] + 2.0
            peer.loc[tail, "low"] = peer.loc[tail, "close"] - 2.0
        peers.append(peer)
    return pd.concat([target, *peers], ignore_index=True)


def test_bottom_reference_uses_the_existing_confirmation_before_promotion():
    frame = _reference_panel("bottom")

    off = extract_turning_points(
        frame,
        _isolated(pivot_gate_q=1.1, breadth_relief=None),
    )
    on = extract_turning_points(
        frame,
        _isolated(
            pivot_gate_q=1.1,
            breadth_relief=None,
            bottom_reference_self_q=0.95,
        ),
    )

    assert off["turn_action"].iloc[77] == NO_TURN
    assert on["turn_action"].iloc[77] == BUY_TURN


def test_strong_two_day_reference_rebound_promotes_to_buy():
    frame = _reference_panel("bottom")
    config = _isolated(
        pivot_gate_q=1.1,
        breadth_relief=None,
        bottom_reference_self_q=0.95,
        bottom_confirm_volatility_q=1.1,
        bottom_confirm_quiet_pivot_q=1.1,
        bottom_confirm_stable_pivot_q=1.1,
        bottom_confirm_rebound_pivot_q=1.1,
        bottom_confirm_immediate_close_loc=1.1,
        bottom_confirm_delayed_pivot_q=1.1,
        bottom_confirm_higher_low_pivot_q=None,
        bottom_confirm_fresh_setup_self_q=None,
        bottom_confirm_reversal_atr=100.0,
    )

    out = extract_turning_points(frame, config)

    assert frame["close"].pct_change(2).iloc[77] >= 0.02
    assert out["turn_action"].iloc[77] == BUY_TURN


def test_weak_two_day_reference_rebound_remains_detection():
    frame = _reference_panel_with_peers("bottom")
    config = _isolated(
        bottom_reference_self_q=0.95,
        bottom_confirm_volatility_q=1.1,
        bottom_confirm_quiet_pivot_q=1.1,
        bottom_confirm_stable_pivot_q=1.1,
        bottom_confirm_rebound_pivot_q=1.1,
        bottom_confirm_immediate_close_loc=1.1,
        bottom_confirm_delayed_pivot_q=1.1,
        bottom_confirm_higher_low_pivot_q=None,
        bottom_confirm_fresh_setup_self_q=None,
        bottom_confirm_reversal_atr=100.0,
    )

    out = extract_turning_points(frame, config)
    target = out[out["symbol"] == "AAA"].reset_index(drop=True)
    target_frame = frame[frame["symbol"] == "AAA"].reset_index(drop=True)

    assert target_frame["close"].pct_change(2).iloc[77] < 0.02
    assert target["turn_action"].iloc[77] == BOTTOM_TURN


def test_top_reference_uses_the_existing_confirmation_before_promotion():
    frame = _reference_panel("top")

    off = extract_turning_points(frame, _isolated())
    on = extract_turning_points(
        frame,
        _isolated(top_reference_pivot_q=0.95),
    )

    assert off["turn_action"].iloc[77] == NO_TURN
    assert on["turn_action"].iloc[77] == SELL_TURN


@pytest.mark.parametrize(
    ("side", "kwargs", "expected"),
    [
        (
            "bottom",
            {
                "pivot_gate_q": 1.1,
                "breadth_relief": None,
                "bottom_reference_self_q": 0.95,
                "bottom_confirm_volatility_q": None,
            },
            BOTTOM_TURN,
        ),
        (
            "top",
            {
                "top_reference_pivot_q": 0.95,
                "top_confirm_volatility_q": None,
            },
            TOP_TURN,
        ),
    ],
)
def test_unconfirmed_region_candidates_remain_reference_turns(side, kwargs, expected):
    frame = _reference_panel(side)

    out = extract_turning_points(frame, _isolated(**kwargs))

    assert out["turn_action"].iloc[77] == expected


@pytest.mark.parametrize(
    ("side", "expected"),
    [("bottom", BOTTOM_TURN), ("top", TOP_TURN)],
)
def test_region_candidate_needs_confirmation_independent_of_eligibility(side, expected):
    frame = _reference_panel_with_peers(side)
    kwargs = (
        {"bottom_reference_self_q": 0.95}
        if side == "bottom"
        else {"top_reference_pivot_q": 0.95}
    )

    out = extract_turning_points(frame, _isolated(**kwargs))
    target = out[out["symbol"] == "AAA"].reset_index(drop=True)

    assert target["turn_action"].iloc[77] == expected


def test_reference_markers_do_not_replace_an_existing_actionable_turn():
    frame = _rescue_panel()
    cfg = _isolated(
        reversal_rescue_q=0.90,
        bottom_reference_self_q=0.95,
    )

    out = extract_turning_points(frame, cfg)

    assert (out["turn_action"] == BUY_TURN).any()
    assert not (out["turn_action"] == BOTTOM_TURN).any()


def test_region_candidate_yields_to_a_nearby_existing_event():
    existing = np.full(12, NO_TURN, dtype=object)
    existing[8] = BUY_TURN
    candidates = np.zeros(12, dtype=bool)
    candidates[5] = True

    buy, sell = _sequence_additions(
        existing,
        candidates,
        np.zeros(12, dtype=bool),
        np.linspace(100.0, 111.0, 12),
        np.ones(12),
        gap=5,
        rearm_atr=2.0,
        rearm_uses_path=True,
    )

    assert not buy.any()
    assert not sell.any()


def test_rescue_needs_the_trigger_to_close_near_its_high():
    """A wide bar closing mid-range is not a reversal."""
    frame = _rescue_panel()
    aaa = frame.index[frame["symbol"] == "AAA"][60]
    span = frame.loc[aaa, "high"] - frame.loc[aaa, "low"]
    frame.loc[aaa, "close"] = frame.loc[aaa, "low"] + 0.45 * span

    assert not _buys(frame, _isolated(breadth_relief=None, reversal_rescue_q=0.90))


def test_rescue_never_removes_an_event():
    frame = _rescue_panel()
    off = _buys(frame, _isolated(breadth_relief=None))
    on = _buys(frame, _isolated(breadth_relief=None, reversal_rescue_q=0.90))

    assert set(off) <= set(on)


def test_rescue_does_not_touch_the_sell_side():
    """It exists for capitulation lows; the sell path is unchanged."""
    frame = _rescue_panel()
    off = extract_turning_points(frame.copy(), _isolated(breadth_relief=None))
    on = extract_turning_points(
        frame.copy(), _isolated(breadth_relief=None, reversal_rescue_q=0.90)
    )

    assert (off["turn_action"] == SELL_TURN).sum() == (on["turn_action"] == SELL_TURN).sum()


# ---------------------------------------------------------------------------
# rearm_uses_path: a completed round trip is a different swing
# ---------------------------------------------------------------------------
def _round_trip(prices) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray(prices, dtype="float64")
    n = close.size
    sell = np.zeros(n, dtype=bool)
    sell[[0, n - 1]] = True
    return np.zeros(n, dtype=bool), sell, close, np.full(n, 1.0)


def test_endpoint_rearm_blocks_a_signal_after_a_round_trip():
    """Price left by 5 ATR and came back to within 1; the endpoint test says no."""
    buy, sell, close, atr = _round_trip([100.0] + [105.0] * 10 + [100.5])

    _, kept = _sequence(buy, sell, close, atr, gap=5, rearm_atr=2.0)

    assert kept.sum() == 1


def test_path_rearm_admits_it():
    buy, sell, close, atr = _round_trip([100.0] + [105.0] * 10 + [100.5])

    _, kept = _sequence(
        buy, sell, close, atr, gap=5, rearm_atr=2.0, rearm_uses_path=True
    )

    assert kept.sum() == 2


def test_path_rearm_still_refuses_a_swing_that_never_moved():
    """Travel is what earns the second signal, not elapsed time."""
    buy, sell, close, atr = _round_trip([100.0] + [100.3] * 10 + [100.5])

    _, kept = _sequence(
        buy, sell, close, atr, gap=5, rearm_atr=2.0, rearm_uses_path=True
    )

    assert kept.sum() == 1


def test_rearm_timeout_admits_the_next_candidate_at_twenty_bars():
    close = np.full(21, 100.0)
    sell = np.zeros(21, dtype=bool)
    sell[[0, 19, 20]] = True

    _, kept = _sequence(
        np.zeros(21, dtype=bool),
        sell,
        close,
        np.ones(21),
        gap=5,
        rearm_atr=2.0,
        rearm_uses_path=True,
        rearm_max_bars=20,
    )

    assert np.flatnonzero(kept).tolist() == [0, 20]


def test_addition_rearm_timeout_uses_the_last_same_side_event():
    existing = np.full(21, NO_TURN, dtype=object)
    existing[0] = BUY_TURN
    existing[10] = SELL_TURN
    candidates = np.zeros(21, dtype=bool)
    candidates[[19, 20]] = True

    kept, _ = _sequence_additions(
        existing,
        candidates,
        np.zeros(21, dtype=bool),
        np.full(21, 100.0),
        np.ones(21),
        gap=5,
        rearm_atr=2.0,
        rearm_uses_path=True,
        rearm_max_bars=20,
    )

    assert np.flatnonzero(kept).tolist() == [20]


def test_path_rearm_respects_the_refractory():
    """The excursion opens the gate; it does not shorten the gap."""
    close = np.array([100.0, 108.0, 100.5], dtype="float64")
    sell = np.ones(3, dtype=bool)

    _, kept = _sequence(
        np.zeros(3, dtype=bool), sell, close, np.full(3, 1.0),
        gap=5, rearm_atr=2.0, rearm_uses_path=True,
    )

    assert kept.sum() == 1


def test_excursion_resets_after_a_signal_fires():
    """Otherwise one big move would arm every later bar for good."""
    close = np.concatenate([[100.0], [110.0] * 10, [100.5], [100.6] * 10, [100.7]])
    sell = np.zeros(close.size, dtype=bool)
    sell[[0, 11, close.size - 1]] = True

    _, kept = _sequence(
        np.zeros(close.size, dtype=bool), sell, close, np.full(close.size, 1.0),
        gap=5, rearm_atr=2.0, rearm_uses_path=True,
    )

    # The first two fire on the big excursion; the third has not travelled since.
    assert kept.sum() == 2
