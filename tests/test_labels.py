"""Tests for the forward-looking label block.

These are the highest-value tests in the package: a one-bar misalignment in a
forward window is invisible in every summary statistic yet inflates every
backtest, so the excursions are checked against brute-force recomputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.labels.swing import (
    _forward_extremes,
    _triple_barrier,
    build_labels,
    zigzag_pivots,
)


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=400)
    frames = []
    for symbol in ("AAA", "BBB"):
        steps = rng.normal(0.0, 0.02, len(dates))
        close = 100.0 * np.exp(np.cumsum(steps))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": close * (1 + rng.normal(0, 0.002, len(dates))),
                    "high": close * (1 + np.abs(rng.normal(0, 0.01, len(dates)))),
                    "low": close * (1 - np.abs(rng.normal(0, 0.01, len(dates)))),
                    "close": close,
                    "volume": rng.integers(1e5, 1e6, len(dates)).astype(float),
                    "price": close,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_forward_extremes_look_strictly_ahead():
    """Bar t's window must cover t+1..t+H — never t itself."""
    high = np.arange(10, dtype="float64")
    low = -high
    fwd_high, fwd_low = _forward_extremes(high, low, 3)
    # From bar 0 the next three highs are 1, 2, 3 -> max 3.
    assert fwd_high[0] == 3.0
    assert fwd_low[0] == -3.0
    assert np.isnan(fwd_high[-3:]).all()


def test_forward_extremes_match_bruteforce():
    rng = np.random.default_rng(1)
    high = rng.normal(100, 5, 200)
    low = high - np.abs(rng.normal(2, 1, 200))
    horizon = 7
    fwd_high, fwd_low = _forward_extremes(high, low, horizon)
    for i in range(200 - horizon):
        assert fwd_high[i] == pytest.approx(high[i + 1 : i + 1 + horizon].max())
        assert fwd_low[i] == pytest.approx(low[i + 1 : i + 1 + horizon].min())


def test_triple_barrier_detects_first_touch():
    close = np.full(10, 100.0)
    atr = np.full(10, 1.0)
    high = np.full(10, 100.0)
    low = np.full(10, 100.0)
    high[3] = 110.0  # upper barrier (100 + 1.5 * 1) breached at bar 3
    label, bars = _triple_barrier(high, low, close, atr, horizon=5, width=1.5)
    assert label[0] == 1.0
    assert bars[0] == 3.0


def test_triple_barrier_same_bar_touch_is_neutral():
    """Without intraday data a bar touching both barriers is genuinely ambiguous."""
    close = np.full(8, 100.0)
    atr = np.full(8, 1.0)
    high = np.full(8, 100.0)
    low = np.full(8, 100.0)
    high[2] = 200.0
    low[2] = 1.0
    label, _ = _triple_barrier(high, low, close, atr, horizon=4, width=1.5)
    assert label[0] == 0.0


def test_build_labels_matches_manual_computation(panel):
    labels = build_labels(panel, horizon=10)
    raw = panel[panel["symbol"] == "AAA"].reset_index(drop=True)
    got = labels[labels["symbol"] == "AAA"].reset_index(drop=True)
    for i in (0, 50, 123, 300):
        expected_mfe = raw["high"][i + 1 : i + 11].max() / raw["close"][i] - 1.0
        expected_mae = raw["low"][i + 1 : i + 11].min() / raw["close"][i] - 1.0
        assert got["fwd_mfe"][i] == pytest.approx(expected_mfe)
        assert got["fwd_mae"][i] == pytest.approx(expected_mae)


def test_swing_position_semantics(panel):
    """0 means the close is the coming low; 1 means it is the coming high."""
    labels = build_labels(panel, horizon=10).dropna(subset=["y_swing_pos"])
    low_bars = labels[labels["y_swing_pos"] < 0.1]
    high_bars = labels[labels["y_swing_pos"] > 0.9]
    assert (low_bars["fwd_mfe"] > 0).mean() > 0.9  # room above -> it was a low
    assert (high_bars["fwd_mae"] < 0).mean() > 0.9  # room below -> it was a high


def test_last_bars_have_no_label(panel):
    """The final H bars cannot be labelled; they are the live prediction rows."""
    labels = build_labels(panel, horizon=10)
    tail = labels[labels["symbol"] == "AAA"].tail(10)
    assert tail["y_swing_pos"].isna().all()


def test_zigzag_alternates_highs_and_lows(panel):
    frame = panel[panel["symbol"] == "AAA"].set_index("date")
    atr = (frame["high"] - frame["low"]).rolling(14).mean()
    pivots = zigzag_pivots(frame["high"], frame["low"], atr, k=2.0)
    marked = pivots[pivots != 0.0]
    assert len(marked) > 2
    # Consecutive pivots must alternate; two highs in a row is a broken zigzag.
    assert (marked.to_numpy()[1:] * marked.to_numpy()[:-1] < 0).all()
