"""Tests for feature engineering.

The contract every feature must satisfy is scale invariance: multiply a price
series by any constant and the feature matrix must not move. That is what lets
a single model serve 700 symbols whose prices span three orders of magnitude,
so it is asserted directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.features.core import (
    MARKET_PREFIX,
    NON_FEATURE_COLUMNS,
    build_features,
    compute_market_features,
    compute_symbol_features,
    feature_columns,
)


def _ohlcv(n: int = 400, scale: float = 1.0, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = scale * 50.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.008, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.008, n))),
            "close": close,
            "volume": rng.integers(1e5, 1e6, n).astype("float64"),
        },
        index=dates,
    )


def test_features_are_scale_invariant():
    """Doubling every price must leave every feature unchanged."""
    base = compute_symbol_features(_ohlcv(scale=1.0))
    scaled = compute_symbol_features(_ohlcv(scale=1000.0))
    # ``amihud`` is a log of price impact, so a pure price rescale shifts it by a
    # constant; every other column must match exactly.
    cols = [c for c in base.columns if not c.startswith("amihud")]
    pd.testing.assert_frame_equal(base[cols], scaled[cols], rtol=1e-6, atol=1e-8)


def test_bounded_features_stay_in_range():
    feats = compute_symbol_features(_ohlcv()).dropna()
    for col, lo, hi in [
        ("rsi_14", 0.0, 1.0),
        ("rsi_6", 0.0, 1.0),
        ("stoch_k", 0.0, 1.0),
        ("mfi_14", 0.0, 1.0),
        ("up_frac_10", 0.0, 1.0),
        ("close_loc", 0.0, 1.0),
        ("pctile_60", 0.0, 1.0),
    ]:
        assert feats[col].between(lo, hi).all(), col


def test_range_position_semantics():
    """``dpos`` is 1.0 at the window high and 0.0 at the window low."""
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.arange(1.0, n + 1.0), index=dates)  # strictly rising
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=dates
    )
    feats = compute_symbol_features(frame)
    assert feats["dpos_20"].dropna().iloc[-1] == pytest.approx(1.0)


def test_self_normalised_columns_exist_and_are_clipped():
    feats = compute_symbol_features(_ohlcv(n=600))
    sn = [c for c in feats.columns if c.endswith("_sn")]
    assert len(sn) > 20
    values = feats[sn].to_numpy()
    finite = values[np.isfinite(values)]
    assert finite.min() >= -6.0 and finite.max() <= 6.0


def test_self_normalised_features_use_no_future_data():
    """Truncating the series must not change any earlier self-normalised value."""
    full = compute_symbol_features(_ohlcv(n=600))
    truncated = compute_symbol_features(_ohlcv(n=600).iloc[:500])
    sn = [c for c in full.columns if c.endswith("_sn")]
    pd.testing.assert_frame_equal(
        full[sn].iloc[:500], truncated[sn], rtol=1e-9, atol=1e-10
    )


def test_feature_columns_excludes_leaky_and_identity_columns():
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": [pd.Timestamp("2020-01-01")],
            "close": [1.0],
            "price": [1.0],
            "high": [1.0],
            "low": [1.0],
            "open_next": [1.0],  # tomorrow's price
            "pivot": [0.0],  # derived with lookahead
            "y_swing_pos": [0.5],
            "fwd_ret": [0.01],
            "exp_gain_pct": [0.02],
            "swing_pos_hat": [0.4],
            "label_scale": [0.02],
            "mkt_ret_5": [0.01],
            "rsi_14": [0.5],
        }
    )
    cols = feature_columns(frame)
    assert cols == ["rsi_14"]
    assert "mkt_ret_5" in feature_columns(frame, include_market=True)
    for leaky in ("open_next", "pivot", "y_swing_pos", "fwd_ret", "swing_pos_hat"):
        assert leaky not in feature_columns(frame, include_market=True)


def test_market_features_are_prefixed():
    bench = _ohlcv(n=400).reset_index(names="date")
    market = compute_market_features(bench)
    assert all(c.startswith(MARKET_PREFIX) for c in market.columns)


def test_build_features_returns_one_row_per_symbol_date():
    panel = []
    for symbol in ("AAA", "BBB"):
        frame = _ohlcv(n=300, seed=hash(symbol) % 100).reset_index(names="date")
        frame.insert(0, "symbol", symbol)
        frame["price"] = frame["close"]
        panel.append(frame)
    panel = pd.concat(panel, ignore_index=True)
    bench = _ohlcv(n=300).reset_index(names="date")

    feats = build_features(panel, bench)
    assert len(feats) == len(panel)
    assert not feats.duplicated(["symbol", "date"]).any()
    assert NON_FEATURE_COLUMNS.issuperset({"symbol", "date"})
