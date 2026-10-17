"""Tests for predicting whether today's price extreme holds in the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.evaluation.metrics import pivot_hold_report
from predixlab_signals.labels.swing import (
    build_labels,
    future_extreme_holds,
)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def test_future_hold_ignores_prior_price_action():
    low = pd.Series([90.0, 100.0, 101.0, 102.0, 103.0])
    high = pd.Series([110.0, 100.0, 99.0, 98.0, 97.0])
    low_hold, high_hold = future_extreme_holds(high, low, window=3)

    assert low_hold[1] == 1.0
    assert high_hold[1] == 1.0


def test_future_hold_lookahead_is_bounded_by_the_window():
    n = 60
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    low_hold, high_hold = future_extreme_holds(close + 1, close - 1, window=5)
    assert low_hold.iloc[-5:].isna().all()
    assert high_hold.iloc[-5:].isna().all()
    assert low_hold.iloc[:-5].notna().all()


def test_future_hold_uses_no_data_beyond_the_window():
    n = 80
    rng = np.random.default_rng(4)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    full_low, _ = future_extreme_holds(close + 1, close - 1, window=5)
    cut_low, _ = future_extreme_holds(
        (close + 1).iloc[:60], (close - 1).iloc[:60], window=5
    )
    pd.testing.assert_series_equal(full_low.iloc[:55], cut_low.iloc[:55], check_names=False)


def test_build_labels_exposes_future_hold_and_return_columns():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
    panel = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
            "price": close,
        }
    )
    labels = build_labels(panel, horizon=10)
    for col in (
        "y_pivot_low_hold",
        "y_pivot_high_hold",
        "y_pivot_low_return",
        "y_pivot_high_return",
    ):
        assert col in labels.columns
    assert {
        "y_pivot_low",
        "y_pivot_high",
        "at_recent_low",
        "at_recent_high",
    }.isdisjoint(labels.columns)
    assert labels["y_pivot_low_hold"].dropna().isin([0.0, 1.0]).all()


def test_pivot_return_is_directional_and_conditional_on_hold():
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=300)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
    panel = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e6,
            "price": close,
        }
    )
    labels = build_labels(panel, horizon=10)
    low = labels["y_pivot_low_hold"] == 1.0
    high = labels["y_pivot_high_hold"] == 1.0
    directional = (labels["fwd_ret"] / labels["label_scale"]).clip(-3.0, 3.0)

    pd.testing.assert_series_equal(
        labels.loc[low, "y_pivot_low_return"],
        directional[low],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        labels.loc[high, "y_pivot_high_return"],
        -directional[high],
        check_names=False,
    )
    assert labels.loc[~low, "y_pivot_low_return"].isna().all()
    assert labels.loc[~high, "y_pivot_high_return"].isna().all()


# --------------------------------------------------------------------------
# hold metric
# --------------------------------------------------------------------------
def _detection_frame(auc_signal: float, n: int = 20_000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    will_hold = rng.random(n) < 0.27
    prob = np.clip(0.2 + auc_signal * will_hold + rng.normal(0, 0.15, n), 0, 1)
    return pd.DataFrame(
        {
            "p_pivot_low": prob,
            "y_pivot_low_hold": will_hold.astype(float),
        }
    )


def test_hold_report_is_at_chance_without_signal():
    report = pivot_hold_report(_detection_frame(0.0))
    assert 0.45 < report["auc"] < 0.55


def test_hold_report_detects_real_forward_signal():
    weak = pivot_hold_report(_detection_frame(0.05))
    strong = pivot_hold_report(_detection_frame(0.40))
    assert strong["auc"] > weak["auc"]
    assert strong["auc"] > 0.7


def test_hold_report_buckets_are_ordered():
    report = pivot_hold_report(_detection_frame(0.40))
    lifts = [b["lift"] for b in report["buckets"]]
    assert lifts == sorted(lifts, reverse=True)  # tighter cut -> higher precision
    assert all(b["lift"] > 1.0 for b in report["buckets"])


def test_hold_report_needs_its_columns():
    assert pivot_hold_report(pd.DataFrame({"a": [1]})) == {}


# --------------------------------------------------------------------------
# model heads
# --------------------------------------------------------------------------
@pytest.fixture
def pivot_training_frame():
    rng = np.random.default_rng(8)
    n = 4000
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "symbol": rng.choice(["A", "B"], n),
            "date": pd.bdate_range("2015-01-01", periods=n // 2).repeat(2)[:n],
            "f1": signal,
            "label_scale": np.full(n, 0.03),
            "y_swing_pos": np.clip(0.5 - 0.2 * signal, -0.5, 1.5),
            "y_swing_pos_cs": np.clip(-0.2 * signal, -1.0, 1.0),
            "y_amplitude": np.full(n, 2.0),
            "y_barrier": np.where(signal > 0, 1.0, -1.0),
            "y_barrier_cs": np.where(signal > 0, 1.0, -1.0),
            "y_pivot_low_hold": (signal > 1.0).astype(float),
            "y_pivot_high_hold": (signal < -1.0).astype(float),
            "y_pivot_low_return": np.where(signal > 1.0, signal, np.nan),
            "y_pivot_high_return": np.where(signal < -1.0, -signal, np.nan),
        }
    )


def _pivot_model():
    from predixlab_signals.model.pivot_model import PivotModel

    return PivotModel(
        features=["f1"],
        horizon=10,
        params={"n_estimators": 40, "num_leaves": 8, "min_child_samples": 50, "verbose": -1},
        market_neutral=False,
    )


def test_pivot_heads_are_fitted_and_scored(pivot_training_frame):
    model = _pivot_model().fit(pivot_training_frame)
    assert model.pivot_low_model is not None and model.pivot_high_model is not None
    out = model.predict(pivot_training_frame)
    assert {"p_pivot_low", "p_pivot_high"} <= set(out.columns)
    assert out["p_pivot_low"].between(0, 1).all()
    # The head must recover the planted relationship.
    assert out["p_pivot_low"].corr(pivot_training_frame["f1"]) > 0.5
    assert out["p_pivot_high"].corr(pivot_training_frame["f1"]) < -0.5


def test_pivot_heads_use_future_hold_targets(pivot_training_frame):
    frame = pivot_training_frame.copy()
    frame["y_pivot_low_hold"] = (frame["f1"] < -0.5).astype(float)
    frame["y_pivot_high_hold"] = (frame["f1"] > 0.5).astype(float)

    model = _pivot_model().fit(frame)
    out = model.predict(frame)

    assert out["p_pivot_low"].corr(frame["f1"]) < -0.5
    assert out["p_pivot_high"].corr(frame["f1"]) > 0.5
    assert model.trained_on["pivot_targets"] == {
        "pivot_low": "y_pivot_low_hold",
        "pivot_high": "y_pivot_high_hold",
    }
    assert model.trained_on["pivot_return_targets"] == {
        "pivot_low_return": "y_pivot_low_return",
        "pivot_high_return": "y_pivot_high_return",
    }


def test_pivot_score_weights_hold_probability_by_expected_return(
    pivot_training_frame,
):
    model = _pivot_model().fit(pivot_training_frame)
    out = model.predict(pivot_training_frame)

    assert {
        "pivot_low_return_hat",
        "pivot_high_return_hat",
        "pivot_low_score",
        "pivot_high_score",
    } <= set(out.columns)
    np.testing.assert_allclose(
        out["pivot_low_score"],
        out["p_pivot_low"] * out["pivot_low_return_hat"].clip(lower=0.0),
    )
    np.testing.assert_allclose(
        out["pivot_high_score"],
        out["p_pivot_high"] * out["pivot_high_return_hat"].clip(lower=0.0),
    )


def test_model_without_pivot_labels_still_fits(pivot_training_frame):
    frame = pivot_training_frame.drop(
        columns=[
            "y_pivot_low_hold",
            "y_pivot_high_hold",
            "y_pivot_low_return",
            "y_pivot_high_return",
        ]
    )
    model = _pivot_model().fit(frame)
    assert model.pivot_low_model is None
    out = model.predict(frame)
    assert out["p_pivot_low"].isna().all()
    assert out["score"].notna().all()
