"""Tests for evaluation metrics and signal scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.evaluation.metrics import (
    _dilate,
    action_buckets,
    cross_sectional_ic,
    probability_report,
    time_series_ic,
    turning_point_report,
)
from predixlab_signals.model.pivot_model import ActionThresholds, PivotModel
from predixlab_signals.signals.scoring import (
    build_scores,
    confidence_in_cross_section,
    enrich_scores,
    format_scores,
)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def _preds(noise: float, seed: int = 0, n_dates: int = 120, n_symbols: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    rows = []
    for date in dates:
        target = rng.normal(size=n_symbols)
        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "symbol": [f"S{i:02d}" for i in range(n_symbols)],
                    "score": target + rng.normal(0, noise, n_symbols),
                    "y_edge": target,
                    "fwd_ret": target * 0.01,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_cross_sectional_ic_detects_a_perfect_signal():
    ic = cross_sectional_ic(_preds(noise=0.0), "score", "y_edge")
    assert ic["rank_ic_mean"] == pytest.approx(1.0, abs=1e-6)
    assert ic["rank_ic_positive_rate"] == 1.0


def test_cross_sectional_ic_is_zero_for_noise():
    rng = np.random.default_rng(1)
    frame = _preds(noise=0.0)
    frame["score"] = rng.normal(size=len(frame))
    ic = cross_sectional_ic(frame, "score", "y_edge")
    assert abs(ic["rank_ic_mean"]) < 0.05


def test_ic_degrades_monotonically_with_noise():
    strong = cross_sectional_ic(_preds(noise=0.2), "score", "y_edge")["rank_ic_mean"]
    weak = cross_sectional_ic(_preds(noise=3.0), "score", "y_edge")["rank_ic_mean"]
    assert strong > weak > 0


def test_time_series_ic_is_computed_per_symbol():
    out = time_series_ic(_preds(noise=0.3), "score", "y_edge", min_obs=50)
    assert out["ts_ic_symbols"] == 30
    assert out["ts_ic_mean"] > 0.5


def test_action_buckets_are_ordered_and_complete():
    rng = np.random.default_rng(3)
    n = 2000
    frame = pd.DataFrame(
        {
            "action": rng.choice(["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"], n),
            "fwd_ret": rng.normal(0, 0.05, n),
            "fwd_mfe": np.abs(rng.normal(0.04, 0.01, n)),
            "fwd_mae": -np.abs(rng.normal(0.04, 0.01, n)),
            "y_swing_pos": rng.uniform(0, 1, n),
            "exp_edge_pct": rng.normal(0, 0.01, n),
        }
    )
    table = action_buckets(frame)
    assert list(table.index) == ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
    assert table["share"].sum() == pytest.approx(1.0)


def test_dilate_widens_a_mask_both_ways():
    mask = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(
        _dilate(mask, 1), np.array([False, True, True, True, False])
    )
    assert _dilate(mask, 0).tolist() == mask.tolist()


def test_turning_point_lift_rewards_a_perfect_detector():
    """Signals placed exactly on the pivots must score well above chance."""
    n = 200
    pivot = np.zeros(n)
    pivot[::20] = -1.0  # a swing low every 20 bars
    action = np.where(pivot == -1.0, "BUY", "HOLD")
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": pd.bdate_range("2022-01-01", periods=n),
            "pivot": pivot,
            "action": action,
        }
    )
    report = turning_point_report(frame, tolerance=1)
    assert report["buy_precision"] == pytest.approx(1.0)
    assert report["buy_recall"] == pytest.approx(1.0)
    assert report["buy_lift"] > 1.0


def test_probability_report_recovers_auc():
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.choice([-1.0, 1.0], n)
    frame = pd.DataFrame({"p_up": np.where(y > 0, rng.uniform(0.5, 1, n), rng.uniform(0, 0.5, n)),
                          "y_barrier": y})
    report = probability_report(frame)
    assert report["auc"] > 0.95
    assert 0.0 <= report["brier"] <= 1.0


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(9)
    n = 3000
    signal = rng.normal(size=n)
    train = pd.DataFrame(
        {
            "symbol": rng.choice(["A", "B"], n),
            "date": pd.bdate_range("2015-01-01", periods=n // 2).repeat(2)[:n],
            "f1": signal,
            "label_scale": np.full(n, 0.03),
            "y_swing_pos": np.clip(0.5 - 0.25 * signal, -0.5, 1.5),
            "y_swing_pos_cs": np.clip(-0.25 * signal, -1.0, 1.0),
            "y_amplitude": np.full(n, 2.0),
            "y_barrier": np.where(signal > 0, 1.0, -1.0),
            "y_barrier_cs": np.where(signal > 0, 1.0, -1.0),
        }
    )
    model = PivotModel(
        features=["f1"],
        horizon=10,
        params={"n_estimators": 40, "num_leaves": 8, "min_child_samples": 50, "verbose": -1},
        market_neutral=False,
    ).fit(train)
    model.calibrate(model.predict(train)["score"].to_numpy())
    return model


@pytest.fixture
def advice_frame():
    return pd.DataFrame(
        {
            "symbol": ["UP", "DOWN", "FLAT"],
            "date": [pd.Timestamp("2026-01-05")] * 3,
            "close": [50.0, 50.0, 50.0],
            "price": [100.0, 100.0, 100.0],
            "f1": [3.0, -3.0, 0.0],
            "label_scale": [0.03, 0.03, 0.03],
            "atr_pct_label": [0.02, 0.02, 0.02],
            "dpos_60": [0.05, 0.95, 0.5],
            "rsi_14": [0.2, 0.85, 0.5],
            "dd_from_high_250": [-0.30, -0.01, -0.05],
            "atr_pct_sn": [1.5, 0.0, 0.0],
            "vol_rel": [0.5, 0.0, 0.0],
            "streak": [-1.0, 1.0, 0.0],
        }
    )


def test_scores_are_generated_for_every_symbol(fitted_model, advice_frame):
    advice = build_scores(fitted_model, advice_frame)
    assert len(advice) == 3
    assert set(advice["symbol"]) == {"UP", "DOWN", "FLAT"}
    assert advice["score"].is_monotonic_decreasing


def test_min_score_filter(fitted_model, advice_frame):
    everything = build_scores(fitted_model, advice_frame)
    filtered = build_scores(fitted_model, advice_frame, min_score=10.0)
    assert len(everything) == 3
    assert filtered.empty


def test_format_scores_handles_empty():
    assert format_scores(pd.DataFrame()) == "No signals generated."


# --------------------------------------------------------------------------
# confidence is a universe percentile, not an among-the-winners percentile
# --------------------------------------------------------------------------
def _universe(date: str = "2024-03-01", n: int = 100) -> pd.DataFrame:
    """A full scored cross-section with evenly spread conviction."""
    return pd.DataFrame(
        {
            "symbol": [f"S{i:03d}" for i in range(n)],
            "date": pd.Timestamp(date),
            "score": np.linspace(-0.05, 0.05, n),
        }
    )


def test_confidence_ranks_against_the_universe_not_the_picks():
    """Three events must not become 0.33 / 0.67 / 1.00."""
    universe = _universe()
    picks = universe.iloc[[50, 60, 99]][["symbol", "date"]].reset_index(drop=True)

    conf = confidence_in_cross_section(universe, picks)

    # S050 sits at the flat middle of a symmetric ramp, so |score| is near zero.
    assert conf[0] < 0.10
    assert conf[2] > 0.98  # the two ramp ends tie on |score|, so 0.995 not 1.0
    assert not set(np.round(conf, 2)).issubset({0.33, 0.67, 1.0})


def test_confidence_is_computed_per_day():
    """Two sessions rank independently; a quiet day does not borrow a loud one."""
    loud = _universe("2024-03-01")
    quiet = _universe("2024-03-04")
    quiet["score"] = quiet["score"] / 100.0
    universe = pd.concat([loud, quiet], ignore_index=True)

    picks = pd.DataFrame(
        {
            "symbol": ["S099", "S099"],
            "date": [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-04")],
        }
    )
    conf = confidence_in_cross_section(universe, picks)
    assert conf[0] == pytest.approx(conf[1])


def test_enrich_scores_uses_the_cross_section_when_given():
    """The parameter is what makes the live and backfill paths agree."""
    universe = _universe()
    events = universe.iloc[[50, 60, 99]].reset_index(drop=True)
    events = events.assign(
        price=100.0,
        exp_gain_pct=0.05,
        exp_loss_pct=-0.03,
        swing_pos_hat=0.2,
        action="BUY",
    )
    without = enrich_scores(events)
    with_universe = enrich_scores(events, cross_section=universe)

    assert sorted(without["confidence"]) == [0.333, 0.667, 1.0]
    assert with_universe["confidence"].iloc[0] < 0.10
