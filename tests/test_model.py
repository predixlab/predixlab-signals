"""Tests for the model heads, walk-forward splitting and score semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.model.pivot_model import (
    ActionThresholds,
    PivotModel,
    classify_action,
    pivot_score,
)
from predixlab_signals.model.walkforward import assert_no_leakage, make_folds, slice_fold


# --------------------------------------------------------------------------
# score semantics
# --------------------------------------------------------------------------
def test_pivot_score_sign_convention():
    """Low predicted swing position (a coming low) must score positive = buy."""
    at_low = pivot_score(np.array([0.0]), np.array([1.0]))
    at_high = pivot_score(np.array([1.0]), np.array([0.0]))
    neutral = pivot_score(np.array([0.5]), np.array([0.5]))
    assert at_low[0] == pytest.approx(1.0)
    assert at_high[0] == pytest.approx(-1.0)
    assert neutral[0] == pytest.approx(0.0)


def test_pivot_score_requires_agreement():
    """Heads that disagree must produce a weaker score than heads that agree."""
    agree = pivot_score(np.array([0.1]), np.array([0.9]))
    disagree = pivot_score(np.array([0.1]), np.array([0.1]))
    assert agree[0] > disagree[0]
    assert abs(disagree[0]) < abs(agree[0])


def test_classify_action_thresholds():
    """Labels follow the *percentile* of the score, not its magnitude.

    Sized so the quantile grid is unambiguous; with a handful of points the
    boundary rounding makes any exact expectation arbitrary.
    """
    th = ActionThresholds(
        strong_buy_q=0.90, buy_q=0.75, sell_q=0.25, strong_sell_q=0.10, min_amplitude=0.02
    )
    score = np.linspace(-1.0, 1.0, 1000)
    amp = np.full(score.size, 0.10)
    action = classify_action(score, amp, th)

    assert action[-1] == "STRONG_BUY" and action[0] == "STRONG_SELL"
    assert action[len(action) // 2] == "HOLD"
    counts = pd.Series(action).value_counts(normalize=True)
    assert counts["STRONG_BUY"] == pytest.approx(0.10, abs=0.01)
    assert counts["STRONG_SELL"] == pytest.approx(0.10, abs=0.01)
    assert counts["HOLD"] == pytest.approx(0.50, abs=0.02)

    # Magnitude alone must not decide: rescaling the scores changes nothing.
    assert list(classify_action(score * 7.5 + 3.0, amp, th)) == list(action)


def test_classify_action_suppresses_tiny_swings():
    """A correct direction call on a move too small to clear costs is not a trade."""
    th = ActionThresholds(min_amplitude=0.05)
    score = np.array([0.9, -0.9])
    amp = np.array([0.01, 0.01])
    assert list(classify_action(score, amp, th)) == ["HOLD", "HOLD"]


# --------------------------------------------------------------------------
# walk-forward splitting
# --------------------------------------------------------------------------
@pytest.fixture
def dates():
    return pd.Series(pd.bdate_range("2015-01-01", "2026-01-01"))


def test_folds_are_chronological_and_disjoint(dates):
    folds = make_folds(dates, horizon=10, n_folds=3, test_months=12)
    assert len(folds) == 3
    for a, b in zip(folds, folds[1:]):
        assert a.test_end < b.test_start
    for fold in folds:
        assert fold.train_end < fold.valid_start < fold.valid_end < fold.test_start


def test_folds_embargo_covers_the_label_horizon(dates):
    """Without a purge gap the last training labels would overlap the test block."""
    horizon = 10
    for fold in make_folds(dates, horizon=horizon, n_folds=3, test_months=12):
        assert_no_leakage(fold, horizon)
        assert (fold.valid_start - fold.train_end).days >= horizon
        assert (fold.test_start - fold.valid_end).days >= horizon


def test_assert_no_leakage_rejects_a_short_gap(dates):
    fold = make_folds(dates, horizon=10, n_folds=1, test_months=12)[0]
    with pytest.raises(ValueError, match="purge gap too small"):
        assert_no_leakage(fold, horizon=200)


def test_slice_fold_respects_boundaries(dates):
    frame = pd.DataFrame({"date": dates, "symbol": "AAA"})
    fold = make_folds(dates, horizon=10, n_folds=2, test_months=12)[0]
    train, valid, test = slice_fold(frame, fold)
    assert train["date"].max() <= fold.train_end
    assert test["date"].min() >= fold.test_start
    assert set(train.index).isdisjoint(test.index)


# --------------------------------------------------------------------------
# end-to-end fit / predict
# --------------------------------------------------------------------------
@pytest.fixture
def training_frame():
    rng = np.random.default_rng(11)
    n = 4000
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    return pd.DataFrame(
        {
            "symbol": rng.choice(["A", "B", "C"], n),
            "date": pd.bdate_range("2015-01-01", periods=n // 4).repeat(4)[:n],
            "f1": signal,
            "f2": noise,
            "label_scale": np.full(n, 0.03),
            # A clean monotone relationship so the heads have something to find.
            "y_swing_pos": np.clip(0.5 - 0.2 * signal + 0.05 * noise, -0.5, 1.5),
            "y_swing_pos_cs": np.clip(-0.2 * signal + 0.05 * noise, -1.0, 1.0),
            "y_amplitude": np.abs(1.5 + 0.3 * noise),
            "y_barrier": np.where(signal > 0, 1.0, -1.0),
            "y_barrier_cs": np.where(signal > 0, 1.0, -1.0),
        }
    )


def _model(**kw) -> PivotModel:
    params = {"n_estimators": 40, "num_leaves": 8, "min_child_samples": 50, "verbose": -1}
    return PivotModel(features=["f1", "f2"], horizon=10, params=params, **kw)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        _model().predict(pd.DataFrame({"f1": [0.0], "f2": [0.0], "label_scale": [0.01]}))


def test_fit_predict_recovers_the_relationship(training_frame):
    model = _model(market_neutral=False).fit(training_frame)
    out = model.predict(training_frame)
    # Higher signal -> lower swing position -> higher score.
    assert out["swing_pos_hat"].corr(training_frame["f1"]) < -0.5
    assert out["score"].corr(training_frame["f1"]) > 0.5


def test_predicted_excursions_are_consistent(training_frame):
    """mfe must be non-negative, mae non-positive, and both rebuild amplitude."""
    model = _model(market_neutral=False).fit(training_frame)
    out = model.predict(training_frame)
    assert (out["mfe_hat"] >= 0).all()
    assert (out["mae_hat"] <= 0).all()
    np.testing.assert_allclose(
        out["mfe_hat"] - out["mae_hat"], out["amplitude_hat"], rtol=1e-9
    )
    assert out["swing_pos_hat"].between(0.0, 1.0).all()


def test_market_neutral_head_is_recentred(training_frame):
    """A demeaned target predicts deviations, so 0.5 must be added back."""
    model = _model(market_neutral=True).fit(training_frame)
    out = model.predict(training_frame)
    assert 0.2 < out["swing_pos_hat"].mean() < 0.8


def test_calibrate_stores_a_reference_distribution(training_frame):
    """Calibration must persist a *distribution*, never absolute cut-offs."""
    model = _model(market_neutral=False).fit(training_frame)
    model.calibrate(model.predict(training_frame)["score"].to_numpy())
    th = model.thresholds
    assert th.strong_sell_q < th.sell_q < th.buy_q < th.strong_buy_q
    assert th.reference is not None and th.reference.size > 100
    assert np.all(np.diff(th.reference) >= 0)


def test_save_and_load_roundtrip(training_frame, tmp_path):
    model = _model(market_neutral=False).fit(training_frame)
    path = model.save(tmp_path / "m.pkl")
    assert path.with_suffix(".meta.json").exists()
    loaded = PivotModel.load(path, verify_integrity=True)
    pd.testing.assert_frame_equal(loaded.predict(training_frame), model.predict(training_frame))


def test_verified_load_rejects_tampered_pickle_before_deserializing(
    training_frame, tmp_path
):
    from predixlab_signals.model.artifacts import ArtifactIntegrityError

    model = _model(market_neutral=False).fit(training_frame)
    path = model.save(tmp_path / "m.pkl")
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ArtifactIntegrityError, match="integrity check failed"):
        PivotModel.load(path, verify_integrity=True)


def test_bundle_keeps_authoritative_manifest_without_sidecar(
    training_frame, tmp_path
):
    model = _model(market_neutral=False).fit(training_frame)
    path = model.save(tmp_path / "m.pkl")
    path.with_suffix(".meta.json").unlink()

    loaded = PivotModel.load(path, verify_integrity=True)

    pd.testing.assert_frame_equal(
        loaded.predict(training_frame), model.predict(training_frame)
    )


def test_feature_importance_covers_all_heads(training_frame):
    model = _model(market_neutral=False).fit(training_frame)
    table = model.feature_importance(top=2)
    assert set(table.columns) == {"swing_pos", "amplitude", "updown", "mean"}


def test_head_estimators_apply_per_head(training_frame):
    """Each head must honour its own tree count, not a shared one."""
    frame = training_frame.copy()
    frame["y_pivot_low_hold"] = (frame["f1"] > 1.0).astype(float)
    frame["y_pivot_high_hold"] = (frame["f1"] < -1.0).astype(float)
    frame["y_pivot_low_return"] = np.where(frame["f1"] > 1.0, frame["f1"], np.nan)
    frame["y_pivot_high_return"] = np.where(frame["f1"] < -1.0, -frame["f1"], np.nan)
    model = _model(market_neutral=False)
    model.head_estimators = {
        "swing_pos": 5,
        "amplitude": 40,
        "updown": 12,
        "pivot_low": 7,
        "pivot_high": 9,
        "pivot_low_return": 11,
        "pivot_high_return": 13,
    }
    model.fit(frame)
    assert model.pos_model.booster_.num_trees() == 5
    assert model.amp_model.booster_.num_trees() == 40
    assert model.updown_model.booster_.num_trees() == 12
    assert model.pivot_low_model.booster_.num_trees() == 7
    assert model.pivot_high_model.booster_.num_trees() == 9
    assert model.pivot_low_return_model.booster_.num_trees() == 11
    assert model.pivot_high_return_model.booster_.num_trees() == 13


def test_head_estimators_fall_back_to_shared_params(training_frame):
    model = _model(market_neutral=False)
    model.head_estimators = {"swing_pos": 7}
    model.fit(training_frame)
    assert model.pos_model.booster_.num_trees() == 7
    assert model.amp_model.booster_.num_trees() == model.params["n_estimators"]


def test_walk_forward_head_estimators_are_median_across_folds():
    """One atypical validation window must not dictate the production model."""
    from predixlab_signals.model.trainer import WalkForwardResult

    class _Stub:
        def __init__(self, best):
            self.trained_on = {"best_iterations": best}

    result = WalkForwardResult(
        predictions=pd.DataFrame(),
        folds=[],
        models=[
            _Stub(
                {
                    "swing_pos": 5,
                    "amplitude": 400,
                    "updown": 50,
                    "pivot_low": 25,
                    "pivot_high": 70,
                    "pivot_low_return": 40,
                    "pivot_high_return": 80,
                }
            ),
            _Stub(
                {
                    "swing_pos": 120,
                    "amplitude": 380,
                    "updown": 60,
                    "pivot_low": 35,
                    "pivot_high": 50,
                    "pivot_low_return": 60,
                    "pivot_high_return": 60,
                }
            ),
            _Stub(
                {
                    "swing_pos": 90,
                    "amplitude": 390,
                    "updown": 55,
                    "pivot_low": 30,
                    "pivot_high": 60,
                    "pivot_low_return": 50,
                    "pivot_high_return": 70,
                }
            ),
        ],
        elapsed=0.0,
    )
    assert result.head_estimators() == {
        "swing_pos": 90,
        "amplitude": 390,
        "updown": 55,
        "pivot_low": 30,
        "pivot_high": 60,
        "pivot_low_return": 50,
        "pivot_high_return": 70,
    }


def test_final_model_calibrates_with_its_own_head_counts(training_frame):
    """Thresholds must come from a model shaped exactly like the deployed one.

    Score spread shrinks as tree count falls, so cut-points borrowed from a
    differently-sized model push the whole universe into HOLD.
    """
    from predixlab_signals.model.pivot_model import classify_action
    from predixlab_signals.model.trainer import train_final_model

    frame = training_frame.copy()
    frame["date"] = pd.bdate_range("2015-01-01", periods=len(frame))
    frame["y_pivot_low_hold"] = (frame["f1"] > 1.0).astype(float)
    frame["y_pivot_high_hold"] = (frame["f1"] < -1.0).astype(float)
    frame["y_pivot_low_return"] = np.where(frame["f1"] > 1.0, frame["f1"], np.nan)
    frame["y_pivot_high_return"] = np.where(frame["f1"] < -1.0, -frame["f1"], np.nan)

    model = train_final_model(
        frame,
        horizon=10,
        valid_months=6,
        params={"n_estimators": 200, "num_leaves": 8, "min_child_samples": 50, "verbose": -1},
        features=["f1", "f2"],
        market_neutral=False,
        head_estimators={
            "swing_pos": 5,
            "amplitude": 30,
            "updown": 5,
            "pivot_low": 7,
            "pivot_high": 9,
            "pivot_low_return": 11,
            "pivot_high_return": 13,
        },
    )
    assert model.head_estimators == {
        "swing_pos": 5,
        "amplitude": 30,
        "updown": 5,
        "pivot_low": 7,
        "pivot_high": 9,
        "pivot_low_return": 11,
        "pivot_high_return": 13,
    }
    assert model.pos_model.booster_.num_trees() == 5
    assert model.updown_model.booster_.num_trees() == 5
    assert model.pivot_low_model.booster_.num_trees() == 7
    assert model.pivot_high_model.booster_.num_trees() == 9
    assert model.pivot_low_return_model.booster_.num_trees() == 11
    assert model.pivot_high_return_model.booster_.num_trees() == 13

    signal = model.predict(frame)
    actions = classify_action(
        signal["score"].to_numpy(), signal["exp_amplitude_pct"].to_numpy(), model.thresholds
    )
    # Cut-points sit at validation quantiles, so most names must not be HOLD.
    hold_share = (actions == "HOLD").mean()
    assert hold_share < 0.75, f"thresholds mis-scaled: {hold_share:.2%} HOLD"
