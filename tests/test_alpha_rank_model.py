from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.model.alpha_rank_model import (
    ALPHA_RANK_BASE_FEATURES,
    FINAL_ALPHA_SCORE_WEIGHTS,
    AlphaRankEnsemble,
    AlphaRankModel,
    _date_equal_weights,
    alpha_rank_score,
    prepare_alpha_rank_training_data,
)


def _data(periods: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for number, symbol in enumerate(("AAA", "BBB", "CCC"), start=1):
        opens = 100.0 * np.cumprod(np.full(periods, 1.0 + number / 1000.0))
        for index, date in enumerate(dates):
            row = {
                "date": date,
                "symbol": symbol,
                "open_next": opens[index],
                "close": opens[index],
            }
            row.update(
                {
                    feature: float(number + index / 100.0)
                    for feature in ALPHA_RANK_BASE_FEATURES
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def test_alpha_rank_targets_use_future_closes_and_daily_ranks():
    prepared = prepare_alpha_rank_training_data(
        _data(), horizon=5, min_cross_section=3
    )
    resolved = prepared.dropna(subset=["alpha_fwd_return"])
    assert resolved["alpha_return_rank"].between(0.0, 1.0).all()
    assert set(resolved.groupby("date")["alpha_top20"].sum().unique()) == {1.0}
    assert prepared.groupby("date")["rank_ret_20"].max().le(1.0).all()
    assert prepared.groupby("date")["rank_ret_20"].min().gt(0.0).all()


def test_alpha_rank_terminal_targets_remain_missing():
    prepared = prepare_alpha_rank_training_data(
        _data(), horizon=5, min_cross_section=3
    )
    tail = prepared.groupby("symbol", sort=False).tail(6)
    assert tail["alpha_fwd_return"].isna().all()


def test_alpha_rank_target_requires_the_exact_calendar_rows():
    data = _data(periods=12)
    missing_date = sorted(data["date"].unique())[1]
    data = data[
        ~((data["symbol"] == "AAA") & (data["date"] == missing_date))
    ]
    prepared = prepare_alpha_rank_training_data(
        data, horizon=3, min_cross_section=3
    )
    first = prepared[
        (prepared["symbol"] == "AAA")
        & (prepared["date"] == prepared["date"].min())
    ].iloc[0]
    assert pd.isna(first["alpha_fwd_return"])


def test_alpha_rank_model_fits_without_pivot_inputs():
    prepared = prepare_alpha_rank_training_data(
        _data(periods=40), horizon=5, min_cross_section=3
    )
    model = AlphaRankModel(
        horizon=5,
        params={
            "n_estimators": 5,
            "num_leaves": 4,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 3,
        },
    ).fit(prepared)
    predictions = model.predict(prepared)
    assert {"return_hat", "rank_hat", "p_top", "alpha_score"} <= set(predictions)
    assert predictions["alpha_score"].between(0.0, 1.0).all()
    assert not any("pivot" in feature for feature in model.features)
    assert model.top_model.class_weight is None
    assert model.trained_on["date_weighting"] == "equal"


def test_alpha_rank_score_uses_rank_head_as_primary_signal():
    predictions = pd.DataFrame(
        {
            "return_hat": [1.0, 2.0, 3.0],
            "rank_hat": [3.0, 2.0, 1.0],
            "p_top": [1.0, 2.0, 3.0],
        }
    )
    score = alpha_rank_score(
        predictions,
        np.zeros(3),
        FINAL_ALPHA_SCORE_WEIGHTS,
    )
    assert dict(FINAL_ALPHA_SCORE_WEIGHTS) == {
        "return_hat": 0.25,
        "rank_hat": 0.55,
        "p_top": 0.20,
    }
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]


def test_alpha_rank_ensemble_trains_three_rolling_windows():
    prepared = prepare_alpha_rank_training_data(
        _data(periods=800), horizon=5, min_cross_section=3
    )
    ensemble = AlphaRankEnsemble(
        horizon=5,
        windows_months=(6, 12, 18),
        params={
            "n_estimators": 3,
            "num_leaves": 4,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_samples": 5,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": 3,
        },
    ).fit(prepared)
    predictions = ensemble.predict(prepared.tail(30))
    assert set(ensemble.models) == {"6", "12", "18"}
    assert {
        "return_hat",
        "rank_hat",
        "p_top",
        "alpha_score",
        "alpha_consensus",
    } <= set(
        predictions
    )
    assert {"alpha_score_6m", "alpha_score_12m", "alpha_score_18m"} <= set(
        predictions
    )
    labelled = prepared.dropna(subset=["alpha_fwd_return"])
    assert ensemble.trained_on["rows"] == len(labelled)
    assert ensemble.trained_on["end"] == str(labelled["date"].max().date())


def test_alpha_rank_masks_labels_for_small_cross_sections():
    prepared = prepare_alpha_rank_training_data(
        _data(periods=12),
        horizon=3,
        min_cross_section=4,
    )
    targets = [
        "alpha_fwd_return",
        "alpha_excess_return",
        "alpha_return_rank",
        "alpha_top20",
    ]
    assert prepared[targets].isna().all().all()


def test_alpha_rank_rejects_duplicate_keys_and_invalid_closes():
    data = _data(periods=12)
    duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with np.testing.assert_raises_regex(ValueError, "duplicate date/symbol"):
        prepare_alpha_rank_training_data(
            duplicate, horizon=3, min_cross_section=3
        )

    invalid = data.copy()
    invalid.loc[0, "close"] = 0.0
    with np.testing.assert_raises_regex(ValueError, "non-positive"):
        prepare_alpha_rank_training_data(
            invalid, horizon=3, min_cross_section=3
        )


def test_alpha_rank_training_dates_have_equal_total_weight():
    frame = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2026-01-02"),
                pd.Timestamp("2026-01-03"),
                pd.Timestamp("2026-01-03"),
                pd.Timestamp("2026-01-03"),
            ]
        }
    )
    frame["weight"] = _date_equal_weights(frame)
    totals = frame.groupby("date")["weight"].sum()
    assert totals.nunique() == 1
    assert np.isclose(frame["weight"].mean(), 1.0)
