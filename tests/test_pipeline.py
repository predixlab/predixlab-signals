"""End-to-end pipeline test on a synthetic collector CSV tree.

Beyond checking that the stages connect, this file contains the package's most
important safety net: a **null test**. The same pipeline is run against labels
that have been shuffled, and it must then find nothing. A harness that reports
skill on randomised labels is measuring leakage, and every other number it
produces is worthless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.backtest import BacktestConfig, run_backtest
from predixlab_signals.config import Paths
from predixlab_signals.data.dataset import build_dataset, build_inference_dataset
from predixlab_signals.evaluation import cross_sectional_ic, evaluate
from predixlab_signals.features import feature_columns
from predixlab_signals.model import run_walk_forward


def _write_csv(path, days, symbol: str, **columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": days, "symbol": symbol, **columns}).to_csv(path, index=False)


@pytest.fixture(scope="module")
def collector_root(tmp_path_factory):
    """A synthetic market: 12 years of daily bars for 24 symbols plus an index."""
    root = tmp_path_factory.mktemp("collector")
    market = root / "normalized_data" / "us_data"
    market.mkdir(parents=True)

    days = pd.bdate_range("2014-01-01", periods=3000)
    (root / "calendar").mkdir()
    (root / "calendar" / "us.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in days)
    )
    (root / "instruments").mkdir()
    n = len(days)
    rng = np.random.default_rng(42)

    market_path = np.cumsum(rng.normal(0.0003, 0.008, n))
    for i in range(24):
        idio = np.cumsum(rng.normal(0.0002, 0.014, n))
        close = 40.0 * np.exp(market_path + idio)
        spread = np.abs(rng.normal(0.008, 0.004, n))
        symbol = f"SYM{i:02d}"
        _write_csv(
            market / f"{symbol}.csv",
            days,
            symbol,
            open=close * (1 + rng.normal(0, 0.002, n)),
            high=close * (1 + spread),
            low=close * (1 - spread),
            close=close,
            volume=rng.integers(1e5, 1e6, n).astype(float),
            factor=np.full(n, 0.5),
        )
    (root / "instruments" / "us.txt").write_text(
        "\n".join(f"SYM{i:02d}\t2014-01-01\t2099-12-31" for i in range(24))
    )

    bench = 1000.0 * np.exp(market_path)
    _write_csv(
        market / "^GSPC.csv",
        days,
        "^GSPC",
        open=bench,
        high=bench * 1.004,
        low=bench * 0.996,
        close=bench,
        volume=np.full(n, 1e9),
        factor=np.ones(n),
    )
    return root


@pytest.fixture(scope="module")
def dataset(collector_root):
    return build_dataset("us", start="2015-01-01", horizon=10, data_root=collector_root)


def test_training_and_inference_use_separate_roots(collector_root, tmp_path):
    training_paths = Paths(
        collector_data=tmp_path / "clean",
        training_data=collector_root,
        artifacts=tmp_path / "artifacts",
    )
    training = build_dataset(
        "us",
        symbols=["SYM00"],
        start="2020-01-01",
        horizon=10,
        paths=training_paths,
    )
    assert not training.empty

    inference_paths = Paths(
        collector_data=collector_root,
        training_data=tmp_path / "history",
        artifacts=tmp_path / "artifacts",
    )
    inference = build_inference_dataset(
        "us",
        symbols=["SYM00"],
        horizon=10,
        paths=inference_paths,
    )
    assert not inference.empty
    assert "volume" in inference


def test_dataset_is_well_formed(dataset):
    assert not dataset.empty
    assert dataset["symbol"].nunique() == 24  # the index is not tradeable
    assert not dataset.duplicated(["symbol", "date"]).any()
    assert dataset["date"].is_monotonic_increasing


def test_dataset_excludes_forward_looking_columns(dataset):
    """The single check that protects every downstream number."""
    feats = feature_columns(dataset, include_market=True)
    for leaky in ("open_next", "pivot", "fwd_ret", "fwd_mfe", "fwd_mae", "label_scale"):
        assert leaky not in feats
    assert "open" in dataset
    assert "open" not in feats
    assert "volume" in dataset
    assert "volume" not in feats
    assert not any(c.startswith("y_") for c in feats)


def test_market_features_are_excluded_by_default(dataset):
    assert not any(c.startswith("mkt_") for c in feature_columns(dataset))
    assert any(c.startswith("mkt_") for c in feature_columns(dataset, include_market=True))


def test_labels_are_missing_only_at_the_tail(dataset):
    """Unlabelled rows are exactly the most recent bars — the live ones."""
    last = dataset["date"].max()
    unlabelled = dataset[dataset["y_swing_pos"].isna()]
    assert not unlabelled.empty
    assert (unlabelled["date"] > last - pd.Timedelta(days=30)).all()


def test_relative_targets_are_centred(dataset):
    """Demeaned targets sum to zero per day, up to float32 storage precision."""
    daily_mean = dataset.groupby("date")["y_swing_pos_cs"].mean().dropna()
    assert daily_mean.abs().max() < 1e-6


def test_dataset_is_stored_as_float32(dataset):
    assert "float64" not in {str(t) for t in dataset.dtypes}


def test_bounded_inference_matches_full_history_on_scored_tail(dataset, collector_root):
    inference = build_inference_dataset(
        "us",
        horizon=10,
        history_sessions=1000,
        data_root=collector_root,
    )
    assert "open" in inference
    sessions = np.sort(dataset["date"].unique())[-250:]
    full_tail = dataset[dataset["date"].isin(sessions)].sort_values(["date", "symbol"])
    live_tail = inference[inference["date"].isin(sessions)].sort_values(["date", "symbol"])

    assert list(full_tail[["date", "symbol"]].itertuples(index=False, name=None)) == list(
        live_tail[["date", "symbol"]].itertuples(index=False, name=None)
    )
    columns = feature_columns(dataset, include_market=True) + ["label_scale"]
    np.testing.assert_allclose(
        live_tail[columns].to_numpy(),
        full_tail[columns].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
        equal_nan=True,
    )


def test_pipeline_runs_end_to_end(dataset):
    result = run_walk_forward(
        dataset,
        horizon=10,
        n_folds=2,
        test_months=12,
        params={"n_estimators": 30, "num_leaves": 8, "min_child_samples": 100, "verbose": -1},
        verbose=False,
    )
    preds = result.predictions
    assert len(result.models) == 2
    assert set(result.head_estimators()) == {
        "swing_pos",
        "amplitude",
        "updown",
        "pivot_low",
        "pivot_high",
        "pivot_low_return",
        "pivot_high_return",
    }
    assert {"score", "action", "swing_pos_hat", "p_up"} <= set(preds.columns)
    assert "volume" in preds
    assert preds["action"].isin(["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]).all()

    report = evaluate(preds)
    assert report["rows"] == len(preds)

    trades = run_backtest(preds, BacktestConfig(horizon=10)).trades
    if not trades.empty:
        assert (trades["bars_held"] >= 1).all()


def test_walk_forward_predictions_are_out_of_sample(dataset):
    """Every prediction must fall strictly inside its own fold's test window."""
    result = run_walk_forward(
        dataset,
        horizon=10,
        n_folds=2,
        test_months=12,
        params={"n_estimators": 20, "num_leaves": 8, "min_child_samples": 100, "verbose": -1},
        verbose=False,
    )
    for fold, model in zip(result.folds, result.models):
        block = result.predictions[result.predictions["fold"] == fold.index]
        assert block["date"].min() >= fold.test_start
        assert pd.Timestamp(model.trained_on["end"]) < fold.test_start
    assert not result.predictions.duplicated(["symbol", "date"]).any()


def test_null_test_shuffled_labels_produce_no_skill(dataset):
    """Randomised labels must yield an IC indistinguishable from zero.

    If this fails, information is reaching the model through a channel other
    than the labels, and every reported metric is contaminated.
    """
    rng = np.random.default_rng(0)
    scrambled = dataset.copy()
    for col in (
        "y_swing_pos",
        "y_swing_pos_cs",
        "y_amplitude",
        "y_barrier",
        "y_barrier_cs",
        "y_pivot_low_hold",
        "y_pivot_high_hold",
        "y_pivot_low_return",
        "y_pivot_high_return",
    ):
        scrambled[col] = rng.permutation(scrambled[col].to_numpy())

    result = run_walk_forward(
        scrambled,
        horizon=10,
        n_folds=2,
        test_months=12,
        params={"n_estimators": 30, "num_leaves": 8, "min_child_samples": 100, "verbose": -1},
        verbose=False,
    )
    ic = cross_sectional_ic(result.predictions, "score", "fwd_ret_cs")
    assert abs(ic["rank_ic_mean"]) < 0.02
    assert abs(ic.get("rank_ic_tstat", 0.0)) < 3.0

    # The future-hold heads must be equally blind.
    from predixlab_signals.evaluation.metrics import pivot_hold_report

    preds = result.predictions.copy()
    detection = pivot_hold_report(preds, "low")
    if detection:
        assert 0.40 < detection["auc"] < 0.60, detection["auc"]
