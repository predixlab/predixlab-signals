"""Real Collector-tree regressions: bad input must never reach either model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.config import Paths
from predixlab_signals.data import dataset
from predixlab_signals.jobs import daily
from predixlab_signals.settings import Settings

TARGET = "2026-09-08"
PREVIOUS = "2026-09-04"
SYMBOLS = [f"SYM{i:02d}" for i in range(20)]


@pytest.fixture
def collector_paths(tmp_path):
    dates = pd.bdate_range(end=PREVIOUS, periods=420).append(pd.DatetimeIndex([TARGET]))
    normalized = tmp_path / "normalized_data" / "us_data"
    normalized.mkdir(parents=True)
    for index, symbol in enumerate([*SYMBOLS, "^GSPC"]):
        step = np.arange(len(dates))
        close = (20.0 + index) * np.exp(0.0004 * step + 0.03 * np.sin(step / 13))
        pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 100000.0 + step * 100,
            "factor": 1.0,
        }).to_csv(normalized / f"{symbol}.csv", index=False)
    (tmp_path / "calendar").mkdir()
    (tmp_path / "calendar" / "us.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")))
    (tmp_path / "instruments").mkdir()
    (tmp_path / "instruments" / "us.txt").write_text(
        "\n".join(f"{symbol}\t2020-01-01\t2099-12-31" for symbol in SYMBOLS)
    )
    return Paths(
        collector_data=tmp_path,
        training_data=tmp_path / "unused-training",
        artifacts=tmp_path / "artifacts",
    )


def _damage(paths: Paths, symbols: list[str], kind: str) -> None:
    for symbol in symbols:
        path = paths.normalized_dir("us") / f"{symbol}.csv"
        if kind == "missing_file":
            path.unlink()
            continue
        if kind == "unreadable":
            path.write_text('date,close\n"unterminated')
            continue
        frame = pd.read_csv(path)
        if kind == "missing":
            frame = frame[frame["date"] != TARGET]
        elif kind == "missing_previous":
            frame = frame[frame["date"] != PREVIOUS]
        elif kind == "unchanged":
            columns = ["open", "high", "low", "close", "volume", "factor"]
            frame.loc[frame["date"] == TARGET, columns] = frame.loc[
                frame["date"] == PREVIOUS, columns
            ].to_numpy()
        elif kind == "invalid":
            frame.loc[frame["date"] == TARGET, "open"] = 0.0
        elif kind == "nonnumeric":
            frame["close"] = frame["close"].astype(object)
            frame.loc[frame["date"] == TARGET, "close"] = "not-a-price"
        elif kind == "schema":
            frame = frame.drop(columns=["volume"])
        elif kind in {
            "duplicate", "duplicate_identical", "duplicate_previous",
            "duplicate_invalid", "duplicate_history",
        }:
            date = PREVIOUS if kind == "duplicate_previous" else TARGET
            if kind == "duplicate_history":
                date = frame["date"].iloc[-3]
            duplicate = frame.loc[frame["date"] == date].copy()
            if kind != "duplicate_identical":
                duplicate[["open", "high", "low", "close"]] *= 1.25
            if kind == "duplicate_invalid":
                duplicate["close"] = np.nan
            frame = pd.concat([frame, duplicate], ignore_index=True).sort_values(
                "date", kind="stable"
            )
        else:
            raise AssertionError(f"Unknown damage type {kind}")
        frame.to_csv(path, index=False)


def _run(
    paths: Paths,
    monkeypatch,
    *,
    dry_run: bool = False,
    end_date: str | None = TARGET,
    market: str = "us",
    require_trading_day: bool = True,
):
    calls = []
    statuses = []
    model_inputs = []

    class Store:
        def ensure_indexes(self):
            calls.append("indexes")

        def write_picks(self, documents):
            calls.append("publish-pivot")
            return len(list(documents))

        def write_alpharank(self, documents):
            calls.append("publish-alpha")
            return len(list(documents))

        def write_job_status(self, status):
            statuses.append(status)
            calls.append("status")

        def close(self):
            calls.append("close")

    def pivot_loader(**kwargs):
        calls.append("load-pivot")
        return object(), {"retrained": False}

    def alpha_loader(**kwargs):
        calls.append("load-alpha")
        return object(), {"retrained": False}

    def pivot_score(model, data, **kwargs):
        calls.append("score-pivot")
        model_inputs.append(data)
        assert kwargs["as_of"] == pd.Timestamp(TARGET)
        return data.assign(score=0.1, action="HOLD")

    def alpha_score(model, data):
        calls.append("score-alpha")
        model_inputs.append(data)
        return pd.DataFrame({
            "return_hat": 0.1,
            "rank_hat": 0.2,
            "p_top": 0.3,
            "alpha_score": 0.5,
            "alpha_consensus": 0.5,
        }, index=data.index)

    monkeypatch.setattr(daily, "select_picks", lambda scored, **kwargs: scored.iloc[0:0])
    run = daily.run_daily(
        settings=Settings(
            market=market,
            require_trading_day=require_trading_day,
            lookback_sessions=2,
            campaign_events_enabled=True,
            campaign_storage_connection_string="UseDevelopmentStorage=true",
        ),
        paths=paths,
        end_date=end_date,
        dry_run=dry_run,
        clock=lambda: datetime(2026, 9, 9, 2, tzinfo=timezone.utc),
        model_loader=pivot_loader,
        panel_scorer=pivot_score,
        alpha_model_loader=alpha_loader,
        alpha_scorer=alpha_score,
        store_factory=lambda settings: Store(),
        notifier=lambda status, settings: calls.append("notify") or True,
        campaign_publisher=lambda *args: calls.append("campaign"),
    )
    return run, calls, statuses, model_inputs


def test_mixed_optional_factor_files_remain_usable(collector_paths, monkeypatch):
    for symbol in SYMBOLS[:2]:
        path = collector_paths.normalized_dir("us") / f"{symbol}.csv"
        pd.read_csv(path).drop(columns=["factor"]).to_csv(path, index=False)

    run, calls, statuses, inputs = _run(collector_paths, monkeypatch, dry_run=True)

    assert run.status == daily.STATUS_SUCCESS, run.error
    assert run.symbols_scored == run.alpha_symbols_scored == 20
    assert run.data_validation["bad_symbols"] == 0
    for model_input in inputs:
        unadjusted = model_input[model_input["symbol"].isin(SYMBOLS[:2])]
        np.testing.assert_allclose(unadjusted["price"], unadjusted["close"])


@pytest.mark.parametrize("factor", [np.nan, 0.0, -1.0])
def test_explicit_invalid_factor_is_not_defaulted(collector_paths, monkeypatch, factor):
    for symbol in SYMBOLS[:2]:
        path = collector_paths.normalized_dir("us") / f"{symbol}.csv"
        frame = pd.read_csv(path)
        frame.loc[frame["date"] == TARGET, "factor"] = factor
        frame.to_csv(path, index=False)

    run, calls, statuses, inputs = _run(collector_paths, monkeypatch, dry_run=True)

    assert run.status == daily.STATUS_FAILED
    assert run.data_validation["excluded_symbols"] == SYMBOLS[:2]
    assert not inputs


@pytest.mark.parametrize(
    ("kind", "count"),
    [
        ("missing", 20),
        ("missing", 2),
        ("missing_file", 2),
        ("unchanged", 20),
        ("unchanged", 2),
        ("invalid", 2),
        ("nonnumeric", 2),
        ("unreadable", 2),
        ("schema", 2),
        ("missing_previous", 2),
        ("duplicate", 2),
        ("duplicate_identical", 2),
        ("duplicate_previous", 2),
        ("duplicate_invalid", 2),
    ],
)
def test_broken_collection_stops_before_features_models_or_snapshots(
    collector_paths, monkeypatch, kind, count
):
    _damage(collector_paths, SYMBOLS[:count], kind)
    monkeypatch.setattr(
        dataset, "build_features",
        lambda *args, **kwargs: pytest.fail("bad raw inputs must not reach features"),
    )
    run, calls, statuses, inputs = _run(collector_paths, monkeypatch)

    assert run.status == daily.STATUS_FAILED, run.error
    assert run.date == TARGET
    assert run.symbols_scored == 0
    assert not inputs
    assert calls == ["indexes", "status", "close", "notify"]
    assert run.campaign_event_status == "blocked"
    assert not (collector_paths.collector_data / "predictions").exists()
    assert not (collector_paths.collector_data / "picks").exists()
    assert not collector_paths.artifacts.exists()
    assert statuses[0]["results"]["data_validation"]["bad_symbols"] == count
    assert statuses[0]["results"]["data_validation"]["expected_symbols"] == 20


@pytest.mark.parametrize(
    "kind", [
        "missing", "missing_file", "unchanged", "invalid", "nonnumeric", "schema", "unreadable",
        "duplicate", "duplicate_identical", "duplicate_previous", "duplicate_invalid",
    ]
)
def test_isolated_bad_data_is_absent_from_both_models_and_daily_artifacts(
    collector_paths, monkeypatch, kind
):
    _damage(collector_paths, SYMBOLS[:1], kind)
    run, calls, statuses, inputs = _run(collector_paths, monkeypatch)

    assert run.status == daily.STATUS_SUCCESS, run.error
    assert run.delivered
    assert run.symbols_scored == run.alpha_symbols_scored == 19
    assert run.data_validation["status"] == "passed"
    assert run.data_validation["excluded_symbols"] == SYMBOLS[:1]
    assert calls.index("score-pivot") < calls.index("score-alpha") < calls.index("campaign")
    assert len(inputs) == 2 and inputs[0] is inputs[1]
    assert SYMBOLS[0] not in set(inputs[0].loc[inputs[0]["date"] == TARGET, "symbol"])
    for artifact in (run.artifact, run.alpha_artifact):
        frame = pd.read_csv(artifact)
        assert set(frame["date"]) == {TARGET}
        assert len(frame) == 19
        assert SYMBOLS[0] not in set(frame["symbol"])
    assert statuses[0]["results"]["data_validation"] == run.data_validation


@pytest.mark.parametrize("end_date", [None, TARGET])
def test_missing_target_never_falls_back_even_when_calendar_skip_is_disabled(
    collector_paths, monkeypatch, end_date
):
    _damage(collector_paths, SYMBOLS, "missing")
    run, calls, statuses, inputs = _run(
        collector_paths,
        monkeypatch,
        end_date=end_date,
        require_trading_day=False,
    )
    assert run.status == daily.STATUS_FAILED
    assert run.date == TARGET
    assert run.data_validation["missing_symbols"] == SYMBOLS
    assert not inputs
    assert run.artifact is None


@pytest.mark.parametrize("dry_run", [False, True])
def test_rejected_run_preserves_existing_snapshots(collector_paths, monkeypatch, dry_run):
    prediction = collector_paths.daily_predictions_dir("us", TARGET) / "us_pivot.csv"
    alpha = prediction.with_name("us_alpharank.csv")
    picks = collector_paths.daily_picks_dir(TARGET) / "us_pivot.csv"
    for path in (prediction, alpha, picks):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("previous valid snapshot\n")
    _damage(collector_paths, SYMBOLS, "unchanged")
    run, calls, statuses, inputs = _run(collector_paths, monkeypatch, dry_run=dry_run)
    assert run.status == daily.STATUS_FAILED
    assert not inputs
    if dry_run:
        assert not calls and not statuses
    else:
        assert calls == ["indexes", "status", "close", "notify"]
        assert statuses[0]["status"] == daily.STATUS_FAILED
    for path in (prediction, alpha, picks):
        assert path.read_text() == "previous valid snapshot\n"


def test_feature_warmup_coverage_is_checked_before_loading_models(collector_paths, monkeypatch):
    build_features = dataset.build_features

    def missing_features(panel, benchmark):
        features = build_features(panel, benchmark)
        mask = (features["date"] == TARGET) & features["symbol"].isin(SYMBOLS[:2])
        features.loc[mask, "rsi_14"] = np.nan
        return features

    monkeypatch.setattr(dataset, "build_features", missing_features)
    run, calls, statuses, inputs = _run(collector_paths, monkeypatch)
    assert run.status == daily.STATUS_FAILED
    assert run.data_validation["unusable_symbols"] == SYMBOLS[:2]
    assert not inputs
    assert "load-pivot" not in calls and "load-alpha" not in calls
    assert run.artifact is None and run.picks_file is None


@pytest.mark.parametrize("kind", [
    "missing", "unchanged", "duplicate", "duplicate_identical",
    "duplicate_previous", "duplicate_invalid",
])
def test_unhealthy_benchmark_blocks_both_models(collector_paths, monkeypatch, kind):
    _damage(collector_paths, ["^GSPC"], kind)
    run, calls, statuses, inputs = _run(collector_paths, monkeypatch)
    assert run.status == daily.STATUS_FAILED
    assert run.data_validation["benchmark_error"]
    assert not inputs
    assert run.artifact is None and run.alpha_artifact is None


def test_older_duplicate_dates_preserve_keep_last_behavior(collector_paths, monkeypatch):
    _damage(collector_paths, [*SYMBOLS[:2], "^GSPC"], "duplicate_history")

    run, calls, statuses, inputs = _run(collector_paths, monkeypatch, dry_run=True)

    assert run.status == daily.STATUS_SUCCESS, run.error
    assert run.data_validation["bad_symbols"] == 0
    assert run.symbols_scored == run.alpha_symbols_scored == 20


def test_validation_and_inference_share_configured_input_paths(collector_paths, monkeypatch):
    root = collector_paths.collector_data
    normalized = root / "custom-normalized"
    instruments = root / "custom-membership.txt"
    collector_paths.normalized_dir("us").rename(normalized)
    collector_paths.instruments_path("us").rename(instruments)
    paths = Paths(
        collector_data=root,
        artifacts=collector_paths.artifacts,
        us_normalized_data_dir=normalized,
        us_index_path=instruments,
    )
    run, calls, statuses, inputs = _run(paths, monkeypatch)
    assert run.status == daily.STATUS_SUCCESS, run.error
    assert run.data_validation["valid_symbols"] == 20


def test_automatic_day_check_uses_the_selected_market_calendar(collector_paths, monkeypatch):
    root = collector_paths.collector_data
    cn_normalized = root / "normalized_data" / "cn_data"
    collector_paths.normalized_dir("us").rename(cn_normalized)
    (cn_normalized / "^GSPC.csv").rename(cn_normalized / "SH000300.csv")
    collector_paths.instruments_path("us").rename(root / "instruments" / "cn.txt")
    collector_paths.calendar_path("us").rename(root / "calendar" / "cn.txt")
    collector_paths.calendar_path("us").write_text(PREVIOUS)

    run, calls, statuses, inputs = _run(
        collector_paths, monkeypatch, market="cn", end_date=None
    )
    assert run.status == daily.STATUS_SUCCESS, run.error
    assert run.date == TARGET
    assert run.data_validation["benchmark"] == "SH000300"
