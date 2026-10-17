"""Tests for model-independent signal generation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import pandas as pd

from predixlab_signals.config import Paths
from predixlab_signals.jobs.daily import STATUS_SUCCESS, run_daily
from predixlab_signals.settings import Settings
from predixlab_signals.signals.models.pivot import TurningPointConfig
from predixlab_signals.signals.models.pivot.bottom_turns import (
    apply_buy_conflict_filters,
    build_bottom_triggers,
    evaluate_bottom_turns,
)
from predixlab_signals.signals.models.pivot.extractor import extract_turning_points
from predixlab_signals.signals.models.pivot.top_turns import (
    _top_followthrough_confirmations,
    build_top_triggers,
    evaluate_top_turns,
)
from predixlab_signals.signals.models.pivot.trend_breakouts import _trend_entry
from predixlab_signals.signals.pipeline import extract_model_events
from predixlab_signals.signals.registry import list_signal_models
from predixlab_signals.signals.specs import EventDefinition, SignalModelSpec
from predixlab_signals.signals.turning_points import (
    TurningPointConfig as LegacyTurningPointConfig,
)


@dataclass(frozen=True)
class MinimumStrengthFilter:
    minimum: float

    def apply(
        self,
        signals: pd.DataFrame,
        model: SignalModelSpec,
    ) -> pd.DataFrame:
        return signals[
            (signals[model.event_column] != model.no_event)
            & (signals["strength"] >= self.minimum)
        ].copy()


def _extract_example_signals(
    scored: pd.DataFrame,
    config: object | None,
) -> pd.DataFrame:
    out = scored.copy()
    out["signal_code"] = out["candidate"]
    return out


def test_pivot_rules_live_in_signal_family_modules():
    assert TurningPointConfig.__module__.endswith(".models.pivot.config")
    assert apply_buy_conflict_filters.__module__.endswith(".pivot.bottom_turns")
    assert build_bottom_triggers.__module__.endswith(".pivot.bottom_turns")
    assert evaluate_bottom_turns.__module__.endswith(".pivot.bottom_turns")
    assert _top_followthrough_confirmations.__module__.endswith(
        ".pivot.top_turns"
    )
    assert build_top_triggers.__module__.endswith(".pivot.top_turns")
    assert evaluate_top_turns.__module__.endswith(".pivot.top_turns")
    assert _trend_entry.__module__.endswith(".pivot.trend_breakouts")
    assert LegacyTurningPointConfig is TurningPointConfig


def test_extractor_contains_orchestration_not_family_thresholds():
    source = inspect.getsource(extract_turning_points)
    family_thresholds = (
        "bottom_confirm_panic",
        "bottom_right_rescue_min",
        "top_guard_boundary",
        "top_confirm_crash",
        "sell_trend_conflict_min",
        "sell_stale_conflict",
    )
    assert not any(threshold in source for threshold in family_thresholds)


def test_builtin_pivot_model_is_registered():
    assert "pivot" in list_signal_models()


def test_an_independent_model_owns_its_events_and_filters():
    model = SignalModelSpec(
        name="example",
        required_columns=("symbol", "date", "candidate", "strength"),
        event_column="signal_code",
        no_event="NONE",
        events={
            "ENTER": EventDefinition(
                direction="long",
                kind="example",
                actionable=True,
                title="Example entry",
            )
        },
        extractor=_extract_example_signals,
        filters=(MinimumStrengthFilter(0.8),),
    )
    scored = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "date": pd.to_datetime(["2026-01-02"] * 3),
            "candidate": ["ENTER", "ENTER", "NONE"],
            "strength": [0.9, 0.7, 1.0],
        }
    )

    events = extract_model_events(scored, model=model)

    assert events["symbol"].tolist() == ["AAA"]
    assert events["signal_model"].tolist() == ["example"]
    assert events["event"].tolist() == ["ENTER"]
    assert events["direction"].tolist() == ["long"]
    assert events["actionable"].tolist() == [True]
    assert "pivot_event" not in events


def test_each_model_validates_its_own_input_contract():
    model = SignalModelSpec(
        name="example",
        required_columns=("symbol", "date", "strength"),
        event_column="signal_code",
        no_event="NONE",
        events={},
        extractor=_extract_example_signals,
    )

    try:
        extract_model_events(pd.DataFrame({"symbol": ["AAA"]}), model=model)
    except ValueError as exc:
        assert "example signal extraction needs" in str(exc)
        assert "date" in str(exc)
        assert "strength" in str(exc)
    else:
        raise AssertionError("missing model inputs must be rejected")


def test_daily_job_accepts_an_independent_model_runtime(
    monkeypatch,
    tmp_path,
):
    source = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": pd.to_datetime(["2026-01-02"]),
        }
    )
    scored = source.assign(
        candidate="ENTER",
        strength=0.9,
        score=0.9,
        close=100.0,
        price=100.0,
    )
    model = SignalModelSpec(
        name="example",
        required_columns=("symbol", "date", "candidate", "strength"),
        event_column="signal_code",
        no_event="NONE",
        events={
            "ENTER": EventDefinition(
                direction="long",
                kind="example",
                actionable=True,
                title="Example entry",
            )
        },
        extractor=_extract_example_signals,
        filters=(MinimumStrengthFilter(0.8),),
    )
    monkeypatch.setattr(
        "predixlab_signals.jobs.daily.build_inference_dataset",
        lambda **kwargs: source,
    )

    run = run_daily(
        settings=Settings(require_trading_day=False, lookback_sessions=2),
        end_date="2026-01-02",
        dry_run=True,
        paths=Paths(
            collector_data=tmp_path,
            training_data=tmp_path,
            artifacts=tmp_path / "artifacts",
        ),
        signal_model=model,
        model_loader=lambda **kwargs: (object(), {"runtime": "example"}),
        panel_scorer=lambda loaded, data, **kwargs: scored,
    )

    assert run.status == STATUS_SUCCESS
    assert run.signal_model == "example"
    assert run.picks is not None
    assert run.picks["signal_model"].tolist() == ["example"]
    assert run.artifact is not None and run.artifact.endswith("us_example.csv")
    assert run.picks_file is not None and run.picks_file.endswith("us_example.csv")
