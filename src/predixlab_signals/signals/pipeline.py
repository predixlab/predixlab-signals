"""Generic model-output to publishable-event pipeline."""

from __future__ import annotations

import pandas as pd

from predixlab_signals.signals.registry import get_signal_model
from predixlab_signals.signals.specs import SignalModelSpec


def resolve_signal_model(model: str | SignalModelSpec) -> SignalModelSpec:
    return get_signal_model(model) if isinstance(model, str) else model


def extract_model_events(
    scored: pd.DataFrame,
    config: object | None = None,
    *,
    model: str | SignalModelSpec = "pivot",
) -> pd.DataFrame:
    """Run one model's extractor and normalize its events for publication."""
    spec = resolve_signal_model(model)
    missing = [column for column in spec.required_columns if column not in scored.columns]
    if missing:
        raise ValueError(
            f"{spec.name} signal extraction needs {missing}; the scored panel does "
            "not satisfy the model's input contract."
        )
    if scored.empty:
        return scored.iloc[0:0].copy()

    signals = spec.extractor(scored, config)
    if spec.event_column not in signals.columns:
        raise ValueError(
            f"{spec.name} extractor did not produce {spec.event_column!r}"
        )
    for signal_filter in spec.filters:
        signals = signal_filter.apply(signals, spec)
    if signals.empty:
        return signals.drop(columns=[spec.event_column], errors="ignore")

    unknown = sorted(set(signals[spec.event_column]) - set(spec.events))
    if unknown:
        raise ValueError(f"{spec.name} extractor produced unknown events: {unknown}")

    events = signals.copy()
    definitions = events[spec.event_column].map(spec.events)
    events["signal_model"] = spec.name
    events["event"] = events[spec.event_column]
    for alias in spec.event_aliases:
        events[alias] = events[spec.event_column]
    events["actionable"] = [definition.actionable for definition in definitions]
    events["direction"] = [definition.direction for definition in definitions]
    events["kind"] = [definition.kind for definition in definitions]
    return events.drop(columns=[spec.event_column])
