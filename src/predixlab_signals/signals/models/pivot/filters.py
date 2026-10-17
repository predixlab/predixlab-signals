"""Pivot-model filters applied before events enter the shared pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from predixlab_signals.signals.specs import SignalModelSpec


@dataclass(frozen=True)
class PivotEventFilter:
    """Publish fired pivot events and discard the model's no-event state."""

    def apply(
        self,
        signals: pd.DataFrame,
        model: SignalModelSpec,
    ) -> pd.DataFrame:
        return signals[signals[model.event_column] != model.no_event].copy()

