"""Contracts for model-specific signal generation and filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class EventDefinition:
    """Stable product metadata for one model event."""

    direction: str
    kind: str
    actionable: bool
    title: str


class SignalFilter(Protocol):
    """A model-owned filter applied after raw signal generation."""

    def apply(
        self,
        signals: pd.DataFrame,
        model: "SignalModelSpec",
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class SignalModelSpec:
    """Everything the generic publishing pipeline needs from a model."""

    name: str
    required_columns: tuple[str, ...]
    event_column: str
    no_event: str
    events: Mapping[str, EventDefinition]
    extractor: Callable[[pd.DataFrame, object | None], pd.DataFrame]
    filters: tuple[SignalFilter, ...] = field(default_factory=tuple)
    event_aliases: tuple[str, ...] = field(default_factory=tuple)
    event_enricher: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None
