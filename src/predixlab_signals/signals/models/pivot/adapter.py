"""Adapter exposing pivot signals through the shared model contract."""

from __future__ import annotations

from predixlab_signals.signals.models.pivot.events import (
    BASE_BREAKOUT,
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
)
from predixlab_signals.signals.models.pivot.extractor import extract_turning_points
from predixlab_signals.signals.models.pivot.filters import PivotEventFilter
from predixlab_signals.signals.models.pivot.publishing import enrich_pivot_events
from predixlab_signals.signals.specs import EventDefinition, SignalModelSpec

PIVOT_SIGNAL_MODEL = SignalModelSpec(
    name="pivot",
    required_columns=("symbol", "date", "high", "low", "close", "volume", "action"),
    event_column="turn_action",
    no_event=NO_TURN,
    events={
        BUY_TURN: EventDefinition(
            "long", "reversal", True, "BUY - confirmed bottom reversal"
        ),
        BOTTOM_TURN: EventDefinition(
            "long",
            "detection",
            False,
            "BOTTOM - confirmed low reversal, directional buy guarded",
        ),
        TREND_BUY: EventDefinition(
            "long", "trend", True, "TREND BUY - joining a confirmed advance"
        ),
        BASE_BREAKOUT: EventDefinition(
            "long",
            "trend",
            True,
            "BASE BREAKOUT - volume-backed breakout from compressed volatility",
        ),
        TOP_TURN: EventDefinition(
            "short",
            "detection",
            False,
            "TOP - confirmed high reversal, directional sell guarded",
        ),
        SELL_TURN: EventDefinition(
            "short", "reversal", True, "SELL - confirmed top reversal"
        ),
    },
    extractor=extract_turning_points,
    filters=(PivotEventFilter(),),
    event_aliases=("pivot_event",),
    event_enricher=enrich_pivot_events,
)
