"""Selection: turn a scored panel into the events worth acting on.

A pick is an **event**, not a ranking. The model scores every name every day,
but a score is a *state*: its lag-1 autocorrelation is 0.88, a ``BUY`` label
covers 31.7% of all bar-days, and real swing lows are 2.6% of bars. Publishing
"today's twenty highest scores" would therefore republish most of the same names
tomorrow, fire in every regime, and say nothing about whether anything actually
happened — it is the model artifact wearing a shorter list.

What the pivot signal model produces instead is sparse
and dated: on the walk-forward set 1.87% of bar-days carry an event, a median
session emits 8 of them, and 14 of 754 sessions emit none at all. A day with no
picks is a real answer, and the honest one.

Published events answer three different questions:

``BUY_TURN`` / ``SELL_TURN``
    A confirmed reversal — price sat in the extreme of its range, the model was
    convinced, and a reversal bar triggered. This is "has it turned".
``BOTTOM_TURN`` / ``TOP_TURN``
    Informational reversals that remain inside a strong directional trend and
    did not clear the corresponding confirmation filter.
``TREND_BUY``
    A trend-confirmation entry: a new 60-day high with sustained model
    conviction behind it. This is "is this advance worth joining", a different
    trade with a different holding pattern, so it carries its own label and is
    never merged into the reversal counts.
``BASE_BREAKOUT``
    A volume-backed break above a 20-day high after ATR contraction, closing
    near the top of its trigger bar and independently endorsed by a bullish
    model state. It joins the first move out of a base rather than requiring
    the established advance that ``TREND_BUY`` does.

Ordering within a session is by conviction, but that is *presentation* only. It
gates nothing: the rules already ran, and a day that produces thirty events
produces thirty picks.
"""

from __future__ import annotations

import pandas as pd

from predixlab_signals.signals.models.pivot import (
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
    TurningPointConfig,
)
from predixlab_signals.signals.models.pivot.adapter import PIVOT_SIGNAL_MODEL
from predixlab_signals.signals.pipeline import (
    extract_model_events,
    resolve_signal_model,
)
from predixlab_signals.signals.specs import SignalModelSpec

#: Position sides attached to published events.
LONG = "long"
SHORT = "short"

#: What question the event answers. Kept distinct from ``direction`` because a
#: reversal long and a trend long are different trades.
REVERSAL = "reversal"
TREND = "trend"
DETECTION = "detection"

#: Event -> (direction, kind). The single source of truth for the mapping.
EVENT_KINDS: dict[str, tuple[str, str]] = {
    event: (definition.direction, definition.kind)
    for event, definition in PIVOT_SIGNAL_MODEL.events.items()
}

#: Backward-compatible alias for the default pivot model's input contract.
REQUIRED_COLUMNS = PIVOT_SIGNAL_MODEL.required_columns

PICK_COLUMNS = [
    "date",
    "symbol",
    "signal_model",
    "event",
    "pivot_event",
    "actionable",
    "direction",
    "kind",
    "rank",
    "action",
    "score",
    "confidence",
    "swing_position",
    "p_up",
    "exp_gain_pct",
    "exp_loss_pct",
    "close",
    "price",
]


def extract_events(
    scored: pd.DataFrame,
    config: TurningPointConfig | None = None,
    *,
    model: str | SignalModelSpec = PIVOT_SIGNAL_MODEL,
) -> pd.DataFrame:
    """Run a model's signal pipeline and keep only the bars that fired.

    ``scored`` must be a *panel* — several sessions of history per symbol, not a
    single day when the selected model uses sequential rules.
    """
    return extract_model_events(scored, config, model=model)


def rank_events(events: pd.DataFrame) -> pd.DataFrame:
    """Order each session's events by conviction and number them.

    Presentation only. Unlike a cross-sectional cut this removes nothing: every
    event that cleared the rules is published, and the rank only says which to
    read first.
    """
    if events is None or events.empty:
        return events

    ranked = events.copy()
    ranked["_conviction"] = (
        pd.to_numeric(ranked["score"], errors="coerce").abs()
        if "score" in ranked
        else 0.0
    )
    ranked = ranked.sort_values(
        ["date", "direction", "_conviction"], ascending=[True, True, False], kind="stable"
    )
    ranked["rank"] = ranked.groupby(["date", "direction"], sort=False).cumcount() + 1
    return ranked.drop(columns="_conviction").reset_index(drop=True)


def select_picks(
    scored: pd.DataFrame,
    config: TurningPointConfig | None = None,
    max_per_side: int | None = None,
    *,
    model: str | SignalModelSpec = PIVOT_SIGNAL_MODEL,
) -> pd.DataFrame:
    """Extract the events and rank them. This is the published pick list.

    ``max_per_side`` is a **safety valve, not a selection rule**, and is off by
    default. The rules already decide what fires; capping the result would
    silently drop real signals on the busiest days, which are exactly the days a
    regime is turning. Set it only to bound a UI.
    """
    events = extract_events(scored, config, model=model)
    if events.empty:
        return events

    picks = rank_events(events)
    if max_per_side is not None and max_per_side > 0:
        picks = picks[picks["rank"] <= max_per_side].reset_index(drop=True)
    return picks


def summarise_picks(
    picks: pd.DataFrame,
    *,
    model: str | SignalModelSpec = PIVOT_SIGNAL_MODEL,
) -> dict[str, int]:
    """Count published events by type, for job status and email."""
    spec = resolve_signal_model(model)
    empty = dict.fromkeys(spec.events, 0)
    if picks is None or picks.empty or "event" not in picks.columns:
        return empty
    counts = picks["event"].value_counts()
    return {key: int(counts.get(key, 0)) for key in empty}


def format_picks(
    picks: pd.DataFrame,
    top: int = 10,
    *,
    model: str | SignalModelSpec = PIVOT_SIGNAL_MODEL,
) -> str:
    """Render the event list as a readable table."""
    if picks is None or picks.empty:
        return (
            "No turning points today."
            if resolve_signal_model(model).name == "pivot"
            else "No signals today."
        )
    spec = resolve_signal_model(model)
    cols = [
        c
        for c in PICK_COLUMNS
        if c in picks.columns and c not in ("date", "direction", "kind")
    ]
    stamp = pd.Timestamp(picks["date"].iloc[0])
    parts = [f"As of {stamp.date()}  ({len(picks)} events)"]
    for event, definition in spec.events.items():
        side = picks[picks["event"] == event].head(top)
        parts.append(f"\n=== {definition.title} ===")
        parts.append(side[cols].to_string(index=False) if not side.empty else "  none")
    return "\n".join(parts)
