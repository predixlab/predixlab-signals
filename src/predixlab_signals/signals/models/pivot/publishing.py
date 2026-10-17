"""Publishing enrichment owned by the pivot signal model."""

from __future__ import annotations

import pandas as pd

from predixlab_signals.signals.scoring import enrich_scores

PIVOT_EVENT_COLUMNS = (
    "signal_model",
    "event",
    "pivot_event",
    "actionable",
    "direction",
    "kind",
    "rank",
)


def enrich_pivot_events(
    events: pd.DataFrame,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Attach pivot score strength while preserving normalized event metadata."""
    if events.empty:
        return events

    scored_columns = [
        column
        for column in (
            "symbol",
            "date",
            "close",
            "price",
            "swing_pos_hat",
            "p_up",
            "exp_gain_pct",
            "exp_loss_pct",
            "exp_amplitude_pct",
            "score",
            "action",
        )
        if column in events.columns
    ]
    enriched = enrich_scores(
        events[scored_columns].reset_index(drop=True),
        cross_section=scored,
    )
    for column in PIVOT_EVENT_COLUMNS:
        enriched[column] = events[column].to_numpy()
    return enriched
