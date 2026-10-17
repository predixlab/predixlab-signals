"""Backward-compatible imports for the pivot signal implementation.

New code should import from :mod:`predixlab_signals.signals.models.pivot`.
"""

from predixlab_signals.signals.models.pivot import (
    BASE_BREAKOUT,
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
    TurningPointConfig,
    extract_turning_points,
    summarise,
)
from predixlab_signals.signals.models.pivot.common import (
    _range_position,
    _recent,
    _recent_while_live,
    _sequence,
    _sequence_additions,
)
from predixlab_signals.signals.models.pivot.top_turns import (
    _top_followthrough_confirmations,
)
from predixlab_signals.signals.models.pivot.trend_breakouts import _apply_refractory

__all__ = [
    "BASE_BREAKOUT",
    "BOTTOM_TURN",
    "BUY_TURN",
    "NO_TURN",
    "SELL_TURN",
    "TOP_TURN",
    "TREND_BUY",
    "TurningPointConfig",
    "extract_turning_points",
    "summarise",
]
