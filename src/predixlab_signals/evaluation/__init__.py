"""Out-of-sample evaluation metrics and systematic diagnostics."""

from predixlab_signals.evaluation.diagnostics import (
    add_trend_context,
    run_diagnostics,
    trend_correlation,
    trend_profile,
    universe_report,
    within_trend_discrimination,
)
from predixlab_signals.evaluation.metrics import (
    action_buckets,
    cross_sectional_ic,
    evaluate,
    pivot_hold_report,
    probability_report,
    time_series_ic,
    turning_point_report,
)

__all__ = [
    "evaluate",
    "cross_sectional_ic",
    "time_series_ic",
    "action_buckets",
    "turning_point_report",
    "probability_report",
    "pivot_hold_report",
    "run_diagnostics",
    "trend_profile",
    "within_trend_discrimination",
    "trend_correlation",
    "universe_report",
    "add_trend_context",
]
