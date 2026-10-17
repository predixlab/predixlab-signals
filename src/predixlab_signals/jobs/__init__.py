"""Scheduled jobs — the production face of the package.

:mod:`predixlab_signals.jobs.daily` is what the Azure Container Apps job runs each
night, after predixlab-collector has refreshed the shared data share.
:mod:`predixlab_signals.jobs.backfill` replays a stored walk-forward run into the
pick history that gives the dashboard something to show on day one.
"""

from predixlab_signals.jobs.backfill import build_historical_picks
from predixlab_signals.jobs.daily import DailyRun, run_daily

__all__ = ["DailyRun", "run_daily", "build_historical_picks"]
