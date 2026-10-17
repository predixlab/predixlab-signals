"""Trend-following buy entries for the pivot model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.config import TurningPointConfig

def _trend_entry(frame: pd.DataFrame, cfg: "TurningPointConfig") -> np.ndarray:
    """Participation in an established advance, rather than a call on its low.

    The buy gate asks price to sit at the bottom of its range, which for a name
    in a confirmed uptrend is the wrong question — measured cross-sectionally,
    buying that dip returns **−0.763% over 60 sessions (t = −4.39)**. Buying the
    *breakout* instead returns +0.428% (t = +3.81), and +0.876% (t = +2.71) once
    the score also reads bullish. It is close to disjoint from the pivot events:
    of 1,955 such bars, 2 coincide with a ``BUY_TURN``.

    Its return profile is genuinely different and the difference is not
    cosmetic. The excess is a right-tail effect — the *median* raw 60-session
    return is +2.50% against +2.51% for the sample as a whole, and the win rate
    is 49.8%. A ``BUY_TURN`` by contrast lifts the median to +3.44%. So this
    signal improves the average without improving the typical outcome, and it is
    emitted under its own name for exactly that reason.
    """
    close = frame["close"].to_numpy(dtype="float64")
    high = frame["high"].to_numpy(dtype="float64")
    n = len(close)
    series = pd.Series(close)

    slow = series.rolling(cfg.trend_slow_ma, min_periods=cfg.trend_slow_ma // 2).mean()
    mid = series.rolling(cfg.trend_mid_ma, min_periods=cfg.trend_mid_ma // 2).mean().to_numpy()
    rising = np.zeros(n, dtype=bool)
    look = cfg.trend_slope_lookback
    if n > look:
        rising[look:] = mid[look:] > mid[:-look]
    established = np.nan_to_num(close > slow.to_numpy(), nan=0.0).astype(bool) & rising

    prior = (
        pd.Series(high)
        .rolling(cfg.trend_breakout_window, min_periods=cfg.trend_breakout_window)
        .max()
        .shift(1)
        .to_numpy()
    )
    breakout = np.nan_to_num(close > prior, nan=0.0).astype(bool)

    endorsed = pd.Series(
        np.isin(frame["action"].to_numpy(), list(cfg.trend_entry_states))
    )
    share = endorsed.rolling(
        cfg.trend_conviction_window,
        min_periods=min(cfg.trend_conviction_window, max(2, cfg.trend_conviction_window // 2)),
    ).mean()
    sustained = np.nan_to_num(
        share.to_numpy() >= cfg.trend_conviction_share, nan=0.0
    ).astype(bool)
    return established & breakout & sustained

def _apply_refractory(events: np.ndarray, gap: int) -> np.ndarray:
    """Keep the first event of a cluster, suppress the rest for ``gap`` bars."""
    if gap <= 0:
        return events
    kept = np.zeros_like(events)
    last = -(10**9)
    for i in np.flatnonzero(events):
        if i - last >= gap:
            kept[i] = True
            last = i
    return kept
