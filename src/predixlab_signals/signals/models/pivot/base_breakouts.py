"""Base-breakout entries from compressed volatility and expanding volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.common import _atr
from predixlab_signals.signals.models.pivot.config import TurningPointConfig


def _base_breakout(frame: pd.DataFrame, cfg: TurningPointConfig) -> np.ndarray:
    """Detect a bullish breakout from a low-volatility 20-session base."""
    n = len(frame)
    if not cfg.base_breakout:
        return np.zeros(n, dtype=bool)

    close = frame["close"].to_numpy(dtype="float64")
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    volume = frame["volume"].to_numpy(dtype="float64")

    prior_high = (
        pd.Series(high)
        .rolling(
            cfg.base_breakout_window,
            min_periods=cfg.base_breakout_window,
        )
        .max()
        .shift(1)
        .to_numpy()
    )

    atr = pd.Series(_atr(frame))
    prior_atr = atr.shift(1)
    prior_atr_mean = prior_atr.rolling(
        cfg.base_breakout_atr_window,
        min_periods=cfg.base_breakout_atr_window,
    ).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        atr_ratio = prior_atr.to_numpy() / prior_atr_mean.to_numpy()

    prior_volume = (
        pd.Series(volume)
        .rolling(
            cfg.base_breakout_volume_window,
            min_periods=cfg.base_breakout_volume_window,
        )
        .mean()
        .shift(1)
        .to_numpy()
    )
    valid_volume = np.isfinite(prior_volume) & (prior_volume > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        close_location = (close - low) / (high - low)

    return (
        np.nan_to_num(high > prior_high, nan=0.0).astype(bool)
        & np.nan_to_num(
            close_location >= cfg.base_breakout_min_close_location,
            nan=0.0,
        ).astype(bool)
        & np.nan_to_num(
            atr_ratio <= cfg.base_breakout_max_atr_ratio,
            nan=0.0,
        ).astype(bool)
        & valid_volume
        & np.nan_to_num(
            volume >= cfg.base_breakout_min_volume_ratio * prior_volume,
            nan=0.0,
        ).astype(bool)
        & np.isin(frame["action"].to_numpy(), list(cfg.base_breakout_states))
    )
