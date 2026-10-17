"""Causal calibration helpers for turning-point gates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from predixlab_signals.signals.models.pivot import TurningPointConfig


def _causal_panel_mean(frame: pd.DataFrame, column: str) -> pd.Series:
    """Mean of all finite observations available on or before each row's date."""
    values = pd.to_numeric(frame[column], errors="coerce")
    daily = pd.DataFrame({"date": frame["date"], "value": values}).groupby(
        "date", sort=True
    )["value"].agg(["sum", "count"])
    expanding = daily["sum"].cumsum() / daily["count"].cumsum().replace(0, np.nan)
    return frame["date"].map(expanding).astype("float64")


def add_gate_percentiles(frame: pd.DataFrame, cfg: "TurningPointConfig") -> None:
    """Attach causal cross-sectional calibration columns to ``frame``.

    Every denominator is computed from the current session and its history.
    Appending future sessions therefore cannot rewrite an existing signal.
    Return-weighted pivot scores are preferred when available; old model bundles
    transparently fall back to raw hold probabilities.
    """
    weight = float(np.clip(cfg.trend_neutral_weight, 0.0, 1.0))
    if weight > 0.0:
        trailing = frame.groupby("symbol", sort=False)["close"].transform(
            lambda s: s / s.shift(cfg.trend_window) - 1.0
        )
        bucket = (
            trailing.groupby(frame["date"])
            .rank(pct=True)
            .mul(cfg.trend_buckets)
            .clip(upper=cfg.trend_buckets - 1e-9)
        ).fillna(-1.0).astype(int)
        frame["_trend_bucket"] = bucket

    for probability, weighted_score in (
        ("p_pivot_low", "pivot_low_score"),
        ("p_pivot_high", "pivot_high_score"),
    ):
        if probability not in frame.columns:
            continue
        source = weighted_score if weighted_score in frame.columns else probability
        overall = frame.groupby("date")[source].rank(pct=True)
        if weight <= 0.0:
            frame[f"_{probability}_q"] = overall
        else:
            within = frame.groupby(["date", "_trend_bucket"], observed=True)[source].rank(
                pct=True
            )
            frame[f"_{probability}_q"] = (
                (1.0 - weight) * overall + weight * within.fillna(overall)
            )

        panel_mean = _causal_panel_mean(frame, source)
        valid_mean = panel_mean.where(panel_mean > 0)
        if cfg.pivot_abs_mult is not None:
            frame[f"_{probability}_abs"] = frame[source] / valid_mean
        if cfg.breadth_relief is not None:
            day_mean = frame.groupby("date")[source].transform("mean")
            frame[f"_{probability}_breadth"] = day_mean / valid_mean
        if cfg.self_gate_window is not None:
            frame[f"_{probability}_s"] = frame.groupby(
                "symbol", sort=False
            )[source].transform(
                lambda s: s.rolling(
                    cfg.self_gate_window,
                    min_periods=cfg.self_gate_min_periods,
                ).rank(pct=True)
            )

    frame.drop(columns=["_trend_bucket"], errors="ignore", inplace=True)
