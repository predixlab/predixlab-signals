"""Evaluation summaries for pivot reversal events."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.events import (
    BOTTOM_TURN,
    BUY_TURN,
    SELL_TURN,
    TOP_TURN,
)

def summarise(preds: pd.DataFrame, tolerance: int = 3) -> dict[str, float]:
    """Density, pivot precision/recall and lift for the extracted events."""
    if "turn_action" not in preds.columns or "pivot" not in preds.columns:
        return {}

    stats = dict.fromkeys(("buy_n", "buy_tp", "sell_n", "sell_tp", "lows", "lows_hit",
                           "highs", "highs_hit"), 0)
    for _, group in preds.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        turn = group["turn_action"].to_numpy()
        pivot = group["pivot"].to_numpy()
        buy = turn == BUY_TURN
        sell = np.isin(turn, (SELL_TURN, TOP_TURN))
        buy = np.isin(turn, (BUY_TURN, BOTTOM_TURN))
        is_low, is_high = pivot == -1.0, pivot == 1.0

        stats["buy_n"] += int(buy.sum())
        stats["buy_tp"] += int((buy & _dilate(is_low, tolerance)).sum())
        stats["sell_n"] += int(sell.sum())
        stats["sell_tp"] += int((sell & _dilate(is_high, tolerance)).sum())
        stats["lows"] += int(is_low.sum())
        stats["lows_hit"] += int((is_low & _dilate(buy, tolerance)).sum())
        stats["highs"] += int(is_high.sum())
        stats["highs_hit"] += int((is_high & _dilate(sell, tolerance)).sum())

    span = 2 * tolerance + 1
    base_low = float((preds["pivot"] == -1.0).mean()) * span
    base_high = float((preds["pivot"] == 1.0).mean()) * span
    buy_prec = _ratio(stats["buy_tp"], stats["buy_n"])
    sell_prec = _ratio(stats["sell_tp"], stats["sell_n"])

    return {
        "buy_events": stats["buy_n"],
        "sell_events": stats["sell_n"],
        "buy_density": _ratio(stats["buy_n"], len(preds)),
        "sell_density": _ratio(stats["sell_n"], len(preds)),
        "buy_precision": buy_prec,
        "sell_precision": sell_prec,
        "buy_recall": _ratio(stats["lows_hit"], stats["lows"]),
        "sell_recall": _ratio(stats["highs_hit"], stats["highs"]),
        "buy_lift": _ratio(buy_prec, base_low),
        "sell_lift": _ratio(sell_prec, base_high),
    }


def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0:
        return mask
    out = mask.copy()
    for shift in range(1, tolerance + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")
