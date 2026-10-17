"""Shared causal helpers used by pivot signal families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.config import TurningPointConfig
from predixlab_signals.signals.models.pivot.events import (
    BASE_BREAKOUT,
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
)

def _atr(frame: pd.DataFrame, window: int = 14) -> np.ndarray:
    """ATR in price units, taken from the label column when it is available."""
    if "atr_pct_label" in frame.columns:
        return (frame["atr_pct_label"] * frame["close"]).to_numpy(dtype="float64")
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean().to_numpy()


def _trend_slope(close: pd.Series, span: int = 50, lookback: int = 20) -> np.ndarray:
    average = close.rolling(span, min_periods=span).mean()
    return (average / average.shift(lookback) - 1.0).to_numpy(dtype="float64")

def _sequence(
    buy: np.ndarray,
    sell: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    gap: int,
    rearm_atr: float,
    rearm_uses_path: bool = False,
    rearm_max_bars: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a candidate stream to one signal per swing.

    Walks bars in order. A side may fire only if the refractory gap has elapsed
    *and* price has moved ``rearm_atr`` ATRs from where that same side last
    fired, so a long one-directional move yields a handful of spaced signals
    rather than one per bounce.

    ``rearm_uses_path`` also re-arms on the largest excursion since that signal,
    not only on where price ended up. The endpoint test cannot tell "we never
    left this level" from "we went away and came back", and those are different
    situations: a completed round trip *is* a new swing, so a second signal is
    new information rather than a duplicate. ``rearm_max_bars`` bounds how long
    price travel alone may keep a side disarmed.
    """
    kept_buy = np.zeros(buy.size, dtype=bool)
    kept_sell = np.zeros(sell.size, dtype=bool)
    last_index = -(10**9)
    last_buy_px = np.nan
    last_sell_px = np.nan
    last_buy_index: int | None = None
    last_sell_index: int | None = None
    buy_excursion = 0.0
    sell_excursion = 0.0

    def _armed(
        last_px: float,
        last_side_index: int | None,
        excursion: float,
        bar_atr: float,
    ) -> bool:
        if last_side_index is None:
            return True
        if rearm_max_bars is not None and i - last_side_index >= rearm_max_bars:
            return True
        if not np.isfinite(last_px) or not np.isfinite(bar_atr):
            return True
        threshold = rearm_atr * bar_atr
        if abs(close[i] - last_px) > threshold:
            return True
        return rearm_uses_path and excursion > threshold

    for i in range(buy.size):
        # Excursions accumulate on every bar, including ones the refractory
        # skips: the travel happened whether or not a signal could fire on it.
        if np.isfinite(last_buy_px):
            buy_excursion = max(buy_excursion, abs(close[i] - last_buy_px))
        if np.isfinite(last_sell_px):
            sell_excursion = max(sell_excursion, abs(close[i] - last_sell_px))

        if i - last_index < gap:
            continue
        bar_atr = atr[i]
        if buy[i]:
            if _armed(last_buy_px, last_buy_index, buy_excursion, bar_atr):
                kept_buy[i] = True
                last_index = i
                last_buy_px = close[i]
                last_buy_index = i
                buy_excursion = 0.0
        elif sell[i]:
            if _armed(last_sell_px, last_sell_index, sell_excursion, bar_atr):
                kept_sell[i] = True
                last_index = i
                last_sell_px = close[i]
                last_sell_index = i
                sell_excursion = 0.0
    return kept_buy, kept_sell


def _sequence_additions(
    existing: np.ndarray,
    buy: np.ndarray,
    sell: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    gap: int,
    rearm_atr: float,
    rearm_uses_path: bool = False,
    rearm_max_bars: int | None = None,
    *,
    preserve_future_events: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Add candidates while retaining finalized events.

    The legacy mode reserves space on both sides of an existing event. Causal
    additions must disable that lookahead: a future event cannot veto today.
    """
    kept_buy = np.zeros(buy.size, dtype=bool)
    kept_sell = np.zeros(sell.size, dtype=bool)
    existing_event = existing != NO_TURN
    blocked = existing_event.copy()
    for shift in range(1, gap):
        blocked[shift:] |= existing_event[:-shift]
        if preserve_future_events:
            blocked[:-shift] |= existing_event[shift:]
    buy &= ~blocked
    sell &= ~blocked

    last_index = -(10**9)
    last_buy_px = np.nan
    last_sell_px = np.nan
    last_buy_index: int | None = None
    last_sell_index: int | None = None
    buy_excursion = 0.0
    sell_excursion = 0.0

    def _armed(
        last_px: float,
        last_side_index: int | None,
        excursion: float,
        bar_atr: float,
    ) -> bool:
        if last_side_index is None:
            return True
        if rearm_max_bars is not None and i - last_side_index >= rearm_max_bars:
            return True
        if not np.isfinite(last_px) or not np.isfinite(bar_atr):
            return True
        threshold = rearm_atr * bar_atr
        if abs(close[i] - last_px) > threshold:
            return True
        return rearm_uses_path and excursion > threshold

    for i in range(buy.size):
        if np.isfinite(last_buy_px):
            buy_excursion = max(buy_excursion, abs(close[i] - last_buy_px))
        if np.isfinite(last_sell_px):
            sell_excursion = max(sell_excursion, abs(close[i] - last_sell_px))

        if existing_event[i]:
            last_index = i
            if existing[i] in (BUY_TURN, BOTTOM_TURN, TREND_BUY, BASE_BREAKOUT):
                last_buy_px = close[i]
                last_buy_index = i
                buy_excursion = 0.0
            elif existing[i] in (SELL_TURN, TOP_TURN):
                last_sell_px = close[i]
                last_sell_index = i
                sell_excursion = 0.0
            continue

        if i - last_index < gap:
            continue
        bar_atr = atr[i]
        if buy[i] and _armed(last_buy_px, last_buy_index, buy_excursion, bar_atr):
            kept_buy[i] = True
            last_index = i
            last_buy_px = close[i]
            last_buy_index = i
            buy_excursion = 0.0
        elif sell[i] and _armed(last_sell_px, last_sell_index, sell_excursion, bar_atr):
            kept_sell[i] = True
            last_index = i
            last_sell_px = close[i]
            last_sell_index = i
            sell_excursion = 0.0

    return kept_buy, kept_sell


def _conviction(frame: pd.DataFrame, cfg: "TurningPointConfig") -> tuple[np.ndarray, np.ndarray]:
    """Which bars the model considers candidate extremes.

    Prefers the dedicated detection head when its output is available, falling
    back to the state label for models trained before those heads existed.
    """
    has_head = cfg.pivot_gate_q is not None and {
        "_p_pivot_low_q",
        "_p_pivot_high_q",
    } <= set(frame.columns)
    if not has_head:
        action = frame["action"].to_numpy()
        return (
            np.isin(action, list(cfg.buy_states)),
            np.isin(action, list(cfg.sell_states)),
        )

    # Percentiles, never raw probabilities, so the gate is reproducible across
    # machines and retraining runs.
    low_q = frame["_p_pivot_low_q"].to_numpy()
    high_q = frame["_p_pivot_high_q"].to_numpy()
    buy = np.nan_to_num(low_q >= cfg.pivot_gate_q, nan=0.0).astype(bool)
    sell = np.nan_to_num(high_q >= cfg.pivot_gate_q, nan=0.0).astype(bool)

    # On a broad day the cross-sectional rank is exhausted before the real lows
    # are exhausted, so lower the bar in proportion to the breadth. See
    # ``breadth_relief``.
    if cfg.breadth_relief is not None:
        for column, side in (("_p_pivot_low_breadth", "buy"), ("_p_pivot_high_breadth", "sell")):
            if column not in frame.columns:
                continue
            lift = frame[column].to_numpy()
            relieved = cfg.pivot_gate_q - cfg.breadth_relief * np.maximum(0.0, lift - 1.0)
            relieved = np.maximum(relieved, cfg.breadth_floor)
            rank = low_q if side == "buy" else high_q
            hit = np.nan_to_num(rank >= relieved, nan=0.0).astype(bool)
            if side == "buy":
                buy |= hit
            else:
                sell |= hit

    # A second, parallel route for days when the cross-section is degenerate.
    if cfg.self_gate_low_q is not None and "_p_pivot_low_s" in frame.columns:
        buy |= np.nan_to_num(
            frame["_p_pivot_low_s"].to_numpy() >= cfg.self_gate_low_q, nan=0.0
        ).astype(bool)
    if cfg.self_gate_high_q is not None and "_p_pivot_high_s" in frame.columns:
        sell |= np.nan_to_num(
            frame["_p_pivot_high_s"].to_numpy() >= cfg.self_gate_high_q, nan=0.0
        ).astype(bool)

    # A third route for the case both ranks miss at once: the probability is
    # simply large against the panel it was produced on. See ``pivot_abs_mult``.
    if cfg.pivot_abs_mult is not None:
        for column, side in (("_p_pivot_low_abs", "buy"), ("_p_pivot_high_abs", "sell")):
            if column not in frame.columns:
                continue
            hit = np.nan_to_num(frame[column].to_numpy() >= cfg.pivot_abs_mult, nan=0.0).astype(
                bool
            )
            if side == "buy":
                buy |= hit
            else:
                sell |= hit

    return buy, sell


def _setup_alive(frame: pd.DataFrame, cfg: "TurningPointConfig") -> tuple[np.ndarray, np.ndarray]:
    """Bars on which a pending setup is still supported by the model.

    Deliberately looser than ``_conviction``: that gate decides whether a bar
    *starts* a setup, this one decides whether an existing setup has decayed.
    Requiring the full conviction threshold to hold every session would make
    the extension almost never fire, since the detection head spikes on the low
    and eases off while the base forms.

    The reading is ranked against the symbol's own history rather than the
    day's cross-section, for the same reason ``self_gate_low_q`` exists: the
    cross-sectional rank is a zero-sum resource that a market-wide sell-off
    exhausts, which is exactly when a setup is waiting to confirm.
    """
    n = len(frame)
    if cfg.setup_decay_q is None:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    def _alive(column: str) -> np.ndarray:
        if column not in frame.columns:
            return np.zeros(n, dtype=bool)
        return np.nan_to_num(
            frame[column].to_numpy(dtype="float64") >= cfg.setup_decay_q, nan=0.0
        ).astype(bool)

    # Prefer the self-relative rank; fall back to the cross-sectional one for
    # panels where the self gate is disabled.
    buy = _alive("_p_pivot_low_s")
    if not buy.any():
        buy = _alive("_p_pivot_low_q")
    sell = _alive("_p_pivot_high_s")
    if not sell.any():
        sell = _alive("_p_pivot_high_q")
    return buy, sell


def _range_position(
    frame: pd.DataFrame, window: int, min_periods: int | None = None
) -> np.ndarray:
    """Where the close sits between the rolling low and high.

    ``min_periods`` below ``window`` makes the window *expanding* while it fills.
    That matters because this runs on the prediction panel, which begins at each
    symbol's index-join date: demanding a full window blanked the gate for the
    first 20 (or 60) sessions of every newly-added name and of the panel itself.
    2.66% of rows carried no 20-day reading, and 478 confirmed pivots — 2.50% of
    all of them — were unreachable for that reason alone, 204 of which the
    detection head had already ranked above the 0.95 gate.
    """
    periods = window if min_periods is None else max(2, min(window, min_periods))
    low = frame["low"].rolling(window, min_periods=periods).min()
    high = frame["high"].rolling(window, min_periods=periods).max()
    span = (high - low).replace(0.0, np.nan)
    return ((frame["close"] - low) / span).to_numpy(dtype="float64")


def _self_relative_position(pos: np.ndarray, cfg: "TurningPointConfig") -> np.ndarray:
    """Rank ``pos`` against the symbol's own trailing history.

    Absolute range position is trend-dependent: a pullback inside a rally never
    reaches the bottom of the 20-day range, so an absolute floor silently
    excludes every low that forms in an uptrend. Ranking within the symbol's own
    recent distribution asks "low *for this stock lately*", which travels with
    the trend.
    """
    if cfg.self_range_window is None:
        return np.full(len(pos), np.nan)
    return (
        pd.Series(pos)
        .rolling(cfg.self_range_window, min_periods=cfg.self_range_min_periods)
        .rank(pct=True)
        .to_numpy(dtype="float64")
    )

def _recent(mask: np.ndarray, window: int) -> np.ndarray:
    """True where ``mask`` held on the current bar or any of the previous ``window``."""
    if window <= 0:
        return mask
    out = mask.copy()
    for shift in range(1, window + 1):
        out[shift:] |= mask[:-shift]
    return out


def _recent_while_live(
    mask: np.ndarray, alive: np.ndarray, window: int, max_window: int
) -> np.ndarray:
    """``_recent`` with the window extended while the setup is still live.

    A setup stays eligible for ``window`` bars unconditionally, and beyond that
    for as long as ``alive`` has held on *every* bar since — the model still
    ranking the name as a candidate extreme. ``max_window`` is a hard stop, so
    a permanently elevated reading cannot keep a stale setup armed forever.

    The extension is contiguous by construction: once ``alive`` breaks the
    setup expires, even if the reading recovers later. A separate rally is a
    separate setup, and reviving a lapsed one would reintroduce the "whole leg
    flagged" failure the refractory period exists to prevent.
    """
    if max_window <= window:
        return _recent(mask, window)

    out = _recent(mask, window)
    n = mask.size
    for i in np.flatnonzero(mask):
        # Walk forward from the end of the fixed window, stopping at the first
        # bar the model no longer endorses.
        stop = min(i + max_window, n - 1)
        for j in range(i + window + 1, stop + 1):
            if not alive[j]:
                break
            out[j] = True
    return out
