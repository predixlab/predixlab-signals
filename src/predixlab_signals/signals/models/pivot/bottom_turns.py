"""BOTTOM_TURN detection and BUY_TURN confirmation for the pivot model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.common import (
    _atr,
    _range_position,
    _recent_while_live,
    _sequence,
    _sequence_additions,
    _setup_alive,
    _trend_slope,
)
from predixlab_signals.signals.models.pivot.config import TurningPointConfig
from predixlab_signals.signals.models.pivot.events import (
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
)


@dataclass(frozen=True)
class BottomCandidateResult:
    actionable: np.ndarray
    detections: np.ndarray
    confirmation_candidates: np.ndarray
    low_conviction_guard: np.ndarray


@dataclass(frozen=True)
class BottomTriggerResult:
    actionable: np.ndarray
    delayed_rescue: np.ndarray
    broke_up: np.ndarray


def build_bottom_triggers(
    group: pd.DataFrame,
    conviction: np.ndarray,
    position: np.ndarray,
    self_position: np.ndarray,
    wide_reversal: np.ndarray,
    cfg: TurningPointConfig,
    *,
    has_confirmation: bool,
) -> BottomTriggerResult:
    """Build the complete bottom setup-to-trigger funnel."""
    n = len(group)
    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    in_trough = np.nan_to_num(
        position <= cfg.trough_pct,
        nan=0.0,
    ).astype(bool) | np.nan_to_num(
        self_position <= cfg.self_trough_pct,
        nan=0.0,
    ).astype(bool)
    setup = conviction & (in_trough | wide_reversal)
    rescue_setup = np.zeros(n, dtype=bool)
    delayed_setup = np.zeros(n, dtype=bool)
    if cfg.reversal_rescue_q is not None and "_p_pivot_low_q" in group:
        marginal = np.nan_to_num(
            group["_p_pivot_low_q"].to_numpy() >= cfg.reversal_rescue_q,
            nan=0.0,
        ).astype(bool)
        rescue_setup = marginal & ~conviction & (in_trough | wide_reversal)
    if cfg.delayed_reversal_rescue_q is not None and "_p_pivot_low_q" in group:
        delayed_marginal = np.nan_to_num(
            group["_p_pivot_low_q"].to_numpy()
            >= cfg.delayed_reversal_rescue_q,
            nan=0.0,
        ).astype(bool)
        delayed_setup = delayed_marginal & ~conviction & in_trough

    if cfg.context_window is not None:
        context = _range_position(
            group,
            cfg.context_window,
            cfg.context_min_periods,
        )
        known = np.isfinite(context)
        context_ok = ~known | (context <= 1.0 - cfg.context_buy_bound)
        setup &= context_ok
        rescue_setup &= context_ok
        delayed_setup &= context_ok

    broke_up = np.r_[False, close[1:] > high[:-1]]
    delayed_rescue = np.zeros(n, dtype=bool)
    if not cfg.require_reversal:
        return BottomTriggerResult(setup, delayed_rescue, broke_up)

    buy_alive, _ = _setup_alive(group, cfg)
    actionable = (
        _recent_while_live(
            setup,
            buy_alive,
            cfg.setup_window,
            cfg.setup_max_window,
        )
        & broke_up
    )
    if cfg.reversal_rescue_q is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
        proof = wide_reversal & np.nan_to_num(
            close_location >= cfg.reversal_rescue_close_loc,
            nan=0.0,
        ).astype(bool)
        actionable |= (
            _recent_while_live(
                rescue_setup,
                buy_alive,
                cfg.setup_window,
                cfg.setup_max_window,
            )
            & broke_up
            & proof
        )
    if (
        has_confirmation
        and cfg.delayed_reversal_rescue_q is not None
        and "_bottom_bars_since_low" in group
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
        return_2 = group["close"].pct_change(2).to_numpy()
        return_3 = group["close"].pct_change(3).to_numpy()
        return_20 = group["close"].pct_change(20).to_numpy()
        slope = _trend_slope(group["close"])
        strong_downtrend = np.zeros(n, dtype=bool)
        if cfg.min_downtrend_slope is not None:
            strong_downtrend |= np.isfinite(slope) & (
                slope <= cfg.min_downtrend_slope
            )
        if cfg.max_downtrend_return_20 is not None:
            strong_downtrend |= np.isfinite(return_20) & (
                return_20 <= cfg.max_downtrend_return_20
            )
        delayed_rescue = (
            _recent_while_live(
                delayed_setup,
                buy_alive,
                cfg.setup_window,
                cfg.setup_max_window,
            )
            & broke_up
            & strong_downtrend
            & np.nan_to_num(
                group["_bottom_bars_since_low"].to_numpy()
                <= cfg.delayed_reversal_rescue_max_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_2 >= cfg.delayed_reversal_rescue_return_2,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_3 >= cfg.delayed_reversal_rescue_return_3,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location >= cfg.delayed_reversal_rescue_close_loc,
                nan=0.0,
            ).astype(bool)
        )

    if cfg.trigger_pct is not None:
        atr = _atr(group)
        with np.errstate(invalid="ignore", divide="ignore"):
            move = np.abs(np.r_[0.0, np.diff(close)]) / np.where(
                np.isfinite(atr) & (atr > 0),
                atr,
                np.nan,
            )
        if cfg.reversal_waiver_atr is None:
            violent = np.zeros(n, dtype=bool)
        elif cfg.waiver_uses_bar_range:
            violent = wide_reversal
        else:
            violent = np.nan_to_num(
                move >= cfg.reversal_waiver_atr,
                nan=0.0,
            ).astype(bool)
        bounded = violent & np.nan_to_num(
            position <= cfg.waiver_pct,
            nan=0.0,
        ).astype(bool)
        actionable &= (
            np.nan_to_num(
                position <= cfg.trigger_pct,
                nan=0.0,
            ).astype(bool)
            | np.nan_to_num(
                self_position <= cfg.self_trigger_pct,
                nan=0.0,
            ).astype(bool)
            | bounded
        )
    return BottomTriggerResult(actionable, delayed_rescue, broke_up)


def prepare_bottom_candidates(
    group: pd.DataFrame,
    actionable: np.ndarray,
    slope: np.ndarray,
    cfg: TurningPointConfig,
    *,
    has_confirmation: bool,
) -> BottomCandidateResult:
    """Apply bottom-only trend and conviction guards before joint sequencing."""
    buy = actionable.copy()
    detections = np.zeros(len(group), dtype=bool)
    confirmation_candidates = np.zeros(len(group), dtype=bool)
    low_conviction_guard = np.zeros(len(group), dtype=bool)

    if cfg.min_downtrend_slope is not None:
        strong_downtrend = np.isfinite(slope) & (
            slope <= cfg.min_downtrend_slope
        )
        if cfg.max_downtrend_return_20 is not None:
            return_20 = group["close"].pct_change(20).to_numpy(dtype="float64")
            strong_downtrend |= np.isfinite(return_20) & (
                return_20 <= cfg.max_downtrend_return_20
            )
        detections = buy & strong_downtrend
        confirmation_candidates = detections.copy()
        buy &= ~strong_downtrend

    if (
        has_confirmation
        and cfg.bottom_guard_low_conviction_q is not None
        and "_p_pivot_low_q" in group
    ):
        return_3 = group["close"].pct_change(3).to_numpy()
        return_10 = group["close"].pct_change(10).to_numpy()
        return_20 = group["close"].pct_change(20).to_numpy()
        low_conviction = np.nan_to_num(
            group["_p_pivot_low_q"].to_numpy()
            < cfg.bottom_guard_low_conviction_q,
            nan=0.0,
        ).astype(bool)
        incomplete_reversal = (
            np.nan_to_num(
                return_3 < cfg.bottom_guard_max_return_3,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_10 < cfg.bottom_guard_max_return_10,
                nan=0.0,
            ).astype(bool)
        )
        stale_low_in_decline = (
            np.nan_to_num(
                group["_bottom_bars_since_low"].to_numpy()
                > cfg.bottom_guard_far_min_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_20 <= cfg.bottom_guard_far_max_return_20,
                nan=0.0,
            ).astype(bool)
        )
        low_conviction_guard = low_conviction & (
            incomplete_reversal | stale_low_in_decline
        )

    return BottomCandidateResult(
        actionable=buy,
        detections=detections,
        confirmation_candidates=confirmation_candidates,
        low_conviction_guard=low_conviction_guard,
    )


def build_bottom_reference_candidates(
    group: pd.DataFrame,
    cfg: TurningPointConfig,
    *,
    enabled: bool,
) -> np.ndarray:
    """Build causal regional bottom candidates independently of top rules."""
    candidates = np.zeros(len(group), dtype=bool)
    if (
        enabled
        and "_p_pivot_low_s" in group
        and "_bottom_bars_since_local_low" in group
    ):
        close = group["close"].to_numpy(dtype="float64")
        high = group["high"].to_numpy(dtype="float64")
        candidates = (
            np.nan_to_num(
                group["_p_pivot_low_s"].to_numpy()
                >= cfg.bottom_reference_self_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_bottom_bars_since_local_low"].to_numpy()
                <= cfg.bottom_reference_max_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change(2).to_numpy()
                >= cfg.bottom_reference_return_2,
                nan=0.0,
            ).astype(bool)
            & np.r_[False, close[1:] > high[:-1]]
        )
    return candidates


def build_bottom_right_rescue(
    group: pd.DataFrame,
    events: np.ndarray,
    broke_up: np.ndarray,
    cfg: TurningPointConfig,
    *,
    enabled: bool,
) -> np.ndarray:
    """Build right-side buy rescues on otherwise empty bars."""
    rescue = np.zeros(len(group), dtype=bool)
    if (
        enabled
        and {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns)
    ):
        close = group["close"].to_numpy(dtype="float64")
        high = group["high"].to_numpy(dtype="float64")
        low = group["low"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
            return_1 = group["close"].pct_change().to_numpy()
            return_2 = group["close"].pct_change(2).to_numpy()
            break_margin = (
                group["close"] / group["high"].shift() - 1.0
            ).to_numpy()
        rescue = (
            (events == NO_TURN)
            & broke_up
            & np.nan_to_num(
                group["_bottom_bars_since_local_low"].to_numpy()
                <= cfg.bottom_right_rescue_max_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_1 >= cfg.bottom_right_rescue_min_return_1,
                nan=0.0,
            ).astype(bool)
            & (
                np.nan_to_num(
                    return_2 >= cfg.bottom_right_rescue_min_return_2,
                    nan=0.0,
                ).astype(bool)
                | np.nan_to_num(
                    break_margin >= cfg.bottom_right_rescue_breakout_waiver,
                    nan=0.0,
                ).astype(bool)
            )
            & np.nan_to_num(
                close_location >= cfg.bottom_right_rescue_min_close_loc,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                >= cfg.bottom_right_rescue_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_s"].to_numpy()
                >= cfg.bottom_right_rescue_self_q,
                nan=0.0,
            ).astype(bool)
        )
    return rescue


@dataclass(frozen=True)
class BottomVolumeReboundShapes:
    strong_close: np.ndarray
    long_lower_wick: np.ndarray
    doji: np.ndarray


def build_bottom_volume_rebound_shapes(
    group: pd.DataFrame,
    cfg: TurningPointConfig,
) -> BottomVolumeReboundShapes:
    """Recognize bottom rejection without requiring a bullish body or daily gain."""
    long_lower_wick = np.zeros(len(group), dtype=bool)
    doji = np.zeros(len(group), dtype=bool)
    if not cfg.bottom_volume_rebound:
        return BottomVolumeReboundShapes(
            np.zeros(len(group), dtype=bool), long_lower_wick, doji
        )

    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    span = high - low
    valid = (
        np.isfinite(span)
        & (span > 0)
        & np.isfinite(close)
        & (close >= low)
        & (close <= high)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        close_location = (close - low) / span
    strong_close = valid & (
        close_location >= cfg.bottom_volume_rebound_min_close_loc
    )
    if (
        cfg.bottom_volume_rebound_min_lower_wick is None
        and cfg.bottom_volume_rebound_max_doji_body is None
    ):
        return BottomVolumeReboundShapes(strong_close, long_lower_wick, doji)
    if "open" not in group:
        raise ValueError(
            "Volume-backed wick/doji confirmation requires current-session open prices"
        )

    session_open = group["open"].to_numpy(dtype="float64")
    valid &= (
        np.isfinite(session_open)
        & (session_open >= low)
        & (session_open <= high)
    )
    strong_close &= valid
    body = np.abs(close - session_open)
    lower_wick = np.minimum(session_open, close) - low
    with np.errstate(invalid="ignore", divide="ignore"):
        lower_wick_ratio = lower_wick / span
        body_ratio = body / span
    if cfg.bottom_volume_rebound_min_lower_wick is not None:
        long_lower_wick = (
            valid
            & (lower_wick_ratio >= cfg.bottom_volume_rebound_min_lower_wick)
            & (lower_wick >= cfg.bottom_volume_rebound_lower_wick_body_ratio * body)
        )
    if cfg.bottom_volume_rebound_max_doji_body is not None:
        doji = (
            valid
            & (body_ratio <= cfg.bottom_volume_rebound_max_doji_body)
            & (lower_wick_ratio >= cfg.bottom_volume_rebound_doji_min_lower_wick)
        )
    return BottomVolumeReboundShapes(strong_close, long_lower_wick, doji)


def build_bottom_volume_rebound(
    group: pd.DataFrame,
    cfg: TurningPointConfig,
) -> np.ndarray:
    """Detect exceptional-volume bottom rejection near a fresh low after a decline."""
    if not cfg.bottom_volume_rebound:
        return np.zeros(len(group), dtype=bool)

    minimum_return = cfg.bottom_volume_rebound_min_return_1
    if (
        minimum_return is None
        or not np.isfinite(minimum_return)
        or minimum_return <= 0
    ):
        raise ValueError(
            "Volume-backed ordinary rebounds require a positive finite daily-return "
            "floor; only lower-wick or doji confirmation may bypass it"
        )

    window = cfg.bottom_volume_rebound_window
    volume = group["volume"].to_numpy(dtype="float64")
    prior_volume = (
        group["volume"].shift(1).rolling(window, min_periods=window).mean().to_numpy()
    )
    return_1 = group["close"].pct_change().to_numpy()
    prior_return = group["close"].shift(1).pct_change(window).to_numpy()
    range_low = group["low"].rolling(window, min_periods=window).min().to_numpy()
    recent_low = (
        group["low"]
        .rolling(cfg.bottom_volume_rebound_max_bars_since_low + 1, min_periods=1)
        .min()
        .to_numpy()
    )
    position = _range_position(group, window, window)
    shapes = build_bottom_volume_rebound_shapes(group, cfg)
    # Comparing price levels avoids cancellation at an exact 2.5% gain.
    previous_close = group["close"].shift(1).to_numpy(dtype="float64")
    ordinary_rebound = shapes.strong_close & (
        group["close"].to_numpy(dtype="float64")
        >= previous_close * (1.0 + minimum_return)
    )

    return (
        np.isfinite(volume)
        & np.isfinite(prior_volume)
        & (prior_volume > 0)
        & (volume >= cfg.bottom_volume_rebound_min_volume_ratio * prior_volume)
        & np.isfinite(return_1)
        & np.isfinite(prior_return)
        & (prior_return <= cfg.bottom_volume_rebound_max_prior_return)
        & (ordinary_rebound | shapes.long_lower_wick | shapes.doji)
        & (recent_low == range_low)
        & np.isfinite(position)
        & (position <= cfg.bottom_volume_rebound_max_position)
    )


def apply_bottom_volume_rebounds(
    group: pd.DataFrame,
    events: np.ndarray,
    cfg: TurningPointConfig,
) -> np.ndarray:
    """Confirm guarded bottoms or add causal buys without replacing other events."""
    if not cfg.bottom_volume_rebound:
        return events

    candidates = build_bottom_volume_rebound(group, cfg)
    confirmed = events.copy()
    confirmed[candidates & (events == BOTTOM_TURN)] = BUY_TURN
    additions, _ = _sequence_additions(
        confirmed,
        candidates & (events == NO_TURN),
        np.zeros(len(group), dtype=bool),
        group["close"].to_numpy(dtype="float64"),
        _atr(group),
        cfg.refractory,
        cfg.rearm_atr,
        cfg.rearm_uses_path,
        cfg.rearm_max_bars,
        preserve_future_events=False,
    )
    confirmed[additions] = BUY_TURN
    return confirmed


def finalize_bottom_events(
    group: pd.DataFrame,
    events: np.ndarray,
    dual_guard: np.ndarray,
    cfg: TurningPointConfig,
) -> np.ndarray:
    """Apply pivot-first promotions and terminal buy conflict filters."""
    filtered = events.copy()
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    close = group["close"].to_numpy(dtype="float64")
    confirmation = np.zeros(len(group), dtype=bool)
    if (
        cfg.bottom_pivot_fresh_q is not None
        and {
            "_p_pivot_low_q",
            "_p_pivot_low_s",
            "_p_pivot_high_q",
            "_bottom_bars_since_local_low",
        }
        <= set(group.columns)
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
        return_1 = group["close"].pct_change().to_numpy()
        return_3 = group["close"].pct_change(3).to_numpy()
        confirmation |= (
            (filtered == BOTTOM_TURN)
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                >= cfg.bottom_pivot_fresh_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_s"].to_numpy()
                >= cfg.bottom_pivot_fresh_self_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                - group["_p_pivot_high_q"].to_numpy()
                >= cfg.bottom_pivot_fresh_min_advantage,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_bottom_bars_since_local_low"].to_numpy()
                <= cfg.bottom_pivot_fresh_max_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_1 >= cfg.bottom_pivot_fresh_min_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_3 >= cfg.bottom_pivot_fresh_min_return_3,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location >= cfg.bottom_pivot_fresh_min_close_loc,
                nan=0.0,
            ).astype(bool)
        )
    if (
        cfg.bottom_pivot_capitulation_q is not None
        and {
            "_p_pivot_low_q",
            "_p_pivot_high_q",
            "_bottom_bars_since_local_low",
        }
        <= set(group.columns)
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
        confirmation |= (
            (filtered == BOTTOM_TURN)
            & dual_guard
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                >= cfg.bottom_pivot_capitulation_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                - group["_p_pivot_high_q"].to_numpy()
                >= cfg.bottom_pivot_capitulation_min_advantage,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_bottom_bars_since_local_low"].to_numpy()
                <= cfg.bottom_pivot_capitulation_max_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change().to_numpy()
                >= cfg.bottom_pivot_capitulation_min_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change(20).to_numpy()
                <= cfg.bottom_pivot_capitulation_max_return_20,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location >= cfg.bottom_pivot_capitulation_min_close_loc,
                nan=0.0,
            ).astype(bool)
        )
    filtered[confirmation] = BUY_TURN
    return apply_buy_conflict_filters(group, filtered, cfg)


def apply_buy_conflict_filters(
    group: pd.DataFrame,
    events: np.ndarray,
    cfg: TurningPointConfig,
) -> np.ndarray:
    """Demote mature or high-conflict buys while retaining bottom detection."""
    filtered = events.copy()
    if cfg.buy_high_conflict_window is not None and "_p_pivot_high_q" in group:
        position = _range_position(
            group,
            cfg.buy_high_conflict_window,
            cfg.buy_high_conflict_min_periods,
        )
        conflict = (
            (filtered == BUY_TURN)
            & np.nan_to_num(
                position >= cfg.buy_high_conflict_min_position,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change(5).to_numpy()
                >= cfg.buy_high_conflict_min_return_5,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                >= cfg.buy_high_conflict_pivot_q,
                nan=0.0,
            ).astype(bool)
        )
        filtered[conflict] = BOTTOM_TURN

    if (
        cfg.buy_late_rebound_min_body_return is not None
        and "_bottom_bars_since_local_low" in group
    ):
        session_open = np.full(len(group), np.nan, dtype="float64")
        if "open" in group:
            session_open = group["open"].to_numpy(dtype="float64")
        elif "open_next" in group:
            session_open = group["open_next"].shift().to_numpy(dtype="float64")

        close = group["close"].to_numpy(dtype="float64")
        high = group["high"].to_numpy(dtype="float64")
        low = group["low"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(span > 0, (close - low) / span, np.nan)
            body_return = close / session_open - 1.0
        mature_rebound = (
            (filtered == BUY_TURN)
            & np.nan_to_num(
                group["_bottom_bars_since_local_low"].to_numpy()
                >= cfg.buy_late_rebound_min_bars_since_low,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location >= cfg.buy_late_rebound_min_close_loc,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                body_return >= cfg.buy_late_rebound_min_body_return,
                nan=0.0,
            ).astype(bool)
        )
        filtered[mature_rebound] = BOTTOM_TURN
    return filtered


@dataclass(frozen=True)
class BottomTurnResult:
    actionable: np.ndarray
    detections: np.ndarray
    guarded: np.ndarray
    dual_guard: np.ndarray
    confirmed: np.ndarray
    confirmation_proof: np.ndarray
    reference_alternate_confirmation: np.ndarray
    setup_memory_proof: np.ndarray
    fresh_setup_proof: np.ndarray
    setup_rebound_proof: np.ndarray
    shock_confirmation_proof: np.ndarray


def evaluate_bottom_turns(
    group: pd.DataFrame,
    actionable: np.ndarray,
    detections: np.ndarray,
    confirmation_candidates: np.ndarray,
    low_conviction_guard: np.ndarray,
    cfg: TurningPointConfig,
    *,
    has_confirmation: bool,
) -> BottomTurnResult:
    """Confirm or guard bottom candidates without top-side policy."""
    n = len(group)
    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    buy = actionable.copy()
    bottom_only = detections.copy()
    bottom_confirmation_candidates = confirmation_candidates
    has_bottom_confirmation = has_confirmation

    low_conviction_bottom = buy & low_conviction_guard
    incomplete_history_bottom = np.zeros(n, dtype=bool)
    incomplete_confirmation_history = np.zeros(n, dtype=bool)
    if (
        has_bottom_confirmation
        and cfg.bottom_guard_requires_confirmation_history
        and "_p_pivot_low_s" in group
    ):
        incomplete_confirmation_history = (
            ~np.isfinite(group["_bottom_bars_since_low"].to_numpy())
            | ~np.isfinite(group["_p_pivot_low_s"].to_numpy())
        )
        incomplete_history_bottom = buy & incomplete_confirmation_history
    guarded_bottom = low_conviction_bottom | incomplete_history_bottom
    dual_guard_bottom = bottom_confirmation_candidates & (
        low_conviction_guard | incomplete_confirmation_history
    )
    buy &= ~guarded_bottom
    bottom_confirmed = np.zeros(n, dtype=bool)
    bottom_confirmation_proof = np.zeros(n, dtype=bool)
    bottom_alternate_confirmation = np.zeros(n, dtype=bool)
    bottom_reference_alternate_confirmation = np.zeros(n, dtype=bool)
    bottom_setup_memory_proof = np.zeros(n, dtype=bool)
    bottom_fresh_setup_proof = np.zeros(n, dtype=bool)
    bottom_setup_rebound_proof = np.zeros(n, dtype=bool)
    bottom_shock_confirmation_proof = np.zeros(n, dtype=bool)
    if has_bottom_confirmation:
        quiet_model_bottom = np.zeros(n, dtype=bool)
        if {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns):
            quiet_model_bottom = (
                np.nan_to_num(
                    group["_p_pivot_low_q"].to_numpy()
                    >= cfg.bottom_confirm_quiet_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_p_pivot_low_s"].to_numpy()
                    >= cfg.bottom_confirm_quiet_self_q,
                    nan=0.0,
                ).astype(bool)
            )
        bottom_atr = _atr(group)
        with np.errstate(invalid="ignore", divide="ignore"):
            bottom_range_atr = (high - low) / np.where(
                np.isfinite(bottom_atr) & (bottom_atr > 0),
                bottom_atr,
                np.nan,
            )
            bottom_span = high - low
            bottom_close_loc = np.where(
                bottom_span > 0,
                (close - low) / bottom_span,
                np.nan,
            )
        bottom_reversal_proof = (
            np.nan_to_num(
                bottom_range_atr >= cfg.bottom_confirm_reversal_atr,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                bottom_close_loc >= cfg.bottom_confirm_reversal_close_loc,
                nan=0.0,
            ).astype(bool)
        )
        panic_model_bottom = np.zeros(n, dtype=bool)
        if {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns):
            panic_model_bottom = (
                np.nan_to_num(
                    group["_p_pivot_low_q"].to_numpy()
                    >= cfg.bottom_confirm_panic_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_p_pivot_low_s"].to_numpy()
                    >= cfg.bottom_confirm_panic_self_q,
                    nan=0.0,
                ).astype(bool)
            )
        bottom_volatility = (
            np.nan_to_num(
                group["_top_volatility_q"].to_numpy()
                >= cfg.bottom_confirm_volatility_q,
                nan=0.0,
            ).astype(bool)
            | np.nan_to_num(
                group["_top_recent_volatility_q"].to_numpy()
                >= cfg.bottom_confirm_recent_volatility_q,
                nan=0.0,
            ).astype(bool)
        )
        bottom_model_support = np.zeros(n, dtype=bool)
        if "_p_pivot_low_s" in group:
            bottom_model_support = np.nan_to_num(
                group["_p_pivot_low_s"].to_numpy()
                >= cfg.bottom_confirm_volatility_self_q,
                nan=0.0,
            ).astype(bool)
        stable_model_bottom = np.zeros(n, dtype=bool)
        rebound_model_bottom = np.zeros(n, dtype=bool)
        if {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns):
            stable_model_bottom = (
                np.nan_to_num(
                    group["_p_pivot_low_q"].to_numpy()
                    >= cfg.bottom_confirm_stable_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_p_pivot_low_s"].to_numpy()
                    >= cfg.bottom_confirm_stable_self_q,
                    nan=0.0,
                ).astype(bool)
            )
            rebound_model_bottom = (
                np.nan_to_num(
                    group["_p_pivot_low_q"].to_numpy()
                    >= cfg.bottom_confirm_rebound_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_p_pivot_low_s"].to_numpy()
                    >= cfg.bottom_confirm_rebound_self_q,
                    nan=0.0,
                ).astype(bool)
            )
        bottom_return_1 = group["close"].pct_change().to_numpy()
        bottom_return_2 = group["close"].pct_change(2).to_numpy()
        bottom_return_3 = group["_top_return_3"].to_numpy()
        bottom_return_5 = group["_top_return_5"].to_numpy()
        bottom_return_10 = group["close"].pct_change(10).to_numpy()
        bottom_return_20 = group["close"].pct_change(20).to_numpy()
        recent_60_day_low = np.nan_to_num(
            group["_bottom_bars_since_low"].to_numpy()
            <= cfg.bottom_confirm_max_bars_since_low,
            nan=0.0,
        ).astype(bool)
        immediate_bottom = np.zeros(n, dtype=bool)
        delayed_bottom = np.zeros(n, dtype=bool)
        higher_low_bottom = np.zeros(n, dtype=bool)
        two_day_bottom = np.zeros(n, dtype=bool)
        rebound_bottom = (
            recent_60_day_low
            & np.nan_to_num(
                bottom_return_1 >= cfg.bottom_confirm_rebound_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                bottom_return_3 >= cfg.bottom_confirm_stable_return_3,
                nan=0.0,
            ).astype(bool)
            & rebound_model_bottom
        )
        panic_bottom = (
            recent_60_day_low
            & np.nan_to_num(
                bottom_return_1 >= cfg.bottom_confirm_panic_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                bottom_range_atr >= cfg.bottom_confirm_panic_atr,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                bottom_close_loc >= cfg.bottom_confirm_panic_close_loc,
                nan=0.0,
            ).astype(bool)
            & panic_model_bottom
        )
        setup_memory_bottom = np.zeros(n, dtype=bool)
        fresh_setup_bottom = np.zeros(n, dtype=bool)
        if {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns):
            recent_low_q = (
                group["_p_pivot_low_q"]
                .rolling(
                    cfg.bottom_confirm_setup_memory_window,
                    min_periods=1,
                )
                .max()
                .to_numpy()
            )
            recent_low_self_q = (
                group["_p_pivot_low_s"]
                .rolling(
                    cfg.bottom_confirm_setup_memory_window,
                    min_periods=1,
                )
                .max()
                .to_numpy()
            )
            bottom_setup_memory_proof = (
                recent_60_day_low
                & np.nan_to_num(
                    recent_low_q
                    >= cfg.bottom_confirm_setup_memory_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    recent_low_self_q
                    >= cfg.bottom_confirm_setup_memory_self_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_p_pivot_low_q"].to_numpy()
                    <= cfg.bottom_confirm_setup_memory_max_current_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_return_1
                    >= cfg.bottom_confirm_setup_memory_return_1,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_return_3
                    >= cfg.bottom_confirm_setup_memory_min_return_3,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_return_10
                    <= cfg.bottom_confirm_setup_memory_max_return_10,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_return_20
                    <= cfg.bottom_confirm_setup_memory_max_return_20,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_close_loc
                    >= cfg.bottom_confirm_setup_memory_close_loc,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    group["_top_recent_volatility_q"].to_numpy()
                    >= cfg.bottom_confirm_setup_memory_volatility_q,
                    nan=0.0,
                ).astype(bool)
            )
            setup_memory_bottom = (
                low_conviction_bottom & bottom_setup_memory_proof
            )
            if cfg.bottom_confirm_setup_rebound_pivot_q is not None:
                bottom_setup_rebound_proof = (
                    np.nan_to_num(
                        group["_bottom_bars_since_low"].to_numpy()
                        <= cfg.bottom_confirm_setup_rebound_max_bars_since_low,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        recent_low_q
                        >= cfg.bottom_confirm_setup_rebound_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        recent_low_self_q
                        >= cfg.bottom_confirm_setup_rebound_self_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_2
                        >= cfg.bottom_confirm_setup_rebound_return_2,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_20
                        <= cfg.bottom_confirm_setup_rebound_max_return_20,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_close_loc
                        >= cfg.bottom_confirm_setup_rebound_close_loc,
                        nan=0.0,
                    ).astype(bool)
                )
            if cfg.bottom_confirm_fresh_setup_self_q is not None:
                bottom_fresh_setup_proof = (
                    np.nan_to_num(
                        group["_bottom_bars_since_low"].to_numpy()
                        <= cfg.bottom_confirm_fresh_setup_max_bars_since_low,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        recent_low_self_q
                        >= cfg.bottom_confirm_fresh_setup_self_q,
                        nan=0.0,
                    ).astype(bool)
                    & (
                        np.nan_to_num(
                            bottom_return_3
                            >= cfg.bottom_confirm_fresh_setup_min_return_3,
                            nan=0.0,
                        ).astype(bool)
                        | np.nan_to_num(
                            bottom_range_atr
                            >= cfg.bottom_confirm_fresh_setup_reversal_atr,
                            nan=0.0,
                        ).astype(bool)
                    )
                )
                fresh_setup_bottom = (
                    bottom_confirmation_candidates | low_conviction_bottom
                ) & bottom_fresh_setup_proof
        if {"_p_pivot_low_q", "_p_pivot_low_s"} <= set(group.columns):
            low_q = group["_p_pivot_low_q"].to_numpy()
            low_self_q = group["_p_pivot_low_s"].to_numpy()
            if cfg.bottom_confirm_shock_pivot_q is not None:
                bottom_shock_confirmation_proof = (
                    np.nan_to_num(
                        group["_bottom_bars_since_low"].to_numpy()
                        <= cfg.bottom_confirm_shock_max_bars_since_low,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_1 >= cfg.bottom_confirm_shock_return_1,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_1
                        <= cfg.bottom_confirm_shock_max_return_1,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_20
                        <= cfg.bottom_confirm_shock_max_return_20,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_close_loc
                        >= cfg.bottom_confirm_shock_close_loc,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        low_q >= cfg.bottom_confirm_shock_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        group["_top_recent_volatility_q"].to_numpy()
                        >= cfg.bottom_confirm_shock_volatility_q,
                        nan=0.0,
                    ).astype(bool)
                )
            immediate_bottom = (
                recent_60_day_low
                & np.nan_to_num(
                    bottom_return_3
                    >= cfg.bottom_confirm_immediate_return_3,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_close_loc
                    >= cfg.bottom_confirm_immediate_close_loc,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    low_q >= cfg.bottom_confirm_immediate_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    low_self_q >= cfg.bottom_confirm_immediate_self_q,
                    nan=0.0,
                ).astype(bool)
            )
            delayed_bottom = (
                np.nan_to_num(
                    group["_bottom_bars_since_local_low"].to_numpy()
                    <= cfg.bottom_confirm_local_max_bars,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    bottom_return_3
                    >= cfg.bottom_confirm_delayed_return_3,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    low_q >= cfg.bottom_confirm_delayed_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    low_self_q >= cfg.bottom_confirm_delayed_self_q,
                    nan=0.0,
                ).astype(bool)
            )
            if cfg.bottom_confirm_higher_low_pivot_q is not None:
                higher_low_bottom = (
                    ~recent_60_day_low
                    & np.nan_to_num(
                        group["_bottom_bars_since_local_low"].to_numpy()
                        <= cfg.bottom_confirm_higher_low_max_local_bars,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_1
                        >= cfg.bottom_confirm_higher_low_return_1,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_3
                        >= cfg.bottom_confirm_higher_low_return_3,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_5
                        >= cfg.bottom_confirm_higher_low_return_5,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_20
                        >= cfg.bottom_confirm_higher_low_return_20,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_close_loc
                        >= cfg.bottom_confirm_higher_low_close_loc,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        low_q >= cfg.bottom_confirm_higher_low_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        low_self_q >= cfg.bottom_confirm_higher_low_self_q,
                        nan=0.0,
                    ).astype(bool)
                )
            if cfg.bottom_confirm_two_day_self_q is not None:
                two_day_bottom = (
                    np.isfinite(
                        group["_bottom_bars_since_low"].to_numpy()
                    )
                    & np.nan_to_num(
                        group["_bottom_bars_since_local_low"].to_numpy()
                        <= cfg.bottom_confirm_two_day_max_local_bars,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        bottom_return_2
                        >= cfg.bottom_confirm_two_day_return_2,
                        nan=0.0,
                    ).astype(bool)
                    & (
                        np.nan_to_num(
                            bottom_return_20
                            > cfg.bottom_confirm_two_day_deep_return_20,
                            nan=0.0,
                        ).astype(bool)
                        | np.nan_to_num(
                            bottom_return_2
                            >= cfg.bottom_confirm_two_day_deep_return_2,
                            nan=0.0,
                        ).astype(bool)
                    )
                    & np.nan_to_num(
                        bottom_range_atr
                        >= cfg.bottom_confirm_two_day_range_atr,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        low_self_q >= cfg.bottom_confirm_two_day_self_q,
                        nan=0.0,
                    ).astype(bool)
                )
        bottom_return_5_ok = np.nan_to_num(
            bottom_return_5 >= cfg.bottom_confirm_min_return_5,
            nan=0.0,
        ).astype(bool)
        strong_reversal_waiver = np.zeros(n, dtype=bool)
        if cfg.bottom_confirm_reversal_waives_return_5:
            strong_reversal_waiver = (
                recent_60_day_low
                & np.nan_to_num(
                    bottom_return_3 >= cfg.bottom_confirm_min_return_3,
                    nan=0.0,
                ).astype(bool)
                & bottom_reversal_proof
            )
        bottom_confirmation_proof = (
            (
                bottom_return_5_ok
                & (
                    (
                        recent_60_day_low
                        & (
                            np.nan_to_num(
                                bottom_return_3
                                >= cfg.bottom_confirm_min_return_3,
                                nan=0.0,
                            ).astype(bool)
                            & (
                                (bottom_volatility & bottom_model_support)
                                | quiet_model_bottom
                                | bottom_reversal_proof
                            )
                        )
                    )
                    | (
                        recent_60_day_low
                        & np.nan_to_num(
                            bottom_return_3
                            >= cfg.bottom_confirm_stable_return_3,
                            nan=0.0,
                        ).astype(bool)
                        & stable_model_bottom
                    )
                    | panic_bottom
                    | rebound_bottom
                    | immediate_bottom
                    | delayed_bottom
                )
            )
            | strong_reversal_waiver
        )
        bottom_alternate_confirmation = higher_low_bottom | two_day_bottom
        # Reference candidates already require local-low recency and a
        # two-session rebound. Promote the subset whose rebound reaches the
        # existing deep-reversal threshold; weaker two-day moves still need
        # an independent evidence family and remain reference detections.
        bottom_reference_alternate_confirmation = higher_low_bottom | (
            two_day_bottom
            & np.nan_to_num(
                bottom_return_2
                >= cfg.bottom_confirm_two_day_deep_return_2,
                nan=0.0,
            ).astype(bool)
        )
        bottom_confirmed = (
            bottom_confirmation_candidates
            & (bottom_confirmation_proof | bottom_alternate_confirmation)
        ) | setup_memory_bottom | fresh_setup_bottom
    bottom_only, _ = _sequence(
        bottom_only,
        np.zeros(n, dtype=bool),
        close,
        _atr(group),
        cfg.refractory,
        cfg.rearm_atr,
        cfg.rearm_uses_path,
        cfg.rearm_max_bars,
    )
    bottom_confirmed, _ = _sequence(
        bottom_confirmed,
        np.zeros(n, dtype=bool),
        close,
        _atr(group),
        cfg.refractory,
        cfg.rearm_atr,
        cfg.rearm_uses_path,
        cfg.rearm_max_bars,
    )
    bottom_only &= ~bottom_confirmed

    return BottomTurnResult(
        actionable=buy,
        detections=bottom_only,
        guarded=guarded_bottom,
        dual_guard=dual_guard_bottom,
        confirmed=bottom_confirmed,
        confirmation_proof=bottom_confirmation_proof,
        reference_alternate_confirmation=bottom_reference_alternate_confirmation,
        setup_memory_proof=bottom_setup_memory_proof,
        fresh_setup_proof=bottom_fresh_setup_proof,
        setup_rebound_proof=bottom_setup_rebound_proof,
        shock_confirmation_proof=bottom_shock_confirmation_proof,
    )
