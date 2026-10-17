"""TOP_TURN detection and SELL_TURN confirmation for the pivot model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from predixlab_signals.signals.models.pivot.common import (
    _atr,
    _range_position,
    _recent_while_live,
    _sequence,
    _setup_alive,
)
from predixlab_signals.signals.models.pivot.config import TurningPointConfig
from predixlab_signals.signals.models.pivot.events import (
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
)


@dataclass(frozen=True)
class TopCandidateResult:
    actionable: np.ndarray
    detections: np.ndarray
    confirmation_candidates: np.ndarray


@dataclass(frozen=True)
class TopTriggerResult:
    actionable: np.ndarray
    confirmation_source: np.ndarray
    setup: np.ndarray
    broke_down: np.ndarray


def build_top_triggers(
    group: pd.DataFrame,
    conviction: np.ndarray,
    position: np.ndarray,
    self_position: np.ndarray,
    wide_reversal: np.ndarray,
    cfg: TurningPointConfig,
) -> TopTriggerResult:
    """Build the complete top setup-to-trigger funnel."""
    n = len(group)
    close = group["close"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    in_peak = np.nan_to_num(
        position >= cfg.peak_pct,
        nan=0.0,
    ).astype(bool) | np.nan_to_num(
        self_position >= cfg.self_peak_pct,
        nan=0.0,
    ).astype(bool)
    setup = conviction & (in_peak | wide_reversal)
    if cfg.context_window is not None:
        context = _range_position(
            group,
            cfg.context_window,
            cfg.context_min_periods,
        )
        known = np.isfinite(context)
        setup &= ~known | (context >= cfg.context_sell_bound)

    broke_down = np.r_[False, close[1:] < low[:-1]]
    if not cfg.require_reversal:
        return TopTriggerResult(setup, setup.copy(), setup, broke_down)

    _, sell_alive = _setup_alive(group, cfg)
    actionable = (
        _recent_while_live(
            setup,
            sell_alive,
            cfg.setup_window,
            cfg.setup_max_window,
        )
        & broke_down
    )
    confirmation_source = (
        _recent_while_live(
            setup,
            sell_alive,
            cfg.top_setup_window,
            cfg.setup_max_window,
        )
        & broke_down
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
            position >= 1.0 - cfg.waiver_pct,
            nan=0.0,
        ).astype(bool)
        trigger = (
            np.nan_to_num(
                position >= 1.0 - cfg.trigger_pct,
                nan=0.0,
            ).astype(bool)
            | np.nan_to_num(
                self_position >= 1.0 - cfg.self_trigger_pct,
                nan=0.0,
            ).astype(bool)
            | bounded
        )
        actionable &= trigger
        confirmation_source &= trigger
    return TopTriggerResult(
        actionable,
        confirmation_source,
        setup,
        broke_down,
    )


@dataclass(frozen=True)
class GuardedTopResult:
    actionable: np.ndarray
    guarded: np.ndarray
    guard_mask: np.ndarray
    reversal_proof: np.ndarray


def prepare_top_candidates(
    group: pd.DataFrame,
    actionable: np.ndarray,
    confirmation_source: np.ndarray,
    slope: np.ndarray,
    cfg: TurningPointConfig,
) -> TopCandidateResult:
    """Apply the top-only uptrend guard before joint sequencing."""
    sell = actionable.copy()
    detections = np.zeros(len(group), dtype=bool)
    confirmation_candidates = np.zeros(len(group), dtype=bool)
    if cfg.max_uptrend_slope is not None:
        trend_ok = np.nan_to_num(
            slope < cfg.max_uptrend_slope,
            nan=0.0,
        ).astype(bool)
        detections = sell & ~trend_ok
        confirmation_candidates = confirmation_source & ~trend_ok
        sell &= trend_ok
    return TopCandidateResult(
        actionable=sell,
        detections=detections,
        confirmation_candidates=confirmation_candidates,
    )


def guard_top_candidates(
    group: pd.DataFrame,
    actionable: np.ndarray,
    cfg: TurningPointConfig,
    *,
    has_confirmation: bool,
) -> GuardedTopResult:
    """Retain weak or stale sell candidates as non-actionable top detections."""
    sell = actionable.copy()
    n = len(group)
    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    guarded = np.zeros(n, dtype=bool)
    guard_mask = np.zeros(n, dtype=bool)
    reversal_proof = np.nan_to_num(
        group["close"].pct_change().to_numpy()
        <= cfg.top_confirm_reversal_return_1,
        nan=0.0,
    ).astype(bool)
    if (
        has_confirmation
        and cfg.top_guard_enabled
        and "_p_pivot_high_q" in group
    ):
        high_q = group["_p_pivot_high_q"].to_numpy()
        bars_since_high = group["_top_bars_since_high"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
            atr = _atr(group)
            range_atr = span / np.where(
                np.isfinite(atr) & (atr > 0),
                atr,
                np.nan,
            )
        weak_boundary = (
            np.nan_to_num(
                bars_since_high == cfg.top_guard_boundary_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                high_q < cfg.top_guard_boundary_pivot_q,
                nan=0.0,
            ).astype(bool)
        )
        weak_wide_boundary = (
            np.nan_to_num(
                bars_since_high == cfg.top_guard_boundary_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                high_q < cfg.top_guard_wide_boundary_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location < cfg.top_guard_wide_boundary_close_loc,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                range_atr >= cfg.top_guard_wide_boundary_atr,
                nan=0.0,
            ).astype(bool)
        )
        obsolete_high = (
            np.nan_to_num(
                bars_since_high > cfg.top_guard_obsolete_min_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                high_q < cfg.top_guard_obsolete_pivot_q,
                nan=0.0,
            ).astype(bool)
        )
        weak_stale_close = (
            np.nan_to_num(
                bars_since_high > cfg.top_guard_stale_min_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                bars_since_high <= cfg.top_guard_stale_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                high_q < cfg.top_guard_stale_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location < cfg.top_guard_stale_close_loc,
                nan=0.0,
            ).astype(bool)
        )
        guard_mask = (
            weak_boundary
            | weak_wide_boundary
            | obsolete_high
            | weak_stale_close
        ) & ~reversal_proof
        guarded = sell & guard_mask
        sell &= ~guarded
    return GuardedTopResult(
        actionable=sell,
        guarded=guarded,
        guard_mask=guard_mask,
        reversal_proof=reversal_proof,
    )


def build_top_reference_candidates(
    group: pd.DataFrame,
    sell_conviction: np.ndarray,
    sell_setup: np.ndarray,
    cfg: TurningPointConfig,
    *,
    enabled: bool,
) -> np.ndarray:
    """Build causal regional top candidates independently of bottom rules."""
    candidates = np.zeros(len(group), dtype=bool)
    if (
        enabled
        and "_p_pivot_high_q" in group
        and "_top_bars_since_high" in group
    ):
        high = group["high"].to_numpy(dtype="float64")
        low = group["low"].to_numpy(dtype="float64")
        close = group["close"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
        candidates = (
            sell_conviction
            & ~sell_setup
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                >= cfg.top_reference_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_bars_since_high"].to_numpy()
                <= cfg.top_reference_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change().to_numpy()
                <= cfg.top_reference_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location <= cfg.top_reference_close_loc,
                nan=0.0,
            ).astype(bool)
        )
    return candidates


def build_top_right_rescue(
    group: pd.DataFrame,
    events: np.ndarray,
    broke_down: np.ndarray,
    cfg: TurningPointConfig,
    *,
    enabled: bool,
) -> np.ndarray:
    """Build right-side sell rescues on otherwise empty bars."""
    rescue = np.zeros(len(group), dtype=bool)
    if (
        enabled
        and {"_p_pivot_high_q", "_p_pivot_high_s"} <= set(group.columns)
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
        rescue = (
            (events == NO_TURN)
            & broke_down
            & np.nan_to_num(
                group["_top_bars_since_local_high"].to_numpy()
                <= cfg.top_right_rescue_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                return_1 <= cfg.top_right_rescue_max_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location <= cfg.top_right_rescue_max_close_loc,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                >= cfg.top_right_rescue_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_s"].to_numpy()
                >= cfg.top_right_rescue_self_q,
                nan=0.0,
            ).astype(bool)
        )
    return rescue


def finalize_top_events(
    group: pd.DataFrame,
    events: np.ndarray,
    cfg: TurningPointConfig,
    *,
    local_confirmation: np.ndarray,
    crash_confirmation: np.ndarray,
    uptrend_action_confirmation: np.ndarray,
    right_rescue: np.ndarray,
    has_right_reversal: bool,
) -> np.ndarray:
    """Apply terminal sell conflicts and delayed follow-through confirmation."""
    filtered = events.copy()
    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    action = group["action"].to_numpy()
    protected = (
        local_confirmation
        | crash_confirmation
        | uptrend_action_confirmation
        | right_rescue
    )
    if (
        has_right_reversal
        and {"_p_pivot_high_q", "_p_pivot_high_s"} <= set(group.columns)
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            span = high - low
            close_location = np.where(
                span > 0,
                (close - low) / span,
                np.nan,
            )
            local_high = (
                group["high"]
                .rolling(
                    cfg.top_confirm_memory_high_window,
                    min_periods=cfg.top_confirm_memory_high_window,
                )
                .max()
                .to_numpy()
            )
            local_high_drawdown = close / local_high - 1.0
        protected |= (
            np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                >= cfg.sell_right_reversal_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_s"].to_numpy()
                >= cfg.sell_right_reversal_self_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_bars_since_local_high"].to_numpy()
                <= cfg.sell_right_reversal_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                close_location <= cfg.sell_right_reversal_max_close_loc,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change().to_numpy()
                <= cfg.sell_right_reversal_max_return_1,
                nan=0.0,
            ).astype(bool)
            & (
                np.nan_to_num(
                    group["_top_return_3"].to_numpy()
                    <= cfg.sell_right_reversal_max_return_3,
                    nan=0.0,
                ).astype(bool)
                | np.nan_to_num(
                    local_high_drawdown <= cfg.sell_right_reversal_max_drawdown,
                    nan=0.0,
                ).astype(bool)
            )
        )

    conflict = np.zeros(len(group), dtype=bool)
    if cfg.sell_trend_conflict_enabled:
        non_bearish_state = np.isin(action, ("HOLD", "BUY", "STRONG_BUY"))
        conflict |= (
            non_bearish_state
            & np.nan_to_num(
                group["close"].pct_change(10).to_numpy()
                > cfg.sell_trend_conflict_min_return_10,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change(20).to_numpy()
                > cfg.sell_trend_conflict_min_return_20,
                nan=0.0,
            ).astype(bool)
        )
    if (
        cfg.sell_stale_conflict_pivot_q is not None
        and "_p_pivot_high_q" in group
    ):
        conflict |= (
            np.nan_to_num(
                group["_top_bars_since_high"].to_numpy()
                > cfg.sell_stale_conflict_min_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change(10).to_numpy()
                > cfg.sell_stale_conflict_min_return_10,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                < cfg.sell_stale_conflict_pivot_q,
                nan=0.0,
            ).astype(bool)
        )
    filtered[(filtered == SELL_TURN) & conflict & ~protected] = TOP_TURN

    if (
        cfg.sell_low_context_window is not None
        and {"_p_pivot_low_q", "_p_pivot_high_q"} <= set(group.columns)
    ):
        long_context = _range_position(
            group,
            cfg.sell_low_context_window,
            cfg.sell_low_context_min_periods,
        )
        low_context_conflict = (
            (filtered == SELL_TURN)
            & np.nan_to_num(
                long_context <= cfg.sell_low_context_bound,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_low_q"].to_numpy()
                >= group["_p_pivot_high_q"].to_numpy(),
                nan=0.0,
            ).astype(bool)
        )
        filtered[low_context_conflict] = TOP_TURN

    filtered[_top_followthrough_confirmations(group, filtered, cfg)] = SELL_TURN
    return filtered


def _top_followthrough_confirmations(
    group: pd.DataFrame,
    pivot_turn: np.ndarray,
    cfg: "TurningPointConfig",
) -> np.ndarray:
    """Confirm strong top detections after a lower-low follow-through bar."""
    confirmed = np.zeros(len(group), dtype=bool)
    if (
        cfg.top_followthrough_window is None
        or not len(group)
        or {"_p_pivot_high_q", "_p_pivot_high_s"} - set(group.columns)
    ):
        return confirmed

    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    high_q = group["_p_pivot_high_q"].to_numpy()
    high_self_q = group["_p_pivot_high_s"].to_numpy()
    bars_since_local_high = group["_top_bars_since_local_high"].to_numpy()
    return_1 = group["close"].pct_change().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        span = high - low
        close_location = np.where(span > 0, (close - low) / span, np.nan)

    for source in np.flatnonzero(pivot_turn == TOP_TURN):
        source_ranks = (high_q[source], high_self_q[source])
        if (
            not np.all(np.isfinite(source_ranks))
            or min(source_ranks) < cfg.top_followthrough_min_rank
            or max(source_ranks) < cfg.top_followthrough_max_rank
            or not np.isfinite(bars_since_local_high[source])
            or bars_since_local_high[source]
            > cfg.top_followthrough_max_source_bars_since_high
        ):
            continue
        stop = min(source + cfg.top_followthrough_window, len(group) - 1)
        for current in range(source + 1, stop + 1):
            if pivot_turn[current] != NO_TURN:
                continue
            cumulative_return = close[current] / close[source] - 1.0
            source_low_break = close[current] / low[source] - 1.0
            if (
                cumulative_return <= cfg.top_followthrough_max_cumulative_return
                and return_1[current] <= cfg.top_followthrough_max_return_1
                and source_low_break
                <= cfg.top_followthrough_max_source_low_break
                and close_location[current]
                <= cfg.top_followthrough_max_close_loc
            ):
                confirmed[current] = True
                break
    return confirmed


@dataclass(frozen=True)
class TopTurnResult:
    detections: np.ndarray
    guarded: np.ndarray
    guarded_confirmed: np.ndarray
    unsupported: np.ndarray
    confirmed: np.ndarray
    confirmation_proof: np.ndarray
    memory_confirmation_proof: np.ndarray
    local_confirmation_proof: np.ndarray
    crash_confirmation_proof: np.ndarray
    uptrend_action_confirmation: np.ndarray
    model_support: np.ndarray


def evaluate_top_turns(
    group: pd.DataFrame,
    detections: np.ndarray,
    confirmation_candidates: np.ndarray,
    guarded: np.ndarray,
    guard_mask: np.ndarray,
    reversal_proof: np.ndarray,
    cfg: TurningPointConfig,
    *,
    has_confirmation: bool,
) -> TopTurnResult:
    """Confirm or retain top candidates without bottom-side policy."""
    n = len(group)
    close = group["close"].to_numpy(dtype="float64")
    high = group["high"].to_numpy(dtype="float64")
    low = group["low"].to_numpy(dtype="float64")
    action = group["action"].to_numpy()
    top_only = detections.copy()
    top_confirmation_candidates = confirmation_candidates
    guarded_top = guarded.copy()
    top_guard_mask = guard_mask
    top_reversal_proof = reversal_proof
    has_top_confirmation = has_confirmation

    top_confirmed = np.zeros(n, dtype=bool)
    top_confirmation_proof = np.zeros(n, dtype=bool)
    top_memory_confirmation_proof = np.zeros(n, dtype=bool)
    top_local_confirmation_proof = np.zeros(n, dtype=bool)
    top_crash_confirmation_proof = np.zeros(n, dtype=bool)
    top_uptrend_action_confirmation = np.zeros(n, dtype=bool)
    top_model_support = np.ones(n, dtype=bool)
    if has_top_confirmation:
        quiet_model_top = np.zeros(n, dtype=bool)
        if {"_p_pivot_high_q", "_p_pivot_high_s"} <= set(group.columns):
            high_q = group["_p_pivot_high_q"].to_numpy()
            high_self_q = group["_p_pivot_high_s"].to_numpy()
            quiet_model_top = (
                np.nan_to_num(
                    high_q >= cfg.top_confirm_quiet_pivot_q,
                    nan=0.0,
                ).astype(bool)
                & np.nan_to_num(
                    high_self_q >= cfg.top_confirm_quiet_self_q,
                    nan=0.0,
                ).astype(bool)
            )
            model_support_paths = []
            if cfg.top_confirm_model_pivot_q is not None:
                model_support_paths.append(
                    np.nan_to_num(
                        high_q >= cfg.top_confirm_model_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                )
            if cfg.top_confirm_model_self_q is not None:
                model_support_paths.append(
                    np.nan_to_num(
                        high_self_q >= cfg.top_confirm_model_self_q,
                        nan=0.0,
                    ).astype(bool)
                )
            if model_support_paths:
                top_model_support = np.logical_or.reduce(model_support_paths)
            if cfg.top_confirm_local_pivot_q is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    local_top_span = high - low
                    local_top_close_loc = np.where(
                        local_top_span > 0,
                        (close - low) / local_top_span,
                        np.nan,
                    )
                top_local_confirmation_proof = (
                    np.nan_to_num(
                        group["_top_bars_since_local_high"].to_numpy()
                        <= cfg.top_confirm_local_max_bars_since_high,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        group["close"].pct_change().to_numpy()
                        <= cfg.top_confirm_local_return_1,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        local_top_close_loc
                        <= cfg.top_confirm_local_close_loc,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        high_q >= cfg.top_confirm_local_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        high_self_q >= cfg.top_confirm_local_self_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        group["_top_recent_volatility_q"].to_numpy()
                        >= cfg.top_confirm_local_volatility_q,
                        nan=0.0,
                    ).astype(bool)
                )
            if cfg.top_confirm_crash_pivot_q is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    crash_top_span = high - low
                    crash_top_close_loc = np.where(
                        crash_top_span > 0,
                        (close - low) / crash_top_span,
                        np.nan,
                    )
                top_crash_confirmation_proof = (
                    np.nan_to_num(
                        group["_top_bars_since_local_high"].to_numpy()
                        <= cfg.top_confirm_crash_max_bars_since_high,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        group["close"].pct_change().to_numpy()
                        <= cfg.top_confirm_crash_return_1,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        crash_top_close_loc
                        <= cfg.top_confirm_crash_close_loc,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        high_q >= cfg.top_confirm_crash_pivot_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        high_self_q >= cfg.top_confirm_crash_self_q,
                        nan=0.0,
                    ).astype(bool)
                    & np.nan_to_num(
                        group["_top_recent_volatility_q"].to_numpy()
                        >= cfg.top_confirm_crash_volatility_q,
                        nan=0.0,
                    ).astype(bool)
                )
        top_model_support |= top_reversal_proof
        top_confirmation_proof = (
            np.nan_to_num(
                group["_top_bars_since_high"].to_numpy()
                <= cfg.top_confirm_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_return_3"].to_numpy()
                <= cfg.top_confirm_max_return_3,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_return_5"].to_numpy()
                <= cfg.top_confirm_max_return_5,
                nan=0.0,
            ).astype(bool)
            & (
                np.nan_to_num(
                    group["_top_volatility_q"].to_numpy()
                    >= cfg.top_confirm_volatility_q,
                    nan=0.0,
                ).astype(bool)
                | np.nan_to_num(
                    group["_top_recent_volatility_q"].to_numpy()
                    >= cfg.top_confirm_recent_volatility_q,
                    nan=0.0,
                ).astype(bool)
                | quiet_model_top
            )
        )
        top_confirmed = top_confirmation_candidates & top_confirmation_proof
    _, top_only = _sequence(
        np.zeros(n, dtype=bool),
        top_only,
        close,
        _atr(group),
        cfg.refractory,
        cfg.rearm_atr,
        cfg.rearm_uses_path,
        cfg.rearm_max_bars,
    )
    top_only &= ~top_confirmed
    _, top_confirmed = _sequence(
        np.zeros(n, dtype=bool),
        top_confirmed,
        close,
        _atr(group),
        cfg.refractory,
        cfg.rearm_atr,
        cfg.rearm_uses_path,
        cfg.rearm_max_bars,
    )
    unsupported_top = top_confirmed & ~top_model_support
    top_confirmed &= top_model_support
    guarded_confirmed_top = top_confirmed & top_guard_mask
    top_confirmed &= ~guarded_confirmed_top
    memory_confirmed_top = np.zeros(n, dtype=bool)
    if (
        has_top_confirmation
        and cfg.top_confirm_memory_self_q is not None
        and "_p_pivot_high_s" in group
    ):
        recent_high_self_q = (
            group["_p_pivot_high_s"]
            .rolling(
                cfg.top_confirm_memory_window,
                min_periods=1,
            )
            .max()
            .to_numpy()
        )
        top_detections = (
            top_only
            | guarded_top
            | guarded_confirmed_top
            | unsupported_top
        )
        top_memory_confirmation_proof = (
            np.nan_to_num(
                group["_top_bars_since_local_high"].to_numpy()
                <= cfg.top_confirm_memory_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_return_5"].to_numpy()
                <= cfg.top_confirm_memory_max_return_5,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                recent_high_self_q >= cfg.top_confirm_memory_self_q,
                nan=0.0,
            ).astype(bool)
        )
        memory_confirmed_top = (
            top_detections & top_memory_confirmation_proof
        )
        top_only &= ~memory_confirmed_top
        guarded_top &= ~memory_confirmed_top
        guarded_confirmed_top &= ~memory_confirmed_top
        unsupported_top &= ~memory_confirmed_top
        top_confirmed |= memory_confirmed_top
    if (
        cfg.top_uptrend_action_confirm_pivot_q is not None
        and {
            "_p_pivot_low_q",
            "_p_pivot_high_q",
            "_p_pivot_high_s",
            "_top_bars_since_local_high",
        }
        <= set(group.columns)
    ):
        with np.errstate(invalid="ignore", divide="ignore"):
            uptrend_top_span = high - low
            uptrend_top_close_loc = np.where(
                uptrend_top_span > 0,
                (close - low) / uptrend_top_span,
                np.nan,
            )
        top_uptrend_action_confirmation = (
            top_only
            & (action == "STRONG_SELL")
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                >= cfg.top_uptrend_action_confirm_pivot_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_s"].to_numpy()
                >= cfg.top_uptrend_action_confirm_self_q,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_p_pivot_high_q"].to_numpy()
                - group["_p_pivot_low_q"].to_numpy()
                >= cfg.top_uptrend_action_confirm_min_advantage,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["_top_bars_since_local_high"].to_numpy()
                <= cfg.top_uptrend_action_confirm_max_bars_since_high,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                group["close"].pct_change().to_numpy()
                <= cfg.top_uptrend_action_confirm_max_return_1,
                nan=0.0,
            ).astype(bool)
            & np.nan_to_num(
                uptrend_top_close_loc
                <= cfg.top_uptrend_action_confirm_max_close_loc,
                nan=0.0,
            ).astype(bool)
        )
        top_only &= ~top_uptrend_action_confirmation
        top_confirmed |= top_uptrend_action_confirmation

    return TopTurnResult(
        detections=top_only,
        guarded=guarded_top,
        guarded_confirmed=guarded_confirmed_top,
        unsupported=unsupported_top,
        confirmed=top_confirmed,
        confirmation_proof=top_confirmation_proof,
        memory_confirmation_proof=top_memory_confirmation_proof,
        local_confirmation_proof=top_local_confirmation_proof,
        crash_confirmation_proof=top_crash_confirmation_proof,
        uptrend_action_confirmation=top_uptrend_action_confirmation,
        model_support=top_model_support,
    )
