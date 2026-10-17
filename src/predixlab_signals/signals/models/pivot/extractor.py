"""Orchestrate the pivot model's independent signal families."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.signals.gates import add_gate_percentiles
from predixlab_signals.signals.models.pivot.base_breakouts import _base_breakout
from predixlab_signals.signals.models.pivot.bottom_turns import (
    apply_bottom_volume_rebounds,
    build_bottom_reference_candidates,
    build_bottom_triggers,
    build_bottom_right_rescue,
    evaluate_bottom_turns,
    finalize_bottom_events,
    prepare_bottom_candidates,
)
from predixlab_signals.signals.models.pivot.common import (
    _atr,
    _conviction,
    _range_position,
    _recent_while_live,
    _self_relative_position,
    _sequence,
    _sequence_additions,
    _setup_alive,
    _trend_slope,
)
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
from predixlab_signals.signals.models.pivot.top_turns import (
    build_top_reference_candidates,
    build_top_triggers,
    build_top_right_rescue,
    evaluate_top_turns,
    finalize_top_events,
    guard_top_candidates,
    prepare_top_candidates,
)
from predixlab_signals.signals.models.pivot.trend_breakouts import (
    _apply_refractory,
    _trend_entry,
)

def extract_turning_points(
    preds: pd.DataFrame, config: TurningPointConfig | None = None
) -> pd.DataFrame:
    """Add a ``turn_action`` column holding sparse turning-point events.

    ``preds`` needs ``symbol``, ``date``, ``high``, ``low``, ``close``,
    ``volume`` and ``action``. The original ``action`` column is left untouched
    so both views stay available: the state answers "is this name attractive at
    all", the event answers "has it turned yet".
    """
    cfg = config or TurningPointConfig()
    frame = preds.sort_values(["symbol", "date"], kind="stable").copy()
    has_base_breakout = cfg.base_breakout and "volume" in frame
    has_bottom_reference = cfg.bottom_reference_self_q is not None
    has_top_reference = cfg.top_reference_pivot_q is not None

    # Rank the detection probabilities *across the day's cross-section* once,
    # up front. Ranking within a symbol's own history instead would make the
    # gate mean "unusual for this stock" rather than "the market's best
    # candidates today", and would fire on every name in every regime.
    if cfg.pivot_gate_q is not None or has_bottom_reference or has_top_reference:
        add_gate_percentiles(frame, cfg)
    has_top_confirmation = cfg.top_confirm_volatility_q is not None
    has_bottom_confirmation = cfg.bottom_confirm_volatility_q is not None
    has_sell_right_reversal = cfg.sell_right_reversal_pivot_q is not None
    has_buy_late_rebound = cfg.buy_late_rebound_min_body_return is not None
    has_bottom_right_rescue = cfg.bottom_right_rescue_pivot_q is not None
    has_top_right_rescue = cfg.top_right_rescue_pivot_q is not None
    has_bottom_pivot_confirmation = (
        cfg.bottom_pivot_fresh_q is not None
        or cfg.bottom_pivot_capitulation_q is not None
    )
    if (
        has_top_confirmation
        or has_bottom_confirmation
        or has_bottom_reference
        or has_top_reference
        or has_sell_right_reversal
        or has_buy_late_rebound
        or has_bottom_right_rescue
        or has_top_right_rescue
        or has_bottom_pivot_confirmation
    ):
        close_grouped = frame.groupby("symbol", sort=False)["close"]
        ret1 = close_grouped.pct_change()
        fast_vol = ret1.groupby(frame["symbol"], sort=False).transform(
            lambda s: s.rolling(
                cfg.top_confirm_fast_vol_window,
                min_periods=cfg.top_confirm_fast_vol_window,
            ).std()
        )
        slow_vol = ret1.groupby(frame["symbol"], sort=False).transform(
            lambda s: s.rolling(
                cfg.top_confirm_slow_vol_window,
                min_periods=cfg.top_confirm_slow_vol_window,
            ).std()
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            volatility_ratio = fast_vol / slow_vol
        bars_since_high = frame.groupby("symbol", sort=False)["high"].transform(
            lambda s: (
                cfg.top_confirm_high_window
                - 1
                - s.rolling(
                    cfg.top_confirm_high_window,
                    min_periods=cfg.top_confirm_high_window,
                ).apply(lambda values: float(np.argmax(values)), raw=True)
            )
        )
        frame["_top_volatility_q"] = volatility_ratio.groupby(frame["date"]).rank(
            pct=True
        )
        frame["_top_recent_volatility_q"] = frame.groupby(
            "symbol", sort=False
        )["_top_volatility_q"].transform(
            lambda s: s.rolling(
                cfg.top_confirm_volatility_lookback,
                min_periods=1,
            ).max()
        )
        frame["_top_bars_since_high"] = bars_since_high
        frame["_top_bars_since_local_high"] = frame.groupby(
            "symbol", sort=False
        )["high"].transform(
            lambda s: (
                cfg.top_confirm_memory_high_window
                - 1
                - s.rolling(
                    cfg.top_confirm_memory_high_window,
                    min_periods=cfg.top_confirm_memory_high_window,
                ).apply(lambda values: float(np.argmax(values)), raw=True)
            )
        )
        frame["_top_return_3"] = close_grouped.pct_change(3)
        frame["_top_return_5"] = close_grouped.pct_change(5)
        frame["_bottom_bars_since_low"] = frame.groupby(
            "symbol", sort=False
        )["low"].transform(
            lambda s: (
                cfg.top_confirm_high_window
                - 1
                - s.rolling(
                    cfg.top_confirm_high_window,
                    min_periods=cfg.top_confirm_high_window,
                ).apply(lambda values: float(np.argmin(values)), raw=True)
            )
        )
        frame["_bottom_bars_since_local_low"] = frame.groupby(
            "symbol", sort=False
        )["low"].transform(
            lambda s: (
                cfg.bottom_confirm_local_window
                - 1
                - s.rolling(
                    cfg.bottom_confirm_local_window,
                    min_periods=cfg.bottom_confirm_local_window,
                ).apply(lambda values: float(np.argmin(values)), raw=True)
            )
        )

    turn = np.full(len(frame), NO_TURN, dtype=object)

    position = 0
    for _, group in frame.groupby("symbol", sort=False):
        n = len(group)
        block = slice(position, position + n)
        position += n

        pos = _range_position(group, cfg.range_window, cfg.range_min_periods)
        action = group["action"].to_numpy()
        close = group["close"].to_numpy(dtype="float64")
        high = group["high"].to_numpy(dtype="float64")
        low = group["low"].to_numpy(dtype="float64")

        # Setup: the model is convinced and price is sitting in the extreme of
        # its range. This is what must have happened *recently*, not today.
        buy_conviction, sell_conviction = _conviction(group, cfg)
        self_pos = _self_relative_position(pos, cfg)
        # A wide reversal bar is shared evidence for either family; each family
        # decides independently how it affects its own setup and trigger gates.
        wide_reversal = np.zeros(n, dtype=bool)
        if cfg.waiver_uses_bar_range and cfg.reversal_waiver_atr is not None:
            setup_atr = _atr(group)
            with np.errstate(invalid="ignore", divide="ignore"):
                bar_range = (high - low) / np.where(
                    np.isfinite(setup_atr) & (setup_atr > 0),
                    setup_atr,
                    np.nan,
                )
            wide_reversal = np.nan_to_num(
                bar_range >= cfg.reversal_waiver_atr,
                nan=0.0,
            ).astype(bool)

        bottom_triggers = build_bottom_triggers(
            group,
            buy_conviction,
            pos,
            self_pos,
            wide_reversal,
            cfg,
            has_confirmation=has_bottom_confirmation,
        )
        buy = bottom_triggers.actionable
        delayed_rescue_buy = bottom_triggers.delayed_rescue
        broke_up = bottom_triggers.broke_up

        top_triggers = build_top_triggers(
            group,
            sell_conviction,
            pos,
            self_pos,
            wide_reversal,
            cfg,
        )
        sell = top_triggers.actionable
        top_sell = top_triggers.confirmation_source
        sell_setup = top_triggers.setup
        broke_down = top_triggers.broke_down

        slope = np.full(n, np.nan, dtype="float64")
        if cfg.max_uptrend_slope is not None or cfg.min_downtrend_slope is not None:
            slope = _trend_slope(group["close"])
        bottom_candidates = prepare_bottom_candidates(
            group,
            buy,
            slope,
            cfg,
            has_confirmation=has_bottom_confirmation,
        )
        buy = bottom_candidates.actionable
        bottom_only = bottom_candidates.detections
        bottom_confirmation_candidates = (
            bottom_candidates.confirmation_candidates
        )
        low_conviction_guard = bottom_candidates.low_conviction_guard

        top_candidates = prepare_top_candidates(
            group,
            sell,
            top_sell,
            slope,
            cfg,
        )
        sell = top_candidates.actionable
        top_only = top_candidates.detections
        top_confirmation_candidates = top_candidates.confirmation_candidates

        # A bar cannot be both; neither side wins a tie.
        both = buy & sell
        buy &= ~both
        sell &= ~both

        buy, sell = _sequence(
            buy,
            sell,
            close,
            _atr(group),
            cfg.refractory,
            cfg.rearm_atr,
            cfg.rearm_uses_path,
            cfg.rearm_max_bars,
        )
        guarded = guard_top_candidates(
            group,
            sell,
            cfg,
            has_confirmation=has_top_confirmation,
        )
        sell = guarded.actionable
        guarded_top = guarded.guarded
        top_guard_mask = guarded.guard_mask
        top_reversal_proof = guarded.reversal_proof
        bottom_result = evaluate_bottom_turns(
            group,
            buy,
            bottom_only,
            bottom_confirmation_candidates,
            low_conviction_guard,
            cfg,
            has_confirmation=has_bottom_confirmation,
        )
        buy = bottom_result.actionable
        bottom_only = bottom_result.detections
        guarded_bottom = bottom_result.guarded
        dual_guard_bottom = bottom_result.dual_guard
        bottom_confirmed = bottom_result.confirmed
        bottom_confirmation_proof = bottom_result.confirmation_proof
        bottom_reference_alternate_confirmation = (
            bottom_result.reference_alternate_confirmation
        )
        bottom_setup_memory_proof = bottom_result.setup_memory_proof
        bottom_fresh_setup_proof = bottom_result.fresh_setup_proof
        bottom_setup_rebound_proof = bottom_result.setup_rebound_proof
        bottom_shock_confirmation_proof = bottom_result.shock_confirmation_proof

        top_result = evaluate_top_turns(
            group,
            top_only,
            top_confirmation_candidates,
            guarded_top,
            top_guard_mask,
            top_reversal_proof,
            cfg,
            has_confirmation=has_top_confirmation,
        )
        top_only = top_result.detections
        guarded_top = top_result.guarded
        guarded_confirmed_top = top_result.guarded_confirmed
        unsupported_top = top_result.unsupported
        top_confirmed = top_result.confirmed
        top_confirmation_proof = top_result.confirmation_proof
        top_memory_confirmation_proof = top_result.memory_confirmation_proof
        top_local_confirmation_proof = top_result.local_confirmation_proof
        top_crash_confirmation_proof = top_result.crash_confirmation_proof
        top_uptrend_action_confirmation = top_result.uptrend_action_confirmation
        top_model_support = top_result.model_support

        reference_bottom = build_bottom_reference_candidates(
            group,
            cfg,
            enabled=has_bottom_reference,
        )
        reference_top = build_top_reference_candidates(
            group,
            sell_conviction,
            sell_setup,
            cfg,
            enabled=has_top_reference,
        )

        block_turn = np.full(n, NO_TURN, dtype=object)
        block_turn[buy] = BUY_TURN
        block_turn[sell] = SELL_TURN
        block_turn[bottom_only] = BOTTOM_TURN
        block_turn[guarded_bottom] = BOTTOM_TURN
        block_turn[top_only] = TOP_TURN
        block_turn[guarded_top] = TOP_TURN
        block_turn[guarded_confirmed_top] = TOP_TURN
        block_turn[unsupported_top] = TOP_TURN
        block_turn[bottom_confirmed] = BUY_TURN
        block_turn[top_confirmed] = SELL_TURN

        if cfg.trend_entry:
            entry = _trend_entry(group, cfg)
            entry &= block_turn == NO_TURN
            block_turn[_apply_refractory(entry, cfg.trend_entry_refractory)] = TREND_BUY

        delayed_rescue_buy &= block_turn == NO_TURN
        delayed_rescue_buy, _ = _sequence_additions(
            block_turn,
            delayed_rescue_buy,
            np.zeros(n, dtype=bool),
            close,
            _atr(group),
            cfg.refractory,
            cfg.rearm_atr,
            cfg.rearm_uses_path,
            cfg.rearm_max_bars,
        )
        block_turn[delayed_rescue_buy] = BUY_TURN

        reference_bottom &= block_turn == NO_TURN
        reference_top &= block_turn == NO_TURN
        reference_conflict = reference_bottom & reference_top
        reference_bottom &= ~reference_conflict
        reference_top &= ~reference_conflict
        reference_bottom_confirmed = reference_bottom & (
            bottom_confirmation_proof
            | bottom_reference_alternate_confirmation
            | bottom_setup_memory_proof
            | bottom_fresh_setup_proof
        )
        reference_bottom, reference_top = _sequence_additions(
            block_turn,
            reference_bottom,
            reference_top,
            close,
            _atr(group),
            cfg.refractory,
            cfg.rearm_atr,
            cfg.rearm_uses_path,
            cfg.rearm_max_bars,
        )
        reference_bottom_confirmed &= reference_bottom
        reference_bottom &= ~reference_bottom_confirmed

        reference_top_standard = reference_top & top_confirmation_proof
        reference_top_memory = reference_top & top_memory_confirmation_proof
        reference_top_confirmed = reference_top_memory | (
            reference_top_standard & top_model_support & ~top_guard_mask
        )
        reference_top_demoted = reference_top_standard & (
            ~top_model_support | top_guard_mask
        )
        reference_top = (
            reference_top | reference_top_demoted
        ) & ~reference_top_confirmed

        block_turn[reference_bottom] = BOTTOM_TURN
        block_turn[reference_top] = TOP_TURN
        block_turn[reference_bottom_confirmed] = BUY_TURN
        block_turn[reference_top_confirmed] = SELL_TURN

        block_turn[
            (block_turn == BOTTOM_TURN)
            & (bottom_setup_rebound_proof | bottom_shock_confirmation_proof)
        ] = BUY_TURN
        block_turn[
            (block_turn == TOP_TURN)
            & (top_local_confirmation_proof | top_crash_confirmation_proof)
        ] = SELL_TURN

        bottom_right_rescue = build_bottom_right_rescue(
            group,
            block_turn,
            broke_up,
            cfg,
            enabled=has_bottom_right_rescue,
        )
        top_right_rescue = build_top_right_rescue(
            group,
            block_turn,
            broke_down,
            cfg,
            enabled=has_top_right_rescue,
        )
        if bottom_right_rescue.any() or top_right_rescue.any():
            bottom_right_rescue, top_right_rescue = _sequence_additions(
                block_turn,
                bottom_right_rescue,
                top_right_rescue,
                close,
                _atr(group),
                cfg.refractory,
                cfg.rearm_atr,
                cfg.rearm_uses_path,
                cfg.rearm_max_bars,
            )
            block_turn[bottom_right_rescue] = BUY_TURN
            block_turn[top_right_rescue] = SELL_TURN

        block_turn = finalize_bottom_events(
            group,
            block_turn,
            dual_guard_bottom,
            cfg,
        )
        block_turn = finalize_top_events(
            group,
            block_turn,
            cfg,
            local_confirmation=top_local_confirmation_proof,
            crash_confirmation=top_crash_confirmation_proof,
            uptrend_action_confirmation=top_uptrend_action_confirmation,
            right_rescue=top_right_rescue,
            has_right_reversal=has_sell_right_reversal,
        )

        if has_base_breakout:
            base_breakout = _base_breakout(group, cfg)
            base_breakout &= block_turn == NO_TURN
            block_turn[base_breakout] = BASE_BREAKOUT

        block_turn = apply_bottom_volume_rebounds(group, block_turn, cfg)
        turn[block] = block_turn

    frame["turn_action"] = turn
    helpers = (
        "_p_pivot_low_q",
        "_p_pivot_high_q",
        "_p_pivot_low_s",
        "_p_pivot_high_s",
        "_p_pivot_low_abs",
        "_p_pivot_high_abs",
        "_p_pivot_low_breadth",
        "_p_pivot_high_breadth",
        "_top_volatility_q",
        "_top_recent_volatility_q",
        "_top_bars_since_high",
        "_top_bars_since_local_high",
        "_top_return_3",
        "_top_return_5",
        "_bottom_bars_since_low",
        "_bottom_bars_since_local_low",
    )
    return frame.drop(columns=[c for c in helpers if c in frame])
