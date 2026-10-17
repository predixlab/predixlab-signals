"""Forward-looking labels that describe *where a bar sits in the coming swing*.

The usual quant target — "return over the next N days" — is a poor teacher for
timing. It rewards a model for calling direction while saying nothing about
whether the entry was near a local low or halfway up an extended move, and its
fat tails let a handful of gap days dominate the fit.

This module targets the swing itself. Over a forward window of ``H`` bars we
measure the two excursions that a trader actually experiences:

``mfe``  maximum favourable excursion — ``max(high[t+1..t+H]) / close[t] - 1``
``mae``  maximum adverse excursion   — ``min(low[t+1..t+H]) / close[t] - 1``

From that pair everything useful follows:

``swing_pos = -mae / (mfe - mae)``   0 → the bar is the coming range's *low*
                                     (a buy point); 1 → its *high* (a sell point)
``amplitude = mfe - mae``            how much room the swing offers at all
``edge      = mfe + mae``            reward/risk asymmetry — the signed answer to
                                     "is this a good place to be long?"

Both excursions are divided by ``atr_pct * sqrt(H)``, the bar's own expected
move over the horizon. That turns the targets into *shape* rather than
*volatility level*, so a quiet utility and a volatile chip stock produce
comparable labels and one model fits both. Predictions are rescaled back into
percent at inference time.

A triple-barrier classification label is produced alongside, giving a calibrated
probability to gate on, and a zigzag pivot marker is provided as independent
ground truth for evaluating turning-point detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9
PIVOT_RETURN_CAP = 3.0

#: Floor on the ATR scaling term, in percent of price. Without it, a stock that
#: went dead-flat for a fortnight would produce explosive normalised labels.
MIN_ATR_PCT = 0.004


def _forward_extremes(high: np.ndarray, low: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling max(high) / min(low) over the *strictly future* window t+1..t+H."""
    n = high.size
    fwd_high = np.full(n, np.nan)
    fwd_low = np.full(n, np.nan)
    if n <= horizon:
        return fwd_high, fwd_low

    # sliding_window_view gives every contiguous H-length block in one shot;
    # block i covers [i, i+H), so aligning it to bar i-1 makes it the future.
    hw = np.lib.stride_tricks.sliding_window_view(high, horizon)
    lw = np.lib.stride_tricks.sliding_window_view(low, horizon)
    fwd_high[: n - horizon] = hw[1:].max(axis=1)
    fwd_low[: n - horizon] = lw[1:].min(axis=1)
    return fwd_high, fwd_low


def _triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    horizon: int,
    width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-touch barrier label and the bar count until that touch.

    Returns ``(label, bars)`` where label is ``1`` if the upper barrier is hit
    first, ``-1`` if the lower one is, and ``0`` on timeout. Ties inside a bar
    (both barriers touched by the same candle) resolve to ``0``: without
    intraday data we genuinely cannot tell which came first, and guessing would
    inject a systematic bias exactly on the most volatile bars.
    """
    n = close.size
    label = np.full(n, np.nan)
    bars = np.full(n, np.nan)
    if n <= horizon:
        return label, bars

    up = close + width * atr
    dn = close - width * atr

    hw = np.lib.stride_tricks.sliding_window_view(high, horizon)[1:]
    lw = np.lib.stride_tricks.sliding_window_view(low, horizon)[1:]
    m = hw.shape[0]

    hit_up = hw >= up[:m, None]
    hit_dn = lw <= dn[:m, None]

    any_up = hit_up.any(axis=1)
    any_dn = hit_dn.any(axis=1)
    first_up = np.where(any_up, hit_up.argmax(axis=1), horizon)
    first_dn = np.where(any_dn, hit_dn.argmax(axis=1), horizon)

    res = np.zeros(m)
    res[first_up < first_dn] = 1.0
    res[first_dn < first_up] = -1.0
    # first_up == first_dn < horizon -> ambiguous same-bar touch -> stays 0.

    label[:m] = res
    bars[:m] = np.minimum(first_up, first_dn) + 1.0
    invalid = ~np.isfinite(atr[:m]) | (atr[:m] <= 0)
    label[:m][invalid] = np.nan
    bars[:m][invalid] = np.nan
    return label, bars


def future_extreme_holds(
    high: pd.Series, low: pd.Series, window: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Whether today's extreme survives the next ``window`` bars.

    The target deliberately ignores all prior price action. A low hold is one
    when no lower low appears from today through the next ``window`` bars; a
    high hold is defined symmetrically.
    """
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    n = h.size
    low_holds = np.full(n, np.nan)
    high_holds = np.full(n, np.nan)
    if n > window:
        low_win = np.lib.stride_tricks.sliding_window_view(lo, window + 1)
        high_win = np.lib.stride_tricks.sliding_window_view(h, window + 1)
        resolved = n - window
        low_holds[:resolved] = (low_win.min(axis=1) == lo[:resolved]).astype(
            "float64"
        )
        high_holds[:resolved] = (high_win.max(axis=1) == h[:resolved]).astype(
            "float64"
        )
    return (
        pd.Series(low_holds, index=high.index),
        pd.Series(high_holds, index=high.index),
    )


def zigzag_pivots(
    high: pd.Series, low: pd.Series, atr: pd.Series, k: float = 3.0
) -> pd.Series:
    """Confirmed swing highs (``1``) and lows (``-1``) via an ATR zigzag.

    This is *not* a model input — it peeks at the future by construction. It
    exists purely so evaluation can ask "did the model fire near a real turning
    point?" against a definition nobody tuned on model output.
    """
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    a = atr.to_numpy(dtype="float64")
    n = h.size
    out = np.zeros(n)
    if n == 0:
        return pd.Series(out, index=high.index)

    direction = 0  # +1 = tracking an up leg, -1 = tracking a down leg
    ext_idx = 0
    ext_val = h[0]

    for i in range(1, n):
        threshold = k * (a[i] if np.isfinite(a[i]) and a[i] > 0 else np.nan)
        if not np.isfinite(threshold):
            continue
        if direction >= 0:
            if h[i] > ext_val:
                ext_val, ext_idx = h[i], i
            elif ext_val - lo[i] > threshold:
                out[ext_idx] = 1.0  # confirmed swing high
                direction = -1
                ext_val, ext_idx = lo[i], i
        else:
            if lo[i] < ext_val:
                ext_val, ext_idx = lo[i], i
            elif h[i] - ext_val > threshold:
                out[ext_idx] = -1.0  # confirmed swing low
                direction = 1
                ext_val, ext_idx = h[i], i
    return pd.Series(out, index=high.index)


def build_labels(
    panel: pd.DataFrame,
    horizon: int = 10,
    barrier_width: float = 1.5,
    zigzag_k: float = 3.0,
    hold_window: int = 5,
) -> pd.DataFrame:
    """Compute the full label block for a long-format OHLCV panel.

    The returned frame carries ``symbol``/``date`` plus:

    ``y_mfe``, ``y_mae``   ATR-normalised forward excursions (model targets)
    ``y_edge``             ``y_mfe + y_mae`` — the signed long-side edge
    ``y_swing_pos``        0 = coming low, 1 = coming high
    ``y_barrier``          triple-barrier class in ``{-1, 0, 1}``
    ``y_pivot_*_hold``     whether today's extreme survives ``hold_window`` bars
    ``y_pivot_*_return``   directional future return conditional on holding
    ``fwd_*``              the same quantities in raw percent, for backtests
    ``pivot``              zigzag ground truth, evaluation only
    """
    blocks: list[pd.DataFrame] = []
    root_h = float(np.sqrt(horizon))

    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.set_index("date").sort_index()
        if len(frame) <= horizon + 20:
            continue

        high = frame["high"].to_numpy(dtype="float64")
        low = frame["low"].to_numpy(dtype="float64")
        close = frame["close"].to_numpy(dtype="float64")

        atr = _atr_series(frame).to_numpy(dtype="float64")
        atr_pct = np.clip(atr / np.maximum(close, EPS), MIN_ATR_PCT, None)
        scale = atr_pct * root_h

        fwd_high, fwd_low = _forward_extremes(high, low, horizon)
        mfe = fwd_high / close - 1.0
        mae = fwd_low / close - 1.0
        amplitude = mfe - mae

        label, bars = _triple_barrier(high, low, close, atr, horizon, barrier_width)

        fwd_close = np.roll(close, -horizon)
        fwd_close[-horizon:] = np.nan
        fwd_ret = fwd_close / close - 1.0

        block = pd.DataFrame(
            {
                "y_mfe": mfe / scale,
                "y_mae": mae / scale,
                "y_edge": (mfe + mae) / scale,
                "y_amplitude": amplitude / scale,
                # Values outside [0, 1] are real - they mean the close sat
                # beyond the entire coming range - but they are rare, so they are
                # clipped to keep the squared-error head from chasing them.
                "y_swing_pos": np.clip(
                    np.where(amplitude > EPS, -mae / np.maximum(amplitude, EPS), np.nan), -0.5, 1.5
                ),
                "y_barrier": label,
                "y_barrier_bars": bars,
                "fwd_mfe": mfe,
                "fwd_mae": mae,
                "fwd_ret": fwd_ret,
                "label_scale": scale,
                "atr_pct_label": atr_pct,
            },
            index=frame.index,
        )
        block["pivot"] = zigzag_pivots(frame["high"], frame["low"], pd.Series(atr, index=frame.index), zigzag_k)
        low_hold, high_hold = future_extreme_holds(
            frame["high"], frame["low"], hold_window
        )
        block["y_pivot_low_hold"] = low_hold
        block["y_pivot_high_hold"] = high_hold
        directional_return = np.clip(
            fwd_ret / scale,
            -PIVOT_RETURN_CAP,
            PIVOT_RETURN_CAP,
        )
        block["y_pivot_low_return"] = np.where(
            low_hold.to_numpy() == 1.0,
            directional_return,
            np.nan,
        )
        block["y_pivot_high_return"] = np.where(
            high_hold.to_numpy() == 1.0,
            -directional_return,
            np.nan,
        )
        block.insert(0, "symbol", symbol)
        blocks.append(block.reset_index())

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True).sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def _atr_series(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


LABEL_COLUMNS = [
    "y_mfe",
    "y_mae",
    "y_edge",
    "y_amplitude",
    "y_swing_pos",
    "y_barrier",
    "y_barrier_bars",
    "fwd_mfe",
    "fwd_mae",
    "fwd_ret",
    "label_scale",
    "atr_pct_label",
    "pivot",
    "y_pivot_low_hold",
    "y_pivot_high_hold",
    "y_pivot_low_return",
    "y_pivot_high_return",
]
