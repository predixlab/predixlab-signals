"""Backfill: turn the stored walk-forward run into three years of daily picks.

The walk-forward parquet is the only honest historical record this project has.
Every row in it was scored by a model fitted on data strictly before that row's
fold, so replaying it reproduces what the pick list *would have said* on each of
those sessions — as opposed to re-scoring history with today's model, which
would leak the future into every past recommendation and make the history
worthless.

The walk-forward frame's own ``fwd_ret`` supplies what happened next — the
  stock's return over the horizon following the signal.

That second one is deliberately *not* a backtest. A trade book would report a
strategy's slot limits, stops and costs; the question here is narrower and more
honest: after this signal fired, what did the name actually do over the next ten
sessions?

Selection reuses :func:`predixlab_signals.signals.picks.select_picks` and the
enrichment reuses :func:`predixlab_signals.signals.scoring.enrich_scores`, so a
backfilled pick and a live one are produced by the same code — which is what
makes the published history comparable to what the job will emit tomorrow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predixlab_signals.config import PATHS
from predixlab_signals.signals.scoring import enrich_scores
from predixlab_signals.signals.picks import (
    DETECTION,
    LONG,
    REVERSAL,
    SHORT,
    select_picks,
)
from predixlab_signals.signals.models.pivot import TurningPointConfig

#: Model outputs carried over from the walk-forward frame.
SCORE_COLUMNS = (
    "symbol",
    "date",
    "close",
    "price",
    "swing_pos_hat",
    "p_up",
    "exp_gain_pct",
    "exp_loss_pct",
    "exp_amplitude_pct",
    "score",
    "action",
)

#: Raw bars the turning-point rules read. The range position, the reversal
#: trigger and the breakout window are all price-action tests, so the OHLC
#: columns are not optional context — without them no event can be detected.
BAR_COLUMNS = ("high", "low", "close", "volume")


def walkforward_path(market: str, horizon: int, tag: str = "default") -> Path:
    return PATHS.report_dir(market) / f"walkforward_h{horizon}_{tag}.parquet"


def load_walkforward(market: str, horizon: int, tag: str = "default") -> pd.DataFrame:
    path = walkforward_path(market, horizon, tag)
    if not path.exists():
        raise FileNotFoundError(
            f"No walk-forward predictions at {path}. Run `predixlab-signals train` first."
        )
    return dedupe_folds(pd.read_parquet(path))


def dedupe_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fold-boundary overlap to one prediction per symbol-day.

    Consecutive walk-forward folds share their boundary date — the last session
    of one test window is the first of the next — so a handful of symbol-days
    carry two scores from two different fold models.

    The later fold wins. Its model was fitted on strictly more history, and on a
    boundary date a live system that retrains on schedule would just have
    switched to it; keeping the earlier score would replay history with a
    staler model than the strategy would actually have used.

    Left in place, the duplicate rows silently fan out every downstream join.
    """
    if predictions.empty or "fold" not in predictions.columns:
        return predictions
    ordered = predictions.sort_values(["symbol", "date", "fold"], kind="stable")
    return ordered.drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def attach_forward_returns(picks: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Attach the raw and direction-adjusted forward return of each signal.

    This measures one thing and measures it plainly: **what the stock did over
    the horizon after the signal fired**. No position sizing, no slots, no stop
    or target, no transaction costs — those belong to a strategy, and mixing
    them in would report the strategy's constraints rather than the signal's
    information.

    Three numbers, and the third is the one that matters:

    ``forward_return``  the stock's own return, ``close[t+H]/close[t] - 1``
    ``signal_return``   the same figure from the position's point of view,
                        negated for shorts so both sides can be averaged
    ``excess_return``   ``signal_return`` measured against the *same day's*
                        cross-section, i.e. with the common market move removed

    The raw figure alone is misleading, and measurably so: buy events average
    +0.547% over the horizon while the universe averages +0.561% on the same
    bars. Reported on its own that reads like an edge; against the benchmark it
    is slightly *below* market. The excess column is what separates "the signal
    knew something" from "the market went up that fortnight".

    All three are ``NaN`` for the most recent ``H`` sessions, where the horizon
    has not elapsed. That is honest: those signals are still open.
    """
    picks = picks.copy()
    picks["date"] = pd.to_datetime(picks["date"])

    if predictions is None or predictions.empty or "fwd_ret" not in predictions.columns:
        picks["forward_return"] = pd.NA
        picks["signal_return"] = pd.NA
        return picks

    columns = ["symbol", "date", "fwd_ret"]
    if "fwd_ret_cs" in predictions.columns:
        columns.append("fwd_ret_cs")
    forward = predictions[columns].copy()
    forward["date"] = pd.to_datetime(forward["date"])
    forward = forward.drop_duplicates(["symbol", "date"], keep="last")

    merged = picks.merge(
        forward.rename(columns={"fwd_ret": "forward_return", "fwd_ret_cs": "_excess"}),
        on=["symbol", "date"],
        how="left",
    )
    # A short profits from a fall, so its return is the negated move — on both
    # the raw and the benchmark-relative figure.
    sign = np.where(merged["direction"].to_numpy() == SHORT, -1.0, 1.0)
    merged["signal_return"] = sign * merged["forward_return"]
    merged["excess_return"] = (
        sign * merged["_excess"] if "_excess" in merged.columns else pd.NA
    )
    return merged.drop(columns=["_excess"], errors="ignore")


#: Bars of slack allowed between a signal and the pivot it claims to have found.
#: A turn is a region, not an instant, and the zigzag's own confirmation is
#: itself approximate; demanding an exact bar match would measure timestamp
#: precision rather than detection.
PIVOT_TOLERANCE = 3


def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    """Widen a boolean mask by ``tolerance`` bars on each side, without wrapping."""
    out = mask.copy()
    n = mask.size
    for k in range(1, tolerance + 1):
        out[k:] |= mask[: n - k]
        out[: n - k] |= mask[k:]
    return out


def attach_pivot_hits(
    picks: pd.DataFrame,
    predictions: pd.DataFrame,
    tolerance: int = PIVOT_TOLERANCE,
) -> pd.DataFrame:
    """Mark whether each reversal landed on a real swing point.

    Ground truth is the ATR zigzag in the walk-forward frame — a definition
    nobody tuned on model output, and one the model never sees. It peeks at the
    future by construction, which is exactly why it is usable for *evaluation*
    and never as a feature.

    Only reversals are scored. ``TREND_BUY`` and ``BASE_BREAKOUT`` are left
    ``NA`` on purpose: they are directional breakout entries, so measuring them
    against pivot labels asks the wrong question and would report a category
    error as a failure.
    """
    picks = picks.copy()
    picks["pivot_hit"] = pd.NA
    if predictions is None or predictions.empty or "pivot" not in predictions.columns:
        return picks

    wanted = {LONG: -1.0, SHORT: 1.0}
    lookup: dict[tuple[str, pd.Timestamp], bool] = {}

    ordered = predictions.sort_values(["symbol", "date"], kind="stable")
    for symbol, group in ordered.groupby("symbol", sort=False):
        pivot = group["pivot"].to_numpy()
        dates = pd.DatetimeIndex(group["date"])
        near_low = _dilate(pivot == -1.0, tolerance)
        near_high = _dilate(pivot == 1.0, tolerance)
        for i, date in enumerate(dates):
            lookup[(symbol, date, -1.0)] = bool(near_low[i])
            lookup[(symbol, date, 1.0)] = bool(near_high[i])

    hits = []
    for symbol, date, direction, kind in zip(
        picks["symbol"], pd.DatetimeIndex(picks["date"]), picks["direction"], picks["kind"]
    ):
        if kind not in (REVERSAL, DETECTION):
            hits.append(pd.NA)
            continue
        target = wanted.get(direction)
        hit = lookup.get((symbol, date, target)) if target is not None else None
        hits.append(pd.NA if hit is None else bool(hit))
    picks["pivot_hit"] = hits
    return picks


def build_historical_picks(
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
    start: str | None = None,
    end: str | None = None,
    with_outcomes: bool = True,
    config: TurningPointConfig | None = None,
    max_per_side: int | None = None,
) -> pd.DataFrame:
    """Replay the walk-forward run into a history of turning-point events.

    ``start`` and ``end`` trim the *published* window, not the panel the rules
    read. The gates are sequential — a trailing range window, a setup that may
    precede its trigger by three sessions, a refractory period — so events are
    always extracted over the full stored run and only then filtered. Trimming
    first would silently suppress every event whose setup fell before ``start``.
    """
    predictions = load_walkforward(market, horizon, tag)
    if predictions.empty:
        return pd.DataFrame()

    missing = [c for c in (*SCORE_COLUMNS, *BAR_COLUMNS) if c not in predictions.columns]
    if missing:
        raise ValueError(f"Walk-forward frame is missing {missing}")

    # Events first, over the whole run; the date filter comes after.
    events = select_picks(predictions, config=config, max_per_side=max_per_side)
    if events.empty:
        return pd.DataFrame()

    if start is not None:
        events = events[events["date"] >= pd.Timestamp(start)]
    if end is not None:
        events = events[events["date"] <= pd.Timestamp(end)]
    if events.empty:
        return pd.DataFrame()
    events = events.reset_index(drop=True)

    scored = events[list(SCORE_COLUMNS)].reset_index(drop=True)
    enriched = enrich_scores(scored, cross_section=predictions)

    for column in (
        "signal_model",
        "event",
        "pivot_event",
        "actionable",
        "direction",
        "kind",
        "rank",
    ):
        enriched[column] = events[column].to_numpy()
    if with_outcomes:
        enriched = attach_forward_returns(enriched, predictions)
        enriched = attach_pivot_hits(enriched, predictions)

    return enriched.sort_values(
        ["date", "direction", "rank"], kind="stable"
    ).reset_index(drop=True)
