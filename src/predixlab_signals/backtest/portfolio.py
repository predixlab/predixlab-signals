"""Portfolio-level accounting on top of the trade list.

Trade statistics alone flatter a strategy: they say nothing about how many
positions were open at once, how much of the time capital sat idle, or what the
path looked like. This module marks positions to market **daily** and runs them
through a fixed-slot equal-weight book, which is the only way to get an honest
Sharpe, CAGR and drawdown.

Capital rules:

* ``max_positions`` slots, each sized ``1 / max_positions`` of equity.
* When more signals fire than there are free slots, the highest-scoring ones win
  and the rest are simply missed — the same constraint a real book faces.
* Unfilled slots earn nothing. Exposure is reported so a high Sharpe achieved by
  sitting in cash is visible rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def build_position_book(
    trades: pd.DataFrame, max_positions: int = 20
) -> pd.DataFrame:
    """Select which trades actually get taken under a slot constraint.

    Trades are considered in date order; on any given day the best-scoring
    candidates fill whatever slots are free.
    """
    if trades.empty:
        return trades

    ordered = trades.sort_values(["entry_date", "signal_score"], ascending=[True, False])
    open_slots: list[pd.Timestamp] = []  # exit dates of currently held positions
    taken: list[int] = []

    for idx, row in zip(ordered.index, ordered.itertuples(index=False)):
        entry = row.entry_date
        open_slots = [d for d in open_slots if d > entry]
        if len(open_slots) >= max_positions:
            continue
        open_slots.append(row.exit_date)
        taken.append(idx)

    return trades.loc[taken].sort_values("entry_date")


def daily_returns(
    preds: pd.DataFrame,
    trades: pd.DataFrame,
    max_positions: int = 20,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Mark every held position to market once per day.

    The first and last days of a trade are marked against the actual fill price
    rather than the previous close. Marking them close-to-close would count the
    position as open on both the entry and the exit bar, inflating measured
    exposure above 100% and overstating returns at every turnover.

    Returns a frame indexed by date with ``ret`` (portfolio return), ``n_open``
    and ``exposure``.
    """
    if trades.empty:
        return pd.DataFrame(columns=["ret", "n_open", "exposure"])

    bars = (
        preds[["symbol", "date", "close"]]
        .drop_duplicates(["symbol", "date"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    bars["prev_close"] = bars.groupby("symbol", sort=False)["close"].shift(1)
    index = {(s, d): i for i, (s, d) in enumerate(zip(bars["symbol"], bars["date"]))}
    close = bars["close"].to_numpy(dtype="float64")
    prev_close = bars["prev_close"].to_numpy(dtype="float64")
    dates = bars["date"].to_numpy()

    # ``held`` accrues the return, ``occupied`` counts slots. They differ on the
    # exit day: the position still earns a return up to its fill, but the slot is
    # already free for a replacement, so counting it as occupied would push
    # measured exposure above 100%.
    rows: list[tuple] = []
    for trade in trades.itertuples(index=False):
        start = index.get((trade.symbol, trade.entry_date))
        stop = index.get((trade.symbol, trade.exit_date))
        if start is None or stop is None or stop < start:
            continue
        sign = 1.0 if trade.direction == "long" else -1.0
        entry_px = float(getattr(trade, "entry_adj", close[start]))
        exit_px = float(getattr(trade, "exit_adj", close[stop]))
        # The whole round trip is charged on entry: simple, and it errs against
        # the strategy rather than for it.
        cost = 2.0 * cost_bps / 10_000.0

        for i in range(start, stop + 1):
            if i == start and i == stop:
                r = exit_px / entry_px - 1.0
            elif i == start:
                r = close[i] / entry_px - 1.0
            elif i == stop:
                r = exit_px / prev_close[i] - 1.0
            else:
                r = close[i] / prev_close[i] - 1.0
            if not np.isfinite(r):
                continue
            contrib = sign * r
            if i == start:
                contrib -= cost
            rows.append((dates[i], contrib, 1.0 if i < stop or start == stop else 0.0))

    if not rows:
        return pd.DataFrame(columns=["ret", "n_open", "exposure"])

    held = pd.DataFrame(rows, columns=["date", "contrib", "occupied"])
    grouped = held.groupby("date").agg(gross=("contrib", "sum"), n_open=("occupied", "sum"))
    calendar = pd.DatetimeIndex(sorted(preds["date"].unique()))
    grouped = grouped.reindex(calendar).fillna({"gross": 0.0, "n_open": 0.0})

    out = pd.DataFrame(index=calendar)
    out["ret"] = grouped["gross"] / float(max_positions)
    out["n_open"] = grouped["n_open"]
    out["exposure"] = grouped["n_open"] / float(max_positions)
    return out


def performance(daily: pd.DataFrame) -> dict[str, float]:
    """Annualised statistics from a daily return series (costs already charged)."""
    if daily.empty:
        return {}
    ret = daily["ret"].astype("float64")

    equity = (1.0 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(ret.std() * np.sqrt(TRADING_DAYS))
    downside = float(ret.clip(upper=0.0).std() * np.sqrt(TRADING_DAYS))
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())

    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS)) if ret.std() > 0 else float("nan"),
        "sortino": float(ret.mean() * TRADING_DAYS / downside) if downside > 0 else float("nan"),
        "max_drawdown": dd,
        "calmar": cagr / abs(dd) if dd < 0 else float("nan"),
        "avg_exposure": float(daily["exposure"].mean()),
        "avg_positions": float(daily["n_open"].mean()),
        "win_days": float((ret > 0).mean()),
        "days": int(len(ret)),
    }


def benchmark_performance(benchmark: pd.DataFrame, start, end) -> dict[str, float]:
    """Buy-and-hold statistics for the benchmark over the same window."""
    frame = benchmark_daily(benchmark, start, end)
    return performance(frame) if not frame.empty else {}


def benchmark_daily(benchmark: pd.DataFrame, start, end) -> pd.DataFrame:
    """Daily benchmark returns shaped like a portfolio return frame."""
    if benchmark.empty:
        return pd.DataFrame(columns=["ret", "n_open", "exposure"])
    b = benchmark.set_index("date").sort_index()
    b = b.loc[(b.index >= pd.Timestamp(start)) & (b.index <= pd.Timestamp(end))]
    if b.empty:
        return pd.DataFrame(columns=["ret", "n_open", "exposure"])
    ret = b["close"].pct_change().dropna()
    return pd.DataFrame({"ret": ret, "n_open": 1.0, "exposure": 1.0})


def relative_performance(daily: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, float]:
    """Beta-adjusted comparison against the benchmark.

    A long-only strategy that stays fully invested in a rising market will post a
    high CAGR whether or not it has any skill. Regressing its daily returns on
    the benchmark separates the part that is simply market exposure (beta) from
    the part that is not (alpha), which is the only figure worth quoting.
    """
    if daily.empty or benchmark.empty:
        return {}
    joined = pd.DataFrame({"p": daily["ret"], "b": benchmark["ret"]}).dropna()
    if len(joined) < 60:
        return {}

    var = float(joined["b"].var())
    beta = float(joined["p"].cov(joined["b"]) / var) if var > 0 else float("nan")
    daily_alpha = float(joined["p"].mean() - beta * joined["b"].mean())
    excess = joined["p"] - joined["b"]

    return {
        "beta": beta,
        "alpha_annual": daily_alpha * TRADING_DAYS,
        "excess_return_annual": float(excess.mean() * TRADING_DAYS),
        "information_ratio": (
            float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else float("nan")
        ),
        "correlation": float(joined["p"].corr(joined["b"])),
    }
