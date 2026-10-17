"""Event-driven, per-symbol trade simulation.

This is not a portfolio optimiser. The objective is to answer one question as
directly as possible: *if you had acted on the buy signals, entered on the next
open, and managed the position with the model's own target and stop, would you
have made money after costs?*

Rules that keep the answer honest:

* A signal generated from bar ``t``'s close fills at bar ``t+1``'s **open**.
  Never at the close that produced it.
* Exits are checked against intraday high/low from ``t+2`` onward. When a bar
  touches both the target and the stop, the **stop** is assumed to have hit
  first — without intraday data the pessimistic assumption is the only safe one.
* Target and stop come from the model's predicted excursions, floored by ATR so
  a tiny prediction cannot produce a hair-trigger stop.
* Commission plus slippage is charged on both legs.
* Concurrent positions per symbol are capped, so a persistent signal cannot
  silently turn into a leveraged bet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BUY_ACTIONS = ("BUY", "STRONG_BUY")
SELL_ACTIONS = ("SELL", "STRONG_SELL")


@dataclass
class BacktestConfig:
    horizon: int = 10
    max_hold: int = 15
    cost_bps: float = 10.0  # round-trip commission + slippage, per leg
    target_atr_mult: float = 1.0  # scales the predicted MFE into a target
    stop_atr_mult: float = 1.0  # scales the predicted MAE into a stop
    min_stop_atr: float = 1.0  # stop never tighter than this many ATRs
    max_stop_atr: float = 3.0
    min_target_atr: float = 1.0
    entry_actions: tuple[str, ...] = BUY_ACTIONS
    #: Labels that close an open position. Empty means "use the built-in
    #: opposite state labels", which is only meaningful for the state signal;
    #: trading turning points requires naming the opposite event explicitly.
    exit_actions: tuple[str, ...] = ()
    exit_on_opposite: bool = True
    max_concurrent_per_symbol: int = 1
    direction: str = "long"
    #: Column holding the entry decision. Point it at ``turn_action`` to trade
    #: discrete turning-point events instead of the persistent state signal.
    action_column: str = "action"


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    stats: dict[str, float]
    equity: pd.Series = field(default_factory=pd.Series)


def _simulate_symbol(
    frame: pd.DataFrame, cfg: BacktestConfig, sign: int
) -> list[dict[str, object]]:
    """Walk one symbol's bars and emit closed trades."""
    dates = frame["date"].to_numpy()
    action = frame[cfg.action_column].to_numpy()
    open_next = frame["open_next"].to_numpy(dtype="float64")
    high = frame["high"].to_numpy(dtype="float64")
    low = frame["low"].to_numpy(dtype="float64")
    close = frame["close"].to_numpy(dtype="float64")
    price = frame["price"].to_numpy(dtype="float64")
    atr = (frame["atr_pct_label"].to_numpy(dtype="float64") * close)
    exp_gain = frame["exp_gain_pct"].to_numpy(dtype="float64")
    exp_loss = frame["exp_loss_pct"].to_numpy(dtype="float64")
    score = frame["score"].to_numpy(dtype="float64")

    entry_set = set(cfg.entry_actions)
    opposite = set(cfg.exit_actions) if cfg.exit_actions else set(
        SELL_ACTIONS if sign > 0 else BUY_ACTIONS
    )
    cost = cfg.cost_bps / 10_000.0

    trades: list[dict[str, object]] = []
    n = len(frame)
    open_until = -1  # bar index before which no new entry may be taken

    for i in range(n - 2):
        if i <= open_until and cfg.max_concurrent_per_symbol <= 1:
            continue
        if action[i] not in entry_set:
            continue
        entry = open_next[i]
        if not np.isfinite(entry) or entry <= 0:
            continue

        bar_atr = atr[i]
        if not np.isfinite(bar_atr) or bar_atr <= 0:
            continue

        # Predicted excursions define the trade envelope; ATR bounds keep it sane.
        raw_target = abs(exp_gain[i] if sign > 0 else exp_loss[i]) * close[i] * cfg.target_atr_mult
        raw_stop = abs(exp_loss[i] if sign > 0 else exp_gain[i]) * close[i] * cfg.stop_atr_mult
        target_dist = max(raw_target, cfg.min_target_atr * bar_atr)
        stop_dist = float(np.clip(raw_stop, cfg.min_stop_atr * bar_atr, cfg.max_stop_atr * bar_atr))

        target = entry + sign * target_dist
        stop = entry - sign * stop_dist

        exit_idx, exit_price, reason = None, None, "timeout"
        last = min(i + 1 + cfg.max_hold, n - 1)
        for j in range(i + 1, last + 1):
            hit_stop = (low[j] <= stop) if sign > 0 else (high[j] >= stop)
            hit_target = (high[j] >= target) if sign > 0 else (low[j] <= target)
            if hit_stop:  # pessimistic: stop wins any same-bar ambiguity
                exit_idx, exit_price, reason = j, stop, "stop"
                break
            if hit_target:
                exit_idx, exit_price, reason = j, target, "target"
                break
            if cfg.exit_on_opposite and j > i and action[j] in opposite:
                exit_idx, exit_price, reason = j, close[j], "signal"
                break
        if exit_idx is None:
            exit_idx, exit_price, reason = last, close[last], "timeout"

        gross = sign * (exit_price / entry - 1.0)
        net = gross - 2.0 * cost
        trades.append(
            {
                "symbol": frame["symbol"].iat[0],
                "entry_date": dates[i + 1],
                "exit_date": dates[exit_idx],
                "bars_held": exit_idx - (i + 1) + 1,
                "direction": "long" if sign > 0 else "short",
                "entry_px": float(entry * price[i] / close[i]) if close[i] > 0 else float(entry),
                "entry_adj": float(entry),
                "exit_adj": float(exit_price),
                "signal_score": float(score[i]),
                "action": action[i],
                "exit_reason": reason,
                "gross_return": float(gross),
                "net_return": float(net),
            }
        )
        open_until = exit_idx

    return trades


def run_backtest(preds: pd.DataFrame, cfg: BacktestConfig | None = None) -> BacktestResult:
    """Simulate every signal in ``preds`` and summarise the outcome."""
    cfg = cfg or BacktestConfig()
    required = {"symbol", "date", cfg.action_column, "open_next", "high", "low", "close"}
    missing = required - set(preds.columns)
    if missing:
        raise KeyError(f"Backtest input missing columns: {sorted(missing)}")

    sign = 1 if cfg.direction == "long" else -1
    frames = (
        preds.sort_values(["symbol", "date"], kind="stable")
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for _, group in frames.groupby("symbol", sort=False):
        rows.extend(_simulate_symbol(group.reset_index(drop=True), cfg, sign))

    trades = pd.DataFrame(rows)
    return BacktestResult(trades=trades, stats=summarise(trades), equity=_equity_curve(trades))


def summarise(trades: pd.DataFrame) -> dict[str, float]:
    """Trade-level performance statistics."""
    if trades.empty:
        return {"trades": 0}
    ret = trades["net_return"]
    wins, losses = ret[ret > 0], ret[ret <= 0]
    gross_win, gross_loss = float(wins.sum()), float(-losses.sum())
    avg_hold = float(trades["bars_held"].mean())

    # Per-trade Sharpe annualised by the average holding period — the right
    # denominator when trades are event-driven rather than daily.
    turns_per_year = 252.0 / max(avg_hold, 1.0)
    sharpe = (
        float(ret.mean() / ret.std() * np.sqrt(turns_per_year)) if float(ret.std()) > 0 else float("nan")
    )

    stats = {
        "trades": int(len(trades)),
        "win_rate": float((ret > 0).mean()),
        "avg_return": float(ret.mean()),
        "median_return": float(ret.median()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf"),
        "expectancy": float(ret.mean()),
        "avg_bars_held": avg_hold,
        "sharpe_per_trade": sharpe,
        "total_return_sum": float(ret.sum()),
        "best": float(ret.max()),
        "worst": float(ret.min()),
    }
    for reason, share in (trades["exit_reason"].value_counts(normalize=True)).items():
        stats[f"exit_{reason}"] = float(share)
    return stats


def _equity_curve(trades: pd.DataFrame) -> pd.Series:
    """Equal-weight compounded equity, ordered by exit date."""
    if trades.empty:
        return pd.Series(dtype="float64")
    ordered = trades.sort_values("exit_date")
    daily = ordered.groupby("exit_date")["net_return"].mean()
    return (1.0 + daily).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def baseline_random(
    preds: pd.DataFrame, cfg: BacktestConfig | None = None, seed: int = 7
) -> BacktestResult:
    """Same machinery, but entries are shuffled within each symbol.

    This is the control group. If the real signal cannot beat its own actions
    randomly re-dated, the edge is in the exit rules, not the prediction.
    """
    cfg = cfg or BacktestConfig()
    rng = np.random.default_rng(seed)
    shuffled = preds.sort_values(["symbol", "date"], kind="stable").copy()
    parts = []
    for _, group in shuffled.groupby("symbol", sort=False):
        g = group.copy()
        g[cfg.action_column] = rng.permutation(g[cfg.action_column].to_numpy())
        parts.append(g)
    return run_backtest(pd.concat(parts, ignore_index=True), cfg)


def buy_and_hold(preds: pd.DataFrame) -> dict[str, float]:
    """Equal-weight buy-and-hold over the same window, for reference."""
    if preds.empty:
        return {}
    firsts = preds.sort_values("date").groupby("symbol")["close"].first()
    lasts = preds.sort_values("date").groupby("symbol")["close"].last()
    rets = (lasts / firsts - 1.0).dropna()
    span_days = (pd.Timestamp(preds["date"].max()) - pd.Timestamp(preds["date"].min())).days
    years = max(span_days / 365.25, 1e-6)
    mean = float(rets.mean())
    return {
        "symbols": int(rets.size),
        "mean_total_return": mean,
        "median_total_return": float(rets.median()),
        "annualised": float((1.0 + mean) ** (1.0 / years) - 1.0),
        "years": years,
    }
