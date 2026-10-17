"""TopNDropouts portfolio for AlphaRankModel predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class TopNDropoutsConfig:
    """Formal AlphaRank portfolio and bear-state configuration."""

    top_n: int = 12
    n_drop: int = 1
    cost_bps: float = 10.0
    min_consensus_percentile: float = 0.55
    bear_entry_market_return20: float = 0.0
    bear_entry_sma50: float = 0.0
    bear_entry_breadth5: float = 0.50
    bear_entry_confirm_days: int = 2
    bear_candidate_ratio20: float = 0.90
    bear_candidate_drop5: float = -0.15
    bear_candidate_max_vol_ratio: float = 1.50
    bear_reversal_return5: float = 0.01
    bear_reversal_breadth: float = 0.50
    bear_reversal_confirm_days: int = 2
    bear_strong_reversal_return1: float = 0.015
    bear_reentry_grace_days: int = 5
    adaptive_volatility: bool = True
    volatility_window: int = 20
    volatility_lookback: int = 252
    volatility_targets: tuple[float, ...] = (
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
    )

    def validate(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.n_drop <= 0 or self.n_drop > self.top_n:
            raise ValueError("n_drop must be between 1 and top_n")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if not 0.0 <= self.min_consensus_percentile <= 1.0:
            raise ValueError(
                "min_consensus_percentile must be between 0 and 1"
            )
        if self.bear_entry_confirm_days <= 0:
            raise ValueError("bear_entry_confirm_days must be positive")
        if self.bear_reversal_confirm_days <= 0:
            raise ValueError("bear_reversal_confirm_days must be positive")
        if self.bear_reentry_grace_days < 0:
            raise ValueError("bear_reentry_grace_days cannot be negative")
        if not 0.0 < self.bear_candidate_ratio20 <= 1.0:
            raise ValueError("bear_candidate_ratio20 must be in (0, 1]")
        if not 0.0 <= self.bear_entry_breadth5 <= 1.0:
            raise ValueError("bear_entry_breadth5 must be between 0 and 1")
        if not 0.0 <= self.bear_reversal_breadth <= 1.0:
            raise ValueError("bear_reversal_breadth must be between 0 and 1")
        if self.volatility_window <= 1:
            raise ValueError("volatility_window must exceed one")
        if self.volatility_lookback < 60:
            raise ValueError("volatility_lookback must be at least 60")
        if not self.volatility_targets or min(self.volatility_targets) <= 0:
            raise ValueError("volatility_targets must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TopNDropoutsResult:
    daily: pd.DataFrame
    selections: pd.DataFrame


def _prepare_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.sort_values(["symbol", "date"], kind="stable").copy()
    calendar = pd.DatetimeIndex(
        pd.to_datetime(frame["date"]).drop_duplicates().sort_values()
    )
    frame["entry_date"] = frame["date"].map(
        pd.Series(calendar, index=calendar).shift(-1)
    )
    frame["return_date"] = frame["date"].map(
        pd.Series(calendar, index=calendar).shift(-2)
    )
    close_lookup = frame.set_index(["symbol", "date"])["close"]
    entry_keys = pd.MultiIndex.from_arrays(
        [frame["symbol"], frame["entry_date"]],
        names=["symbol", "date"],
    )
    return_keys = pd.MultiIndex.from_arrays(
        [frame["symbol"], frame["return_date"]],
        names=["symbol", "date"],
    )
    entry_close = close_lookup.reindex(entry_keys).to_numpy(dtype="float64")
    return_close = close_lookup.reindex(return_keys).to_numpy(dtype="float64")
    frame["strategy_return"] = return_close / entry_close - 1.0
    breadth = frame.groupby("date", sort=True)["ret_20"].apply(
        lambda values: float((values > 0.0).mean())
    )
    frame["breadth"] = frame["date"].map(breadth)
    frame["breadth5"] = frame["date"].map(breadth.rolling(5).mean())
    frame["breadth_change5"] = frame["date"].map(breadth - breadth.shift(5))
    return frame


def _eligible(
    block: pd.DataFrame,
    config: TopNDropoutsConfig,
) -> pd.DataFrame:
    consensus_rank = block["alpha_consensus"].rank(pct=True)
    return block[
        (consensus_rank >= config.min_consensus_percentile)
        & (block["ret_20"] > 0.0)
        & (block["px_sma60"] > 0.0)
    ]


def _scaled_history_score(
    returns: pd.Series,
    target_volatility: float,
    cost: float,
    window: int,
) -> float:
    realized = returns.rolling(window).std().shift(1) * np.sqrt(TRADING_DAYS)
    exposure = (target_volatility / realized).clip(upper=1.0).fillna(1.0)
    scaled = (
        exposure * returns
        - exposure.diff().abs().fillna(0.0) * cost
    )
    if len(scaled) < 60 or float(scaled.std()) <= 0:
        return -np.inf
    equity = (1.0 + scaled).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    drawdown = float((equity / equity.cummax() - 1.0).min())
    sharpe = float(
        scaled.mean() / scaled.std() * np.sqrt(TRADING_DAYS)
    )
    return sharpe + 0.5 * total + 0.5 * drawdown


def _select_target_volatility(
    shadow_returns: list[float],
    config: TopNDropoutsConfig,
    cost: float,
) -> float | None:
    if not config.adaptive_volatility:
        return None
    if len(shadow_returns) < config.volatility_lookback:
        return None
    history = pd.Series(
        shadow_returns[-config.volatility_lookback :],
        dtype="float64",
    )
    return max(
        config.volatility_targets,
        key=lambda target: _scaled_history_score(
            history,
            target,
            cost,
            config.volatility_window,
        ),
    )


def run_topn_dropouts(
    predictions: pd.DataFrame,
    config: TopNDropoutsConfig | None = None,
    portfolio_start: str | pd.Timestamp | None = None,
) -> TopNDropoutsResult:
    """Run AlphaRank TopNDropouts with next-close execution."""
    config = config or TopNDropoutsConfig()
    config.validate()
    required = {
        "symbol",
        "date",
        "close",
        "ret_20",
        "px_sma60",
        "alpha_score",
        "alpha_consensus",
        "mkt_ret_1",
        "mkt_ret_5",
        "mkt_ret_20",
        "mkt_px_sma50",
        "mkt_vol_ratio",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(
            f"AlphaRank TopN input missing columns: {sorted(missing)}"
        )

    frame = _prepare_panel(predictions)
    calendar = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    candidate_counts = (
        frame.groupby("date", sort=True)
        .apply(lambda block: len(_eligible(block, config)), include_groups=False)
        .reindex(calendar, fill_value=0)
        .astype("float64")
    )
    frame["candidate_ratio20"] = frame["date"].map(
        candidate_counts / candidate_counts.rolling(20).median()
    )
    frame["candidate_drop5"] = frame["date"].map(
        candidate_counts / candidate_counts.shift(5) - 1.0
    )

    holdings: list[str] = []
    daily_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    start_date = (
        pd.Timestamp(portfolio_start)
        if portfolio_start is not None
        else None
    )
    cost = config.cost_bps / 10_000.0
    bear_active = False
    bear_entry_count = 0
    candidate_entry_count = 0
    reversal_count = 0
    reentry_grace = 0
    shadow_returns: list[float] = []
    current_month: tuple[int, int] | None = None
    target_volatility: float | None = None
    prior_exposure = 1.0
    prior_positions = 0
    portfolio_reset = start_date is None

    for signal_date, raw in frame.groupby("date", sort=True):
        trend_entry = (
            float(raw["mkt_ret_20"].iloc[0])
            <= config.bear_entry_market_return20
            and float(raw["mkt_px_sma50"].iloc[0])
            <= config.bear_entry_sma50
            and float(raw["breadth5"].iloc[0])
            <= config.bear_entry_breadth5
        )
        candidate_entry = (
            float(raw["candidate_ratio20"].iloc[0])
            <= config.bear_candidate_ratio20
            and float(raw["candidate_drop5"].iloc[0])
            <= config.bear_candidate_drop5
            and float(raw["mkt_ret_20"].iloc[0]) <= 0.0
            and float(raw["mkt_px_sma50"].iloc[0]) <= 0.0
            and float(raw["mkt_vol_ratio"].iloc[0])
            <= config.bear_candidate_max_vol_ratio
        )
        reversal = (
            float(raw["mkt_ret_5"].iloc[0])
            >= config.bear_reversal_return5
            and float(raw["breadth"].iloc[0])
            >= config.bear_reversal_breadth
            and float(raw["breadth_change5"].iloc[0]) > 0.0
        )
        strong_reversal = (
            float(raw["mkt_ret_1"].iloc[0])
            >= config.bear_strong_reversal_return1
        )

        bear_trigger = False
        bear_reversal = False
        bear_entry_reason: str | None = None
        if not bear_active:
            if reentry_grace > 0:
                reentry_grace -= 1
                bear_entry_count = 0
                candidate_entry_count = 0
            else:
                bear_entry_count = (
                    bear_entry_count + 1 if trend_entry else 0
                )
                candidate_entry_count = (
                    candidate_entry_count + 1 if candidate_entry else 0
                )
                if (
                    bear_entry_count >= config.bear_entry_confirm_days
                    or candidate_entry_count >= 1
                ):
                    bear_active = True
                    bear_trigger = True
                    bear_entry_reason = (
                        "candidate_collapse"
                        if candidate_entry_count >= 1
                        else "trend_breadth"
                    )
                    bear_entry_count = 0
                    candidate_entry_count = 0
                    reversal_count = 0
        else:
            reversal_count = reversal_count + 1 if reversal else 0
            if strong_reversal or (
                reversal_count >= config.bear_reversal_confirm_days
            ):
                bear_active = False
                bear_reversal = True
                reentry_grace = config.bear_reentry_grace_days
                bear_entry_count = 0
                candidate_entry_count = 0
                reversal_count = 0

        portfolio_active = (
            start_date is None or pd.Timestamp(signal_date) >= start_date
        )
        if portfolio_active and not portfolio_reset:
            holdings = []
            prior_exposure = 0.0
            prior_positions = 0
            portfolio_reset = True

        target_positions = 0 if bear_active else config.top_n
        regime = "bear" if bear_active else "full"
        candidates = _eligible(raw, config).sort_values(
            ["alpha_score", "symbol"],
            ascending=[False, True],
            kind="stable",
        )
        score = raw.set_index("symbol")["alpha_score"]
        eligible_symbols = set(candidates["symbol"])
        absent_holdings = [
            symbol for symbol in holdings if symbol not in score.index
        ]
        current = [symbol for symbol in holdings if symbol in score.index]
        exit_priority = [
            symbol for symbol in current if symbol not in eligible_symbols
        ]
        ranked_current = sorted(
            (symbol for symbol in current if symbol in eligible_symbols),
            key=lambda symbol: (float(score.loc[symbol]), symbol),
        )
        required_reduction = max(len(current) - target_positions, 0)
        rotation_allowance = max(config.n_drop - len(absent_holdings), 0)
        sell_limit = max(
            required_reduction,
            min(rotation_allowance, len(current)),
        )
        sold_present = list(
            dict.fromkeys(exit_priority + ranked_current)
        )[:sell_limit]
        sold = absent_holdings + sold_present
        sold_set = set(sold)
        kept = [symbol for symbol in current if symbol not in sold_set]
        kept_set = set(kept)
        bought = [
            symbol
            for symbol in candidates["symbol"]
            if symbol not in kept_set and symbol not in sold_set
        ][: max(target_positions - len(kept), 0)]
        holdings = kept + bought

        by_symbol = raw.set_index("symbol")
        returns = by_symbol.reindex(holdings)["strategy_return"].fillna(0.0)
        gross_return = float(returns.sum() / config.top_n)
        turnover = (len(sold) + len(bought)) / float(config.top_n)
        shadow_return = gross_return - turnover * cost
        month = (pd.Timestamp(signal_date).year, pd.Timestamp(signal_date).month)
        if month != current_month:
            current_month = month
            target_volatility = _select_target_volatility(
                shadow_returns,
                config,
                cost,
            )
        if target_positions == 0:
            exposure = 0.0
        elif target_volatility is None or len(shadow_returns) < config.volatility_window:
            exposure = 1.0
        else:
            realized = float(
                pd.Series(
                    shadow_returns[-config.volatility_window :],
                    dtype="float64",
                ).std()
                * np.sqrt(TRADING_DAYS)
            )
            exposure = (
                min(1.0, target_volatility / realized)
                if realized > 0
                else 1.0
            )
        exposure_change_cost = abs(exposure - prior_exposure) * cost
        if prior_positions == 0 and len(holdings) > 0:
            exposure_change_cost = 0.0
        portfolio_return = exposure * shadow_return - exposure_change_cost
        prior_exposure = exposure
        prior_positions = len(holdings)
        shadow_returns.append(shadow_return)
        return_date = raw["return_date"].iloc[0]
        entry_date = raw["entry_date"].iloc[0]
        if portfolio_active and pd.notna(return_date):
            daily_rows.append(
                {
                    "date": return_date,
                    "signal_date": signal_date,
                    "ret": portfolio_return,
                    "gross_return": gross_return,
                    "shadow_return": shadow_return,
                    "turnover": turnover,
                    "positions": len(holdings),
                    "target_positions": target_positions,
                    "exposure_scale": exposure,
                    "target_volatility": target_volatility,
                    "cash_weight": 1.0
                    - exposure * len(holdings) / float(config.top_n),
                    "exposure_regime": regime,
                    "bear_trigger": bear_trigger,
                    "bear_reversal": bear_reversal,
                    "bear_entry_reason": bear_entry_reason,
                    "candidates": len(candidates),
                    "execution_price": "next_close",
                }
            )
        for rank, symbol in enumerate(holdings, start=1):
            if not portfolio_active:
                continue
            selection_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "return_date": return_date,
                    "symbol": symbol,
                    "rank": rank,
                    "score": float(score.loc[symbol]),
                    "status": "bought" if symbol in bought else "held",
                    "stock_return": float(returns.get(symbol, 0.0)),
                    "execution_price": "next_close",
                }
            )

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily = daily.set_index("date").sort_index()
    selection_columns = [
        "signal_date",
        "entry_date",
        "return_date",
        "symbol",
        "rank",
        "score",
        "status",
        "stock_return",
        "execution_price",
    ]
    return TopNDropoutsResult(
        daily=daily,
        selections=pd.DataFrame(selection_rows, columns=selection_columns),
    )


def topn_performance(
    daily: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Summarize a daily, after-cost portfolio return series."""
    if daily.empty:
        return {}
    report = daily
    if start is not None:
        report = report[report.index >= pd.Timestamp(start)]
    if report.empty:
        return {}

    ret = report["ret"].astype("float64")
    equity = (1.0 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    peak = equity.cummax()
    yearly = (1.0 + ret).groupby(ret.index.year).prod() - 1.0
    result = {
        "start": str(pd.Timestamp(report.index.min()).date()),
        "end": str(pd.Timestamp(report.index.max()).date()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "volatility": float(ret.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": (
            float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))
            if ret.std() > 0
            else float("nan")
        ),
        "max_drawdown": float((equity / peak - 1.0).min()),
        "avg_turnover": float(report["turnover"].mean()),
        "avg_positions": float(report["positions"].mean()),
        "avg_target_positions": float(report["target_positions"].mean()),
        "avg_cash_weight": float(report["cash_weight"].mean()),
        "avg_candidates": float(report["candidates"].mean()),
        "yearly_returns": {
            str(year): float(value) for year, value in yearly.items()
        },
        "regime_days": {
            str(regime): int(count)
            for regime, count in report["exposure_regime"].value_counts().items()
        },
    }
    return result
