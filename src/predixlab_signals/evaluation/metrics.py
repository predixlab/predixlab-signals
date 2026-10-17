"""Out-of-sample evaluation: does the score actually locate highs and lows?

Five independent views, because any single one is easy to fool:

* **Rank IC** — cross-sectional rank correlation of the score against the
  realised edge. Answers "does it order names correctly on a given day?".
* **Time-series IC** — the same correlation *within* each symbol over time.
  Answers "does it time a single stock?", which is the actual objective here
  and is a strictly harder test than cross-sectional ranking.
* **Action buckets** — realised forward outcomes grouped by recommendation.
  Answers "would following the advice have paid?".
* **Turning-point detection** — precision and recall of signals against zigzag
  pivots that were derived without any reference to the model.
* **Probability quality** — AUC, Brier score and a decile calibration curve for
  the barrier head, so the confidence number means something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BUY_ACTIONS = {"BUY", "STRONG_BUY"}
SELL_ACTIONS = {"SELL", "STRONG_SELL"}


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 5:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def cross_sectional_ic(
    preds: pd.DataFrame, score_col: str = "score", target_col: str = "y_edge", min_names: int = 10
) -> dict[str, float]:
    """Daily cross-sectional rank IC summary."""
    sub = preds[["date", score_col, target_col]].dropna()
    if sub.empty:
        return {}
    daily = (
        sub.groupby("date")
        .apply(
            lambda g: _spearman(g[score_col], g[target_col]) if len(g) >= min_names else np.nan,
            include_groups=False,
        )
        .dropna()
    )
    if daily.empty:
        return {}
    mean, std = float(daily.mean()), float(daily.std())
    n = int(daily.size)
    return {
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_icir": mean / std * np.sqrt(252) if std > 0 else float("nan"),
        "rank_ic_tstat": mean / std * np.sqrt(n) if std > 0 else float("nan"),
        "rank_ic_positive_rate": float((daily > 0).mean()),
        "rank_ic_days": n,
    }


def time_series_ic(
    preds: pd.DataFrame, score_col: str = "score", target_col: str = "y_edge", min_obs: int = 60
) -> dict[str, float]:
    """Per-symbol IC over time — the timing question, not the ranking question."""
    sub = preds[["symbol", score_col, target_col]].dropna()
    if sub.empty:
        return {}
    per_symbol = (
        sub.groupby("symbol")
        .apply(
            lambda g: _spearman(g[score_col], g[target_col]) if len(g) >= min_obs else np.nan,
            include_groups=False,
        )
        .dropna()
    )
    if per_symbol.empty:
        return {}
    return {
        "ts_ic_mean": float(per_symbol.mean()),
        "ts_ic_median": float(per_symbol.median()),
        "ts_ic_positive_rate": float((per_symbol > 0).mean()),
        "ts_ic_symbols": int(per_symbol.size),
    }


def action_buckets(preds: pd.DataFrame) -> pd.DataFrame:
    """Realised forward outcomes grouped by recommendation."""
    cols = ["action", "fwd_ret", "fwd_mfe", "fwd_mae", "y_swing_pos", "exp_edge_pct"]
    sub = preds[[c for c in cols if c in preds.columns]].dropna(subset=["fwd_ret"])
    if sub.empty:
        return pd.DataFrame()
    grouped = sub.groupby("action")
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "share": grouped.size() / len(sub),
            "fwd_ret_mean": grouped["fwd_ret"].mean(),
            "fwd_ret_median": grouped["fwd_ret"].median(),
            "win_rate": grouped["fwd_ret"].apply(lambda s: float((s > 0).mean())),
            "mfe_mean": grouped["fwd_mfe"].mean(),
            "mae_mean": grouped["fwd_mae"].mean(),
            "realized_swing_pos": grouped["y_swing_pos"].mean(),
        }
    )
    # Reward/risk actually realised: how much upside per unit of drawdown.
    table["mfe_mae_ratio"] = table["mfe_mean"] / table["mae_mean"].abs().replace(0, np.nan)
    order = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]
    return table.reindex([a for a in order if a in table.index])


def turning_point_report(preds: pd.DataFrame, tolerance: int = 3) -> dict[str, float]:
    """Precision / recall of signals against zigzag pivots.

    A buy signal counts as correct if a confirmed swing **low** occurs within
    ``tolerance`` bars either side; a sell signal likewise for a swing high.
    Recall asks the mirror question: of all real pivots, how many did we flag?
    """
    if "pivot" not in preds.columns:
        return {}
    frame = preds.sort_values(["symbol", "date"], kind="stable")

    stats = dict.fromkeys(
        ("tp_buy", "sig_buy", "tp_sell", "sig_sell", "lows", "lows_hit", "highs", "highs_hit"), 0
    )

    for _, group in frame.groupby("symbol", sort=False):
        pivot = group["pivot"].to_numpy()
        action = group["action"].to_numpy()
        if pivot.size == 0:
            continue
        is_low = pivot == -1.0
        is_high = pivot == 1.0
        buy = np.isin(action, list(BUY_ACTIONS))
        sell = np.isin(action, list(SELL_ACTIONS))

        stats["sig_buy"] += int(buy.sum())
        stats["tp_buy"] += int((buy & _dilate(is_low, tolerance)).sum())
        stats["sig_sell"] += int(sell.sum())
        stats["tp_sell"] += int((sell & _dilate(is_high, tolerance)).sum())
        stats["lows"] += int(is_low.sum())
        stats["lows_hit"] += int((is_low & _dilate(buy, tolerance)).sum())
        stats["highs"] += int(is_high.sum())
        stats["highs_hit"] += int((is_high & _dilate(sell, tolerance)).sum())

    span = 2 * tolerance + 1
    base_low = float((frame["pivot"] == -1.0).mean()) * span
    base_high = float((frame["pivot"] == 1.0).mean()) * span
    out = {
        "buy_precision": _ratio(stats["tp_buy"], stats["sig_buy"]),
        "buy_recall": _ratio(stats["lows_hit"], stats["lows"]),
        "sell_precision": _ratio(stats["tp_sell"], stats["sig_sell"]),
        "sell_recall": _ratio(stats["highs_hit"], stats["highs"]),
        "pivot_low_base_rate": base_low,
        "pivot_high_base_rate": base_high,
    }
    # Lift over chance is the only number that matters here: precision alone is
    # meaningless when the base rate of "near a pivot" is already high.
    out["buy_lift"] = _ratio(out["buy_precision"], base_low)
    out["sell_lift"] = _ratio(out["sell_precision"], base_high)
    return out


def probability_report(preds: pd.DataFrame, deciles: int = 10) -> dict[str, object]:
    """AUC, Brier score and a calibration table for the barrier probability."""
    sub = preds[["p_up", "y_barrier"]].dropna()
    sub = sub[sub["y_barrier"].isin([-1.0, 1.0])]
    if len(sub) < 100:
        return {}
    y = (sub["y_barrier"] > 0).astype(int).to_numpy()
    p = sub["p_up"].to_numpy()

    order = np.argsort(p)
    ranks = np.empty(p.size, dtype="float64")
    ranks[order] = np.arange(1, p.size + 1)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = (
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        if n_pos and n_neg
        else float("nan")
    )

    bucket = pd.qcut(p, deciles, labels=False, duplicates="drop")
    calib = (
        pd.DataFrame({"bucket": bucket, "p": p, "y": y})
        .groupby("bucket")
        .agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    )
    return {
        "auc": float(auc),
        "brier": float(np.mean((p - y) ** 2)),
        "base_rate": float(y.mean()),
        "calibration": calib.reset_index().to_dict(orient="records"),
    }


def pivot_hold_report(
    preds: pd.DataFrame, side: str = "low", quantiles: tuple[float, ...] = (0.99, 0.95, 0.90)
) -> dict[str, object]:
    """Measure whether the model predicts that today's extreme will hold."""
    prob_col = "p_pivot_low" if side == "low" else "p_pivot_high"
    label_col = "y_pivot_low_hold" if side == "low" else "y_pivot_high_hold"
    needed = {prob_col, label_col}
    if not needed <= set(preds.columns):
        return {}

    frame = preds[list(needed)].dropna()
    if len(frame) < 500:
        return {}

    y = frame[label_col].to_numpy(dtype="float64")
    p = frame[prob_col].to_numpy(dtype="float64")
    base = float(y.mean())

    order = np.argsort(p)
    ranks = np.empty(p.size, dtype="float64")
    ranks[order] = np.arange(1, p.size + 1)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = (
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        if n_pos and n_neg
        else float("nan")
    )

    buckets = []
    for q in quantiles:
        selected = p >= np.quantile(p, q)
        if not selected.any():
            continue
        precision = float(y[selected].mean())
        buckets.append(
            {
                "top_frac": round(1.0 - q, 4),
                "n": int(selected.sum()),
                "precision": precision,
                "lift": precision / base if base else float("nan"),
            }
        )

    return {
        "side": side,
        "auc": float(auc),
        "base_rate": base,
        "rows": int(len(frame)),
        "buckets": buckets,
    }


def evaluate(preds: pd.DataFrame, tolerance: int = 3) -> dict[str, object]:
    """Run every view and return a single report dictionary."""
    report: dict[str, object] = {
        "rows": int(len(preds)),
        "symbols": int(preds["symbol"].nunique()),
        "start": str(pd.Timestamp(preds["date"].min()).date()),
        "end": str(pd.Timestamp(preds["date"].max()).date()),
    }
    report["ic_edge"] = cross_sectional_ic(preds, "score", "y_edge")
    report["ic_swing_pos"] = cross_sectional_ic(preds, "score", "y_swing_pos")
    report["ic_fwd_ret"] = cross_sectional_ic(preds, "score", "fwd_ret")
    report["ts_ic_edge"] = time_series_ic(preds, "score", "y_edge")
    report["ts_ic_fwd_ret"] = time_series_ic(preds, "score", "fwd_ret")
    report["buckets"] = action_buckets(preds).reset_index().to_dict(orient="records")
    report["turning_points"] = turning_point_report(preds, tolerance=tolerance)
    report["probability"] = probability_report(preds)
    report["pivot_low_detection"] = pivot_hold_report(preds, "low")
    report["pivot_high_detection"] = pivot_hold_report(preds, "high")
    return report


# --------------------------------------------------------------------------
def _dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    """Widen a boolean mask by ``tolerance`` bars in both directions."""
    if tolerance <= 0:
        return mask
    out = mask.copy()
    for shift in range(1, tolerance + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")
