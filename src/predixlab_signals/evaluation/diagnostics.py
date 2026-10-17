"""Systematic diagnostics — the antidote to tuning on anecdotes.

Chasing individual missed signals is how a filter gets overfitted: each fix is
justified by one symbol on one date, and the aggregate quietly degrades. This
module answers the same questions in aggregate, so a change can be judged before
any single case is inspected.

Three views, each addressing a failure mode that individual cases cannot reveal:

* :func:`trend_profile` — does the model merely fade whatever moved? Splits
  signals and future-hold outcomes by trailing trend, so the model's tilt can
  be compared against the tilt that genuinely exists.
* :func:`within_trend_discrimination` — the question that actually matters.
  Real lows cluster after declines (11.7% vs 2.5% of bars), so a detector that
  simply fires on losers looks skilled. This measures lift *inside* each trend
  bucket, where that free tilt is held constant.
* :func:`universe_report` — how much of the panel is outside the index, an
  artefact, or otherwise not a tradeable single stock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_EDGES = [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf]
TREND_LABELS = ["steep_down", "down", "flat", "up", "steep_up"]


def add_trend_context(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Attach a trailing return and its trend bucket, computed per symbol."""
    out = panel.sort_values(["symbol", "date"], kind="stable").copy()
    out["trailing_return"] = out.groupby("symbol", sort=False)["close"].transform(
        lambda s: s / s.shift(window) - 1.0
    )
    out["trend_bucket"] = pd.cut(
        out["trailing_return"], TREND_EDGES, labels=TREND_LABELS
    )
    return out


def trend_profile(panel: pd.DataFrame) -> pd.DataFrame:
    """Signal mix and true future-hold rates by trailing trend.

    The two must be read together. A model that issues 60% sells into strong
    uptrends looks broken until you notice that true highs really are 4x more
    common there than true lows — the tilt is a property of turning points, not
    a defect. What *would* be a defect is a tilt materially steeper than the
    ground truth's.
    """
    frame = add_trend_context(panel)
    grouped = frame.groupby("trend_bucket", observed=True)

    table = pd.DataFrame({"bars": grouped.size()})
    if "action" in frame.columns:
        table["buy_share"] = grouped["action"].apply(
            lambda s: float(s.isin(["BUY", "STRONG_BUY"]).mean())
        )
        table["sell_share"] = grouped["action"].apply(
            lambda s: float(s.isin(["SELL", "STRONG_SELL"]).mean())
        )
    for column, name in (
        ("y_pivot_low_hold", "true_low_hold_rate"),
        ("y_pivot_high_hold", "true_high_hold_rate"),
    ):
        if column in frame.columns:
            table[name] = grouped[column].mean()
    if {"true_low_hold_rate", "true_high_hold_rate"} <= set(table.columns):
        # How lopsided the ground truth itself is, as a yardstick for the model.
        table["true_low_high_hold_ratio"] = (
            table["true_low_hold_rate"] / table["true_high_hold_rate"]
        )
    if "fwd_ret_cs" in frame.columns:
        table["fwd_excess"] = grouped["fwd_ret_cs"].mean()
    return table


def within_trend_discrimination(
    panel: pd.DataFrame, side: str = "low", top_frac: float = 0.10
) -> pd.DataFrame:
    """Detection lift measured *inside* each trend bucket.

    Holding the trend constant tests whether the head predicts future survival
    rather than merely repeating the direction of the trailing move.
    """
    prob_col = f"p_pivot_{side}"
    label_col = f"y_pivot_{side}_hold"
    needed = {prob_col, label_col, "close"}
    if not needed <= set(panel.columns):
        return pd.DataFrame()

    frame = add_trend_context(panel).dropna(subset=[prob_col, label_col, "trend_bucket"])
    rows = []
    for bucket, block in frame.groupby("trend_bucket", observed=True):
        if len(block) < 200:
            continue
        base = float(block[label_col].mean())
        cut = block[prob_col].quantile(1.0 - top_frac)
        selected = block[block[prob_col] >= cut]
        precision = float(selected[label_col].mean()) if len(selected) else np.nan
        rows.append(
            {
                "trend_bucket": bucket,
                "bars": int(len(block)),
                "base_rate": base,
                "precision": precision,
                "lift": precision / base if base else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("trend_bucket")


def trend_correlation(panel: pd.DataFrame, window: int = 20) -> dict[str, float]:
    """Rank correlation of each model output against the trailing trend.

    Compared against the same correlation for the *labels*, this separates
    "the model fades trend" from "turning points are distributed that way".
    """
    frame = add_trend_context(panel, window).dropna(subset=["trailing_return"])
    out: dict[str, float] = {}
    for column in (
        "score",
        "p_pivot_low",
        "p_pivot_high",
        "y_pivot_low_hold",
        "y_pivot_high_hold",
        "fwd_ret_cs",
    ):
        if column in frame.columns:
            series = frame[[column, "trailing_return"]].dropna()
            if len(series) > 1000:
                out[column] = float(
                    series[column].corr(series["trailing_return"], method="spearman")
                )
    for side in ("low", "high"):
        model, truth = f"p_pivot_{side}", f"y_pivot_{side}_hold"
        if model in out and truth in out and out[truth]:
            # >1 means the model tilts harder than reality warrants.
            out[f"{side}_overfade"] = out[model] / out[truth]
    return out


def universe_report(panel: pd.DataFrame, membership: pd.DataFrame | None = None) -> dict[str, object]:
    """What is actually in the panel: indices, non-members, coverage."""
    symbols = sorted(panel["symbol"].unique())
    indices = [s for s in symbols if s.startswith("^")]
    report: dict[str, object] = {
        "symbols": len(symbols),
        "bars": int(len(panel)),
        "index_instruments": indices,
        "start": str(pd.Timestamp(panel["date"].min()).date()),
        "end": str(pd.Timestamp(panel["date"].max()).date()),
    }
    if membership is not None and not membership.empty:
        known = set(membership["symbol"])
        report["not_in_membership"] = sorted(set(symbols) - known)
        from predixlab_signals.data.csv_data import membership_mask

        outside = 0
        for symbol, block in panel.groupby("symbol", sort=False):
            if symbol not in known:
                outside += len(block)
                continue
            outside += int((~membership_mask(symbol, block["date"], membership)).sum())
        report["bars_outside_index"] = outside
        report["share_outside_index"] = outside / max(len(panel), 1)
    return report


def run_diagnostics(panel: pd.DataFrame) -> dict[str, object]:
    """Every view at once, for a single before/after comparison."""
    return {
        "trend_profile": trend_profile(panel),
        "buy_discrimination": within_trend_discrimination(panel, "low"),
        "sell_discrimination": within_trend_discrimination(panel, "high"),
        "trend_correlation": trend_correlation(panel),
    }
