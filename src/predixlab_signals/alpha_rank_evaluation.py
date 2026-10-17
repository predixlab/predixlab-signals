"""Per-fold evaluation for AlphaRank walk-forward predictions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from predixlab_signals.config import PATHS
from predixlab_signals.model import alpha_rank_walkforward_path

TOP_K_VALUES = (20, 50, 100)
SCORE_COLUMNS = (
    "alpha_score",
    "alpha_score_36m",
    "alpha_score_60m",
    "alpha_score_96m",
)
OPTIONAL_SCORE_COLUMNS = ("alpha_consensus",)


@dataclass(frozen=True)
class AlphaRankEvaluationResult:
    json_path: Path
    topk_path: Path
    html_path: Path
    report: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _rank_ic(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    daily = (
        frame[["date", score_column, "alpha_fwd_return"]]
        .dropna()
        .groupby("date", sort=True)
        .apply(
            lambda block: block[score_column].corr(
                block["alpha_fwd_return"],
                method="spearman",
            ),
            include_groups=False,
        )
        .dropna()
    )
    standard_deviation = float(daily.std())
    return {
        "dates": int(len(daily)),
        "mean": float(daily.mean()),
        "median": float(daily.median()),
        "std": standard_deviation,
        "icir": (
            float(daily.mean() / standard_deviation * np.sqrt(252))
            if standard_deviation > 0
            else float("nan")
        ),
        "positive_rate": float((daily > 0).mean()),
    }


def _topk_metrics(
    frame: pd.DataFrame,
    top_k: int,
    comparison_frame: pd.DataFrame | None = None,
) -> dict[str, float | int]:
    ordered = frame.sort_values(
        ["date", "alpha_score", "symbol"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()
    selected = ordered.groupby("date", sort=False).head(top_k)
    bottom = ordered.groupby("date", sort=False).tail(top_k)

    selected_daily = selected.groupby("date", sort=True)[
        "alpha_fwd_return"
    ].mean()
    bottom_daily = bottom.groupby("date", sort=True)[
        "alpha_fwd_return"
    ].mean()
    universe = comparison_frame if comparison_frame is not None else frame
    universe_daily = universe.groupby("date", sort=True)[
        "alpha_fwd_return"
    ].mean()
    spread_universe = selected_daily - universe_daily
    spread_bottom = selected_daily - bottom_daily
    base_rate = float(universe["alpha_top20"].mean())
    precision = float(selected["alpha_top20"].mean())
    return {
        "top_k": int(top_k),
        "rows": int(len(selected)),
        "dates": int(selected["date"].nunique()),
        "mean_forward_return": float(selected["alpha_fwd_return"].mean()),
        "median_forward_return": float(selected["alpha_fwd_return"].median()),
        "mean_excess_return": float(selected["alpha_excess_return"].mean()),
        "win_rate": float((selected["alpha_fwd_return"] > 0).mean()),
        "beat_median_rate": float(
            (selected["alpha_return_rank"] > 0.50).mean()
        ),
        "top20_precision": precision,
        "top20_base_rate": base_rate,
        "top20_lift": precision / base_rate if base_rate > 0 else float("nan"),
        "spread_vs_universe": float(spread_universe.mean()),
        "spread_vs_universe_positive_rate": float(
            (spread_universe > 0).mean()
        ),
        "spread_vs_bottom": float(spread_bottom.mean()),
        "spread_vs_bottom_positive_rate": float((spread_bottom > 0).mean()),
    }


def _filtered_topk_metrics(
    frame: pd.DataFrame,
    top_k: int,
) -> dict[str, float | int]:
    consensus_rank = frame.groupby("date", sort=False)[
        "alpha_consensus"
    ].rank(pct=True)
    eligible = frame[
        (consensus_rank >= 0.55)
        & (frame["ret_20"] > 0.0)
        & (frame["px_sma60"] > 0.0)
    ]
    metrics = _topk_metrics(eligible, top_k, comparison_frame=frame)
    metrics["average_names"] = float(
        len(
            eligible.sort_values(
                ["date", "alpha_score"],
                ascending=[True, False],
            )
            .groupby("date")
            .head(top_k)
        )
        / max(eligible["date"].nunique(), 1)
    )
    return metrics


def _head_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    top = frame[["p_top", "alpha_top20"]].dropna()
    regression = frame[
        ["return_hat", "alpha_excess_return", "rank_hat", "alpha_return_rank"]
    ].dropna()
    return {
        "top20_auc": (
            float(roc_auc_score(top["alpha_top20"], top["p_top"]))
            if top["alpha_top20"].nunique() > 1
            else float("nan")
        ),
        "top20_brier": float(
            brier_score_loss(top["alpha_top20"], top["p_top"])
        ),
        "return_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        regression["return_hat"]
                        - regression["alpha_excess_return"]
                    )
                    ** 2
                )
            )
        ),
        "rank_rmse": float(
            np.sqrt(
                np.mean(
                    (
                        regression["rank_hat"]
                        - regression["alpha_return_rank"]
                    )
                    ** 2
                )
            )
        ),
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "symbols": int(frame["symbol"].nunique()),
    }


def evaluate_alpha_rank_predictions(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    required = {
        "date",
        "symbol",
        "fold",
        "alpha_fwd_return",
        "alpha_excess_return",
        "alpha_return_rank",
        "alpha_top20",
        "return_hat",
        "rank_hat",
        "p_top",
        *SCORE_COLUMNS,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"AlphaRank evaluation input missing columns: {sorted(missing)}"
        )
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    folds = []
    for fold, block in frame.groupby("fold", sort=True):
        block = block.dropna(subset=["alpha_fwd_return"]).copy()
        folds.append(
            {
                "fold": int(fold),
                "period": {
                    "start": str(block["date"].min().date()),
                    "end": str(block["date"].max().date()),
                },
                "universe": _head_metrics(block),
                "rank_ic": {
                    column: _rank_ic(block, column)
                    for column in (
                        *SCORE_COLUMNS,
                        *(
                            column
                            for column in OPTIONAL_SCORE_COLUMNS
                            if column in block.columns
                        ),
                    )
                },
                "topk": [
                    _topk_metrics(block, top_k)
                    for top_k in TOP_K_VALUES
                ],
                "filtered_topk": (
                    [
                        _filtered_topk_metrics(block, top_k)
                        for top_k in TOP_K_VALUES
                    ]
                    if {
                        "alpha_consensus",
                        "ret_20",
                        "px_sma60",
                    }.issubset(block.columns)
                    else []
                ),
            }
        )
    return {
        "definition": {
            "target": "next-close to future-close return over the configured horizon",
            "rank_ic": "daily Spearman correlation against realized forward return",
            "topk": "highest alpha_score names per date before strategy filters",
            "filtered_topk": "formal consensus, momentum, and trend entry filters applied before alpha_score ranking",
        },
        "folds": folds,
    }


def _render_html(report: dict[str, Any]) -> str:
    rank_rows = []
    top_rows = []
    filtered_rows = []
    head_rows = []
    for fold in report["folds"]:
        for score, metrics in fold["rank_ic"].items():
            rank_rows.append(
                f"<tr><td>{fold['fold']}</td><td>{score}</td>"
                f"<td>{metrics['mean']:+.4f}</td>"
                f"<td>{metrics['median']:+.4f}</td>"
                f"<td>{metrics['icir']:+.2f}</td>"
                f"<td>{metrics['positive_rate']:.1%}</td></tr>"
            )
        for metrics in fold["topk"]:
            top_rows.append(
                f"<tr><td>{fold['fold']}</td><td>{metrics['top_k']}</td>"
                f"<td>{metrics['mean_forward_return']:+.2%}</td>"
                f"<td>{metrics['mean_excess_return']:+.2%}</td>"
                f"<td>{metrics['win_rate']:.1%}</td>"
                f"<td>{metrics['top20_precision']:.1%}</td>"
                f"<td>{metrics['top20_lift']:.2f}x</td>"
                f"<td>{metrics['spread_vs_universe']:+.2%}</td>"
                f"<td>{metrics['spread_vs_bottom']:+.2%}</td></tr>"
            )
        for metrics in fold["filtered_topk"]:
            filtered_rows.append(
                f"<tr><td>{fold['fold']}</td><td>{metrics['top_k']}</td>"
                f"<td>{metrics['average_names']:.1f}</td>"
                f"<td>{metrics['mean_forward_return']:+.2%}</td>"
                f"<td>{metrics['mean_excess_return']:+.2%}</td>"
                f"<td>{metrics['win_rate']:.1%}</td>"
                f"<td>{metrics['top20_precision']:.1%}</td>"
                f"<td>{metrics['spread_vs_universe']:+.2%}</td></tr>"
            )
        universe = fold["universe"]
        head_rows.append(
            f"<tr><td>{fold['fold']}</td>"
            f"<td>{fold['period']['start']}..{fold['period']['end']}</td>"
            f"<td>{universe['rows']:,}</td><td>{universe['symbols']}</td>"
            f"<td>{universe['top20_auc']:.4f}</td>"
            f"<td>{universe['top20_brier']:.4f}</td>"
            f"<td>{universe['return_rmse']:.4f}</td>"
            f"<td>{universe['rank_rmse']:.4f}</td></tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AlphaRank fold evaluation</title>
<style>body{{font:14px system-ui;margin:32px;color:#172126}}table{{width:100%;border-collapse:collapse;margin:12px 0 28px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#eef2ef}}</style>
</head><body><h1>AlphaRank per-fold evaluation</h1>
<h2>Head diagnostics</h2><table><thead><tr><th>Fold</th><th>Period</th><th>Rows</th><th>Symbols</th><th>Top20 AUC</th><th>Brier</th><th>Return RMSE</th><th>Rank RMSE</th></tr></thead><tbody>{''.join(head_rows)}</tbody></table>
<h2>Rank IC</h2><table><thead><tr><th>Fold</th><th>Score</th><th>Mean IC</th><th>Median IC</th><th>ICIR</th><th>Positive days</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table>
<h2>Top-K diagnostics</h2><table><thead><tr><th>Fold</th><th>Top K</th><th>Forward return</th><th>Excess return</th><th>Win rate</th><th>Top20 precision</th><th>Lift</th><th>Spread vs universe</th><th>Spread vs bottom</th></tr></thead><tbody>{''.join(top_rows)}</tbody></table>
<h2>Top-K after formal entry filters</h2><table><thead><tr><th>Fold</th><th>Top K</th><th>Average names</th><th>Forward return</th><th>Excess return</th><th>Win rate</th><th>Top20 precision</th><th>Spread vs universe</th></tr></thead><tbody>{''.join(filtered_rows)}</tbody></table>
</body></html>"""


def build_alpha_rank_evaluation(
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
) -> AlphaRankEvaluationResult:
    predictions_path = alpha_rank_walkforward_path(market, horizon, tag)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"AlphaRank predictions not found: {predictions_path}"
        )
    report = evaluate_alpha_rank_predictions(
        pd.read_parquet(predictions_path)
    )
    root = PATHS.report_dir(market)
    stem = f"alpha_rank_fold_evaluation_h{horizon}_{tag}"
    json_path = root / f"{stem}.json"
    topk_path = root / f"{stem}_topk.csv"
    html_path = root / f"{stem}.html"
    json_path.write_text(json.dumps(_json_ready(report), indent=2))
    pd.DataFrame(
        [
            {"fold": fold["fold"], **metrics}
            for fold in report["folds"]
            for metrics in fold["topk"]
        ]
    ).to_csv(topk_path, index=False)
    html_path.write_text(_render_html(report), encoding="utf-8")
    return AlphaRankEvaluationResult(
        json_path=json_path,
        topk_path=topk_path,
        html_path=html_path,
        report=report,
    )
