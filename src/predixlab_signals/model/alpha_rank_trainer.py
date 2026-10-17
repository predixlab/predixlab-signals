"""Walk-forward training for AlphaRankModel."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from predixlab_signals.config import PATHS, Paths
from predixlab_signals.model.alpha_rank_model import (
    AlphaRankEnsemble,
    prepare_alpha_rank_training_data,
)
from predixlab_signals.model.walkforward import Fold, make_folds, slice_fold


@dataclass
class AlphaRankWalkForwardResult:
    predictions: pd.DataFrame
    folds: list[Fold]
    models: list[AlphaRankEnsemble]
    elapsed: float

    def head_estimators(self) -> dict[str, dict[str, int]]:
        values: dict[str, dict[str, list[int]]] = {}
        for ensemble in self.models:
            for window, metadata in ensemble.trained_on["windows"].items():
                for head, count in metadata["best_iterations"].items():
                    values.setdefault(window, {}).setdefault(head, []).append(
                        int(count)
                    )
        return {
            window: {
                head: int(np.median(counts))
                for head, counts in heads.items()
                if counts
            }
            for window, heads in values.items()
        }


def alpha_rank_walkforward_path(
    market: str,
    horizon: int,
    tag: str,
) -> Path:
    return (
        PATHS.report_dir(market)
        / f"alpha_rank_walkforward_h{horizon}_{tag}.parquet"
    )


def staged_alpha_rank_model_path(
    market: str,
    horizon: int,
    tag: str,
    paths: Paths = PATHS,
) -> Path:
    return (
        paths.staging_model_dir(market)
        / f"alpha_rank_h{horizon}_{tag}.pkl"
    )


def alpha_rank_model_path(
    market: str,
    horizon: int,
    tag: str = "default",
    paths: Paths = PATHS,
) -> Path:
    """The production AlphaRank artifact installed after independent validation."""
    return paths.alpha_rank_model_path(market, horizon, tag)


def run_alpha_rank_walk_forward(
    data: pd.DataFrame,
    horizon: int = 10,
    n_folds: int = 3,
    test_months: int = 12,
    valid_months: int = 6,
) -> AlphaRankWalkForwardResult:
    prepared = prepare_alpha_rank_training_data(data, horizon)
    folds = make_folds(
        prepared["date"],
        horizon=horizon,
        n_folds=n_folds,
        test_months=test_months,
        valid_months=valid_months,
        embargo_days=5,
        expanding=True,
    )
    started = time.time()
    predictions = []
    models = []
    carry = [
        "date",
        "symbol",
        "open_next",
        "close",
        "ret_20",
        "px_sma60",
        "amihud",
        "atr_pct",
        "mkt_ret_20",
        "mkt_ret_1",
        "mkt_ret_5",
        "mkt_px_sma50",
        "mkt_vol_ratio",
        "alpha_fwd_return",
        "alpha_excess_return",
        "alpha_return_rank",
        "alpha_top20",
    ]
    for fold in folds:
        train, valid, test = slice_fold(prepared, fold)
        model = AlphaRankEnsemble(horizon=horizon).fit(train, valid)
        scored = pd.concat(
            [
                test[carry].reset_index(drop=True),
                model.predict(test).reset_index(drop=True),
            ],
            axis=1,
        )
        scored["fold"] = fold.index
        predictions.append(scored)
        models.append(model)
        ic = (
            scored.groupby("date")
            .apply(
                lambda block: block["alpha_score"].corr(
                    block["alpha_fwd_return"],
                    method="spearman",
                ),
                include_groups=False,
            )
            .mean()
        )
        print(
            f"  {fold.describe()} | train={len(train):,} test={len(test):,} "
            f"| IC_cs={ic:+.4f}",
            flush=True,
        )
    return AlphaRankWalkForwardResult(
        predictions=pd.concat(predictions, ignore_index=True),
        folds=folds,
        models=models,
        elapsed=time.time() - started,
    )


def save_alpha_rank_walk_forward(
    result: AlphaRankWalkForwardResult,
    market: str,
    horizon: int,
    tag: str,
) -> Path:
    path = alpha_rank_walkforward_path(market, horizon, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(path, index=False)
    rows = [
        {
            "fold": fold.index,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "valid_start": str(fold.valid_start.date()),
            "valid_end": str(fold.valid_end.date()),
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
        }
        for fold in result.folds
    ]
    path.with_suffix(".folds.json").write_text(json.dumps(rows, indent=2))
    return path


def train_final_alpha_rank_model(
    data: pd.DataFrame,
    horizon: int,
    head_estimators: dict[str, dict[str, int]],
) -> AlphaRankEnsemble:
    prepared = prepare_alpha_rank_training_data(data, horizon)
    model = AlphaRankEnsemble(
        horizon=horizon,
        head_estimators=head_estimators,
    )
    return model.fit(prepared)
