from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.alpha_rank_evaluation import (
    evaluate_alpha_rank_predictions,
)


def _predictions() -> pd.DataFrame:
    rows = []
    for fold, start in (
        (1, "2024-01-02"),
        (2, "2025-01-02"),
    ):
        for date in pd.bdate_range(start, periods=5):
            for rank in range(120):
                realized_rank = (120 - rank) / 120.0
                rows.append(
                    {
                        "fold": fold,
                        "date": date,
                        "symbol": f"S{rank:03d}",
                        "alpha_fwd_return": realized_rank / 10.0 - 0.05,
                        "alpha_excess_return": realized_rank / 10.0 - 0.05,
                        "alpha_return_rank": realized_rank,
                        "alpha_top20": float(realized_rank >= 0.80),
                        "return_hat": realized_rank / 10.0 - 0.05,
                        "rank_hat": realized_rank,
                        "p_top": realized_rank,
                        "alpha_score": realized_rank,
                        "alpha_score_36m": realized_rank,
                        "alpha_score_60m": realized_rank,
                        "alpha_score_96m": realized_rank,
                    }
                )
    return pd.DataFrame(rows)


def test_alpha_rank_fold_evaluation_detects_perfect_ranking():
    report = evaluate_alpha_rank_predictions(_predictions())
    assert len(report["folds"]) == 2
    for fold in report["folds"]:
        assert fold["rank_ic"]["alpha_score"]["mean"] == 1.0
        top20 = next(
            row for row in fold["topk"] if row["top_k"] == 20
        )
        assert top20["top20_precision"] == 1.0
        assert top20["spread_vs_universe"] > 0
        assert top20["spread_vs_bottom"] > 0
        assert np.isfinite(fold["universe"]["top20_auc"])
