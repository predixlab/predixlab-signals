"""Model heads, walk-forward splitting and training orchestration."""

from predixlab_signals.model.pivot_model import (
    ActionThresholds,
    PivotModel,
    classify_action,
    pivot_score,
)
from predixlab_signals.model.alpha_rank_model import (
    AlphaRankEnsemble,
    AlphaRankModel,
    prepare_alpha_rank_features,
    prepare_alpha_rank_training_data,
)
from predixlab_signals.model.alpha_rank_trainer import (
    AlphaRankWalkForwardResult,
    alpha_rank_walkforward_path,
    alpha_rank_model_path,
    run_alpha_rank_walk_forward,
    save_alpha_rank_walk_forward,
    staged_alpha_rank_model_path,
    train_final_alpha_rank_model,
)
from predixlab_signals.model.trainer import (
    WalkForwardResult,
    model_path,
    staged_model_path,
    run_walk_forward,
    save_walk_forward,
    train_final_model,
)
from predixlab_signals.model.walkforward import Fold, make_folds, slice_fold

__all__ = [
    "PivotModel",
    "ActionThresholds",
    "pivot_score",
    "classify_action",
    "run_walk_forward",
    "train_final_model",
    "save_walk_forward",
    "model_path",
    "staged_model_path",
    "WalkForwardResult",
    "Fold",
    "make_folds",
    "slice_fold",
    "AlphaRankModel",
    "AlphaRankEnsemble",
    "AlphaRankWalkForwardResult",
    "prepare_alpha_rank_features",
    "prepare_alpha_rank_training_data",
    "run_alpha_rank_walk_forward",
    "save_alpha_rank_walk_forward",
    "train_final_alpha_rank_model",
    "alpha_rank_walkforward_path",
    "alpha_rank_model_path",
    "staged_alpha_rank_model_path",
]
