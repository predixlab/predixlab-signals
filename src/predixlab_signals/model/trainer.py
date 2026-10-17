"""Walk-forward training orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from predixlab_signals.config import PATHS, Paths
from predixlab_signals.features.core import feature_columns
from predixlab_signals.model.pivot_model import DEFAULT_PARAMS, PivotModel, classify_action
from predixlab_signals.model.walkforward import Fold, assert_no_leakage, make_folds, slice_fold


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: list[Fold]
    models: list[PivotModel]
    elapsed: float

    def head_estimators(self) -> dict[str, int]:
        """Median per-head tree count chosen by early stopping across folds.

        A single validation window can pick a wildly atypical count - the
        direction head has stopped anywhere between a handful and hundreds of
        trees depending on the regime it was validated against. Taking the
        median across folds is far more stable than trusting whichever window
        happened to be last.
        """
        counts: dict[str, list[int]] = {}
        for model in self.models:
            for head, value in (model.trained_on.get("best_iterations") or {}).items():
                if value:
                    counts.setdefault(head, []).append(int(value))
        return {head: int(np.median(values)) for head, values in counts.items() if values}

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": f.index,
                    "train_start": f.train_start.date(),
                    "train_end": f.train_end.date(),
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    "train_rows": m.trained_on.get("rows"),
                }
                for f, m in zip(self.folds, self.models)
            ]
        )


CARRY_COLUMNS = [
    "symbol",
    "date",
    "close",
    "price",
    "open",
    "high",
    "low",
    "volume",
    "open_next",
    "label_scale",
    "atr_pct_label",
    "atr_pct",
    "fwd_mfe",
    "fwd_mae",
    "fwd_ret",
    "pivot",
    "y_mfe",
    "y_mae",
    "y_edge",
    "y_swing_pos",
    "y_amplitude",
    "y_barrier",
    "y_swing_pos_cs",
    "y_edge_cs",
    "fwd_ret_cs",
    "y_barrier_cs",
    "y_pivot_low_hold",
    "y_pivot_high_hold",
    "y_pivot_low_return",
    "y_pivot_high_return",
]


def run_walk_forward(
    data: pd.DataFrame,
    horizon: int = 10,
    n_folds: int = 5,
    test_months: int = 12,
    valid_months: int = 6,
    embargo_days: int = 5,
    expanding: bool = True,
    params: dict | None = None,
    features: list[str] | None = None,
    market_neutral: bool = True,
    verbose: bool = True,
) -> WalkForwardResult:
    """Fit one model per fold and collect strictly out-of-sample predictions."""
    feats = features or feature_columns(data)
    folds = make_folds(
        data["date"],
        horizon=horizon,
        n_folds=n_folds,
        test_months=test_months,
        valid_months=valid_months,
        embargo_days=embargo_days,
        expanding=expanding,
    )
    if not folds:
        raise ValueError("Not enough history to build walk-forward folds")

    started = time.time()
    preds: list[pd.DataFrame] = []
    models: list[PivotModel] = []

    for fold in folds:
        assert_no_leakage(fold, horizon)
        train, valid, test = slice_fold(data, fold)
        if train.empty or valid.empty or test.empty:
            continue

        model = PivotModel(
            features=feats,
            horizon=horizon,
            params=dict(params or DEFAULT_PARAMS),
            market_neutral=market_neutral,
        )
        model.fit(train, features=feats, valid=valid)
        model.calibrate(model.predict(valid)["score"].to_numpy())

        signal = model.predict(test)
        block = test[[c for c in CARRY_COLUMNS if c in test.columns]].reset_index(drop=True)
        block = pd.concat([block, signal.reset_index(drop=True)], axis=1)
        block["fold"] = fold.index
        block["action"] = classify_action(
            block["score"].to_numpy(),
            block["exp_amplitude_pct"].to_numpy(),
            model.thresholds,
            groups=block["date"].to_numpy(),
        )
        preds.append(block)
        models.append(model)

        if verbose:
            cs_ic, ts_ic = _quick_ic(block)
            print(
                f"  {fold.describe()} | train={len(train):,} test={len(test):,} "
                f"| IC_cs={cs_ic:+.4f} IC_ts={ts_ic:+.4f}",
                flush=True,
            )

    if not preds:
        raise ValueError("No fold produced predictions")

    return WalkForwardResult(
        predictions=pd.concat(preds, ignore_index=True),
        folds=folds[: len(models)],
        models=models,
        elapsed=time.time() - started,
    )


def _quick_ic(block: pd.DataFrame) -> tuple[float, float]:
    """(cross-sectional rank IC, within-symbol rank IC) against forward return."""
    sub = block[["date", "symbol", "score", "fwd_ret"]].dropna()
    if sub.empty:
        return float("nan"), float("nan")
    cs = (
        sub.groupby("date")
        .apply(
            lambda g: g["score"].corr(g["fwd_ret"], method="spearman") if len(g) > 5 else np.nan,
            include_groups=False,
        )
        .mean()
    )
    ts = (
        sub.groupby("symbol")
        .apply(
            lambda g: g["score"].corr(g["fwd_ret"], method="spearman") if len(g) > 60 else np.nan,
            include_groups=False,
        )
        .mean()
    )
    return float(cs), float(ts)


def train_final_model(
    data: pd.DataFrame,
    horizon: int = 10,
    valid_months: int = 6,
    embargo_days: int = 5,
    params: dict | None = None,
    features: list[str] | None = None,
    market_neutral: bool = True,
    head_estimators: dict[str, int] | None = None,
) -> PivotModel:
    """Fit the production model on all history, holding out a recent tail.

    The tail is used only to calibrate action thresholds - never to select
    hyper-parameters - so the deployed model still sees the most recent regime
    in its training data while the cut-points stay honest.

    Three fits, and the ordering matters:

    1. If per-head tree counts were not supplied, a probe fit with early
       stopping discovers them.
    2. A calibration fit uses **exactly** those counts on the training block,
       and its scores on the held-out tail set the action thresholds. Calibrating
       from a differently-sized model would mis-scale every cut-point: score
       spread shrinks with tree count, so thresholds taken from a deeper model
       classify almost everything as HOLD.
    3. The production fit repeats the same configuration over all history.
    """
    feats = features or feature_columns(data)
    last = pd.Timestamp(data["date"].max())
    purge = pd.Timedelta(days=int(horizon * 1.6) + embargo_days)
    valid_start = last - pd.DateOffset(months=valid_months)
    train_end = valid_start - purge

    train = data[data["date"] <= train_end]
    valid = data[data["date"] >= valid_start]
    if train.empty or valid.empty:
        raise ValueError("Not enough history to fit the final model")

    def _build(counts: dict[str, int] | None) -> PivotModel:
        return PivotModel(
            features=feats,
            horizon=horizon,
            params=dict(params or DEFAULT_PARAMS),
            market_neutral=market_neutral,
            head_estimators=dict(counts or {}),
        )

    chosen = {k: int(v) for k, v in (head_estimators or {}).items() if v}
    if not chosen:
        probe = _build(None)
        probe.fit(train, features=feats, valid=valid)
        chosen = {
            k: int(v)
            for k, v in (probe.trained_on.get("best_iterations") or {}).items()
            if v
        }

    calibrator = _build(chosen)
    calibrator.fit(train, features=feats)
    calibrator.calibrate(calibrator.predict(valid)["score"].to_numpy())

    full = _build(chosen)
    full.fit(data, features=feats)
    full.thresholds = calibrator.thresholds
    full.trained_on["head_estimators"] = chosen
    full.trained_on["calibrated_on"] = {
        "start": str(pd.Timestamp(valid["date"].min()).date()),
        "end": str(pd.Timestamp(valid["date"].max()).date()),
    }
    return full


def model_path(
    market: str, horizon: int, tag: str = "default", paths: Paths = PATHS
) -> Path:
    """The production model: what the daily job scores with.

    Installed by copying a validated candidate here; never written by ``train``
    and never by the scheduled job.
    """
    return paths.model_dir(market) / f"pivot_h{horizon}_{tag}.pkl"


def staged_model_path(
    market: str, horizon: int, tag: str = "default", paths: Paths = PATHS
) -> Path:
    """The candidate model: what ``train`` just fitted, pending validation."""
    return paths.staging_model_dir(market) / f"pivot_h{horizon}_{tag}.pkl"


def save_walk_forward(result: WalkForwardResult, market: str, horizon: int, tag: str = "default") -> Path:
    out = PATHS.report_dir(market)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"walkforward_h{horizon}_{tag}.parquet"
    result.predictions.to_parquet(path, index=False)
    (out / f"walkforward_h{horizon}_{tag}.folds.json").write_text(
        json.dumps(result.fold_table().astype(str).to_dict(orient="records"), indent=2)
    )
    return path
