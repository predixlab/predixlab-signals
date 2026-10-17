"""LightGBM model for ranking stocks by forward relative return."""

from __future__ import annotations

import inspect
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from predixlab_signals.model.artifacts import (
    BUNDLE_MAGIC,
    atomic_write_bytes,
    atomic_write_json,
    build_bundle,
    metadata_path,
    read_verified_bytes,
)

ALPHA_RANK_BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ret_120",
    "ret_5_z",
    "ret_20_z",
    "mom_accel",
    "px_sma5",
    "px_sma10",
    "px_sma20",
    "px_sma60",
    "px_sma120",
    "sma5_sma20",
    "sma20_sma60",
    "sma60_sma120",
    "sma20_slope",
    "macd",
    "macd_hist",
    "macd_hist_chg",
    "dpos_20",
    "dpos_60",
    "dpos_120",
    "dpos_20_chg",
    "pctile_60",
    "pctile_250",
    "rsi_6",
    "rsi_14",
    "rsi_14_chg",
    "bb_pctb",
    "bb_width",
    "z_20",
    "z_60",
    "adx",
    "di_spread",
    "atr_pct",
    "atr_ratio",
    "vol_5",
    "vol_20",
    "vol_ratio_5_60",
    "vol_ratio_20_60",
    "downside_vol_20",
    "skew_60",
    "kurt_60",
    "vol_rel",
    "vol_rel_5",
    "vol_z",
    "vol_trend",
    "obv_slope",
    "cmf_20",
    "mfi_14",
    "amihud",
    "close_loc",
    "gap_atr",
    "streak",
    "up_frac_10",
    "up_frac_20",
    "dd_from_high_250",
    "rally_from_low_250",
    "beta_20",
    "corr_20",
    "rs_20",
    "alpha_20",
    "beta_60",
    "corr_60",
    "rs_60",
    "alpha_60",
]

DEFAULT_ALPHA_RANK_PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_samples": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 8.0,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 23,
}

TARGET_COLUMNS = (
    "alpha_fwd_return",
    "alpha_excess_return",
    "alpha_return_rank",
    "alpha_top20",
)

ROLLING_WINDOWS_MONTHS = (36, 60, 96)
MIN_ALPHA_RANK_CROSS_SECTION = 100
CONSENSUS_SCORE_WEIGHTS = (
    ("return_hat", 0.25),
    ("rank_hat", 0.55),
    ("p_top", 0.20),
)
FINAL_ALPHA_SCORE_WEIGHTS = CONSENSUS_SCORE_WEIGHTS

ScoreWeights = tuple[tuple[str, float], ...]


def _score_weights_metadata() -> dict[str, dict[str, float]]:
    return {
        "consensus": dict(CONSENSUS_SCORE_WEIGHTS),
        "final": dict(FINAL_ALPHA_SCORE_WEIGHTS),
    }


def alpha_rank_feature_columns(
    base_features: list[str] | None = None,
) -> list[str]:
    return [
        f"rank_{column}"
        for column in (base_features or ALPHA_RANK_BASE_FEATURES)
    ]


def alpha_rank_score(
    predictions: pd.DataFrame,
    groups: pd.Series | np.ndarray,
    weights: ScoreWeights = CONSENSUS_SCORE_WEIGHTS,
) -> pd.Series:
    """Combine cross-sectional head ranks into the portfolio score."""
    score = pd.Series(0.0, index=predictions.index)
    for column, weight in weights:
        score += weight * predictions[column].groupby(
            groups,
            sort=False,
        ).rank(pct=True)
    return score


def prepare_alpha_rank_features(
    data: pd.DataFrame,
    base_features: list[str] | None = None,
) -> pd.DataFrame:
    """Convert same-day model inputs to cross-sectional percentiles."""
    columns = base_features or ALPHA_RANK_BASE_FEATURES
    missing = set(columns) - set(data.columns)
    if missing:
        raise KeyError(f"AlphaRank input missing features: {sorted(missing)}")
    ranked = (
        data.groupby("date", sort=False)[columns]
        .rank(pct=True)
        .astype("float32")
        .rename(columns={column: f"rank_{column}" for column in columns})
    )
    return pd.concat([data, ranked], axis=1)


def prepare_alpha_rank_training_data(
    data: pd.DataFrame,
    horizon: int = 10,
    base_features: list[str] | None = None,
    min_cross_section: int = MIN_ALPHA_RANK_CROSS_SECTION,
) -> pd.DataFrame:
    """Add globally aligned next-close targets and same-day rank features."""
    if horizon <= 0:
        raise ValueError("AlphaRank horizon must be positive")
    if min_cross_section <= 0:
        raise ValueError("AlphaRank min_cross_section must be positive")
    required = {"date", "symbol", "close"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(
            f"AlphaRank training input missing columns: {sorted(missing)}"
        )
    frame = data.sort_values(["symbol", "date"], kind="stable").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame[["date", "symbol"]].isna().any().any():
        raise ValueError("AlphaRank training input contains missing date or symbol")
    duplicates = frame.duplicated(["date", "symbol"], keep=False)
    if duplicates.any():
        examples = (
            frame.loc[duplicates, ["date", "symbol"]]
            .drop_duplicates()
            .head(5)
        )
        raise ValueError(
            "AlphaRank training input contains duplicate date/symbol rows: "
            f"{examples.to_dict('records')}"
        )
    close = pd.to_numeric(frame["close"], errors="coerce")
    invalid_close = ~np.isfinite(close) | (close <= 0.0)
    if invalid_close.any():
        examples = frame.loc[invalid_close, ["date", "symbol", "close"]].head(5)
        raise ValueError(
            "AlphaRank training input contains non-positive or non-finite close "
            f"values: {examples.to_dict('records')}"
        )
    frame["close"] = close
    calendar = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    entry_dates = frame["date"].map(
        pd.Series(calendar, index=calendar).shift(-1)
    )
    exit_dates = frame["date"].map(
        pd.Series(calendar, index=calendar).shift(-(horizon + 1))
    )
    close_lookup = frame.set_index(["symbol", "date"])["close"]
    entry_keys = pd.MultiIndex.from_arrays(
        [frame["symbol"], entry_dates],
        names=["symbol", "date"],
    )
    exit_keys = pd.MultiIndex.from_arrays(
        [frame["symbol"], exit_dates],
        names=["symbol", "date"],
    )
    entry_close = close_lookup.reindex(entry_keys).to_numpy(dtype="float64")
    exit_close = close_lookup.reindex(exit_keys).to_numpy(dtype="float64")
    frame["alpha_fwd_return"] = exit_close / entry_close - 1.0
    by_date = frame.groupby("date", sort=False)
    frame["alpha_excess_return"] = (
        frame["alpha_fwd_return"]
        - by_date["alpha_fwd_return"].transform("median")
    ).clip(-0.30, 0.30)
    frame["alpha_return_rank"] = by_date["alpha_fwd_return"].rank(pct=True)
    frame["alpha_top20"] = (
        frame["alpha_return_rank"] >= 0.80
    ).astype("float32")
    frame.loc[frame["alpha_return_rank"].isna(), "alpha_top20"] = np.nan
    valid_cross_section = by_date["alpha_fwd_return"].transform("count")
    insufficient = valid_cross_section < min_cross_section
    frame.loc[insufficient, list(TARGET_COLUMNS)] = np.nan
    return prepare_alpha_rank_features(frame, base_features)


def _date_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weight rows so every training date contributes equal total mass."""
    counts = frame.groupby("date", sort=False)["date"].transform("size")
    weights = len(frame) / (frame["date"].nunique() * counts)
    return weights.to_numpy(dtype="float64")


@dataclass
class AlphaRankModel:
    """Three-head model aligned with forward alpha ranking."""

    horizon: int = 10
    base_features: list[str] = field(
        default_factory=lambda: list(ALPHA_RANK_BASE_FEATURES)
    )
    params: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_ALPHA_RANK_PARAMS)
    )
    head_estimators: dict[str, int] = field(default_factory=dict)
    return_model: lgb.LGBMRegressor | None = None
    rank_model: lgb.LGBMRegressor | None = None
    top_model: lgb.LGBMClassifier | None = None
    trained_on: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return alpha_rank_feature_columns(self.base_features)

    def _params(self, head: str, **extra: Any) -> dict[str, Any]:
        params = {**self.params, **extra}
        count = self.head_estimators.get(head)
        if count:
            params["n_estimators"] = max(int(count), 1)
        return params

    def _eval_kwargs(
        self,
        valid: pd.DataFrame | None,
        target: str,
        metric: str,
    ) -> dict[str, Any]:
        if valid is None or valid.empty:
            return {}
        block = valid[valid[target].notna()]
        if block.empty:
            return {}
        kwargs: dict[str, Any] = {
            "eval_metric": metric,
            "eval_sample_weight": [_date_equal_weights(block)],
            "callbacks": [
                lgb.early_stopping(75, verbose=False),
                lgb.log_evaluation(0),
            ],
        }
        if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
            kwargs["eval_X"] = block[self.features]
            kwargs["eval_y"] = block[target]
        else:
            kwargs["eval_set"] = [(block[self.features], block[target])]
        return kwargs

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
    ) -> "AlphaRankModel":
        self.return_model = lgb.LGBMRegressor(
            objective="huber",
            alpha=0.9,
            **self._params("return"),
        )
        self.rank_model = lgb.LGBMRegressor(
            objective="regression",
            **self._params("rank"),
        )
        self.top_model = lgb.LGBMClassifier(
            objective="binary",
            **self._params("top"),
        )
        for model, target, metric in (
            (self.return_model, "alpha_excess_return", "l2"),
            (self.rank_model, "alpha_return_rank", "l2"),
            (self.top_model, "alpha_top20", "auc"),
        ):
            block = train[train[target].notna()]
            model.fit(
                block[self.features],
                block[target],
                sample_weight=_date_equal_weights(block),
                **self._eval_kwargs(valid, target, metric),
            )
        labelled = train[train["alpha_fwd_return"].notna()]
        self.trained_on = {
            "rows": int(len(labelled)),
            "symbols": int(labelled["symbol"].nunique()),
            "start": str(pd.Timestamp(labelled["date"].min()).date()),
            "end": str(pd.Timestamp(labelled["date"].max()).date()),
            "dates": int(labelled["date"].nunique()),
            "minimum_labelled_cross_section": int(
                labelled.groupby("date", sort=False).size().min()
            ),
            "date_weighting": "equal",
            "top20_class_weight": None,
            "best_iterations": {
                name: int(value)
                for name, model in (
                    ("return", self.return_model),
                    ("rank", self.rank_model),
                    ("top", self.top_model),
                )
                if (value := getattr(model, "best_iteration_", None))
            },
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (
            self.return_model is None
            or self.rank_model is None
            or self.top_model is None
        ):
            raise RuntimeError("AlphaRankModel is not fitted")
        prepared = (
            frame
            if set(self.features) <= set(frame.columns)
            else prepare_alpha_rank_features(frame, self.base_features)
        )
        x = prepared[self.features]
        out = pd.DataFrame(index=prepared.index)
        out["return_hat"] = self.return_model.predict(x)
        out["rank_hat"] = self.rank_model.predict(x)
        out["p_top"] = self.top_model.predict_proba(x)[:, 1]
        out["alpha_score"] = alpha_rank_score(
            out,
            prepared["date"].to_numpy(),
        )
        return out

    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        frames = []
        for name, model in (
            ("return", self.return_model),
            ("rank", self.rank_model),
            ("top", self.top_model),
        ):
            if model is None:
                continue
            values = pd.Series(
                model.feature_importances_,
                index=self.base_features,
                dtype="float64",
            )
            frames.append((values / values.sum()).rename(name))
        table = pd.concat(frames, axis=1)
        table["mean"] = table.mean(axis=1)
        return table.sort_values("mean", ascending=False).head(top)

    def save(self, path: Path) -> Path:
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "model_type": "alpha_rank",
            "horizon": self.horizon,
            "score_weights": _score_weights_metadata(),
            "base_features": self.base_features,
            "n_features": len(self.features),
            "params": self.params,
            "head_estimators": self.head_estimators,
            "trained_on": self.trained_on,
        }
        bundle, manifest = build_bundle(payload, metadata)
        atomic_write_bytes(path, bundle)
        atomic_write_json(metadata_path(path), manifest)
        return path

    @staticmethod
    def load(
        path: Path,
        *,
        verify_integrity: bool = False,
    ) -> "AlphaRankModel":
        path = Path(path)
        if verify_integrity or path.read_bytes().startswith(BUNDLE_MAGIC):
            payload, _ = read_verified_bytes(path)
            return pickle.loads(payload)
        with path.open("rb") as handle:
            return pickle.load(handle)


@dataclass
class AlphaRankEnsemble:
    """Three AlphaRank models trained on short, medium, and long windows."""

    horizon: int = 10
    windows_months: tuple[int, ...] = ROLLING_WINDOWS_MONTHS
    base_features: list[str] = field(
        default_factory=lambda: list(ALPHA_RANK_BASE_FEATURES)
    )
    params: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_ALPHA_RANK_PARAMS)
    )
    head_estimators: dict[str, dict[str, int]] = field(default_factory=dict)
    models: dict[str, AlphaRankModel] = field(default_factory=dict)
    trained_on: dict[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return alpha_rank_feature_columns(self.base_features)

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
    ) -> "AlphaRankEnsemble":
        labelled = train[train["alpha_fwd_return"].notna()]
        if labelled.empty:
            raise ValueError("AlphaRank training data has no completed targets")
        end = pd.Timestamp(labelled["date"].max())
        models: dict[str, AlphaRankModel] = {}
        windows: dict[str, dict[str, Any]] = {}
        for months in self.windows_months:
            key = str(months)
            start = end - pd.DateOffset(months=months)
            block = train[
                (train["date"] >= start)
                & (train["date"] <= end)
            ]
            if block.empty:
                raise ValueError(f"AlphaRank {months}-month training window is empty")
            model = AlphaRankModel(
                horizon=self.horizon,
                base_features=list(self.base_features),
                params=dict(self.params),
                head_estimators=dict(self.head_estimators.get(key, {})),
            ).fit(block, valid)
            models[key] = model
            windows[key] = dict(model.trained_on)
        self.models = models
        self.trained_on = {
            "rows": int(len(labelled)),
            "symbols": int(labelled["symbol"].nunique()),
            "start": str(pd.Timestamp(labelled["date"].min()).date()),
            "end": str(end.date()),
            "dates": int(labelled["date"].nunique()),
            "windows": windows,
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("AlphaRankEnsemble is not fitted")
        predictions = {
            window: model.predict(frame)
            for window, model in self.models.items()
        }
        out = pd.DataFrame(index=frame.index)
        for column in ("return_hat", "rank_hat", "p_top"):
            out[column] = pd.concat(
                [
                    prediction[column]
                    for prediction in predictions.values()
                ],
                axis=1,
            ).mean(axis=1)
        long_window = str(max(int(window) for window in predictions))
        long_prediction = predictions[long_window]
        groups = frame["date"].to_numpy()
        out["alpha_score"] = alpha_rank_score(
            long_prediction,
            groups,
            FINAL_ALPHA_SCORE_WEIGHTS,
        )
        consensus = []
        for prediction in predictions.values():
            consensus.append(
                prediction["alpha_score"]
                .groupby(groups, sort=False)
                .rank(pct=True)
            )
        out["alpha_consensus"] = pd.concat(consensus, axis=1).mean(axis=1)
        for window, prediction in predictions.items():
            for column in ("return_hat", "rank_hat", "p_top", "alpha_score"):
                out[f"{column}_{window}m"] = prediction[column]
        return out

    def feature_importance(self, top: int = 20) -> pd.DataFrame:
        frames = []
        for window, model in self.models.items():
            table = model.feature_importance(len(self.base_features))
            frames.append(table["mean"].rename(f"{window}m"))
        result = pd.concat(frames, axis=1).fillna(0.0)
        result["mean"] = result.mean(axis=1)
        return result.sort_values("mean", ascending=False).head(top)

    def save(self, path: Path) -> Path:
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "model_type": "alpha_rank_ensemble",
            "horizon": self.horizon,
            "score_weights": _score_weights_metadata(),
            "windows_months": list(self.windows_months),
            "base_features": self.base_features,
            "n_features": len(self.features),
            "params": self.params,
            "head_estimators": self.head_estimators,
            "trained_on": self.trained_on,
        }
        bundle, manifest = build_bundle(payload, metadata)
        atomic_write_bytes(path, bundle)
        atomic_write_json(metadata_path(path), manifest)
        return path

    @staticmethod
    def load(
        path: Path,
        *,
        verify_integrity: bool = False,
    ) -> "AlphaRankEnsemble":
        path = Path(path)
        if verify_integrity or path.read_bytes().startswith(BUNDLE_MAGIC):
            payload, _ = read_verified_bytes(path)
            return pickle.loads(payload)
        with path.open("rb") as handle:
            return pickle.load(handle)
