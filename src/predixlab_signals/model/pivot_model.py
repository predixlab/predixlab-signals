"""The PivotModel: three compact LightGBM heads over one shared feature matrix.

The decomposition is the whole design. A naive model predicts the two forward
excursions (MFE and MAE) directly — and fails, because both are dominated by
volatility. Volatility is easy to forecast, so the model spends all of its
capacity there, and the *difference* that actually carries direction cancels out
to noise. Measured on the S&P universe, that formulation produces a rank IC
indistinguishable from zero.

Splitting the swing into two orthogonal questions fixes it:

``swing_pos``  *Where* in the coming range does today's close sit? 0 means the
               close is the bottom of the next H bars — a low, i.e. a buy. 1
               means it is the top — a high, i.e. a sell. Bounded in [0, 1] and
               volatility-free by construction, so this head can only learn
               direction.
``amplitude``  *How wide* is that range going to be, in ATR units? Pure
               volatility clustering — strongly predictable, and it is what
               sizes the target and the stop.

The excursions are then reconstructed rather than predicted:

    mfe = amplitude * (1 - swing_pos)      mae = -amplitude * swing_pos

A third head, ``updown``, is a binary classifier on the triple-barrier outcome
("does price travel +1.5 ATR before -1.5 ATR?"). It fails differently from a
regression — insensitive to outliers, directly calibratable — so requiring the
two directional views to agree is what keeps the signal honest.

Two hold classifiers estimate whether today's low/high survives its bounded
future window. Two conditional regressors estimate directional future return
when that hold succeeds. Their product is the expected-return-weighted strength
used by the turning-point gates.
"""

from __future__ import annotations

import inspect
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from predixlab_signals.labels.swing import PIVOT_RETURN_CAP
from predixlab_signals.model.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    BUNDLE_MAGIC,
    build_bundle,
    metadata_path,
    read_verified_bytes,
)

#: LightGBM 4.6 replaced ``eval_set`` with ``eval_X``/``eval_y`` in the sklearn
#: API. Detect once at import rather than catching a warning at every fit.
_SUPPORTS_EVAL_XY = "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters

#: Deliberately small trees with strong regularisation. The signal here is weak
#: and noisy; capacity buys overfitting, not accuracy.
DEFAULT_PARAMS: dict[str, Any] = {
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.04,
    "n_estimators": 400,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "n_jobs": -1,
    "verbose": -1,
}

PIVOT_HEADS = (
    ("pivot_low", "pivot_low_model", "y_pivot_low_hold"),
    ("pivot_high", "pivot_high_model", "y_pivot_high_hold"),
)
PIVOT_RETURN_HEADS = (
    ("pivot_low_return", "pivot_low_return_model", "y_pivot_low_return"),
    ("pivot_high_return", "pivot_high_return_model", "y_pivot_high_return"),
)


@dataclass
class ActionThresholds:
    """Percentile cut-points, plus the reference distribution they index into.

    Absolute score cut-offs are not reproducible. The raw score is a blend of
    boosted-tree outputs whose scale depends on the training window, the
    LightGBM build, thread count and floating-point order; retraining the same
    architecture on a different window already moves its standard deviation by
    3.5x in this project, and a different chip would move it again. Measured on
    the walk-forward set, a uniform shift of 1e-3 in score — well inside that
    variation — moves the STRONG_BUY count by 11% and relabels 14.6% of bars.

    Storing **quantiles against a reference distribution** removes the problem
    entirely: percentiles are invariant under any monotone transformation of the
    score, so the same shift relabels exactly 0.0% of bars and the number of
    names selected does not move at all. The reference grid is a snapshot of the
    calibration model's own scores, so it travels with the model.
    """

    strong_buy_q: float = 0.90
    buy_q: float = 0.75
    sell_q: float = 0.25
    strong_sell_q: float = 0.10
    #: Expected swing below which nothing is worth trading. This one stays
    #: absolute on purpose: it is a percentage of price, an economic quantity,
    #: not a model score.
    min_amplitude: float = 0.03
    #: Sorted score grid from calibration. ``None`` falls back to ranking within
    #: whatever batch is being scored.
    reference: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        out = {
            "strong_buy_q": self.strong_buy_q,
            "buy_q": self.buy_q,
            "sell_q": self.sell_q,
            "strong_sell_q": self.strong_sell_q,
            "min_amplitude": self.min_amplitude,
            "reference_size": 0 if self.reference is None else int(self.reference.size),
        }
        if self.reference is not None and self.reference.size:
            out["reference_quantiles"] = {
                q: float(np.quantile(self.reference, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            }
        return out

    def percentile(self, score: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Map raw scores onto ``[0, 1]`` percentiles.

        Three routes, in descending order of robustness:

        1. ``groups`` given (normally the date) — rank **within each group**.
           This needs no stored state at all and is invariant to *any* monotone
           change in the score, so it reproduces exactly on any machine.
        2. A stored reference — percentile against the calibration snapshot.
           Invariant provided the reference was produced by the same model, so
           it survives retraining as long as calibration is redone with it.
        3. Neither — rank within whatever batch was passed.
        """
        score = np.asarray(score, dtype="float64")
        if groups is not None:
            return (
                pd.Series(score).groupby(np.asarray(groups)).rank(pct=True).to_numpy()
            )
        if self.reference is not None and self.reference.size:
            return np.searchsorted(self.reference, score, side="left") / float(
                self.reference.size
            )
        return pd.Series(score).rank(pct=True).to_numpy()


@dataclass
class PivotModel:
    """Container for the fitted direction, hold, and return heads."""

    features: list[str] = field(default_factory=list)
    horizon: int = 10
    barrier_width: float = 1.5
    #: ``True`` trains the directional heads on cross-sectionally demeaned
    #: targets, so the model predicts the stock-specific component rather than
    #: the market move every name shares.
    market_neutral: bool = True
    #: Per-head tree counts. The objectives reach their optimum at different
    #: depths, so a shared count either starves one or overfits another.
    head_estimators: dict[str, int] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    thresholds: ActionThresholds = field(default_factory=ActionThresholds)
    pos_model: lgb.LGBMRegressor | None = None
    amp_model: lgb.LGBMRegressor | None = None
    updown_model: lgb.LGBMClassifier | None = None
    #: Direct hold detectors: whether today's high or low survives the next
    #: bounded window, regardless of the trend or price action before today.
    pivot_low_model: lgb.LGBMClassifier | None = None
    pivot_high_model: lgb.LGBMClassifier | None = None
    #: Directional future return conditional on the corresponding hold event.
    pivot_low_return_model: lgb.LGBMRegressor | None = None
    pivot_high_return_model: lgb.LGBMRegressor | None = None
    trained_on: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(
        self,
        train: pd.DataFrame,
        features: list[str] | None = None,
        valid: pd.DataFrame | None = None,
        early_stopping_rounds: int = 50,
    ) -> "PivotModel":
        """Fit all available heads, optionally early-stopping on a held-out block.

        ``valid`` must be separated from ``train`` by a purge gap; this method
        does not enforce that, the fold builder does.
        """
        self.features = features or self.features
        x = train[self.features]

        # The direction head sees a bounded target, so plain L2 is well behaved.
        # The size head sees a heavy right tail, so Huber caps outlier influence.
        self.pos_model = lgb.LGBMRegressor(objective="l2", **self._head_params("swing_pos"))
        self.amp_model = lgb.LGBMRegressor(objective="huber", alpha=1.0, **self._head_params("amplitude"))

        pos_target = "y_swing_pos_cs" if self.market_neutral else "y_swing_pos"
        for model, target, metric in (
            (self.pos_model, pos_target, "l2"),
            (self.amp_model, "y_amplitude", "l2"),
        ):
            mask = train[target].notna()
            fit_kwargs = self._eval_kwargs(valid, target, early_stopping_rounds, metric)
            model.fit(x[mask], train.loc[mask, target], **fit_kwargs)

        # Timeouts carry no directional information, so they are excluded rather
        # than folded into a third class that would blur the decision boundary.
        clf_target = "y_barrier_cs" if self.market_neutral else "y_barrier"
        clf_mask = train[clf_target].isin([-1.0, 1.0])
        self.updown_model = lgb.LGBMClassifier(objective="binary", **self._head_params("updown"))
        self.updown_model.fit(
            x[clf_mask],
            (train.loc[clf_mask, clf_target] > 0).astype(int),
            **self._eval_kwargs(valid, clf_target, early_stopping_rounds, "auc", binary=True),
        )
        self._pos_target = pos_target
        self._clf_target = clf_target

        pivot_targets: dict[str, str] = {}
        for head, attr, target in PIVOT_HEADS:
            if target not in train.columns:
                continue
            mask = train[target].notna()
            if not mask.any():
                continue
            model = lgb.LGBMClassifier(objective="binary", **self._head_params(head))
            model.fit(
                x[mask],
                train.loc[mask, target].astype(int),
                **self._eval_kwargs(valid, target, early_stopping_rounds, "auc"),
            )
            setattr(self, attr, model)
            pivot_targets[head] = target

        pivot_return_targets: dict[str, str] = {}
        for head, attr, target in PIVOT_RETURN_HEADS:
            if target not in train.columns:
                continue
            mask = train[target].notna()
            if not mask.any():
                continue
            model = lgb.LGBMRegressor(
                objective="huber",
                alpha=0.9,
                **self._head_params(head),
            )
            model.fit(
                x[mask],
                train.loc[mask, target],
                **self._eval_kwargs(
                    valid,
                    target,
                    early_stopping_rounds,
                    "l2",
                ),
            )
            setattr(self, attr, model)
            pivot_return_targets[head] = target

        self.trained_on = {
            "rows": int(len(train)),
            "symbols": int(train["symbol"].nunique()),
            "start": str(pd.Timestamp(train["date"].min()).date()),
            "end": str(pd.Timestamp(train["date"].max()).date()),
            "pivot_targets": pivot_targets,
            "pivot_return_targets": pivot_return_targets,
            # Only meaningful when a validation block drove early stopping; a
            # plain fit reports 0, which would read as "zero trees".
            "best_iterations": {
                name: value
                for name, model in (
                    ("swing_pos", self.pos_model),
                    ("amplitude", self.amp_model),
                    ("updown", self.updown_model),
                    ("pivot_low", self.pivot_low_model),
                    ("pivot_high", self.pivot_high_model),
                    ("pivot_low_return", self.pivot_low_return_model),
                    ("pivot_high_return", self.pivot_high_return_model),
                )
                if (value := getattr(model, "best_iteration_", None))
            },
        }
        return self

    def _head_params(self, head: str) -> dict[str, Any]:
        """Base parameters with this head's own tree count applied, if one is set."""
        params = dict(self.params)
        count = (self.head_estimators or {}).get(head)
        if count:
            params["n_estimators"] = max(int(count), 1)
        return params

    def _eval_kwargs(
        self,
        valid: pd.DataFrame | None,
        target: str,
        rounds: int,
        metric: str,
        binary: bool = False,
    ) -> dict[str, Any]:
        if valid is None or valid.empty or rounds <= 0:
            return {}
        if target not in valid.columns:
            return {}
        mask = valid[target].isin([-1.0, 1.0]) if binary else valid[target].notna()
        block = valid[mask]
        if block.empty:
            return {}
        y = (block[target] > 0).astype(int) if binary else block[target]
        if metric == "auc" and not binary:
            y = block[target].astype(int)
        kwargs: dict[str, Any] = {
            "eval_metric": metric,
            "callbacks": [lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)],
        }
        # LightGBM 4.6 renamed the sklearn-API holdout arguments. Support both so
        # the package works across the versions its dependency range allows.
        if _SUPPORTS_EVAL_XY:
            kwargs["eval_X"] = block[self.features]
            kwargs["eval_y"] = y
        else:
            kwargs["eval_set"] = [(block[self.features], y)]
        return kwargs

    # ------------------------------------------------------------------
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score a frame and return the full derived signal block."""
        if self.pos_model is None or self.amp_model is None or self.updown_model is None:
            raise RuntimeError("PivotModel is not fitted")

        x = frame[self.features]
        raw_pos = np.asarray(self.pos_model.predict(x), dtype="float64")
        # A market-neutral head outputs a deviation centred on zero; adding back
        # the neutral 0.5 midpoint puts it on the same [0, 1] scale as the
        # absolute head so everything downstream is identical.
        if self.market_neutral:
            raw_pos = raw_pos + 0.5
        # Predictions are shrunk toward the middle by regression-to-the-mean, so
        # clipping to the label's own bounds is a restoration, not a distortion.
        swing_pos = np.clip(raw_pos, 0.0, 1.0)
        amplitude = np.maximum(np.asarray(self.amp_model.predict(x), dtype="float64"), 1e-3)
        p_up = np.asarray(self.updown_model.predict_proba(x)[:, 1], dtype="float64")

        # Reconstruct the excursions from direction x size.
        mfe_hat = amplitude * (1.0 - swing_pos)
        mae_hat = -amplitude * swing_pos

        out = pd.DataFrame(index=frame.index)
        out["swing_pos_hat"] = swing_pos
        out["amplitude_hat"] = amplitude
        out["mfe_hat"] = mfe_hat
        out["mae_hat"] = mae_hat
        out["edge_hat"] = mfe_hat + mae_hat
        out["p_up"] = p_up

        scale = frame["label_scale"].to_numpy(dtype="float64")
        out["exp_gain_pct"] = mfe_hat * scale
        out["exp_loss_pct"] = mae_hat * scale
        out["exp_edge_pct"] = out["edge_hat"] * scale
        out["exp_amplitude_pct"] = amplitude * scale

        out["score"] = pivot_score(swing_pos, p_up)

        for attr, column in (
            ("pivot_low_model", "p_pivot_low"),
            ("pivot_high_model", "p_pivot_high"),
        ):
            model = getattr(self, attr, None)
            out[column] = (
                np.asarray(model.predict_proba(x)[:, 1], dtype="float64")
                if model is not None
                else np.nan
            )

        for side in ("low", "high"):
            probability = out[f"p_pivot_{side}"].to_numpy(dtype="float64")
            model = getattr(self, f"pivot_{side}_return_model", None)
            if model is None:
                out[f"pivot_{side}_return_hat"] = np.nan
                out[f"pivot_{side}_return_pct_hat"] = np.nan
                out[f"pivot_{side}_score"] = probability
                continue
            return_hat = np.clip(
                np.asarray(model.predict(x), dtype="float64"),
                -PIVOT_RETURN_CAP,
                PIVOT_RETURN_CAP,
            )
            out[f"pivot_{side}_return_hat"] = return_hat
            out[f"pivot_{side}_return_pct_hat"] = return_hat * scale
            out[f"pivot_{side}_score"] = probability * np.maximum(
                return_hat,
                0.0,
            )
        return out

    # ------------------------------------------------------------------
    def calibrate(
        self,
        valid_scores: np.ndarray,
        top_frac: float = 0.10,
        mid_frac: float = 0.25,
        grid: int = 2001,
    ) -> None:
        """Store the validation score distribution the cut-points index into.

        Only the *shape* is kept, as an evenly spaced quantile grid. Nothing
        downstream ever compares a raw score to a stored number.
        """
        finite = valid_scores[np.isfinite(valid_scores)]
        if finite.size < 100:
            return
        reference = np.quantile(finite, np.linspace(0.0, 1.0, grid))
        self.thresholds = ActionThresholds(
            strong_buy_q=1.0 - top_frac,
            buy_q=1.0 - mid_frac,
            sell_q=mid_frac,
            strong_sell_q=top_frac,
            min_amplitude=self.thresholds.min_amplitude,
            reference=np.sort(reference),
        )

    # ------------------------------------------------------------------
    def feature_importance(self, top: int = 30) -> pd.DataFrame:
        frames = []
        for name, model in (
            ("swing_pos", self.pos_model),
            ("amplitude", self.amp_model),
            ("updown", self.updown_model),
            ("pivot_low", self.pivot_low_model),
            ("pivot_high", self.pivot_high_model),
            (
                "pivot_low_return",
                getattr(self, "pivot_low_return_model", None),
            ),
            (
                "pivot_high_return",
                getattr(self, "pivot_high_return_model", None),
            ),
        ):
            if model is None:
                continue
            imp = pd.Series(model.feature_importances_, index=self.features, dtype="float64")
            frames.append((imp / imp.sum()).rename(name))
        if not frames:
            return pd.DataFrame()
        table = pd.concat(frames, axis=1)
        table["mean"] = table.mean(axis=1)
        return table.sort_values("mean", ascending=False).head(top)

    # ------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        payload = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        metadata = {
            "horizon": self.horizon,
            "barrier_width": self.barrier_width,
            "n_features": len(self.features),
            "features": self.features,
            "params": self.params,
            "head_estimators": self.head_estimators,
            "thresholds": self.thresholds.as_dict(),
            "trained_on": self.trained_on,
        }
        bundle, manifest = build_bundle(payload, metadata)
        atomic_write_bytes(path, bundle)
        atomic_write_json(
            metadata_path(path),
            manifest,
        )
        return path

    @staticmethod
    def load(path: Path, *, verify_integrity: bool = False) -> "PivotModel":
        path = Path(path)
        if verify_integrity or path.read_bytes().startswith(BUNDLE_MAGIC):
            payload, _ = read_verified_bytes(path)
            return pickle.loads(payload)
        with path.open("rb") as fh:
            return pickle.load(fh)


def pivot_score(swing_pos_hat: np.ndarray, p_up: np.ndarray) -> np.ndarray:
    """Blend the two directional views into a single ``[-1, 1]`` score.

    ``+1`` means "as close to a local low as the model ever calls" (buy) and
    ``-1`` the mirror image (sell). The two halves are weighted equally: neither
    head has earned the right to outvote the other, and requiring agreement is
    what suppresses the false positives each one produces alone.
    """
    pos_part = 1.0 - 2.0 * np.asarray(swing_pos_hat, dtype="float64")
    prob_part = 2.0 * np.asarray(p_up, dtype="float64") - 1.0
    return 0.5 * pos_part + 0.5 * prob_part


def classify_action(
    score: np.ndarray,
    amplitude_pct: np.ndarray,
    thresholds: ActionThresholds,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Map score + expected swing size onto a recommendation label.

    Classification is by **percentile**, never by raw score, so the output is
    reproducible across machines, LightGBM builds and retraining windows: any
    monotone change to the score leaves every label untouched.
    """
    percentile = thresholds.percentile(score, groups)
    amplitude_pct = np.asarray(amplitude_pct, dtype="float64")

    action = np.full(percentile.shape, "HOLD", dtype=object)
    action[percentile >= thresholds.buy_q] = "BUY"
    action[percentile >= thresholds.strong_buy_q] = "STRONG_BUY"
    action[percentile <= thresholds.sell_q] = "SELL"
    action[percentile <= thresholds.strong_sell_q] = "STRONG_SELL"
    # A correct direction call on a swing too small to clear costs is not a trade.
    action[amplitude_pct < thresholds.min_amplitude] = "HOLD"
    return action
