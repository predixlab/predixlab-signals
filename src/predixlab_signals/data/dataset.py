"""Dataset assembly: OHLCV panel -> features + labels -> a single parquet file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from predixlab_signals.config import BENCHMARKS, DEFAULT_HORIZON, MIN_HISTORY, PATHS, Paths
from predixlab_signals.contracts import DATASET_CONTRACT
from predixlab_signals.data.csv_data import read_calendar, read_membership
from predixlab_signals.data.loader import load_benchmark, load_panel
from predixlab_signals.data_quality import DEFAULT_MAX_BAD_DATA_RATIO, DailyDataValidator
from predixlab_signals.features.core import build_features, feature_columns
from predixlab_signals.labels.swing import MIN_ATR_PCT, build_labels

#: Raw sessions retained by the live path. This covers the 250 sessions scored
#: by the sequential rules, the nested 250/252-session feature windows, and a
#: generous initialization buffer for exponentially weighted indicators.
DEFAULT_INFERENCE_HISTORY = 1000

#: Columns kept alongside the features for backtesting and reporting.
CONTEXT_COLUMNS = [
    "symbol",
    "date",
    "close",
    "price",
    "open",
    "high",
    "low",
    "volume",
    "open_next",
    "fwd_mfe",
    "fwd_mae",
    "fwd_ret",
    "label_scale",
    "atr_pct_label",
    "pivot",
]


def build_dataset(
    market: str = "us",
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    barrier_width: float = 1.5,
    min_history: int = MIN_HISTORY,
    members_only: bool = True,
    data_root: Path | None = None,
    paths: Paths = PATHS,
) -> pd.DataFrame:
    """Assemble the modelling table.

    Features are computed on the *full* history and only trimmed afterwards, so
    a ``start`` filter never truncates the rolling warm-up windows. Unless
    explicitly overridden, research datasets read from ``paths.training_data``;
    the clean production tree is reserved for live inference.
    """
    source_root = data_root or paths.training_data
    panel = load_panel(
        market,
        symbols=symbols,
        start=None,
        end=end,
        min_history=min_history,
        members_only=members_only,
        data_root=source_root,
        paths=paths,
    )
    if panel.empty:
        return pd.DataFrame()

    benchmark = load_benchmark(
        market,
        BENCHMARKS.get(market.lower(), "^GSPC"),
        data_root=source_root,
        paths=paths,
    )
    features = build_features(panel, benchmark)
    labels = build_labels(panel, horizon=horizon, barrier_width=barrier_width)

    # Raw bars the backtest needs, plus next session's open: every simulated
    # fill happens there, never on the close that produced the signal.
    bars = panel[
        ["symbol", "date", "open", "high", "low", "volume", "is_member"]
    ].copy()
    bars["open_next"] = bars.groupby("symbol", sort=False)["open"].shift(-1)

    data = features.merge(labels, on=["symbol", "date"], how="inner").merge(
        bars, on=["symbol", "date"], how="left"
    )

    # Only now, with every rolling window warmed up on real history, are the
    # pre-membership bars discarded. Dropping them earlier would have crippled
    # the features of anything recently added to the index.
    if "is_member" in data.columns:
        data = data[data["is_member"].fillna(True).astype(bool)]
        data = data.drop(columns=["is_member"])

    if start is not None:
        data = data[data["date"] >= pd.Timestamp(start)]

    data = _add_relative_targets(data)

    feats = feature_columns(data)
    # A row is usable only if the slow rolling windows have warmed up. Requiring
    # every column to be finite would throw away far too much, so demand the
    # core block and let LightGBM handle the rest as missing values.
    core = [c for c in ("px_sma120", "pctile_250", "rsi_14", "atr_pct", "vol_ratio_20_60") if c in feats]
    data = data.dropna(subset=core)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = _downcast(data)
    return data.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def build_inference_dataset(
    market: str = "us",
    symbols: list[str] | None = None,
    end: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    history_sessions: int = DEFAULT_INFERENCE_HISTORY,
    min_history: int = MIN_HISTORY,
    members_only: bool = True,
    data_root: Path | None = None,
    paths: Paths = PATHS,
    validation_date: str | None = None,
    max_bad_data_ratio: float = DEFAULT_MAX_BAD_DATA_RATIO,
) -> pd.DataFrame:
    """Build the bounded, causal frame required by production scoring.

    Unlike :func:`build_dataset`, this path neither computes future labels nor
    writes a reusable training artifact. It retains enough raw history to warm
    nested rolling features and sequential turning-point rules, then returns
    the model inputs plus the OHLC context used by those rules.

    Production callers must supply ``validation_date``. It pins the input end
    and requires fresh, usable target-session coverage before returning data.
    """
    validator = None
    if validation_date is not None:
        if end is not None and pd.Timestamp(end) != pd.Timestamp(validation_date):
            raise ValueError("end must match validation_date for daily inference")
        end = validation_date
        calendar_path = (
            data_root / "calendar" / f"{market.lower()}.txt"
            if data_root is not None else paths.calendar_path(market)
        )
        instruments_path = (
            data_root / "instruments" / f"{market.lower()}.txt"
            if data_root is not None else paths.instruments_path(market)
        )
        validator = DailyDataValidator(
            market,
            validation_date,
            read_calendar(calendar_path),
            read_membership(instruments_path),
            max_bad_ratio=max_bad_data_ratio,
        )
    panel = load_panel(
        market,
        symbols=symbols,
        end=end,
        min_history=min_history,
        members_only=members_only,
        history_sessions=history_sessions,
        data_root=data_root,
        paths=paths,
        on_read_error=validator.record_read_error if validator is not None else None,
    )
    if panel.empty and validator is None:
        return pd.DataFrame()

    benchmark = load_benchmark(
        market,
        BENCHMARKS.get(market.lower(), "^GSPC"),
        end=end,
        history_sessions=history_sessions,
        data_root=data_root,
        paths=paths,
    )
    if validator is not None:
        panel = validator.validate_panel(panel, benchmark)
    data = build_features(panel, benchmark)
    bars = panel[["symbol", "date", "open", "high", "low", "volume", "is_member"]]
    data = data.merge(bars, on=["symbol", "date"], how="left")

    if "is_member" in data.columns:
        data = data[data["is_member"].fillna(True).astype(bool)]
        data = data.drop(columns=["is_member"])

    # The model rescales normalized amplitude predictions with this causal ATR
    # value. ``atr_pct`` is computed from the same Wilder EWM as the label path,
    # so no forward-looking label calculation is needed in production.
    data["atr_pct_label"] = data["atr_pct"].clip(lower=MIN_ATR_PCT)
    data["label_scale"] = data["atr_pct_label"] * float(np.sqrt(horizon))

    feats = feature_columns(data)
    core = [
        c
        for c in ("px_sma120", "pctile_250", "rsi_14", "atr_pct", "vol_ratio_20_60")
        if c in feats
    ]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=core)
    data = _downcast(data)
    data = data.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
    if validator is not None:
        data = validator.validate_dataset(data, required_features=core)
        data.attrs["data_validation"] = validator.report
    return data


def _downcast(data: pd.DataFrame) -> pd.DataFrame:
    """Store float columns as float32.

    Halves the dataset on disk and in memory (1.9 GB -> 0.9 GB for the US
    universe) at no cost to the model: LightGBM bins features into 8-bit
    histograms internally, so float64 precision was never reaching the trees.
    """
    floats = data.select_dtypes(include=["float64"]).columns
    data[floats] = data[floats].astype("float32")
    return data


def _add_relative_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectionally neutral versions of the directional targets.

    Subtracting each day's universe mean strips out the common market move that
    every symbol shares. What remains is the stock-specific component - the only
    part a per-symbol timing model can honestly claim to predict, and the part
    that survives out of sample.
    """
    by_date = data.groupby("date", sort=False)
    data["y_swing_pos_cs"] = data["y_swing_pos"] - by_date["y_swing_pos"].transform("mean")
    data["y_edge_cs"] = data["y_edge"] - by_date["y_edge"].transform("mean")
    data["fwd_ret_cs"] = data["fwd_ret"] - by_date["fwd_ret"].transform("mean")
    # Relative barrier: did this name beat the median name over the horizon?
    median = by_date["fwd_ret"].transform("median")
    data["y_barrier_cs"] = np.where(
        data["fwd_ret"].isna(), np.nan, np.where(data["fwd_ret"] > median, 1.0, -1.0)
    )
    return data


def dataset_path(
    market: str, horizon: int, tag: str = "default", paths: Paths = PATHS
) -> Path:
    return paths.dataset_dir(market) / f"pivot_h{horizon}_{tag}.parquet"


def save_dataset(
    data: pd.DataFrame,
    market: str,
    horizon: int,
    tag: str = "default",
    paths: Paths = PATHS,
) -> Path:
    DATASET_CONTRACT.validate(data, allow_empty=False)
    path = dataset_path(market, horizon, tag, paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)
    meta = {
        "rows": int(len(data)),
        "symbols": int(data["symbol"].nunique()),
        "start": str(data["date"].min().date()),
        "end": str(data["date"].max().date()),
        "horizon": horizon,
        "features": feature_columns(data),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path


def load_dataset(
    market: str, horizon: int, tag: str = "default", paths: Paths = PATHS
) -> pd.DataFrame:
    path = dataset_path(market, horizon, tag, paths)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run `predixlab-signals build-dataset` first.")
    data = pd.read_parquet(path)
    DATASET_CONTRACT.validate(data, allow_empty=False)
    return data
