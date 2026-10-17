"""Score panels and expose compact, comparable signal output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predixlab_signals.model.pivot_model import PivotModel, classify_action


def confidence_in_cross_section(
    cross_section: pd.DataFrame, signals: pd.DataFrame
) -> np.ndarray:
    """Return each signal's absolute-score percentile within its session."""
    scored = cross_section[["symbol", "date", "score"]].copy()
    scored["date"] = pd.to_datetime(scored["date"])
    scored["pct"] = scored["score"].abs().groupby(scored["date"]).rank(pct=True).round(3)
    keys = signals[["symbol", "date"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    merged = keys.merge(
        scored[["symbol", "date", "pct"]].drop_duplicates(["symbol", "date"]),
        on=["symbol", "date"],
        how="left",
    )
    return merged["pct"].to_numpy()


def enrich_scores(
    scored: pd.DataFrame, cross_section: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach comparable signal strength and stable display fields."""
    out = scored.reset_index(drop=True).copy()
    if cross_section is not None:
        out["confidence"] = confidence_in_cross_section(cross_section, out)
    else:
        out["confidence"] = np.round(out["score"].abs().rank(pct=True), 3)
    out["swing_position"] = np.round(out["swing_pos_hat"], 3)

    for col in ("score", "p_up", "exp_gain_pct", "exp_loss_pct", "exp_amplitude_pct"):
        if col in out.columns:
            out[col] = out[col].round(4)

    return out


def score_panel(
    model: PivotModel,
    frame: pd.DataFrame,
    lookback: int = 250,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Score a trailing panel of bars, one row per symbol-day.

    Turning-point detection is *sequential*: it reads a trailing range window,
    tolerates three sessions between a setup and its trigger, and enforces a
    refractory period. Scoring only the latest bar would therefore emit nothing,
    so the live path has to reproduce the same panel shape the walk-forward run
    stored.

    ``lookback`` is in sessions and is generous on purpose — the longest rule
    window is the 120-day trend average, so 250 leaves the gates fully warmed up
    without scoring a decade of history every night.

    Actions are classified against each day's own cross-section, exactly as they
    were during walk-forward, so a bar scored here carries the same label it
    would have carried then.
    """
    if frame.empty:
        return pd.DataFrame()

    panel = frame if as_of is None else frame[frame["date"] <= pd.Timestamp(as_of)]
    if panel.empty:
        return pd.DataFrame()

    sessions = np.sort(panel["date"].unique())
    if lookback and lookback > 0 and len(sessions) > lookback:
        panel = panel[panel["date"] >= sessions[-lookback]]

    panel = panel.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    signal = model.predict(panel)

    carry = [
        c
        for c in (
            "symbol",
            "date",
            "close",
            "price",
            "open",
            "high",
            "low",
            "volume",
        )
        if c in panel.columns
    ]
    out = pd.concat([panel[carry], signal], axis=1)
    out["action"] = classify_action(
        out["score"].to_numpy(),
        out["exp_amplitude_pct"].to_numpy(),
        model.thresholds,
        groups=out["date"].to_numpy(),
    )
    return out


def build_scores(
    model: PivotModel,
    frame: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Score the latest requested cross-section."""
    if frame.empty:
        return pd.DataFrame()

    if as_of is not None:
        frame = frame[frame["date"] == pd.Timestamp(as_of)]
    else:
        frame = frame[frame["date"] == frame["date"].max()]
    if frame.empty:
        return pd.DataFrame()
    frame = frame.reset_index(drop=True)
    signal = model.predict(frame)
    out = pd.concat([frame[["symbol", "date", "close", "price"]], signal], axis=1)

    out["action"] = classify_action(
        out["score"].to_numpy(),
        out["exp_amplitude_pct"].to_numpy(),
        model.thresholds,
        groups=out["date"].to_numpy() if len(out) >= 50 else None,
    )

    out = enrich_scores(out)

    if min_score is not None:
        out = out[out["score"].abs() >= min_score]

    return out.sort_values("score", ascending=False).reset_index(drop=True)


SIGNAL_COLUMNS = [
    "date",
    "symbol",
    "action",
    "score",
    "confidence",
    "swing_position",
    "p_up",
    "exp_gain_pct",
    "exp_loss_pct",
]


def format_scores(signals: pd.DataFrame, top: int = 20) -> str:
    """Render the strongest positive and negative signals."""
    if signals.empty:
        return "No signals generated."
    cols = [c for c in SIGNAL_COLUMNS if c in signals.columns]
    buys = signals[signals["action"].isin(["BUY", "STRONG_BUY"])].head(top)
    sells = signals[signals["action"].isin(["SELL", "STRONG_SELL"])].tail(top).iloc[::-1]

    parts = [f"As of {pd.Timestamp(signals['date'].iloc[0]).date()}  ({len(signals)} symbols scored)"]
    parts.append("\n=== BUY signals (closest to a local low) ===")
    parts.append(buys[cols].to_string(index=False) if not buys.empty else "  none")
    parts.append("\n=== SELL signals (closest to a local high) ===")
    parts.append(sells[cols].to_string(index=False) if not sells.empty else "  none")
    return "\n".join(parts)
