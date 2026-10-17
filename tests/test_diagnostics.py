"""Tests for the systematic diagnostics.

These exist so a change can be judged in aggregate rather than by whichever
symbol happened to be inspected. The metrics they cover — trend tilt versus
ground truth, and lift measured *inside* a trend bucket — are what distinguish
a genuine turn detector from a dressed-up trend indicator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.evaluation.diagnostics import (
    add_trend_context,
    trend_correlation,
    trend_profile,
    universe_report,
    within_trend_discrimination,
)


def _panel(n=8000, seed=0, tilt=1.0):
    """Synthetic panel where true lows follow declines and the model can tilt harder.

    Noise is deliberately large relative to the trend term: with a near-noiseless
    model the rank correlation saturates near -1 and the tilt ratio stops
    responding, which measures the fixture rather than the metric.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n // 4)
    rows = []
    for sym in ("AAA", "BBB", "CCC", "DDD"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        trail = pd.Series(close).pct_change(20).fillna(0.0).to_numpy()
        # A latent "this really is a turn" factor, independent of trend. The
        # model sees a noisy version of it, so within-bucket lift is genuine
        # rather than an artefact of the trend term.
        latent = rng.normal(0, 1, len(dates))
        p_true = np.clip(0.08 - 0.5 * trail + 0.05 * latent, 0.01, 0.6)
        y_low = (rng.random(len(dates)) < p_true).astype(float)
        model = np.clip(
            0.08 - 0.5 * tilt * trail + 0.05 * latent + rng.normal(0, 0.05, len(dates)),
            0,
            1,
        )
        rows.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "close": close,
                    "p_pivot_low": model,
                    "p_pivot_high": 1.0 - model,
                    "y_pivot_low_hold": y_low,
                    "y_pivot_high_hold": (
                        rng.random(len(dates)) < 0.06
                    ).astype(float),
                    "action": rng.choice(["HOLD", "BUY", "STRONG_BUY", "SELL"], len(dates)),
                    "fwd_ret_cs": rng.normal(0, 0.03, len(dates)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_trend_context_is_computed_per_symbol():
    panel = _panel()
    out = add_trend_context(panel)
    assert out["trailing_return"].notna().any()
    for _, block in out.groupby("symbol"):
        assert block["trailing_return"].head(20).isna().all()


def test_trend_profile_reports_signal_and_truth_together():
    """Both are needed: a lopsided signal is only a defect if truth is not."""
    table = trend_profile(_panel())
    assert {
        "buy_share",
        "sell_share",
        "true_low_hold_rate",
        "true_high_hold_rate",
    } <= set(table.columns)
    # Truth must tilt: lows commoner after declines than after rallies.
    assert (
        table.loc["steep_down", "true_low_hold_rate"]
        > table.loc["steep_up", "true_low_hold_rate"]
    )


def test_overfade_flags_a_model_that_tilts_harder_than_reality():
    mild = trend_correlation(_panel(tilt=1.0))
    harsh = trend_correlation(_panel(tilt=4.0))
    assert harsh["low_overfade"] > mild["low_overfade"]
    # Both must point the same way as the truth they are compared against.
    assert mild["low_overfade"] > 0 and harsh["low_overfade"] > 0


def test_within_trend_discrimination_holds_the_free_tilt_constant():
    """Lift inside a bucket is the honest number; a pure trend proxy scores ~1."""
    panel = _panel()
    real = within_trend_discrimination(panel, "low")
    assert not real.empty
    assert (real["lift"] > 1.0).any()

    # A model that is *only* trailing return carries no within-bucket skill.
    proxy = panel.copy()
    ctx = add_trend_context(proxy)
    proxy["p_pivot_low"] = -ctx["trailing_return"].fillna(0.0).to_numpy()
    flat = within_trend_discrimination(proxy, "low")
    assert not flat.empty
    assert flat["lift"].mean() < real["lift"].mean()


def test_within_trend_discrimination_needs_its_columns():
    assert within_trend_discrimination(pd.DataFrame({"close": [1.0]}), "low").empty


def test_universe_report_flags_indices_and_non_members():
    panel = _panel()
    panel = pd.concat(
        [panel, panel.head(50).assign(symbol="^GSPC")], ignore_index=True
    )
    membership = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "start": pd.to_datetime(["2020-01-01"] * 2),
            "end": pd.to_datetime(["2099-12-31"] * 2),
        }
    )
    report = universe_report(panel, membership)
    assert report["index_instruments"] == ["^GSPC"]
    assert set(report["not_in_membership"]) >= {"CCC", "DDD", "^GSPC"}
    assert report["share_outside_index"] > 0.0
