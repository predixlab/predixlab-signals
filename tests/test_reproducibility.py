"""Reproducibility: no decision may depend on a raw model score.

Boosted-tree outputs are not portable numbers. Their scale depends on the
training window, the LightGBM build, thread count and floating-point summation
order — retraining the same architecture on a different window already shifts
the score standard deviation by 3.5x inside this project, and a different chip
would shift it again. Measured on the walk-forward set, a uniform shift of 1e-3
against *fixed* cut-offs moves the STRONG_BUY count by 11% and relabels 14.6%
of bars.

Every gate in the pipeline therefore consumes a **percentile**, never a score.
These tests pin that: apply any monotone transformation to the scores and the
labels must not move at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.model.pivot_model import ActionThresholds, classify_action

#: Transformations a different machine, build or training run could plausibly
#: introduce. All are strictly increasing, so the *ranking* is untouched.
MONOTONE = {
    "tiny shift": lambda x: x + 1e-9,
    "small shift": lambda x: x + 1e-3,
    "large shift": lambda x: x + 0.05,
    "scale up": lambda x: x * 1.5,
    "scale down": lambda x: x * 0.4,
    "affine": lambda x: 2.3 * x - 0.07,
    "tanh squash": lambda x: np.tanh(x * 30.0),
    "cube": lambda x: np.sign(x) * np.abs(x) ** 3,
}


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    n = 12_000
    return {
        "score": rng.normal(0, 0.02, n),
        "amplitude": np.full(n, 0.10),
        "dates": np.repeat(np.arange(60), n // 60),
    }


def _reference(scale: float = 0.02, seed: int = 1, transform=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample = rng.normal(0, scale, 40_000)
    if transform is not None:
        sample = transform(sample)
    return np.sort(np.quantile(sample, np.linspace(0.0, 1.0, 2001)))


# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(MONOTONE))
def test_cross_sectional_labels_survive_any_monotone_change(batch, name):
    """The primary route: ranking within a date needs no stored state."""
    th = ActionThresholds()
    base = classify_action(batch["score"], batch["amplitude"], th, groups=batch["dates"])
    moved = classify_action(
        MONOTONE[name](batch["score"]), batch["amplitude"], th, groups=batch["dates"]
    )
    assert (base != moved).sum() == 0, name


@pytest.mark.parametrize("name", ["small shift", "scale up", "affine"])
def test_reference_route_survives_when_recalibrated(batch, name):
    """The fallback route holds as long as calibration is redone on the same model."""
    transform = MONOTONE[name]
    before = classify_action(batch["score"], batch["amplitude"], ActionThresholds(reference=_reference()))
    after = classify_action(
        transform(batch["score"]),
        batch["amplitude"],
        ActionThresholds(reference=_reference(transform=transform)),
    )
    assert (before != after).sum() == 0, name


def test_selected_count_is_exactly_the_configured_quantile(batch):
    th = ActionThresholds(strong_buy_q=0.90, buy_q=0.75, sell_q=0.25, strong_sell_q=0.10)
    action = classify_action(batch["score"], batch["amplitude"], th, groups=batch["dates"])
    n = len(action)
    assert abs((action == "STRONG_BUY").sum() / n - 0.10) < 0.01
    assert abs((action == "STRONG_SELL").sum() / n - 0.10) < 0.01


def test_absolute_cutoffs_would_have_failed(batch):
    """Documents the defect this design replaces — a guard against regressing."""
    score = batch["score"]
    cut = float(np.quantile(score, 0.90))
    naive_flips = int(((score >= cut) != ((score + 1e-3) >= cut)).sum())

    th = ActionThresholds()
    robust_before = classify_action(score, batch["amplitude"], th, groups=batch["dates"])
    robust_after = classify_action(score + 1e-3, batch["amplitude"], th, groups=batch["dates"])
    robust_flips = int((robust_before != robust_after).sum())

    # The contrast is the point: a fixed cut-off moves, a percentile does not.
    assert naive_flips > 0
    assert robust_flips == 0


def test_percentile_is_monotone_in_score(batch):
    """Higher score must never map to a lower percentile.

    For the grouped route that holds *within* a date, which is the whole point
    of grouping — percentiles are only comparable inside their own cross-section.
    """
    th = ActionThresholds(reference=_reference())

    pct = th.percentile(batch["score"], batch["dates"])
    frame = pd.DataFrame({"d": batch["dates"], "s": batch["score"], "p": pct})
    for _, block in frame.groupby("d"):
        ordered = block.sort_values("s")
        assert np.all(np.diff(ordered["p"].to_numpy()) >= -1e-12)

    global_pct = th.percentile(batch["score"], None)
    order = np.argsort(batch["score"])
    assert np.all(np.diff(global_pct[order]) >= -1e-12)


def test_single_symbol_query_falls_back_to_the_reference():
    """One name has no cross-section; the stored distribution must carry it."""
    th = ActionThresholds(reference=_reference())
    high = classify_action(np.array([0.06]), np.array([0.10]), th)
    low = classify_action(np.array([-0.06]), np.array([0.10]), th)
    assert high[0] == "STRONG_BUY"
    assert low[0] == "STRONG_SELL"


def test_amplitude_gate_is_economic_not_a_model_score(batch):
    """`min_amplitude` stays absolute on purpose: it is a percentage of price.

    It binds on 0.14% of live bars, and unlike the score it carries a unit that
    means the same thing on every machine.
    """
    th = ActionThresholds(min_amplitude=0.05)
    tiny = np.full(batch["score"].size, 0.01)
    action = classify_action(batch["score"], tiny, th, groups=batch["dates"])
    assert (action == "HOLD").all()


def test_thresholds_serialise_without_raw_cutoffs():
    th = ActionThresholds(reference=_reference())
    payload = th.as_dict()
    assert {"strong_buy_q", "buy_q", "sell_q", "strong_sell_q"} <= set(payload)
    # No absolute score cut-off may be persisted as a decision boundary.
    assert "strong_buy" not in payload and "strong_sell" not in payload
    assert payload["reference_size"] == 2001


def test_calibrate_stores_a_sorted_reference():
    from predixlab_signals.model.pivot_model import PivotModel

    model = PivotModel()
    rng = np.random.default_rng(3)
    model.calibrate(rng.normal(0, 0.02, 5000))
    ref = model.thresholds.reference
    assert ref is not None and ref.size == 2001
    assert np.all(np.diff(ref) >= 0)
    assert model.thresholds.strong_buy_q == pytest.approx(0.90)
