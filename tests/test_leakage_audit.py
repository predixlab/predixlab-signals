"""Leakage and scale-invariance audit.

Two failure modes would invalidate every number this project reports, and
neither shows up in ordinary metrics — a leaking model simply looks good.

**Leakage.** Any feature that reads a bar it could not have seen, any label
whose lookahead exceeds the purge gap, or any computation that lets one symbol
see another, silently inflates every result. The decisive test is truncation:
recompute on a prefix of the series and require bit-identical values for every
bar before the cut. A quantity that passes cannot be using the future, because
the future was not present.

**Scale dependence.** The model is trained across ~700 symbols whose prices span
three orders of magnitude, on collector data where prices are additionally rebased
so each symbol starts at 1.0 — an arbitrary constant. Any feature or label that
responds to absolute price or share count is therefore encoding "which stock is
this", or worse, an artefact of the rebasing. The test multiplies the inputs and
demands the outputs do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.features.core import (
    compute_market_features,
    compute_relative_features,
    compute_symbol_features,
    feature_columns,
)
from predixlab_signals.labels.swing import build_labels

CUT = 500
HORIZON = 10
HOLD_WINDOW = 5

#: Training targets and evaluation labels. Every one must be dimensionless.
SCALE_FREE_LABELS = [
    "y_mfe",
    "y_mae",
    "y_edge",
    "y_amplitude",
    "y_swing_pos",
    "y_barrier",
    "y_pivot_low_hold",
    "y_pivot_high_hold",
    "y_pivot_low_return",
    "y_pivot_high_return",
    "pivot",
    "label_scale",
    "atr_pct_label",
]


def _bars(n: int = 800, price_scale: float = 1.0, volume_scale: float = 1.0, seed: int = 7):
    """Synthetic OHLCV.

    Shadows are deliberately *asymmetric*. A fixture with ``high = close * 1.01``
    and ``low = close * 0.99`` makes the Chaikin multiplier identically zero, so
    CMF and its z-score degenerate to 0/0 and every invariance check on them
    compares noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    upper = np.abs(rng.normal(0.006, 0.004, n))
    lower = np.abs(rng.normal(0.006, 0.004, n))
    volume = rng.integers(1e5, 1e6, n).astype("float64")
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)) * price_scale,
            "high": close * (1 + upper) * price_scale,
            "low": close * (1 - lower) * price_scale,
            "close": close * price_scale,
            "volume": volume * volume_scale,
        },
        index=dates,
    )


def _panel(frame: pd.DataFrame, symbol: str = "AAA") -> pd.DataFrame:
    out = frame.reset_index(names="date")
    out.insert(0, "symbol", symbol)
    out["price"] = out["close"]
    return out


def _mismatches(a: pd.DataFrame, b: pd.DataFrame, columns=None) -> list[str]:
    """Columns whose finite values differ, ignoring rows missing from either."""
    bad = []
    for col in columns or a.columns:
        left = np.asarray(a[col], dtype="float64")
        right = np.asarray(b[col], dtype="float64")
        n = min(left.size, right.size)
        left, right = left[:n], right[:n]
        both = np.isfinite(left) & np.isfinite(right)
        if both.sum() and not np.allclose(left[both], right[both], rtol=1e-8, atol=1e-10):
            bad.append(col)
    return bad


# --------------------------------------------------------------------------
# leakage: nothing may read a bar it could not have seen
# --------------------------------------------------------------------------
def test_no_symbol_feature_reads_the_future():
    """Truncation test over every engineered column — the decisive leak check."""
    frame = _bars()
    full = compute_symbol_features(frame)
    truncated = compute_symbol_features(frame.iloc[:CUT])
    assert _mismatches(full.iloc[:CUT], truncated) == []


def test_no_market_feature_reads_the_future():
    bench = _bars().reset_index(names="date")
    full = compute_market_features(bench)
    truncated = compute_market_features(bench.iloc[:CUT])
    assert _mismatches(full.iloc[:CUT], truncated) == []


def test_no_relative_feature_reads_the_future():
    """Beta, correlation and residual strength all use rolling market joins."""
    frame = _bars()
    bench = _bars(seed=11).reset_index(names="date")
    market = compute_market_features(bench)["mkt_ret_1"]
    full = compute_relative_features(frame["close"], market)
    truncated = compute_relative_features(frame["close"].iloc[:CUT], market.iloc[:CUT])
    assert _mismatches(full.iloc[:CUT], truncated) == []


def test_labels_do_not_change_outside_their_lookahead():
    panel = _panel(_bars())
    full = build_labels(panel, horizon=HORIZON)
    truncated = build_labels(panel.iloc[:CUT], horizon=HORIZON)
    safe = CUT - HORIZON - HOLD_WINDOW
    assert _mismatches(full.iloc[:safe], truncated.iloc[:safe], SCALE_FREE_LABELS) == []


def test_label_lookahead_fits_inside_the_purge_gap():
    """The gap the fold builder inserts must cover the longest label lookahead."""
    panel = _panel(_bars())
    labels = build_labels(panel, horizon=HORIZON)
    # Excursion labels resolve H bars ahead; hold labels resolve W ahead.
    assert labels["y_swing_pos"].tail(HORIZON).isna().all()
    assert labels["y_swing_pos"].iloc[-HORIZON - 1] == labels["y_swing_pos"].iloc[-HORIZON - 1]
    assert labels["y_pivot_low_hold"].tail(HOLD_WINDOW).isna().all()
    assert labels["y_pivot_high_hold"].tail(HOLD_WINDOW).isna().all()

    purge_days = int(HORIZON * 1.6) + 5  # matches make_folds
    assert purge_days > HORIZON >= HOLD_WINDOW


def test_symbols_are_computed_independently():
    """One symbol's bars must not influence another's features or labels."""
    from predixlab_signals.features.core import build_features

    a = _panel(_bars(seed=1), "AAA")
    b = _panel(_bars(seed=2), "BBB")
    together = build_features(pd.concat([a, b], ignore_index=True))
    alone = build_features(a)

    cols = feature_columns(together)
    left = together[together["symbol"] == "AAA"].reset_index(drop=True)
    assert _mismatches(left, alone.reset_index(drop=True), cols) == []

    labels_together = build_labels(pd.concat([a, b], ignore_index=True), horizon=HORIZON)
    labels_alone = build_labels(a, horizon=HORIZON)
    left_labels = labels_together[labels_together["symbol"] == "AAA"].reset_index(drop=True)
    assert _mismatches(left_labels, labels_alone, SCALE_FREE_LABELS) == []


def test_future_labels_never_enter_the_feature_matrix():
    panel = _panel(_bars())
    from predixlab_signals.features.core import build_features

    merged = build_features(panel).merge(
        build_labels(panel, horizon=HORIZON), on=["symbol", "date"], how="inner"
    )
    cols = feature_columns(merged, include_market=True)
    for leaky in (
        "y_pivot_low_hold",
        "y_pivot_high_hold",
        "y_pivot_low_return",
        "y_pivot_high_return",
        "pivot",
        "label_scale",
    ):
        assert leaky not in cols, leaky


# --------------------------------------------------------------------------
# scale invariance: absolute price and share count must not matter
# --------------------------------------------------------------------------
def test_every_label_is_invariant_to_price_scale():
    """Collector prices are rebased per symbol; labels must ignore that."""
    base = build_labels(_panel(_bars()), horizon=HORIZON)
    scaled = build_labels(_panel(_bars(price_scale=1000.0)), horizon=HORIZON)
    assert _mismatches(base, scaled, SCALE_FREE_LABELS) == []


def test_every_label_is_invariant_to_volume_scale():
    base = build_labels(_panel(_bars()), horizon=HORIZON)
    scaled = build_labels(_panel(_bars(volume_scale=1000.0)), horizon=HORIZON)
    assert _mismatches(base, scaled, SCALE_FREE_LABELS) == []


def test_all_features_are_invariant_at_constant_dollar_volume():
    """Scale price up and share count down: nothing real changed, nothing may move."""
    base = compute_symbol_features(_bars())
    scaled = compute_symbol_features(_bars(price_scale=1e6, volume_scale=1e-6))
    assert _mismatches(base, scaled) == []


def test_only_raw_amihud_responds_to_dollar_volume():
    """Amihud is liquidity, so it legitimately tracks dollar volume — alone."""
    base = compute_symbol_features(_bars())
    scaled = compute_symbol_features(_bars(price_scale=1e6))
    assert _mismatches(base, scaled) == ["amihud"]


def test_self_normalised_amihud_is_scale_free():
    """The z-score must cancel the units.

    It did not, because a fixed epsilon floor inside the log clipped the ratio
    for high dollar-volume names — flattening the feature on 32% of AAPL's bars
    and 25% of NVDA's, and distorting the z-score that was supposed to be
    immune. The floor is gone; this pins it.
    """
    base = compute_symbol_features(_bars())
    scaled = compute_symbol_features(_bars(price_scale=1e6))
    left = base["amihud_sn"].to_numpy()
    right = scaled["amihud_sn"].to_numpy()
    both = np.isfinite(left) & np.isfinite(right)
    assert both.sum() > 100
    np.testing.assert_allclose(left[both], right[both], rtol=1e-8, atol=1e-10)


def test_amihud_has_no_artificial_floor():
    """Values must not pile up on a constant, whatever the dollar volume."""
    for scale in (1.0, 1e3, 1e6):
        amihud = compute_symbol_features(_bars(price_scale=scale))["amihud"].dropna()
        assert amihud.nunique() > len(amihud) * 0.9, f"saturated at price_scale={scale}"


@pytest.mark.parametrize("price_scale", [0.01, 1.0, 100.0])
def test_bounded_features_stay_bounded_at_any_price(price_scale):
    feats = compute_symbol_features(_bars(price_scale=price_scale)).dropna()
    for col in ("rsi_14", "stoch_k", "mfi_14", "close_loc", "pctile_60", "dpos_20"):
        assert feats[col].between(0.0, 1.0).all(), col


# --------------------------------------------------------------------------
# index membership: no bar may predate a symbol joining the universe
# --------------------------------------------------------------------------
def _membership_tree(tmp_path):
    """A tree where BBB joins the index halfway through its price history."""
    root = tmp_path / "data"
    normalized = root / "normalized_data" / "us_data"
    normalized.mkdir(parents=True)
    days = pd.bdate_range("2020-01-01", periods=600)
    (root / "calendar").mkdir(parents=True)
    (root / "calendar" / "us.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in days)
    )
    (root / "instruments").mkdir(parents=True)
    join = days[300].strftime("%Y-%m-%d")
    (root / "instruments" / "us.txt").write_text(
        f"AAA\t2020-01-01\t2099-12-31\nBBB\t{join}\t2099-12-31\n"
    )

    rng = np.random.default_rng(0)
    for symbol in ("AAA", "BBB", "CCC"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
        pd.DataFrame(
            {
                "date": days,
                "symbol": symbol,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(days), 1e6),
                "factor": np.ones(len(days)),
            }
        ).to_csv(normalized / f"{symbol}.csv", index=False)
    return root, days[300]


def test_bars_before_index_entry_are_excluded(tmp_path):
    """The feature tree keeps full history; *training* on it is a look-ahead.

    A symbol's pre-membership years exist only because it *later* grew enough to
    be indexed, so fitting on them uses knowledge unavailable at the time.

    The panel still carries those bars, because rolling features need real
    history to warm up on and a stock's own past prices leak nothing. The
    exclusion is applied once the features exist, so the guarantee is asserted
    on the modelling table rather than on the raw panel.
    """
    root, join_date = _membership_tree(tmp_path)
    from predixlab_signals.data.dataset import build_dataset
    from predixlab_signals.data.loader import load_panel

    panel = load_panel("us", min_history=1, data_root=root, members_only=True)
    bbb = panel[panel["symbol"] == "BBB"]
    assert bbb["date"].min() < join_date, "warm-up history must survive"
    assert not bbb.loc[bbb["date"] < join_date, "is_member"].any()
    assert bbb.loc[bbb["date"] >= join_date, "is_member"].all()
    assert panel.loc[panel["symbol"] == "AAA", "is_member"].all()

    data = build_dataset(
        market="us", min_history=1, data_root=root, members_only=True
    )
    if not data.empty and (data["symbol"] == "BBB").any():
        assert data[data["symbol"] == "BBB"]["date"].min() >= join_date
    assert "is_member" not in data.columns

    unfiltered = load_panel("us", min_history=1, data_root=root, members_only=False)
    assert unfiltered[unfiltered["symbol"] == "BBB"]["date"].min() < join_date


def test_recent_additions_keep_their_warm_up_history(tmp_path):
    """A name added late must not be dropped for having a short membership span.

    Truncating the price series at the join date used to fail ``min_history``
    for every recent addition — 34 current constituents, HOOD among them,
    vanished from the universe entirely.
    """
    root, join_date = _membership_tree(tmp_path)
    from predixlab_signals.data.loader import load_panel

    panel = load_panel("us", min_history=400, data_root=root, members_only=True)
    bbb = panel[panel["symbol"] == "BBB"]
    member_bars = int(bbb["is_member"].sum())

    assert not bbb.empty, "a recent addition must still load"
    assert member_bars < 400, "fixture must have a membership span below min_history"
    assert len(bbb) >= 400, "full history is what satisfies the warm-up"


def test_non_constituents_are_dropped(tmp_path):
    """CCC has price data but never appears in the instruments file."""
    root, _ = _membership_tree(tmp_path)
    from predixlab_signals.data.loader import load_panel

    filtered = load_panel("us", min_history=1, data_root=root, members_only=True)
    assert "CCC" not in set(filtered["symbol"])
    unfiltered = load_panel("us", min_history=1, data_root=root, members_only=False)
    assert "CCC" in set(unfiltered["symbol"])


def test_membership_mask_handles_multiple_spans(tmp_path):
    """Symbols that leave and rejoin must keep both spans and drop the gap."""
    from predixlab_signals.data.csv_data import membership_mask

    membership = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "start": pd.to_datetime(["2020-01-01", "2022-01-01"]),
            "end": pd.to_datetime(["2020-12-31", "2099-12-31"]),
        }
    )
    dates = pd.Series(pd.to_datetime(["2020-06-01", "2021-06-01", "2023-06-01"]))
    assert list(membership_mask("AAA", dates, membership)) == [True, False, True]
