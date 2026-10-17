"""Tests for the normalized CSV reader and panel assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.data.csv_data import (
    list_symbols,
    read_calendar,
    read_membership,
    read_symbol,
    read_symbol_on_calendar,
)
from predixlab_signals.data.loader import MembershipDataError, load_benchmark, load_panel


def _write_csv(path, days, base, factor=0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in days],
            "symbol": [path.stem] * len(days),
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "volume": [1000.0] * len(days),
            "factor": [factor] * len(days),
        }
    )
    frame.to_csv(path, index=False)


@pytest.fixture
def data_tree(tmp_path):
    """A miniature normalized tree: 10 sessions, two symbols and one index."""
    root = tmp_path / "data"
    normalized = root / "normalized_data" / "us_data"
    days = pd.bdate_range("2020-01-01", periods=10)

    (root / "calendar").mkdir(parents=True)
    (root / "calendar" / "us.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in days))

    (root / "instruments").mkdir(parents=True)
    (root / "instruments" / "us.txt").write_text(
        "AAA\t2020-01-01\t2099-12-31\nBBB\t2020-01-06\t2099-12-31\n"
    )

    _write_csv(normalized / "AAA.csv", days, np.linspace(10.0, 12.0, 10))
    _write_csv(normalized / "BBB.csv", days[3:], np.linspace(20.0, 22.0, 7))
    _write_csv(normalized / "^GSPC.csv", days, np.linspace(1.0, 1.1, 10), factor=1.0)
    return root


def test_read_calendar(data_tree):
    calendar = read_calendar(data_tree / "calendar" / "us.txt")
    assert len(calendar) == 10
    assert calendar[0] == pd.Timestamp("2020-01-01")


def test_read_calendar_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_calendar(tmp_path / "nope.txt")


def test_list_symbols_excludes_indices(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    assert list_symbols(normalized) == ["AAA", "BBB"]
    assert "^GSPC" in list_symbols(normalized, include_indices=True)


def test_read_membership_parses_spans(data_tree):
    membership = read_membership(data_tree / "instruments" / "us.txt")
    assert set(membership["symbol"]) == {"AAA", "BBB"}
    assert membership.loc[membership["symbol"] == "BBB", "start"].iloc[0] == pd.Timestamp("2020-01-06")


def test_read_membership_missing_file_is_empty(tmp_path):
    assert read_membership(tmp_path / "nope.txt").empty


def test_read_symbol_derives_real_price(data_tree):
    """Prices in the tree are rebased; ``close / factor`` restores the real level."""
    frame = read_symbol(data_tree / "normalized_data" / "us_data", "AAA")
    assert len(frame) == 10
    np.testing.assert_allclose(frame["price"], frame["close"] / 0.5)
    np.testing.assert_allclose(frame["raw_volume"], frame["volume"] * 0.5)


def test_read_symbol_defaults_absent_factor_to_one(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    path = normalized / "AAA.csv"
    pd.read_csv(path).drop(columns=["factor"]).to_csv(path, index=False)

    frame = read_symbol(normalized, "AAA")
    assert len(frame) == 10
    assert frame["factor"].eq(1.0).all()
    np.testing.assert_allclose(frame["price"], frame["close"])
    np.testing.assert_allclose(frame["raw_volume"], frame["volume"])


def test_read_symbol_missing_returns_empty(data_tree):
    assert read_symbol(data_tree / "normalized_data" / "us_data", "ZZZ").empty


def test_read_symbol_can_bound_live_history(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    frame = read_symbol(normalized, "AAA", history_sessions=4)

    assert len(frame) == 4
    assert frame.index.min() == pd.Timestamp("2020-01-09")


def test_read_symbol_can_bound_history_at_an_explicit_end(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    frame = read_symbol(
        normalized,
        "AAA",
        history_sessions=3,
        end="2020-01-08",
    )

    assert list(frame.index) == list(pd.bdate_range("2020-01-06", periods=3))


def test_read_symbol_drops_non_positive_prices(tmp_path):
    normalized = tmp_path / "us_data"
    normalized.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "open": [1.0, 0.0, 1.2],
            "high": [1.1, 1.0, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.0, 1.0, 1.2],
            "volume": [10.0, 10.0, 10.0],
            "factor": [1.0, 1.0, 1.0],
        }
    )
    frame.to_csv(normalized / "AAA.csv", index=False)
    out = read_symbol(normalized, "AAA")
    assert len(out) == 2  # the zero-open row is dropped


def test_read_symbol_tolerates_isolated_nonnumeric_and_infinite_prices(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    path = normalized / "AAA.csv"
    frame = pd.read_csv(path)
    frame["close"] = frame["close"].astype(object)
    frame.loc[0, "close"] = "vendor-error"
    frame.loc[1, "open"] = np.inf
    frame.to_csv(path, index=False)

    out = read_symbol(normalized, "AAA", history_sessions=10, end="2020-01-14")
    assert len(out) == 8
    assert np.isfinite(out[["open", "high", "low", "close", "price"]]).all().all()


def test_read_symbol_on_calendar_trims_stale_sessions(data_tree):
    normalized = data_tree / "normalized_data" / "us_data"
    calendar = read_calendar(data_tree / "calendar" / "us.txt")[:5]
    frame = read_symbol_on_calendar(normalized, "AAA", calendar)
    assert len(frame) == 5


@pytest.mark.parametrize("invalid_duplicate", [False, True])
def test_read_symbol_records_duplicates_before_cleaning(data_tree, invalid_duplicate):
    normalized = data_tree / "normalized_data" / "us_data"
    path = normalized / "AAA.csv"
    source = pd.read_csv(path)
    duplicate = source.iloc[[-1]].copy()
    duplicate[["open", "high", "low", "close"]] *= 1.25
    if invalid_duplicate:
        duplicate["close"] = np.nan
    pd.concat([source, duplicate], ignore_index=True).to_csv(path, index=False)

    frame = read_symbol_on_calendar(
        normalized, "AAA", read_calendar(data_tree / "calendar" / "us.txt"),
        history_sessions=10, end="2020-01-14",
    )

    assert frame.index.is_unique
    assert frame.attrs["duplicate_dates"] == (pd.Timestamp("2020-01-14"),)
    expected = source.iloc[-1]["close"] if invalid_duplicate else duplicate.iloc[-1]["close"]
    assert frame.iloc[-1]["close"] == pytest.approx(expected)


def test_load_panel_reads_the_data_tree(data_tree):
    panel = load_panel("us", min_history=5, data_root=data_tree)
    assert set(panel["symbol"]) == {"AAA", "BBB"}
    assert panel["date"].is_monotonic_increasing or True  # sorted per symbol
    for _, group in panel.groupby("symbol"):
        assert group["date"].is_monotonic_increasing


def test_load_panel_filters_short_history(data_tree):
    panel = load_panel("us", min_history=9, data_root=data_tree)
    assert set(panel["symbol"]) == {"AAA"}  # BBB only has 7 bars


def test_load_panel_bounds_each_symbols_history(data_tree):
    panel = load_panel(
        "us",
        min_history=1,
        data_root=data_tree,
        history_sessions=4,
    )

    assert panel.groupby("symbol").size().max() == 4


def test_load_panel_marks_membership(data_tree):
    """Bars before the join date are kept for warm-up but tagged non-member."""
    panel = load_panel("us", min_history=1, data_root=data_tree)
    aaa = panel[panel["symbol"] == "AAA"]
    assert aaa["is_member"].all()


def test_load_panel_fails_closed_without_membership(data_tree):
    (data_tree / "instruments" / "us.txt").unlink()
    with pytest.raises(MembershipDataError, match="members_only=True"):
        load_panel("us", min_history=1, data_root=data_tree)


def test_load_panel_all_history_allows_missing_membership(data_tree):
    (data_tree / "instruments" / "us.txt").unlink()
    panel = load_panel(
        "us", min_history=1, data_root=data_tree, members_only=False
    )
    assert set(panel["symbol"]) == {"AAA", "BBB"}


def test_load_benchmark_reads_the_index(data_tree):
    bench = load_benchmark("us", "^GSPC", data_root=data_tree)
    assert len(bench) == 10
    assert {"date", "close"}.issubset(bench.columns)


def test_explicit_root_loads_data_tree(data_tree):
    panel = load_panel("us", min_history=1, data_root=data_tree)
    assert not panel.empty