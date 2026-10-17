"""Tests for the pick-only accuracy report."""

from __future__ import annotations

import pandas as pd
import pytest

from predixlab_signals.report import render_pick_report, summarize_pick_accuracy, write_pick_report


def _picks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "event": ["BUY_TURN", "SELL_TURN", "TREND_BUY", "BUY_TURN"],
            "direction": ["long", "short", "long", "long"],
            "signal_return": [0.05, -0.02, 0.01, None],
            "pivot_hit": [True, False, None, True],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-05",
                    "2025-01-07",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-06",
                    "2025-01-07",
                ]
            ),
            "symbol": ["AAA"] * 4 + ["BBB"] * 4,
            "price": [100.0, 102.0, 101.0, 104.0, 50.0, 49.0, 51.0, 52.0],
        }
    )


def test_accuracy_uses_resolved_pick_returns_only():
    report = summarize_pick_accuracy(_picks())

    assert report["overall"] == {
        "signals": 4,
        "evaluated": 3,
        "pending": 1,
        "wins": 2,
        "accuracy": pytest.approx(2 / 3),
        "mean_signal_return": pytest.approx(0.04 / 3),
    }


def test_report_breaks_accuracy_down_by_symbol_and_event():
    report = summarize_pick_accuracy(_picks())

    assert report["by_symbol"]["AAA"]["accuracy"] == pytest.approx(0.5)
    assert report["by_symbol"]["BBB"]["evaluated"] == 1
    assert report["by_event"]["TREND_BUY"]["accuracy"] == pytest.approx(1.0)
    assert report["by_event"]["TREND_BUY"]["accuracy_kind"] == "directional"


def test_base_breakout_uses_directional_accuracy():
    base = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-08")],
            "symbol": ["CCC"],
            "event": ["BASE_BREAKOUT"],
            "direction": ["long"],
            "signal_return": [0.03],
            "pivot_hit": [None],
        }
    )

    report = summarize_pick_accuracy(pd.concat([_picks(), base], ignore_index=True))

    assert report["by_event"]["BASE_BREAKOUT"]["accuracy"] == pytest.approx(1.0)
    assert report["by_event"]["BASE_BREAKOUT"]["accuracy_kind"] == "directional"


def test_pick_type_accuracy_excludes_reference_turns():
    picks = pd.concat(
        [
            _picks(),
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2025-01-08")],
                    "symbol": ["CCC"],
                    "event": ["TOP_TURN"],
                    "direction": ["short"],
                    "signal_return": [-0.10],
                    "pivot_hit": [True],
                }
            ),
        ],
        ignore_index=True,
    )

    report = summarize_pick_accuracy(picks)

    assert "TOP_TURN" not in report["by_event"]
    assert set(report["by_event"]) == {"BUY_TURN", "SELL_TURN", "TREND_BUY"}


def test_aggregate_accuracy_excludes_reference_turns():
    reference = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-08", "2025-01-09"]),
            "symbol": ["CCC", "DDD"],
            "event": ["BOTTOM_TURN", "TOP_TURN"],
            "direction": ["long", "short"],
            "signal_return": [0.50, 0.50],
            "pivot_hit": [True, True],
        }
    )

    report = summarize_pick_accuracy(pd.concat([_picks(), reference], ignore_index=True))

    assert report["overall"]["signals"] == 4
    assert report["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert report["pivot_accuracy"]["evaluated"] == 3
    assert set(report["pivot_accuracy"]["by_event"]) == {"BUY_TURN", "SELL_TURN"}
    assert set(report["by_event"]) == {"BUY_TURN", "SELL_TURN", "TREND_BUY"}


def test_pivot_accuracy_excludes_non_reversal_rows():
    report = summarize_pick_accuracy(_picks())

    assert report["top_accuracy"] == {
        "evaluated": 1,
        "hits": 0,
        "accuracy": pytest.approx(0.0),
    }
    assert report["pivot_accuracy"] == {
        "evaluated": 3,
        "hits": 2,
        "accuracy": pytest.approx(2 / 3),
        "by_event": {
            "BUY_TURN": {
                "evaluated": 2,
                "hits": 2,
                "accuracy": pytest.approx(1.0),
            },
            "SELL_TURN": {
                "evaluated": 1,
                "hits": 0,
                "accuracy": pytest.approx(0.0),
            },
        },
    }


def test_missing_pick_columns_are_rejected():
    with pytest.raises(ValueError, match="signal_return"):
        summarize_pick_accuracy(_picks().drop(columns="signal_return"))


def test_html_contains_interactive_stock_view_and_pick_data():
    html = render_pick_report(
        _picks(), _prices(), market="us", horizon=10, tag="test"
    )

    assert 'id="symbol-search"' in html
    assert 'list="symbol-options"' in html
    assert "function matchSymbol(query)" in html
    assert 'id="signal-chart"' in html
    assert 'class: "price-line"' in html
    assert 'class: "signal-marker"' in html
    assert 'class: "hover-guide"' in html
    assert 'class: "hover-tooltip"' in html
    assert 'class: "chart-overlay"' in html
    assert 'overlay.addEventListener("pointermove"' in html
    assert "Date: ${point.date}" in html
    assert "Price: ${point.price.toFixed(2)}" in html
    assert "Signal: ${signalLabel}" in html
    assert '"symbol":"AAA"' in html
    assert '"price_history":{"AAA":[["2025-01-02",100.0]' in html
    assert '"2025-01-05",101.0' in html
    assert "BOTTOM_TURN and TOP_TURN are chart-only reference detections" in html
    assert 'id="total-events"' in html
    assert 'id="pivot-accuracy"' in html
    assert 'item.accuracy_kind === "pivot"' in html
    assert (
        'const actionableEvents = new Set(["BUY_TURN", "SELL_TURN", "TREND_BUY", '
        '"BASE_BREAKOUT"])' in html
    )
    assert "const rows = chartRows.filter(row => actionableEvents.has(row.event))" in html
    assert '"var(--event-bottom-turn)"' in html
    assert 'class="bottom-turn">L: Bottom detection' in html
    assert 'class="base-breakout">R: Base breakout' in html


def test_writer_persists_html_and_its_exact_pick_frame(tmp_path):
    result = write_pick_report(
        _picks(),
        _prices(),
        tmp_path / "report.html",
        market="us",
        horizon=10,
        tag="test",
    )

    assert result.html_path.exists()
    assert result.picks_path.exists()
    stored = pd.read_parquet(result.picks_path)
    pd.testing.assert_frame_equal(stored, _picks())