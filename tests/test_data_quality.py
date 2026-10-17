"""Daily completeness uses membership; freshness uses the previous session."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.data_quality import DataQualityError, DailyDataValidator, OHLCV_COLUMNS
from predixlab_signals.settings import Settings, load_settings

TARGET = pd.Timestamp("2026-09-08")
PREVIOUS = pd.Timestamp("2026-09-04")
SYMBOLS = [f"SYM{i:02d}" for i in range(20)]


def _panel() -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(SYMBOLS):
        for offset, date in enumerate((PREVIOUS, TARGET)):
            close = 10.0 + index + offset
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "price": close,
                "volume": 1000.0 + offset * 100,
                "is_member": True,
            })
    return pd.DataFrame(rows)


def _benchmark() -> pd.DataFrame:
    return _panel().query("symbol == 'SYM00'").drop(columns=["symbol", "is_member"])


def _membership() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": SYMBOLS,
        "start": pd.Timestamp("2020-01-01"),
        "end": pd.Timestamp("2099-12-31"),
    })


def _validator(**overrides) -> DailyDataValidator:
    values = {
        "market": "us",
        "target": str(TARGET.date()),
        "calendar": pd.DatetimeIndex([TARGET, PREVIOUS, PREVIOUS]),
        "membership": _membership(),
    }
    values.update(overrides)
    return DailyDataValidator(**values)


def _repeat_bars(panel: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    panel = panel.copy()
    for symbol in symbols:
        previous = panel.loc[
            (panel["date"] == PREVIOUS) & (panel["symbol"] == symbol), OHLCV_COLUMNS
        ]
        panel.loc[
            (panel["date"] == TARGET) & (panel["symbol"] == symbol), OHLCV_COLUMNS
        ] = previous.to_numpy()
    return panel


def test_healthy_data_uses_previous_trading_session_not_previous_calendar_day():
    validator = _validator()
    panel = validator.validate_panel(_panel(), _benchmark())
    result = validator.validate_dataset(panel)

    pd.testing.assert_frame_equal(result, _panel())
    assert validator.report["status"] == "passed"
    assert validator.report["previous_date"] == "2026-09-04"
    assert validator.report["valid_symbols"] == 20
    assert validator.report["excluded_symbols"] == []


@pytest.mark.parametrize("missing_count", [2, 20])
def test_missing_target_rows_cannot_shrink_the_denominator(missing_count):
    panel = _panel()
    panel = panel[
        ~((panel["date"] == TARGET) & panel["symbol"].isin(SYMBOLS[:missing_count]))
    ]
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())

    report = error.value.report
    assert report["status"] == "failed"
    assert report["expected_symbols"] == 20
    assert report["missing_symbols"] == SYMBOLS[:missing_count]
    assert report["bad_symbols"] == missing_count


def test_entirely_absent_symbols_still_count_as_missing():
    panel = _panel()
    panel = panel[~panel["symbol"].isin(SYMBOLS[:2])]
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())
    assert error.value.report["missing_symbols"] == SYMBOLS[:2]


@pytest.mark.parametrize("count", [2, 20])
def test_repeated_ohlcv_blocks_the_session(count):
    panel = _repeat_bars(_panel(), SYMBOLS[:count])
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())
    assert error.value.report["unchanged_symbols"] == SYMBOLS[:count]


@pytest.mark.parametrize("kind", ["missing", "unchanged", "invalid"])
def test_one_bad_symbol_at_the_limit_is_excluded_but_history_is_retained(kind):
    panel = _panel()
    target_row = (panel["date"] == TARGET) & (panel["symbol"] == SYMBOLS[0])
    if kind == "missing":
        panel = panel[~target_row]
    elif kind == "unchanged":
        panel = _repeat_bars(panel, SYMBOLS[:1])
    else:
        panel.loc[target_row, "high"] = 0.0

    validator = _validator()
    filtered = validator.validate_panel(panel, _benchmark())
    filtered = validator.validate_dataset(filtered)
    assert validator.report["bad_ratio"] == 0.05
    assert validator.report["excluded_symbols"] == SYMBOLS[:1]
    assert len(filtered[filtered["date"] == TARGET]) == 19
    assert len(filtered[filtered["date"] == PREVIOUS]) == 20


def test_different_rejection_reasons_share_one_error_budget():
    panel = _repeat_bars(_panel(), SYMBOLS[:1])
    panel = panel[~((panel["date"] == TARGET) & (panel["symbol"] == SYMBOLS[1]))]
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())
    assert error.value.report["bad_symbols"] == 2
    assert error.value.report["unchanged_symbols"] == SYMBOLS[:1]
    assert error.value.report["missing_symbols"] == SYMBOLS[1:2]


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", 0.0),
        ("close", -1.0),
        ("high", 1.0),
        ("low", 1e6),
        ("volume", 0.0),
        ("volume", -1.0),
        ("volume", np.nan),
        ("close", np.inf),
        ("price", 0.0),
        ("factor", -1.0),
    ],
)
def test_invalid_bars_are_rejected(column, value):
    panel = _panel()
    if column == "factor":
        panel[column] = 1.0
    target_rows = (panel["date"] == TARGET) & panel["symbol"].isin(SYMBOLS[:2])
    panel.loc[target_rows, column] = value
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())
    assert error.value.report["invalid_symbols"] == SYMBOLS[:2]


def test_equal_prices_with_updated_volume_are_not_stale():
    panel = _repeat_bars(_panel(), SYMBOLS)
    panel.loc[panel["date"] == TARGET, "volume"] += 1
    validator = _validator()
    validator.validate_panel(panel, _benchmark())
    assert validator.report["unchanged_symbols"] == []


def test_benchmark_can_have_zero_volume_when_prices_are_updated():
    validator = _validator()
    validator.validate_panel(_panel(), _benchmark().assign(volume=0.0))
    assert validator.report["benchmark_error"] is None


def test_missing_previous_session_cannot_disable_freshness_validation():
    panel = _panel()
    panel = panel[
        ~((panel["date"] == PREVIOUS) & panel["symbol"].isin(SYMBOLS[:2]))
    ]
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(panel, _benchmark())
    assert error.value.report["missing_previous_symbols"] == SYMBOLS[:2]


def test_duplicate_target_keys_are_invalid_not_extra_coverage():
    panel = _panel()
    duplicates = panel[(panel["date"] == TARGET) & panel["symbol"].isin(SYMBOLS[:2])]
    with pytest.raises(DataQualityError) as error:
        _validator().validate_panel(pd.concat([panel, duplicates]), _benchmark())
    assert error.value.report["invalid_symbols"] == SYMBOLS[:2]


@pytest.mark.parametrize("kind", ["missing", "unchanged", "invalid", "missing_previous"])
def test_bad_benchmark_is_fatal_even_with_healthy_constituents(kind):
    benchmark = _benchmark()
    if kind == "missing":
        benchmark = benchmark[benchmark["date"] != TARGET]
    elif kind == "missing_previous":
        benchmark = benchmark[benchmark["date"] != PREVIOUS]
    elif kind == "unchanged":
        benchmark.loc[benchmark["date"] == TARGET, OHLCV_COLUMNS] = benchmark.loc[
            benchmark["date"] == PREVIOUS, OHLCV_COLUMNS
        ].to_numpy()
    else:
        benchmark.loc[benchmark["date"] == TARGET, "close"] = np.nan
    with pytest.raises(DataQualityError, match=r"benchmark \^GSPC") as error:
        _validator().validate_panel(_panel(), benchmark)
    assert error.value.report["bad_symbols"] == 0
    assert error.value.report["benchmark_error"]


def test_current_membership_excludes_expired_future_and_benchmark_spans():
    membership = pd.concat([
        _membership(),
        pd.DataFrame({
            "symbol": ["EXPIRED", "FUTURE", "^GSPC"],
            "start": pd.to_datetime(["2020-01-01", "2026-09-09", "2020-01-01"]),
            "end": pd.to_datetime(["2026-09-04", "2099-12-31", "2099-12-31"]),
        }),
        _membership().iloc[:1],
    ])
    validator = _validator(membership=membership)
    validator.validate_panel(_panel(), _benchmark())
    assert validator.report["expected_symbols"] == 20


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"calendar": pd.DatetimeIndex([PREVIOUS])}, "not in"),
        ({"calendar": pd.DatetimeIndex([TARGET])}, "No previous trading"),
        ({"membership": _membership().iloc[0:0]}, "No active constituent"),
    ],
)
def test_metadata_cannot_silently_disable_validation(overrides, message):
    with pytest.raises(DataQualityError, match=message):
        _validator(**overrides)


def test_warmup_losses_are_added_to_the_same_budget():
    validator = _validator()
    panel = validator.validate_panel(_repeat_bars(_panel(), SYMBOLS[:1]), _benchmark())
    panel = panel[~((panel["date"] == TARGET) & (panel["symbol"] == SYMBOLS[1]))]
    with pytest.raises(DataQualityError) as error:
        validator.validate_dataset(panel)
    assert error.value.report["unusable_symbols"] == SYMBOLS[1:2]
    assert error.value.report["bad_symbols"] == 2


def test_nonfinite_core_features_are_excluded_before_scoring():
    validator = _validator()
    panel = validator.validate_panel(_panel(), _benchmark()).assign(rsi_14=50.0)
    panel.loc[(panel["date"] == TARGET) & (panel["symbol"] == SYMBOLS[0]), "rsi_14"] = np.inf
    result = validator.validate_dataset(panel, required_features=["rsi_14"])
    assert validator.report["unusable_symbols"] == SYMBOLS[:1]
    assert len(result[result["date"] == TARGET]) == 19


def test_threshold_is_configurable_but_cannot_allow_all_bad_data(monkeypatch):
    monkeypatch.setenv("PIVOT_DATA_MAX_BAD_RATIO", "0.1")
    assert load_settings().max_bad_data_ratio == 0.1
    validator = _validator(max_bad_ratio=0.1)
    validator.validate_panel(_repeat_bars(_panel(), SYMBOLS[:2]), _benchmark())
    assert validator.report["bad_ratio"] == 0.1


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0, float("nan"), float("inf")])
def test_invalid_tolerances_fail_closed(value):
    with pytest.raises(ValueError, match="PIVOT_DATA_MAX_BAD_RATIO"):
        Settings(max_bad_data_ratio=value)
    with pytest.raises(ValueError, match="max_bad_ratio"):
        _validator(max_bad_ratio=value)


def test_nonnumeric_environment_tolerance_is_reported(monkeypatch):
    monkeypatch.setenv("PIVOT_DATA_MAX_BAD_RATIO", "five")
    with pytest.raises(ValueError, match="PIVOT_DATA_MAX_BAD_RATIO must be a number"):
        load_settings()
