"""Fail-closed quality checks for the session used by production inference."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from predixlab_signals.config import BENCHMARKS

DEFAULT_MAX_BAD_DATA_RATIO = 0.05
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataQualityError(ValueError):
    """A rejected daily input, including the report persisted in job status."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _valid_bars(frame: pd.DataFrame, *, allow_zero_volume: bool = False) -> pd.Series:
    numbers = frame.reindex(columns=OHLCV_COLUMNS).apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(
        np.isfinite(numbers.to_numpy(dtype=float)).all(axis=1),
        index=frame.index,
    )
    valid &= (numbers[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= numbers["volume"] >= 0 if allow_zero_volume else numbers["volume"] > 0
    valid &= numbers["high"] >= numbers[["open", "low", "close"]].max(axis=1)
    valid &= numbers["low"] <= numbers[["open", "high", "close"]].min(axis=1)
    for column in ("factor", "price", "raw_volume"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid &= np.isfinite(values) & (
                values >= 0 if column == "raw_volume" else values > 0
            )
    return valid


def _unchanged_bars(current: pd.DataFrame, previous: pd.DataFrame) -> pd.Series:
    current_values = current.reindex(columns=OHLCV_COLUMNS).apply(pd.to_numeric, errors="coerce")
    previous_values = previous.reindex(columns=OHLCV_COLUMNS).apply(pd.to_numeric, errors="coerce")
    return pd.Series(
        np.isclose(
            current_values.to_numpy(dtype=float),
            previous_values.to_numpy(dtype=float),
            rtol=1e-9,
            atol=0.0,
            equal_nan=False,
        ).all(axis=1),
        index=current.index,
    )


class DailyDataValidator:
    """Validate raw bars, then confirm usable coverage after feature warm-up.

    The denominator comes from target-date membership, never from the files or
    rows that happened to arrive. All rejection reasons share one error budget.
    """

    def __init__(
        self,
        market: str,
        target: str,
        calendar: pd.DatetimeIndex,
        membership: pd.DataFrame,
        max_bad_ratio: float = DEFAULT_MAX_BAD_DATA_RATIO,
    ):
        if not 0 <= max_bad_ratio < 1:
            raise ValueError("max_bad_ratio must be in [0, 1)")
        self.market = market.lower()
        self.target = pd.Timestamp(target).normalize()
        self.previous: pd.Timestamp | None = None
        self.max_bad_ratio = max_bad_ratio
        self.expected: set[str] = set()
        self.present: set[str] = set()
        self.missing: set[str] = set()
        self.invalid: set[str] = set()
        self.unchanged: set[str] = set()
        self.missing_previous: set[str] = set()
        self.unusable: set[str] = set()
        self.read_errors: dict[str, str] = {}
        self.benchmark = BENCHMARKS.get(self.market, "^GSPC")
        self.benchmark_error: str | None = None
        self.status = "checking"

        calendar = calendar.dropna().sort_values().unique()
        if pd.isna(self.target) or self.target not in calendar:
            self._fail(f"Target date {target} is not in the {self.market} trading calendar")
        prior = calendar[calendar < self.target]
        if prior.empty:
            self._fail(f"No previous trading session in the calendar before {target}")
        self.previous = prior[-1]

        if not {"symbol", "start", "end"} <= set(membership.columns):
            self._fail("Daily validation requires valid membership spans")
        active = membership[
            (membership["start"] <= self.target) & (membership["end"] >= self.target)
        ]
        self.expected = {
            str(symbol).strip().upper()
            for symbol in active["symbol"].dropna()
            if str(symbol).strip()
        }
        self.expected = {
            symbol for symbol in self.expected
            if not symbol.startswith("^") and symbol != self.benchmark
        }
        if not self.expected:
            self._fail(f"No active constituent membership for {self.market} on {target}")

    @property
    def bad_symbols(self) -> set[str]:
        return (
            self.missing | self.invalid | self.unchanged
            | self.missing_previous | self.unusable | set(self.read_errors)
        ) & self.expected

    @property
    def report(self) -> dict[str, Any]:
        expected = len(self.expected)
        bad = len(self.bad_symbols)
        return {
            "status": self.status,
            "market": self.market,
            "date": str(self.target.date()),
            "previous_date": str(self.previous.date()) if self.previous is not None else None,
            "expected_symbols": expected,
            "present_symbols": len(self.present),
            "valid_symbols": expected - bad,
            "bad_symbols": bad,
            "bad_ratio": bad / expected if expected else None,
            "max_bad_ratio": self.max_bad_ratio,
            "missing_symbols": sorted(self.missing),
            "invalid_symbols": sorted(self.invalid),
            "unchanged_symbols": sorted(self.unchanged),
            "missing_previous_symbols": sorted(self.missing_previous),
            "unusable_symbols": sorted(self.unusable),
            "excluded_symbols": sorted(self.bad_symbols),
            "read_errors": dict(self.read_errors),
            "benchmark": self.benchmark,
            "benchmark_error": self.benchmark_error,
        }

    def _fail(self, reason: str) -> None:
        self.status = "failed"
        raise DataQualityError(
            f"Daily data validation failed for {self.market} {self.target.date()}: {reason}",
            self.report,
        )

    def record_read_error(self, symbol: str, error: Exception) -> None:
        """Account for an unreadable symbol without hiding a partial collection."""
        self.read_errors[symbol] = str(error)
        logging.getLogger(__name__).warning("Excluding unreadable %s data: %s", symbol, error)

    def _enforce(self) -> None:
        bad = len(self.bad_symbols)
        expected = len(self.expected)
        reasons = []
        if bad / expected > self.max_bad_ratio:
            reasons.append(
                f"{bad}/{expected} constituents are missing, invalid, stale, or unusable "
                f"({bad / expected:.1%}; maximum {self.max_bad_ratio:.1%}); "
                f"missing={len(self.missing)}, invalid={len(self.invalid)}, "
                f"unchanged={len(self.unchanged)}, "
                f"missing_previous={len(self.missing_previous)}, "
                f"unusable={len(self.unusable)}; "
                f"examples={', '.join(sorted(self.bad_symbols)[:10])}"
            )
        if self.benchmark_error:
            reasons.append(f"benchmark {self.benchmark}: {self.benchmark_error}")
        if reasons:
            self._fail("; ".join(reasons))

    def _check_benchmark(self, benchmark: pd.DataFrame) -> str | None:
        if "date" not in benchmark:
            return "missing target-session data"
        duplicate_dates = benchmark.attrs.get("duplicate_dates", ())
        if self.target in duplicate_dates:
            return "duplicate target-session records"
        if self.previous in duplicate_dates:
            return "duplicate previous-session records"
        current = benchmark[benchmark["date"] == self.target]
        previous = benchmark[benchmark["date"] == self.previous]
        if len(current) != 1 or not _valid_bars(current, allow_zero_volume=True).all():
            return "missing or invalid target-session OHLCV"
        if len(previous) != 1 or not _valid_bars(previous, allow_zero_volume=True).all():
            return "missing or invalid previous-session OHLCV"
        if _unchanged_bars(
            current.reset_index(drop=True), previous.reset_index(drop=True)
        ).all():
            return "OHLCV is unchanged from the previous trading session"
        return None

    def validate_panel(self, panel: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        """Reject a broken session before any feature or prediction calculation."""
        if not {"symbol", "date"} <= set(panel.columns):
            self._fail("Price panel is missing symbol/date columns")
        current = panel[panel["date"] == self.target]
        previous = panel[panel["date"] == self.previous]
        self.present = set(current["symbol"]) & self.expected
        self.missing = self.expected - self.present
        duplicate_symbols = set(current.loc[current["symbol"].duplicated(False), "symbol"])
        duplicate_previous = set(previous.loc[previous["symbol"].duplicated(False), "symbol"])
        duplicate_dates = panel.attrs.get("duplicate_dates_by_symbol", {})
        duplicate_symbols.update(
            symbol for symbol, dates in duplicate_dates.items() if self.target in dates
        )
        duplicate_previous.update(
            symbol for symbol, dates in duplicate_dates.items() if self.previous in dates
        )
        current = (
            current.drop_duplicates("symbol", keep="last")
            .set_index("symbol").reindex(sorted(self.expected))
        )
        previous = (
            previous.drop_duplicates("symbol", keep="last")
            .set_index("symbol").reindex(current.index)
        )
        valid = _valid_bars(current)
        previous_valid = _valid_bars(previous)
        self.invalid = (set(current.index[~valid]) | duplicate_symbols) & self.present
        candidates = self.present - self.invalid
        self.missing_previous = (
            set(previous.index[~previous_valid]) | duplicate_previous
        ) & candidates
        self.unchanged = (
            set(current.index[_unchanged_bars(current, previous)])
            & (candidates - self.missing_previous)
        )
        self.benchmark_error = self._check_benchmark(benchmark)
        self._enforce()
        # Keep historical bars for temporal-rule warm-up, but never let rejected
        # target rows influence features, ranks, or either model's daily output.
        return panel[
            ~((panel["date"] == self.target) & panel["symbol"].isin(self.bad_symbols))
        ].copy()

    def validate_dataset(
        self, data: pd.DataFrame, required_features: Sequence[str] = ()
    ) -> pd.DataFrame:
        """Require the same healthy universe to survive feature construction."""
        current = data[data["date"] == self.target]
        valid = _valid_bars(current)
        if required_features:
            features = current[list(required_features)].apply(pd.to_numeric, errors="coerce")
            valid &= np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
        valid &= ~current["symbol"].duplicated(keep=False)
        usable = set(current.loc[valid, "symbol"])
        self.unusable = self.expected - self.bad_symbols - usable
        self._enforce()
        self.status = "passed"
        return data[
            ~((data["date"] == self.target) & data["symbol"].isin(self.bad_symbols))
        ].copy()
