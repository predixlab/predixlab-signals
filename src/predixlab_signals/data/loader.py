"""Panel loading: turn the collector's normalized CSV tree into an OHLCV panel.

The normalized tree is mounted directly by the nightly job, so nothing sits
between the data source and this package. See
:mod:`predixlab_signals.data.csv_data` for the on-disk shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from predixlab_signals.config import MIN_HISTORY, PATHS, Paths
from predixlab_signals.contracts import PANEL_CONTRACT, DataContractError
from predixlab_signals.data import csv_data
from predixlab_signals.data.csv_data import membership_mask

PANEL_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price",
    "raw_volume",
    "is_member",
]


class MembershipDataError(ValueError):
    """Raised when membership filtering was requested without valid spans."""


class _Tree:
    """One market's normalized CSV tree and the files that describe it."""

    def __init__(self, market: str, root: Path | None = None, paths: Paths = PATHS):
        self.market = market.lower()
        self.market_dir = (
            root / "normalized_data" / f"{self.market}_data"
            if root is not None else paths.normalized_dir(self.market)
        )
        self.calendar_path = (
            root / "calendar" / f"{self.market}.txt"
            if root is not None else paths.calendar_path(self.market)
        )
        self.instruments_path = (
            root / "instruments" / f"{self.market}.txt"
            if root is not None else paths.instruments_path(self.market)
        )

    def calendar(self) -> pd.DatetimeIndex | None:
        # Advisory only: each CSV carries its own dates, so a missing calendar
        # is not fatal.
        if self.calendar_path.exists():
            return csv_data.read_calendar(self.calendar_path)
        return None

    def symbols(self) -> list[str]:
        return csv_data.list_symbols(self.market_dir)

    def membership(self) -> pd.DataFrame:
        return csv_data.read_membership(self.instruments_path)

    def bars(
        self,
        symbol: str,
        calendar: pd.DatetimeIndex | None,
        history_sessions: int | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        return csv_data.read_symbol_on_calendar(
            self.market_dir,
            symbol,
            calendar,
            history_sessions=history_sessions,
            end=end,
        )


def load_panel(
    market: str = "us",
    symbols: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    min_history: int = MIN_HISTORY,
    members_only: bool = True,
    data_root: Path | None = None,
    paths: Paths = PATHS,
    history_sessions: int | None = None,
    on_read_error: Callable[[str, Exception], None] | None = None,
) -> pd.DataFrame:
    """Load a long-format panel of adjusted OHLCV bars.

    Two different concerns get conflated here, and keeping them apart is what
    lets a recent addition like HOOD be predicted at all.

    *Universe selection* is a genuine look-ahead: the tree stores full history
    for every name that was **ever** a constituent, so training on the years
    before a company was large enough to be indexed selects on an outcome.
    That must be excluded, and it is worth 5.3% of all bars.

    *Feature warm-up* is not. Computing HOOD's 250-day percentile today needs
    HOOD's prices from the past 250 days, which were public and known at the
    time; there is nothing to leak. Truncating the price series at the join date
    used to do both at once, which left every newly-added name with unusable
    features for its first year and then dropped it outright for failing
    ``min_history``. 34 current constituents disappeared that way.

    So the bars are returned in full and tagged with ``is_member``. Callers
    compute features over everything and drop non-member rows afterwards; with
    ``members_only`` the symbol must be a constituent at some point, and
    ``min_history`` applies to the *price history* available for warm-up rather
    than to the membership span.

    ``history_sessions`` bounds each symbol before feature construction. It is
    intended for live inference; research callers leave it unset so rolling
    features can warm on the complete history.

    Returns a frame sorted by ``(symbol, date)`` with a ``RangeIndex``.
    """
    tree = _Tree(market, data_root, paths)
    calendar = tree.calendar()
    universe = [s.upper() for s in symbols] if symbols else tree.symbols()

    membership = tree.membership() if members_only else None
    if membership is not None:
        required = {"symbol", "start", "end"}
        if membership.empty or not required <= set(membership.columns):
            raise MembershipDataError(
                f"members_only=True requires valid membership spans for {market!r}; "
                f"none were loaded from {tree.instruments_path}"
            )
        membership = membership.dropna(subset=["symbol", "start", "end"]).copy()
        membership = membership[membership["start"] <= membership["end"]]
        if membership.empty:
            raise MembershipDataError(
                f"members_only=True requires at least one valid membership span for "
                f"{market!r}; check {tree.instruments_path}"
            )
        universe = [s for s in universe if s in set(membership["symbol"])]

    frames: list[pd.DataFrame] = []
    duplicate_dates: dict[str, tuple[pd.Timestamp, ...]] = {}
    for symbol in universe:
        try:
            frame = tree.bars(
                symbol,
                calendar,
                history_sessions=history_sessions,
                end=end,
            )
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            if on_read_error is None:
                raise
            on_read_error(symbol, exc)
            continue
        if frame.empty:
            continue
        if frame.attrs.get("duplicate_dates"):
            duplicate_dates[symbol] = frame.attrs["duplicate_dates"]
        if on_read_error is not None:
            missing = sorted({"open", "high", "low", "close", "volume", "price"} - set(frame.columns))
            if missing:
                on_read_error(symbol, DataContractError(f"Missing OHLCV columns: {missing}"))
                continue
        frame = frame.reset_index()
        if len(frame) < min_history:
            continue
        if membership is not None:
            member = membership_mask(symbol, frame["date"], membership)
            if not member.any():
                continue
            frame["is_member"] = member
        else:
            frame["is_member"] = True
        frame.insert(0, "symbol", symbol)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = pd.concat(frames, ignore_index=True)
    if start is not None:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["date"] <= pd.Timestamp(end)]

    panel = panel.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    panel.attrs["duplicate_dates_by_symbol"] = duplicate_dates
    PANEL_CONTRACT.validate(panel)
    return panel


def load_benchmark(
    market: str = "us",
    benchmark: str = "^GSPC",
    data_root: Path | None = None,
    paths: Paths = PATHS,
    end: str | None = None,
    history_sessions: int | None = None,
) -> pd.DataFrame:
    """Load the benchmark series used for market-relative features."""
    tree = _Tree(market, data_root, paths)
    frame = tree.bars(
        benchmark,
        tree.calendar(),
        history_sessions=history_sessions,
        end=end,
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "close"])
    return frame.reset_index()[["date", "open", "high", "low", "close", "volume"]]


def market_dir(
    market: str = "us", data_root: Path | None = None, paths: Paths = PATHS
) -> Path:
    """Directory the panel for ``market`` is read from, for diagnostics."""
    return _Tree(market, data_root, paths).market_dir
