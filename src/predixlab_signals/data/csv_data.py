"""Reader for the normalized CSV market-data tree.

The data source provides one CSV per symbol under
``<PREDIXLAB_DATA>/normalized_data/<market>_data/<SYMBOL>.csv`` with the columns
``date, symbol, open, high, low, close, volume, factor``. Prices are
split/dividend adjusted and then rebased so each symbol's first close is
``1.0``, with ``factor`` holding the rebasing constant. The tradeable price is
therefore ``close / factor`` and the raw share count is ``volume * factor``.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_FIELDS = ("open", "high", "low", "close")
REQUIRED_FIELDS = ("date", "close")


def read_calendar(calendar_path: Path) -> pd.DatetimeIndex:
    """Load the trading calendar (one ``YYYY-MM-DD`` per line)."""
    if not Path(calendar_path).exists():
        raise FileNotFoundError(f"Calendar not found: {calendar_path}")
    stamps = Path(calendar_path).read_text().split()
    return pd.DatetimeIndex(pd.to_datetime(stamps, errors="coerce"), name="date").dropna()


def read_membership(instruments_path: Path) -> pd.DataFrame:
    """Index-membership spans from the instruments file.

    Each row is tab-separated ``(symbol, start, end)``. A symbol may appear
    several times if it left the index and rejoined; ``end`` of ``2099-12-31``
    means still a member.

    This matters because the tree carries *full history* for every symbol that
    was ever a constituent, including the years before it joined. Training on
    those bars is a look-ahead: the universe was chosen with knowledge of which
    companies would later become large enough to be indexed.
    """
    path = Path(instruments_path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "start", "end"])
    frame = pd.read_csv(path, sep="\t", header=None, names=["symbol", "start", "end"])
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["start"] = pd.to_datetime(frame["start"], errors="coerce")
    frame["end"] = pd.to_datetime(frame["end"], errors="coerce")
    return frame.dropna(subset=["start", "end"])


def membership_mask(
    symbol: str, dates: pd.Series, membership: pd.DataFrame
) -> np.ndarray:
    """Boolean mask of the bars on which ``symbol`` was an index constituent."""
    spans = membership[membership["symbol"] == symbol.upper()]
    if spans.empty:
        return np.zeros(len(dates), dtype=bool)
    dates = pd.DatetimeIndex(dates)
    mask = np.zeros(len(dates), dtype=bool)
    for start, end in zip(spans["start"], spans["end"]):
        mask |= (dates >= start) & (dates <= end)
    return mask


def list_symbols(normalized_dir: Path, include_indices: bool = False) -> list[str]:
    """Every symbol with a CSV in the normalized tree.

    Index instruments (``^GSPC`` and friends) are stored alongside the tradeable
    names but are not positions, so they are excluded unless asked for.
    """
    directory = Path(normalized_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Normalized data directory not found: {directory}")
    out = []
    for child in sorted(directory.glob("*.csv")):
        name = child.stem.upper()
        if not include_indices and name.startswith("^"):
            continue
        out.append(name)
    return out


def _read_csv_tail(
    path: Path,
    history_sessions: int,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Read only the trailing rows needed by live inference.

    Normal live files are ordered and unbounded, so retaining their final CSV
    records avoids materialising years of bars. An explicit historical end date
    still requires a streaming scan, but memory remains bounded to one chunk
    plus the requested tail.
    """
    if history_sessions <= 0:
        raise ValueError("history_sessions must be positive")

    if end is None:
        with path.open("rb") as handle:
            header = handle.readline()
            data_start = handle.tell()
            handle.seek(0, 2)
            position = handle.tell()
            chunks: list[bytes] = []
            line_breaks = 0

            while position > data_start and line_breaks <= history_sessions:
                size = min(64 * 1024, position - data_start)
                position -= size
                handle.seek(position)
                chunk = handle.read(size)
                chunks.append(chunk)
                line_breaks += chunk.count(b"\n")

        rows = b"".join(reversed(chunks)).splitlines()[-history_sessions:]
        return pd.read_csv(BytesIO(header + b"\n".join(rows)))

    end_stamp = pd.Timestamp(end)
    tail = pd.DataFrame()
    for chunk in pd.read_csv(path, chunksize=max(4096, history_sessions)):
        if "date" not in chunk.columns:
            return chunk.head(0)
        dates = pd.to_datetime(chunk["date"], errors="coerce")
        eligible = chunk[dates <= end_stamp]
        if not eligible.empty:
            tail = pd.concat([tail, eligible], ignore_index=True).tail(history_sessions)
    return tail


def read_symbol(
    normalized_dir: Path,
    symbol: str,
    history_sessions: int | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one instrument into a tidy, NaN-free OHLCV frame indexed by date.

    ``history_sessions`` activates the bounded live-inference reader. Research
    callers leave it unset and continue to receive the complete history.
    ``attrs["duplicate_dates"]`` records dates repeated before cleaning so daily
    validation can reject ambiguous sessions despite keep-last deduplication.
    """
    path = Path(normalized_dir) / f"{symbol.upper()}.csv"
    if not path.exists():
        return pd.DataFrame()

    frame = (
        _read_csv_tail(path, history_sessions, end)
        if history_sessions is not None
        else pd.read_csv(path)
    )
    if frame.empty or not set(REQUIRED_FIELDS).issubset(frame.columns):
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    duplicate_dates = tuple(
        frame.loc[frame["date"].duplicated(keep=False), "date"].dropna().drop_duplicates()
    )
    numeric_columns = [
        column for column in (*PRICE_FIELDS, "volume", "factor") if column in frame
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["date", "close"])
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if frame.empty:
        return pd.DataFrame()

    frame = frame.set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    # A handful of vendor rows carry zero/negative prices; they break every
    # ratio downstream, so drop them rather than let them poison the features.
    price_cols = [c for c in PRICE_FIELDS if c in frame.columns]
    if price_cols:
        frame = frame[(frame[price_cols] > 0).all(axis=1)]
    if frame.empty:
        return pd.DataFrame()

    if "factor" in frame.columns:
        factor = frame["factor"].replace(0.0, np.nan)
        frame["price"] = frame["close"] / factor
        frame["raw_volume"] = frame.get("volume", np.nan) * factor
    else:
        frame["factor"] = 1.0
        frame["price"] = frame["close"]
        frame["raw_volume"] = frame.get("volume", np.nan)

    keep = [c for c in ("open", "high", "low", "close", "volume", "factor", "price", "raw_volume")
            if c in frame.columns]
    frame = frame[keep].dropna(subset=["price"])
    frame.index.name = "date"
    frame.attrs["duplicate_dates"] = duplicate_dates
    return frame


def read_symbol_on_calendar(
    normalized_dir: Path,
    symbol: str,
    calendar: pd.DatetimeIndex | None,
    history_sessions: int | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read a symbol and, when a calendar is given, drop off-calendar bars.

    The source normally reindexes onto the calendar, but a stale CSV left over
    from an earlier universe can carry sessions the current calendar does not
    know about. Trimming here keeps every symbol on the same grid, which the
    market-relative features assume.
    """
    frame = read_symbol(
        normalized_dir,
        symbol,
        history_sessions=history_sessions,
        end=end,
    )
    if frame.empty or calendar is None or len(calendar) == 0:
        return frame
    return frame[frame.index.isin(calendar)]
