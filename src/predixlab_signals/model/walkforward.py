"""Purged, embargoed walk-forward validation.

Two leaks make most stock-model backtests worthless, and both are structural
rather than accidental:

1. **Label overlap.** A label at bar ``t`` looks ``H`` bars into the future, so
   the last ``H`` training bars overlap the first test bars. Training on them
   lets the model see its own test period. We therefore drop (*purge*) the final
   ``H`` bars of every training block and add a further *embargo* gap.
2. **Cross-sectional contamination.** Splitting rows randomly puts the same
   calendar day in both train and test; with ~700 correlated names that is
   almost pure leakage. Every split here is by **date**, never by row.

Folds walk forward in time with a fixed test span, so the reported numbers are
what a model retrained on a schedule would actually have produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"fold{self.index:02d} "
            f"train {self.train_start.date()}..{self.train_end.date()} | "
            f"valid {self.valid_start.date()}..{self.valid_end.date()} | "
            f"test {self.test_start.date()}..{self.test_end.date()}"
        )


def make_folds(
    dates: pd.Series,
    horizon: int,
    n_folds: int = 5,
    test_months: int = 12,
    valid_months: int = 6,
    min_train_months: int = 36,
    embargo_days: int = 5,
    expanding: bool = True,
) -> list[Fold]:
    """Build walk-forward folds over the dataset's calendar span.

    The final fold always ends at the last available date; earlier folds step
    back by ``test_months`` each. ``expanding=True`` grows the training window
    fold over fold (more data, one regime mix); ``False`` keeps it a fixed-width
    rolling window (less data, fresher regime).
    """
    unique = pd.DatetimeIndex(pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values())
    if unique.empty:
        return []

    first, last = unique[0], unique[-1]
    purge = pd.Timedelta(days=int(horizon * 1.6) + embargo_days)
    test_span = pd.DateOffset(months=test_months)
    valid_span = pd.DateOffset(months=valid_months)
    min_train = pd.DateOffset(months=min_train_months)

    folds: list[Fold] = []
    test_end = last
    for i in range(n_folds):
        test_start = test_end - test_span
        valid_end = test_start - purge
        valid_start = valid_end - valid_span
        train_end = valid_start - purge
        train_start = first if expanding else train_end - pd.DateOffset(months=min_train_months)

        if train_end - min_train < first and train_end < first + min_train:
            break
        if train_start >= train_end or valid_start >= valid_end:
            break

        folds.append(
            Fold(
                index=n_folds - i,
                train_start=max(train_start, first),
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        # Fold slices are inclusive at both ends. Step back one calendar day so
        # the boundary session is predicted by exactly one model.
        test_end = test_start - pd.Timedelta(days=1)

    folds.sort(key=lambda f: f.test_start)
    for n, fold in enumerate(folds, start=1):
        object.__setattr__(fold, "index", n)
    return folds


def slice_fold(data: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into ``(train, valid, test)`` frames for one fold."""
    d = data["date"]
    train = data[(d >= fold.train_start) & (d <= fold.train_end)]
    valid = data[(d >= fold.valid_start) & (d <= fold.valid_end)]
    test = data[(d >= fold.test_start) & (d <= fold.test_end)]
    return train, valid, test


def assert_no_leakage(fold: Fold, horizon: int) -> None:
    """Fail loudly if a fold's purge gap cannot cover the label horizon."""
    gap_valid = (fold.valid_start - fold.train_end).days
    gap_test = (fold.test_start - fold.valid_end).days
    needed = int(horizon * 1.4)
    if gap_valid < needed or gap_test < needed:
        raise ValueError(
            f"{fold.describe()}: purge gap too small "
            f"(train->valid {gap_valid}d, valid->test {gap_test}d, need >= {needed}d)"
        )


def date_blocks(dates: pd.Series, n: int) -> list[np.ndarray]:
    """Split unique dates into ``n`` contiguous blocks (helper for diagnostics)."""
    unique = np.sort(pd.to_datetime(pd.Series(dates)).drop_duplicates().to_numpy())
    return [b for b in np.array_split(unique, n) if b.size]
