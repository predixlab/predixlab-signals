"""Runtime contracts for DataFrame and persistence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import pandas as pd


class DataContractError(ValueError):
    """Raised when a boundary receives a structurally invalid frame."""


@dataclass(frozen=True)
class FrameContract:
    name: str
    required: frozenset[str]

    def validate(self, frame: pd.DataFrame, *, allow_empty: bool = True) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise DataContractError(f"{self.name} must be a pandas DataFrame")
        missing = sorted(self.required - set(frame.columns))
        if missing:
            raise DataContractError(f"{self.name} is missing required columns: {missing}")
        if not allow_empty and frame.empty:
            raise DataContractError(f"{self.name} must not be empty")


PANEL_CONTRACT = FrameContract(
    "panel",
    frozenset({"symbol", "date", "open", "high", "low", "close", "price", "is_member"}),
)
DATASET_CONTRACT = FrameContract(
    "dataset", frozenset({"symbol", "date", "close", "price"})
)
PREDICTION_CONTRACT = FrameContract(
    "prediction", frozenset({"symbol", "date", "close", "score", "action"})
)
PICK_CONTRACT = FrameContract(
    "pick list", frozenset({"symbol", "date", "event", "direction", "rank"})
)


class PickDocument(TypedDict, total=False):
    market: str
    date: str
    symbol: str
    signal_model: str
    event: str
    pivot_event: str
    actionable: bool
    horizon: int
    tag: str
    generated_at: Any
