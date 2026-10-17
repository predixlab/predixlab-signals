from __future__ import annotations

import pandas as pd
import pytest

from predixlab_signals.contracts import DataContractError, PICK_CONTRACT
from predixlab_signals.settings import Settings, load_settings


def test_pick_contract_reports_missing_columns():
    frame = pd.DataFrame({"symbol": ["AAA"], "date": ["2026-01-02"]})

    with pytest.raises(DataContractError, match="direction.*event.*rank"):
        PICK_CONTRACT.validate(frame, allow_empty=False)


def test_invalid_integer_environment_is_not_silently_defaulted(monkeypatch):
    monkeypatch.setenv("PIVOT_HORIZON", "ten")

    with pytest.raises(ValueError, match="PIVOT_HORIZON must be an integer"):
        Settings()


def test_invalid_boolean_environment_is_not_silently_defaulted(monkeypatch):
    monkeypatch.setenv("PIVOT_REQUIRE_TRADING_DAY", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        Settings()


def test_alpharank_tag_defaults_to_pivot_tag_and_can_be_independent(monkeypatch):
    monkeypatch.setenv("PIVOT_TAG", "default")
    monkeypatch.delenv("ALPHARANK_TAG", raising=False)
    assert Settings().alpharank_tag == "default"

    monkeypatch.setenv("ALPHARANK_TAG", "rank552520")
    assert Settings().alpharank_tag == "rank552520"
    assert Settings().alpharank_generations_collection == "alpharank_generations"


@pytest.mark.parametrize(
    "name",
    [
        "PIVOT_ADVICE_COLLECTION",
        "PIVOT_COLLECTOR_DATA",
        "PIVOT_DATA_SOURCE",
        "PIVOT_MODEL_MAX_AGE_DAYS",
        "PIVOT_QLIB_DATA",
        "PIVOT_RETENTION_DAYS",
        "PIVOT_RUNS_COLLECTION",
    ],
)
def test_unknown_pivot_environment_is_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "legacy")

    with pytest.raises(ValueError, match="Unknown predixlab-signals"):
        load_settings()
