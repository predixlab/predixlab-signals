"""Runtime settings for the scheduled job — everything comes from the environment.

The research package deliberately has no configuration framework: paths live in
:mod:`predixlab_signals.config` and everything else is an argument. The nightly
job does need ambient configuration (database, mailbox, schedule tolerances), so
it is collected here and nowhere else, read with :func:`os.getenv` so the core
package keeps its four dependencies.

Nothing here is imported by the model code. Importing this module never opens a
connection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from predixlab_signals.data_quality import DEFAULT_MAX_BAD_DATA_RATIO

#: Collections owned by pivot. They are deliberately *separate* from the legacy
#: ``predictions`` / ``picks`` collections written by predixlab-signal, so the
#: two datasets can coexist while the old pipeline is retired.
#:
#: Only the *selected* pick list is published. The full scored cross-section is
#: a model artifact — it stays in the artifacts tree where it can be audited,
#: because 500 rows of signed scores is not something a reader can act on.
DEFAULT_PICKS_COLLECTION = "pivot_picks"
DEFAULT_ALPHARANK_COLLECTION = "alpharank"
DEFAULT_ALPHARANK_GENERATIONS_COLLECTION = "alpharank_generations"
DEFAULT_JOBS_COLLECTION = "jobs"

#: Internal job identifier persisted with every run, and its display name.
JOB_NAME = "predixlab_signals"
JOB_DISPLAY_NAME = "PredixLab Signals"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean; got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; got {raw!r}") from exc


KNOWN_PIVOT_ENV = {
    "PIVOT_PICKS_COLLECTION",
    "PIVOT_MARKET",
    "PIVOT_TAG",
    "PIVOT_HORIZON",
    "PIVOT_TRAIN_START",
    "PIVOT_LOOKBACK",
    "PIVOT_MAX_PER_SIDE",
    "PIVOT_REQUIRE_TRADING_DAY",
    "PIVOT_DATA_MAX_BAD_RATIO",
    "PIVOT_ARTIFACTS",
    "ALPHARANK_TAG",
    "ALPHARANK_COLLECTION",
    "ALPHARANK_GENERATIONS_COLLECTION",
    "ALPHARANK_TOP_N",
}


def _validate_known_environment() -> None:
    unknown = sorted(
        name
        for name in os.environ
        if (
            name.startswith("PIVOT_") or name.startswith("ALPHARANK_")
        )
        and name not in KNOWN_PIVOT_ENV
    )
    if unknown:
        raise ValueError(f"Unknown predixlab-signals environment variables: {unknown}")


@dataclass(frozen=True)
class Settings:
    """Ambient configuration for the scheduled pivot job."""

    # -- database ----------------------------------------------------------
    mongodb_url: str = field(default_factory=lambda: _env("MONGODB_URL", "mongodb://localhost:27017"))
    database_name: str = field(default_factory=lambda: _env("DATABASE_NAME", "predixlab"))
    picks_collection: str = field(
        default_factory=lambda: _env("PIVOT_PICKS_COLLECTION", DEFAULT_PICKS_COLLECTION)
    )
    alpharank_collection: str = field(
        default_factory=lambda: _env("ALPHARANK_COLLECTION", DEFAULT_ALPHARANK_COLLECTION)
    )
    alpharank_generations_collection: str = field(
        default_factory=lambda: _env(
            "ALPHARANK_GENERATIONS_COLLECTION",
            DEFAULT_ALPHARANK_GENERATIONS_COLLECTION,
        )
    )
    alpharank_top_n: int = field(default_factory=lambda: _env_int("ALPHARANK_TOP_N", 15))
    jobs_collection: str = field(
        default_factory=lambda: _env("JOBS_COLLECTION", DEFAULT_JOBS_COLLECTION)
    )

    # -- campaign events ---------------------------------------------------
    campaign_events_enabled: bool = field(
        default_factory=lambda: _env_bool("CAMPAIGN_EVENTS_ENABLED", False)
    )
    campaign_storage_connection_string: str = field(
        default_factory=lambda: _env("CAMPAIGN_STORAGE_CONNECTION_STRING")
    )
    campaign_events_queue: str = field(
        default_factory=lambda: _env("CAMPAIGN_EVENTS_QUEUE", "campaign-events")
    )

    # -- notifications -----------------------------------------------------
    acs_connection_string: str = field(default_factory=lambda: _env("ACS_CONNECTION_STRING"))
    acs_sender_email: str = field(default_factory=lambda: _env("ACS_SENDER_EMAIL"))
    acs_to_emails: str = field(default_factory=lambda: _env("ACS_TO_EMAILS"))

    # -- pipeline ----------------------------------------------------------
    market: str = field(default_factory=lambda: _env("PIVOT_MARKET", "us").lower())
    tag: str = field(default_factory=lambda: _env("PIVOT_TAG", "default"))
    alpharank_tag: str = field(default_factory=lambda: _env("ALPHARANK_TAG"))
    horizon: int = field(default_factory=lambda: _env_int("PIVOT_HORIZON", 10))
    #: Deprecated compatibility field. Daily inference no longer rebuilds the
    #: training dataset, but accepting this value keeps existing deployments
    #: from failing the strict environment validation during rollout.
    train_start: str = field(default_factory=lambda: _env("PIVOT_TRAIN_START", "2015-01-01"))
    #: Sessions of history scored so the turning-point rules have their windows
    #: warmed up. The longest rule window is the 120-day trend average.
    lookback_sessions: int = field(default_factory=lambda: _env_int("PIVOT_LOOKBACK", 250))
    #: Safety valve on a single session's event count, **not** a selection rule.
    #: ``0`` (the default) publishes every event that fired: the rules already
    #: decide what qualifies, and capping would drop real signals on exactly the
    #: days a regime is turning.
    max_per_side: int = field(default_factory=lambda: _env_int("PIVOT_MAX_PER_SIDE", 0))
    #: Skip the run when the requested date is not a trading day.
    require_trading_day: bool = field(
        default_factory=lambda: _env_bool("PIVOT_REQUIRE_TRADING_DAY", True)
    )
    max_bad_data_ratio: float = field(
        default_factory=lambda: _env_float(
            "PIVOT_DATA_MAX_BAD_RATIO", DEFAULT_MAX_BAD_DATA_RATIO
        )
    )

    @property
    def recipients(self) -> list[str]:
        return [e.strip() for e in self.acs_to_emails.split(",") if e.strip()]

    @property
    def email_configured(self) -> bool:
        return bool(self.acs_connection_string and self.acs_sender_email and self.recipients)

    def __post_init__(self) -> None:
        if not 0 <= self.max_bad_data_ratio < 1:
            raise ValueError(
                f"PIVOT_DATA_MAX_BAD_RATIO must be in [0, 1); got {self.max_bad_data_ratio}"
            )
        if not self.alpharank_tag:
            object.__setattr__(self, "alpharank_tag", self.tag)
        if not self.alpharank_collection:
            raise ValueError("ALPHARANK_COLLECTION must not be empty")
        if not self.alpharank_generations_collection:
            raise ValueError("ALPHARANK_GENERATIONS_COLLECTION must not be empty")
        if self.alpharank_top_n <= 0:
            raise ValueError(f"ALPHARANK_TOP_N must be greater than zero; got {self.alpharank_top_n}")
        if self.campaign_events_enabled:
            if not self.campaign_storage_connection_string:
                raise ValueError(
                    "CAMPAIGN_STORAGE_CONNECTION_STRING is required when "
                    "CAMPAIGN_EVENTS_ENABLED is true"
                )
            if not self.campaign_events_queue:
                raise ValueError(
                    "CAMPAIGN_EVENTS_QUEUE is required when "
                    "CAMPAIGN_EVENTS_ENABLED is true"
                )


def load_settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    _validate_known_environment()
    return Settings()
