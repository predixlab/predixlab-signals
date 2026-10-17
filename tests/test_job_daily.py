"""Tests for the scheduled job: calendar handling, publication and orchestration.

The daily run is the only part of the package with side effects, so the sinks
are injected and asserted on rather than mocked at the driver level.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.config import PATHS, Paths
from predixlab_signals.jobs.daily import (
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    DailyRun,
    is_trading_day,
    read_trading_days,
    run_daily,
)
from predixlab_signals.publish.mongo import (
    AdviceStore,
    pick_counts,
    pick_documents,
)
from predixlab_signals.settings import Settings


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _write_symbol(directory, symbol, days, close, factor=0.5):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    spread = np.abs(rng.normal(0.008, 0.004, len(days)))
    open_prices = close * (1 + rng.normal(0, 0.002, len(days)))
    pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in days],
            "symbol": symbol,
            "open": open_prices,
            "high": np.maximum(open_prices, close) * (1 + spread),
            "low": np.minimum(open_prices, close) * (1 - spread),
            "close": close,
            "volume": rng.integers(1e5, 1e6, len(days)).astype(float),
            "factor": factor,
        }
    ).to_csv(directory / f"{symbol}.csv", index=False)


@pytest.fixture(scope="module")
def collector_root(tmp_path_factory):
    """A synthetic collector tree: ~7 years of bars for 12 names plus the index."""
    root = tmp_path_factory.mktemp("collector")
    normalized = root / "normalized_data" / "us_data"
    days = pd.bdate_range("2018-01-01", periods=1800)
    n = len(days)
    rng = np.random.default_rng(7)

    (root / "calendar").mkdir(parents=True)
    (root / "calendar" / "us.txt").write_text("\n".join(d.strftime("%Y-%m-%d") for d in days))

    # A cyclical component on top of the random walk. A pure random walk has no
    # swing structure by construction, so no turning point can legitimately
    # fire in it and the publication path would never be exercised.
    market_path = np.cumsum(rng.normal(0.0003, 0.008, n))
    symbols = [f"SYM{i:02d}" for i in range(12)]
    for i, symbol in enumerate(symbols):
        idio = np.cumsum(rng.normal(0.0002, 0.010, n))
        cycle = 0.22 * np.sin(np.linspace(0, 14 * np.pi, n) + i * 0.7)
        _write_symbol(normalized, symbol, days, 40.0 * np.exp(market_path + idio + cycle))

    _write_symbol(normalized, "^GSPC", days, 1000.0 * np.exp(market_path), factor=1.0)

    (root / "instruments").mkdir(parents=True)
    (root / "instruments" / "us.txt").write_text(
        "\n".join(f"{s}\t2018-01-01\t2099-12-31" for s in symbols)
    )
    return root


@pytest.fixture
def pinned_paths(collector_root, tmp_path, monkeypatch):
    """Point the package's path singleton at the synthetic tree.

    ``PATHS`` is a frozen dataclass imported by value across the package, so it
    is patched in place and restored rather than re-created.

    A production model is part of the world the job runs in, not something the
    job creates: it never trains. So the first test to need one trains it and
    writes it to the production path, and the module-scoped tree keeps it for
    the rest.
    """
    original = (PATHS.collector_data, PATHS.training_data, PATHS.artifacts)
    object.__setattr__(PATHS, "collector_data", collector_root)
    object.__setattr__(PATHS, "training_data", collector_root)
    object.__setattr__(PATHS, "artifacts", tmp_path / "artifacts")
    _pin_job_clock(monkeypatch, max(read_trading_days("us")))
    # Daily now consumes a separately installed AlphaRank production artifact.
    # Keep the legacy Pivot orchestration fixture lightweight while supplying a
    # deterministic validated-runtime stand-in; artifact loading is tested
    # independently in the AlphaRank-focused tests.
    from predixlab_signals.jobs import daily

    class _AlphaRankFixtureModel:
        def predict(self, frame):
            return pd.DataFrame(
                {
                    "return_hat": 0.1,
                    "rank_hat": 0.2,
                    "p_top": 0.3,
                    "alpha_score": 0.5,
                    "alpha_consensus": 0.5,
                },
                index=frame.index,
            )

    monkeypatch.setattr(
        daily,
        "load_production_alpha_rank_model",
        lambda **kwargs: (_AlphaRankFixtureModel(), {"retrained": False}),
    )
    _ensure_production_model()
    yield collector_root
    for name, value in zip(("collector_data", "training_data", "artifacts"), original):
        object.__setattr__(PATHS, name, value)


def _ensure_production_model(horizon: int = 10, tag: str = "default") -> Path:
    """Train once and install it, mirroring the local workflow the server depends on."""
    from predixlab_signals.data import build_dataset
    from predixlab_signals.features import feature_columns
    from predixlab_signals.model import model_path, train_final_model

    path = model_path("us", horizon, tag)
    if path.exists():
        return path
    data = build_dataset(market="us", start="2019-01-01", horizon=horizon)
    model = train_final_model(data, horizon=horizon, features=feature_columns(data))
    model.save(path)
    return path


class FakeCollection:
    """Just enough of a pymongo collection to observe what the job writes."""

    def __init__(self):
        self.documents: list[dict] = []
        self.indexes: list[tuple] = []
        self.deleted = 0

    def create_index(self, keys, **options):
        self.indexes.append((tuple(keys), options))

    def bulk_write(self, operations, ordered=True):
        for op in operations:
            self.documents.append(op)
        return type("Result", (), {"upserted_count": len(operations), "modified_count": 0})()

    def update_one(self, key, update, upsert=False):
        self.documents.append({"key": key, "update": update})

    def insert_one(self, document):
        self.documents.append(document)

    def distinct(self, field, query=None):
        return []

class FakeStore(AdviceStore):
    """An :class:`AdviceStore` with the driver replaced by in-memory doubles."""

    def __init__(self, settings):
        self.settings = settings
        self._owns_client = False
        self._picks = FakeCollection()
        self._alpharank = FakeCollection()
        self._jobs = FakeCollection()

    @property
    def picks(self):
        return self._picks

    @property
    def alpharank(self):
        return self._alpharank

    @property
    def jobs(self):
        return self._jobs

    def ensure_indexes(self):
        self._picks.create_index([("market", 1)], name="stub")

    def write_picks(self, documents):
        documents = list(documents)
        self._picks.documents.extend(documents)
        return len(documents)

    def write_alpharank(self, documents):
        documents = list(documents)
        self._alpharank.documents.extend(documents)
        return len(documents)

    def write_job_status(self, job_status):
        self._jobs.insert_one(job_status)

    def close(self):
        return None


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------
def test_paths_separate_training_and_prediction_defaults(monkeypatch):
    monkeypatch.delenv("PREDIXLAB_DATA", raising=False)
    monkeypatch.delenv("PREDIXLAB_TRAINING_DATA", raising=False)

    paths = Paths()

    assert paths.collector_data == Path("~/.predixlab").expanduser().resolve()
    assert paths.training_data == (Path(__file__).resolve().parents[1] / "data").resolve()
    assert paths.instruments_path("us") == paths.collector_data / "instruments" / "us.txt"
    assert paths.normalized_dir("us") == paths.collector_data / "normalized_data" / "us_data"
    assert paths.model_dir("us") == paths.collector_data / "models" / "us"
    assert paths.daily_predictions_dir("us", "2026-08-04") == (
        paths.collector_data / "predictions" / "2026-08-04" / "us"
    )
    assert paths.daily_picks_dir("2026-08-04") == (
        paths.collector_data / "picks" / "2026-08-04"
    )


def test_predixlab_signal_path_overrides_take_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDIXLAB_DATA", str(tmp_path / "shared"))
    monkeypatch.setenv("PREDIXLAB_TRAINING_DATA", str(tmp_path / "history"))
    monkeypatch.setenv("US_INDEX_PATH", str(tmp_path / "custom" / "us.txt"))
    monkeypatch.setenv("US_NORMALIZED_DATA_DIR", str(tmp_path / "custom" / "us_data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "custom" / "models"))
    monkeypatch.setenv("PREDICTIONS_DIR", str(tmp_path / "custom" / "predictions"))
    monkeypatch.setenv("PICKS_DIR", str(tmp_path / "custom" / "picks"))

    paths = Paths()

    assert paths.training_data == (tmp_path / "history").resolve()
    assert paths.instruments_path("us") == (tmp_path / "custom" / "us.txt").resolve()
    assert paths.normalized_dir("us") == (tmp_path / "custom" / "us_data").resolve()
    assert paths.model_dir("us") == (tmp_path / "custom" / "models" / "us").resolve()
    assert paths.daily_predictions_dir("us", "2026-08-04") == (
        tmp_path / "custom" / "predictions" / "2026-08-04" / "us"
    ).resolve()
    assert paths.daily_picks_dir("2026-08-04") == (
        tmp_path / "custom" / "picks" / "2026-08-04"
    ).resolve()


def test_read_trading_days(pinned_paths):
    days = read_trading_days("us")
    assert "2018-01-01" in days
    assert all(len(d) == 10 for d in days)


def test_is_trading_day_uses_us_calendar_under_predixlab_data(tmp_path):
    paths = Paths(collector_data=tmp_path, artifacts=tmp_path / "artifacts")
    calendar = tmp_path / "calendar" / "us.txt"
    calendar.parent.mkdir(parents=True)
    calendar.write_text("bad row\n2026-08-04\n\n")

    assert is_trading_day(datetime(2026, 8, 4), paths)
    assert not is_trading_day(datetime(2026, 8, 3), paths)


def test_read_trading_days_without_calendar(tmp_path):
    original = PATHS.collector_data
    object.__setattr__(PATHS, "collector_data", tmp_path)
    try:
        assert read_trading_days("us") == set()
    finally:
        object.__setattr__(PATHS, "collector_data", original)


def _picks_frame():
    """A published event list, in the shape the job hands to the sinks."""
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-04-24")] * 3,
            "symbol": ["AAA", "BBB", "CCC"],
            "event": ["BUY_TURN", "TREND_BUY", "SELL_TURN"],
            "direction": ["long", "long", "short"],
            "kind": ["reversal", "trend", "reversal"],
            "rank": [1, 2, 1],
            "action": ["STRONG_BUY", "STRONG_BUY", "STRONG_SELL"],
            "score": [0.31, 0.18, -0.22],
            "confidence": [0.98, 0.8, 0.9],
            "price": [10.0, 20.0, 30.0],
        }
    )


def test_pick_documents_are_keyed_by_event():
    docs = pick_documents(_picks_frame(), "us", 10, "default")
    keys = {(d["market"], d["date"], d["symbol"], d["event"]) for d in docs}
    assert len(keys) == len(docs)


def test_pick_counts_report_each_event_type():
    counts = pick_counts(_picks_frame())
    assert counts["BUY_TURN"] == 1
    assert counts["TREND_BUY"] == 1
    assert counts["SELL_TURN"] == 1


def test_daily_run_job_status_includes_top_picks():
    run = DailyRun(
        market="us",
        horizon=10,
        tag="default",
        status=STATUS_SUCCESS,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        date="2026-04-24",
        symbols_scored=3,
        counts={"BUY_TURN": 1, "SELL_TURN": 1},
        picks=_picks_frame(),
    )
    status = run.job_status()
    assert status["job_name"] == "predixlab_signals"
    assert status["results"]["top_buys"][0]["symbol"] == "AAA"
    assert status["results"]["top_sells"][0]["symbol"] == "CCC"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _settings(**overrides) -> Settings:
    base = dict(
        mongodb_url="mongodb://unused",
        database_name="test",
        market="us",
        tag="default",
        horizon=10,
        train_start="2019-01-01",
        lookback_sessions=250,
        max_per_side=0,
        require_trading_day=False,
        acs_connection_string="",
        acs_sender_email="",
        acs_to_emails="",
        campaign_events_enabled=False,
        campaign_storage_connection_string="",
        campaign_events_queue="campaign-events",
    )
    base.update(overrides)
    return Settings(**base)


def test_run_daily_checks_yesterday_before_building_data(pinned_paths, monkeypatch):
    from predixlab_signals.jobs import daily

    monkeypatch.setattr(daily, "is_trading_day", lambda date, paths, market: False)
    monkeypatch.setattr(
        daily,
        "build_inference_dataset",
        lambda **kwargs: pytest.fail("inference data must not be built on a skipped day"),
    )
    sent: list[dict] = []
    run = run_daily(
        settings=_settings(require_trading_day=True),
        store_factory=FakeStore,
        notifier=lambda status, settings: sent.append(status) or True,
        clock=lambda: datetime(2026, 8, 5, 2, tzinfo=timezone.utc),
    )
    assert run.status == STATUS_SKIPPED
    assert run.error == "Yesterday was not a trading day"
    assert sent and sent[0]["status"] == STATUS_SKIPPED


def test_run_daily_reports_failure_without_raising(tmp_path):
    """An empty data tree must produce a failed run, not an exception."""
    original = PATHS.collector_data
    object.__setattr__(PATHS, "collector_data", tmp_path / "empty")
    try:
        run = run_daily(
            settings=_settings(require_trading_day=False),
            store_factory=FakeStore,
            notifier=lambda status, settings: True,
        )
    finally:
        object.__setattr__(PATHS, "collector_data", original)
    assert run.status == STATUS_FAILED
    assert run.error


def _active_session(settings) -> str:
    """A session on which at least one event fires.

    Events are sparse by design — 1.87% of bar-days on the walk-forward set —
    so with a twelve-name universe the *expected* count on an arbitrary day is
    well under one. Asserting the publication path against a fixed date would
    therefore test the calendar rather than the pipeline.
    """
    from predixlab_signals.data import build_inference_dataset
    from predixlab_signals.model import PivotModel, model_path
    from predixlab_signals.signals.scoring import score_panel
    from predixlab_signals.signals.picks import select_picks

    data = build_inference_dataset(market=settings.market, horizon=settings.horizon)
    model = PivotModel.load(model_path(settings.market, settings.horizon, settings.tag))
    events = select_picks(score_panel(model, data, lookback=settings.lookback_sessions))
    assert not events.empty, "the fixture must produce at least one event"
    return str(pd.Timestamp(events["date"].max()).date())


def _pin_job_clock(monkeypatch, target: str) -> None:
    """Request the fixture's session instead of silently using its latest row."""
    from predixlab_signals.jobs import daily

    fixed = (pd.Timestamp(target) + pd.Timedelta(days=1, hours=2)).to_pydatetime()
    fixed = fixed.replace(tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(daily, "datetime", FixedDateTime)


def test_run_daily_publishes_and_notifies(pinned_paths, monkeypatch):
    settings = _settings(
        campaign_events_enabled=True,
        campaign_storage_connection_string="UseDevelopmentStorage=true",
    )
    # Warm the model cache so the helper below can reuse it.
    run_daily(settings=settings, store_factory=FakeStore,
              notifier=lambda status, s: True)
    target = _active_session(settings)
    _pin_job_clock(monkeypatch, target)

    stores: list[FakeStore] = []
    sent: list[dict] = []
    campaign_events: list[tuple[str, str]] = []

    def factory(s):
        store = FakeStore(s)
        stores.append(store)
        return store

    run = run_daily(
        settings=settings,
        store_factory=factory,
        notifier=lambda status, s: sent.append(status) or True,
        campaign_publisher=lambda market, date, s: campaign_events.append(
            (market, date)
        ),
    )

    assert run.status == STATUS_SUCCESS, run.error
    # The whole universe is scored; only the bars that fired are published.
    assert run.symbols_scored == 12
    assert run.date == target
    assert run.artifact and run.artifact.endswith(".csv")

    store = stores[0]
    published = store.picks.documents
    assert 0 < len(published) <= run.symbols_scored
    assert len(published) == sum(run.counts.values())
    assert {d["market"] for d in published} == {"us"}
    assert {d["direction"] for d in published} <= {"long", "short"}
    assert {d["event"] for d in published} <= {
        "BOTTOM_TURN",
        "BUY_TURN",
        "SELL_TURN",
        "TOP_TURN",
        "TREND_BUY",
        "BASE_BREAKOUT",
    }
    assert all(d["confidence"] is not None for d in published)
    assert all(d["price"] is not None for d in published)
    assert store.jobs.documents
    assert sent[0]["results"]["published"] == len(published)
    assert campaign_events == [("us", target)]
    assert run.campaign_event_status == "sent"


def test_run_daily_treats_a_quiet_session_as_success(pinned_paths, monkeypatch):
    """Most sessions fire nothing, and that is an answer rather than a failure."""
    from predixlab_signals.jobs import daily

    settings = _settings()
    run_daily(settings=settings, store_factory=FakeStore,
              notifier=lambda status, s: True)

    stores: list[FakeStore] = []

    def factory(s):
        store = FakeStore(s)
        stores.append(store)
        return store

    monkeypatch.setattr(
        daily,
        "select_picks",
        lambda scored, **kwargs: scored.iloc[0:0].copy(),
    )
    run = run_daily(
        settings=settings,
        store_factory=factory,
        notifier=lambda status, s: True,
    )

    assert run.status == STATUS_SUCCESS
    assert run.picks is not None and run.picks.empty
    assert sum(run.counts.values()) == 0
    assert stores[0].picks.documents == []
    # The job status still records that the run reached a quiet conclusion.
    assert stores[0].jobs.documents


def test_run_daily_explicit_end_date_bypasses_automatic_trading_day_check(pinned_paths, monkeypatch):
    from predixlab_signals.jobs import daily

    monkeypatch.setattr(
        daily, "is_trading_day",
        lambda *args, **kwargs: pytest.fail("explicit date bypasses the automatic check"),
    )
    target = max(read_trading_days("us"))
    run = run_daily(
        settings=_settings(require_trading_day=True),
        store_factory=FakeStore,
        notifier=lambda status, s: True,
        end_date=target,
    )

    assert run.status == STATUS_SUCCESS, run.error
    assert run.date == target


def test_run_daily_explicit_end_date_never_falls_back_to_old_data(pinned_paths):
    run = run_daily(
        settings=_settings(require_trading_day=True),
        dry_run=True,
        end_date="2099-01-02",
    )
    assert run.status == STATUS_FAILED
    assert run.date == "2099-01-02"
    assert run.artifact is None
    assert "not in the us trading calendar" in run.error


def test_run_daily_never_trains(pinned_paths):
    """The cloud job is a consumer of a validated model, never a producer.

    A model it fitted itself has passed no review, yet its signals would reach
    the product surface indistinguishable from a reviewed one.
    """
    from predixlab_signals.model import model_path

    path = model_path("us", 10, "default")
    before = path.stat().st_mtime

    run = run_daily(settings=_settings(), store_factory=FakeStore,
                    notifier=lambda status, settings: True)

    assert run.status == STATUS_SUCCESS, run.error
    assert run.model["retrained"] is False
    assert path.stat().st_mtime == before


def test_run_daily_dry_run_publishes_nothing(pinned_paths):
    stores: list[FakeStore] = []
    run = run_daily(
        settings=_settings(),
        dry_run=True,
        store_factory=lambda s: stores.append(FakeStore(s)) or stores[-1],
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SUCCESS
    assert not stores


def test_run_daily_survives_a_publication_failure(pinned_paths):
    """The computation stands, but the failure must not read as a clean run."""

    stores: list[FakeStore] = []

    class BrokenStore(FakeStore):
        def write_picks(self, documents):
            raise RuntimeError("cosmos is down")

    def factory(settings):
        store = BrokenStore(settings)
        stores.append(store)
        return store

    sent: list[dict] = []
    run = run_daily(
        settings=_settings(
            campaign_events_enabled=True,
            campaign_storage_connection_string="UseDevelopmentStorage=true",
        ),
        store_factory=factory,
        notifier=lambda status, settings: sent.append(status) or True,
        campaign_publisher=lambda market, date, settings: pytest.fail(
            "campaign event must not be published after a database failure"
        ),
    )
    assert run.status == STATUS_SUCCESS
    assert run.publish_error == "cosmos is down"
    assert run.delivered is False
    assert "cosmos is down" in sent[0]["warning"]
    assert stores[0].jobs.documents
    assert "cosmos is down" in stores[0].jobs.documents[0]["warning"]
    assert run.campaign_event_status == "blocked"


def test_run_daily_surfaces_campaign_queue_failure(pinned_paths):
    run = run_daily(
        settings=_settings(
            campaign_events_enabled=True,
            campaign_storage_connection_string="UseDevelopmentStorage=true",
        ),
        store_factory=FakeStore,
        notifier=lambda status, settings: True,
        campaign_publisher=lambda market, date, settings: (_ for _ in ()).throw(
            RuntimeError("queue unavailable")
        ),
    )

    assert run.status == STATUS_SUCCESS
    assert run.campaign_event_status == "failed"
    assert run.campaign_event_error == "queue unavailable"
    assert run.delivered is False


def test_delivered_is_true_for_a_clean_run(pinned_paths):
    run = run_daily(
        settings=_settings(),
        store_factory=FakeStore,
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SUCCESS
    assert run.publish_error is None
    assert run.delivered is True


def test_delivered_is_true_for_a_skipped_run(pinned_paths, monkeypatch):
    """Nothing was computed, so there is nothing that failed to publish."""
    from predixlab_signals.jobs import daily

    monkeypatch.setattr(daily, "is_trading_day", lambda date, paths, market: False)
    run = run_daily(
        settings=_settings(require_trading_day=True),
        store_factory=FakeStore,
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SKIPPED
    assert run.delivered is True


def test_run_daily_records_an_unreachable_store(pinned_paths):
    def factory(settings):
        raise RuntimeError("connection refused")

    run = run_daily(
        settings=_settings(),
        store_factory=factory,
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SUCCESS
    assert run.publish_error == "connection refused"
    assert run.delivered is False


# ---------------------------------------------------------------------------
# Shared-root outputs: pivot writes where predixlab-signal writes
# ---------------------------------------------------------------------------
def _run_for_outputs(pinned_paths):
    """One successful session, returning the run and the shared root."""
    settings = _settings()
    # Warm the model cache so the active-session helper can reuse it.
    run_daily(settings=settings, store_factory=FakeStore,
              notifier=lambda status, s: True)
    run = run_daily(settings=settings,
                    store_factory=FakeStore, notifier=lambda status, s: True)
    assert run.status == STATUS_SUCCESS, run.error
    return run, run.date


def test_predictions_land_under_the_signal_convention(pinned_paths):
    """Both jobs mount the same share, so both publish into the same layout."""
    run, target = _run_for_outputs(pinned_paths)

    expected = pinned_paths / "predictions" / target / "us" / "us_pivot.csv"
    assert Path(run.artifact) == expected
    assert expected.exists()

    frame = pd.read_csv(expected)
    # The *whole* cross-section, not just what fired: it is the audit trail.
    assert len(frame) == run.symbols_scored
    assert set(frame["date"].astype(str).str[:10]) == {target}


def test_picks_land_under_the_signal_convention(pinned_paths):
    run, target = _run_for_outputs(pinned_paths)

    expected = pinned_paths / "picks" / target / "us_pivot.csv"
    assert Path(run.picks_file) == expected
    assert expected.exists()

    frame = pd.read_csv(expected)
    assert len(frame) == sum(run.counts.values())
    assert set(frame["event"]) <= {
        "BOTTOM_TURN",
        "BUY_TURN",
        "SELL_TURN",
        "TOP_TURN",
        "TREND_BUY",
        "BASE_BREAKOUT",
    }


def test_filenames_do_not_collide_with_signal(pinned_paths):
    """`us.csv` and `top_rising.csv` belong to signal; pivot namespaces its own."""
    _, target = _run_for_outputs(pinned_paths)

    assert not (pinned_paths / "predictions" / target / "us" / "us.csv").exists()
    picks = pinned_paths / "picks" / target
    assert not (picks / "top_rising.csv").exists()
    assert not (picks / "top_declining.csv").exists()


def test_the_run_record_carries_both_paths(pinned_paths):
    run, _ = _run_for_outputs(pinned_paths)
    results = run.job_status()["results"]

    assert results["artifact"] == run.artifact
    assert results["picks_file"] == run.picks_file


# ---------------------------------------------------------------------------
# Scratch vs state: what has to survive an empty container
# ---------------------------------------------------------------------------
def test_model_is_kept_on_the_shared_root_not_in_scratch(pinned_paths):
    """PIVOT_ARTIFACTS starts empty every run, so a model there would be refit
    nightly — which the retrain-interval setting exists to prevent."""
    from predixlab_signals.model import model_path

    path = model_path("us", 10, "default")

    assert path == pinned_paths / "models" / "us" / "pivot_h10_default.pkl"
    assert PATHS.artifacts not in path.parents


def test_the_daily_job_writes_the_model_to_the_shared_root(pinned_paths):
    settings = _settings()
    run = run_daily(settings=settings, store_factory=FakeStore,
                    notifier=lambda status, s: True)

    assert run.status == STATUS_SUCCESS, run.error
    saved = pinned_paths / "models" / "us" / f"pivot_h{settings.horizon}_{settings.tag}.pkl"
    assert saved.exists()


def test_a_wiped_scratch_tree_still_runs(pinned_paths):
    """The next night's container: same share, empty scratch.

    A bounded inference frame is built from the collector CSVs and the model is
    read off the share, so the run succeeds without training or a dataset cache.
    """
    import shutil

    from predixlab_signals.model import model_path

    shutil.rmtree(PATHS.artifacts, ignore_errors=True)
    assert not PATHS.dataset_dir("us").exists()
    before = model_path("us", 10, "default").stat().st_mtime

    run = run_daily(settings=_settings(), store_factory=FakeStore,
                    notifier=lambda status, s: True)

    assert run.status == STATUS_SUCCESS, run.error
    assert run.model["retrained"] is False
    assert not PATHS.dataset_dir("us").exists()
    assert model_path("us", 10, "default").stat().st_mtime == before


def test_a_missing_production_model_fails_the_run(pinned_paths):
    """Loudly, with instructions — not by quietly fitting a replacement."""
    from predixlab_signals.model import model_path

    path = model_path("us", 10, "default")
    stashed = path.with_suffix(".stashed")
    path.rename(stashed)
    try:
        run = run_daily(settings=_settings(), store_factory=FakeStore,
                        notifier=lambda status, s: True)
    finally:
        stashed.rename(path)

    assert run.status == STATUS_FAILED
    assert "No production model" in run.error
    assert not run.counts
