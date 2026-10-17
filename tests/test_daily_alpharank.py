from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predixlab_signals.config import Paths
from predixlab_signals.jobs import daily
from predixlab_signals.jobs.daily import STATUS_FAILED, STATUS_SUCCESS, run_daily
from predixlab_signals.publish.mongo import AdviceStore, alpha_rank_documents
from predixlab_signals.settings import Settings


def _settings(**overrides) -> Settings:
    values = dict(require_trading_day=False, lookback_sessions=2)
    values.update(overrides)
    return Settings(**values)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 3),
            "symbol": ["CCC", "AAA", "BBB"],
            "close": [30.0, 10.0, 20.0],
        }
    )


class _GenerationSignals:
    def __init__(self):
        self.docs = []
        self.fail_after = None

    @staticmethod
    def _matches(row, query):
        for key, value in query.items():
            if key == "$or":
                if not any(_GenerationSignals._matches(row, item) for item in value):
                    return False
            elif isinstance(value, dict) and "$lt" in value:
                if row.get(key) is None or row[key] >= value["$lt"]:
                    return False
            elif isinstance(value, dict) and "$exists" in value:
                if (key in row) != value["$exists"]:
                    return False
            elif row.get(key) != value:
                return False
        return True

    def bulk_write(self, operations, ordered=False):
        for index, operation in enumerate(operations):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("partial AlphaRank bulk failure")
            if "$set" in operation._doc and "$setOnInsert" in operation._doc:
                raise AssertionError("overlapping Mongo update operators")
            found = next(
                (row for row in self.docs if self._matches(row, operation._filter)),
                None,
            )
            if found is not None:
                found.update(operation._doc.get("$set", {}))
            elif operation._upsert:
                self.docs.append(dict(operation._doc.get("$set", {})))
        return type(
            "Result",
            (),
            {"upserted_count": len(operations), "modified_count": 0},
        )()

    def count_documents(self, query):
        return sum(1 for row in self.docs if self._matches(row, query))

    def delete_many(self, query):
        self.docs = [row for row in self.docs if not self._matches(row, query)]


class _GenerationManifests:
    def __init__(self):
        self.docs = {}
        self.interleave = None
        self.interleaved = False

    def find_one(self, query, sort=None):
        key = (query["market"], query["date"], query["signal_model"])
        if self.interleave is not None and not self.interleaved:
            self.interleaved = True
            callback = self.interleave
            self.interleave = None
            callback()
            return None
        return self.docs.get(key)

    def update_one(self, query, update, upsert=False):
        key = (query["market"], query["date"], query["signal_model"])
        current = self.docs.get(key)
        can_update = current is None or any(
            (
                "active_generation_version" in clause
                and "$lt" in clause["active_generation_version"]
                and (
                    clause["active_generation_version"].get("$lt")
                    > current["active_generation_version"]
                )
            )
            for clause in query.get("$or", [])
        )
        if can_update and "$set" in update:
            self.docs[key] = dict(update["$set"])
            return type("Result", (), {"matched_count": 1})()
        if current is None and upsert and "$setOnInsert" in update:
            self.docs[key] = dict(update["$setOnInsert"])
            return type("Result", (), {"matched_count": 0})()
        return type("Result", (), {"matched_count": 0})()


class _GenerationStore(AdviceStore):
    def __init__(self, signals, manifests):
        self._owns_client = False
        self.settings = _settings()
        self._signals = signals
        self._manifests = manifests

    @property
    def alpharank(self):
        return self._signals

    @property
    def alpharank_generations(self):
        return self._manifests


def _alpha_rows(symbols, generated_at):
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-02")] * len(symbols),
            "symbol": symbols,
            "close": [100.0 + index for index in range(len(symbols))],
            "rank": list(range(1, len(symbols) + 1)),
            "alpha_score": [0.9 - index / 100 for index in range(len(symbols))],
            "alpha_consensus": 0.5,
            "return_hat": 0.1,
            "rank_hat": 0.5,
            "p_top": 0.5,
        }
    )
    return alpha_rank_documents(frame, "us", 10, generated_at=generated_at)


def test_alpha_rank_selection_is_deterministic():
    scored = _panel().assign(
        return_hat=0.1,
        rank_hat=0.2,
        p_top=0.3,
        alpha_score=[0.9, 0.9, 0.8],
        alpha_consensus=0.7,
    )
    selected = daily.select_alpha_rank_rows(scored, "2026-01-02", top_n=2)
    assert selected["symbol"].tolist() == ["AAA", "CCC"]
    assert selected["rank"].tolist() == [1, 2]


def test_daily_builds_one_panel_and_scores_pivot_before_alpharank(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(daily, "build_inference_dataset", lambda **kwargs: calls.append("build") or _panel())
    monkeypatch.setattr(
        daily,
        "select_picks",
        lambda scored, **kwargs: calls.append("pivot-select") or scored.iloc[0:0].copy(),
    )

    pivot_scored = _panel().assign(
        score=0.1,
        action="HOLD",
        swing_pos_hat=0.5,
        p_up=0.5,
        exp_gain_pct=0.0,
        exp_loss_pct=0.0,
        exp_amplitude_pct=0.0,
    )
    alpha_scored = _panel().assign(
        return_hat=0.1,
        rank_hat=0.2,
        p_top=0.3,
        alpha_score=[0.1, 0.3, 0.2],
        alpha_consensus=0.5,
    )

    class Store:
        def ensure_indexes(self):
            pass

        def write_picks(self, docs):
            return len(list(docs))

        def write_alpharank(self, docs):
            return len(list(docs))

        def write_job_status(self, status):
            pass

        def close(self):
            pass

    def pivot_score(*args, **kwargs):
        calls.append("pivot-score")
        return pivot_scored

    def alpha_score(*args, **kwargs):
        calls.append("alpha-score")
        return alpha_scored

    run = run_daily(
        settings=_settings(),
        end_date="2026-01-02",
        dry_run=True,
        paths=Paths(collector_data=tmp_path, training_data=tmp_path, artifacts=tmp_path / "a"),
        model_loader=lambda **kwargs: (object(), {"retrained": False}),
        panel_scorer=pivot_score,
        alpha_model_loader=lambda **kwargs: (object(), {"retrained": False}),
        alpha_scorer=alpha_score,
        store_factory=lambda settings: Store(),
        notifier=lambda status, settings: True,
    )

    assert run.status == STATUS_SUCCESS, run.error
    assert calls == ["build", "pivot-score", "pivot-select", "alpha-score"]
    assert run.alpha_picks["symbol"].tolist() == ["AAA", "BBB", "CCC"]
    assert (tmp_path / "predictions" / "2026-01-02" / "us" / "us_alpharank.csv").exists()


def test_pivot_and_alpharank_use_independent_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "build_inference_dataset", lambda **kwargs: _panel())
    monkeypatch.setattr(daily, "select_picks", lambda scored, **kwargs: scored.iloc[0:0].copy())
    pivot = _panel().assign(
        score=0.1,
        action="HOLD",
        swing_pos_hat=0.5,
        p_up=0.5,
        exp_gain_pct=0.0,
        exp_loss_pct=0.0,
        exp_amplitude_pct=0.0,
    )
    alpha = _panel().assign(
        return_hat=0.1,
        rank_hat=0.2,
        p_top=0.3,
        alpha_score=0.5,
        alpha_consensus=0.5,
    )
    loaded_tags = []
    published = []

    class Store:
        def ensure_indexes(self):
            pass

        def write_picks(self, docs):
            return len(list(docs))

        def write_alpharank(self, docs):
            published.extend(docs)
            return len(published)

        def write_job_status(self, status):
            pass

        def close(self):
            pass

    run = run_daily(
        settings=_settings(tag="default", alpharank_tag="rank552520"),
        end_date="2026-01-02",
        paths=Paths(collector_data=tmp_path, training_data=tmp_path, artifacts=tmp_path / "a"),
        model_loader=lambda **kwargs: (loaded_tags.append(("pivot", kwargs["tag"])) or (object(), {})),
        panel_scorer=lambda *args, **kwargs: pivot,
        alpha_model_loader=lambda **kwargs: (
            loaded_tags.append(("alpha", kwargs["tag"])) or (object(), {})
        ),
        alpha_scorer=lambda *args, **kwargs: alpha[
            ["return_hat", "rank_hat", "p_top", "alpha_score", "alpha_consensus"]
        ],
        store_factory=lambda settings: Store(),
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SUCCESS, run.error
    assert loaded_tags == [("pivot", "default"), ("alpha", "rank552520")]
    assert published and {doc["tag"] for doc in published} == {"rank552520"}
    alpha_status = run.job_status()["results"]["alpha_rank"]
    assert alpha_status["tag"] == "rank552520"
    assert run.alpha_model["tag"] == "rank552520"


def test_alpha_documents_use_public_schema_and_clean_values():
    rows = _panel().head(1).assign(
        rank=1,
        alpha_score=0.9,
        alpha_consensus=0.8,
        return_hat=0.1,
        rank_hat=0.7,
        p_top=0.6,
    )
    doc = alpha_rank_documents(
        rows,
        "us",
        10,
        generated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )[0]
    assert doc["signal_model"] == "alpha_rank"
    assert doc["score"] == 0.9
    assert doc["predicted_excess_return"] == 0.1
    assert set(doc) == {
        "market",
        "date",
        "symbol",
        "rank",
        "horizon",
        "tag",
        "generated_at",
        "generation_id",
        "generation_version",
        "signal_model",
        "score",
        "consensus",
        "predicted_excess_return",
        "predicted_rank",
        "top20_probability",
        "close",
    }


def test_alpha_documents_clip_percentile_fields_and_keep_signed_return():
    rows = _panel().head(1).assign(
        rank=1,
        alpha_score=1.5,
        alpha_consensus=-0.5,
        return_hat=-12.0,
        rank_hat=2.0,
        p_top=0.25,
    )
    doc = alpha_rank_documents(rows, "us", 10)[0]
    assert doc["score"] == 1.0
    assert doc["consensus"] == 0.0
    assert doc["predicted_rank"] == 1.0
    assert doc["top20_probability"] == 0.25
    assert doc["predicted_excess_return"] == -12.0


def test_alpha_documents_reject_nonfinite_required_scores():
    base = _panel().head(1).assign(
        rank=1,
        alpha_score=0.5,
        alpha_consensus=0.5,
        return_hat=float("nan"),
        rank_hat=0.5,
        p_top=0.5,
    )
    for field in ("alpha_score", "alpha_consensus", "p_top", "rank_hat"):
        rows = base.copy()
        rows[field] = float("nan")
        try:
            alpha_rank_documents(rows, "us", 10)
        except ValueError as exc:
            assert "finite number" in str(exc)
        else:
            raise AssertionError(f"{field} NaN must be rejected")
    base["return_hat"] = -1.25
    doc = alpha_rank_documents(base, "us", 10)[0]
    assert doc["predicted_excess_return"] == -1.25


def test_alpha_documents_reject_nonfinite_return_hat():
    rows = _panel().head(1).assign(
        rank=1,
        alpha_score=0.5,
        alpha_consensus=0.5,
        return_hat=float("nan"),
        rank_hat=0.5,
        p_top=0.5,
    )
    try:
        alpha_rank_documents(rows, "us", 10)
    except ValueError as exc:
        assert "predicted_excess_return" in str(exc)
    else:
        raise AssertionError("NaN return_hat must be rejected")


def test_alpha_generated_at_is_utc_bson_millisecond_precision():
    BSON = pytest.importorskip("bson").BSON
    rows = _panel().head(1).assign(
        rank=1,
        alpha_score=0.5,
        alpha_consensus=0.5,
        return_hat=0.1,
        rank_hat=0.5,
        p_top=0.5,
    )
    doc = alpha_rank_documents(
        rows,
        "us",
        10,
        generated_at=datetime(2026, 1, 2, 3, 4, 5, 123456),
    )[0]
    assert doc["generated_at"].tzinfo == timezone.utc
    assert doc["generated_at"].microsecond == 123000
    decoded = BSON(BSON.encode(doc)).decode()["generated_at"]
    assert decoded.microsecond == 123000
    assert isinstance(doc["generation_id"], str)
    assert len(doc["generation_id"]) >= 32
    assert doc["generation_version"] == 1767323045123


def test_same_millisecond_runs_get_distinct_generation_ids():
    rows = _panel().head(1).assign(
        rank=1,
        alpha_score=0.5,
        alpha_consensus=0.5,
        return_hat=0.1,
        rank_hat=0.5,
        p_top=0.5,
    )
    stamp = datetime(2026, 1, 2, 3, 4, 5, 123000, tzinfo=timezone.utc)
    first = alpha_rank_documents(rows, "us", 10, generated_at=stamp)[0]
    second = alpha_rank_documents(rows, "us", 10, generated_at=stamp)[0]
    assert first["generation_version"] == second["generation_version"]
    assert first["generation_id"] != second["generation_id"]


def test_generation_double_rejects_overlapping_update_paths():
    UpdateOne = pytest.importorskip("pymongo").UpdateOne
    signals = _GenerationSignals()
    operation = UpdateOne(
        {"market": "us", "symbol": "AAA"},
        {"$set": {"score": 1}, "$setOnInsert": {"score": 1}},
        upsert=True,
    )
    with pytest.raises(AssertionError, match="overlapping"):
        signals.bulk_write([operation])


def test_required_generation_indexes_are_fatal_when_creation_fails():
    class IndexCollection:
        name = "alpharank"

        def create_index(self, keys, **options):
            if options.get("name") == "market_date_model_generation_symbol":
                raise RuntimeError("index unavailable")

        def index_information(self):
            return {}

    class Store(AdviceStore):
        def __init__(self):
            self._owns_client = False
            self.settings = _settings()
            self._collection = IndexCollection()

        @property
        def picks(self):
            return self._collection

        @property
        def alpharank(self):
            return self._collection

        @property
        def alpharank_generations(self):
            return self._collection

        @property
        def jobs(self):
            return self._collection

    with pytest.raises(RuntimeError, match="required AlphaRank index"):
        Store().ensure_indexes()


def test_required_manifest_index_must_verify_unique():
    class IndexCollection:
        name = "alpharank"

        def create_index(self, keys, **options):
            return options["name"]

        def index_information(self):
            return {
                "market_date_model_generation_symbol": {
                    "key": [
                        ("market", 1),
                        ("date", -1),
                        ("signal_model", 1),
                        ("generation_id", 1),
                        ("symbol", 1),
                    ],
                    "unique": True,
                },
                "market_date_model_manifest": {
                    "key": [
                        ("market", 1),
                        ("date", -1),
                        ("signal_model", 1),
                    ],
                    "unique": False,
                },
            }

    class Store(AdviceStore):
        def __init__(self):
            self._owns_client = False
            self.settings = _settings()
            self._collection = IndexCollection()

        @property
        def picks(self):
            return self._collection

        @property
        def alpharank(self):
            return self._collection

        @property
        def alpharank_generations(self):
            return self._collection

        @property
        def jobs(self):
            return self._collection

    with pytest.raises(RuntimeError, match="required AlphaRank index"):
        Store().ensure_indexes()


def test_alpharank_history_index_is_created_for_api_queries():
    class IndexCollection:
        name = "alpharank"

        def __init__(self):
            self.created = {}

        def create_index(self, keys, **options):
            self.created[options["name"]] = (list(keys), options)

        def index_information(self):
            return {
                "market_date_model_generation_symbol": {
                    "key": [
                        ("market", 1),
                        ("date", -1),
                        ("signal_model", 1),
                        ("generation_id", 1),
                        ("symbol", 1),
                    ],
                    "unique": True,
                },
                "market_date_model_manifest": {
                    "key": [("market", 1), ("date", -1), ("signal_model", 1)],
                    "unique": True,
                },
            }

    class Store(AdviceStore):
        def __init__(self):
            self._owns_client = False
            self.settings = _settings()
            self._collection = IndexCollection()

        @property
        def picks(self):
            return self._collection

        @property
        def alpharank(self):
            return self._collection

        @property
        def alpharank_generations(self):
            return self._collection

        @property
        def jobs(self):
            return self._collection

    store = Store()
    store.ensure_indexes()
    keys, options = store._collection.created["market_model_symbol_date_generation"]
    assert keys == [
        ("market", 1),
        ("signal_model", 1),
        ("symbol", 1),
        ("date", -1),
        ("generation_id", 1),
    ]
    assert options.get("unique") is None


def test_committed_generation_stages_exact_rows_and_survives_partial_failure():
    signals = _GenerationSignals()
    manifests = _GenerationManifests()
    store = _GenerationStore(signals, manifests)
    first_time = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
    second_time = datetime(2026, 1, 2, 4, tzinfo=timezone.utc)
    first = _alpha_rows([f"S{i:02d}" for i in range(15)], first_time)
    second = _alpha_rows([f"T{i:02d}" for i in range(15)], second_time)

    assert store.write_alpharank(first) == 15
    key = ("us", "2026-01-02", "alpha_rank")
    assert manifests.docs[key]["active_generation_id"] == first[0]["generation_id"]
    assert signals.count_documents(
        {"market": "us", "date": "2026-01-02", "generation_id": first[0]["generation_id"]}
    ) == 15

    signals.fail_after = 5
    try:
        store.write_alpharank(second)
    except RuntimeError as exc:
        assert "partial AlphaRank bulk failure" in str(exc)
    else:
        raise AssertionError("partial stage failure must propagate")
    assert manifests.docs[key]["active_generation_id"] == first[0]["generation_id"]
    assert signals.count_documents(
        {"market": "us", "date": "2026-01-02", "generation_id": first[0]["generation_id"]}
    ) == 15
    assert signals.count_documents(
        {"market": "us", "date": "2026-01-02", "generation_id": second[0]["generation_id"]}
    ) == 0

    signals.fail_after = None
    assert store.write_alpharank(second) == 15
    assert manifests.docs[key]["active_generation_id"] == second[0]["generation_id"]
    assert signals.count_documents(
        {"market": "us", "date": "2026-01-02", "generation_id": second[0]["generation_id"]}
    ) == 15
    assert signals.count_documents(
        {"market": "us", "date": "2026-01-02", "generation_id": first[0]["generation_id"]}
    ) == 0


def test_committed_generation_promotion_loses_to_interleaved_newer_writer():
    signals = _GenerationSignals()
    manifests = _GenerationManifests()
    store = _GenerationStore(signals, manifests)
    older_time = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
    newer_time = datetime(2026, 1, 2, 4, tzinfo=timezone.utc)
    older = _alpha_rows(["OLD_A", "OLD_B"], older_time)
    newer = _alpha_rows(["NEW_A", "NEW_B"], newer_time)
    manifests.interleave = lambda: store.write_alpharank(newer)

    store.write_alpharank(older)
    key = ("us", "2026-01-02", "alpha_rank")
    assert manifests.docs[key]["active_generation_id"] == newer[0]["generation_id"]
    assert [
        row["symbol"]
        for row in signals.docs
        if row["generation_id"] == newer[0]["generation_id"]
    ] == ["NEW_A", "NEW_B"]
    assert not any(
        row["generation_id"] == older[0]["generation_id"] for row in signals.docs
    )


def test_missing_alpha_model_is_a_failed_daily_run(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "build_inference_dataset", lambda **kwargs: _panel())
    monkeypatch.setattr(daily, "select_picks", lambda scored, **kwargs: scored.iloc[0:0].copy())
    run = run_daily(
        end_date="2026-01-02",
        settings=_settings(),
        dry_run=True,
        paths=Paths(collector_data=tmp_path, training_data=tmp_path, artifacts=tmp_path / "a"),
        model_loader=lambda **kwargs: (object(), {}),
        panel_scorer=lambda *args, **kwargs: _panel().assign(
            score=0.1,
            action="HOLD",
            swing_pos_hat=0.5,
            p_up=0.5,
            exp_gain_pct=0.0,
            exp_loss_pct=0.0,
            exp_amplitude_pct=0.0,
        ),
    )
    assert run.status == STATUS_FAILED
    assert "No production AlphaRank model" in run.error
    assert run.alpha_error == run.error
    assert run.delivered is False
    alpha_status = run.job_status()["results"]["alpha_rank"]
    assert alpha_status["status"] == "failed"
    assert alpha_status["execution_status"] == "failed"
    assert alpha_status["publication_status"] == "not_run"


def test_alpharank_session_replacement_removes_stale_rows():
    class Collection:
        def __init__(self):
            self.docs = []

        @staticmethod
        def matches(row, query):
            for key, value in query.items():
                if key == "$or":
                    if not any(Collection.matches(row, option) for option in value):
                        return False
                elif isinstance(value, dict) and "$nin" in value:
                    if row.get(key) in value["$nin"]:
                        return False
                elif isinstance(value, dict) and "$lte" in value:
                    if row.get(key) is None or row[key] > value["$lte"]:
                        return False
                elif isinstance(value, dict) and "$lt" in value:
                    if row.get(key) is None or row[key] >= value["$lt"]:
                        return False
                elif isinstance(value, dict) and "$exists" in value:
                    if (key in row) != value["$exists"]:
                        return False
                elif row.get(key) != value:
                    return False
            return True

        def find_one(self, query, sort=None):
            matches = [row for row in self.docs if self.matches(row, query)]
            if not matches:
                return None
            return max(matches, key=lambda row: row["generated_at"])

        def delete_many(self, query):
            self.docs = [doc for doc in self.docs if not self.matches(doc, query)]

        def bulk_write(self, operations, ordered=False):
            for operation in operations:
                found = next(
                    (row for row in self.docs if self.matches(row, operation._filter)),
                    None,
                )
                if found is not None:
                    if "$set" in operation._doc:
                        found.update(operation._doc["$set"])
                elif operation._upsert and "$set" in operation._doc:
                    self.docs.append(dict(operation._doc["$set"]))
            return type(
                "Result", (), {"upserted_count": 0, "modified_count": len(operations)}
            )()

    class Store(AdviceStore):
        def __init__(self, collection):
            self._owns_client = False
            self.settings = _settings()
            self._collection = collection

        @property
        def alpharank(self):
            return self._collection

    collection = Collection()
    store = Store(collection)
    first = alpha_rank_documents(
        _panel().assign(
            rank=[1, 2, 3],
            alpha_score=[0.9, 0.8, 0.7],
            alpha_consensus=0.5,
            return_hat=0.1,
            rank_hat=0.2,
            p_top=0.3,
        ),
        "us",
        10,
    )
    second = [dict(first[0]), dict(first[1])]
    second[1]["symbol"] = "ZZZ"
    second_stamp = first[0]["generated_at"] + timedelta(seconds=1)
    for document in second:
        document["generated_at"] = second_stamp
        document["generation_version"] = int(second_stamp.timestamp() * 1000)
        document["generation_id"] = str(document["generation_version"])
    store.write_alpharank(first)
    store.write_alpharank(second)
    assert sorted(row["symbol"] for row in collection.docs) == ["CCC", "ZZZ"]


def test_alpharank_bulk_failure_preserves_previous_session():
    class Collection:
        def __init__(self):
            self.docs = []
            self.fail = False

        @staticmethod
        def matches(row, query):
            for key, value in query.items():
                if key == "$or":
                    if not any(Collection.matches(row, option) for option in value):
                        return False
                elif isinstance(value, dict) and "$nin" in value:
                    if row.get(key) in value["$nin"]:
                        return False
                elif isinstance(value, dict) and "$lt" in value:
                    if row.get(key) is None or row[key] >= value["$lt"]:
                        return False
                elif isinstance(value, dict) and "$exists" in value:
                    if (key in row) != value["$exists"]:
                        return False
                elif row.get(key) != value:
                    return False
            return True

        def find_one(self, query, sort=None):
            matches = [
                row
                for row in self.docs
                if row["market"] == query["market"]
                and row["date"] == query["date"]
                and row["signal_model"] == query["signal_model"]
            ]
            return max(matches, key=lambda row: row["generated_at"]) if matches else None

        def delete_many(self, query):
            self.docs = [doc for doc in self.docs if not self.matches(doc, query)]

        def bulk_write(self, operations, ordered=False):
            if self.fail:
                raise RuntimeError("mongo write failed")
            for operation in operations:
                if "$setOnInsert" in operation._doc:
                    if not any(
                        row["symbol"] == operation._filter["symbol"] for row in self.docs
                    ):
                        self.docs.append(dict(operation._doc["$setOnInsert"]))
                elif operation._upsert:
                    self.docs.append(dict(operation._doc["$set"]))
                else:
                    for row in self.docs:
                        if row["symbol"] == operation._filter["symbol"]:
                            row.update(operation._doc["$set"])
            return type("Result", (), {"upserted_count": len(operations), "modified_count": 0})()

    class Store(AdviceStore):
        def __init__(self, collection):
            self._owns_client = False
            self.settings = _settings()
            self._collection = collection

        @property
        def alpharank(self):
            return self._collection

    collection = Collection()
    store = Store(collection)
    rows = alpha_rank_documents(
        _panel().assign(
            rank=[1, 2, 3],
            alpha_score=[0.9, 0.8, 0.7],
            alpha_consensus=0.5,
            return_hat=0.1,
            rank_hat=0.2,
            p_top=0.3,
        ),
        "us",
        10,
    )
    store.write_alpharank(rows)
    previous = [dict(row) for row in collection.docs]
    collection.fail = True
    replacement = [dict(rows[0]), dict(rows[1])]
    replacement[0]["symbol"] = "NEW"
    replacement_stamp = rows[0]["generated_at"] + timedelta(seconds=1)
    for document in replacement:
        document["generated_at"] = replacement_stamp
        document["generation_version"] = int(replacement_stamp.timestamp() * 1000)
        document["generation_id"] = str(document["generation_version"])
    try:
        store.write_alpharank(replacement)
    except RuntimeError as exc:
        assert str(exc) == "mongo write failed"
    else:
        raise AssertionError("bulk failure must propagate")
    assert collection.docs == previous


def test_older_alpharank_generation_cannot_overwrite_or_delete_newer():
    class Collection:
        def __init__(self):
            self.docs = []

        def find_one(self, query, sort=None):
            matches = [
                row
                for row in self.docs
                if row["market"] == query["market"]
                and row["date"] == query["date"]
                and row["signal_model"] == query["signal_model"]
            ]
            return max(matches, key=lambda row: row["generated_at"]) if matches else None

        def bulk_write(self, operations, ordered=False):
            for operation in operations:
                symbol = operation._filter["symbol"]
                existing = next((row for row in self.docs if row["symbol"] == symbol), None)
                if "$set" in operation._doc and existing is not None:
                    if operation._filter["$or"][0]["generated_at"]["$lte"] >= existing["generated_at"]:
                        existing.update(operation._doc["$set"])
                elif operation._upsert and existing is None:
                    self.docs.append(dict(operation._doc.get("$set", {})))
            return type("Result", (), {"upserted_count": 0, "modified_count": 0})()

        def delete_many(self, query):
            def matches(row, clause):
                for key, value in clause.items():
                    if isinstance(value, dict) and "$lt" in value:
                        if row[key] >= value["$lt"]:
                            return False
                    elif isinstance(value, dict) and "$nin" in value:
                        if row[key] in value["$nin"]:
                            return False
                    elif row.get(key) != value:
                        return False
                return True

            self.docs = [
                row
                for row in self.docs
                if not (
                    row["market"] == query["market"]
                    and row["date"] == query["date"]
                    and row["signal_model"] == query["signal_model"]
                    and (
                        (
                            "generation_id" in query
                            and row["generation_id"] == query["generation_id"]
                        )
                        or (
                            "$or" in query
                            and any(matches(row, clause) for clause in query["$or"])
                        )
                    )
                )
            ]

    class Store(AdviceStore):
        def __init__(self, collection):
            self._owns_client = False
            self.settings = _settings()
            self._collection = collection

        @property
        def alpharank(self):
            return self._collection

    def rows(symbols, generation):
        frame = _panel()
        frame = frame[frame["symbol"].isin(symbols)].copy()
        return alpha_rank_documents(
            frame.assign(
                rank=range(1, len(frame) + 1),
                alpha_score=0.8,
                alpha_consensus=0.5,
                return_hat=0.1,
                rank_hat=0.2,
                p_top=0.3,
            ),
            "us",
            10,
            generated_at=generation,
        )

    collection = Collection()
    store = Store(collection)
    newer = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
    older = datetime(2026, 1, 2, 2, tzinfo=timezone.utc)
    store.write_alpharank(rows(["AAA", "CCC"], newer))
    store.write_alpharank(rows(["AAA", "BBB"], older))
    assert sorted(row["symbol"] for row in collection.docs) == ["AAA", "CCC"]
    assert {row["generated_at"] for row in collection.docs} == {newer}


def test_old_writer_reconciles_rows_inserted_after_newer_completion():
    class Collection:
        def __init__(self):
            self.docs = []
            self.interleave = None
            self.interleaved = False

        @staticmethod
        def matches(row, query):
            for key, value in query.items():
                if key == "$or":
                    if not any(Collection.matches(row, option) for option in value):
                        return False
                elif isinstance(value, dict) and "$nin" in value:
                    if row.get(key) in value["$nin"]:
                        return False
                elif isinstance(value, dict) and "$lt" in value:
                    if row.get(key) is None or row[key] >= value["$lt"]:
                        return False
                elif isinstance(value, dict) and "$exists" in value:
                    if (key in row) != value["$exists"]:
                        return False
                elif row.get(key) != value:
                    return False
            return True

        def find_one(self, query, sort=None):
            if self.interleave is not None and not self.interleaved:
                self.interleaved = True
                callback = self.interleave
                self.interleave = None
                callback()
                # Return the old pre-read result intentionally. The outer
                # writer must discover the newer generation on its post-read.
                return None
            matches = [row for row in self.docs if self.matches(row, query)]
            return max(matches, key=lambda row: row["generated_at"]) if matches else None

        def bulk_write(self, operations, ordered=False):
            for operation in operations:
                symbol = operation._filter["symbol"]
                existing = next(
                    (
                        row
                        for row in self.docs
                        if row["symbol"] == symbol
                        and row["generation_id"] == operation._filter["generation_id"]
                    ),
                    None,
                )
                if "$set" in operation._doc and existing is not None:
                    existing.update(operation._doc["$set"])
                elif operation._upsert and existing is None:
                    self.docs.append(dict(operation._doc.get("$set", {})))
            return type("Result", (), {"upserted_count": 0, "modified_count": 0})()

        def delete_many(self, query):
            self.docs = [row for row in self.docs if not self.matches(row, query)]

    class Store(AdviceStore):
        def __init__(self, collection):
            self._owns_client = False
            self.settings = _settings()
            self._collection = collection

        @property
        def alpharank(self):
            return self._collection

    def rows(symbols, generation):
        frame = _panel()
        frame = frame[frame["symbol"].isin(symbols)].copy()
        return alpha_rank_documents(
            frame.assign(
                rank=range(1, len(frame) + 1),
                alpha_score=0.8,
                alpha_consensus=0.5,
                return_hat=0.1,
                rank_hat=0.2,
                p_top=0.3,
            ),
            "us",
            10,
            generated_at=generation,
        )

    collection = Collection()
    store = Store(collection)
    newer = datetime(2026, 1, 2, 3, tzinfo=timezone.utc)
    older = datetime(2026, 1, 2, 2, tzinfo=timezone.utc)
    newer_rows = rows(["AAA", "CCC"], newer)
    collection.interleave = lambda: store.write_alpharank(newer_rows)
    store.write_alpharank(rows(["AAA", "BBB"], older))
    assert sorted(row["symbol"] for row in collection.docs) == ["AAA", "CCC"]
    assert {row["generated_at"] for row in collection.docs} == {newer}


def test_global_mongo_failure_marks_alpha_publication_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "build_inference_dataset", lambda **kwargs: _panel())
    monkeypatch.setattr(daily, "select_picks", lambda scored, **kwargs: scored.iloc[0:0].copy())
    pivot = _panel().assign(
        score=0.1,
        action="HOLD",
        swing_pos_hat=0.5,
        p_up=0.5,
        exp_gain_pct=0.0,
        exp_loss_pct=0.0,
        exp_amplitude_pct=0.0,
    )
    alpha = _panel().assign(
        return_hat=0.1,
        rank_hat=0.2,
        p_top=0.3,
        alpha_score=0.5,
        alpha_consensus=0.5,
    )
    run = run_daily(
        end_date="2026-01-02",
        settings=_settings(),
        paths=Paths(collector_data=tmp_path, training_data=tmp_path, artifacts=tmp_path / "a"),
        model_loader=lambda **kwargs: (object(), {}),
        panel_scorer=lambda *args, **kwargs: pivot,
        alpha_model_loader=lambda **kwargs: (object(), {}),
        alpha_scorer=lambda *args, **kwargs: alpha[
            ["return_hat", "rank_hat", "p_top", "alpha_score", "alpha_consensus"]
        ],
        store_factory=lambda settings: (_ for _ in ()).throw(
            RuntimeError("mongo unavailable")
        ),
        notifier=lambda status, settings: True,
    )
    assert run.status == STATUS_SUCCESS
    assert run.alpha_execution_status == "success"
    assert run.alpha_publication_status == "failed"
    assert run.alpha_error == "mongo unavailable"
    assert run.delivered is False
    alpha_status = run.job_status()["results"]["alpha_rank"]
    assert alpha_status["status"] == "failed"
    assert alpha_status["execution_status"] == "success"
    assert alpha_status["publication_status"] == "failed"
    assert alpha_status["delivered"] is False
