"""MongoDB persistence for pivot picks.

``pivot_picks``
    One document per ``(market, date, signal_model, symbol, event)`` — the
    signals that actually fired, not the cross-section they were detected in.
    Upserted, so a re-run of the same session corrects itself rather than
    duplicating.
Neither touches the legacy ``predictions`` or ``picks`` collections written by
predixlab-signal. That separation is the point: the old output stays queryable
while the pivot pipeline takes over, and a bad pivot deployment cannot corrupt
data the product already serves.

``pymongo`` is imported lazily so the research package keeps working — and the
test suite keeps running — without a database driver installed.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from predixlab_signals.contracts import PICK_CONTRACT, PickDocument
from predixlab_signals.settings import Settings, load_settings

#: Fields copied verbatim from a pick row into its document. The forward-return
#: block is only populated once the horizon has elapsed.
PICK_FIELDS = (
    "signal_model",
    "event",
    "pivot_event",
    "actionable",
    "direction",
    "kind",
    "rank",
    "action",
    "score",
    "confidence",
    "swing_position",
    "p_up",
    "exp_gain_pct",
    "exp_loss_pct",
    "close",
    "price",
    # What the name did over the horizon after the signal. `NaN` while the
    # horizon has not elapsed, which is the case for the most recent sessions.
    "forward_return",
    "signal_return",
    "excess_return",
    # Whether a reversal landed within a few bars of a real swing point. `None`
    # for trend entries, which are not turning points and must not be scored
    # against pivot labels.
    "pivot_hit",
)

ALPHARANK_FIELDS = (
    "score",
    "consensus",
    "predicted_excess_return",
    "predicted_rank",
    "top20_probability",
    "close",
)


def _clean(value: Any) -> Any:
    """Coerce numpy/pandas scalars into something BSON accepts.

    Every flavour of missing — ``None``, ``NaN``, ``NaT``, ``pd.NA`` — collapses
    to ``None``. The API models these fields as optional, a JSON-serialised
    ``NaN`` is invalid JSON, and a missing timestamp that fell through to
    ``str()`` would be stored as the literal string ``"NaT"``, which reads as a
    present value to every consumer.
    """
    if value is None:
        return None
    # Checked before the scalar branches: pd.NA raises on bool() and NaT is not
    # a Timestamp, so neither is caught by the type tests below.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # arrays and other non-scalars are handled below
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(as_float) or math.isinf(as_float):
        return None
    return as_float


def _normalize_generated_at(value: Any) -> tuple[datetime, int]:
    """Normalize a timestamp to UTC BSON millisecond precision."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("AlphaRank generated_at must be a valid timestamp")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    version = int(stamp.value // 1_000_000)
    normalized = datetime.fromtimestamp(version / 1000, tz=timezone.utc)
    return normalized, version


def pick_documents(
    picks: pd.DataFrame,
    market: str,
    horizon: int,
    tag: str = "default",
    generated_at: datetime | None = None,
) -> list[PickDocument]:
    """Turn the selected pick list into upsertable documents.

    Keyed by model and event identity rather than by rank: a rank is a position
    in today's list, so keying on it would let a re-run that reordered the book
    leave the previous ordering behind as duplicates.
    """
    if picks is None or picks.empty:
        return []
    PICK_CONTRACT.validate(picks, allow_empty=False)

    stamp = generated_at or datetime.now(timezone.utc)
    docs: list[PickDocument] = []
    for row in picks.to_dict(orient="records"):
        date = row.get("date")
        doc: dict[str, Any] = {
            "market": market.lower(),
            "date": str(pd.Timestamp(date).date()) if date is not None else None,
            "symbol": str(row.get("symbol", "")).upper(),
            "signal_model": str(row.get("signal_model") or "pivot"),
            "horizon": int(horizon),
            "tag": tag,
            "generated_at": stamp,
        }
        for field in PICK_FIELDS:
            if field in row:
                value = _clean(row[field])
                # Ranks and holding periods are counts; a float rank would sort
                # correctly but reads badly in an API response.
                if field in ("rank", "bars_held") and value is not None:
                    value = int(value)
                doc[field] = value
        doc.setdefault("expected_hold_days", int(horizon))
        docs.append(doc)
    return docs


def pick_counts(picks: pd.DataFrame) -> dict[str, int]:
    """Published events by type, for job status and email."""
    if picks is None or picks.empty or "event" not in picks.columns:
        return {}
    return {str(k): int(v) for k, v in picks["event"].value_counts().items()}


def alpha_rank_documents(
    rows: pd.DataFrame,
    market: str,
    horizon: int,
    tag: str = "default",
    generated_at: datetime | None = None,
    generation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Serialize the stable AlphaRank publication contract."""
    if rows is None or rows.empty:
        return []
    required = {
        "date",
        "symbol",
        "rank",
        "alpha_score",
        "alpha_consensus",
        "return_hat",
        "rank_hat",
        "p_top",
        "close",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"AlphaRank rows are missing required columns: {missing}")
    stamp, generation_version = _normalize_generated_at(
        generated_at or datetime.now(timezone.utc)
    )
    generation_id = str(generation_id or uuid.uuid4())
    docs: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        bounded = {
            "score": _bounded_probability(row["alpha_score"], "score"),
            "consensus": _bounded_probability(
                row["alpha_consensus"], "consensus"
            ),
            "predicted_rank": _bounded_probability(row["rank_hat"], "predicted_rank"),
            "top20_probability": _bounded_probability(row["p_top"], "top20_probability"),
        }
        predicted_excess_return = _finite_number(
            row["return_hat"], "predicted_excess_return"
        )
        doc: dict[str, Any] = {
            "market": str(market).lower(),
            "date": str(pd.Timestamp(row["date"]).date()),
            "symbol": str(row["symbol"]).upper(),
            "rank": int(row["rank"]),
            "horizon": int(horizon),
            "tag": str(tag),
            "generated_at": stamp,
            "generation_id": generation_id,
            "generation_version": generation_version,
            "signal_model": "alpha_rank",
            "score": bounded["score"],
            "consensus": bounded["consensus"],
            "predicted_excess_return": predicted_excess_return,
            "predicted_rank": bounded["predicted_rank"],
            "top20_probability": bounded["top20_probability"],
            "close": _clean(row["close"]),
        }
        docs.append(doc)
    return docs


def _bounded_probability(value: Any, field: str) -> float:
    """Validate and normalize a required probability/percentile value."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AlphaRank {field} must be a finite number in [0, 1]") from exc
    if not math.isfinite(number):
        raise ValueError(f"AlphaRank {field} must be a finite number in [0, 1]")
    return float(min(1.0, max(0.0, number)))


def _finite_number(value: Any, field: str) -> float:
    """Validate a required signed numeric value without bounding its range."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AlphaRank {field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"AlphaRank {field} must be a finite number")
    return number


class AdviceStore:
    """Thin, synchronous MongoDB writer for pivot picks and job telemetry."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None):
        self.settings = settings or load_settings()
        if client is None:
            from pymongo import MongoClient  # imported lazily; see module docstring

            client = MongoClient(self.settings.mongodb_url)
            self._owns_client = True
        else:
            self._owns_client = False
        self.client = client
        self.db = client[self.settings.database_name]

    # ------------------------------------------------------------------
    @property
    def picks(self):
        return self.db[self.settings.picks_collection]

    @property
    def alpharank(self):
        return self.db[self.settings.alpharank_collection]

    @property
    def alpharank_generations(self):
        return self.db[self.settings.alpharank_generations_collection]

    @property
    def jobs(self):
        return self.db[self.settings.jobs_collection]

    # ------------------------------------------------------------------
    def ensure_indexes(self) -> None:
        """Create the query indexes the API depends on.

        Failures are logged rather than raised: on Cosmos DB for MongoDB an
        index may already exist with different options, and that must not take
        the nightly run down.
        """
        from pymongo import ASCENDING, DESCENDING

        def ensure_index(
            collection: Any,
            keys: list[tuple[str, int]],
            options: dict[str, Any],
            *,
            required: bool = False,
        ) -> None:
            name = options["name"]
            try:
                collection.create_index(keys, **options)
            except Exception as exc:  # pragma: no cover - driver/server specific
                if required:
                    raise RuntimeError(
                        f"required AlphaRank index {name!r} could not be created"
                    ) from exc
                _log_warning(f"index creation on {collection.name} failed: {exc}")
                return
            if not required:
                return
            inspector = getattr(collection, "index_information", None)
            if inspector is None:
                raise RuntimeError(
                    f"required AlphaRank index {name!r} cannot be verified"
                )
            try:
                indexes = inspector()
                actual = indexes.get(name)
                actual_keys = (
                    [tuple(item) for item in actual.get("key", [])]
                    if actual is not None
                    else []
                )
            except Exception as exc:  # pragma: no cover - driver/server specific
                raise RuntimeError(
                    f"required AlphaRank index {name!r} cannot be verified"
                ) from exc
            if (
                actual is None
                or not actual.get("unique", False)
                or actual_keys != [tuple(item) for item in keys]
            ):
                raise RuntimeError(
                    f"required AlphaRank index {name!r} is missing or not unique"
                )

        specs = (
            (
                self.picks,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("signal_model", ASCENDING),
                    ("symbol", ASCENDING),
                    ("event", ASCENDING),
                ],
                {"unique": True, "name": "market_date_model_symbol_event"},
            ),
            (
                self.picks,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("signal_model", ASCENDING),
                    ("direction", ASCENDING),
                    ("rank", ASCENDING),
                ],
                {"name": "market_date_model_direction_rank"},
            ),
            (
                self.picks,
                [("market", ASCENDING), ("symbol", ASCENDING), ("date", DESCENDING)],
                {"name": "market_symbol_date"},
            ),
        )
        for collection, keys, options in specs:
            ensure_index(collection, keys, options)

        # The pre-manifest deployment used a unique index without
        # generation_id. Remove that obsolete constraint before staging a
        # second generation for the same symbol.
        drop_index = getattr(self.alpharank, "drop_index", None)
        if drop_index is not None:
            try:
                drop_index("market_date_model_symbol")
            except Exception as exc:
                inspector = getattr(self.alpharank, "index_information", None)
                if inspector is not None and "market_date_model_symbol" in inspector():
                    raise RuntimeError(
                        "obsolete AlphaRank unique index could not be removed"
                    ) from exc

        alpha_specs = (
            (
                self.alpharank,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("signal_model", ASCENDING),
                    ("generation_id", ASCENDING),
                    ("symbol", ASCENDING),
                ],
                {"unique": True, "name": "market_date_model_generation_symbol"},
            ),
            (
                self.alpharank,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("rank", ASCENDING),
                ],
                {"name": "market_date_rank"},
            ),
            (
                self.alpharank,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("signal_model", ASCENDING),
                    ("generation_version", DESCENDING),
                ],
                {"name": "market_date_model_generation"},
            ),
            (
                self.alpharank,
                [
                    ("market", ASCENDING),
                    ("signal_model", ASCENDING),
                    ("symbol", ASCENDING),
                    ("date", DESCENDING),
                    ("generation_id", ASCENDING),
                ],
                {"name": "market_model_symbol_date_generation"},
            ),
        )
        for index, (collection, keys, options) in enumerate(alpha_specs):
            ensure_index(collection, keys, options, required=index == 0)

        manifest_specs = (
            (
                self.alpharank_generations,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("signal_model", ASCENDING),
                ],
                {"unique": True, "name": "market_date_model_manifest"},
            ),
            (
                self.alpharank_generations,
                [
                    ("market", ASCENDING),
                    ("date", DESCENDING),
                    ("active_generation_version", DESCENDING),
                ],
                {"name": "market_date_active_generation"},
            ),
        )
        for index, (collection, keys, options) in enumerate(manifest_specs):
            ensure_index(collection, keys, options, required=index == 0)

    # ------------------------------------------------------------------
    def write_picks(
        self,
        documents: Iterable[dict[str, Any]],
    ) -> int:
        """Upsert pick documents without deleting any existing data."""
        documents = list(documents)
        if not documents:
            return 0

        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for document in documents:
            grouped.setdefault((document["market"], document["date"]), []).append(document)
        written = 0
        for session_docs in grouped.values():
            stamps = {doc.get("generated_at") for doc in session_docs}
            if None in stamps or len(stamps) != 1:
                raise ValueError("all picks for one session need one generation")
            written += self._upsert_session(session_docs)
        return written

    def _upsert_session(self, documents: list[dict[str, Any]]) -> int:
        if not documents:
            return 0
        from pymongo import UpdateOne

        operations = [
            UpdateOne(
                {
                    "market": doc["market"],
                    "date": doc["date"],
                    "signal_model": doc.get("signal_model", "pivot"),
                    "symbol": doc["symbol"],
                    "event": doc["event"],
                },
                {"$set": doc},
                upsert=True,
            )
            for doc in documents
        ]
        written = 0
        # Cosmos DB caps request size, so the batch is chunked rather than sent
        # as one bulk write of several thousand operations.
        for start in range(0, len(operations), 500):
            chunk = operations[start : start + 500]
            result = self.picks.bulk_write(chunk, ordered=False)
            written += (result.upserted_count or 0) + (result.modified_count or 0)
        return written

    def write_alpharank(self, documents: Iterable[dict[str, Any]]) -> int:
        """Stage, compare-and-set promote, and garbage-collect AlphaRank rows."""
        documents = list(documents)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for document in documents:
            grouped.setdefault((document["market"], document["date"]), []).append(document)
        if not grouped:
            return 0

        from pymongo import UpdateOne

        try:
            manifests = self.alpharank_generations
        except AttributeError:  # lightweight test doubles from older callers
            manifests = self.alpharank
        memory_manifests: dict[tuple[str, str, str], dict[str, Any]] | None = None
        if not hasattr(manifests, "update_one"):
            # Compatibility only for pre-manifest in-memory test doubles. The
            # production AdviceStore always uses the separate Mongo collection.
            memory_manifests = self.__dict__.setdefault(
                "_alpha_manifest_memory", {}
            )

        def find_manifest(session: dict[str, Any]) -> dict[str, Any] | None:
            if memory_manifests is not None:
                # Let legacy in-memory doubles observe reads for deterministic
                # interleaving tests while keeping their manifest state local.
                before = memory_manifests.get(
                    (session["market"], session["date"], session["signal_model"])
                )
                finder = getattr(manifests, "find_one", None)
                if finder is not None:
                    finder(session)
                if before is None:
                    return None
                return memory_manifests.get(
                    (session["market"], session["date"], session["signal_model"])
                )
            finder = getattr(manifests, "find_one", None)
            return finder(session) if finder is not None else None

        def delete_generation(
            session: dict[str, Any],
            generation_id: str,
        ) -> None:
            active = find_manifest(session)
            if active and active.get("active_generation_id") == generation_id:
                return
            self.alpharank.delete_many(
                {**session, "generation_id": generation_id}
            )

        def count_generation(
            session: dict[str, Any],
            generation_id: str,
            fallback: int,
        ) -> int:
            counter = getattr(self.alpharank, "count_documents", None)
            if counter is not None:
                return int(
                    counter({**session, "generation_id": generation_id})
                )
            rows = getattr(self.alpharank, "docs", None)
            if rows is not None:
                return sum(
                    1
                    for row in rows
                    if all(row.get(key) == value for key, value in session.items())
                    and row.get("generation_id") == generation_id
                )
            return fallback

        def promote(
            session: dict[str, Any],
            manifest: dict[str, Any],
        ) -> bool:
            version = manifest["active_generation_version"]
            if memory_manifests is not None:
                key = (session["market"], session["date"], session["signal_model"])
                current = memory_manifests.get(key)
                if (
                    current is None
                    or current["active_generation_version"] < version
                ):
                    memory_manifests[key] = manifest
                current = memory_manifests.get(key)
                return bool(
                    current
                    and current["active_generation_id"]
                    == manifest["active_generation_id"]
                    and current["active_generation_version"] == version
                )
            update = manifests.update_one(
                {
                    **session,
                    "$or": [
                        {"active_generation_version": {"$lt": version}},
                        {"active_generation_version": {"$exists": False}},
                    ],
                },
                {"$set": manifest},
                upsert=False,
            )
            if not getattr(update, "matched_count", 0):
                try:
                    manifests.update_one(
                        session,
                        {"$setOnInsert": manifest},
                        upsert=True,
                    )
                except Exception as exc:
                    # A concurrent winner can race the insert-only CAS and
                    # surface a duplicate-key error. Re-read below so the
                    # loser can clean its staged generation without treating
                    # the active winner as corrupted.
                    if exc.__class__.__name__ != "DuplicateKeyError":
                        raise
            current = find_manifest(session)
            return bool(
                current
                and current.get("active_generation_id") == manifest["active_generation_id"]
                and current.get("active_generation_version") == version
            )

        written = 0
        for (market, date), session_docs in grouped.items():
            generation_ids = {doc.get("generation_id") for doc in session_docs}
            generation_versions = {
                doc.get("generation_version") for doc in session_docs
            }
            generated_at = {doc.get("generated_at") for doc in session_docs}
            if (
                None in generation_ids
                or None in generation_versions
                or len(generation_ids) != 1
                or len(generation_versions) != 1
                or len(generated_at) != 1
            ):
                raise ValueError(
                    "all AlphaRank rows for one session need one generation"
                )
            generation_id = next(iter(generation_ids))
            generation_version = int(next(iter(generation_versions)))
            generation_stamp = next(iter(generated_at))
            session = {
                "market": market,
                "date": date,
                "signal_model": "alpha_rank",
            }
            manifest = {
                **session,
                "active_generation_id": generation_id,
                "active_generation_version": generation_version,
                "generated_at": generation_stamp,
                "row_count": len(session_docs),
                "horizon": session_docs[0].get("horizon"),
                "tag": session_docs[0].get("tag"),
            }
            active = find_manifest(session)
            if active and (
                active.get("active_generation_version", -1) > generation_version
                or (
                    active.get("active_generation_version") == generation_version
                    and active.get("active_generation_id") != generation_id
                )
            ):
                delete_generation(session, generation_id)
                continue
            if active and active.get("active_generation_id") == generation_id:
                written += count_generation(session, generation_id, len(session_docs))
                continue

            # A previous failed attempt may have left a partial staged set.
            # It is not active, so clear it before rebuilding the generation.
            delete_generation(session, generation_id)
            operations = [
                UpdateOne(
                    {
                        **session,
                        "generation_id": generation_id,
                        "symbol": doc["symbol"],
                    },
                    {"$set": doc},
                    upsert=True,
                )
                for doc in session_docs
            ]
            try:
                for start in range(0, len(operations), 500):
                    result = self.alpharank.bulk_write(
                        operations[start : start + 500],
                        ordered=False,
                    )
                    written += (result.upserted_count or 0) + (
                        result.modified_count or 0
                    )
                staged_count = count_generation(
                    session, generation_id, len(session_docs)
                )
                if staged_count != len(session_docs):
                    raise RuntimeError(
                        f"AlphaRank generation {generation_id} staged "
                        f"{staged_count} rows; expected {len(session_docs)}"
                    )
            except Exception:
                delete_generation(session, generation_id)
                raise

            try:
                promoted = promote(session, manifest)
            except Exception:
                delete_generation(session, generation_id)
                raise
            if not promoted:
                delete_generation(session, generation_id)
                continue

            # Keep newer staged generations intact so they can still promote;
            # only generations strictly older than the committed manifest are
            # eligible for cleanup.
            self.alpharank.delete_many(
                {
                    **session,
                    "$or": [
                        {"generation_version": {"$lt": generation_version}},
                        {"generation_version": {"$exists": False}},
                    ],
                }
            )
            written = len(session_docs)
        return written


    def write_job_status(self, job_status: dict[str, Any]) -> None:
        """Append to the shared ``jobs`` collection every worker writes to."""
        payload = dict(job_status)
        payload.setdefault("created_at", datetime.now(timezone.utc))
        self.jobs.insert_one(payload)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "AdviceStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def _log_warning(message: str) -> None:
    try:
        from loguru import logger

        logger.warning(message)
    except ImportError:  # pragma: no cover - loguru ships with the job extra
        import logging

        logging.getLogger(__name__).warning(message)
