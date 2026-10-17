"""The nightly pipeline: collector output in, published advice out.

Ordering is the whole design. The job runs as an Azure Container Apps job whose
*init container* is predixlab-collector. Successful collection does not guarantee
fresh vendor data, so the shared input is validated before any model runs.

Steps
-----
1. Unless an explicit end date was supplied, stop early when yesterday is not
    present in the market's trading calendar.
2. Validate target-session completeness and freshness against active membership
   and the previous trading session, then build a bounded labels-free table.
3. Load the production model. This job **never trains**: models are fitted and
   validated on a workstation and copied onto the shared root by hand, so a
   model running in production is always one a human reviewed. A missing or
   unloadable model fails the run.
4. Score the latest bar into full recommendations.
5. Write the session's CSVs to the shared root and publish Pivot plus AlphaRank
    into MongoDB. A quiet Pivot session writes no pick documents.
6. After both database publications commit, enqueue the daily campaign event.
7. Record shared job status and notify.

Every step is separately testable and the whole thing is a pure function of
``(data tree, settings)`` apart from the two side-effecting sinks, which are
injectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from predixlab_signals import __version__
from predixlab_signals.config import PATHS, Paths
from predixlab_signals.data import build_inference_dataset
from predixlab_signals.data.dataset import DEFAULT_INFERENCE_HISTORY
from predixlab_signals.data_quality import DataQualityError
from predixlab_signals.model import (
    AlphaRankEnsemble,
    PivotModel,
    alpha_rank_model_path,
    model_path,
)
from predixlab_signals.publish.mongo import (
    AdviceStore,
    alpha_rank_documents,
    pick_documents,
)
from predixlab_signals.publish.notification import send_email_notification
from predixlab_signals.publish.campaign import enqueue_daily_signals_campaign
from predixlab_signals.settings import JOB_DISPLAY_NAME, JOB_NAME, Settings, load_settings
from predixlab_signals.signals.scoring import score_panel
from predixlab_signals.signals.picks import (
    PICK_COLUMNS,
    select_picks,
    summarise_picks,
)
from predixlab_signals.signals.models.pivot.adapter import PIVOT_SIGNAL_MODEL
from predixlab_signals.signals.pipeline import resolve_signal_model
from predixlab_signals.signals.specs import SignalModelSpec
from predixlab_signals.signals.models.pivot import (
    BASE_BREAKOUT,
    BOTTOM_TURN,
    BUY_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
)

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

#: How many names each side of the book the notification lists.
EMAIL_PICKS = 10


def _log(message: str, level: str = "info") -> None:
    try:
        from loguru import logger

        getattr(logger, level)(message)
    except ImportError:  # pragma: no cover - loguru ships with the job extra
        import logging

        getattr(logging.getLogger(__name__), level)(message)


def is_trading_day(
    date: datetime | None = None, paths: Paths = PATHS, market: str = "us"
) -> bool:
    """Return whether ``date`` appears in the market's Collector calendar."""
    date = date or datetime.now()
    calendar_path = paths.calendar_path(market)

    try:
        if not calendar_path.exists():
            _log(f"{market.upper()} calendar file not found at {calendar_path}", "error")
            return False

        calendar = pd.read_csv(
            calendar_path,
            header=None,
            names=["date"],
            dtype=str,
            skip_blank_lines=True,
            encoding="utf-8",
        )
        dates = calendar["date"].str.strip()
        valid_dates = dates[dates.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
        return date.strftime("%Y-%m-%d") in set(valid_dates)
    except Exception as exc:  # noqa: BLE001 - a bad calendar means do not run
        _log(f"Error checking trading day from calendar file: {exc}", "error")
        return False


def read_trading_days(market: str = "us", paths: Paths = PATHS) -> set[str]:
    """Trading sessions from the collector calendar, as ``YYYY-MM-DD`` strings."""
    path = paths.calendar_path(market)
    if not path.exists():
        _log(f"Calendar not found at {path}; trading-day check disabled", "warning")
        return set()
    days = set()
    for line in path.read_text().split():
        token = line.strip()
        if len(token) == 10 and token[4] == "-" and token[7] == "-":
            days.add(token)
    return days


def _event_rows(picks: pd.DataFrame, events: tuple[str, ...], top: int) -> list[dict[str, Any]]:
    """Serialise one class of published event for the email."""
    if picks is None or picks.empty or "event" not in picks.columns:
        return []
    side = picks[picks["event"].isin(events)].head(top)
    if side.empty:
        return []
    columns = ["rank", "symbol", "event", "action", "score", "confidence", "price"]
    rows = []
    for record in side[[c for c in columns if c in side.columns]].to_dict(orient="records"):
        rows.append(
            {
                key: (None if isinstance(value, float) and pd.isna(value) else value)
                for key, value in record.items()
            }
        )
    return rows


@dataclass
class DailyRun:
    """Outcome of one execution, in a shape both sinks can consume."""

    market: str
    horizon: int
    tag: str
    status: str
    started_at: datetime
    signal_model: str = "pivot"
    finished_at: datetime | None = None
    date: str | None = None
    symbols_scored: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    model: dict[str, Any] | None = None
    published: int = 0
    #: The scored cross-section written under ``predictions/<date>/<market>/``.
    artifact: str | None = None
    #: The pick list written under ``picks/<date>/``.
    picks_file: str | None = None
    error: str | None = None
    data_validation: dict[str, Any] | None = None
    #: Set when the model ran but its output could not reach MongoDB. The run is
    #: still a success — the computation is correct and does not need repeating —
    #: but the product surface is stale, so the process must not exit 0 silently.
    publish_error: str | None = None
    #: The published short list. The scored cross-section it came from is not
    #: retained here: it goes to the predictions CSV and nowhere else.
    picks: pd.DataFrame | None = None
    #: AlphaRank's latest-session top-N list and audit artifact.
    alpha_picks: pd.DataFrame | None = None
    alpha_model: dict[str, Any] | None = None
    alpha_tag: str | None = None
    alpha_artifact: str | None = None
    alpha_published: int = 0
    alpha_error: str | None = None
    alpha_symbols_scored: int = 0
    alpha_execution_status: str = "not_run"
    alpha_publication_status: str = "not_run"
    campaign_event_required: bool = False
    campaign_event_status: str = "disabled"
    campaign_event_error: str | None = None

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    @property
    def delivered(self) -> bool:
        """Whether this run's picks actually reached the collections."""
        if self.status == STATUS_SKIPPED:
            return True
        if self.status != STATUS_SUCCESS:
            return False
        database_delivered = self.publish_error is None and self.alpha_error is None
        campaign_delivered = (
            not self.campaign_event_required
            or self.campaign_event_status in {"sent", "skipped"}
        )
        return database_delivered and campaign_delivered

    def job_status(self) -> dict[str, Any]:
        """The document written to the shared ``jobs`` collection."""
        finished = self.finished_at or datetime.now(timezone.utc)
        if self.alpha_execution_status in {"failed", "not_run", "skipped"}:
            alpha_status = self.alpha_execution_status
        elif self.alpha_publication_status == "failed":
            alpha_status = "failed"
        elif self.alpha_publication_status in {"pending", "not_run"}:
            alpha_status = "pending"
        else:
            alpha_status = "success"
        status: dict[str, Any] = {
            "job_name": JOB_NAME,
            "job_display_name": JOB_DISPLAY_NAME,
            "status": self.status,
            "start_time": self.started_at.isoformat(),
            "end_time": finished.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "version": __version__,
            "results": {
                "market": self.market,
                "date": self.date,
                "horizon": self.horizon,
                "tag": self.tag,
                "signal_model": self.signal_model,
                "symbols_scored": self.symbols_scored,
                "data_validation": self.data_validation,
                "counts": self.counts,
                "published": self.published,
                "alpha_artifact": self.alpha_artifact,
                "alpha_published": self.alpha_published,
                "alpha_symbols_scored": self.alpha_symbols_scored,
                "alpha_model": self.alpha_model,
                "alpha_tag": self.alpha_tag,
                "alpha_rank": {
                    "status": alpha_status,
                    "execution_status": self.alpha_execution_status,
                    "publication_status": self.alpha_publication_status,
                    "delivered": (
                        self.alpha_execution_status == "success"
                        and self.alpha_publication_status in {"success", "skipped"}
                    ),
                    "symbols_scored": self.alpha_symbols_scored,
                    "top_n": int(self.alpha_picks.shape[0]) if self.alpha_picks is not None else 0,
                    "published": self.alpha_published,
                    "artifact": self.alpha_artifact,
                    "model": self.alpha_model,
                    "tag": self.alpha_tag,
                    "error": self.alpha_error,
                },
                "campaign_event": {
                    "required": self.campaign_event_required,
                    "status": self.campaign_event_status,
                    "error": self.campaign_event_error,
                },
                "picks_file": self.picks_file,
                "artifact": self.artifact,
                "model": self.model,
                "top_buys": _event_rows(
                    self.picks,
                    (BUY_TURN, BOTTOM_TURN, TREND_BUY, BASE_BREAKOUT),
                    EMAIL_PICKS,
                ),
                "top_sells": _event_rows(
                    self.picks,
                    (SELL_TURN, TOP_TURN),
                    EMAIL_PICKS,
                ),
            },
        }
        if self.error:
            status["error"] = self.error
        if self.publish_error:
            status["warning"] = f"MongoDB publication failed: {self.publish_error}"
        if self.alpha_error:
            status["warning"] = (
                f"{status.get('warning', '')}; AlphaRank publication failed: "
                f"{self.alpha_error}"
            ).lstrip("; ")
        if self.campaign_event_error:
            status["warning"] = (
                f"{status.get('warning', '')}; Campaign event publication failed: "
                f"{self.campaign_event_error}"
            ).lstrip("; ")
        return status

class MissingProductionModel(RuntimeError):
    """Raised when the scheduled job has no validated model to score with.

    Deliberately fatal. The job used to train one on the spot when the file was
    absent or stale, which quietly defeats the entire release process: a model
    fitted at 02:00 UTC inside a container has passed no validation, yet its
    signals would reach the product surface indistinguishable from a reviewed
    one. A loud failure is recoverable in minutes; a silent unreviewed model is
    not detectable at all.
    """


class MissingProductionAlphaRankModel(MissingProductionModel):
    """Raised when the validated production AlphaRank ensemble is unavailable."""


def load_production_model(
    market: str,
    horizon: int,
    tag: str,
    paths: Paths = PATHS,
) -> tuple[PivotModel, dict[str, Any]]:
    """Load the production model. Never trains.

    Models are trained and validated on a workstation and copied onto the shared
    root by hand. The scheduled job is a consumer of that artifact and nothing
    else.

    """
    path = model_path(market, horizon, tag, paths)
    if not path.exists():
        raise MissingProductionModel(
            f"No production model at {path}. Train and validate one locally, then "
            f"copy the candidate there:\n"
            f"  predixlab-signals train --market={market} --horizon={horizon} --tag={tag}\n"
            f"  predixlab-signals evaluate --market={market} --horizon={horizon} --tag={tag}\n"
            f"  cp <candidate> {path}"
        )

    try:
        model = PivotModel.load(path, verify_integrity=True)
    except Exception as exc:
        raise MissingProductionModel(
            f"The production model at {path} could not be loaded ({exc}). It is "
            "corrupt or was written by an incompatible build; copy a known-good "
            "model over it rather than letting the job fit its own."
        ) from exc

    meta = dict(model.trained_on or {})
    meta.update({"retrained": False, "path": str(path)})
    meta.setdefault("features", len(model.features))
    _log(f"Scoring with {path.name}")
    return model, meta


def load_production_alpha_rank_model(
    market: str,
    horizon: int,
    tag: str,
    paths: Paths = PATHS,
) -> tuple[AlphaRankEnsemble, dict[str, Any]]:
    """Load the independently validated 36/60/96-month production ensemble."""
    path = alpha_rank_model_path(market, horizon, tag, paths)
    if not path.exists():
        raise MissingProductionAlphaRankModel(
            f"No production AlphaRank model at {path}. Train and independently "
            "validate the formal 36/60/96-month ensemble, then install it there."
        )
    try:
        model = AlphaRankEnsemble.load(path, verify_integrity=True)
        if not isinstance(model, AlphaRankEnsemble):
            raise TypeError(f"expected AlphaRankEnsemble, got {type(model).__name__}")
        if tuple(model.windows_months) != (36, 60, 96):
            raise ValueError(
                f"expected formal windows (36, 60, 96), got {model.windows_months}"
            )
        if int(model.horizon) != int(horizon):
            raise ValueError(
                f"artifact horizon {model.horizon} does not match requested {horizon}"
            )
        if set(model.models) != {"36", "60", "96"}:
            raise ValueError("ensemble does not contain all formal production windows")
    except Exception as exc:
        raise MissingProductionAlphaRankModel(
            f"The production AlphaRank model at {path} could not be loaded ({exc}). "
            "It is corrupt, incompatible, or not the independently validated "
            "formal ensemble; install a known-good artifact."
        ) from exc
    meta = dict(model.trained_on or {})
    meta.update(
        {
            "retrained": False,
            "path": str(path),
            "windows_months": [36, 60, 96],
            "model_type": "alpha_rank_ensemble",
        }
    )
    return model, meta


def write_artifact(
    scored: pd.DataFrame,
    market: str,
    target: str,
    paths: Paths = PATHS,
    signal_model: str = "pivot",
) -> Path:
    """Persist the day's scored cross-section for audit.

    This is the model artifact the product deliberately does not publish to the
    API. It is kept so a published event can always be traced back to the scores
    it was derived from, including the names that did *not* fire.

    It lands in the shared data root under predixlab-signal's ``predictions``
    convention rather than in pivot's own artifacts tree: the two jobs mount the
    same share, and an operator looking for "what did the models say on the 3rd"
    should find both answers in one directory. ``horizon`` stays in the frame
    rather than the filename so the path matches signal's exactly.
    """
    out_dir = paths.daily_predictions_dir(market, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = scored[scored["date"] == pd.Timestamp(target)]
    path = out_dir / f"{market.lower()}_{signal_model}.csv"
    day.to_csv(path, index=False)
    return path


def write_alpha_rank_artifact(
    scored: pd.DataFrame,
    market: str,
    target: str,
    paths: Paths = PATHS,
) -> Path:
    """Persist the latest AlphaRank cross-section beside the Pivot artifact."""
    out_dir = paths.daily_predictions_dir(market, target)
    out_dir.mkdir(parents=True, exist_ok=True)
    day = scored[scored["date"] == pd.Timestamp(target)]
    path = out_dir / f"{market.lower()}_alpharank.csv"
    day.to_csv(path, index=False)
    return path


def select_alpha_rank_rows(
    scored: pd.DataFrame,
    target: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Select a deterministic latest-session AlphaRank top-N."""
    if top_n <= 0:
        raise ValueError("AlphaRank top_n must be greater than zero")
    required = {
        "date",
        "symbol",
        "close",
        "return_hat",
        "rank_hat",
        "p_top",
        "alpha_score",
        "alpha_consensus",
    }
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"AlphaRank scoring output is missing required columns: {missing}")
    day = scored[scored["date"] == pd.Timestamp(target)].copy()
    if day.empty:
        return day.assign(rank=pd.Series(dtype="int64"))
    day["symbol"] = day["symbol"].astype(str).str.upper()
    day = day.sort_values(
        ["alpha_score", "symbol"],
        ascending=[False, True],
        kind="mergesort",
        na_position="last",
    ).head(top_n).reset_index(drop=True)
    day.insert(0, "rank", range(1, len(day) + 1))
    return day


def write_picks_file(
    picks: pd.DataFrame,
    market: str,
    target: str,
    paths: Paths = PATHS,
    signal_model: str = "pivot",
) -> Path:
    """Persist the published pick list beside signal's, in the shared root.

    MongoDB stays the source of truth for the dashboard; this file is the
    on-share record, which is what the archiver snapshots and what makes a run
    reproducible after the collection has been re-published.

    A session that fired nothing still writes the file, with headers only. An
    absent file cannot be told apart from a job that never ran, and "no turning
    points today" is a real answer the product makes deliberately.
    """
    out_dir = paths.daily_picks_dir(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{market.lower()}_{signal_model}.csv"
    columns = [c for c in PICK_COLUMNS if picks is None or c in picks.columns]
    frame = picks[columns] if picks is not None and not picks.empty else pd.DataFrame(columns=PICK_COLUMNS)
    frame.to_csv(path, index=False)
    return path


def run_daily(
    settings: Settings | None = None,
    dry_run: bool = False,
    store_factory: Callable[[Settings], AdviceStore] | None = None,
    notifier: Callable[[dict[str, Any], Settings], bool] | None = None,
    paths: Paths = PATHS,
    clock: Callable[[], datetime] | None = None,
    end_date: str | None = None,
    signal_model: str | SignalModelSpec = PIVOT_SIGNAL_MODEL,
    model_loader: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    panel_scorer: Callable[..., pd.DataFrame] | None = None,
    alpha_model_loader: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
    alpha_scorer: Callable[..., pd.DataFrame] | None = None,
    campaign_publisher: Callable[[str, str, Settings], Any] | None = None,
) -> DailyRun:
    """Run the nightly pipeline end to end.

    Reads the shared data root (``PREDIXLAB_DATA``, the mounted file share) and
    writes the session's two outputs back into it under predixlab-signal's
    layout — ``predictions/<date>/<market>/`` and ``picks/<date>/`` — before
    publishing the pick list to MongoDB.

    ``dry_run`` computes and writes both files but publishes nothing, which is
    how the job is smoke-tested against production data without touching the
    collections the dashboard reads.

    ``end_date`` pins the required trading session and bypasses the automatic
    yesterday check. Neither mode may fall back to an older date in the data.
    """
    settings = settings or load_settings()
    notifier = notifier or send_email_notification
    clock = clock or (lambda: datetime.now(timezone.utc))
    signal_spec = resolve_signal_model(signal_model)
    model_loader = model_loader or load_production_model
    panel_scorer = panel_scorer or score_panel
    alpha_model_loader = alpha_model_loader or load_production_alpha_rank_model
    started = clock()
    run = DailyRun(
        market=settings.market,
        horizon=settings.horizon,
        tag=settings.tag,
        status=STATUS_FAILED,
        started_at=started,
        signal_model=signal_spec.name,
        alpha_tag=settings.alpharank_tag,
        campaign_event_required=settings.campaign_events_enabled,
        campaign_event_status=(
            "pending" if settings.campaign_events_enabled else "disabled"
        ),
    )

    try:
        if settings.require_trading_day and end_date is None:
            yesterday = started - timedelta(days=1)
            if not is_trading_day(yesterday, paths, market=settings.market):
                _log("Yesterday was not a trading day. Skipping.")
                run.status = STATUS_SKIPPED
                run.error = "Yesterday was not a trading day"
                return _finish(
                    run,
                    settings,
                    store_factory,
                    notifier,
                    dry_run,
                    clock,
                    campaign_publisher,
                )

        target = str(
            pd.Timestamp(end_date).date()
            if end_date is not None else (started - timedelta(days=1)).date()
        )
        run.date = target
        history_sessions = max(
            DEFAULT_INFERENCE_HISTORY,
            settings.lookback_sessions + 500,
        )
        _log(
            f"Building {history_sessions}-session inference frame for {settings.market} "
            f"from {paths.normalized_dir(settings.market)}"
        )
        data = build_inference_dataset(
            market=settings.market,
            end=target,
            horizon=settings.horizon,
            history_sessions=history_sessions,
            paths=paths,
            validation_date=target,
            max_bad_data_ratio=settings.max_bad_data_ratio,
        )
        run.data_validation = data.attrs.get("data_validation")
        if data.empty:
            raise RuntimeError(
                "No data produced — check that predixlab-collector has populated "
                f"{paths.normalized_dir(settings.market)}"
            )
        _log(f"Inference frame contains {len(data):,} rows")
        if run.data_validation:
            quality = run.data_validation
            _log(
                f"Data validation passed for {target}: "
                f"{quality['valid_symbols']}/{quality['expected_symbols']} valid constituents; "
                f"excluded={quality['excluded_symbols']}",
                "warning" if quality["bad_symbols"] or quality["read_errors"] else "info",
            )

        model, model_meta = model_loader(
            market=settings.market,
            horizon=settings.horizon,
            tag=settings.tag,
            paths=paths,
        )
        run.model = model_meta

        # Turning-point detection is sequential, so a trailing panel is scored
        # rather than a single bar: the range window, the three-session setup
        # tolerance and the refractory period all read the preceding sessions.
        scored = panel_scorer(
            model, data, lookback=settings.lookback_sessions, as_of=pd.Timestamp(target)
        )
        if scored.empty:
            raise RuntimeError(f"Nothing scored for {target}")

        events = select_picks(
            scored,
            max_per_side=settings.max_per_side or None,
            model=signal_spec,
        )
        today = events[events["date"] == pd.Timestamp(target)] if not events.empty else events

        run.date = target
        run.symbols_scored = int((scored["date"] == pd.Timestamp(target)).sum())
        run.artifact = str(
            write_artifact(
                scored,
                settings.market,
                target,
                paths,
                signal_model=signal_spec.name,
            )
        )

        if today.empty:
            # A quiet session is a real answer, not a failure: 14 of 754
            # walk-forward sessions produced no event at all. Publishing nothing
            # is the honest outcome, and the run is still a success.
            run.counts = summarise_picks(today, model=signal_spec)
            run.picks = today
            run.status = STATUS_SUCCESS
            _log(f"No turning points on {target}; {run.symbols_scored:,} names scored")
        else:
            picks = today.reset_index(drop=True)
            if signal_spec.event_enricher is not None:
                picks = signal_spec.event_enricher(picks, scored)
            picks = picks[[column for column in PICK_COLUMNS if column in picks]]
            run.counts = summarise_picks(picks, model=signal_spec)
            run.picks = picks
            run.status = STATUS_SUCCESS
            _log(
                f"Picks for {target}: {len(picks)} events from "
                f"{run.symbols_scored:,} names scored, {run.counts}"
            )

        # Written for every successful session, including quiet ones: the file
        # is the record that the job ran and reached a conclusion.
        run.picks_file = str(
            write_picks_file(
                run.picks,
                settings.market,
                target,
                paths,
                signal_model=signal_spec.name,
            )
        )
        _log(f"Wrote {run.artifact} and {run.picks_file}")

        # AlphaRank consumes the exact same labels-free panel. Pivot has
        # completed first, including its artifact, so model execution order is
        # explicit and the two models can never drift onto different inputs.
        if signal_spec.name == "pivot":
            run.alpha_execution_status = "running"
            alpha_model, alpha_meta = alpha_model_loader(
                market=settings.market,
                horizon=settings.horizon,
                tag=settings.alpharank_tag,
                paths=paths,
            )
            run.alpha_tag = settings.alpharank_tag
            run.alpha_model = {**alpha_meta, "tag": settings.alpharank_tag}
            if alpha_scorer is None:
                alpha_signal = alpha_model.predict(data)
            else:
                alpha_signal = alpha_scorer(alpha_model, data)
            if alpha_signal.empty:
                raise RuntimeError(f"AlphaRank produced no scores for {target}")
            alpha_context = data[
                [column for column in ("date", "symbol", "close") if column in data.columns]
            ].copy()
            # Custom scorers may already return context; the production model
            # returns only prediction columns.
            alpha_scored = (
                alpha_signal
                if {"date", "symbol", "close"} <= set(alpha_signal.columns)
                else pd.concat(
                    [
                        alpha_context.reset_index(drop=True),
                        alpha_signal.reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            alpha_scored["date"] = pd.to_datetime(alpha_scored["date"])
            alpha_scored = alpha_scored.sort_values(
                ["date", "symbol"], kind="stable"
            ).reset_index(drop=True)
            run.alpha_symbols_scored = int(
                (alpha_scored["date"] == pd.Timestamp(target)).sum()
            )
            if run.alpha_symbols_scored == 0:
                raise RuntimeError(f"AlphaRank scored no symbols for {target}")
            run.alpha_artifact = str(
                write_alpha_rank_artifact(
                    alpha_scored,
                    settings.market,
                    target,
                    paths,
                )
            )
            run.alpha_picks = select_alpha_rank_rows(
                alpha_scored,
                target,
                settings.alpharank_top_n,
            )
            run.alpha_execution_status = "success"
            run.alpha_publication_status = "pending"
            _log(
                f"AlphaRank top {len(run.alpha_picks)} for {target} from "
                f"{run.alpha_symbols_scored:,} names"
            )
        else:
            run.alpha_model = {"skipped": True, "reason": "non-pivot signal model"}
            run.alpha_execution_status = "skipped"
            run.alpha_publication_status = "skipped"

    except Exception as exc:  # noqa: BLE001 - the job must always report
        run.status = STATUS_FAILED
        run.error = str(exc)
        if isinstance(exc, DataQualityError):
            run.data_validation = exc.report
        if run.alpha_execution_status == "running":
            run.alpha_execution_status = "failed"
            run.alpha_error = str(exc)
        _log(f"Daily {signal_spec.name} run failed: {exc}", "exception")

    return _finish(
        run,
        settings,
        store_factory,
        notifier,
        dry_run,
        clock,
        campaign_publisher,
    )


def _finish(
    run: DailyRun,
    settings: Settings,
    store_factory: Callable[[Settings], AdviceStore] | None,
    notifier: Callable[[dict[str, Any], Settings], bool],
    dry_run: bool,
    clock: Callable[[], datetime] | None = None,
    campaign_publisher: Callable[[str, str, Settings], Any] | None = None,
) -> DailyRun:
    """Publish, record and notify.

    Two sinks, deliberately handled differently. The advice write *is* the
    product: if it fails the model output never reaches the dashboard, so the
    failure is recorded on the run and surfaced through a non-zero exit code
    even though the computation itself succeeded and does not need repeating.
    The shared job status and email are best-effort telemetry — a failure there
    is logged and otherwise ignored.
    """
    run.finished_at = (clock or (lambda: datetime.now(timezone.utc)))()

    if dry_run:
        if run.alpha_execution_status == "success":
            run.alpha_publication_status = "skipped"
        if run.campaign_event_required:
            run.campaign_event_status = "skipped"
        _log("Dry run: skipping MongoDB publication and notification")
        return run

    store: AdviceStore | None = None
    try:
        store = (store_factory or (lambda s: AdviceStore(s)))(settings)
        store.ensure_indexes()

        if run.status == STATUS_SUCCESS and run.picks is not None:
            try:
                documents = pick_documents(
                    run.picks,
                    settings.market,
                    settings.horizon,
                    settings.tag,
                    generated_at=run.started_at,
                )
                run.published = store.write_picks(documents)
                _log(f"Published {run.published:,} picks")
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                run.publish_error = str(exc)
                _log(f"Advice publication failed: {exc}", "error")

        if run.status == STATUS_SUCCESS and run.alpha_picks is not None:
            try:
                documents = alpha_rank_documents(
                    run.alpha_picks,
                    settings.market,
                    settings.horizon,
                    settings.alpharank_tag,
                    generated_at=run.started_at,
                )
                run.alpha_published = store.write_alpharank(documents)
                run.alpha_publication_status = "success"
                _log(f"Published {run.alpha_published:,} AlphaRank rows")
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                run.alpha_publication_status = "failed"
                run.alpha_error = str(exc)
                _log(f"AlphaRank publication failed: {exc}", "error")

    except Exception as exc:  # noqa: BLE001 - the store itself is unreachable
        if run.status == STATUS_SUCCESS and run.publish_error is None:
            run.publish_error = str(exc)
        if (
            run.alpha_execution_status == "success"
            and run.alpha_publication_status == "pending"
        ):
            run.alpha_publication_status = "failed"
            run.alpha_error = str(exc)
        _log(f"MongoDB is unreachable: {exc}", "error")

    if run.campaign_event_required:
        if run.status == STATUS_SKIPPED:
            run.campaign_event_status = "skipped"
        elif (
            run.status == STATUS_SUCCESS
            and run.date is not None
            and run.publish_error is None
            and run.alpha_error is None
            and run.alpha_publication_status == "success"
        ):
            try:
                publisher = campaign_publisher or enqueue_daily_signals_campaign
                publisher(run.market, run.date, settings)
                run.campaign_event_status = "sent"
                _log(
                    f"Enqueued daily campaign event for {run.market} on {run.date}"
                )
            except Exception as exc:  # noqa: BLE001 - recorded and surfaced
                run.campaign_event_status = "failed"
                run.campaign_event_error = str(exc)
                _log(f"Campaign event publication failed: {exc}", "error")
        else:
            run.campaign_event_status = "blocked"

    if store is not None:
        try:
            store.write_job_status(run.job_status())
        except Exception as exc:  # noqa: BLE001 - telemetry is best effort
            _log(f"Job status could not be written: {exc}", "error")
        finally:
            store.close()

    try:
        notifier(run.job_status(), settings)
    except Exception as exc:  # noqa: BLE001 - notification is best effort
        _log(f"Notification failed: {exc}", "error")

    return run
