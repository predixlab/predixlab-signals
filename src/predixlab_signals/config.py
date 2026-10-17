"""Filesystem roots and tunable defaults for predixlab-signals.

Everything is overridable through environment variables so the package can use
the full research history for training and a separate clean production tree for
daily prediction.

The panel always comes from the normalized per-symbol CSVs written by
``predixlab-collector normalize_data``: the nightly job mounts the shared
PredixLab data share and reads straight from it, so pivot depends on nothing
between the collector and itself.

Environment variables
---------------------
* ``PREDIXLAB_TRAINING_DATA`` — full historical Collector tree used only to
  build research/training datasets. Defaults to ``data`` below the repository.
* ``PREDIXLAB_DATA`` — clean production tree used for inference input and
  persistent production output (``models/``, ``predictions/``, ``picks/``).
  Defaults to ``~/.predixlab``. Its individual path environment variables take
  precedence.
* ``PIVOT_ARTIFACTS`` — scratch for derived files (the dataset parquet, research
  reports). The scheduled job's container starts with this empty every time by
  design, so nothing may depend on its contents surviving. Defaults to
  ``<repo>/artifacts``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    # ``<repo>/src/predixlab_signals/config.py`` -> ``<repo>``
    return Path(__file__).resolve().parents[2]


def _default_collector_data() -> Path:
    return Path(os.getenv("PREDIXLAB_DATA") or "~/.predixlab").expanduser().resolve()


def _default_training_data() -> Path:
    value = os.getenv("PREDIXLAB_TRAINING_DATA")
    return Path(value).expanduser().resolve() if value else (_project_root() / "data").resolve()


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None


def _default_artifacts() -> Path:
    env = os.getenv("PIVOT_ARTIFACTS")
    if env:
        return Path(env).expanduser().resolve()
    return (_project_root() / "artifacts").resolve()


@dataclass(frozen=True)
class Paths:
    collector_data: Path = field(default_factory=_default_collector_data)
    training_data: Path = field(default_factory=_default_training_data)
    artifacts: Path = field(default_factory=_default_artifacts)
    index_dir: Path | None = field(default_factory=lambda: _optional_path("INDEX_DIR"))
    us_index_path: Path | None = field(default_factory=lambda: _optional_path("US_INDEX_PATH"))
    cn_index_path: Path | None = field(default_factory=lambda: _optional_path("CN_INDEX_PATH"))
    normalized_data_dir: Path | None = field(
        default_factory=lambda: _optional_path("NORMALIZED_DATA_DIR")
    )
    us_normalized_data_dir: Path | None = field(
        default_factory=lambda: _optional_path("US_NORMALIZED_DATA_DIR")
    )
    cn_normalized_data_dir: Path | None = field(
        default_factory=lambda: _optional_path("CN_NORMALIZED_DATA_DIR")
    )
    models_dir: Path | None = field(default_factory=lambda: _optional_path("MODELS_DIR"))
    predictions_dir: Path | None = field(
        default_factory=lambda: _optional_path("PREDICTIONS_DIR")
    )
    picks_dir: Path | None = field(default_factory=lambda: _optional_path("PICKS_DIR"))

    # -- collector tree ----------------------------------------------------
    def normalized_dir(self, market: str) -> Path:
        market = market.lower()
        specific = {
            "us": self.us_normalized_data_dir,
            "cn": self.cn_normalized_data_dir,
        }.get(market)
        root = self.normalized_data_dir or self.collector_data / "normalized_data"
        return specific or root / f"{market}_data"

    def instruments_path(self, market: str) -> Path:
        market = market.lower()
        specific = {
            "us": self.us_index_path,
            "cn": self.cn_index_path,
        }.get(market)
        root = self.index_dir or self.collector_data / "instruments"
        return specific or root / f"{market}.txt"

    def calendar_path(self, market: str) -> Path:
        return self.collector_data / "calendar" / f"{market.lower()}.txt"

    # -- scratch (PIVOT_ARTIFACTS) ----------------------------------------
    #
    # Everything below is *derived* and cheap to rebuild. In production the job
    # runs in a container whose filesystem starts empty every time by design, so
    # treat these as within-run scratch: they make a single run faster and give
    # a local research session somewhere to put its output, and nothing depends
    # on them surviving. The dataset is a pure function of the collector CSVs
    # and the horizon, so rebuilding it costs minutes and loses no information.

    def dataset_dir(self, market: str) -> Path:
        return self.artifacts / market.lower() / "datasets"

    def staging_model_dir(self, market: str) -> Path:
        """Where ``train`` writes a *candidate* model.

        Training produces a candidate, not a production model. It stays local
        and unused until an operator validates it and copies it to
        :meth:`model_dir`. Keeping the two apart is what
        makes "only a locally validated model reaches the server" a property of
        the layout rather than of whoever ran the command.
        """
        return self.artifacts / market.lower() / "models"

    def report_dir(self, market: str) -> Path:
        return self.artifacts / market.lower() / "reports"

    # -- shared data root outputs -----------------------------------------
    #
    # These persist. predixlab-signal keeps its state and its daily output in
    # the shared data root rather than in its own tree, and pivot follows:
    #
    #     models/<market>/                       trained models
    #     predictions/<date>/<market>/           the day's scored cross-section
    #     picks/<date>/                          the day's published pick list
    #
    # In production both projects mount the same Azure file share
    # (``predixlab-data``) at ``/data`` (``PREDIXLAB_DATA``), so following the
    # convention is what lets anything downstream — the archiver, a notebook, an
    # operator browsing the share — find pivot's output where it already looks
    # for signal's, without knowing that two different models produced it.
    #
    # Per-session filenames are namespaced with the registered signal model
    # because those directories are shared: ``us.csv`` belongs to the legacy
    # signal job and ``us_pivot.csv`` to the current pivot runtime.

    def model_dir(self, market: str) -> Path:
        """Where the **production** model lives: the one the daily job scores with.

        Models are trained and validated locally; a candidate is copied here by
        hand once it passes review. The scheduled job never writes to this
        directory — it only reads — so a model on the server is always one a
        human validated, never one a container fitted at 02:00 UTC on its own.

        It is on the shared root rather than in scratch because the job's
        container filesystem starts empty every run by design.
        """
        return (self.models_dir or self.collector_data / "models") / market.lower()

    def alpha_rank_model_path(self, market: str, horizon: int, tag: str = "default") -> Path:
        """The independently validated production AlphaRank ensemble."""
        return self.model_dir(market) / f"alpha_rank_h{horizon}_{tag}.pkl"

    def daily_predictions_dir(self, market: str, date: str) -> Path:
        root = self.predictions_dir or self.collector_data / "predictions"
        return root / date / market.lower()

    def daily_picks_dir(self, date: str) -> Path:
        return (self.picks_dir or self.collector_data / "picks") / date


PATHS = Paths()

#: Benchmark instrument used for market-relative features, per market.
BENCHMARKS: dict[str, str] = {"us": "^GSPC", "cn": "SH000300"}

#: Prediction horizon in trading days. Ten sessions is roughly two weeks — long
#: enough for a swing to develop, short enough that the label stays informative.
DEFAULT_HORIZON = 10

#: Minimum number of bars a symbol must have before it is usable.
MIN_HISTORY = 400
