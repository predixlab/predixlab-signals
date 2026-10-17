"""CLI surface contracts."""

from __future__ import annotations

import pytest

from predixlab_signals.cli import build_parser, main


def test_operational_commands_accept_only_run_scoped_options():
    parser = build_parser()

    daily = parser.parse_args(["daily", "--end-date", "2026-04-24", "--dry-run"])
    assert daily.command == "daily"
    assert daily.end_date == "2026-04-24"
    backfill = parser.parse_args(["backfill-picks", "--start", "2024-01-01", "--dry-run"])
    assert backfill.command == "backfill-picks"


def test_daily_data_rejection_is_reported_with_a_nonzero_exit(monkeypatch, capsys):
    from datetime import datetime, timezone

    from predixlab_signals import jobs
    from predixlab_signals.jobs.daily import DailyRun

    rejected = DailyRun(
        market="us",
        horizon=10,
        tag="default",
        status="failed",
        started_at=datetime.now(timezone.utc),
        date="2026-09-08",
        error="Daily data validation failed: 20/20 constituents are missing",
        data_validation={
            "status": "failed",
            "valid_symbols": 0,
            "expected_symbols": 20,
            "bad_symbols": 20,
        },
    )
    monkeypatch.setattr(jobs, "run_daily", lambda **kwargs: rejected)
    assert main(["daily", "--end-date", "2026-09-08", "--dry-run"]) == 1
    output = capsys.readouterr()
    assert "data validation : failed; 0/20 valid, 20 excluded" in output.out
    assert rejected.error in output.err


def test_report_accepts_research_scope_and_output_options():
    report = build_parser().parse_args(
        [
            "report",
            "--market",
            "us",
            "--horizon",
            "10",
            "--start",
            "2024-01-01",
            "--output",
            "report.html",
        ]
    )

    assert report.command == "report"
    assert report.output == "report.html"


def test_alpha_rank_commands_have_formal_options_only():
    parser = build_parser()
    train = parser.parse_args(
        [
            "train-alpha-rank",
            "--dataset-tag",
            "default",
            "--tag",
            "default",
        ]
    )
    assert train.command == "train-alpha-rank"
    assert train.dataset_tag == "default"

    backtest = parser.parse_args(
        [
            "backtest-alpha-rank",
            "--portfolio-start",
            "2025-01-02",
        ]
    )
    assert backtest.command == "backtest-alpha-rank"
    assert backtest.portfolio_start == "2025-01-02"

    report = parser.parse_args(
        [
            "alpha-rank-report",
            "--tag",
            "default",
            "--output",
            "portfolio.html",
        ]
    )
    assert report.command == "alpha-rank-report"

    evaluation = parser.parse_args(["evaluate-alpha-rank"])
    assert evaluation.command == "evaluate-alpha-rank"


@pytest.mark.parametrize(
    "argv",
    [
        ["train-cross-sectional"],
        ["topn-report"],
        ["backtest", "--signal", "topn"],
        ["backtest-alpha-rank", "--top-n", "10"],
        ["backtest-alpha-rank", "--bear-market"],
        ["train-alpha-rank", "--return-only"],
        ["train-alpha-rank", "--absolute-return"],
        ["train-alpha-rank", "--dataset-horizon", "10"],
    ],
)
def test_removed_experimental_alpha_rank_paths_stay_removed(argv):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["daily", "--force"],
        ["daily", "--market", "us"],
        ["daily", "--horizon", "5"],
        ["daily", "--tag", "candidate"],
        ["daily", "--date", "2026-04-24"],
        ["backfill-picks", "--market", "us"],
        ["backfill-picks", "--max-per-side", "10"],
    ],
)
def test_operational_commands_reject_environment_overrides(argv):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2


@pytest.mark.parametrize("argv", [["advise"], ["train", "--skip-walkforward"]])
def test_removed_research_shortcuts_stay_removed(argv):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    assert exc_info.value.code == 2