"""Command line interface for Pivot events and AlphaRank portfolios.

    predixlab-signals build-dataset --market us --horizon 10
    predixlab-signals train         --market us --horizon 10
    predixlab-signals evaluate      --market us --horizon 10
    predixlab-signals backtest      --market us --horizon 10
    predixlab-signals report        --market us --horizon 10
    predixlab-signals daily                              # scheduled job entrypoint
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from predixlab_signals import __version__
from predixlab_signals.backtest import (
    BacktestConfig,
    baseline_random,
    benchmark_daily,
    benchmark_performance,
    build_position_book,
    buy_and_hold,
    daily_returns,
    performance,
    relative_performance,
    run_backtest,
    run_topn_dropouts,
    topn_performance,
    TopNDropoutsConfig,
)
from predixlab_signals.config import BENCHMARKS, DEFAULT_HORIZON, PATHS
from predixlab_signals.data import (
    build_dataset,
    load_benchmark,
    load_dataset,
    market_dir,
    save_dataset,
)
from predixlab_signals.evaluation import evaluate
from predixlab_signals.features import feature_columns
from predixlab_signals.model import (
    alpha_rank_model_path,
    alpha_rank_walkforward_path,
    model_path,
    run_alpha_rank_walk_forward,
    save_alpha_rank_walk_forward,
    staged_model_path,
    staged_alpha_rank_model_path,
    run_walk_forward,
    save_walk_forward,
    train_final_alpha_rank_model,
    train_final_model,
)
from predixlab_signals.report import build_pick_report
from predixlab_signals.alpha_rank_evaluation import (
    build_alpha_rank_evaluation,
)
from predixlab_signals.alpha_rank_report import build_alpha_rank_report
from predixlab_signals.signals.models.pivot import (
    BUY_TURN,
    SELL_TURN,
    TurningPointConfig,
    extract_turning_points,
    summarise as summarise_turns,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
def _walkforward_path(market: str, horizon: int, tag: str) -> Path:
    return PATHS.report_dir(market) / f"walkforward_h{horizon}_{tag}.parquet"


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2))


# --------------------------------------------------------------------------
def cmd_build_dataset(args: argparse.Namespace) -> int:
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    data = build_dataset(
        market=args.market,
        symbols=symbols,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        barrier_width=args.barrier_width,
        members_only=not args.all_history,
    )
    if data.empty:
        print("No data produced — check the normalized_data tree path.", file=sys.stderr)
        return 1
    path = save_dataset(data, args.market, args.horizon, args.tag)
    print(
        f"rows={len(data):,} symbols={data['symbol'].nunique()} "
        f"features={len(feature_columns(data))} "
        f"span={data['date'].min().date()}..{data['date'].max().date()}"
    )
    print(f"source {PATHS.training_data}")
    print(f"wrote {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    data = load_dataset(
        args.market,
        args.horizon,
        args.dataset_tag or args.tag,
    )
    feats = feature_columns(data, include_market=args.include_market)
    print(f"dataset rows={len(data):,} features={len(feats)}")

    print("walk-forward validation:")
    result = run_walk_forward(
        data,
        horizon=args.horizon,
        n_folds=args.folds,
        test_months=args.test_months,
        valid_months=args.valid_months,
        features=feats,
        market_neutral=not args.absolute,
    )
    path = save_walk_forward(result, args.market, args.horizon, args.tag)
    head_estimators = result.head_estimators()
    print(f"  wrote {path}  ({result.elapsed:.0f}s)")
    print(f"  per-head trees (fold median): {head_estimators}")

    print("fitting production model on full history...")
    model = train_final_model(
        data,
        horizon=args.horizon,
        valid_months=args.valid_months,
        features=feats,
        market_neutral=not args.absolute,
        head_estimators=head_estimators,
    )
    path = staged_model_path(args.market, args.horizon, args.tag)
    model.save(path)
    print(f"  wrote candidate {path}")
    print("\ntop features:")
    print(model.feature_importance(15).round(4).to_string())
    print(
        "\nThis is a candidate, not the production model. Validate it with\n"
        f"  predixlab-signals evaluate --market={args.market} --horizon={args.horizon}\n"
        f"then copy it to {model_path(args.market, args.horizon, args.tag)}"
    )
    return 0


def cmd_train_alpha_rank(args: argparse.Namespace) -> int:
    data = load_dataset(
        args.market,
        args.horizon,
        args.dataset_tag or args.tag,
    )
    print(
        f"dataset rows={len(data):,} symbols={data['symbol'].nunique():,} "
        f"span={data['date'].min().date()}..{data['date'].max().date()}"
    )
    print("AlphaRank walk-forward validation:")
    result = run_alpha_rank_walk_forward(
        data,
        horizon=args.horizon,
        n_folds=args.folds,
        test_months=args.test_months,
        valid_months=args.valid_months,
    )
    path = save_alpha_rank_walk_forward(
        result,
        args.market,
        args.horizon,
        args.tag,
    )
    head_estimators = result.head_estimators()
    print(f"  wrote {path} ({result.elapsed:.0f}s)")
    print(f"  per-head trees (fold median): {head_estimators}")

    print("fitting AlphaRank rolling ensemble candidate...")
    model = train_final_alpha_rank_model(
        data,
        horizon=args.horizon,
        head_estimators=head_estimators,
    )
    path = staged_alpha_rank_model_path(
        args.market,
        args.horizon,
        args.tag,
    )
    model.save(path)
    print(f"  wrote candidate {path}")
    print("\ntop features:")
    print(model.feature_importance(15).round(4).to_string())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    path = _walkforward_path(args.market, args.horizon, args.tag)
    if not path.exists():
        print(f"No walk-forward predictions at {path}. Run `train` first.", file=sys.stderr)
        return 1
    preds = pd.read_parquet(path)
    report = evaluate(preds, tolerance=args.tolerance)

    out = PATHS.report_dir(args.market) / f"evaluation_h{args.horizon}_{args.tag}.json"
    _write_json(out, report)

    print(f"out-of-sample: {report['rows']:,} rows, {report['symbols']} symbols, "
          f"{report['start']}..{report['end']}\n")

    for key, label in (("pivot_low_detection", "buy pivots"), ("pivot_high_detection", "sell pivots")):
        block = report.get(key) or {}
        if not block:
            continue
        print(f"-- future-hold detection: {label} --")
        print(f"   AUC                      {block['auc']:+.4f}")
        print(f"   base rate                {block['base_rate']:.4f}")
        for bucket in block["buckets"]:
            print(f"     top {bucket['top_frac'] * 100:4.1f}%  n={bucket['n']:6d}  "
                  f"precision={bucket['precision']:.4f}  lift={bucket['lift']:.2f}x")
        print()

    print("-- cross-sectional rank IC (score vs forward return) --")
    for k, v in report["ic_fwd_ret"].items():
        print(f"   {k:24s} {v:+.4f}")
    print("\n-- within-symbol IC (timing a single name) --")
    for k, v in report["ts_ic_fwd_ret"].items():
        print(f"   {k:24s} {v:+.4f}")
    print("\n-- realised outcome by recommendation --")
    print(pd.DataFrame(report["buckets"]).to_string(index=False))
    print("\n-- turning-point detection vs zigzag pivots --")
    for k, v in report["turning_points"].items():
        print(f"   {k:24s} {v:+.4f}")
    if report["probability"]:
        print(f"\n-- barrier probability --  AUC={report['probability']['auc']:.4f} "
              f"Brier={report['probability']['brier']:.4f}")
    print(f"\nwrote {out}")
    return 0


def cmd_backtest_alpha_rank(args: argparse.Namespace) -> int:
    path = alpha_rank_walkforward_path(
        args.market,
        args.horizon,
        args.tag,
    )
    if not path.exists():
        print(
            f"No AlphaRank predictions at {path}. "
            "Run `train-alpha-rank` first.",
            file=sys.stderr,
        )
        return 1
    preds = pd.read_parquet(path)
    config = TopNDropoutsConfig(cost_bps=args.cost_bps)
    result = run_topn_dropouts(
        preds,
        config,
        portfolio_start=args.portfolio_start,
    )
    perf = topn_performance(result.daily, start=args.start)
    if not perf:
        print("No AlphaRank returns generated.", file=sys.stderr)
        return 1
    payload = {
        "strategy": "TopNDropouts",
        "model": "alpha_rank",
        "config": config.as_dict(),
        "performance": perf,
    }
    out = (
        PATHS.report_dir(args.market)
        / f"alpha_rank_backtest_h{args.horizon}_{args.tag}.json"
    )
    _write_json(out, payload)
    result.daily.reset_index().to_parquet(
        out.with_name(out.stem + "_daily.parquet"),
        index=False,
    )
    result.selections.to_parquet(
        out.with_name(out.stem + "_selections.parquet"),
        index=False,
    )
    print("-- AlphaRank Top12Drop1, next-close execution --")
    for year, value in perf["yearly_returns"].items():
        print(f"   {year:20s} {value:+.2%}")
    for key in (
        "cagr",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
        "avg_cash_weight",
    ):
        print(f"   {key:20s} {perf[key]:+.4f}")
    print(f"   {'regime_days':20s} {perf['regime_days']}")
    print(f"\nwrote {out}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    path = _walkforward_path(args.market, args.horizon, args.tag)
    if not path.exists():
        print(f"No walk-forward predictions at {path}. Run `train` first.", file=sys.stderr)
        return 1
    preds = pd.read_parquet(path)

    turn_stats: dict = {}
    if args.signal == "turn":
        # Enter on a confirmed buy turn, exit on the opposite turn (or the usual
        # target / stop / timeout, whichever comes first).
        preds = extract_turning_points(preds, TurningPointConfig())
        turn_stats = summarise_turns(preds)
        cfg = BacktestConfig(
            horizon=args.horizon,
            max_hold=args.max_hold,
            cost_bps=args.cost_bps,
            entry_actions=(BUY_TURN,),
            exit_actions=(SELL_TURN,),
            action_column="turn_action",
        )
        actions = (BUY_TURN,)
    else:
        actions = tuple(a.strip().upper() for a in args.actions.split(","))
        cfg = BacktestConfig(
            horizon=args.horizon,
            max_hold=args.max_hold,
            cost_bps=args.cost_bps,
            entry_actions=actions,
        )
    result = run_backtest(preds, cfg)
    if result.trades.empty:
        print("No trades generated.", file=sys.stderr)
        return 1

    book = build_position_book(result.trades, max_positions=args.max_positions)
    daily = daily_returns(preds, book, max_positions=args.max_positions, cost_bps=args.cost_bps)
    perf = performance(daily)

    bench = load_benchmark(
        args.market,
        BENCHMARKS.get(args.market.lower(), "^GSPC"),
        data_root=PATHS.training_data,
    )
    bench_perf = benchmark_performance(bench, preds["date"].min(), preds["date"].max())
    bench_daily = benchmark_daily(bench, preds["date"].min(), preds["date"].max())
    relative = relative_performance(daily, bench_daily)
    random_stats = baseline_random(preds, cfg).stats

    payload = {
        "turning_points": turn_stats,
        "config": {
            "signal": args.signal,
            "entry_actions": list(actions),
            "max_hold": args.max_hold,
            "cost_bps": args.cost_bps,
            "max_positions": args.max_positions,
        },
        "trades": result.stats,
        "portfolio": perf,
        "benchmark": bench_perf,
        "vs_benchmark": relative,
        "random_baseline": random_stats,
        "buy_and_hold": buy_and_hold(preds),
    }
    suffix = "" if args.signal == "state" else f"_{args.signal}"
    out = PATHS.report_dir(args.market) / f"backtest_h{args.horizon}_{args.tag}{suffix}.json"
    _write_json(out, payload)
    book.to_parquet(out.with_name(out.stem + "_trades.parquet"), index=False)

    if turn_stats:
        print(f"-- turning points --  buy={turn_stats['buy_events']:,} "
              f"(lift {turn_stats['buy_lift']:.2f}x)  sell={turn_stats['sell_events']:,} "
              f"(lift {turn_stats['sell_lift']:.2f}x)\n")
    print(f"-- trades ({result.stats['trades']:,} signals, {len(book):,} taken) --")
    for k in ("win_rate", "avg_return", "profit_factor", "avg_bars_held", "sharpe_per_trade"):
        print(f"   {k:20s} {result.stats.get(k, float('nan')):+.4f}")
    print(f"\n-- portfolio ({args.max_positions} slots, {args.cost_bps:.0f}bps/leg) --")
    for k in ("cagr", "sharpe", "sortino", "max_drawdown", "calmar", "avg_exposure", "volatility"):
        print(f"   {k:20s} {perf.get(k, float('nan')):+.4f}")
    print("\n-- benchmark buy & hold --")
    for k in ("cagr", "sharpe", "max_drawdown"):
        print(f"   {k:20s} {bench_perf.get(k, float('nan')):+.4f}")
    print("\n-- vs benchmark (beta-adjusted) --")
    for k in ("beta", "alpha_annual", "excess_return_annual", "information_ratio"):
        print(f"   {k:20s} {relative.get(k, float('nan')):+.4f}")
    print("\n-- random-entry control --")
    for k in ("win_rate", "avg_return", "profit_factor"):
        print(f"   {k:20s} {random_stats.get(k, float('nan')):+.4f}")
    print(f"\nwrote {out}")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Replay the stored walk-forward run into a published pick history.

    This is what gives the product something to show on day one: three years of
    daily pick lists, every one of them scored out of sample by a model that had
    not seen the session it is recommending.
    """
    from predixlab_signals.jobs.backfill import build_historical_picks
    from predixlab_signals.publish.mongo import AdviceStore, pick_documents
    from predixlab_signals.settings import load_settings

    settings = load_settings()

    picks = build_historical_picks(
        market=settings.market,
        horizon=settings.horizon,
        tag=settings.tag,
        start=args.start,
        end=args.end,
        with_outcomes=not args.no_outcomes,
        max_per_side=settings.max_per_side or None,
    )
    if picks.empty:
        print("No events produced — check the walk-forward artifacts.", file=sys.stderr)
        return 1

    sessions = picks["date"].nunique()
    evaluated = int(picks["signal_return"].notna().sum()) if "signal_return" in picks else 0
    print(f"events    : {len(picks):,} over {sessions:,} sessions "
          f"({len(picks) / sessions:.1f} per active session)")
    print(f"span      : {pd.Timestamp(picks['date'].min()).date()}"
          f" .. {pd.Timestamp(picks['date'].max()).date()}")
    print(f"by type   : {picks['event'].value_counts().to_dict()}")
    print(f"evaluated : {evaluated:,} events have a completed {settings.horizon}-day horizon")
    if evaluated:
        done = picks.dropna(subset=["signal_return"])
        print(f"raw       : mean {done['signal_return'].mean():+.4%}  "
              f"win {(done['signal_return'] > 0).mean():.1%}")
        if "excess_return" in done and done["excess_return"].notna().any():
            print(f"excess    : mean {done['excess_return'].mean():+.4%}  "
                  f"win {(done['excess_return'] > 0).mean():.1%}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    documents = pick_documents(picks, settings.market, settings.horizon, settings.tag)
    with AdviceStore(settings) as store:
        store.ensure_indexes()
        written = store.write_picks(documents)
    print(f"published : {written:,} documents to {settings.picks_collection}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Build an interactive report from out-of-sample pick events only."""
    try:
        result = build_pick_report(
            market=args.market,
            horizon=args.horizon,
            tag=args.tag,
            start=args.start,
            end=args.end,
            output_path=Path(args.output) if args.output else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    overall = result.summary["overall"]
    directional = overall["accuracy"]
    directional_text = "N/A" if directional is None else f"{directional:.1%}"
    pivot = result.summary["pivot_accuracy"]["accuracy"]
    pivot_text = "N/A" if pivot is None else f"{pivot:.1%}"
    print(
        f"events    : {result.summary['events']:,} total"
    )
    print(
        f"trades    : {overall['signals']:,} total, {overall['evaluated']:,} evaluated, "
        f"{overall['pending']:,} pending"
    )
    print(f"pivot acc.: {pivot_text}")
    print(f"direction : {directional_text}")
    print(f"data      : {result.picks_path}")
    print(f"report    : {result.html_path}")
    return 0


def cmd_alpha_rank_report(args: argparse.Namespace) -> int:
    try:
        result = build_alpha_rank_report(
            market=args.market,
            horizon=args.horizon,
            tag=args.tag,
            start=args.start,
            end=args.end,
            output_path=Path(args.output) if args.output else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"period     : {result.performance['start']}..{result.performance['end']}")
    print(f"CAGR       : {result.performance['cagr']:+.2%}")
    print(f"max DD     : {result.performance['max_drawdown']:+.2%}")
    print(f"report     : {result.html_path}")
    return 0


def cmd_evaluate_alpha_rank(args: argparse.Namespace) -> int:
    try:
        result = build_alpha_rank_evaluation(
            market=args.market,
            horizon=args.horizon,
            tag=args.tag,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for fold in result.report["folds"]:
        rank_ic = fold["rank_ic"]["alpha_score"]
        print(
            f"fold {fold['fold']}: {fold['period']['start']}.."
            f"{fold['period']['end']}  rankIC={rank_ic['mean']:+.4f}  "
            f"ICIR={rank_ic['icir']:+.2f}"
        )
    print(f"json       : {result.json_path}")
    print(f"top-k csv  : {result.topk_path}")
    print(f"report     : {result.html_path}")
    return 0


def cmd_daily(args: argparse.Namespace) -> int:
    """Scheduled-job entrypoint: build, score, publish, notify.

    Imported here rather than at module scope so the research commands keep
    working without ``pymongo`` installed.
    """
    from predixlab_signals.jobs import run_daily
    from predixlab_signals.settings import load_settings

    run = run_daily(
        settings=load_settings(),
        dry_run=args.dry_run,
        end_date=args.end_date,
    )

    print(f"status          : {run.status}")
    print(f"market / date   : {run.market} / {run.date}")
    print(f"symbols scored  : {run.symbols_scored:,}")
    if run.data_validation:
        quality = run.data_validation
        print(
            f"data validation : {quality['status']}; "
            f"{quality['valid_symbols']}/{quality['expected_symbols']} valid, "
            f"{quality['bad_symbols']} excluded"
        )
    if run.counts:
        print("recommendations : " + "  ".join(f"{k}={v}" for k, v in sorted(run.counts.items())))
    if run.model:
        print(f"model           : rows={run.model.get('rows')} "
              f"features={run.model.get('features')}")
    if run.artifact:
        print(f"artifact        : {run.artifact}")
    if run.alpha_artifact:
        print(f"AlphaRank audit : {run.alpha_artifact}")
        alpha_count = 0 if run.alpha_picks is None else len(run.alpha_picks)
        print(f"AlphaRank top   : {alpha_count:,}")
    print(f"published       : {run.published:,}")
    print(f"AlphaRank pub.  : {run.alpha_published:,}")
    if run.error:
        print(f"error           : {run.error}", file=sys.stderr)
    if run.publish_error:
        print(f"publish error   : {run.publish_error}", file=sys.stderr)
    if run.alpha_error:
        print(f"AlphaRank error : {run.alpha_error}", file=sys.stderr)
    if run.campaign_event_error:
        print(f"campaign error  : {run.campaign_event_error}", file=sys.stderr)

    # A publication failure leaves the model output stranded on the artifacts
    # share, so the process must not exit 0: the scheduler is the only thing
    # that can surface "today's advice never reached the dashboard".
    if run.status not in ("success", "skipped") or not run.delivered:
        return 1
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    training_dir = market_dir(args.market, data_root=PATHS.training_data)
    prediction_dir = market_dir(args.market)
    calendar = PATHS.calendar_path(args.market)
    instruments = PATHS.instruments_path(args.market)
    print(f"predixlab-signals {__version__}")
    print(f"training tree : {PATHS.training_data}")
    print(f"prediction tree: {PATHS.collector_data}")
    print(f"artifacts     : {PATHS.artifacts}  (scratch; empty on every job start)")
    print(f"training data : {training_dir}  exists={training_dir.exists()}")
    print(f"prediction data: {prediction_dir}  exists={prediction_dir.exists()}")
    print(f"calendar      : {calendar}  exists={calendar.exists()}")
    print(f"instruments   : {instruments}  exists={instruments.exists()}")
    # Daily only loads an operator-approved production model from this path.
    model = model_path(args.market, args.horizon, args.tag)
    print(f"model         : {model}  exists={model.exists()}")
    alpha_model = alpha_rank_model_path(args.market, args.horizon, args.tag)
    print(f"AlphaRank model: {alpha_model}  exists={alpha_model.exists()}")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predixlab-signals",
        description=(
            "Train Pivot turning-point models and AlphaRank stock-selection "
            "portfolios from shared market data."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--market", default="us")
        p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
        p.add_argument("--tag", default="default")

    p = sub.add_parser("build-dataset", help="assemble features + labels into a parquet")
    common(p)
    p.add_argument("--symbols", default=None, help="comma-separated subset")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--barrier-width", type=float, default=1.5)
    p.add_argument("--all-history", action="store_true",
                   help="include bars from before a symbol joined the index (look-ahead; off by default)")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("train", help="walk-forward validate, then fit the production model")
    common(p)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--valid-months", type=int, default=6)
    p.add_argument("--include-market", action="store_true",
                   help="also feed market-wide features to the model (overfits; off by default)")
    p.add_argument("--absolute", action="store_true",
                   help="train on absolute rather than market-neutral targets")
    p.add_argument(
        "--dataset-tag",
        default=None,
        help="read this dataset tag while writing model/report artifacts under --tag",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "train-alpha-rank",
        help="train the AlphaRank forward-relative-return model",
    )
    common(p)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--test-months", type=int, default=12)
    p.add_argument("--valid-months", type=int, default=6)
    p.add_argument(
        "--dataset-tag",
        default=None,
        help="read this dataset tag while writing artifacts under --tag",
    )
    p.set_defaults(func=cmd_train_alpha_rank)

    p = sub.add_parser("evaluate", help="score the walk-forward predictions")
    common(p)
    p.add_argument("--tolerance", type=int, default=3, help="bars of slack around a pivot")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("backtest", help="simulate trades and portfolio performance")
    common(p)
    p.add_argument("--actions", default="STRONG_BUY")
    p.add_argument("--signal", choices=("state", "turn"), default="state")
    p.add_argument("--max-hold", type=int, default=15)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--max-positions", type=int, default=20)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser(
        "backtest-alpha-rank",
        help="run the formal AlphaRank Top12Drop1 portfolio",
    )
    common(p)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--start", default=None)
    p.add_argument(
        "--portfolio-start",
        default=None,
        help="reset holdings and initialize the portfolio on this signal date",
    )
    p.set_defaults(func=cmd_backtest_alpha_rank)

    p = sub.add_parser(
        "evaluate-alpha-rank",
        help="evaluate AlphaRank models by fold and Top-K bucket",
    )
    common(p)
    p.set_defaults(func=cmd_evaluate_alpha_rank)

    p = sub.add_parser(
        "backfill-picks",
        help="replay the stored walk-forward run into a published pick history",
    )
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-outcomes", action="store_true",
                   help="skip the realised-return join from the backtest trade book")
    p.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser(
        "report",
        help="build an interactive HTML accuracy report from historical picks",
    )
    common(p)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--output", default=None, help="HTML output path")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "alpha-rank-report",
        help="build the formal AlphaRank portfolio report",
    )
    common(p)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_alpha_rank_report)

    p = sub.add_parser("daily", help="scheduled job: build, score, publish to MongoDB, notify")
    p.add_argument(
        "--end-date",
        default=None,
        help="required target trading date; validates that session without falling back to older data",
    )
    p.add_argument("--dry-run", action="store_true", help="compute and write artifacts, publish nothing")
    p.set_defaults(func=cmd_daily)

    p = sub.add_parser("info", help="show resolved paths")
    common(p)
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
