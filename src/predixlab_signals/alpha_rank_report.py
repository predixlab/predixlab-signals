"""Self-contained AlphaRank portfolio report."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predixlab_signals.backtest import topn_performance
from predixlab_signals.config import PATHS


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


@dataclass(frozen=True)
class AlphaRankReportResult:
    html_path: Path
    daily_path: Path
    selections_path: Path
    performance: dict[str, object]


def _artifact_stem(horizon: int, tag: str) -> str:
    return f"alpha_rank_backtest_h{horizon}_{tag}"


def _holding_rows(
    daily: pd.DataFrame,
    selections: pd.DataFrame,
) -> list[dict[str, Any]]:
    selection_groups = {
        pd.Timestamp(date): group.sort_values("rank", kind="stable")
        for date, group in selections.groupby("signal_date", sort=True)
    }
    previous: set[str] = set()
    rows = []
    equity = 1.0
    peak = 1.0
    for row in daily.sort_values("date", kind="stable").itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        group = selection_groups.get(signal_date, pd.DataFrame())
        holdings = (
            group["symbol"].astype(str).tolist() if not group.empty else []
        )
        current = set(holdings)
        bought = (
            group.loc[group["status"] == "bought", "symbol"].astype(str).tolist()
            if not group.empty
            else []
        )
        sold = sorted(previous - current)
        equity *= 1.0 + float(row.ret)
        peak = max(peak, equity)
        rows.append(
            {
                "date": str(pd.Timestamp(row.date).date()),
                "signal_date": str(signal_date.date()),
                "return": float(row.ret),
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "turnover": float(row.turnover),
                "positions": int(row.positions),
                "cash_weight": float(getattr(row, "cash_weight", 0.0)),
                "exposure_scale": float(getattr(row, "exposure_scale", 1.0)),
                "target_volatility": (
                    float(row.target_volatility)
                    if pd.notna(getattr(row, "target_volatility", np.nan))
                    else None
                ),
                "regime": str(getattr(row, "exposure_regime", "full")),
                "shock": bool(getattr(row, "shock_trigger", False)),
                "bear_trigger": bool(getattr(row, "bear_trigger", False)),
                "bear_reversal": bool(getattr(row, "bear_reversal", False)),
                "holdings": holdings,
                "bought": bought,
                "sold": sold,
            }
        )
        previous = current
    return rows


def _holding_episodes(
    selections: pd.DataFrame,
    *,
    cost_bps: float = 10.0,
    open_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate consecutive daily selections into per-symbol holding periods."""
    scores = selections.copy()
    if scores.empty:
        return []
    scores["signal_date"] = pd.to_datetime(scores["signal_date"])
    required = {"entry_date", "return_date", "stock_return"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"AlphaRank selections missing report columns: {sorted(missing)}"
        )
    joined = scores.copy()
    joined["entry_date"] = pd.to_datetime(joined["entry_date"])
    joined["return_date"] = pd.to_datetime(joined["return_date"])
    episodes = []
    cost = cost_bps / 10_000.0
    for symbol, symbol_rows in joined.groupby("symbol", sort=True):
        symbol_rows = symbol_rows.sort_values("signal_date", kind="stable").copy()
        symbol_rows["_episode"] = (
            symbol_rows["signal_date"]
            .ne(symbol_rows["entry_date"].shift(1))
            .cumsum()
        )
        for _, group in symbol_rows.groupby("_episode", sort=True):
            group = group.sort_values("signal_date", kind="stable")
            returns = pd.to_numeric(
                group["stock_return"],
                errors="coerce",
            ).fillna(0.0)
            path = (1.0 + returns).cumprod()
            gross_return = float(path.iloc[-1] - 1.0)
            open_episode = bool(
                symbol in (open_symbols or set())
                and group["signal_date"].iloc[-1] == symbol_rows["signal_date"].iloc[-1]
            )
            entry_legs = 1 if group["status"].iloc[0] == "bought" else 0
            exit_legs = 0 if open_episode else 1
            legs = entry_legs + exit_legs
            net_return = float((1.0 + gross_return) * (1.0 - cost) ** legs - 1.0)
            drawdown = path / path.cummax() - 1.0
            daily_path = [
                {
                    "date": (
                        str(pd.Timestamp(return_date).date())
                        if pd.notna(return_date)
                        else str(pd.Timestamp(signal_date).date())
                    ),
                    "return": float(stock_return),
                    "cumulative_return": float(cumulative - 1.0),
                }
                for signal_date, return_date, stock_return, cumulative in zip(
                    group["signal_date"],
                    group["return_date"],
                    returns,
                    path,
                )
            ]
            episodes.append(
                {
                    "symbol": str(symbol),
                    "entry_signal_date": str(
                        pd.Timestamp(group["signal_date"].iloc[0]).date()
                    ),
                    "entry_date": (
                        str(pd.Timestamp(group["entry_date"].iloc[0]).date())
                        if pd.notna(group["entry_date"].iloc[0])
                        else None
                    ),
                    "exit_date": (
                        None
                        if open_episode
                        else str(pd.Timestamp(group["return_date"].iloc[-1]).date())
                        if pd.notna(group["return_date"].iloc[-1])
                        else None
                    ),
                    "sessions": int(len(group)),
                    "entry_score": float(group["score"].iloc[0]),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "max_runup": float(path.max() - 1.0),
                    "max_drawdown": float(drawdown.min()),
                    "status": "open" if open_episode else "closed",
                    "path": daily_path,
                }
            )
    return sorted(
        episodes,
        key=lambda row: (row["entry_signal_date"], row["symbol"]),
    )


def render_alpha_rank_report(
    daily: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    market: str,
    horizon: int,
    tag: str,
    cost_bps: float = 10.0,
) -> tuple[str, dict[str, object]]:
    if daily.empty:
        raise ValueError("AlphaRank report daily input is empty")
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    picks = selections.copy()
    picks["signal_date"] = pd.to_datetime(picks["signal_date"])
    performance = topn_performance(frame.set_index("date"))
    holdings = _holding_rows(frame, picks)
    open_symbols = set(holdings[-1]["holdings"]) if holdings else set()
    episodes = _holding_episodes(
        picks,
        cost_bps=cost_bps,
        open_symbols=open_symbols,
    )
    report_start = str(pd.Timestamp(frame["date"].min()).date())
    report_end = str(pd.Timestamp(frame["date"].max()).date())
    episodes = [
        episode
        for episode in episodes
        if (episode["exit_date"] or report_end) >= report_start
        and (episode["entry_date"] or report_start) <= report_end
    ]
    resolved = [episode for episode in episodes if episode["status"] == "closed"]
    episode_summary = {
        "episodes": len(episodes),
        "closed": len(resolved),
        "win_rate": (
            float(np.mean([episode["net_return"] > 0 for episode in resolved]))
            if resolved
            else None
        ),
        "avg_net_return": (
            float(np.mean([episode["net_return"] for episode in resolved]))
            if resolved
            else None
        ),
    }
    monthly_rows = []
    monthly_returns = (
        (1.0 + frame.set_index("date")["ret"])
        .groupby(frame["date"].dt.to_period("M").to_numpy())
        .prod()
        - 1.0
    )
    for month, monthly_return in monthly_returns.items():
        block = frame[frame["date"].dt.to_period("M") == month].set_index("date")
        equity = (1.0 + block["ret"]).cumprod()
        monthly_rows.append(
            {
                "month": str(month),
                "return": float(monthly_return),
                "max_drawdown": float(
                    (equity / equity.cummax() - 1.0).min()
                ),
                "avg_exposure": float(
                    (1.0 - block["cash_weight"]).mean()
                ),
                "bear_days": int(
                    (block["exposure_regime"] == "bear").sum()
                ),
            }
        )
    monthly_series = pd.Series(
        {
            row["month"]: row["return"]
            for row in monthly_rows
        },
        dtype="float64",
    )
    rolling_months = {}
    for window in (3, 6, 12):
        values = (1.0 + monthly_series).rolling(window).apply(
            np.prod,
            raw=True,
        ) - 1.0
        resolved_values = values.dropna()
        rolling_months[str(window)] = {
            "windows": int(len(resolved_values)),
            "min": (
                float(resolved_values.min())
                if len(resolved_values)
                else None
            ),
            "median": (
                float(resolved_values.median())
                if len(resolved_values)
                else None
            ),
            "positive_rate": (
                float((resolved_values > 0).mean())
                if len(resolved_values)
                else None
            ),
        }
    payload = {
        "meta": {
            "market": market.upper(),
            "horizon": horizon,
            "tag": tag,
            "model": "alpha_rank",
            "execution_price": "next_close",
            "cost_bps": cost_bps,
            "start": holdings[0]["date"],
            "end": holdings[-1]["date"],
        },
        "performance": performance,
        "yearly": [
            {"year": year, "return": value}
            for year, value in performance["yearly_returns"].items()
        ],
        "monthly": monthly_rows,
        "rolling_months": rolling_months,
        "history": holdings,
        "episodes": episodes,
        "episode_summary": episode_summary,
        "shocks": [row for row in holdings if row["regime"] == "shock"],
        "risk_sessions": [
            row
            for row in holdings
            if row["regime"] != "full" or row["bear_reversal"]
        ],
    }
    payload_json = json.dumps(
        _json_ready(payload),
        ensure_ascii=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    title = escape(f"{market.upper()} AlphaRank Portfolio Report")
    return (
        _HTML_TEMPLATE.replace("__TITLE__", title).replace(
            "__REPORT_DATA__",
            payload_json,
        ),
        performance,
    )


def build_alpha_rank_report(
    *,
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
    start: str | None = None,
    end: str | None = None,
    output_path: Path | None = None,
) -> AlphaRankReportResult:
    stem = _artifact_stem(horizon, tag)
    root = PATHS.report_dir(market)
    daily_path = root / f"{stem}_daily.parquet"
    selections_path = root / f"{stem}_selections.parquet"
    if not daily_path.exists() or not selections_path.exists():
        raise FileNotFoundError(
            f"Missing AlphaRank backtest artifacts for {stem}; run backtest-alpha-rank first"
        )
    daily = pd.read_parquet(daily_path)
    selections = pd.read_parquet(selections_path)
    config_path = root / f"{stem}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing AlphaRank backtest config: {config_path}")
    backtest_payload = json.loads(config_path.read_text())
    cost_bps = float(backtest_payload["config"]["cost_bps"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
    if start is not None:
        daily = daily[daily["date"] >= pd.Timestamp(start)]
    if end is not None:
        daily = daily[daily["date"] <= pd.Timestamp(end)]
    if not selections.empty:
        selections["signal_date"] = pd.to_datetime(selections["signal_date"])
        last_signal_date = daily["signal_date"].max()
        selections = selections[
            selections["signal_date"] <= last_signal_date
        ]

    html, performance = render_alpha_rank_report(
        daily,
        selections,
        market=market,
        horizon=horizon,
        tag=tag,
        cost_bps=cost_bps,
    )
    destination = output_path or root / f"alpha_rank_report_h{horizon}_{tag}.html"
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return AlphaRankReportResult(
        html_path=destination,
        daily_path=daily_path,
        selections_path=selections_path,
        performance=performance,
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--ink:#172126;--muted:#65736f;--paper:#f5f7f5;--card:#fff;--line:#d8dfda;--green:#087f72;--red:#d85b42;--gold:#d9a91e}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--paper);font-family:"Avenir Next","Segoe UI",sans-serif}
header{padding:30px 24px;background:var(--ink);color:#fff}header div,main{width:min(1280px,calc(100% - 32px));margin:auto}
h1,h2{font-family:Georgia,serif;font-weight:500}h1{margin:5px 0;font-size:clamp(30px,5vw,48px)}.sub{color:#cbd5d1}
main{padding:24px 0 50px}.cards{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:7px}
.card{padding:15px;border-top:4px solid var(--green)}.card:nth-child(3),.card:nth-child(4){border-top-color:var(--red)}.label{color:var(--muted);font-size:11px;font-weight:700;text-transform:uppercase}
.value{display:block;margin-top:5px;font-family:Georgia,serif;font-size:27px}.panel{margin-top:22px;padding:17px}h2{margin:0 0 12px;font-size:22px}
.range-tools,.tools,.pager{display:flex;flex-wrap:wrap;align-items:center;gap:9px}.range-tools{margin:0 0 12px}.tools{margin-bottom:10px}.range-tools input,.tools input,.tools select,.pager select{height:38px;padding:0 10px;border:1px solid #aeb9b4;border-radius:4px;background:#fff;color:var(--ink)}
.range-tools button,.pager button,.expand-button{height:34px;padding:0 11px;border:1px solid #aeb9b4;border-radius:4px;background:#fff;color:var(--ink);cursor:pointer}.range-tools button:hover,.pager button:hover,.expand-button:hover{border-color:var(--green);color:var(--green)}
.charts{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px}.chart-shell{position:relative}.chart{width:100%;height:300px;display:block}.grid{stroke:#e6ebe7}.axis{stroke:#aeb9b4}.equity{fill:none;stroke:var(--green);stroke-width:2}.drawdown{fill:none;stroke:var(--red);stroke-width:2}.cash{fill:none;stroke:var(--gold);stroke-width:2}.hover-guide{stroke:#66736f;stroke-dasharray:4 3}.hover-point{fill:#fff;stroke:var(--ink);stroke-width:2}.chart-overlay{fill:transparent;cursor:crosshair}.chart-tooltip{position:absolute;display:none;pointer-events:none;z-index:4;min-width:154px;padding:8px 10px;border-radius:5px;background:rgba(23,33,38,.94);color:#fff;font-size:11px;line-height:1.45;box-shadow:0 5px 15px rgba(0,0,0,.22)}
.table-wrap{overflow:auto;max-height:660px;border:1px solid var(--line);border-radius:6px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}th{position:sticky;top:0;background:#eaf0ec;color:var(--muted);font-size:10px;text-transform:uppercase;z-index:1}th:first-child,td:first-child{text-align:left}th.sortable,th.episode-sortable{cursor:pointer;user-select:none}th.sortable:hover,th.episode-sortable:hover{color:var(--green)}th.sort-asc::after{content:" ▲"}th.sort-desc::after{content:" ▼"}
.positive{color:var(--green);font-weight:700}.negative{color:var(--red);font-weight:700}.shock{background:#fff2ef}.detail-row td{padding:0;background:#f8faf8;text-align:left}.detail-content{display:grid;grid-template-columns:2fr 1fr 1fr;gap:14px;padding:14px}.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{padding:4px 7px;border-radius:12px;background:#e7eeea;font-size:11px}.chip.buy{background:#dff2ed;color:#07685e}.chip.sell{background:#f9e4df;color:#a63f2d}.pager{justify-content:space-between;margin-top:10px;color:var(--muted);font-size:12px}.pager-group{display:flex;align-items:center;gap:8px}
.year-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.year{padding:13px;background:#eef3ef;border-radius:5px}.year strong{display:block;font:24px Georgia,serif;margin-top:3px}
.episode-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}.episode-stat{padding:12px;background:#eef3ef;border-radius:5px}.episode-stat strong{display:block;margin-top:3px;font:22px Georgia,serif}
footer{margin-top:20px;color:var(--muted);font-size:12px;line-height:1.5}
@media(max-width:900px){.cards{grid-template-columns:repeat(3,1fr)}.charts{grid-template-columns:1fr}.episode-summary{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.cards{grid-template-columns:repeat(2,1fr)}.year-grid,.episode-summary{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><span class="label" style="color:#d9a91e">Strict walk-forward OOS</span><h1>__TITLE__</h1><p class="sub" id="period"></p></div></header>
<main>
<div class="cards">
<div class="card"><span class="label">CAGR</span><strong class="value" id="cagr"></strong></div>
<div class="card"><span class="label">Sharpe</span><strong class="value" id="sharpe"></strong></div>
<div class="card"><span class="label">Max drawdown</span><strong class="value" id="mdd"></strong></div>
<div class="card"><span class="label">Volatility</span><strong class="value" id="vol"></strong></div>
<div class="card"><span class="label">Total return</span><strong class="value" id="total"></strong></div>
<div class="card"><span class="label">Avg cash</span><strong class="value" id="cash"></strong></div>
</div>
<section class="panel">
<div class="range-tools">
<span class="label">Visible range</span>
<input id="range-start" type="date" aria-label="Chart start date">
<input id="range-end" type="date" aria-label="Chart end date">
<button type="button" data-range="126">6M</button>
<button type="button" data-range="252">1Y</button>
<button type="button" data-range="504">2Y</button>
<button type="button" data-range="all">All</button>
</div>
<div class="charts">
<div><h2>Net equity</h2><div class="chart-shell"><svg class="chart" id="equity-chart"></svg><div class="chart-tooltip" id="equity-tooltip"></div></div></div>
<div><h2>Drawdown</h2><div class="chart-shell"><svg class="chart" id="drawdown-chart"></svg><div class="chart-tooltip" id="drawdown-tooltip"></div></div></div>
<div><h2>Cash exposure</h2><div class="chart-shell"><svg class="chart" id="cash-chart"></svg><div class="chart-tooltip" id="cash-tooltip"></div></div></div>
</div>
</section>
<section class="panel"><h2>Calendar-year returns</h2><div class="year-grid" id="years"></div></section>
<section class="panel"><h2>Monthly rolling stability</h2><div class="year-grid" id="rolling-months"></div><div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>Month</th><th>Return</th><th>Max drawdown</th><th>Average exposure</th><th>Bear days</th></tr></thead><tbody id="monthly-body"></tbody></table></div></section>
<section class="panel"><h2>Risk-off sessions</h2><div class="table-wrap"><table><thead><tr><th>Return date</th><th>Signal date</th><th>Regime</th><th>Transition</th><th>Positions</th><th>Daily return</th></tr></thead><tbody id="shock-body"></tbody></table></div></section>
<section class="panel">
<h2>Individual holding-period returns</h2>
<div class="episode-summary">
<div class="episode-stat"><span class="label">Holding episodes</span><strong id="episode-count"></strong></div>
<div class="episode-stat"><span class="label">Closed episodes</span><strong id="episode-closed"></strong></div>
<div class="episode-stat"><span class="label">Win rate after costs</span><strong id="episode-win-rate"></strong></div>
<div class="episode-stat"><span class="label">Average net return</span><strong id="episode-avg-return"></strong></div>
</div>
<div class="tools">
<input id="episode-symbol-filter" placeholder="Filter stock symbol" aria-label="Filter holding episode symbol">
<select id="episode-status-filter"><option value="">All statuses</option><option value="closed">Closed</option><option value="open">Open</option></select>
<select id="episode-outcome-filter"><option value="">All outcomes</option><option value="win">Profitable</option><option value="loss">Loss</option></select>
<button type="button" id="clear-episode-filters" class="expand-button">Clear</button>
</div>
<div class="table-wrap"><table><thead><tr>
<th class="episode-sortable" data-episode-key="symbol">Symbol</th>
<th class="episode-sortable" data-episode-key="entry_date">Entry</th>
<th class="episode-sortable" data-episode-key="exit_date">Exit</th>
<th class="episode-sortable" data-episode-key="sessions">Sessions</th>
<th class="episode-sortable" data-episode-key="entry_score">Entry score</th>
<th class="episode-sortable" data-episode-key="gross_return">Gross return</th>
<th class="episode-sortable" data-episode-key="net_return">Net return</th>
<th class="episode-sortable" data-episode-key="max_runup">Max run-up</th>
<th class="episode-sortable" data-episode-key="max_drawdown">Max drawdown</th>
<th class="episode-sortable" data-episode-key="status">Status</th><th>Path</th>
</tr></thead><tbody id="episode-body"></tbody></table></div>
<div class="pager"><span id="episode-row-count"></span><div class="pager-group"><label for="episode-page-size">Rows</label><select id="episode-page-size"><option>25</option><option selected>50</option><option>100</option></select><button type="button" id="episode-prev-page">Previous</button><span id="episode-page-info"></span><button type="button" id="episode-next-page">Next</button></div></div>
</section>
<section class="panel">
<h2>Daily portfolio log</h2>
<div class="tools">
<input id="date-filter" type="date" aria-label="Filter exact return date">
<input id="symbol-filter" placeholder="Filter holding / trade symbol" aria-label="Filter symbol">
<select id="regime-filter" aria-label="Filter regime"><option value="">All regimes</option><option value="full">Full</option><option value="shock">Shock</option><option value="light">Light</option><option value="cash">Cash</option></select>
<button type="button" id="clear-filters" class="expand-button">Clear</button>
</div>
<div class="table-wrap"><table><thead><tr>
<th class="sortable" data-key="date">Return date</th>
<th class="sortable" data-key="signal_date">Signal date</th>
<th class="sortable" data-key="return">Net return</th>
<th class="sortable" data-key="equity">Equity</th>
<th class="sortable" data-key="drawdown">Drawdown</th>
<th class="sortable" data-key="positions">Positions</th>
<th class="sortable" data-key="regime">Regime</th>
<th class="sortable" data-key="turnover">Turnover</th>
<th>Trades</th><th>Details</th>
</tr></thead><tbody id="history-body"></tbody></table></div>
<div class="pager"><span id="row-count"></span><div class="pager-group"><label for="page-size">Rows</label><select id="page-size"><option>25</option><option selected>50</option><option>100</option></select><button type="button" id="prev-page">Previous</button><span id="page-info"></span><button type="button" id="next-page">Next</button></div></div>
</section>
<footer>Returns include <span id="cost-bps"></span> bps per buy or sell leg and compound daily. Signals use close-t information and trade at the configured next-session execution price. Market impact, taxes, and live fill uncertainty are excluded.</footer>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
const report=JSON.parse(document.getElementById("report-data").textContent);
const pct=(v,d=2)=>`${v>=0?"+":""}${(v*100).toFixed(d)}%`;
const cls=v=>v>=0?"positive":"negative";
document.getElementById("period").textContent=`${report.meta.start} to ${report.meta.end} | ${report.meta.market} | h${report.meta.horizon} | ${report.meta.model} | ${report.meta.execution_price.replace("_"," ")}`;
const p=report.performance;
document.getElementById("cagr").textContent=pct(p.cagr);
document.getElementById("sharpe").textContent=p.sharpe==null?"N/A":p.sharpe.toFixed(2);
document.getElementById("mdd").textContent=pct(p.max_drawdown);
document.getElementById("vol").textContent=pct(p.volatility);
document.getElementById("total").textContent=pct(p.total_return);
document.getElementById("cash").textContent=pct(p.avg_cash_weight);
document.getElementById("cost-bps").textContent=report.meta.cost_bps.toFixed(1);
document.getElementById("years").innerHTML=report.yearly.map(x=>`<div class="year"><span class="label">${x.year}</span><strong class="${cls(x.return)}">${pct(x.return)}</strong></div>`).join("");
document.getElementById("rolling-months").innerHTML=Object.entries(report.rolling_months).map(([window,item])=>`<div class="year"><span class="label">${window}-month windows</span><strong class="${cls(item.min)}">min ${pct(item.min)}</strong><span>median ${pct(item.median)} | positive ${pct(item.positive_rate)}</span></div>`).join("");
document.getElementById("monthly-body").innerHTML=[...report.monthly].reverse().map(x=>`<tr><td>${x.month}</td><td class="${cls(x.return)}">${pct(x.return)}</td><td class="${cls(x.max_drawdown)}">${pct(x.max_drawdown)}</td><td>${pct(x.avg_exposure)}</td><td>${x.bear_days}</td></tr>`).join("");
document.getElementById("shock-body").innerHTML=report.risk_sessions.map(x=>{const transition=x.bear_trigger?"Bear entry":x.bear_reversal?"Reversal":x.shock?"Shock":"Ongoing";return `<tr class="shock"><td>${x.date}</td><td>${x.signal_date}</td><td>${x.regime}</td><td>${transition}</td><td>${x.positions}</td><td class="${cls(x.return)}">${pct(x.return)}</td></tr>`;}).join("")||`<tr><td colspan="6">No risk-off sessions</td></tr>`;
const episodeSummary=report.episode_summary;
document.getElementById("episode-count").textContent=episodeSummary.episodes.toLocaleString();
document.getElementById("episode-closed").textContent=episodeSummary.closed.toLocaleString();
document.getElementById("episode-win-rate").textContent=episodeSummary.win_rate==null?"N/A":pct(episodeSummary.win_rate);
document.getElementById("episode-avg-return").textContent=episodeSummary.avg_net_return==null?"N/A":pct(episodeSummary.avg_net_return);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const allHistory=report.history;
let visibleHistory=[...allHistory];
const rangeStart=document.getElementById("range-start"),rangeEnd=document.getElementById("range-end");
rangeStart.value=report.meta.start;rangeEnd.value=report.meta.end;
const ns="http://www.w3.org/2000/svg",chartRegistry={};
function make(tag,attrs,text=null){const node=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==null)node.textContent=text;return node;}
function tooltipText(id,row){
 if(id==="equity-chart")return `<strong>${row.date}</strong><br>Net return ${pct(row.return)}<br>Equity ${row.equity.toFixed(4)}`;
 if(id==="drawdown-chart")return `<strong>${row.date}</strong><br>Drawdown ${pct(row.drawdown)}<br>Net return ${pct(row.return)}`;
 return `<strong>${row.date}</strong><br>Cash ${pct(row.cash_weight)}<br>Positions ${row.positions}<br>Regime ${esc(row.regime)}`;
}
function syncHover(index){
 const row=visibleHistory[index];if(!row)return;
 Object.entries(chartRegistry).forEach(([id,c])=>{
  const xx=c.x(index),yy=c.y(row[c.field]);c.guide.setAttribute("x1",xx);c.guide.setAttribute("x2",xx);c.guide.style.display="";
  c.point.setAttribute("cx",xx);c.point.setAttribute("cy",yy);c.point.style.display="";
  c.tooltip.innerHTML=tooltipText(id,row);c.tooltip.style.display="block";
  c.tooltip.style.left=`${Math.max(6,Math.min(c.width-166,xx+9))}px`;c.tooltip.style.top="8px";
 });
}
function hideHover(){Object.values(chartRegistry).forEach(c=>{c.guide.style.display="none";c.point.style.display="none";c.tooltip.style.display="none";});}
function drawChart(svgId,field,css,fixedMin=null,fixedMax=null){
 const svg=document.getElementById(svgId),tooltip=document.getElementById(svgId.replace("-chart","-tooltip"));svg.replaceChildren();
 const data=visibleHistory,w=svg.clientWidth||500,h=svg.clientHeight||300,m={l:48,r:12,t:14,b:26};svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
 if(!data.length)return;const values=data.map(row=>Number(row[field])),lo=fixedMin??Math.min(...values),hi=fixedMax??Math.max(...values),span=Math.max(hi-lo,1e-9);
 const x=i=>m.l+(data.length===1?0.5:i/(data.length-1))*(w-m.l-m.r),y=v=>m.t+(hi-v)/span*(h-m.t-m.b);
 for(let i=0;i<=4;i++){const yy=m.t+i/4*(h-m.t-m.b),value=hi-i/4*span;svg.appendChild(make("line",{class:"grid",x1:m.l,x2:w-m.r,y1:yy,y2:yy}));svg.appendChild(make("text",{x:m.l-6,y:yy+4,"text-anchor":"end","font-size":"10",fill:"#65736f"},field==="equity"?value.toFixed(2):pct(value,0)));}
 svg.appendChild(make("polyline",{class:css,points:values.map((v,i)=>`${x(i)},${y(v)}`).join(" ")}));
 svg.appendChild(make("text",{x:m.l,y:h-6,"font-size":"10",fill:"#65736f"},data[0].date));svg.appendChild(make("text",{x:w-m.r,y:h-6,"text-anchor":"end","font-size":"10",fill:"#65736f"},data[data.length-1].date));
 const guide=make("line",{class:"hover-guide",x1:m.l,x2:m.l,y1:m.t,y2:h-m.b});guide.style.display="none";svg.appendChild(guide);
 const point=make("circle",{class:"hover-point",cx:m.l,cy:m.t,r:4});point.style.display="none";svg.appendChild(point);
 const overlay=make("rect",{class:"chart-overlay",x:m.l,y:m.t,width:w-m.l-m.r,height:h-m.t-m.b});svg.appendChild(overlay);
 chartRegistry[svgId]={field,x,y,guide,point,tooltip,width:w};
 overlay.addEventListener("mousemove",event=>{const rect=svg.getBoundingClientRect(),mouse=(event.clientX-rect.left)/rect.width*w,index=Math.max(0,Math.min(data.length-1,Math.round((mouse-m.l)/(w-m.l-m.r)*(data.length-1))));syncHover(index);});
 overlay.addEventListener("mouseleave",hideHover);
}
function drawCharts(){
 Object.keys(chartRegistry).forEach(key=>delete chartRegistry[key]);
 drawChart("equity-chart","equity","equity");
 drawChart("drawdown-chart","drawdown","drawdown",Math.min(...visibleHistory.map(x=>x.drawdown)),0);
 drawChart("cash-chart","cash_weight","cash",0,1);
}
const episodeBody=document.getElementById("episode-body"),episodeSymbolFilter=document.getElementById("episode-symbol-filter"),episodeStatusFilter=document.getElementById("episode-status-filter"),episodeOutcomeFilter=document.getElementById("episode-outcome-filter");
const episodePageSize=document.getElementById("episode-page-size"),episodeRowCount=document.getElementById("episode-row-count"),episodePageInfo=document.getElementById("episode-page-info"),episodePrevPage=document.getElementById("episode-prev-page"),episodeNextPage=document.getElementById("episode-next-page");
let episodePage=1,episodeSortKey="entry_date",episodeSortDirection="desc";const expandedEpisodes=new Set();
function filteredEpisodes(){
 const symbol=episodeSymbolFilter.value.trim().toUpperCase(),status=episodeStatusFilter.value,outcome=episodeOutcomeFilter.value,start=rangeStart.value,end=rangeEnd.value;
 const rows=report.episodes.filter(row=>(!start||(row.exit_date??row.entry_date)>=start)&&(!end||row.entry_date<=end)&&(!symbol||row.symbol.includes(symbol))&&(!status||row.status===status)&&(!outcome||(outcome==="win"?row.net_return>0:row.net_return<=0)));
 return rows.sort((a,b)=>{const av=a[episodeSortKey]??"",bv=b[episodeSortKey]??"",cmp=typeof av==="number"?av-bv:String(av).localeCompare(String(bv));return episodeSortDirection==="asc"?cmp:-cmp;});
}
function episodePathTable(path){
 return `<div style="grid-column:1/-1"><span class="label">Daily open-to-open return path</span><div class="table-wrap" style="max-height:260px;margin-top:8px"><table><thead><tr><th>Date</th><th>Daily return</th><th>Cumulative return</th></tr></thead><tbody>${path.map(point=>`<tr><td>${point.date}</td><td class="${cls(point.return)}">${pct(point.return)}</td><td class="${cls(point.cumulative_return)}">${pct(point.cumulative_return)}</td></tr>`).join("")}</tbody></table></div></div>`;
}
function renderEpisodes(){
 const rows=filteredEpisodes(),size=Number(episodePageSize.value),pages=Math.max(1,Math.ceil(rows.length/size));episodePage=Math.min(episodePage,pages);
 const shown=rows.slice((episodePage-1)*size,episodePage*size);
 episodeBody.innerHTML=shown.map(row=>{const key=`${row.symbol}|${row.entry_signal_date}`,open=expandedEpisodes.has(key);return `<tr><td><strong>${esc(row.symbol)}</strong></td><td>${row.entry_date??"N/A"}</td><td>${row.exit_date??"Open"}</td><td>${row.sessions}</td><td>${row.entry_score.toFixed(4)}</td><td class="${cls(row.gross_return)}">${pct(row.gross_return)}</td><td class="${cls(row.net_return)}">${pct(row.net_return)}</td><td class="${cls(row.max_runup)}">${pct(row.max_runup)}</td><td class="${cls(row.max_drawdown)}">${pct(row.max_drawdown)}</td><td>${row.status}</td><td><button class="expand-button" data-episode-expand="${esc(key)}">${open?"Hide":"View"}</button></td></tr>${open?`<tr class="detail-row"><td colspan="11"><div class="detail-content">${episodePathTable(row.path)}</div></td></tr>`:""}`;}).join("");
 episodeRowCount.textContent=`${rows.length.toLocaleString()} holding periods`;episodePageInfo.textContent=`Page ${episodePage} / ${pages}`;episodePrevPage.disabled=episodePage<=1;episodeNextPage.disabled=episodePage>=pages;
}
function updateEpisodeSortIndicators(){document.querySelectorAll("th.episode-sortable").forEach(th=>{th.classList.remove("sort-asc","sort-desc");if(th.dataset.episodeKey===episodeSortKey)th.classList.add(episodeSortDirection==="asc"?"sort-asc":"sort-desc");});}
const body=document.getElementById("history-body"),dateFilter=document.getElementById("date-filter"),symbolFilter=document.getElementById("symbol-filter"),regimeFilter=document.getElementById("regime-filter");
const pageSize=document.getElementById("page-size"),rowCount=document.getElementById("row-count"),pageInfo=document.getElementById("page-info"),prevPage=document.getElementById("prev-page"),nextPage=document.getElementById("next-page");
let page=1,sortKey="date",sortDirection="desc";const expanded=new Set();
function filteredRows(){
 const exact=dateFilter.value,symbol=symbolFilter.value.trim().toUpperCase(),regime=regimeFilter.value;
 const start=rangeStart.value,end=rangeEnd.value;
 const rows=allHistory.filter(row=>(!start||row.date>=start)&&(!end||row.date<=end)&&(!exact||row.date===exact)&&(!regime||row.regime===regime)&&(!symbol||[...row.holdings,...row.bought,...row.sold].some(v=>v.includes(symbol))));
 return rows.sort((a,b)=>{const av=a[sortKey],bv=b[sortKey],cmp=typeof av==="number"?av-bv:String(av).localeCompare(String(bv));return sortDirection==="asc"?cmp:-cmp;});
}
function chips(values,type=""){return values.length?`<div class="chips">${values.map(v=>`<span class="chip ${type}">${esc(v)}</span>`).join("")}</div>`:"—";}
function renderRows(){
 const rows=filteredRows(),size=Number(pageSize.value),pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(page,pages);
 const shown=rows.slice((page-1)*size,page*size);
 body.innerHTML=shown.map(row=>{const key=row.date,open=expanded.has(key);return `<tr class="${row.regime==="shock"?"shock":""}"><td>${row.date}</td><td>${row.signal_date}</td><td class="${cls(row.return)}">${pct(row.return)}</td><td>${row.equity.toFixed(4)}</td><td class="${cls(row.drawdown)}">${pct(row.drawdown)}</td><td>${row.positions}</td><td>${esc(row.regime)}</td><td>${pct(row.turnover)}</td><td><span class="positive">+${row.bought.length}</span> / <span class="negative">-${row.sold.length}</span></td><td><button class="expand-button" data-expand="${key}">${open?"Hide":"View"}</button></td></tr>${open?`<tr class="detail-row"><td colspan="10"><div class="detail-content"><div><span class="label">Holdings (${row.holdings.length})</span>${chips(row.holdings)}</div><div><span class="label">Bought</span>${chips(row.bought,"buy")}</div><div><span class="label">Sold</span>${chips(row.sold,"sell")}</div></div></td></tr>`:""}`;}).join("");
 rowCount.textContent=`${rows.length.toLocaleString()} matching sessions`;pageInfo.textContent=`Page ${page} / ${pages}`;prevPage.disabled=page<=1;nextPage.disabled=page>=pages;
}
function updateSortIndicators(){document.querySelectorAll("th.sortable").forEach(th=>{th.classList.remove("sort-asc","sort-desc");if(th.dataset.key===sortKey)th.classList.add(sortDirection==="asc"?"sort-asc":"sort-desc");});}
function applyRange(){visibleHistory=allHistory.filter(row=>(!rangeStart.value||row.date>=rangeStart.value)&&(!rangeEnd.value||row.date<=rangeEnd.value));if(!visibleHistory.length)return;page=1;episodePage=1;drawCharts();renderRows();renderEpisodes();}
document.querySelectorAll("[data-range]").forEach(button=>button.addEventListener("click",()=>{const value=button.dataset.range;if(value==="all"){rangeStart.value=report.meta.start;}else{const count=Number(value),index=Math.max(0,allHistory.length-count);rangeStart.value=allHistory[index].date;}rangeEnd.value=report.meta.end;applyRange();}));
[rangeStart,rangeEnd].forEach(input=>input.addEventListener("change",applyRange));
[dateFilter,symbolFilter,regimeFilter].forEach(input=>input.addEventListener("input",()=>{page=1;renderRows();}));
[episodeSymbolFilter,episodeStatusFilter,episodeOutcomeFilter].forEach(input=>input.addEventListener("input",()=>{episodePage=1;renderEpisodes();}));
episodePageSize.addEventListener("change",()=>{episodePage=1;renderEpisodes();});episodePrevPage.addEventListener("click",()=>{episodePage=Math.max(1,episodePage-1);renderEpisodes();});episodeNextPage.addEventListener("click",()=>{episodePage+=1;renderEpisodes();});
document.getElementById("clear-episode-filters").addEventListener("click",()=>{episodeSymbolFilter.value="";episodeStatusFilter.value="";episodeOutcomeFilter.value="";episodePage=1;renderEpisodes();});
document.querySelectorAll("th.episode-sortable").forEach(th=>th.addEventListener("click",()=>{if(episodeSortKey===th.dataset.episodeKey)episodeSortDirection=episodeSortDirection==="asc"?"desc":"asc";else{episodeSortKey=th.dataset.episodeKey;episodeSortDirection="desc";}episodePage=1;updateEpisodeSortIndicators();renderEpisodes();}));
episodeBody.addEventListener("click",event=>{const button=event.target.closest("[data-episode-expand]");if(!button)return;const key=button.dataset.episodeExpand;expandedEpisodes.has(key)?expandedEpisodes.delete(key):expandedEpisodes.add(key);renderEpisodes();});
pageSize.addEventListener("change",()=>{page=1;renderRows();});prevPage.addEventListener("click",()=>{page=Math.max(1,page-1);renderRows();});nextPage.addEventListener("click",()=>{page+=1;renderRows();});
document.getElementById("clear-filters").addEventListener("click",()=>{dateFilter.value="";symbolFilter.value="";regimeFilter.value="";page=1;renderRows();});
document.querySelectorAll("th.sortable").forEach(th=>th.addEventListener("click",()=>{if(sortKey===th.dataset.key)sortDirection=sortDirection==="asc"?"desc":"asc";else{sortKey=th.dataset.key;sortDirection="desc";}page=1;updateSortIndicators();renderRows();}));
body.addEventListener("click",event=>{const button=event.target.closest("[data-expand]");if(!button)return;const key=button.dataset.expand;expanded.has(key)?expanded.delete(key):expanded.add(key);renderRows();});
window.addEventListener("resize",drawCharts);updateSortIndicators();updateEpisodeSortIndicators();renderEpisodes();applyRange();
</script>
</body>
</html>
"""
