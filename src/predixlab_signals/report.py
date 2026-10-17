"""Standalone HTML reporting for out-of-sample historical picks."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predixlab_signals.config import PATHS
from predixlab_signals.signals.models.pivot import (
    BASE_BREAKOUT,
    BUY_TURN,
    SELL_TURN,
    TREND_BUY,
)


REQUIRED_PICK_COLUMNS = (
    "date",
    "symbol",
    "event",
    "direction",
    "signal_return",
)

REQUIRED_PRICE_COLUMNS = ("date", "symbol", "price")
ACCURACY_EVENTS = (BUY_TURN, SELL_TURN, TREND_BUY, BASE_BREAKOUT)
PIVOT_ACCURACY_EVENTS = (BUY_TURN, SELL_TURN)


def _accuracy_block(picks: pd.DataFrame) -> dict[str, Any]:
    outcomes = pd.to_numeric(picks["signal_return"], errors="coerce")
    resolved = picks[np.isfinite(outcomes)]
    wins = resolved["signal_return"] > 0
    return {
        "signals": int(len(picks)),
        "evaluated": int(len(resolved)),
        "pending": int(len(picks) - len(resolved)),
        "wins": int(wins.sum()),
        "accuracy": float(wins.mean()) if len(resolved) else None,
        "mean_signal_return": (
            float(resolved["signal_return"].mean()) if len(resolved) else None
        ),
    }


def _event_accuracy_block(picks: pd.DataFrame) -> dict[str, Any]:
    """Use pivot precision for turn events and directional accuracy otherwise."""
    directional = _accuracy_block(picks)
    if "pivot_hit" not in picks.columns:
        return {**directional, "accuracy_kind": "directional"}

    resolved = picks[picks["pivot_hit"].notna()]
    if resolved.empty:
        return {**directional, "accuracy_kind": "directional"}

    hits = resolved["pivot_hit"].astype(bool)
    return {
        **directional,
        "evaluated": int(len(resolved)),
        "pending": int(len(picks) - len(resolved)),
        "wins": int(hits.sum()),
        "accuracy": float(hits.mean()),
        "accuracy_kind": "pivot",
    }


def summarize_pick_accuracy(picks: pd.DataFrame) -> dict[str, Any]:
    """Summarize directional accuracy using resolved pick outcomes only."""
    missing = [column for column in REQUIRED_PICK_COLUMNS if column not in picks.columns]
    if missing:
        raise ValueError(f"Pick report input is missing {missing}")

    frame = picks.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    directional = frame[frame["event"].isin(ACCURACY_EVENTS)]
    result: dict[str, Any] = {
        "events": int(len(frame)),
        "overall": _accuracy_block(directional),
        "by_event": {
            str(event): _event_accuracy_block(group)
            for event, group in directional.groupby("event", sort=True)
        },
        "by_symbol": {
            str(symbol): _accuracy_block(
                group[group["event"].isin(ACCURACY_EVENTS)]
            )
            for symbol, group in frame.groupby("symbol", sort=True)
        },
    }

    if "pivot_hit" in frame.columns:
        pivot_event = (
            frame["pivot_event"] if "pivot_event" in frame.columns else frame["event"]
        )
        reversals = frame[
            pivot_event.isin(PIVOT_ACCURACY_EVENTS)
            & frame["pivot_hit"].notna()
        ].copy()
        reversals["_pivot_event"] = pivot_event.loc[reversals.index]
        tops = reversals[reversals["_pivot_event"] == SELL_TURN]
        result["top_accuracy"] = {
            "evaluated": int(len(tops)),
            "hits": int(tops["pivot_hit"].astype(bool).sum()),
            "accuracy": (
                float(tops["pivot_hit"].astype(bool).mean()) if len(tops) else None
            ),
        }
        result["pivot_accuracy"] = {
            "evaluated": int(len(reversals)),
            "hits": int(reversals["pivot_hit"].astype(bool).sum()),
            "accuracy": (
                float(reversals["pivot_hit"].astype(bool).mean())
                if len(reversals)
                else None
            ),
            "by_event": {
                str(event): {
                    "evaluated": int(len(group)),
                    "hits": int(group["pivot_hit"].astype(bool).sum()),
                    "accuracy": float(group["pivot_hit"].astype(bool).mean()),
                }
                for event, group in reversals.groupby("_pivot_event", sort=True)
            },
        }
    else:
        result["top_accuracy"] = {"evaluated": 0, "hits": 0, "accuracy": None}
        result["pivot_accuracy"] = {
            "evaluated": 0,
            "hits": 0,
            "accuracy": None,
            "by_event": {},
        }
    return result


REPORT_COLUMNS = (
    "date",
    "symbol",
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
    "close",
    "price",
    "forward_return",
    "signal_return",
    "excess_return",
    "pivot_hit",
)


@dataclass(frozen=True)
class PickReportResult:
    """Files and summary produced by a pick report run."""

    html_path: Path
    picks_path: Path
    summary: dict[str, Any]


def _price_history(prices: pd.DataFrame, picks: pd.DataFrame) -> dict[str, list[list[Any]]]:
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"Pick report price history is missing {missing}")

    frame = prices[list(REQUIRED_PRICE_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    pick_dates = pd.to_datetime(picks["date"])
    symbols = set(picks["symbol"].astype(str))
    frame = frame[
        frame["symbol"].astype(str).isin(symbols)
        & frame["date"].between(pick_dates.min(), pick_dates.max())
        & np.isfinite(frame["price"])
    ]
    frame = frame.sort_values(["symbol", "date"], kind="stable").drop_duplicates(
        ["symbol", "date"], keep="last"
    )
    available = set(frame["symbol"].astype(str))
    if missing_symbols := sorted(symbols - available):
        raise ValueError(f"Pick report has no price history for {missing_symbols}")

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return {
        str(symbol): group[["date", "price"]].to_numpy().tolist()
        for symbol, group in frame.groupby("symbol", sort=True)
    }


def _report_payload(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    market: str,
    horizon: int,
    tag: str,
) -> dict[str, Any]:
    frame = picks.sort_values(["date", "symbol", "event"], kind="stable").copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    columns = [column for column in REPORT_COLUMNS if column in frame.columns]
    records = json.loads(frame[columns].to_json(orient="records"))
    return {
        "meta": {
            "market": market.upper(),
            "horizon": int(horizon),
            "tag": tag,
            "start": frame["date"].min(),
            "end": frame["date"].max(),
            "symbols": int(frame["symbol"].nunique()),
        },
        "summary": summarize_pick_accuracy(picks),
        "picks": records,
        "price_history": _price_history(prices, picks),
    }


def render_pick_report(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
) -> str:
    """Render a self-contained pick report with sample-out price context."""
    if picks.empty:
        raise ValueError("Pick report input is empty")

    payload = _report_payload(picks, prices, market, horizon, tag)
    payload_json = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":")
    ).replace("</", "<\\/")
    title = escape(f"{market.upper()} Pivot Detection Report")
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace(
        "__REPORT_DATA__", payload_json
    )


def write_pick_report(
    picks: pd.DataFrame,
    prices: pd.DataFrame,
    output_path: Path,
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
) -> PickReportResult:
    """Persist the exact pick frame used by the self-contained HTML report."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    picks_path = output_path.with_name(f"{output_path.stem}_picks.parquet")
    picks.to_parquet(picks_path, index=False)
    output_path.write_text(
        render_pick_report(picks, prices, market=market, horizon=horizon, tag=tag),
        encoding="utf-8",
    )
    return PickReportResult(
        html_path=output_path,
        picks_path=picks_path,
        summary=summarize_pick_accuracy(picks),
    )


def build_pick_report(
    market: str = "us",
    horizon: int = 10,
    tag: str = "default",
    start: str | None = None,
    end: str | None = None,
    output_path: Path | None = None,
) -> PickReportResult:
    """Replay sample-out picks and build their reproducible report artifacts."""
    from predixlab_signals.jobs.backfill import build_historical_picks, load_walkforward

    picks = build_historical_picks(
        market=market,
        horizon=horizon,
        tag=tag,
        start=start,
        end=end,
        with_outcomes=True,
    )
    if picks.empty:
        raise ValueError("No historical picks were produced for the requested period")
    prices = load_walkforward(market, horizon, tag)[list(REQUIRED_PRICE_COLUMNS)]
    destination = output_path or (
        PATHS.report_dir(market) / f"picks_report_h{horizon}_{tag}.html"
    )
    return write_pick_report(
        picks,
        prices,
        destination,
        market=market,
        horizon=horizon,
        tag=tag,
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__TITLE__</title>
    <style>
        :root {
            --ink: #172126;
            --muted: #627076;
            --paper: #f7f9f7;
            --line: #d9dfdb;
            --teal: #087f72;
            --coral: #d85b42;
            --yellow: #f1c75b;
            --white: #ffffff;
            --event-buy-turn: #d9a91e;
            --event-bottom-turn: #c47f17;
            --event-sell-turn: #d85b42;
            --event-top-turn: #8b5fbf;
            --event-trend-buy: #087f72;
            --event-base-breakout: #2474a6;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background-color: var(--paper);
            background-image: linear-gradient(rgba(23,33,38,.035) 1px, transparent 1px),
                                                linear-gradient(90deg, rgba(23,33,38,.035) 1px, transparent 1px);
            background-size: 28px 28px;
            font-family: "Avenir Next", "Gill Sans", sans-serif;
            letter-spacing: 0;
        }
        header { background: var(--ink); color: var(--white); padding: 34px 24px 30px; }
        header .inner, main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; }
        .eyebrow { color: var(--yellow); font-size: 12px; font-weight: 700; text-transform: uppercase; }
        h1 { margin: 7px 0 4px; font-family: Georgia, serif; font-size: clamp(30px, 5vw, 52px); font-weight: 500; }
        .subtitle { color: #cbd5d1; margin: 0; }
        main { padding: 26px 0 48px; }
        .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
        .metric { min-height: 112px; background: var(--white); border: 1px solid var(--line); border-top: 4px solid var(--teal); border-radius: 6px; padding: 17px; }
        .metric:nth-child(2) { border-top-color: var(--coral); }
        .metric:nth-child(3) { border-top-color: var(--yellow); }
        .metric:nth-child(4) { border-top-color: var(--ink); }
        .metric-label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
        .metric-value { display: block; margin-top: 5px; font-family: Georgia, serif; font-size: 32px; }
        .metric-note { color: var(--muted); font-size: 12px; }
        section { margin-top: 28px; }
        .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 10px; }
        h2 { margin: 0; font-family: Georgia, serif; font-size: 23px; font-weight: 500; }
        .method { max-width: 720px; color: var(--muted); font-size: 13px; line-height: 1.5; }
        .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--white); }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 11px 13px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
        th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
        th { color: var(--muted); background: #eef2ef; font-size: 11px; text-transform: uppercase; }
        tbody tr:last-child td { border-bottom: 0; }
        .symbol-tools { display: flex; align-items: center; gap: 10px; }
        label { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
        .symbol-search { min-width: 220px; height: 38px; border: 1px solid #aeb9b4; border-radius: 4px; background: var(--white); color: var(--ink); padding: 0 10px; font: inherit; text-transform: uppercase; }
        .symbol-search:focus { border-color: var(--teal); outline: 2px solid rgba(8,127,114,.18); outline-offset: 1px; }
        .symbol-summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); border-bottom: 0; background: var(--white); }
        .symbol-stat { padding: 14px 16px; border-right: 1px solid var(--line); }
        .symbol-stat:last-child { border-right: 0; }
        .symbol-stat strong { display: block; margin-top: 2px; font-family: Georgia, serif; font-size: 22px; }
        .chart-shell { height: 320px; overflow-x: auto; border: 1px solid var(--line); background: var(--white); padding: 12px; }
        #signal-chart { width: 100%; min-width: 780px; height: 100%; display: block; }
        .axis { stroke: #aeb9b4; stroke-width: 1; }
        .grid-line { stroke: #e7ebe8; stroke-width: 1; }
        .price-line { fill: none; stroke: var(--ink); stroke-width: 2; vector-effect: non-scaling-stroke; }
        .chart-label { fill: var(--muted); font-size: 11px; }
        .hover-guide { stroke: #6f7d77; stroke-width: 1; stroke-dasharray: 4 3; pointer-events: none; }
        .hover-point { fill: var(--white); stroke: var(--ink); stroke-width: 2; pointer-events: none; }
        .hover-tooltip { pointer-events: none; }
        .hover-tooltip rect { fill: rgba(20,31,27,.94); stroke: rgba(255,255,255,.2); }
        .hover-tooltip text { fill: #ffffff; font-size: 11px; }
        .hover-tooltip .hover-date { font-weight: 700; }
        .chart-overlay { fill: transparent; cursor: crosshair; }
        .legend { display: flex; flex-wrap: wrap; gap: 16px; padding: 9px 12px; color: var(--muted); background: #eef2ef; font-size: 12px; }
        .legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; margin-right: 6px; border-radius: 50%; vertical-align: 1px; }
        .legend .price::before { width: 18px; height: 2px; border-radius: 0; background: var(--ink); vertical-align: 4px; }
        .legend .buy-turn::before { background: var(--event-buy-turn); }
        .legend .bottom-turn::before { background: var(--event-bottom-turn); }
        .legend .sell-turn::before { background: var(--event-sell-turn); }
        .legend .top-turn::before { background: var(--event-top-turn); }
        .legend .trend-buy::before { background: var(--event-trend-buy); }
        .legend .base-breakout::before { background: var(--event-base-breakout); }
        .positive { color: var(--teal); font-weight: 700; }
        .negative { color: var(--coral); font-weight: 700; }
        .pending-text { color: var(--muted); }
        .event { display: inline-block; border-left: 3px solid var(--event-buy-turn); padding-left: 7px; font-weight: 700; }
        .event.buy-turn { border-left-color: var(--event-buy-turn); }
        .event.bottom-turn { border-left-color: var(--event-bottom-turn); }
        .event.sell-turn { border-left-color: var(--event-sell-turn); }
        .event.top-turn { border-left-color: var(--event-top-turn); }
        .event.trend-buy { border-left-color: var(--event-trend-buy); }
        .event.base-breakout { border-left-color: var(--event-base-breakout); }
        footer { margin-top: 28px; color: var(--muted); font-size: 12px; line-height: 1.5; }
        @media (max-width: 760px) {
            .metrics, .symbol-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .section-head { align-items: stretch; flex-direction: column; }
            .symbol-tools { align-items: stretch; flex-direction: column; }
            .symbol-search { width: 100%; }
            .symbol-stat:nth-child(2) { border-right: 0; }
            .symbol-stat { border-bottom: 1px solid var(--line); }
            .chart-shell { height: 270px; }
        }
        @media (max-width: 420px) {
            .metrics { grid-template-columns: 1fr; }
            header .inner, main { width: min(100% - 20px, 1180px); }
        }
    </style>
</head>
<body>
    <header>
        <div class="inner">
            <div class="eyebrow">Out-of-sample pick evaluation</div>
            <h1>__TITLE__</h1>
            <p class="subtitle" id="report-period"></p>
        </div>
    </header>
    <main>
        <div class="metrics">
            <div class="metric"><span class="metric-label">Total events</span><strong class="metric-value" id="total-events"></strong><span class="metric-note">all published signal events</span></div>
            <div class="metric"><span class="metric-label">Actionable trades</span><strong class="metric-value" id="trade-count"></strong><span class="metric-note" id="pending-trades"></span></div>
            <div class="metric"><span class="metric-label">Directional accuracy</span><strong class="metric-value" id="overall-accuracy"></strong><span class="metric-note">resolved actionable trades</span></div>
            <div class="metric"><span class="metric-label">Pivot accuracy</span><strong class="metric-value" id="pivot-accuracy"></strong><span class="metric-note">actionable buy/sell turns</span></div>
        </div>

        <section>
            <div class="section-head">
                <div><h2>Accuracy by pick type</h2><div class="method">Only actionable events are included. Buy/sell turns use pivot-hit accuracy; trend buys use resolved directional returns.</div></div>
            </div>
            <div class="table-wrap">
                <table><thead><tr><th>Event</th><th>Metric</th><th>Signals</th><th>Evaluated</th><th>Hits / wins</th><th>Accuracy</th><th>Mean return</th></tr></thead><tbody id="event-body"></tbody></table>
            </div>
        </section>

        <section>
            <div class="section-head">
                <div><h2>Signals by stock</h2><div class="method">Tables contain actionable picks only. Bottom/top detections remain visible on the chart as reference markers.</div></div>
                <div class="symbol-tools">
                    <label for="symbol-search">Stock</label>
                    <input class="symbol-search" id="symbol-search" type="search" list="symbol-options" placeholder="Search symbol" autocomplete="off" spellcheck="false" aria-label="Search stock symbol">
                    <datalist id="symbol-options"></datalist>
                </div>
            </div>
            <div class="symbol-summary">
                <div class="symbol-stat"><span class="metric-label">Signals</span><strong id="symbol-signals"></strong></div>
                <div class="symbol-stat"><span class="metric-label">Evaluated</span><strong id="symbol-evaluated"></strong></div>
                <div class="symbol-stat"><span class="metric-label">Accuracy</span><strong id="symbol-accuracy"></strong></div>
                <div class="symbol-stat"><span class="metric-label">Mean return</span><strong id="symbol-return"></strong></div>
            </div>
            <div class="chart-shell"><svg id="signal-chart" role="img" aria-label="Daily tradeable-price history with pick markers"></svg></div>
            <div class="legend"><span class="price">Daily price</span><span class="buy-turn">B: Buy turn</span><span class="bottom-turn">L: Bottom detection</span><span class="sell-turn">S: Sell turn</span><span class="top-turn">P: Top detection</span><span class="trend-buy">T: Trend buy</span><span class="base-breakout">R: Base breakout</span></div>
            <div class="table-wrap">
                <table><thead><tr><th>Date</th><th>Event</th><th>Direction</th><th>Signal price</th><th>Signal return</th><th>Excess return</th><th>Pivot hit</th></tr></thead><tbody id="signal-body"></tbody></table>
            </div>
        </section>
        <footer>Accuracy includes BUY_TURN, SELL_TURN, TREND_BUY, and BASE_BREAKOUT. BOTTOM_TURN and TOP_TURN are chart-only reference detections and do not enter any accuracy denominator.</footer>
    </main>
    <script id="report-data" type="application/json">__REPORT_DATA__</script>
    <script>
        const report = JSON.parse(document.getElementById("report-data").textContent);
        const pct = value => value == null ? "N/A" : `${(value * 100).toFixed(1)}%`;
        const ret = value => value == null ? "Pending" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
        const num = value => Number(value).toLocaleString();
        const esc = value => String(value ?? "").replace(/[&<>\"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
        const actionableEvents = new Set(["BUY_TURN", "SELL_TURN", "TREND_BUY", "BASE_BREAKOUT"]);
        const eventClass = event => String(event).toLowerCase().replaceAll("_", "-");
        const overall = report.summary.overall;
        document.getElementById("report-period").textContent = `${report.meta.start} to ${report.meta.end} | ${report.meta.symbols} stocks | ${report.meta.horizon}-session horizon | ${report.meta.tag}`;
        document.getElementById("total-events").textContent = num(report.summary.events);
        document.getElementById("trade-count").textContent = num(overall.signals);
        document.getElementById("pending-trades").textContent = `${num(overall.evaluated)} evaluated | ${num(overall.pending)} pending`;
        document.getElementById("overall-accuracy").textContent = pct(overall.accuracy);
        document.getElementById("pivot-accuracy").textContent = pct(report.summary.pivot_accuracy.accuracy);

        const eventBody = document.getElementById("event-body");
        eventBody.innerHTML = Object.entries(report.summary.by_event).map(([event, item]) => `<tr><td><span class="event ${eventClass(event)}">${esc(event)}</span></td><td>${item.accuracy_kind === "pivot" ? "Pivot hit" : "Directional"}</td><td>${num(item.signals)}</td><td>${num(item.evaluated)}</td><td>${num(item.wins)}</td><td>${pct(item.accuracy)}</td><td>${pct(item.mean_signal_return)}</td></tr>`).join("");

        const search = document.getElementById("symbol-search");
        const options = document.getElementById("symbol-options");
        const symbols = Object.keys(report.summary.by_symbol).sort();
        options.innerHTML = symbols.map(symbol => `<option value="${esc(symbol)}"></option>`).join("");

        function renderChart(rows, history) {
            const svg = document.getElementById("signal-chart");
            const ns = "http://www.w3.org/2000/svg";
            svg.replaceChildren();
            if (!history.length) return;

            const width = Math.max(svg.clientWidth, 780);
            const height = svg.clientHeight;
            svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
            const left = 62, right = width - 18, top = 18, bottom = height - 30;
            const points = history.map(([date, price]) => ({
                date,
                price: Number(price),
                time: new Date(`${date}T00:00:00`).getTime(),
            }));
            const minTime = points[0].time, maxTime = points[points.length - 1].time;
            const prices = points.map(point => point.price);
            const rawMin = Math.min(...prices), rawMax = Math.max(...prices);
            const padding = Math.max((rawMax - rawMin) * 0.08, rawMax * 0.01, 0.01);
            const minPrice = rawMin - padding, maxPrice = rawMax + padding;
            const x = time => minTime === maxTime ? (left + right) / 2 : left + ((time - minTime) / (maxTime - minTime)) * (right - left);
            const y = price => bottom - ((price - minPrice) / (maxPrice - minPrice)) * (bottom - top);
            const make = (tag, attributes, text = null) => {
                const element = document.createElementNS(ns, tag);
                Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, value));
                if (text != null) element.textContent = text;
                return element;
            };

            for (let index = 0; index <= 4; index += 1) {
                const ratio = index / 4;
                const ypos = top + ratio * (bottom - top);
                const price = maxPrice - ratio * (maxPrice - minPrice);
                svg.appendChild(make("line", { x1: left, x2: right, y1: ypos, y2: ypos, class: "grid-line" }));
                svg.appendChild(make("text", { x: left - 8, y: ypos + 4, "text-anchor": "end", class: "chart-label" }, price.toFixed(2)));
            }

            const pathData = points.map((point, index) => `${index ? "L" : "M"}${x(point.time).toFixed(2)},${y(point.price).toFixed(2)}`).join(" ");
            svg.appendChild(make("path", { d: pathData, class: "price-line" }));

            const priceByDate = new Map(points.map(point => [point.date, point.price]));
            const eventsByDate = new Map();
            rows.forEach(row => {
                if (!eventsByDate.has(row.date)) eventsByDate.set(row.date, []);
                eventsByDate.get(row.date).push(row.event);
            });
            const markerStyle = {
                BOTTOM_TURN: { color: "var(--event-bottom-turn)", label: "L", offset: 20 },
                BUY_TURN: { color: "var(--event-buy-turn)", label: "B", offset: 20 },
                SELL_TURN: { color: "var(--event-sell-turn)", label: "S", offset: -20 },
                TOP_TURN: { color: "var(--event-top-turn)", label: "P", offset: -20 },
                TREND_BUY: { color: "var(--event-trend-buy)", label: "T", offset: 20 },
                BASE_BREAKOUT: { color: "var(--event-base-breakout)", label: "R", offset: 20 },
            };
            rows.forEach(row => {
                const price = priceByDate.get(row.date) ?? row.price;
                if (!Number.isFinite(Number(price))) return;
                const style = markerStyle[row.event] ?? markerStyle.BUY_TURN;
                const xpos = x(new Date(`${row.date}T00:00:00`).getTime());
                const priceY = y(Number(price));
                const labelY = Math.max(top + 10, Math.min(bottom - 10, priceY + style.offset));
                const group = make("g", { class: "signal-marker", "data-event": row.event });
                group.appendChild(make("line", { x1: xpos, x2: xpos, y1: priceY, y2: labelY, stroke: style.color, "stroke-width": 1.5 }));
                group.appendChild(make("circle", { cx: xpos, cy: priceY, r: 3.5, fill: "#ffffff", stroke: style.color, "stroke-width": 2 }));
                group.appendChild(make("circle", { cx: xpos, cy: labelY, r: 9, fill: style.color }));
                group.appendChild(make("text", { x: xpos, y: labelY + 3.5, "text-anchor": "middle", fill: "#ffffff", "font-size": 10, "font-weight": 700 }, style.label));
                const title = make("title", {}, `${row.date} ${row.event} | price ${Number(price).toFixed(2)} | ${ret(row.signal_return)}`);
                group.appendChild(title);
                svg.appendChild(group);
            });

            const dateLabels = [points[0], points[Math.floor(points.length / 2)], points[points.length - 1]];
            dateLabels.forEach((point, index) => {
                svg.appendChild(make("text", { x: x(point.time), y: height - 8, "text-anchor": index === 0 ? "start" : index === 2 ? "end" : "middle", class: "chart-label" }, point.date));
            });

            const guide = make("line", {
                x1: left, x2: left, y1: top, y2: bottom,
                class: "hover-guide", visibility: "hidden",
            });
            const hoverPoint = make("circle", {
                cx: left, cy: top, r: 4.5,
                class: "hover-point", visibility: "hidden",
            });
            const tooltip = make("g", {
                class: "hover-tooltip", visibility: "hidden",
            });
            const tooltipRect = make("rect", {
                x: 0, y: 0, width: 210, height: 68, rx: 5, ry: 5,
            });
            const dateText = make("text", { x: 10, y: 18, class: "hover-date" });
            const priceText = make("text", { x: 10, y: 37 });
            const signalText = make("text", { x: 10, y: 56 });
            tooltip.append(tooltipRect, dateText, priceText, signalText);
            svg.append(guide, hoverPoint, tooltip);

            const nearestPoint = targetTime => {
                let low = 0, high = points.length - 1;
                while (low < high) {
                    const mid = Math.floor((low + high) / 2);
                    if (points[mid].time < targetTime) low = mid + 1;
                    else high = mid;
                }
                if (low === 0) return points[0];
                const before = points[low - 1];
                const after = points[low];
                return targetTime - before.time <= after.time - targetTime ? before : after;
            };
            const overlay = make("rect", {
                x: left, y: top, width: right - left, height: bottom - top,
                class: "chart-overlay",
            });
            overlay.addEventListener("pointermove", event => {
                const bounds = svg.getBoundingClientRect();
                const pointerX = (event.clientX - bounds.left) * (width / bounds.width);
                const clampedX = Math.max(left, Math.min(right, pointerX));
                const ratio = (clampedX - left) / Math.max(right - left, 1);
                const point = nearestPoint(minTime + ratio * (maxTime - minTime));
                const xpos = x(point.time);
                const ypos = y(point.price);
                const signals = eventsByDate.get(point.date) ?? [];
                const signalLabel = signals.length ? [...new Set(signals)].join(", ") : "None";
                const tooltipWidth = Math.min(
                    290,
                    Math.max(210, 18 + Math.max(point.date.length, signalLabel.length + 8) * 6.5),
                );
                const tooltipX = xpos + tooltipWidth + 12 <= right
                    ? xpos + 10
                    : xpos - tooltipWidth - 10;
                const tooltipY = top + 8;

                guide.setAttribute("x1", xpos);
                guide.setAttribute("x2", xpos);
                guide.setAttribute("visibility", "visible");
                hoverPoint.setAttribute("cx", xpos);
                hoverPoint.setAttribute("cy", ypos);
                hoverPoint.setAttribute("visibility", "visible");
                tooltipRect.setAttribute("width", tooltipWidth);
                tooltip.setAttribute("transform", `translate(${tooltipX},${tooltipY})`);
                tooltip.setAttribute("visibility", "visible");
                dateText.textContent = `Date: ${point.date}`;
                priceText.textContent = `Price: ${point.price.toFixed(2)}`;
                signalText.textContent = `Signal: ${signalLabel}`;
            });
            overlay.addEventListener("pointerleave", () => {
                guide.setAttribute("visibility", "hidden");
                hoverPoint.setAttribute("visibility", "hidden");
                tooltip.setAttribute("visibility", "hidden");
            });
            svg.appendChild(overlay);
        }

        function renderSymbol(symbol) {
            const summary = report.summary.by_symbol[symbol];
            const chartRows = report.picks.filter(row => row.symbol === symbol);
            const rows = chartRows.filter(row => actionableEvents.has(row.event));
            document.getElementById("symbol-signals").textContent = num(summary.signals);
            document.getElementById("symbol-evaluated").textContent = num(summary.evaluated);
            document.getElementById("symbol-accuracy").textContent = pct(summary.accuracy);
            document.getElementById("symbol-return").textContent = pct(summary.mean_signal_return);
            renderChart(chartRows, report.price_history[symbol] ?? []);
            document.getElementById("signal-body").innerHTML = rows.slice().reverse().map(row => {
                const outcomeClass = row.signal_return == null ? "pending-text" : row.signal_return > 0 ? "positive" : "negative";
                const price = row.price ?? row.close;
                return `<tr><td>${esc(row.date)}</td><td><span class="event ${eventClass(row.event)}">${esc(row.event)}</span></td><td>${esc(row.direction)}</td><td>${price == null ? "N/A" : Number(price).toFixed(2)}</td><td class="${outcomeClass}">${ret(row.signal_return)}</td><td>${ret(row.excess_return)}</td><td>${row.pivot_hit == null ? "N/A" : row.pivot_hit ? "Yes" : "No"}</td></tr>`;
            }).join("");
        }

        function matchSymbol(query) {
            const normalized = query.trim().toUpperCase();
            if (!normalized) return null;
            return symbols.find(symbol => symbol === normalized)
                ?? symbols.find(symbol => symbol.startsWith(normalized))
                ?? symbols.find(symbol => symbol.includes(normalized))
                ?? null;
        }

        function commitSymbol() {
            const symbol = matchSymbol(search.value);
            if (!symbol) return;
            search.value = symbol;
            renderSymbol(symbol);
        }

        search.addEventListener("input", () => {
            const exact = symbols.find(symbol => symbol === search.value.trim().toUpperCase());
            if (exact) renderSymbol(exact);
        });
        search.addEventListener("change", commitSymbol);
        search.addEventListener("keydown", event => {
            if (event.key === "Enter") {
                event.preventDefault();
                commitSymbol();
            }
        });
        if (symbols.length) {
            search.value = symbols[0];
            renderSymbol(symbols[0]);
        }
    </script>
</body>
</html>
"""