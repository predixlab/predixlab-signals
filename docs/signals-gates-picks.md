# Scores, states, and events

## Model boundaries

Signal publication is model-independent. `SignalModelSpec` defines one model's
input columns, extractor, event vocabulary, event metadata, and ordered
`SignalFilter` chain. The shared pipeline validates that contract, runs the
model-owned filters, and normalizes fired rows for ranking and publication.
Every published row carries `signal_model`; model-specific aliases such as
`pivot_event` are added only by that model.

The current pivot implementation lives under
`predixlab_signals.signals.models.pivot`:

```text
config.py             pivot thresholds and windows
bottom_turns.py       BOTTOM_TURN detection and BUY_TURN confirmation
top_turns.py          TOP_TURN detection and SELL_TURN confirmation
trend_breakouts.py    independent TREND_BUY generation
base_breakouts.py     independent BASE_BREAKOUT generation
common.py             shared causal sequencing/range helpers
extractor.py          orchestration and conflict resolution
adapter.py            SignalModelSpec and pivot-owned filter chain
```

`signals.turning_points` remains a compatibility import only. A future model
should add its own package and `SignalModelSpec`; it does not need to add
conditions to the pivot extractor or reuse pivot filters.

## A state is not an event

The model produces a continuous `score` for each eligible symbol and trading day, then calibrates it into persistent states:

```text
STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
```

States reflect the current cross-sectional direction and amplitude assessment. Events are sparser because they require return-weighted Pivot strength, price location, reversal confirmation, cross-sectional and own-history gates, and duplicate suppression.

| `event` | `direction` | `kind` | User meaning |
| --- | --- | --- | --- |
| `BUY_TURN` | `long` | `reversal` | Confirmed reversal after a low setup |
| `SELL_TURN` | `short` | `reversal` | Confirmed reversal after a high setup |
| `BOTTOM_TURN` | `long` | `detection` | Low reversal detected but directional-confirmation protection did not pass |
| `TOP_TURN` | `short` | `detection` | High reversal detected but directional-confirmation protection did not pass |
| `TREND_BUY` | `long` | `trend` | Breakout with persistent bullish state; not a bottom reversal |
| `BASE_BREAKOUT` | `long` | `trend` | Volume-backed breakout from compressed volatility |

`BOTTOM_TURN` and `TOP_TURN` are reference detections that have not passed the
stricter pivot-confirmation evidence. Confirmation is tuned primarily for
pivot precision and true-pivot coverage; whether a downstream strategy trades
the confirmed event is a separate decision. `TREND_BUY` and `BASE_BREAKOUT`
are directional events, not turning-point labels.

Published rows expose confirmation explicitly through `pivot_event`; `event`
uses the same result for the app and turn backtest. A confirmed `BUY_TURN`
remains confirmed even inside a persistent decline. Trend context may inform a
separate trading strategy, but it must not relabel or hide a real low pivot.

## Why history is required

Daily processing scores a trailing panel and then extracts target-day events. Temporal rules need history for recent and long-window price location; trend and volatility context; a permitted setup-to-trigger interval; same-day cross-sectional and own-history percentiles; event cooldown and ATR rearming with a 20-session timeout; and trend-breakout windows.

A single-day cross section can produce no event even when a symbol state is strong. Latest-cross-section methods such as `build_scores()` suit state display, not event-sequence input.

## Event pipeline

For each symbol and date, the implementation:

1. checks model state and return-weighted Pivot-head gates;
2. tests proximity to recent range extremes;
3. filters directional conflicts with longer trend/range context;
4. pairs setup and price-reversal triggers within a bounded window;
5. removes records satisfying conflicting long and short paths;
6. suppresses duplicates by cooldown and ATR displacement;
7. independently detects qualified trend and base breakouts.

`BASE_BREAKOUT` requires the current high to exceed the prior 20-session high
and the close to finish in the top 35% of the trigger bar. The previous
session's 14-day ATR must be no more than 90% of its trailing 60-session mean,
current volume must be at least 1.25 times the prior 20-session mean, and the
current model state must be `BUY` or `STRONG_BUY`. It is evaluated only after
the reversal and established-trend families finish, so it never replaces an
existing event on the same bar.

Restricted delayed-confirmation and regional-candidate paths handle reversal evidence after a model peak. They still require causal history, price confirmation, and existing event priority; they cannot revise determined historical events.

Buy candidates formed while the 50-day trend is non-rising, or after a sharp
20-session decline that the slower average has not yet reflected, must pass
bottom confirmation. Reference bottoms can also become actionable after a
two-session rebound of at least 2%; weaker rebounds remain `BOTTOM_TURN`
detections unless another confirmation family passes. An exceptional rebound
can retain the prior session's strong low setup when the current Pivot rank
cools: the low must be fresh, the two-session rebound at least 3%, and the
preceding 20-session decline at least 8%.
An independent price-shock path handles capitulation rebounds that still have
weak own-history rank: price must rebound at least 7% in one session, close near
the bar high, remain close to a fresh low after an 8% decline, retain strong
cross-sectional low evidence, and occur under top-decile recent volatility.
Rebounds above 10% stay detections because their OOS follow-through was less
stable.

The sell side mirrors these protections. A fresh 20-session high followed by a
high-volatility decline of at least 5% that closes near the bar low can confirm
a top even when an older 60-session high is stale. A delayed confirmation path
also accepts an 8% crash within four sessions of a local high when both high
evidence ranks remain exceptional. Conversely, a sell in the bottom quarter of
the 250-session range is kept only as `TOP_TURN` when the low Pivot evidence is
at least as strong as the high evidence.

Independent right-side rescues cover pivots whose detection head strengthens
only on the confirmation bar. A fresh low can create a confirmed `BUY_TURN`
from an otherwise empty bar when both low ranks clear the 35th percentile, the
daily rebound is at least 2.5%, and the close sits in the upper 35% of the bar.
To reject one-day false breakouts, two-day momentum must be at least 2%; a
decisive close at least 1.5% above the prior high can waive that floor. This
raises right-side true-low hits from 733 to 1,013 OOS while slightly improving
precision from 71.93% to 72.25%. The mirrored downside path creates
`SELL_TURN` under 85th percentile high evidence and a 5% lower-range break; its
pivot precision is 79.4% in research and 77.0% in validation. State-label agreement and portfolio
returns do not decide whether these high-quality pivots are confirmed.

High-confidence `TOP_TURN` detections can also confirm causally on either of
the next two sessions. The source must be at a fresh 20-session high, one high
rank must be at least 93%, and the other at least 65%. Confirmation requires a
subsequent daily loss of at least 1%, a cumulative decline of at least 0.5%,
a close below the detection bar's low, and a close in the bottom 30% of the
current bar. The original detection remains `TOP_TURN`; the later bar publishes
`SELL_TURN`. This adds 180 OOS confirmations at 77.22% pivot precision,
including CHTR 2025-05-21 and 2025-07-09.

Two pivot-first low confirmations cover guarded detections without disabling
the broad protections. A fresh 20-session low with both low ranks above 60%,
positive one-/three-session reversal, and a clear advantage over high evidence
becomes `BUY_TURN`; this includes APO 2025-05-27. A deep capitulation candidate
blocked by both downtrend and incomplete-history protection can also confirm
when cross-sectional low evidence exceeds 80%, the prior 20-session decline is
at least 8%, and the rebound closes in the upper part of its bar; this includes
APO 2025-03-11.

A top detected inside a still-rising trend can also become `SELL_TURN` without
disabling the general uptrend guard. It requires a current `STRONG_SELL` state,
both high ranks above 85%, high evidence leading low by at least 30 percentage
points, a fresh one-session high, and a downside break that closes in the
bottom 10% of the bar. This covers strong distribution reversals such as APO
2026-06-18 while leaving ordinary pullbacks as `TOP_TURN`.

Two terminal conflict guards prevent confirmed events from becoming poor
entries after the original setup:

- a `BUY_TURN` returns to `BOTTOM_TURN` after price reaches at least 40% of its
  20-session range, five-session momentum is non-negative, and competing high
  Pivot evidence ranks in the day's top 30%;
- a `BUY_TURN` also returns to `BOTTOM_TURN` when the 20-session low is already
  at least five bars old and the current candle has rallied at least 2.5% from
  its open while closing in the top 15% of its range; this is a mature rebound,
  not a fresh low confirmation;
- a `SELL_TURN` returns to `TOP_TURN` when either the current model state is no
  longer bearish during a positive 10-/20-session advance, or the 60-session
  high is stale while current high evidence has fallen below the top decile.

Strong local-high and crash confirmations are exempt from the sell filters. A
narrow right-side reversal path is also protected when both current high Pivot
ranks remain exceptional, the signal is within three sessions of a 20-session
high, the bar closes in its bottom 10%, and price has either fallen at least 2%
over three sessions or 5% from that local high. These rules reduce validation
buys near true highs and remove the two largest stable false-sell categories
without discarding strongly confirmed tops.

Long consolidations followed by breakouts remain a separate trend problem.
Chronological research on 40–60 bar compressed bases did not remain stable in
the latest validation year, so those events are not forced through the reversal
filter. Qualified starts continue to use the independent `TREND_BUY` path.

## Volume-backed bottom confirmation

Volume-backed bottom confirmation is **enabled by default** in the shared Pivot
v1 extractor, so daily publishing, historical replay, and reports all use it.
`TurningPointConfig(bottom_volume_rebound=False)` explicitly disables this
family without disabling the other Pivot rules.
The candidate requires all of the following background conditions:

- volume at least 3 times the mean of the preceding 20 completed sessions;
- a decline of at least 8% over the 20 sessions ending on the previous session;
- a 20-session low formed or retested within the last 3 bars, with the current
  close still in the bottom 35% of the 20-session range.

The ordinary rebound still requires a **daily gain of at least 2.5% and a
strong close**. Only a long lower wick or bottom doji can waive the daily-return
floor, allowing flat or negative daily returns and bearish candle bodies.
The current bar must pass at least one of these three confirmation paths:

| Shape | Confirmation |
| --- | --- |
| Ordinary rebound | Daily gain at least 2.5% and close in the top 25% of the bar |
| Long lower wick | Lower wick at least 50% of the bar and twice the real body |
| Bottom doji | Real body at most 10% of the bar, lower wick at least 40% |

The real body is `abs(close - open)` and the lower wick is
`min(open, close) - low`. Ratios use `high - low`; zero-range or invalid candles
cannot confirm. Wick/doji paths require actual current-session `open` prices,
which both training and live scored panels carry. Upper-shadow-only gravestone
doji do not qualify. These are parallel paths, so doji and lower-wick bars do
not also need a top-quarter close.
An enabled rule requires a positive finite ordinary-return floor; `None`,
zero, or a negative floor cannot reopen an unrestricted strong-close path.

The trigger's volume is excluded from its reference mean, and all rolling
windows require their full history. This price-volume evidence can create
`BUY_TURN` without breaking the prior high or passing the ordinary Pivot-rank,
state, and bottom-confirmation gates. It can also promote an existing
`BOTTOM_TURN`; other existing events retain priority.

New additions observe earlier events, cooldown, and ATR rearming. Future
baseline events cannot retrospectively suppress a candidate; those later events
are preserved, so evaluation must account for multiple signals near one real
low rather than counting each as additional coverage.

The fixed three-fold comparison produced 50 incremental events: 31 positive
10-session returns (62.00%) and 40 ordinary-or-major low matches (80.00%).
One-to-one matching added coverage of 31 unique structural lows, including
14 major lows, without removing baseline events. Every below-floor event had
lower-wick or doji confirmation. These are limited historical signal outcomes,
not executable portfolio returns; the known AAPL example is not independent
validation evidence.

## Gates

- **Expected Pivot strength:** hold probability multiplied by positive expected directional return.
- **Same-day cross-sectional percentile:** compares expected Pivot strength across symbols on one date.
- **Trend-neutral percentile:** compares within 20-day-return groups, preventing one trend group from monopolizing events.
- **Own-history percentile:** compares a symbol with its prior expected Pivot strength.
- **Absolute/market-breadth gates:** compare causal panel means through the event date.
- **Price context:** requires recent extremes and observable reversal/breakout evidence.

All extended statistics must use the event date or earlier. Appending future panel dates must not change historical gates or events; the test suite verifies this. Thresholds are internal research parameters, numerous, and implementation-dependent, so this user document does not duplicate them. Rebuild walk-forward predictions and reevaluate after rule changes; do not tune only a final model or one period.

## Published events and ranking

`select_picks()` retains every triggered event and ranks it by `abs(score)` per date and direction. `rank` controls display order only, not event eligibility.

`PIVOT_MAX_PER_SIDE` is a consumer-side safety cap:

- `0` (default) does not truncate;
- a positive integer retains events whose per-direction rank does not exceed it;
- it is neither a model threshold nor a default Top-N strategy.

Published CSV fields have a fixed order:

```text
date, symbol, signal_model, event, pivot_event, actionable, direction, kind, rank,
action, score,
confidence, swing_position, p_up, exp_gain_pct, exp_loss_pct,
close, price
```

MongoDB documents also include `market`, `horizon`, `tag`, and `generated_at`.
Historical replay may add `forward_return`, `signal_return`, `excess_return`,
and `pivot_hit`.

## Related documents

- [Evaluation, backtesting, and reports](evaluation-backtesting.md)
- [Artifacts and output locations](artifacts-and-reports.md)
- [Production operations](operations.md)
