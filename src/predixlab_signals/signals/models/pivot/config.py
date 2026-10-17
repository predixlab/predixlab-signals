"""Configuration for pivot reversal and trend-breakout signals."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class TurningPointConfig:
    """Rules converting the continuous score into discrete turning points."""

    range_window: int = 20
    #: Bars needed before the range gates report anything; below the window they
    #: expand as history accrues, which is what lets a newly-added constituent be
    #: scored at all.
    range_min_periods: int = 10
    context_min_periods: int = 20
    #: Buy only when the close sits in the bottom fraction of its recent range.
    trough_pct: float = 0.35
    #: Sell only when it sits in the top fraction.
    peak_pct: float = 0.65
    #: Trailing window over which the range position is ranked *against the
    #: symbol's own history*, and the rank thresholds that open a second,
    #: parallel path through the range gate. ``None`` on the window disables it.
    #:
    #: The absolute thresholds above encode an assumption that does not hold for
    #: a stock in an uptrend: that a real low prints near the bottom of the
    #: 20-day range. Measured on the walk-forward set, the range position of a
    #: confirmed swing low has a median of 0.079 in the steepest downtrends but
    #: **0.620 in mild uptrends and 0.746 in strong ones** — a pullback inside a
    #: rally simply never reaches the bottom 35%. The consequence was a
    #: structural exclusion rather than a threshold that was merely tight: of
    #: all true lows occurring in an uptrend, only **1.3%** could clear
    #: ``trough_pct``, versus 19.2% across the whole sample. AAPL between May
    #: and August 2025 is the canonical case — 55 of 84 sessions carried a
    #: ``STRONG_BUY`` state and price rose 8.8%, yet *zero* sessions satisfied
    #: the absolute gate, so the only event the period could produce was a sell.
    #:
    #: Ranking the range position within the symbol's own trailing year removes
    #: that trend dependence: "low for this stock lately" travels with the trend
    #: while "low in absolute range terms" does not. It works as advertised on
    #: its own terms — coverage of uptrend lows rises from 1.3% to 11.7%, buy
    #: recall from 13.2% to 16.1%, and locking the trend band the extra uptrend
    #: buys still discriminate at 4.5x-6.4x against the pivot labels.
    #:
    #: **It is off by default because those extra lows are economically empty.**
    #: Raw forward returns hid this: buys and sells both sit near +0.5%, so the
    #: comparison is dominated by trend beta. Subtracting each trend band's own
    #: mean removes the confound and reverses the verdict:
    #:
    #: =========================  ==============  ==============  ==========
    #: variant                    buy excess      sell excess     spread
    #: =========================  ==============  ==============  ==========
    #: absolute gate only         -0.028%         -0.124%         **+0.095%**
    #: + self-relative (0.50)     -0.192%         -0.064%         -0.127%
    #: + self-relative (0.30)     —               —               +0.014%
    #: + self-relative (0.15)     —               —               +0.041%
    #: =========================  ==============  ==============  ==========
    #:
    #: The degradation is monotone in how many uptrend buys are admitted, and at
    #: the setting that actually fixes the coverage gap the spread changes sign:
    #: the buy signal ends up predicting *worse* forward returns than the sell
    #: signal. The pivot labels call a shallow dip inside a rally a low, and by
    #: that definition detection genuinely improves; but the bar carries no
    #: information, because the name was rising and keeps rising. A timing signal
    #: that does not separate future direction is not accurate in any useful
    #: sense, whatever the label agreement says.
    #:
    #: The honest reading of the AAPL case is therefore the opposite of the first
    #: one: a steady rally with no real pullback offers no entry, and declining
    #: to signal is correct. Set a window here to trade recall for that spread
    #: deliberately.
    self_range_window: int | None = None
    self_range_min_periods: int = 60
    self_trough_pct: float = 0.50
    self_peak_pct: float = 0.50
    #: Self-relative counterpart of ``trigger_pct``: how far up its own recent
    #: distribution price may have travelled by the trigger bar.
    self_trigger_pct: float = 0.60
    #: Longer window used only as a *context guard*, and the absolute bound a
    #: signal may not contradict: a sell needs ``ctx >= context_sell_bound``, a
    #: buy needs ``ctx <= 1 - context_buy_bound``. ``None`` disables the guard.
    #:
    #: Selling a name that has already fallen is usually wrong, and the 20-day
    #: setup window is too short to notice: a stock in a sustained decline
    #: bounces inside its own recent range often enough to look like a top
    #: locally while sitting near multi-month lows. The guard supplies the
    #: longer context that the setup gate lacks.
    #:
    #: Both sides fail the same way and by the same margin. Grouped by 60-day
    #: range position, hit rate against confirmed pivots is:
    #:
    #: ===================  ==================  ==================
    #: 60-day position      sell hits a top     buy hits a low
    #: ===================  ==================  ==================
    #: bottom 25%           **39.3%**           53.2%
    #: 25-50%               47.9%               56.6%
    #: 50-75%               61.7%               49.3%
    #: top 25%              49.5%               **39.3%**
    #: ===================  ==================  ==================
    #:
    #: Forward 10-day return agrees and is the sharper test, because it does not
    #: depend on the pivot definition: sells fired at 60-day lows are followed by
    #: **+0.70%**, the highest of any bucket and the worst possible outcome for a
    #: sell. Buys fired at 60-day highs return +0.42% against +0.46% at the lows.
    #:
    #: Unusually for a filter, ``0.25`` on the sell side improves every headline
    #: number at once —
    #: buy lift 3.02x -> 3.04x, sell lift 2.82x -> 2.83x, trend-neutral buy/sell
    #: spread +0.095% -> **+0.135%** — while recall falls only 13.15% -> 13.05%
    #: (buy) and 15.36% -> 15.24% (sell). Sells at 60-day lows drop from 62 to 34
    #: and buys at 60-day highs from 89 to 44. Tightening further overshoots: at
    #: ``0.35`` the spread collapses to -0.001%.
    #: The two sides do not want the same bound. Selling a name at its
    #: multi-month lows is the failure this guard was built for; refusing to buy
    #: one near its multi-month highs is a weaker claim, because a shallow
    #: pullback inside a strong advance keeps a high 60-day reading by
    #: construction — the same trend dependence documented above. Splitting the
    #: bound keeps the sell-side protection intact while letting those pullbacks
    #: through.
    #:
    #: The buy side moved to 0.10 when the bar-range waiver landed. A wide
    #: reversal bar drives the 20-day range position up by construction — the
    #: recovery is what confirms the turn — and the 60-day reading inherits the
    #: same tilt, so a bound tuned against close-based setups was rejecting the
    #: new path's best cases on the margin. TSLA 2025-10-23 cleared every other
    #: gate and died here at 0.874 against a 0.85 ceiling. Measured across the
    #: panel the bound is nearly inert on the buy side: 0.15 -> 0.10 moves
    #: recall 19.7% -> 19.8% and precision 56.3% -> 56.2%, and further loosening
    #: to 0.0 buys nothing (19.8%, 55.8%). The sell side is untouched by this
    #: parameter and keeps its 0.25, which is where the guard actually earns its
    #: place.
    context_window: int | None = 60
    context_sell_bound: float = 0.25
    context_buy_bound: float = 0.10
    #: Terminal sell-side conflict guard. A setup can form above the 60-session
    #: floor and trigger only after price has fallen into the bottom of its
    #: one-year range. When the low Pivot evidence is then at least as strong as
    #: the high evidence, retain the detection as ``TOP_TURN`` but do not publish
    #: an actionable sell. This removes MSFT 2025-04-16 and only 23 OOS sells;
    #: raw and market-relative returns improve in both research and validation.
    sell_low_context_window: int | None = 250
    sell_low_context_min_periods: int = 120
    sell_low_context_bound: float = 0.25
    #: Terminal buy-side conflict guard. A confirmed low is no longer
    #: actionable after price has recovered into the upper 60% of its 20-day
    #: range while five-day momentum remains positive and the competing high
    #: Pivot score ranks in the day's top 30%. On validation this lowers buys
    #: near true highs from 7.56% to 5.01% while improving low precision,
    #: directional return, and win rate.
    buy_high_conflict_window: int | None = 20
    buy_high_conflict_min_periods: int = 20
    buy_high_conflict_min_position: float = 0.40
    buy_high_conflict_pivot_q: float = 0.70
    buy_high_conflict_min_return_5: float = 0.0
    #: Terminal late-rebound guard. Once the 20-session low is at least five
    #: bars old, a 2.5% bullish candle that closes in the top 15% of its range is
    #: a mature rebound rather than a fresh bottom confirmation. Demote it to
    #: ``BOTTOM_TURN``. Across both research folds and untouched validation this
    #: removes 15 false buys with negative forward returns and does not remove a
    #: single buy within three sessions of a true low.
    buy_late_rebound_min_body_return: float | None = 0.025
    buy_late_rebound_min_bars_since_low: int = 5
    buy_late_rebound_min_close_loc: float = 0.85
    #: False-sell category 1: the current state is no longer bearish and both
    #: 10- and 20-session momentum still describe an active advance. A pullback
    #: trigger in that context is retained as ``TOP_TURN`` rather than sold.
    sell_trend_conflict_enabled: bool = True
    sell_trend_conflict_min_return_10: float = 0.0
    sell_trend_conflict_min_return_20: float = 0.05
    #: False-sell category 2: the 60-session high is stale, price remains above
    #: its level ten sessions ago, and current high evidence has cooled below
    #: the top decile. This is an expired setup, not a fresh reversal.
    sell_stale_conflict_pivot_q: float | None = 0.90
    sell_stale_conflict_min_bars_since_high: int = 3
    sell_stale_conflict_min_return_10: float = 0.0
    #: Protect a narrow class of genuine right-side reversals from the two
    #: terminal sell-quality filters above. Both high-evidence ranks must remain
    #: exceptional near a fresh 20-session high, the trigger must close in the
    #: bottom tenth of its bar, and price must have either fallen 2% over three
    #: sessions or drawn down 5% from that local high. On OOS data this restores
    #: only 52 of 1,395 demoted sells; 75% are near true highs, while research
    #: and validation precision, market-relative return, and win rate improve.
    sell_right_reversal_pivot_q: float | None = 0.95
    sell_right_reversal_self_q: float = 0.90
    sell_right_reversal_max_bars_since_high: int = 3
    sell_right_reversal_max_close_loc: float = 0.10
    sell_right_reversal_max_return_1: float = 0.0
    sell_right_reversal_max_return_3: float = -0.02
    sell_right_reversal_max_drawdown: float = -0.05
    #: Minimum bars between two events on the same symbol. Without it a single
    #: multi-week setup fires dozens of near-identical signals.
    refractory: int = 5
    #: Require the bar to close beyond the previous bar's extreme, i.e. the turn
    #: has visibly begun. Raises the pivot lift from 1.77x to 2.43x.
    require_reversal: bool = True
    #: Bars the *setup* (strong state + price in the trough) may precede the
    #: *trigger* (the reversal bar). A turn does not happen in one session: on
    #: the day price bottoms there is no reversal yet, and by the day the
    #: reversal confirms price has usually left the trough zone. Demanding both
    #: on the same bar therefore misses most real bottoms - it held recall to
    #: 8.5%. Allowing a short lag lifts recall to ~22% at unchanged precision.
    #:
    #: Recalibrated after the return-weighted pivot model was introduced, using
    #: folds 1-2 for research and fold 3 as untouched validation. Extending the
    #: ordinary setup by one session, together with a 0.50 trigger cap, raises
    #: validation buy recall 16.34% -> 17.80% and sell recall 27.80% -> 29.06%.
    #: Buy precision moves only 54.17% -> 53.79%; sell precision improves
    #: 55.25% -> 55.54%. The gain also holds across both research folds.
    setup_window: int = 4
    #: Guarded-top detections may wait one additional session for a downside
    #: break. They are informational rather than actionable sells, so the extra
    #: confirmation latency is preferable to losing a genuine blow-off top.
    top_setup_window: int = 4
    #: Percentile of the detection head, ranked against the symbol's own recent
    #: history, above which a setup stays eligible **past** ``setup_window``.
    #: ``None`` restores the fixed-window behaviour.
    #:
    #: ``setup_window`` counts sessions, but the thing it approximates — how
    #: long a bottom takes to confirm — has no fixed length. A sharp V confirms
    #: the next session; a deep, drawn-out base can take a fortnight. TSLA
    #: 2025-11 is the case: the true low printed on 11-14 with the detection
    #: head at the 98.8th percentile of the symbol's own history, and the
    #: reversal bar did not arrive until 11-24, six sessions later. The setup
    #: was still obviously live the whole time and the window had closed.
    #:
    #: Widening ``setup_window`` is the wrong fix, because it extends *every*
    #: setup rather than the ones still supported. Measured: reaching 11-24 by
    #: window alone needs ``setup_window=6`` together with ``rearm_atr=1.0``,
    #: and costs 4.7pp of buy precision (56.2% -> 51.5%).
    #:
    #: Decay is the better clock. The setup survives while the model keeps
    #: ranking the name a candidate extreme and expires the moment it stops,
    #: which is what "the setup is still there" actually means. Measured:
    #:
    #: =========  ========  ========  =========  ========
    #: decay_q    buy prec  buy rec   sell prec  sell rec
    #: =========  ========  ========  =========  ========
    #: ``None``     56.2%     19.8%     51.6%      19.4%
    #: 0.80         56.3%     20.0%     —          —
    #: **0.60**   **55.7%** **20.8%** **50.8%**  **19.7%**
    #: 0.50         55.1%     21.7%     —          —
    #: 0.40         54.0%     22.7%     —          —
    #: =========  ========  ========  =========  ========
    #:
    #: 0.60 buys 1.0pp of buy recall for 0.5pp of precision, holds the
    #: market-neutral best point (+0.25% -> +0.24%) and improves the run-up to
    #: drawdown share (52.4% -> 52.9%). Recall rises in all four years
    #: (22.6->22.9, 17.8->19.4, 20.3->21.0, 18.5->19.8). Below 0.50 precision
    #: falls faster than recall rises.
    #:
    #: **It does not rescue TSLA 2025-11, and the reason is worth recording.**
    #: That was the case the mechanism was designed for, but the premise turned
    #: out to be false: the self-relative reading did not stay high while the
    #: base formed, it decayed — 0.988 on the 11-14 low, then 0.724, 0.792,
    #: **0.456**, 0.648, 0.860, and 0.364 on the 11-24 reversal bar. The chain
    #: breaks on 11-19 at any threshold that is not so loose as to be no gate at
    #: all. The model did not hold a quiet conviction through the base; it
    #: called the low once and then stopped seeing one. No setup-window rule,
    #: fixed or adaptive, can bridge that — it is a detection-head limitation,
    #: not a gating one.
    setup_decay_q: float | None = 0.60
    #: Hard stop on the decay extension, so a permanently elevated reading
    #: cannot keep a stale setup armed indefinitely.
    setup_max_window: int = 10
    #: Cap on how far price may already have travelled back into its range by
    #: the trigger bar. Without it, a lagged trigger buys after the bounce has
    #: run and per-trade quality falls; ``None`` disables the check.
    trigger_pct: float | None = 0.50
    #: Price travel, in ATRs, required before the *same* side may fire again.
    #: Without it a single multi-week decline produces a run of near-identical
    #: buy signals — the "whole leg flagged" failure. Measured in either
    #: direction, so a side re-arms whether the previous call worked out or was
    #: invalidated; keying it off the *opposite* signal instead deadlocks any
    #: name whose other side is filtered out (a permanent uptrend never
    #: generates a sell, which would lock the buy side forever after one event).
    #: A reversal bar this large (in ATRs) waives ``trigger_pct``. The sharpest
    #: turns collapse range position in a single session — a violent down day
    #: from a top can drop it from 0.89 to 0.55 — so demanding price still sit
    #: near the extreme rejects exactly the turns worth catching. The size of
    #: the bar *is* the confirmation. Measured: sell-side market-neutral return
    #: flips from +0.082% to -0.104% (i.e. into the correct sign) and buy lift
    #: rises from 2.55x to 2.58x.
    #:
    #: **The travel is measured close-to-close, and deliberately so.** A wide
    #: intraday range that closes back where it started is not evidence of
    #: anything; only a bar that *ends* somewhere new has confirmed a move.
    #:
    #: The alternative was measured, because it looks obviously better and is
    #: not. TSLA 2025-10-23 is the canonical case: the bar opened at 420, traded
    #: down to 413.90 and closed at 448.98 — its low held over the label window
    #: (``y_pivot_low_hold = 1``), detection head in the 98.8th percentile of the
    #: day's cross-section and the 99.9th of the symbol's own history, and it
    #: closed above the prior bar's high. It fired nothing, because the 20-day
    #: range position of the *close* was 0.633 against a ``trough_pct`` of 0.35.
    #: The bar's low sat at position 0.041; recovering 8.5% off it is what
    #: carried the close out of the trough zone. Close-to-close the bar moved
    #: 0.53 ATR and the waiver did not apply; its full range was 1.89 ATR and a
    #: range-based waiver would have admitted it.
    #:
    #: Across the panel, 562 confirmed lows are rejected this way after clearing
    #: both the detection head and the reversal trigger, and 68.5% of them span
    #: 1.5 ATR or more while only 21.7% do close-to-close. Admitting them on
    #: bar range is a large *detection* win and an *economic* loss.
    #:
    #: Judged on the day-10 close alone that verdict would be suspect, since an
    #: endpoint measure is exactly what cannot see a run that was given back.
    #: It survives the path measures, which is why it is recorded here:
    #:
    #: ======================  ======  ======  ========  ========  =========  =======
    #: buy list                n       lows    best      best_cs   worst      best>dd
    #: ======================  ======  ======  ========  ========  =========  =======
    #: shipped                 3,070   8.3%    +4.80%    **+0.26%** -4.37%    53.0%
    #: bars a range waiver adds  631   60.7%   +4.25%    **-0.28%** -4.73%    46.8%
    #: combined                3,701   17.2%   +4.71%    +0.17%    -4.43%     51.9%
    #: *panel baseline*        —       6.1%    +4.59%    0         -4.07%     52.5%
    #: ======================  ======  ======  ========  ========  =========  =======
    #:
    #: ``best`` is the highest the position reached at any point in the window
    #: and ``best_cs`` is that figure with the day's cross-section removed. The
    #: added bars reach a *lower* best point than an average bar of the panel,
    #: carry a deeper drawdown, and their ratio of run-up to drawdown is 0.900
    #: against the shipped funnel's 1.099. Normalised by ATR, to rule out the
    #: reading being a volatility artefact, the same gap holds: 1.65 against
    #: 1.79. Market-neutral best point is negative in three of four years
    #: (-0.00%, +0.11%, -0.61%, -0.33%) where the shipped funnel is positive in
    #: all four. The day-10 close agrees: excess +0.180% -> -0.280%.
    #:
    #: The added bars are 7.3x better at *being* pivots and worse on every
    #: measure of what followed. Requiring the close to finish in the upper part
    #: of its own bar does not rescue them: sweeping that threshold from 0.0 to
    #: 0.8 moves excess monotonically the wrong way, -0.280% -> -0.448%. Neither
    #: does tightening the 60-day context, the detection percentile, or how far
    #: the bar undercut the prior close; every one of them raises label
    #: precision and lowers excess.
    #:
    #: The reading is the module docstring's thesis in its sharpest form:
    #: ``trough_pct`` is not really a pivot detector, it is an economic filter.
    #: The edge comes from buying while price is still *at* the low, not from
    #: correctly identifying that a low occurred. A bar that has already
    #: recovered across its range has spent the thing being paid for, however
    #: well it marks the turn in hindsight. TSLA's own bar reached +5.59%
    #: against a -4.19% drawdown and was worth having; the population of bars
    #: that a rule admitting it would also admit is not.
    #:
    #: After moving the Pivot heads to future-hold targets, a time-isolated
    #: retune for published-pick accuracy selected 2.5 ATR. On the first two OOS
    #: folds it raised accuracy from 48.50% to 51.28% when combined with the
    #: uptrend guard below; on the untouched final fold it reached 50.79% versus
    #: 49.58%, while retaining 50.1% of evaluated picks and turning mean
    #: direction-adjusted return positive.
    reversal_waiver_atr: float | None = 2.5
    #: Lowered from 2.5 after measuring where confirmed pivots actually die.
    #: Of the true lows that clear every gate, the sequencing step was discarding
    #: **2,022 of 3,583** — more than half the survivors, and the single largest
    #: loss after the detection head itself. Requiring 2.5 ATR of travel before
    #: the same side may fire again is simply wider than the spacing of real
    #: pivots in a choppy name. At 2.0 every headline number improves at once:
    #: buy recall 16.23% -> 18.34%, sell recall 15.85% -> 17.84%, buy F 0.253 ->
    #: 0.277, sell F 0.242 -> 0.265, with lift unchanged and both the
    #: trend-neutral and cross-sectional spreads wider (+0.049% -> +0.101% and
    #: +0.278% -> +0.307%). Below 1.75 the spreads collapse again.
    rearm_atr: float = 2.0
    #: Maximum trading-session interval before a side re-arms without the
    #: required ATR excursion. This only admits the next otherwise-valid
    #: candidate; it does not create a signal when the timeout expires.
    #:
    #: At 20 sessions the full OOS panel adds 143 buys at 64.3% low precision
    #: and +1.29% mean 10-session market-relative return. Ten older signals are
    #: displaced by the newly admitted event sequence; those averaged -2.11%.
    #: Overall buy precision rises 59.07% -> 59.23% and relative return rises
    #: +0.28% -> +0.32%.
    rearm_max_bars: int | None = 20
    #: Re-arm on the largest excursion since the last same-side signal, not only
    #: on where price ended up.
    #:
    #: The endpoint test asks "how far is price from where I last sold?", which
    #: cannot distinguish never having left that level from a completed round
    #: trip back to it. ARM's 2026-06-18 top is the case: a confirmed pivot on
    #: 2.4x volume with a 3%-of-range body, at the end of a +43% six-session
    #: run, which then fell 39% over the next month. Every gate and the trigger
    #: passed on 06-22 — and it was dropped because the 06-05 sell sat 1.019
    #: away against a 1.123 threshold. Price had gone *down* to 4.83 and back
    #: up to 6.91 in between, an unambiguously different swing, and the run
    #: itself had inflated ATR 17% so the bar it had to clear rose with it.
    #:
    #: Measured, this is the largest single coverage gain in the extractor: buy
    #: recall 27.5% -> 41.4% and sell recall 26.9% -> 40.4%, with precision
    #: *rising* on both sides (55.1% -> 55.9%, 48.0% -> 48.7%). The events it
    #: newly admits are better than the incumbents (buy 57.1%, sell 50.2%),
    #: which is what you would expect: the endpoint test correlates with nothing
    #: about signal quality, so what it blocked was close to a random sample of
    #: candidates that had already cleared every quality gate.
    #:
    #: Signals stay spaced. The refractory still holds the floor at 7 days, the
    #: 10th-percentile gap moves 22 -> 15 days, and a name goes from 11 to 16
    #: signals over three years. Per year, buy precision improves in 2023-2025
    #: and gives back 2.4pp in 2026; sell precision is flat or better in all
    #: four. Density does rise: 14.5 -> 21 events per active session.
    #:
    #: It does not rescue TSLA 2025-11, UNH 2025-08 or AMD 2026-03. Those are
    #: blocked upstream on conviction and still need a better model.
    rearm_uses_path: bool = True
    #: Trade-layer guard: suppress actionable sells when the 50-day average has
    #: risen more than this over 20 sessions.
    #:
    #: Candidates blocked only here are preserved as ``TOP_TURN`` detector
    #: events, so the guard can reduce false directional sells without erasing
    #: confirmed high reversals from reports and diagnostics.
    max_uptrend_slope: float | None = 0.02
    #: Buy-side mirror of ``max_uptrend_slope``. Every non-rising 50-day average,
    #: or a 4% 20-session price decline before the average has reacted, routes a
    #: candidate through bottom confirmation before it becomes actionable.
    #: Rejected candidates remain ``BOTTOM_TURN`` detections rather than being
    #: erased. On the untouched latest fold this removes 11.2% of falling-edge
    #: buys, raises precision 53.79% -> 56.94%, and keeps recall nearly flat
    #: (17.80% -> 17.45%).
    min_downtrend_slope: float | None = 0.0
    max_downtrend_return_20: float | None = -0.04
    bottom_guard_low_conviction_q: float | None = 0.905
    bottom_guard_max_return_3: float = 0.0
    bottom_guard_max_return_10: float = 0.0
    bottom_guard_far_min_bars_since_low: int = 10
    bottom_guard_far_max_return_20: float = -0.04
    bottom_guard_requires_confirmation_history: bool = True
    #: Promote guarded top detections when price is still within three sessions
    #: of a 60-session high, volatility is elevated now or was extreme during
    #: the last three sessions, and short-horizon momentum has cooled. This
    #: keeps the filter causal while allowing the volatility burst to precede
    #: the right-side confirmation:
    #:
    #: ==========  =========  =========  ========
    #: period      retained   precision  lift
    #: ==========  =========  =========  ========
    #: 2023-2024      50.8%      59.4%   +11.1pp
    #: 2025           50.4%      61.1%   +11.9pp
    #: 2026           52.2%      57.0%    +8.7pp
    #: ==========  =========  =========  ========
    #:
    #: Volatility inputs are daily percentiles. No symbol, date, sector, or
    #: future-return condition is involved.
    top_confirm_volatility_q: float | None = 0.60
    top_confirm_recent_volatility_q: float = 0.70
    top_confirm_quiet_pivot_q: float = 0.96
    top_confirm_quiet_self_q: float = 0.96
    top_confirm_model_pivot_q: float | None = None
    top_confirm_model_self_q: float | None = None
    top_confirm_reversal_return_1: float = -0.04
    top_confirm_volatility_lookback: int = 3
    top_confirm_max_bars_since_high: int = 3
    top_confirm_max_return_3: float = 0.05
    top_confirm_max_return_5: float = 0.08
    top_confirm_fast_vol_window: int = 5
    top_confirm_slow_vol_window: int = 60
    top_confirm_high_window: int = 60
    #: Preserve exceptionally strong setup evidence after five-session momentum
    #: has stopped advancing near a fresh local high. This narrow path is
    #: applied after sequencing, so promoting a reference event cannot release
    #: a later sell candidate.
    top_confirm_memory_self_q: float | None = 0.97
    top_confirm_memory_window: int = 3
    top_confirm_memory_high_window: int = 20
    top_confirm_memory_max_bars_since_high: int = 3
    top_confirm_memory_max_return_5: float = 0.0
    #: A fresh local high followed by an extreme, high-volatility downside bar
    #: is sufficient price confirmation even when an older 60-session high
    #: makes ``top_confirm_max_bars_since_high`` stale. This covers the
    #: TSLA 2022-01-04 high, whose confirming bar arrived on 2022-01-05.
    top_confirm_local_pivot_q: float | None = 0.80
    top_confirm_local_self_q: float = 0.80
    top_confirm_local_max_bars_since_high: int = 1
    top_confirm_local_return_1: float = -0.05
    top_confirm_local_close_loc: float = 0.20
    top_confirm_local_volatility_q: float = 0.90
    #: Delayed crash confirmation for a setup that remains near a recent local
    #: high but did not reverse immediately. The current bar must fall at least
    #: 8%, close in its lower 35%, and retain top-decile cross-sectional and
    #: own-history high evidence under elevated volatility.
    top_confirm_crash_pivot_q: float | None = 0.90
    top_confirm_crash_self_q: float = 0.95
    top_confirm_crash_max_bars_since_high: int = 4
    top_confirm_crash_return_1: float = -0.08
    top_confirm_crash_close_loc: float = 0.35
    top_confirm_crash_volatility_q: float = 0.90
    #: Mild right-side confirmation for a high-confidence ``TOP_TURN`` that
    #: needs one or two more sessions to prove the break. The source detection
    #: must have one exceptional and one strong high rank near a fresh local
    #: high. Confirmation then requires a lower close beneath the source low.
    #: The admitted cohort has 81.9% research and 73.2% validation precision.
    top_followthrough_window: int | None = 2
    top_followthrough_min_rank: float = 0.65
    top_followthrough_max_rank: float = 0.93
    top_followthrough_max_source_bars_since_high: int = 1
    top_followthrough_max_cumulative_return: float = -0.005
    top_followthrough_max_return_1: float = -0.01
    top_followthrough_max_source_low_break: float = 0.0
    top_followthrough_max_close_loc: float = 0.30
    #: Confirm a narrow top inside a still-rising trend when the state is already
    #: strongly bearish and both Pivot ranks agree with a fresh lower-range
    #: break. The admitted cohort has 60.0% research and 62.5% validation pivot
    #: precision. This recovers APO 2026-06-18 without disabling the broad
    #: uptrend protection.
    top_uptrend_action_confirm_pivot_q: float | None = 0.85
    top_uptrend_action_confirm_self_q: float = 0.85
    top_uptrend_action_confirm_max_bars_since_high: int = 1
    top_uptrend_action_confirm_max_return_1: float = -0.005
    top_uptrend_action_confirm_max_close_loc: float = 0.10
    top_uptrend_action_confirm_min_advantage: float = 0.30
    top_guard_enabled: bool = True
    top_guard_boundary_bars_since_high: int = 3
    top_guard_boundary_pivot_q: float = 0.60
    top_guard_wide_boundary_pivot_q: float = 0.90
    top_guard_wide_boundary_close_loc: float = 0.15
    top_guard_wide_boundary_atr: float = 2.0
    top_guard_obsolete_min_bars_since_high: int = 20
    top_guard_obsolete_pivot_q: float = 0.50
    top_guard_stale_min_bars_since_high: int = 3
    top_guard_stale_max_bars_since_high: int = 10
    top_guard_stale_pivot_q: float = 0.93
    top_guard_stale_close_loc: float = 0.02
    #: Mirrored bottom confirmation thresholds. The 60-session-low paths handle
    #: immediate reversals; the local-low path allows high-confidence evidence
    #: to arrive up to ten sessions after a 20-session low. Against the prior
    #: 60-session-only filter, actionable buys rise from 2,628 to 2,819 while
    #: ``y_pivot_low_hold`` precision rises from 55.58% to 56.54%, with an
    #: improvement in every available OOS year.
    bottom_confirm_volatility_q: float | None = 0.60
    bottom_confirm_recent_volatility_q: float = 0.70
    bottom_confirm_volatility_self_q: float = 0.80
    bottom_confirm_quiet_pivot_q: float = 0.96
    bottom_confirm_quiet_self_q: float = 0.96
    bottom_confirm_stable_pivot_q: float = 0.93
    bottom_confirm_stable_self_q: float = 0.93
    bottom_confirm_stable_return_3: float = 0.0
    bottom_confirm_rebound_pivot_q: float = 0.90
    bottom_confirm_rebound_self_q: float = 0.93
    bottom_confirm_rebound_return_1: float = 0.02
    bottom_confirm_immediate_pivot_q: float = 0.70
    bottom_confirm_immediate_self_q: float = 0.80
    bottom_confirm_immediate_return_3: float = -0.01
    bottom_confirm_immediate_close_loc: float = 0.94
    bottom_confirm_local_window: int = 20
    bottom_confirm_local_max_bars: int = 10
    bottom_confirm_delayed_pivot_q: float = 0.94
    bottom_confirm_delayed_self_q: float = 0.90
    bottom_confirm_delayed_return_3: float = 0.02
    bottom_confirm_higher_low_pivot_q: float | None = 0.88
    bottom_confirm_higher_low_self_q: float = 0.90
    bottom_confirm_higher_low_max_local_bars: int = 3
    bottom_confirm_higher_low_return_1: float = 0.04
    bottom_confirm_higher_low_return_3: float = 0.01
    bottom_confirm_higher_low_return_5: float = -0.10
    bottom_confirm_higher_low_return_20: float = -0.10
    bottom_confirm_higher_low_close_loc: float = 0.90
    bottom_confirm_two_day_self_q: float | None = 0.90
    bottom_confirm_two_day_max_local_bars: int = 3
    bottom_confirm_two_day_return_2: float = 0.01
    bottom_confirm_two_day_deep_return_20: float = -0.08
    bottom_confirm_two_day_deep_return_2: float = 0.02
    bottom_confirm_two_day_range_atr: float = 0.75
    bottom_confirm_fresh_setup_self_q: float | None = 0.97
    bottom_confirm_fresh_setup_max_bars_since_low: int = 1
    bottom_confirm_fresh_setup_min_return_3: float = 0.0
    bottom_confirm_fresh_setup_reversal_atr: float = 2.0
    #: Preserve an exceptional setup peak when the confirming rebound causes
    #: the current Pivot rank to cool. The move must still be a fresh 60-session
    #: low, follow top-decile setup evidence, rebound at least 3% in two
    #: sessions, and remain inside a decline of at least 8% over 20 sessions.
    #: This promotes TSLA 2025-03-12 without relaxing ordinary bottom proof.
    bottom_confirm_setup_rebound_pivot_q: float | None = 0.93
    bottom_confirm_setup_rebound_self_q: float = 0.93
    bottom_confirm_setup_rebound_max_bars_since_low: int = 1
    bottom_confirm_setup_rebound_return_2: float = 0.03
    bottom_confirm_setup_rebound_max_return_20: float = -0.08
    bottom_confirm_setup_rebound_close_loc: float = 0.60
    #: Price-dominant capitulation confirmation. This covers cases where the
    #: rebound is decisive but the own-history Pivot rank has not recovered:
    #: a fresh low, at least a 7% one-day rebound, an 8% preceding decline, a
    #: close near the bar high, strong cross-sectional low evidence, and recent
    #: volatility in the top decile. Five of six OOS promotions were profitable.
    bottom_confirm_shock_pivot_q: float | None = 0.85
    bottom_confirm_shock_max_bars_since_low: int = 2
    bottom_confirm_shock_return_1: float = 0.07
    bottom_confirm_shock_max_return_1: float = 0.10
    bottom_confirm_shock_max_return_20: float = -0.08
    bottom_confirm_shock_close_loc: float = 0.85
    bottom_confirm_shock_volatility_q: float = 0.90
    bottom_confirm_max_bars_since_low: int = 3
    bottom_confirm_min_return_3: float = 0.01
    bottom_confirm_min_return_5: float = -0.08
    bottom_confirm_reversal_atr: float = 2.0
    bottom_confirm_reversal_close_loc: float = 0.80
    bottom_confirm_reversal_waives_return_5: bool = True
    bottom_confirm_panic_pivot_q: float = 0.96
    bottom_confirm_panic_self_q: float = 0.96
    bottom_confirm_panic_return_1: float = 0.03
    bottom_confirm_panic_atr: float = 1.80
    bottom_confirm_panic_close_loc: float = 0.65
    bottom_confirm_setup_memory_window: int = 3
    bottom_confirm_setup_memory_pivot_q: float = 0.96
    bottom_confirm_setup_memory_self_q: float = 0.96
    bottom_confirm_setup_memory_max_current_q: float = 0.50
    bottom_confirm_setup_memory_return_1: float = 0.02
    bottom_confirm_setup_memory_min_return_3: float = -0.01
    bottom_confirm_setup_memory_max_return_10: float = -0.05
    bottom_confirm_setup_memory_max_return_20: float = -0.05
    bottom_confirm_setup_memory_close_loc: float = 0.60
    bottom_confirm_setup_memory_volatility_q: float = 0.60
    #: Pivot-first confirmations for detections blocked by the broad trend and
    #: history guards. The fresh-local path confirms a rebound within one bar
    #: of a 20-session low when both low ranks clear 60% and lead high evidence.
    #: It runs at 70.2% research and 64.5% validation pivot precision and
    #: recovers APO 2025-05-27.
    bottom_pivot_fresh_q: float | None = 0.60
    bottom_pivot_fresh_self_q: float = 0.60
    bottom_pivot_fresh_max_bars_since_low: int = 1
    bottom_pivot_fresh_min_return_1: float = 0.0
    bottom_pivot_fresh_min_return_3: float = 0.0
    bottom_pivot_fresh_min_close_loc: float = 0.70
    bottom_pivot_fresh_min_advantage: float = 0.30
    #: The capitulation path covers candidates hit by both the downtrend and
    #: confirmation-history guards. It requires a deep decline, a fresh rebound,
    #: strong cross-sectional low evidence, and a clear advantage over the high
    #: head. Precision is 61.3% in research and 85.7% in validation; it recovers
    #: APO 2025-03-11 without weakening the general history guard.
    bottom_pivot_capitulation_q: float | None = 0.80
    bottom_pivot_capitulation_max_bars_since_low: int = 3
    bottom_pivot_capitulation_min_return_1: float = 0.02
    bottom_pivot_capitulation_max_return_20: float = -0.08
    bottom_pivot_capitulation_min_close_loc: float = 0.65
    bottom_pivot_capitulation_min_advantage: float = 0.50
    #: Emit a distinct entry when price leaves a low-volatility base on
    #: expanding volume and the model independently agrees with the direction.
    #: All windows use completed bars; only the breakout close, volume, and
    #: model state come from the trigger session.
    #:
    #: A predeclared OOS grid selected an intraday break of the prior 20-session
    #: high that closes in the top 35% of its bar, prior ATR ratio <= 0.90,
    #: volume >= 1.25x, and a BUY-class state. After existing-event collisions
    #: the regenerated strict OOS run emits 317 incremental events with +0.80%
    #: mean 10-session excess return and 64.4% raw win rate. Excess return is
    #: positive in 2024-2026; the partial 2023 cohort is negative. Allowing HOLD
    #: expands the cohort but cuts excess; range-width contraction is unstable
    #: and deliberately not used.
    base_breakout: bool = True
    base_breakout_window: int = 20
    base_breakout_min_close_location: float = 0.65
    base_breakout_atr_window: int = 60
    base_breakout_max_atr_ratio: float = 0.90
    base_breakout_volume_window: int = 20
    base_breakout_min_volume_ratio: float = 1.25
    base_breakout_states: tuple[str, ...] = ("BUY", "STRONG_BUY")
    #: Emit trend-confirmation entries alongside the pivot events. They answer a
    #: different question — "is this advance worth joining" rather than "has this
    #: turned" — and carry their own label so the two are never conflated.
    trend_entry: bool = True
    trend_slow_ma: int = 120
    trend_mid_ma: int = 50
    trend_slope_lookback: int = 20
    #: The trigger is a **new high**, not a moving-average cross, and the two are
    #: not close. Holding everything else fixed and swapping only the trigger:
    #:
    #: =========================  =====  ==========  =======  ==========
    #: trigger                    n      excess      raw med  win rate
    #: =========================  =====  ==========  =======  ==========
    #: cross above the 20-day MA  811    +0.077%     +0.81%   44.9%
    #: cross above the 50-day MA  423    -0.021%     +0.05%   42.7%
    #: hold above the 20-day MA   8,296  +0.500%     +2.19%   48.8%
    #: cross the upper Bollinger  470    +1.219%     +2.25%   48.8%
    #: new 20-day high            1,243  +1.648%     +3.42%   52.4%
    #: **new 60-day high**        1,043  **+2.153%** +3.71%   **53.9%**
    #: =========================  =====  ==========  =======  ==========
    #:
    #: The sample's own 60-session median is +2.51%, so every moving-average
    #: construction is *negative* on the median — crossing back above a mean is
    #: something a stock does on the way down as readily as on the way up. Only
    #: taking out a prior extreme carries information.
    #:
    #: Window length matters and 60 is the optimum: t rises 3.84 -> 4.34 -> 4.52
    #: across 20, 40 and 60 sessions and falls away after (3.82 at 90, 3.56 at
    #: 120). 60 also earns the widest breadth of the strong settings (266
    #: symbols, none over 4.9%) and is positive in all four years, which 20 is
    #: not. A 250-day window scores a higher mean but reads -17.3% in 2026 on a
    #: thin sample.
    trend_breakout_window: int = 60
    trend_entry_refractory: int = 20
    #: The model's endorsement is read as *sustained conviction*, not a
    #: single-day label, and only ``STRONG_BUY`` counts toward it.
    #:
    #: Accepting a same-day ``BUY`` was the weaker construction on every measure.
    #: Before the refractory is applied, requiring instead that at least 30% of
    #: the last ten sessions carried a ``STRONG_BUY`` roughly doubles the 60-day
    #: cross-sectional excess, from +0.876% (t = +2.71) to +1.648% (t = +3.84),
    #: and lifts the *median* raw 60-session return from +2.50% — identical to
    #: the sample's own +2.51%, i.e. no median edge at all — to +3.42%.
    #:
    #: Measured on the emitted events, after conflict resolution and the 20-bar
    #: refractory have taken their share, the gain is smaller but holds:
    #:
    #: ==================  ======  ==========  ============  ========
    #: endorsement         events  excess      raw median    win rate
    #: ==================  ======  ==========  ============  ========
    #: same-day BUY-class  987     +0.979%     +2.53%        49.5%
    #: sustained STRONG    515     **+1.475%** **+2.66%**    50.8%
    #: ==================  ======  ==========  ============  ========
    #:
    #: The sample median is +2.51%, so the old rule had no median advantage
    #: whatsoever and the new one has a slim positive one. Yearly excess is
    #: positive throughout: +1.88%, +0.19%, +2.64%, +0.93%. A 20-session window
    #: is comparably stable but gives back the median edge, which is the property
    #: worth keeping.
    trend_conviction_window: int = 10
    trend_conviction_share: float = 0.30
    trend_entry_states: tuple[str, ...] = ("STRONG_BUY",)
    buy_states: tuple[str, ...] = ("STRONG_BUY",)
    sell_states: tuple[str, ...] = ("STRONG_SELL",)
    #: Cross-sectional percentile of the dedicated pivot-detection head above
    #: which a bar qualifies as a setup, used *instead of* the state label when
    #: ``p_pivot_low`` / ``p_pivot_high`` are present.
    #:
    #: The state label comes from the mean-reversion score, which answers "is
    #: this name attractive" — a different question from "is this bar a local
    #: extreme". On AMG the gap is stark: at the genuine 2025 low the detection
    #: head read 0.487 while the state label said HOLD, and at the February 2026
    #: breakout the head read 0.021 while the state label said STRONG_BUY. It
    #: was right both times and the funnel ignored it. Switching the gate lifts
    #: pivot precision from 2.60x to 3.01x and recall from 9.3% to 11.2%.
    #: Top 5% of the day's cross-section. At 0.97 the gate covered only 26% of
    #: confirmed lows and missed obvious ones — MU's July 2025 bottom peaked at
    #: the 96.4th percentile, just under the cut. Loosening to 0.95 raises buy
    #: recall from 11.2% to 15.2% and sell recall from 5.3% to 7.1% at
    #: essentially unchanged precision (3.01x -> 2.99x).
    pivot_gate_q: float | None = 0.95
    #: Trailing window and threshold for a *second* gate that ranks the
    #: detection head against the symbol's own history instead of the day's
    #: cross-section. ``None`` on either disables it.
    #:
    #: A cross-sectional percentile forces the same 5% of names through every
    #: session, whatever the day actually looks like. On a market-wide bottom
    #: that is exactly backwards. On 2025-08-01, 33.4% of the universe printed a
    #: true swing low against a 6.1% baseline; AAPL's raw ``p_pivot_low`` was
    #: 0.206, more than three times the full-sample mean of 0.063 — the model saw
    #: it — but 248 of 482 names scored higher, so its percentile was 0.485 and
    #: the gate refused it. The rank is a zero-sum resource and a systematic day
    #: exhausts it.
    #:
    #: Ranking the head against the symbol's own trailing year is not zero-sum
    #: and stays a percentile, so the cross-machine reproducibility guarantee is
    #: untouched. Coverage of true lows by market regime:
    #:
    #: ==================  ===============  ===============
    #: market that day     cross-section    own history
    #: ==================  ===============  ===============
    #: heavy decline       13.5%            **54.4%**
    #: mild decline        23.9%            55.2%
    #: flat                34.8%            58.9%
    #: mild advance        40.5%            69.1%
    #: ==================  ===============  ===============
    #:
    #: The cross-sectional gate is weakest precisely where turns cluster.
    #: The mechanism is **not** symmetric, and the asymmetry has a cause:
    #: market bottoms are synchronised — panic prints them on one session — while
    #: tops form by gradual distribution. True-top density barely moves with the
    #: market (5.1%-8.2% across regimes) whereas true-low density reaches 33% on
    #: a washout day. So the cross-section only degenerates on the low side, and
    #: measured on the high side the self-percentile is the *weaker* of the two
    #: (16%-43% coverage against 24%-47%). It is therefore enabled for lows only;
    #: ``self_gate_high_q`` exists to make that choice explicit rather than
    #: implicit.
    self_gate_window: int | None = 250
    self_gate_min_periods: int = 60
    self_gate_low_q: float | None = 0.98
    self_gate_high_q: float | None = None
    #: A third path through the conviction gate: the detection probability as a
    #: multiple of the panel's own mean for this run. ``None`` disables it.
    #:
    #: Both percentile gates fail together on a synchronised bottom, and for
    #: different reasons. The cross-sectional rank is zero-sum, so a washout
    #: exhausts it — that is what ``self_gate_low_q`` exists to catch. But the
    #: self-relative rank is taken against the symbol's *own* trailing year, and
    #: a name that has been falling all year has spent that year printing
    #: elevated readings, so it sets itself an unusually high bar exactly when
    #: it finally bottoms.
    #:
    #: UNH 2025-08-01 is the case. It bottomed at 237.77 and ran to 369.92 by
    #: October, +55%. The detection head read 0.245, nearly 4x the panel mean of
    #: 0.063, and both gates refused it: 166 of 498 names printed a true low
    #: that day so the cross-sectional rank was 0.535, and UNH's own trailing
    #: 98th percentile had been inflated to 0.597 by the April-July crash, so
    #: the self rank was 0.896. The model saw the bottom; the two ranks it is
    #: read through both happened to be blind to it.
    #:
    #: Coverage of confirmed lows by how synchronised the day was, before this
    #: path: quiet days 53.9%, 5-10% days 35.8%, 10-20% days 26.6%, washouts
    #: 30.5%. The gap is not a threshold being tight, it is both rank measures
    #: degrading at once.
    #:
    #: Expressed as a multiple of the run's own panel mean rather than a raw
    #: probability, so it self-calibrates when the model is retrained and stays
    #: reproducible across machines — the property the percentile gates were
    #: chosen for. Measured on the bars the percentile gates refuse:
    #:
    #: =========  ===========  ==========  ======
    #: multiple   extra bars   precision   lift
    #: =========  ===========  ==========  ======
    #: 3x            26,012      27.3%     4.47x
    #: 4x            13,830      32.3%     5.29x
    #: **5x**         6,600      37.1%     6.08x
    #: 6x             2,636      40.4%     6.61x
    #: =========  ===========  ==========  ======
    #:
    #: For reference the percentile gates themselves run at 39.1% and 6.40x, so
    #: from 5x upward this path is of comparable quality rather than a
    #: relaxation. That is the stopping rule: **5x is the loosest setting whose
    #: admitted bars still match the population already being let through**, and
    #: it is chosen on that basis rather than on any individual case.
    #:
    #: End to end it is close to free. Buy precision *rises* (55.7% -> 55.9%)
    #: while buy recall gains 4.1pp (20.8% -> 24.9%) and sell recall 4.8pp
    #: (19.7% -> 24.5%); recall improves in all four years. The run-up to
    #: drawdown share improves (52.9% -> 53.4%) and market-neutral best point
    #: eases from +0.24% to +0.18%.
    #:
    #: Looser settings keep raising F (0.344 at 5x, 0.367 at 4x, 0.380 at 3.5x)
    #: because recall outruns precision, but from 4x down the admitted bars are
    #: measurably worse than the ones already passing and buy precision starts
    #: to fall. UNH's own bar reads 3.90x and is therefore *not* recovered here;
    #: tuning to 3.5x to capture it would be fitting the threshold to a single
    #: example.
    pivot_abs_mult: float | None = 5.0
    #: Relief applied to ``pivot_gate_q`` in proportion to how broadly the panel
    #: is reading as a turn that day. ``None`` disables it.
    #:
    #: The cross-sectional gate is a zero-sum resource: it admits a fixed 5% of
    #: names no matter how many are genuinely turning. On a synchronised washout
    #: that is exactly wrong, and the self-relative gate does not rescue it,
    #: because a name that has been falling all year has an inflated trailing
    #: distribution of its own. Both ranks fail together precisely when the most
    #: valuable lows print. Measured over confirmed lows, coverage runs backwards
    #: to the size of the advance that follows:
    #:
    #: ==============  =======  ==========
    #: advance          lows     flagged
    #: ==============  =======  ==========
    #: <10%              8,464     35.8%
    #: 10-20%            7,064     28.5%
    #: 20-30%            3,135     26.9%
    #: 30-40%            1,043     27.4%
    #: 40-60%              627     29.3%
    #: >=60%               237     25.7%
    #: ==============  =======  ==========
    #:
    #: The relief is driven by the day's own mean detection probability over the
    #: run's mean, so it uses only the same end-of-day cross-section the
    #: percentile ranks already use and adds no lookahead.
    #:
    #: Chosen by the same stopping rule as ``pivot_abs_mult`` — the loosest
    #: setting whose newly admitted events still match the population already
    #: passing (81.7% precision, 4.44x lift):
    #:
    #: ========  =======  ==========  ======
    #: relief     added    precision   lift
    #: ========  =======  ==========  ======
    #: 0.10          287     83.6%     4.54x
    #: 0.20          677     79.8%     4.33x
    #: **0.30**      870     80.0%     4.34x
    #: 0.50        1,047     79.0%     4.29x
    #: ========  =======  ==========  ======
    #:
    #: Every setting qualifies, so the choice falls to where the gain stops
    #: paying: 0.30 buys 2.6pp of buy recall (24.9% -> 27.5%) and 2.5pp of sell
    #: recall for 0.8pp of buy precision, while 0.50 adds a further 177 events
    #: for 0.2pp. Coverage of the >=40% advances above rises 28.4% -> 31.2%.
    #: Buy recall improves in all four years (2023 27.9 -> 30.4, 2024 23.6 ->
    #: 26.9, 2025 23.8 -> 25.8, 2026 25.8 -> 28.3).
    #:
    #: It does not rescue every washout low. AMD's 2026-03-30 bottom, which ran
    #: +77% in a month, gets no relief at all: the panel's mean reading that day
    #: was 0.939x its own average, *below* normal, even though 24% of names
    #: printed a true low. The model did not register the washout as a washout,
    #: so there is no breadth to relieve against. AMD is also blocked
    #: independently on the trigger side, and its own reading is only 2.31x the
    #: panel mean. That case needs a better model, not a looser gate.
    breadth_relief: float | None = 0.30
    #: Floor on the relieved gate, so a violently broad day cannot open it wide.
    breadth_floor: float = 0.80
    #: Rescue a near-miss setup when the *trigger* bar is itself proof of the
    #: turn. ``None`` disables it; otherwise it is the floor on the
    #: cross-sectional percentile a setup may have and still be admitted.
    #:
    #: This targets a structural asymmetry rather than a threshold. The
    #: detection head answers "is *this bar* a swing low?", so it has to fire on
    #: the low itself — but the evidence that confirms a capitulation low
    #: arrives on the *next* bar, in the reversal. MSFT's 2026-06-25 bottom is
    #: the clean example: a confirmed pivot (``pivot = -1``) reading 0.913
    #: against a 0.95 gate, where the following session ran +5.7% on 3.7x
    #: average volume, a 3.9-sigma volume event, closing at 0.83 of its own
    #: range and taking out the prior high. Every downstream condition was
    #: already satisfied on that bar; only the setup gate had refused.
    #:
    #: So the rescue is deliberately not "lower the gate". It admits a weaker
    #: setup *only* when the trigger bar is a wide-range reversal closing near
    #: its high, which is what a volume climax looks like in price. Volume
    #: itself is not on the prediction panel, and does not need to be.
    #:
    #: Simply lowering ``pivot_gate_q`` to 0.91 also catches MSFT, and costs
    #: 1.6pp of buy precision (55.1% -> 53.5%) and 0.09 of lift for 1.3pp of
    #: recall, while dragging the sell side open as a side effect (5,412 ->
    #: 5,918 events). The rescue buys 0.4pp of recall for 0.2pp of precision and
    #: leaves the sell side untouched, because it only reaches events the
    #: trigger bar has independently confirmed.
    #:
    #: 0.90 is where the marginal population stops resembling the existing one,
    #: which is the same stopping rule used for ``pivot_abs_mult`` and
    #: ``breadth_relief``. The events it newly admits run 54.2% precision
    #: against the existing population's 55.1% — the same kind of event. One
    #: notch lower the quality falls off a cliff (0.85 admits 139 at 48.2%), so
    #: the data picks the threshold rather than a preference for round numbers.
    #: Recall improves in all four years and precision moves at most 0.2pp.
    reversal_rescue_q: float | None = 0.90
    #: How far up its own bar the rescuing trigger must close. At 0.70 the same
    #: gate admits 177 events at 49.2%, so the stricter close is doing real
    #: work: it is the difference between a reversal and a merely wide bar.
    reversal_rescue_close_loc: float = 0.80
    #: A slower rescue for marginal setups that form a small double bottom
    #: instead of a single wide capitulation bar. It is restricted to strong
    #: declines near a fresh 60-session low and needs explicit two-/three-day
    #: recovery before becoming actionable.
    delayed_reversal_rescue_q: float | None = 0.875
    delayed_reversal_rescue_max_bars_since_low: int = 3
    delayed_reversal_rescue_return_2: float = 0.015
    delayed_reversal_rescue_return_3: float = 0.0
    delayed_reversal_rescue_close_loc: float = 0.80
    #: Independent right-side rescue for pivots whose detection head is weak on
    #: the extreme itself but strengthens on the confirming bar. These candidates
    #: start only on otherwise empty bars and still pass through sequencing and
    #: terminal conflict guards.
    #:
    #: Bottom confirmation combines looser Pivot ranks with explicit two-day
    #: follow-through. A bar may waive the cumulative-momentum floor only by
    #: closing at least 1.5% above the prior high. This rejects isolated weak
    #: spikes while preserving decisive V-bottom bars. Against the prior
    #: rank-only rescue, confirmed true-low hits rise 733 -> 1,013 OOS while
    #: rescue precision improves 71.93% -> 72.25%; validation detector recall
    #: rises 38.15% -> 41.32% and detector precision 54.49% -> 55.60%.
    bottom_right_rescue_pivot_q: float | None = 0.35
    bottom_right_rescue_self_q: float = 0.35
    bottom_right_rescue_max_bars_since_low: int = 3
    bottom_right_rescue_min_return_1: float = 0.025
    bottom_right_rescue_min_return_2: float = 0.02
    bottom_right_rescue_breakout_waiver: float = 0.015
    bottom_right_rescue_min_close_loc: float = 0.65
    #: Volume-backed bottom confirmation, enabled in the shared Pivot pipeline.
    #: Strong closes, lower-wick rejection, and doji can bypass the old gates.
    bottom_volume_rebound: bool = True
    bottom_volume_rebound_window: int = 20
    bottom_volume_rebound_min_volume_ratio: float = 3.0
    #: Only lower-wick or doji confirmation may waive this ordinary-rebound floor.
    bottom_volume_rebound_min_return_1: float = 0.025
    bottom_volume_rebound_min_close_loc: float = 0.75
    bottom_volume_rebound_min_lower_wick: float | None = 0.50
    bottom_volume_rebound_lower_wick_body_ratio: float = 2.0
    bottom_volume_rebound_max_doji_body: float | None = 0.10
    bottom_volume_rebound_doji_min_lower_wick: float = 0.40
    bottom_volume_rebound_max_prior_return: float = -0.08
    bottom_volume_rebound_max_bars_since_low: int = 3
    bottom_volume_rebound_max_position: float = 0.35
    #: Top detection is deliberately stricter because true highs do not imply a
    #: profitable short. Both ranks must clear 85%, price must break down at
    #: least 5% within two bars of a local high, and the close must sit in the
    #: bottom 20% of its range. The detected cohort has 79.4% research and 77.0%
    #: validation pivot precision, so every sequenced rescue is confirmed as a
    #: ``SELL_TURN``. This recovers ARM 2026-06-18 without loosening the broad
    #: top detector.
    top_right_rescue_pivot_q: float | None = 0.85
    top_right_rescue_self_q: float = 0.85
    top_right_rescue_max_bars_since_high: int = 2
    top_right_rescue_max_return_1: float = -0.05
    top_right_rescue_max_close_loc: float = 0.20
    #: Region candidates recover reversals that the ordinary setup funnel
    #: rejects. They run in independent sequences on otherwise empty bars, then
    #: pass through the same confirmation proofs as existing detections:
    #: confirmed candidates become ``BUY_TURN``/``SELL_TURN`` and the remainder
    #: stay ``BOTTOM_TURN``/``TOP_TURN``.
    bottom_reference_self_q: float | None = 0.95
    bottom_reference_max_bars_since_low: int = 3
    bottom_reference_return_2: float = 0.015
    top_reference_pivot_q: float | None = 0.95
    top_reference_max_bars_since_high: int = 3
    top_reference_return_1: float = -0.01
    top_reference_close_loc: float = 0.25
    #: How much of the detection probability's trend tilt to neutralise, from
    #: 0 (raw) to 1 (fully trend-neutral).
    #:
    #: Real lows genuinely cluster after declines — 9.6% of bars in the weakest
    #: trend quintile versus 3.2% in the strongest — so *some* tilt is correct
    #: and forcing it to zero destroys information (AUC 0.667 -> 0.625). But the
    #: raw model tilts about **three times harder than the ground truth**
    #: (rank correlation with trailing return -0.349 versus -0.114), which is
    #: what produces blanket buys through a decline and blanket sells through a
    #: rally. Blending the global rank with a within-trend-bucket rank at 0.5
    #: brings the tilt to 1.7x truth while costing almost nothing in accuracy
    #: (AUC 0.652, lift 2.18x vs 2.25x), and makes the selection rate across
    #: trend quintiles track the true pivot rate closely.
    trend_neutral_weight: float = 0.5
    #: Trailing window and bucket count used to define the trend context.
    trend_window: int = 20
    trend_buckets: int = 5
    #: Cap on how far price may already have retraced when the trigger bar is
    #: itself violent. Without a cap a single large up-day three sessions after
    #: a low is read as a reversal even when price has broken to new highs —
    #: which is exactly how AMG's 2026-02-12 top was mislabelled a buy.
    #: Upper bound on the violent-reversal waiver. A sharp turn collapses range
    #: position in one session, so demanding price still sit near the extreme
    #: rejects the turns worth catching; but a violent bar that has carried price
    #: all the way across its range is a breakout, not a reversal off a low, and
    #: without any bound AMG's +6.77% breakout day in February 2026 registered as
    #: a bottom. ``0.85`` keeps the breakout guard while admitting gap-style
    #: V-bottoms, and improves every headline number: buy lift 3.12x -> 3.16x,
    #: sell lift 2.82x -> 2.86x, buy recall 15.51% -> 15.87%, sell recall 15.18%
    #: -> 15.64%, trend-neutral spread +0.141% -> +0.152%.
    waiver_pct: float = 0.85
    #: Measure the violent-bar waiver on the bar's **full range** (high to low)
    #: rather than close-to-close, and let it waive the *setup* range gate as
    #: well as ``trigger_pct``.
    #:
    #: This is the wide reversal bar: price trades down into new low ground and
    #: closes back near the top of its own range. Close-to-close such a bar can
    #: be almost flat, so the shipped waiver never sees it, and because the
    #: recovery is what carries the close out of the trough zone, ``trough_pct``
    #: rejects it too. The turn confirms itself out of eligibility.
    #:
    #: TSLA 2025-10-23 is the canonical case: opened 420.00, traded to 413.90,
    #: closed 448.98. A confirmed swing low, detection head in the 98.8th
    #: percentile of the day's cross-section and the 99.9th of the symbol's own
    #: history, closing above the prior bar's high. It fired nothing: the 20-day
    #: range position of the close was 0.633 against a ``trough_pct`` of 0.35,
    #: while the position of the bar's *low* was 0.041. Close-to-close it moved
    #: 0.53 ATR; its full range was 1.89 ATR.
    #:
    #: **On its own the candidate population looks bad, and that reading is
    #: misleading.** Scored in isolation the 631 admitted bars are 60.7% true
    #: lows against the shipped funnel's 8.3%, yet their market-neutral best
    #: point is -0.28% against +0.26%. Run through the actual pipeline — where
    #: ranking, the refractory period and the ATR re-arm decide which of them
    #: survives — the dilution nearly vanishes: market-neutral best point
    #: +0.26% -> +0.26%, run-up to drawdown 53.0% -> 52.5%, day-10 excess
    #: +0.180% -> +0.133%. Candidate quality is not event quality, and the
    #: sequencing rules were already doing the discriminating.
    #:
    #: Measured on the funnel, with ``context_buy_bound`` at 0.10:
    #:
    #: ====================  =======  =========  ========  =======
    #: variant               buy rec  buy prec   sell rec  s prec
    #: ====================  =======  =========  ========  =======
    #: shipped                 18.3%     56.6%      17.9%    51.8%
    #: bar-range waiver      **19.8%**   56.2%    **19.4%**  51.6%
    #: ====================  =======  =========  ========  =======
    #:
    #: Recall rises on both sides for 0.4pp of precision, and the gain holds in
    #: every year (buy recall 21.3->22.6, 16.7->17.8, 18.3->20.3, 17.2->18.5).
    #: The threshold is reused from ``reversal_waiver_atr``, so no new constant
    #: is introduced.
    #:
    #: This is the right trade **because the product's goal is locating the
    #: turn**: the events are a timing reference and the exit belongs to the
    #: strategy consuming them. Set it back to ``False`` if the events are ever
    #: consumed directly as entries at the close, where ``trough_pct`` acting as
    #: an economic filter is worth more than the extra recall.
    waiver_uses_bar_range: bool = True
