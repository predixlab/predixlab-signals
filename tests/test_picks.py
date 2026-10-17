"""Tests for event selection and the historical backfill.

Selection is the boundary between a model artifact and a product, so the rule is
pinned here: a pick is a dated *event*, not a cross-sectional ranking, and the
event — not the state label it happened to carry — decides the side.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from predixlab_signals.jobs.backfill import (
    attach_forward_returns,
    attach_pivot_hits,
    dedupe_folds,
)
from predixlab_signals.publish.mongo import (
    PICK_FIELDS,
    AdviceStore,
    _clean,
    pick_counts,
    pick_documents,
)
from predixlab_signals.signals.picks import (
    DETECTION,
    LONG,
    REVERSAL,
    SHORT,
    TREND,
    extract_events,
    format_picks,
    rank_events,
    select_picks,
    summarise_picks,
)
from predixlab_signals.signals.turning_points import (
    BASE_BREAKOUT,
    BOTTOM_TURN,
    BUY_TURN,
    NO_TURN,
    SELL_TURN,
    TOP_TURN,
    TREND_BUY,
    TurningPointConfig,
)
from predixlab_signals.signals.models.pivot.bottom_turns import apply_bottom_volume_rebounds

TEST_CONFIG = TurningPointConfig(
    max_uptrend_slope=None,
    reversal_waiver_atr=1.5,
)


def _panel(n_symbols: int = 6, n_days: int = 400) -> pd.DataFrame:
    """A scored panel with enough history for the sequential rules to warm up."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames = []
    for i in range(n_symbols):
        # A slow cycle guarantees genuine tops and bottoms to detect.
        wave = np.sin(np.linspace(0, 6 * np.pi, n_days) + i)
        close = 100.0 * np.exp(0.15 * wave + np.cumsum(rng.normal(0, 0.004, n_days)))
        score = 0.2 * -wave + rng.normal(0, 0.02, n_days)
        action = np.where(score > 0.1, "STRONG_BUY", np.where(score < -0.1, "STRONG_SELL", "HOLD"))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": f"SYM{i:02d}",
                    "date": dates,
                    "open": close,
                    "close": close,
                    "price": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": np.full(n_days, 1_000_000.0),
                    "swing_pos_hat": 0.5 - 0.4 * np.sign(score),
                    "p_up": 0.5 + 0.1 * np.sign(score),
                    "exp_gain_pct": np.abs(rng.normal(0.05, 0.01, n_days)),
                    "exp_loss_pct": -np.abs(rng.normal(0.04, 0.01, n_days)),
                    "exp_amplitude_pct": np.abs(rng.normal(0.09, 0.01, n_days)),
                    "score": score,
                    "action": action,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _volume_bottom_panel(candle=(90.0, 94.0, 89.5, 93.0)) -> pd.DataFrame:
    close = np.r_[np.linspace(140.0, 110.0, 60), 90.0, 93.0]
    panel = _panel(1, len(close)).assign(
        open=close, high=close + 1.0, low=close - 1.0,
        close=close, price=close, action="HOLD", score=0.1,
    )
    panel.loc[panel.index[-2], ["open", "high", "low"]] = [95.0, 100.0, 89.0]
    panel.loc[panel.index[-1], ["open", "high", "low", "close"]] = candle
    panel.loc[panel.index[-1], ["price", "volume"]] = [candle[-1], 3_000_000.0]
    return panel


class TestVolumeBottomFilter:
    @pytest.mark.parametrize(
        ("candle", "expected"),
        [
            ((90.0, 94.0, 89.5, 93.0), True),
            ((90.8, 92.0, 85.0, 89.8), True),
            ((88.0, 92.5, 84.5, 88.0), True),
            ((89.1, 91.0, 89.0, 90.6), False),
            ((88.4, 90.2, 88.2, 89.8), False),
            ((88.6, 90.5, 88.5, 90.0), False),
            ((85.0, 92.5, 84.5, 85.1), False),
            ((90.0, 99.0, 89.0, 93.0), False),
            ((93.0, 93.0, 93.0, 93.0), False),
        ],
    )
    def test_default_confirms_only_strong_rebounds_or_rejection_shapes(self, candle, expected):
        assert TurningPointConfig().bottom_volume_rebound
        panel = _volume_bottom_panel(candle)
        picks = select_picks(panel)
        if expected:
            assert picks["event"].tolist() == [BUY_TURN]
            assert picks["pivot_event"].tolist() == [BUY_TURN]
            assert picks["signal_model"].tolist() == ["pivot"]
            assert picks["actionable"].tolist() == [True]
            assert picks["action"].tolist() == ["HOLD"]
            assert picks["date"].tolist() == [panel["date"].iloc[-1]]
        else:
            assert picks.empty

    def test_explicit_opt_out_preserves_original_pivot_behavior(self):
        assert select_picks(
            _volume_bottom_panel(), TurningPointConfig(bottom_volume_rebound=False)
        ).empty

    @pytest.mark.parametrize("gate", ["volume", "decline", "fresh_low", "position"])
    def test_rejection_does_not_bypass_volume_or_bottom_context(self, gate):
        panel = _volume_bottom_panel()
        if gate == "volume":
            panel.loc[panel.index[-1], "volume"] = 2_990_000.0
        elif gate == "decline":
            panel.loc[panel.index[-22], ["open", "high", "low", "close"]] = [
                90.0, 91.0, 89.0, 90.0
            ]
        elif gate == "fresh_low":
            panel.loc[panel.index[-6], "low"] = 80.0
        else:
            panel.loc[panel.index[-20:-2], ["open", "high", "low", "close"]] = [
                98.0, 99.0, 97.0, 98.0
            ]
        assert select_picks(panel).empty

    @pytest.mark.parametrize("close,expected", [(92.25, True), (92.24, False)])
    @pytest.mark.parametrize("scale", [1.0, 0.041296902])
    def test_ordinary_gain_boundary_is_price_scale_invariant(self, close, expected, scale):
        panel = _volume_bottom_panel((89.1, 93.0, 89.0, close))
        panel[["open", "high", "low", "close", "price"]] *= scale
        assert select_picks(panel).empty == (not expected)

    @pytest.mark.parametrize("floor", [None, 0.0, -0.01, np.nan, np.inf])
    def test_ordinary_gain_floor_cannot_be_removed(self, floor):
        with pytest.raises(ValueError, match="positive finite daily-return"):
            select_picks(
                _volume_bottom_panel(),
                TurningPointConfig(bottom_volume_rebound_min_return_1=floor),
            )

    def test_candle_confirmation_requires_current_open_not_future_open(self):
        panel = _volume_bottom_panel().drop(columns="open").assign(open_next=90.0)
        with pytest.raises(ValueError, match="current-session open"):
            select_picks(panel)

    @pytest.mark.parametrize(
        "event", [BUY_TURN, SELL_TURN, TOP_TURN, TREND_BUY, BASE_BREAKOUT, BOTTOM_TURN]
    )
    def test_existing_event_priority_and_bottom_promotion(self, event):
        panel = _volume_bottom_panel()
        existing = np.full(len(panel), NO_TURN, dtype=object)
        existing[-1] = event
        result = apply_bottom_volume_rebounds(panel, existing, TurningPointConfig())
        assert result[-1] == (BUY_TURN if event == BOTTOM_TURN else event)
        assert existing[-1] == event

    def test_prior_cooldown_is_respected_without_using_future_events(self):
        panel = _volume_bottom_panel()
        existing = np.full(len(panel), NO_TURN, dtype=object)
        existing[-3] = SELL_TURN
        cfg = TurningPointConfig()
        assert apply_bottom_volume_rebounds(panel, existing, cfg)[-1] == NO_TURN

        existing[:] = NO_TURN
        prefix = apply_bottom_volume_rebounds(panel, existing, cfg)
        future = panel.iloc[[-1]].copy()
        future["date"] += pd.offsets.BDay(1)
        full = pd.concat([panel, future], ignore_index=True)
        complete = apply_bottom_volume_rebounds(full, np.r_[existing, [SELL_TURN]], cfg)
        assert prefix[-1] == BUY_TURN
        assert complete[-1] == SELL_TURN
        np.testing.assert_array_equal(complete[:-1], prefix)


# ---------------------------------------------------------------------------
# a pick is an event
# ---------------------------------------------------------------------------
class TestExtractEvents:
    def test_events_are_sparse(self):
        """The whole point: far fewer picks than bar-days."""
        panel = _panel()
        events = extract_events(panel, TEST_CONFIG)
        assert not events.empty
        assert len(events) < 0.15 * len(panel)

    def test_only_fired_bars_survive(self):
        events = extract_events(_panel(), TEST_CONFIG)
        assert "turn_action" not in events.columns
        assert "turn_trade_action" not in events.columns
        assert "pivot_event" in events.columns
        assert "actionable" in events.columns
        assert events["event"].equals(events["pivot_event"])
        assert set(events["event"]) <= {
            BOTTOM_TURN,
            BUY_TURN,
            SELL_TURN,
            TOP_TURN,
            TREND_BUY,
            BASE_BREAKOUT,
        }
        assert NO_TURN not in set(events["event"])

    def test_direction_and_kind_follow_the_event(self):
        events = extract_events(_panel(), TEST_CONFIG)
        mapping = events.groupby("event")[["direction", "kind"]].agg(lambda s: set(s))
        for event, row in mapping.iterrows():
            if event == BUY_TURN:
                assert row["direction"] == {LONG} and row["kind"] == {REVERSAL}
            elif event == BOTTOM_TURN:
                assert row["direction"] == {LONG} and row["kind"] == {DETECTION}
            elif event == SELL_TURN:
                assert row["direction"] == {SHORT} and row["kind"] == {REVERSAL}
            elif event == TOP_TURN:
                assert row["direction"] == {SHORT} and row["kind"] == {DETECTION}
            elif event == TREND_BUY:
                assert row["direction"] == {LONG} and row["kind"] == {TREND}
            elif event == BASE_BREAKOUT:
                assert row["direction"] == {LONG} and row["kind"] == {TREND}

    def test_trend_entries_are_never_merged_into_reversals(self):
        """They answer different questions and must stay distinguishable."""
        events = extract_events(_panel(), TEST_CONFIG)
        if TREND_BUY in set(events["event"]):
            trend = events[events["event"] == TREND_BUY]
            assert (trend["kind"] == TREND).all()
            assert (events[events["kind"] == REVERSAL]["event"] != TREND_BUY).all()

    def test_missing_bars_raise_rather_than_emit_nothing(self):
        """Without OHLC no rule can fire; silence would look like 'a quiet day'."""
        panel = _panel().drop(columns=["high", "low"])
        with pytest.raises(ValueError, match="high"):
            extract_events(panel, TEST_CONFIG)

    def test_missing_volume_raises_rather_than_hide_base_breakouts(self):
        panel = _panel().drop(columns="volume")
        with pytest.raises(ValueError, match="volume"):
            extract_events(panel, TEST_CONFIG)

    def test_empty_panel(self):
        assert extract_events(_panel().iloc[0:0], TEST_CONFIG).empty


class TestRanking:
    def test_rank_is_one_based_per_side_per_day(self):
        picks = select_picks(_panel(), TEST_CONFIG)
        for (_, _), group in picks.groupby(["date", "direction"]):
            assert list(group["rank"]) == list(range(1, len(group) + 1))

    def test_ranking_removes_nothing(self):
        """Unlike a cross-sectional cut, ordering is presentation only."""
        events = extract_events(_panel(), TEST_CONFIG)
        assert len(rank_events(events)) == len(events)

    def test_longs_are_ordered_by_conviction(self):
        picks = select_picks(_panel(), TEST_CONFIG)
        day = picks[picks["direction"] == LONG].groupby("date").head(50)
        for _, group in day.groupby("date"):
            magnitudes = group["score"].abs().tolist()
            assert magnitudes == sorted(magnitudes, reverse=True)

    def test_max_per_side_is_off_by_default(self):
        panel = _panel()
        assert len(select_picks(panel, TEST_CONFIG)) == len(
            extract_events(panel, TEST_CONFIG)
        )

    def test_max_per_side_caps_when_asked(self):
        picks = select_picks(_panel(), TEST_CONFIG, max_per_side=1)
        assert (picks["rank"] <= 1).all()


class TestSummarise:
    def test_counts_each_event_type(self):
        counts = summarise_picks(select_picks(_panel(), TEST_CONFIG))
        assert set(counts) == {
            BOTTOM_TURN,
            BUY_TURN,
            SELL_TURN,
            TOP_TURN,
            TREND_BUY,
            BASE_BREAKOUT,
        }
        assert sum(counts.values()) > 0

    def test_empty_is_all_zero(self):
        assert summarise_picks(pd.DataFrame()) == {
            BOTTOM_TURN: 0,
            BUY_TURN: 0,
            SELL_TURN: 0,
            TOP_TURN: 0,
            TREND_BUY: 0,
            BASE_BREAKOUT: 0,
        }

    def test_format_is_readable(self):
        assert "BUY" in format_picks(select_picks(_panel(), TEST_CONFIG))

    def test_format_says_so_when_nothing_fired(self):
        assert "No turning points" in format_picks(pd.DataFrame())


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------
class TestDedupeFolds:
    def test_boundary_date_keeps_the_later_fold(self):
        frame = pd.DataFrame(
            {
                "symbol": ["A", "A", "B"],
                "date": [pd.Timestamp("2024-04-24")] * 3,
                "score": [0.1, 0.2, 0.3],
                "fold": [1, 2, 1],
            }
        )
        out = dedupe_folds(frame)
        assert len(out) == 2
        assert out[out["symbol"] == "A"]["score"].iloc[0] == pytest.approx(0.2)

    def test_no_fold_column_is_a_passthrough(self):
        frame = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-01")]})
        assert len(dedupe_folds(frame)) == 1


class TestAttachForwardReturns:
    """The outcome measure: what the name did over the horizon after the signal."""

    def _picks(self):
        return pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "date": pd.to_datetime(["2024-01-02"] * 3),
                "direction": [LONG, SHORT, LONG],
            }
        )

    def _predictions(self):
        return pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "date": pd.to_datetime(["2024-01-02"] * 2),
                "fwd_ret": [0.05, 0.04],
                "fwd_ret_cs": [0.03, 0.02],
            }
        )

    def test_long_keeps_the_raw_sign(self):
        out = attach_forward_returns(self._picks(), self._predictions())
        aaa = out[out["symbol"] == "AAA"].iloc[0]
        assert aaa["forward_return"] == pytest.approx(0.05)
        assert aaa["signal_return"] == pytest.approx(0.05)

    def test_short_return_is_negated(self):
        """A short profits from a fall, so a +4% move is a -4% result."""
        out = attach_forward_returns(self._picks(), self._predictions())
        bbb = out[out["symbol"] == "BBB"].iloc[0]
        assert bbb["forward_return"] == pytest.approx(0.04)
        assert bbb["signal_return"] == pytest.approx(-0.04)

    def test_excess_is_negated_for_shorts_too(self):
        out = attach_forward_returns(self._picks(), self._predictions())
        assert out[out["symbol"] == "AAA"].iloc[0]["excess_return"] == pytest.approx(0.03)
        assert out[out["symbol"] == "BBB"].iloc[0]["excess_return"] == pytest.approx(-0.02)

    def test_unresolved_signals_stay_null(self):
        """Recent signals are still open, not missing."""
        out = attach_forward_returns(self._picks(), self._predictions())
        assert pd.isna(out[out["symbol"] == "CCC"].iloc[0]["signal_return"])

    def test_no_predictions_yields_the_columns_anyway(self):
        out = attach_forward_returns(self._picks(), pd.DataFrame())
        for column in ("forward_return", "signal_return"):
            assert column in out.columns
            assert out[column].isna().all()

    def test_duplicate_predictions_do_not_fan_out(self):
        doubled = pd.concat([self._predictions(), self._predictions()], ignore_index=True)
        assert len(attach_forward_returns(self._picks(), doubled)) == 3

    def test_missing_excess_column_is_tolerated(self):
        predictions = self._predictions().drop(columns=["fwd_ret_cs"])
        out = attach_forward_returns(self._picks(), predictions)
        assert out[out["symbol"] == "AAA"].iloc[0]["signal_return"] == pytest.approx(0.05)


class TestAttachPivotHits:
    """Did the reversal actually land on a swing point?"""

    def _picks(self):
        return pd.DataFrame(
            {
                "symbol": ["AAA", "AAA", "BBB", "CCC"],
                "date": pd.to_datetime(
                    ["2024-01-03", "2024-01-10", "2024-01-03", "2024-01-03"]
                ),
                "direction": [LONG, LONG, SHORT, LONG],
                "kind": [REVERSAL, REVERSAL, REVERSAL, TREND],
            }
        )

    def _predictions(self):
        dates = pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-10"]
        )
        frames = []
        for symbol, pivots in (
            # AAA prints a swing low on the 2nd; the pick on the 3rd is one bar away.
            ("AAA", [0.0, -1.0, 0.0, 0.0, 0.0]),
            ("BBB", [0.0, 0.0, 1.0, 0.0, 0.0]),
            ("CCC", [0.0, 0.0, 0.0, 0.0, 0.0]),
        ):
            frames.append(pd.DataFrame({"symbol": symbol, "date": dates, "pivot": pivots}))
        return pd.concat(frames, ignore_index=True)

    def test_a_near_miss_within_tolerance_counts(self):
        """A turn is a region, not an instant."""
        out = attach_pivot_hits(self._picks(), self._predictions(), tolerance=3)
        hit = out[(out["symbol"] == "AAA") & (out["date"] == pd.Timestamp("2024-01-03"))]
        assert bool(hit["pivot_hit"].iloc[0]) is True

    def test_a_distant_signal_misses(self):
        out = attach_pivot_hits(self._picks(), self._predictions(), tolerance=1)
        miss = out[(out["symbol"] == "AAA") & (out["date"] == pd.Timestamp("2024-01-10"))]
        assert bool(miss["pivot_hit"].iloc[0]) is False

    def test_shorts_are_scored_against_swing_highs(self):
        out = attach_pivot_hits(self._picks(), self._predictions(), tolerance=3)
        assert bool(out[out["symbol"] == "BBB"]["pivot_hit"].iloc[0]) is True

    def test_trend_entries_are_left_unscored(self):
        """A breakout is the opposite of a swing low; scoring it that way is a
        category error, and a zero would read as a failure rather than as
        'this metric does not apply'."""
        out = attach_pivot_hits(self._picks(), self._predictions(), tolerance=3)
        assert pd.isna(out[out["kind"] == TREND]["pivot_hit"].iloc[0])

    def test_tolerance_does_not_wrap_around_the_series(self):
        """Dilation must not leak from the end of one symbol to its start."""
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        predictions = pd.DataFrame(
            {"symbol": "AAA", "date": dates, "pivot": [0.0, 0.0, -1.0]}
        )
        picks = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": [pd.Timestamp("2024-01-01")],
                "direction": [LONG],
                "kind": [REVERSAL],
            }
        )
        out = attach_pivot_hits(picks, predictions, tolerance=1)
        assert bool(out["pivot_hit"].iloc[0]) is False

    def test_no_ground_truth_leaves_everything_unscored(self):
        out = attach_pivot_hits(self._picks(), pd.DataFrame())
        assert out["pivot_hit"].isna().all()


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
class TestPickDocuments:
    def _picks(self):
        picks = select_picks(_panel(), TEST_CONFIG).head(6).copy()
        picks["forward_return"] = np.nan
        picks["signal_return"] = np.nan
        picks["excess_return"] = np.nan
        return picks

    def test_key_fields_are_present(self):
        docs = pick_documents(self._picks(), "us", 10, "default")
        for key in ("market", "date", "symbol", "event", "direction", "kind", "rank"):
            assert key in docs[0]

    def test_every_pick_field_is_carried(self):
        docs = pick_documents(self._picks(), "us", 10)
        assert {"event", "kind", "direction", "rank"} <= {f for f in PICK_FIELDS if f in docs[0]}

    def test_rank_is_stored_as_an_integer(self):
        assert all(isinstance(d["rank"], int) for d in pick_documents(self._picks(), "us", 10))

    def test_missing_values_become_none(self):
        docs = pick_documents(self._picks(), "us", 10)
        assert docs[0]["signal_return"] is None
        assert docs[0]["excess_return"] is None

    def test_empty_frame(self):
        assert pick_documents(pd.DataFrame(), "us", 10) == []

    def test_counts_are_by_event_type(self):
        counts = pick_counts(self._picks())
        assert set(counts) <= {
            BOTTOM_TURN,
            BUY_TURN,
            SELL_TURN,
            TOP_TURN,
            TREND_BUY,
        }
        assert pick_counts(pd.DataFrame()) == {}


class _FakePicks:
    """In-memory stand-in for the picks collection, keyed like the real one."""

    def __init__(self):
        self.docs: list[dict] = []

    def bulk_write(self, operations, ordered=True):
        upserted = modified = 0
        for op in operations:
            key = op._filter
            doc = op._doc["$set"]
            existing = next(
                (d for d in self.docs if all(d.get(k) == v for k, v in key.items())), None
            )
            if existing is None:
                self.docs.append(dict(doc))
                upserted += 1
            else:
                existing.update(doc)
                modified += 1
        return type("R", (), {"upserted_count": upserted, "modified_count": modified})()

    def create_index(self, keys, **options):
        return None


class _FakeStore(AdviceStore):
    def __init__(self):
        self.settings = None
        self._owns_client = False
        self._picks = _FakePicks()

    @property
    def picks(self):
        return self._picks

    def _upsert_session(self, documents):
        written = 0
        for document in documents:
            key = ("market", "date", "symbol", "event")
            existing = next(
                (
                    row
                    for row in self._picks.docs
                    if all(row.get(field) == document.get(field) for field in key)
                ),
                None,
            )
            if existing is None:
                self._picks.docs.append(dict(document))
            else:
                existing.update(document)
            written += 1
        return written

class TestUpsertsWithoutDeleting:
    """Publishing only adds or updates documents; it never removes data."""

    def _docs(self, symbols, stamp):
        return [
            {
                "market": "us",
                "date": "2026-04-24",
                "symbol": s,
                "event": BUY_TURN,
                "direction": LONG,
                "rank": i + 1,
                "generated_at": stamp,
            }
            for i, s in enumerate(symbols)
        ]

    def test_events_not_present_in_a_later_write_are_preserved(self):
        store = _FakeStore()
        first = datetime(2026, 4, 25, 2, 0, tzinfo=timezone.utc)
        second = datetime(2026, 4, 25, 3, 0, tzinfo=timezone.utc)

        store.write_picks(self._docs(["AAA", "BBB", "CCC"], first))
        store.write_picks(self._docs(["AAA", "DDD"], second))

        assert sorted(d["symbol"] for d in store.picks.docs) == ["AAA", "BBB", "CCC", "DDD"]

    def test_other_sessions_are_untouched(self):
        store = _FakeStore()
        stamp = datetime(2026, 4, 25, 2, 0, tzinfo=timezone.utc)
        store.write_picks(self._docs(["AAA"], stamp))
        store.write_picks([dict(d, date="2026-04-23") for d in self._docs(["ZZZ"], stamp)])
        store.write_picks(self._docs(["BBB"], datetime(2026, 4, 25, 4, 0, tzinfo=timezone.utc)))

        assert sorted({d["date"] for d in store.picks.docs}) == ["2026-04-23", "2026-04-24"]
        assert any(d["symbol"] == "ZZZ" for d in store.picks.docs)

    def test_empty_write_deletes_nothing(self):
        store = _FakeStore()
        stamp = datetime(2026, 4, 25, 2, 0, tzinfo=timezone.utc)
        store.write_picks(self._docs(["AAA"], stamp))
        store.write_picks([dict(d, date="2026-04-23") for d in self._docs(["ZZZ"], stamp)])

        written = store.write_picks([])

        assert written == 0
        assert sorted(d["symbol"] for d in store.picks.docs) == ["AAA", "ZZZ"]


class TestClean:
    @pytest.mark.parametrize("value", [None, np.nan, pd.NaT, pd.NA, np.inf, -np.inf])
    def test_every_missing_flavour_becomes_none(self, value):
        assert _clean(value) is None

    def test_timestamps_become_date_strings(self):
        assert _clean(pd.Timestamp("2024-01-02 15:30")) == "2024-01-02"

    def test_scalars_pass_through(self):
        assert _clean(np.float64(2.5)) == 2.5
        assert _clean("x") == "x"
        assert _clean(True) is True
