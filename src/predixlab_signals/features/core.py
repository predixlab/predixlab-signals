"""Scale-free feature engineering for swing-point detection.

Design rule: **every** feature must be dimensionless. Ratios, percentile
positions, z-scores and bounded oscillators only — never a raw price, never a
raw share count. That single constraint is what lets one small model serve the
whole universe without per-symbol normalisation, cross-sectional ranking, or
retraining when a stock splits or drifts an order of magnitude in price.

The set is deliberately compact (~70 columns). It covers the six things that
actually decide whether a bar is a local extreme:

1. where price sits inside its recent range (Donchian / percentile position),
2. how stretched it is versus its own trend (moving-average ratios, z-scores),
3. how exhausted momentum is (bounded oscillators),
4. what the volatility regime is doing (ATR ratios, vol-of-vol),
5. whether volume confirms or diverges, and
6. how the bar behaves relative to the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------
# small indicator helpers (vectorised, single-symbol series in / series out)
# --------------------------------------------------------------------------
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _atr(high, low, close, window)
    alpha = 1.0 / window
    plus_di = 100.0 * _safe_div(
        pd.Series(plus_dm, index=high.index).ewm(alpha=alpha, min_periods=window, adjust=False).mean(),
        atr,
    )
    minus_di = 100.0 * _safe_div(
        pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, min_periods=window, adjust=False).mean(),
        atr,
    )
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    adx = dx.ewm(alpha=alpha, min_periods=window, adjust=False).mean()
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


def _range_position(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Donchian position: 0.0 at the window low, 1.0 at the window high."""
    hi = high.rolling(window, min_periods=window).max()
    lo = low.rolling(window, min_periods=window).min()
    return _safe_div(close - lo, hi - lo)


def _streak(sign: pd.Series) -> pd.Series:
    """Signed run length of consecutive same-direction closes."""
    values = sign.to_numpy(dtype="float64")
    out = np.zeros(values.size, dtype="float64")
    run = 0.0
    prev = 0.0
    for i, value in enumerate(values):
        if np.isnan(value) or value == 0.0:
            run = 0.0
        elif value == prev:
            run += value
        else:
            run = value
        out[i] = run
        prev = value if not np.isnan(value) else prev
    return pd.Series(out, index=sign.index)


def _bars_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    """Bars elapsed since the rolling window's max (or min), scaled to [0, 1]."""
    arg = series.rolling(window, min_periods=window).apply(
        (lambda x: float(np.argmax(x))) if mode == "max" else (lambda x: float(np.argmin(x))),
        raw=True,
    )
    return (window - 1 - arg) / float(window - 1)


# --------------------------------------------------------------------------
# per-symbol feature block
# --------------------------------------------------------------------------
def compute_symbol_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature block for a single symbol's OHLCV frame.

    ``frame`` must be sorted ascending by date and carry ``open/high/low/close/
    volume``. The returned frame shares ``frame``'s index.
    """
    o, h, low, c = frame["open"], frame["high"], frame["low"], frame["close"]
    v = frame["volume"].astype("float64")
    out: dict[str, pd.Series] = {}

    log_close = np.log(c)
    ret1 = log_close.diff()

    # --- 1. momentum over several look-backs, plus vol-scaled momentum -----
    for w in (1, 3, 5, 10, 20, 60, 120):
        out[f"ret_{w}"] = log_close.diff(w)
    vol20 = ret1.rolling(20, min_periods=20).std()
    out["ret_5_z"] = _safe_div(out["ret_5"], vol20 * np.sqrt(5))
    out["ret_20_z"] = _safe_div(out["ret_20"], vol20 * np.sqrt(20))
    # Momentum acceleration: is the recent push stronger than the older one?
    out["mom_accel"] = out["ret_5"] - out["ret_20"] / 4.0

    # --- 2. trend structure: price vs its own moving averages -------------
    smas = {w: c.rolling(w, min_periods=w).mean() for w in (5, 10, 20, 60, 120)}
    for w, sma in smas.items():
        out[f"px_sma{w}"] = _safe_div(c, sma) - 1.0
    out["sma5_sma20"] = _safe_div(smas[5], smas[20]) - 1.0
    out["sma20_sma60"] = _safe_div(smas[20], smas[60]) - 1.0
    out["sma60_sma120"] = _safe_div(smas[60], smas[120]) - 1.0
    # Slope of the 20d mean, expressed in daily-return units.
    out["sma20_slope"] = _safe_div(smas[20], smas[20].shift(5)) - 1.0

    ema12 = c.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = c.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    out["macd"] = _safe_div(macd, c)
    out["macd_hist"] = _safe_div(macd - signal, c)
    out["macd_hist_chg"] = out["macd_hist"].diff(3)

    # --- 3. range position — the single most direct high/low signal -------
    for w in (10, 20, 60, 120):
        out[f"dpos_{w}"] = _range_position(c, h, low, w)
        out[f"to_high_{w}"] = _safe_div(c, h.rolling(w, min_periods=w).max()) - 1.0
        out[f"to_low_{w}"] = _safe_div(c, low.rolling(w, min_periods=w).min()) - 1.0
    out["dpos_20_chg"] = out["dpos_20"].diff(5)
    out["pctile_60"] = c.rolling(60, min_periods=60).rank(pct=True)
    out["pctile_250"] = c.rolling(250, min_periods=120).rank(pct=True)
    out["bars_since_high_60"] = _bars_since_extreme(h, 60, "max")
    out["bars_since_low_60"] = _bars_since_extreme(low, 60, "min")

    # --- 4. bounded oscillators: momentum exhaustion ----------------------
    out["rsi_6"] = _rsi(c, 6) / 100.0
    out["rsi_14"] = _rsi(c, 14) / 100.0
    out["rsi_14_chg"] = out["rsi_14"].diff(5)
    stoch_k = _range_position(c, h, low, 14)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3, min_periods=3).mean()
    out["williams_r"] = stoch_k - 1.0
    typical = (h + low + c) / 3.0
    tp_mean = typical.rolling(20, min_periods=20).mean()
    tp_mad = typical.rolling(20, min_periods=20).apply(
        lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True
    )
    out["cci_20"] = _safe_div(typical - tp_mean, 0.015 * tp_mad) / 100.0

    std20 = c.rolling(20, min_periods=20).std()
    out["bb_pctb"] = _safe_div(c - (smas[20] - 2 * std20), 4 * std20)
    out["bb_width"] = _safe_div(4 * std20, smas[20])
    out["z_20"] = _safe_div(c - smas[20], std20)
    out["z_60"] = _safe_div(c - smas[60], c.rolling(60, min_periods=60).std())

    dmi = _adx(h, low, c, 14)
    out["adx"] = dmi["adx"] / 100.0
    out["di_spread"] = (dmi["plus_di"] - dmi["minus_di"]) / 100.0

    # --- 5. volatility regime --------------------------------------------
    atr14 = _atr(h, low, c, 14)
    atr_pct = _safe_div(atr14, c)
    out["atr_pct"] = atr_pct
    out["atr_ratio"] = _safe_div(atr_pct, atr_pct.rolling(60, min_periods=60).mean())
    vol5 = ret1.rolling(5, min_periods=5).std()
    vol60 = ret1.rolling(60, min_periods=60).std()
    out["vol_5"] = vol5
    out["vol_20"] = vol20
    out["vol_ratio_5_60"] = _safe_div(vol5, vol60)
    out["vol_ratio_20_60"] = _safe_div(vol20, vol60)
    out["vol_of_vol"] = _safe_div(vol20.rolling(60, min_periods=60).std(), vol20)
    # Parkinson estimator — uses the bar range, so it reacts faster than close-to-close.
    park = np.sqrt((np.log(_safe_div(h, low)) ** 2).rolling(20, min_periods=20).mean() / (4 * np.log(2)))
    out["parkinson_ratio"] = _safe_div(park, vol20)
    out["range_atr"] = _safe_div(h - low, atr14)
    out["downside_vol_20"] = ret1.clip(upper=0.0).rolling(20, min_periods=20).std()
    out["skew_60"] = ret1.rolling(60, min_periods=60).skew()
    out["kurt_60"] = ret1.rolling(60, min_periods=60).kurt()

    # --- 6. volume and money flow ----------------------------------------
    vol_ma20 = v.rolling(20, min_periods=20).mean()
    out["vol_rel"] = np.log(_safe_div(v, vol_ma20).clip(lower=EPS))
    out["vol_rel_5"] = np.log(
        _safe_div(v.rolling(5, min_periods=5).mean(), vol_ma20).clip(lower=EPS)
    )
    out["vol_z"] = _safe_div(v - vol_ma20, v.rolling(20, min_periods=20).std())
    out["vol_trend"] = _safe_div(vol_ma20, v.rolling(60, min_periods=60).mean()) - 1.0

    obv = (np.sign(ret1).fillna(0.0) * v).cumsum()
    out["obv_slope"] = _safe_div(obv - obv.shift(20), vol_ma20 * 20.0)
    # Chaikin money flow: where in the bar the close landed, volume weighted.
    mf_mult = _safe_div((c - low) - (h - c), h - low)
    out["cmf_20"] = _safe_div(
        (mf_mult * v).rolling(20, min_periods=20).sum(), v.rolling(20, min_periods=20).sum()
    )
    raw_mf = typical * v
    pos_mf = raw_mf.where(typical.diff() > 0, 0.0).rolling(14, min_periods=14).sum()
    neg_mf = raw_mf.where(typical.diff() < 0, 0.0).rolling(14, min_periods=14).sum()
    out["mfi_14"] = 1.0 - 1.0 / (1.0 + _safe_div(pos_mf, neg_mf))
    out["pv_corr_20"] = ret1.rolling(20, min_periods=20).corr(np.log(v.clip(lower=EPS)).diff())
    # Amihud illiquidity, log-scaled: price impact per unit of turnover.
    # ``v * c`` is genuine dollar volume — the qlib rebasing factor cancels,
    # since stored volume is divided by it and stored close multiplied by it.
    # A fixed epsilon floor must NOT be used here: dollar volume for a mega-cap
    # runs to 1e10, driving the ratio below any plausible epsilon and pinning
    # the feature to a constant. That silently flattened it on 32% of AAPL's
    # bars, 25% of NVDA's and 23% of TSLA's, and broke the self-normalised
    # variant, whose z-score should be immune to the units entirely.
    impact = _safe_div(ret1.abs(), v * c).rolling(20, min_periods=20).mean()
    out["amihud"] = np.log(impact.where(impact > 0))

    # --- 7. candle geometry ----------------------------------------------
    bar_range = (h - low).replace(0.0, np.nan)
    out["upper_shadow"] = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), bar_range)
    out["lower_shadow"] = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - low, bar_range)
    out["body_ratio"] = _safe_div(c - o, bar_range)
    out["close_loc"] = _safe_div(c - low, bar_range)
    out["gap_atr"] = _safe_div(o - c.shift(1), atr14)
    out["streak"] = _streak(np.sign(ret1)) / 5.0
    up = (ret1 > 0).astype("float64")
    out["up_frac_10"] = up.rolling(10, min_periods=10).mean()
    out["up_frac_20"] = up.rolling(20, min_periods=20).mean()

    # --- 8. drawdown / rally state ---------------------------------------
    out["dd_from_high_250"] = _safe_div(c, c.rolling(250, min_periods=120).max()) - 1.0
    out["rally_from_low_250"] = _safe_div(c, c.rolling(250, min_periods=120).min()) - 1.0
    new_high20 = (c >= c.rolling(20, min_periods=20).max()).astype("float64")
    new_low20 = (c <= c.rolling(20, min_periods=20).min()).astype("float64")
    out["new_high_frac_10"] = new_high20.rolling(10, min_periods=10).mean()
    out["new_low_frac_10"] = new_low20.rolling(10, min_periods=10).mean()

    features = pd.DataFrame(out, index=frame.index)
    features = features.join(_self_normalised(features))
    return features


#: Features whose *level* differs systematically between symbols. A quiet
#: utility and a volatile chip stock never share an ``atr_pct`` scale, so a tree
#: split on the raw value encodes "which stock is this", not "what is happening".
#: Re-expressing them as a trailing z-score against the symbol's own two-year
#: history turns each one into a time-series statement — "unusually volatile
#: *for this name*" — which is what actually predicts a turning point.
SELF_NORM_FEATURES = (
    "atr_pct",
    "vol_5",
    "vol_20",
    "downside_vol_20",
    "bb_width",
    "amihud",
    "beta_60",
    "corr_60",
    "adx",
    "rsi_14",
    "dpos_60",
    "dpos_120",
    "to_high_60",
    "to_high_120",
    "to_low_60",
    "dd_from_high_250",
    "px_sma20",
    "px_sma60",
    "px_sma120",
    "ret_20",
    "ret_60",
    "ret_120",
    "macd",
    "z_60",
    "obv_slope",
    "cmf_20",
    "mfi_14",
    "vol_trend",
    "range_atr",
    "pv_corr_20",
    "skew_60",
)

SELF_NORM_WINDOW = 252
SELF_NORM_MIN = 120


def _self_normalised(features: pd.DataFrame, window: int = SELF_NORM_WINDOW) -> pd.DataFrame:
    """Trailing z-score of each level feature against the symbol's own history.

    Strictly backward-looking: the window ends at the current bar, so no future
    information enters. Values are clipped because a single regime break would
    otherwise produce z-scores large enough to dominate a split.
    """
    cols = [c for c in SELF_NORM_FEATURES if c in features.columns]
    if not cols:
        return pd.DataFrame(index=features.index)
    block = features[cols]
    roll = block.rolling(window, min_periods=SELF_NORM_MIN)
    z = (block - roll.mean()) / roll.std().replace(0.0, np.nan)
    return z.clip(-6.0, 6.0).add_suffix("_sn")


# --------------------------------------------------------------------------
# market-relative block
# --------------------------------------------------------------------------
def compute_market_features(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Features describing the market regime, shared by every symbol."""
    bench = benchmark.set_index("date").sort_index()
    c, h, low = bench["close"], bench["high"], bench["low"]
    log_close = np.log(c)
    ret1 = log_close.diff()

    out = pd.DataFrame(index=bench.index)
    out["mkt_ret_1"] = ret1
    out["mkt_ret_5"] = log_close.diff(5)
    out["mkt_ret_20"] = log_close.diff(20)
    out["mkt_rsi_14"] = _rsi(c, 14) / 100.0
    out["mkt_dpos_60"] = _range_position(c, h, low, 60)
    out["mkt_vol_20"] = ret1.rolling(20, min_periods=20).std()
    out["mkt_vol_ratio"] = _safe_div(
        ret1.rolling(5, min_periods=5).std(), ret1.rolling(60, min_periods=60).std()
    )
    sma50 = c.rolling(50, min_periods=50).mean()
    sma200 = c.rolling(200, min_periods=200).mean()
    out["mkt_px_sma50"] = _safe_div(c, sma50) - 1.0
    out["mkt_regime"] = _safe_div(sma50, sma200) - 1.0
    out["mkt_dd_250"] = _safe_div(c, c.rolling(250, min_periods=120).max()) - 1.0
    return out


def compute_relative_features(
    close: pd.Series, market_ret1: pd.Series, windows: tuple[int, ...] = (20, 60)
) -> pd.DataFrame:
    """Beta, correlation and residual strength of a symbol versus the market."""
    ret1 = np.log(close).diff()
    aligned_mkt = market_ret1.reindex(close.index)
    out = pd.DataFrame(index=close.index)
    for w in windows:
        cov = ret1.rolling(w, min_periods=w).cov(aligned_mkt)
        var = aligned_mkt.rolling(w, min_periods=w).var()
        beta = _safe_div(cov, var)
        out[f"beta_{w}"] = beta.clip(-5.0, 5.0)
        out[f"corr_{w}"] = ret1.rolling(w, min_periods=w).corr(aligned_mkt)
        cum_sym = ret1.rolling(w, min_periods=w).sum()
        cum_mkt = aligned_mkt.rolling(w, min_periods=w).sum()
        out[f"rs_{w}"] = cum_sym - cum_mkt
        out[f"alpha_{w}"] = cum_sym - beta.clip(-5.0, 5.0) * cum_mkt
    return out


def build_features(panel: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the full feature matrix for a long-format OHLCV panel.

    Returns the panel's identity columns (``symbol``, ``date``, ``close``,
    ``price``, ``atr_pct``-style helpers stay inside the feature block) joined
    with every engineered column.
    """
    market = compute_market_features(benchmark) if benchmark is not None and not benchmark.empty else None
    market_ret1 = market["mkt_ret_1"] if market is not None else None

    blocks: list[pd.DataFrame] = []
    for symbol, group in panel.groupby("symbol", sort=True):
        frame = group.set_index("date").sort_index()
        feats = compute_symbol_features(frame)
        if market_ret1 is not None:
            feats = feats.join(compute_relative_features(frame["close"], market_ret1))
        feats.insert(0, "symbol", symbol)
        feats["close"] = frame["close"]
        feats["price"] = frame["price"]
        blocks.append(feats.reset_index())

    if not blocks:
        return pd.DataFrame()

    result = pd.concat(blocks, ignore_index=True)
    if market is not None:
        result = result.merge(market.reset_index(), on="date", how="left")
    return result.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


#: Columns that must never be handed to a model. Some are identity (``symbol``),
#: some are raw price levels that would break scale-invariance (``high``,
#: ``low``), and some look into the future (``open_next``, ``pivot``) — a single
#: one of those slipping into the matrix would silently invalidate every metric.
NON_FEATURE_COLUMNS = {
    "symbol",
    "date",
    "close",
    "price",
    "open",
    "high",
    "low",
    "volume",
    "raw_volume",
    "factor",
    "open_next",
    "pivot",
    "label_scale",
    "atr_pct_label",
    "score",
    "action",
    "fold",
}

#: Prefix marking market-wide columns. They are identical for every symbol on a
#: given day, so ten years of data contain only ~2,500 independent observations
#: of them - yet they appear on all 1.7M rows. A gradient-booster reads that as
#: overwhelming evidence and builds a market-timing model that tests near zero.
#: They are therefore excluded from the model by default and kept only as
#: reporting context.
MARKET_PREFIX = "mkt_"


def feature_columns(frame: pd.DataFrame, include_market: bool = False) -> list[str]:
    """Feature column names, i.e. everything that is not an identity column."""
    return [
        c
        for c in frame.columns
        if c not in NON_FEATURE_COLUMNS
        and not c.startswith(("y_", "label_", "fwd_", "exp_", "p_up"))
        and not c.endswith("_hat")
        and (include_market or not c.startswith(MARKET_PREFIX))
    ]
