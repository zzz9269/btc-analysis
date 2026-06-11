"""
Pure technical-indicator math, extracted from btc_analysis_app.py
(module split stage 1). No Streamlit, no network, no app state —
just numpy/pandas/scipy transforms. Safe to import anywhere.
"""
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.signal

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_g / avg_l.replace(0, np.nan)   # FIX: avoid div-by-zero → NaN instead of inf
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period * 2:
        return np.nan
    try:
        d = df.copy()
        d["tr"] = np.maximum.reduce([
            d["High"] - d["Low"],
            abs(d["High"] - d["Close"].shift(1)),
            abs(d["Low"]  - d["Close"].shift(1)),
        ])
        d["up"]   = d["High"] - d["High"].shift(1)
        d["down"] = d["Low"].shift(1) - d["Low"]
        d["pdm"]  = np.where((d["up"] > d["down"]) & (d["up"] > 0),   d["up"],   0.0)
        d["mdm"]  = np.where((d["down"] > d["up"]) & (d["down"] > 0), d["down"], 0.0)
        a = 1 / period
        d["tr_s"]  = d["tr"].ewm(alpha=a,  adjust=False).mean()
        d["pdm_s"] = d["pdm"].ewm(alpha=a, adjust=False).mean()
        d["mdm_s"] = d["mdm"].ewm(alpha=a, adjust=False).mean()
        pdi = 100 * d["pdm_s"] / d["tr_s"]
        mdi = 100 * d["mdm_s"] / d["tr_s"]
        dx  = 100 * abs(pdi - mdi) / (pdi + mdi)
        return dx.ewm(alpha=a, adjust=False).mean().iloc[-1]
    except Exception:
        return np.nan


def calculate_obv_trend(df: pd.DataFrame, period: int = 20) -> str:
    try:
        if df is None or len(df) < period + 5:
            return "neutral"
        closes  = df["Close"].tolist()
        volumes = df["Volume"].tolist()
        obv = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:   obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i - 1]: obv.append(obv[-1] - volumes[i])
            else:                            obv.append(obv[-1])
        obv         = np.array(obv)
        obv_trend   = (obv[-1] - obv[-period]) / (abs(obv[-period]) + 1)
        price_trend = (closes[-1] - closes[-period]) / closes[-period]
        if   obv_trend > 0.10 and price_trend < 0.0:    return "strong_accumulation"
        elif obv_trend > 0.05 and price_trend < 0.02:  return "accumulation"
        elif obv_trend < -0.10 and price_trend > 0.0:  return "strong_distribution"
        elif obv_trend < -0.05 and price_trend > -0.02: return "distribution"
        elif obv_trend > 0.03:  return "mild_accumulation"
        elif obv_trend < -0.03: return "mild_distribution"
        return "neutral"
    except Exception:
        return "neutral"


def detect_price_traps(ohlcv: pd.DataFrame, lookback: int = 20) -> tuple:
    """
    Detect bull/bear traps and liquidity sweep candles from OHLCV price action.

    Bull trap  — price breaks above rolling high, closes back below → bearish (longs trapped)
    Bear trap  — price breaks below rolling low, closes back above  → bullish (shorts trapped)
    Sweep up   — long lower wick + close near high (pin bar)         → bullish rejection of low
    Sweep down — long upper wick + close near low  (shooting star)   → bearish rejection of high
    Failed break — price tags a level multiple times without closing through → exhaustion

    Returns (category, score, note) where score ∈ [−1, +1].
    """
    try:
        if ohlcv is None or len(ohlcv) < lookback + 4:
            return "none", 0.0, "Insufficient data for trap detection"

        # Reference window excludes the last 3 bars (those are the "test" bars)
        ref   = ohlcv.iloc[-(lookback + 3):-3]
        last3 = ohlcv.iloc[-3:]
        curr  = ohlcv.iloc[-1]

        ref_high = float(ref["High"].max())
        ref_low  = float(ref["Low"].min())

        c_high  = float(curr["High"])
        c_low   = float(curr["Low"])
        c_close = float(curr["Close"])
        c_open  = float(curr["Open"])
        c_range = c_high - c_low
        c_body  = abs(c_close - c_open)

        # ── 1. Bull Trap: any of last 3 bars pierced ref_high but current close < ref_high
        pierce_high = max((float(r["High"]) for _, r in last3.iterrows()), default=0)
        if pierce_high > ref_high and c_close < ref_high:
            pierce_pct  = (pierce_high - ref_high) / ref_high * 100
            pullback_pct = (ref_high - c_close) / ref_high * 100
            # Stronger trap if the pierce was small (real fakeout) and pull-back is sharp
            strength = float(np.tanh((pierce_pct + pullback_pct * 1.5) / 1.5))
            strength = max(0.25, min(1.0, strength))
            return ("bull_trap", -strength,
                    f"Bull trap: broke ${ref_high:,.0f} (+{pierce_pct:.2f}%), closed back below "
                    f"(trapped longs, -{pullback_pct:.2f}%)")

        # ── 2. Bear Trap: any of last 3 bars pierced ref_low but current close > ref_low
        pierce_low = min((float(r["Low"]) for _, r in last3.iterrows()), default=float("inf"))
        if pierce_low < ref_low and c_close > ref_low:
            pierce_pct   = (ref_low - pierce_low) / ref_low * 100
            recovery_pct = (c_close - ref_low) / ref_low * 100
            strength = float(np.tanh((pierce_pct + recovery_pct * 1.5) / 1.5))
            strength = max(0.25, min(1.0, strength))
            return ("bear_trap", +strength,
                    f"Bear trap: swept ${ref_low:,.0f} (-{pierce_pct:.2f}%), recovered above "
                    f"(trapped shorts, +{recovery_pct:.2f}%)")

        # ── 3. Sweep candle on current bar (pin bars / shooting stars)
        if c_range > 0:
            upper_wick = c_high - max(c_close, c_open)
            lower_wick = min(c_close, c_open) - c_low
            body_ratio = c_body / c_range   # near 0 = doji/pin, near 1 = marubozu

            # Bearish sweep: large upper wick, close in lower half
            if (upper_wick > 0.55 * c_range and lower_wick < 0.20 * c_range
                    and c_close < (c_high + c_low) / 2):
                wick_pct = upper_wick / c_close * 100
                score = -float(np.tanh(wick_pct / 0.8))
                score = max(-0.85, score)
                return ("sweep_down", score,
                        f"Bearish sweep candle — upper wick {wick_pct:.2f}% of price "
                        f"(rejection at high, body ratio {body_ratio:.2f})")

            # Bullish sweep: large lower wick, close in upper half
            if (lower_wick > 0.55 * c_range and upper_wick < 0.20 * c_range
                    and c_close > (c_high + c_low) / 2):
                wick_pct = lower_wick / c_close * 100
                score = +float(np.tanh(wick_pct / 0.8))
                score = min(+0.85, score)
                return ("sweep_up", score,
                        f"Bullish sweep candle — lower wick {wick_pct:.2f}% of price "
                        f"(rejection at low, body ratio {body_ratio:.2f})")

        # ── 4. Failed breakout test (multiple tags without close-through)
        # Price has tagged ref_high 2+ times in last 3 bars without closing above → exhaustion
        tags_high = sum(1 for _, r in last3.iterrows() if float(r["High"]) >= ref_high * 0.998)
        tags_low  = sum(1 for _, r in last3.iterrows() if float(r["Low"])  <= ref_low  * 1.002)
        if tags_high >= 2 and c_close < ref_high * 0.998:
            return ("failed_breakout", -0.40,
                    f"Failed breakout — {tags_high}x tag of ${ref_high:,.0f} resistance, no close-through (bearish exhaustion)")
        if tags_low >= 2 and c_close > ref_low * 1.002:
            return ("failed_breakdown", +0.40,
                    f"Failed breakdown — {tags_low}x tag of ${ref_low:,.0f} support, no close-through (bullish absorption)")

        return "none", 0.0, "No trap or sweep pattern detected"

    except Exception:
        return "none", 0.0, "Trap detection N/A"


def calculate_cvd(df: pd.DataFrame, lookback: int = 24) -> tuple:
    """
    Approximate Cumulative Volume Delta from OHLCV.
    Buy pressure per bar = volume * (close - low) / (high - low).
    Returns (category_str, raw_score) where score is in [-1, +1].
    """
    try:
        if df is None or len(df) < lookback + 5:
            return "neutral", 0.0
        d   = df.iloc[-(lookback + 5):].copy()
        hl  = (d["High"] - d["Low"]).replace(0, float("nan"))
        buy_vol  = d["Volume"] * (d["Close"] - d["Low"]) / hl
        sell_vol = d["Volume"] * (d["High"] - d["Close"]) / hl
        delta    = (buy_vol - sell_vol).fillna(0)
        cvd      = delta.cumsum()
        cvd_now  = float(cvd.iloc[-1])
        cvd_prev = float(cvd.iloc[-lookback])
        price_now  = float(d["Close"].iloc[-1])
        price_prev = float(d["Close"].iloc[-lookback])
        price_up   = price_now > price_prev
        cvd_up     = cvd_now  > cvd_prev
        # Normalised slope: how much CVD moved relative to average bar volume
        avg_vol    = float(d["Volume"].mean()) * lookback + 1e-8
        cvd_slope  = (cvd_now - cvd_prev) / avg_vol   # positive = net buying
        raw = float(np.tanh(cvd_slope * 2.5))          # scale so ±1 ≈ strong imbalance
        if   cvd_up and not price_up: category = "bull_divergence"
        elif not cvd_up and price_up: category = "bear_divergence"
        elif cvd_up and price_up:     category = "accumulation"
        elif not cvd_up and not price_up: category = "distribution"
        else:                         category = "neutral"
        return category, raw
    except Exception:
        return "neutral", 0.0


def rsi_trajectory(rsi_series: pd.Series, lookback: int = 10) -> str:
    try:
        valid = rsi_series.dropna()
        if len(valid) < lookback + 2:
            return "neutral"
        rsi_now  = float(valid.iloc[-1])
        rsi_prev = float(valid.iloc[-lookback])
        delta    = rsi_now - rsi_prev
        if delta > 8:    return "ascending"
        elif delta > 3:  return "mild_ascending"
        elif delta < -8: return "descending"
        elif delta < -3: return "mild_descending"
        return "neutral"
    except Exception:
        return "neutral"


def detect_rsi_divergence(closes: List[float], rsi: pd.Series, order: int = 5) -> str:
    if len(closes) < 30:
        return "none"
    price = np.array(closes)
    rsi_a = rsi.fillna(50).values
    p_mins = scipy.signal.argrelextrema(price, np.less,    order=order)[0]
    p_maxs = scipy.signal.argrelextrema(price, np.greater, order=order)[0]
    if len(p_mins) >= 2:
        li, pi = p_mins[-1], p_mins[-2]
        if len(closes) - li < 20 and price[li] < price[pi] and rsi_a[li] > rsi_a[pi]:
            return "bull"
    if len(p_maxs) >= 2:
        li, pi = p_maxs[-1], p_maxs[-2]
        if len(closes) - li < 20 and price[li] > price[pi] and rsi_a[li] < rsi_a[pi]:
            return "bear"
    return "none"


def detect_macd_divergence(closes: List[float], macd_hist: pd.Series, order: int = 5) -> str:
    if len(closes) < 40:
        return "none"
    price  = np.array(closes)
    hist   = macd_hist.fillna(0).values
    p_mins = scipy.signal.argrelextrema(price, np.less, order=order)[0]
    h_mins = scipy.signal.argrelextrema(hist,  np.less, order=order)[0]
    if len(p_mins) >= 2 and len(h_mins) >= 2:
        li, pi = p_mins[-1], p_mins[-2]
        if len(closes) - li < 20 and price[li] < price[pi] and hist[li] > hist[pi]:
            return "bull"
    return "none"


def find_levels(closes: List[float], order: int = 5) -> Tuple[List[float], List[float]]:
    arr  = np.array(closes)
    maxs = scipy.signal.argrelextrema(arr, np.greater, order=order)[0]
    mins = scipy.signal.argrelextrema(arr, np.less,    order=order)[0]

    def consolidate(levels, pct=0.015):
        if not levels: return []
        levels.sort()
        groups, grp = [], [levels[0]]
        for v in levels[1:]:
            if v <= grp[-1] * (1 + pct): grp.append(v)
            else:
                groups.append(sum(grp) / len(grp)); grp = [v]
        groups.append(sum(grp) / len(grp))
        return groups

    return consolidate(arr[mins].tolist()), consolidate(arr[maxs].tolist())


def nearest_level(levels: List[float], price: float, kind: str, pct: float = 0.15) -> Optional[float]:
    if not levels: return None
    relevant = [l for l in levels if abs(l - price) / price < pct]
    if kind == "support":
        below = [l for l in relevant if l < price]
        return max(below) if below else None
    else:
        above = [l for l in relevant if l > price]
        return min(above) if above else None


def week52_metrics(closes: List[float]) -> dict:
    if len(closes) < 20:
        return {"w52_low": None, "w52_high": None, "pct_from_low": None, "pct_from_high": None}
    arr   = np.array(closes[-252:]) if len(closes) >= 252 else np.array(closes)
    low   = float(arr.min())
    high  = float(arr.max())
    price = closes[-1]
    return {
        "w52_low":       round(low, 2),
        "w52_high":      round(high, 2),
        "pct_from_low":  round((price - low)  / low  * 100, 1),
        "pct_from_high": round((price - high) / high * 100, 1),
    }


# ════════════════════════════════════════════════════════════════
#  NEW INDICATORS
# ════════════════════════════════════════════════════════════════

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """Middle band (SMA), upper/lower bands, %B, and bandwidth."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b     = (series - lower) / (upper - lower)           # 0=at lower, 1=at upper
    bandwidth = (upper - lower) / mid * 100                  # % of middle
    return {"mid": mid, "upper": upper, "lower": lower,
            "pct_b": pct_b, "bandwidth": bandwidth}


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """%K and %D lines."""
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return {"k": k, "d": d}


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_c = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_c).abs(),
        (df["Low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    """Awesome Oscillator: SMA5 - SMA34 of midprice."""
    mid = (df["High"] + df["Low"]) / 2
    return mid.rolling(5).mean() - mid.rolling(34).mean()


def calculate_accelerator_oscillator(df: pd.DataFrame) -> pd.Series:
    """Accelerator Oscillator: AO - SMA5(AO)."""
    ao = calculate_awesome_oscillator(df)
    return ao - ao.rolling(5).mean()


def calculate_ichimoku(df: pd.DataFrame) -> dict:
    """
    Tenkan (9), Kijun (26), Senkou A (26 forward), Senkou B (52, 26 forward),
    Chikou (close shifted 26 back).
    """
    def _hl_avg(n):
        return (df["High"].rolling(n).max() + df["Low"].rolling(n).min()) / 2

    tenkan = _hl_avg(9)
    kijun  = _hl_avg(26)
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = _hl_avg(52).shift(26)
    chikou = df["Close"].shift(-26)
    return {"tenkan": tenkan, "kijun": kijun,
            "span_a": span_a, "span_b": span_b, "chikou": chikou}


def calculate_fibonacci_levels(closes: list, lookback: int = 365) -> dict:
    """
    Auto-detect the most recent significant swing high and low over `lookback` bars,
    then return Fibonacci retracement levels (0→100% = top→bottom of the swing).
    """
    arr  = np.array(closes[-lookback:])
    high = float(arr.max())
    low  = float(arr.min())
    diff = high - low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {f"{r*100:.1f}%": round(high - diff * r, 2) for r in ratios}
    return {"high": high, "low": low, "levels": levels}


