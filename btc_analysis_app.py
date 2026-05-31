"""
BTC Analysis — Streamlit App
Converted from btc_analysis.ipynb
"""

import warnings, os, csv as _csv, urllib.request, urllib.parse, json as _json, io as _io
from pathlib import Path as _Path
from datetime import datetime as _dt, timezone as _tz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal
from typing import List, Tuple, Optional
import yfinance as yf
import streamlit as st
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")

# ── Optional TA-Lib ──────────────────────────────────────────────
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    class _DummyTalib:
        def __getattr__(self, name):
            return lambda *args, **kwargs: np.zeros(len(args[0]))
    talib = _DummyTalib()

# ── Constants ────────────────────────────────────────────────────
LOOKBACK_DAYS = 400
MIN_DAYS      = 60

PATTERNS_BULLISH = {
    "Hammer": "CDLHAMMER", "Inverted Hammer": "CDLINVERTEDHAMMER",
    "Bullish Engulfing": "CDLENGULFING", "Piercing Line": "CDLPIERCING",
    "Bullish Marubozu": "CDLMARUBOZU", "Three White Soldiers": "CDL3WHITESOLDIERS",
    "Three Inside Up": "CDL3INSIDE", "Bullish Harami": "CDLHARAMI",
    "Morning Star": "CDLMORNINGSTAR", "Dragonfly Doji": "CDLDRAGONFLYDOJI",
}
PATTERNS_BEARISH = {
    "Hanging Man": "CDLHANGINGMAN", "Dark Cloud Cover": "CDLDARKCLOUDCOVER",
    "Bearish Engulfing": "CDLENGULFING", "Bearish Marubozu": "CDLMARUBOZU",
    "Three Black Crows": "CDL3BLACKCROWS", "Shooting Star": "CDLSHOOTINGSTAR",
    "Evening Star": "CDLEVENINGSTAR", "Gravestone Doji": "CDLGRAVESTONEDOJI",
}

# Order-book endpoints — aggregated across the 3 largest BTC perp venues
BINANCE_SPOT_DEPTH = "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5000"
BINANCE_FUT_DEPTH  = "https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000"
BYBIT_FUT_DEPTH    = "https://api.bybit.com/v5/market/orderbook?category=linear&symbol=BTCUSDT&limit=500"
OKX_SWAP_DEPTH     = "https://www.okx.com/api/v5/market/books?instId=BTC-USDT-SWAP&sz=400"

CLUSTER_BIN       = 50    # $50 bins — tight enough to see individual walls at BTC prices
HEATMAP_BIN       = 25
WALL_RANGE_PCT    = 15.0
HEATMAP_RANGE_PCT = 6.0
WALL_MIN_NOTL     = 1_000_000
WALL_MIN_DEPTH_PCT = 0.02   # 2% threshold
ALL_TIERS         = [2, 3, 5, 10, 15, 20, 25, 50, 75, 100]
HIGH_TIERS        = [25, 50, 75, 100]

_SIGNAL_LOG    = _Path(__file__).parent / "signal_log.csv"
_LOG_FIELDS    = ["ts", "score", "label", "direction",
                  "entry_price", "exit_price", "pct_move", "correct"]
_LOG_INTERVAL  = 300    # seconds between log writes (5 min)
_OUTCOME_HOURS = 72.0   # hours to wait before resolving outcome (matches "72h bias" name)

# Backtest cutoff: rows logged BEFORE this UTC timestamp were produced by an
# older scoring engine (e.g. the double-EMA-smoothing bug fixed in commit
# 97d0537 on 2026-05-31 00:24:19 SGT). They still render on the history chart
# for continuity, but are excluded from winrate and calibration so the numbers
# reflect ONLY the current engine.
# Bump this whenever you change the scoring engine in a way that breaks
# comparability with prior logs.
_BACKTEST_CUTOFF_UTC_ISO = "2026-05-30T16:24:19+00:00"  # = 2026-05-31 00:24 SGT


# ════════════════════════════════════════════════════════════════
#  SUPABASE HELPERS  (used when SUPABASE_URL + SUPABASE_KEY are set)
# ════════════════════════════════════════════════════════════════

def _supa_url() -> str:
    try: return st.secrets.get("SUPABASE_URL", "") or ""
    except Exception: return os.environ.get("SUPABASE_URL", "")

def _supa_key() -> str:
    try: return st.secrets.get("SUPABASE_KEY", "") or ""
    except Exception: return os.environ.get("SUPABASE_KEY", "")

def _supa_available() -> bool:
    return bool(_supa_url() and _supa_key())

def _supa_request(method: str, path: str, body=None) -> "dict | list | None":
    url = _supa_url().rstrip("/") + path
    data = _json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "apikey":        _supa_key(),
        "Authorization": f"Bearer {_supa_key()}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    })
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return _json.loads(r.read())
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  SIGNAL OUTCOME LOGGER
# ════════════════════════════════════════════════════════════════

def _log_bias_signal(score: float, label: str, price: float,
                     bias_72h: dict = None, poly: dict = None) -> None:
    """Append signal-log rows for this tick. Writes three Supabase tables:

      signal_log     — one row: aggregate score/label/direction/entry_price
      signal_detail  — N rows:  per-signal (name, raw_value, weight, contribution)
      polymarket_log — M rows:  per-market (question, strike, prob, weight, score)

    Local CSV always written as offline backup of signal_log. The detail and
    polymarket tables are Supabase-only — they exist purely to enable
    IC-weighting, SHAP, calibration, and probability-velocity in the future.

    All writes are best-effort. Missing tables / failures don't crash the app.
    """
    direction = "LONG" if score >= 25 else ("SHORT" if score <= -25 else "HOLD")
    ts = _dt.now(_tz.utc).isoformat()

    # ── 1. signal_log (Supabase + CSV) ────────────────────────────────────
    if _supa_available():
        try:
            _supa_request("POST", "/rest/v1/signal_log", body={
                "ts":          ts,
                "score":       round(score, 1),
                "label":       label,
                "direction":   direction,
                "entry_price": round(price, 2),
                "exit_price":  None,
                "pct_move":    None,
                "correct":     None,
            })
        except Exception:
            pass

    write_hdr = not _SIGNAL_LOG.exists()
    try:
        with open(_SIGNAL_LOG, "a", newline="") as f:
            w = _csv.writer(f)
            if write_hdr:
                w.writerow(_LOG_FIELDS)
            w.writerow([ts, round(score, 1), label, direction,
                        round(price, 2), "", "", ""])
    except Exception:
        pass

    # ── 2. signal_detail (per-signal contributions for IC / SHAP later) ──
    if _supa_available() and bias_72h:
        try:
            _sigs = bias_72h.get("signals") or {}
            _wts  = bias_72h.get("weights") or {}
            rows  = []
            for _name, _tup in _sigs.items():
                try:
                    _raw   = float(_tup[0]) if _tup else 0.0
                    _w     = float(_wts.get(_name, 0.0))
                    _contr = _raw * _w * 100   # scaled to 0–100 like the score
                    rows.append({
                        "ts":           ts,
                        "signal_name":  str(_name),
                        "raw_value":    round(_raw, 4),
                        "weight":       round(_w, 4),
                        "contribution": round(_contr, 4),
                    })
                except Exception:
                    continue
            if rows:
                _supa_request("POST", "/rest/v1/signal_detail", body=rows)
        except Exception:
            pass

    # ── 3. polymarket_log (per-strike probabilities for velocity later) ──
    if _supa_available() and poly:
        try:
            _mkts = (poly or {}).get("markets") or []
            rows  = []
            for _m in _mkts:
                try:
                    _q       = str(_m.get("question") or _m.get("event") or "")[:200]
                    _buckets = _m.get("buckets") or []
                    for _b in _buckets:
                        # buckets are tuples or lists: (label, prob, is_bull)
                        if not _b or len(_b) < 2:
                            continue
                        _lbl    = str(_b[0])
                        _prob   = float(_b[1])
                        _is_bull = None
                        if len(_b) >= 3 and _b[2] is not None:
                            _is_bull = bool(_b[2])
                        rows.append({
                            "ts":          ts,
                            "question":    _q,
                            "strike_lbl":  _lbl[:80],
                            "probability": round(_prob, 4),
                            "is_bull":     _is_bull,
                            "mkt_score":   round(float(_m.get("score", 0.0)), 2),
                            "mkt_weight":  float(_m.get("weight", 0)),
                        })
                except Exception:
                    continue
            if rows:
                # Polymarket can have ~50+ strikes per tick across all markets.
                # Insert in chunks of 100 to keep request bodies small.
                for i in range(0, len(rows), 100):
                    _supa_request("POST", "/rest/v1/polymarket_log", body=rows[i:i+100])
        except Exception:
            pass


def _supa_fetch_signals() -> "list | None":
    """Fetch all signal_log rows from Supabase, normalised to the CSV
    row format the rest of the app expects. Returns None on failure."""
    if not _supa_available():
        return None
    try:
        # Pull up to 10k rows ordered by ts ascending — plenty for live history.
        raw = _supa_request(
            "GET",
            "/rest/v1/signal_log?select=ts,score,label,direction,entry_price,"
            "exit_price,pct_move,correct&order=ts.asc&limit=10000",
        )
        if not isinstance(raw, list):
            return None
        rows = []
        for r in raw:
            rows.append({
                "ts":          r.get("ts") or "",
                "score":       "" if r.get("score") is None       else str(r["score"]),
                "label":       r.get("label") or "",
                "direction":   r.get("direction") or "",
                "entry_price": "" if r.get("entry_price") is None else str(r["entry_price"]),
                "exit_price":  "" if r.get("exit_price")  is None else str(r["exit_price"]),
                "pct_move":    "" if r.get("pct_move")    is None else str(r["pct_move"]),
                "correct":     "" if r.get("correct")     is None else str(r["correct"]),
            })
        return rows
    except Exception:
        return None


def _resolve_signal_outcomes(current_price: float) -> list:
    """Return all signal rows with outcomes filled where age ≥ _OUTCOME_HOURS.
    Merges the local CSV (historical, written when this PC ran the app) with
    Supabase (server-side, written by the GitHub Actions cron). Supabase wins
    on duplicate timestamps because it's authoritative for resolved outcomes."""

    now = _dt.now(_tz.utc)

    # ── Load Supabase rows (may be empty list if cron hasn't logged yet) ─
    supa_rows = _supa_fetch_signals() if _supa_available() else None
    supa_rows = supa_rows or []
    supa_ts   = {r.get("ts") for r in supa_rows if r.get("ts")}

    # ── Load local CSV rows (skipping any ts already present in Supabase) ─
    csv_rows = []
    if _SIGNAL_LOG.exists():
        try:
            with open(_SIGNAL_LOG, newline="") as f:
                for row in _csv.DictReader(f):
                    if row.get("ts") and row["ts"] not in supa_ts:
                        csv_rows.append(row)
        except Exception:
            pass

    # ── Resolve outcomes on CSV rows (in-memory + write back if changed) ─
    csv_changed = False
    for row in csv_rows:
        if row.get("correct") or not row.get("entry_price"):
            continue
        try:
            age = (now - _dt.fromisoformat(row["ts"])).total_seconds() / 3600
            if age < _OUTCOME_HOURS:
                continue
            ep  = float(row["entry_price"])
            pct = (current_price - ep) / ep * 100
            d   = row["direction"]
            _dir_right = (d == "LONG" and pct > 0) or (d == "SHORT" and pct < 0)
            row["exit_price"] = str(round(current_price, 2))
            row["pct_move"]   = f"{pct:+.2f}"
            row["correct"] = (
                "2"   if _dir_right and abs(pct) >= 3.0
                else "1"   if _dir_right
                else "N/A" if d == "HOLD"
                else "0"
            )
            csv_changed = True
        except Exception:
            pass
    if csv_changed:
        try:
            # Re-read full CSV (including rows we skipped above) and write back
            all_rows = []
            with open(_SIGNAL_LOG, newline="") as f:
                for r in _csv.DictReader(f):
                    all_rows.append(r)
            # Overlay our resolved updates
            by_ts = {r["ts"]: r for r in csv_rows}
            for i, r in enumerate(all_rows):
                if r.get("ts") in by_ts:
                    all_rows[i] = by_ts[r["ts"]]
            with open(_SIGNAL_LOG, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=_LOG_FIELDS)
                w.writeheader()
                w.writerows(all_rows)
        except Exception:
            pass

    # ── Resolve outcomes on Supabase rows (PATCH back to remote) ─────────
    for row in supa_rows:
        if row.get("correct") or not row.get("entry_price"):
            continue
        try:
            age = (now - _dt.fromisoformat(row["ts"])).total_seconds() / 3600
            if age < _OUTCOME_HOURS:
                continue
            ep  = float(row["entry_price"])
            pct = (current_price - ep) / ep * 100
            d   = row["direction"]
            _dir_right = (d == "LONG" and pct > 0) or (d == "SHORT" and pct < 0)
            correct = (
                "2"   if _dir_right and abs(pct) >= 3.0
                else "1"   if _dir_right
                else "N/A" if d == "HOLD"
                else "0"
            )
            _supa_request(
                "PATCH",
                f"/rest/v1/signal_log?ts=eq.{urllib.parse.quote(row['ts'])}",
                body={
                    "exit_price": round(current_price, 2),
                    "pct_move":   round(pct, 2),
                    "correct":    correct,
                },
            )
            row["exit_price"] = str(round(current_price, 2))
            row["pct_move"]   = f"{pct:+.2f}"
            row["correct"]    = correct
        except Exception:
            pass

    # ── Merge + sort by ts ascending ─────────────────────────────────────
    merged = csv_rows + supa_rows
    merged.sort(key=lambda r: r.get("ts") or "")
    return merged


def _accuracy_stats(rows: list) -> dict:
    """Accuracy + avg move by score tier (only resolved LONG/SHORT signals)."""
    buckets = [
        ("≥ +70  Strong Bull",   70,  100),
        ("+40 to +69  Bull",     40,   69),
        ("+25 to +39  Mild Bull",25,   39),
        ("−25 to −39  Mild Bear",-39, -25),
        ("−40 to −69  Bear",    -69,  -40),
        ("≤ −70  Strong Bear",  -100, -70),
    ]
    out = {}
    for label, lo, hi in buckets:
        vals, moves = [], []
        for r in rows:
            if r.get("correct") not in ("0", "1", "2"):
                continue
            try:
                s = float(r["score"])
                if lo <= s <= hi:
                    vals.append(1 if r["correct"] in ("1", "2") else 0)
                    if r.get("pct_move"):
                        moves.append(float(r["pct_move"]))
            except Exception:
                pass
        out[label] = {
            "n":        len(vals),
            "acc":      round(sum(vals) / len(vals) * 100, 1) if vals else None,
            "avg_move": round(sum(moves) / len(moves), 2)     if moves else None,
        }
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _backtest_tech_signals(days: int = 55) -> dict:
    """
    Backtest the technical component of the 72h bias on 55 days of 1h data.
    Liquidity signals (cascade, hunt zones, depth) cannot be backtested — no
    historical order book.  This tests the 43% technical slice only, normalised
    to a full-score equivalent so tiers (±25, ±40, ±70) stay meaningful.
    """
    _empty = {"stats": {}, "total": 0, "overall_acc": None, "bars": 0, "days": days}
    try:
        df = yf.Ticker("BTC-USD").history(period=f"{days}d", interval="1h", auto_adjust=True)
        if df is None or len(df) < 100:
            return _empty
        df.columns = [c.capitalize() for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()

        # Pre-compute all indicators once on the full series
        rsi_s       = calculate_rsi(df["Close"])
        _, _, mhist = calculate_macd(df["Close"])
        stoch       = calculate_stochastic(df)
        ema8_s      = df["Close"].ewm(span=8,  adjust=False).mean()   # was wrongly span=20
        ema21_s     = df["Close"].ewm(span=21, adjust=False).mean()   # was wrongly span=50
        ma50_s      = df["Close"].rolling(50,  min_periods=20).mean() # regime filter
        closes      = df["Close"].values

        # Weights mirror compute_72h_bias (tech-only slice)
        W       = {"rsi": 0.14, "macd": 0.10, "ema": 0.09, "stoch": 0.07, "mom": 0.03}
        W_TOTAL = sum(W.values())   # 0.43 — normalise up to full-score equivalent
        HORIZON = 72                # 72 × 1h = 72 hours (matches live log window)

        rows = []
        for i in range(50, len(df) - HORIZON):
            try:
                ws = 0.0

                # RSI
                rv = rsi_s.iloc[i]
                if pd.isna(rv): continue
                rsi  = float(rv)
                rsi5 = float(rsi_s.iloc[i - 5]) if i >= 5 and not pd.isna(rsi_s.iloc[i - 5]) else rsi
                slp  = 0.15 if (rsi > rsi5 and rsi < 50) else (-0.15 if (rsi < rsi5 and rsi > 50) else 0.0)
                if   rsi < 20: base =  1.00
                elif rsi < 30: base =  0.75
                elif rsi < 40: base =  0.40
                elif rsi < 50: base =  0.15
                elif rsi < 60: base = -0.15
                elif rsi < 70: base = -0.40
                elif rsi < 80: base = -0.75
                else:          base = -1.00
                ws += max(-1.0, min(1.0, base + slp)) * W["rsi"]

                # MACD histogram
                hv = mhist.iloc[i]; hp = mhist.iloc[i - 3] if i >= 3 else hv
                if not pd.isna(hv):
                    if   hv > 0 and hv > hp: ws +=  1.0 * W["macd"]
                    elif hv > 0:             ws +=  0.4 * W["macd"]
                    elif hv < 0 and hv < hp: ws += -1.0 * W["macd"]
                    elif hv < 0:             ws += -0.4 * W["macd"]

                # EMA structure
                e8  = float(ema8_s.iloc[i]);  e8p  = float(ema8_s.iloc[i - 1])  if i >= 1 else e8
                e21 = float(ema21_s.iloc[i]); e21p = float(ema21_s.iloc[i - 1]) if i >= 1 else e21
                e8sl = (e8 - float(ema8_s.iloc[i - 5])) / e8 * 100 if i >= 5 else 0
                bx = e8 > e21 and e8p <= e21p; nx = e8 < e21 and e8p >= e21p
                if bx:     er =  1.0
                elif nx:   er = -1.0
                elif e8 > e21: er =  min(1.0,  0.5 + min(0.4, abs(e8sl) * 15))
                else:          er =  max(-1.0, -0.5 - min(0.4, abs(e8sl) * 15))
                ws += er * W["ema"]

                # Stochastic
                k = stoch["k"].iloc[i]
                if not pd.isna(k):
                    k = float(k)
                    if   k < 20: ws +=  0.7 * W["stoch"]
                    elif k < 30: ws +=  0.4 * W["stoch"]
                    elif k > 80: ws += -0.7 * W["stoch"]
                    elif k > 70: ws += -0.4 * W["stoch"]

                # 24h momentum (24 bars = 24h on 1h candles)
                m6 = closes[i - 24] if i >= 24 else closes[0]
                r6 = (closes[i] - m6) / m6 * 100
                if   r6 >  5: ws +=  1.0 * W["mom"]
                elif r6 >  2: ws +=  0.5 * W["mom"]
                elif r6 < -5: ws += -1.0 * W["mom"]
                elif r6 < -2: ws += -0.5 * W["mom"]

                tech_score = (ws / W_TOTAL) * 100
                direction  = "LONG" if tech_score >= 25 else ("SHORT" if tech_score <= -25 else None)
                if direction is None:
                    continue

                # Trend regime filter — only take signals aligned with the MA50 regime.
                # Longs in downtrends and shorts in uptrends fight the dominant trend and
                # statistically underperform, so we skip them rather than count them as losses.
                ma50_val = ma50_s.iloc[i]
                if not pd.isna(ma50_val):
                    if direction == "LONG"  and closes[i] < float(ma50_val): continue
                    if direction == "SHORT" and closes[i] > float(ma50_val): continue

                pct     = (closes[i + HORIZON] - closes[i]) / closes[i] * 100
                correct = (direction == "LONG" and pct > 0) or (direction == "SHORT" and pct < 0)
                rows.append({"score": str(round(tech_score, 1)),
                             "direction": direction,
                             "pct_move": str(round(pct, 3)),
                             "correct": "1" if correct else "0"})
            except Exception:
                continue

        stats   = _accuracy_stats(rows)
        total   = len(rows)
        overall = round(sum(1 for r in rows if r["correct"] in ("1", "2")) / total * 100, 1) if total else None
        return {"stats": stats, "total": total, "overall_acc": overall,
                "bars": len(df), "days": days, "rows": rows}
    except Exception:
        return _empty


# ════════════════════════════════════════════════════════════════
#  DATA FETCHING (cached)
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    # Prefer Binance daily klines for BTC — single source of truth across the app.
    # Last candle is the live current-day candle (Close = live spot price).
    if "BTC" in ticker.upper():
        _bn = _fetch_binance_klines("1d", min(days, 1000))
        if _bn is not None and len(_bn) >= MIN_DAYS:
            return _bn
    # yfinance fallback (and primary path for non-BTC tickers)
    for method in ["history", "download"]:
        try:
            if method == "history":
                df = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
            else:
                df = yf.download(ticker, period=f"{days}d", progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < MIN_DAYS:
                continue
            df.columns = [c.capitalize() for c in df.columns]
            # Drop the last row if volume is 0 (incomplete session)
            if df["Volume"].iloc[-1] == 0:
                df = df.iloc[:-1]
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_crypto_signals(ticker: str) -> dict:
    result = {
        "fear_greed_value":  "N/A",
        "fear_greed_label":  "N/A",
        "etf_flow_trend":    "N/A",
        "etf_flow_detail":   "N/A",
        "btc_dominance":     "N/A",
        "dominance_trend":   "N/A",
        "funding_sentiment": "N/A",
        "momentum_7d":       "N/A",
        "momentum_30d":      "N/A",
        "crypto_qual_score": 0,
    }

    coin_id = "bitcoin"
    t = ticker.upper()
    if   "ETH"  in t: coin_id = "ethereum"
    elif "SOL"  in t: coin_id = "solana"
    elif "BNB"  in t: coin_id = "binancecoin"
    elif "XRP"  in t: coin_id = "ripple"
    elif "ADA"  in t: coin_id = "cardano"
    elif "AVAX" in t: coin_id = "avalanche-2"
    elif "DOGE" in t: coin_id = "dogecoin"
    elif "LINK" in t: coin_id = "chainlink"

    qual_adj = 0

    def _get(url, timeout=8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _json.loads(r.read().decode())
        except Exception:
            return None

    # 1. Fear & Greed
    try:
        data = _get("https://api.alternative.me/fng/?limit=2")
        if data and data.get("data"):
            current  = data["data"][0]
            fgi      = int(current["value"])
            label    = current["value_classification"]
            prev_fgi = int(data["data"][1]["value"]) if len(data["data"]) > 1 else fgi
            trend    = "up" if fgi > prev_fgi else ("down" if fgi < prev_fgi else "flat")
            result["fear_greed_value"] = fgi
            if fgi <= 20:
                qual_adj += 3
                result["fear_greed_label"] = f"{label} ({fgi}, {trend}) -- extreme fear = contrarian BUY"
            elif fgi <= 35:
                qual_adj += 2
                result["fear_greed_label"] = f"{label} ({fgi}, {trend})"
            elif fgi <= 55:
                result["fear_greed_label"] = f"{label} ({fgi}, {trend})"
            elif fgi <= 75:
                qual_adj -= 1
                result["fear_greed_label"] = f"{label} ({fgi}, {trend}) -- elevated greed, be cautious"
            else:
                qual_adj -= 3
                result["fear_greed_label"] = f"{label} ({fgi}, {trend}) -- extreme greed = contrarian SELL"
    except Exception:
        pass

    # 2. BTC Spot ETF Flow Proxy
    if coin_id == "bitcoin":
        try:
            ETF_UNIVERSE = {
                "IBIT": "BlackRock", "FBTC": "Fidelity",
                "ARKB": "ARK/21Shares", "BITB": "Bitwise",
                "HODL": "VanEck", "BTCO": "Invesco", "EZBC": "Franklin",
            }
            etf_daily    = {}
            flow_stats   = {}

            for etf in ETF_UNIVERSE:
                try:
                    h = yf.download(etf, period="400d", interval="1d",
                                    progress=False, auto_adjust=True)
                    if isinstance(h.columns, pd.MultiIndex):
                        h.columns = h.columns.get_level_values(0)
                    if h.empty or len(h) < 15:
                        continue
                    direction  = np.where(h["Close"] >= h["Open"], 1.0, -1.0)
                    signed_vol = pd.Series(
                        h["Volume"].values * direction, index=h.index, name=etf
                    )
                    etf_daily[etf] = signed_vol
                except Exception:
                    continue

            if etf_daily:
                norm_series = []
                for etf, sv in etf_daily.items():
                    std = sv.std()
                    if std > 0:
                        norm_series.append(sv / std)

                if norm_series:
                    aligned = pd.DataFrame(
                        {s.name: s for s in norm_series}
                    ).fillna(0)
                    composite_sv = aligned.sum(axis=1)

                    # Strip timezone
                    if isinstance(composite_sv.index, pd.DatetimeIndex) and composite_sv.index.tz is not None:
                        composite_sv.index = composite_sv.index.tz_convert(None)

                    composite_cum = composite_sv.cumsum()
                    result["etf_composite_cum"]  = composite_cum
                    result["etf_daily_composite"] = composite_sv
                    result["etf_tickers_loaded"]  = list(etf_daily.keys())
                    result["_etf_daily_raw"]      = etf_daily

                    # Signal derivation
                    sv_idx = composite_sv.index
                    if hasattr(sv_idx, "tz") and sv_idx.tz is not None:
                        sv_idx = sv_idx.tz_convert(None)
                    sv_datestr_dict = dict(zip(
                        pd.to_datetime(sv_idx).strftime("%Y-%m-%d"),
                        composite_sv.values
                    ))
                    sorted_dates   = sorted(sv_datestr_dict.keys())[-60:]
                    sv_aligned_vals = np.array([sv_datestr_dict[d] for d in sorted_dates], dtype=float)
                    roll20_aligned  = pd.Series(sv_aligned_vals).rolling(20, min_periods=5).mean().values
                    roll20_last     = float(roll20_aligned[-1])
                    roll20_5d_ago   = float(roll20_aligned[-5]) if len(roll20_aligned) >= 5 else roll20_last
                    is_inflow       = roll20_last > 0
                    improving       = (roll20_last > roll20_5d_ago) if is_inflow else (roll20_last < roll20_5d_ago)
                    flow_std        = float(np.std(sv_aligned_vals))
                    min_threshold   = max(flow_std * 0.4, 0.2)
                    accelerating    = (abs(roll20_last) > min_threshold and improving and
                                       abs(roll20_last - roll20_5d_ago) > min_threshold * 0.3)

                    n_inflow, n_outflow, n_total = 0, 0, 0
                    top_notes = []
                    for etf, sv_series in etf_daily.items():
                        try:
                            etf_idx = sv_series.index
                            if hasattr(etf_idx, "tz") and etf_idx.tz is not None:
                                etf_idx = etf_idx.tz_convert(None)
                            etf_dict = dict(zip(
                                pd.to_datetime(etf_idx).strftime("%Y-%m-%d"),
                                sv_series.values
                            ))
                            etf_vals = np.array([etf_dict.get(d, 0.0) for d in sorted_dates], dtype=float)
                            etf_roll20_last = float(
                                pd.Series(etf_vals).rolling(20, min_periods=5).mean().iloc[-1]
                            )
                            n_total += 1
                            if etf_roll20_last > 0:
                                n_inflow += 1; top_notes.append(f"{etf} (↑)")
                            else:
                                n_outflow += 1; top_notes.append(f"{etf} (↓)")
                        except Exception:
                            continue

                    top_notes_sorted = sorted(
                        top_notes,
                        key=lambda x: (0 if "↑" in x else 1) if is_inflow else (0 if "↓" in x else 1)
                    )[:3]
                    pct_agree = (n_inflow if is_inflow else n_outflow) / n_total if n_total > 0 else 0

                    if is_inflow and pct_agree >= 0.5:
                        trend  = "Positive"
                        detail = f"{'Accelerating i' if accelerating else 'I'}nflows - {n_inflow}/{n_total} ETFs positive ({', '.join(top_notes_sorted)})"
                        qual_adj += 2 if accelerating else 1
                    elif not is_inflow and pct_agree >= 0.5:
                        trend  = "Negative"
                        detail = f"{'Accelerating o' if accelerating else 'O'}utflows - {n_outflow}/{n_total} ETFs negative ({', '.join(top_notes_sorted)})"
                        qual_adj -= 2 if accelerating else 1
                    else:
                        trend  = "Mixed"
                        detail = f"Mixed - {n_inflow} inflow / {n_outflow} outflow ({', '.join(top_notes_sorted)})"

                    result["etf_flow_trend"]  = trend
                    result["etf_flow_detail"] = detail
                    result["etf_flow_stats"]  = {
                        "n_etfs": n_total, "n_inflow": n_inflow, "n_outflow": n_outflow,
                        "roll20_last": roll20_last, "accelerating": accelerating,
                    }
                else:
                    result["etf_flow_trend"]  = "Neutral"
                    result["etf_flow_detail"] = "ETF data insufficient"
            else:
                result["etf_flow_trend"]  = "Neutral"
                result["etf_flow_detail"] = "No ETF data available"
        except Exception:
            result["etf_flow_trend"]  = "Neutral"
            result["etf_flow_detail"] = "ETF fetch error"

    # 3. BTC Dominance
    try:
        mkt = _get("https://api.coingecko.com/api/v3/global")
        if mkt and mkt.get("data"):
            dom = mkt["data"]["market_cap_percentage"].get("btc")
            if dom is not None:
                result["btc_dominance"] = f"{dom:.1f}%"
                if coin_id == "bitcoin":
                    if dom >= 55:
                        qual_adj += 1
                        result["dominance_trend"] = f"{dom:.1f}% -- high BTC season"
                    elif dom >= 48:
                        result["dominance_trend"] = f"{dom:.1f}% -- balanced"
                    else:
                        result["dominance_trend"] = f"{dom:.1f}% -- alt season may reduce BTC flows"
                else:
                    if dom < 48:
                        qual_adj += 1
                        result["dominance_trend"] = f"{dom:.1f}% -- alt season favoured"
                    elif dom >= 58:
                        qual_adj -= 1
                        result["dominance_trend"] = f"{dom:.1f}% -- BTC dominance headwind for alts"
                    else:
                        result["dominance_trend"] = f"{dom:.1f}%"
    except Exception:
        pass

    # 4. Momentum + Funding via CoinGecko
    try:
        coin_data = _get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            f"?localization=false&tickers=false&market_data=true"
            f"&community_data=false&developer_data=false&sparkline=false"
        )
        if coin_data and coin_data.get("market_data"):
            md  = coin_data["market_data"]
            d7  = md.get("price_change_percentage_7d",  0) or 0
            d30 = md.get("price_change_percentage_30d", 0) or 0
            result["momentum_7d"]  = f"{d7:+.1f}%"
            result["momentum_30d"] = f"{d30:+.1f}%"
            if d7 > 20 and d30 > 30:
                result["funding_sentiment"] = f"Crowded longs (7d {d7:+.0f}%, 30d {d30:+.0f}%) -- longs likely paying high funding"
                qual_adj -= 2
            elif d7 > 8 and d30 > 10:
                result["funding_sentiment"] = f"Bullish (7d {d7:+.0f}%, 30d {d30:+.0f}%)"
                qual_adj += 1
            elif d7 < -15 and d30 < -20:
                result["funding_sentiment"] = f"Oversold shorts (7d {d7:+.0f}%, 30d {d30:+.0f}%) -- short squeeze potential"
                qual_adj += 2
            elif d7 < -5:
                result["funding_sentiment"] = f"Bearish (7d {d7:+.0f}%, 30d {d30:+.0f}%)"
                qual_adj -= 1
            else:
                result["funding_sentiment"] = f"Neutral (7d {d7:+.0f}%, 30d {d30:+.0f}%)"
    except Exception:
        pass

    result["crypto_qual_score"] = max(-10, min(10, qual_adj))
    return result


# ════════════════════════════════════════════════════════════════
#  POLYMARKET AI EXTRACTION LAYER
# ════════════════════════════════════════════════════════════════

_POLYMARKET_AI_SYSTEM_PROMPT = (
    "You are a Bitcoin market data extraction agent for a Streamlit dashboard.\n\n"
    "Your task is to parse Polymarket Bitcoin prediction market questions and convert them "
    "into structured sentiment data for a BTC directional scoring engine.\n\n"
    "The scoring engine predicts whether Bitcoin is bullish or bearish over the next 24 hours.\n\n"
    "You MUST extract:\n"
    "market type\ntimeframe\nbullish probabilities\nbearish probabilities\n"
    "implied directional bias\nconfidence weighting\nrelevant strike levels\n\n"
    "Ignore irrelevant markets.\n\n"
    "INPUT FORMAT\n\n"
    "You will receive raw market strings like:\n\n"
    "BTC Up or Down 5m (51% Up)\n"
    "BTC Up or Down 4h (14% Up)\n"
    "BTC Up or Down Daily (53% Up)\n"
    "What price will Bitcoin hit May 4-10? (↓ 78,000 at 36% | ↑ 84,000 at 8%)\n"
    "Bitcoin price on May 8? (78,000-80,000 at 48% | 80,000-82,000 at 44%)\n"
    "Bitcoin above ___ on May 8? (68,000 at 100% | 70,000 at 100%)\n"
    "What price will Bitcoin hit in May? (↑ 85,000 at 51% | ↓ 75,000 at 51%)\n\n"
    "OUTPUT FORMAT\n\n"
    "Return ONLY valid JSON.\n\n"
    'Example structure:\n\n{ "markets": [ { "question": "BTC Up or Down 5m", '
    '"category": "directional", "timeframe": "5m", "bullish_probability": 51, '
    '"bearish_probability": 49, "bias": "bullish", "confidence": 0.02, "weight": 0.1 }, '
    '{ "question": "BTC Up or Down 4h", "category": "directional", "timeframe": "4h", '
    '"bullish_probability": 14, "bearish_probability": 86, "bias": "bearish", '
    '"confidence": 0.72, "weight": 0.7 } ], "aggregate": { "bullish_score": 42.7, '
    '"bearish_score": 57.3, "net_sentiment": -14.6, "overall_bias": "bearish" } }\n\n'
    "EXTRACTION RULES\n\n"
    "1. Directional Markets\n"
    "Examples: BTC Up or Down 5m, BTC Up or Down 4h, BTC Up or Down Daily\n"
    "Interpret: '51% Up' = bullish_probability = 51, bearish_probability = 100 - bullish_probability\n"
    "Bias rules: >55 bullish → bullish, <45 bullish → bearish, otherwise → neutral\n"
    "Confidence formula: confidence = abs(bullish_probability - 50) / 50\n"
    "Weighting: 5m = 0.1, 1h = 0.4, 4h = 0.7, Daily = 1.0, Weekly/monthly = 0.5\n\n"
    "2. Price Target Markets\n"
    "Examples: What price will Bitcoin hit May 4-10?, What price will Bitcoin hit in May?\n"
    "Extract upside targets, downside targets, and associated probabilities.\n"
    "Higher upside probabilities = bullish; higher downside probabilities = bearish.\n"
    "directional_score = sum(upside_target_probability) - sum(downside_target_probability), "
    "normalized -100 to +100.\n"
    "Example: ↑ 85,000 at 51% | ↓ 75,000 at 51% → Net = 0 → neutral\n\n"
    "3. Range Markets\n"
    "Example: Bitcoin price on May 8? (78,000-80,000 at 48% | 80,000-82,000 at 44%)\n"
    "Determine weighted expected price: expected_price = Σ(midpoint × probability).\n"
    "If expected price > current BTC price: bullish. Else: bearish.\n\n"
    "4. Threshold Markets\n"
    "Example: Bitcoin above ___ on May 8? (68,000 at 100% | 70,000 at 100%)\n"
    "Very low thresholds with 100% probability are weak bullish signals.\n"
    "Thresholds near spot price have stronger predictive value.\n"
    "Assign low confidence unless thresholds are close to market price.\n\n"
    "5. Missing or Invalid Data\n"
    "If no probability exists: skip the market.\n"
    'If parsing fails: return: { "status": "unparseable" }\n'
    "Do not hallucinate values.\n\n"
    "FINAL AGGREGATION LOGIC\n"
    "weighted_bullish = Σ(bullish_probability × weight × confidence)\n"
    "weighted_bearish = Σ(bearish_probability × weight × confidence)\n"
    "Normalize to percentages. Net sentiment: bullish_score - bearish_score.\n"
    "Interpretation: >+15 = bullish, <-15 = bearish, otherwise neutral.\n\n"
    "IMPORTANT RULES\n"
    "Return ONLY JSON. No markdown. No explanations. No commentary. No code fences.\n"
    "Do not invent probabilities. Preserve exact question text.\n"
    "Always convert percentages into numbers.\n"
    "Always include confidence and weight.\n"
    "Always include aggregate section."
)


def _scored_markets_to_ai_strings(scored_markets: list) -> list:
    """Convert scored Polymarket markets to normalized strings for AI extraction input."""
    strings = []
    for m in scored_markets:
        q     = m.get("question", "")
        dtype = m.get("display_type", "")
        od    = m.get("outcomes_display", [])
        try:
            if dtype == "updown" and len(od) >= 1:
                up_pct = int(od[0][1] * 100)
                strings.append(f"{q} ({up_pct}% Up)")
            elif dtype == "yesno" and len(od) >= 2:
                y_lbl = str(od[0][0]).strip()
                y_pct = int(od[0][1] * 100)
                n_lbl = str(od[1][0]).strip()
                n_pct = int(od[1][1] * 100)
                strings.append(f"{q} ({y_lbl} at {y_pct}% | {n_lbl} at {n_pct}%)")
            elif dtype == "range" and od:
                parts = [f"{str(e[0]).strip()} at {int(e[1]*100)}%" for e in od[:4]]
                strings.append(f"{q}? ({' | '.join(parts)})")
        except Exception:
            continue
    return strings


def _call_polymarket_ai(market_strings: list, api_key: str) -> "dict | None":
    """Send formatted Polymarket market strings to Claude for AI-powered sentiment extraction."""
    if not api_key or not market_strings:
        return None
    payload = _json.dumps({
        "model":      "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
        "system":     _POLYMARKET_AI_SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": "\n".join(market_strings)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":           api_key,
            "anthropic-version":   "2023-06-01",
            "content-type":        "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = _json.loads(r.read().decode())
            text = body["content"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return _json.loads(text)
    except Exception:
        return None


_CLAUDE_SYSTEM_PROMPT = """You are an expert Bitcoin trading analyst embedded in a real-time BTC dashboard. Synthesise every signal into one clear, actionable verdict. Be direct. No hedging words.

## ENGINE ARCHITECTURE
Two parallel engines: **72h bias** (24–72h horizon, score ±100, 16 signals) and **24h bias** (intraday, score ±100, 13 signals). Both use ADX-based regime switching (TREND / RANGE / TRANSITION) which dynamically shifts signal weights. The actual current weights (post-regime, post-decay) are shown next to each signal in the data — trust those numbers, not generic priors.

## 72h SIGNAL HIERARCHY (BASE weights — TREND/RANGE shift them, then horizon-decay applies)

**Tier 1 — Structural anchors:**
- Polymarket Sentiment (16%, raised in RANGE): real-money crowd thesis, horizon-weighted toward 24–72h questions; same-day questions discounted to 0.2×. High-agreement (>85%) markets get a tanh steepening boost. Highest-quality external signal.
- EMA Structure (20% BASE → 28% TREND → 10% RANGE): 4h EMA20/50 cross + slope. 4h preferred over 1h.
- OI Funding (13%): combined funding rate (centered at 0.01%) + 24h OI delta. Crowded longs = bearish; rising OI on price up = conviction.
- 4h RSI (13% BASE → 6% TREND → 18% RANGE): RSI(14) on 4h with **trend-conditional center**: 44 in downtrend (EMA20<EMA50), 56 in uptrend, 50 neutral. RSI=37 in a downtrend is NOT oversold — it's normal.

**Tier 2 — Reversal & structure:**
- RSI Divergence (8%, boosted to 13% in RANGE): 4h preferred. Binary ±1.0 — high alpha when present.
- Price Structure (5–7%): 4h trap/sweep detection. Bull/bear traps, failed breakouts.
- CVD Trend (6%, boosted to 8% in TREND): 24h cumulative volume delta. Bull/bear divergence boosted +0.3.

**Tier 3 — Liquidation cluster signals (forward-looking + reactive pair):**
- Hunt Zone Pull (4–5%): **liquidation clusters within 7%** acting as price magnets (NOT order book walls). side=ASK → short-liq cluster above (squeeze magnet pulling up); side=BID → long-liq cluster below (flush magnet pulling down). Swept zones discounted 65%.
- Swept Reversal (3–5%, strongest in RANGE): **reactive twin** to Hunt Zone Pull — fires when last 3×4h candles wicked into a cluster ≥$1M then closed back through it (bear when ASK swept-and-rejected; bull when BID swept-and-rejected). High alpha at range edges.
- Cascade Direction (3–5%): liq fuel ratio (long-liq vs short-liq notional). Dampened 50% when opposing 6-day trend.

**Tier 4 — Divergence & microstructure (minimal weight):**
- MACD Divergence (2–4%): 4h preferred.
- Order Book Depth (1–3%), RSI Level (1–2%), MACD Momentum (1–2%), Short Momentum (1%), Stochastic Zone (~1% floor).

## 24h ENGINE (13 signals, separate weight table)
- Liq Imbalance (14–16%): forced trades over last 4h — strongest unique 24h signal.
- CVD (13–14%): 6h volume delta, volume-normalized.
- Momentum (11–14%): blend of 1h + 4h returns.
- Funding (9–11%): same neutral as 72h (0.01% center).
- Fast MACD (7–12%): 6/13/4 params.
- EMA Align (4–10%, boosted in TREND): 1h/15m EMA cross.
- RSI(7) (4–12%, boosted in RANGE), Stochastic (2–9%, boosted in RANGE), Candle Pos (6–7%), Order Book (4–6%).
- **Nearby Liq Magnet (3–6%, strongest in RANGE)**: NEW — projected liquidation clusters within ±2.5% pulling price intraday. Same cluster data as 72h Hunt Zone Pull but tighter window and no daily-coherence damping. Strongest signal at range edges.
- Polymarket (3%): uses `signal_24h` — same-day questions boosted 1.5×, far markets discounted to 0.15×.
- Fear & Greed (1–4%, extremes only, <20 or >80).

## LIQUIDATION DATA SOURCE QUALITY
Liquidity maps drive Hunt Zone Pull, Swept Reversal, Cascade Direction, and Nearby Liq Magnet. Source is reported as `liq_map_source`:
- **coinglass**: paid API, highest quality (if available).
- **hyblock**: free fallback, good coverage.
- **binance_synthetic**: built locally from Binance OI history + taker ratio + leverage tier fan-out (3x/5x/10x/25x/50x/100x with empirical weights 0.10/0.18/0.30/0.24/0.12/0.06), 7-day half-life decay. Reasonable proxy but coarser — discount liq-cluster signals by ~20% confidence.
- **orderbook_proxy**: last-resort fallback using book walls as a stand-in. Liq-cluster signals are weak in this mode; lean on technicals + Polymarket.

## CONFLICT RULES
1. **Polymarket + EMA agree** → highest conviction. Both capital-anchored.
2. **4h RSI bullish + EMA bearish** → counter-trend signal. The center fix already discounts this; don't double-count.
3. **72h and 24h disagree** → 72h owns the direction, 24h owns the timing.
4. **Hunt Zone Pull + Swept Reversal disagree** → Swept Reversal wins (reactive beats projection). Pull says "magnet exists"; Swept says "magnet rejected price".
5. **Hunt Zone Pull + Swept Reversal agree on bear (ASK above, then ASK rejection)** → high-quality fade short setup.
6. **Nearby Liq Magnet + Liq Imbalance agree** → intraday squeeze imminent in the indicated direction.
7. **CVD divergence + price extreme** → high-quality reversal flag.
8. **Daily trend vs 72h bias divergence** → favour 72h bias for the trade but note the headwind.
9. **Cycle phase extreme (>+16 or <-16)** → overrides mild 72h signals.

## POLYMARKET HORIZON
The engine already routes questions: same-day → 24h signal (1.5× boost), 24–168h → 72h signal (1.5× boost in 24–72h sweet spot). You don't need to re-filter by expiry. But still call out questions expiring in <12h as low forward-value when summarizing.

## OVERSOLD/OVERBOUGHT
If RSI < 32 or Stoch < 22 → SHORT carries reversal risk; name invalidation level. If RSI > 68 or Stoch > 78 → LONG carries same risk.

## SCORE THRESHOLDS
≥70 STRONG · 40–69 BULL/BEAR · 15–39 MILD · ±14 NEUTRAL. Actionable trade only at |score| ≥ 40.

## POSITION SIZING
STRONG: 100% · BUY/SELL: 75% · LEAN: 40% · NEUTRAL: 0%. Reduce one tier if `liq_map_source` = orderbook_proxy AND liq-cluster signals are dominant drivers.

## OUTPUT FORMAT — EXACTLY this structure:

**VERDICT: [STRONG BUY / BUY / LEAN BUY / NEUTRAL / LEAN SELL / SELL / STRONG SELL]** · Confidence: [LOW / MEDIUM / HIGH] · Size: [X%]
[One sentence: the single clearest reason for the verdict.]

**72h read:** [2 sentences. Lead with 2 highest-weight signals (name + value + weight%). Note the regime.]

**24h read:** [1–2 sentences. Lead with the dominant 24h signal. Does it confirm or contradict the 72h direction?]

**Cycle / far-term:** [1–2 sentences on cycle phase, dominance, momentum.]

**Key conflict or alignment:** [One sentence. Name the two signals most opposed — or confirm conviction.]

**Trade setup:** "Direction · entry (price + 1h close trigger) · stop (invalidation) · target · R:R · structural reason." Calculate R:R explicitly. If |score| < 40, state the exact level/signal change that would validate the trade.

**Confidence:**
- HIGH: Tier 1 signals agree + 24h confirms + no major counter-signal
- MEDIUM: 1 Tier 1 signal clear with momentum support, or strong signal offset by meaningful counter
- LOW: Tier 1 split, 24h contradicts 72h, or dominant signal opposes daily trend

Under 300 words total."""


def _call_claude_interpretation(
    price: float,
    bias_score: float,
    bias_label: str,
    bias_signals: dict,
    pm_thesis: float,
    pm_label: str,
    pm_markets: list,
    pred_label: str,
    pred_score: int,
    fear_greed: str,
    api_key: str,
    btc_dominance: str = "N/A",
    momentum_7d: str = "N/A",
    momentum_30d: str = "N/A",
    pct_from_high: "float | None" = None,
    pct_from_low:  "float | None" = None,
    cycle_phase: str = "N/A",
    cycle_score: int = 0,
    adx_val: float = float("nan"),
    signal_weights: "dict | None" = None,
    # 24h engine
    bias_24h_score: float = 0.0,
    bias_24h_label: str = "N/A",
    bias_24h_regime: str = "N/A",
    bias_24h_signals: "dict | None" = None,
    bias_24h_weights: "dict | None" = None,
    # 15-min chart signals
    ema_cross_15m: str = "N/A",
    vwap_bias_15m: str = "N/A",
    poc_vs_price_15m: str = "N/A",
    atr_15m: str = "N/A",
    atr_pct_15m: str = "N/A",
    # Liquidation map source quality
    liq_map_source: str = "N/A",
) -> str:
    """Call Claude with full system prompt to interpret all dashboard signals."""
    weights = signal_weights or {}

    # Format individual signals with their weight so Claude knows what matters most
    signal_lines = []
    if isinstance(bias_signals, dict):
        for name, val in bias_signals.items():
            if isinstance(val, tuple) and len(val) >= 2:
                raw_v, note_v = val[0], val[1]
                w_pct = f"{weights.get(name, 0)*100:.0f}%" if name in weights else ""
                signal_lines.append(f"  • [{w_pct} weight] {name} ({raw_v:+.2f}): {note_v}")

    # 24h engine signal lines
    sig24_lines = []
    if isinstance(bias_24h_signals, dict):
        w24 = bias_24h_weights or {}
        for name, val in bias_24h_signals.items():
            try:
                v = float(val)
                w_pct = f"{w24.get(name, 0)*100:.0f}%" if name in w24 else ""
                sig24_lines.append(f"  • [{w_pct} weight] {name}: {v:+.2f}")
            except Exception:
                pass

    pm_lines = []
    for m in (pm_markets or [])[:12]:
        q   = m.get("question", "")
        sc  = m.get("individual_score", 0)
        wsc = m.get("weighted_score", 0)
        rat = m.get("rationale", "")
        od  = m.get("outcomes_display", [])
        tops = " | ".join(f"{e[0]} {e[1]:.0%}" for e in od[:3]) if od else ""
        pm_lines.append(f"  • [score {sc:+.1f} → weighted {wsc:+.1f}] {q}: {tops} ({rat})")

    _adx_str = f"{adx_val:.0f}" if adx_val == adx_val else "N/A"  # nan check
    _pfh_str = f"{pct_from_high:.0f}% from ATH area" if pct_from_high is not None else "N/A"
    _pfl_str = f"+{pct_from_low:.0f}% from 52w low" if pct_from_low is not None else "N/A"

    user_msg = f"""=== MACRO CONTEXT ===
BTC Price: ${price:,.0f}
Fear & Greed: {fear_greed}
BTC Dominance: {btc_dominance}  (rising dominance = BTC outperforming alts = risk-on for BTC)
7d Return: {momentum_7d}  |  30d Return: {momentum_30d}
52-Week Range: {_pfl_str}, {_pfh_str}
ADX Trend Strength: {_adx_str} (>25 = strong directional trend, <20 = choppy/ranging)
Cycle Phase: {cycle_phase} ({cycle_score:+d}/24)  (positive = accumulation/bull; negative = distribution/bear)

=== 72h DIRECTIONAL BIAS: {bias_score:+.0f}/100 ({bias_label}) ===
Signal breakdown (weight → influence on composite score):
{chr(10).join(signal_lines) or "  (no signal data)"}

=== 24h DIRECTIONAL BIAS: {bias_24h_score:+.0f}/100 ({bias_24h_label}) · regime: {bias_24h_regime} ===
Intraday engine — separate from 72h. Use to confirm/contradict 72h direction and set entry timing.
{chr(10).join(sig24_lines) or "  (no signal data)"}

=== POLYMARKET CROWD THESIS: {pm_thesis:+.2f}/10 ({pm_label}) ===
Real money at risk — treat as highest-conviction external signal.
Markets expiring ≤72h are most relevant to the 72h bias; weekly/range markets inform the far-term.
{chr(10).join(pm_lines) or "  (no markets)"}

=== DAILY TREND MODEL: {pred_label} ({pred_score:+d}/14) ===
Built on daily candles — the primary far-term signal. Captures multi-day structure.

=== 15-MIN INTRADAY CHART (last 24h, entry-timing context) ===
EMA8 vs EMA21: {ema_cross_15m}
Price vs VWAP: {vwap_bias_15m}
POC vs Price: {poc_vs_price_15m}
ATR (volatility): {atr_15m} = {atr_pct_15m}% of price

=== LIQUIDATION MAP SOURCE: {liq_map_source} ===
Drives Hunt Zone Pull, Swept Reversal, Cascade Direction (72h) and Nearby Liq Magnet (24h).
Quality ladder: coinglass > hyblock > binance_synthetic > orderbook_proxy.
If source is binance_synthetic, treat liq-cluster signals with ~80% confidence.
If source is orderbook_proxy, treat liq-cluster signals as noise and lean on technicals + Polymarket.

Now give your analysis using the OUTPUT FORMAT in the system prompt: 72h read, 24h read, cycle/far-term, conflict/alignment, trade setup."""

    payload = _json.dumps({
        "model":      "claude-sonnet-4-6",
        "max_tokens": 900,
        "system":     _CLAUDE_SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_msg}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = _json.loads(r.read().decode())
            return body["content"][0]["text"].strip()
    except Exception as e:
        return f"Error calling Claude: {e}"


@st.cache_data(ttl=60, show_spinner=False)
def fetch_polymarket_btc_sentiment(current_price: float) -> dict:
    """
    Fetches active BTC prediction markets from Polymarket and scores them against
    the 72h directional thesis: is BTC bullish or bearish over the next 72 hours?

    Thesis weights by question category:
      1x — 5-min noise, trivial floors far below current
      2x — 4h directional, floor support 5-15% below, long-term targets
      3x — daily/24h directional, near-level support, downside targets
      4x — price range for today/this week, hard weekly price targets

    Individual score: -10 to +10.  thesis_score = sum(score×weight) / sum(weights).
    No pre-filter on near-resolved or near-50/50 markets — the dampening multipliers
    in the scoring formula handle signal strength naturally.
    """
    import re
    from datetime import datetime, timezone

    result = {
        "signal": 0.0, "confidence": 0.0, "markets_used": 0,
        "markets": [], "detail": "Polymarket N/A",
        "thesis_score": 0.0, "thesis_label": "NEUTRAL",
    }
    now = datetime.now(timezone.utc)

    def _get(url, timeout=10):
        import urllib.error
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept":     "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                data = _json.loads(raw)
                # Polymarket sometimes returns {"error": ...} with HTTP 200
                if isinstance(data, dict) and data.get("error"):
                    return None
                return data
        except urllib.error.HTTPError as e:
            result["detail"] = f"Polymarket HTTP {e.code}: {e.reason}"
            return None
        except Exception:
            return None

    def _parse_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return _json.loads(val)
            except Exception:
                return []
        return []

    def _extract_price(text):
        """Extract the primary dollar price from text, e.g. '$80,000', '80k'."""
        m = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*([kK]?)\b', text)
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
                if m.group(2).lower() == "k":
                    num *= 1000
                if num > 10_000:
                    return num
            except Exception:
                pass
        m = re.search(r'\b([\d]{5,}(?:,[\d]{3})*)\b', text)
        if m:
            try:
                num = float(m.group(1).replace(",", ""))
                if num > 10_000:
                    return num
            except Exception:
                pass
        return None

    def _parse_bucket_price(s):
        """Parse a dollar price from a range-market outcome label like '$78k', '78,000'."""
        for digits, suffix in re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*([kK]?)\b', s):
            try:
                v = float(digits.replace(",", ""))
                if suffix.lower() == "k":
                    v *= 1000
                if v > 10_000:
                    return v
            except Exception:
                pass
        return None

    def _near_price_buckets(bkts, cur_p, prob_thresh=0.005, min_keep=3):
        """Drop 0%-probability buckets far from the current price.

        Keeps a bucket if it has meaningful probability (≥0.5%, i.e. shows as
        '1%' or more) OR if its price range actually contains the current price.
        Always returns at least min_keep entries (nearest by price) as a fallback.
        """
        def _lo(lbl):
            nums = [float(x.replace(",", "")) for x in re.findall(r'[\d,]+', lbl)]
            nums = [n for n in nums if n > 10_000]
            return min(nums) if nums else None

        def _hi(lbl):
            nums = [float(x.replace(",", "")) for x in re.findall(r'[\d,]+', lbl)]
            nums = [n for n in nums if n > 10_000]
            return max(nums) if nums else None

        filtered = []
        for b in bkts:
            if b[1] >= prob_thresh:
                filtered.append(b)
                continue
            lo = _lo(b[0])
            if lo is None:
                continue
            # "Above $X" has lo==hi from the regex; treat as open-ended upward
            hi = _hi(b[0])
            bkt_hi = float("inf") if (hi is None or hi == lo) else hi
            if lo <= cur_p <= bkt_hi:
                filtered.append(b)

        if len(filtered) < min_keep and bkts:
            seen = {id(b) for b in filtered}
            by_dist = sorted(bkts, key=lambda b: abs((_lo(b[0]) or cur_p) - cur_p))
            for b in by_dist:
                if id(b) not in seen:
                    filtered.append(b)
                    seen.add(id(b))
                    if len(filtered) >= min_keep:
                        break
            filtered.sort(key=lambda b: _lo(b[0]) or 0, reverse=True)

        return filtered

    def _score_range_market(outcomes, probs, cur_price):
        """
        Score a multi-outcome price range market.
        Returns (score, summary_str, bucket_list) or (None, None, None).
        bucket_list: [(label, prob, is_bull)] sorted by prob desc for display.

        Handles two Polymarket formats:
          A) Explicit ranges: ">$90k", "$80k-$82k", "<$76k"
          B) Ladder floors:  "+$90,000", "+$88,000" ... each label is the FLOOR
             of a $Xk-wide bucket; the upper bound is inferred from the next
             higher level.  This is the format used by "Bitcoin price on May X?"
        """
        def _extract_prices(s):
            out = []
            for digits, suffix in re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*([kK]?)\b', s):
                try:
                    v = float(digits.replace(",", ""))
                    if suffix.lower() == "k": v *= 1000
                    if v > 10_000: out.append(v)
                except Exception: pass
            return out

        # ── Parse all outcomes ────────────────────────────────────────────
        raw = [(str(o).strip(), p) for o, p in zip(outcomes, probs)]

        # Regex that matches ONLY Polymarket's "+$FLOOR" ladder tokens:
        #   "+$80,000"  "+$80k"  "+80000"  (plus-sign, optional $, digits, optional k)
        _floor_re = re.compile(
            r'^\+\s*\$?([\d,]+(?:\.\d+)?)\s*([kK]?)\s*$'
        )

        def _floor_price(s):
            m = _floor_re.match(s)
            if not m:
                return None
            v = float(m.group(1).replace(",", ""))
            if m.group(2).lower() == "k":
                v *= 1000
            return v if v > 10_000 else None

        # Classify each outcome
        _ladder_rungs  = []   # (price, prob, orig_label)  — the +$X floor entries
        _tail_entries  = []   # outcomes NOT matching the +$X pattern
        for s, p in raw:
            fp = _floor_price(s)
            if fp is not None:
                _ladder_rungs.append((fp, p, s))
            else:
                _tail_entries.append((s, p))

        # It's a ladder if at least 2 outcomes are "+$X" floor-format rungs
        _is_ladder = len(_ladder_rungs) >= 2

        bull_prob   = 0.0
        bear_prob   = 0.0
        bucket_list = []

        if _is_ladder:
            # Sort rungs by price descending — highest floor first
            ladder = sorted(_ladder_rungs, key=lambda x: x[0], reverse=True)

            for i, (price, prob, orig_lbl) in enumerate(ladder):
                if i == 0:
                    lo, hi = price, float("inf")
                    label  = f">${price:,.0f}"
                else:
                    hi    = ladder[i - 1][0]
                    lo    = price
                    label = f"${lo:,.0f}–${hi:,.0f}"

                if lo >= cur_price:
                    bull_prob += prob
                    is_bull = True
                elif hi != float("inf") and hi <= cur_price:
                    bear_prob += prob
                    is_bull = False
                else:
                    # Range straddles current price → neutral (grey)
                    hi_eff    = hi if hi != float("inf") else cur_price * 1.5
                    bull_frac = (hi_eff - cur_price) / max(hi_eff - lo, 1)
                    bull_prob += prob * bull_frac
                    bear_prob += prob * (1 - bull_frac)
                    is_bull   = None

                bucket_list.append((label, prob, is_bull))

            # Handle non-ladder tail entries (e.g. "Below $X", "<$X")
            for s, prob in _tail_entries:
                sl  = s.lower()
                pp  = _extract_prices(s)
                ref = pp[0] if pp else cur_price
                if sl.startswith(("<", "below", "under")):
                    is_bull = False if ref <= cur_price else True
                    if is_bull:
                        bull_prob += prob
                    else:
                        bear_prob += prob
                elif sl.startswith((">", "above", "over")):
                    is_bull = True if ref >= cur_price else True
                    bull_prob += prob
                else:
                    is_bull = None
                bucket_list.append((s, prob, is_bull))

        else:
            # ── Explicit range / above / below labels ─────────────────────
            for (s, prob) in raw:
                sl         = s.lower()
                all_prices = _extract_prices(s)
                is_above   = sl.startswith(">") or sl.startswith("above") or sl.startswith("over")
                is_below   = sl.startswith("<") or sl.startswith("below") or sl.startswith("under")
                is_bull    = False

                if is_above:
                    ref = all_prices[0] if all_prices else cur_price
                    if ref >= cur_price:
                        bull_prob += prob; is_bull = True
                    else:
                        bull_prob += prob * 0.3; is_bull = True
                elif is_below:
                    ref = all_prices[0] if all_prices else cur_price
                    bear_prob += prob if ref <= cur_price else prob * 0.3
                elif len(all_prices) >= 2:
                    lo, hi = min(all_prices), max(all_prices)
                    if lo >= cur_price:
                        bull_prob += prob; is_bull = True
                    elif hi <= cur_price:
                        bear_prob += prob
                    else:
                        bf = (hi - cur_price) / max(hi - lo, 1)
                        bull_prob += prob * bf
                        bear_prob += prob * (1 - bf)
                        is_bull = bf >= 0.5
                elif len(all_prices) == 1:
                    if all_prices[0] >= cur_price:
                        bull_prob += prob; is_bull = True
                    else:
                        bear_prob += prob

                bucket_list.append((s, prob, is_bull))

        total = bull_prob + bear_prob
        if total < 0.05:
            return None, None, None

        score   = (bull_prob - bear_prob) / total * 10
        summary = f"↑{bull_prob*100:.0f}% bull / ↓{bear_prob*100:.0f}% bear"
        bucket_list.sort(key=lambda x: x[1], reverse=True)
        return round(score, 2), summary, bucket_list

    def _classify_market(question, outcomes, probs, hours_left, cur_price):
        """
        Returns dict with keys:
          category, weight, individual_score, market_data, rationale,
          display_type, outcomes_display, target_price
        or None to skip.

        display_type: "updown" | "yesno" | "range"
        outcomes_display:
          updown  → [(label, prob, color), ...]
          yesno   → [(label, prob, color), ...]
          range   → [(label, prob, is_bull), ...] (top N by prob)
        """
        q = question.lower()
        n = len(outcomes)

        # ── Multi-outcome price range markets ──────────────────────────────
        if n > 2:
            score, summary, buckets = _score_range_market(outcomes, probs, cur_price)
            if score is None:
                return None
            if hours_left is not None and hours_left <= 72:
                w, rat = 5, "Resolves within 72h — direct bias window"
            elif hours_left is not None and hours_left <= 168:
                w, rat = 3, "This week's price range — partially relevant to 72h bias"
            elif hours_left is not None and hours_left <= 720:
                w, rat = 1, "Monthly price range — marginal relevance to 72h"
            elif hours_left is not None and hours_left > 720:
                return None  # exclude markets resolving >30 days out
            else:
                w, rat = 1, "Price range — unknown horizon"
            return {
                "category": "PRICE_RANGE", "weight": w,
                "individual_score": score, "market_data": summary,
                "rationale": rat, "display_type": "range",
                "outcomes_display": buckets[:6],   # top 6 by prob
                "target_price": None,
            }

        # ── Race-to-target: "Will BTC hit $X or $Y first?" ─────────────────
        # Outcomes: lower price first (bearish) vs higher price first (bullish).
        # Extract both prices from the question, assign direction, score as spread.
        if "first" in q and ("or" in q):
            _rt_prices = []
            for _dg, _sf in re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*([kK]?)\b', question):
                try:
                    _v = float(_dg.replace(",", ""))
                    if _sf.lower() == "k": _v *= 1000
                    if _v > 10_000: _rt_prices.append(_v)
                except Exception: pass
            if len(_rt_prices) >= 2:
                _rt_prices = sorted(set(_rt_prices))
                _lo, _hi = _rt_prices[0], _rt_prices[-1]

                # Match outcome labels to lo/hi price
                _lo_idx, _hi_idx = 0, 1
                for _ri, _ro in enumerate(outcomes):
                    _rol = str(_ro).lower().replace(",", "")
                    if (str(int(_lo)) in _rol or
                            f"{_lo/1000:.0f}k" in _rol or
                            f"${_lo:,.0f}".lower().replace(",", "") in _rol):
                        _lo_idx = _ri
                        _hi_idx = 1 - _ri
                        break

                _lo_p = probs[_lo_idx]   # prob lower target hits first (bearish)
                _hi_p = probs[_hi_idx]   # prob higher target hits first (bullish)
                score  = round((_hi_p - _lo_p) * 10, 2)

                _lo_pct = abs(_lo - cur_price) / cur_price
                _hi_pct = abs(_hi - cur_price) / cur_price
                _avg_d  = (_lo_pct + _hi_pct) / 2
                if _avg_d < 0.15:   w = 4
                elif _avg_d < 0.30: w = 3
                else:               w = 2

                rat  = (f"Race: ↑${_hi:,.0f} ({_hi_p:.0%}) vs"
                        f" ↓${_lo:,.0f} ({_lo_p:.0%})")
                mdata = f"↑${_hi:,.0f} first: {_hi_p:.0%}"
                od = [
                    (f"↑${_hi:,.0f} first", _hi_p,
                     "#3fb950" if _hi_p >= 0.5 else "#f85149"),
                    (f"↓${_lo:,.0f} first", _lo_p,
                     "#f85149" if _lo_p >= 0.5 else "#3fb950"),
                ]
                return {
                    "category": "RACE_TARGET", "weight": w,
                    "individual_score": score, "market_data": mdata,
                    "rationale": rat, "display_type": "updown",
                    "outcomes_display": od, "target_price": None,
                }

        # ── Binary markets ──────────────────────────────────────────────────
        # Find the bullish outcome index
        bullish_words = [
            "up", "yes", "higher", "above",
            "over", "bull", "rise"
        ]
        
        bearish_words = [
            "down", "no", "lower", "below",
            "under", "bear", "fall"
        ]
        
        up_idx = 0
        
        for i, o in enumerate(outcomes):
            ol = str(o).lower()
        
            if any(w in ol for w in bullish_words):
                up_idx = i
                break
        dn_idx  = 1 - up_idx
        up_prob = probs[up_idx]
        dn_prob = probs[dn_idx]
        up_lbl  = str(outcomes[up_idx])
        dn_lbl  = str(outcomes[dn_idx])

        # ── A. Up/down directional questions ──────────────────────────────
        if "up or down" in q or "up/down" in q or "higher or lower" in q:
            score = round((up_prob - 0.5) * 20, 2)
            mdata = f"{up_prob:.0%} {up_lbl} / {dn_prob:.0%} {dn_lbl}"
            od    = [(f"{up_lbl} ↑", up_prob, "#3fb950"),
                     (f"{dn_lbl} ↓", dn_prob, "#f85149")]
            if any(x in q for x in ["5 min", "5m", "5-min", "five min"]):
                cat, w, rat = "ULTRA_SHORT", 1, "5-min micro-timeframe (noise)"
            elif any(x in q for x in ["1 hour", "1h ", "1-hour", "one hour"]):
                cat, w, rat = "SHORT_DIR", 2, "1h intraday directional"
            elif any(x in q for x in ["4 hour", "4h ", "4-hour", "four hour"]):
                cat, w, rat = "MID_DIR", 2, "4h directional — meaningful intraday signal"
            elif any(x in q for x in ["today", "daily", "24 hour", "24h ", "this day", "end of day"]):
                cat, w, rat = "INTRADAY_DIR", 3, "Daily directional — direct 24h thesis"
            else:
                if hours_left is not None:
                    if hours_left <= 6:
                        cat, w, rat = "SHORT_DIR", 2, f"Short-term directional ({hours_left:.0f}h)"
                    elif hours_left <= 72:
                        cat, w, rat = "INTRADAY_DIR", 3, f"72h directional ({hours_left:.0f}h window)"
                    else:
                        cat, w, rat = "MID_DIR", 2, "Generic directional"
                else:
                    cat, w, rat = "SHORT_DIR", 1, "Unknown timeframe directional"
            return {
                "category": cat, "weight": w, "individual_score": score,
                "market_data": mdata, "rationale": rat,
                "display_type": "updown", "outcomes_display": od, "target_price": None,
            }

        # ── B. Price level (above/below $X) questions ─────────────────────
        target  = _extract_price(question)
        is_bull = any(kw in q for kw in ["above", "over $", "exceed", "reach",
                                          "rally", "higher", "btc hit", "bitcoin hit",
                                          "at least", "or higher", "stays above", "close above",
                                          "end above", "trade above", "go above"])
        is_bear = any(kw in q for kw in ["below", "under $", "drop", "fall", "crash",
                                          "decline", "lose", "break below", "close below"])

        if is_bull and target:
            pct_diff = (target - cur_price) / cur_price
            t_fmt    = f"${target:,.0f}"

            if pct_diff > 0.15:
                score = round((up_prob - 0.5) * 20, 2)
                cat, w, rat = ("ABOVE_TARGET_LONG", 1,
                               f"Reach {t_fmt} (+{pct_diff:.0%}) — long-term bull target")
            elif pct_diff > 0.03:
                score = round((up_prob - 0.5) * 20, 2)
                if hours_left is not None and hours_left <= 168:
                    cat, w, rat = ("ABOVE_TARGET_WEEK", 3,
                                   f"Hit {t_fmt} (+{pct_diff:.0%}) this week — directional bet")
                else:
                    cat, w, rat = ("ABOVE_TARGET_LONG", 1,
                                   f"Hit {t_fmt} (+{pct_diff:.0%}) — longer-term target")
            elif pct_diff > -0.05:
                score = round((up_prob - 0.5) * 15, 2)
                cat, w, rat = ("NEAR_LEVEL", 3,
                               f"Hold {t_fmt} near-level (±{abs(pct_diff):.0%} from current)")
            elif pct_diff > -0.15:
                score = round((up_prob - 0.5) * 10, 2)
                cat, w, rat = ("ABOVE_FLOOR", 1,
                               f"{t_fmt} floor ({abs(pct_diff):.0%} below) — not a pump signal")
            else:
                score = round((up_prob - 0.5) * 5, 2)
                cat, w, rat = ("TRIVIAL_FLOOR", 1,
                               f"{t_fmt} trivial floor ({abs(pct_diff):.0%} below)")

            mdata = f"{up_prob:.0%} Yes (≥{t_fmt})"
            od    = [(f"Yes ≥{t_fmt}", up_prob, "#3fb950" if score > 0 else "#f85149"),
                     (f"No  <{t_fmt}", dn_prob, "#f85149" if score > 0 else "#3fb950")]
            return {
                "category": cat, "weight": w, "individual_score": score,
                "market_data": mdata, "rationale": rat,
                "display_type": "yesno", "outcomes_display": od, "target_price": target,
            }

        elif is_bear and target:
            pct_diff = (cur_price - target) / cur_price
            t_fmt    = f"${target:,.0f}"
            score    = round(-(up_prob - 0.5) * 20, 2)

            if pct_diff > 0.05:
                cat, w, rat = ("BELOW_TARGET", 3,
                               f"Drop to {t_fmt} ({pct_diff:.0%} down) — downside target")
            else:
                cat, w, rat = ("BELOW_NEAR", 2,
                               f"Break {t_fmt} near-support — key bear trigger")

            mdata = f"{up_prob:.0%} Yes (≤{t_fmt})"
            od    = [(f"Yes ≤{t_fmt}", up_prob, "#f85149" if score < 0 else "#3fb950"),
                     (f"No  >{t_fmt}", dn_prob, "#3fb950" if score < 0 else "#f85149")]
            return {
                "category": cat, "weight": w, "individual_score": score,
                "market_data": mdata, "rationale": rat,
                "display_type": "yesno", "outcomes_display": od, "target_price": target,
            }

        return None  # unclassifiable

    def _make_entry(display_q, score, w, rat, dtype, od, tprice, liq, hl, mdata):
        """Build a scored market dict for the final list."""
        return {
            "question":         display_q,
            "market_data":      mdata,
            "individual_score": round(score, 2),
            "weight":           w,
            "weighted_score":   round(score * w, 2),
            "rationale":        rat,
            "category":         dtype.upper(),
            "display_type":     dtype,
            "outcomes_display": od,
            "target_price":     tprice,
            "liquidity":        liq,
            "hours_left":       round(hl, 1) if hl is not None else None,
        }

    try:
        # ascending=true (oldest first) surfaces price-target and perpetual Up/Down
        # markets which have old startDates.  ascending=false floods the list with
        # freshly-created 15-min time-slot markets (new one every 15 min) and none
        # of the useful markets appear within the first 200 results.
        events = (
            _get("https://gamma-api.polymarket.com/events"
                 "?tag_slug=bitcoin&active=true&closed=false"
                 "&limit=500&order=startDate&ascending=true")
            or []
        )

        scored_markets = []

        for ev in events:
            try:
                event_title = (ev.get("title") or "").strip()
                ev_mkts     = _parse_list(ev.get("markets", []))

                # ── Skip event types that don't fit directional scoring ────────
                # 1. 5-min time-slot noise: "Bitcoin Up or Down - May 9, 8:10AM-8:15AM ET"
                if re.search(
                    r'Bitcoin Up or Down\s*-\s*.*\d{1,2}(:\d{2})?\s*[AaPp][Mm]',
                    event_title,
                    re.IGNORECASE
                ):
                    continue

                # ── Parse all qualifying sub-markets for this event ────────────
                sub_mkts = []
                for m in ev_mkts:
                    if not isinstance(m, dict):
                        continue
                    outcomes_m   = _parse_list(m.get("outcomes"))
                    out_prices_m = _parse_list(m.get("outcomePrices"))
                    liq_m        = float(m.get("liquidityNum") or m.get("liquidity") or 0)
                    end_str_m    = m.get("endDateIso") or m.get("endDate") or ""

                    if len(outcomes_m) < 2 or len(out_prices_m) < 2 or liq_m < 200:
                        continue

                    probs_m = [float(p) for p in out_prices_m]
                    hl_m    = None
                    if end_str_m:
                        try:
                            # Polymarket returns date-only strings like "2026-05-10" for
                            # endDateIso. fromisoformat() produces a naive datetime which
                            # can't be subtracted from the timezone-aware `now`. Append
                            # end-of-day UTC so the comparison works.
                            if "T" not in end_str_m:
                                _end_norm = end_str_m + "T23:59:59+00:00"
                            else:
                                _end_norm = end_str_m.replace("Z", "+00:00")
                            end_dt_m = datetime.fromisoformat(_end_norm)
                            hl_m     = (end_dt_m - now).total_seconds() / 3600
                            if hl_m < 0:
                                continue
                        except Exception:
                            pass

                    sub_mkts.append({
                        "question":  (m.get("question") or "").strip(),
                        "outcomes":  outcomes_m,
                        "probs":     probs_m,
                        "liquidity": liq_m,
                        "hours_left": hl_m,
                    })

                if not sub_mkts:
                    continue

                all_hl    = [sm["hours_left"] for sm in sub_mkts if sm["hours_left"] is not None]
                evt_hours = min(all_hl) if all_hl else None
                evt_liq   = sum(sm["liquidity"] for sm in sub_mkts)
                evt_name  = event_title or sub_mkts[0]["question"]

                # ── Skip events whose referenced date has already passed ───────
                # "What price will Bitcoin hit on May 10?" asked on May 11 has a
                # settlement date of May 11 on Polymarket, so hl_m is still +20h.
                # We detect the date IN THE TITLE and skip if it's before today.
                # Find ALL dates mentioned in the title; use the LATEST one.
                # "May 11-17" should stay visible until May 17 has passed.
                _date_pat = re.compile(
                    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
                    r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|'
                    r'nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})\b',
                    re.IGNORECASE)
                _all_ref_dates = []
                for _dm in _date_pat.finditer(evt_name):
                    try:
                        _all_ref_dates.append(datetime.strptime(
                            f"{_dm.group(0)} {now.year}", "%B %d %Y"
                        ).replace(tzinfo=timezone.utc))
                    except Exception:
                        pass
                # Also catch trailing day numbers in ranges like "May 11-17"
                for _dm2 in re.finditer(r'-(\d{1,2})\b', evt_name):
                    try:
                        # Pair with the month from the first found date
                        if _all_ref_dates:
                            _mo = _all_ref_dates[0].month
                            _yr = now.year
                            _all_ref_dates.append(datetime(
                                _yr, _mo, int(_dm2.group(1)), tzinfo=timezone.utc))
                    except Exception:
                        pass
                if _all_ref_dates:
                    _latest_ref = max(_all_ref_dates)
                    if _latest_ref.date() < now.date():
                        continue

                # ══════════════════════════════════════════════════════════════
                # CASE W: "When will Bitcoin hit $X?" — timeframe sub-markets
                # Sub-markets ask "by Q3 2025?", "before June?" etc. — not prices.
                # Scoring: nearest timeframe's YES probability is the signal.
                # High prob soon → bullish; low prob across all dates → bearish.
                # ══════════════════════════════════════════════════════════════
                if re.search(r'\bwhen will\b', evt_name, re.IGNORECASE):
                    _ww_target = _extract_price(evt_name)
                    if _ww_target is None:
                        continue
                    _ww_pct = (_ww_target - current_price) / current_price

                    # Collect (hours_left, yes_prob, display_label) per sub-market
                    _ww_subs = []
                    for _sm in sub_mkts:
                        _up_i = 0
                        for _i, _o in enumerate(_sm["outcomes"]):
                            if str(_o).lower() == "yes":
                                _up_i = _i; break
                        _yp = _sm["probs"][_up_i]
                        _hl = _sm.get("hours_left")
                        if _hl is None or _hl <= 0:
                            continue
                        # Use the end date as the label — sub-market questions like
                        # "Will Bitcoin hit $150k by June 30, 2026?" don't share
                        # a strippable prefix with the parent event title.
                        from datetime import timedelta as _td
                        _end_dt = now + _td(hours=_hl)
                        _lbl    = _end_dt.strftime("By %b %d, %Y")
                        _ww_subs.append((_hl, _yp, _lbl))

                    if not _ww_subs:
                        continue

                    _ww_subs.sort(key=lambda x: x[0])   # nearest date first
                    _near_hl, _near_p, _near_lbl = _ww_subs[0]

                    # "When will" markets are context-only — not scored, not
                    # weighted, displayed grey so they don't mislead the thesis.
                    rat = (f"${_ww_target:,.0f} (+{_ww_pct:.0%}); "
                           f"{_near_lbl}: {_near_p:.0%}")

                    # All buckets grey (is_bull=None) — timeframe labels only
                    _ww_buckets = [
                        (_lbl, _yp, None)
                        for _, _yp, _lbl in _ww_subs[:6]
                    ]
                    scored_markets.append(_make_entry(
                        evt_name, 0, 0, rat, "range", _ww_buckets,
                        _ww_target, evt_liq, _near_hl,
                        f"{_near_lbl}: {_near_p:.0%}"))
                    continue

                # ══════════════════════════════════════════════════════════════
                # CASE A: Single sub-market → use existing classify logic
                # ══════════════════════════════════════════════════════════════
                if len(sub_mkts) == 1:
                    sm = sub_mkts[0]
                    if len(sm["outcomes"]) > 2:
                        # Multi-outcome range market (e.g. "Bitcoin price on May 8?")
                        score, summary, buckets = _score_range_market(
                            sm["outcomes"], sm["probs"], current_price)
                        if score is None:
                            continue
                        if evt_hours is not None and evt_hours <= 48:
                            w, rat = 4, "Today/tomorrow price range — hard directional stake"
                        elif evt_hours is not None and evt_hours <= 168:
                            w, rat = 4, "This week's price range — hard directional stake"
                        else:
                            w, rat = 2, "Monthly price range"
                        scored_markets.append(_make_entry(
                            evt_name, score, w, rat, "range", buckets[:6],
                            None, evt_liq, evt_hours, summary))
                    else:
                        # Single binary market
                        cls = _classify_market(sm["question"], sm["outcomes"],
                                               sm["probs"], sm["hours_left"], current_price)
                        if cls is None:
                            continue
                        scored_markets.append({
                            "question":         evt_name,
                            "market_data":      cls["market_data"],
                            "individual_score": cls["individual_score"],
                            "weight":           cls["weight"],
                            "weighted_score":   round(cls["individual_score"] * cls["weight"], 2),
                            "rationale":        cls["rationale"],
                            "category":         cls["category"],
                            "display_type":     cls["display_type"],
                            "outcomes_display": cls["outcomes_display"],
                            "target_price":     cls["target_price"],
                            "liquidity":        evt_liq,
                            "hours_left":       round(evt_hours, 1) if evt_hours is not None else None,
                        })

                # ══════════════════════════════════════════════════════════════
                # CASE B: Multiple sub-markets in one event
                # Two event formats land here:
                #   • "What price will Bitcoin hit May 4-10?"
                #     → sub-markets: ↓78k (36%), ↑84k (8%) — compare bull vs bear
                #   • "Bitcoin above ___ on May 8?"
                #     → sub-markets: above 68k (100%), above 70k (100%) — floor ladder
                #   • "Bitcoin price on May 11?" (bucket-floor ladder)
                #     → sub-markets: "+$80,000" YES=94%, "+$82,000" YES=4% etc.
                #     Each question IS the "+$X" floor label, not a normal question.
                # ══════════════════════════════════════════════════════════════
                else:
                    # ── Range-bucket detection ────────────────────────────────────
                    # "Bitcoin price on May X?" has sub-markets like:
                    #   "Will the price of Bitcoin be between $80,000 and $82,000 on May 11?"
                    #   "Will the price of Bitcoin be less than $72,000 on May 11?"
                    #   "Will the price of Bitcoin be greater than $90,000 on May 11?"
                    # Each is binary YES/NO.  We extract (lo, hi) per sub-market and
                    # classify each bucket relative to current_price.
                    _bet_re = re.compile(
                        r'between\s+\$?([\d,]+)\s+and\s+\$?([\d,]+)', re.IGNORECASE)
                    _lt_re  = re.compile(r'less than\s+\$?([\d,]+)', re.IGNORECASE)
                    _gt_re  = re.compile(r'greater than\s+\$?([\d,]+)', re.IGNORECASE)

                    _rbkts_raw = []   # (lo, hi, yes_prob)
                    for _rsm in sub_mkts:
                        _rq = _rsm["question"]
                        _up_i = 0
                        for _ri, _ro in enumerate(_rsm["outcomes"]):
                            if str(_ro).lower() == "yes": _up_i = _ri; break
                        _ryp = _rsm["probs"][_up_i]

                        _m = _bet_re.search(_rq)
                        if _m:
                            _rlo = float(_m.group(1).replace(",", ""))
                            _rhi = float(_m.group(2).replace(",", ""))
                            if _rlo > 10_000 and _rhi > 10_000:
                                _rbkts_raw.append((_rlo, _rhi, _ryp)); continue
                        _m = _lt_re.search(_rq)
                        if _m:
                            _rv = float(_m.group(1).replace(",", ""))
                            if _rv > 10_000:
                                _rbkts_raw.append((0.0, _rv, _ryp)); continue
                        _m = _gt_re.search(_rq)
                        if _m:
                            _rv = float(_m.group(1).replace(",", ""))
                            if _rv > 10_000:
                                _rbkts_raw.append((_rv, float("inf"), _ryp)); continue

                    if len(_rbkts_raw) >= 3:
                        _rbkts_raw.sort(key=lambda x: x[0], reverse=True)
                        _rb = 0.0; _rbr = 0.0; _rbkts = []
                        for _rlo, _rhi, _ryp in _rbkts_raw:
                            if _rhi == float("inf"):
                                _rlbl = f">${_rlo:,.0f}"
                            elif _rlo == 0.0:
                                _rlbl = f"<${_rhi:,.0f}"
                            else:
                                _rlbl = f"${_rlo:,.0f}–${_rhi:,.0f}"
                            if _rlo >= current_price:
                                _rb += _ryp; _r_ib = True
                            elif _rhi != float("inf") and _rhi <= current_price:
                                _rbr += _ryp; _r_ib = False
                            else:
                                _r_hie = _rhi if _rhi != float("inf") else current_price * 1.5
                                _r_bf  = (_r_hie - current_price) / max(_r_hie - _rlo, 1)
                                _rb   += _ryp * _r_bf
                                _rbr  += _ryp * (1 - _r_bf)
                                _r_ib  = None
                            _rbkts.append((_rlbl, _ryp, _r_ib))
                        _rt = _rb + _rbr
                        if _rt >= 0.05:
                            _rs = round((_rb - _rbr) / _rt * 10, 2)
                            # sort by lower-bound price descending (highest range first)
                            def _rbkt_price(item):
                                import re as _re2
                                m2 = _re2.search(r'\$([\d,]+)', item[0])
                                return float(m2.group(1).replace(',','')) if m2 else 0
                            _rbkts.sort(key=_rbkt_price, reverse=True)
                            _rsum = f"↑{_rb*100:.0f}% bull / ↓{_rbr*100:.0f}% bear"
                            if evt_hours is not None and evt_hours <= 48:
                                _rw, _rrat = 4, "Today directional price range"
                            elif evt_hours is not None and evt_hours <= 168:
                                _rw, _rrat = 3, "This week price range"
                            else:
                                _rw, _rrat = 2, "Monthly price range"
                            scored_markets.append(_make_entry(
                                evt_name, _rs, _rw, _rrat, "range",
                                _near_price_buckets(_rbkts, current_price)[:8], None, evt_liq, evt_hours, _rsum))
                        continue   # handled as range buckets

                    bull_hits = []  # (target_price, yes_prob) for bullish targets
                    bear_hits = []  # (target_price, yes_prob) for bearish targets

                    for sm in sub_mkts:
                        q_sm = sm["question"].lower()
                        # Find YES probability
                        up_idx_sm = 0
                        for i, o in enumerate(sm["outcomes"]):
                            if str(o).lower() in ("up", "yes", "higher", "above"):
                                up_idx_sm = i
                                break
                        yes_prob = sm["probs"][up_idx_sm]
                        target   = _extract_price(sm["question"])
                        if target is None:
                            continue

                        pct     = (target - current_price) / current_price
                        is_bull_kw = any(kw in q_sm for kw in
                                         ["above", "exceed", "reach", "hit", "higher", "rally"])
                        is_bear_kw = any(kw in q_sm for kw in
                                         ["below", "drop", "fall", "crash", "under"])

                        # Classify based on keyword + price position
                        if is_bear_kw or (not is_bull_kw and pct < 0):
                            # Bear-framed: YES = bearish outcome
                            # Convert to directional probability
                            bear_hits.append((target, yes_prob))
                        else:
                            # Bull-framed or above-current target
                            if pct > 0:
                                # Target is above current — directional bull signal
                                bull_hits.append((target, yes_prob))
                            else:
                                # Floor below current — filter trivial 100% ones,
                                # use as floor support only if not fully resolved
                                if yes_prob < 0.98:
                                    bull_hits.append((target, yes_prob))
                                # else skip trivially-resolved floors

                    # Split bull_hits into targets ABOVE vs BELOW current price.
                    # Below-current targets are already exceeded floors — they tell
                    # us nothing about direction and should not inflate p_bull.
                    # Deduplicate by price target: keep highest probability per price
                    # (handles "When will BTC hit $150k?" sub-markets Q2/Q3/etc. that
                    # all extract to the same price).
                    _seen_t: dict = {}
                    for _t, _p in bull_hits:
                        if _t not in _seen_t or _p > _seen_t[_t]:
                            _seen_t[_t] = _p
                    bull_hits = list(_seen_t.items())

                    bull_dir   = [(t, p) for t, p in bull_hits if t > current_price]
                    bull_floor = [(t, p) for t, p in bull_hits if t <= current_price]

                    p_bull = sum(p for _, p in bull_dir)   # directional only
                    p_bear = sum(p for _, p in bear_hits)
                    total  = p_bull + p_bear

                    # ── Floor ladder: ALL bull targets below current (pure support) ──
                    _is_floor_ladder = (
                        bull_hits and not bear_hits and not bull_dir
                    )

                    # ── Ceiling/mixed: targets above current price exist (no bear) ──
                    # e.g. "Bitcoin above $80k/$82k/$84k on May 11?" at $81.2k.
                    # The nearest above-current threshold's probability is the signal:
                    # 22% chance of clearing $82k → bearish (market not expecting breakout).
                    _is_ceiling = bool(bull_dir) and not bear_hits

                    if _is_floor_ladder:
                        _top_t, _top_p = max(bull_hits, key=lambda x: x[0])
                        _gap = abs((_top_t - current_price) / current_price)
                        score = round((_top_p - 0.5) * 20, 2)
                        if _gap > 0.15:
                            w, rat = 1, f"Trivial floor ladder (>{_gap:.0%} below current)"
                        elif _gap > 0.05:
                            w, rat = 2, f"Floor support — highest contested ${_top_t:,.0f} ({_gap:.0%} below)"
                        else:
                            w, rat = 3, f"Near-level floor — ${_top_t:,.0f} closely contested"

                    elif _is_ceiling or bull_dir:
                        # Score by absolute probability of the nearest above-current target.
                        # 6% chance of hitting $83k today → (6%-50%)*20 = -8.8 (bearish).
                        # Applies to both pure ceiling markets AND reach/hit markets where
                        # resolved (already-hit) sub-markets drop out, leaving only the
                        # unresolved above-current targets with their raw Polymarket probs.
                        _near_t, _near_p = min(bull_dir, key=lambda x: x[0])
                        _pct_above = (_near_t - current_price) / current_price
                        score = round((_near_p - 0.5) * 20, 2)
                        if evt_hours is not None and evt_hours <= 48:
                            w, rat = 3, (f"Nearest resistance ${_near_t:,.0f}"
                                         f" (+{_pct_above:.1%}): {_near_p:.0%} prob today")
                        elif evt_hours is not None and evt_hours <= 168:
                            w, rat = 2, (f"Week resistance ${_near_t:,.0f}"
                                         f" (+{_pct_above:.1%}): {_near_p:.0%} prob")
                        else:
                            w, rat = 1, (f"Resistance ${_near_t:,.0f}"
                                         f" (+{_pct_above:.1%}): {_near_p:.0%} prob")

                    else:
                        # Bear-only targets (no above-current bull direction)
                        if total < 0.02:
                            continue
                        score = (p_bull - p_bear) / max(total, 0.01) * 10
                        if evt_hours is not None and evt_hours <= 48:
                            w, rat = 4, "Today directional price targets"
                        elif evt_hours is not None and evt_hours <= 168:
                            w, rat = 3, "Week directional price targets"
                        elif evt_hours is not None and evt_hours <= 744:
                            w, rat = 2, "Monthly directional price targets"
                        else:
                            w, rat = 1, "Long-term directional targets"

                    if abs(score) < 0.2:
                        continue

                    # Build bucket display sorted by price descending.
                    # For "above" markets show every option exactly as Polymarket does.
                    _is_above_mkt = all("above" in _dsm["question"].lower() for _dsm in sub_mkts)
                    if _is_above_mkt:
                        buckets = sorted(
                            [(f"Above ${t:,.0f}", p, True)  for t, p in bull_dir]
                          + [(f"Above ${t:,.0f}", p, True)  for t, p in bull_floor]
                          + [(f"Below ${t:,.0f}", p, False) for t, p in bear_hits],
                            key=lambda x: float(x[0].replace("Above $","").replace("Below $","").replace(",","")),
                            reverse=True
                        )
                    else:
                        buckets = sorted(
                            [(f"↑${t:,.0f}", p, True)  for t, p in bull_dir]
                          + [(f"↓${t:,.0f}", p, False)  for t, p in bear_hits]
                          + [(f"~${t:,.0f}", p, None)   for t, p in bull_floor],
                            key=lambda x: float(x[0].replace("↑$","").replace("↓$","").replace("~$","").replace(",","")),
                            reverse=True
                        )

                    if bull_dir and bear_hits:
                        summary = f"↑{p_bull:.0%} above vs ↓{p_bear:.0%} below"
                    elif bull_dir:
                        _near_t2, _near_p2 = min(bull_dir, key=lambda x: x[0])
                        summary = f"Nearest resistance ${_near_t2:,.0f}: {_near_p2:.0%}"
                    elif bull_floor:
                        summary = f"Floor support only"
                    else:
                        summary = f"↓{p_bear:.0%} bear targets"

                    scored_markets.append(_make_entry(
                        evt_name, score, w, rat, "range",
                        _near_price_buckets(buckets, current_price),
                        None, evt_liq, evt_hours, summary))

            except Exception:
                continue

        if not scored_markets:
            result["detail"] = (
                f"No qualifying BTC markets found on Polymarket "
                f"({len(events)} events fetched)"
            )
            return result

        # ── Horizon-split aggregation ──────────────────────────────────────────
        # Same question list, two different horizon multipliers so each engine
        # gets a signal calibrated to its window:
        #
        # 72h engine: wants markets resolving in 24–168h (tomorrow / this week).
        #             Same-day markets (≤24h) resolve before the window matters → discounted.
        # 24h engine: wants markets resolving today (≤24h). Longer-dated markets
        #             are forward of the window → discounted.

        def _horizon_mul(hl, window):
            """Return weight multiplier for a market given hours_left and target window."""
            if hl is None:
                return 0.5
            if window == 24:
                if hl <= 24:   return 1.5   # primary window
                if hl <= 48:   return 0.6
                return 0.15                  # too far out for 24h
            else:  # 72h
                if hl <= 24:   return 0.2   # resolves before half the window
                if hl <= 72:   return 1.5   # sweet spot
                if hl <= 168:  return 1.0   # still relevant
                return 0.2                   # too far out for 72h

        def _aggregate(markets, window):
            ws = sum(m["weight"] * _horizon_mul(m.get("hours_left"), window) for m in markets)
            if ws == 0:
                return 0.0, 0.0, 0
            wsum = sum(m["individual_score"] * m["weight"] * _horizon_mul(m.get("hours_left"), window)
                       for m in markets)
            ts   = max(-10.0, min(10.0, wsum / ws))
            wagreeing = sum(
                m["weight"] * _horizon_mul(m.get("hours_left"), window)
                for m in markets
                if (m["individual_score"] >= 0) == (ts >= 0)
            )
            return ts, wagreeing / ws, len(markets)

        thesis_score,  confidence,  n = _aggregate(scored_markets, 72)
        thesis_score24, conf24,    _  = _aggregate(scored_markets, 24)
        signal    = thesis_score   / 10.0
        signal_24 = thesis_score24 / 10.0

        total_weight = sum(m["weight"] for m in scored_markets)
        weighted_agreement = confidence * total_weight  # approximate for legacy compat

        scored_markets.sort(key=lambda x: abs(x["weighted_score"]), reverse=True)

        if   thesis_score >=  6.0: thesis_label = "STRONGLY BULLISH"
        elif thesis_score >=  3.0: thesis_label = "BULLISH"
        elif thesis_score >=  1.0: thesis_label = "MILDLY BULLISH"
        elif thesis_score >= -1.0: thesis_label = "NEUTRAL"
        elif thesis_score >= -3.0: thesis_label = "MILDLY BEARISH"
        elif thesis_score >= -6.0: thesis_label = "BEARISH"
        else:                      thesis_label = "STRONGLY BEARISH"

        result.update({
            "signal":       round(signal, 3),
            "signal_24h":   round(signal_24, 3),
            "confidence":   round(confidence, 2),
            "markets_used": n,
            "markets":      scored_markets,
            "thesis_score": round(thesis_score, 2),
            "thesis_label": thesis_label,
            "detail":       (f"72h Thesis: {thesis_label} ({thesis_score:+.2f}/10) "
                             f"| {n} mkts, {confidence:.0%} agree"),
        })

        # ── AI extraction enhancement ──────────────────────────────────────────
        # Formats already-parsed markets as normalized strings, sends to Claude,
        # and overlays AI-derived directional scores onto the rule-based result.
        # Falls back silently to rule-based output if the API key is absent or the
        # call fails — the result dict is already populated above.
        try:
            _api_key = ""
            try:
                _api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
            except Exception:
                pass
            if not _api_key:
                _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            if False:  # AI per-market override disabled — use Ask Claude button instead
                _ai_strings = _scored_markets_to_ai_strings(scored_markets)
                if _ai_strings:
                    _ai_json = _call_polymarket_ai(_ai_strings, _api_key)
                    if (isinstance(_ai_json, dict)
                            and _ai_json.get("status") != "unparseable"):
                        _ai_mkts = _ai_json.get("markets", [])
                        _agg     = _ai_json.get("aggregate", {})
                        if _ai_mkts and _agg:
                            for _i, _am in enumerate(_ai_mkts[:len(scored_markets)]):
                                _sm   = scored_markets[_i]
                                _bp   = float(_am.get("bullish_probability", 50))
                                _ai_w = float(_am.get("weight", 0.5))
                                _sc   = round((_bp - 50) * 0.2, 2)
                                _dw   = max(1, min(4, round(_ai_w * 4)))
                                _sm["individual_score"] = _sc
                                _sm["weight"]           = _dw
                                _sm["weighted_score"]   = round(_sc * _dw, 2)
                                _sm["rationale"] = (
                                    _sm["rationale"]
                                    + f"  [AI: {str(_am.get('bias','?')).upper()}]"
                                )
                            scored_markets.sort(
                                key=lambda x: abs(x["weighted_score"]), reverse=True)

                            # Compute new aggregate from AI output
                            _net     = float(_agg.get("net_sentiment", 0.0))
                            _new_ts  = max(-10.0, min(10.0, _net / 10.0))
                            _new_sig = max(-1.0,  min(1.0,  _net / 100.0))
                            _tw      = sum(float(_am.get("weight", 0.5)) for _am in _ai_mkts)
                            _wc      = sum(
                                float(_am.get("confidence", 0.5)) * float(_am.get("weight", 0.5))
                                for _am in _ai_mkts
                            )
                            _new_conf = round(_wc / max(0.001, _tw), 2)

                            if   _new_ts >= 6.0:   _new_lbl = "STRONGLY BULLISH"
                            elif _new_ts >= 3.0:   _new_lbl = "BULLISH"
                            elif _new_ts >= 1.0:   _new_lbl = "MILDLY BULLISH"
                            elif _new_ts >= -1.0:  _new_lbl = "NEUTRAL"
                            elif _new_ts >= -3.0:  _new_lbl = "MILDLY BEARISH"
                            elif _new_ts >= -6.0:  _new_lbl = "BEARISH"
                            else:                  _new_lbl = "STRONGLY BEARISH"

                            result.update({
                                "signal":       round(_new_sig, 3),
                                "confidence":   _new_conf,
                                "markets":      scored_markets,
                                "thesis_score": round(_new_ts, 2),
                                "thesis_label": _new_lbl,
                                "detail": (
                                    f"AI 24h Thesis: {_new_lbl} ({_new_ts:+.2f}/10)"
                                    f" | {n} mkts, {_new_conf:.0%} agree"
                                ),
                                "ai_aggregate": {
                                    "bullish_score": float(_agg.get("bullish_score", 0)),
                                    "bearish_score": float(_agg.get("bearish_score", 0)),
                                    "net_sentiment": _net,
                                    "overall_bias":  _agg.get("overall_bias", "neutral"),
                                },
                            })
        except Exception:
            pass  # AI enhancement is optional; rule-based result already in place

    except Exception as exc:
        result["detail"] = f"Polymarket error: {str(exc)[:80]}"

    return result


@st.cache_data(ttl=120, show_spinner=False)
def fetch_btc_liquidity_cached(current_price: float) -> dict:
    """Thin wrapper so we can cache with the price rounded to nearest $100."""
    return _fetch_btc_liquidity(current_price)


# ════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════

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


def nearest_level(levels: List[float], price: float, kind: str) -> Optional[float]:
    if not levels: return None
    relevant = [l for l in levels if abs(l - price) / price < 0.15]
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


# ════════════════════════════════════════════════════════════════
#  CYCLE PHASE DETECTOR
# ════════════════════════════════════════════════════════════════

def detect_btc_cycle_phase(closes, rsi_series, crypto_sig, obv_trend, div_rsi) -> dict:
    signals = {}
    total   = 0
    price   = closes[-1] if closes else 0

    # 1. Fear & Greed
    fgi = crypto_sig.get("fear_greed_value", "N/A")
    try:
        fgi = int(fgi)
        if fgi <= 10:  s, e = +3, f"Extreme Fear {fgi} - capitulation zone"
        elif fgi <= 20: s, e = +2, f"Extreme Fear {fgi} - strong contrarian buy"
        elif fgi <= 30: s, e = +1, f"Fear {fgi} - below average sentiment"
        elif fgi <= 55: s, e =  0, f"Neutral {fgi}"
        elif fgi <= 70: s, e = -1, f"Greed {fgi} - elevated sentiment"
        elif fgi <= 85: s, e = -2, f"Greed {fgi} - high greed, distribution likely"
        else:           s, e = -3, f"Extreme Greed {fgi} - euphoria zone"
    except Exception:
        s, e = 0, "Fear & Greed N/A"
    signals["Fear & Greed"] = (s, e); total += s

    # 2. ETF Flow
    etf_trend = crypto_sig.get("etf_flow_trend", "N/A")
    fs        = crypto_sig.get("etf_flow_stats", {})
    n_in  = fs.get("n_inflow",  0)
    n_out = fs.get("n_outflow", 0)
    n_tot = fs.get("n_etfs",    0)
    accel = fs.get("accelerating", False)
    if etf_trend == "Positive":
        s = +3 if accel else +2
        e = f"ETF inflows {'accelerating' if accel else 'positive'} - {n_in}/{n_tot} ETFs buying"
    elif etf_trend == "Negative":
        s = -3 if accel else -2
        e = f"ETF outflows {'accelerating' if accel else 'negative'} - {n_out}/{n_tot} ETFs selling"
    elif etf_trend == "Mixed":
        s, e = -1, f"Mixed ETF flows - {n_in} in / {n_out} out"
    else:
        s, e =  0, "ETF flows neutral / unavailable"
    signals["ETF Flows"] = (s, e); total += s

    # 3. RSI + Divergence
    valid_rsi = rsi_series.dropna()
    rsi = float(valid_rsi.iloc[-1]) if not valid_rsi.empty else 50
    if rsi <= 25:   rsi_score, rsi_e = +2, f"RSI {rsi:.0f} - deeply oversold"
    elif rsi <= 35: rsi_score, rsi_e = +1, f"RSI {rsi:.0f} - oversold"
    elif rsi >= 80: rsi_score, rsi_e = -2, f"RSI {rsi:.0f} - extremely overbought"
    elif rsi >= 70: rsi_score, rsi_e = -1, f"RSI {rsi:.0f} - overbought"
    else:           rsi_score, rsi_e =  0, f"RSI {rsi:.0f} - neutral"
    if div_rsi == "bull" and rsi <= 50:
        rsi_score += 1; rsi_e += " + bullish divergence"
    elif div_rsi == "bear" and rsi >= 50:
        rsi_score -= 1; rsi_e += " + bearish divergence"
    rsi_score = max(-3, min(3, rsi_score))
    signals["RSI + Divergence"] = (rsi_score, rsi_e); total += rsi_score

    # 4. Funding / Momentum
    m7  = crypto_sig.get("momentum_7d",  "N/A")
    m30 = crypto_sig.get("momentum_30d", "N/A")
    try:
        m7f  = float(str(m7).replace("%", "").replace("+", ""))
        m30f = float(str(m30).replace("%", "").replace("+", ""))
        if m7f < -15 and m30f < -20:   s, e = +3, f"Capitulation: 7d {m7}, 30d {m30}"
        elif m7f < -8 and m30f < -10:  s, e = +2, f"Sharp pullback: 7d {m7}, 30d {m30}"
        elif m7f < -5:                 s, e = +1, f"Moderate selloff: 7d {m7}"
        elif m7f > 20 and m30f > 30:   s, e = -3, f"Euphoric rally: 7d {m7}, 30d {m30}"
        elif m7f > 10 and m30f > 15:   s, e = -2, f"Strong rally: 7d {m7}, 30d {m30}"
        elif m7f > 5:                  s, e = -1, f"Moderate rally: 7d {m7}"
        else:                          s, e =  0, f"Neutral: 7d {m7}, 30d {m30}"
    except Exception:
        s, e = 0, f"Momentum: {m7} / {m30}"
    signals["Price Momentum"] = (s, e); total += s

    # 5. BTC Dominance
    dom_raw = str(crypto_sig.get("btc_dominance", "N/A")).replace("%", "")
    try:
        dom = float(dom_raw)
        if dom >= 65:   s, e = +3, f"Dominance {dom:.1f}% - extreme alt washout, max BTC season"
        elif dom >= 60: s, e = +2, f"Dominance {dom:.1f}% - extreme BTC season"
        elif dom >= 55: s, e = +1, f"Dominance {dom:.1f}% - high, alts underperforming"
        elif dom >= 48: s, e =  0, f"Dominance {dom:.1f}% - balanced, mid-cycle"
        elif dom >= 40: s, e = -1, f"Dominance {dom:.1f}% - alt season building"
        elif dom >= 32: s, e = -2, f"Dominance {dom:.1f}% - alt mania, classic top zone"
        else:           s, e = -3, f"Dominance {dom:.1f}% - extreme alt bubble, peak greed"
    except Exception:
        s, e = 0, "BTC dominance N/A"
    signals["BTC Dominance"] = (s, e); total += s

    # 6. Price vs MA200
    ma200 = pd.Series(closes).rolling(200).mean().iloc[-1]
    if not np.isnan(ma200) and ma200 > 0:
        dev = (price - ma200) / ma200 * 100
        if dev < -40:   s, e = +3, f"Price {dev:+.0f}% below MA200 - capitulation"
        elif dev < -25: s, e = +2, f"Price {dev:+.0f}% below MA200 - stretched below mean"
        elif dev < -10: s, e = +1, f"Price {dev:+.0f}% below MA200"
        elif dev < +15: s, e =  0, f"Price {dev:+.0f}% vs MA200 - near mean"
        elif dev < +40: s, e = -1, f"Price {dev:+.0f}% above MA200 - extended"
        elif dev < +70: s, e = -2, f"Price {dev:+.0f}% above MA200 - bull peak territory"
        else:           s, e = -3, f"Price {dev:+.0f}% above MA200 - parabolic, top risk"
    else:
        s, e = 0, "MA200 unavailable"
    signals["Price vs MA200"] = (s, e); total += s

    # 7. 52-Week Position
    w52 = week52_metrics(closes)
    pfl = w52.get("pct_from_low")
    pfh = w52.get("pct_from_high")
    if pfl is not None and pfh is not None:
        if pfl <= 10:    s, e = +3, f"Only {pfl:.0f}% above 52W low - near floor"
        elif pfl <= 25:  s, e = +2, f"{pfl:.0f}% above 52W low - lower quartile"
        elif pfl <= 50:  s, e = +1, f"{pfl:.0f}% above 52W low - lower half"
        elif pfh > -10:  s, e = -3, f"Only {abs(pfh):.0f}% from 52W high - near ceiling"
        elif pfh > -20:  s, e = -2, f"{abs(pfh):.0f}% from 52W high - upper quartile"
        elif pfh > -35:  s, e = -1, f"{abs(pfh):.0f}% from 52W high - upper half"
        else:            s, e =  0, f"{pfl:.0f}% from low / {abs(pfh):.0f}% from high - mid"
    else:
        s, e = 0, "52W data unavailable"
    signals["52W Position"] = (s, e); total += s

    # 8. OBV Divergence
    if obv_trend == "strong_accumulation":   s, e = +3, "OBV strong accumulation - aggressive smart money loading"
    elif obv_trend == "accumulation":        s, e = +2, "OBV accumulation - smart money loading"
    elif obv_trend == "mild_accumulation":   s, e = +1, "OBV mild accumulation"
    elif obv_trend == "strong_distribution": s, e = -3, "OBV strong distribution - aggressive smart money exit"
    elif obv_trend == "distribution":        s, e = -2, "OBV distribution - smart money exiting"
    elif obv_trend == "mild_distribution":   s, e = -1, "OBV mild distribution"
    else:                                    s, e =  0, "OBV neutral"
    signals["OBV Divergence"] = (s, e); total += s

    # Phase label
    if total >= 12:    phase, phase_col, emoji, advice = "PROBABLE BOTTOM", "green",  "🟢", "High confluence of bottom signals. Strong case for accumulation."
    elif total >= 6:   phase, phase_col, emoji, advice = "BOTTOM FORMING",  "yellow", "🟡", "Multiple bottom signals aligning. Start building position in tranches."
    elif total >= -5:  phase, phase_col, emoji, advice = "MID-CYCLE",       "gray",   "⚪", "No clear cycle extreme. Follow trend, manage risk normally."
    elif total >= -11: phase, phase_col, emoji, advice = "TOP FORMING",     "yellow", "🟡", "Multiple top signals aligning. Reduce exposure, tighten stops."
    else:              phase, phase_col, emoji, advice = "PROBABLE TOP",    "red",    "🔴", "High confluence of top signals. Strong case for taking profits."

    return {
        "total": total, "max": 24,
        "phase": phase, "emoji": emoji, "advice": advice, "signals": signals,
    }


# ════════════════════════════════════════════════════════════════
#  LIQUIDITY (Binance depth)
# ════════════════════════════════════════════════════════════════

def _get_raw(url, timeout=8):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _json.loads(r.read().decode())
    except Exception:
        return None


def _fusd(v):
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:.0f}"

def _fprice(v):
    return f"${v:,.0f}" if v >= 1000 else f"${v:.2f}"

# FIX: matplotlib text with usetex=False doesn't need \$ but it also
# doesn't break — kept for consistency with original rendering intent.
def _fusd_m(v):   return _fusd(v).replace("$", r"\$")
def _fprice_m(v): return _fprice(v).replace("$", r"\$")


def _cluster(levels, bsz):
    c = {}
    for ps, qs in levels:
        p, q = float(ps), float(qs)
        b = round(p / bsz) * bsz
        c[b] = c.get(b, 0.0) + p * q
    return c


def _walls(clusters, cprice, rpct, mnotl, side):
    lo, hi = cprice * (1 - rpct / 100), cprice * (1 + rpct / 100)
    cands  = [(p, n) for p, n in clusters.items()
              if lo <= p <= hi and n >= mnotl
              and ((side == "bid" and p < cprice) or (side == "ask" and p > cprice))]
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands[:20]


def _snap_grid(lo, hi, bsz):
    return np.arange(np.ceil(lo / bsz) * bsz, np.floor(hi / bsz) * bsz + bsz, bsz)


def _fetch_depth(cprice=None):
    """
    Aggregate BTC/USD order book from Binance (spot+fut), Bybit, and OKX in
    parallel.  Failing exchanges are silently skipped — the result degrades
    gracefully rather than returning empty data.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    r = {"bid_walls": [], "ask_walls": [], "bid_c": {}, "ask_c": {},
         "price": 0, "ratio": 1.0, "bid_usd": 0, "ask_usd": 0,
         "spread": 0, "source": "N/A", "raw_bids": [], "raw_asks": []}

    # ── Fetch all four endpoints in parallel ─────────────────────────────────
    _urls = {
        "bn_spot": BINANCE_SPOT_DEPTH,
        "bn_fut":  BINANCE_FUT_DEPTH,
        "bybit":   BYBIT_FUT_DEPTH,
        "okx":     OKX_SWAP_DEPTH,
    }
    _raw = {}
    with ThreadPoolExecutor(max_workers=4) as _ex:
        _futs = {_ex.submit(_get_raw, url): name for name, url in _urls.items()}
        try:
            for _f in as_completed(_futs, timeout=12):
                try:
                    _raw[_futs[_f]] = _f.result()
                except Exception:
                    _raw[_futs[_f]] = None   # individual exchange error — skip it
        except TimeoutError:
            # One or more exchanges didn't respond in time — use whatever finished.
            # Cancel any still-pending futures so the executor can shut down promptly.
            for _f in _futs:
                if not _f.done():
                    _f.cancel()
                else:
                    try:
                        _raw.setdefault(_futs[_f], _f.result(timeout=0))
                    except Exception:
                        _raw.setdefault(_futs[_f], None)

    # ── Parse and normalise each source to [(price_str, qty_str), ...] ───────
    bids: list = []
    asks: list = []
    sources: list = []

    # Binance (spot + futures) — format: [["price", "qty"], ...]
    for key in ("bn_spot", "bn_fut"):
        d = _raw.get(key)
        if d and "bids" in d:
            bids += d["bids"];  asks += d["asks"]
            sources.append(key)

    # Use Binance best bid/ask to set initial cprice / spread before adding others
    if bids and asks and cprice is None:
        bb = float(bids[0][0]) if bids else 0
        ba = float(asks[0][0]) if asks else 0
        cprice = (bb + ba) / 2 if bb and ba else 0
    if not cprice:
        # Last resort: try Bybit mid
        d = _raw.get("bybit")
        if d and d.get("retCode") == 0:
            res = d["result"]
            bb = float(res["b"][0][0]) if res.get("b") else 0
            ba = float(res["a"][0][0]) if res.get("a") else 0
            cprice = (bb + ba) / 2 if bb and ba else 0
    if not cprice:
        return r

    # Spread from Binance spot best quote
    if "bn_spot" in sources and _raw["bn_spot"].get("bids"):
        _bb = float(_raw["bn_spot"]["bids"][0][0])
        _ba = float(_raw["bn_spot"]["asks"][0][0])
        if _bb > 0:
            r["spread"] = round((_ba - _bb) / _bb * 10000, 2)

    # Bybit — format: {"retCode": 0, "result": {"b": [["p","q"],...], "a": [...]}}
    d = _raw.get("bybit")
    if d and d.get("retCode") == 0:
        res = d.get("result", {})
        bids += res.get("b", [])
        asks += res.get("a", [])
        sources.append("bybit")

    # OKX — format: {"code":"0","data":[{"bids":[["p","q","0","n"],...], "asks":[...]}]}
    d = _raw.get("okx")
    if d and d.get("code") == "0":
        book = (d.get("data") or [{}])[0]
        # OKX rows have 4 fields; only first two (price, qty) are needed
        bids += [[row[0], row[1]] for row in book.get("bids", [])]
        asks += [[row[0], row[1]] for row in book.get("asks", [])]
        sources.append("okx")

    if not sources:
        return r

    r["price"]    = cprice
    r["source"]   = "+".join(sources)
    r["raw_bids"] = bids
    r["raw_asks"] = asks

    r["bid_c"]       = _cluster(bids, CLUSTER_BIN)
    r["ask_c"]       = _cluster(asks, CLUSTER_BIN)

    lo, hi = cprice * (1 - WALL_RANGE_PCT / 100), cprice * (1 + WALL_RANGE_PCT / 100)
    r["bid_usd"]  = sum(v for p, v in r["bid_c"].items() if lo <= p <= cprice)
    r["ask_usd"]  = sum(v for p, v in r["ask_c"].items() if cprice <= p <= hi)
    r["ratio"]    = round(r["bid_usd"] / r["ask_usd"], 2) if r["ask_usd"] > 0 else 99.0

    bid_min = max(WALL_MIN_NOTL, r["bid_usd"] * WALL_MIN_DEPTH_PCT)
    ask_min = max(WALL_MIN_NOTL, r["ask_usd"] * WALL_MIN_DEPTH_PCT)
    r["bid_walls"] = _walls(r["bid_c"], cprice, WALL_RANGE_PCT, bid_min, "bid")
    r["ask_walls"] = _walls(r["ask_c"], cprice, WALL_RANGE_PCT, ask_min, "ask")
    return r


def _build_liq(raw_bids, raw_asks, cprice, grid, depth_ratio, tiers=None):
    if tiers is None: tiers = ALL_TIERS
    bsz = HEATMAP_BIN; n = len(grid)
    g_lo, g_hi = grid[0], grid[-1]
    layers = {}; la = np.zeros(n); sa = np.zeros(n)
    lw = min(2.0, max(0.3, 0.7 + 0.3 * depth_ratio))
    sw = min(2.0, max(0.3, 0.7 + 0.3 / max(depth_ratio, 0.1)))
    for lev in tiers:
        layer = np.zeros(n)
        for ps, qs in raw_bids:
            p, q = float(ps), float(qs)
            lp   = p * (1.0 - 1.0 / lev)
            if g_lo <= lp <= g_hi:
                idx = int(round((lp - g_lo) / bsz))
                if 0 <= idx < n:
                    w = p * q * (lev / 50.0) * lw
                    layer[idx] += w; la[idx] += w
        for ps, qs in raw_asks:
            p, q = float(ps), float(qs)
            lp   = p * (1.0 + 1.0 / lev)
            if g_lo <= lp <= g_hi:
                idx = int(round((lp - g_lo) / bsz))
                if 0 <= idx < n:
                    w = p * q * (lev / 50.0) * sw
                    layer[idx] += w; sa[idx] += w
        layers[lev] = layer
    return layers, la, sa, la + sa


def _analyze_liq(la, sa, grid, cprice, bid_walls, ask_walls,
                 bid_c: dict = None, ask_c: dict = None):
    """
    Derives liquidation structure, hunt zones, and cascade potential.

    la/sa    = leverage-math heat arrays (None → skip)
    bid_c/ask_c = raw $50-bin order-book clusters (used for hunt zone engine)
    """
    bid_c = bid_c or {}
    ask_c = ask_c or {}

    # ── Leverage-math liq clusters (la/sa arrays) ─────────────────────────────
    nl = ns = None
    lt = st = 0.0
    long_clusters  = []
    short_clusters = []

    if la is not None and sa is not None and grid is not None and len(grid) > 0:
        import numpy as np
        grid  = np.asarray(grid)
        below = grid < cprice
        above = grid > cprice
        lt = float(la[below].sum())
        st = float(sa[above].sum())

        def _top_peaks(arr, mask, g, n=3):
            sub = arr * mask
            peaks = []
            if sub.max() == 0:
                return peaks
            for _ in range(n):
                idx = int(sub.argmax())
                if sub[idx] == 0:
                    break
                peaks.append((float(g[idx]), float(sub[idx])))
                lo = max(0, idx - 3); hi = min(len(sub), idx + 4)
                sub[lo:hi] = 0
            return sorted(peaks, key=lambda x: x[0], reverse=True)

        long_clusters  = _top_peaks(la.copy(), below, grid)
        short_clusters = _top_peaks(sa.copy(), above, grid)
        if long_clusters:
            nl = max(long_clusters, key=lambda x: x[0])
        if short_clusters:
            ns = min(short_clusters, key=lambda x: x[0])

    # ── Cascade engine: uses bid_c/ask_c cluster fuel ─────────────────────────
    # Long-liq fuel = bids below price (forced sells if swept)
    # Short-liq fuel = asks above price (forced buys if swept)
    CASC_RANGE = 0.06   # ±6% from current price
    lo_c = cprice * (1 - CASC_RANGE)
    hi_c = cprice * (1 + CASC_RANGE)

    long_fuel  = sum(v for p, v in bid_c.items() if lo_c <= p < cprice)
    short_fuel = sum(v for p, v in ask_c.items() if cprice < p <= hi_c)

    # Cascade chain: consecutive clusters within $500 of each other amplify
    def _chain_len(prices_sorted):
        if len(prices_sorted) < 2:
            return len(prices_sorted)
        n = 1
        for i in range(1, len(prices_sorted)):
            if abs(prices_sorted[i] - prices_sorted[i-1]) <= 500:
                n += 1
        return n

    down_prices = sorted([p for p in bid_c if lo_c <= p < cprice], reverse=True)
    up_prices   = sorted([p for p in ask_c if cprice < p <= hi_c])
    chain_down  = _chain_len(down_prices)
    chain_up    = _chain_len(up_prices)

    # Cascade ratio: long fuel / short fuel, amplified by chain length
    casc_ratio_raw = (long_fuel * (1 + 0.2 * chain_down)) / max(
                      short_fuel * (1 + 0.2 * chain_up), 1)

    # Fall back to order-book wall ratio when cluster data is absent
    if long_fuel == 0 and short_fuel == 0:
        ob_wall_long  = sum(wn for _, wn in bid_walls)
        ob_wall_short = sum(wn for _, wn in ask_walls)
        casc_ratio_raw = (ob_wall_long / max(ob_wall_short, 1)
                          if ob_wall_short else (99.0 if ob_wall_long else 1.0))

    cd = ("DOWN" if casc_ratio_raw > 1.4 else
          "UP"   if casc_ratio_raw < 0.7 else "BALANCED")

    # ── Hunt zone engine ──────────────────────────────────────────────────────
    # A "hunt zone" is a price level that the market is likely to sweep because
    # of the concentrated liquidation fuel sitting just beyond it.
    # Score = notional / dist_pct² × cascade_multiplier
    # BID clusters below price → price hunts DOWN to flush longs
    # ASK clusters above price → price hunts UP to squeeze shorts
    HUNT_RANGE = 0.08   # ±8% search window
    hz = []

    def _add_zones(clusters, side, is_bid):
        sorted_c = sorted(clusters.items(),
                          key=lambda x: abs(x[0] - cprice))  # nearest first
        for i, (p, v) in enumerate(sorted_c):
            dist_pct = abs(p - cprice) / cprice
            if dist_pct > HUNT_RANGE:
                continue
            if dist_pct < 1e-4:
                continue
            # Cascade multiplier: clusters of same side stacked within $1000
            nearby = sum(1 for p2, _ in sorted_c[i+1:]
                         if abs(p2 - p) <= 1000)
            cascade_mult = 1.0 + 0.25 * min(nearby, 4)
            hunt_score   = (v / 1e6) / (dist_pct ** 2) * cascade_mult
            # Nearest order-book wall sitting between price and this zone
            if is_bid:
                wall_res = max((wn for wp, wn in bid_walls if p < wp < cprice),
                               default=0)
            else:
                wall_res = max((wn for wp, wn in ask_walls if cprice < wp < p),
                               default=0)
            hz.append({
                "price":         p,
                "side":          "BID" if is_bid else "ASK",
                "notional":      v,
                "dist_pct":      dist_pct,
                "hunt_score":    hunt_score,
                "cascade_chain": nearby,
                "wall_res":      wall_res,
                # Legacy keys kept for 72h bias compatibility
                "wall": v,
                "fuel": v,
            })

    _add_zones({p: v for p, v in bid_c.items() if p < cprice}, "BID", True)
    _add_zones({p: v for p, v in ask_c.items() if p > cprice}, "ASK", False)

    # Fall back to top walls when clusters are absent
    if not hz:
        for wp, wn in bid_walls:
            if wn >= WALL_MIN_NOTL:
                hz.append({"price": wp, "side": "BID", "notional": wn,
                           "dist_pct": abs(wp - cprice) / cprice,
                           "hunt_score": wn / max(abs(wp - cprice) / cprice, 1e-4) ** 2,
                           "cascade_chain": 0, "wall_res": 0, "wall": wn, "fuel": wn})
        for wp, wn in ask_walls:
            if wn >= WALL_MIN_NOTL:
                hz.append({"price": wp, "side": "ASK", "notional": wn,
                           "dist_pct": abs(wp - cprice) / cprice,
                           "hunt_score": wn / max(abs(wp - cprice) / cprice, 1e-4) ** 2,
                           "cascade_chain": 0, "wall_res": 0, "wall": wn, "fuel": wn})

    hz.sort(key=lambda z: z["hunt_score"], reverse=True)

    return {
        "nearest_long_liq":  nl,   "nearest_short_liq":  ns,
        "long_liq_total":    lt,   "short_liq_total":    st,
        "long_clusters":     long_clusters,
        "short_clusters":    short_clusters,
        "long_fuel":         long_fuel,  "short_fuel":   short_fuel,
        "cascade_direction": cd,   "cascade_ratio":  casc_ratio_raw,
        "chain_down":        chain_down, "chain_up":    chain_up,
        "hunt_zones":        hz,
    }


def _narrative(liq, a):
    """Plain-facts liquidity structure summary — no directional claims."""
    cp         = liq.get("liq_current_price", 0)
    ratio      = liq.get("liq_depth_ratio", 1.0)
    bid_usd    = liq.get("liq_total_bid", 0)
    ask_usd    = liq.get("liq_total_ask", 0)
    lt, sliq   = a.get("long_liq_total", 0), a.get("short_liq_total", 0)
    nl, ns     = a.get("nearest_long_liq"), a.get("nearest_short_liq")
    # Prefer clusters from the heatmap render (same source as visual) over the
    # order-book projection used as fallback in _analyze_liq.
    _hm = st.session_state.get("_liq_clusters", {})
    _hm_err   = st.session_state.get("_liq_clusters_err", "")
    _hm_long  = _hm.get("long",  [])   # [(price, raw_heat, vis), ...]
    _hm_short = _hm.get("short", [])
    _hm_real  = _hm.get("real_events", False)
    long_c    = [(p, r) for p, r, _ in _hm_long]  if _hm_long  else a.get("long_clusters",  [])
    short_c   = [(p, r) for p, r, _ in _hm_short] if _hm_short else a.get("short_clusters", [])
    hz         = a.get("hunt_zones", [])
    cd         = a.get("cascade_direction", "BALANCED")
    bid_walls  = liq.get("liq_bid_walls", [])
    ask_walls  = liq.get("liq_ask_walls", [])

    L = []
    L.append(f"=== LIQUIDITY MAP at {_fprice(cp)} ===\n")

    # ── 1. Order-book walls (these are limit orders, NOT liquidations) ────────
    bw = bid_walls[0] if bid_walls else None
    aw = ask_walls[0] if ask_walls else None
    bw_str = (f"{_fprice(bw[0])}  ({_fusd(bw[1])})  "
              f"[{(cp - bw[0]) / cp * 100:.1f}% below]") if bw else "none detected"
    aw_str = (f"{_fprice(aw[0])}  ({_fusd(aw[1])})  "
              f"[{(aw[0] - cp) / cp * 100:.1f}% above]") if aw else "none detected"
    if ratio > 1.4:
        imb_txt = f"{ratio:.1f}x more bids than asks — order book is bid-heavy"
    elif ratio < 0.7:
        imb_txt = f"{(1/ratio if ratio > 0 else float('inf')):.1f}x more asks than bids — order book is ask-heavy"
    else:
        imb_txt = f"balanced ({ratio:.1f}x bid/ask)"
    L.append("ORDER BOOK WALLS  (large resting limit orders — not liquidations)")
    L.append(f"  Nearest bid wall : {bw_str}")
    L.append(f"  Nearest ask wall : {aw_str}")
    L.append(f"  Depth imbalance  : {imb_txt}\n")

    # ── 2 & 3. Liquidation clusters from heatmap ─────────────────────────────
    has_hm_data  = bool(_hm_long or _hm_short)
    has_liq_data = has_hm_data or lt > 0 or sliq > 0
    if has_hm_data:
        src_note = f"from heatmap {'(real liq events)' if _hm_real else '(lev-math estimate)'}"
    elif _hm_err:
        src_note = f"heatmap error: {_hm_err}"
    else:
        src_note = "order-book estimate — updates after heatmap renders (scroll down first)"

    def _fmt_cluster(price, usd, vis, side):
        pct     = (cp - price) / cp * 100 if side == "long" else (price - cp) / cp * 100
        dir_str = "below" if side == "long" else "above"
        usd_str = _fusd(usd) if usd > 0 else "n/a"
        vis_str = f"  [{vis*100:.0f}% intensity]" if vis > 0 else ""
        return f"  {_fprice(price)}  ({pct:.1f}% {dir_str})  ~{usd_str} est. liq{vis_str}"

    # Long clusters sit BELOW price. If price sweeps them, force-sells cascade.
    L.append(f"LONG LIQUIDATION CLUSTERS  (leveraged longs get force-closed below price)")
    L.append(f"  Source: {src_note}")
    L.append( "  If price drops here, long positions are force-sold → cascading sell pressure.")
    if long_c:
        for i, (price, usd) in enumerate(long_c[:4]):
            vis  = _hm_long[i][2] if has_hm_data and i < len(_hm_long) else 0
            row  = _fmt_cluster(price, usd, vis, "long")
            tag  = "  ← nearest" if i == 0 else ""
            star = "★" if i == 0 else " "
            L.append(f"  {star}{row}{tag}")
    else:
        L.append("  No significant clusters below current price.")
    L.append("")

    # Short clusters sit ABOVE price. If price sweeps them, force-buys squeeze.
    L.append(f"SHORT LIQUIDATION CLUSTERS  (leveraged shorts get force-closed above price)")
    L.append(f"  Source: {src_note}")
    L.append( "  If price rises here, short positions are force-bought → cascading buy pressure.")
    if short_c:
        for i, (price, usd) in enumerate(short_c[:4]):
            vis  = _hm_short[i][2] if has_hm_data and i < len(_hm_short) else 0
            row  = _fmt_cluster(price, usd, vis, "short")
            tag  = "  ← nearest" if i == 0 else ""
            star = "★" if i == 0 else " "
            L.append(f"  {star}{row}{tag}")
    else:
        L.append("  No significant clusters above current price.")
    L.append("")

    # ── 4. Interpretation ─────────────────────────────────────────────────────
    L.append("INTERPRETATION")
    if has_hm_data:
        long_top_vis  = max((v for _, _, v in _hm_long[:3]),  default=0)
        short_top_vis = max((v for _, _, v in _hm_short[:3]), default=0)
        long_near  = long_c[0]  if long_c  else None
        short_near = short_c[0] if short_c else None
        if long_top_vis > 0 or short_top_vis > 0:
            dominant = ("short" if short_top_vis > long_top_vis * 1.2
                        else "long" if long_top_vis > short_top_vis * 1.2
                        else None)
            if long_near:
                L.append(f"  Nearest long cluster : {_fprice(long_near[0])} "
                         f"({(cp-long_near[0])/cp*100:.1f}% below)  ~{_fusd(long_near[1])} est. liq")
            if short_near:
                L.append(f"  Nearest short cluster: {_fprice(short_near[0])} "
                         f"({(short_near[0]-cp)/cp*100:.1f}% above)  ~{_fusd(short_near[1])} est. liq")
            if dominant == "short":
                L.append(f"  Short clusters are denser — upward move has more cascade fuel.")
                L.append(f"  Price may be drawn up to squeeze leveraged shorts.")
            elif dominant == "long":
                L.append(f"  Long clusters are denser — downward move has more cascade fuel.")
                L.append(f"  Price may be drawn down to flush leveraged longs.")
            else:
                L.append(f"  Long and short cluster density roughly balanced.")
    elif long_c or short_c:
        L.append(f"  (Order-book estimate — heatmap clusters load after first render)")
    if bid_walls or ask_walls:
        L.append(f"  Order book: {ratio:.1f}x {'bid' if ratio>1 else 'ask'}-heavy "
                 f"({'support bias' if ratio>1 else 'resistance bias'})")

    # Largest order-book wall (clearly labelled as OB, not liquidation)
    if hz:
        h    = hz[0]
        side = "Bid" if h["side"] == "BID" else "Ask"
        L.append(f"\nLARGEST ORDER BOOK WALL  (resting limit order — not a liquidation cluster)")
        L.append(f"  {side} wall {_fusd(h['wall'])} @ {_fprice(h['price'])} — "
                 f"large resting order that price may be drawn toward")

    return "\n".join(f"  {ln}" if ln and not ln.startswith("===") else ln for ln in L)


# ════════════════════════════════════════════════════════════════
#  MULTI-SIGNAL DIRECTION PREDICTION ENGINE
# ════════════════════════════════════════════════════════════════

def predict_direction(df: pd.DataFrame, liq: dict) -> dict:
    """
    Score directional bias across 7 independent signal groups.
    Each group scores −2 to +2 (positive = bullish).
    Returns a dict with score, label, confidence, and per-signal breakdown.
    """
    signals = {}
    total   = 0

    closes = df["Close"].tolist()
    price  = closes[-1]

    # ── 1. Short-term momentum (last 6 candles) ─────────────────
    try:
        p6 = closes[-6] if len(closes) >= 6 else closes[0]
        ret6 = (price - p6) / p6 * 100
        if   ret6 >  4: s, e = +2, f"Strong rally last 6 bars (+{ret6:.1f}%)"
        elif ret6 >  1: s, e = +1, f"Mild upside last 6 bars (+{ret6:.1f}%)"
        elif ret6 < -4: s, e = -2, f"Strong selloff last 6 bars ({ret6:.1f}%)"
        elif ret6 < -1: s, e = -1, f"Mild downside last 6 bars ({ret6:.1f}%)"
        else:           s, e =  0, f"Flat last 6 bars ({ret6:.1f}%)"
    except Exception:
        s, e = 0, "Momentum N/A"
    signals["Short Momentum"] = (s, e); total += s

    # ── 2. RSI level + direction ─────────────────────────────────
    try:
        rsi_s = calculate_rsi(df["Close"])
        rsi   = float(rsi_s.dropna().iloc[-1])
        rsi_5 = float(rsi_s.dropna().iloc[-5]) if len(rsi_s.dropna()) >= 5 else rsi
        rising = rsi > rsi_5
        if   rsi < 30: s, e = +2 if rising else +1, f"RSI {rsi:.0f} oversold {'↑' if rising else '→'}"
        elif rsi < 45: s, e = +1, f"RSI {rsi:.0f} below mid, {'recovering' if rising else 'weak'}"
        elif rsi > 70: s, e = -2 if not rising else -1, f"RSI {rsi:.0f} overbought {'↓' if not rising else '→'}"
        elif rsi > 55: s, e = -1, f"RSI {rsi:.0f} above mid, {'fading' if not rising else 'extended'}"
        else:          s, e =  0, f"RSI {rsi:.0f} neutral"
    except Exception:
        s, e = 0, "RSI N/A"
    signals["RSI"] = (s, e); total += s

    # ── 3. MACD histogram momentum ───────────────────────────────
    try:
        _, _, hist = calculate_macd(df["Close"])
        h_now  = float(hist.dropna().iloc[-1])
        h_prev = float(hist.dropna().iloc[-3]) if len(hist.dropna()) >= 3 else h_now
        if   h_now > 0 and h_now > h_prev: s, e = +2, f"MACD hist positive and rising ({h_now:+.0f})"
        elif h_now > 0:                    s, e = +1, f"MACD hist positive but fading ({h_now:+.0f})"
        elif h_now < 0 and h_now < h_prev: s, e = -2, f"MACD hist negative and falling ({h_now:+.0f})"
        elif h_now < 0:                    s, e = -1, f"MACD hist negative but recovering ({h_now:+.0f})"
        else:                              s, e =  0, "MACD hist near zero"
    except Exception:
        s, e = 0, "MACD N/A"
    signals["MACD Histogram"] = (s, e); total += s

    # ── 4. Stochastic crossover + zone ───────────────────────────
    try:
        st_d = calculate_stochastic(df)
        k = float(st_d["k"].dropna().iloc[-1])
        d = float(st_d["d"].dropna().iloc[-1])
        k_prev = float(st_d["k"].dropna().iloc[-2]) if len(st_d["k"].dropna()) >= 2 else k
        d_prev = float(st_d["d"].dropna().iloc[-2]) if len(st_d["d"].dropna()) >= 2 else d
        bull_cross = k > d and k_prev <= d_prev
        bear_cross = k < d and k_prev >= d_prev
        if   k < 20 and bull_cross: s, e = +2, f"Stoch bullish cross in oversold zone (%K {k:.0f})"
        elif k < 25:                s, e = +1, f"Stoch oversold %K {k:.0f}"
        elif k > 80 and bear_cross: s, e = -2, f"Stoch bearish cross in overbought zone (%K {k:.0f})"
        elif k > 75:                s, e = -1, f"Stoch overbought %K {k:.0f}"
        elif bull_cross:            s, e = +1, f"Stoch bullish cross %K {k:.0f}"
        elif bear_cross:            s, e = -1, f"Stoch bearish cross %K {k:.0f}"
        else:                       s, e =  0, f"Stoch neutral %K {k:.0f} / %D {d:.0f}"
    except Exception:
        s, e = 0, "Stoch N/A"
    signals["Stochastic"] = (s, e); total += s

    # ── 5. Price vs MA structure ──────────────────────────────────
    try:
        ma50  = float(df["MA50"].dropna().iloc[-1])
        ma200 = float(df["MA200"].dropna().iloc[-1])
        above50  = price > ma50
        above200 = price > ma200
        ma50_slope  = (float(df["MA50"].dropna().iloc[-1]) - float(df["MA50"].dropna().iloc[-5])) / ma50 * 100
        if above50 and above200 and ma50_slope > 0:
            s, e = +2, f"Price above both MAs, MA50 rising"
        elif above50 and above200:
            s, e = +1, f"Price above both MAs"
        elif not above50 and not above200 and ma50_slope < 0:
            s, e = -2, f"Price below both MAs, MA50 falling"
        elif not above50 and not above200:
            s, e = -1, f"Price below both MAs"
        else:
            s, e =  0, f"Price between MA50/MA200 — mixed"
    except Exception:
        s, e = 0, "MA structure N/A"
    signals["MA Structure"] = (s, e); total += s

    # ── 6. Order book depth ratio ─────────────────────────────────
    try:
        ratio    = liq.get("liq_depth_ratio", 1.0) if liq else 1.0
        bid_usd  = liq.get("liq_total_bid", 0)     if liq else 0
        ask_usd  = liq.get("liq_total_ask", 0)     if liq else 0
        if   ratio >= 2.0: s, e = +2, f"Order book strongly bid-heavy ({ratio:.1f}x, {_fusd(bid_usd)} bids)"
        elif ratio >= 1.4: s, e = +1, f"Order book bid-heavy ({ratio:.1f}x)"
        elif ratio <= 0.5: s, e = -2, f"Order book strongly ask-heavy ({ratio:.1f}x, {_fusd(ask_usd)} asks)"
        elif ratio <= 0.7: s, e = -1, f"Order book ask-heavy ({ratio:.1f}x)"
        else:              s, e =  0, f"Order book balanced ({ratio:.1f}x)"
    except Exception:
        s, e = 0, "Orderbook N/A"
    signals["Order Book"] = (s, e); total += s

    # ── 7. Nearest liquidity asymmetry ────────────────────────────
    # Which magnet is closer AND bigger relative to the wall protecting it?
    try:
        analysis  = liq.get("liq_analysis", {}) if liq else {}
        nl_pair   = analysis.get("nearest_long_liq")    # (price, notional) below
        ns_pair   = analysis.get("nearest_short_liq")   # (price, notional) above
        bid_walls = liq.get("liq_bid_walls", []) if liq else []
        ask_walls = liq.get("liq_ask_walls", []) if liq else []
        bid_wall1 = bid_walls[0][1] if bid_walls else 0
        ask_wall1 = ask_walls[0][1] if ask_walls else 0

        down_dist = (price - nl_pair[0]) / price if nl_pair else 1.0
        up_dist   = (ns_pair[0] - price) / price if ns_pair else 1.0
        down_fuel = nl_pair[1] if nl_pair else 0
        up_fuel   = ns_pair[1] if ns_pair else 0

        # Score: closer + bigger fuel relative to wall protecting it = stronger magnet
        down_score = (down_fuel / max(bid_wall1, 1)) * (1 - down_dist * 10)
        up_score   = (up_fuel  / max(ask_wall1, 1)) * (1 - up_dist  * 10)

        if   up_score > down_score * 2 and up_dist < 0.02:
            s, e = +2, f"Short liq {_fusd(up_fuel)} at {_fprice(ns_pair[0])} — strong upside magnet"
        elif up_score > down_score:
            s, e = +1, f"Upside liq magnet stronger ({_fprice(ns_pair[0])} if ns_pair else 'N/A')"
        elif down_score > up_score * 2 and down_dist < 0.02:
            s, e = -2, f"Long liq {_fusd(down_fuel)} at {_fprice(nl_pair[0])} — strong downside magnet"
        elif down_score > up_score:
            s, e = -1, f"Downside liq magnet stronger ({_fprice(nl_pair[0]) if nl_pair else 'N/A'})"
        else:
            s, e =  0, "Liq magnets balanced"
    except Exception:
        s, e = 0, "Liq asymmetry N/A"
    signals["Liq Asymmetry"] = (s, e); total += s

    # ── Composite label ───────────────────────────────────────────
    max_score = len(signals) * 2   # = 14
    pct       = (total + max_score) / (2 * max_score) * 100   # 0–100 scale

    if   total >= 8:  label, col = "STRONG BULL",  "#3fb950"
    elif total >= 4:  label, col = "MILD BULL",    "#58a6ff"
    elif total >= 1:  label, col = "SLIGHT BULL",  "#8bc4ff"
    elif total <= -8: label, col = "STRONG BEAR",  "#f85149"
    elif total <= -4: label, col = "MILD BEAR",    "#f0883e"
    elif total <= -1: label, col = "SLIGHT BEAR",  "#ffb347"
    else:             label, col = "NEUTRAL",       "#8b949e"

    # Confidence = how many signals agree with the net direction
    net_dir  = 1 if total > 0 else (-1 if total < 0 else 0)
    agreeing = sum(1 for s, _ in signals.values() if (s > 0 and net_dir > 0) or
                                                      (s < 0 and net_dir < 0))
    conf_pct = int(agreeing / len(signals) * 100)

    return {
        "total": total, "max": max_score, "pct": pct,
        "label": label, "color": col, "confidence": conf_pct,
        "signals": signals,
    }


# ════════════════════════════════════════════════════════════════
#  12-HOUR DIRECTIONAL BIAS SCORE  (−100 to +100)
# ════════════════════════════════════════════════════════════════

def compute_72h_bias(df: pd.DataFrame, liq: dict,
                     short_df: "pd.DataFrame | None" = None,
                     poly: "dict | None" = None,
                     oi_data: "dict | None" = None,
                     df_4h: "pd.DataFrame | None" = None) -> dict:
    """
    Weighted 72-hour directional bias score from −100 (strong bear) to +100 (strong bull).

    Timeframe architecture:
      - df_4h  (4h candles, 30d) → EMA structure, 4h RSI, RSI divergence  [primary 72h anchor]
      - short_df (1h candles, 72h) → MACD, Stoch, CVD, momentum            [medium refinement]
      - df     (daily candles)     → regime detection (ADX), daily coherence [slow anchor]
      - liq    (real-time)         → cascade, hunt zones, OB depth           [short-term, decay-weighted]

    Microstructure signals (OB, cascade, hunt zones) receive horizon-decay multipliers
    because they have effective horizons of hours–1 day, not 72h.
    """
    signals      = {}
    weighted_sum = 0.0
    # Timeframe layers
    _ti   = short_df if (short_df is not None and len(short_df) >= 20) else df   # 1h candles
    _tf4h = df_4h    if (df_4h   is not None and len(df_4h)   >= 20) else None   # 4h candles
    closes       = _ti["Close"].tolist()
    price        = closes[-1]
    analysis     = (liq or {}).get("liq_analysis", {})

    # ── Daily trend direction for coherence filtering ──────────────────────────
    try:
        _d_cls   = df["Close"].tolist()
        _d6ret   = (_d_cls[-1] - _d_cls[-6]) / _d_cls[-6] if len(_d_cls) >= 6 else 0.0
        _daily_dir = +1 if _d6ret > 0.025 else (-1 if _d6ret < -0.025 else 0)
    except Exception:
        _daily_dir = 0
    _COHERENCE_DAMP = 0.50

    # ── Regime detection via ADX on 1h candles ────────────────────────────────
    # ADX > 28 → trend regime (EMAs + momentum reliable; mean-reversion weak)
    # ADX < 18 → range regime (RSI/Stoch/divergence reliable; EMAs lag)
    # 18–28    → transition (use base weights)
    try:
        _adx_1h = calculate_adx(_ti, period=14)
    except Exception:
        _adx_1h = float("nan")
    _adx_ok = not np.isnan(_adx_1h)
    if _adx_ok and _adx_1h > 28:
        regime = "trend"
    elif _adx_ok and _adx_1h < 18:
        regime = "range"
    else:
        regime = "transition"

    # ── Volatility conditioning via ATR percentile on daily candles ───────────
    # ATR percentile > 70 → high vol → dampen mean-reversion signals
    # ATR percentile < 30 → low vol / squeeze → dampen breakout signals
    try:
        _atr_series = calculate_atr(df)
        _atr_pctile = float(_atr_series.rank(pct=True).iloc[-1]) if len(_atr_series) >= 20 else 0.5
    except Exception:
        _atr_pctile = 0.5
    _vol_high  = _atr_pctile > 0.70   # high vol → mean-reversion less reliable
    _vol_low   = _atr_pctile < 0.30   # low vol / squeeze → momentum less reliable

    # ── Regime-switched weights (13 signals, sum to 1.00) ─────────────────────
    # BASE (transition / no clear regime):
    # Weights redesigned for genuine 72h horizon:
    # Structural/daily signals (Polymarket, EMA, 4h RSI, OI, RSI Div) dominate.
    # Fast 1h oscillators (Stoch, Short Momentum) drastically reduced — they
    # describe hours, not days, and cause score churn without predictive value.
    _W_BASE = {
        "Polymarket Sentiment": 0.16,  # forward-looking but slow on BTC — moves rarely
        "EMA Structure":        0.20,  # primary directional anchor — tracks the trend
        "OI Funding":           0.13,  # positioning trap detector; structural
        "4h RSI":               0.13,  # structural momentum; raised to compensate for PM cut
        "RSI Divergence":       0.08,  # reliable reversal signal on 4h
        "Price Structure":      0.05,  # trap/sweep detection (overlaps Swept Reversal)
        "CVD Trend":            0.06,  # buy/sell pressure over 24h
        "Hunt Zone Pull":       0.04,  # liquidity magnet (forward-looking)
        "Swept Reversal":       0.04,  # confirmed hunt completion (reactive)
        "Cascade Direction":    0.03,  # real-time liq fuel ratio
        "MACD Divergence":      0.03,  # structural divergence on 4h
        "Order Book Depth":     0.02,  # very fast-decaying; reduced
        "RSI Level":            0.01,  # 1h signal; noisy for 72h horizon
        "MACD Momentum":        0.01,  # 1h MACD histogram; noisy
        "Short Momentum":       0.01,  # 24h price change; partially redundant with 4h RSI
        "Stochastic Zone":      0.00,  # 1h oscillator; too short for 72h
    }
    # TREND: EMA dominant; mean-reversion signals suppressed; volume/momentum confirm
    _W_TREND = {
        "Polymarket Sentiment": 0.16,
        "EMA Structure":        0.28,  # primary trend anchor
        "OI Funding":           0.13,  # trend-confirming positioning
        "4h RSI":               0.06,  # CUT — mean-reversion fights trend in this regime
        "RSI Divergence":       0.04,
        "Price Structure":      0.05,  # trimmed — overlaps Swept Reversal
        "CVD Trend":            0.08,  # boosted — volume confirms trend
        "Hunt Zone Pull":       0.05,
        "Swept Reversal":       0.03,  # confirmed hunt completion — small in trend (counter-trend bias)
        "Cascade Direction":    0.05,
        "MACD Divergence":      0.02,
        "Order Book Depth":     0.01,
        "RSI Level":            0.01,
        "MACD Momentum":        0.02,  # boosted — momentum confirms trend
        "Short Momentum":       0.01,
        "Stochastic Zone":      0.00,
    }
    # RANGE: 4h RSI + RSI Div lead; EMA suppressed; CVD and traps confirm reversal zones
    _W_RANGE = {
        "Polymarket Sentiment": 0.13,
        "EMA Structure":        0.10,  # lags in ranging market but small bump from PM cut
        "OI Funding":           0.10,
        "4h RSI":               0.18,  # mean-reversion king in range
        "RSI Divergence":       0.13,  # highly reliable in range
        "Price Structure":      0.07,  # trimmed — overlaps Swept Reversal
        "CVD Trend":            0.05,
        "Hunt Zone Pull":       0.05,
        "Swept Reversal":       0.05,  # ranges live & die at hunt zones — strongest here
        "Cascade Direction":    0.03,
        "MACD Divergence":      0.04,
        "Order Book Depth":     0.03,
        "RSI Level":            0.02,
        "MACD Momentum":        0.01,
        "Short Momentum":       0.01,
        "Stochastic Zone":      0.00,
    }
    WEIGHTS = {"trend": _W_TREND, "range": _W_RANGE, "transition": _W_BASE}[regime]

    # ── Horizon-decay: microstructure signals decay fast vs 72h target ────────
    # Order book, cascade, and hunt zones have effective horizons of hours–1 day.
    # We down-weight them relative to their face weight to prevent short-term state
    # from dominating what is claimed to be a 72h forecast.
    # Decay factor = fraction of 72h budget the signal is genuinely predictive over.
    _HORIZON_DECAY = {
        "Order Book Depth":     0.25,  # minutes → almost no 72h predictive value
        "Stochastic Zone":      0.30,  # hours → negligible weight
        "Swept Reversal":       0.40,  # 4–12h reversal confirmation; fades fast
        "Short Momentum":       0.45,  # 24h return; half-life well under 72h
        "Cascade Direction":    0.50,  # hours–1 day
        "Hunt Zone Pull":       0.55,  # positional but reshapes within a day
        "CVD Trend":            0.65,  # 1–2 days
        "RSI Level":            0.65,  # 1h RSI; directional only 12–24h
        "MACD Momentum":        0.75,  # 1–3 days; trend-confirming but lags
        "Price Structure":      0.85,  # 4h traps last 1–3 days
        "RSI Divergence":       0.95,  # 4h divergences persist days
        "MACD Divergence":      0.95,  # 4h divergences persist multiple days
        "OI Funding":           0.95,  # funding cycles are 8h–days level
        "4h RSI":               1.00,  # structural — directly 72h-relevant
        "EMA Structure":        1.00,  # daily — no decay
        "Polymarket Sentiment": 1.00,  # 24h+ crowd consensus — no decay
    }
    _decay_total_before = sum(WEIGHTS.values())
    for _dk, _dd in _HORIZON_DECAY.items():
        if _dk in WEIGHTS:
            WEIGHTS[_dk] *= _dd
    # Renormalise so weights still sum to 1.0 after decay
    _decay_total_after = sum(WEIGHTS.values())
    if _decay_total_after > 0:
        WEIGHTS = {k: v / _decay_total_after for k, v in WEIGHTS.items()}

    # ── Dynamic EMA weight: scale by ADX trend strength ───────────────────────
    # EMA is lagging and unreliable in ranging/compressing markets.
    # trend_strength = 0 when ADX ≤ 18 (fully ranging), 1.0 when ADX ≥ 30 (strong trend).
    # Freed weight is redistributed proportionally to the liquidity+sentiment signals.
    _trend_strength = float(np.clip((_adx_1h - 18) / 12, 0.0, 1.0)) if _adx_ok else 0.6
    if _trend_strength < 0.99:
        _ema_raw = WEIGHTS["EMA Structure"]
        WEIGHTS["EMA Structure"] = _ema_raw * _trend_strength
        _freed = _ema_raw - WEIGHTS["EMA Structure"]
        # Redistribute freed weight to liquidity signals (they don't lag like EMA)
        _redist_keys = ["Hunt Zone Pull", "Cascade Direction", "Order Book Depth",
                        "Polymarket Sentiment", "OI Funding"]
        _redist_total = sum(WEIGHTS[k] for k in _redist_keys)
        if _redist_total > 0:
            for _rk in _redist_keys:
                WEIGHTS[_rk] += _freed * (WEIGHTS[_rk] / _redist_total)
        # Ensure sum == 1.0 (floating point safety)
        _wsum = sum(WEIGHTS.values())
        WEIGHTS = {k: v / _wsum for k, v in WEIGHTS.items()}

    # Weight floor: no signal fully silenced — ADX/regime may zero signals
    # which lets 2-3 hot signals dominate uncontested. Floor at 1% per signal.
    _W_FLOOR = 0.01
    WEIGHTS = {k: max(v, _W_FLOOR) for k, v in WEIGHTS.items()}
    _wsum = sum(WEIGHTS.values())
    WEIGHTS = {k: v / _wsum for k, v in WEIGHTS.items()}

    _SMOOTH_N = 5   # rolling window for noisy liq signals (≈10 min at 2-min refresh)

    # ── Session-state rolling buffers for volatile order-book signals ──────────
    # Cascade ratio and depth ratio are raw order-book snapshots; averaging the
    # last _SMOOTH_N readings prevents a single noisy order-book tick from swinging the score.
    def _smooth_push(key: str, value: float) -> float:
        buf = st.session_state.setdefault(key, [])
        buf.append(value)
        if len(buf) > _SMOOTH_N:
            buf.pop(0)
        return float(np.mean(buf))

    # ── 1. Polymarket Thesis Score (20%) ──────────────────────────
    # Thesis-weighted crowd sentiment: each market scored -10..+10 by question category,
    # normalized to [-1,+1] via thesis_score/10. Confidence dampens signal towards 0.
    try:
        p         = poly or {}
        pm_sig    = float(p.get("signal", 0.0))         # already thesis_score/10
        pm_thesis = float(p.get("thesis_score", pm_sig * 10))
        n_mkts    = int(p.get("markets_used", 0))
        conf      = float(p.get("confidence", 0.0))
        if n_mkts == 0:
            raw, note = 0.0, "Polymarket: no qualifying markets (neutral)"
        else:
            conf_factor = 0.3 + 0.7 * conf              # [0.3, 1.0] — low-confidence markets barely count
            if conf > 0.85:
                # Near-unanimous agreement: steepen the curve so consensus hits harder
                _boost = 1.0 + (conf - 0.85) * 4.0     # 1.0→1.6x as conf goes 85%→100%
                raw = float(np.clip(np.tanh(pm_sig * conf_factor * _boost), -1.0, 1.0))
            else:
                raw = float(np.clip(pm_sig * conf_factor, -1.0, 1.0))
            label       = p.get("thesis_label", "BULL" if raw > 0.05 else ("BEAR" if raw < -0.05 else "NEUTRAL"))
            note        = f"Polymarket {label} ({pm_thesis:+.2f}/10, {n_mkts} mkts, {conf:.0%} agree)"
    except Exception:
        raw, note = 0.0, "Polymarket N/A"
    signals["Polymarket Sentiment"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Polymarket Sentiment"]

    # ── 2. Cascade Direction ───────────────────────────────────────
    # Cascade ratio = long-liq-fuel / short-liq-fuel.
    # HIGH ratio → lots of long-liq fuel below → price likely hunts DOWN.
    try:
        rat_raw  = float(analysis.get("cascade_ratio", 1.0))
        rat      = _smooth_push("_b12_cascade_ratio", rat_raw)
        # Continuous linear interpolation — no cliff-edge jumps at thresholds.
        # Anchor points: 0.4→+1.0, 0.7→+0.55, 1.0→0.0, 1.4→−0.55, 2.0→−1.0
        if   rat <= 0.4:  raw = +1.00
        elif rat <= 0.7:  raw = float(np.interp(rat, [0.4, 0.7],  [+1.00, +0.55]))
        elif rat <= 1.0:  raw = float(np.interp(rat, [0.7, 1.0],  [+0.55,  0.00]))
        elif rat <= 1.4:  raw = float(np.interp(rat, [1.0, 1.4],  [ 0.00, -0.55]))
        elif rat <= 2.0:  raw = float(np.interp(rat, [1.4, 2.0],  [-0.55, -1.00]))
        else:             raw = -1.00
        direction = "UP" if rat < 0.7 else ("DOWN" if rat > 1.4 else "BALANCED")
        note = f"Cascade {direction} (smoothed ratio {rat:.2f}x)"
    except Exception:
        raw, note = 0.0, "Cascade N/A"
    # Coherence filter: dampen cascade when it contradicts the daily trend
    if _daily_dir != 0 and raw != 0 and ((raw < 0 and _daily_dir > 0) or (raw > 0 and _daily_dir < 0)):
        raw  *= _COHERENCE_DAMP
        note += f" [dampened — opposes daily trend]"
    signals["Cascade Direction"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Cascade Direction"]

    # ── 3. Hunt Zone Pull ──────────────────────────────────────────
    # Hunt zones are LIQUIDATION CLUSTERS (not order-book walls):
    #   side="ASK" → short-liq cluster ABOVE price → magnet pulling UP
    #                (price hunts up to trigger forced buys → squeeze)
    #   side="BID" → long-liq cluster BELOW price  → magnet pulling DOWN
    #                (price hunts down to trigger forced sells → flush)
    # Order-book walls only enter via `wall_res`: a real resting wall sitting
    # between price and the cluster dampens the magnet pull (layered resistance).
    # For a 72H horizon, clusters even 0.1% away are legitimate sweep targets.
    try:
        hz            = analysis.get("hunt_zones", [])
        ask_pull      = 0.0
        bid_pull      = 0.0
        MIN_NOTIONAL  = 1_000_000   # $1M minimum — filter noise clusters
        MAX_HUNT_DIST = 0.07        # 7% window — BTC can move ~10% in 72h
        active_zones  = [
            z for z in hz
            if abs(z["price"] - price) / price <= MAX_HUNT_DIST
            and z.get("notional", z.get("wall", 0)) >= MIN_NOTIONAL
        ]
        for zone in active_zones:
            # hunt_score = (notional/1e6) / dist² × cascade_chain_mult — no squaring
            score     = zone.get("hunt_score", 0.0)
            # ⑩ Post-sweep discount: a zone already swept has far less magnet pull.
            # Check if price passed through the zone level in the last 4 candles.
            try:
                _zp      = float(zone["price"])
                _rh4     = _ti["High"].dropna().iloc[-4:]
                _rl4     = _ti["Low"].dropna().iloc[-4:]
                _swept   = any(lo <= _zp <= hi for hi, lo in zip(_rh4, _rl4))
                if _swept:
                    score *= 0.35  # recently swept → 65% pull reduction
            except Exception:
                pass
            # Dampen by any order-book wall sitting between price and the zone.
            # A $20M wall in the path → 50% reduction; $100M → 83% reduction.
            wall_res  = zone.get("wall_res", 0.0)
            wall_damp = 1.0 / (1.0 + wall_res / 2e7)
            strength  = score * wall_damp
            if zone["side"] == "ASK":
                ask_pull += strength
            else:
                bid_pull += strength
        total_pull = ask_pull + bid_pull
        if total_pull > 0:
            raw      = (ask_pull - bid_pull) / total_pull
            ask_pct  = ask_pull / total_pull * 100
            bid_pct  = 100 - ask_pct
            dominant = "Upside" if ask_pull >= bid_pull else "Downside"
            note     = (f"{dominant} hunt pull within 7% "
                        f"({ask_pct:.0f}% short-liq vs {bid_pct:.0f}% long-liq, "
                        f"{len(active_zones)}/{len(hz)} zones ≥$1M)")
        elif hz:
            raw, note = 0.0, "Hunt zones too far or too small for 72h horizon"
        else:
            raw, note = 0.0, "No significant hunt zones detected"
    except Exception:
        raw, note = 0.0, "Hunt zones N/A"
    # Coherence filter: dampen hunt pull when it contradicts the daily trend
    if _daily_dir != 0 and raw != 0 and ((raw < 0 and _daily_dir > 0) or (raw > 0 and _daily_dir < 0)):
        raw  *= _COHERENCE_DAMP
        note += f" [dampened — opposes daily trend]"
    signals["Hunt Zone Pull"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Hunt Zone Pull"]

    # ── 3b. Swept Reversal ─────────────────────────────────────────
    # Confirms hunt completion. When price wicks through a known hunt zone
    # AND closes back on the originating side within the last 1–3 × 4h bars,
    # the liquidations actually triggered and got rejected — the strongest
    # short-term reversal tell. Complements Hunt Zone Pull, which is
    # forward-looking (where price MIGHT go); this is reactive (where it
    # just went and reversed).
    #   +1 → long-liq cluster swept & rejected (longs flushed, bounce → bullish)
    #   -1 → short-liq cluster swept & rejected (shorts squeezed, fade → bearish)
    try:
        hz_all   = analysis.get("hunt_zones", [])
        bull_acc = 0.0
        bear_acc = 0.0
        swept_n  = 0
        if hz_all and _tf4h is not None and len(_tf4h) >= 3:
            _last3   = _tf4h.iloc[-3:]
            _recency = [0.3, 0.6, 1.0]   # oldest → newest
            _cands   = [z for z in hz_all
                        if abs(z["price"] - price) / price <= 0.05
                        and z.get("notional", z.get("wall", 0)) >= 1_000_000]
            for r_idx, (_, row) in enumerate(_last3.iterrows()):
                hi, lo, cl = float(row["High"]), float(row["Low"]), float(row["Close"])
                for z in _cands:
                    zp = float(z["price"])
                    # cap per-zone weight so a single huge cluster can't dominate
                    w  = min(z.get("hunt_score", 0.0), 50_000.0) * _recency[r_idx]
                    if z["side"] == "ASK":
                        # short-liq cluster above: swept-then-rejected = bearish
                        if hi >= zp and cl < zp:
                            bear_acc += w
                            swept_n  += 1
                    else:
                        # long-liq cluster below: swept-then-rejected = bullish
                        if lo <= zp and cl > zp:
                            bull_acc += w
                            swept_n  += 1
        total = bull_acc + bear_acc
        if total > 0:
            direction = (bull_acc - bear_acc) / total
            magnitude = float(np.tanh(total / 30_000.0))
            raw       = direction * magnitude
            tag       = ("Long-flush rejected (bull)" if bull_acc > bear_acc
                         else "Short-squeeze rejected (bear)")
            note      = f"{tag} — {swept_n} sweep(s) in last 12h, mag {magnitude:.2f}"
        else:
            raw, note = 0.0, "No hunt-zone sweeps in last 12h"
    except Exception:
        raw, note = 0.0, "Swept Reversal N/A"
    # No daily-trend coherence damping: counter-trend reversal signals are
    # exactly when this signal is most valuable.
    signals["Swept Reversal"] = (raw, note)
    weighted_sum += raw * WEIGHTS.get("Swept Reversal", 0.0)

    # ── 4. Order Book Depth Ratio ─────────────────────────────────
    # tanh((ratio-1)/0.6) → smooth +1 when bid-heavy, -1 when ask-heavy.
    try:
        ratio_raw = float((liq or {}).get("liq_depth_ratio", 1.0))
        ratio     = _smooth_push("_b12_ob_ratio", ratio_raw)
        raw       = float(np.tanh((ratio - 1.0) / 0.6))
        note      = f"OB ratio {ratio:.2f}x — {'bid' if ratio > 1 else 'ask'}-heavy (tanh {raw:+.2f})"
    except Exception:
        raw, note = 0.0, "OB depth N/A"
    signals["Order Book Depth"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Order Book Depth"]

    # ── 5. RSI Level — uses 1h candles ────────────────────────────
    # Probabilistic tanh scoring: smooth, no cliff-edges at integer thresholds.
    # ⑨ Trend-conditional center: shift neutral from 50 toward 55 in uptrend, 45 in
    # downtrend. RSI=75 is continuation in a bull market, not overbought.
    try:
        rsi_s  = calculate_rsi(_ti["Close"])
        rsi    = float(rsi_s.dropna().iloc[-1])
        rsi_5  = float(rsi_s.dropna().iloc[-5]) if len(rsi_s.dropna()) >= 5 else rsi
        rising = rsi > rsi_5
        slope_bonus = (+0.15 if (rising and rsi < 45) else
                       -0.15 if (not rising and rsi > 55) else
                       +0.08 if rising else -0.08)
        # ⑨ Shift neutral center based on EMA20/EMA50 trend direction
        _rsi_ema_src = _tf4h if _tf4h is not None else _ti
        _r_e20 = float(_rsi_ema_src["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        _r_e50 = float(_rsi_ema_src["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
        _rsi_center = 56.0 if _r_e20 > _r_e50 else (44.0 if _r_e20 < _r_e50 else 50.0)
        base = float(np.tanh((_rsi_center - rsi) / 18.0))
        raw  = max(-1.0, min(1.0, base + slope_bonus))
        if _vol_high and regime == "trend":
            raw *= 0.65   # dampen mean-reversion read during high-vol trending
        note = f"RSI {rsi:.0f} {'↑' if rising else '↓'} (center {_rsi_center:.0f}, tanh {raw:+.2f})"
    except Exception:
        raw, note = 0.0, "RSI N/A"
    signals["RSI Level"] = (raw, note)
    weighted_sum += raw * WEIGHTS["RSI Level"]

    # ── 6. MACD Momentum (8%) — uses 1h candles ───────────────────
    try:
        _, _, hist = calculate_macd(_ti["Close"])
        h_now  = float(hist.dropna().iloc[-1])
        h_prev = float(hist.dropna().iloc[-3]) if len(hist.dropna()) >= 3 else h_now
        if   h_now > 0 and h_now > h_prev: raw, note = +1.0, f"MACD hist positive & accelerating ({h_now:+.0f})"
        elif h_now > 0 and h_now < h_prev: raw, note = +0.4, f"MACD hist positive, decelerating ({h_now:+.0f})"
        elif h_now > 0:                    raw, note = +0.2, f"MACD hist positive, stalling ({h_now:+.0f})"
        elif h_now < 0 and h_now < h_prev: raw, note = -1.0, f"MACD hist negative & accelerating down ({h_now:+.0f})"
        elif h_now < 0 and h_now > h_prev: raw, note = -0.4, f"MACD hist negative, recovering ({h_now:+.0f})"
        elif h_now < 0:                    raw, note = -0.2, f"MACD hist negative, stalling ({h_now:+.0f})"
        else:                              raw, note =  0.0, "MACD hist near zero"
    except Exception:
        raw, note = 0.0, "MACD N/A"
    signals["MACD Momentum"] = (raw, note)
    weighted_sum += raw * WEIGHTS["MACD Momentum"]

    # ── 7. Stochastic Zone — uses 1h candles ──────────────────────
    # Probabilistic tanh: smooth gradient from oversold to overbought.
    # Cross bonus preserved — %K/%D crossover adds directional conviction.
    try:
        st_d   = calculate_stochastic(_ti)
        k      = float(st_d["k"].dropna().iloc[-1])
        d      = float(st_d["d"].dropna().iloc[-1])
        k_prev = float(st_d["k"].dropna().iloc[-2]) if len(st_d["k"].dropna()) >= 2 else k
        d_prev = float(st_d["d"].dropna().iloc[-2]) if len(st_d["d"].dropna()) >= 2 else d
        bull_x = k > d and k_prev <= d_prev
        bear_x = k < d and k_prev >= d_prev
        base   = float(np.tanh((50.0 - k) / 20.0))
        cross_bonus = (+0.25 if bull_x and k < 40 else
                       -0.25 if bear_x and k > 60 else
                       +0.12 if bull_x else
                       -0.12 if bear_x else 0.0)
        raw    = max(-1.0, min(1.0, base + cross_bonus))
        if _vol_high and regime == "trend":
            raw *= 0.65
        note   = f"Stoch %K {k:.0f}/%D {d:.0f} {'↗ bull cross' if bull_x else '↘ bear cross' if bear_x else ''} (tanh {raw:+.2f})"
    except Exception:
        raw, note = 0.0, "Stoch N/A"
    signals["Stochastic Zone"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Stochastic Zone"]

    # ── 8. 24h Momentum ───────────────────────────────────────────
    # tanh((ret24)/4) → ±1 at ±8% move; smooth gradient through neutral.
    # Damped in low-vol squeeze (momentum less reliable pre-expansion).
    try:
        p24   = closes[-24] if len(closes) >= 24 else closes[0]
        ret24 = (price - p24) / p24 * 100
        raw   = float(np.tanh(ret24 / 4.0))
        if _vol_low:
            raw *= 0.60   # squeeze → momentum signal unreliable
        note  = f"24h momentum {ret24:+.1f}% (tanh {raw:+.2f})"
    except Exception:
        raw, note = 0.0, "Momentum N/A"
    signals["Short Momentum"] = (raw, note)
    weighted_sum += raw * WEIGHTS["Short Momentum"]

    # ── 9. EMA Structure — 4h candles preferred, 1h fallback ────────
    # 4h EMA20 spans ~80h, EMA50 spans ~200h — genuinely predictive for 72h.
    # 1h EMA20 only spans ~20h; too reactive to be a 72h structural anchor.
    try:
        _ema_src = _tf4h if _tf4h is not None else _ti
        _tf_label = "4h" if _tf4h is not None else "1h"
        ema20   = _ema_src["Close"].ewm(span=20, adjust=False).mean()
        ema50   = _ema_src["Close"].ewm(span=50, adjust=False).mean()
        e20_now  = float(ema20.iloc[-1]); e20_prev  = float(ema20.iloc[-2]) if len(ema20) >= 2 else e20_now
        e50_now  = float(ema50.iloc[-1]); e50_prev  = float(ema50.iloc[-2]) if len(ema50) >= 2 else e50_now
        bull_x   = e20_now > e50_now and e20_prev <= e50_prev
        bear_x   = e20_now < e50_now and e20_prev >= e50_prev
        e20_slope = (e20_now - float(ema20.iloc[-3])) / e20_now * 100 if len(ema20) >= 3 else 0
        if   bull_x:            raw, note = +1.0, f"EMA20/50 bull cross [{_tf_label}] ({e20_now:,.0f} vs {e50_now:,.0f})"
        elif bear_x:            raw, note = -1.0, f"EMA20/50 bear cross [{_tf_label}] ({e20_now:,.0f} vs {e50_now:,.0f})"
        elif e20_now > e50_now:
            bonus = min(0.4, abs(e20_slope) * 20)
            raw   = min(1.0, 0.5 + bonus)
            note  = f"EMA20 above EMA50 [{_tf_label}], slope {e20_slope:+.3f}%/bar"
        else:
            bonus = min(0.4, abs(e20_slope) * 20)
            raw   = max(-1.0, -0.5 - bonus)
            note  = f"EMA20 below EMA50 [{_tf_label}], slope {e20_slope:+.3f}%/bar"
    except Exception:
        raw, note = 0.0, "EMA N/A"
    signals["EMA Structure"] = (raw, note)
    weighted_sum += raw * WEIGHTS["EMA Structure"]

    # ── 10. RSI Divergence — 4h preferred, 1h fallback ───────────
    # 4h divergences span multi-day swing structure — far more reliable for 72h.
    # 1h divergences are often noise in a strong trend.
    try:
        _div_src   = _tf4h if _tf4h is not None else _ti
        _div_label = "4h" if _tf4h is not None else "1h"
        _rsi_div_s = calculate_rsi(_div_src["Close"])
        _div       = detect_rsi_divergence(_div_src["Close"].tolist(), _rsi_div_s, order=3)
        if   _div == "bull": raw, note = +1.0, f"Bullish RSI divergence [{_div_label}] (price ↓ low, RSI ↑ low)"
        elif _div == "bear": raw, note = -1.0, f"Bearish RSI divergence [{_div_label}] (price ↑ high, RSI ↓ high)"
        else:                raw, note =  0.0, f"No RSI divergence [{_div_label}]"
    except Exception:
        raw, note = 0.0, "RSI div N/A"
    signals["RSI Divergence"] = (raw, note)
    weighted_sum += raw * WEIGHTS["RSI Divergence"]

    # ── 10c. MACD Divergence — 4h preferred, 1h fallback (structural reversal) ──
    try:
        _mdiv_src   = _tf4h if _tf4h is not None else _ti
        _mdiv_label = "4h" if _tf4h is not None else "1h"
        _, _, _mhist = calculate_macd(_mdiv_src["Close"])
        _mdiv = detect_macd_divergence(_mdiv_src["Close"].tolist(), _mhist, order=3)
        if   _mdiv == "bull": raw, note = +1.0, f"Bullish MACD divergence [{_mdiv_label}] (price ↓ low, MACD hist ↑ low)"
        elif _mdiv == "bear": raw, note = -1.0, f"Bearish MACD divergence [{_mdiv_label}] (price ↑ high, MACD hist ↓ high)"
        else:                 raw, note =  0.0, f"No MACD divergence [{_mdiv_label}]"
    except Exception:
        raw, note = 0.0, "MACD div N/A"
    signals["MACD Divergence"] = (raw, note)
    weighted_sum += raw * WEIGHTS.get("MACD Divergence", 0.0)

    # ── 10b. 4h RSI — structural momentum anchor ──────────────────
    # RSI on 4h measures exhaustion over ~3-day windows — directly relevant to 72h.
    # This is separate from the 1h RSI (signal 5) which reads short-term momentum.
    # Only computed when 4h data is available; otherwise signal is muted.
    if _tf4h is not None:
        try:
            _rsi4h_s = calculate_rsi(_tf4h["Close"])
            _rsi4h   = float(_rsi4h_s.dropna().iloc[-1])
            _rsi4h_p = float(_rsi4h_s.dropna().iloc[-3]) if len(_rsi4h_s.dropna()) >= 3 else _rsi4h
            _rising4h = _rsi4h > _rsi4h_p
            _slope_b4h = (+0.12 if (_rising4h and _rsi4h < 50) else
                          -0.12 if (not _rising4h and _rsi4h > 50) else
                          +0.06 if _rising4h else -0.06)
            # Trend-conditional center: mirrors 1h RSI logic. In downtrend (EMA20 < EMA50),
            # RSI=37 is not truly oversold — shift neutral zone from 50 → 44.
            _4h_bull = e20_now > e50_now
            _4h_bear = e20_now < e50_now
            _4h_rsi_center = 56.0 if _4h_bull else (44.0 if _4h_bear else 50.0)
            raw  = max(-1.0, min(1.0, float(np.tanh((_4h_rsi_center - _rsi4h) / 18.0)) + _slope_b4h))
            note = f"4h RSI {_rsi4h:.0f} {'↑' if _rising4h else '↓'} (center {_4h_rsi_center:.0f}, tanh {raw:+.2f}) — structural"
        except Exception:
            raw, note = 0.0, "4h RSI N/A"
    else:
        raw, note = 0.0, "4h RSI: data unavailable"
    signals["4h RSI"] = (raw, note)
    weighted_sum += raw * WEIGHTS.get("4h RSI", 0.0)

    # ── 11. Price Structure — traps, sweeps, failed breaks ────────
    # Primary: trap/sweep detection on 4h candles (multi-day structure).
    # Secondary: same on 1h (intraday confirmation).
    # Bonus: TA-Lib named patterns if available (tiebreaker only).
    # Bull trap / bear trap / sweep candles are the highest-alpha output here.
    try:
        _ps_scores = []
        _ps_notes  = []

        # 4h trap detection (primary — structural, 72h-relevant)
        if _tf4h is not None and len(_tf4h) >= 24:
            _cat4h, _sc4h, _nt4h = detect_price_traps(_tf4h, lookback=20)
            if _cat4h != "none":
                _ps_scores.append(_sc4h * 1.0)   # full weight for 4h signal
                _ps_notes.append(f"[4h] {_nt4h}")

        # 1h trap/sweep detection (secondary — confirms or contradicts 4h)
        if len(_ti) >= 24:
            _cat1h, _sc1h, _nt1h = detect_price_traps(_ti, lookback=24)
            if _cat1h != "none":
                _ps_scores.append(_sc1h * 0.6)   # 60% weight for 1h signal
                _ps_notes.append(f"[1h] {_nt1h}")

        # TA-Lib named patterns (bonus — low weight, complements trap signal)
        _talib_bonus = 0.0
        if TALIB_AVAILABLE and len(_ti) >= 10:
            try:
                _o = _ti["Open"].values.astype(float)
                _h = _ti["High"].values.astype(float)
                _l = _ti["Low"].values.astype(float)
                _c = _ti["Close"].values.astype(float)
                _bull_hits = _bear_hits = 0.0
                for fn in PATTERNS_BULLISH.values():
                    res = getattr(talib, fn)(_o, _h, _l, _c)
                    if res[-1] > 0:   _bull_hits += 1.0
                    elif res[-2] > 0: _bull_hits += 0.5
                for fn in PATTERNS_BEARISH.values():
                    res = getattr(talib, fn)(_o, _h, _l, _c)
                    if res[-1] < 0:   _bear_hits += 1.0
                    elif res[-2] < 0: _bear_hits += 0.5
                _total_hits = _bull_hits + _bear_hits
                if _total_hits > 0:
                    _talib_bonus = (_bull_hits - _bear_hits) / _total_hits * 0.3
                    _ps_notes.append(f"Patterns: {_bull_hits:.0f}B/{_bear_hits:.0f}bear")
            except Exception:
                pass

        if _ps_scores:
            # Blend: traps dominate, TA-Lib is a nudge
            raw = max(-1.0, min(1.0, float(np.mean(_ps_scores)) + _talib_bonus))
        elif _talib_bonus != 0.0:
            raw = max(-1.0, min(1.0, _talib_bonus))
        else:
            raw = 0.0

        note = " · ".join(_ps_notes) if _ps_notes else "No traps, sweeps, or patterns detected"

    except Exception:
        raw, note = 0.0, "Price structure N/A"
    signals["Price Structure"] = (raw, note)
    weighted_sum += raw * WEIGHTS.get("Price Structure", WEIGHTS.get("Candlestick", 0.0))

    # ── 12. CVD (Cumulative Volume Delta) — on 1h candles ────────────
    # CVD = net buyer/seller pressure estimated from OHLCV bar positions.
    # More reliable than OBV on BTC: OBV is fragmented across venues + noisy.
    # Divergences (CVD rising while price falls, or vice versa) are strongest signal.
    try:
        _cvd_cat, raw = calculate_cvd(_ti, lookback=24)
        if   _cvd_cat == "bull_divergence":  raw = min(+1.0, abs(raw) + 0.3); note = f"CVD: hidden buying — buyers active vs falling price (strong bull div, {raw:+.2f})"
        elif _cvd_cat == "bear_divergence":  raw = max(-1.0, -abs(raw) - 0.3); note = f"CVD: hidden selling — sellers active into rally (strong bear div, {raw:+.2f})"
        elif _cvd_cat == "accumulation":     note = f"CVD: net buy pressure (CVD↑ with price↑, {raw:+.2f})"
        elif _cvd_cat == "distribution":     note = f"CVD: net sell pressure (CVD↓ with price↓, {raw:+.2f})"
        else:                                raw = 0.0; note = "CVD: neutral — balanced buy/sell pressure"
    except Exception:
        raw, note = 0.0, "CVD N/A"
    signals["CVD Trend"] = (raw, note)
    weighted_sum += raw * WEIGHTS["CVD Trend"]

    # ── 13. OI / Funding Regime ────────────────────────────────────
    # price↑ + OI↑ = trend continuation (bull); price↑ + OI↓ = short covering (weaker)
    # flat price + rising OI = squeeze brewing; high positive funding = overleveraged longs
    # Funding tanh: +0.01% rate (neutral) → 0; +0.05% (crowded longs) → -0.8 (bearish)
    try:
        _oi = oi_data or {}
        _fr  = _oi.get("funding_rate", None)
        _oi24 = _oi.get("oi_24h_delta_pct", None)
        if _fr is None and _oi24 is None:
            raw, note = 0.0, "OI/Funding: N/A"
        else:
            # Funding component: crowded longs (high +funding) = bearish signal
            # tanh centered at 0.01% (typical neutral) with scale 0.025%
            _fr_val  = _fr if _fr is not None else 0.0
            _fr_comp = float(-np.tanh((_fr_val - 0.0001) / 0.00025))
            # OI delta component: rising OI = conviction, falling = fading
            _oi_comp = float(np.tanh(_oi24 / 5.0)) if _oi24 is not None else 0.0
            # Price-OI divergence: if OI falling while price rising → weaker bull
            _p1h  = closes[-1]
            _pm1h = closes[-2] if len(closes) >= 2 else _p1h
            _price_rising = _p1h > _pm1h
            if _oi24 is not None and _price_rising and _oi24 < -2:
                _oi_comp *= 0.5   # short-covering rally, not conviction
            raw  = max(-1.0, min(1.0, 0.5 * _fr_comp + 0.5 * _oi_comp))
            _fr_pct = _fr_val * 100 if _fr is not None else None
            note = (f"Funding {_fr_pct:+.4f}%/8h" if _fr_pct is not None else "Funding N/A") + \
                   (f", OI 24h {_oi24:+.1f}%" if _oi24 is not None else ", OI N/A") + \
                   f" (raw {raw:+.2f})"
    except Exception:
        raw, note = 0.0, "OI/Funding N/A"
    signals["OI Funding"] = (raw, note)
    weighted_sum += raw * WEIGHTS["OI Funding"]

    # ── Signal family normalization — cap each family before summing ──────────
    # Prevents correlated signals from compounding false confidence.
    # Each family is capped at ±1 effective unit; the individual weights still apply.
    def _family_cap(keys: list, cap: float = 1.0) -> None:
        total = sum(signals[k][0] * WEIGHTS[k] for k in keys if k in signals)
        max_possible = sum(WEIGHTS[k] for k in keys if k in signals)
        if max_possible == 0:
            return
        utilization = total / max_possible  # how much of max the family is using
        if abs(utilization) > cap:
            scale = cap / abs(utilization)
            for k in keys:
                if k in signals:
                    signals[k] = (signals[k][0] * scale, signals[k][1])

    _family_cap(["EMA Structure", "MACD Momentum"],                           cap=0.95)
    _family_cap(["RSI Level", "Stochastic Zone", "RSI Divergence", "4h RSI", "MACD Divergence"],  cap=0.95)
    _family_cap(["Hunt Zone Pull", "Swept Reversal", "Cascade Direction", "Order Book Depth"],  cap=0.95)
    _family_cap(["CVD Trend", "OI Funding"],                                  cap=0.95)

    # Recompute weighted_sum after family normalization
    weighted_sum = sum(sig * WEIGHTS.get(name, 0) for name, (sig, _) in signals.items())

    # ⑫ Signal dispersion — std of raw signal values measures genuine consensus.
    # Low std = all signals agree → real conviction.
    # High std = half at +1, half at -1 → misleading average → flag as weak.
    _sig_vals   = [v[0] for v in signals.values()]
    _sig_std    = float(np.std(_sig_vals)) if _sig_vals else 0.5
    # Normalize: std=0 → conviction=1.0, std=0.7 (max for ±1 distribution) → conviction≈0
    _conviction = round(max(0.0, 1.0 - _sig_std / 0.70), 2)

    # ── Final score — raw, clipped to [-100, +100]. Smoothing happens once
    # at the gauge layer (α=0.50) so behavior is consistent across local app
    # and headless cron (where st.session_state is stubbed). Previously this
    # block applied a second α=0.65 EMA that compounded with the gauge EMA
    # locally but was a no-op in cron, contaminating logged data.
    _raw_score = max(-100.0, min(100.0, weighted_sum * 100))
    score = round(_raw_score, 1)

    if   score >=  70: label, color = "STRONG BULL", "#3fb950"
    elif score >=  40: label, color = "BULL BIAS",   "#58a6ff"
    elif score >=  15: label, color = "MILD BULL",   "#8bc4ff"
    elif score >  -15: label, color = "NEUTRAL",     "#8b949e"
    elif score >  -40: label, color = "MILD BEAR",   "#ffb347"
    elif score >  -70: label, color = "BEAR BIAS",   "#f0883e"
    else:              label, color = "STRONG BEAR",  "#f85149"

    # ── Probabilistic outputs ─────────────────────────────────────────────────
    # bull_prob: logistic transform of score, calibrated so score=0 → 50%, ±60 → ~80/20%
    # NOTE: the /30 scaling is currently a hand-tuned heuristic. Once enough
    # resolved signals accumulate, this should be replaced with a logistic
    # regression fit on (score → actual 72h direction) — i.e. Brier-calibrated.
    bull_prob     = round(100 / (1 + np.exp(-score / 30)), 1)
    # expected_move: realized-vol-based 72h range, split into directional halves
    try:
        # Realized 72h volatility: std of all 3-day rolling returns in the lookback window.
        # More accurate than ATR×3 because BTC vol clusters — ATR×3 overestimates in calm periods.
        _cls_daily  = df["Close"].dropna()
        _ret_72h    = _cls_daily.pct_change(3).dropna() * 100   # 3-day % returns
        _vol_72h    = float(_ret_72h.std()) if len(_ret_72h) >= 10 else 3.0
        _price_now  = float(_cls_daily.iloc[-1])
        expected_range    = round(_price_now * _vol_72h / 100, 0)
        # Legacy single-direction expected_move (kept for back-compat with UI)
        expected_move     = round(expected_range * (score / 100) * 0.7, 0)
        # ── Magnitude split (Phase 1.1) ──────────────────────────────────────
        # Up/down expected moves derived from bull_prob, so they don't collapse
        # at score=0 the way direction-weighted expected_move does. At 50/50
        # they're both ~half of expected_range; as bull_prob climbs, upside
        # expands and downside contracts.
        _bp_dec               = bull_prob / 100.0
        expected_move_up      = round(expected_range * _bp_dec, 0)
        expected_move_down    = round(expected_range * (1 - _bp_dec), 0)
    except Exception:
        expected_move      = None
        expected_range     = None
        expected_move_up   = None
        expected_move_down = None
        _vol_72h           = None

    # ── Volatility regime overlay (Phase 1.2) ─────────────────────────────────
    # Independent of the trend/range/transition classifier. Tells you how
    # MUCH movement to expect, not which way. Helps filter false signals
    # (a +30 score in compression is noise; the same +30 in expansion is real).
    try:
        # Bollinger-band width percentile on daily candles (20-period, 2σ)
        _bb_mid   = df["Close"].rolling(20).mean()
        _bb_std   = df["Close"].rolling(20).std()
        _bb_width = (_bb_std * 4) / _bb_mid     # full width = 2σ above + 2σ below, normalised to price
        _bb_width = _bb_width.dropna()
        _bb_pct   = float(_bb_width.rank(pct=True).iloc[-1]) if len(_bb_width) >= 20 else 0.5
    except Exception:
        _bb_pct   = 0.5
    try:
        # Realized-vol percentile (vs the lookback window itself)
        _rv_pct   = float(_ret_72h.rank(pct=True).abs().iloc[-1]) if _vol_72h is not None and len(_ret_72h) >= 20 else 0.5
    except Exception:
        _rv_pct   = 0.5
    # Combine: average of ATR + BB-width + realized-vol percentiles → vol_score [0,1]
    _vol_score = float(np.mean([_atr_pctile, _bb_pct, _rv_pct]))
    if   _vol_score >= 0.90: vol_regime = "panic"
    elif _vol_score >= 0.70: vol_regime = "expansion"
    elif _vol_score <= 0.30: vol_regime = "compression"
    else:                    vol_regime = "normal"

    # ── Composite confidence (Phase 1.3) ──────────────────────────────────────
    # Combines four orthogonal signals of how much to trust today's score:
    #   1. Signal agreement (existing _conviction)
    #   2. Data-source quality (Coinglass real > Binance synthetic > N/A)
    #   3. Regime stability (transition regime is less stable than trend/range)
    #   4. Volatility regime (panic regimes are inherently low-confidence)
    _liq_src         = (liq or {}).get("liq_map_source", "N/A")
    _data_quality    = (
        1.00 if "Coinglass"   in str(_liq_src)
        else 0.85 if "Binance"   in str(_liq_src)
        else 0.70                            # N/A or unknown
    )
    _regime_stability = (
        1.00 if regime in ("trend", "range")
        else 0.80                            # transition = less stable
    )
    _vol_stability    = (
        0.70 if vol_regime == "panic"        # extreme vol → low confidence
        else 0.95 if vol_regime == "compression"  # squeeze → slightly low
        else 1.00
    )
    composite_confidence = round(
        max(0.0, min(1.0,
            _conviction * _data_quality * _regime_stability * _vol_stability
        )), 2
    )

    return {"score": score, "label": label, "color": color,
            "signals": signals, "weights": WEIGHTS,
            "regime": regime, "adx_1h": round(_adx_1h, 1) if _adx_ok else None,
            "atr_percentile": round(_atr_pctile * 100, 0),
            "bull_prob": bull_prob,
            "expected_move": expected_move,
            "expected_range": expected_range,
            # Phase 1.1 — directional magnitudes
            "expected_move_up":   expected_move_up,
            "expected_move_down": expected_move_down,
            # Phase 1.2 — volatility regime overlay
            "vol_regime":         vol_regime,
            "vol_score":          round(_vol_score, 3),
            "bb_width_pct":       round(_bb_pct * 100, 0),
            "realized_vol_pct":   round(_rv_pct * 100, 0),
            # Phase 1.3 — composite confidence
            "composite_confidence": composite_confidence,
            "data_quality":         round(_data_quality, 2),
            "regime_stability":     round(_regime_stability, 2),
            "vol_stability":        round(_vol_stability, 2),
            # Legacy fields kept for back-compat
            "conviction": _conviction,
            "signal_std": round(_sig_std, 3)}


def compute_24h_bias(df: pd.DataFrame, btc_liq: dict = None,
                     short_df: pd.DataFrame = None,
                     oi_data: dict = None,
                     crypto_sig: dict = None,
                     poly: dict = None) -> dict:
    """
    24-hour directional bias with full ADX regime switching (④).
    Distinct from 72h engine:
      - RSI(7) vs RSI(14), Fast MACD (6,13,4) vs standard (12,26,9)
      - Liquidation imbalance and candle close position are unique to this engine
      - CVD uses 6h window with rolling-std normalization (⑬)
      - Microstructure signals (OB, CVD, liq) carry more weight (⑭)
      - Polymarket daily thesis re-added at light weight (⑥)
    """
    tf = short_df if (short_df is not None and not short_df.empty) else df

    # ④ Full ADX regime detection — same pattern as 72h engine
    try:
        _adx24 = calculate_adx(tf, period=14)
        _adx24_ok = not np.isnan(_adx24)
    except Exception:
        _adx24 = float("nan"); _adx24_ok = False
    if _adx24_ok and _adx24 > 25:
        _regime24 = "trend"
    elif _adx24_ok and _adx24 < 15:
        _regime24 = "range"
    else:
        _regime24 = "transition"
    _mr = 0.45 if _regime24 == "trend" else (1.10 if _regime24 == "range" else 1.0)

    # ⑭ Weight tables: microstructure signals get boosted vs structural ones.
    # OB depth, CVD, liq imbalance, and momentum matter most over 24h.
    # RSI/Stoch get less weight than in 72h since mean-reversion over 24h is noisy.
    _W24_BASE = {
        "RSI7":              0.07,
        "Stochastic":        0.04,
        "Fast MACD":         0.10,
        "Momentum":          0.11,
        "Candle Pos":        0.07,
        "Order Book":        0.06,   # trimmed for Nearby Liq Magnet
        "Funding":           0.11,
        "CVD":               0.14,
        "Liq Imbalance":     0.14,   # trimmed for Nearby Liq Magnet
        "Nearby Liq Magnet": 0.04,   # NEW — projected liq clusters within ±2.5%
        "EMA Align":         0.08,
        "Polymarket":        0.03,
        "Fear Greed":        0.01,
    }
    _W24_TREND = {
        "RSI7":              0.04, "Stochastic":        0.02, "Fast MACD":     0.12,
        "Momentum":          0.14, "Candle Pos":        0.06, "Order Book":    0.04,
        "Funding":           0.09, "CVD":               0.13, "Liq Imbalance": 0.16,
        "Nearby Liq Magnet": 0.03, "EMA Align":         0.10, "Polymarket":    0.03,
        "Fear Greed":        0.04,
    }
    _W24_RANGE = {
        "RSI7":              0.12, "Stochastic":        0.09, "Fast MACD":     0.07,
        "Momentum":          0.08, "Candle Pos":        0.07, "Order Book":    0.05,
        "Funding":           0.10, "CVD":               0.13, "Liq Imbalance": 0.13,
        "Nearby Liq Magnet": 0.06, "EMA Align":         0.04, "Polymarket":    0.03,
        "Fear Greed":        0.03,
    }
    _W24 = {"trend": _W24_TREND, "range": _W24_RANGE, "transition": _W24_BASE}[_regime24]
    # Normalise weights to sum to 1.0
    _w24_sum = sum(_W24.values())
    _W24 = {k: v / _w24_sum for k, v in _W24.items()}
    # Weight floor: no signal silenced below 1%
    _W24 = {k: max(v, 0.01) for k, v in _W24.items()}
    _w24_sum = sum(_W24.values())
    _W24 = {k: v / _w24_sum for k, v in _W24.items()}

    _sigs24 = {}

    # 1. RSI(7) — faster than 72h engine's RSI(14)
    try:
        _delta = tf["Close"].diff()
        _gain  = _delta.clip(lower=0).ewm(com=6, adjust=False).mean()
        _loss  = (-_delta).clip(lower=0).ewm(com=6, adjust=False).mean()
        _rsi7  = float((100 - 100 / (1 + _gain / (_loss + 1e-9))).dropna().iloc[-1])
        _sigs24["RSI7"] = float(np.tanh((50.0 - _rsi7) / 10.0)) * _mr
    except Exception:
        _sigs24["RSI7"] = 0.0

    # 2. Stochastic %K/%D with cross bonus
    try:
        _st = calculate_stochastic(tf)
        k = float(_st["k"].dropna().iloc[-1])
        d = float(_st["d"].dropna().iloc[-1])
        s = float(np.tanh((50.0 - k) / 12.0))
        cross = 0.40 if (k > d and k < 35) else (-0.40 if (k < d and k > 65) else 0.0)
        _sigs24["Stochastic"] = (s + cross * (1.0 - abs(s))) * _mr
    except Exception:
        _sigs24["Stochastic"] = 0.0

    # 3. Fast MACD (6,13,4) — intraday momentum vs 72h's standard (12,26,9)
    try:
        _ema6  = tf["Close"].ewm(span=6,  adjust=False).mean()
        _ema13 = tf["Close"].ewm(span=13, adjust=False).mean()
        _fmacd = _ema6 - _ema13
        _fsig  = _fmacd.ewm(span=4, adjust=False).mean()
        _fhist = (_fmacd - _fsig).dropna()
        h_now  = float(_fhist.iloc[-1])
        h_prev = float(_fhist.iloc[-2]) if len(_fhist) >= 2 else h_now
        h_mean = float(_fhist.abs().mean()) or 1.0
        s = float(np.tanh(h_now / h_mean))
        accel = (0.30 if (h_now > h_prev > 0 or h_now > 0 > h_prev) else
                -0.30 if (h_now < h_prev < 0 or h_now < 0 < h_prev) else 0.0)
        _sigs24["Fast MACD"] = s * 0.70 + accel * 0.30
    except Exception:
        _sigs24["Fast MACD"] = 0.0

    # 4. Short-horizon momentum: 1h/4h only (24h momentum belongs to 72h engine)
    try:
        _cl = tf["Close"].dropna()
        c0  = float(_cl.iloc[-1])
        c1  = float(_cl.iloc[-2]) if len(_cl) >= 2 else c0
        c4  = float(_cl.iloc[-5]) if len(_cl) >= 5 else c0
        mom = np.tanh((c0 - c1) / c1 * 100 / 0.5) * 0.60 + np.tanh((c0 - c4) / c4 * 100 / 1.2) * 0.40
        _sigs24["Momentum"] = float(mom)
    except Exception:
        _sigs24["Momentum"] = 0.0

    # 5. EMA Align — structural trend anchor on 1h candles (EMA8 vs EMA21)
    # ±0.5% spread → ±1.0 signal. Bullish when EMA8 > EMA21, bearish when below.
    try:
        _ema8_24  = tf["Close"].ewm(span=8,  adjust=False).mean()
        _ema21_24 = tf["Close"].ewm(span=21, adjust=False).mean()
        _e8  = float(_ema8_24.iloc[-1])
        _e21 = float(_ema21_24.iloc[-1])
        _sigs24["EMA Align"] = float(np.tanh((_e8 - _e21) / _e21 * 100 / 0.5))
    except Exception:
        _sigs24["EMA Align"] = 0.0

    # 6. Candle close position — where last 3 bars closed within their H/L range
    try:
        _hi  = tf["High"].dropna().iloc[-3:]
        _lo  = tf["Low"].dropna().iloc[-3:]
        _cl3 = tf["Close"].dropna().iloc[-3:]
        _rng = (_hi - _lo).replace(0, np.nan)
        _pos = ((_cl3 - _lo) / _rng - 0.5) * 2
        _wts = np.array([0.20, 0.30, 0.50])[:len(_pos)]; _wts /= _wts.sum()
        _sigs24["Candle Pos"] = float(np.dot(_pos.fillna(0).values, _wts))
    except Exception:
        _sigs24["Candle Pos"] = 0.0

    # 6. Order book depth ratio — smoothed over last 5 readings to cut noise
    try:
        bid = btc_liq.get("total_bid_depth", 0) if btc_liq else 0
        ask = btc_liq.get("total_ask_depth", 0) if btc_liq else 0
        _raw_ob24 = float(np.tanh((bid / (ask + 1e-9) - 1.0) / 0.35)) if bid + ask > 0 else 0.0
        _ob24_buf = st.session_state.setdefault("_ob24_buf", [])
        _ob24_buf.append(_raw_ob24)
        if len(_ob24_buf) > 5:
            _ob24_buf.pop(0)
        _sigs24["Order Book"] = float(np.mean(_ob24_buf))
    except Exception:
        _sigs24["Order Book"] = 0.0

    # 7. Funding rate — center at 0.01% (same neutral as 72h engine)
    try:
        fr = oi_data.get("funding_rate") if oi_data else None
        _sigs24["Funding"] = float(-np.tanh((fr - 0.0001) / 0.00025)) if fr is not None else 0.0
    except Exception:
        _sigs24["Funding"] = 0.0

    # 8. ⑬ CVD with rolling-std normalization (not arbitrary fixed divisor)
    try:
        _tf6h = tf.iloc[-7:] if len(tf) >= 7 else tf   # ~6h of 1h candles
        _, _cvd_raw = calculate_cvd(_tf6h)
        # Rolling 20-period std for calibration — adapts to current volume conditions
        _cvd_series = []
        for _i in range(max(1, len(tf) - 19), len(tf) + 1):
            try:
                _, _c = calculate_cvd(tf.iloc[max(0, _i - 7): _i])
                _cvd_series.append(_c)
            except Exception:
                pass
        _cvd_std = float(np.std(_cvd_series)) if len(_cvd_series) >= 3 else 30.0
        _sigs24["CVD"] = float(np.tanh(_cvd_raw / (_cvd_std + 1e-9)))
    except Exception:
        _sigs24["CVD"] = 0.0

    # 9. Liquidation imbalance — UNIQUE to 24h engine
    try:
        import time as _time
        _liq_ev = st.session_state.get(_LIQ_EVENTS_KEY, {})
        _sigs24["Liq Imbalance"] = 0.0
        if _liq_ev:
            _cutoff = (_time.time() - 4 * 3600) * 1000
            _ll = _sl = 0.0
            for _ts_ms, _ev_list in _liq_ev.items():
                if float(_ts_ms) < _cutoff:
                    continue
                for _ev in (_ev_list if isinstance(_ev_list, list) else [_ev_list]):
                    _side = str(_ev.get("side", "")).upper()
                    _notl = float(_ev.get("notional", 0))
                    if _side in ("SELL", "SHORT"):
                        _ll += _notl
                    elif _side in ("BUY", "LONG"):
                        _sl += _notl
            _tl = _ll + _sl
            if _tl > 0:
                _sigs24["Liq Imbalance"] = float(np.tanh((_sl - _ll) / _tl / 0.4))
    except Exception:
        _sigs24["Liq Imbalance"] = 0.0

    # 9b. Nearby Liq Magnet — projected liquidation clusters within ±2.5%
    # Forward-looking complement to Liq Imbalance (which is reactive — measures
    # liquidations that ALREADY fired). This signal reads the same synthetic /
    # Coinglass map that feeds the 72h Hunt Zone Pull, but with an intraday
    # window and tighter notional floor.
    #   +1 → short-liq cluster above pulls UP (squeeze magnet)
    #   -1 → long-liq cluster below pulls DOWN (flush magnet)
    try:
        _hz24       = (btc_liq or {}).get("liq_analysis", {}).get("hunt_zones", [])
        _ask_pull24 = 0.0
        _bid_pull24 = 0.0
        _px24       = float((btc_liq or {}).get("liq_current_price")
                            or tf["Close"].iloc[-1])
        for _z in _hz24:
            _zp   = float(_z.get("price", 0))
            if _zp <= 0:
                continue
            _dist = abs(_zp - _px24) / _px24
            if _dist > 0.025 or _dist < 1e-4:
                continue
            _ntl = float(_z.get("notional", _z.get("wall", 0)))
            if _ntl < 500_000:
                continue
            # Linear 1/dist weighting — intraday distance matters but doesn't
            # dominate as steeply as the 72h's 1/dist² treatment.
            _strength = (_ntl / 1e6) / _dist
            if _z.get("side") == "ASK":
                _ask_pull24 += _strength
            else:
                _bid_pull24 += _strength
        _tot24 = _ask_pull24 + _bid_pull24
        if _tot24 > 0:
            _dir24 = (_ask_pull24 - _bid_pull24) / _tot24
            _mag24 = float(np.tanh(_tot24 / 200.0))
            _sigs24["Nearby Liq Magnet"] = _dir24 * _mag24
        else:
            _sigs24["Nearby Liq Magnet"] = 0.0
    except Exception:
        _sigs24["Nearby Liq Magnet"] = 0.0

    # 10. ⑥ Polymarket daily thesis — light weight, avoids heavy correlation with 72h
    # Uses thesis_score (range -10..+10) normalised to -1..+1. Only applies when
    # ≥1 qualifying market found and confidence > 20% (thin markets excluded).
    try:
        _pm = poly or {}
        # 24h engine uses signal_24h — horizon-weighted toward same-day questions
        _pm_sig  = float(_pm.get("signal_24h", _pm.get("signal", 0.0)))
        _pm_conf = float(_pm.get("confidence", 0.0))
        _pm_n    = int(_pm.get("markets_used", 0))
        if _pm_n >= 1 and _pm_conf > 0.20:
            _cf24 = 0.2 + 0.8 * _pm_conf
            if _pm_conf > 0.85:
                _boost24 = 1.0 + (_pm_conf - 0.85) * 4.0
                _sigs24["Polymarket"] = float(np.clip(np.tanh(_pm_sig * _cf24 * _boost24), -1.0, 1.0))
            else:
                _sigs24["Polymarket"] = float(np.clip(_pm_sig * _cf24, -1.0, 1.0))
        else:
            _sigs24["Polymarket"] = 0.0
    except Exception:
        _sigs24["Polymarket"] = 0.0

    # 11. Fear & Greed — only at panic (<20) or euphoria (>80) extremes ⑤
    try:
        fg = crypto_sig.get("fear_greed_value") if crypto_sig else None
        if isinstance(fg, (int, float)) and (fg < 20 or fg > 80):
            _sigs24["Fear Greed"] = float(np.tanh((50.0 - fg) / 15.0))
        else:
            _sigs24["Fear Greed"] = 0.0
    except Exception:
        _sigs24["Fear Greed"] = 0.0

    # Weighted sum
    _ws24 = sum(_sigs24.get(k, 0.0) * w for k, w in _W24.items())
    score_100 = float(np.clip(_ws24 * 100, -100, 100))

    if   score_100 >= 40: lab, col = "BULLISH",   "#3fb950"
    elif score_100 >= 20: lab, col = "MILD BULL", "#58a6ff"
    elif score_100 > -20: lab, col = "NEUTRAL",   "#8b949e"
    elif score_100 > -40: lab, col = "MILD BEAR", "#f0883e"
    else:                 lab, col = "BEARISH",   "#f85149"

    return {"score": score_100, "label": lab, "color": col,
            "regime": _regime24, "adx": round(_adx24, 1) if _adx24_ok else None,
            "signals": _sigs24, "weights": _W24}


def _fetch_btc_liquidity(current_price=None):
    d     = _fetch_depth(current_price)
    ratio = d["ratio"]
    if   ratio >= 2.0:  bias, stren = "BID", "strong"
    elif ratio >= 1.4:  bias, stren = "BID", "moderate"
    elif ratio >= 1.15: bias, stren = "BID", "mild"
    elif ratio <= 0.5:  bias, stren = "ASK", "strong"
    elif ratio <= 0.7:  bias, stren = "ASK", "moderate"
    elif ratio <= 0.85: bias, stren = "ASK", "mild"
    else:               bias, stren = "BALANCED", "neutral"
    sadj = (+1 if bias == "BID" and stren in ("strong", "moderate") else
            -1 if bias == "ASK" and stren in ("strong", "moderate") else 0)

    # ── Liquidation map for hunt-zone / cascade engines ──────────────────────
    # Priority: Coinglass (paid) → Hyblock (paid) → Binance synthetic (free)
    # When any is available we feed REAL liquidation levels into _analyze_liq
    # instead of using order-book bid_c/ask_c as a proxy.
    _liq_map     = None
    _liq_map_src = "orderbook_proxy"
    if d["price"] > 0:
        _liq_map = _fetch_coinglass_liq_map()
        if _liq_map:
            _liq_map_src = "coinglass"
        else:
            _liq_map = _fetch_hyblock_liq_map()
            if _liq_map:
                _liq_map_src = "hyblock"
            else:
                _liq_map = _fetch_binance_synthetic_liqmap(d["price"])
                if _liq_map:
                    _liq_map_src = _liq_map.get("source", "binance_synthetic")

    # If we have a real liq map, build {price: usd} dicts that _analyze_liq's
    # cascade/hunt-zone engines can consume in place of order-book clusters.
    # long-liq levels (below price) → bid_c slot; short-liq (above) → ask_c slot.
    if _liq_map:
        _bidc_for_engine = {float(p): float(u) for p, u in _liq_map.get("long",  [])}
        _askc_for_engine = {float(p): float(u) for p, u in _liq_map.get("short", [])}
    else:
        _bidc_for_engine = d.get("bid_c", {})
        _askc_for_engine = d.get("ask_c", {})

    analysis = {}
    if d["bid_walls"] or d["ask_walls"] or d["raw_bids"] or d["raw_asks"]:
        lo   = d["price"] * (1 - WALL_RANGE_PCT / 100)
        hi   = d["price"] * (1 + WALL_RANGE_PCT / 100)
        grid = _snap_grid(lo, hi, HEATMAP_BIN)
        # Build actual liquidation heat arrays from order-book positions (same
        # data the visual heatmap uses) so clusters are real, not bid/ask walls.
        try:
            _, la, sa, _ = _build_liq(
                d["raw_bids"], d["raw_asks"], d["price"], grid, ratio)
        except Exception:
            la = sa = None
        analysis = _analyze_liq(la, sa, grid, d["price"], d["bid_walls"], d["ask_walls"],
                                bid_c=_bidc_for_engine, ask_c=_askc_for_engine)
        analysis["liq_map_source"] = _liq_map_src

    result = {
        "liq_bid_walls": d["bid_walls"], "liq_ask_walls": d["ask_walls"],
        "liq_bid_clusters": d["bid_c"],  "liq_ask_clusters": d["ask_c"],
        "liq_raw_bids": d["raw_bids"],   "liq_raw_asks": d["raw_asks"],
        "liq_current_price": d["price"],
        "liq_depth_ratio": ratio,
        "liq_total_bid": d["bid_usd"],   "liq_total_ask": d["ask_usd"],
        "liq_spread_bps": d["spread"],
        "liq_bias": bias, "liq_bias_strength": stren,
        "liq_source": d["source"], "liq_score_adj": sadj,
        "liq_cg_available": _liq_map_src != "orderbook_proxy",
        "liq_map_source":   _liq_map_src,
        "liq_analysis": analysis,
    }
    result["liq_narrative"] = _narrative(result, analysis)
    parts = []
    for p, n in d["bid_walls"][:1]: parts.append(f"{_fprice(p)} bid ({_fusd(n)})")
    for p, n in d["ask_walls"][:1]: parts.append(f"{_fprice(p)} ask ({_fusd(n)})")
    parts.append(f"{ratio:.1f}x {bias}")
    cd = analysis.get("cascade_direction", "")
    if cd and cd != "BALANCED": parts.append(f"cascade {cd}")
    hz = analysis.get("hunt_zones", [])
    if hz: parts.append(f"{len(hz)} hunt zone(s)")
    result["liq_signal"] = "  |  ".join(parts)
    return result


# ════════════════════════════════════════════════════════════════
#  COLORMAPS
# ════════════════════════════════════════════════════════════════

def _cg_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("cg", [
        (0.00, "#0d0221"), (0.04, "#150935"), (0.10, "#1f1360"),
        (0.16, "#3b1578"), (0.22, "#6b2090"), (0.28, "#8b2080"),
        (0.34, "#4b3fa0"), (0.41, "#1b6ea0"), (0.48, "#1b8a8a"),
        (0.56, "#2eb872"), (0.65, "#5cc840"), (0.74, "#7dd71d"),
        (0.83, "#a8e30a"), (0.92, "#d4f006"), (1.00, "#ffff00"),
    ], N=512)

def _cmap_bids():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bids", ["#001133", "#003388", "#0055cc", "#0088ff", "#44ccff"], N=256)

def _cmap_asks():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("asks", ["#220000", "#880011", "#cc2200", "#ff4400", "#ff9944"], N=256)


def _fetch_binance_klines(interval: str, limit: int, symbol: str = "BTCUSDT"):
    """Single source of truth for BTC OHLCV — Binance spot klines.
    Returns DataFrame with UTC DatetimeIndex and Open/High/Low/Close/Volume columns,
    or None on failure. The most recent row is the currently-forming candle (live)."""
    try:
        import datetime as _dt
        url  = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = _get_raw(url)
        if not data:
            return None
        timestamps, rows = [], []
        for k in data:
            timestamps.append(_dt.datetime.fromtimestamp(k[0] / 1000, tz=_dt.timezone.utc))
            rows.append({
                "Open":   float(k[1]),
                "High":   float(k[2]),
                "Low":    float(k[3]),
                "Close":  float(k[4]),
                "Volume": float(k[5]),
            })
        return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps))
    except Exception:
        return None


def _fetch_short_term_ohlcv():
    """1h BTC candles for last 72 hours (used for 72h bias computation)."""
    df = _fetch_binance_klines("1h", 72)
    if df is not None and len(df) >= 10:
        return df
    # yfinance fallback if Binance unreachable
    try:
        import datetime as _dt
        end   = _dt.datetime.now(_dt.timezone.utc)
        start = end - _dt.timedelta(hours=72)
        df    = yf.Ticker("BTC-USD").history(start=start, end=end, interval="1h")
        if df is not None and len(df) >= 10:
            df.index = pd.to_datetime(df.index)
            return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return None


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_4h_ohlcv():
    """4h BTC candles for last 30 days — primary timeframe for 72h structural signals.
    4h EMA20/50 span ~80h/200h, which is far more appropriate for a 72h forecast
    than 1h equivalents that only span ~20h/50h."""
    df = _fetch_binance_klines("4h", 180)
    if df is not None and len(df) >= 20:
        return df
    # yfinance fallback
    try:
        import datetime as _dt
        end   = _dt.datetime.now(_dt.timezone.utc)
        start = end - _dt.timedelta(days=30)
        df    = yf.Ticker("BTC-USD").history(start=start, end=end, interval="4h")
        if df is not None and len(df) >= 20:
            df.index = pd.to_datetime(df.index)
            return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_oi_funding() -> dict:
    """Fetch Binance futures open interest + funding rate. Returns dict or empty dict on failure."""
    result = {}
    try:
        oi_data = _get_raw("https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT")
        if isinstance(oi_data, dict) and "openInterest" in oi_data:
            result["open_interest"] = float(oi_data["openInterest"])
            result["open_interest_usd"] = float(oi_data.get("openInterestValue", 0))
    except Exception:
        pass
    try:
        # Historical OI (last 30 periods, 1h) for delta
        hist = _get_raw(
            "https://fapi.binance.com/futures/data/openInterestHist"
            "?symbol=BTCUSDT&period=1h&limit=30"
        )
        if isinstance(hist, list) and len(hist) >= 2:
            oi_vals = [float(r["sumOpenInterestValue"]) for r in hist if "sumOpenInterestValue" in r]
            if len(oi_vals) >= 2:
                result["oi_hist"] = oi_vals
                result["oi_1h_delta_pct"] = (oi_vals[-1] - oi_vals[-2]) / oi_vals[-2] * 100
                result["oi_24h_delta_pct"] = (oi_vals[-1] - oi_vals[0])  / oi_vals[0]  * 100
    except Exception:
        pass
    try:
        # Last 8 funding rates
        fr_data = _get_raw(
            "https://fapi.binance.com/fapi/v1/fundingRate"
            "?symbol=BTCUSDT&limit=8"
        )
        if isinstance(fr_data, list) and fr_data:
            rates = [float(r["fundingRate"]) for r in fr_data if "fundingRate" in r]
            if rates:
                result["funding_rate"]       = rates[-1]
                result["funding_rate_avg8h"] = float(np.mean(rates))
    except Exception:
        pass
    return result


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_15min_ohlcv():
    """~5 days of 15-min BTC/USDT candles from Binance (free, no API key).
    We fetch 500 candles (~5 days) so the heat matrix captures positions opened
    well above/below current price; only the last 24h (96 candles) is displayed."""
    return _fetch_binance_klines("15m", 500)


def _build_liq_heat_2d(plot_df, grid, bsz, tiers=None):
    """
    Build a 2D (n_prices × n_candles) liquidation heat matrix.
    For each historical candle, project where leveraged positions opened at that
    price would get liquidated, weighted by candle volume. Levels swept by price
    are drained (those positions got stopped out). This produces time-accurate
    heat accumulation that matches Coinglass's approach.
    """
    if tiers is None:
        tiers = [10, 15, 20, 25, 50, 75, 100]
    n_prices  = len(grid)
    n_candles = len(plot_df)
    g_lo      = grid[0]

    # 50x/75x/100x get highest weight — they project to near-price clusters
    # (~1-2% from current price) which are the dominant yellow bands in Coinglass.
    lev_weights = {10: 0.8, 15: 0.6, 20: 1.0, 25: 0.9, 50: 2.0, 75: 1.8, 100: 1.5}

    closes   = plot_df["Close"].values.astype(float)
    highs    = plot_df["High"].values.astype(float)
    lows     = plot_df["Low"].values.astype(float)
    vols     = plot_df["Volume"].values.astype(float)
    vol_max  = vols.max() or 1.0
    vol_norm = 0.2 + 0.8 * (vols / vol_max)  # keep low-vol candles at 0.2 min

    cumulative = np.zeros(n_prices)
    liq_heat   = np.zeros((n_prices, n_candles))

    for t in range(n_candles):
        cp  = closes[t]
        vol = vol_norm[t]

        for lev in tiers:
            lw = lev_weights.get(lev, 0.5)

            lp_long = cp * (1.0 - 1.0 / lev)
            idx = int(round((lp_long - g_lo) / bsz))
            if 0 <= idx < n_prices:
                cumulative[idx] += vol * lw

            lp_short = cp * (1.0 + 1.0 / lev)
            idx = int(round((lp_short - g_lo) / bsz))
            if 0 <= idx < n_prices:
                cumulative[idx] += vol * lw

        # Positions at levels swept by this candle got liquidated
        swept = (grid >= lows[t]) & (grid <= highs[t])
        cumulative[swept] *= 0.55

        liq_heat[:, t] = cumulative

    # No blur — keep bands at their exact projected price levels so the bar
    # chart renderer shows discrete clusters rather than a smeared gradient.
    return liq_heat


def _build_orderbook_liq_heat(bid_c, ask_c, cprice, grid, bsz, n_candles):
    """
    Build a 2D liquidation heat matrix from the LIVE ORDER BOOK.

    More accurate than OHLCV-based projection because the order book directly
    shows where trading activity is concentrated right now.  Large bid walls at
    price P → many long positions entered near P → strong liquidation cluster at
    P*(1 - 1/lev) for each leverage tier.  Equivalent to the vsching open-source
    liquidation-heatmap approach.

    Returns (n_prices × n_candles) matrix where each row is constant across the
    time axis (snapshot) with a slight left-fade for visual depth.
    """
    n_prices = len(grid)
    g_lo     = grid[0]

    # Same leverage tiers as lev-math but weighted toward high-lev (near-price clusters)
    tiers = [(10, 0.7), (15, 0.6), (20, 0.9), (25, 0.8), (50, 2.2), (75, 2.0), (100, 1.8)]

    liq_density = np.zeros(n_prices)

    all_vals = list(bid_c.values()) + list(ask_c.values())
    if not all_vals:
        return np.zeros((n_prices, n_candles))
    max_val = max(all_vals) or 1.0

    # Bids at price P → long positions → liquidate BELOW P
    for bp, bv in bid_c.items():
        if bp <= 0 or bv <= 0:
            continue
        w = (bv / max_val) ** 0.6   # compress outliers so mid-size walls still show
        for lev, lw in tiers:
            lp  = bp * (1.0 - 1.0 / lev)
            idx = int(round((lp - g_lo) / bsz))
            if 0 <= idx < n_prices:
                liq_density[idx] += w * lw

    # Asks at price P → short positions → liquidate ABOVE P
    for ap, av in ask_c.items():
        if ap <= 0 or av <= 0:
            continue
        w = (av / max_val) ** 0.6
        for lev, lw in tiers:
            lp  = ap * (1.0 + 1.0 / lev)
            idx = int(round((lp - g_lo) / bsz))
            if 0 <= idx < n_prices:
                liq_density[idx] += w * lw

    # Broadcast 1D snapshot to 2D with a slight left-fade for visual depth
    weights = np.linspace(0.45, 1.0, n_candles)
    heat    = liq_density[:, np.newaxis] * weights[np.newaxis, :]
    return heat


# ── Real liquidation event heatmap ──────────────────────────────────────────
# Fetches actual forced-liquidation orders from Binance + Bybit and accumulates
# them in session state. Yellow clusters = where real cascades happened.

_LIQ_EVENTS_KEY = "_liq_events"
_LIQ_MAX_EVENTS = 10_000


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_hyblock_liq_map():
    """
    Fetch BTC liquidation levels from Hyblock Capital (free tier available).
    Requires HYBLOCK_CLIENT_ID, HYBLOCK_CLIENT_SECRET, HYBLOCK_API_KEY in secrets.toml.
    """
    import os, requests as _req

    def _sec(k):
        try:
            v = st.secrets.get(k, "") or ""
        except Exception:
            v = ""
        return v or os.environ.get(k, "")

    client_id     = _sec("HYBLOCK_CLIENT_ID")
    client_secret = _sec("HYBLOCK_CLIENT_SECRET")
    api_key       = _sec("HYBLOCK_API_KEY")
    if not (client_id and client_secret and api_key):
        return None

    try:
        # ── Step 1: OAuth2 token ─────────────────────────────────────────────
        # Authorization: Basic base64(client_id:client_secret)  ← per Hyblock docs
        # x-api-key in header, grant_type in body as form data
        tok = _req.post(
            "https://api.hyblockcapital.com/v2/oauth2/token",
            auth=(client_id, client_secret),   # sets Authorization: Basic base64(id:secret)
            headers={"x-api-key": api_key},
            data={"grant_type": "client_credentials"},  # requests sets Content-Type automatically
            timeout=10,
        )
        tok.raise_for_status()
        token = tok.json()["access_token"]

        # ── Step 2: Fetch price-level liquidation data ────────────────────────
        liq = _req.get(
            "https://api.hyblockcapital.com/v2/liquidationLevels",
            headers={"Authorization": f"Bearer {token}", "x-api-key": api_key},
            params={"coin": "btc", "exchange": "binance"},
            timeout=12,
        )
        liq.raise_for_status()
        raw  = liq.json()
        data = raw.get("data", raw)

        # ── Step 3: Parse — store raw response in session state for debugging ─
        st.session_state["_hyblock_last_raw"] = str(raw)[:500]

        def _parse_levels(candidates):
            for key in candidates:
                items = data.get(key) or []
                if not items:
                    continue
                out = []
                for item in items:
                    if isinstance(item, dict):
                        p = float(item.get("price", item.get("level", item.get("priceLevel", 0))))
                        u = float(item.get("amount", item.get("size", item.get("usdAmount", 0))))
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        p, u = float(item[0]), float(item[1])
                    else:
                        continue
                    if p > 0 and u > 0:
                        out.append((p, u))
                if out:
                    return out
            return []

        longs  = _parse_levels(["longLevels", "longLiquidationLevels", "long", "longs", "buyLevels"])
        shorts = _parse_levels(["shortLevels", "shortLiquidationLevels", "short", "shorts", "sellLevels"])

        if not longs and not shorts:
            return None

        all_usd = [u for _, u in longs + shorts]
        max_usd = max(all_usd) if all_usd else 1.0
        return {"long": longs, "short": shorts, "max_usd": max_usd, "source": "hyblock"}

    except Exception as e:
        st.session_state["_hyblock_err"] = str(e)
        return None


# ════════════════════════════════════════════════════════════════════════════
#  BINANCE SYNTHETIC LIQUIDATION MAP — free, no API key
# ════════════════════════════════════════════════════════════════════════════
#
# Reconstructs a Coinglass-style long/short liquidation heatmap from public
# Binance futures-data endpoints. Methodology:
#
#   1. Walk 1h bars over the last ~20 days.
#   2. For each bar, OI delta = OI[t] - OI[t-1] (USD notional change).
#   3. When OI rises → new positions opened at this bar's midprice.
#      Split between longs/shorts using the taker-buy ratio (closest live
#      proxy for new directional flow).
#   4. When OI falls → close oldest tranches first (FIFO) until the drop
#      is absorbed. Survivors = the position book that's still open.
#   5. For each surviving tranche, fan its notional across realistic
#      Binance leverage tiers (3x/5x/10x/25x/50x/100x with empirical
#      retail+pro distribution) and compute the liquidation price for
#      each tier using Binance's ~0.4% maintenance margin floor.
#   6. Apply soft time decay (7-day half-life) — positions still open
#      after weeks are typically reduced or hedged.
#   7. Bin all (liq_price, weighted_notional) pairs into the same grid
#      Coinglass returns.
#
# Limits we're honest about: Binance only (~40-50% of BTC perp OI), no
# margin/spot leverage. Leverage tier weights are empirically calibrated to
# match observed Coinglass cluster shapes but cannot replicate exact strikes.

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_binance_oi_hist(period: str = "1h", limit: int = 480):
    """Open interest history. Returns list of (ts_ms, oi_btc, oi_usd) or None.
    Free, no auth. Binance returns max ~500 entries; 480 × 1h ≈ 20 days."""
    url = ("https://fapi.binance.com/futures/data/openInterestHist"
           f"?symbol=BTCUSDT&period={period}&limit={limit}")
    data = _get_raw(url, timeout=10)
    if not isinstance(data, list) or not data:
        return None
    out = []
    for row in data:
        try:
            ts  = int(row["timestamp"])
            btc = float(row["sumOpenInterest"])
            usd = float(row["sumOpenInterestValue"])
            if btc > 0 and usd > 0:
                out.append((ts, btc, usd))
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_binance_taker_ls(period: str = "1h", limit: int = 480):
    """Taker buy/sell ratio. Returns {ts_ms: buy_ratio_0_1}.
    buy_ratio = buyVol / (buyVol + sellVol). Best live proxy for the
    directional flow that drove each bar's OI delta."""
    url = ("https://fapi.binance.com/futures/data/takerlongshortRatio"
           f"?symbol=BTCUSDT&period={period}&limit={limit}")
    data = _get_raw(url, timeout=10)
    if not isinstance(data, list) or not data:
        return None
    out = {}
    for row in data:
        try:
            ts = int(row["timestamp"])
            bv = float(row["buyVol"])
            sv = float(row["sellVol"])
            tot = bv + sv
            if tot > 0:
                out[ts] = bv / tot
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_binance_ls_ratio(period: str = "1h", limit: int = 480):
    """Global account long/short ratio. Fallback when taker ratio is sparse."""
    url = ("https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
           f"?symbol=BTCUSDT&period={period}&limit={limit}")
    data = _get_raw(url, timeout=10)
    if not isinstance(data, list) or not data:
        return None
    out = {}
    for row in data:
        try:
            ts = int(row["timestamp"])
            la = float(row["longAccount"])
            # longAccount is fraction in [0,1] — directly usable
            if 0 < la < 1:
                out[ts] = la
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


# Empirical Binance BTC perp leverage distribution. Tier weights calibrated
# against observed Coinglass cluster intensity ratios across multiple
# regime snapshots (bull, range, post-flush). Sum to 1.00.
_LEV_TIERS = [
    (3,   0.10),   # conservative whales, hedgers
    (5,   0.18),   # mid-leverage
    (10,  0.30),   # most common retail
    (25,  0.24),   # aggressive retail
    (50,  0.12),   # gambler tier
    (100, 0.06),   # YOLO / liquidation farming bait
]
# Binance BTCUSDT perp maintenance margin floor (effective fraction)
_MAINT_MARGIN = 0.004


def _liq_price_long(entry: float, lev: int) -> float:
    """Long liquidation: entry × (1 − 1/lev + maint_margin × something).
    Simplified Binance formula: liq ≈ entry × (1 − 1/lev × (1 − maint))."""
    return entry * (1.0 - (1.0 - _MAINT_MARGIN) / lev)


def _liq_price_short(entry: float, lev: int) -> float:
    """Short liquidation: mirror of long."""
    return entry * (1.0 + (1.0 - _MAINT_MARGIN) / lev)


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_binance_synthetic_liqmap(current_price: float):
    """
    Build a Coinglass-style liquidation heatmap from free Binance data.
    Returns dict matching the _fetch_coinglass_liq_map shape:
        {"long":  [(price, usd), ...],
         "short": [(price, usd), ...],
         "max_usd": float,
         "source": "binance_synthetic"}
    or None if data is unavailable.
    """
    try:
        oi_hist = _fetch_binance_oi_hist("1h", 480)
        if not oi_hist or len(oi_hist) < 24:
            return None
        taker = _fetch_binance_taker_ls("1h", 480) or {}
        lsr   = _fetch_binance_ls_ratio("1h", 480) or {}

        # We also need a per-bar entry price. The OI hist row gives
        # sumOpenInterestValue (USD) and sumOpenInterest (BTC); their ratio
        # is the average mark price during that bar — exactly what we want
        # for "where positions were opened".
        # tranches: list of {"entry": price, "long_usd": x, "short_usd": x, "ts": ms}
        tranches: list = []
        prev_usd = None

        for ts, oi_btc, oi_usd in oi_hist:
            mid_price = oi_usd / oi_btc if oi_btc > 0 else current_price
            if prev_usd is None:
                prev_usd = oi_usd
                continue
            delta_usd = oi_usd - prev_usd

            if delta_usd > 0:
                # New positions opened. Allocate by taker buy ratio.
                # If taker data missing, fall back to longAccount fraction.
                if ts in taker:
                    long_frac = taker[ts]
                elif ts in lsr:
                    long_frac = lsr[ts]
                else:
                    long_frac = 0.5
                # Clip to [0.2, 0.8] — extreme ratios are noisy and would
                # produce single-sided liq maps that don't match reality.
                long_frac = max(0.2, min(0.8, long_frac))
                tranches.append({
                    "entry":     mid_price,
                    "long_usd":  delta_usd * long_frac,
                    "short_usd": delta_usd * (1.0 - long_frac),
                    "ts":        ts,
                })
            elif delta_usd < 0:
                # Positions closed. Bleed FIFO from oldest tranches first.
                close_usd = -delta_usd
                # Each tranche has long + short components; close them
                # proportionally (we don't know which side actually closed,
                # so prorate by tranche composition).
                i = 0
                while close_usd > 0 and i < len(tranches):
                    tr = tranches[i]
                    tr_total = tr["long_usd"] + tr["short_usd"]
                    if tr_total <= 0:
                        i += 1
                        continue
                    take = min(close_usd, tr_total)
                    ratio = take / tr_total
                    tr["long_usd"]  -= tr["long_usd"]  * ratio
                    tr["short_usd"] -= tr["short_usd"] * ratio
                    close_usd       -= take
                    if tr["long_usd"] + tr["short_usd"] < 1.0:
                        i += 1     # tranche exhausted
                    else:
                        break
                # Drop fully-exhausted tranches
                tranches = [t for t in tranches if (t["long_usd"] + t["short_usd"]) >= 1.0]

            prev_usd = oi_usd

        if not tranches:
            return None

        # Apply 7-day half-life time decay
        now_ms       = oi_hist[-1][0]
        HALF_LIFE_MS = 7 * 24 * 3600 * 1000

        # Bin to $50 grid (same as order-book heatmap)
        BIN  = 50.0
        long_map: dict  = {}
        short_map: dict = {}

        for tr in tranches:
            age_ms = max(0, now_ms - tr["ts"])
            decay  = 0.5 ** (age_ms / HALF_LIFE_MS)
            entry  = tr["entry"]
            for lev, w in _LEV_TIERS:
                # Long liq price + weighted notional
                if tr["long_usd"] > 0:
                    lp = _liq_price_long(entry, lev)
                    if lp > 0:
                        b = round(lp / BIN) * BIN
                        long_map[b] = long_map.get(b, 0.0) + tr["long_usd"] * w * decay
                # Short liq price
                if tr["short_usd"] > 0:
                    sp = _liq_price_short(entry, lev)
                    if sp > 0:
                        b = round(sp / BIN) * BIN
                        short_map[b] = short_map.get(b, 0.0) + tr["short_usd"] * w * decay

        # Filter: longs by definition liquidate BELOW current price, shorts ABOVE.
        # Liq prices that ended up on the wrong side of current price are
        # tranches that are already underwater / already liquidated — drop them.
        longs  = [(p, u) for p, u in long_map.items()  if p < current_price * 0.999]
        shorts = [(p, u) for p, u in short_map.items() if p > current_price * 1.001]

        if not longs and not shorts:
            return None

        # Sort by intensity desc for top-cluster display
        longs.sort(key=lambda x: x[1], reverse=True)
        shorts.sort(key=lambda x: x[1], reverse=True)

        all_usd = [u for _, u in longs + shorts]
        max_usd = max(all_usd) if all_usd else 1.0

        return {
            "long":    longs,
            "short":   shorts,
            "max_usd": max_usd,
            "source":  "binance_synthetic",
        }
    except Exception as e:
        st.session_state["_bn_synth_err"] = str(e)
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _fetch_coinglass_liq_map():
    """
    Fetch BTC liquidation heatmap from Coinglass API (requires COINGLASS_API_KEY
    in .streamlit/secrets.toml or env var).  Returns dict with:
      "long":  [(price, usd_amount), ...]  — levels below current price
      "short": [(price, usd_amount), ...]  — levels above current price
      "max_usd": float                     — largest single level
    Returns None if no API key or fetch fails.
    """
    import os, json
    api_key = ""
    try:
        api_key = st.secrets.get("COINGLASS_API_KEY", "") or ""
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("COINGLASS_API_KEY", "")
    if not api_key:
        return None

    try:
        import urllib.request as _ur
        url = ("https://open-api.coinglass.com/public/v2/liquidation_map"
               "?ex=Binance&pair=BTCUSDT&interval=12h")
        req = _ur.Request(url, headers={
            "coinglassSecret": api_key,
            "Content-Type":    "application/json",
        })
        with _ur.urlopen(req, timeout=12) as resp:
            raw = json.loads(resp.read().decode())

        # Accept both {"success":true,...} and {"code":"0",...} shapes
        if not (raw.get("success") or raw.get("code") == "0"):
            return None

        data = raw.get("data", raw)  # some versions nest, some don't

        # Parse long / short maps — handle several key-name variants
        def _parse(key_candidates):
            for k in key_candidates:
                v = data.get(k)
                if v:
                    # list of {price, amount} or list of [price, amount]
                    out = []
                    for item in v:
                        if isinstance(item, dict):
                            p = float(item.get("price", 0))
                            a = float(item.get("amount", item.get("liqAmount", 0)))
                        else:
                            p, a = float(item[0]), float(item[1])
                        if p > 0 and a > 0:
                            out.append((p, a))
                    if out:
                        return out
            return []

        longs  = _parse(["longLiquidationMap",  "long",  "longs",  "buyMap"])
        shorts = _parse(["shortLiquidationMap", "short", "shorts", "sellMap"])
        if not longs and not shorts:
            return None

        all_usd  = [a for _, a in longs + shorts]
        max_usd  = max(all_usd) if all_usd else 1.0
        return {"long": longs, "short": shorts, "max_usd": max_usd}

    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_liq_events():
    """Fetch recent BTC liquidation events from Binance + Bybit (both free, no API key).
    Binance returns up to 1000 events per call; Bybit returns every liquidation (not
    filtered to largest-per-second like Binance), giving better coverage."""
    out = []

    # ── Binance forced-liquidation orders (last 24h, up to 1000) ────────────
    try:
        import time as _time
        start_ms = int((_time.time() - 86400) * 1000)
        url  = (f"https://fapi.binance.com/fapi/v1/allForceOrders"
                f"?symbol=BTCUSDT&limit=1000&startTime={start_ms}")
        data = _get_raw(url)
        if isinstance(data, list):
            for o in data:
                try:
                    price = float(o.get("averagePrice") or o.get("price") or 0)
                    qty   = float(o.get("executedQty") or 0)
                    side  = str(o.get("side", ""))
                    ts_ms = int(o.get("time") or 0)
                    if price > 0 and qty > 0 and ts_ms > 0:
                        out.append({"price": price, "notional": price * qty,
                                    "side": side, "ts_ms": ts_ms,
                                    "src": "binance"})
                except Exception:
                    pass
    except Exception:
        pass

    # ── Bybit liquidations (most recent 200 — emits ALL liquidations) ────────
    try:
        url  = ("https://api.bybit.com/v5/market/liquidation"
                "?category=linear&symbol=BTCUSDT&limit=200")
        data = _get_raw(url)
        if isinstance(data, dict):
            items = data.get("result", {}).get("list", [])
            for o in items:
                try:
                    price = float(o.get("price", 0))
                    size  = float(o.get("size", 0))
                    # Bybit: side="Buy" means a short position was liquidated (forced buy)
                    side  = "BUY" if o.get("side") == "Buy" else "SELL"
                    ts_ms = int(o.get("time", 0))
                    if price > 0 and size > 0 and ts_ms > 0:
                        out.append({"price": price, "notional": price * size,
                                    "side": side, "ts_ms": ts_ms + 1,  # +1 avoids ts collision
                                    "src": "bybit"})
                except Exception:
                    pass
    except Exception:
        pass

    return out


def _push_liq_events(events):
    """Merge new liq events into session state (ts_ms key deduplicates)."""
    if not events:
        return
    store = st.session_state.setdefault(_LIQ_EVENTS_KEY, {})
    for e in events:
        store[e["ts_ms"]] = e
    if len(store) > _LIQ_MAX_EVENTS:
        for k in sorted(store)[:-_LIQ_MAX_EVENTS]:
            del store[k]


def _build_liq_event_heat_2d(plot_df, grid, bsz):
    """
    Build a 2D (n_prices × n_candles) heatmap from real liquidation events.
    Each event is placed in the 15-min candle column that contains its timestamp.
    Returns (matrix, raw_max_notional).
    """
    store     = st.session_state.get(_LIQ_EVENTS_KEY, {})
    n_prices  = len(grid)
    n_candles = len(plot_df)
    g_lo      = grid[0]
    heat      = np.zeros((n_prices, n_candles))
    if not store:
        return heat, 0.0

    # Pre-compute candle start times in ms for fast lookup
    candle_ms = []
    for ct in plot_df.index:
        try:
            ts = ct.timestamp() * 1000
        except Exception:
            import datetime as _dt
            ts = _dt.datetime.fromtimestamp(float(ct) / 1e9).timestamp() * 1000
        candle_ms.append(int(ts))

    interval_ms = 15 * 60 * 1000

    for e in store.values():
        ts   = e["ts_ms"]
        # Binary-search: find the last candle whose start ≤ event timestamp
        lo, hi, t = 0, n_candles - 1, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if candle_ms[mid] <= ts:
                t = mid; lo = mid + 1
            else:
                hi = mid - 1
        if t < 0 or ts > candle_ms[t] + interval_ms:
            continue  # event outside display window
        idx = int(round((e["price"] - g_lo) / bsz))
        if 0 <= idx < n_prices:
            heat[idx, t] += e["notional"]

    # Light time-axis blur only — keeps bar shape, removes single-candle spikes
    try:
        from scipy.ndimage import gaussian_filter
        heat = gaussian_filter(heat, sigma=(0.4, 0.8))
    except Exception:
        pass

    raw_max = float(heat.max()) if heat.max() > 0 else 0.0
    return heat, raw_max


_OB_HISTORY_KEY = "_ob_history"
_OB_MAX_SNAPS   = 600   # ~20h at 2-min refresh cadence

def _push_ob_snapshot(raw_bids, raw_asks, cprice):
    """Append a timestamped, pre-binned order-book snapshot to session state."""
    import datetime as _dt
    snaps = st.session_state.setdefault(_OB_HISTORY_KEY, [])
    now = _dt.datetime.now(_dt.timezone.utc)
    # Skip if last push was within 90 s (same cache window re-render)
    if snaps and (now - snaps[-1]["ts"]).total_seconds() < 90:
        return
    bsz = HEATMAP_BIN
    bd, ad = {}, {}
    for ps, qs in raw_bids:
        p = round(float(ps) / bsz) * bsz
        bd[p] = bd.get(p, 0.0) + float(ps) * float(qs)
    for ps, qs in raw_asks:
        p = round(float(ps) / bsz) * bsz
        ad[p] = ad.get(p, 0.0) + float(ps) * float(qs)
    snaps.append({"ts": now, "bids": bd, "asks": ad})
    if len(snaps) > _OB_MAX_SNAPS:
        del snaps[0]


def _build_ob_heat_2d(plot_df, grid, bsz):
    """
    Build a 2D (n_prices × n_candles) order-book heatmap from stored snapshots.
    Each column uses the most recent snapshot taken at or before that candle's
    timestamp; gaps are forward-filled so walls persist until changed.
    Returns (matrix, raw_max_notional).
    """
    import datetime as _dt
    snaps     = st.session_state.get(_OB_HISTORY_KEY, [])
    n_prices  = len(grid)
    n_candles = len(plot_df)
    g_lo      = grid[0]
    heat      = np.zeros((n_prices, n_candles))
    if not snaps:
        return heat, 0.0

    prev_col = np.zeros(n_prices)
    for t in range(n_candles):
        cts = plot_df.index[t]
        if hasattr(cts, "tzinfo") and cts.tzinfo is None:
            cts = cts.replace(tzinfo=_dt.timezone.utc)
        elif not hasattr(cts, "tzinfo"):
            cts = _dt.datetime.fromtimestamp(float(cts) / 1e9, tz=_dt.timezone.utc)

        snap = None
        for s in reversed(snaps):
            if s["ts"] <= cts:
                snap = s
                break

        if snap is None:
            heat[:, t] = prev_col
            continue

        col = np.zeros(n_prices)
        for p, notional in snap["bids"].items():
            idx = int(round((p - g_lo) / bsz))
            if 0 <= idx < n_prices:
                col[idx] += notional
        for p, notional in snap["asks"].items():
            idx = int(round((p - g_lo) / bsz))
            if 0 <= idx < n_prices:
                col[idx] += notional
        heat[:, t] = col
        prev_col = col

    raw_max = float(heat.max()) if heat.max() > 0 else 0.0
    return heat, raw_max


# ════════════════════════════════════════════════════════════════
#  LIQUIDITY CHART PANEL
# ════════════════════════════════════════════════════════════════

def plot_liquidity_depth(ax, liq, short_df=None):
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cprice   = liq.get("liq_current_price", 0)
    raw_bids = liq.get("liq_raw_bids", [])
    raw_asks = liq.get("liq_raw_asks", [])
    bid_c    = liq.get("liq_bid_clusters", {})
    ask_c    = liq.get("liq_ask_clusters", {})
    _full_15m = _fetch_15min_ohlcv()  # 288 candles = 72h
    _display_n = 96                   # show only last 24h on the chart
    if _full_15m is not None and not _full_15m.empty:
        _heat_df = _full_15m          # full 72h used to build heat accumulation
        plot_df  = _full_15m.iloc[-_display_n:]  # last 24h for price line / x-axis
    else:
        _fb      = short_df if (short_df is not None and not short_df.empty) else _fetch_short_term_ohlcv()
        _heat_df = _fb
        plot_df  = _fb

    if not raw_bids or not raw_asks or cprice <= 0 or plot_df is None or plot_df.empty:
        ax.text(0.5, 0.5, "Liquidity Data Unavailable",
                ha="center", color="#8b949e", transform=ax.transAxes)
        return

    # Accumulate real data into session state every refresh
    _push_ob_snapshot(raw_bids, raw_asks, cprice)
    _push_liq_events(_fetch_liq_events())

    ratio     = liq.get("liq_depth_ratio", 1.0)
    n_candles = len(plot_df)

    # Dynamic viewport: ±4% base, extended to include nearest walls if within ±6%
    _an_pre = liq.get("liq_analysis", {})
    _nl_pre = _an_pre.get("nearest_long_liq")
    _ns_pre = _an_pre.get("nearest_short_liq")
    _vp_lo  = cprice * 0.960
    _vp_hi  = cprice * 1.040
    for _wp in [_nl_pre[0] if _nl_pre else None, _ns_pre[0] if _ns_pre else None]:
        if _wp and abs(_wp - cprice) / cprice <= 0.06:
            _vp_lo = min(_vp_lo, _wp * 0.997)
            _vp_hi = max(_vp_hi, _wp * 1.003)
    grid_lo = _vp_lo * 0.999
    grid_hi = _vp_hi * 1.001

    grid      = _snap_grid(grid_lo, grid_hi, HEATMAP_BIN)
    n_prices  = len(grid)
    g_lo      = grid[0]
    zoom_lo, zoom_hi = grid[0], grid[-1]
    zoom_range = zoom_hi - zoom_lo

    if n_prices < 5 or n_candles < 5:
        return

    # ── Layer 1: liquidation heatmap ────────────────────────────────────────
    # Priority order:
    #   A) Coinglass API           — real OI liquidation map (best, paid)
    #   B) Hyblock Capital         — real liquidation levels (free tier, needs auth)
    #   C) Binance synthetic       — reconstructed from public OI+taker+L/S (FREE, NEW)
    #   D) Order-book projection   — last-resort proxy
    cg_data         = _fetch_coinglass_liq_map()
    hb_data         = _fetch_hyblock_liq_map() if not cg_data else None
    bn_synth        = (_fetch_binance_synthetic_liqmap(cprice)
                       if not (cg_data or hb_data) else None)
    oi_data         = cg_data or hb_data or bn_synth
    using_coinglass = oi_data is not None

    if using_coinglass:
        # ── A: Coinglass discrete-bar rendering ──────────────────────────────
        oi_max  = oi_data["max_usd"] or 1.0
        cg_cmap = _cg_cmap()

        for price, usd in oi_data.get("long", []):
            if zoom_lo <= price <= zoom_hi:
                t = min(1.0, usd / oi_max)
                ax.barh(price, t * n_candles * 0.9, height=HEATMAP_BIN * 0.85,
                        left=n_candles - t * n_candles * 0.9, align="center",
                        color=cg_cmap(t), alpha=0.85, zorder=1)
        for price, usd in oi_data.get("short", []):
            if zoom_lo <= price <= zoom_hi:
                t = min(1.0, usd / oi_max)
                ax.barh(price, t * n_candles * 0.9, height=HEATMAP_BIN * 0.85,
                        left=n_candles - t * n_candles * 0.9, align="center",
                        color=cg_cmap(t), alpha=0.85, zorder=1)

        liq_raw_max = oi_max
        using_real_events = False
        # Use order-book projection for narrative cluster ranking (no OHLCV needed)
        lev_heat  = _build_orderbook_liq_heat(bid_c, ask_c, cprice, grid, HEATMAP_BIN, n_candles)
        liq_heat  = lev_heat / (lev_heat.max() or 1.0)

    else:
        # ── B: Order-book projection (base) + real liq events (overlay) ──────
        # Order-book projection: uses live bid/ask depth to project where leveraged
        # positions are concentrated, which is more accurate than OHLCV-based math.
        lev_heat    = _build_orderbook_liq_heat(bid_c, ask_c, cprice, grid, HEATMAP_BIN, n_candles)
        lev_nrm     = lev_heat / (lev_heat.max() or 1.0)

        # Real liquidation events from Binance + Bybit (accumulated in session state)
        liq_ev_heat, liq_ev_max = _build_liq_event_heat_2d(plot_df, grid, HEATMAP_BIN)
        n_liq_events = len(st.session_state.get(_LIQ_EVENTS_KEY, {}))

        using_real_events = n_liq_events >= 5 and liq_ev_max > 0
        if using_real_events:
            # Blend: real events dominate where they exist, OB projection fills gaps
            ev_nrm      = liq_ev_heat / liq_ev_max
            ev_covered  = np.any(liq_ev_heat > 0, axis=0)
            liq_heat    = np.where(ev_covered[None, :], ev_nrm, lev_nrm)
            # Reinforce real-event rows with the OB projection signal
            _ev_rows    = liq_ev_heat.max(axis=1) > 0
            liq_heat[_ev_rows] = np.maximum(liq_heat[_ev_rows],
                                            lev_nrm[_ev_rows] * 0.4)
            liq_raw_max = liq_ev_max
        else:
            liq_heat    = lev_nrm.copy()
            liq_raw_max = float(lev_heat.max() or 1.0)

        if using_real_events:
            _row_max      = liq_heat.max(axis=1)
            _sorted_rows  = np.argsort(_row_max)[::-1]
            _tier_targets = [1.0, 0.65, 0.45]
            for _ri, _pidx in enumerate(_sorted_rows):
                if _row_max[_pidx] < 0.04:
                    break
                target = _tier_targets[_ri] if _ri < len(_tier_targets) else 0.28
                if _row_max[_pidx] > 0:
                    liq_heat[_pidx, :] *= target / _row_max[_pidx]
            liq_heat = np.power(liq_heat.clip(0, 1), 1.5)
        else:
            liq_heat = np.power(liq_heat.clip(0, 1), 0.7)

        pi  = int(round((cprice - g_lo) / HEATMAP_BIN))
        gap = max(1, int(n_prices * 0.008))
        liq_heat[max(0, pi - gap):min(n_prices, pi + gap + 1)] *= 0.4

    # ── Store top clusters in session state so narrative matches heatmap ──────
    try:
        _below_m = grid < cprice
        _above_m = grid > cprice

        if using_coinglass:
            # Coinglass data has real USD values — sort by proximity to current price
            _lc = sorted(
                [(p, u, min(1.0, u / cg_data["max_usd"])) for p, u in cg_data.get("long", [])
                 if zoom_lo <= p < cprice],
                key=lambda x: x[0], reverse=True)[:5]
            _sc = sorted(
                [(p, u, min(1.0, u / cg_data["max_usd"])) for p, u in cg_data.get("short", [])
                 if cprice < p <= zoom_hi],
                key=lambda x: x[0])[:5]
            st.session_state["_liq_clusters"] = {
                "long":  _lc, "short": _sc,
                "cprice": cprice, "real_events": True,
            }
        else:
            _raw_row  = lev_heat.max(axis=1)
            _vis_row  = liq_heat.max(axis=1)
            _raw_max  = _raw_row.max() or 1.0

            # Build USD arrays from current order-book snapshot
            _depth_ratio = liq.get("liq_depth_ratio", 1.0)
            _, _la_usd, _sa_usd, _ = _build_liq(raw_bids, raw_asks, cprice, grid, _depth_ratio)

        def _pick_peaks(raw, vis, la_usd, sa_usd, mask, grid, side, n=5):
            sub_r = (raw * mask.astype(float)).copy()
            sub_v = (vis * mask.astype(float)).copy()
            usd_arr = la_usd if side == "long" else sa_usd
            out = []
            for _ in range(n):
                if sub_r.max() < _raw_max * 0.005:
                    break
                idx = int(sub_r.argmax())
                usd_val = float(usd_arr[idx])
                out.append((float(grid[idx]), usd_val, float(sub_v[idx])))
                lo = max(0, idx - 5); hi = min(len(sub_r), idx + 6)
                sub_r[lo:hi] = 0
            return out   # [(price, usd_notional, vis_intensity), ...]

            _lc = _pick_peaks(_raw_row, _vis_row, _la_usd, _sa_usd, _below_m, grid, "long")
            _sc = _pick_peaks(_raw_row, _vis_row, _la_usd, _sa_usd, _above_m, grid, "short")
            st.session_state["_liq_clusters"] = {
                "long":  sorted(_lc, key=lambda x: x[0], reverse=True),
                "short": sorted(_sc, key=lambda x: x[0]),
                "cprice": cprice,
                "real_events": using_real_events,
            }
    except Exception as _e:
        st.session_state["_liq_clusters_err"] = str(_e)

    # ── Layer 2: discrete-bar heatmap (matches Coinglass visual style) ───────
    # Draw each price level as a horizontal bar extending left from the current
    # candle.  Bar length ∝ heat intensity — makes clusters look like Coinglass
    # instead of a smooth gradient.  Works for all data sources.
    cg = _cg_cmap()   # always needed for colorbar + key-level bars
    if not using_coinglass:
        # Collapse time axis: use row-max so the strongest point in 24h defines bar length
        _row_h   = lev_heat.max(axis=1)
        _row_max = _row_h.max() or 1.0

        # Dark purple background fill (matches Coinglass background colour)
        ax.set_facecolor("#0d0012")

        # Only show the most significant clusters — top 50% of peak intensity
        _threshold = _row_max * 0.50
        # Find nearest grid index for each key level using argmin (robust against grid offsets)
        _nl_tup   = _an_pre.get("nearest_long_liq") if _an_pre else None
        _ns_tup   = _an_pre.get("nearest_short_liq") if _an_pre else None
        _nl_price = _nl_tup[0] if _nl_tup else None
        _ns_price = _ns_tup[0] if _ns_tup else None
        _nl_idx   = int(np.argmin(np.abs(grid - _nl_price))) if _nl_price else -1
        _ns_idx   = int(np.argmin(np.abs(grid - _ns_price))) if _ns_price else -1

        for i in range(n_prices):
            h = float(_row_h[i])
            is_key = i in (_nl_idx, _ns_idx)
            if h < _threshold and not is_key:
                continue
            t         = max(h / _row_max, 0.55 if is_key else 0.0)  # key levels always visible
            color     = "#ffee00" if is_key else cg(t)               # bright yellow for key levels
            bar_len   = t * n_candles
            bar_left  = n_candles - bar_len
            _alpha    = 0.97 if is_key else min(0.95, 0.30 + 0.65 * t)
            _height   = HEATMAP_BIN * 0.75 if is_key else HEATMAP_BIN * 0.55
            ax.barh(grid[i], bar_len, height=_height,
                    left=bar_left, align="center",
                    color=color, alpha=_alpha,
                    linewidth=0, zorder=2 if is_key else 1)

    # ── Layer 3: resting order-book walls — bold band + spine style ───────────
    # Bids = cyan bands below price, Asks = orange bands above price.
    # Each wall gets: a semi-transparent fill band (axhspan) + a solid spine line.
    # This makes walls visually distinct from the liq bars (which are barh rectangles).
    _bid_max = max(bid_c.values(), default=1) or 1
    _ask_max = max(ask_c.values(), default=1) or 1
    _half_band = HEATMAP_BIN * 0.30          # half-height of the fill band
    for _wp, _wn in bid_c.items():
        if zoom_lo <= _wp <= zoom_hi and _wp < cprice:
            _s = _wn / _bid_max
            if _s > 0.08:                    # only the meaningful walls
                _band_alpha = 0.08 + 0.22 * _s
                _line_alpha = 0.55 + 0.45 * _s
                _lw         = 1.0 + 2.5 * _s
                # faint fill band
                ax.axhspan(_wp - _half_band, _wp + _half_band,
                           color="#3fb950", alpha=_band_alpha, zorder=2, linewidth=0)
                # crisp spine on top of band
                ax.axhline(_wp, color="#3fb950", lw=_lw,
                           alpha=_line_alpha, zorder=4,
                           linestyle="--", dashes=(6, 3), solid_capstyle="butt")
    for _wp, _wn in ask_c.items():
        if zoom_lo <= _wp <= zoom_hi and _wp > cprice:
            _s = _wn / _ask_max
            if _s > 0.08:
                _band_alpha = 0.08 + 0.22 * _s
                _line_alpha = 0.55 + 0.45 * _s
                _lw         = 1.0 + 2.5 * _s
                ax.axhspan(_wp - _half_band, _wp + _half_band,
                           color="#f85149", alpha=_band_alpha, zorder=2, linewidth=0)
                ax.axhline(_wp, color="#f85149", lw=_lw,
                           alpha=_line_alpha, zorder=4,
                           linestyle="--", dashes=(6, 3), solid_capstyle="butt")

    # Colorbar
    fig = ax.get_figure()
    cax = inset_axes(ax, width="2.2%", height="38%", loc="lower left",
                     bbox_to_anchor=(0.035, 0.035, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    sm = ScalarMappable(cmap=cg, norm=Normalize(vmin=0, vmax=liq_raw_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=5.5, colors="#8b949e", length=2)
    cbar.set_label("Liq (USD)", color="#8b949e", fontsize=5.5, labelpad=2)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _fusd(v)))
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_color("#8b949e")
    cbar.outline.set_edgecolor("#21262d")
    cbar.ax.set_facecolor("#0d1117")

    # ── Right-axis wall labels ──────────────────────────────────
    bid_walls = liq.get("liq_bid_walls", [])
    ask_walls = liq.get("liq_ask_walls", [])
    bid_usd_total = liq.get("liq_total_bid", 0)
    ask_usd_total = liq.get("liq_total_ask", 0)

    ax_right = ax.twinx()
    ax_right.set_ylim(zoom_lo, zoom_hi)
    ax_right.set_yticks([])
    ax_right.tick_params(right=False)
    for sp in ax_right.spines.values():
        sp.set_visible(False)

    # ── Layer 4: price line with glow ────────────────────────────────────────
    x        = np.arange(n_candles)
    closes_v = plot_df["Close"].values
    highs_v  = plot_df["High"].values
    lows_v   = plot_df["Low"].values
    opens_v  = plot_df["Open"].values
    # Wide faint glow → medium → crisp bright core
    for _lw, _a in [(6, 0.06), (3, 0.15), (1.3, 0.95)]:
        ax.plot(x, closes_v, color="#ffffff", lw=_lw, alpha=_a, zorder=6 + _lw,
                solid_capstyle="round")
    # Current price tag on right axis
    ax_right.annotate(
        f" ${closes_v[-1]:,.0f}",
        xy=(1.0, closes_v[-1]), xycoords=ax_right.get_yaxis_transform(),
        fontsize=8.5, color="#ffffff", fontweight="bold",
        va="center", ha="left", annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#ffffff",
                  alpha=0.95, lw=1.2),
    )

    def _render_wall_labels(walls, color, side, depth_total):
        if not walls:
            return

        in_view  = [(p, n) for p, n in walls if zoom_lo <= p <= zoom_hi][:3]
        out_view = [(p, n) for p, n in walls if p < zoom_lo or p > zoom_hi]

        # Sort bids high→low, asks low→high (natural price order on chart)
        in_view.sort(key=lambda x: x[0], reverse=(side == "bid"))

        if in_view:
            max_n   = max(n for _, n in in_view)
            min_gap = zoom_range * 0.028

            slots = []
            for wp, wn in in_view:
                y = wp
                for _ in range(40):
                    if all(abs(y - s) >= min_gap for s in slots):
                        break
                    y += min_gap * (1 if side == "ask" else -1)
                y = max(zoom_lo + zoom_range * 0.015,
                        min(zoom_hi - zoom_range * 0.015, y))
                slots.append(y)

                alpha = 0.25 + 0.30 * (wn / max_n)
                lw    = 0.5  + 0.8  * (wn / max_n)
                ax.axhline(wp, color=color, lw=lw, ls=(0, (4, 3)), alpha=alpha, zorder=3)

                dist_pct = abs(wp - cprice) / cprice * 100
                tag = "BID" if side == "bid" else "ASK"
                ax_right.annotate(
                    f"{_fprice_m(wp)} {tag} ({_fusd_m(wn)})  {dist_pct:.1f}%",
                    xy=(1.01, y),
                    xycoords=ax_right.get_yaxis_transform(),
                    fontsize=6.5, color=color,
                    va="center", ha="left", fontweight="bold",
                    annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.13", fc="#0d1117",
                              ec=color, alpha=0.92, lw=0.6),
                )

        # Always show total depth pill — out-of-view count + full notional
        n_shown = len(in_view)
        n_hidden = len(out_view)
        hidden_total = sum(n for _, n in out_view)
        tag = "BID" if side == "bid" else "ASK"

        if side == "bid":
            pill_y = zoom_lo + zoom_range * 0.035
            # push up if a wall label is sitting there
            if slots and min(slots) < pill_y + zoom_range * 0.04:
                pill_y = zoom_lo + zoom_range * 0.015
            if n_hidden > 0:
                pill = f"▼ +{n_hidden} {tag} below view  {_fusd_m(hidden_total)}"
            else:
                pill = f"TOTAL {tag} DEPTH  {_fusd_m(depth_total)}"
        else:
            pill_y = zoom_hi - zoom_range * 0.035
            if slots and max(slots) > pill_y - zoom_range * 0.04:
                pill_y = zoom_hi - zoom_range * 0.015
            if n_hidden > 0:
                pill = f"▲ +{n_hidden} {tag} above view  {_fusd_m(hidden_total)}"
            else:
                pill = f"TOTAL {tag} DEPTH  {_fusd_m(depth_total)}"

        ax_right.annotate(
            pill,
            xy=(1.01, pill_y),
            xycoords=ax_right.get_yaxis_transform(),
            fontsize=7, color=color,
            va="center", ha="left", fontweight="bold",
            annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.18", fc="#0d1117",
                      ec=color, alpha=0.94, lw=0.9),
        )

    _render_wall_labels(bid_walls, "#3fb950", "bid", bid_usd_total)
    _render_wall_labels(ask_walls, "#f85149", "ask", ask_usd_total)

    # Price marker
    ax.axhline(cprice, color="#ffffff", lw=0.6, ls=":", alpha=0.25, zorder=3)

    # ── Sweep markers: wick-through-wall then close on opposite side ──────────
    # ▼S = bid sweep (wick below wall, close above) → potential long entry signal
    # ▲S = ask sweep (wick above wall, close below) → potential short entry signal
    # Only scan the last 8 candles (2h) — current wall prices are stale beyond that
    _sw_walls = ([(p, "bid") for p, _ in bid_walls[:3] if zoom_lo <= p <= zoom_hi] +
                 [(p, "ask") for p, _ in ask_walls[:3] if zoom_lo <= p <= zoom_hi])
    for _t in range(max(1, n_candles - 8), n_candles):
        for _swp, _sws in _sw_walls:
            if _sws == "bid" and lows_v[_t] < _swp < closes_v[_t] and closes_v[_t] > opens_v[_t]:
                ax.annotate("▼S", xy=(_t, _swp), fontsize=6, color="#3fb950",
                            ha="center", va="top", zorder=9, fontweight="bold",
                            annotation_clip=True)
            elif _sws == "ask" and highs_v[_t] > _swp > closes_v[_t] and closes_v[_t] < opens_v[_t]:
                ax.annotate("▲S", xy=(_t, _swp), fontsize=6, color="#f85149",
                            ha="center", va="bottom", zorder=9, fontweight="bold",
                            annotation_clip=True)

    # ── Directional intent ────────────────────────────────────────
    analysis  = liq.get("liq_analysis", {})
    nl        = analysis.get("nearest_long_liq")
    ns        = analysis.get("nearest_short_liq")
    lt        = analysis.get("long_liq_total", 0)
    st_liq    = analysis.get("short_liq_total", 0)
    hz        = analysis.get("hunt_zones", [])
    cd        = analysis.get("cascade_direction", "BALANCED")
    bid_walls = liq.get("liq_bid_walls", [])
    ask_walls = liq.get("liq_ask_walls", [])
    ratio     = liq.get("liq_depth_ratio", 1.0)

    # Which liq target is the primary magnet vs the secondary?
    if cd == "UP":
        primary_liq, secondary_liq   = ns, nl
        primary_col, secondary_col   = "#3fb950", "#f85149"
        dir_label                    = "HUNTING UP"
    elif cd == "DOWN":
        primary_liq, secondary_liq   = nl, ns
        primary_col, secondary_col   = "#f85149", "#3fb950"
        dir_label                    = "HUNTING DOWN"
    else:
        dir_label = "BALANCED"
        primary_col, secondary_col = "#ffd700", "#8b949e"
        # Score = pool_size / distance_pct — larger pool at similar distance wins.
        # This correctly identifies the stronger magnet (bigger payoff for hunters).
        ns_dist  = abs(ns[0] - cprice) / cprice * 100 if ns else float("inf")
        nl_dist  = abs(nl[0] - cprice) / cprice * 100 if nl else float("inf")
        ns_score = (ns[1] / ns_dist) if ns and ns_dist > 0 else 0
        nl_score = (nl[1] / nl_dist) if nl and nl_dist > 0 else 0
        if nl_score >= ns_score:
            primary_liq, secondary_liq = nl, ns
        else:
            primary_liq, secondary_liq = ns, nl

    # Hunt zone: if it aligns with cascade, mark as the specific target level
    hunt_target = None
    for h in hz[:1]:
        aligned = (cd == "UP" and h["side"] == "ASK") or (cd == "DOWN" and h["side"] == "BID")
        if aligned:
            hunt_target = h

    # ── Liq-pool glow lines ───────────────────────────────────────
    # Use the top of the heatmap spectrum (bright yellow) — same encoding as the
    # yellow key-level bars.  Red/green are reserved for directional arrows only.
    _liq_spectrum_top = _cg_cmap()(1.0)   # brightest end of the colormap
    def _liq_glow(price):
        if not (zoom_lo <= price <= zoom_hi):
            return
        for lw, a in [(12, 0.04), (6, 0.10), (2.5, 0.65)]:
            ax.axhline(price, color=_liq_spectrum_top, lw=lw, alpha=a,
                       zorder=3, solid_capstyle="butt")

    _liq_glow(nl[0]) if nl else None
    _liq_glow(ns[0]) if ns else None

    # ── Secondary target (muted dashed + small label) ───────────
    _spec_col = _liq_spectrum_top   # spectrum yellow, consistent with bars and glow
    if secondary_liq and zoom_lo <= secondary_liq[0] <= zoom_hi:
        ax.axhline(secondary_liq[0], color=_spec_col, lw=0.9,
                   ls="--", alpha=0.35, zorder=4)
        _sec_dist = abs(secondary_liq[0] - cprice) / cprice * 100
        _sec_sign = "+" if secondary_liq[0] > cprice else "-"
        _sec_lbl  = "LONG LIQ" if secondary_liq[0] < cprice else "SHORT LIQ"
        ax.text(n_candles * 0.015, secondary_liq[0] + zoom_range * 0.007,
                f"{_sec_lbl}  {_sec_sign}{_sec_dist:.1f}%  {_fusd_m(secondary_liq[1])} est.",
                fontsize=6.5, color=_spec_col, alpha=0.65, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0d111780", ec="none"))

    # ── Primary target: path shade + thick line + distance label ─
    if primary_liq and zoom_lo <= primary_liq[0] <= zoom_hi:
        shade_lo = min(cprice, primary_liq[0])
        shade_hi = max(cprice, primary_liq[0])
        ax.axhspan(shade_lo, shade_hi, alpha=0.05, color=_spec_col, zorder=1)
        ax.axhline(primary_liq[0], color=_spec_col, lw=2.2, ls="-", alpha=0.88, zorder=5)
        dist_pct  = abs(primary_liq[0] - cprice) / cprice * 100
        sign      = "+" if primary_liq[0] > cprice else "-"
        wall_notl = primary_liq[1]
        tgt_label = "LONG LIQ TARGET" if primary_liq[0] < cprice else "SHORT LIQ TARGET"
        ax.text(n_candles * 0.015, primary_liq[0] + zoom_range * 0.009,
                f"{tgt_label}  {sign}{dist_pct:.1f}%  {_fusd_m(wall_notl)} est.",
                fontsize=7.5, color=_spec_col, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="#0d111780", ec="none"))

    # ── Hunt zone line (gold, most specific target) ──────────────
    if hunt_target and zoom_lo <= hunt_target["price"] <= zoom_hi:
        ax.axhline(hunt_target["price"], color="#ffd700", lw=1.8,
                   ls=(0, (5, 2)), alpha=0.85, zorder=5)

    # ── Directional intent box (between price and primary target) ─
    if cd != "BALANCED" and primary_liq:
        dist_pct = abs(primary_liq[0] - cprice) / cprice * 100
        sign     = "+" if primary_liq[0] > cprice else "-"
        arrow    = "▲" if cd == "UP" else "▼"
        mid_y    = (cprice + primary_liq[0]) / 2
        if zoom_lo < mid_y < zoom_hi:
            ax.text(n_candles * 0.68, mid_y,
                    f"{arrow}  {dir_label}\n{_fprice_m(primary_liq[0])}  ({sign}{dist_pct:.1f}%)",
                    fontsize=9.5, color=primary_col, fontweight="bold",
                    ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#0d1117",
                              ec=primary_col, alpha=0.93, lw=1.8))

    # ── KEY LEVELS box (top-left, compact) ───────────────────────
    imb_lbl   = "bid-heavy" if ratio > 1.2 else ("ask-heavy" if ratio < 0.8 else "balanced")
    lvl_lines = [" KEY LEVELS"]
    if ns:
        d = (ns[0] - cprice) / cprice * 100
        lvl_lines.append(f" Short liq {_fprice_m(ns[0])}  +{d:.1f}%  {_fusd_m(ns[1])} est.")
    if nl:
        d = (cprice - nl[0]) / cprice * 100
        lvl_lines.append(f" Long liq  {_fprice_m(nl[0])}  -{d:.1f}%  {_fusd_m(nl[1])} est.")
    if hz:
        h = hz[0]
        lvl_lines.append(f" Hunt({'ASK' if h['side']=='ASK' else 'BID'}) "
                         f"{_fprice_m(h['price'])}  {_fusd_m(h['wall'])} wall")
    lvl_lines.append(f" Book {ratio:.1f}x {imb_lbl}")

    # ── Book imbalance badge (bottom-right) ──────────────────────────────────
    _bid_d = liq.get("liq_total_bid", 0)
    _ask_d = liq.get("liq_total_ask", 0)
    _imb   = _bid_d / _ask_d if _ask_d > 0 else 1.0
    if _imb > 1.1:
        _ic, _il = "#3fb950", f"BID {_imb:.2f}x  ▲ bullish"
    elif _imb < 0.9:
        _ic, _il = "#f85149", f"ASK {1/_imb:.2f}x  ▼ bearish"
    else:
        _ic, _il = "#8b949e", f"BALANCED  {_imb:.2f}x"
    ax.text(0.988, 0.022, f"Book Imbalance  {_il}",
            transform=ax.transAxes, fontsize=7.5, color=_ic,
            ha="right", va="bottom", fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=_ic,
                      alpha=0.93, lw=0.9))

    ax.text(0.012, 0.978, "\n".join(lvl_lines), transform=ax.transAxes,
            fontsize=7, color="#c9d1d9", va="top", ha="left", family="monospace",
            linespacing=1.5, zorder=7,
            bbox=dict(boxstyle="round,pad=0.4", fc="#0d111799", ec="#58a6ff", alpha=0.95, lw=0.8))

    ax.set_xlim(-0.5, n_candles - 0.5)
    ax.set_ylim(zoom_lo, zoom_hi)
    ax_right.set_ylim(zoom_lo, zoom_hi)

    step   = max(1, n_candles // 8)
    xticks = list(range(0, n_candles, step))
    xlbls  = [plot_df.index[i].strftime("%H:%M") if i < n_candles else "" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlbls, fontsize=7, color="#8b949e")

    zoom_g = _snap_grid(zoom_lo, zoom_hi, HEATMAP_BIN)
    ystep  = max(1, len(zoom_g) // 12)
    ax.set_yticks(zoom_g[::ystep])
    ax.set_yticklabels([f"${p:,.0f}" for p in zoom_g[::ystep]], fontsize=7)
    src = liq.get("liq_source", "")
    n_ev = len(st.session_state.get(_LIQ_EVENTS_KEY, {}))
    if cg_data:
        ev_lbl = "Coinglass open-interest data"
    elif hb_data:
        ev_lbl = "Hyblock liquidation levels"
    elif ss_data:
        ev_lbl = f"Screenshot extract · {ss_data.get('n_clusters', '?')} clusters (Claude vision)"
    elif n_ev >= 5:
        ev_lbl = f"OB projection + {n_ev} real liq events (Binance+Bybit)"
    else:
        ev_lbl = f"Order-book projection · collecting real events ({n_ev})"
    ax.set_title(f"BTC Liquidation Heatmap · 24h/15m  ·  {ev_lbl}  ·  {src}",
                 color="#8b949e", fontsize=8)

    # ── Legend (top-right, inside axes, below the right-side wall labels) ────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    _leg_items = [
        Patch(facecolor="#3d2b00", edgecolor="#ffee00", lw=0.8, label="Liq bar (spectrum: purple→yellow = low→high)"),
        Line2D([0], [0], color="#ffee00", lw=2.0,               label="Primary liq cluster / hunt target"),
        Line2D([0], [0], color="#ffee00", lw=0.9, ls="--",      label="Secondary liq cluster"),
        Line2D([0], [0], color="#ffd700", lw=1.4, ls=(0,(5,2)), label="Specific hunt zone"),
        Line2D([0], [0], color="#f85149", lw=1.5, ls="--",      label="Ask wall (resting sell orders)"),
        Line2D([0], [0], color="#3fb950", lw=1.5, ls="--",      label="Bid wall (resting buy orders)"),
    ]
    # Place outside the axes — below the x-axis — so it never overlaps chart content
    _leg = ax.legend(
        handles=_leg_items,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.09),   # just below x-axis labels
        bbox_transform=ax.transAxes,
        fontsize=6.0,
        framealpha=0.90,
        facecolor="#0d1117",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        handlelength=2.4,
        handleheight=0.9,
        borderpad=0.55,
        labelspacing=0.35,
        ncols=3,                        # 2 rows × 3 cols fits cleanly under the chart
    )
    _leg.set_zorder(10)


# ════════════════════════════════════════════════════════════════
#  MAIN FIGURE BUILDER
# ════════════════════════════════════════════════════════════════

def _ax_style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")


# ════════════════════════════════════════════════════════════════
#  DATA LAYER — single fetch, all charts share this
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(ticker: str = "BTC-USD") -> dict:
    """Fetch + compute everything. Returns a plain dict — no figures."""
    df = fetch_ohlc(ticker)
    if df.empty:
        return {}

    closes = df["Close"].tolist()
    price  = closes[-1]
    # Override with live Binance spot price — yfinance daily lags by 5-30 min via cache+latency.
    # Only applies for BTC; other tickers keep yfinance value.
    if "BTC" in ticker.upper():
        try:
            _live = _get_raw("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
            if isinstance(_live, dict) and "price" in _live:
                _p = float(_live["price"])
                if _p > 0:
                    price = _p
        except Exception:
            pass
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    rsi_series  = calculate_rsi(df["Close"])
    df["RSI"]   = rsi_series
    macd, macd_sig, macd_hist = calculate_macd(df["Close"])
    df["MACD"] = macd; df["MACD_Signal"] = macd_sig; df["MACD_Hist"] = macd_hist
    obv_trend             = calculate_obv_trend(df)
    div_rsi               = detect_rsi_divergence(closes, rsi_series)
    adx_val               = calculate_adx(df)
    supports, resistances = find_levels(closes)
    imm_sup = nearest_level(supports,    price, "support")
    imm_res = nearest_level(resistances, price, "resistance")
    w52     = week52_metrics(closes)

    # ── New indicators ────────────────────────────────────────
    bb      = calculate_bollinger_bands(df["Close"])
    df["BB_mid"]   = bb["mid"]
    df["BB_upper"] = bb["upper"]
    df["BB_lower"] = bb["lower"]
    df["BB_pctb"]  = bb["pct_b"]
    df["BB_bw"]    = bb["bandwidth"]

    stoch   = calculate_stochastic(df)
    df["Stoch_K"] = stoch["k"]
    df["Stoch_D"] = stoch["d"]

    df["ATR"]  = calculate_atr(df)
    df["AO"]   = calculate_awesome_oscillator(df)
    df["AC"]   = calculate_accelerator_oscillator(df)

    ichi    = calculate_ichimoku(df)
    df["Ichi_Tenkan"] = ichi["tenkan"]
    df["Ichi_Kijun"]  = ichi["kijun"]
    df["Ichi_SpanA"]  = ichi["span_a"]
    df["Ichi_SpanB"]  = ichi["span_b"]
    df["Ichi_Chikou"] = ichi["chikou"]

    fib     = calculate_fibonacci_levels(closes)

    crypto_sig = fetch_crypto_signals(ticker)
    price_key  = round(price / 100) * 100
    btc_liq    = fetch_btc_liquidity_cached(price_key)
    has_liq    = bool(btc_liq and btc_liq.get("liq_bid_clusters"))
    cycle      = detect_btc_cycle_phase(
        closes=closes, rsi_series=rsi_series,
        crypto_sig=crypto_sig, obv_trend=obv_trend, div_rsi=div_rsi
    )

    short_df      = _fetch_short_term_ohlcv()
    df_4h         = _fetch_4h_ohlcv()
    # ⑯ Track 4h fetch freshness: compare session-state timestamp to now.
    # If score refreshes faster than the 15-min cache, EMA Structure may be stale.
    _4h_prev_len = st.session_state.get("_4h_cached_len", -1)
    _4h_cur_len  = len(df_4h) if df_4h is not None else 0
    if _4h_cur_len != _4h_prev_len:
        st.session_state["_4h_fetch_ts"]   = _dt.now(_tz.utc).timestamp()
        st.session_state["_4h_cached_len"] = _4h_cur_len
    poly_sentiment = fetch_polymarket_btc_sentiment(price)
    oi_funding    = fetch_oi_funding()
    prediction    = predict_direction(df, btc_liq)
    bias_72h      = compute_72h_bias(df, btc_liq, short_df=short_df, poly=poly_sentiment,
                                     oi_data=oi_funding, df_4h=df_4h)
    bias_24h      = compute_24h_bias(df, btc_liq, short_df=short_df,
                                     oi_data=oi_funding, crypto_sig=crypto_sig,
                                     poly=poly_sentiment)

    # 24h price change from last two rows
    prev_close = closes[-2] if len(closes) >= 2 else price
    chg_24h    = (price - prev_close) / prev_close * 100

    plot_df = df.iloc[-365:].copy()

    return dict(
        df=df, plot_df=plot_df, short_df=short_df, closes=closes, price=price,
        rsi_series=rsi_series, obv_trend=obv_trend, div_rsi=div_rsi,
        adx_val=adx_val, imm_sup=imm_sup, imm_res=imm_res, w52=w52,
        fib=fib, bb=bb,
        crypto_sig=crypto_sig, btc_liq=btc_liq, has_liq=has_liq,
        cycle=cycle, prediction=prediction, bias_72h=bias_72h, bias_24h=bias_24h,
        poly_sentiment=poly_sentiment,
        chg_24h=chg_24h, ticker=ticker,
    )


# ════════════════════════════════════════════════════════════════
#  CHART BUILDERS  (each returns a small, focused figure)
# ════════════════════════════════════════════════════════════════

def fig_price_chart(a: dict) -> plt.Figure:
    """Price + Volume + RSI(14) — 3-panel daily chart."""
    plot_df = a["plot_df"]
    price   = a["price"]
    fib     = a["fib"]

    fig, (ax, ax_vol, ax_rsi) = plt.subplots(
        3, 1, figsize=(13, 7.0), sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.04}
    )
    fig.patch.set_facecolor("#0d1117")
    for _a in (ax, ax_vol, ax_rsi): _ax_style(_a)

    # ── Price line + fill ─────────────────────────────────────────
    ax.plot(plot_df.index, plot_df["Close"], color="#e6edf3", lw=1.4, alpha=0.9, zorder=3)
    ax.fill_between(plot_df.index, plot_df["Close"], float(plot_df["Close"].min()),
                    alpha=0.06, color="#58a6ff")

    # ── MAs ───────────────────────────────────────────────────────
    ax.plot(plot_df.index, plot_df["MA50"],  color="#58a6ff", alpha=0.75, lw=1.3, label="MA50")
    ax.plot(plot_df.index, plot_df["MA200"], color="#f0883e", alpha=0.75, lw=1.3, label="MA200")

    # EMA8 — short-term dynamic S/R, tighter than EMA21 and non-redundant with MA50
    ema8_d = plot_df["Close"].ewm(span=8, adjust=False).mean()
    ax.plot(plot_df.index, ema8_d, color="#39d353", alpha=0.65, lw=0.9, ls="--", label="EMA8")

    # Golden / Death cross markers
    ma50_v  = plot_df["MA50"].values
    ma200_v = plot_df["MA200"].values
    gc_mask = (ma50_v[1:] > ma200_v[1:]) & (ma50_v[:-1] <= ma200_v[:-1])
    dc_mask = (ma50_v[1:] < ma200_v[1:]) & (ma50_v[:-1] >= ma200_v[:-1])
    gc_idx  = plot_df.index[1:][gc_mask]
    dc_idx  = plot_df.index[1:][dc_mask]
    if len(gc_idx):
        ax.scatter(gc_idx, plot_df["Close"].loc[gc_idx], marker="*", s=120,
                   color="#ffd700", zorder=6, label="Golden ✕")
    if len(dc_idx):
        ax.scatter(dc_idx, plot_df["Close"].loc[dc_idx], marker="X", s=80,
                   color="#f85149", zorder=6, label="Death ✕")

    # ── Bollinger Bands ───────────────────────────────────────────
    ax.plot(plot_df.index, plot_df["BB_upper"], color="#bc8cff", alpha=0.6, lw=0.9, ls="--", label="BB Upper")
    ax.plot(plot_df.index, plot_df["BB_lower"], color="#bc8cff", alpha=0.6, lw=0.9, ls="--", label="BB Lower")
    ax.fill_between(plot_df.index, plot_df["BB_upper"], plot_df["BB_lower"],
                    alpha=0.04, color="#bc8cff")

    # ── S/R levels ────────────────────────────────────────────────
    if a["imm_sup"]: ax.axhline(a["imm_sup"], ls="--", color="#3fb950", alpha=0.65, lw=1.2, label=f"Sup ${a['imm_sup']:,.0f}")
    if a["imm_res"]: ax.axhline(a["imm_res"], ls="--", color="#f85149", alpha=0.65, lw=1.2, label=f"Res ${a['imm_res']:,.0f}")

    # ── Fibonacci levels (within ±25% of price) ───────────────────
    fib_colors = {"0.0%": "#8b949e", "23.6%": "#58a6ff", "38.2%": "#3fb950",
                  "50.0%": "#ffd700", "61.8%": "#f0883e", "78.6%": "#f85149", "100.0%": "#8b949e"}
    for label, lvl in fib["levels"].items():
        if abs(lvl - price) / price < 0.25:
            col = fib_colors.get(label, "#8b949e")
            ax.axhline(lvl, color=col, alpha=0.45, lw=0.85, ls=(0, (5, 4)))
            ax.text(plot_df.index[1], lvl * 1.001, f"Fib {label}",
                    fontsize=6.5, color=col, alpha=0.75, va="bottom")

    # ── Cycle badge + BB squeeze annotation ───────────────────────
    _cycle  = a["cycle"]
    _col    = ("#3fb950" if "PROBABLE BOTTOM" in _cycle["phase"] else
               "#f0883e" if "FORMING"         in _cycle["phase"] else
               "#f85149" if "PROBABLE TOP"    in _cycle["phase"] else "#8b949e")
    _bar_str = "█" * int((_cycle["total"] + 24) / 48 * 14) + "░" * (14 - int((_cycle["total"] + 24) / 48 * 14))
    ax.text(0.995, 0.97, f"{_cycle['emoji']} {_cycle['phase']}  {_cycle['total']:+d}/24  [{_bar_str}]",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color=_col,
            bbox=dict(boxstyle="round,pad=0.4", fc="#0d1117", ec=_col, alpha=0.9, lw=0.8))
    bw_now  = float(plot_df["BB_bw"].dropna().iloc[-1]) if not plot_df["BB_bw"].dropna().empty else 0
    bw_mean = float(plot_df["BB_bw"].dropna().mean())   if not plot_df["BB_bw"].dropna().empty else 1
    if bw_now < bw_mean * 0.6:
        ax.text(0.005, 0.03, "⚡ BB Squeeze — breakout likely",
                transform=ax.transAxes, fontsize=7.5, color="#ffd700", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#ffd700", alpha=0.85, lw=0.7))

    ax.set_title(f"BTC-USD  Daily (1Y)  ·  OBV: {a['obv_trend']}  ·  ADX: {a['adx_val']:.0f}  ·  BB bw: {bw_now:.1f}%",
                 color="#8b949e", fontsize=9, loc="left", pad=6)
    ax.legend(loc="upper left", fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9",
              framealpha=0.9, ncol=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

    # ── Volume panel ──────────────────────────────────────────────
    close_arr = plot_df["Close"].values
    prev_close = np.concatenate([[close_arr[0]], close_arr[:-1]])
    vol_colors = np.where(close_arr >= prev_close, "#3fb950", "#f85149")
    ax_vol.bar(plot_df.index, plot_df["Volume"].values, color=vol_colors, alpha=0.55, width=0.7)
    vol_ma20 = pd.Series(plot_df["Volume"].values).rolling(20, min_periods=5).mean().values
    ax_vol.plot(plot_df.index, vol_ma20, color="#58a6ff", lw=0.9, alpha=0.7)
    ax_vol.set_ylabel("Volume", color="#8b949e", fontsize=7)
    ax_vol.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.0f}M"))

    # ── RSI(14) panel ─────────────────────────────────────────────
    rsi = a["rsi_series"].iloc[-len(plot_df):]
    rsi_signal = pd.Series(rsi.values).ewm(span=9, adjust=False).mean().values
    ax_rsi.plot(plot_df.index, rsi.values, color="#f0883e", lw=1.1)
    ax_rsi.plot(plot_df.index, rsi_signal,  color="#58a6ff", lw=0.8, alpha=0.7, ls="--")
    ax_rsi.axhline(70, color="#f85149", ls="--", lw=0.7, alpha=0.5)
    ax_rsi.axhline(30, color="#3fb950", ls="--", lw=0.7, alpha=0.5)
    ax_rsi.fill_between(plot_df.index, rsi.values, 70, where=(rsi.values > 70), color="#f85149", alpha=0.12)
    ax_rsi.fill_between(plot_df.index, rsi.values, 30, where=(rsi.values < 30), color="#3fb950", alpha=0.12)
    ax_rsi.set_ylim(10, 90)
    ax_rsi.set_ylabel("RSI 14", color="#8b949e", fontsize=7)
    rsi_now = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50
    rsi_col = "#f85149" if rsi_now > 70 else ("#3fb950" if rsi_now < 30 else "#8b949e")
    ax_rsi.text(0.995, 0.92, f"RSI {rsi_now:.0f}", transform=ax_rsi.transAxes,
                fontsize=8, va="top", ha="right", color=rsi_col)

    plt.tight_layout(pad=0.5)
    return fig


def fig_intraday_15m(a: dict) -> "plt.Figure | None":
    """24h 15-min chart: EMA ribbon, session shading, market structure labels,
    relative-volume z-score, ATR regime percentile, HVN/LVN profile, liq walls."""
    _raw15 = _fetch_15min_ohlcv()
    if _raw15 is not None and len(_raw15) >= 10:
        df15 = _raw15.iloc[-96:].copy()
    else:
        sdf = a.get("short_df")
        if sdf is None or len(sdf) < 10:
            return None
        df15 = sdf.iloc[-96:].copy()
    if df15.empty:
        return None

    closes = df15["Close"].values
    opens  = df15["Open"].values
    highs  = df15["High"].values
    lows   = df15["Low"].values
    vols   = df15["Volume"].values
    n      = len(df15)
    xs     = np.arange(n)
    # Use the live Binance ticker price (a["price"]) instead of the last 15m kline
    # close — kline cache can lag the ticker by 5–15s, causing the right-side price
    # tag to disagree with the header BTC Price. Fall back to closes[-1] if missing.
    _live_price = float(a.get("price") or closes[-1])
    cur    = _live_price
    p_lo   = lows.min();  p_hi = highs.max()
    _pad   = (p_hi - p_lo) * 0.06
    colors = np.where(closes >= opens, "#3fb950", "#f85149")

    # EMAs
    ema8  = df15["Close"].ewm(span=8,  adjust=False).mean().values
    ema21 = df15["Close"].ewm(span=21, adjust=False).mean().values

    # VWAP — reset each UTC session
    vwap = None
    vwap_sessions = []
    try:
        _idx   = df15.index
        _dates = (_idx.normalize() if (hasattr(_idx, "tz") and _idx.tz is not None)
                  else pd.to_datetime(_idx).normalize())
        df15["_s"]    = _dates
        df15["_tp"]   = (df15["High"] + df15["Low"] + df15["Close"]) / 3
        df15["_tpv"]  = df15["_tp"] * df15["Volume"]
        df15["_ctpv"] = df15.groupby("_s")["_tpv"].cumsum()
        df15["_cv"]   = df15.groupby("_s")["Volume"].cumsum()
        vwap = (df15["_ctpv"] / df15["_cv"].replace(0, float("nan"))).values
        for _, _grp in df15.groupby("_s"):
            _mask = _grp.index
            _xi   = np.where(np.isin(df15.index, _mask))[0]
            vwap_sessions.append((xs[_xi], vwap[_xi]))
    except Exception:
        pass

    # ATR (Wilder 14)
    prev_c  = np.concatenate([[closes[0]], closes[:-1]])
    tr      = np.maximum(highs - lows,
              np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr     = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_now = atr[-1]
    atr_pct = float(pd.Series(atr).rank(pct=True).iloc[-1]) * 100   # rolling percentile
    atr_rolling_pct = pd.Series(atr).rank(pct=True).values * 100
    atr_mean20  = float(np.mean(atr[-20:])) if len(atr) >= 20 else float(np.mean(atr))
    atr_chg_pct = (atr_now - atr_mean20) / atr_mean20 * 100
    if atr_pct < 30:
        atr_state, atr_sc = "LOW VOL", "#58a6ff"
    elif atr_pct > 70:
        atr_state, atr_sc = "HIGH VOL", "#f85149"
    else:
        atr_state, atr_sc = "NORMAL",   "#e3b341"
    # Compression zones: ATR below 30th pctile for ≥2 consecutive bars → blue shading
    _atr_30 = np.percentile(atr, 30)
    _compress = atr < _atr_30

    # Relative volume z-score
    vol_mean = float(np.mean(vols))
    vol_std  = float(np.std(vols)) if np.std(vols) > 0 else 1.0
    vol_z    = (vols - vol_mean) / vol_std
    _bull_mask   = closes >= opens
    _bright_mask = vol_z >= 1.5   # highlight spikes ≥1.5σ

    # Volume profile (40 bins)
    N_BINS    = 40
    bin_edges = np.linspace(p_lo, p_hi, N_BINS + 1)
    bin_sz    = bin_edges[1] - bin_edges[0]
    vol_prof  = np.zeros(N_BINS)
    for i in range(n):
        lo_c, hi_c, v = lows[i], highs[i], vols[i]
        span = max(hi_c - lo_c, bin_sz * 0.01)
        for b in range(N_BINS):
            overlap = max(0.0, min(bin_edges[b+1], hi_c) - max(bin_edges[b], lo_c))
            vol_prof[b] += v * overlap / span
    poc_idx   = int(np.argmax(vol_prof))
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    vp_max    = vol_prof.max() or 1
    vp_hvn    = np.percentile(vol_prof, 72)   # HVN threshold
    vp_lvn    = np.percentile(vol_prof, 28)   # LVN threshold

    # Market structure: local swing highs/lows (order=3 bars)
    _ord = 3
    swing_highs, swing_lows = [], []
    for i in range(_ord, n - _ord):
        if all(highs[i] >= highs[i-j] for j in range(1, _ord+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, _ord+1)):
            swing_highs.append((i, highs[i]))
        if all(lows[i]  <= lows[i-j]  for j in range(1, _ord+1)) and \
           all(lows[i]  <= lows[i+j]  for j in range(1, _ord+1)):
            swing_lows.append((i, lows[i]))
    struct_labels = []   # (x, y, label, color, "high"|"low")
    for k in range(1, len(swing_highs)):
        xi, hi = swing_highs[k]; _, hi_p = swing_highs[k-1]
        lbl = "HH" if hi > hi_p else "LH"
        struct_labels.append((xi, hi, lbl, "#3fb950" if lbl == "HH" else "#f85149", "high"))
    for k in range(1, len(swing_lows)):
        xi, lo = swing_lows[k]; _, lo_p = swing_lows[k-1]
        lbl = "HL" if lo > lo_p else "LL"
        struct_labels.append((xi, lo, lbl, "#3fb950" if lbl == "HL" else "#f85149", "low"))

    # Liquidity walls from btc_liq
    _liq      = a.get("btc_liq") or {}
    _liq_anal = _liq.get("liq_analysis", {})
    _bid_walls = _liq_anal.get("bid_walls", []) or []
    _ask_walls = _liq_anal.get("ask_walls", []) or []

    # ── Layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6.5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 0.65, 0.9],
                          width_ratios=[5, 0.7], hspace=0.06, wspace=0.015)
    ax_c  = fig.add_subplot(gs[0, 0])
    ax_vp = fig.add_subplot(gs[0, 1], sharey=ax_c)
    ax_v  = fig.add_subplot(gs[1, 0], sharex=ax_c)
    ax_a  = fig.add_subplot(gs[2, 0], sharex=ax_c)
    for _ax in [ax_c, ax_vp, ax_v, ax_a]: _ax_style(_ax)

    # ── Session shading (Asia / London / NY) ───────────────────────────
    # Asia: 00–07 UTC  London: 07–13 UTC  NY: 13–21 UTC
    # Off-hours (21–00 UTC) shaded grey so the gap doesn't look like a void
    _SESSIONS = [("Asia",    0,  7,  "#58a6ff", 0.045),
                 ("London",  7,  13, "#bc8cff", 0.045),
                 ("NY",      13, 21, "#f0883e", 0.045),
                 ("Off",     21, 24, "#484f58", 0.18 )]
    try:
        _ts_utc = pd.to_datetime(df15.index)
        if _ts_utc.tz is not None:
            _ts_utc = _ts_utc.tz_convert("UTC")
        for _sname, _sh, _eh, _sc, _sa in _SESSIONS:
            _in = np.array([_sh <= t.hour < _eh for t in _ts_utc])
            _seg_start = None
            for i in range(len(_in)):
                if _in[i] and _seg_start is None:
                    _seg_start = i
                elif not _in[i] and _seg_start is not None:
                    for _axs in (ax_c, ax_v):
                        _axs.axvspan(_seg_start - 0.5, i - 0.5, color=_sc, alpha=_sa, zorder=0)
                    ax_c.text(_seg_start + 0.5, p_hi + _pad * 0.55, _sname,
                              fontsize=5, color=_sc, alpha=0.65, va="bottom", zorder=1)
                    _seg_start = None
            if _seg_start is not None:
                for _axs in (ax_c, ax_v):
                    _axs.axvspan(_seg_start - 0.5, n - 0.5, color=_sc, alpha=_sa, zorder=0)
                ax_c.text(_seg_start + 0.5, p_hi + _pad * 0.55, _sname,
                          fontsize=5, color=_sc, alpha=0.65, va="bottom", zorder=1)
    except Exception:
        pass

    # ── Volatility compression zones (amber — distinct from blue Asia shading) ──
    _cseg = None
    for i in range(n):
        if _compress[i] and _cseg is None:
            _cseg = i
        elif not _compress[i] and _cseg is not None:
            ax_c.axvspan(_cseg - 0.5, i - 0.5, color="#e3b341", alpha=0.09, zorder=0)
            _cseg = None
    if _cseg is not None:
        ax_c.axvspan(_cseg - 0.5, n - 0.5, color="#e3b341", alpha=0.09, zorder=0)

    # ── Liquidity wall overlays ─────────────────────────────────────────
    for _wp, _wn in _bid_walls[:4]:
        if p_lo * 0.98 <= _wp <= p_hi * 1.02:
            _wt = min(2.5, 0.6 + float(_wn) / 3e7)
            ax_c.axhline(_wp, color="#3fb950", lw=_wt, ls="--", alpha=0.55, zorder=2)
            ax_c.text(1, _wp, " bid", transform=ax_c.get_yaxis_transform(),
                      fontsize=5.5, color="#3fb950", va="bottom", alpha=0.8)
    for _wp, _wn in _ask_walls[:4]:
        if p_lo * 0.98 <= _wp <= p_hi * 1.02:
            _wt = min(2.5, 0.6 + float(_wn) / 3e7)
            ax_c.axhline(_wp, color="#f85149", lw=_wt, ls="--", alpha=0.55, zorder=2)
            ax_c.text(1, _wp, " ask", transform=ax_c.get_yaxis_transform(),
                      fontsize=5.5, color="#f85149", va="top", alpha=0.8)

    # ── Candles ────────────────────────────────────────────────────────
    ax_c.bar(xs, closes - opens, 0.55, bottom=opens, color=colors, alpha=1.0, zorder=3)
    ax_c.bar(xs, highs  - lows,  0.12, bottom=lows,  color=colors, alpha=0.9, zorder=3)

    # ── EMA ribbon: green fill when EMA8 > EMA21, red when below ───────
    ax_c.fill_between(xs, ema8, ema21, where=(ema8 >= ema21),
                      color="#3fb950", alpha=0.12, zorder=2, interpolate=True)
    ax_c.fill_between(xs, ema8, ema21, where=(ema8 < ema21),
                      color="#f85149", alpha=0.12, zorder=2, interpolate=True)
    ax_c.plot(xs, ema8,  color="#58a6ff", lw=0.85, alpha=0.70, label="EMA8",  zorder=4)
    ax_c.plot(xs, ema21, color="#f0883e", lw=1.75,             label="EMA21", zorder=4)

    # ── VWAP ──────────────────────────────────────────────────────────
    if vwap_sessions:
        for i, (_xs, _vw) in enumerate(vwap_sessions):
            ax_c.plot(_xs, _vw, color="#d2a8ff", lw=1.05, ls="--",
                      label="VWAP" if i == 0 else "_nolegend_", zorder=4)

    # ── POC line ──────────────────────────────────────────────────────
    ax_c.axhline(poc_price, color="#e3b341", lw=0.85, ls=(0,(4,4)), alpha=0.75, zorder=2)
    ax_c.text(0.002, poc_price, " POC \${:,.0f}".format(poc_price),
              transform=ax_c.get_yaxis_transform(),
              color="#e3b341", fontsize=6.5, va="bottom", zorder=5)

    # ── Market structure labels (last 6 chronologically — mix of highs & lows) ─
    _struct_sorted = sorted(struct_labels, key=lambda t: t[0])
    for _sx, _sy, _slbl, _scol, _sside in _struct_sorted[-6:]:
        _yoff = _pad * 0.35 if _sside == "high" else -_pad * 0.35
        ax_c.text(_sx, _sy + _yoff, _slbl, fontsize=6.5, color=_scol,
                  ha="center", va="bottom" if _sside == "high" else "top",
                  fontweight="bold", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.1", fc="#0d1117", ec=_scol, lw=0.5, alpha=0.75))

    # ── Current price tag ─────────────────────────────────────────────
    _cur_col = "#3fb950" if closes[-1] >= opens[-1] else "#f85149"
    ax_c.axhline(cur, color=_cur_col, lw=0.6, ls=":", alpha=0.45, zorder=2)

    # ── Axes ──────────────────────────────────────────────────────────
    tick_pos  = list(range(0, n, 8))
    tick_lbls = []
    for i in tick_pos:
        try:   tick_lbls.append(pd.Timestamp(df15.index[i]).strftime("%H:%M"))
        except: tick_lbls.append("")

    ax_c.set_xlim(-1, n)
    ax_c.set_ylim(p_lo - _pad, p_hi + _pad)
    ax_c.set_xticks([])
    ax_c.tick_params(bottom=False, which="both")
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "\${:,.0f}".format(v)))
    ax_c.tick_params(axis="y", labelsize=7, labelcolor="#8b949e")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    ema_lbl  = "EMA8 > EMA21 ▲" if ema8[-1] > ema21[-1] else "EMA8 < EMA21 ▼"
    ema_col  = "#3fb950" if ema8[-1] > ema21[-1] else "#f85149"
    vwap_str = ("▲ above VWAP" if vwap is not None and cur > vwap[-1] else
                "▼ below VWAP" if vwap is not None else "")
    _atr_chg_str = "{:+.0f}% ATR".format(atr_chg_pct) if abs(atr_chg_pct) >= 5 else ""

    ax_c.set_title("BTC / USDT  ·  15-min  ·  24h", color="#c9d1d9",
                   fontsize=9, loc="left", pad=18, fontweight="bold")
    _meta = "   ".join(filter(None, [vwap_str,
                                     "ATR {} ({:.2f}%)".format(atr_state, atr_now / cur * 100),
                                     _atr_chg_str,
                                     "POC \${:,.0f}".format(poc_price)]))
    ax_c.text(0.0, 1.02, ema_lbl + "   " + _meta,
              transform=ax_c.transAxes, fontsize=6.8, color="#8b949e", va="bottom", ha="left")
    ax_c.text(0.0, 1.02, ema_lbl, transform=ax_c.transAxes,
              fontsize=6.8, color=ema_col, va="bottom", ha="left")

    import matplotlib.lines  as _mlines
    import matplotlib.patches as _mpatches
    _leg_handles = [
        _mlines.Line2D([], [], color="#58a6ff", lw=0.9, alpha=0.7, label="EMA 8"),
        _mlines.Line2D([], [], color="#f0883e", lw=1.8,             label="EMA 21"),
        _mlines.Line2D([], [], color="#d2a8ff", lw=1.0, ls="--",   label="VWAP"),
        _mlines.Line2D([], [], color="#e3b341", lw=0.9, ls=(0,(4,4)), label="POC"),
        _mpatches.Patch(fc="#3fb950", alpha=0.40, ec="none", label="Bull ribbon"),
        _mpatches.Patch(fc="#f85149", alpha=0.40, ec="none", label="Bear ribbon"),
        _mpatches.Patch(fc="#58a6ff", alpha=0.30, ec="none", label="Asia 00–07"),
        _mpatches.Patch(fc="#bc8cff", alpha=0.30, ec="none", label="London 07–13"),
        _mpatches.Patch(fc="#f0883e", alpha=0.30, ec="none", label="NY 13–21"),
        _mpatches.Patch(fc="#484f58", alpha=0.50, ec="none", label="Off 21–00"),
        _mpatches.Patch(fc="#e3b341", alpha=0.35, ec="none", label="Squeeze zone"),
        _mlines.Line2D([], [], color="#3fb950", lw=0, marker="s", ms=5,
                       markerfacecolor="#3fb950", label="HH / HL"),
        _mlines.Line2D([], [], color="#f85149", lw=0, marker="s", ms=5,
                       markerfacecolor="#f85149", label="LH / LL"),
        _mlines.Line2D([], [], color="#3fb950", lw=1.4, ls="--", label="Bid wall"),
        _mlines.Line2D([], [], color="#f85149", lw=1.4, ls="--", label="Ask wall"),
        _mpatches.Patch(fc="#3fb950", alpha=0.9, ec="none", label="HVN (high vol)"),
        _mpatches.Patch(fc="#f85149", alpha=0.9, ec="none", label="LVN (low vol)"),
    ]
    # Legend placed OUTSIDE below the chart — no chart overlap

    # ── Volume Profile with HVN/LVN coloring ───────────────────────────
    bin_mid   = (bin_edges[:-1] + bin_edges[1:]) / 2
    vp_colors = []
    for i in range(N_BINS):
        if i == poc_idx:             vp_colors.append("#e3b341")   # POC — gold
        elif vol_prof[i] >= vp_hvn:  vp_colors.append("#3fb950")   # HVN — green (acceptance)
        elif vol_prof[i] <= vp_lvn:  vp_colors.append("#f85149")   # LVN — red (fast-move zone)
        else:                        vp_colors.append("#21262d")
    ax_vp.barh(bin_mid, vol_prof / vp_max, height=bin_sz * 0.9,
               color=vp_colors, alpha=0.9, zorder=1)
    ax_vp.axhline(cur, color=_cur_col, lw=0.6, ls=":", alpha=0.4, zorder=2)
    ax_vp.text(0.05, cur, "\${:,.0f}".format(cur),
               color=_cur_col, fontsize=7, va="center", fontweight="bold", zorder=5,
               bbox=dict(boxstyle="round,pad=0.18", fc="#0d1117", ec=_cur_col, lw=0.9, alpha=0.95))
    ax_vp.set_xlim(0, 1.1); ax_vp.set_xticks([])
    ax_vp.tick_params(labelleft=False, bottom=False, left=False, right=False)
    ax_vp.set_ylim(p_lo - _pad, p_hi + _pad)
    ax_vp.set_title("Vol\nProfile", color="#484f58", fontsize=6, pad=2)
    for _sp in ax_vp.spines.values(): _sp.set_visible(False)

    # ── Relative-volume panel with z-score coloring ─────────────────────
    # Dim bars when normal; bright when ≥ 1.5σ spike
    _v2sig = vol_mean + 2.0 * vol_std
    ax_v.bar(xs[~_bright_mask &  _bull_mask], vols[~_bright_mask &  _bull_mask],
             0.65, color="#1e3a2a", alpha=0.7, zorder=2)
    ax_v.bar(xs[~_bright_mask & ~_bull_mask], vols[~_bright_mask & ~_bull_mask],
             0.65, color="#2d1b1b", alpha=0.7, zorder=2)
    ax_v.bar(xs[ _bright_mask &  _bull_mask], vols[ _bright_mask &  _bull_mask],
             0.65, color="#3fb950", alpha=0.95, zorder=3)
    ax_v.bar(xs[ _bright_mask & ~_bull_mask], vols[ _bright_mask & ~_bull_mask],
             0.65, color="#f85149", alpha=0.95, zorder=3)
    ax_v.axhline(vol_mean, color="#30363d", lw=0.7, ls="--", alpha=0.65, zorder=1)
    ax_v.axhline(_v2sig,   color="#ffd700", lw=0.6, ls=":",  alpha=0.55, zorder=1)
    # R.Vol legend
    _rvol_legend = [
        _mpatches.Patch(fc="#3fb950", alpha=0.9, ec="none", label="Bull spike ≥1.5σ"),
        _mpatches.Patch(fc="#f85149", alpha=0.9, ec="none", label="Bear spike ≥1.5σ"),
        _mpatches.Patch(fc="#1e3a2a", alpha=0.9, ec="none", label="Normal vol"),
        _mlines.Line2D([], [], color="#ffd700", lw=0.8, ls=":", label="2σ threshold"),
    ]
    ax_v.legend(handles=_rvol_legend, loc="upper left", fontsize=5.2,
                facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.85,
                ncol=4, edgecolor="#30363d", borderpad=0.4,
                handlelength=1.0, handletextpad=0.4, columnspacing=0.6)
    ax_v.set_xticks([])
    ax_v.set_xlim(-1, n + 2)
    ax_v.set_ylim(0, max(vols.max() * 1.55, _v2sig * 1.1))
    ax_v.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: "{:.0f}k".format(v / 1e3) if v >= 1000 else "{:.1f}".format(v)))
    ax_v.tick_params(labelsize=5.5, labelcolor="#6e7681")
    ax_v.set_ylabel("R.Vol", color="#484f58", fontsize=6.5)
    ax_v.spines["top"].set_visible(False); ax_v.spines["right"].set_visible(False)

    # ── ATR regime panel (percentile bars, not raw values) ──────────────
    _atr_reg_cols = np.where(atr_rolling_pct < 30, "#58a6ff",
                    np.where(atr_rolling_pct > 70, "#f85149", "#e3b341"))
    ax_a.bar(xs, atr_rolling_pct, 0.65, color=_atr_reg_cols, alpha=0.75)
    ax_a.axhline(30, color="#30363d", lw=0.6, ls="--", alpha=0.5)
    ax_a.axhline(70, color="#30363d", lw=0.6, ls="--", alpha=0.5)
    ax_a.fill_between(xs, np.minimum(atr_rolling_pct, 30), 0,
                      color="#58a6ff", alpha=0.08, zorder=0)   # low-vol shading
    ax_a.set_ylim(0, 112)
    ax_a.set_yticks([30, 70])
    ax_a.set_yticklabels(["30", "70"], fontsize=5, color="#6e7681")
    ax_a.set_xticks(tick_pos)
    ax_a.set_xticklabels(tick_lbls, fontsize=6, color="#6e7681")
    ax_a.set_xlim(-1, n + 2)
    ax_a.set_ylabel("ATR%ile", color="#484f58", fontsize=6)
    _chg_tag = "  {}{:.0f}% vs avg".format("+" if atr_chg_pct >= 0 else "", atr_chg_pct) \
               if abs(atr_chg_pct) >= 5 else ""
    ax_a.text(0.99, 0.88,
              "ATR {}  {:.2f}%{}".format(atr_state, atr_now / cur * 100, _chg_tag),
              transform=ax_a.transAxes, fontsize=7, color=atr_sc, ha="right", va="top",
              fontweight="bold")
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    # ATR%ile inline legend (small, inside thin panel — doesn't block price data)
    _atr_legend = [
        _mpatches.Patch(fc="#58a6ff", alpha=0.85, ec="none", label="LOW <30th"),
        _mpatches.Patch(fc="#e3b341", alpha=0.85, ec="none", label="NORMAL"),
        _mpatches.Patch(fc="#f85149", alpha=0.85, ec="none", label="HIGH >70th"),
    ]
    ax_a.legend(handles=_atr_legend, loc="upper left", fontsize=5.2,
                facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.85,
                ncol=3, edgecolor="#30363d", borderpad=0.4,
                handlelength=1.0, handletextpad=0.4, columnspacing=0.6)

    # ── Main legend below chart (outside all panels, no overlap) ────────
    # tight_layout reserves the bottom strip; fig.legend sits in that margin.
    plt.tight_layout(pad=0.6, rect=[0, 0.10, 1, 1])
    fig.legend(handles=_leg_handles,
               loc="lower center",
               bbox_to_anchor=(0.42, 0.01),   # centred under candle area
               fontsize=5.8,
               facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.92,
               ncol=7, edgecolor="#30363d", borderpad=0.55,
               handlelength=1.25, handletextpad=0.45, columnspacing=0.7)
    return fig


def fig_rsi(a: dict) -> plt.Figure:
    """RSI panel with BTC price below (matches expanded chart format)."""
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(13, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for _ax in (ax, axp): _ax_style(_ax)
    ax.plot(plot_df.index, plot_df["RSI"], color="#bc8cff", lw=1.2)
    ax.axhline(70, color="#f85149", ls="--", alpha=0.45, lw=0.9)
    ax.axhline(50, color="#484f58", ls=":",  alpha=0.4,  lw=0.8)
    ax.axhline(30, color="#3fb950", ls="--", alpha=0.45, lw=0.9)
    ax.fill_between(plot_df.index, plot_df["RSI"], 30, where=(plot_df["RSI"] < 30), color="#3fb950", alpha=0.15)
    ax.fill_between(plot_df.index, plot_df["RSI"], 70, where=(plot_df["RSI"] > 70), color="#f85149", alpha=0.12)
    rsi_now = float(plot_df["RSI"].dropna().iloc[-1]) if not plot_df["RSI"].dropna().empty else 50
    state = "Overbought" if rsi_now > 70 else ("Oversold" if rsi_now < 30 else "Neutral")
    col   = "#f85149" if rsi_now > 70 else ("#3fb950" if rsi_now < 30 else "#bc8cff")
    ax.text(0.995, 0.95, f"RSI {rsi_now:.1f}  ·  {state}", transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color=col)
    ax.set_ylim(0, 100); ax.set_ylabel("RSI", color="#8b949e", fontsize=8)
    ax.set_title("RSI (14)", color="#8b949e", fontsize=9, loc="left", pad=4)
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_macd(a: dict) -> plt.Figure:
    """MACD panel with BTC price below (matches expanded chart format)."""
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(13, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for _ax in (ax, axp): _ax_style(_ax)
    plot_hist = a["df"]["MACD_Hist"].iloc[-365:]
    hist_cols = np.where(plot_hist >= 0, "#3fb950", "#f85149")
    ax.bar(plot_df.index, plot_hist.values, color=hist_cols, alpha=0.55, width=0.6)
    ax.plot(plot_df.index, a["df"]["MACD"].iloc[-365:].values,        color="#58a6ff", lw=1.1, label="MACD")
    ax.plot(plot_df.index, a["df"]["MACD_Signal"].iloc[-365:].values, color="#f0883e", lw=1.1, label="Signal")
    ax.axhline(0, color="#484f58", lw=0.7)
    ax.set_ylabel("MACD", color="#8b949e", fontsize=8)
    ax.set_title("MACD (12, 26, 9)", color="#8b949e", fontsize=9, loc="left", pad=4)
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left", framealpha=0.85)
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_etf_flows(a: dict) -> plt.Figure:
    """ETF composite flows (top) + BTC price (bottom), shared x-axis."""
    crypto_sig    = a["crypto_sig"]
    plot_df       = a["plot_df"]
    composite_sv  = crypto_sig.get("etf_daily_composite")
    etfs_loaded   = crypto_sig.get("etf_tickers_loaded", [])
    flow_stats    = crypto_sig.get("etf_flow_stats", {})
    etf_trend_str = crypto_sig.get("etf_flow_trend", "Neutral")

    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.04)
    ax  = fig.add_subplot(gs[0])          # ETF flow panel
    ax_price = fig.add_subplot(gs[1], sharex=ax)  # BTC price panel, shared x
    _ax_style(ax); _ax_style(ax_price)

    if composite_sv is None or len(composite_sv) < 5:
        ax.text(0.5, 0.5, "ETF Flow Data Unavailable", ha="center",
                color="#8b949e", transform=ax.transAxes)
        return fig

    try:
        def _to_dict(s):
            idx = pd.to_datetime(s.index)
            if idx.tz is not None: idx = idx.tz_convert(None)
            return dict(zip(idx.strftime("%Y-%m-%d"), s.values))

        sv_dict = _to_dict(composite_sv)

        dates_in_plot   = pd.to_datetime(plot_df.index).strftime("%Y-%m-%d").tolist()
        first_data_date = min(sv_dict.keys()) if sv_dict else None

        sv_vals, cum_vals, x_dates = [], [], []
        cum_sum = 0.0
        for ds in dates_in_plot:
            if first_data_date and ds < first_data_date:
                continue
            x_dates.append(ds)
            if ds in sv_dict:
                val = sv_dict[ds]
                cum_sum += val
                sv_vals.append(val)
            else:
                sv_vals.append(np.nan)
            cum_vals.append(cum_sum)

        sv_arr  = np.array(sv_vals,  dtype=float)
        cum_arr = np.array(cum_vals, dtype=float)

        # Align x_vals to the trimmed date set
        mask   = plot_df.index.strftime("%Y-%m-%d").isin(x_dates)
        x_vals = plot_df.loc[mask].index
        if len(x_vals) != len(sv_arr):
            x_vals = pd.to_datetime(x_dates)

        roll20 = pd.Series(sv_arr).rolling(20, min_periods=5).mean().values
        roll5  = pd.Series(sv_arr).rolling(5,  min_periods=2).mean().values
        _std   = np.nanstd(roll20[~np.isnan(roll20)])
        _mean  = np.nanmean(roll20[~np.isnan(roll20)])

        # ── Top panel: ETF flow ──────────────────────────────────────
        flow_col = "#3fb950" if float(np.nanmax([roll20[-1], 0])) >= 0 and not np.isnan(roll20[-1]) and roll20[-1] > 0 else "#f85149"
        ax.plot(x_vals, roll20, color=flow_col, lw=1.8, label="20d Net Flow", zorder=3)
        ax.fill_between(x_vals, roll20, 0, where=(roll20 >= 0), color="#3fb950", alpha=0.13, zorder=1)
        ax.fill_between(x_vals, roll20, 0, where=(roll20 < 0),  color="#f85149", alpha=0.13, zorder=1)
        ax.plot(x_vals, roll5, color="#bc8cff", lw=0.9, ls="--", alpha=0.55, label="5d Fast", zorder=2)
        ax.axhline(0, color="#8b949e", lw=1.0, ls="--", alpha=0.7)
        if _std > 0:
            ax.axhline(_mean + 2 * _std, color="#3fb950", lw=0.4, ls=":", alpha=0.35)
            ax.axhline(_mean - 2 * _std, color="#f85149", lw=0.4, ls=":", alpha=0.35)

        # Cumulative on right axis of top panel
        ax_c = ax.twinx()
        ax_c.plot(x_vals, cum_arr, color="#ffd700", lw=1.3, alpha=0.7, label="Cumulative")
        ax_c.set_ylabel("Cumulative", color="#ffd700", fontsize=7, alpha=0.8)
        ax_c.tick_params(colors="#ffd700", labelsize=6)
        for sp in ax_c.spines.values(): sp.set_visible(False)
        ax_c.spines["right"].set_visible(True)
        ax_c.spines["right"].set_color("#444c56")
        ax_c.spines["right"].set_linewidth(0.5)

        trend_col = "#3fb950" if etf_trend_str == "Positive" else "#f85149" if etf_trend_str == "Negative" else "#8b949e"
        n_in  = flow_stats.get("n_inflow",  0)
        n_out = flow_stats.get("n_outflow", 0)
        n_tot = flow_stats.get("n_etfs",    len(etfs_loaded))
        accel = "⚡" if flow_stats.get("accelerating") else ""
        ax.text(0.005, 0.97,
                f"{etf_trend_str} {accel}  ·  {n_in} inflow / {n_out} outflow  ({n_tot} ETFs)",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left", color=trend_col,
                bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=trend_col, alpha=0.85, lw=0.7))
        etf_str = " · ".join(etfs_loaded)
        ax.set_title(f"BTC Spot ETF Composite Flow  ({etf_str})", color="#8b949e",
                     fontsize=8, loc="left", pad=5)
        ax.set_ylabel("20d Net Flow", color="#8b949e", fontsize=8)
        ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9",
                  loc="lower left", framealpha=0.85)
        plt.setp(ax.get_xticklabels(), visible=False)  # hide top panel x labels

        # ── Bottom panel: BTC price on same time scale ───────────────
        price_full = plot_df["Close"].values.astype(float)
        price_vals = plot_df.loc[mask, "Close"].values.astype(float) if len(x_vals) == mask.sum() else price_full[-len(x_vals):]
        ax_price.plot(x_vals, price_vals, color="#58a6ff", lw=1.4, zorder=3)
        ax_price.fill_between(x_vals, price_vals, price_vals.min(),
                              color="#58a6ff", alpha=0.08, zorder=1)
        ax_price.set_ylabel("BTC Price", color="#8b949e", fontsize=7)
        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
        ax_price.tick_params(axis="y", colors="#8b949e", labelsize=6)
        ax_price.tick_params(axis="x", colors="#8b949e", labelsize=6)

    except Exception as e:
        ax.text(0.5, 0.5, f"ETF error: {e}", ha="center",
                color="#f85149", transform=ax.transAxes, fontsize=7)

    fig.subplots_adjust(right=0.88, left=0.06, top=0.94, bottom=0.08)
    return fig


def fig_liquidity_panel(a: dict):
    """Panel 5: Liquidity heatmap. Returns (fig, narrative_text)."""
    btc_liq = a["btc_liq"]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0d1117"); _ax_style(ax)
    try:
        plot_liquidity_depth(ax, btc_liq, short_df=a.get("short_df"))
    except Exception as e:
        ax.text(0.5, 0.5, f"Heatmap error: {e}", ha="center", color="#f85149", transform=ax.transAxes)
    # Extra bottom margin so the legend below the x-axis isn't clipped
    plt.subplots_adjust(bottom=0.14, top=0.93, left=0.06, right=0.88)
    narrative = btc_liq.get("liq_narrative", "") if btc_liq else ""
    return fig, narrative



def fig_bollinger_detail(a: dict) -> plt.Figure:
    """Bollinger Bands detail: %B oscillator + bandwidth."""
    plot_df = a["plot_df"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 3.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.1})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2): _ax_style(ax)

    # %B — where price sits in the band (0=lower, 1=upper)
    pctb = plot_df["BB_pctb"]
    ax1.plot(plot_df.index, pctb, color="#bc8cff", lw=1.2)
    ax1.axhline(1.0, color="#f85149", ls="--", lw=0.8, alpha=0.5)
    ax1.axhline(0.5, color="#484f58", ls=":",  lw=0.7, alpha=0.4)
    ax1.axhline(0.0, color="#3fb950", ls="--", lw=0.8, alpha=0.5)
    ax1.fill_between(plot_df.index, pctb, 1.0, where=(pctb > 1.0), color="#f85149", alpha=0.2)
    ax1.fill_between(plot_df.index, pctb, 0.0, where=(pctb < 0.0), color="#3fb950", alpha=0.2)
    ax1.set_ylabel("%B", color="#8b949e", fontsize=8)
    pctb_now = float(pctb.dropna().iloc[-1]) if not pctb.dropna().empty else 0.5
    ax1.text(0.995, 0.92, f"%B {pctb_now:.2f}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right", color="#bc8cff")

    # Bandwidth — squeeze detector
    bw = plot_df["BB_bw"]
    bw_mean = float(bw.dropna().mean()) if not bw.dropna().empty else 1
    ax2.plot(plot_df.index, bw, color="#58a6ff", lw=1.2, label="Bandwidth %")
    ax2.axhline(bw_mean, color="#484f58", ls="--", lw=0.8, alpha=0.5, label=f"Mean {bw_mean:.1f}%")
    ax2.fill_between(plot_df.index, bw, bw_mean * 0.6,
                     where=(bw < bw_mean * 0.6), color="#ffd700", alpha=0.2, label="Squeeze zone")
    ax2.set_ylabel("Bandwidth %", color="#8b949e", fontsize=8)
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left", framealpha=0.85)

    ax1.set_title("Bollinger Bands  ·  %B position  ·  Bandwidth squeeze detector",
                  color="#8b949e", fontsize=9, loc="left", pad=5)
    plt.tight_layout(pad=0.5)
    return fig


def fig_stochastic(a: dict) -> plt.Figure:
    """Stochastic Oscillator %K / %D."""
    plot_df = a["plot_df"]
    fig, ax = plt.subplots(figsize=(13, 2.5))
    fig.patch.set_facecolor("#0d1117"); _ax_style(ax)

    k = plot_df["Stoch_K"]
    d = plot_df["Stoch_D"]
    ax.plot(plot_df.index, k, color="#58a6ff", lw=1.3, label="%K (fast)")
    ax.plot(plot_df.index, d, color="#f0883e", lw=1.0, ls="--", label="%D (signal)")
    ax.axhline(80, color="#f85149", ls="--", lw=0.85, alpha=0.5)
    ax.axhline(50, color="#484f58", ls=":",  lw=0.75, alpha=0.4)
    ax.axhline(20, color="#3fb950", ls="--", lw=0.85, alpha=0.5)
    ax.fill_between(plot_df.index, k, 80, where=(k > 80), color="#f85149", alpha=0.12)
    ax.fill_between(plot_df.index, k, 20, where=(k < 20), color="#3fb950", alpha=0.12)

    # Crossover signals
    k_arr = k.fillna(50).values
    d_arr = d.fillna(50).values
    buy_sig  = (k_arr[1:] > d_arr[1:]) & (k_arr[:-1] <= d_arr[:-1]) & (k_arr[1:] < 30)
    sell_sig = (k_arr[1:] < d_arr[1:]) & (k_arr[:-1] >= d_arr[:-1]) & (k_arr[1:] > 70)
    x_idx = plot_df.index[1:]
    if buy_sig.any():
        ax.scatter(x_idx[buy_sig],  k_arr[1:][buy_sig],  marker="^", color="#3fb950", s=40, zorder=5, label="Buy cross")
    if sell_sig.any():
        ax.scatter(x_idx[sell_sig], k_arr[1:][sell_sig], marker="v", color="#f85149", s=40, zorder=5, label="Sell cross")

    k_now = float(k.dropna().iloc[-1]) if not k.dropna().empty else 50
    d_now = float(d.dropna().iloc[-1]) if not d.dropna().empty else 50
    state = "Overbought" if k_now > 80 else ("Oversold" if k_now < 20 else "Neutral")
    col   = "#f85149" if k_now > 80 else ("#3fb950" if k_now < 20 else "#8b949e")
    ax.text(0.995, 0.95, f"Stoch %K {k_now:.0f}  %D {d_now:.0f}  ·  {state}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", color=col)

    ax.set_ylim(-5, 105)
    ax.set_ylabel("Stochastic", color="#8b949e", fontsize=8)
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left",
              ncol=2, framealpha=0.85)
    ax.set_title("Stochastic Oscillator (14,3)  ·  %K/%D crossovers",
                 color="#8b949e", fontsize=9, loc="left", pad=5)
    plt.tight_layout(pad=0.5)
    return fig


def fig_ichimoku(a: dict) -> plt.Figure:
    """Ichimoku Cloud chart."""
    plot_df = a["plot_df"]
    price   = a["price"]
    fig, ax = plt.subplots(figsize=(13, 5.0))
    fig.patch.set_facecolor("#0d1117"); _ax_style(ax)

    ax.plot(plot_df.index, plot_df.Close, color="#c9d1d9", lw=1.3, label="Price", zorder=5)

    tenkan = plot_df["Ichi_Tenkan"]
    kijun  = plot_df["Ichi_Kijun"]
    span_a = plot_df["Ichi_SpanA"]
    span_b = plot_df["Ichi_SpanB"]
    chikou = plot_df["Ichi_Chikou"]

    ax.plot(plot_df.index, tenkan, color="#f85149", lw=1.1, label="Tenkan (9)")
    ax.plot(plot_df.index, kijun,  color="#58a6ff", lw=1.2, label="Kijun (26)")
    ax.plot(plot_df.index, chikou, color="#ffd700", lw=0.9, ls="--", alpha=0.6, label="Chikou (lag)")

    # Cloud fill — green when Span A > Span B, red otherwise
    ax.plot(plot_df.index, span_a, color="#3fb950", lw=0.7, alpha=0.5)
    ax.plot(plot_df.index, span_b, color="#f85149", lw=0.7, alpha=0.5)
    ax.fill_between(plot_df.index, span_a, span_b,
                    where=(span_a >= span_b), color="#3fb950", alpha=0.08, label="Cloud (bull)")
    ax.fill_between(plot_df.index, span_a, span_b,
                    where=(span_a <  span_b), color="#f85149", alpha=0.08, label="Cloud (bear)")

    # Signal annotation
    try:
        t_now = float(tenkan.dropna().iloc[-1])
        k_now = float(kijun.dropna().iloc[-1])
        sa_now = float(span_a.dropna().iloc[-1])
        sb_now = float(span_b.dropna().iloc[-1])
        cloud_top = max(sa_now, sb_now)
        cloud_bot = min(sa_now, sb_now)
        if price > cloud_top:
            signal, sig_col = "Price ABOVE cloud — Bullish", "#3fb950"
        elif price < cloud_bot:
            signal, sig_col = "Price BELOW cloud — Bearish", "#f85149"
        else:
            signal, sig_col = "Price INSIDE cloud — Indecision", "#8b949e"
        tk_cross = "TK Cross ↑" if t_now > k_now else "TK Cross ↓"
        ax.text(0.005, 0.97, f"{signal}  ·  {tk_cross}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left", color=sig_col,
                bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=sig_col, alpha=0.9, lw=0.7))
    except Exception:
        pass

    ax.set_title("Ichimoku Cloud  ·  Tenkan / Kijun / Chikou / Kumo",
                 color="#8b949e", fontsize=9, loc="left", pad=5)
    ax.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9",
              framealpha=0.9, ncol=3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    plt.tight_layout(pad=0.8)
    return fig


def fig_ac_atr(a: dict) -> plt.Figure:
    """Accelerator Oscillator + ATR side by side."""
    plot_df = a["plot_df"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 2.8))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2): _ax_style(ax)

    # Accelerator Oscillator
    ac     = plot_df["AC"]
    ao     = plot_df["AO"]
    ac_col = np.where(ac >= 0, "#3fb950", "#f85149")
    ax1.bar(plot_df.index, ac.values, color=ac_col, alpha=0.7, width=0.6)
    ax1.plot(plot_df.index, ao, color="#58a6ff", lw=1.0, alpha=0.6, label="AO")
    ax1.axhline(0, color="#484f58", lw=0.8)
    ac_now = float(ac.dropna().iloc[-1]) if not ac.dropna().empty else 0
    ao_now = float(ao.dropna().iloc[-1]) if not ao.dropna().empty else 0
    accel_state = "Accelerating ↑" if ac_now > 0 else "Decelerating ↓"
    ac_col_text = "#3fb950" if ac_now > 0 else "#f85149"
    ax1.text(0.995, 0.95, f"AC {ac_now:+.0f}  ·  {accel_state}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right", color=ac_col_text)
    ax1.set_title("Accelerator Oscillator (AC)", color="#8b949e", fontsize=9, loc="left", pad=4)
    ax1.set_ylabel("AC / AO", color="#8b949e", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="lower left")

    # ATR
    atr = plot_df["ATR"]
    ax2.plot(plot_df.index, atr, color="#f0883e", lw=1.3, label="ATR (14)")
    atr_mean = float(atr.dropna().mean()) if not atr.dropna().empty else 1
    ax2.axhline(atr_mean, color="#484f58", ls="--", lw=0.8, alpha=0.5, label=f"Mean ${atr_mean:,.0f}")
    ax2.fill_between(plot_df.index, atr, atr_mean,
                     where=(atr > atr_mean * 1.3), color="#f0883e", alpha=0.15, label="High vol zone")
    atr_now = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0
    price   = a["price"]
    atr_pct = atr_now / price * 100
    vol_state = "High" if atr_now > atr_mean * 1.3 else ("Low" if atr_now < atr_mean * 0.7 else "Normal")
    col = "#f85149" if vol_state == "High" else ("#3fb950" if vol_state == "Low" else "#8b949e")
    ax2.text(0.995, 0.95, f"ATR ${atr_now:,.0f}  ({atr_pct:.1f}%)  ·  {vol_state} vol",
             transform=ax2.transAxes, fontsize=8, va="top", ha="right", color=col)
    ax2.set_title("Average True Range  ·  Volatility", color="#8b949e", fontsize=9, loc="left", pad=4)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left")

    plt.tight_layout(pad=0.8)
    return fig


def fig_fibonacci_detail(a: dict) -> plt.Figure:
    """Fibonacci retracement levels with price context."""
    plot_df = a["plot_df"]
    price   = a["price"]
    fib     = a["fib"]
    fig, ax = plt.subplots(figsize=(13, 4.0))
    fig.patch.set_facecolor("#0d1117"); _ax_style(ax)

    colors = np.where(plot_df.Close >= plot_df.Open, "#3fb950", "#f85149")
    ax.bar(plot_df.index, plot_df.Close - plot_df.Open, 0.6, bottom=plot_df.Open, color=colors, alpha=0.8)
    ax.bar(plot_df.index, plot_df.High  - plot_df.Low,  0.1, bottom=plot_df.Low,  color=colors, alpha=0.8)

    fib_palette = {
        "0.0%": "#e6edf3", "23.6%": "#58a6ff", "38.2%": "#3fb950",
        "50.0%": "#ffd700", "61.8%": "#f0883e", "78.6%": "#f85149", "100.0%": "#e6edf3",
    }
    for label, lvl in fib["levels"].items():
        col = fib_palette.get(label, "#8b949e")
        ax.axhline(lvl, color=col, lw=1.1, alpha=0.75, ls="--")
        ax.annotate(f"  {label}  ${lvl:,.0f}", xy=(1.002, lvl),
                    xycoords=ax.get_yaxis_transform(),
                    fontsize=7.5, color=col, va="center", ha="left", clip_on=False)

    # Highlight where price currently sits
    ax.axhline(price, color="#ffffff", lw=1.2, ls="-", alpha=0.6)
    # Current price label — nudge away from nearest fib level to avoid overlap
    _fib_prices = list(fib["levels"].values())
    _nearest_fib = min(_fib_prices, key=lambda v: abs(v - price))
    _price_y = price
    if abs(price - _nearest_fib) / price < 0.015:
        _price_y = price - price * 0.02 if _nearest_fib > price else price + price * 0.02
    ax.annotate(f"  ▶ ${price:,.0f}",
                xy=(1.002, _price_y), xycoords=ax.get_yaxis_transform(),
                fontsize=7.5, color="#ffffff", va="center", ha="left", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.2", fc="#161b22", ec="#ffffff", alpha=0.9, lw=0.7))

    # Find nearest levels
    lvl_vals = sorted(fib["levels"].values())
    below = [v for v in lvl_vals if v < price]
    above = [v for v in lvl_vals if v > price]
    if below and above:
        range_size = above[0] - below[-1]
        pct_in = (price - below[-1]) / range_size * 100 if range_size > 0 else 50
        ax.text(0.005, 0.97,
                f"Price \${price:,.0f}  ·  Between {[k for k,v in fib['levels'].items() if v==below[-1]][0]} and {[k for k,v in fib['levels'].items() if v==above[0]][0]}  ·  {pct_in:.0f}% through zone",
                transform=ax.transAxes, fontsize=8, va="top", ha="left", color="#ffd700",
                bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#ffd700", alpha=0.85, lw=0.7))

    ax.set_title(f"Fibonacci Retracement  ·  Swing High \${fib['high']:,.0f}  →  Swing Low \${fib['low']:,.0f}",
                 color="#8b949e", fontsize=9, loc="left", pad=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    plt.tight_layout(pad=0.8)
    return fig


# ── Expanded indicator views (price chart + indicator, shared x-axis) ────────

def _add_price_panel(ax, plot_df, price):
    """Draws a BTC/USD close price line into an existing axes."""
    ax.plot(plot_df.index, plot_df["Close"], color="#58a6ff", lw=1.3)
    ax.fill_between(plot_df.index, plot_df["Close"], float(plot_df["Close"].min()), color="#58a6ff", alpha=0.06)
    ax.axhline(price, color="#ffffff", lw=0.8, ls="--", alpha=0.5)
    ax.text(0.995, 0.95, f"BTC/USD  ${price:,.0f}", transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="#c9d1d9")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.set_title("BTC/USD — Close Price", color="#8b949e", fontsize=9, loc="left", pad=4)


def fig_speedometer_24h(score: float, label: str, color: str) -> plt.Figure:
    """Half-circle speedometer gauge for the 24h bias score."""
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(4.6, 2.9))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0d1117")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.72, 1.3)

    sc = float(np.clip(score, -100, 100))

    def s_to_deg(s):
        return 180.0 - (s + 100.0) / 200.0 * 180.0

    # Dark background track
    ax.add_patch(mpatches.Arc((0, 0), 2.0, 2.0, angle=0,
                              theta1=0, theta2=180,
                              color="#1c2128", lw=22, zorder=1))

    # Colored zone bands
    for lo, hi, col, alpha in [
        (-100, -40, "#f85149", 0.55),
        (-40,  -20, "#f0883e", 0.55),
        (-20,   20, "#484f58", 0.45),
        (20,    40, "#58a6ff", 0.55),
        (40,   100, "#3fb950", 0.55),
    ]:
        ax.add_patch(mpatches.Arc((0, 0), 2.0, 2.0, angle=0,
                                  theta1=s_to_deg(hi), theta2=s_to_deg(lo),
                                  color=col, lw=22, alpha=alpha, zorder=2))

    # Active highlight from 0 → score
    if abs(sc) > 2:
        t0, ts = s_to_deg(0), s_to_deg(sc)
        ax.add_patch(mpatches.Arc((0, 0), 2.0, 2.0, angle=0,
                                  theta1=min(t0, ts), theta2=max(t0, ts),
                                  color=color, lw=22, alpha=0.92, zorder=3))

    # Tick marks at key levels
    for ts in (-40, -20, 0, 20, 40):
        rad = np.deg2rad(s_to_deg(ts))
        ax.plot([0.87 * np.cos(rad), 1.02 * np.cos(rad)],
                [0.87 * np.sin(rad), 1.02 * np.sin(rad)],
                color="#8b949e", lw=1.5, zorder=4)

    # Needle
    needle_rad = np.deg2rad(s_to_deg(sc))
    ax.plot([0, 0.80 * np.cos(needle_rad)],
            [0, 0.80 * np.sin(needle_rad)],
            color="white", lw=2.8, zorder=6, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.065, color="#58a6ff", zorder=7))

    # Zone text labels
    for s_pos, txt, tcol in [(-72, "BEAR", "#f85149"),
                              (0,   "NEUT", "#484f58"),
                              (72,  "BULL", "#3fb950")]:
        rad = np.deg2rad(s_to_deg(s_pos))
        ax.text(1.21 * np.cos(rad), 1.21 * np.sin(rad), txt,
                ha="center", va="center", fontsize=6.5, color=tcol, fontweight="700")

    # Score + label
    sign = "+" if sc >= 0 else ""
    ax.text(0, -0.10, f"{sign}{sc:.0f}", ha="center", va="center",
            fontsize=26, fontweight="900", color=color, fontfamily="monospace")
    ax.text(0, -0.36, label, ha="center", va="center",
            fontsize=9.5, color="#c9d1d9", fontweight="700")
    ax.text(0, -0.54, "24H MARKET BIAS", ha="center", va="center",
            fontsize=6.5, color="#484f58", fontweight="600")

    plt.tight_layout(pad=0.1)
    return fig


def fig_bb_expanded(a: dict) -> plt.Figure:
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax1, ax2, axp) = plt.subplots(3, 1, figsize=(13, 6.5), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1, 1.2], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2, axp): _ax_style(ax)
    # %B
    pctb = plot_df["BB_pctb"]
    ax1.plot(plot_df.index, pctb, color="#bc8cff", lw=1.2)
    ax1.axhline(1.0, color="#f85149", ls="--", lw=0.8, alpha=0.5)
    ax1.axhline(0.5, color="#484f58", ls=":",  lw=0.7, alpha=0.4)
    ax1.axhline(0.0, color="#3fb950", ls="--", lw=0.8, alpha=0.5)
    ax1.fill_between(plot_df.index, pctb, 1.0, where=(pctb > 1.0), color="#f85149", alpha=0.2)
    ax1.fill_between(plot_df.index, pctb, 0.0, where=(pctb < 0.0), color="#3fb950", alpha=0.2)
    ax1.set_ylabel("%B", color="#8b949e", fontsize=8)
    pctb_now = float(pctb.dropna().iloc[-1]) if not pctb.dropna().empty else 0.5
    ax1.text(0.995, 0.92, f"%B {pctb_now:.2f}", transform=ax1.transAxes, fontsize=8, va="top", ha="right", color="#bc8cff")
    ax1.set_title("Bollinger Bands  ·  %B position  ·  Bandwidth squeeze detector", color="#8b949e", fontsize=9, loc="left", pad=5)
    # Bandwidth
    bw = plot_df["BB_bw"]
    bw_mean = float(bw.dropna().mean()) if not bw.dropna().empty else 1
    ax2.plot(plot_df.index, bw, color="#58a6ff", lw=1.2, label="Bandwidth %")
    ax2.axhline(bw_mean, color="#484f58", ls="--", lw=0.8, alpha=0.5, label=f"Mean {bw_mean:.1f}%")
    ax2.fill_between(plot_df.index, bw, bw_mean * 0.6, where=(bw < bw_mean * 0.6), color="#ffd700", alpha=0.2, label="Squeeze zone")
    ax2.set_ylabel("Bandwidth %", color="#8b949e", fontsize=8)
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left", framealpha=0.85)
    # Price
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_stoch_expanded(a: dict) -> plt.Figure:
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax1, axp) = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, axp): _ax_style(ax)
    # Stochastic
    k = plot_df["Stoch_K"]; d = plot_df["Stoch_D"]
    ax1.plot(plot_df.index, k, color="#58a6ff", lw=1.3, label="%K (fast)")
    ax1.plot(plot_df.index, d, color="#f0883e", lw=1.0, ls="--", label="%D (signal)")
    ax1.axhline(80, color="#f85149", ls="--", lw=0.85, alpha=0.5)
    ax1.axhline(50, color="#484f58", ls=":",  lw=0.75, alpha=0.4)
    ax1.axhline(20, color="#3fb950", ls="--", lw=0.85, alpha=0.5)
    ax1.fill_between(plot_df.index, k, 80, where=(k > 80), color="#f85149", alpha=0.12)
    ax1.fill_between(plot_df.index, k, 20, where=(k < 20), color="#3fb950", alpha=0.12)
    k_arr = k.fillna(50).values; d_arr = d.fillna(50).values
    buy_sig  = (k_arr[1:] > d_arr[1:]) & (k_arr[:-1] <= d_arr[:-1]) & (k_arr[1:] < 30)
    sell_sig = (k_arr[1:] < d_arr[1:]) & (k_arr[:-1] >= d_arr[:-1]) & (k_arr[1:] > 70)
    x_idx = plot_df.index[1:]
    if buy_sig.any():
        ax1.scatter(x_idx[buy_sig],  k_arr[1:][buy_sig],  marker="^", color="#3fb950", s=40, zorder=5, label="Buy cross")
    if sell_sig.any():
        ax1.scatter(x_idx[sell_sig], k_arr[1:][sell_sig], marker="v", color="#f85149", s=40, zorder=5, label="Sell cross")
    k_now = float(k.dropna().iloc[-1]) if not k.dropna().empty else 50
    d_now = float(d.dropna().iloc[-1]) if not d.dropna().empty else 50
    state = "Overbought" if k_now > 80 else ("Oversold" if k_now < 20 else "Neutral")
    col   = "#f85149" if k_now > 80 else ("#3fb950" if k_now < 20 else "#8b949e")
    ax1.text(0.995, 0.95, f"Stoch %K {k_now:.0f}  %D {d_now:.0f}  ·  {state}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right", color=col)
    ax1.set_ylim(-5, 105)
    ax1.set_ylabel("Stochastic", color="#8b949e", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left", ncol=2, framealpha=0.85)
    ax1.set_title("Stochastic Oscillator (14,3)  ·  %K/%D crossovers", color="#8b949e", fontsize=9, loc="left", pad=5)
    # Price
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_ac_atr_expanded(a: dict) -> plt.Figure:
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax1, ax2, axp) = plt.subplots(3, 1, figsize=(13, 7.0), sharex=True,
                                         gridspec_kw={"height_ratios": [1.2, 1.2, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2, axp): _ax_style(ax)
    # Accelerator Oscillator
    ac = plot_df["AC"]; ao = plot_df["AO"]
    ac_col = np.where(ac >= 0, "#3fb950", "#f85149")
    ax1.bar(plot_df.index, ac.values, color=ac_col, alpha=0.7, width=0.6)
    ax1.plot(plot_df.index, ao, color="#58a6ff", lw=1.0, alpha=0.6, label="AO")
    ax1.axhline(0, color="#484f58", lw=0.8)
    ac_now = float(ac.dropna().iloc[-1]) if not ac.dropna().empty else 0
    ao_now = float(ao.dropna().iloc[-1]) if not ao.dropna().empty else 0
    accel_state = "Accelerating ↑" if ac_now > 0 else "Decelerating ↓"
    ax1.text(0.995, 0.95, f"AC {ac_now:+.0f}  ·  {accel_state}",
             transform=ax1.transAxes, fontsize=8, va="top", ha="right",
             color="#3fb950" if ac_now > 0 else "#f85149")
    ax1.set_title("Accelerator Oscillator (AC)", color="#8b949e", fontsize=9, loc="left", pad=4)
    ax1.set_ylabel("AC / AO", color="#8b949e", fontsize=8)
    ax1.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="lower left")
    # ATR
    atr = plot_df["ATR"]
    atr_mean = float(atr.dropna().mean()) if not atr.dropna().empty else 1
    ax2.plot(plot_df.index, atr, color="#f0883e", lw=1.3, label="ATR (14)")
    ax2.axhline(atr_mean, color="#484f58", ls="--", lw=0.8, alpha=0.5, label=f"Mean ${atr_mean:,.0f}")
    ax2.fill_between(plot_df.index, atr, atr_mean, where=(atr > atr_mean * 1.3), color="#f0883e", alpha=0.15, label="High vol zone")
    atr_now = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 0
    atr_pct = atr_now / price * 100
    vol_state = "High" if atr_now > atr_mean * 1.3 else ("Low" if atr_now < atr_mean * 0.7 else "Normal")
    ax2.text(0.995, 0.95, f"ATR ${atr_now:,.0f}  ({atr_pct:.1f}%)  ·  {vol_state}",
             transform=ax2.transAxes, fontsize=8, va="top", ha="right",
             color="#f85149" if vol_state == "High" else ("#3fb950" if vol_state == "Low" else "#8b949e"))
    ax2.set_title("Average True Range  ·  Volatility", color="#8b949e", fontsize=9, loc="left", pad=4)
    ax2.set_ylabel("ATR", color="#8b949e", fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", loc="upper left")
    # Price
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_fib_expanded(a: dict) -> plt.Figure:
    plot_df = a["plot_df"]; price = a["price"]; fib = a["fib"]
    fib_palette = {"0.0%": "#e6edf3", "23.6%": "#58a6ff", "38.2%": "#3fb950",
                   "50.0%": "#ffd700", "61.8%": "#f0883e", "78.6%": "#f85149", "100.0%": "#e6edf3"}
    fig, (ax1, axp) = plt.subplots(2, 1, figsize=(13, 7.0), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, axp): _ax_style(ax)
    # Fibonacci candlestick chart (unchanged from original)
    colors = np.where(plot_df.Close >= plot_df.Open, "#3fb950", "#f85149")
    ax1.bar(plot_df.index, plot_df.Close - plot_df.Open, 0.6, bottom=plot_df.Open, color=colors, alpha=0.8)
    ax1.bar(plot_df.index, plot_df.High  - plot_df.Low,  0.1, bottom=plot_df.Low,  color=colors, alpha=0.8)
    for label, lvl in fib["levels"].items():
        col = fib_palette.get(label, "#8b949e")
        ax1.axhline(lvl, color=col, lw=1.1, alpha=0.75, ls="--")
        ax1.annotate(f"  {label}  ${lvl:,.0f}", xy=(1.002, lvl),
                     xycoords=ax1.get_yaxis_transform(),
                     fontsize=7.5, color=col, va="center", ha="left", clip_on=False)
    ax1.axhline(price, color="#ffffff", lw=1.2, ls="-", alpha=0.6)
    # Current price label — nudge away from nearest fib level to avoid overlap
    _fib_prices = list(fib["levels"].values())
    _nearest_fib = min(_fib_prices, key=lambda v: abs(v - price))
    _price_y = price
    if abs(price - _nearest_fib) / price < 0.015:
        _price_y = price - price * 0.02 if _nearest_fib > price else price + price * 0.02
    ax1.annotate(f"  ▶ ${price:,.0f}",
                 xy=(1.002, _price_y), xycoords=ax1.get_yaxis_transform(),
                 fontsize=7.5, color="#ffffff", va="center", ha="left", clip_on=False,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#161b22", ec="#ffffff", alpha=0.9, lw=0.7))
    lvl_vals = sorted(fib["levels"].values())
    below = [v for v in lvl_vals if v < price]; above = [v for v in lvl_vals if v > price]
    if below and above:
        range_size = above[0] - below[-1]
        pct_in = (price - below[-1]) / range_size * 100 if range_size > 0 else 50
        ax1.text(0.005, 0.97,
                 f"Price ${price:,.0f}  ·  Between {[k for k,v in fib['levels'].items() if v==below[-1]][0]} and {[k for k,v in fib['levels'].items() if v==above[0]][0]}  ·  {pct_in:.0f}% through zone",
                 transform=ax1.transAxes, fontsize=8, va="top", ha="left", color="#ffd700",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#ffd700", alpha=0.85, lw=0.7))
    ax1.set_title(f"Fibonacci Retracement  ·  Swing High ${fib['high']:,.0f}  →  Swing Low ${fib['low']:,.0f}",
                  color="#8b949e", fontsize=9, loc="left", pad=5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    # Price
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.5)
    return fig


def fig_ichi_expanded(a: dict) -> plt.Figure:
    plot_df = a["plot_df"]; price = a["price"]
    fig, (ax1, axp) = plt.subplots(2, 1, figsize=(13, 8.0), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, axp): _ax_style(ax)
    ax1.plot(plot_df.index, plot_df.Close, color="#c9d1d9", lw=1.3, label="Price", zorder=5)
    tenkan = plot_df["Ichi_Tenkan"]; kijun = plot_df["Ichi_Kijun"]
    span_a = plot_df["Ichi_SpanA"];  span_b = plot_df["Ichi_SpanB"]
    chikou = plot_df["Ichi_Chikou"]
    ax1.plot(plot_df.index, tenkan, color="#f85149", lw=1.1, label="Tenkan (9)")
    ax1.plot(plot_df.index, kijun,  color="#58a6ff", lw=1.2, label="Kijun (26)")
    ax1.plot(plot_df.index, chikou, color="#ffd700", lw=0.9, ls="--", alpha=0.6, label="Chikou (lag)")
    ax1.plot(plot_df.index, span_a, color="#3fb950", lw=0.7, alpha=0.5)
    ax1.plot(plot_df.index, span_b, color="#f85149", lw=0.7, alpha=0.5)
    ax1.fill_between(plot_df.index, span_a, span_b, where=(span_a >= span_b), color="#3fb950", alpha=0.08, label="Cloud (bull)")
    ax1.fill_between(plot_df.index, span_a, span_b, where=(span_a <  span_b), color="#f85149", alpha=0.08, label="Cloud (bear)")
    try:
        t_now = float(tenkan.dropna().iloc[-1]); k_now = float(kijun.dropna().iloc[-1])
        sa_now = float(span_a.dropna().iloc[-1]); sb_now = float(span_b.dropna().iloc[-1])
        cloud_top = max(sa_now, sb_now); cloud_bot = min(sa_now, sb_now)
        if price > cloud_top:   signal, sig_col = "Price ABOVE cloud — Bullish", "#3fb950"
        elif price < cloud_bot: signal, sig_col = "Price BELOW cloud — Bearish", "#f85149"
        else:                   signal, sig_col = "Price INSIDE cloud — Indecision", "#8b949e"
        tk_cross = "TK Cross ↑" if t_now > k_now else "TK Cross ↓"
        ax1.text(0.005, 0.97, f"{signal}  ·  {tk_cross}",
                 transform=ax1.transAxes, fontsize=8, va="top", ha="left", color=sig_col,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec=sig_col, alpha=0.9, lw=0.7))
    except Exception:
        pass
    ax1.set_title("Ichimoku Cloud  ·  Tenkan / Kijun / Chikou / Kumo", color="#8b949e", fontsize=9, loc="left", pad=5)
    ax1.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", framealpha=0.9, ncol=3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    # Price
    _add_price_panel(axp, plot_df, price)
    plt.tight_layout(pad=0.8)
    return fig


# ── Full-screen dialogs for each expanded indicator ──────────────────────────

def _exp_btn(key: str, label: str) -> None:
    """Toggle expand/close button for an indicator section."""
    _open = st.session_state.get("_exp_open") == key
    _btn_label = "✕ Close" if _open else "⛶ Expand"
    if st.button(_btn_label, key=f"btn_exp_{key}", use_container_width=True):
        st.session_state["_exp_open"] = None if _open else key


def _exp_inline(key: str, render_fn, a: dict) -> None:
    """If this key is expanded, render the full chart inline."""
    if st.session_state.get("_exp_open") != key:
        return
    with st.spinner("Rendering full chart…"):
        _fig = render_fn(a)
    st.pyplot(_fig, use_container_width=True)
    plt.close(_fig)


# ════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BTC Analysis", page_icon="₿", layout="wide")

st.markdown("""
<style>
/* Base */
.stApp { background-color: #0d1117; }
h1,h2,h3,h4,p,label,span { color: #e6edf3 !important; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 2px; gap: 2px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8b949e !important; font-size: 13px; padding: 6px 16px; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #e6edf3 !important; }
/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] p { color: #8b949e !important; font-size: 11px !important; }
[data-testid="stMetricValue"]   { font-size: 18px !important; }
/* Signal pills */
.sig-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 0;
}
.sig-bull { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid #3fb950; }
.sig-bear { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
.sig-neut { background: rgba(139,148,158,0.12); color: #8b949e; border: 1px solid #30363d; }
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 13px;
}
div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ₿ BTC Analysis")
    st.caption("Live market dashboard")
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Real auto-refresh — triggers a full rerun every 5 minutes
    _refresh_count = st_autorefresh(interval=5 * 60 * 1000, key="autorefresh_main")
    if _refresh_count > 0:
        st.cache_data.clear()
    _SGT = _tz(offset=__import__("datetime").timedelta(hours=8))
    _last_refreshed = _dt.now(_SGT).strftime("%H:%M:%S SGT")
    st.caption(f"Auto-refreshes every 5 min · Liquidity every 2 min")
    st.caption(f"🕐 Last refreshed: {_last_refreshed}")
    st.divider()
    st.markdown("**Data Sources**")
    st.caption("📈 Price: Binance spot (yfinance fallback)")
    st.caption("😱 Fear & Greed: alternative.me")
    st.caption("🌍 Dominance: CoinGecko")
    st.caption("📊 ETF Flows: yfinance (IBIT/FBTC/ARKB…)")
    st.caption("🔥 Liquidity: Binance orderbook")
    st.divider()
    st.markdown("**About**")
    st.caption("Cycle phase scored across 8 signals: "
               "Fear & Greed, ETF Flows, RSI, Momentum, "
               "Dominance, MA200 deviation, 52W position, OBV.  "
               "Range: −24 (top) to +24 (bottom).")


# ── Fetch data ───────────────────────────────────────────────────
with st.spinner("Loading…"):
    try:
        a = run_analysis("BTC-USD")
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if not a:
    st.error("Could not fetch BTC data. Check your internet connection.")
    st.stop()

# run_analysis() is cached for 5 minutes — its embedded live-ticker call only
# fires on a cache miss. Fetch a fresh ticker on every render so the header
# BTC Price tracks Binance spot in real time, and recompute the 24h % change
# from the cached prior daily close.
try:
    _live_tick = _get_raw("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
    if isinstance(_live_tick, dict) and "price" in _live_tick:
        _lp = float(_live_tick["price"])
        if _lp > 0:
            a["price"] = _lp
            _cached_closes = a.get("closes") or []
            if len(_cached_closes) >= 2 and _cached_closes[-2]:
                a["chg_24h"] = (_lp - _cached_closes[-2]) / _cached_closes[-2] * 100
except Exception:
    pass

price      = a["price"]
chg_24h    = a["chg_24h"]
cycle      = a["cycle"]
crypto_sig = a["crypto_sig"]
w52        = a["w52"]
btc_liq    = a["btc_liq"]
prediction = a["prediction"]
bias_72h       = a["bias_72h"]
bias_24h       = a.get("bias_24h", {"score": 0, "label": "N/A", "color": "#8b949e"})
poly_sentiment = a.get("poly_sentiment", {})

# ── Top metrics bar (2 rows of 4 — mobile-friendly) ──────────────
_mr1c1, _mr1c2, _mr1c3, _mr1c4 = st.columns(4)
with _mr1c1:
    st.metric("BTC Price", f"${price:,.0f}", delta=f"{chg_24h:+.2f}%")
with _mr1c2:
    fgi = crypto_sig.get("fear_greed_value", "N/A")
    fg_raw = str(crypto_sig.get("fear_greed_label", "N/A")).split("(")[0].strip()
    st.metric("Fear & Greed", f"{fgi} — {fg_raw}" if isinstance(fgi, int) else "N/A")
with _mr1c3:
    st.metric("BTC Dominance", crypto_sig.get("btc_dominance", "N/A"))
with _mr1c4:
    m7  = crypto_sig.get("momentum_7d",  "N/A")
    m30 = crypto_sig.get("momentum_30d", "N/A")
    st.metric("7d / 30d Return", f"{m7}", delta=f"30d {m30}")

_mr2c1, _mr2c2, _mr2c3, _mr2c4 = st.columns(4)
with _mr2c1:
    pfl = w52.get("pct_from_low")
    pfh = w52.get("pct_from_high")
    st.metric("52W Range", f"+{pfl:.0f}% low" if pfl else "N/A",
              delta=f"{pfh:.0f}% high" if pfh else None)
with _mr2c2:
    adx_v = a.get("adx_val", float("nan"))
    trend_str = "Strong" if adx_v > 25 else ("Weak" if adx_v < 20 else "Moderate")
    st.metric("ADX (Trend)", f"{adx_v:.0f}" if not np.isnan(adx_v) else "N/A", delta=trend_str)
with _mr2c3:
    st.metric("Cycle Phase", f"{cycle['emoji']} {cycle['phase']}",
              delta=f"{cycle['total']:+d} / 24")
with _mr2c4:
    pred_score = prediction.get("total", 0)
    pred_label = prediction.get("label", "N/A")
    pred_conf  = prediction.get("confidence", 0)
    st.metric("Daily Trend", pred_label, delta=f"{pred_score:+d}/14  {pred_conf}% agree")

st.divider()

# ── 72-Hour Directional Bias Gauge ───────────────────────────────
_b12_score_raw = bias_72h["score"]
_b12_sigs   = bias_72h["signals"]
_b12_wts    = bias_72h["weights"]
_b12_regime = bias_72h.get("regime", "transition")
_b12_adx1h  = bias_72h.get("adx_1h")
_b12_atrpct = bias_72h.get("atr_percentile", 50)
_b12_bprob  = bias_72h.get("bull_prob", 50.0)
_b12_emove  = bias_72h.get("expected_move")
_b12_erange = bias_72h.get("expected_range")
_b12_conv   = bias_72h.get("conviction", 0.5)   # ⑫ 0=split signals, 1=full consensus
_b12_sigstd = bias_72h.get("signal_std", 0.5)

# ① EMA smoothing: α=0.50 = equal-weight blend of raw + previous.
# Responsive enough to track genuine 5%+ BTC moves within 1–2 refreshes,
# while still smoothing single-candle spikes.
# ⑮ Cold-start: initialize to 0 (neutral) not raw score, so the first
# reading after an app restart doesn't show an unsmoothed spike.
_EMA_ALPHA  = 0.50
_prev_ema   = st.session_state.get("_bias_ema", 0.0)
_b12_score  = round(_EMA_ALPHA * _b12_score_raw + (1 - _EMA_ALPHA) * _prev_ema, 1)
st.session_state["_bias_ema"] = _b12_score

# Derive label/color from smoothed score
def _score_to_label_color(s):
    if s >= 70:  return "STRONG BULL", "#3fb950"
    if s >= 40:  return "BULL",        "#3fb950"
    if s >= 25:  return "MILD BULL",   "#58a6ff"
    if s > -25:  return "NEUTRAL",     "#8b949e"
    if s > -40:  return "MILD BEAR",   "#f0883e"
    if s > -70:  return "BEAR",        "#f85149"
    return "STRONG BEAR", "#f85149"

_b12_label, _b12_color = _score_to_label_color(_b12_score)
_b12_pct    = (_b12_score + 100) / 200 * 100      # 0–100 for CSS positioning
_b12_sign   = "+" if _b12_score >= 0 else ""

# ── Regime pill + probabilistic outputs ──────────────────────────
_regime_color = {"trend": "#ffd700", "range": "#bc8cff", "transition": "#8b949e"}[_b12_regime]
_regime_label = {"trend": "TRENDING", "range": "RANGING", "transition": "TRANSITION"}[_b12_regime]
_adx_str = f" · ADX {_b12_adx1h:.0f}" if _b12_adx1h is not None else ""
_vol_str  = ("· vol HIGH" if _b12_atrpct > 70 else "· vol LOW" if _b12_atrpct < 30 else f"· vol {_b12_atrpct:.0f}th pct")
_bprob_color = "#3fb950" if _b12_bprob >= 60 else "#f85149" if _b12_bprob <= 40 else "#8b949e"
_emove_str = (f"Expected move: <b>${abs(_b12_emove):,.0f}</b> {'↑' if _b12_emove and _b12_emove > 0 else '↓'}"
              f" over 72h · range ±${_b12_erange:,.0f}"
              if _b12_emove is not None and _b12_erange is not None else "")
# ⑫ Conviction pill: color-coded consensus quality
_conv_pct   = int(_b12_conv * 100)
_conv_color = "#3fb950" if _b12_conv >= 0.65 else "#ffd700" if _b12_conv >= 0.40 else "#f85149"
_conv_label = "HIGH" if _b12_conv >= 0.65 else ("MED" if _b12_conv >= 0.40 else "LOW")
st.markdown(f"""
<div style="display:flex; gap:10px; align-items:center; margin-bottom:10px; flex-wrap:wrap;">
  <span style="background:{_regime_color}22; border:1px solid {_regime_color}66;
        color:{_regime_color}; border-radius:6px; padding:3px 10px; font-size:11px; font-weight:700;
        letter-spacing:.08em;">{_regime_label}{_adx_str} {_vol_str}</span>
  <span style="background:#161b22; border:1px solid #30363d; border-radius:6px;
        padding:3px 10px; font-size:11px; color:{_bprob_color}; font-weight:700;">
    Bull prob: {_b12_bprob:.0f}%</span>
  <span style="background:{_conv_color}18; border:1px solid {_conv_color}55; border-radius:6px;
        padding:3px 10px; font-size:11px; color:{_conv_color}; font-weight:700;"
        title="Conviction = 1 − signal std. High = all signals agree. Low = half bullish, half bearish averaging to noise.">
    Conviction: {_conv_label} ({_conv_pct}%)</span>
  {"<span style='font-size:11px; color:#6e7681;'>" + _emove_str + "</span>" if _emove_str else ""}
</div>""", unsafe_allow_html=True)

# Gradient track: red → grey → green, pointer dot at the right position
_gauge_html = f"""
<div style="margin: 4px 0 18px 0;">
  <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
    <span style="font-size:13px; font-weight:600; color:#8b949e; letter-spacing:.05em;">
      72H DIRECTIONAL BIAS
    </span>
    <span style="font-size:22px; font-weight:800; color:{_b12_color}; letter-spacing:-.5px;">
      {_b12_sign}{_b12_score:.0f}%
      &nbsp;<span style="font-size:14px; font-weight:600;">{_b12_label}</span>
    </span>
  </div>

  <!-- Track -->
  <div style="position:relative; height:16px; border-radius:8px;
       background: linear-gradient(to right, #f85149 0%, #f0883e 20%, #8b949e 45%, #8b949e 55%, #58a6ff 80%, #3fb950 100%);
       box-shadow: inset 0 1px 3px rgba(0,0,0,.4);">
    <!-- Filled overlay that fades from centre toward the active side -->
    <div style="position:absolute; top:0; left:0; right:0; bottom:0; border-radius:8px;
         background: linear-gradient(to right,
           rgba(13,17,23,.7) 0%,
           rgba(13,17,23,.0) {_b12_pct:.1f}%,
           rgba(13,17,23,.0) {_b12_pct:.1f}%,
           rgba(13,17,23,.7) 100%);"></div>
    <!-- Pointer -->
    <div style="position:absolute; top:50%; left:{_b12_pct:.1f}%;
         transform:translate(-50%,-50%);
         width:20px; height:20px; border-radius:50%;
         background:#0d1117; border:3px solid {_b12_color};
         box-shadow:0 0 8px {_b12_color}88;"></div>
  </div>

  <!-- Scale labels with ±40 action-zone markers -->
  <div style="position:relative; height:6px; margin-top:2px;">
    <div style="position:absolute; left:30%; width:2px; height:6px; background:#3fb95055;"></div>
    <div style="position:absolute; left:70%; width:2px; height:6px; background:#f8514955;"></div>
  </div>
  <div style="display:flex; justify-content:space-between;
       font-size:10px; color:#484f58; margin-top:1px; font-family:monospace;">
    <span>−100</span>
    <span style="color:#f8514988;">−40 ← trade zone</span>
    <span style="color:#555;">0</span>
    <span style="color:#3fb95088;">+40 trade zone →</span>
    <span>+100</span>
  </div>
</div>
"""
st.markdown(_gauge_html, unsafe_allow_html=True)

# ── ETF Action Recommendation ─────────────────────────────────
# Backtest shows ±25–39 is near-random (44–55%). Only call trades at ±40+
# where the technical component shows demonstrated edge (56–62%).
# Daily trend is a risk modifier: with-trend = clean, counter-trend = caution.
_pred_total = prediction.get("total", 0)
_pred_bull  = _pred_total > 0
_pred_bear  = _pred_total < 0
_CALL_THRESHOLD = 40   # minimum score to generate a trade call

if _b12_score >= _CALL_THRESHOLD:
    _etf_color = "#3fb950"
    _etf_icon  = "▲"
    if _pred_bull:
        _etf_action = "BUY LONG BTC ETF"
        _etf_note   = f"Score +{_b12_score:.0f} · with daily trend · clean setup (e.g. BITO, IBIT, FBTC)"
    else:
        _etf_action = "BUY LONG — COUNTER-TREND"
        _etf_note   = f"Score +{_b12_score:.0f} · daily trend bearish · valid call but size down, higher risk"

elif _b12_score <= -_CALL_THRESHOLD:
    _etf_color = "#f85149"
    _etf_icon  = "▼"
    if _pred_bear:
        _etf_action = "BUY INVERSE / SHORT ETF"
        _etf_note   = f"Score {_b12_score:.0f} · with daily trend · clean setup (e.g. BITI, SBIT)"
    else:
        _etf_action = "BUY INVERSE — COUNTER-TREND"
        _etf_note   = f"Score {_b12_score:.0f} · daily trend bullish · valid call but size down, higher risk"

elif abs(_b12_score) >= 25:
    # Weak signal tier — backtest shows ~50% here, not worth trading
    _etf_color  = "#ffd700"
    _etf_icon   = "⏸"
    _etf_action = "WEAK SIGNAL — WAIT"
    _dir        = "bullish" if _b12_score > 0 else "bearish"
    _etf_note   = (f"Score {_b12_sign}{_b12_score:.0f} · {_dir} lean but below ±40 action threshold · "
                   f"backtest shows ~50% accuracy here — not enough edge to trade")
else:
    _etf_color  = "#8b949e"
    _etf_icon   = "⏸"
    _etf_action = "HOLD / STAY FLAT"
    _etf_note   = f"Score {_b12_sign}{_b12_score:.0f} · no meaningful directional conviction"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:14px;
     background:{_etf_color}14; border:1px solid {_etf_color}55;
     border-radius:8px; padding:10px 16px; margin-bottom:14px;">
  <span style="font-size:26px; color:{_etf_color};">{_etf_icon}</span>
  <div>
    <div style="font-size:16px; font-weight:700; color:{_etf_color}; letter-spacing:.03em;">
      {_etf_action}
    </div>
    <div style="font-size:11px; color:#8b949e; margin-top:2px;">{_etf_note}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 24h Market Bias Speedometer ──────────────────────────────────
_c24_gauge, _c24_spacer = st.columns([1, 2])
with _c24_gauge:
    _f24 = fig_speedometer_24h(bias_24h["score"], bias_24h["label"], bias_24h["color"])
    st.pyplot(_f24, use_container_width=True)
    plt.close(_f24)
with _c24_spacer:
    st.markdown("")
    st.markdown(
        "**24h Market Bias** — reactive reading of current sentiment using 1h/4h indicators "
        "(RSI, Stochastic, MACD, momentum, EMA align, order book depth, funding rate, CVD, Fear & Greed). "
        "Volatile by design: moves with the market in real-time.\n\n"
        "**72h Directional Bias** above is the structural forecast — EMA-smoothed, "
        "driven by 4h/daily signals. Use both together: 72h tells you the tide, "
        "24h tells you the current wave."
    )

st.divider()

# ⑯ 4h data staleness indicator — EMA Structure is 18% of score but cached 15 min
_4h_fetch_ts   = st.session_state.get("_4h_fetch_ts", 0)
_4h_age_min    = int((_dt.now(_tz.utc).timestamp() - _4h_fetch_ts) / 60) if _4h_fetch_ts else 0
_4h_stale      = _4h_age_min >= 10   # flag if 4h data is ≥10 min old
_4h_stale_note = (f" · ⚠ 4h data {_4h_age_min}min old — EMA Structure may lag"
                  if _4h_stale else f" · 4h data {_4h_age_min}min ago")

# Signal breakdown inside expander — 2 rows of 6 (dynamic)
with st.expander(f"📊 72h Bias — Signal Breakdown ({len(_b12_sigs)} signals){_4h_stale_note}", expanded=False):
    _sig_items = list(_b12_sigs.items())
    # Row 1: first 6 signals
    _row1_cols = st.columns(6)
    for _ci, (_sn, (_sv, _se)) in enumerate(_sig_items[:6]):
        _wt      = _b12_wts[_sn]
        _contrib = _sv * _wt * 100
        _pcol    = "#3fb950" if _sv > 0.05 else ("#f85149" if _sv < -0.05 else "#8b949e")
        _sign    = "+" if _contrib >= 0 else ""
        with _row1_cols[_ci]:
            st.markdown(f"""
<div class="info-box" style="text-align:center;">
  <div style="font-size:10px; color:#8b949e; margin-bottom:4px;">{_sn}</div>
  <div style="font-size:16px; font-weight:700; color:{_pcol};">{_sign}{_contrib:.1f}%</div>
  <div style="font-size:9px; color:#484f58; margin-top:2px;">wt {int(_wt*100)}%</div>
  <div style="font-size:9px; color:#8b949e; margin-top:5px; line-height:1.3;">{_se}</div>
</div>
""", unsafe_allow_html=True)
    # Row 2: remaining signals (up to 6)
    _row2_items = _sig_items[6:]
    _row2_cols = st.columns(len(_row2_items))
    for _ci, (_sn, (_sv, _se)) in enumerate(_row2_items):
        _wt      = _b12_wts[_sn]
        _contrib = _sv * _wt * 100
        _pcol    = "#3fb950" if _sv > 0.05 else ("#f85149" if _sv < -0.05 else "#8b949e")
        _sign    = "+" if _contrib >= 0 else ""
        with _row2_cols[_ci]:
            st.markdown(f"""
<div class="info-box" style="text-align:center;">
  <div style="font-size:10px; color:#8b949e; margin-bottom:4px;">{_sn}</div>
  <div style="font-size:16px; font-weight:700; color:{_pcol};">{_sign}{_contrib:.1f}%</div>
  <div style="font-size:9px; color:#484f58; margin-top:2px;">wt {int(_wt*100)}%</div>
  <div style="font-size:9px; color:#8b949e; margin-top:5px; line-height:1.3;">{_se}</div>
</div>
""", unsafe_allow_html=True)

# Polymarket 72h thesis scoring panel
_pm_mkts      = poly_sentiment.get("markets", [])
_pm_n         = poly_sentiment.get("markets_used", 0)
_pm_thesis    = poly_sentiment.get("thesis_score", 0.0)
_pm_thlabel   = poly_sentiment.get("thesis_label", "NEUTRAL")
_pm_conf      = poly_sentiment.get("confidence", 0.0)
_pm_detail    = poly_sentiment.get("detail", "N/A")
_pm_color     = ("#3fb950" if _pm_thesis >= 1.0 else
                 "#f85149" if _pm_thesis <= -1.0 else "#8b949e")

def _pm_q_safe(q):
    """Escape $ so Streamlit's markdown parser doesn't treat them as LaTeX delimiters."""
    return q.replace("$", "&#36;")

def _pm_card_updown(mk):
    """Horizontal split probability bar for up/down directional markets."""
    od   = mk.get("outcomes_display", [])
    sc   = mk["individual_score"]
    wsc  = mk["weighted_score"]
    exp  = f"{mk['hours_left']:.0f}h" if mk.get("hours_left") is not None else "—"
    liq  = f"&#36;{mk['liquidity']/1000:.0f}k" if mk.get("liquidity") else ""
    sc_c = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#8b949e")
    wc   = "#3fb950" if wsc > 0 else ("#f85149" if wsc < 0 else "#8b949e")
    bdr  = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#30363d")
    ref  = mk.get("reference_price")
    ref_html = (f' · <span style="color:#e3b341; font-weight:600;">Price to beat: '
                f'&#36;{ref:,.0f}</span>' if ref else "")

    if len(od) >= 2:
        u_lbl, u_p, _ = od[0]
        d_lbl, d_p, _ = od[1]
        u_pct = round(u_p * 100)
        d_pct = 100 - u_pct
        # Ensure labels fit — shorten if over 50% bar
        u_txt = f"{u_pct}%" if u_pct >= 20 else ""
        d_txt = f"{d_pct}%" if d_pct >= 20 else ""
        bar = (
            f'<div style="display:flex; border-radius:5px; overflow:hidden; height:22px; margin:6px 0;">'
            f'<div style="width:{u_pct}%; background:#3fb950; display:flex; align-items:center;'
            f' justify-content:center; font-size:11px; font-weight:700; color:#fff;">{u_txt}</div>'
            f'<div style="width:{d_pct}%; background:#f85149; display:flex; align-items:center;'
            f' justify-content:center; font-size:11px; font-weight:700; color:#fff;">{d_txt}</div>'
            f'</div>'
            f'<div style="display:flex; justify-content:space-between; font-size:10px; color:#8b949e; margin-bottom:4px;">'
            f'<span style="color:#3fb950;">↑ {u_lbl} {u_pct}%</span>'
            f'<span style="color:#f85149;">{d_pct}% {d_lbl} ↓</span>'
            f'</div>'
        )
    else:
        bar = ""

    return (
        f'<div style="background:#161b22; border:1px solid #30363d; border-left:3px solid {bdr};'
        f' border-radius:8px; padding:10px 14px; margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:2px;">'
        f'<div style="font-size:12px; color:#e6edf3; font-weight:600; line-height:1.4; max-width:72%;">{_pm_q_safe(mk["question"])}</div>'
        f'<div style="text-align:right; white-space:nowrap;">'
        f'<span style="font-size:15px; font-weight:800; color:{sc_c};">{sc:+.1f}</span>'
        f'<span style="font-size:10px; color:#484f58;"> ×{mk["weight"]}x = </span>'
        f'<span style="font-size:13px; font-weight:700; color:{wc};">{wsc:+.1f}</span>'
        f'</div></div>'
        f'{bar}'
        f'<div style="font-size:9px; color:#484f58;">{exp} · {liq} · {_pm_q_safe(mk["rationale"])}{ref_html}</div>'
        f'</div>'
    )

def _pm_card_yesno(mk):
    """Confidence progress bar for yes/no price-threshold markets."""
    od   = mk.get("outcomes_display", [])
    sc   = mk["individual_score"]
    wsc  = mk["weighted_score"]
    exp  = f"{mk['hours_left']:.0f}h" if mk.get("hours_left") is not None else "—"
    liq  = f"&#36;{mk['liquidity']/1000:.0f}k" if mk.get("liquidity") else ""
    sc_c = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#8b949e")
    wc   = "#3fb950" if wsc > 0 else ("#f85149" if wsc < 0 else "#8b949e")
    bdr  = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#30363d")

    if len(od) >= 2:
        y_lbl, y_p, y_c = od[0]
        n_lbl, n_p, n_c = od[1]
        y_pct = round(y_p * 100)
        bar = (
            f'<div style="margin:6px 0 3px;">'
            f'<div style="display:flex; justify-content:space-between; font-size:10px; margin-bottom:3px;">'
            f'<span style="color:{y_c}; font-weight:600;">{_pm_q_safe(y_lbl)}</span>'
            f'<span style="color:{y_c}; font-weight:800; font-size:13px;">{y_pct}%</span>'
            f'</div>'
            f'<div style="background:#21262d; border-radius:4px; height:10px; overflow:hidden;">'
            f'<div style="width:{y_pct}%; height:100%; background:{y_c};"></div>'
            f'</div>'
            f'<div style="font-size:10px; color:#484f58; margin-top:3px;">'
            f'{_pm_q_safe(n_lbl)}: {n_p:.0%}'
            f'</div>'
            f'</div>'
        )
    else:
        bar = f'<div style="font-size:11px; color:#8b949e; margin:4px 0;">{_pm_q_safe(mk.get("market_data",""))}</div>'

    return (
        f'<div style="background:#161b22; border:1px solid #30363d; border-left:3px solid {bdr};'
        f' border-radius:8px; padding:10px 14px; margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start;">'
        f'<div style="font-size:12px; color:#e6edf3; font-weight:600; line-height:1.4; max-width:72%;">{_pm_q_safe(mk["question"])}</div>'
        f'<div style="text-align:right; white-space:nowrap;">'
        f'<span style="font-size:15px; font-weight:800; color:{sc_c};">{sc:+.1f}</span>'
        f'<span style="font-size:10px; color:#484f58;"> ×{mk["weight"]}x = </span>'
        f'<span style="font-size:13px; font-weight:700; color:{wc};">{wsc:+.1f}</span>'
        f'</div></div>'
        f'{bar}'
        f'<div style="font-size:9px; color:#484f58;">{exp} · {liq} · {_pm_q_safe(mk["rationale"])}</div>'
        f'</div>'
    )

def _pm_card_range(mk):
    """Proportional bucket bars for price-range and multi-target markets."""
    od      = mk.get("outcomes_display", [])
    sc      = mk["individual_score"]
    wsc     = mk["weighted_score"]
    wt      = mk.get("weight", 1)
    mdata   = mk.get("market_data", "")
    exp     = f"{mk['hours_left']:.0f}h" if mk.get("hours_left") is not None else "—"
    liq     = f"&#36;{mk['liquidity']/1000:.0f}k" if mk.get("liquidity") else ""
    ctx_only = (wt == 0)   # "When will" markets — display only, no score

    if ctx_only:
        sc_c = "#6e7681"; wc = "#6e7681"; bdr = "#30363d"
    else:
        sc_c = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#8b949e")
        wc   = "#3fb950" if wsc > 0 else ("#f85149" if wsc < 0 else "#8b949e")
        bdr  = "#3fb950" if sc > 0.5 else ("#f85149" if sc < -0.5 else "#30363d")

    # Build bucket bars (show up to 6).
    # is_bull=True → green (directional bull target above current price)
    # is_bull=False → red (bear target)
    # is_bull=None → grey floor (already-exceeded support, de-emphasised)
    visible = od[:8]
    # Scale bars to the highest probability across ALL shown options so widths
    # are directly comparable — a 93% floor bar and an 8% ceiling bar both reflect
    # their true probability relative to the same scale.
    max_p = max((p for _, p, _ in visible), default=1) or 1
    rows  = ""

    def _grad(t, lo, hi):
        # interpolate RGB tuple lo→hi by factor t ∈ [0,1]
        return "#{:02x}{:02x}{:02x}".format(
            int(lo[0] + t * (hi[0] - lo[0])),
            int(lo[1] + t * (hi[1] - lo[1])),
            int(lo[2] + t * (hi[2] - lo[2])),
        )

    for bl, bp, is_bull in visible:
        t = min(1.0, bp / 0.85)   # 85%+ → max brightness
        lbl_col = "#e6edf3"        # label text always white
        pct_col = "#8b949e"        # percentage number always grey
        if is_bull is None:
            bar_col = "#484f58"    # floor — grey bar
        elif is_bull is True:
            bar_col = _grad(t, (13, 68, 41), (86, 211, 100))   # dark→vivid green
        else:
            bar_col = _grad(t, (67, 17, 26), (248, 81, 73))    # dark→vivid red
        bar_pct = round(min(bp, max_p) / max_p * 100)
        rows += (
            f'<div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">'
            f'<div style="font-size:10px; color:{lbl_col}; width:90px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{_pm_q_safe(bl)}</div>'
            f'<div style="flex:1; background:#21262d; border-radius:3px; height:9px;">'
            f'<div style="width:{bar_pct}%; height:100%; background:{bar_col}; border-radius:3px;"></div>'
            f'</div>'
            f'<div style="font-size:10px; color:{pct_col}; width:32px; text-align:right;">{bp:.0%}</div>'
            f'</div>'
        )

    _score_html = (
        f'<span style="font-size:10px; color:#484f58; font-style:italic;">context only</span>'
        if ctx_only else
        f'<span style="font-size:15px; font-weight:800; color:{sc_c};">{sc:+.1f}</span>'
        f'<span style="font-size:10px; color:#484f58;"> ×{wt}x = </span>'
        f'<span style="font-size:13px; font-weight:700; color:{wc};">{wsc:+.1f}</span>'
    )
    _q_col = "#6e7681" if ctx_only else "#e6edf3"
    return (
        f'<div style="background:#161b22; border:1px solid #30363d; border-left:3px solid {bdr};'
        f' border-radius:8px; padding:10px 14px; margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px;">'
        f'<div style="font-size:12px; color:{_q_col}; font-weight:600; line-height:1.4; max-width:72%;">{_pm_q_safe(mk["question"])}</div>'
        f'<div style="text-align:right; white-space:nowrap;">{_score_html}</div>'
        f'</div>'
        f'{rows}'
        f'<div style="display:flex; justify-content:space-between; margin-top:4px;">'
        f'<div style="font-size:9px; color:#8b949e;">{_pm_q_safe(mdata)}</div>'
        f'<div style="font-size:9px; color:#484f58;">{exp} · {liq} · {_pm_q_safe(mk["rationale"])}</div>'
        f'</div>'
        f'</div>'
    )

# ── Claude AI Interpretation ─────────────────────────────────────
_ai_interp_key = ""
try:
    _ai_interp_key = st.secrets.get("ANTHROPIC_API_KEY", "") or ""
except Exception:
    pass
if not _ai_interp_key:
    import os as _os2
    _ai_interp_key = _os2.environ.get("ANTHROPIC_API_KEY", "")

if _ai_interp_key:
    _ai_col, _ai_btn_col = st.columns([5, 1])
    with _ai_btn_col:
        _ask_claude = st.button("🤖 Ask Claude", use_container_width=True)
    # ── Compute 15m chart signals for Claude ─────────────────────────
    _sdf15 = a.get("short_df")
    _ema_cross_15m = _vwap_bias_15m = _poc_vs_price_15m = "N/A"
    _atr_15m = _atr_pct_15m = "N/A"
    try:
        _df15c = _sdf15.iloc[-96:].copy() if _sdf15 is not None and len(_sdf15) >= 10 else None
        if _df15c is not None:
            _c15 = _df15c["Close"].values
            _h15 = _df15c["High"].values
            _l15 = _df15c["Low"].values
            _v15 = _df15c["Volume"].values
            _e8  = float(_df15c["Close"].ewm(span=8,  adjust=False).mean().iloc[-1])
            _e21 = float(_df15c["Close"].ewm(span=21, adjust=False).mean().iloc[-1])
            _ema_cross_15m = (f"EMA8 ${_e8:,.0f} > EMA21 ${_e21:,.0f} — BULLISH crossover"
                              if _e8 > _e21 else
                              f"EMA8 ${_e8:,.0f} < EMA21 ${_e21:,.0f} — BEARISH crossover")
            # VWAP
            _tp  = (_h15 + _l15 + _c15) / 3
            _vw  = float(np.cumsum(_tp * _v15)[-1] / (np.cumsum(_v15)[-1] or 1))
            _cur = float(_c15[-1])
            _vwap_bias_15m = (f"Price ${_cur:,.0f} ABOVE VWAP ${_vw:,.0f} — bullish intraday"
                              if _cur > _vw else
                              f"Price ${_cur:,.0f} BELOW VWAP ${_vw:,.0f} — bearish intraday")
            # POC
            _p_lo = _l15.min(); _p_hi = _h15.max()
            _N_B  = 40
            _be   = np.linspace(_p_lo, _p_hi, _N_B + 1)
            _bsz  = _be[1] - _be[0]
            _vp   = np.zeros(_N_B)
            for _ii in range(len(_c15)):
                _sp = max(_h15[_ii] - _l15[_ii], _bsz * 0.01)
                for _bb in range(_N_B):
                    _ov = max(0.0, min(_be[_bb+1], _h15[_ii]) - max(_be[_bb], _l15[_ii]))
                    _vp[_bb] += _v15[_ii] * _ov / _sp
            _poc_p = float((_be[int(np.argmax(_vp))] + _be[int(np.argmax(_vp))+1]) / 2)
            _poc_vs_price_15m = (f"POC ${_poc_p:,.0f} — price ${_cur:,.0f} trading ABOVE POC (bullish structure)"
                                 if _cur > _poc_p else
                                 f"POC ${_poc_p:,.0f} — price ${_cur:,.0f} trading BELOW POC (bearish structure)")
            # ATR
            _pc   = np.concatenate([[_c15[0]], _c15[:-1]])
            _tr   = np.maximum(_h15 - _l15, np.maximum(np.abs(_h15 - _pc), np.abs(_l15 - _pc)))
            _atr  = float(pd.Series(_tr).ewm(span=14, adjust=False).mean().iloc[-1])
            _atr_15m     = f"${_atr:,.0f}"
            _atr_pct_15m = f"{_atr / _cur * 100:.2f}"
    except Exception:
        pass

    if _ask_claude or st.session_state.get("_claude_interp"):
        if _ask_claude:
            with st.spinner("Claude is reading the dashboard…"):
                _interp_text = _call_claude_interpretation(
                    price        = price,
                    bias_score   = _b12_score,
                    bias_label   = _b12_label,
                    bias_signals = _b12_sigs,
                    pm_thesis    = _pm_thesis,
                    pm_label     = _pm_thlabel,
                    pm_markets   = _pm_mkts,
                    pred_label   = prediction.get("label", "N/A"),
                    pred_score   = prediction.get("total", 0),
                    fear_greed   = str(crypto_sig.get("fear_greed_label", "N/A")),
                    api_key      = _ai_interp_key,
                    # Additional indicators
                    btc_dominance  = str(crypto_sig.get("btc_dominance", "N/A")),
                    momentum_7d    = str(crypto_sig.get("momentum_7d",  "N/A")),
                    momentum_30d   = str(crypto_sig.get("momentum_30d", "N/A")),
                    pct_from_high  = w52.get("pct_from_high"),
                    pct_from_low   = w52.get("pct_from_low"),
                    cycle_phase    = cycle.get("phase", "N/A"),
                    cycle_score    = cycle.get("total", 0),
                    adx_val        = float(a.get("adx_val", float("nan"))),
                    signal_weights = _b12_wts,
                    # 24h engine
                    bias_24h_score   = float(bias_24h.get("score", 0.0)),
                    bias_24h_label   = str(bias_24h.get("label", "N/A")),
                    bias_24h_regime  = str(bias_24h.get("regime", "N/A")),
                    bias_24h_signals = bias_24h.get("signals"),
                    bias_24h_weights = bias_24h.get("weights"),
                    # 15-min chart signals
                    ema_cross_15m     = _ema_cross_15m,
                    vwap_bias_15m     = _vwap_bias_15m,
                    poc_vs_price_15m  = _poc_vs_price_15m,
                    atr_15m           = _atr_15m,
                    atr_pct_15m       = _atr_pct_15m,
                    # Liquidation map source quality
                    liq_map_source    = str((btc_liq or {}).get("liq_map_source", "N/A")),
                )
            st.session_state["_claude_interp"] = _interp_text
        _interp_text = st.session_state.get("_claude_interp", "")
        if _interp_text:
            _interp_color = ("#3fb950" if "LONG" in _interp_text.upper()[:60]
                             else "#f85149" if "SHORT" in _interp_text.upper()[:60]
                             else "#8b949e")
            # Render as native markdown so bold/bullets display correctly.
            # Escape $ signs first — Streamlit treats $...$ as LaTeX math,
            # which swallows price levels like $80,500 ... $82,000.
            _safe_interp = _interp_text.replace("$", r"\$")
            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d;'
                f' border-left:4px solid {_interp_color}; border-radius:10px;'
                f' padding:4px 14px 10px 14px; margin-bottom:12px;">'
                f'<p style="font-size:11px; color:#8b949e; margin-bottom:0;">'
                f'🤖 CLAUDE INTERPRETATION'
                f'<span style="float:right; font-size:10px; color:#484f58;">claude-haiku · ~$0.005/call</span>'
                f'</p></div>',
                unsafe_allow_html=True
            )
            st.markdown(_safe_interp)

with st.expander(
    f"🎯 Polymarket 24h Thesis — {_pm_n} markets · {_pm_thlabel} ({_pm_thesis:+.2f}/10)",
    expanded=False
):
    if not _pm_mkts:
        st.caption(_pm_detail)
    else:
        _pm_total_w = sum(m["weight"] for m in _pm_mkts)
        _pm_wsum    = sum(m["weighted_score"] for m in _pm_mkts)

        # ── Thesis score summary box ───────────────────────────────────────
        st.markdown(f"""
<div style="display:flex; align-items:center; gap:20px;
     background:{_pm_color}12; border:1px solid {_pm_color}44;
     border-radius:8px; padding:10px 16px; margin-bottom:14px;">
  <div style="text-align:center; min-width:90px;">
    <div style="font-size:28px; font-weight:800; color:{_pm_color}; letter-spacing:-.5px;">{_pm_thesis:+.2f}</div>
    <div style="font-size:10px; color:#8b949e; margin-top:1px;">/ 10</div>
  </div>
  <div>
    <div style="font-size:15px; font-weight:700; color:{_pm_color};">{_pm_thlabel}</div>
    <div style="font-size:11px; color:#8b949e; margin-top:3px;">
      {_pm_wsum:.1f} weighted sum &#247; {_pm_total_w:.0f} total weight &nbsp;&middot;&nbsp;
      {_pm_conf:.0%} agreement &nbsp;&middot;&nbsp; {_pm_n} markets
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Chronological order: nearest expiry first, None/long-term last ──
        _sorted_mkts = sorted(
            _pm_mkts,
            key=lambda m: m.get("hours_left") if m.get("hours_left") is not None else float("inf")
        )

        # Inject 24h open price into updown cards so the card can show "price to beat"
        _sdf_ref = a.get("short_df")
        _day_open = None
        try:
            if _sdf_ref is not None and len(_sdf_ref) > 0:
                _today = pd.Timestamp.utcnow().normalize()
                _today_rows = _sdf_ref[_sdf_ref.index.normalize() == _today]
                if not _today_rows.empty:
                    _day_open = float(_today_rows.iloc[0]["Open"])
                else:
                    _day_open = float(_sdf_ref.iloc[-96]["Open"]) if len(_sdf_ref) >= 96 else float(_sdf_ref.iloc[0]["Open"])
        except Exception:
            _day_open = None

        _card_fn = {
            "updown": _pm_card_updown,
            "yesno":  _pm_card_yesno,
            "range":  _pm_card_range,
        }
        for _mk in _sorted_mkts:
            _hl = _mk.get("hours_left")
            if (_mk.get("display_type") == "updown" and _day_open is not None
                    and _hl is not None and _hl <= 24):
                _mk = {**_mk, "reference_price": _day_open}
            _fn = _card_fn.get(_mk.get("display_type"), _pm_card_range)
            st.markdown(_fn(_mk), unsafe_allow_html=True)

        # ── Final calculation footer ───────────────────────────────────────
        st.markdown(
            f"<div style='font-size:11px; color:#8b949e; text-align:right; padding-top:6px;'>"
            f"Weighted sum: <b style='color:#e6edf3;'>{_pm_wsum:.1f}</b>"
            f" &nbsp;&#247;&nbsp; "
            f"Total weight: <b style='color:#e6edf3;'>{_pm_total_w:.0f}</b>"
            f" &nbsp;=&nbsp; "
            f"<b style='color:{_pm_color}; font-size:13px;'>{_pm_thesis:+.2f} / 10</b>"
            f"</div>",
            unsafe_allow_html=True)

# ── Signal outcome logging + stability + accuracy ─────────────────

# Resolve any outstanding signals and log current reading (max every 5 min)
_sig_rows = _resolve_signal_outcomes(price)
_now_ts   = _dt.now(_tz.utc).timestamp()
_last_log = st.session_state.get("_last_log_ts", 0)
if _now_ts - _last_log >= _LOG_INTERVAL:
    # Log the raw (unsmoothed) score + label so values match what the
    # headless cron writes — calibration must be environment-independent.
    # Also pass bias_72h + poly_sentiment so per-signal and per-market
    # detail rows are written for IC weighting / SHAP / velocity later.
    _log_bias_signal(_b12_score_raw, bias_72h.get("label", _b12_label), price,
                     bias_72h=bias_72h, poly=poly_sentiment)
    st.session_state["_last_log_ts"] = _now_ts

# Signal stability — consecutive same-direction refreshes
_cur_dir  = "LONG" if _b12_score >= 25 else ("SHORT" if _b12_score <= -25 else "HOLD")
_prev_dir = st.session_state.get("_prev_dir", _cur_dir)
_streak   = st.session_state.get("_dir_streak", 1)
_streak   = _streak + 1 if _cur_dir == _prev_dir else 1
st.session_state["_prev_dir"]   = _cur_dir
st.session_state["_dir_streak"] = _streak
_stability_mins = _streak * 2   # ~2 min per refresh
_stable_color   = "#3fb950" if _stability_mins >= 30 else ("#ffd700" if _stability_mins >= 10 else "#8b949e")
_stable_label   = f"Stable {_stability_mins}+ min" if _stability_mins >= 10 else "Just updated"

# Accuracy stats from log
_acc_stats     = _accuracy_stats(_sig_rows)
_total_resolved = sum(v["n"] for v in _acc_stats.values())

with st.expander(f"📈 72h Bias Accuracy", expanded=False):

    # ── Backtest cutoff: only count rows from the current scoring engine. ──
    # Pre-cutoff rows came from an older engine (double-EMA bug) and would
    # poison the winrate. They still appear on the history chart below.
    try:
        _cutoff_dt = _dt.fromisoformat(_BACKTEST_CUTOFF_UTC_ISO)
    except Exception:
        _cutoff_dt = None

    def _post_cutoff(_r):
        if _cutoff_dt is None:
            return True
        try:
            return _dt.fromisoformat(_r.get("ts", "")) >= _cutoff_dt
        except Exception:
            return False

    _bt_rows = [r for r in _sig_rows if _post_cutoff(r)]

    # Live log winrate: all logged signals (already filtered to |score| >= 40 at log time)
    _live_wins = sum(1 for r in _bt_rows if r.get("correct") in ("1", "2"))
    _live_loss = sum(1 for r in _bt_rows if r.get("correct") == "0")
    _live_n    = _live_wins + _live_loss
    _live_wr   = round(_live_wins / _live_n * 100) if _live_n > 0 else None
    _live_col  = ("#8b949e" if _live_wr is None
                  else "#3fb950" if _live_wr >= 60
                  else "#ffd700" if _live_wr >= 50
                  else "#f85149")

    _live_display = f"{_live_wr}%" if _live_wr is not None else "building..."
    # Show how many pre-cutoff rows were excluded so the user knows what's
    # in/out. Pre-cutoff rows came from older scoring engines.
    _excluded    = sum(1 for r in _sig_rows
                       if r.get("correct") in ("0", "1", "2")
                       and not _post_cutoff(r))
    if _live_n > 0:
        _live_note = f"{_live_n} trades resolved"
        if _excluded > 0:
            _live_note += f"  ({_excluded} pre-patch excluded)"
    else:
        _live_note = "building... (excludes pre-patch rows)" if _excluded > 0 else "logs every 5 min"

    st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
     padding:16px 22px; display:flex; gap:32px; align-items:center;">
  <div style="text-align:center;">
    <div style="font-size:11px; color:#8b949e; margin-bottom:4px;">LIVE WINRATE</div>
    <div style="font-size:48px; font-weight:800; color:{_live_col}; line-height:1;">{_live_display}</div>
    <div style="font-size:10px; color:#484f58; margin-top:6px;">{_live_note}</div>
  </div>
  <div style="font-size:11px; color:#6e7681; max-width:300px; line-height:1.6;">
    <b style="color:#8b949e;">This number uses the FULL 16-signal score</b> from the live log.<br>
    Every 5 min when |score| ≥ 25, the timestamp + score + price get logged.
    72h later, the row is resolved against the actual price:
    if score sign matched the price move = WIN.<br>
    <span style="color:#484f58; font-size:10px;">(The legacy technicals-only backtest in a separate panel is just a cold-start placeholder — this live winrate is the real number.)</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Score history chart (last 72h from log) ───────────────────
    _cutoff_72h = _dt.now(_tz.utc).timestamp() - 72 * 3600
    _hist_rows  = []
    for _r in _sig_rows:
        try:
            _ts = _dt.fromisoformat(_r["ts"]).timestamp()
            if _ts >= _cutoff_72h:
                _sgt = _tz(offset=__import__("datetime").timedelta(hours=8))
                _local_ts = _dt.fromisoformat(_r["ts"]).astimezone(_sgt).replace(tzinfo=None)
                _hist_rows.append((_local_ts, float(_r["score"])))
        except Exception:
            continue

    # Always render a 72h-wide chart — empty space on the left until data fills in
    import matplotlib.dates as mdates
    import datetime as _datetime

    _sgt_tz   = _tz(offset=_datetime.timedelta(hours=8))
    _now_sgt  = _dt.now(_tz.utc).astimezone(_sgt_tz).replace(tzinfo=None)
    _x_end    = _now_sgt
    _x_start  = _now_sgt - _datetime.timedelta(hours=72)

    _sf, _sa = plt.subplots(figsize=(13, 2.6))
    _sf.patch.set_facecolor("#0d1117")
    _sa.set_facecolor("#0d1117")

    if len(_hist_rows) >= 1:
        _hist_times  = [h[0] for h in _hist_rows]
        _hist_scores = [h[1] for h in _hist_rows]
        _sv = np.array(_hist_scores)
        if len(_hist_rows) >= 2:
            _sa.fill_between(_hist_times, _sv, 0,
                             where=(_sv >= 25),  color="#3fb950", alpha=0.20)
            _sa.fill_between(_hist_times, _sv, 0,
                             where=(_sv <= -25), color="#f85149", alpha=0.20)
            _sa.fill_between(_hist_times, _sv, 0,
                             where=((_sv > -25) & (_sv < 25)), color="#8b949e", alpha=0.08)
        _sa.plot(_hist_times, _sv, color="#58a6ff", lw=1.5, zorder=3)

    _sa.axhline(0,   color="#484f58", lw=0.7, ls="--")
    _sa.axhline(25,  color="#3fb950", lw=0.6, ls=":", alpha=0.5)
    _sa.axhline(-25, color="#f85149", lw=0.6, ls=":", alpha=0.5)
    _sa.axhline(40,  color="#3fb950", lw=0.6, ls="--", alpha=0.4)
    _sa.axhline(-40, color="#f85149", lw=0.6, ls="--", alpha=0.4)

    _sa.text(1.002, 40,  "+40",  transform=_sa.get_yaxis_transform(), fontsize=6, color="#3fb950", va="center")
    _sa.text(1.002, 25,  "+25",  transform=_sa.get_yaxis_transform(), fontsize=6, color="#3fb950", va="center")
    _sa.text(1.002, -25, "−25",  transform=_sa.get_yaxis_transform(), fontsize=6, color="#f85149", va="center")
    _sa.text(1.002, -40, "−40",  transform=_sa.get_yaxis_transform(), fontsize=6, color="#f85149", va="center")

    _sa.set_xlim(_x_start, _x_end)
    _sa.set_ylim(-75, 75)
    _sa.set_ylabel("Bias Score", color="#8b949e", fontsize=7)
    _sa.tick_params(colors="#8b949e", labelsize=7)
    for _sp in _sa.spines.values(): _sp.set_edgecolor("#30363d")
    _sa.set_title("72h Bias Score History  (every 5 min · SGT)",
                  color="#8b949e", fontsize=8, loc="left", pad=4)
    _sa.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    _sa.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(pad=0.4)
    st.pyplot(_sf, use_container_width=True)
    plt.close(_sf)

    if len(_hist_rows) == 0:
        st.caption("No data yet — first point will appear within 5 minutes.")

    # ── Calibration buckets ────────────────────────────────────────
    # Shows actual hit-rate per score bucket. Monotonicity = well-calibrated model.
    # Needs resolved signals (correct==1/0) to populate.
    _resolved = [r for r in _bt_rows if r.get("correct") in ("0", "1", "2")]
    if len(_resolved) >= 5:
        _buckets = [
            ("< −40",  lambda s: s <= -40),
            ("−40–−25", lambda s: -40 < s <= -25),
            ("−25–0",  lambda s: -25 < s < 0),
            ("0–+25",  lambda s: 0 <= s < 25),
            ("+25–+40", lambda s: 25 <= s < 40),
            ("> +40",  lambda s: s >= 40),
        ]
        _cal_rows = []
        for _blabel, _bfn in _buckets:
            _bin_rows = [r for r in _resolved if _bfn(float(r.get("score", 0)))]
            if _bin_rows:
                _wins  = sum(1 for r in _bin_rows if r.get("correct") in ("1", "2"))
                _wr    = round(_wins / len(_bin_rows) * 100)
                _col   = "#3fb950" if _wr >= 60 else "#ffd700" if _wr >= 50 else "#f85149"
                _cal_rows.append((_blabel, len(_bin_rows), _wr, _col))
        if _cal_rows:
            st.markdown("**Calibration** — hit-rate by score bucket (monotonic = well-calibrated)")
            _cal_html = '<div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">'
            for _bl, _bn, _bwr, _bc in _cal_rows:
                _cal_html += (f'<div style="background:#161b22; border:1px solid #30363d; border-radius:8px;'
                              f' padding:8px 14px; text-align:center; min-width:80px;">'
                              f'<div style="font-size:10px; color:#8b949e;">{_bl}</div>'
                              f'<div style="font-size:20px; font-weight:800; color:{_bc};">{_bwr}%</div>'
                              f'<div style="font-size:9px; color:#484f58;">n={_bn}</div></div>')
            _cal_html += '</div>'
            st.markdown(_cal_html, unsafe_allow_html=True)
    else:
        st.caption(f"Calibration buckets will appear once ≥5 signals resolve (currently {len(_resolved)} resolved).")



# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Price & Technicals",
    "🔍  Cycle Signals",
    "🏦  ETF Flows",
    "🔥  Liquidity",
    "🧮  Advanced Indicators",
])


# ── Tab 1: Price & Technicals ─────────────────────────────────
with tab1:
    # ── Shared computation (used by scorer + Technical Snapshot table) ──
    pdf  = a["plot_df"]
    obv  = a["obv_trend"].replace("_", " ").title()
    div  = a["div_rsi"]
    fund = crypto_sig.get("funding_sentiment", "N/A")

    def _safe_last(col):
        s = pdf[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    bw_now   = _safe_last("BB_bw")
    pctb_now = _safe_last("BB_pctb")
    bw_mean  = float(pdf["BB_bw"].dropna().mean()) if not pdf["BB_bw"].dropna().empty else 1
    bb_sq    = "⚡ Squeeze!" if bw_now and bw_now < bw_mean * 0.6 else ""
    k_now    = _safe_last("Stoch_K")
    d_now    = _safe_last("Stoch_D")
    stoch_s  = ("🔴 Overbought" if k_now and k_now > 80 else
                "🟢 Oversold"   if k_now and k_now < 20 else "⚪ Neutral")
    atr_now  = _safe_last("ATR")
    atr_pct  = f"{atr_now / a['price'] * 100:.1f}%" if atr_now else "N/A"
    t_now    = _safe_last("Ichi_Tenkan")
    kj_now   = _safe_last("Ichi_Kijun")
    sa_now   = _safe_last("Ichi_SpanA")
    sb_now   = _safe_last("Ichi_SpanB")
    if sa_now and sb_now:
        cloud_top, cloud_bot = max(sa_now, sb_now), min(sa_now, sb_now)
        ichi_s = ("🟢 Above cloud" if a["price"] > cloud_top else
                  "🔴 Below cloud" if a["price"] < cloud_bot else "🟡 Inside cloud")
    else:
        cloud_top = cloud_bot = None
        ichi_s = "N/A"
    tk_s  = ("TK ↑ bull" if t_now and kj_now and t_now > kj_now else
             "TK ↓ bear" if t_now and kj_now else "N/A")
    ac_now = _safe_last("AC")
    ao_now = _safe_last("AO")
    ac_s   = ("🟢 Accel ↑" if ac_now and ac_now > 0 else
              "🔴 Decel ↓" if ac_now and ac_now < 0 else "⚪ Neutral")

    # ── Prediction / Daily Trend variables ─────────────────────────────
    pred_col   = prediction.get("color", "#8b949e")
    pred_label = prediction.get("label", "N/A")
    pred_score = prediction.get("total", 0)
    pred_max   = prediction.get("max", 14)
    pred_conf  = prediction.get("confidence", 0)
    pred_sigs  = prediction.get("signals", {})

    # ── Instantaneous Signal Scorer computation ─────────────────────────
    _rsi_now      = _safe_last("RSI")
    _macd_hist    = _safe_last("MACD_Hist")
    _macd_sig_v   = _safe_last("MACD_Signal")
    _macd_line    = _safe_last("MACD")
    _pdf_mh       = pdf["MACD_Hist"].dropna()
    _macd_hist_p  = float(_pdf_mh.iloc[-2]) if len(_pdf_mh) >= 2 else None

    _close_s = pdf["Close"].dropna()
    _ema8_d  = float(_close_s.ewm(span=8,  adjust=False).mean().iloc[-1]) if not _close_s.empty else None
    _ema21_d = float(_close_s.ewm(span=21, adjust=False).mean().iloc[-1]) if not _close_s.empty else None
    _ema50_d = float(_close_s.ewm(span=50, adjust=False).mean().iloc[-1]) if not _close_s.empty else None
    _ma50_d  = float(pdf["MA50"].dropna().iloc[-1])  if not pdf["MA50"].dropna().empty  else None
    _ma200_d = float(pdf["MA200"].dropna().iloc[-1]) if not pdf["MA200"].dropna().empty else None

    _chikou_s    = pdf["Ichi_Chikou"].dropna()
    _chikou_v    = float(_chikou_s.iloc[-1])  if not _chikou_s.empty else None
    _close_26ago = float(_close_s.iloc[-27])  if len(_close_s) >= 27 else None

    _cl     = a["closes"]
    _mom24h = a["chg_24h"]
    _mom6d  = (_cl[-1] - _cl[-7])  / _cl[-7]  * 100 if len(_cl) >= 7  else 0
    _mom7d  = (_cl[-1] - _cl[-8])  / _cl[-8]  * 100 if len(_cl) >= 8  else 0
    _mom30d = (_cl[-1] - _cl[-31]) / _cl[-31] * 100 if len(_cl) >= 31 else 0

    _etf_trend   = crypto_sig.get("etf_flow_trend", "N/A")
    _fg_val      = crypto_sig.get("fear_greed_value", 50)
    _dom_trend   = crypto_sig.get("dominance_trend", "")
    _fund_l      = fund.lower()

    _la          = btc_liq.get("liq_analysis", {}) if isinstance(btc_liq, dict) else {}
    _cascade     = _la.get("cascade_direction", "")
    _liq_bias    = btc_liq.get("liq_bias", "")     if isinstance(btc_liq, dict) else ""
    _depth_ratio = btc_liq.get("liq_depth_ratio", 1.0) if isinstance(btc_liq, dict) else 1.0
    _hz          = _la.get("hunt_zones", [])
    _ask_pull = _bid_pull = 0.0
    for _z in _hz:
        if abs(_z.get("price", price) - price) / price <= 0.05:
            _dp = max(abs(_z.get("price", price) - price) / price, 1e-4)
            _st = (_z.get("wall", 0) / 1e6) * (_z.get("fuel", 0) / 1e6) / (_dp ** 2)
            if _z.get("side") == "ASK": _ask_pull += _st
            else:                       _bid_pull += _st
    _hunt_pull = (_ask_pull - _bid_pull) / (_ask_pull + _bid_pull) if (_ask_pull + _bid_pull) > 0 else 0.0
    _long_liq  = _la.get("long_liq_total",  0)
    _short_liq = _la.get("short_liq_total", 0)

    _fib_lvls = a.get("fib", {}).get("levels", {})
    _fib_382  = _fib_lvls.get("38.2%")
    _fib_618  = _fib_lvls.get("61.8%")
    _pfl_s    = w52.get("pct_from_low")
    _sr_score = 0
    if a.get("imm_sup") and a.get("imm_res") and price:
        _ds = (price - a["imm_sup"]) / price
        _dr = (a["imm_res"] - price) / price
        _sr_score = 1 if _dr < _ds * 0.5 else (-1 if _ds < _dr * 0.5 else 0)

    _poly_sig  = poly_sentiment.get("signal", 0) if isinstance(poly_sentiment, dict) else 0
    _candle_sc = (bias_72h.get("signals", {}).get("Candlestick", (0, ""))[0]
                  if isinstance(bias_72h, dict) else 0)

    _sg = {
        "Momentum": {
            "RSI":        (1 if _rsi_now and _rsi_now < 45 else -1 if _rsi_now and _rsi_now > 55 else 0),
            "MACD Hist":  (1 if _macd_hist and _macd_hist > 0 else -1 if _macd_hist and _macd_hist < 0 else 0),
            "MACD Accel": (1 if _macd_hist and _macd_hist_p and _macd_hist > _macd_hist_p else
                           -1 if _macd_hist and _macd_hist_p and _macd_hist < _macd_hist_p else 0),
            "MACD Cross": (1 if _macd_line and _macd_sig_v and _macd_line > _macd_sig_v else
                           -1 if _macd_line and _macd_sig_v and _macd_line < _macd_sig_v else 0),
            "Stochastic": (1 if k_now and k_now < 20 else -1 if k_now and k_now > 80 else 0),
            "ADX":        (1 if a["adx_val"] and a["adx_val"] > 25 and _ma50_d and price > _ma50_d else
                           -1 if a["adx_val"] and a["adx_val"] > 25 and _ma50_d and price < _ma50_d else 0),
            "AO":         (1 if ao_now and ao_now > 0 else -1 if ao_now and ao_now < 0 else 0),
            "AC":         (1 if ac_now and ac_now > 0 else -1 if ac_now and ac_now < 0 else 0),
        },
        "Trend": {
            "EMA8/21":    (1 if _ema8_d and _ema21_d and _ema8_d > _ema21_d else
                           -1 if _ema8_d and _ema21_d else 0),
            "EMA21/50":   (1 if _ema21_d and _ema50_d and _ema21_d > _ema50_d else
                           -1 if _ema21_d and _ema50_d else 0),
            "vs MA50":    (1 if _ma50_d and price > _ma50_d else -1 if _ma50_d else 0),
            "vs MA200":   (1 if _ma200_d and price > _ma200_d else -1 if _ma200_d else 0),
            "Ichi Cloud": (1 if "above" in ichi_s.lower() else -1 if "below" in ichi_s.lower() else 0),
            "TK Cross":   (1 if t_now and kj_now and t_now > kj_now else
                           -1 if t_now and kj_now and t_now < kj_now else 0),
            "Chikou":     (1 if _chikou_v and _close_26ago and _chikou_v > _close_26ago else
                           -1 if _chikou_v and _close_26ago and _chikou_v < _close_26ago else 0),
            "24h Mom":    (1 if _mom24h and _mom24h > 1 else -1 if _mom24h and _mom24h < -1 else 0),
            "6d Mom":     (1 if _mom6d > 2 else -1 if _mom6d < -2 else 0),
            "7d Mom":     (1 if _mom7d > 2 else -1 if _mom7d < -2 else 0),
            "30d Mom":    (1 if _mom30d > 5 else -1 if _mom30d < -5 else 0),
        },
        "Volatility": {
            "BB %B":      (1 if pctb_now is not None and pctb_now < 0.2 else
                           -1 if pctb_now is not None and pctb_now > 0.8 else 0),
            "BB Squeeze": 0,
        },
        "Vol / Flow": {
            "OBV":        (1 if "accum" in obv.lower() else -1 if "distrib" in obv.lower() else 0),
            "ETF Flows":  (1 if "positive" in _etf_trend.lower() else
                           -1 if "negative" in _etf_trend.lower() else 0),
            "Funding":    (1 if "negative" in _fund_l else
                           -1 if "extreme" in _fund_l and "positive" in _fund_l else 0),
        },
        "Liquidity": {
            "OB Depth":   (1 if _liq_bias == "BID" else -1 if _liq_bias == "ASK" else 0),
            "Cascade":    (1 if _cascade == "UP"  else -1 if _cascade == "DOWN" else 0),
            "Hunt Zone":  (1 if _hunt_pull > 0.1  else -1 if _hunt_pull < -0.1 else 0),
            "Depth Ratio":(1 if _depth_ratio > 1.15 else -1 if _depth_ratio < 0.85 else 0),
            "Liq Asym":   (1 if _short_liq > _long_liq * 1.2 else
                           -1 if _long_liq  > _short_liq * 1.2 else 0),
        },
        "Price Levels": {
            "Fib Zone":   (1 if _fib_618 and price > _fib_618 else
                           -1 if _fib_382 and price < _fib_382 else 0),
            "Sup/Res":    _sr_score,
            "52W Pos":    (1 if _pfl_s is not None and _pfl_s < 33 else
                           -1 if _pfl_s is not None and _pfl_s > 66 else 0),
        },
        "Sentiment": {
            "Polymarket": (1 if _poly_sig > 0.1 else -1 if _poly_sig < -0.1 else 0),
            "Fear/Greed": (1 if isinstance(_fg_val, (int, float)) and _fg_val < 30 else
                           -1 if isinstance(_fg_val, (int, float)) and _fg_val > 70 else 0),
            "BTC Dom":    (1 if "rising" in _dom_trend.lower() or "increas" in _dom_trend.lower() else
                           -1 if "falling" in _dom_trend.lower() or "decreas" in _dom_trend.lower() else 0),
            "Candlestick":(1 if _candle_sc > 0 else -1 if _candle_sc < 0 else 0),
            "RSI Div":    (1 if "bull" in div.lower() else -1 if "bear" in div.lower() else 0),
        },
    }

    _all_flat = {f"{g}:{k}": v for g, d in _sg.items() for k, v in d.items()}
    _bull  = sum(1 for v in _all_flat.values() if v > 0)
    _bear  = sum(1 for v in _all_flat.values() if v < 0)
    _neut  = sum(1 for v in _all_flat.values() if v == 0)
    _total = len(_all_flat)
    _net   = _bull - _bear

    _thresh = _total * 0.15
    if   _net >=  _thresh * 2: _bias_label, _bias_col = "Strongly Bullish", "#3fb950"
    elif _net >=  _thresh:     _bias_label, _bias_col = "Leaning Bullish",  "#3fb950"
    elif _net <= -_thresh * 2: _bias_label, _bias_col = "Strongly Bearish", "#f85149"
    elif _net <= -_thresh:     _bias_label, _bias_col = "Leaning Bearish",  "#f85149"
    else:                      _bias_label, _bias_col = "Mixed / Neutral",  "#e3b341"

    _bw = round(_bull / _total * 20) if _total else 0
    _rw = round(_bear / _total * 20) if _total else 0
    _nw = max(0, 20 - _bw - _rw)

    def _pill(label, score):
        bg  = "#1a3a1f" if score > 0 else ("#3a1a1a" if score < 0 else "#1c1c24")
        col = "#3fb950" if score > 0 else ("#f85149" if score < 0 else "#8b949e")
        bdr = "#3fb95040" if score > 0 else ("#f8514940" if score < 0 else "#30363d")
        ico = "↑" if score > 0 else ("↓" if score < 0 else "→")
        return (f'<span style="display:inline-block;padding:2px 6px;margin:2px 2px 2px 0;'
                f'border-radius:4px;font-size:10.5px;font-weight:600;background:{bg};'
                f'color:{col};border:1px solid {bdr};">{ico} {label}</span>')

    def _cat_row(cat_name, cat_scores):
        b = sum(1 for v in cat_scores.values() if v > 0)
        r = sum(1 for v in cat_scores.values() if v < 0)
        n = sum(1 for v in cat_scores.values() if v == 0)
        col = "#3fb950" if b > r else ("#f85149" if r > b else "#e3b341")
        pills = "".join(_pill(k, v) for k, v in cat_scores.items())
        return (f'<div style="margin-top:10px;">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                f'<span style="font-size:11px;font-weight:600;color:#8b949e;min-width:90px;">{cat_name}</span>'
                f'<span style="font-size:10px;color:{col};font-weight:700;">▲{b} ▼{r} →{n}</span></div>'
                f'<div>{pills}</div></div>')

    _cats_html = "".join(_cat_row(cat, scores) for cat, scores in _sg.items())

    # ── Technical Snapshot — all indicators ─────────────────────────────
    with st.expander("📋 Technical Snapshot — All Indicators", expanded=False):
        # Note variables (shared with table rows below)
        pctb_val   = pctb_now if pctb_now is not None else 0.5
        pctb_sig   = ("🔴 At upper band" if pctb_val > 0.9 else "🟢 At lower band" if pctb_val < 0.1 else "⚪ Mid-band")
        pctb_note  = ("Price pressing the upper Bollinger band — statistically stretched, pullback risk." if pctb_val > 0.8 else
                      "Price near the lower Bollinger band — potential mean-reversion bounce zone." if pctb_val < 0.2 else
                      "Price in the middle of the bands — no extreme compression signal.")
        bb_note    = ("Bands are unusually narrow (< 60% of average width). Volatility is compressing; a sharp breakout is likely imminent." if bb_sq else
                      f"Band width ({bw_now:.1f}%) is within normal range. No volatility squeeze pending." if bw_now else "N/A")
        stoch_note = ("Both %K and %D above 80 — overextended upside. Watch for a bearish cross." if k_now and k_now > 80 else
                      "Both %K and %D below 20 — momentum exhausted downside. Mean-reversion bounce zone." if k_now and k_now < 20 else
                      f"%K={k_now:.0f}, %D={d_now:.0f} — neutral territory, no extreme." if k_now and d_now else "N/A")
        atr_note   = (f"ATR ${atr_now:,.0f} = {atr_pct} of price. " +
                      ("Elevated — wider stops and larger moves expected." if atr_now and atr_now / price > 0.03 else
                       "Moderate volatility — normal intraday swings.")) if atr_now else "N/A"
        if sa_now and sb_now:
            _dist_pct2 = abs(price - cloud_top) / price * 100 if price > cloud_top else abs(price - cloud_bot) / price * 100
            ichi_note  = (f"Price {_dist_pct2:.1f}% above Kumo cloud — bullish structure, cloud is dynamic support." if price > cloud_top else
                          f"Price {_dist_pct2:.1f}% below Kumo cloud — bearish structure, cloud is overhead resistance." if price < cloud_bot else
                          "Price inside the cloud — indecision zone, trend undefined.")
        else:
            ichi_note = "Ichimoku data unavailable."
        tk_note    = ("Tenkan above Kijun — short-term momentum bullish, buying pressure confirmed." if t_now and kj_now and t_now > kj_now else
                      "Tenkan below Kijun — short-term momentum bearish, selling pressure dominant." if t_now and kj_now else "N/A")
        ac_note    = ("AC positive — momentum accelerating upward. Stronger entry conditions." if ac_now and ac_now > 0 else
                      "AC negative — momentum decelerating or reversing. Caution on fresh longs." if ac_now and ac_now < 0 else "N/A")
        obv_note   = ("OBV trending up — volume flowing in, suggests sustained/institutional buying." if "accum" in obv.lower() else
                      "OBV trending down — volume flowing out, selling pressure beneath the surface." if "distrib" in obv.lower() else
                      "OBV flat/mixed — no clear volume-based directional conviction.")
        div_note   = ("RSI making higher lows while price makes lower lows — hidden bullish divergence, downtrend losing strength." if "bull" in div.lower() else
                      "RSI making lower highs while price makes higher highs — hidden bearish divergence, uptrend weakening." if "bear" in div.lower() else
                      "No divergence — RSI and price direction aligned. No hidden reversal signal.")
        _pfl  = w52.get("pct_from_low")
        _pfh  = w52.get("pct_from_high")
        w52_note = (f"Price {_pfl:.1f}% above 52W low and {_pfh:.1f}% below 52W high. " +
                    ("Lower third of annual range — long-term value zone." if _pfl and _pfl < 33 else
                     "Upper third of annual range — approaching prior highs, more resistance." if _pfl and _pfl > 66 else
                     "Mid-range within annual high/low — no extreme valuation signal.")) if _pfl is not None else "52W data N/A"

        # Extra values for new rows
        _ma50_dev  = (price - _ma50_d) / _ma50_d * 100 if _ma50_d else None
        _ma200_dev = (price - _ma200_d) / _ma200_d * 100 if _ma200_d else None
        _cascade_r = _la.get("cascade_ratio", 1.0)
        _near_bid  = _la.get("nearest_long_liq")
        _near_ask  = _la.get("nearest_short_liq")
        _bid_str   = f"${_near_bid[0]:,.0f} (${_near_bid[1]/1e6:.0f}M)" if _near_bid else "None"
        _ask_str   = f"${_near_ask[0]:,.0f} (${_near_ask[1]/1e6:.0f}M)" if _near_ask else "None"
        _etf_n_in  = crypto_sig.get("etf_flow_stats", {}).get("n_inflow",  0)
        _etf_n_out = crypto_sig.get("etf_flow_stats", {}).get("n_outflow", 0)
        _fg_label  = crypto_sig.get("fear_greed_label", "N/A")
        _btc_dom   = crypto_sig.get("btc_dominance", "N/A")
        _poly_thesis = poly_sentiment.get("thesis_label", "N/A") if isinstance(poly_sentiment, dict) else "N/A"
        _candle_note = (bias_72h.get("signals", {}).get("Candlestick", (0, ""))[1]
                        if isinstance(bias_72h, dict) else "No pattern detected.")
        _fib_dists = {k: abs(price - v) for k, v in _fib_lvls.items() if v}
        _fib_near  = min(_fib_dists, key=_fib_dists.get) if _fib_dists else None
        _fib_near_p = _fib_lvls.get(_fib_near) if _fib_near else None

        st.markdown("#### Momentum / Oscillators")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| RSI (14) | {f"{_rsi_now:.0f}" if _rsi_now else "N/A"} | {"🔴 Overbought" if _rsi_now and _rsi_now > 70 else "🟢 Oversold" if _rsi_now and _rsi_now < 30 else "🔴 Bearish" if _rsi_now and _rsi_now < 45 else "🟢 Bullish" if _rsi_now and _rsi_now > 55 else "⚪ Neutral"} | RSI > 55 = bullish momentum; < 45 = bearish. > 70 = overbought mean-reversion risk; < 30 = oversold bounce zone. |
| MACD Histogram | {f"{_macd_hist:+.0f}" if _macd_hist else "N/A"} | {"🟢 Positive" if _macd_hist and _macd_hist > 0 else "🔴 Negative" if _macd_hist and _macd_hist < 0 else "⚪ Flat"} | Histogram above zero = bullish pressure building; below = bearish. Currently {"accelerating" if _macd_hist and _macd_hist_p and abs(_macd_hist) > abs(_macd_hist_p) else "decelerating"}. |
| MACD Acceleration | {f"{(_macd_hist - _macd_hist_p):+.0f}" if _macd_hist and _macd_hist_p else "N/A"} | {"🟢 Accel ↑" if _macd_hist and _macd_hist_p and _macd_hist > _macd_hist_p else "🔴 Decel ↓" if _macd_hist and _macd_hist_p else "⚪ N/A"} | Change in histogram from prior bar. Positive delta = momentum building; negative = fading. Key early-warning signal. |
| MACD Cross | {f"Line {_macd_line:+.0f} / Sig {_macd_sig_v:+.0f}" if _macd_line and _macd_sig_v else "N/A"} | {"🟢 Bullish cross" if _macd_line and _macd_sig_v and _macd_line > _macd_sig_v else "🔴 Bearish cross" if _macd_line and _macd_sig_v else "⚪ N/A"} | MACD line above signal = bullish crossover regime; below = bearish. Crossovers mark momentum regime changes. |
| Stochastic %K/%D | {f"{k_now:.0f} / {d_now:.0f}" if k_now and d_now else "N/A"} | {stoch_s} | {stoch_note} |
| ADX | {f"{a['adx_val']:.0f}" if a.get("adx_val") else "N/A"} | {"🟢 Trending ↑" if a.get("adx_val") and a["adx_val"] > 25 and _ma50_d and price > _ma50_d else "🔴 Trending ↓" if a.get("adx_val") and a["adx_val"] > 25 else "⚪ Ranging"} | ADX > 25 = strong directional trend; < 20 = ranging market. Direction assigned via price vs MA50. |
| AO (Awesome Osc.) | {f"{ao_now:+.0f}" if ao_now else "N/A"} | {"🟢 Positive" if ao_now and ao_now > 0 else "🔴 Negative" if ao_now and ao_now < 0 else "⚪ Flat"} | Difference of 5 and 34-period midpoint SMAs. Positive = bullish market force, negative = bearish. |
| AC (Accel. Osc.) | {f"{ac_now:+.1f}" if ac_now else "N/A"} | {ac_s} | {ac_note} |
""")

        st.markdown("#### Trend Indicators")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| EMA8 vs EMA21 | {f"EMA8 {'>' if _ema8_d and _ema21_d and _ema8_d > _ema21_d else '<'} EMA21" if _ema8_d and _ema21_d else "N/A"} | {"🟢 Bullish stack" if _ema8_d and _ema21_d and _ema8_d > _ema21_d else "🔴 Bearish stack" if _ema8_d and _ema21_d else "⚪ N/A"} | Fast EMA above slow = upward short-term momentum; below = downward. Most sensitive crossover in the stack. |
| EMA21 vs EMA50 | {f"EMA21 {'>' if _ema21_d and _ema50_d and _ema21_d > _ema50_d else '<'} EMA50" if _ema21_d and _ema50_d else "N/A"} | {"🟢 Bullish" if _ema21_d and _ema50_d and _ema21_d > _ema50_d else "🔴 Bearish" if _ema21_d and _ema50_d else "⚪ N/A"} | Medium-term trend alignment. EMA21 above EMA50 = sustained uptrend structure; below = sustained downtrend. |
| Price vs MA50 | {f"{_ma50_dev:+.1f}%" if _ma50_dev is not None else "N/A"} | {"🟢 Above" if _ma50_d and price > _ma50_d else "🔴 Below"} | 50-day MA is the medium-term trend floor. Above = bullish structure; crossing below is a meaningful trend break. |
| Price vs MA200 | {f"{_ma200_dev:+.1f}%" if _ma200_dev is not None else "N/A"} | {"🟢 Above" if _ma200_d and price > _ma200_d else "🔴 Below"} | 200-day MA is the long-term secular trend divider. Above = macro bull; below = macro bear regime. |
| Ichimoku Cloud | — | {ichi_s} | {ichi_note} |
| TK Cross | — | {tk_s} | {tk_note} |
| Chikou Span | {f"${_chikou_v:,.0f}" if _chikou_v else "N/A"} | {"🟢 Above price-26" if _chikou_v and _close_26ago and _chikou_v > _close_26ago else "🔴 Below price-26" if _chikou_v and _close_26ago else "⚪ N/A"} | Chikou (close plotted 26 bars back) above historical price = bullish cross-time confirmation; below = bearish. |
| 24h Momentum | {f"{_mom24h:+.1f}%" if _mom24h else "N/A"} | {"🟢 Up" if _mom24h and _mom24h > 1 else "🔴 Down" if _mom24h and _mom24h < -1 else "⚪ Flat"} | 24-hour close-to-close change. Immediate direction bias for intraday decisions. |
| 6-Day Momentum | {f"{_mom6d:+.1f}%"} | {"🟢 Up" if _mom6d > 2 else "🔴 Down" if _mom6d < -2 else "⚪ Flat"} | 6-day rolling return. Captures short-to-medium trend shifts. |
| 7-Day Momentum | {f"{_mom7d:+.1f}%"} | {"🟢 Up" if _mom7d > 2 else "🔴 Down" if _mom7d < -2 else "⚪ Flat"} | Weekly return. Trend context over the recent trading week. |
| 30-Day Momentum | {f"{_mom30d:+.1f}%"} | {"🟢 Up" if _mom30d > 5 else "🔴 Down" if _mom30d < -5 else "⚪ Flat"} | Monthly return. Reflects the medium-term cycle phase. |
""")

        st.markdown("#### Volatility")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| ATR (14) | {f"${atr_now:,.0f}" if atr_now else "N/A"} | {atr_pct} daily range | {atr_note} |
| Bollinger %B | {f"{pctb_now:.2f}" if pctb_now is not None else "N/A"} | {pctb_sig} | {pctb_note} |
| BB Bandwidth | {f"{bw_now:.1f}%" if bw_now else "N/A"} | {bb_sq if bb_sq else "Normal"} | {bb_note} |
""")

        st.markdown("#### Volume / Flow")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| OBV | — | {obv} | {obv_note} |
| ETF Flows | {f"{_etf_n_in} in / {_etf_n_out} out"} | {"🟢 Net Inflow" if "positive" in _etf_trend.lower() else "🔴 Net Outflow" if "negative" in _etf_trend.lower() else "⚪ Mixed"} | Spot Bitcoin ETF inflow/outflow. Net inflows = institutional demand; outflows = institutional selling pressure. |
| Funding Sentiment | — | {fund} | Funding reflects perpetual trader positioning cost. Neutral = balanced. Negative = shorts paying longs (contrarian bull signal). |
""")

        st.markdown("#### Liquidity / Order Book")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| OB Depth Bias | {_liq_bias} ({_depth_ratio:.2f}x) | {"🟢 Bid-heavy" if _liq_bias == "BID" else "🔴 Ask-heavy" if _liq_bias == "ASK" else "⚪ Balanced"} | Ratio of bid vs ask USD depth within ±15% of price. > 1 = more buy support; < 1 = more sell pressure. |
| Cascade Direction | {_cascade} (ratio {_cascade_r:.2f}x) | {"🟢 UP" if _cascade == "UP" else "🔴 DOWN" if _cascade == "DOWN" else "⚪ Balanced"} | Long vs short liquidation fuel. UP = short squeeze potential; DOWN = long cascade risk. Strongest market-structure signal. |
| Hunt Zone Pull | {"Upside" if _hunt_pull > 0.1 else "Downside" if _hunt_pull < -0.1 else "Neutral"} ({_hunt_pull:+.2f}) | {"🟢 Upside" if _hunt_pull > 0.1 else "🔴 Downside" if _hunt_pull < -0.1 else "⚪ Neutral"} | Short-liq clusters above pull price UP (squeeze magnet); long-liq clusters below pull DOWN (flush magnet). Score = notional / distance² × cascade chain mult. Order-book walls in the path dampen the pull. |
| Nearest Bid Wall | {_bid_str} | — | Largest bid-side liquidity cluster below price. Acts as a magnet for downside hunts or strong support. |
| Nearest Ask Wall | {_ask_str} | — | Largest ask-side liquidity cluster above price. Acts as a short-squeeze target or resistance ceiling. |
| Depth Ratio | {f"{_depth_ratio:.2f}"} | {"🟢 > 1.15" if _depth_ratio > 1.15 else "🔴 < 0.85" if _depth_ratio < 0.85 else "⚪ Balanced"} | Total USD bids / asks. Smoothed over recent snapshots. Sustained > 1 = buyers absorbing; < 1 = sellers dominant. |
| Liq Asymmetry | {f"Long ${_long_liq/1e6:.0f}M / Short ${_short_liq/1e6:.0f}M" if _long_liq or _short_liq else "N/A"} | {"🟢 Short-heavy" if _short_liq > _long_liq * 1.2 else "🔴 Long-heavy" if _long_liq > _short_liq * 1.2 else "⚪ Balanced"} | More short fuel = price can squeeze up to liquidate shorts; more long fuel = cascading longs below. |
""")

        st.markdown("#### Price Level Tools")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| Fibonacci Zone | {f"{_fib_near} = ${_fib_near_p:,.0f}" if _fib_near_p else "N/A"} | {"🟢 Above 61.8%" if _fib_618 and price > _fib_618 else "🔴 Below 38.2%" if _fib_382 and price < _fib_382 else "⚪ Mid-range"} | Price vs 52-week swing Fibonacci levels. Above 61.8% retracement = strong recovery; below 38.2% = deep retracement zone. |
| Support / Resistance | {f"S ${a['imm_sup']:,.0f} / R ${a['imm_res']:,.0f}" if a.get("imm_sup") and a.get("imm_res") else "N/A"} | {"🟢 Near support" if _sr_score > 0 else "🔴 Near resistance" if _sr_score < 0 else "⚪ Mid-range"} | Nearest support and resistance levels. Proximity score signals which level is dominating price behavior. |
| 52W Low / High | ${w52.get("w52_low","—"):,} / ${w52.get("w52_high","—"):,} | {w52.get("pct_from_low","—")}% from low | {w52_note} |
""")

        st.markdown("#### Sentiment / External")
        st.markdown(f"""
| Indicator | Value | Signal | What it means |
|-----------|-------|--------|---------------|
| RSI Divergence | — | {div.replace("none","None detected").title()} | {div_note} |
| Polymarket | {f"{_poly_thesis} ({_poly_sig:+.2f})" if isinstance(_poly_sig, float) else "N/A"} | {"🟢 Bullish" if _poly_sig > 0.1 else "🔴 Bearish" if _poly_sig < -0.1 else "⚪ Neutral"} | Prediction-market thesis score. Real money positioned; directional signal with genuine skin in the game. |
| Fear & Greed | {f"{_fg_val} — {_fg_label}" if isinstance(_fg_val, (int, float)) else "N/A"} | {"🟢 Extreme Fear (contrarian buy)" if isinstance(_fg_val, (int, float)) and _fg_val < 25 else "🔴 Extreme Greed (contrarian sell)" if isinstance(_fg_val, (int, float)) and _fg_val > 75 else "⚪ Neutral zone"} | Crypto Fear & Greed Index. Extreme fear historically marks bottoms (contrarian buy); extreme greed marks tops. |
| BTC Dominance | {_btc_dom} | {"🟢 Rising" if "rising" in _dom_trend.lower() or "increas" in _dom_trend.lower() else "🔴 Falling" if "falling" in _dom_trend.lower() or "decreas" in _dom_trend.lower() else "⚪ Stable"} | BTC share of total crypto market cap. Rising = capital rotating into BTC; falling = alt-season or risk-off rotation. |
| Candlestick Pattern | — | {"🟢 Bullish" if _candle_sc > 0 else "🔴 Bearish" if _candle_sc < 0 else "⚪ Neutral"} | {_candle_note if _candle_note else "No significant candlestick pattern on current daily candle."} |
""")

    st.divider()

    # Key levels row
    lv1, lv2, lv3, lv4 = st.columns(4)
    with lv1:
        st.metric("Support", f"${a['imm_sup']:,.0f}" if a["imm_sup"] else "—")
    with lv2:
        st.metric("Resistance", f"${a['imm_res']:,.0f}" if a["imm_res"] else "—")
    with lv3:
        ma50  = a["df"]["MA50"].iloc[-1]
        ma200 = a["df"]["MA200"].iloc[-1]
        above = price > ma50
        st.metric("MA50", f"${ma50:,.0f}", delta="Above ✓" if above else "Below ✗")
    with lv4:
        above200 = price > ma200
        dev200   = (price - ma200) / ma200 * 100
        st.metric("MA200", f"${ma200:,.0f}", delta=f"{dev200:+.1f}%")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    f_price = fig_price_chart(a)
    st.pyplot(f_price, use_container_width=True)
    plt.close(f_price)

    with st.expander("📖 Indicator glossary — what each line/panel measures", expanded=False):
        st.markdown("""
| Indicator | Colour / Style | What it measures | Signal to watch |
|-----------|---------------|-----------------|-----------------|
| **Price** | White line | BTC-USD closing price | Baseline — everything else is relative to this |
| **MA50** | Blue solid | 50-day simple moving average | Medium-term trend direction; price above = bullish structure |
| **MA200** | Orange solid | 200-day simple moving average | Long-term secular trend; classic bull/bear divider |
| **EMA8** | Green dashed | 8-day exponential MA — reacts faster than MA50 | Acts as a short-term dynamic support/resistance; price bouncing off it = momentum intact |
| **Golden ✕** | Gold star | MA50 crosses above MA200 | Historically reliable long-term buy signal |
| **Death ✕** | Red X | MA50 crosses below MA200 | Long-term bearish crossover; signals trend breakdown |
| **BB Upper / Lower** | Purple dashed + fill | Bollinger Bands — 2 std deviations around the 20-day MA | Price near upper band = extended/overbought; near lower = oversold. Narrow bands (squeeze) precede big moves |
| **Support** | Green dashed horizontal | Nearest key price floor below current price | Likely area where buyers step in; break below is bearish |
| **Resistance** | Red dashed horizontal | Nearest key ceiling above current price | Likely area where sellers appear; break above is bullish |
| **Fib levels** | Colour-coded horizontals | Fibonacci retracement levels from the last major swing high→low | Common reversal/consolidation zones: 50%, 61.8%, and 78.6% are the most watched |
| **Volume bars** | Green/red bars | Daily trading volume (green = up day, red = down day) | Rising price + rising volume = conviction; rising price + falling volume = weak move |
| **Volume MA20** | Blue line (volume panel) | 20-day average volume | Bars above the line = above-average participation |
| **RSI 14** | Orange line (bottom panel) | Relative Strength Index over 14 days — momentum oscillator 0–100 | >70 overbought (red fill), <30 oversold (green fill), 50 = neutral |
| **RSI Signal** | Blue dashed (bottom panel) | EMA9 of RSI — smoothed signal line | RSI crossing above signal = momentum building; crossing below = fading |
| **OBV** | Title label | On-Balance Volume trend — accumulation vs. distribution | "mild_distribution" / "mild_accumulation" tells you whether big money is quietly buying or selling |
| **ADX** | Title label | Average Directional Index — trend strength 0–100 | <20 = no clear trend (ranging); >25 = trending; >40 = strong trend |
| **BB bw** | Title label | Bollinger Band bandwidth — measures how tight/wide the bands are | Low % = squeeze (breakout pending); high % = band expansion already underway |
""")

    # 24h 15-min intraday chart — entry-timing context alongside the 72h bias
    f_15m = fig_intraday_15m(a)
    if f_15m is not None:
        st.pyplot(f_15m, use_container_width=True)
        plt.close(f_15m)

# ── Tab 2: Cycle Signals ─────────────────────────────────────
with tab2:
    # ── Instantaneous Trend Score ───────────────────────────────
    st.markdown(f"""
<div style="background:#161b22;border:1px solid {_bias_col}55;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <div>
      <span style="font-size:15px;font-weight:700;color:{_bias_col};">{_bias_label}</span>
      <span style="font-size:12px;color:#8b949e;margin-left:10px;">Instantaneous Trend Score</span>
    </div>
    <div style="font-size:12px;color:#8b949e;">
      <span style="color:#3fb950;font-weight:600;">▲ {_bull} bull</span>
      &nbsp;·&nbsp;<span style="color:#f85149;font-weight:600;">▼ {_bear} bear</span>
      &nbsp;·&nbsp;<span style="color:#8b949e;">→ {_neut} neutral</span>
      &nbsp;·&nbsp;<span style="color:{_bias_col};font-weight:700;">net {_net:+d} / {_total}</span>
    </div>
  </div>
  <div style="margin-top:10px;font-family:monospace;font-size:12px;">
    <span style="color:#3fb950;">{"█" * _bw}</span><span style="color:#f85149;">{"█" * _rw}</span><span style="color:#30363d;">{"░" * _nw}</span>
    <span style="color:#484f58;font-size:10px;margin-left:6px;">{_bull}/{_total} bull · {_bear}/{_total} bear</span>
  </div>
  {_cats_html}
  <div style="font-size:10px;color:#484f58;margin-top:10px;border-top:1px solid #21262d;padding-top:6px;">
    {_total} indicators · Momentum · Trend · Volatility · Vol/Flow · Liquidity · Price Levels · Sentiment · independent of 72h bias
  </div>
</div>
""", unsafe_allow_html=True)

    # Big cycle phase banner
    _col_map = {
        "PROBABLE BOTTOM": "#3fb950",
        "BOTTOM FORMING":  "#f0883e",
        "MID-CYCLE":       "#8b949e",
        "TOP FORMING":     "#f0883e",
        "PROBABLE TOP":    "#f85149",
    }
    banner_col = _col_map.get(cycle["phase"], "#8b949e")
    bar_filled = int((cycle["total"] + 24) / 48 * 20)
    bar_str    = "█" * bar_filled + "░" * (20 - bar_filled)
    st.markdown(f"""
<div style="background:{banner_col}18; border:1px solid {banner_col};
     border-radius:10px; padding:16px 20px; margin-bottom:16px;">
  <div style="font-size:22px; font-weight:700; color:{banner_col};">
    {cycle['emoji']} {cycle['phase']} &nbsp; <span style="font-size:14px; font-weight:400;">{cycle['total']:+d} / 24</span>
  </div>
  <div style="font-family:monospace; font-size:13px; color:{banner_col}; margin:4px 0;">[{bar_str}]</div>
  <div style="font-size:13px; color:#c9d1d9; margin-top:6px;">{cycle['advice']}</div>
</div>
""", unsafe_allow_html=True)

    # 8 signal cards
    signals  = cycle.get("signals", {})
    sig_list = list(signals.items())
    row1     = st.columns(4)
    row2     = st.columns(4)
    for i, (name, (score, explanation)) in enumerate(sig_list):
        col     = row1[i] if i < 4 else row2[i - 4]
        pill_cls = "sig-bull" if score > 0 else ("sig-bear" if score < 0 else "sig-neut")
        icon     = "↑" if score > 0 else ("↓" if score < 0 else "→")
        with col:
            st.markdown(f"""
<div class="info-box">
  <div style="font-size:12px; color:#8b949e; margin-bottom:4px;">{name}</div>
  <span class="sig-pill {pill_cls}">{icon} {score:+d}</span>
  <div style="font-size:11px; color:#8b949e; margin-top:6px; line-height:1.4;">{explanation}</div>
</div>
""", unsafe_allow_html=True)



# ── Tab 3: ETF Flows ─────────────────────────────────────────
with tab3:
    etf_trend  = crypto_sig.get("etf_flow_trend",  "N/A")
    etf_detail = crypto_sig.get("etf_flow_detail", "N/A")
    fs         = crypto_sig.get("etf_flow_stats",  {})
    n_in  = fs.get("n_inflow",  0)
    n_out = fs.get("n_outflow", 0)
    n_tot = fs.get("n_etfs",    0)
    accel = fs.get("accelerating", False)

    ef1, ef2, ef3 = st.columns(3)
    with ef1:
        st.metric("Flow Regime", etf_trend, delta="Accelerating ⚡" if accel else None)
    with ef2:
        st.metric("ETFs Inflow / Total", f"{n_in} / {n_tot}")
    with ef3:
        roll = fs.get("roll20_last", 0)
        st.metric("20d Rolling Signal", f"{roll:+.2f}")

    st.markdown(f'<div class="info-box" style="font-size:12px;">{etf_detail}</div>',
                unsafe_allow_html=True)

    f_etf = fig_etf_flows(a)
    st.pyplot(f_etf, use_container_width=True)
    plt.close(f_etf)


# ── Tab 4: Liquidity ─────────────────────────────────────────
with tab4:
    import streamlit.components.v1 as _stc

    if not a["has_liq"]:
        st.warning("Order book data unavailable — check your internet connection.")
    else:
        bias   = btc_liq.get("liq_bias", "N/A")
        stren  = btc_liq.get("liq_bias_strength", "N/A")
        ratio  = btc_liq.get("liq_depth_ratio", 1.0)
        spread = btc_liq.get("liq_spread_bps", 0)
        liq_a  = btc_liq.get("liq_analysis", {})
        cd     = liq_a.get("cascade_direction", "N/A")

        # ── Metrics row ───────────────────────────────────────────────────────
        ll1, ll2, ll3, ll4 = st.columns(4)
        with ll1:
            bid_usd = btc_liq.get("liq_total_bid", 0)
            ask_usd = btc_liq.get("liq_total_ask", 0)
            st.metric("Depth Bias", f"{bias} ({stren})", delta=f"{ratio:.2f}× bid/ask")
        with ll2:
            st.metric("Bid Depth", _fusd(bid_usd))
        with ll3:
            st.metric("Ask Depth", _fusd(ask_usd))
        with ll4:
            st.metric("Spread", f"{spread:.1f} bps", delta=f"Cascade: {cd}")

        # Exchange source badges
        _ob_src   = btc_liq.get("liq_source", "")
        _src_map  = {"bn_spot": "Binance spot", "bn_fut": "Binance fut",
                     "bybit": "Bybit", "okx": "OKX"}
        _src_html = "".join(
            f'<span style="background:#21262d;color:#8b949e;padding:1px 7px;'
            f'border-radius:10px;font-size:11px;margin-right:4px;">{_src_map.get(s,s)}</span>'
            for s in _ob_src.split("+") if s
        )
        if _src_html:
            st.markdown(f'<div style="margin:2px 0 10px 0">{_src_html}</div>',
                        unsafe_allow_html=True)

    # ── Coinglass Liquidation Heatmap (embedded) ──────────────────────────────
    _CG_URL = "https://www.coinglass.com/pro/futures/LiquidationHeatMap?coin=BTC"
    _cg_col1, _cg_col2 = st.columns([6, 1])
    with _cg_col1:
        st.markdown("**Coinglass — Liquidation Heatmap**")
    with _cg_col2:
        st.link_button("↗ Open in browser", _CG_URL)
    _stc.html(
        f'<iframe src="{_CG_URL}" width="100%" height="620" '
        f'frameborder="0" scrolling="yes" '
        f'sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox">'
        f'</iframe>',
        height=625,
    )

    # ── Key Walls + Hunt Zones strip ──────────────────────────────────────────
    if a["has_liq"]:
        _walls_col, _hz_col = st.columns([1, 1])

        # ── Left: Key bid/ask walls ───────────────────────────────────────────
        with _walls_col:
            st.markdown(
                '<div style="font-size:11px;font-weight:600;color:#8b949e;'
                'letter-spacing:.08em;margin-bottom:8px;">KEY ORDER BOOK WALLS</div>',
                unsafe_allow_html=True,
            )
            _bid_walls = btc_liq.get("liq_bid_walls", [])[:3]
            _ask_walls = btc_liq.get("liq_ask_walls", [])[:3]

            # Pre-compute hunt likelihood for each wall
            _hz_prices = {_z["price"] for _z in liq_a.get("hunt_zones", [])}

            def _hunt_tier(wp, wn):
                dist_pct = abs(wp - price) / price * 100 + 0.01
                score = wn / (dist_pct ** 1.5)
                in_zone = any(abs(wp - _hz) < 150 for _hz in _hz_prices)
                return score, in_zone

            _all_walls = (
                [(_wp, _wn, "ask") for _wp, _wn in _ask_walls] +
                [(_wp, _wn, "bid") for _wp, _wn in _bid_walls]
            )
            _wall_scores = {(_wp, _wn, side): _hunt_tier(_wp, _wn) for _wp, _wn, side in _all_walls}
            _max_score = max((s for s, _ in _wall_scores.values()), default=1)

            def _wall_row(wp, wn, side):
                col   = "#3fb950" if side == "bid" else "#f85149"
                arrow = "▼" if side == "bid" else "▲"
                dist  = abs(wp - price) / price * 100
                sc, in_zone = _wall_scores.get((wp, wn, side), (0, False))
                rel = sc / _max_score if _max_score > 0 else 0

                if rel >= 0.85 or (rel >= 0.6 and in_zone):
                    tier_badge = '<span style="background:#f0883e33;color:#f0883e;font-size:9px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px;letter-spacing:.05em;">HIGH HUNT</span>'
                    border     = f"border:1px solid {col}88;box-shadow:0 0 6px {col}44;"
                elif rel >= 0.45 or in_zone:
                    tier_badge = '<span style="background:#ffd70022;color:#ffd700;font-size:9px;padding:1px 6px;border-radius:4px;margin-left:6px;">watch</span>'
                    border     = f"border:1px solid {col}44;"
                else:
                    tier_badge = ""
                    border     = f"border:1px solid {col}22;"

                return (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'background:#161b22;{border}border-radius:6px;'
                    f'padding:6px 10px;margin-bottom:5px;">'
                    f'<span style="color:{col};font-size:13px;font-weight:700;">'
                    f'{arrow} ${wp:,.0f}{tier_badge}</span>'
                    f'<span style="color:#8b949e;font-size:11px;">{dist:.1f}% away</span>'
                    f'<span style="color:{col};font-size:13px;font-weight:600;">{_fusd(wn)}</span>'
                    f'</div>'
                )

            _wall_html = ""
            # Asks first (above price), bids below — mirrors how a real order book looks
            for _wp, _wn in reversed(_ask_walls):
                _wall_html += _wall_row(_wp, _wn, "ask")
            _wall_html += (
                f'<div style="text-align:center;font-size:10px;color:#484f58;'
                f'margin:3px 0;letter-spacing:.05em;">── ${price:,.0f} current ──</div>'
            )
            for _wp, _wn in _bid_walls:
                _wall_html += _wall_row(_wp, _wn, "bid")

            if not _bid_walls and not _ask_walls:
                _wall_html = '<div style="color:#484f58;font-size:12px;">No significant walls detected</div>'

            st.markdown(_wall_html, unsafe_allow_html=True)

        # ── Right: Hunt zones ─────────────────────────────────────────────────
        with _hz_col:
            st.markdown(
                '<div style="font-size:11px;font-weight:600;color:#8b949e;'
                'letter-spacing:.08em;margin-bottom:8px;">HUNT ZONES</div>',
                unsafe_allow_html=True,
            )
            _hz_list = liq_a.get("hunt_zones", [])[:6]

            if _hz_list:
                _hz_html = ""
                for _z in _hz_list:
                    _zp    = _z["price"]
                    _zn    = _z.get("notional", _z.get("wall", 0))
                    _zside = _z["side"]
                    _zdist = _z.get("dist_pct", abs(_zp - price) / price) * 100
                    _zchain= _z.get("cascade_chain", 0)
                    _col   = "#3fb950" if _zside == "BID" else "#f85149"
                    _dir   = "Hunt ↓" if _zside == "BID" else "Hunt ↑"
                    _chain_badge = (
                        f'<span style="background:#30363d;color:#8b949e;'
                        f'font-size:9px;padding:1px 5px;border-radius:4px;margin-left:4px;">'
                        f'chain {_zchain}</span>'
                        if _zchain > 0 else ""
                    )
                    _hz_html += (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'background:#161b22;border-left:3px solid {_col};border-radius:0 6px 6px 0;'
                        f'padding:5px 10px;margin-bottom:5px;">'
                        f'<span style="color:{_col};font-size:12px;font-weight:700;">'
                        f'${_zp:,.0f}</span>'
                        f'<span style="color:#8b949e;font-size:11px;">{_zdist:.1f}% &nbsp;{_dir}{_chain_badge}</span>'
                        f'<span style="color:#cdd9e5;font-size:12px;font-weight:600;">{_fusd(_zn)}</span>'
                        f'</div>'
                    )
                st.markdown(_hz_html, unsafe_allow_html=True)
            else:
                st.caption("No hunt zones detected in current range.")


# ── Tab 5: Advanced Indicators ───────────────────────────────
with tab5:
    pdf = a["plot_df"]

    # ── RSI + MACD ─────────────────────────────────────────
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.markdown("##### RSI (14)")
        st.caption("Relative Strength Index. >70 = overbought, <30 = oversold. Dashed line = EMA-9 signal of RSI.")
        f_rsi = fig_rsi(a)
        st.pyplot(f_rsi, use_container_width=True)
        plt.close(f_rsi)
    with col_macd:
        st.markdown("##### MACD (12, 26, 9)")
        st.caption("Histogram bars show momentum. Blue = MACD line, orange = signal line. Bar above zero & growing = acceleration up.")
        f_macd = fig_macd(a)
        st.pyplot(f_macd, use_container_width=True)
        plt.close(f_macd)

    st.divider()

    # ── Quick metrics row ──────────────────────────────────
    def _sl(col):
        s = pdf[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    bw_v  = _sl("BB_bw");   pctb_v = _sl("BB_pctb")
    k_v   = _sl("Stoch_K"); d_v    = _sl("Stoch_D")
    ac_v  = _sl("AC");      ao_v   = _sl("AO")
    atr_v = _sl("ATR")
    t_v   = _sl("Ichi_Tenkan"); kj_v  = _sl("Ichi_Kijun")
    sa_v  = _sl("Ichi_SpanA");  sb_v  = _sl("Ichi_SpanB")

    bw_mean = float(pdf["BB_bw"].dropna().mean()) if not pdf["BB_bw"].dropna().empty else 1

    am1, am2, am3, am4, am5 = st.columns(5)
    with am1:
        bb_state = ("Squeeze ⚡" if bw_v and bw_v < bw_mean * 0.6 else
                    "Expanded" if bw_v and bw_v > bw_mean * 1.4 else "Normal")
        st.metric("BB Bandwidth", f"{bw_v:.1f}%" if bw_v else "N/A", delta=bb_state)
    with am2:
        stoch_state = ("Overbought" if k_v and k_v > 80 else
                       "Oversold"   if k_v and k_v < 20 else "Neutral")
        st.metric("Stochastic %K", f"{k_v:.0f}" if k_v else "N/A", delta=stoch_state)
    with am3:
        ac_state = "Accel ↑" if ac_v and ac_v > 0 else "Decel ↓"
        st.metric("Accel. Osc.", f"{ac_v:+.1f}" if ac_v else "N/A", delta=ac_state)
    with am4:
        atr_pct = f"{atr_v / price * 100:.1f}% daily" if atr_v else "N/A"
        st.metric("ATR (14)", f"${atr_v:,.0f}" if atr_v else "N/A", delta=atr_pct)
    with am5:
        if sa_v and sb_v:
            ct = max(sa_v, sb_v); cb = min(sa_v, sb_v)
            ichi_s = ("Above ✅" if price > ct else "Below ❌" if price < cb else "Inside ⚠️")
        else:
            ichi_s = "N/A"
        tk_s = ("↑ bull" if t_v and kj_v and t_v > kj_v else "↓ bear" if t_v and kj_v else "N/A")
        st.metric("Ichimoku", ichi_s, delta=f"TK {tk_s}")

    st.divider()

    # ── Ichimoku Cloud ─────────────────────────────────────
    st.markdown("##### Ichimoku Cloud")
    with st.expander("How to read this chart", expanded=False):
        st.markdown(
            """
**The cloud (Kumo)** — the shaded region.
- **Green cloud** = bullish regime · **Red cloud** = bearish regime.
- **Price above the cloud** → uptrend, cloud acts as support.
- **Price below the cloud** → downtrend, cloud acts as resistance.
- **Price inside the cloud** → indecision / no-trade zone.
- **Thick cloud** = strong S/R · **Thin cloud** = weak / about to flip.

**Tenkan (red, 9-period)** — fast trend line. Reacts quickly to price changes.
**Kijun (blue, 26-period)** — slow trend line. Major support/resistance level.
- **Tenkan crosses ABOVE Kijun (TK Cross ↑)** → bullish momentum.
- **Tenkan crosses BELOW Kijun (TK Cross ↓)** → bearish momentum.
- Strongest signals happen when the cross occurs **above the cloud** (bull) or **below** (bear).

**Chikou (gold dashed, lagging line)** — current close shifted 26 bars back.
- **Chikou above price 26 bars ago** → bullish confirmation.
- **Chikou below price 26 bars ago** → bearish confirmation.
- **Chikou tangled with price** → consolidation, weak signal.

**The strongest setups** stack all three:
1. Price on the correct side of the cloud
2. TK cross in the same direction
3. Chikou confirming above/below historical price
            """
        )
    f_ichi = fig_ichi_expanded(a)
    st.pyplot(f_ichi, use_container_width=True)
    plt.close(f_ichi)

    st.divider()

    # ── Bollinger Bands + Stochastic side by side ──────────
    col_bb, col_stoch = st.columns(2)
    with col_bb:
        st.markdown("##### Bollinger Bands — %B & Bandwidth")
        st.caption("**%B** shows where price sits within the band. **Bandwidth** detects squeezes — low bandwidth often precedes a sharp breakout.")
        f_bb = fig_bb_expanded(a)
        st.pyplot(f_bb, use_container_width=True)
        plt.close(f_bb)
        if pctb_v is not None and bw_v is not None:
            _squeeze = bw_v < bw_mean * 0.6
            _expand  = bw_v > bw_mean * 1.4
            if pctb_v > 1.0:
                _pos_msg = "🔴 **Price above upper band** — statistically extended; often reverts or signals strong breakout momentum."
            elif pctb_v > 0.8:
                _pos_msg = "🟠 **Price near upper band** — overbought territory; watch for rejection or continuation with volume."
            elif pctb_v < 0.0:
                _pos_msg = "🟢 **Price below lower band** — statistically oversold; often reverts or signals strong breakdown momentum."
            elif pctb_v < 0.2:
                _pos_msg = "🟡 **Price near lower band** — oversold territory; watch for bounce or continuation with volume."
            else:
                _pos_msg = f"⚪ **Price mid-band** (%B {pctb_v:.2f}) — no directional edge from band position alone."
            if _squeeze:
                _bw_msg = f"⚡ **Bandwidth squeeze** ({bw_v:.1f}% vs mean {bw_mean:.1f}%) — volatility historically low; explosive move (either direction) often follows."
            elif _expand:
                _bw_msg = f"📈 **Bandwidth expanding** ({bw_v:.1f}%) — volatility in expansion phase; trend likely in progress."
            else:
                _bw_msg = f"➡️ **Bandwidth normal** ({bw_v:.1f}%) — no squeeze or expansion signal."
            st.markdown(_pos_msg)
            st.markdown(_bw_msg)
    with col_stoch:
        st.markdown("##### Stochastic Oscillator (14, 3)")
        st.caption("**%K** vs **%D** crossovers in the overbought (>80) or oversold (<20) zones are the primary signals.")
        f_st = fig_stoch_expanded(a)
        st.pyplot(f_st, use_container_width=True)
        plt.close(f_st)

    st.divider()

    # ── Fibonacci + AC/ATR side by side ───────────────────
    col_fib, col_ac = st.columns([3, 2])
    with col_fib:
        st.markdown("##### Fibonacci Retracement")
        st.caption("Auto-detected swing high/low over the last 6 months. Key levels: **38.2%**, **50%**, **61.8%** act as strongest S/R zones.")
        f_fib = fig_fib_expanded(a)
        st.pyplot(f_fib, use_container_width=True)
        plt.close(f_fib)
    with col_ac:
        st.markdown("##### Accelerator Osc. + ATR")
        st.caption("**AC** catches momentum shifts before price moves. **ATR** quantifies daily volatility — rising ATR = expanding moves.")
        f_ac = fig_ac_atr_expanded(a)
        st.pyplot(f_ac, use_container_width=True)
        plt.close(f_ac)

