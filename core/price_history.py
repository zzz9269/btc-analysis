"""
Historical BTC price lookup at exact timestamps.

Fixes the outcome-resolution flaw where signals ≥72h old were graded against
the *current* price at whatever time the resolver happened to run (could be
75h, 90h+ after entry). Every consumer (app resolver, cron logger, regrade
script) now grades against the price at exactly entry + 72h.

Price definition: the OPEN of the 5-minute candle that contains the target
timestamp — i.e. the traded price within 5 minutes of the exact moment.

Stdlib-only (urllib) so log_signal.py can use it on the slim GitHub Actions
runner without extra deps.

Endpoint fallback chain (first that answers wins):
  1. api.binance.com          — primary
  2. data-api.binance.vision  — Binance's public market-data mirror; NOT
                                geo-blocked (api.binance.com returns HTTP 451
                                from US-hosted servers incl. Streamlit Cloud)
  3. api.bybit.com (v5 spot)  — independent venue; BTCUSDT spot tracks
                                Binance spot within bps at 5m granularity
"""
from __future__ import annotations

import json
import urllib.request
import urllib.parse
from datetime import datetime, timezone

_5M_MS = 5 * 60 * 1000

_BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
]
_BYBIT_HOST = "https://api.bybit.com"


def _http_json(url: str, timeout: int = 10):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-analysis/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _floor_5m_ms(ts_ms: int) -> int:
    return (ts_ms // _5M_MS) * _5M_MS


def _to_utc_ms(ts) -> int:
    """Accept datetime, ISO string, or epoch ms int; return epoch ms (UTC)."""
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)


def _binance_klines_5m(host: str, start_ms: int, end_ms: int) -> "dict[int, float]":
    """{candle_open_ms: open_price} for [start_ms, end_ms]. Paginates at 1000."""
    out: dict[int, float] = {}
    cursor = start_ms
    while cursor <= end_ms:
        url = (f"{host}/api/v3/klines?symbol=BTCUSDT&interval=5m"
               f"&startTime={cursor}&endTime={end_ms}&limit=1000")
        rows = _http_json(url)
        if not rows:
            break
        for k in rows:
            out[int(k[0])] = float(k[1])
        last_open = int(rows[-1][0])
        if last_open + _5M_MS <= cursor:   # no forward progress — bail
            break
        cursor = last_open + _5M_MS
    return out


def _bybit_klines_5m(start_ms: int, end_ms: int) -> "dict[int, float]":
    """Bybit v5 spot klines. Returns newest-first; we normalise to the same
    {open_ms: open_price} mapping as Binance."""
    out: dict[int, float] = {}
    cursor_end = end_ms
    while cursor_end >= start_ms:
        url = (f"{_BYBIT_HOST}/v5/market/kline?category=spot&symbol=BTCUSDT"
               f"&interval=5&start={start_ms}&end={cursor_end}&limit=1000")
        data = _http_json(url)
        rows = (((data or {}).get("result") or {}).get("list")) or []
        if not rows:
            break
        for k in rows:                      # k = [startMs, open, high, low, close, ...]
            out[int(k[0])] = float(k[1])
        oldest = min(int(k[0]) for k in rows)
        if oldest <= start_ms:
            break
        cursor_end = oldest - 1
    return out


def fetch_5m_open_prices(start_ts, end_ts) -> "dict[int, float]":
    """Bulk fetch: {5m-candle open_ms: open_price} covering [start_ts, end_ts].
    Tries each endpoint in the fallback chain; returns {} only if all fail."""
    start_ms = _floor_5m_ms(_to_utc_ms(start_ts))
    end_ms   = _floor_5m_ms(_to_utc_ms(end_ts))
    for host in _BINANCE_HOSTS:
        try:
            prices = _binance_klines_5m(host, start_ms, end_ms)
            if prices:
                return prices
        except Exception:
            continue
    try:
        return _bybit_klines_5m(start_ms, end_ms)
    except Exception:
        return {}


def btc_spot() -> "float | None":
    """Current BTCUSDT spot price, cheap (one ticker call, no engine needed).
    Same endpoint fallback chain as the kline helpers."""
    for host in _BINANCE_HOSTS:
        try:
            data = _http_json(f"{host}/api/v3/ticker/price?symbol=BTCUSDT")
            return float(data["price"])
        except Exception:
            continue
    try:
        data = _http_json(f"{_BYBIT_HOST}/v5/market/tickers?category=spot&symbol=BTCUSDT")
        return float(data["result"]["list"][0]["lastPrice"])
    except Exception:
        return None


def btc_price_at(ts) -> "float | None":
    """BTC price at an exact historical timestamp (datetime or ISO string).
    Returns the open of the 5m candle containing ts, or None if unavailable
    (all endpoints down, or ts is in the future / before listing)."""
    target_ms = _floor_5m_ms(_to_utc_ms(ts))
    prices = fetch_5m_open_prices(target_ms, target_ms + _5M_MS)
    return prices.get(target_ms)


def btc_prices_at_bulk(timestamps) -> "dict":
    """Map each input timestamp (datetime or ISO string) → price at that moment.
    One ranged fetch covering min..max, then an index lookup per timestamp.
    Timestamps whose candle is missing map to None."""
    ts_list = list(timestamps)
    if not ts_list:
        return {}
    ms_list = [_floor_5m_ms(_to_utc_ms(t)) for t in ts_list]
    prices = fetch_5m_open_prices(min(ms_list), max(ms_list) + _5M_MS)
    return {t: prices.get(ms) for t, ms in zip(ts_list, ms_list)}
