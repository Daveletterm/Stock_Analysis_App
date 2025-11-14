"""Market data helper that favors Alpaca and falls back to yfinance."""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd
import requests
import yfinance as yf

from paper_trading import AlpacaPaperBroker

try:  # pragma: no cover - optional import varies by yfinance version
    from yfinance.shared.exceptions import YFRateLimitError  # type: ignore
except Exception:  # pragma: no cover - fallback when module layout changes
    YFRateLimitError = ()  # type: ignore

logger = logging.getLogger("market_data")

ALPACA_DATA_BASE_URL = (
    os.getenv("ALPACA_DATA_BASE_URL")
    or os.getenv("ALPACA_MARKET_DATA_URL")
    or "https://data.alpaca.markets/v2"
).rstrip("/")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

_broker = AlpacaPaperBroker()
_data_session = requests.Session()
_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
_cache_lock = threading.Lock()


class PriceDataError(RuntimeError):
    """Raised when price history cannot be retrieved from upstream providers."""


def _isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _alpaca_timeframe(interval: str) -> str:
    mapping = {
        "1m": "1Min",
        "5m": "5Min",
        "15m": "15Min",
        "30m": "30Min",
        "1h": "1Hour",
        "1d": "1Day",
        "1wk": "1Week",
        "1mo": "1Month",
    }
    return mapping.get(interval, "1Day")


def _canonical_column(name: str) -> str:
    return name.replace(" ", "").replace("_", "").lower()


def _ensure_series(obj, symbol: str) -> pd.Series:
    """Return a Series from pandas objects, preferring columns matching symbol."""
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return pd.Series(dtype=float, index=obj.index)
        sym_upper = symbol.upper()
        for col in obj.columns:
            if str(col).upper() == sym_upper:
                return obj[col]
        return obj.iloc[:, 0]
    return pd.Series(obj)


def _flatten_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    sym_upper = symbol.upper()
    for level in range(df.columns.nlevels - 1, -1, -1):
        labels = [str(v).upper() for v in df.columns.get_level_values(level)]
        if sym_upper in labels:
            try:
                return df.xs(sym_upper, axis=1, level=level)
            except KeyError:
                try:
                    return df.xs(symbol, axis=1, level=level)
                except KeyError:
                    continue

    flattened = [
        "_".join(str(part) for part in col if part not in (None, ""))
        for col in df.columns.to_flat_index()
    ]
    df = df.copy()
    df.columns = flattened
    return df


def _normalize_ohlcv_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = _flatten_columns(df, symbol)
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    normalized = pd.DataFrame(index=pd.to_datetime(df.index, utc=True))
    name_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }

    reverse_lookup = {}
    for column in df.columns:
        reverse_lookup.setdefault(_canonical_column(column), column)

    for canonical, final_name in name_map.items():
        source = reverse_lookup.get(canonical)
        if not source:
            continue
        series = _ensure_series(df[source], symbol)
        normalized[final_name] = pd.to_numeric(series, errors="coerce")

    if "Adj Close" not in normalized and "Close" in normalized:
        normalized["Adj Close"] = normalized["Close"]

    ordered_cols = [
        col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if col in normalized.columns
    ]
    normalized = normalized[ordered_cols]
    normalized = normalized.dropna(how="all")
    return normalized.sort_index()


def _fetch_alpaca_history(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Retrieve bars from Alpaca's data API."""
    if not _broker.enabled:
        raise PriceDataError("Alpaca credentials are not configured")

    headers = _broker._headers()  # type: ignore[attr-defined]
    params = {
        "start": _isoformat(start),
        "end": _isoformat(end),
        "timeframe": _alpaca_timeframe(interval),
        "adjustment": "all",
        "feed": ALPACA_DATA_FEED,
        "limit": 10000,
    }
    url = f"{ALPACA_DATA_BASE_URL}/stocks/{symbol}/bars"

    try:
        response = _data_session.get(url, headers=headers, params=params, timeout=20)
    except Exception as exc:
        raise PriceDataError(f"Alpaca request failed: {exc}") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:500].strip()
        logger.warning(
            "Alpaca response error %s for %s: %s", response.status_code, symbol, body
        )
        raise PriceDataError(
            f"Alpaca request failed: {response.status_code} {body}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise PriceDataError("Alpaca returned invalid JSON") from exc

    bars = payload.get("bars") if isinstance(payload, dict) else None
    if not bars:
        raise PriceDataError("Alpaca returned no data")

    records = []
    for bar in bars:
        try:
            ts = pd.to_datetime(bar.get("t"))
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            records.append(
                {
                    "timestamp": ts,
                    "Open": float(bar.get("o", 0.0)),
                    "High": float(bar.get("h", 0.0)),
                    "Low": float(bar.get("l", 0.0)),
                    "Close": float(bar.get("c", 0.0)),
                    "Volume": float(bar.get("v", 0.0)),
                }
            )
        except Exception:
            continue

    if not records:
        raise PriceDataError("Alpaca returned malformed data")

    df = pd.DataFrame.from_records(records).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = _normalize_ohlcv_frame(df, symbol)
    if df is None or df.empty:
        raise PriceDataError("Alpaca returned no usable data")
    return df


def _fetch_yfinance_history(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Retrieve bars from yfinance as a fallback."""
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
    except Exception as exc:
        message = str(exc)
        if "rate limit" in message.lower() or isinstance(exc, YFRateLimitError):
            raise PriceDataError(f"Yahoo Finance rate limit reached for {symbol}") from exc
        raise PriceDataError(f"Yahoo Finance error: {exc}") from exc

    if df is None or df.empty:
        raise PriceDataError("Yahoo Finance returned no data")

    df = _normalize_ohlcv_frame(df, symbol)
    if df is None or df.empty:
        raise PriceDataError("Yahoo Finance returned no usable data")
    return df


def get_price_history(symbol: str, start: datetime, end: datetime, interval: str = "1d") -> pd.DataFrame:
    """
    Return OHLCV history between start/end using Alpaca first, yfinance second.

    Results are cached by (symbol, interval, start_iso, end_iso) so repeated
    requests inside a single process avoid additional API calls.
    """

    start_iso = _isoformat(start)
    end_iso = _isoformat(end)
    cache_key = (symbol.upper(), interval, start_iso, end_iso)

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached.copy()

    try:
        df = _fetch_alpaca_history(symbol, start, end, interval)
    except PriceDataError as alpaca_err:
        logger.warning("Alpaca history failed for %s: %s", symbol, alpaca_err)
        try:
            df = _fetch_yfinance_history(symbol, start, end, interval)
        except PriceDataError as yf_err:
            raise PriceDataError(f"All data sources failed for {symbol}: {yf_err}") from yf_err
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Unexpected Alpaca error for %s", symbol)
        try:
            df = _fetch_yfinance_history(symbol, start, end, interval)
        except PriceDataError as yf_err:
            raise PriceDataError(f"All data sources failed for {symbol}: {yf_err}") from yf_err
    if df is None or df.empty:
        raise PriceDataError(f"No price data for {symbol}")

    df = df.sort_index()

    with _cache_lock:
        _cache[cache_key] = df.copy()

    return df.copy()


__all__ = ["get_price_history", "PriceDataError"]
