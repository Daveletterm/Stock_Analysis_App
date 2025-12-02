"""Market data helper that favors Alpaca and falls back to yfinance."""
from __future__ import annotations

import logging
import os
import threading
import time
from urllib.parse import urlparse, urlunparse
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from paper_trading import AlpacaPaperBroker

try:  # pragma: no cover - optional import varies by yfinance version
    from yfinance.shared.exceptions import YFRateLimitError  # type: ignore
except Exception:  # pragma: no cover - fallback when module layout changes
    YFRateLimitError = ()  # type: ignore

logger = logging.getLogger("market_data")

# Option liquidity and pricing guards
MIN_OPTION_PRICE = 0.20  # minimum acceptable option price in dollars
MIN_OPTION_BID = 0.15  # minimum bid required
MAX_OPTION_SPREAD_PCT = 0.60  # max (ask - bid) / ask for normal pass
MIN_OPTION_OI = 10  # minimum open interest for preferred contracts
MIN_OPTION_VOLUME = 5  # minimum volume for preferred contracts
MIN_OPTION_DTE = 10  # minimum days to expiration
MAX_OPTION_DTE = 90  # maximum days to expiration


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

_ALPACA_DATA_DEFAULT_BASE_URL = "https://paper-api.alpaca.markets/v2"
_ALPACA_STOCK_DATA_DEFAULT_BASE_URL = "https://data.alpaca.markets/v2"
ALPACA_DATA_BASE_URL = (
    os.getenv("ALPACA_DATA_BASE_URL")
    or _ALPACA_STOCK_DATA_DEFAULT_BASE_URL
).rstrip("/")
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

_ALPACA_TRADING_DEFAULT_BASE_URL = (
    os.getenv("APCA_API_BASE_URL")
    or "https://paper-api.alpaca.markets/v2"
).rstrip("/")


def _ensure_options_contracts_path(url: str | None) -> str:
    """Ensure *url* targets Alpaca's /v2/options/contracts endpoint."""

    candidate = (url or _ALPACA_DATA_DEFAULT_BASE_URL).strip()
    if not candidate:
        candidate = _ALPACA_DATA_DEFAULT_BASE_URL

    # Make sure we have a scheme and netloc
    if "://" not in candidate:
        candidate = f"https://{candidate.lstrip('/')}"
    parsed = urlparse(candidate)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="https")
    if not parsed.netloc:
        parsed = urlparse(f"https://{candidate.lstrip('/')}")

    path = parsed.path.rstrip("/")
    if not path or path == "/":
        path = "/v2"

    # If there is already an /options segment, make sure it ends with /contracts
    if "/options" in path:
        if not path.endswith("/contracts"):
            path = f"{path}/contracts"
    else:
        prefix = path if path and path != "/" else "/v2"
        versioned_prefix = prefix.rstrip("/") or "/v2"
        path = f"{versioned_prefix}/options/contracts"

    normalized = urlunparse(parsed._replace(path=path.rstrip("/")))
    return normalized.rstrip("/")


def _resolve_alpaca_options_base_url() -> str:
    """Return the fully-qualified Alpaca options contracts endpoint."""

    env_options_url = (os.getenv("ALPACA_OPTIONS_DATA_URL") or "").strip()
    base_candidate = env_options_url or (
        os.getenv("ALPACA_DATA_BASE_URL")
        or os.getenv("ALPACA_MARKET_DATA_URL")
        or _ALPACA_DATA_DEFAULT_BASE_URL
    )
    resolved = _ensure_options_contracts_path(base_candidate)
    return resolved or _DEFAULT_ALPACA_OPTIONS_BASE_URL


_DEFAULT_ALPACA_OPTIONS_BASE_URL = "https://paper-api.alpaca.markets/v2/options/contracts"
ALPACA_OPTIONS_BASE_URL = _resolve_alpaca_options_base_url()
logger.debug("Resolved Alpaca options base URL: %s", ALPACA_OPTIONS_BASE_URL)

_broker = AlpacaPaperBroker()
_data_session = requests.Session()
_cache: Dict[Tuple[str, str, str, str], pd.DataFrame] = {}
_cache_lock = threading.Lock()

try:
    _ALPACA_MIN_REQUEST_INTERVAL = max(
        0.0, float(os.getenv("ALPACA_MIN_REQUEST_INTERVAL", "0.35"))
    )
except Exception:  # pragma: no cover - defensive default
    _ALPACA_MIN_REQUEST_INTERVAL = 0.35

try:
    _YFINANCE_MIN_REQUEST_INTERVAL = max(
        0.0, float(os.getenv("YFINANCE_MIN_REQUEST_INTERVAL", "0.6"))
    )
except Exception:  # pragma: no cover - defensive default
    _YFINANCE_MIN_REQUEST_INTERVAL = 0.6

_alpaca_rate_lock = threading.Lock()
_alpaca_last_request = 0.0
_yfinance_rate_lock = threading.Lock()
_yfinance_last_request = 0.0


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


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _respect_alpaca_rate_limit() -> None:
    """Ensure Alpaca requests observe the configured minimum spacing."""

    if _ALPACA_MIN_REQUEST_INTERVAL <= 0:
        return

    global _alpaca_last_request
    with _alpaca_rate_lock:
        now = time.monotonic()
        wait = _ALPACA_MIN_REQUEST_INTERVAL - (now - _alpaca_last_request)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _alpaca_last_request = now


def _respect_yfinance_rate_limit() -> None:
    """Throttle yfinance downloads to reduce HTTP 429 responses."""

    if _YFINANCE_MIN_REQUEST_INTERVAL <= 0:
        return

    global _yfinance_last_request
    with _yfinance_rate_lock:
        now = time.monotonic()
        wait = _YFINANCE_MIN_REQUEST_INTERVAL - (now - _yfinance_last_request)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _yfinance_last_request = now


def _alpaca_request(
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, object],
    timeout: float,
    resource: str,
    max_retries: int = 3,
) -> requests.Response:
    """Perform a throttled Alpaca GET request with 429 handling."""

    last_response: requests.Response | None = None
    for attempt in range(max_retries):
        _respect_alpaca_rate_limit()
        try:
            response = _data_session.get(url, headers=headers, params=params, timeout=timeout)
        except Exception as exc:  # pragma: no cover - network failure
            raise PriceDataError(f"Alpaca request failed: {exc}") from exc

        if response.status_code != 429:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                body = response.text[:500].strip()
                logger.warning(
                    "Alpaca response error %s for %s: %s",
                    response.status_code,
                    resource,
                    body,
                )
                raise PriceDataError(
                    f"Alpaca request failed: {response.status_code} {body}"
                ) from exc
            return response

        body = response.text[:200].strip()
        logger.warning(
            "Alpaca rate limit hit for %s (attempt %d/%d): %s",
            resource,
            attempt + 1,
            max_retries,
            body or "429 Too Many Requests",
        )
        retry_after = _safe_float(response.headers.get("Retry-After"), None)
        delay = retry_after if retry_after and retry_after > 0 else 1.5 * (attempt + 1)
        time.sleep(min(delay, 10.0))
        last_response = response

    if last_response is not None:
        body = last_response.text[:500].strip()
        raise PriceDataError(f"Alpaca request failed: 429 {body}")

    raise PriceDataError("Alpaca request failed: rate limited")


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

    response = _alpaca_request(
        url,
        headers=headers,
        params=params,
        timeout=20,
        resource=f"{symbol} bars",
    )

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


def fetch_option_contracts(
    symbol: str,
    *,
    expiration_date_from: date | None = None,
    expiration_date_to: date | None = None,
    option_type: str | None = None,
    limit: int = 500,
) -> List[dict]:
    """Retrieve a slice of the Alpaca option chain for *symbol* using /v2/options/contracts."""

    if not _broker.enabled:
        raise PriceDataError("Alpaca credentials are not configured")

    headers = _broker._headers()  # type: ignore[attr-defined]

    # Build query parameters for the contracts endpoint
    params: Dict[str, object] = {
        # Alpaca's /v2/options/contracts limit is capped; clamp for safety
        "limit": max(1, min(limit, 1000)),
        # Primary, documented parameter for underlying
        "underlying_symbols": symbol.upper(),
        # Avoid halted/expired contracts unless explicitly requested
        "status": "active",
    }

    # Optional filters supported by the contracts endpoint
    if expiration_date_from:
        params["expiration_date_gte"] = expiration_date_from.isoformat()
    if expiration_date_to:
        params["expiration_date_lte"] = expiration_date_to.isoformat()
    if option_type:
        # "call" or "put"
        params["type"] = option_type.lower()

    # ALPACA_OPTIONS_BASE_URL should normally be:
    #   https://paper-api.alpaca.markets/v2/options/contracts
    # Use it as-is and do NOT append any /options/chain style paths.
    base_url = ALPACA_OPTIONS_BASE_URL.rstrip("/")

    try:
        response = _alpaca_request(
            base_url,
            headers=headers,
            params=params,
            timeout=20,
            resource=f"{symbol.upper()} option contracts",
        )
    except PriceDataError as exc:
        # Surface a clear error if the contracts endpoint itself fails
        message = f"Alpaca option contracts endpoint failed for {symbol.upper()}: {exc}"
        logger.warning(message)
        logger.debug(
            "Option contracts request debug -- url=%s params=%s",
            base_url,
            params,
        )
        raise PriceDataError(message) from exc

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected response
        raise PriceDataError("Alpaca returned invalid JSON for option contracts") from exc

    options: List[dict] = []

    if isinstance(payload, dict):
        # New, documented key
        if isinstance(payload.get("option_contracts"), list):
            options = payload["option_contracts"]
        # Backwards-compatibility fallbacks, just in case
        elif isinstance(payload.get("options"), list):
            options = payload["options"]
        elif isinstance(payload.get("result"), list):
            options = payload["result"]
        elif isinstance(payload.get("data"), dict):
            data_obj = payload.get("data", {})
            if isinstance(data_obj, dict):
                for key in ("option_contracts", "options", "result", "contracts"):
                    candidate = data_obj.get(key)
                    if isinstance(candidate, list):
                        options = candidate
                        break

    if not isinstance(options, list):
        return []

    normalized: List[dict] = []
    priced_contracts = 0
    for contract in options:
        if not isinstance(contract, dict):
            continue

        last = _safe_float(
            contract.get("last_price") or contract.get("last_trade_price")
        )
        bid = _safe_float(contract.get("bid_price") or contract.get("bid"))
        ask = _safe_float(contract.get("ask_price") or contract.get("ask"))
        mark = _safe_float(contract.get("mark_price") or contract.get("mark"))
        close = _safe_float(contract.get("close_price") or contract.get("close"))

        price: float | None = None
        if last is not None and last > 0:
            price = last
        elif bid is not None and bid > 0 and ask is not None and ask > 0:
            price = (bid + ask) / 2.0
        elif mark is not None and mark > 0:
            price = mark
        elif close is not None and close > 0:
            price = close

        if price is not None:
            contract["price"] = price
            if mark is None:
                contract.setdefault("mark_price", price)
            priced_contracts += 1

        normalized.append(contract)

    logger.info(
        "Fetched %d option contracts for %s via %s (%d priced)",
        len(normalized),
        symbol.upper(),
        response.url,
        priced_contracts,
    )

    return normalized


def _latest_spot_price(symbol: str, as_of: datetime) -> Optional[float]:
    """Return the most recent close for *symbol* before *as_of*.

    A small helper to avoid circular imports when option sizing needs
    the underlying's current price.
    """

    start = as_of - timedelta(days=7)
    end = as_of + timedelta(days=1)
    try:
        hist = get_price_history(symbol, start, end, interval="1d")
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    price = _safe_float(hist["Close"].iloc[-1])
    return price if price and price > 0 else None


def _derive_option_price(contract: dict, *, bid: float | None = None, ask: float | None = None) -> float | None:
    last = _safe_float(contract.get("last_price") or contract.get("last_trade_price"))
    mark = _safe_float(contract.get("mark_price") or contract.get("mark"))
    local_bid = bid if bid is not None else _safe_float(contract.get("bid_price") or contract.get("bid"))
    local_ask = ask if ask is not None else _safe_float(contract.get("ask_price") or contract.get("ask"))

    price: float | None = None
    if last is not None and last > 0:
        price = last
    elif local_bid is not None and local_bid > 0 and local_ask is not None and local_ask > 0:
        price = (local_bid + local_ask) / 2.0
    elif mark is not None and mark > 0:
        price = mark

    if price is not None and price > 0:
        contract["price"] = price
    else:
        contract.pop("price", None)
    return contract.get("price")


def _choose_option_contract(symbol: str, now: datetime, option_type: str) -> Optional[dict]:
    """Internal helper to select a reasonably liquid contract near the money."""

    kind = option_type.lower()
    logger.info("Choosing %s contract for %s", kind, symbol.upper())

    spot = _latest_spot_price(symbol, now)
    if not spot:
        logger.warning(
            "Skipping %s selection for %s: no recent price", kind, symbol.upper()
        )
        return None

    min_expiry = now.date() + timedelta(days=MIN_OPTION_DTE)
    max_expiry = now.date() + timedelta(days=MAX_OPTION_DTE)

    try:
        chain = fetch_option_contracts(
            symbol,
            expiration_date_from=min_expiry,
            expiration_date_to=max_expiry,
            option_type=kind,
            limit=500,
        )
    except Exception as exc:
        logger.warning("%s chain fetch failed for %s: %s", kind.title(), symbol.upper(), exc)
        return None

    if not chain:
        logger.info("No %s contracts available for %s", kind, symbol.upper())
        return None

    best: dict | None = None
    best_sort: tuple[int, float] | None = None
    total = len(chain)
    rejected = 0
    for contract in chain:
        try:
            status = str(contract.get("status", "")).lower()
            if status and status != "active":
                rejected += 1
                continue

            raw_exp = contract.get("expiration_date") or contract.get("expiry")
            expiration = None
            if isinstance(raw_exp, str):
                try:
                    expiration = datetime.fromisoformat(raw_exp).date()
                except ValueError:
                    expiration = None
            elif isinstance(raw_exp, date):
                expiration = raw_exp
            if not expiration:
                rejected += 1
                continue
            days_out = (expiration - now.date()).days
            if days_out < MIN_OPTION_DTE or days_out > MAX_OPTION_DTE:
                rejected += 1
                continue
            if expiration < min_expiry or expiration > max_expiry:
                rejected += 1
                continue

            strike = _safe_float(contract.get("strike_price") or contract.get("strike"))
            if not strike:
                rejected += 1
                continue
            if strike < spot * 0.75 or strike > spot * 1.25:
                rejected += 1
                continue

            bid = _safe_float(contract.get("bid_price") or contract.get("bid"))
            ask = _safe_float(contract.get("ask_price") or contract.get("ask"))

            if bid is None or ask is None or bid <= 0 or ask <= 0:
                rejected += 1
                continue
            if bid < MIN_OPTION_BID:
                rejected += 1
                continue

            spread_pct = (ask - bid) / ask if ask else 1.0
            if spread_pct > MAX_OPTION_SPREAD_PCT:
                rejected += 1
                continue

            open_int = _safe_float(contract.get("open_interest"))
            volume = _safe_float(contract.get("volume"))
            if open_int is not None and volume is not None and open_int == 0 and volume == 0:
                rejected += 1
                continue
            if (open_int or 0) < MIN_OPTION_OI and (volume or 0) < MIN_OPTION_VOLUME:
                # Prefer contracts with some participation but do not hard fail unless both zero
                pass

            price = _derive_option_price(contract, bid=bid, ask=ask)
            if price is None or price <= 0 or price < MIN_OPTION_PRICE:
                rejected += 1
                continue

            mid = (bid + ask) / 2.0 if bid and ask else price
            option_symbol = str(contract.get("symbol", "")).strip()
            if not option_symbol:
                rejected += 1
                continue

            sort_key = (days_out, abs(strike - spot))
            if best_sort is None or sort_key < best_sort:
                best_sort = sort_key
                best = {
                    "underlying": symbol.upper(),
                    "option_symbol": option_symbol,
                    "strike": strike,
                    "expiration": expiration,
                    "bid": bid,
                    "ask": ask,
                    "last": _safe_float(contract.get("last_price") or contract.get("last_trade_price")),
                    "mid": mid,
                    "price": price,
                }
        except Exception:
            rejected += 1
            continue

    logger.info(
        "Option chain filtered %d/%d %s contracts for %s due to quote/liquidity rules",
        rejected,
        total,
        kind,
        symbol.upper(),
    )

    if not best:
        logger.info(
            "No suitable %s contract for %s (filtered out by liquidity or strike rules)",
            kind,
            symbol.upper(),
        )
        return None

    logger.info(
        "Selected %s for %s: option_symbol=%s strike=%.2f expiry=%s mid_price=%.2f",
        kind,
        symbol.upper(),
        best["option_symbol"],
        best["strike"],
        best["expiration"],
        best.get("mid", 0.0),
    )
    return best


def choose_call_contract(symbol: str, now: datetime) -> Optional[dict]:
    """Select a reasonably liquid call near the money, skipping illiquid quotes.

    Contracts with very low bids, wide spreads, or weak open interest/volume are
    filtered out before selection.
    """

    return _choose_option_contract(symbol, now, "call")


def choose_put_contract(symbol: str, now: datetime) -> Optional[dict]:
    """Select a reasonably liquid put near the money with liquidity guards.

    Contracts with very low bids, wide spreads, or weak open interest/volume are
    filtered out before selection.
    """

    return _choose_option_contract(symbol, now, "put")

def _fetch_yfinance_history(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    """Retrieve bars from yfinance as a fallback."""
    _respect_yfinance_rate_limit()
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


__all__ = [
    "get_price_history",
    "PriceDataError",
    "fetch_option_contracts",
    "choose_call_contract",
    "choose_put_contract",
]
