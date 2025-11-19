import math
import os
import re
import json
import random
import threading
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO
from typing import Any, Dict, List, Tuple, Optional

import requests

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from paper_trading import AlpacaPaperBroker, NoAvailableBidError, OptionCloseRejectedError
from market_data import PriceDataError
from market_data import fetch_option_contracts
from market_data import get_price_history as load_price_history

load_dotenv()

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "appdata.sqlite3")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stockapp")

try:  # Optional dependency used for some advanced indicators
    import pandas_ta as _pandas_ta  # type: ignore

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    _pandas_ta = None
    logger.info(
        "pandas-ta not installed; advanced technical indicators will be skipped."
    )

# Paper trading configuration
paper_broker = AlpacaPaperBroker()
PAPER_MAX_POSITION_PCT = float(os.getenv("PAPER_MAX_POSITION_PCT", "0.1"))
PAPER_MAX_POSITION_NOTIONAL = float(os.getenv("PAPER_MAX_POSITION_NOTIONAL", "8000"))
PAPER_DEFAULT_STOP_LOSS_PCT = float(os.getenv("PAPER_STOP_LOSS_PCT", "0.05"))
PAPER_DEFAULT_TAKE_PROFIT_PCT = float(os.getenv("PAPER_TAKE_PROFIT_PCT", "0.1"))

# -----------------------------
# Globals & caches
# -----------------------------
_lock = threading.Lock()
_sp500 = {"tickers": [], "updated": datetime.min}
_price_cache: Dict[Tuple[str, str, str, bool], Tuple[datetime, pd.DataFrame]] = {}
PRICE_CACHE_TTL = timedelta(minutes=15)
_recommendations = []
_rec_state = {"refreshing": False, "last_completed": None, "last_error": None}
_background_jobs_started = False
_background_jobs_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")
OPTION_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
OPTION_CONTRACT_MULTIPLIER = int(os.getenv("ALPACA_OPTION_MULTIPLIER", "100")) or 100
FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "UNH",
    "V",
]


@dataclass
class OptionSelection:
    contract: Optional[dict[str, Any]]
    premium: Optional[float]
    meta: Optional[dict[str, Any]]
    diagnostics: Optional[dict[str, Any]] = None


# -----------------------------
# Autopilot configuration
# -----------------------------

AUTOPILOT_STRATEGIES = {
    "conservative": {
        "label": "Conservative Growth",
        "description": "Focus on highest scoring names with tight risk controls and limited exposure.",
        "min_score": 3.6,
        "exit_score": 2.2,
        "max_positions": 4,
        "max_position_pct": 0.08,
        "max_total_allocation": 0.55,
        "min_entry_notional": 300.0,
        "stop_loss_pct": 0.03,
        "take_profit_pct": 0.06,
        "lookback": "1y",
    },
    "balanced": {
        "label": "Balanced Momentum",
        "description": "Blend of momentum and trend with moderate diversification and bracket exits.",
        "min_score": 3.2,
        "exit_score": 2.0,
        "max_positions": 6,
        "max_position_pct": 0.12,
        "max_total_allocation": 0.8,
        "min_entry_notional": 200.0,
        "stop_loss_pct": 0.045,
        "take_profit_pct": 0.12,
        "lookback": "1y",
    },
    "aggressive": {
        "label": "Aggressive Breakouts",
        "description": "Targets early breakouts with wider stops and more simultaneous bets.",
        "min_score": 2.8,
        "exit_score": 1.6,
        "max_positions": 9,
        "max_position_pct": 0.16,
        "max_total_allocation": 1.05,
        "min_entry_notional": 150.0,
        "stop_loss_pct": 0.065,
        "take_profit_pct": 0.2,
        "lookback": "9mo",
    },
    "options_momentum": {
        "label": "Momentum Options",
        "description": (
            "Seek liquid, near-dated contracts on top-scoring names with conservative risk controls."
        ),
        "asset_class": "option",
        "min_score": 3.4,
        "exit_score": 2.4,
        "max_positions": 4,
        "max_position_pct": 0.06,
        "max_total_allocation": 0.4,
        "min_entry_notional": 150.0,
        "options_take_profit_pct": 0.85,
        "options_stop_loss_pct": 0.45,
        "options_expiry_buffer": 5,
        "options_expiry_window": (21, 45),
        "min_open_interest": 75,
        "min_volume": 10,
        "target_delta": 0.4,
        "min_delta": 0.25,
        "max_delta": 0.65,
        "max_spread_pct": 0.45,
        "max_implied_volatility": 3.0,
        "max_premium_pct_of_spot": 0.35,
        "contract_type": "auto",
        "allow_opposite_contract": True,
        "max_contracts_per_trade": 5,
        "lookback": "6mo",
    },
}

AUTOPILOT_RISK_LEVELS = {
    "low": {
        "label": "Low",
        "position_multiplier": 0.65,
        "stop_loss_multiplier": 0.75,
        "take_profit_multiplier": 0.85,
    },
    "medium": {
        "label": "Medium",
        "position_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.0,
    },
    "high": {
        "label": "High",
        "position_multiplier": 1.3,
        "stop_loss_multiplier": 1.35,
        "take_profit_multiplier": 1.2,
    },
}

AUTOPILOT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "autopilot_state.json")

_autopilot_state = {
    "enabled": False,
    "paused": False,
    "strategy": "balanced",
    "risk": "medium",
    "last_run": None,
    "last_actions": [],
    "last_error": None,
}
_autopilot_lock = threading.Lock()
_autopilot_runtime_lock = threading.Lock()
_autopilot_last_run: dict[str, Any] | None = None
_autopilot_uncovered_exits: set[str] = set()


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    try:
        conn = _get_db_connection()
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS recommendation_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS autopilot_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    summary TEXT,
                    error TEXT
                );
                """
            )
    except Exception:
        logger.exception("Failed to initialize local sqlite cache")


def _load_latest_recommendations_from_db() -> None:
    try:
        conn = _get_db_connection()
        cur = conn.execute(
            "SELECT payload, created_at FROM recommendation_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return
        payload = json.loads(row["payload"])
        created_at = row["created_at"]
        with _lock:
            if isinstance(payload, list):
                _recommendations.clear()
                _recommendations.extend(payload)
            _rec_state["last_completed"] = datetime.fromisoformat(created_at)
            _rec_state["last_error"] = None
        logger.info("Loaded cached recommendations snapshot from %s", created_at)
    except Exception:
        logger.exception("Failed to load cached recommendations from sqlite")


def _record_recommendations_snapshot(recs: list[dict[str, Any]]) -> None:
    try:
        conn = _get_db_connection()
        with conn:
            conn.execute(
                "INSERT INTO recommendation_snapshots (created_at, payload) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(to_plain(recs))),
            )
    except Exception:
        logger.exception("Failed to persist recommendation snapshot")


def _load_last_autopilot_run() -> None:
    global _autopilot_last_run
    try:
        conn = _get_db_connection()
        cur = conn.execute(
            "SELECT run_at, success, summary, error FROM autopilot_runs ORDER BY run_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return
        _autopilot_last_run = {
            "run_at": datetime.fromisoformat(row["run_at"]),
            "success": bool(row["success"]),
            "summary": row["summary"],
            "error": row["error"],
        }
        with _autopilot_lock:
            _autopilot_state["last_run"] = _autopilot_last_run["run_at"]
            _autopilot_state["last_actions"] = [row["summary"]] if row["summary"] else []
            _autopilot_state["last_error"] = row["error"]
        logger.info("Loaded last autopilot run from %s", row["run_at"])
    except Exception:
        logger.exception("Failed to load last autopilot run from sqlite")


def _record_autopilot_run(success: bool, summary: str | None, error: str | None) -> None:
    global _autopilot_last_run
    run_at = datetime.now()
    _autopilot_last_run = {
        "run_at": run_at,
        "success": success,
        "summary": summary,
        "error": error,
    }
    try:
        conn = _get_db_connection()
        with conn:
            conn.execute(
                "INSERT INTO autopilot_runs (run_at, success, summary, error) VALUES (?, ?, ?, ?)",
                (run_at.isoformat(), int(success), summary, error),
            )
    except Exception:
        logger.exception("Failed to persist autopilot run")


def _load_autopilot_state() -> None:
    """Load persisted autopilot settings if present."""

    if not os.path.exists(AUTOPILOT_CONFIG_FILE):
        return

    try:
        with open(AUTOPILOT_CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception as exc:  # pragma: no cover - best-effort restore
        logger.warning("Failed to load autopilot config: %s", exc)
        return

    if not isinstance(saved, dict):
        return

    with _autopilot_lock:
        for key in ("enabled", "paused"):
            if key in saved:
                _autopilot_state[key] = bool(saved.get(key))
        for key in ("strategy", "risk"):
            if key in saved and isinstance(saved.get(key), str):
                _autopilot_state[key] = str(saved.get(key))


def _persist_autopilot_state() -> None:
    """Persist the current autopilot settings to disk."""

    with _autopilot_lock:
        payload = {
            "enabled": bool(_autopilot_state.get("enabled")),
            "paused": bool(_autopilot_state.get("paused")),
            "strategy": _autopilot_state.get("strategy"),
            "risk": _autopilot_state.get("risk"),
        }

    try:  # pragma: no cover - minimal persistence wrapper
        with open(AUTOPILOT_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as exc:
        logger.warning("Failed to persist autopilot config: %s", exc)


_init_db()
_load_latest_recommendations_from_db()
_load_last_autopilot_run()
_load_autopilot_state()

# -----------------------------
# Sentiment stubs (no external calls for now)
# -----------------------------

def fetch_news_sentiment(_ticker: str) -> float:
    return 0.0


def fetch_reddit_sentiment(_ticker: str) -> float:
    return 0.0


# -----------------------------
# Helpers
# -----------------------------


def _fallback_recommendations(reason: str | None = None) -> list[dict[str, Any]]:
    """Return placeholder picks so the UI isn't empty when data is unreachable."""

    why_text = reason or "Data unavailable; using placeholder symbols."
    return [
        {
            "Symbol": sym,
            "Recommendation": "HOLD",
            "Score": 0.0,
            "Why": [why_text],
        }
        for sym in FALLBACK_TICKERS[:5]
    ]


def _format_recommendation_status(recs: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    last_completed = state.get("last_completed")
    if isinstance(last_completed, datetime):
        last_completed_str = last_completed.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_completed_str = None

    refreshing = bool(state.get("refreshing"))
    last_error = state.get("last_error")

    if not recs:
        message = "No completed scans yet; first-time refresh can take a few minutes."
    elif refreshing:
        when = f" from {last_completed_str}" if last_completed_str else ""
        message = f"Refreshing in the background; showing last completed scan{when}."
    else:
        when = f" from {last_completed_str}" if last_completed_str else ""
        message = f"Showing latest recommendations{when}."
    if last_error:
        message += f" Last refresh error: {last_error}."

    return {
        "refreshing": refreshing,
        "last_completed": last_completed_str,
        "last_error": last_error,
        "message": message,
    }
# -----------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def last_value(s: pd.Series):
    if s is None or len(s) == 0:
        return None
    idx = s.last_valid_index()
    if idx is None:
        return None
    val = s.loc[idx]
    try:
        return float(val)
    except Exception:
        try:
            return val.item()
        except Exception:
            return None


def _col_series(hist: pd.DataFrame, col: str, ticker: str) -> pd.Series:
    """Return a 1D Series for an OHLC column even if yfinance returns a DataFrame."""
    s = hist[col]
    if isinstance(s, pd.DataFrame):
        if ticker in s.columns:
            s = s[ticker]
        else:
            s = s.iloc[:, 0]
    return s.astype(float)


def to_plain(obj):
    """Recursively convert pandas/numpy objects to plain Python types and stringify dict keys."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, (pd.Series, pd.Index)):
        return to_plain(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_plain(obj.to_dict(orient='records'))
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    if obj is pd.NaT:
        return None
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def parse_option_symbol(contract_symbol: str) -> dict[str, Any] | None:
    """Return metadata about an OCC option symbol (underlying, expiry, strike)."""

    if not contract_symbol:
        return None
    cleaned = contract_symbol.replace(" ", "").upper()
    match = OPTION_SYMBOL_RE.match(cleaned)
    if not match:
        return None
    underlying, yymmdd, cp_flag, strike_str = match.groups()
    try:
        expiration = datetime.strptime(yymmdd, "%y%m%d").date()
    except ValueError:
        return None
    strike_val = safe_float(int(strike_str) / 1000.0, 0.0)
    if strike_val is None:
        strike_val = 0.0
    return {
        "symbol": cleaned,
        "underlying": underlying,
        "expiration": expiration,
        "type": "call" if cp_flag == "C" else "put",
        "strike": float(strike_val),
    }


def option_days_to_expiration(meta: dict[str, Any] | None) -> int | None:
    if not meta or "expiration" not in meta:
        return None
    expiration = meta["expiration"]
    if isinstance(expiration, str):
        try:
            expiration = datetime.strptime(expiration, "%Y-%m-%d").date()
        except ValueError:
            return None
    if isinstance(expiration, datetime):
        expiration = expiration.date()
    if not isinstance(expiration, date):
        return None
    return (expiration - date.today()).days


def option_contract_delta(contract: dict[str, Any]) -> float | None:
    if not isinstance(contract, dict):
        return None
    delta = contract.get("delta")
    if delta is None and isinstance(contract.get("greeks"), dict):
        delta = contract["greeks"].get("delta")
    return safe_float(delta, None)


def _option_quote(contract: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(contract, dict):
        return None
    for key in ("last_quote", "quote", "latest_quote"):
        value = contract.get(key)
        if isinstance(value, dict):
            return value
    return None


def option_bid_ask(contract: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    quote = _option_quote(contract)
    bid: Optional[float] = None
    ask: Optional[float] = None

    def _extract_bid_ask(source: dict[str, Any] | None) -> tuple[Optional[float], Optional[float]]:
        local_bid: Optional[float] = None
        local_ask: Optional[float] = None
        if not source:
            return None, None
        for bid_key in ("bid_price", "bid", "bp", "best_bid_price"):
            local_bid = safe_float(source.get(bid_key), None)
            if local_bid is not None:
                break
        for ask_key in ("ask_price", "ask", "ap", "best_ask_price"):
            local_ask = safe_float(source.get(ask_key), None)
            if local_ask is not None:
                break
        return local_bid, local_ask

    bid, ask = _extract_bid_ask(quote)

    # Some Alpaca contract payloads expose bid/ask at the top level, not just within quotes.
    if bid is None or ask is None:
        top_bid, top_ask = _extract_bid_ask(contract)
        bid = bid if bid is not None else top_bid
        ask = ask if ask is not None else top_ask

    return bid, ask


def _option_last_price(contract: dict[str, Any]) -> float | None:
    last_price = safe_float(contract.get("last_price"), None)
    if last_price is not None:
        return last_price

    trade = None
    for key in ("last_trade", "trade", "latest_trade"):
        value = contract.get(key)
        if isinstance(value, dict):
            trade = value
            break
    if trade:
        for price_key in ("price", "last_price", "p"):
            price_val = trade.get(price_key)
            price = safe_float(price_val, None)
            if price is not None:
                return price
    return None


def option_implied_volatility(contract: dict[str, Any]) -> float | None:
    if not isinstance(contract, dict):
        return None
    iv = contract.get("implied_volatility")
    if iv is None and isinstance(contract.get("greeks"), dict):
        greeks = contract["greeks"]
        iv = greeks.get("iv") or greeks.get("implied_volatility")
    return safe_float(iv, None)


def _derive_option_price(
    contract: dict[str, Any],
    *,
    bid: Optional[float] | None = None,
    ask: Optional[float] | None = None,
) -> tuple[Optional[float], bool]:
    """
    Determine a usable option price from available fields.

    Preference order:
      1) last_price / latest trade price if present and positive
      2) Midpoint of bid/ask when both are present and positive
      3) mark_price / mark if present and positive

    Returns a tuple of (price, has_price_fields) where has_price_fields
    indicates whether the payload contained any price-related fields at all.
    """

    if not isinstance(contract, dict):
        return None, False

    local_bid, local_ask = bid, ask
    if local_bid is None or local_ask is None:
        local_bid, local_ask = option_bid_ask(contract)

    last_price = _option_last_price(contract)
    mark_price = safe_float(contract.get("mark_price", contract.get("mark")), None)

    has_price_fields = any(
        val is not None for val in (last_price, local_bid, local_ask, mark_price)
    )

    price: Optional[float] = None
    if last_price is not None and last_price > 0:
        price = last_price
    elif (
        local_bid is not None
        and local_bid > 0
        and local_ask is not None
        and local_ask > 0
    ):
        price = (local_bid + local_ask) / 2.0
    elif mark_price is not None and mark_price > 0:
        price = mark_price

    if price is not None and price > 0:
        contract["price"] = price
        if mark_price is None:
            contract.setdefault("mark_price", price)
    else:
        contract.pop("price", None)

    return contract.get("price"), has_price_fields


def option_mid_price(contract: dict[str, Any]) -> float | None:
    price, _ = _derive_option_price(contract)
    return price


_PERIOD_RE = re.compile(r"^(\d+)([a-zA-Z]+)$")


def enrich_option_mark_prices(contracts: list[dict[str, Any]]) -> None:
    """
    For each contract, ensure there is a usable price/mark_price.
    Preference order:
      1) last_price / latest trade price if present and positive
      2) Midpoint of bid/ask when both are present and positive
      3) mark_price / mark if present and positive
    Only leave mark_price/price as None if no price information exists at all.
    This function mutates the contracts list in place.
    """

    for contract in contracts:
        if not isinstance(contract, dict):
            continue

        _derive_option_price(contract)


def _period_to_range(period: str, *, interval: str | None = None) -> tuple[datetime, datetime, str]:
    """Translate shorthand period strings into UTC start/end datetimes."""

    interval_out = interval or "1d"
    end = datetime.now(timezone.utc)

    match = _PERIOD_RE.match(period.lower().strip())
    if not match:
        raise ValueError(f"Unsupported period string: {period}")

    amount = int(match.group(1))
    unit = match.group(2)

    if unit in {"d", "day", "days"}:
        start = end - timedelta(days=amount)
    elif unit in {"w", "wk", "week", "weeks"}:
        start = end - timedelta(weeks=amount)
    elif unit in {"mo", "mon", "month", "months"}:
        start = (pd.Timestamp(end) - pd.DateOffset(months=amount)).to_pydatetime()
    elif unit in {"y", "yr", "year", "years"}:
        start = (pd.Timestamp(end) - pd.DateOffset(years=amount)).to_pydatetime()
    else:
        raise ValueError(f"Unsupported period unit: {unit}")

    # Alpaca expects timezone-aware timestamps in UTC.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    else:
        start = start.astimezone(timezone.utc)

    return start, end, interval_out


def _synthetic_price_history(
    ticker: str, start: datetime, end: datetime, interval: str
) -> pd.DataFrame:
    """Generate a deterministic, minimal OHLCV frame when real data is unavailable."""

    freq = {
        "1d": "B",
        "1h": "H",
        "1wk": "W",
        "1mo": "M",
    }.get(interval, "B")

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    index = pd.date_range(start=start, end=end, freq=freq)
    if len(index) < 30:
        index = pd.date_range(end=end, periods=30, freq=freq)

    seed = abs(hash(ticker.upper())) % (2**32)
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0008, 0.0005)
    noise = rng.normal(0, 0.01, size=len(index))
    steps = np.cumsum(drift + noise)
    base = 50 + (seed % 75)
    close = base * (1 + steps).clip(min=0.2)
    high = close * (1 + rng.normal(0.003, 0.004, size=len(index))).clip(min=1.0005)
    low = close * (1 - rng.normal(0.003, 0.004, size=len(index))).clip(min=0.001)
    open_ = close * (1 + rng.normal(0, 0.002, size=len(index)))
    volume = np.abs(rng.normal(2_000_000, 400_000, size=len(index)))

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )

    return df


def get_price_history(
    ticker: str,
    period: str,
    *,
    interval: str | None = None,
    auto_adjust: bool = True,
    max_age: timedelta | None = None,
) -> pd.DataFrame:
    """Load price history using the shared market data helper."""

    _ = auto_adjust  # retained for backward compatibility; handled upstream
    _ = max_age

    start, end, resolved_interval = _period_to_range(period, interval=interval)

    try:
        return load_price_history(ticker, start, end, interval=resolved_interval)
    except PriceDataError as exc:
        logger.warning(
            "Price history unavailable for %s (%s); using synthetic data instead",
            ticker,
            exc,
        )
        return _synthetic_price_history(ticker, start, end, resolved_interval)
    except Exception:
        logger.exception("Unexpected price history failure for %s", ticker)
        return _synthetic_price_history(ticker, start, end, resolved_interval)


@lru_cache(maxsize=256)
def fetch_latest_price(ticker: str) -> float:
    """Fetch the most recent closing price for guardrail calculations."""
    hist = get_price_history(ticker, "5d")
    if hist is None or hist.empty:
        raise ValueError(f"No recent price data for {ticker}")
    price = hist["Close"].iloc[-1]
    return float(price)


def parse_percent(value, default):
    try:
        if value is None or value == "":
            raise ValueError
        val = float(value)
        if val > 1:
            val = val / 100.0
        if val < 0:
            raise ValueError
        return val
    except Exception:
        return default


def _coerce_numeric_series(values, index) -> pd.Series:
    """Return a 1D float Series aligned to *index* regardless of the input shape."""

    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return pd.Series(index=index, dtype=float)
        values = values.iloc[:, 0]
    if isinstance(values, pd.Series):
        series = values.reindex(index)
    else:
        series = pd.Series(values, index=index)
    return pd.to_numeric(series, errors="coerce")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = _coerce_numeric_series(df[col], df.index)

    close_series = df.get("Close")
    if close_series is None:
        raise ValueError("Close column missing for indicator computation")
    close_series = _coerce_numeric_series(close_series, df.index)
    df["Close"] = close_series

    df["SMA_50"] = close_series.rolling(50, min_periods=1).mean()
    df["SMA_200"] = close_series.rolling(200, min_periods=1).mean()
    df["RSI"] = compute_rsi(close_series)  # 14

    atr = compute_atr(df, 14)
    df["ATR14"] = _coerce_numeric_series(atr, df.index)
    df["ATR_pct"] = (df["ATR14"] / close_series).clip(lower=0)
    df["HH_20"] = df["High"].rolling(20, min_periods=1).max().astype(float)
    df["LL_20"] = df["Low"].rolling(20, min_periods=1).min().astype(float)
    df["HH_252"] = df["High"].rolling(252, min_periods=1).max().astype(float)
    df["pct_from_52w_high"] = (df["Close"] / df["HH_252"] - 1.0).replace([np.inf, -np.inf], np.nan)
    if HAS_PANDAS_TA:
        try:
            df["EMA_21"] = _pandas_ta.ema(df["Close"], length=21)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("pandas-ta EMA calculation failed: %s", exc)
            df["EMA_21"] = pd.NA
    else:
        df["EMA_21"] = pd.NA
    return df


def score_stock(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Return (score, reasons) based on last row of indicators."""
    reasons = []
    if df.empty:
        return 0.0, ["insufficient price history"]
    row = df.iloc[-1]
    score = 0.0

    # Trend
    if pd.notna(row["SMA_200"]) and row["Close"] > row["SMA_200"]:
        score += 2.0
        reasons.append("price > 200d MA")
    else:
        reasons.append("below 200d MA")

    # Alignment
    if row["SMA_50"] > row["SMA_200"]:
        score += 1.0
        reasons.append("50d MA > 200d MA")

    # Momentum window
    if 50 <= (row["RSI"] or 0) <= 65:
        score += 1.0
        reasons.append("RSI in 50–65 sweet spot")
    elif 65 < (row["RSI"] or 0) <= 70:
        score += 0.5
        reasons.append("RSI 65–70 (strong)")
    elif (row["RSI"] or 0) < 40:
        score -= 0.5
        reasons.append("RSI < 40 (weak)")

    # Breakout / proximity to highs
    if row["Close"] >= row["HH_20"] * 0.999:  # near 20d breakout
        score += 1.0
        reasons.append("near 20d high breakout")
    if pd.notna(row["pct_from_52w_high"]) and row["pct_from_52w_high"] > -0.05:
        score += 1.0
        reasons.append("within 5% of 52w high")

    # Volatility filter
    if pd.notna(row["ATR_pct"]) and row["ATR_pct"] < 0.04:
        score += 0.5
        reasons.append("low volatility (<4% ATR)")

    return round(float(score), 2), reasons


def place_guarded_paper_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    time_in_force: str = "day",
    *,
    asset_class: str | None = None,
    price_hint: float | None = None,
    support_brackets: bool = True,
    position_effect: str | None = None,
):
    if not paper_broker.enabled:
        raise RuntimeError("Paper trading credentials are not configured")
    symbol = symbol.strip()
    if not str(asset_class or "").lower().startswith("option") and not TICKER_RE.match(symbol.upper()):
        raise ValueError("Enter a valid ticker symbol")
    if qty <= 0:
        raise ValueError("Quantity must be positive")
    if side not in {"buy", "sell"}:
        raise ValueError("Side must be 'buy' or 'sell'")
    if order_type not in {"market", "limit"}:
        raise ValueError("Order type must be market or limit")

    symbol = symbol.upper()
    is_option = str(asset_class or "").lower() == "option"
    multiplier = OPTION_CONTRACT_MULTIPLIER if is_option else 1

    entry_price = None
    if price_hint is not None and price_hint > 0:
        entry_price = float(price_hint)
    elif order_type == "limit":
        if limit_price is None or limit_price <= 0:
            raise ValueError("Provide a positive limit price")
        entry_price = float(limit_price)
    elif not is_option:
        entry_price = fetch_latest_price(symbol)
    else:
        entry_price = 0.0

    notional = float(entry_price) * qty * multiplier

    account = paper_broker.get_account()
    equity = float(account.get("equity") or account.get("cash") or 0.0)
    buying_power = float(account.get("buying_power") or account.get("cash") or 0.0)

    if PAPER_MAX_POSITION_NOTIONAL and notional > PAPER_MAX_POSITION_NOTIONAL:
        raise ValueError(
            f"Order notional ${notional:,.2f} exceeds max allowed ${PAPER_MAX_POSITION_NOTIONAL:,.2f}"
        )
    if PAPER_MAX_POSITION_PCT and equity > 0 and notional > equity * PAPER_MAX_POSITION_PCT:
        allowed = equity * PAPER_MAX_POSITION_PCT
        raise ValueError(
            f"Order notional ${notional:,.2f} exceeds {PAPER_MAX_POSITION_PCT*100:.1f}% of equity (${allowed:,.2f})"
        )
    if side == "buy" and buying_power and notional > buying_power:
        raise ValueError("Not enough buying power for this order")

    payload: dict[str, object] = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if order_type == "limit":
        payload["limit_price"] = round(float(limit_price), 2)
    if asset_class:
        payload["asset_class"] = asset_class
    if position_effect:
        payload["position_effect"] = position_effect

    if side == "buy" and support_brackets and not is_option:
        stop_loss_pct = PAPER_DEFAULT_STOP_LOSS_PCT if stop_loss_pct is None else max(stop_loss_pct, 0.0)
        take_profit_pct = (
            PAPER_DEFAULT_TAKE_PROFIT_PCT if take_profit_pct is None else max(take_profit_pct, 0.0)
        )
        if stop_loss_pct or take_profit_pct:
            payload["order_class"] = "bracket"
            if take_profit_pct:
                payload["take_profit"] = {
                    "limit_price": round(entry_price * (1 + take_profit_pct), 2)
                }
            if stop_loss_pct:
                payload["stop_loss"] = {
                    "stop_price": round(entry_price * (1 - stop_loss_pct), 2)
                }

    return paper_broker.submit_order(payload)


# -----------------------------
# Autopilot routines
# -----------------------------


def get_autopilot_status() -> dict:
    with _autopilot_lock:
        snapshot = dict(_autopilot_state)
    last_run = snapshot.get("last_run")
    if isinstance(last_run, datetime):
        snapshot["last_run"] = last_run.isoformat(timespec="seconds")
    if _autopilot_last_run:
        snapshot["last_run_summary"] = _autopilot_last_run.get("summary")
        snapshot["last_run_error"] = _autopilot_last_run.get("error")
        snapshot["last_run_success"] = bool(_autopilot_last_run.get("success"))
    snapshot.setdefault("strategy", "balanced")
    snapshot.setdefault("risk", "medium")
    snapshot.setdefault("paused", False)
    actions = snapshot.get("last_actions")
    if isinstance(actions, list):
        snapshot["last_actions"] = actions
    elif actions:
        snapshot["last_actions"] = [str(actions)]
    else:
        snapshot["last_actions"] = []
    return snapshot


def update_autopilot_config(
    *, enabled: bool | None = None, paused: bool | None = None, strategy: str | None = None, risk: str | None = None
) -> dict:
    with _autopilot_lock:
        if enabled is not None:
            _autopilot_state["enabled"] = bool(enabled)
            if paused is None:
                _autopilot_state["paused"] = not bool(enabled)
        if paused is not None:
            _autopilot_state["paused"] = bool(paused)
        if strategy and strategy in AUTOPILOT_STRATEGIES:
            _autopilot_state["strategy"] = strategy
        if risk and risk in AUTOPILOT_RISK_LEVELS:
            _autopilot_state["risk"] = risk
    _persist_autopilot_state()
    return get_autopilot_status()


def _strategy_asset_class(strategy: dict[str, Any]) -> str:
    asset_class = str(strategy.get("asset_class", "equity")).lower().strip()
    if asset_class in {"option", "options"}:
        return "option"
    return "equity"


def _autopilot_order_blocked(symbol: str, open_orders: list[dict]) -> bool:
    for order in open_orders:
        try:
            if str(order.get("symbol", "")).upper() != symbol.upper():
                continue
            status = str(order.get("status", "")).lower()
            if status in {"new", "accepted", "open", "pending_new", "partially_filled"}:
                return True
        except Exception:
            continue
    return False


def _option_position_quantity(position: dict[str, Any] | None) -> int:
    if not position:
        return 0
    qty = safe_float(position.get("qty"), safe_float(position.get("quantity"), 0.0))
    return int(abs(qty))


def _find_option_position(symbol: str, positions_snapshot: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = symbol.replace(" ", "").upper()
    for pos in positions_snapshot:
        try:
            if str(pos.get("symbol", "")).replace(" ", "").upper() != target:
                continue
            asset = str(pos.get("asset_class", "")).lower()
            if "option" not in asset:
                continue
            qty = _option_position_quantity(pos)
            if qty <= 0:
                continue
            return pos
        except Exception:
            continue
    return None


def _autopilot_select_option_contract(
    symbol: str,
    strategy: dict[str, Any],
    *,
    underlying_price: float,
    score: float | None = None,
) -> OptionSelection:
    """Pick an option contract that matches the strategy's risk profile."""

    diagnostics: dict[str, Any] = {"symbol": symbol.upper()}

    window = strategy.get("options_expiry_window", (21, 45))
    if (
        not isinstance(window, (list, tuple))
        or len(window) != 2
        or any(v is None for v in window)
    ):
        window = (21, 45)
    try:
        min_days = max(7, int(window[0]))
    except Exception:
        min_days = 21
    try:
        max_days = max(min_days + 1, int(window[1]))
    except Exception:
        max_days = max(min_days + 1, 45)
    diagnostics["expiry_window_days"] = {"min": min_days, "max": max_days}

    expiration_from = date.today() + timedelta(days=min_days)
    expiration_to = date.today() + timedelta(days=max_days)

    contract_pref = str(strategy.get("contract_type", "call")).lower().strip()
    allow_opposite = bool(strategy.get("allow_opposite_contract", False))
    contract_types: list[str] = []
    if contract_pref in {"call", "put"}:
        contract_types.append(contract_pref)
        if allow_opposite:
            alt = "put" if contract_pref == "call" else "call"
            if alt not in contract_types:
                contract_types.append(alt)
    else:
        contract_types = ["call", "put"]
    if score is not None and score < 0 and "put" in contract_types:
        contract_types = ["put"] + [ct for ct in contract_types if ct != "put"]
    diagnostics["contract_types_considered"] = contract_types

    target_delta = abs(safe_float(strategy.get("target_delta"), 0.35)) or 0.35
    min_open_interest_base = max(0, int(safe_float(strategy.get("min_open_interest"), 0)))
    min_volume_base = max(0, int(safe_float(strategy.get("min_volume"), 0)))

    min_delta_cfg = strategy.get("min_delta")
    max_delta_cfg = strategy.get("max_delta")
    min_delta = (
        safe_float(min_delta_cfg, None)
        if min_delta_cfg is not None and min_delta_cfg != ""
        else None
    )
    max_delta = (
        safe_float(max_delta_cfg, None)
        if max_delta_cfg is not None and max_delta_cfg != ""
        else None
    )

    spread_cfg = strategy.get("max_spread_pct")
    max_spread_pct = (
        safe_float(spread_cfg, None)
        if spread_cfg is not None and spread_cfg != ""
        else None
    )
    if not max_spread_pct or max_spread_pct <= 0:
        max_spread_pct = 0.5

    premium_cap_cfg = strategy.get("max_premium_pct_of_spot")
    max_premium_pct = (
        safe_float(premium_cap_cfg, None)
        if premium_cap_cfg is not None and premium_cap_cfg != ""
        else None
    )

    max_iv_cfg = strategy.get("max_implied_volatility")
    max_iv = (
        safe_float(max_iv_cfg, None)
        if max_iv_cfg is not None and max_iv_cfg != ""
        else None
    )

    diagnostics["risk_filters"] = {
        "target_delta": target_delta,
        "min_delta": min_delta,
        "max_delta": max_delta,
        "max_spread_pct": max_spread_pct,
        "max_premium_pct": max_premium_pct,
        "max_implied_vol": max_iv,
        "min_open_interest": min_open_interest_base,
        "min_volume": min_volume_base,
    }

    chains: dict[str, list[dict[str, Any]]] = {}
    chain_sizes: dict[str, int] = {}
    priced_counts: dict[str, int] = {}
    chain_errors: dict[str, str] = {}
    for option_type in contract_types:
        try:
            chain = fetch_option_contracts(
                symbol,
                expiration_date_from=expiration_from,
                expiration_date_to=expiration_to,
                option_type=option_type,
                limit=800,
            )
        except PriceDataError as exc:
            chain = []
            chain_errors[option_type] = str(exc)
        priced = 0
        for contract in chain:
            price, _ = _derive_option_price(contract)
            if price is not None and price > 0:
                priced += 1
        chains[option_type] = chain
        chain_sizes[option_type] = len(chain)
        priced_counts[option_type] = priced
    diagnostics["chain_sizes"] = chain_sizes
    diagnostics["priced_contracts"] = priced_counts
    if chain_errors:
        diagnostics["chain_errors"] = chain_errors

    total_contracts = sum(chain_sizes.values())
    total_priced_contracts = sum(priced_counts.values())
    diagnostics["total_contracts"] = total_contracts
    diagnostics["total_priced_contracts"] = total_priced_contracts
    logger.info(
        "Option chain summary for %s: %d total contracts (%d priced)",
        symbol.upper(),
        total_contracts,
        total_priced_contracts,
    )
    if total_contracts == 0:
        diagnostics["base_rejections"] = {"empty_chain": len(contract_types)}
        return OptionSelection(None, None, None, diagnostics)

    base_rejections: Counter[str] = Counter()
    price_fallbacks: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for option_type, chain in chains.items():
        for contract in chain:
            contract_symbol = str(contract.get("symbol", "")).strip()
            if not contract_symbol:
                base_rejections["missing_symbol"] += 1
                continue
            parsed = parse_option_symbol(contract_symbol)
            if not parsed:
                base_rejections["unparseable_symbol"] += 1
                continue
            days_out = option_days_to_expiration(parsed)
            if days_out is None:
                base_rejections["invalid_expiration"] += 1
                continue
            if days_out < min_days or days_out > max_days:
                base_rejections["expiry_window"] += 1
                continue
            bid, ask = option_bid_ask(contract)
            price, has_price_fields = _derive_option_price(
                contract, bid=bid, ask=ask
            )
            if price is None or price <= 0:
                if not has_price_fields:
                    base_rejections["no_mark_price"] += 1
                else:
                    base_rejections["invalid_price"] += 1
                continue
            if ask is None or ask <= 0:
                # Alpaca's options endpoint frequently omits real-time quotes for
                # thin contracts. When we have a reliable last/mark price,
                # synthesize a conservative ask instead of discarding the
                # contract outright so the strategy can still evaluate it.
                ask = price * 1.01
                price_fallbacks["synthetic_ask_used"] += 1
            if bid is None or bid <= 0:
                # Likewise synthesize a bid slightly below the derived price so
                # spreads remain small and risk filters still apply.
                bid = price * 0.99
                price_fallbacks["synthetic_bid_used"] += 1
            spread_pct = max(0.0, (ask - bid) / ask) if ask else None
            if spread_pct is None:
                base_rejections["invalid_spread"] += 1
                continue
            oi = safe_float(contract.get("open_interest"), None)
            volume = safe_float(contract.get("volume"), None)
            delta = option_contract_delta(contract)
            iv = option_implied_volatility(contract)
            premium_pct = (price / underlying_price) if underlying_price else None
            candidates.append(
                {
                    "contract": contract,
                    "symbol": contract_symbol,
                    "parsed": parsed,
                    "option_type": option_type,
                    "days_out": days_out,
                    "open_interest": oi,
                    "volume": volume,
                    "price": price,
                    "spread_pct": spread_pct,
                    "delta": delta,
                    "delta_abs": abs(delta) if delta is not None else None,
                    "iv": iv,
                    "premium_pct": premium_pct,
                }
            )
    diagnostics["base_rejections"] = dict(base_rejections)
    if price_fallbacks:
        diagnostics["price_fallbacks"] = dict(price_fallbacks)
    diagnostics["candidates_considered"] = len(candidates)

    if not candidates:
        logger.warning(
            "Option selection aborted for %s: no candidates with usable prices (%s)",
            symbol.upper(),
            diagnostics.get("base_rejections"),
        )
        return OptionSelection(None, None, None, diagnostics)

    passes = [
        {"label": "strict", "oi_scale": 1.0, "vol_scale": 1.0, "spread_scale": 1.0, "delta_widen": 0.0},
        {"label": "relaxed", "oi_scale": 0.6, "vol_scale": 0.6, "spread_scale": 1.35, "delta_widen": 0.15},
    ]
    diagnostics["evaluation_passes"] = [p["label"] for p in passes]

    final_rejections: Counter[str] = Counter()
    relaxed_liquidity_flags: Counter[str] = Counter()
    best_choice: Optional[dict[str, Any]] = None
    best_score: Optional[tuple[float, float, float, float, float, float]] = None
    best_pass = None
    target_mid = (min_days + max_days) / 2.0

    for idx, pass_cfg in enumerate(passes, start=1):
        pass_label = pass_cfg["label"]
        min_oi = min_open_interest_base
        min_vol = min_volume_base
        if pass_label == "strict":
            if min_oi:
                # Even the strict pass should allow thinner markets when paper
                # liquidity is sparse, but keep a modest floor.
                min_oi = max(5, int(round(min_oi * pass_cfg["oi_scale"])))
            if min_vol:
                min_vol = max(1, int(round(min_vol * pass_cfg["vol_scale"])))
        else:
            # Alpaca's paper option chains often report zero volume/OI. On the
            # relaxed pass allow requirements to scale all the way to zero so
            # we can still consider contracts that meet every other risk check.
            if min_oi is None:
                min_oi = 0
            else:
                min_oi = max(0, int(round(min_oi * pass_cfg["oi_scale"])))
            if min_vol is None:
                min_vol = 0
            else:
                min_vol = max(0, int(round(min_vol * pass_cfg["vol_scale"])))
        spread_limit = max_spread_pct * pass_cfg["spread_scale"] if max_spread_pct else None
        if spread_limit:
            spread_limit = min(spread_limit, 0.9)
        min_delta_pass = min_delta
        max_delta_pass = max_delta
        if min_delta_pass is not None:
            min_delta_pass = max(0.05, min_delta_pass * (1 - pass_cfg["delta_widen"]))
        if max_delta_pass is not None:
            max_delta_pass = min(0.95, max_delta_pass + pass_cfg["delta_widen"])

        for metrics in candidates:
            reasons: list[str] = []
            oi = metrics["open_interest"]
            vol = metrics["volume"]
            spread_pct = metrics["spread_pct"]
            delta_abs = metrics["delta_abs"]
            iv = metrics["iv"]
            premium_pct = metrics["premium_pct"]

            oi_for_check = oi if oi is not None else (0 if pass_label == "relaxed" else None)
            if min_oi and (oi_for_check is None or oi_for_check < min_oi):
                if pass_label == "strict":
                    reasons.append("open_interest")
                else:
                    # Track that the relaxed pass tolerated thin liquidity so
                    # diagnostics still reflect the weaker tape, but do not
                    # let this be a hard rejection for paper trading.
                    relaxed_liquidity_flags["open_interest"] += 1
            vol_for_check = vol if vol is not None else (0 if pass_label == "relaxed" else None)
            if min_vol and (vol_for_check is None or vol_for_check < min_vol):
                if pass_label == "strict":
                    reasons.append("volume")
                else:
                    # Same idea for reported volume: log it but keep the
                    # contract in the running when every other risk test
                    # passes.
                    relaxed_liquidity_flags["volume"] += 1
            if spread_limit and (spread_pct is None or spread_pct > spread_limit):
                reasons.append("bid_ask_spread")
            if min_delta_pass is not None and delta_abs is not None and delta_abs < min_delta_pass:
                reasons.append("delta_below_min")
            if max_delta_pass is not None and delta_abs is not None and delta_abs > max_delta_pass:
                reasons.append("delta_above_max")
            if max_iv and iv and iv > max_iv:
                reasons.append("implied_volatility")
            if max_premium_pct and premium_pct and premium_pct > max_premium_pct:
                reasons.append("premium_to_spot")

            if reasons:
                if idx == len(passes):
                    for reason in reasons:
                        final_rejections[reason] += 1
                continue

            delta_for_score = delta_abs if delta_abs is not None else target_delta
            delta_gap = abs(delta_for_score - target_delta)
            expiry_gap = abs(metrics["days_out"] - target_mid)
            spread_component = spread_pct if spread_pct is not None else 1.0
            liquidity_penalty = 1.0 / ((oi or 0.0) + 1.0)
            type_penalty = 0.0 if metrics["option_type"] == contract_types[0] else 0.25
            score_tuple = (
                delta_gap,
                expiry_gap,
                spread_component,
                liquidity_penalty,
                type_penalty,
                metrics["price"],
            )
            if best_score is None or score_tuple < best_score:
                best_score = score_tuple
                best_choice = metrics
                best_pass = pass_cfg["label"]

        if best_choice:
            break

    diagnostics["final_rejections"] = dict(final_rejections)
    if relaxed_liquidity_flags:
        diagnostics["relaxed_liquidity_flags"] = dict(relaxed_liquidity_flags)

    if not best_choice:
        return OptionSelection(None, None, None, diagnostics)

    diagnostics["selected_contract"] = {
        "symbol": best_choice["symbol"],
        "option_type": best_choice["option_type"],
        "days_out": best_choice["days_out"],
        "open_interest": best_choice["open_interest"],
        "volume": best_choice["volume"],
        "spread_pct": best_choice["spread_pct"],
        "delta": best_choice["delta"],
        "selection_pass": best_pass,
    }

    return OptionSelection(
        best_choice["contract"],
        best_choice["price"],
        best_choice["parsed"],
        diagnostics,
    )


def _autopilot_prepare_dataframe(symbol: str, period: str) -> pd.DataFrame | None:
    try:
        hist = get_price_history(symbol, period)
    except PriceDataError as exc:
        logger.warning("Autopilot skip %s: %s", symbol, exc)
        return None
    if hist is None or hist.empty:
        return None
    df = pd.DataFrame(index=hist.index.copy())
    try:
        df["Close"] = _col_series(hist, "Close", symbol)
        df["High"] = _col_series(hist, "High", symbol)
        df["Low"] = _col_series(hist, "Low", symbol)
    except Exception as exc:
        logger.debug("Autopilot failed to shape history for %s: %s", symbol, exc)
        return None
    df = df.dropna()
    if df.empty:
        return None
    try:
        df = compute_indicators(df)
    except Exception as exc:
        logger.warning("Autopilot indicators failed for %s: %s", symbol, exc)
        return None
    return df


def run_autopilot_cycle(force: bool = False) -> None:
    logger.info("Autopilot cycle starting%s", " (forced)" if force else "")

    if not _autopilot_runtime_lock.acquire(blocking=force):
        logger.debug("Autopilot cycle skipped; previous cycle still running")
        return

    summary_lines: list[str] = []
    errors: list[str] = []
    orders_placed = 0
    candidate_count = 0
    try:
        with _autopilot_lock:
            config = dict(_autopilot_state)
        logger.info(
            "Autopilot config loaded: enabled=%s paused=%s strategy=%s risk=%s",
            config.get("enabled"),
            config.get("paused"),
            config.get("strategy"),
            config.get("risk"),
        )

        if config.get("enabled") is False or config.get("paused") is True:
            summary_lines.append("Autopilot is paused.")
            return

        prerequisites_met = True
        if not paper_broker.enabled:
            summary_lines.append("Paper trading disabled; autopilot idle.")
            prerequisites_met = False

        strategy: dict[str, Any] = {}
        risk_cfg: dict[str, Any] = {}
        asset_class = "equity"
        equity = 0.0
        positions: list[dict] = []
        open_orders: list[dict] = []
        recs_snapshot: list[dict] = []

        if prerequisites_met:
            strategy_key = config.get("strategy", "balanced")
            risk_key = config.get("risk", "medium")
            strategy = AUTOPILOT_STRATEGIES.get(strategy_key, AUTOPILOT_STRATEGIES["balanced"])
            risk_cfg = AUTOPILOT_RISK_LEVELS.get(risk_key, AUTOPILOT_RISK_LEVELS["medium"])
            asset_class = _strategy_asset_class(strategy)

            try:
                account = paper_broker.get_account()
            except Exception as exc:
                logger.exception("Autopilot failed to fetch account")
                errors.append(f"account error: {exc}")
                prerequisites_met = False
            else:
                equity = safe_float(account.get("equity"), safe_float(account.get("cash")))
                if equity <= 0:
                    summary_lines.append("Account equity unavailable; skipping cycle.")
                    prerequisites_met = False

        if prerequisites_met:
            try:
                positions = list(paper_broker.get_positions())
            except Exception as exc:
                logger.exception("Autopilot failed to fetch positions")
                errors.append(f"positions error: {exc}")
                positions = []

            try:
                open_orders = list(paper_broker.list_orders(status="open", limit=200))
            except Exception as exc:
                logger.exception("Autopilot failed to fetch open orders")
                errors.append(f"orders error: {exc}")
                open_orders = []

            held_positions: dict[str, dict[str, dict[str, Any]]] = {"equity": {}, "option": {}}
            gross_equity_notional = 0.0
            gross_option_notional = 0.0
            held_option_underlyings: set[str] = set()

            for pos in positions:
                symbol_raw = str(pos.get("symbol", ""))
                symbol = symbol_raw.replace(" ", "").upper()
                qty = safe_float(pos.get("qty"), safe_float(pos.get("quantity")))
                if not symbol or qty <= 0:
                    continue
                market_value = abs(
                    safe_float(
                        pos.get("market_value"),
                        qty * safe_float(pos.get("avg_entry_price"), 0.0),
                    )
                )
                asset = str(pos.get("asset_class", "")).lower()
                is_option_pos = "option" in asset
                entry: dict[str, Any] = {"position": pos, "qty": qty, "market_value": market_value}
                if is_option_pos:
                    parsed = parse_option_symbol(symbol)
                    if parsed:
                        entry["meta"] = parsed
                        underlying = parsed.get("underlying")
                        if underlying:
                            held_option_underlyings.add(underlying)
                    held_positions["option"][symbol] = entry
                    gross_option_notional += market_value
                else:
                    held_positions["equity"][symbol] = entry
                    gross_equity_notional += market_value

            with _lock:
                recs_snapshot = [dict(r) for r in _recommendations]

            if not recs_snapshot:
                seek_recommendations()
                with _lock:
                    recs_snapshot = [dict(r) for r in _recommendations]
            candidate_count = len(recs_snapshot)
            logger.info("Autopilot evaluating %d recommendation candidates", candidate_count)

            position_multiplier = max(risk_cfg.get("position_multiplier", 1.0), 0.25)
            stop_loss_multiplier = max(risk_cfg.get("stop_loss_multiplier", 1.0), 0.25)
            take_profit_multiplier = max(risk_cfg.get("take_profit_multiplier", 1.0), 0.25)

            max_position_pct = min(
                strategy.get("max_position_pct", 0.1) * position_multiplier, 0.95
            )
            max_total_allocation = max(
                0.1, strategy.get("max_total_allocation", 1.0) * position_multiplier
            )
            min_entry_notional = max(50.0, strategy.get("min_entry_notional", 200.0))

            exit_threshold = strategy.get("exit_score", 2.0)
            lookback = strategy.get("lookback", "1y")

            current_positions = (
                held_positions["option"] if asset_class == "option" else held_positions["equity"]
            )
            if asset_class == "option":
                _autopilot_uncovered_exits.intersection_update(set(current_positions.keys()))
            else:
                _autopilot_uncovered_exits.clear()

            if asset_class == "option":
                option_profit = safe_float(strategy.get("options_take_profit_pct"), 0.8)
                option_stop = abs(safe_float(strategy.get("options_stop_loss_pct"), 0.45))
                expiry_buffer = max(0, int(safe_float(strategy.get("options_expiry_buffer"), 5)))

                for contract_symbol, entry in current_positions.items():
                    if contract_symbol in _autopilot_uncovered_exits:
                        notice = (
                            f"Manual attention required for {contract_symbol}; previous exit rejected as uncovered."
                        )
                        logger.info(notice)
                        summary_lines.append(notice)
                        continue
                    pos = entry.get("position")
                    qty = int(abs(entry.get("qty", 0)))
                    available_qty = max(qty, _option_position_quantity(pos))
                    if available_qty <= 0:
                        refreshed = _find_option_position(contract_symbol, positions)
                        if not refreshed:
                            try:
                                positions = list(paper_broker.get_positions())
                            except Exception as refresh_exc:
                                logger.warning(
                                    "Skip exit for %s: failed to refresh positions (%s)",
                                    contract_symbol,
                                    refresh_exc,
                                )
                                summary_lines.append(
                                    f"Skip exit for {contract_symbol}; unable to confirm position."
                                )
                                continue
                            refreshed = _find_option_position(contract_symbol, positions)
                        if refreshed:
                            entry["position"] = refreshed
                            available_qty = _option_position_quantity(refreshed)
                            entry["qty"] = available_qty
                    if available_qty <= 0:
                        logger.warning("Skip exit for %s: no matching position found", contract_symbol)
                        summary_lines.append(
                            f"Skip exit for {contract_symbol}; no matching position found."
                        )
                        continue
                    if qty <= 0:
                        qty = available_qty
                    if qty > available_qty:
                        logger.info(
                            "Clamping exit qty for %s to %d from %d", contract_symbol, available_qty, qty
                        )
                        qty = available_qty
                    pos = entry.get("position") or {}
                    parsed = entry.get("meta") or parse_option_symbol(contract_symbol)
                    underlying = parsed.get("underlying") if parsed else None
                    reasons: list[str] = []
                    plpc = safe_float(pos.get("unrealized_plpc"), None)
                    if plpc is None:
                        percent = safe_float(pos.get("unrealized_pl_percent"), None)
                        if percent is not None:
                            plpc = percent / 100.0
                    if plpc is not None and option_profit and plpc >= option_profit:
                        reasons.append(f"profit {plpc*100:.0f}% >= target")
                    if plpc is not None and option_stop and plpc <= -option_stop:
                        reasons.append(f"loss {plpc*100:.0f}% beyond stop")
                    days_out = option_days_to_expiration(parsed)
                    if days_out is not None and days_out <= expiry_buffer:
                        reasons.append(f"{days_out}d to expiry")
                    if underlying:
                        df = _autopilot_prepare_dataframe(underlying, lookback)
                        if df is None:
                            errors.append(f"no data for {underlying}; option exit skipped")
                        else:
                            underlying_score, _ = score_stock(df)
                            if underlying_score < exit_threshold:
                                reasons.append(f"score {underlying_score:.2f} < {exit_threshold}")
                    if not reasons:
                        continue
                    open_exit_orders = paper_broker.list_open_orders_for_symbol(
                        contract_symbol,
                        asset_class="option",
                        side="sell",
                        orders=open_orders,
                    )
                    if open_exit_orders:
                        summary_lines.append(
                            f"Exit pending for {contract_symbol}; open exit order already open."
                        )
                        continue
                    market_price = safe_float(
                        pos.get("current_price"), safe_float(pos.get("market_price"), None)
                    )
                    avg_entry_price = safe_float(pos.get("avg_entry_price"), None)
                    # Alpaca's paper API rejects market exits when no NBBO quote exists,
                    # so synthesize a conservative limit using whatever reference price
                    # we have to keep the order accepted and resting on the book.
                    price_hint = market_price or avg_entry_price
                    limit_price = None
                    if market_price and market_price > 0:
                        limit_price = round(max(market_price * 0.98, 0.01), 2)
                    elif avg_entry_price and avg_entry_price > 0:
                        limit_price = round(max(avg_entry_price * 0.5, 0.01), 2)
                    order_type = "limit" if limit_price else "market"
                    price_hint = price_hint if (price_hint and price_hint > 0) else limit_price
                    try:
                        result = place_guarded_paper_order(
                            contract_symbol,
                            qty,
                            "sell",
                            order_type=order_type,
                            limit_price=limit_price,
                            stop_loss_pct=None,
                            take_profit_pct=None,
                            time_in_force="day",
                            asset_class="option",
                            price_hint=price_hint,
                            support_brackets=False,
                            position_effect="close",
                        )
                        if isinstance(result, dict) and result.get("status") == "rejected_uncovered":
                            warning = (
                                f"Exit for {contract_symbol} rejected as uncovered; manual attention required."
                            )
                            logger.warning(warning)
                            summary_lines.append(warning)
                            _autopilot_uncovered_exits.add(contract_symbol)
                            continue
                        orders_placed += 1
                        summary_lines.append(
                            f"Exit {qty} {contract_symbol} ({'; '.join(reasons)})"
                        )
                    except NoAvailableBidError:
                        postpone_msg = (
                            f"Exit for {contract_symbol} postponed; Alpaca reports no available bid."
                        )
                        logger.info(postpone_msg)
                        summary_lines.append(postpone_msg)
                    except OptionCloseRejectedError as exc:
                        warning = (
                            f"Exit for {contract_symbol} rejected: {exc.api_message or exc}"
                        )
                        logger.warning(warning)
                        summary_lines.append(warning)
                        _autopilot_uncovered_exits.add(contract_symbol)
                    except Exception as exc:
                        logger.exception("Autopilot option exit failed for %s", contract_symbol)
                        errors.append(f"sell {contract_symbol} failed: {exc}")
            else:
                for symbol, entry in current_positions.items():
                    pos = entry["position"]
                    qty = abs(safe_float(pos.get("qty"), safe_float(pos.get("quantity"))))
                    if qty < 1:
                        continue
                    df = _autopilot_prepare_dataframe(symbol, lookback)
                    if df is None:
                        errors.append(f"no data for {symbol}; exit review skipped")
                        continue
                    score, _ = score_stock(df)
                    if score < exit_threshold:
                        if _autopilot_order_blocked(symbol, open_orders):
                            summary_lines.append(
                                f"Exit pending for {symbol}; open order detected."
                            )
                            continue
                        qty_int = int(math.floor(qty))
                        if qty_int <= 0:
                            continue
                        try:
                            place_guarded_paper_order(symbol, qty_int, "sell", time_in_force="day")
                            summary_lines.append(
                                f"Exit {qty_int} {symbol} (score {score:.2f} < {exit_threshold})"
                            )
                            orders_placed += 1
                        except Exception as exc:
                            logger.exception("Autopilot equity exit failed for %s", symbol)
                            errors.append(f"sell {symbol} failed: {exc}")

            held_and_pending = set(current_positions.keys())
            pending_underlyings = (
                set(held_option_underlyings) if asset_class == "option" else set()
            )
            for order in open_orders:
                try:
                    if str(order.get("side", "")).lower() != "buy":
                        continue
                    order_symbol = str(order.get("symbol", "")).replace(" ", "").upper()
                    asset = str(order.get("asset_class", "")).lower()
                    if asset_class == "option":
                        if "option" in asset:
                            held_and_pending.add(order_symbol)
                            parsed = parse_option_symbol(order_symbol)
                            if parsed and parsed.get("underlying"):
                                pending_underlyings.add(parsed["underlying"])
                    else:
                        if asset != "option":
                            held_and_pending.add(order_symbol)
                except Exception:
                    continue

            buy_candidates: list[tuple[str, float]] = []
            for rec in recs_snapshot:
                symbol = str(rec.get("Symbol", "")).upper()
                if not symbol:
                    continue
                score = safe_float(rec.get("Score"))
                if score < strategy.get("min_score", 3.0):
                    continue
                if asset_class == "option":
                    if symbol in pending_underlyings:
                        continue
                else:
                    if symbol in held_and_pending:
                        continue
                buy_candidates.append((symbol, score))

            if not buy_candidates:
                summary_lines.append("No new symbols met entry criteria.")

            buy_candidates.sort(key=lambda item: item[1], reverse=True)
            max_positions = max(1, int(strategy.get("max_positions", 5)))
            available_slots = max(0, max_positions - len(current_positions))
            gross_notional = (
                gross_option_notional if asset_class == "option" else gross_equity_notional
            )

            allocation_warning_logged = False

            for symbol, score in buy_candidates:
                if available_slots <= 0:
                    break

                if asset_class == "option":
                    try:
                        spot_price = fetch_latest_price(symbol)
                    except Exception as exc:
                        logger.exception("Autopilot price fetch failed for %s", symbol)
                        errors.append(f"price {symbol} failed: {exc}")
                        continue
                    try:
                        selection = _autopilot_select_option_contract(
                            symbol,
                            strategy,
                            underlying_price=spot_price,
                            score=score,
                        )
                    except PriceDataError as exc:
                        errors.append(f"options {symbol} chain failed: {exc}")
                        continue
                    diag = selection.diagnostics or {}
                    if diag:
                        try:
                            logger.debug(
                                "Option selection diagnostics for %s: %s",
                                symbol.upper(),
                                json.dumps(to_plain(diag)),
                            )
                        except Exception:
                            logger.debug(
                                "Option selection diagnostics for %s: %s",
                                symbol.upper(),
                                diag,
                            )
                    contract_data = selection.contract
                    premium = selection.premium
                    parsed = selection.meta or {}
                    if not contract_data or premium is None or premium <= 0:
                        reason_bits = []
                        rejection_counts = Counter()
                        base_rejects = (
                            diag.get("base_rejections")
                            if isinstance(diag.get("base_rejections"), dict)
                            else {}
                        )
                        final_rejects = (
                            diag.get("final_rejections")
                            if isinstance(diag.get("final_rejections"), dict)
                            else {}
                        )
                        try:
                            rejection_counts.update(Counter(base_rejects))
                            rejection_counts.update(Counter(final_rejects))
                        except Exception:
                            rejection_counts = Counter()
                        chain_errors = (
                            diag.get("chain_errors")
                            if isinstance(diag.get("chain_errors"), dict)
                            else {}
                        )
                        for chain_type, message in chain_errors.items():
                            if not message:
                                continue
                            reason_bits.append(
                                f"{chain_type}_chain:{str(message).strip()[:80]}"
                            )
                        if rejection_counts:
                            for reason, count in rejection_counts.most_common(3):
                                reason_bits.append(f"{reason}:{count}")
                        if reason_bits:
                            logger.warning(
                                "Option selection rejected %s: %s",
                                symbol.upper(),
                                ", ".join(reason_bits),
                            )
                        else:
                            logger.warning(
                                "Option selection rejected %s: no priced contracts",
                                symbol.upper(),
                            )
                        summary_lines.append(
                            "No option contracts matched filters for {}.{}".format(
                                symbol,
                                f" ({', '.join(reason_bits)})" if reason_bits else "",
                            )
                        )
                        continue
                    contract_symbol = (
                        str(contract_data.get("symbol", "")).replace(" ", "").upper()
                        if isinstance(contract_data, dict)
                        else ""
                    )
                    if not contract_symbol or contract_symbol in held_and_pending:
                        continue
                    unit_cost = premium * OPTION_CONTRACT_MULTIPLIER
                    notional_cap_pct = min(max_position_pct, max_total_allocation)
                    target_notional = max(min_entry_notional, equity * notional_cap_pct)
                    max_debit = safe_float(strategy.get("max_position_debit"), None)
                    if max_debit:
                        target_notional = min(target_notional, max_debit)
                    if PAPER_MAX_POSITION_NOTIONAL:
                        target_notional = min(target_notional, PAPER_MAX_POSITION_NOTIONAL)
                    if gross_notional + target_notional > equity * max_total_allocation:
                        if not allocation_warning_logged:
                            summary_lines.append(
                                "Max portfolio allocation reached; skipping new entries."
                            )
                            allocation_warning_logged = True
                        break
                    qty = max(1, int(target_notional // max(unit_cost, 1e-6)))
                    max_contracts = int(safe_float(strategy.get("max_contracts_per_trade"), 0))
                    if max_contracts:
                        qty = min(qty, max_contracts)
                    if qty <= 0:
                        continue
                    try:
                        place_guarded_paper_order(
                            contract_symbol,
                            qty,
                            "buy",
                            order_type="limit",
                            limit_price=round(premium, 2),
                            stop_loss_pct=None,
                            take_profit_pct=None,
                            time_in_force="day",
                            asset_class="option",
                            price_hint=premium,
                            support_brackets=False,
                        )
                        gross_notional += qty * unit_cost
                        available_slots -= 1
                        held_and_pending.add(contract_symbol)
                        if parsed and parsed.get("underlying"):
                            pending_underlyings.add(parsed["underlying"])
                        summary_lines.append(
                            f"Buy {qty} {contract_symbol} ({symbol} {parsed.get('type', 'call')} {parsed.get('strike', 0):.2f} exp {parsed.get('expiration')}, premium ${premium:.2f})"
                        )
                        logger.info(
                            "Autopilot option entry: %s qty=%d premium=%.2f notional=%.2f",
                            contract_symbol,
                            qty,
                            premium,
                            qty * unit_cost,
                        )
                        orders_placed += 1
                    except Exception as exc:
                        logger.exception("Autopilot option entry failed for %s", contract_symbol)
                        errors.append(f"buy {contract_symbol} failed: {exc}")
                else:
                    if _autopilot_order_blocked(symbol, open_orders):
                        continue
                    try:
                        price = fetch_latest_price(symbol)
                    except Exception as exc:
                        logger.exception("Autopilot price fetch failed for %s", symbol)
                        errors.append(f"price {symbol} failed: {exc}")
                        continue
                    notional_cap_pct = min(max_position_pct, max_total_allocation)
                    target_notional = max(min_entry_notional, equity * notional_cap_pct)
                    if PAPER_MAX_POSITION_NOTIONAL:
                        target_notional = min(target_notional, PAPER_MAX_POSITION_NOTIONAL)
                    if gross_notional + target_notional > equity * max_total_allocation:
                        if not allocation_warning_logged:
                            summary_lines.append(
                                "Max portfolio allocation reached; skipping new entries."
                            )
                            allocation_warning_logged = True
                        break
                    qty = int(target_notional // price)
                    if qty <= 0:
                        continue
                    stop_loss_pct = max(
                        0.01, strategy.get("stop_loss_pct", 0.04) * stop_loss_multiplier
                    )
                    take_profit_pct = max(
                        0.02, strategy.get("take_profit_pct", 0.1) * take_profit_multiplier
                    )
                    try:
                        place_guarded_paper_order(
                            symbol,
                            qty,
                            "buy",
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            time_in_force="gtc",
                        )
                        gross_notional += qty * price
                        available_slots -= 1
                        held_and_pending.add(symbol)
                        summary_lines.append(
                            f"Buy {qty} {symbol} (score {score:.2f}, stop {stop_loss_pct*100:.1f}%, take {take_profit_pct*100:.1f}%)"
                        )
                        orders_placed += 1
                    except Exception as exc:
                        logger.exception("Autopilot equity entry failed for %s", symbol)
                        errors.append(f"buy {symbol} failed: {exc}")

            if not summary_lines:
                summary_lines.append("Cycle complete with no trades.")
    except Exception as exc:
        logger.exception("Autopilot cycle failed")
        errors.append(str(exc))
    finally:
        with _autopilot_lock:
            _autopilot_state["last_run"] = datetime.now()
            _autopilot_state["last_actions"] = summary_lines or ["Cycle complete with no trades."]
            _autopilot_state["last_error"] = "; ".join(errors) if errors else None
        _autopilot_runtime_lock.release()

        summary_text = " | ".join(summary_lines) if summary_lines else "Cycle complete with no trades."
        error_text = "; ".join(errors) if errors else None
        _record_autopilot_run(not errors, summary_text, error_text)
        logger.info(
            "Autopilot cycle finished; candidates=%d, orders=%d, errors=%d",
            candidate_count,
            orders_placed,
            len(errors),
        )

        if errors:
            logger.warning("Autopilot cycle completed with errors: %s", "; ".join(errors))
        else:
            logger.info("Autopilot cycle completed: %s", " | ".join(summary_lines))

# Helper to trigger an autopilot cycle without blocking the caller
def trigger_autopilot_run(*, force: bool = False) -> None:
    threading.Thread(target=run_autopilot_cycle, args=(force,), daemon=True).start()


# -----------------------------
# Data sources
# -----------------------------

def update_sp500() -> None:
    logger.info("Updating S&P 500 list if stale…")
    if datetime.now() - _sp500["updated"] <= timedelta(hours=24):
        return

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch S&P 500 constituents: %s", exc)
        return

    try:
        tables = pd.read_html(StringIO(response.text))
    except Exception as exc:
        logger.warning("Failed to parse S&P 500 table: %s", exc)
        return

    if not tables:
        logger.warning("No tables found when parsing S&P 500 page")
        return

    table = None
    symbol_col = None
    for candidate in tables:
        df = candidate.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                " ".join(str(part).strip() for part in col if str(part) != "nan").strip()
                for col in df.columns
            ]
        else:
            df.columns = [str(col).strip() for col in df.columns]

        for col in df.columns:
            normalized = col.lower()
            if normalized in {"symbol", "ticker symbol", "ticker"}:
                table = df
                symbol_col = col
                break
        if table is not None:
            break

    if table is None or symbol_col is None:
        logger.warning("S&P 500 table missing 'Symbol' column")
        return

    tickers = [
        str(t).strip().upper()
        for t in table[symbol_col].dropna()
        if isinstance(t, (str, int, float)) and str(t).strip()
    ]
    if not tickers:
        logger.warning("Parsed S&P 500 table but found no tickers")
        return

    with _lock:
        _sp500["tickers"] = tickers
        _sp500["updated"] = datetime.now()
    logger.info("Cached %d S&P 500 tickers", len(tickers))


# -----------------------------
# Analysis
# -----------------------------

def analyze_stock(ticker: str) -> dict:
    """Compute metrics and chart data for a single ticker (flattening any MultiIndex)."""
    logger.info("Analyze %s", ticker)
    try:
        hist = get_price_history(ticker, "2y")
    except PriceDataError as exc:
        logger.warning("Analyze %s failed: %s", ticker, exc)
        raise ValueError(str(exc))

    # Build a simple, single-index DataFrame for techs/plotting
    close = _col_series(hist, "Close", ticker)
    high  = _col_series(hist, "High",  ticker)
    low   = _col_series(hist, "Low",   ticker)

    df = pd.DataFrame(index=hist.index.copy())
    df["Close"] = close
    df["High"]  = high
    df["Low"]   = low
    df = compute_indicators(df)

    # Chart data: last ~180 days with ISO date strings
    chart_df = df.tail(180).reset_index().rename(columns={df.index.name or "index": "Date"})
    if pd.api.types.is_datetime64_any_dtype(chart_df["Date"]):
        chart_df["Date"] = chart_df["Date"].dt.strftime("%Y-%m-%d")
    chart_data = chart_df[["Date", "Close", "RSI", "SMA_50", "SMA_200"]].to_dict(orient="records")

    latest_close = last_value(df["Close"])
    latest_rsi = last_value(df["RSI"])
    latest_sma50 = last_value(df["SMA_50"])
    latest_sma200 = last_value(df["SMA_200"])

    sentiment = fetch_news_sentiment(ticker) + fetch_reddit_sentiment(ticker)
    above_sma50 = (
        latest_close is not None
        and latest_sma50 is not None
        and latest_close > latest_sma50
    )
    above_sma200 = (
        latest_close is not None
        and latest_sma200 is not None
        and latest_close > latest_sma200
    )
    rec = "BUY" if above_sma50 else "HOLD"

    summary_bits: list[str] = []
    if above_sma50:
        summary_bits.append(
            "Price is above the 50-day moving average, indicating positive momentum."
        )
    elif latest_close is not None and latest_sma50 is not None:
        summary_bits.append(
            "Price is below the 50-day moving average, so momentum looks soft."
        )
    if above_sma200:
        summary_bits.append(
            "Price is also above the 200-day moving average, supporting an uptrend view."
        )
    elif latest_close is not None and latest_sma200 is not None:
        summary_bits.append(
            "Price is under the 200-day moving average, so the long-term trend is weaker."
        )
    if latest_rsi is not None:
        if latest_rsi >= 70:
            summary_bits.append("RSI is elevated, suggesting overbought conditions.")
        elif latest_rsi <= 30:
            summary_bits.append("RSI is low, suggesting the stock may be oversold.")
        else:
            summary_bits.append("RSI is neutral, so momentum is balanced.")
    if not summary_bits:
        summary_bits.append("Insufficient indicator data to form an opinion.")
    analysis_summary = " ".join(summary_bits)

    out = {
        "Symbol": ticker,
        "Close": round(latest_close, 2) if latest_close is not None else None,
        "RSI": round(latest_rsi, 2) if latest_rsi is not None else None,
        "50-day MA": round(latest_sma50, 2) if latest_sma50 is not None else None,
        "200-day MA": round(latest_sma200, 2) if latest_sma200 is not None else None,
        "Recommendation": rec,
        "Analysis Summary": analysis_summary,
        "Chart Data": chart_data,
    }
    logger.info("Analyze %s done: rows=%d chart_rows=%d", ticker, len(df), len(chart_data))
    return out


# -----------------------------
# Recommendations
# -----------------------------

def fast_filter_ticker(ticker: str) -> bool:
    try:
        hist = get_price_history(ticker, "90d")
    except PriceDataError:
        return False
    if hist is None or hist.empty or "Volume" not in hist.columns:
        return False
    volume_series = _coerce_numeric_series(_col_series(hist, "Volume", ticker), hist.index)
    if volume_series.empty:
        return False
    avg_vol = volume_series.tail(30).mean()
    avg_vol = float(avg_vol) if pd.notna(avg_vol) else 0.0
    return avg_vol > 1_000_000


def seek_recommendations() -> None:
    global _recommendations
    logger.info("Refreshing recommendations…")
    with _lock:
        _rec_state["refreshing"] = True
        _rec_state["last_error"] = None

    try:
        update_sp500()
        with _lock:
            tickers = _sp500["tickers"][:]

        if not tickers:
            fallback_env = os.getenv("FALLBACK_TICKERS", "")
            fallback_cfg = [t.strip().upper() for t in fallback_env.split(",") if t.strip()]
            fallback = fallback_cfg or FALLBACK_TICKERS
            tickers = fallback[:]
            logger.warning(
                "S&P 500 cache is empty; using %d fallback tickers for recommendations",
                len(tickers),
            )
            if not tickers:
                logger.warning("No fallback tickers configured; skipping recommendation refresh")
                return

        # deterministic daily shuffle to avoid A* bias
        rng = random.Random(date.today().toordinal())
        rng.shuffle(tickers)

        # pick first ~30 that pass volume filter
        filtered = []
        for t in tickers:
            try:
                if fast_filter_ticker(t):
                    filtered.append(t)
                    if len(filtered) >= 30:
                        break
            except Exception as exc:
                logger.warning("Liquidity filter failed for %s: %s", t, exc)
                continue

        if not filtered:
            logger.warning(
                "No tickers passed the liquidity filter; keeping existing recommendations"
            )
            with _lock:
                if not _recommendations:
                    _recommendations = _fallback_recommendations(
                        "No tickers passed the liquidity filter"
                    )
            return

        def worker(sym: str) -> dict | None:
            try:
                df = get_price_history(sym, "1y")
            except PriceDataError as exc:
                logger.warning("Skipping %s: %s", sym, exc)
                return None
            except Exception as exc:
                logger.warning("Unexpected error loading %s: %s", sym, exc)
                return None
            try:
                if df is None or df.empty:
                    logger.warning("Skipping %s: no historical data", sym)
                    return None

                # Ensure we have needed cols
                missing = [col for col in ("High", "Low", "Close") if col not in df.columns]
                if missing:
                    logger.warning("Skipping %s: missing columns %s", sym, ", ".join(missing))
                    return None

                df = df[["High", "Low", "Close"]].dropna().copy()
                if df.empty:
                    logger.warning("Skipping %s: insufficient price data after cleaning", sym)
                    return None
                df = compute_indicators(df)
                score, reasons = score_stock(df)
                rec = "BUY" if score >= 3.5 else "HOLD"
                return {"Symbol": sym, "Recommendation": rec, "Score": score, "Why": reasons}
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Skipping %s due to processing error: %s", sym, exc)
                return None

        with ThreadPoolExecutor(max_workers=5) as ex:
            results = [res for res in ex.map(worker, filtered) if res]

        # rank by score desc, then symbol
        results.sort(key=lambda x: (-x.get("Score", 0.0), x.get("Symbol", "")))
        buys = [r for r in results if r["Recommendation"] == "BUY"]
        logger.info("Recommendations computed: %d BUY out of %d", len(buys), len(results))
        completed_at = datetime.now()
        with _lock:
            if results:
                _recommendations = results[:5]  # top 5 overall, already ranked
                _rec_state["last_completed"] = completed_at
                _rec_state["last_error"] = None
            elif not _recommendations:
                _recommendations = _fallback_recommendations(
                    "Unable to compute recommendations; using placeholders"
                )
        if results:
            _record_recommendations_snapshot(results[:5])
    except Exception as exc:
        logger.exception("Recommendation refresh failed")
        with _lock:
            _rec_state["last_error"] = str(exc)
        with _lock:
            if not _recommendations:
                _recommendations = _fallback_recommendations(
                    "Refreshing recommendations failed; using placeholders"
                )
    finally:
        with _lock:
            _rec_state["refreshing"] = False
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# -----------------------------
# Backtest
# -----------------------------

def backtest_ticker(ticker: str, years: int = 3, cost_bps: float = 5.0) -> dict:
    """Very simple long-only backtest for the rules used in scoring.
    cost_bps: round-trip cost in basis points per trade leg (5 = 0.05%).
    """
    logger.info("Backtest %s", ticker)
    hist = get_price_history(ticker, f"{years}y")
    if hist is None or hist.empty:
        raise ValueError("No price data")

    # Indicators
    close = _col_series(hist, "Close", ticker)
    high  = _col_series(hist, "High",  ticker)
    low   = _col_series(hist,  "Low",   ticker)
    df = pd.DataFrame(index=hist.index.copy())
    df["Close"] = close
    df["High"]  = high
    df["Low"]   = low
    df = df.dropna()
    df = compute_indicators(df)

    close = df["Close"]
    sma50 = df["SMA_50"]
    sma200 = df["SMA_200"]
    rsi = df["RSI"]

    buy = (close > sma50) & (sma50 > sma200) & (rsi.between(45, 65))
    sell = (close < sma50) | (rsi < 40) | (rsi > 75)

    pos = pd.Series(index=df.index, dtype=float)
    pos[buy] = 1.0
    pos[sell] = 0.0
    pos = pos.ffill().fillna(0.0)

    daily_ret = close.pct_change().fillna(0.0)
    # apply cost when position changes
    turns = pos.diff().abs().fillna(0.0)
    cost = (cost_bps / 10000.0) * turns
    strat_ret = (pos.shift(1).fillna(0.0) * daily_ret) - cost
    equity = (1.0 + strat_ret).cumprod()

    # Metrics
    days = max(len(df), 1)
    cagr = float(equity.iloc[-1] ** (252.0 / days) - 1.0)
    roll_max = equity.cummax()
    drawdown = (equity / roll_max - 1.0)
    mdd = float(drawdown.min())
    vol = float(strat_ret.std() * np.sqrt(252.0)) if strat_ret.std() > 0 else 0.0
    sharpe = float((strat_ret.mean() * 252.0) / (strat_ret.std() * np.sqrt(252.0) + 1e-12)) if strat_ret.std() > 0 else 0.0

    # Trade stats
    entries = (pos.diff() > 0).astype(int)
    trade_id = entries.cumsum()
    trade_ret = ((1.0 + strat_ret).groupby(trade_id).prod() - 1.0)
    trade_ret = trade_ret[trade_id[trade_id > 0].unique()]  # ignore id 0 when no trade
    wins = int((trade_ret > 0).sum())
    losses = int((trade_ret <= 0).sum())
    win_rate = float(wins / max(wins + losses, 1))
    avg_win = float(trade_ret[trade_ret > 0].mean()) if wins else 0.0
    avg_loss = float(trade_ret[trade_ret <= 0].mean()) if losses else 0.0

    curve = (
        equity.reset_index()
        .rename(columns={equity.index.name or "index": "Date", 0: "Equity"})
    )
    if pd.api.types.is_datetime64_any_dtype(curve["Date"]):
        curve["Date"] = curve["Date"].dt.strftime("%Y-%m-%d")

    return {
        "metrics": {
            "CAGR": round(cagr, 4),
            "Max Drawdown": round(mdd, 4),
            "Volatility": round(vol, 4),
            "Sharpe": round(sharpe, 3),
            "Trades": int(wins + losses),
            "Win Rate": round(win_rate, 3),
            "Avg Win": round(avg_win, 4),
            "Avg Loss": round(avg_loss, 4),
        },
        "equity_curve": curve.to_dict(orient="records"),
    }
# -----------------------------
# Scheduler (avoid double-start)
# -----------------------------

def start_background_jobs():
    global _background_jobs_started, _scheduler
    should_start = not app.debug or str(os.environ.get("WERKZEUG_RUN_MAIN", "")).lower() == "true"
    if not should_start:
        logger.debug("Skipping scheduler start in reloader parent process")
        return
    with _background_jobs_lock:
        if _background_jobs_started:
            return
        _background_jobs_started = True
    logger.info("Starting background scheduler")
    # Run both routines immediately so the UI has fresh data without waiting
    threading.Thread(target=seek_recommendations, daemon=True).start()
    trigger_autopilot_run()
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        seek_recommendations,
        "interval",
        hours=1,
        next_run_time=datetime.now(),
        id="seek_recommendations",
        replace_existing=True,
    )
    _scheduler.add_job(
        run_autopilot_cycle,
        "interval",
        minutes=5,
        next_run_time=datetime.now(),
        id="run_autopilot_cycle",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Background jobs started")

def _ensure_background_jobs() -> None:
    if not _background_jobs_started:
        start_background_jobs()


app.before_request(_ensure_background_jobs)


# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        t = request.form.get("ticker", "").strip().upper()
        if not TICKER_RE.match(t):
            flash("Enter a valid ticker symbol like AAPL or MSFT.")
        else:
            try:
                result = analyze_stock(t)
            except Exception as e:
                if isinstance(e, ValueError):
                    logger.warning("Analyze error for %s: %s", t, e)
                else:
                    logger.exception("Analyze error for %s", t)
                flash(f"Analyze error for {t}: {e}")

    with _lock:
        recs = [dict(r) for r in _recommendations]
        rec_state = dict(_rec_state)

    if not recs:
        try:
            if not rec_state.get("refreshing"):
                threading.Thread(target=seek_recommendations, daemon=True).start()
                with _lock:
                    _rec_state["refreshing"] = True
                    rec_state["refreshing"] = True
            recs = _fallback_recommendations(
                "No recommendations yet; first-time scan can take a few minutes."
            )
        except Exception:
            logger.exception("Initial recommendation refresh failed")

    rec_status = _format_recommendation_status(recs, rec_state)
    # Pre-serialize for safe JS usage in template
    result_clean = to_plain(result or {})
    recs_clean = to_plain(recs or [])
    result_json = json.dumps(result_clean, ensure_ascii=False)
    recs_json = json.dumps(recs_clean, ensure_ascii=False)
    return render_template(
        "index.html",
        result_json=result_json,
        recommendations_json=recs_json,
        rec_status=rec_status,
    )


@app.route("/seek")
def seek_and_redirect():
    # Keep legacy redirect endpoint (still works if user hits it)
    try:
        seek_recommendations()
    except Exception:
        logger.exception("Recommendation refresh failed")
    return redirect(url_for("home"))


@app.route("/api/recommendations")
def api_recommendations():
    with _lock:
        data = [dict(r) for r in _recommendations]
        state = dict(_rec_state)
    if not data:
        data = _fallback_recommendations("No recommendations available; using placeholders")
    status = _format_recommendation_status(data, state)
    return jsonify({"ok": True, "data": data, "status": status})


@app.route("/api/recommendations/refresh", methods=["POST"])
def api_refresh_recommendations():
    try:
        seek_recommendations()
        with _lock:
            data = [dict(r) for r in _recommendations]
            state = dict(_rec_state)
        status = _format_recommendation_status(data, state)
        return jsonify({"ok": True, "data": data, "status": status})
    except Exception as exc:
        logger.exception("Recommendation refresh failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/backtest/<ticker>")
def api_backtest(ticker: str):
    try:
        years = int(request.args.get("years", 3))
    except Exception:
        years = 3
    try:
        data = backtest_ticker(ticker.upper(), years=years)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        logger.exception("Backtest error for %s", ticker)
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/analyze/<ticker>")
def api_analyze(ticker: str):
    try:
        data = analyze_stock(ticker.upper())
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/paper", methods=["GET"])
def paper_dashboard():
    account = None
    positions = []
    orders = []
    broker_error = None
    todays_pl = None
    autopilot_status = get_autopilot_status()
    if not paper_broker.enabled:
        broker_error = "Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY to enable paper trading."
    else:
        try:
            account = paper_broker.get_account()
            positions = list(paper_broker.get_positions())
            orders = list(paper_broker.list_orders(status="all", limit=25))
            if account:
                try:
                    equity = float(account.get("equity") or 0.0)
                    last_equity = float(account.get("last_equity") or equity)
                    todays_pl = equity - last_equity
                except Exception:
                    todays_pl = None
        except Exception as exc:
            broker_error = str(exc)
            logger.exception("Failed to load paper trading data")
    return render_template(
        "paper.html",
        broker_enabled=paper_broker.enabled,
        broker_error=broker_error,
        account=account,
        positions=positions,
        orders=orders,
        todays_pl=todays_pl,
        PAPER_MAX_POSITION_PCT=PAPER_MAX_POSITION_PCT,
        PAPER_MAX_POSITION_NOTIONAL=PAPER_MAX_POSITION_NOTIONAL,
        PAPER_DEFAULT_STOP_LOSS_PCT=PAPER_DEFAULT_STOP_LOSS_PCT,
        PAPER_DEFAULT_TAKE_PROFIT_PCT=PAPER_DEFAULT_TAKE_PROFIT_PCT,
        autopilot=autopilot_status,
        autopilot_strategies=AUTOPILOT_STRATEGIES,
        autopilot_risks=AUTOPILOT_RISK_LEVELS,
    )


@app.route("/paper/order", methods=["POST"])
def paper_order_submit():
    try:
        symbol = request.form.get("symbol", "").strip().upper()
        qty = int(request.form.get("quantity", "0"))
        side = request.form.get("side", "buy").lower()
        order_type = request.form.get("order_type", "market").lower()
        limit_price = request.form.get("limit_price")
        tif = request.form.get("time_in_force", "day")
        stop_loss_pct = parse_percent(request.form.get("stop_loss_pct"), PAPER_DEFAULT_STOP_LOSS_PCT)
        take_profit_pct = parse_percent(request.form.get("take_profit_pct"), PAPER_DEFAULT_TAKE_PROFIT_PCT)
        limit_price_val = float(limit_price) if limit_price else None
        order = place_guarded_paper_order(
            symbol,
            qty,
            side,
            order_type,
            limit_price_val,
            stop_loss_pct,
            take_profit_pct,
            tif,
        )
        flash(f"Order submitted: {order.get('id', 'n/a')}")
    except Exception as exc:
        flash(f"Order failed: {exc}")
    return redirect(url_for("paper_dashboard"))


@app.route("/paper/autopilot", methods=["POST"])
def paper_autopilot_update():
    if not paper_broker.enabled:
        flash("Enable paper trading to use the autopilot.")
        return redirect(url_for("paper_dashboard"))

    enabled = request.form.get("autopilot_enabled") == "on"
    strategy = request.form.get("autopilot_strategy", "balanced")
    risk = request.form.get("autopilot_risk", "medium")
    status = update_autopilot_config(enabled=enabled, strategy=strategy, risk=risk)
    strategy_cfg = AUTOPILOT_STRATEGIES.get(status.get("strategy", "balanced"), {})
    risk_cfg = AUTOPILOT_RISK_LEVELS.get(status.get("risk", "medium"), {})
    state_text = "enabled" if status.get("enabled") else "paused"
    flash(
        "Autopilot %s – %s (%s risk)" % (
            state_text,
            strategy_cfg.get("label", status.get("strategy", "balanced")).strip(),
            risk_cfg.get("label", status.get("risk", "medium")),
        )
    )
    if status.get("enabled"):
        run_autopilot_cycle(force=True)
    return redirect(url_for("paper_dashboard"))


@app.route("/paper/autopilot/run", methods=["POST"])
def paper_autopilot_run_now():
    if not paper_broker.enabled:
        flash("Enable paper trading to use the autopilot.")
        return redirect(url_for("paper_dashboard"))

    try:
        run_autopilot_cycle(force=True)
        status = get_autopilot_status()
        if status.get("last_run_error"):
            flash(f"Autopilot run finished with warnings: {status.get('last_run_error')}")
        else:
            summary = status.get("last_run_summary") or "Run complete."
            flash(summary)
    except Exception as exc:
        logger.exception("Forced autopilot run failed")
        flash(f"Autopilot run failed: {exc}")
    return redirect(url_for("paper_dashboard"))


@app.route("/api/paper/account")
def api_paper_account():
    if not paper_broker.enabled:
        return jsonify({"ok": False, "error": "paper trading disabled"}), 400
    try:
        return jsonify({"ok": True, "data": paper_broker.get_account()})
    except Exception as exc:
        logger.exception("Account fetch failed")
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/api/paper/positions")
def api_paper_positions():
    if not paper_broker.enabled:
        return jsonify({"ok": False, "error": "paper trading disabled"}), 400
    try:
        return jsonify({"ok": True, "data": list(paper_broker.get_positions())})
    except Exception as exc:
        logger.exception("Positions fetch failed")
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/api/paper/orders")
def api_paper_orders():
    if not paper_broker.enabled:
        return jsonify({"ok": False, "error": "paper trading disabled"}), 400
    try:
        status = request.args.get("status", "all")
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "data": list(paper_broker.list_orders(status=status, limit=limit))})
    except Exception as exc:
        logger.exception("Orders fetch failed")
        return jsonify({"ok": False, "error": str(exc)}), 502


@app.route("/api/paper/autopilot", methods=["GET", "POST"])
def api_paper_autopilot():
    if request.method == "GET":
        return jsonify({"ok": True, "data": get_autopilot_status()})

    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception as exc:
        return jsonify({"ok": False, "error": f"invalid payload: {exc}"}), 400

    enabled = payload.get("enabled")
    if isinstance(enabled, str):
        enabled = enabled.lower() in {"true", "1", "yes", "on"}
    elif enabled is not None:
        enabled = bool(enabled)

    paused = payload.get("paused")
    if isinstance(paused, str):
        paused = paused.lower() in {"true", "1", "yes", "on"}
    elif paused is not None:
        paused = bool(paused)

    strategy = payload.get("strategy")
    risk = payload.get("risk")

    status = update_autopilot_config(
        enabled=enabled,
        paused=paused,
        strategy=strategy,
        risk=risk,
    )
    if status.get("enabled") and paper_broker.enabled:
        run_autopilot_cycle(force=True)
    return jsonify({"ok": True, "data": status})


@app.route("/api/paper/order", methods=["POST"])
def api_paper_order():
    if not paper_broker.enabled:
        return jsonify({"ok": False, "error": "paper trading disabled"}), 400
    try:
        payload = request.get_json(force=True, silent=False)
        symbol = str(payload.get("symbol", "")).strip().upper()
        qty = int(payload.get("qty") or payload.get("quantity") or 0)
        side = str(payload.get("side", "buy")).lower()
        order_type = str(payload.get("type", payload.get("order_type", "market"))).lower()
        limit_price = payload.get("limit_price")
        tif = str(payload.get("time_in_force", "day"))
        stop_loss_pct = None
        take_profit_pct = None
        if "stop_loss_pct" in payload:
            stop_loss_pct = parse_percent(payload.get("stop_loss_pct"), PAPER_DEFAULT_STOP_LOSS_PCT)
        if "take_profit_pct" in payload:
            take_profit_pct = parse_percent(payload.get("take_profit_pct"), PAPER_DEFAULT_TAKE_PROFIT_PCT)
        order = place_guarded_paper_order(
            symbol,
            qty,
            side,
            order_type,
            float(limit_price) if limit_price else None,
            stop_loss_pct,
            take_profit_pct,
            tif,
        )
        return jsonify({"ok": True, "data": order})
    except Exception as exc:
        logger.exception("Paper order failed")
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.route("/api/health")
def api_health():
    with _lock:
        return jsonify({"tickers_cached": len(_sp500["tickers"]), "recs": _recommendations})


if __name__ == "__main__":
    start_background_jobs()
    app.run(debug=True)
