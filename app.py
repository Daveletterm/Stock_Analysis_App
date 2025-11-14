import math
import os
import re
import json
import random
import threading
import logging
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO
from typing import Dict, Tuple

import requests

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from paper_trading import AlpacaPaperBroker
from market_data import PriceDataError
from market_data import get_price_history as load_price_history

load_dotenv()

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

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
PAPER_MAX_POSITION_NOTIONAL = float(os.getenv("PAPER_MAX_POSITION_NOTIONAL", "5000"))
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
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")


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

_autopilot_state = {
    "enabled": False,
    "strategy": "balanced",
    "risk": "medium",
    "last_run": None,
    "last_actions": [],
    "last_error": None,
}
_autopilot_lock = threading.Lock()
_autopilot_runtime_lock = threading.Lock()

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


_PERIOD_RE = re.compile(r"^(\d+)([a-zA-Z]+)$")


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
    return load_price_history(ticker, start, end, interval=resolved_interval)


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
):
    if not paper_broker.enabled:
        raise RuntimeError("Paper trading credentials are not configured")
    if not TICKER_RE.match(symbol):
        raise ValueError("Enter a valid ticker symbol")
    if qty <= 0:
        raise ValueError("Quantity must be positive")
    if side not in {"buy", "sell"}:
        raise ValueError("Side must be 'buy' or 'sell'")
    if order_type not in {"market", "limit"}:
        raise ValueError("Order type must be market or limit")

    symbol = symbol.upper()
    entry_price = None
    if order_type == "limit":
        if limit_price is None or limit_price <= 0:
            raise ValueError("Provide a positive limit price")
        entry_price = float(limit_price)
    if entry_price is None:
        entry_price = fetch_latest_price(symbol)
    notional = float(entry_price) * qty

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

    if side == "buy":
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
    snapshot.setdefault("strategy", "balanced")
    snapshot.setdefault("risk", "medium")
    actions = snapshot.get("last_actions")
    if isinstance(actions, list):
        snapshot["last_actions"] = actions
    elif actions:
        snapshot["last_actions"] = [str(actions)]
    else:
        snapshot["last_actions"] = []
    return snapshot


def update_autopilot_config(*, enabled: bool | None = None, strategy: str | None = None, risk: str | None = None) -> dict:
    with _autopilot_lock:
        if enabled is not None:
            _autopilot_state["enabled"] = bool(enabled)
        if strategy and strategy in AUTOPILOT_STRATEGIES:
            _autopilot_state["strategy"] = strategy
        if risk and risk in AUTOPILOT_RISK_LEVELS:
            _autopilot_state["risk"] = risk
    return get_autopilot_status()


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


def run_autopilot_cycle() -> None:
    if not paper_broker.enabled:
        return

    with _autopilot_lock:
        config = dict(_autopilot_state)

    if not config.get("enabled"):
        return

    if not _autopilot_runtime_lock.acquire(blocking=False):
        logger.debug("Autopilot cycle skipped; previous cycle still running")
        return

    summary_lines: list[str] = []
    errors: list[str] = []
    try:
        strategy_key = config.get("strategy", "balanced")
        risk_key = config.get("risk", "medium")
        strategy = AUTOPILOT_STRATEGIES.get(strategy_key, AUTOPILOT_STRATEGIES["balanced"])
        risk_cfg = AUTOPILOT_RISK_LEVELS.get(risk_key, AUTOPILOT_RISK_LEVELS["medium"])

        account = paper_broker.get_account()
        equity = safe_float(account.get("equity"), safe_float(account.get("cash")))
        if equity <= 0:
            summary_lines.append("Account equity unavailable; skipping cycle.")
            return

        try:
            positions = list(paper_broker.get_positions())
        except Exception as exc:
            errors.append(f"positions error: {exc}")
            positions = []

        try:
            open_orders = list(paper_broker.list_orders(status="open", limit=200))
        except Exception as exc:
            errors.append(f"orders error: {exc}")
            open_orders = []

        held_symbols: dict[str, dict] = {}
        gross_notional = 0.0
        for pos in positions:
            symbol = str(pos.get("symbol", "")).upper()
            qty = safe_float(pos.get("qty"), safe_float(pos.get("quantity")))
            if not symbol or qty <= 0:
                continue
            held_symbols[symbol] = pos
            gross_notional += abs(safe_float(pos.get("market_value")))

        with _lock:
            recs_snapshot = [dict(r) for r in _recommendations]

        if not recs_snapshot:
            seek_recommendations()
            with _lock:
                recs_snapshot = [dict(r) for r in _recommendations]

        position_multiplier = max(risk_cfg.get("position_multiplier", 1.0), 0.25)
        stop_loss_multiplier = max(risk_cfg.get("stop_loss_multiplier", 1.0), 0.25)
        take_profit_multiplier = max(risk_cfg.get("take_profit_multiplier", 1.0), 0.25)

        max_position_pct = min(strategy.get("max_position_pct", 0.1) * position_multiplier, 0.95)
        max_total_allocation = max(0.1, strategy.get("max_total_allocation", 1.0) * position_multiplier)
        min_entry_notional = max(50.0, strategy.get("min_entry_notional", 200.0))

        # Exit review
        exit_threshold = strategy.get("exit_score", 2.0)
        lookback = strategy.get("lookback", "1y")
        for symbol, pos in held_symbols.items():
            qty = abs(safe_float(pos.get("qty"), safe_float(pos.get("quantity"))))
            if qty < 1:
                continue
            df = _autopilot_prepare_dataframe(symbol, lookback)
            if df is None:
                errors.append(f"no data for {symbol}; exit review skipped")
                continue
            score, reasons = score_stock(df)
            if score < exit_threshold:
                if _autopilot_order_blocked(symbol, open_orders):
                    summary_lines.append(f"Exit pending for {symbol}; open order detected.")
                    continue
                qty_int = int(math.floor(qty))
                if qty_int <= 0:
                    continue
                try:
                    place_guarded_paper_order(symbol, qty_int, "sell", time_in_force="day")
                    summary_lines.append(
                        f"Exit {qty_int} {symbol} (score {score:.2f} < {exit_threshold})"
                    )
                except Exception as exc:
                    errors.append(f"sell {symbol} failed: {exc}")

        # Entry candidates
        held_and_pending = set(held_symbols.keys())
        for order in open_orders:
            try:
                if str(order.get("side", "")).lower() == "buy":
                    held_and_pending.add(str(order.get("symbol", "")).upper())
            except Exception:
                continue

        buy_candidates: list[tuple[str, float]] = []
        for rec in recs_snapshot:
            symbol = str(rec.get("Symbol", "")).upper()
            if not symbol or symbol in held_and_pending:
                continue
            score = safe_float(rec.get("Score"))
            if score < strategy.get("min_score", 3.0):
                continue
            buy_candidates.append((symbol, score))

        if not buy_candidates:
            summary_lines.append("No new symbols met entry criteria.")

        buy_candidates.sort(key=lambda item: item[1], reverse=True)
        max_positions = max(1, int(strategy.get("max_positions", 5)))
        available_slots = max(0, max_positions - len(held_symbols))

        for symbol, score in buy_candidates:
            if available_slots <= 0:
                break
            if _autopilot_order_blocked(symbol, open_orders):
                continue
            try:
                price = fetch_latest_price(symbol)
            except Exception as exc:
                errors.append(f"price {symbol} failed: {exc}")
                continue

            notional_cap_pct = min(max_position_pct, max_total_allocation)
            target_notional = max(min_entry_notional, equity * notional_cap_pct)
            if PAPER_MAX_POSITION_NOTIONAL:
                target_notional = min(target_notional, PAPER_MAX_POSITION_NOTIONAL)
            if gross_notional + target_notional > equity * max_total_allocation:
                summary_lines.append("Max portfolio allocation reached; skipping new entries.")
                break

            qty = int(target_notional // price)
            if qty <= 0:
                continue

            stop_loss_pct = max(0.01, strategy.get("stop_loss_pct", 0.04) * stop_loss_multiplier)
            take_profit_pct = max(0.02, strategy.get("take_profit_pct", 0.1) * take_profit_multiplier)

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
                summary_lines.append(
                    f"Buy {qty} {symbol} (score {score:.2f}, stop {stop_loss_pct*100:.1f}%, take {take_profit_pct*100:.1f}%)"
                )
            except Exception as exc:
                errors.append(f"buy {symbol} failed: {exc}")

        if not summary_lines:
            summary_lines.append("Cycle complete with no trades.")

    except Exception as exc:
        logger.exception("Autopilot cycle failed")
        errors.append(str(exc))
    finally:
        with _autopilot_lock:
            _autopilot_state["last_run"] = datetime.now()
            _autopilot_state["last_actions"] = summary_lines
            _autopilot_state["last_error"] = "; ".join(errors) if errors else None
        _autopilot_runtime_lock.release()

    if errors:
        logger.warning("Autopilot cycle completed with errors: %s", "; ".join(errors))
    else:
        logger.info("Autopilot cycle completed: %s", " | ".join(summary_lines))


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
    rec = (
        "BUY"
        if (latest_close is not None and latest_sma50 is not None and latest_close > latest_sma50)
        else "HOLD"
    )

    out = {
        "Symbol": ticker,
        "Close": round(latest_close, 2) if latest_close is not None else None,
        "RSI": round(latest_rsi, 2) if latest_rsi is not None else None,
        "50-day MA": round(latest_sma50, 2) if latest_sma50 is not None else None,
        "200-day MA": round(latest_sma200, 2) if latest_sma200 is not None else None,
        "Recommendation": rec,
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
    logger.info("Refreshing recommendations…")
    update_sp500()
    with _lock:
        tickers = _sp500["tickers"][:]

    if not tickers:
        logger.warning("S&P 500 cache is empty; skipping recommendation refresh")
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
        except Exception:
            continue

    if not filtered:
        logger.warning("No tickers passed the liquidity filter; keeping existing recommendations")
        return

    def worker(sym: str) -> dict:
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
        except Exception as e:
            logger.exception("Worker failed for %s", sym)
            return {"Symbol": sym, "Recommendation": "HOLD", "Score": 0.0, "Why": [f"error: {e}"]}

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = [res for res in ex.map(worker, filtered) if res]

    # rank by score desc, then symbol
    results.sort(key=lambda x: (-x.get("Score", 0.0), x.get("Symbol", "")))
    buys = [r for r in results if r["Recommendation"] == "BUY"]
    logger.info("Recommendations computed: %d BUY out of %d", len(buys), len(results))
    with _lock:
        global _recommendations
        _recommendations = results[:5]  # top 5 overall, already ranked


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
    should_start = not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if not should_start:
        return
    threading.Thread(target=seek_recommendations, daemon=True).start()
    sched = BackgroundScheduler()
    sched.add_job(seek_recommendations, "interval", hours=1, next_run_time=datetime.now())
    sched.add_job(run_autopilot_cycle, "interval", minutes=5, next_run_time=datetime.now() + timedelta(seconds=30))
    sched.start()
    logger.info("Background jobs started")


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
        recs = _recommendations[:]
    # Pre-serialize for safe JS usage in template
    result_clean = to_plain(result or {})
    recs_clean = to_plain(recs or [])
    result_json = json.dumps(result_clean, ensure_ascii=False)
    recs_json = json.dumps(recs_clean, ensure_ascii=False)
    return render_template("index.html", result_json=result_json, recommendations_json=recs_json)


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
        return jsonify({"ok": True, "data": _recommendations})


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
        threading.Thread(target=run_autopilot_cycle, daemon=True).start()
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

    strategy = payload.get("strategy")
    risk = payload.get("risk")

    status = update_autopilot_config(enabled=enabled, strategy=strategy, risk=risk)
    if status.get("enabled") and paper_broker.enabled:
        threading.Thread(target=run_autopilot_cycle, daemon=True).start()
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
