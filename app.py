import math
import os
import re
import csv
import json
import random
import threading
import logging
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, date, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Literal

import requests
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore

import pandas as pd
try:
    import plotly.express as px  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    px = None
    logging.getLogger("stockapp").warning(
        "plotly not installed; dashboard charts will be unavailable. "
        "Install with 'pip install plotly'."
    )
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from paper_trading import (
    AlpacaAPIError,
    AlpacaPaperBroker,
    PAPER_TRADES_COLUMNS,
    build_paper_trades_export,
    get_daily_account_snapshot,
    NoAvailableBidError,
    OptionCloseRejectedError,
    is_regular_equity_trading_hours,
    option_pnl_percent,
)
from market_data import PriceDataError
from market_data import choose_put_contract, fetch_option_contracts
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
ENABLE_ZOMBIE_DELETE = True

# -----------------------------
# Globals & caches
# -----------------------------
_lock = threading.Lock()
_sp500 = {"tickers": [], "updated": datetime.min}
_price_cache: Dict[Tuple[str, str, str, bool], Tuple[datetime, pd.DataFrame]] = {}
PRICE_CACHE_TTL = timedelta(minutes=15)
_recommendations: dict[str, Any] = {
    "timestamp": None,
    "bullish": [],
    "bearish": [],
    "top": [],
}
_rec_state = {"refreshing": False, "last_completed": None, "last_error": None}
_background_jobs_started = False
_background_jobs_lock = threading.Lock()
_scheduler: BackgroundScheduler | None = None
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")
OPTION_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
OPTION_CONTRACT_MULTIPLIER = int(os.getenv("ALPACA_OPTION_MULTIPLIER", "100")) or 100
MAX_OPTION_NOTIONAL_FRACTION = 0.05  # 5 percent of equity per option trade
MAX_OPTION_CONTRACTS_PER_TRADE = 10  # hard cap on single-trade contract count
OPTION_STOP_LOSS_PCT = 0.5  # close if price falls to 50 percent of entry
OPTION_TAKE_PROFIT_PCT = 1.5  # close if price reaches 150 percent of entry
VOL_RETURN_WINDOW = 10  # number of cycles to use for rolling volatility
VOL_BASELINE_STD = 0.01  # target daily return std for scaling
VOL_POSITION_SCALE_MIN = 0.6  # floor for position scale when volatility spikes
VOL_POSITION_SCALE_MAX = 1.4  # ceiling for position scale when volatility is calm
BALANCED_GROWTH_CONFIG = {
    "max_equity_position_pct": 0.10,
    "max_options_notional_pct": 0.01,
    "max_total_options_pct": 0.10,
    "target_num_equity_positions": 10,
    "equity_trend_lookback": 50,
    "equity_min_relative_strength": 0.0,
    "equity_stop_loss_pct": 0.12,
    "equity_trailing_stop_pct": 0.15,
    "option_max_loss_pct": 0.50,
    "option_min_dte": 20,
    "option_max_dte": 90,
}
AI_LOG_PATH = Path("data/ai_training_log.csv")
AI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
AI_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "asset_class",
    "strategy_key",
    "contract_symbol",
    "direction",
    "score",
    "spot_price",
    "entry_price",
    "rsi",
    "macd",
    "volatility_20d",
    "volume_rel_20d",
    "sector_strength",
    "market_trend",
    "congress_score",
    "news_sentiment",
    "decision",
    "label_good_trade",
    "realized_return_5d",
]
# AI training log captures entry snapshots for future model training.
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
# Autopilot risk profiles
# -----------------------------


@dataclass
class RiskProfile:
    name: str
    max_open_positions: int
    max_new_trades_per_run: int
    base_position_pct_equity_stock: float
    base_position_pct_equity_option: float
    max_pyramids_per_position: int
    winner_add_threshold_plpc: float
    loser_cut_threshold_plpc: float
    idle_cash_target_pct: float
    min_entry_score: float


# -----------------------------
# Autopilot configuration
# -----------------------------

OPTION_MOMENTUM_DEFAULTS = {
    "options_expiry_buffer": 5,
    "options_expiry_window": (14, 60),
    "min_open_interest": 25,
    "min_volume": 2,
    "target_delta": 0.4,
    "min_delta": 0.25,
    "max_delta": 0.65,
    "max_spread_pct": 0.65,
    "max_implied_volatility": 3.0,
    "max_premium_pct_of_spot": 0.35,
    "contract_type": "auto",
    "allow_opposite_contract": True,
    "max_contracts_per_trade": 5,
    "max_position_debit": None,
    "lookback": "1y",
}

HYBRID_STRATEGIES = {
    "hybrid_conservative": {
        **OPTION_MOMENTUM_DEFAULTS,
        "label": "Conservative (hybrid)",
        "description": "Favor shares with occasional options on very strong setups.",
        "max_positions": 2,
        "stock_position_fraction": 0.02,
        "option_position_fraction": 0.01,
        "min_bullish_score": 5.0,
        "min_bearish_score": 3.5,
        "option_bias": "rare",
        "option_stop_loss_pct": 0.30,
        "option_take_profit_pct": 0.60,
        "stock_stop_loss_pct": 0.05,
        "stock_take_profit_pct": 0.12,
        "max_total_allocation": 0.55,
        "min_entry_notional": 200.0,
        "exit_score": 2.2,
    },
    "hybrid_balanced": {
        **OPTION_MOMENTUM_DEFAULTS,
        "label": "Balanced (hybrid)",
        "description": "Blend shares and options depending on price level and score strength.",
        "max_positions": 4,
        "stock_position_fraction": 0.02,
        "option_position_fraction": 0.02,
        "min_bullish_score": 4.5,
        "min_bearish_score": 3.0,
        "option_bias": "mixed",
        "option_stop_loss_pct": 0.40,
        "option_take_profit_pct": 0.80,
        "stock_stop_loss_pct": 0.06,
        "stock_take_profit_pct": 0.15,
        "max_total_allocation": 0.8,
        "min_entry_notional": 200.0,
        "exit_score": 2.0,
    },
    # Options-forward with capped portfolio allocation and tighter default stops
    "hybrid_aggressive": {
        **OPTION_MOMENTUM_DEFAULTS,
        "label": "Aggressive (hybrid)",
        "description": "Prefer options when available; fall back to shares when needed.",
        "max_positions": 5,
        "stock_position_fraction": 0.02,
        "option_position_fraction": 0.05,
        "min_bullish_score": 3.8,
        "min_bearish_score": 2.3,
        "option_bias": "prefer_options",
        "contract_type": "call",
        "option_stop_loss_pct": 0.45,
        "option_take_profit_pct": 1.20,
        "stock_stop_loss_pct": 0.08,
        "stock_take_profit_pct": 0.22,
        "max_total_allocation": 0.95,
        "min_entry_notional": 150.0,
        "exit_score": 1.6,
    },
}

RISK_PROFILES: dict[str, RiskProfile] = {
    "hybrid_conservative": RiskProfile(
        name="conservative",
        max_open_positions=2,
        max_new_trades_per_run=2,
        base_position_pct_equity_stock=0.02,
        base_position_pct_equity_option=0.01,
        max_pyramids_per_position=0,
        winner_add_threshold_plpc=0.08,
        loser_cut_threshold_plpc=-0.08,
        idle_cash_target_pct=0.6,
        min_entry_score=5.0,
    ),
    "hybrid_balanced": RiskProfile(
        name="balanced",
        max_open_positions=4,
        max_new_trades_per_run=4,
        base_position_pct_equity_stock=0.02,
        base_position_pct_equity_option=0.02,
        max_pyramids_per_position=0,
        winner_add_threshold_plpc=0.07,
        loser_cut_threshold_plpc=-0.07,
        idle_cash_target_pct=0.45,
        min_entry_score=4.5,
    ),
    "hybrid_aggressive": RiskProfile(
        name="high",
        max_open_positions=25,
        max_new_trades_per_run=10,
        base_position_pct_equity_stock=0.05,
        base_position_pct_equity_option=0.02,
        max_pyramids_per_position=3,
        winner_add_threshold_plpc=0.03,
        loser_cut_threshold_plpc=-0.04,
        idle_cash_target_pct=0.20,
        min_entry_score=3.8,
    ),
}

HYBRID_STRATEGY_ALIASES = {
    "conservative": "hybrid_conservative",
    "balanced": "hybrid_balanced",
    "aggressive": "hybrid_aggressive",
    "high": "hybrid_aggressive",
    "options_momentum": "hybrid_balanced",
}

AUTOPILOT_STRATEGIES = HYBRID_STRATEGIES

AUTOPILOT_RISK_LEVELS = {
    "low": {
        "label": "Low",
        "position_multiplier": 0.7,
        "stop_loss_multiplier": 0.9,
        "take_profit_multiplier": 0.9,
    },
    "medium": {
        "label": "Medium",
        "position_multiplier": 1.0,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.0,
    },
    "high": {
        "label": "High",
        "position_multiplier": 1.25,
        "stop_loss_multiplier": 1.0,
        "take_profit_multiplier": 1.35,
    },
}

WINNER_OPTION_THRESHOLD_MULTIPLIER = 5
LOSER_OPTION_THRESHOLD_MULTIPLIER = 7
PYRAMID_POSITION_SCALE = 0.5

AUTOPILOT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "autopilot_state.json")
DEFAULT_HYBRID_STRATEGY = "hybrid_balanced"

_autopilot_state = {
    "enabled": False,
    "paused": False,
    "strategy": DEFAULT_HYBRID_STRATEGY,
    "risk": "medium",
    "last_run": None,
    "last_actions": [],
    "last_error": None,
}
_autopilot_lock = threading.Lock()
_autopilot_runtime_lock = threading.Lock()
_autopilot_last_run: dict[str, Any] | None = None
_autopilot_uncovered_exits: set[str] = set()
_autopilot_stale_exits: set[str] = set()
_autopilot_option_cooldowns: dict[str, datetime] = {}
_autopilot_position_params: dict[str, Any] = {}
_autopilot_pyramid_counts: dict[str, int] = {}
_autopilot_return_history: list[float] = []
_autopilot_last_equity: float | None = None
ZOMBIE_POSITIONS: set[str] = set()


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


def _normalize_recommendations_payload(payload: Any) -> dict[str, Any]:
    """Ensure recommendation payloads are stored with bullish/bearish buckets."""

    bullish: list[dict[str, Any]] = []
    bearish: list[dict[str, Any]] = []
    timestamp = None

    if isinstance(payload, dict):
        bullish = [dict(r) for r in payload.get("bullish", []) if isinstance(r, dict)]
        bearish = [dict(r) for r in payload.get("bearish", []) if isinstance(r, dict)]
        timestamp = payload.get("timestamp")
        if not bullish and isinstance(payload.get("top"), list):
            bullish = [dict(r) for r in payload.get("top", []) if isinstance(r, dict)]
    elif isinstance(payload, list):
        bullish = [dict(r) for r in payload if isinstance(r, dict)]

    top = bullish[:]
    return {"timestamp": timestamp, "bullish": bullish, "bearish": bearish, "top": top}


def _load_latest_recommendations_from_db() -> None:
    try:
        conn = _get_db_connection()
        cur = conn.execute(
            "SELECT payload, created_at FROM recommendation_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            return
        payload = _normalize_recommendations_payload(json.loads(row["payload"]))
        created_at = row["created_at"]
        with _lock:
            _recommendations.clear()
            _recommendations.update(payload)
            _rec_state["last_completed"] = datetime.fromisoformat(created_at)
            _rec_state["last_error"] = None
        logger.info("Loaded cached recommendations snapshot from %s", created_at)
    except Exception:
        logger.exception("Failed to load cached recommendations from sqlite")


def _record_recommendations_snapshot(recs: dict[str, Any]) -> None:
    try:
        conn = _get_db_connection()
        with conn:
            conn.execute(
                "INSERT INTO recommendation_snapshots (created_at, payload) VALUES (?, ?)",
                (datetime.now().isoformat(), json.dumps(to_plain(recs))),
            )
    except Exception:
        logger.exception("Failed to persist recommendation snapshot")


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


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


def _resolve_hybrid_strategy_key(strategy: str | None) -> str:
    if not strategy:
        return DEFAULT_HYBRID_STRATEGY
    key = str(strategy).strip()
    if key in HYBRID_STRATEGIES:
        return key
    if key in HYBRID_STRATEGY_ALIASES:
        return HYBRID_STRATEGY_ALIASES[key]
    logger.warning(
        "Unsupported autopilot strategy '%s'; falling back to %s",
        key,
        DEFAULT_HYBRID_STRATEGY,
    )
    return DEFAULT_HYBRID_STRATEGY


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
        if "strategy" in saved and isinstance(saved.get("strategy"), str):
            _autopilot_state["strategy"] = _resolve_hybrid_strategy_key(saved.get("strategy"))
        if "risk" in saved and isinstance(saved.get("risk"), str):
            _autopilot_state["risk"] = str(saved.get("risk"))
        if "position_params" in saved and isinstance(saved.get("position_params"), dict):
            _autopilot_position_params.clear()
            _autopilot_position_params.update(saved.get("position_params"))
        if "return_history" in saved and isinstance(saved.get("return_history"), list):
            _autopilot_return_history.clear()
            for val in saved.get("return_history"):
                val_float = safe_float(val, None)
                if val_float is not None:
                    _autopilot_return_history.append(val_float)
        if "last_equity" in saved:
            _autopilot_last_equity = safe_float(saved.get("last_equity"), None)


def _persist_autopilot_state() -> None:
    """Persist the current autopilot settings to disk."""

    with _autopilot_lock:
        payload = {
            "enabled": bool(_autopilot_state.get("enabled")),
            "paused": bool(_autopilot_state.get("paused")),
            "strategy": _autopilot_state.get("strategy"),
            "risk": _autopilot_state.get("risk"),
            "position_params": _autopilot_position_params,
            "return_history": _autopilot_return_history,
            "last_equity": _autopilot_last_equity,
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


def _fallback_recommendations(reason: str | None = None) -> dict[str, Any]:
    """Return placeholder picks so the UI isn't empty when data is unreachable."""

    why_text = reason or "Data unavailable; using placeholder symbols."
    bullish = [
        {
            "Symbol": sym,
            "Recommendation": "HOLD",
            "Score": 0.0,
            "BearishScore": 0.0,
            "Why": [why_text],
            "WhyBearish": [why_text],
        }
        for sym in FALLBACK_TICKERS[:5]
    ]
    return {
        "timestamp": None,
        "bullish": bullish,
        "bearish": [],
        "top": bullish,
    }


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


def score_equity_growth(symbol: str, df: pd.DataFrame, *, benchmark: pd.DataFrame | None = None) -> tuple[float, list[str]]:
    """Growth-oriented equity score emphasizing trend, RS, and sane volatility."""

    reasons: list[str] = []
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0, ["insufficient data"]
    df_local = df.copy()
    lookback = max(10, int(BALANCED_GROWTH_CONFIG.get("equity_trend_lookback", 50)))
    df_local["ma_trend"] = df_local["Close"].rolling(lookback).mean()
    price = safe_float(df_local["Close"].iloc[-1], None)
    ma = safe_float(df_local["ma_trend"].iloc[-1], None)
    score = 0.0

    if price and ma and price > ma:
        score += 2.0
        reasons.append("above trend")
    else:
        reasons.append("below trend")

    if len(df_local) > lookback + 5 and ma:
        prev_ma = safe_float(df_local["ma_trend"].iloc[-5], None)
        if prev_ma and ma > prev_ma:
            score += 1.0
            reasons.append("trend rising")
        else:
            reasons.append("trend flat/declining")

    rs_contrib = 0.0
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        try:
            span = min(len(df_local), 90)
            rs_window = min(span, 60)
            sym_ret = df_local["Close"].pct_change(rs_window).iloc[-1]
            bench_ret = benchmark["Close"].pct_change(rs_window).iloc[-1]
            rel = sym_ret - bench_ret
            if rel > BALANCED_GROWTH_CONFIG.get("equity_min_relative_strength", 0.0):
                score += 1.0
                reasons.append(f"rs positive {rel:.2%}")
            else:
                reasons.append(f"rs weak {rel:.2%}")
            rs_contrib = rel * 10
        except Exception:
            reasons.append("rs unavailable")
    score += rs_contrib

    if len(df_local) >= 30:
        vol30 = df_local["Close"].pct_change().rolling(30).std().iloc[-1]
        if vol30 and vol30 > 0.07:
            score -= 2.0
            reasons.append("volatility high")
        else:
            reasons.append("volatility acceptable")

    vol_col = "Volume" if "Volume" in df_local.columns else ("volume" if "volume" in df_local.columns else None)
    if vol_col and len(df_local) >= 20:
        try:
            vol_rel = df_local[vol_col].iloc[-1] / df_local[vol_col].rolling(20).mean().iloc[-1]
            if vol_rel < 0.8:
                score -= 0.5
                reasons.append("volume light")
            else:
                reasons.append("volume healthy")
        except Exception:
            reasons.append("volume check failed")

    return round(float(score), 2), reasons


def score_equity_growth(symbol: str, df: pd.DataFrame, *, benchmark: pd.DataFrame | None = None) -> tuple[float, list[str]]:
    """Growth-oriented equity score emphasizing trend, RS, and sane volatility."""

    reasons: list[str] = []
    if df is None or df.empty or "Close" not in df.columns:
        return 0.0, ["insufficient data"]
    df_local = df.copy()
    lookback = max(10, int(BALANCED_GROWTH_CONFIG.get("equity_trend_lookback", 50)))
    df_local["ma_trend"] = df_local["Close"].rolling(lookback).mean()
    price = safe_float(df_local["Close"].iloc[-1], None)
    ma = safe_float(df_local["ma_trend"].iloc[-1], None)
    score = 0.0

    if price and ma and price > ma:
        score += 2.0
        reasons.append("above trend")
    else:
        reasons.append("below trend")

    if len(df_local) > lookback + 5 and ma:
        prev_ma = safe_float(df_local["ma_trend"].iloc[-5], None)
        if prev_ma and ma > prev_ma:
            score += 1.0
            reasons.append("trend rising")
        else:
            reasons.append("trend flat/declining")

    rs_contrib = 0.0
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        try:
            span = min(len(df_local), 90)
            rs_window = min(span, 60)
            sym_ret = df_local["Close"].pct_change(rs_window).iloc[-1]
            bench_ret = benchmark["Close"].pct_change(rs_window).iloc[-1]
            rel = sym_ret - bench_ret
            if rel > BALANCED_GROWTH_CONFIG.get("equity_min_relative_strength", 0.0):
                score += 1.0
                reasons.append(f"rs positive {rel:.2%}")
            else:
                reasons.append(f"rs weak {rel:.2%}")
            rs_contrib = rel * 10
        except Exception:
            reasons.append("rs unavailable")
    score += rs_contrib

    if len(df_local) >= 30:
        vol30 = df_local["Close"].pct_change().rolling(30).std().iloc[-1]
        if vol30 and vol30 > 0.07:
            score -= 2.0
            reasons.append("volatility high")
        else:
            reasons.append("volatility acceptable")

    vol_col = "Volume" if "Volume" in df_local.columns else ("volume" if "volume" in df_local.columns else None)
    if vol_col and len(df_local) >= 20:
        try:
            vol_rel = df_local[vol_col].iloc[-1] / df_local[vol_col].rolling(20).mean().iloc[-1]
            if vol_rel < 0.8:
                score -= 0.5
                reasons.append("volume light")
            else:
                reasons.append("volume healthy")
        except Exception:
            reasons.append("volume check failed")

    return round(float(score), 2), reasons


def score_bearish(df: pd.DataFrame) -> tuple[float, list[str]]:
    """Return a bearish bias score using the same core indicators."""

    reasons: list[str] = []
    if df.empty:
        return 0.0, ["insufficient price history"]
    row = df.iloc[-1]
    score = 0.0

    if pd.notna(row.get("SMA_200")) and row["Close"] < row["SMA_200"]:
        score += 2.0
        reasons.append("price below 200d MA")
    else:
        reasons.append("above 200d MA")

    if pd.notna(row.get("SMA_50")) and row["SMA_50"] < row["SMA_200"]:
        score += 1.0
        reasons.append("50d MA < 200d MA")

    rsi_val = row.get("RSI") or 0
    if rsi_val >= 70:
        score += 1.0
        reasons.append("RSI overbought")
    elif rsi_val <= 35:
        score += 0.5
        reasons.append("RSI weak (<35)")

    if pd.notna(row.get("LL_20")) and row["Close"] <= row["LL_20"] * 1.001:
        score += 1.0
        reasons.append("near 20d low breakdown")

    if pd.notna(row.get("ATR_pct")) and row["ATR_pct"] < 0.05:
        score += 0.25
        reasons.append("contained volatility")

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
    snapshot.setdefault("strategy", DEFAULT_HYBRID_STRATEGY)
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
        if strategy:
            _autopilot_state["strategy"] = _resolve_hybrid_strategy_key(strategy)
        if risk and risk in AUTOPILOT_RISK_LEVELS:
            _autopilot_state["risk"] = risk
    _persist_autopilot_state()
    return get_autopilot_status()


def _autopilot_order_blocked(symbol: str, open_orders: list[dict]) -> bool:
    active_status = {"new", "accepted", "open", "pending_new", "partially_filled"}
    active_buy = 0
    for order in open_orders:
        try:
            if str(order.get("symbol", "")).upper() != symbol.upper():
                continue
            status = str(order.get("status", "")).lower()
            side = str(order.get("side", "")).lower()
            if status in active_status and side == "buy":
                active_buy += 1
                if active_buy >= 1:
                    return True
        except Exception:
            continue
    return False


def find_bearish_candidates(
    recs: list[dict],
    *,
    lookback: str,
    min_score: float,
    held_underlyings: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return symbols that meet the bearish threshold using existing indicators."""

    held = held_underlyings or set()
    bearish: list[tuple[str, float]] = []
    seen: set[str] = set()
    for rec in recs:
        symbol = str(rec.get("Symbol", "")).upper()
        if not symbol or symbol in seen or symbol in held:
            continue
        seen.add(symbol)
        df = _autopilot_prepare_dataframe(symbol, lookback)
        if df is None:
            continue
        score, reasons = score_bearish(df)
        if score >= min_score:
            bearish.append((symbol, score))
            logger.info(
                "Bearish candidate %s score %.2f (%s)",
                symbol,
                score,
                "; ".join(reasons[:3]),
            )
    bearish.sort(key=lambda item: item[1], reverse=True)
    return bearish


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


def _set_option_cooldown(underlying: str, now: datetime, minutes: int = 60) -> None:
    """Prevent new option entries on this underlying for a short window."""

    if not underlying:
        return
    underlying = underlying.upper()
    _autopilot_option_cooldowns[underlying] = now + timedelta(minutes=minutes)


def _option_on_cooldown(underlying: str, now: datetime) -> bool:
    """Return True if this underlying is in a cooldown window."""

    if not underlying:
        return False
    underlying = underlying.upper()
    until = _autopilot_option_cooldowns.get(underlying)
    if not until:
        return False
    if until <= now:
        # Cooldown expired, clear it
        _autopilot_option_cooldowns.pop(underlying, None)
        return False
    return True


def is_recently_traded(symbol: str) -> bool:
    """Return True if the account traded *symbol* during the current day."""

    symbol_cmp = symbol.replace(" ", "").upper()
    try:
        activities = list(paper_broker.get_activities(activity_types="FILL", date=date.today()))
    except Exception as exc:
        logger.debug("recent trade check failed for %s: %s", symbol_cmp, exc)
        return False
    for act in activities or []:
        try:
            act_symbol = str(act.get("symbol", "")).replace(" ", "").upper()
            if act_symbol != symbol_cmp:
                continue
            return True
        except Exception:
            continue
    return False


def _recently_closed_symbols() -> set[str]:
    """Best-effort detection of symbols closed earlier today to avoid wash trades."""

    closed: set[str] = set()
    try:
        activities = list(paper_broker.get_activities(activity_types="FILL", date=date.today()))
    except Exception as exc:
        logger.debug("recent close lookup failed: %s", exc)
        return closed
    for act in activities or []:
        try:
            act_symbol = str(act.get("symbol", "")).replace(" ", "").upper()
            side = str(act.get("side", "")).lower()
            qty = safe_float(act.get("qty"), None)
            if not act_symbol:
                continue
            if side.startswith("sell") or (qty is not None and qty < 0):
                closed.add(act_symbol)
        except Exception:
            continue
    return closed


def _option_trade_block_reason(
    contract_symbol: str,
    positions_snapshot: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    closed_today: set[str],
    *,
    now: datetime,
) -> str | None:
    """Return reason string if a trade should be blocked for wash/zombie protection."""

    symbol_cmp = contract_symbol.replace(" ", "").upper()
    if symbol_cmp in ZOMBIE_POSITIONS:
        return "zombie contract flagged"

    for pos in positions_snapshot:
        try:
            if str(pos.get("symbol", "")).replace(" ", "").upper() != symbol_cmp:
                continue
            qty = _option_position_quantity(pos)
            status = str(pos.get("status", "")).lower()
            if qty > 0:
                return "position already open"
            if qty == 0 and status and status != "closed":
                ZOMBIE_POSITIONS.add(symbol_cmp)
                return "alpaca reports zero qty but position not closed"
        except Exception:
            continue

    active_orders = paper_broker.list_open_orders_for_symbol(
        symbol_cmp, asset_class="option", orders=open_orders
    )
    if active_orders:
        return "pending order already exists"
    if symbol_cmp in closed_today:
        return "closed earlier today"
    if is_recently_traded(symbol_cmp):
        return "recently traded today"
    return None


def _mark_zombie_position(
    symbol: str, *, reason: str | None = None, summary_lines: list[str] | None = None
) -> None:
    """Mark an option contract as a zombie so autopilot ignores it going forward."""

    contract_symbol = symbol.replace(" ", "").upper()
    if not contract_symbol or contract_symbol in ZOMBIE_POSITIONS:
        return
    ZOMBIE_POSITIONS.add(contract_symbol)
    _autopilot_uncovered_exits.discard(contract_symbol)
    _autopilot_stale_exits.discard(contract_symbol)
    message = f"Marking {contract_symbol} as zombie; {reason or 'alpaca rejection'}"
    logger.warning(message)
    if summary_lines is not None:
        summary_lines.append(f"{contract_symbol} marked zombie ({reason or 'ignored'})")


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
    raw_chain_sizes: dict[str, int] = {}
    priced_counts: dict[str, int] = {}
    chain_filtered_counts: dict[str, int] = {}
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
        raw_count = len(chain)
        filtered_chain: list[dict[str, Any]] = []
        filtered_out = 0
        for contract in chain:
            status_value = str(contract.get("status", "") or contract.get("state", "")).lower()
            if status_value and status_value != "active":
                filtered_out += 1
                continue
            bid, ask = option_bid_ask(contract)
            bid_val = safe_float(bid, None)
            ask_val = safe_float(ask, None)
            price, has_price_fields = _derive_option_price(
                contract,
                bid=bid_val,
                ask=ask_val,
            )
            if not has_price_fields or price is None or price <= 0:
                filtered_out += 1
                continue
            filtered_chain.append(contract)
        if filtered_out:
            logger.info(
                "Option chain filtered %d/%d %s contracts for %s due to quote/liquidity rules",
                filtered_out,
                raw_count,
                option_type,
                symbol.upper(),
            )
        chain_filtered_counts[option_type] = filtered_out
        chain = filtered_chain
        priced = 0
        for contract in chain:
            price, _ = _derive_option_price(contract)
            if price is not None and price > 0:
                priced += 1
        chains[option_type] = chain
        chain_sizes[option_type] = len(chain)
        raw_chain_sizes[option_type] = raw_count
        priced_counts[option_type] = priced
    diagnostics["chain_sizes"] = chain_sizes
    diagnostics["chain_raw_sizes"] = raw_chain_sizes
    diagnostics["chain_filtered_out"] = chain_filtered_counts
    diagnostics["priced_contracts"] = priced_counts
    if chain_errors:
        diagnostics["chain_errors"] = chain_errors

    total_raw_contracts = sum(raw_chain_sizes.values())
    total_contracts = sum(chain_sizes.values())
    total_filtered_contracts = sum(chain_filtered_counts.values())
    total_priced_contracts = sum(priced_counts.values())
    diagnostics["total_raw_contracts"] = total_raw_contracts
    diagnostics["total_contracts"] = total_contracts
    diagnostics["total_filtered_contracts"] = total_filtered_contracts
    diagnostics["total_priced_contracts"] = total_priced_contracts
    logger.info(
        "Option chain summary for %s: kept %d/%d after filters (%d priced, %d filtered out)",
        symbol.upper(),
        total_contracts,
        total_raw_contracts,
        total_priced_contracts,
        total_filtered_contracts,
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


def _remember_position_params(
    symbol: str,
    *,
    asset_class: str,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    strategy: str,
    opened_at: datetime | None = None,
    highest_price: float | None = None,
) -> None:
    """Store stop/take parameters for a position so exits use entry-time values."""

    if not symbol:
        return
    payload = {
        "asset_class": asset_class,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "strategy": strategy,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "opened_at": (opened_at or datetime.now(timezone.utc)).isoformat(),
    }
    if highest_price is not None:
        payload["highest_price"] = highest_price
    with _autopilot_lock:
        _autopilot_position_params[symbol.upper()] = payload
    _persist_autopilot_state()


def _resolve_risk_profile(strategy_key: str) -> RiskProfile:
    resolved = _resolve_hybrid_strategy_key(strategy_key)
    return RISK_PROFILES.get(resolved, RISK_PROFILES[DEFAULT_HYBRID_STRATEGY])


def _prune_pyramid_counts(active_symbols: set[str]) -> None:
    with _autopilot_lock:
        stale = set(_autopilot_pyramid_counts) - active_symbols
        for sym in stale:
            _autopilot_pyramid_counts.pop(sym, None)


def _position_unrealized_plpc(pos: dict[str, Any]) -> float | None:
    plpc = safe_float(pos.get("unrealized_plpc"), None)
    if plpc is not None:
        return plpc
    percent = safe_float(pos.get("unrealized_pl_percent"), None)
    if percent is not None:
        return percent / 100.0
    unrealized_pl = safe_float(pos.get("unrealized_pl"), None)
    market_value = abs(safe_float(pos.get("market_value"), 0.0))
    if unrealized_pl is not None and market_value:
        return unrealized_pl / market_value
    return None


def _position_price(pos: dict[str, Any], asset_class: str) -> float | None:
    price = safe_float(pos.get("current_price"), None)
    if price is None:
        price = safe_float(pos.get("market_price"), None)
    if price is None:
        price = safe_float(pos.get("avg_entry_price"), None)
    if price is None and str(asset_class).lower() == "option":
        qty = abs(safe_float(pos.get("qty"), 0.0)) or 1.0
        mv = abs(safe_float(pos.get("market_value"), 0.0))
        if qty:
            price = mv / (qty * OPTION_CONTRACT_MULTIPLIER)
    return price


def _update_return_history(current_equity: float) -> float | None:
    """Track rolling equity returns for volatility-aware sizing."""

    global _autopilot_last_equity
    prev = _autopilot_last_equity
    ret = None
    if prev and prev > 0 and current_equity > 0:
        ret = (current_equity - prev) / prev
    with _autopilot_lock:
        _autopilot_last_equity = current_equity
        if ret is not None:
            _autopilot_return_history.append(ret)
            if len(_autopilot_return_history) > VOL_RETURN_WINDOW:
                _autopilot_return_history[:] = _autopilot_return_history[-VOL_RETURN_WINDOW:]
        _autopilot_state["last_equity"] = current_equity
        _autopilot_state["return_history"] = list(_autopilot_return_history)
    _persist_autopilot_state()
    return ret


def _volatility_position_scale() -> float:
    """Return a multiplier for sizing based on recent return volatility."""

    with _autopilot_lock:
        history = list(_autopilot_return_history)
    if len(history) < 2:
        return 1.0
    if np is not None:
        std_dev = float(np.std(history))
    else:
        std_dev = float(statistics.pstdev(history))
    baseline = VOL_BASELINE_STD or 0.01
    scale = 1.0 + (baseline - std_dev) / max(baseline, 1e-6) * 0.5
    return max(VOL_POSITION_SCALE_MIN, min(VOL_POSITION_SCALE_MAX, scale))


def _extract_indicator_features(df: pd.DataFrame | None) -> dict[str, float | None]:
    """Return a best-effort snapshot of indicators for AI logging."""

    features: dict[str, float | None] = {
        "rsi": None,
        "macd": None,
        "volatility_20d": None,
        "volume_rel_20d": None,
    }
    if df is None or df.empty:
        return features
    try:
        if "rsi_14" in df.columns:
            features["rsi"] = safe_float(df["rsi_14"].iloc[-1], None)
        if "macd" in df.columns:
            features["macd"] = safe_float(df["macd"].iloc[-1], None)
        close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
        volume_col = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
        if close_col:
            close_series = df[close_col]
            if len(close_series) >= 20:
                features["volatility_20d"] = safe_float(
                    close_series.pct_change().rolling(20).std().iloc[-1],
                    None,
                )
        if volume_col:
            volume_series = df[volume_col]
            if len(volume_series) >= 20:
                rolling_mean = volume_series.rolling(20).mean().iloc[-1]
                if rolling_mean and rolling_mean != 0:
                    features["volume_rel_20d"] = safe_float(
                        volume_series.iloc[-1] / rolling_mean,
                        None,
                    )
    except Exception:
        logger.debug("Failed to extract indicator features", exc_info=True)
    return features


def log_ai_snapshot(
    *,
    symbol: str,
    asset_class: str,
    strategy_key: str,
    contract_symbol: str | None,
    direction: str,
    score: float,
    spot_price: float | None,
    entry_price: float | None,
    rsi: float | None = None,
    macd: float | None = None,
    volatility_20d: float | None = None,
    volume_rel_20d: float | None = None,
    sector_strength: float | None = None,
    market_trend: str | None = None,
    congress_score: float | None = None,
    news_sentiment: float | None = None,
    decision: str = "enter",
    label_good_trade: str | None = "",
    realized_return_5d: str | None = "",
) -> None:
    """
    Append one training example row for a future AI portfolio model.

    This must never raise uncaught exceptions. If logging fails, log a warning
    and continue normal execution.
    """

    try:
        new_file = not AI_LOG_PATH.exists()
        with AI_LOG_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AI_LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "asset_class": asset_class,
                    "strategy_key": strategy_key,
                    "contract_symbol": contract_symbol or "",
                    "direction": direction,
                    "score": float(score),
                    "spot_price": "" if spot_price is None else float(spot_price),
                    "entry_price": "" if entry_price is None else float(entry_price),
                    "rsi": "" if rsi is None else float(rsi),
                    "macd": "" if macd is None else float(macd),
                    "volatility_20d": "" if volatility_20d is None else float(volatility_20d),
                    "volume_rel_20d": "" if volume_rel_20d is None else float(volume_rel_20d),
                    "sector_strength": "" if sector_strength is None else float(sector_strength),
                    "market_trend": market_trend or "",
                    "congress_score": "" if congress_score is None else float(congress_score),
                    "news_sentiment": "" if news_sentiment is None else float(news_sentiment),
                    "decision": decision,
                    "label_good_trade": label_good_trade or "",
                    "realized_return_5d": realized_return_5d or "",
                }
            )
    except Exception:
        logger.exception("Failed to log AI snapshot for %s", symbol)

def compute_position_qty(
    symbol: str,
    asset_class: str,
    price: float,
    account_equity: float,
    profile: RiskProfile,
    *,
    position_multiplier: float = 1.0,
    min_entry_notional: float | None = None,
    scale: float = 1.0,
    contract_multiplier: int | None = None,
) -> tuple[int, float]:
    """Return quantity and target notional for an entry based on risk settings."""

    if price is None or price <= 0 or account_equity <= 0:
        return 0, 0.0

    asset_class_norm = str(asset_class or "").lower()
    is_option = "option" in asset_class_norm
    target_pct = (
        profile.base_position_pct_equity_option
        if is_option
        else profile.base_position_pct_equity_stock
    )
    # stock_position_fraction and option_position_fraction define the target equity fraction
    # for new entries; risk multipliers adjust them by position_multiplier.
    target_pct = max(target_pct * position_multiplier, 0.0)

    target_notional = account_equity * target_pct * max(scale, 0.0)
    if min_entry_notional:
        target_notional = max(target_notional, min_entry_notional)

    multiplier = contract_multiplier or (OPTION_CONTRACT_MULTIPLIER if is_option else 1)
    qty = int(target_notional // (price * multiplier)) if price > 0 else 0
    qty = max(qty, 1) if target_notional > 0 else 0
    return qty, target_notional


def _prune_position_params(active_symbols: set[str]) -> None:
    """Drop stored params for positions that are no longer active."""

    if not active_symbols:
        active_symbols = set()
    active_upper = {s.upper() for s in active_symbols}
    with _autopilot_lock:
        stale = [sym for sym in _autopilot_position_params if sym not in active_upper]
        for sym in stale:
            _autopilot_position_params.pop(sym, None)
    if stale:
        _persist_autopilot_state()


def choose_instrument_for_candidate(
    *,
    bias: str,
    hybrid_params: dict[str, Any],
    has_option: bool,
    score: float,
    min_score: float,
    equity_price: float | None,
) -> tuple[Literal["stock", "option", "skip"], str]:
    """Return instrument choice and reason for a candidate."""

    bias = bias or "bullish"
    option_bias = str(hybrid_params.get("option_bias", "mixed")).lower()
    strong_signal = score >= (min_score + 0.5)
    price_high = equity_price is not None and equity_price > 100

    if bias == "bearish":
        if has_option:
            return "option", "bearish path uses options"
        return "skip", "no suitable option and bearish path requires options"

    if option_bias == "rare":
        if has_option and strong_signal:
            return "option", "very strong signal permits option"
        return "stock", "prefers shares by default"

    if option_bias == "mixed":
        if has_option and (price_high or strong_signal):
            return "option", "price/score favor option"
        return "stock", "balanced bias leans to shares"

    if option_bias == "prefer_options":
        if has_option:
            return "option", "prefers options when available"
        return "stock", "no suitable option; falling back to shares"

    return "stock", "defaulting to shares"


def run_autopilot_cycle(force: bool = False) -> None:
    logger.info("Autopilot cycle starting%s", " (forced)" if force else "")

    if not _autopilot_runtime_lock.acquire(blocking=force):
        logger.debug("Autopilot cycle skipped; previous cycle still running")
        return

    now = datetime.now(timezone.utc)
    summary_lines: list[str] = []
    errors: list[str] = []
    orders_placed = 0
    candidate_count = 0
    vol_position_scale = 1.0
    try:
        with _autopilot_lock:
            config = dict(_autopilot_state)
        strategy_key_raw = config.get("strategy", DEFAULT_HYBRID_STRATEGY)
        strategy_key = _resolve_hybrid_strategy_key(strategy_key_raw)
        if strategy_key != strategy_key_raw:
            logger.info("Migrated autopilot strategy %s -> %s", strategy_key_raw, strategy_key)
            with _autopilot_lock:
                _autopilot_state["strategy"] = strategy_key
                _persist_autopilot_state()
        hybrid = HYBRID_STRATEGIES.get(strategy_key, HYBRID_STRATEGIES[DEFAULT_HYBRID_STRATEGY])
        strategy = hybrid
        profile = _resolve_risk_profile(strategy_key)
        risk_key = config.get("risk", "medium")
        risk_cfg = AUTOPILOT_RISK_LEVELS.get(risk_key, AUTOPILOT_RISK_LEVELS["medium"])
        logger.info(
            "Autopilot config loaded: enabled=%s paused=%s strategy=%s risk=%s",
            config.get("enabled"),
            config.get("paused"),
            strategy_key,
            risk_key,
        )

        if config.get("enabled") is False or config.get("paused") is True:
            summary_lines.append("Autopilot is paused.")
            return

        prerequisites_met = True
        if not paper_broker.enabled:
            summary_lines.append("Paper trading disabled; autopilot idle.")
            prerequisites_met = False

        equity = 0.0
        cash_balance = 0.0
        positions: list[dict] = []
        open_orders: list[dict] = []
        recs_snapshot: list[dict] = []

        if prerequisites_met:
            try:
                account = paper_broker.get_account()
            except Exception as exc:
                logger.exception("Autopilot failed to fetch account")
                errors.append(f"account error: {exc}")
                prerequisites_met = False
            else:
                equity = safe_float(account.get("equity"), safe_float(account.get("cash")))
                cash_balance = safe_float(account.get("cash"), 0.0)
                if equity <= 0:
                    summary_lines.append("Account equity unavailable; skipping cycle.")
                    prerequisites_met = False
                else:
                    _update_return_history(equity)
                    vol_position_scale = _volatility_position_scale()

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
            closed_today_symbols = _recently_closed_symbols()

            held_positions: dict[str, dict[str, dict[str, Any]]] = {"equity": {}, "option": {}}
            gross_equity_notional = 0.0
            gross_option_notional = 0.0
            held_option_underlyings: set[str] = set()
            held_put_underlyings: set[str] = set()
            active_symbols: set[str] = set()

            for pos in positions:
                symbol_raw = str(pos.get("symbol", ""))
                symbol = symbol_raw.replace(" ", "").upper()
                qty = safe_float(pos.get("qty"), safe_float(pos.get("quantity")))
                if not symbol or qty <= 0:
                    continue
                if symbol in ZOMBIE_POSITIONS:
                    logger.debug("Skipping zombie position %s from autopilot snapshot", symbol)
                    continue
                active_symbols.add(symbol)
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
                            if parsed.get("type") == "put":
                                held_put_underlyings.add(underlying)
                    held_positions["option"][symbol] = entry
                    gross_option_notional += market_value
                else:
                    held_positions["equity"][symbol] = entry
                    gross_equity_notional += market_value

            _prune_pyramid_counts(active_symbols)
            options_budget_limit = equity * BALANCED_GROWTH_CONFIG["max_total_options_pct"] if equity else 0.0
            options_budget_remaining = max(0.0, options_budget_limit - gross_option_notional)

            pending_order_symbols = {
                str(o.get("symbol", "")).replace(" ", "").upper()
                for o in open_orders
                if str(o.get("side", "")).lower() == "buy"
            }
            _prune_position_params(active_symbols | pending_order_symbols)

            with _lock:
                recs_payload = _normalize_recommendations_payload(_recommendations)

            if not recs_payload.get("bullish"):
                seek_recommendations()
                with _lock:
                    recs_payload = _normalize_recommendations_payload(_recommendations)
            if not recs_payload.get("bullish"):
                recs_payload = _fallback_recommendations(
                    "No recommendations yet; using placeholders for autopilot"
                )

            bullish_recs = [dict(r) for r in recs_payload.get("bullish", [])]
            bearish_recs = [dict(r) for r in recs_payload.get("bearish", [])]
            bullish_slice = bullish_recs[:8]
            bearish_slice = bearish_recs[:5]
            candidate_entries: list[dict[str, Any]] = []
            for rec in bullish_slice:
                symbol = str(rec.get("Symbol", "")).upper()
                if not symbol:
                    continue
                candidate_entries.append(
                    {
                        "symbol": symbol,
                        "score": safe_float(rec.get("Score")),
                        "bias": "bullish",
                        "rec": rec,
                    }
                )
            for rec in bearish_slice:
                symbol = str(rec.get("Symbol", "")).upper()
                if not symbol:
                    continue
                candidate_entries.append(
                    {
                        "symbol": symbol,
                        "score": safe_float(rec.get("BearishScore")),
                        "bias": "bearish",
                        "rec": rec,
                    }
                )
            recs_snapshot = bullish_slice + bearish_slice
            bearish_symbols = {str(rec.get("Symbol", "")).upper() for rec in bearish_slice}
            candidate_count = len(candidate_entries)
            logger.info(
                "Autopilot evaluating %d recommendation candidates (bullish=%d bearish=%d strategy=%s)",
                candidate_count,
                len(bullish_slice),
                len(bearish_slice),
                strategy_key,
            )

            position_multiplier = max(risk_cfg.get("position_multiplier", 1.0), 0.25)
            position_multiplier = max(position_multiplier * max(vol_position_scale, 0.1), 0.25)
            stop_loss_multiplier = max(risk_cfg.get("stop_loss_multiplier", 1.0), 0.25)
            take_profit_multiplier = max(risk_cfg.get("take_profit_multiplier", 1.0), 0.25)

            max_total_allocation = max(
                0.1, hybrid.get("max_total_allocation", 1.0) * position_multiplier
            )
            min_entry_notional = max(50.0, hybrid.get("min_entry_notional", 200.0))

            exit_threshold = hybrid.get(
                "exit_score", hybrid.get("min_bullish_score", 3.0) - 1.0
            )
            lookback = hybrid.get("lookback", "1y")
            benchmark_df = _autopilot_prepare_dataframe("SPY", lookback)

            max_positions = max(1, int(profile.max_open_positions))
            current_positions_equity = held_positions["equity"]
            current_positions_option = {
                sym: entry for sym, entry in held_positions["option"].items() if sym not in ZOMBIE_POSITIONS
            }
            _autopilot_uncovered_exits.intersection_update(set(current_positions_option.keys()))
            _autopilot_stale_exits.intersection_update(set(current_positions_option.keys()))

            current_positions: dict[str, dict[str, Any]] = {}
            current_positions.update(current_positions_equity)
            current_positions.update(current_positions_option)
            available_slots = max(0, max_positions - len(current_positions))
            current_equity_positions = len(current_positions_equity)

            option_profit_default = safe_float(hybrid.get("option_take_profit_pct"), 0.8)
            option_stop_default = abs(
                safe_float(hybrid.get("option_stop_loss_pct"), 0.45) * stop_loss_multiplier
            )
            option_stop_default = min(
                option_stop_default,
                BALANCED_GROWTH_CONFIG.get("option_max_loss_pct", option_stop_default),
            )
            option_profit_default = (
                option_profit_default * take_profit_multiplier if option_profit_default is not None else None
            )
            expiry_buffer = max(0, int(safe_float(hybrid.get("options_expiry_buffer"), 5)))
            stock_stop_default = max(
                0.0,
                safe_float(hybrid.get("stock_stop_loss_pct"), PAPER_DEFAULT_STOP_LOSS_PCT)
                * stop_loss_multiplier,
            )
            stock_take_default = max(
                0.0,
                safe_float(hybrid.get("stock_take_profit_pct"), PAPER_DEFAULT_TAKE_PROFIT_PCT)
                * take_profit_multiplier,
            )

            if current_positions_option:
                # Option exits now enforce hard max-loss and expiry safety nets.
                for contract_symbol, entry in current_positions_option.items():
                    if contract_symbol in ZOMBIE_POSITIONS:
                        logger.debug("Skipping zombie contract %s during exit review", contract_symbol)
                        continue
                    if contract_symbol in _autopilot_stale_exits:
                        logger.debug(
                            "Skipping stale exit for %s; previously cleared", contract_symbol
                        )
                        continue
                    if contract_symbol in _autopilot_uncovered_exits:
                        notice = (
                            f"Manual attention required for {contract_symbol}; previous exit rejected as uncovered."
                        )
                        logger.info(notice)
                        summary_lines.append(notice)
                        continue
                    pos = entry.get("position")
                    qty = int(abs(entry.get("qty", 0)))
                    available_qty = _option_position_quantity(pos)
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
                        logger.info("No open quantity for %s; skipping option close.", contract_symbol)
                        summary_lines.append(
                            f"No open quantity for {contract_symbol}; skipping exit."
                        )
                        continue
                    if qty <= 0:
                        qty = available_qty
                    if qty > available_qty:
                        logger.info(
                            "Clamping exit qty for %s to %d from %d", contract_symbol, available_qty, qty
                        )
                        qty = available_qty
                    current_position_qty = available_qty
                    exit_qty = min(current_position_qty, qty)
                    if current_position_qty <= 0 or exit_qty <= 0:
                        logger.info("No open quantity for %s; skipping option close.", contract_symbol)
                        summary_lines.append(f"No open quantity for {contract_symbol}; skipping exit.")
                        continue
                    qty = exit_qty
                    pos = entry.get("position") or {}
                    parsed = entry.get("meta") or parse_option_symbol(contract_symbol)
                    underlying = parsed.get("underlying") if parsed else None
                    position_params = _autopilot_position_params.get(contract_symbol, {})
                    option_profit = safe_float(
                        position_params.get("take_profit_pct"), option_profit_default
                    )
                    option_stop = abs(
                        safe_float(
                            position_params.get("stop_loss_pct"),
                            option_stop_default,
                        )
                    )
                    avg_entry_price = safe_float(pos.get("avg_entry_price"), None)
                    price_snapshot = dict(pos) if isinstance(pos, dict) else {}
                    if avg_entry_price is not None:
                        price_snapshot.setdefault("last_price", avg_entry_price)
                        price_snapshot.setdefault("mark_price", avg_entry_price)
                    last_hint = safe_float(
                        pos.get("current_price"), safe_float(pos.get("market_price"), None)
                    )
                    if last_hint is not None:
                        price_snapshot["last_price"] = last_hint
                        price_snapshot.setdefault("mark_price", last_hint)
                    bid_hint = safe_float(pos.get("bid_price"), None)
                    ask_hint = safe_float(pos.get("ask_price"), None)
                    if bid_hint is not None:
                        price_snapshot.setdefault("bid_price", bid_hint)
                    if ask_hint is not None:
                        price_snapshot.setdefault("ask_price", ask_hint)
                    current_price, _ = _derive_option_price(price_snapshot)
                    stop_price = avg_entry_price * OPTION_STOP_LOSS_PCT if avg_entry_price else None
                    take_price = avg_entry_price * OPTION_TAKE_PROFIT_PCT if avg_entry_price else None
                    reasons: list[str] = []
                    if current_price is not None:
                        if stop_price is not None and current_price <= stop_price:
                            reasons.append(
                                f"stop hit ${current_price:.2f} <= ${stop_price:.2f}"
                            )
                        elif take_price is not None and current_price >= take_price:
                            reasons.append(
                                f"target hit ${current_price:.2f} >= ${take_price:.2f}"
                            )
                    loss_reason_added = False
                    loss_pct = option_pnl_percent(avg_entry_price, current_price)
                    if loss_pct is not None and loss_pct <= -50.0:
                        reasons.insert(0, f"max_loss_exit {loss_pct:.1f}%")
                        loss_reason_added = True
                    plpc = safe_float(pos.get("unrealized_plpc"), None)
                    if plpc is None:
                        percent = safe_float(pos.get("unrealized_pl_percent"), None)
                        if percent is not None:
                            plpc = percent / 100.0
                    opened_at = None
                    opened_raw = position_params.get("opened_at") if isinstance(position_params, dict) else None
                    if opened_raw:
                        try:
                            opened_at = datetime.fromisoformat(str(opened_raw))
                        except Exception:
                            opened_at = None
                    if plpc is not None and option_profit and plpc >= option_profit:
                        reasons.append(f"profit {plpc*100:.0f}% >= target")
                    if plpc is not None and option_stop and plpc <= -option_stop:
                        reasons.append(f"loss {plpc*100:.0f}% beyond stop")
                    if (
                        profile.name == "high"
                        and plpc is not None
                        and plpc <= profile.loser_cut_threshold_plpc * LOSER_OPTION_THRESHOLD_MULTIPLIER
                    ):
                        reasons.append(
                            f"loss {plpc*100:.0f}% beyond aggressive stop"
                        )
                    days_out = option_days_to_expiration(parsed)
                    if days_out is not None and days_out <= expiry_buffer:
                        reasons.append(f"{days_out}d to expiry")
                    if days_out is not None and days_out <= 5:
                        reasons.insert(0, f"expiry_safety_exit DTE={days_out} days")
                    if underlying:
                        df = _autopilot_prepare_dataframe(underlying, lookback)
                        if df is None:
                            errors.append(f"no data for {underlying}; option exit skipped")
                        else:
                            underlying_score, _ = score_stock(df)
                            if underlying_score < exit_threshold:
                                reasons.append(f"score {underlying_score:.2f} < {exit_threshold}")
                    if (
                        profile.name == "high"
                        and opened_at
                        and plpc is not None
                        and abs(plpc) < 0.01
                        and (now - opened_at) > timedelta(days=4)
                        ):
                        reasons.append("stagnant option freeing capital")
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
                    market_price = current_price or safe_float(
                        pos.get("current_price"), safe_float(pos.get("market_price"), None)
                    )
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
                            _mark_zombie_position(
                                contract_symbol,
                                reason="uncovered rejection",
                                summary_lines=summary_lines,
                            )
                            if ENABLE_ZOMBIE_DELETE:
                                try:
                                    delete_result = paper_broker.delete_position(contract_symbol)
                                    logger.warning(
                                        "Hard delete for zombie position %s completed after uncovered rejection: %s",
                                        contract_symbol,
                                        delete_result,
                                    )
                                    summary_lines.append(
                                        f"Hard-deleted zombie position {contract_symbol} via API after uncovered rejection"
                                    )
                                    if underlying:
                                        _set_option_cooldown(underlying, now, minutes=60)
                                    _autopilot_stale_exits.add(contract_symbol)
                                    continue
                                except AlpacaAPIError as delete_err:
                                    msg_lower = str(getattr(delete_err, "api_message", "") or delete_err).lower()
                                    if getattr(delete_err, "status_code", None) == 422 or "options market orders are only allowed during market hours" in msg_lower:
                                        logger.warning(
                                            "Hard delete for %s failed due to market hours; leaving zombie flag in place.",
                                            contract_symbol,
                                        )
                                    else:
                                        logger.error(
                                            "Hard delete for %s failed after uncovered rejection: %s (status=%s)",
                                            contract_symbol,
                                            delete_err,
                                            getattr(delete_err, "status_code", None),
                                        )
                            else:
                                logger.warning(
                                    "Zombie delete for %s skipped because ENABLE_ZOMBIE_DELETE is False",
                                    contract_symbol,
                                )
                            continue
                        orders_placed += 1
                        summary_lines.append(
                            f"Exit {qty} {contract_symbol} ({'; '.join(reasons)})"
                        )
                        # Apply cooldown to the underlying so we do not immediately reenter
                        if underlying:
                            _set_option_cooldown(underlying, now, minutes=60)
                    except NoAvailableBidError:
                        postpone_msg = (
                            f"Exit for {contract_symbol} postponed; Alpaca reports no available bid."
                        )
                        logger.info(postpone_msg)
                        summary_lines.append(postpone_msg)
                        if underlying:
                            _set_option_cooldown(underlying, now, minutes=60)
                    except OptionCloseRejectedError as exc:
                        warning = (
                            f"Exit for {contract_symbol} rejected: {exc.api_message or exc}"
                        )
                        logger.warning(warning)
                        summary_lines.append(warning)
                        _mark_zombie_position(
                            contract_symbol,
                            reason="uncovered rejection",
                            summary_lines=summary_lines,
                        )
                        if ENABLE_ZOMBIE_DELETE:
                            try:
                                delete_result = paper_broker.delete_position(contract_symbol)
                                logger.warning(
                                    "Hard delete for zombie position %s completed after close rejection: %s",
                                    contract_symbol,
                                    delete_result,
                                )
                                summary_lines.append(
                                    f"Hard-deleted zombie position {contract_symbol} via API after close rejection"
                                )
                                if underlying:
                                    _set_option_cooldown(underlying, now, minutes=60)
                                _autopilot_stale_exits.add(contract_symbol)
                                continue
                            except AlpacaAPIError as delete_err:
                                msg_lower = str(getattr(delete_err, "api_message", "") or delete_err).lower()
                                if getattr(delete_err, "status_code", None) == 422 or "options market orders are only allowed during market hours" in msg_lower:
                                    logger.warning(
                                        "Hard delete for %s failed due to market hours; leaving zombie flag in place.",
                                        contract_symbol,
                                    )
                                else:
                                    logger.error(
                                        "Hard delete for %s failed after close rejection: %s (status=%s)",
                                        contract_symbol,
                                        delete_err,
                                        getattr(delete_err, "status_code", None),
                                    )
                        else:
                            logger.warning(
                                "Zombie delete for %s skipped because ENABLE_ZOMBIE_DELETE is False",
                                contract_symbol,
                            )
                    except AlpacaAPIError as exc:
                        msg = str(exc).lower()
                        underlying = parsed.get("underlying") if parsed else None
                        if exc.status_code == 403:
                            if "insufficient qty available for order" in msg:
                                logger.warning(
                                    "Marking zombie for %s: insufficient qty rejection (%s)",
                                    contract_symbol,
                                    exc,
                                )
                                _mark_zombie_position(
                                    contract_symbol,
                                    reason="insufficient qty rejection",
                                    summary_lines=summary_lines,
                                )
                                _autopilot_stale_exits.add(contract_symbol)
                                _autopilot_uncovered_exits.discard(contract_symbol)
                                if underlying:
                                    _set_option_cooldown(underlying, now, minutes=60)
                                continue
                            if "account not eligible to trade uncovered option contracts" in msg:
                                logger.warning(
                                    "Marking zombie for %s: uncovered rejection (%s)",
                                    contract_symbol,
                                    exc,
                                )
                                _mark_zombie_position(
                                    contract_symbol,
                                    reason="uncovered rejection",
                                    summary_lines=summary_lines,
                                )
                                if underlying:
                                    _set_option_cooldown(underlying, now, minutes=60)
                                continue

                        if "no available bid for symbol" in msg:
                            logger.warning(
                                "Exit failed for %s due to illiquidity (no available bid): %s - attempting hard delete via positions API",
                                contract_symbol,
                                exc,
                            )
                            if ENABLE_ZOMBIE_DELETE:
                                try:
                                    delete_result = paper_broker.delete_position(contract_symbol)
                                    logger.warning(
                                        "Hard delete for zombie position %s completed: %s",
                                        contract_symbol,
                                        delete_result,
                                    )
                                    summary_lines.append(
                                        f"Hard-deleted zombie position {contract_symbol} via API"
                                    )
                                    if underlying:
                                        _set_option_cooldown(underlying, now, minutes=60)
                                    _autopilot_stale_exits.add(contract_symbol)
                                    continue
                                except AlpacaAPIError as delete_err:
                                    msg_lower = str(getattr(delete_err, "api_message", "") or delete_err).lower()
                                    if getattr(delete_err, "status_code", None) == 422 or "options market orders are only allowed during market hours" in msg_lower:
                                        logger.warning(
                                            "Hard delete for %s failed due to market hours; leaving zombie flag in place.",
                                            contract_symbol,
                                        )
                                    else:
                                        logger.error(
                                            "Hard delete for %s failed: %s (status=%s)",
                                            contract_symbol,
                                            delete_err,
                                            getattr(delete_err, "status_code", None),
                                        )
                            else:
                                logger.warning(
                                    "Zombie delete for %s skipped because ENABLE_ZOMBIE_DELETE is False",
                                    contract_symbol,
                                )

                        logger.error(
                            "Autopilot option exit failed for %s: %s", contract_symbol, exc
                        )
                        errors.append(f"sell {contract_symbol} failed: {exc}")
                    except Exception as exc:
                        logger.exception("Autopilot option exit failed for %s", contract_symbol)
                        errors.append(f"sell {contract_symbol} failed: {exc}")
            for symbol, entry in current_positions_equity.items():
                pos = entry["position"]
                qty = abs(safe_float(pos.get("qty"), safe_float(pos.get("quantity"))))
                available_qty = qty
                if available_qty < 1:
                    refreshed = None
                    for candidate in positions:
                        try:
                            if str(candidate.get("symbol", "")).replace(" ", "").upper() != symbol.upper():
                                continue
                            asset_class = str(candidate.get("asset_class", "")).lower()
                            if "option" in asset_class:
                                continue
                            refreshed = candidate
                            break
                        except Exception:
                            continue
                    if refreshed is None:
                        try:
                            positions = list(paper_broker.get_positions())
                        except Exception as refresh_exc:
                            logger.warning(
                                "Skip exit for %s: failed to refresh positions (%s)", symbol, refresh_exc
                            )
                            summary_lines.append(
                                f"Skip exit for {symbol}; unable to confirm position."
                            )
                            continue
                        for candidate in positions:
                            try:
                                if str(candidate.get("symbol", "")).replace(" ", "").upper() != symbol.upper():
                                    continue
                                asset_class = str(candidate.get("asset_class", "")).lower()
                                if "option" in asset_class:
                                    continue
                                refreshed = candidate
                                break
                            except Exception:
                                continue
                    if refreshed:
                        entry["position"] = refreshed
                        pos = refreshed
                        available_qty = abs(
                            safe_float(pos.get("qty"), safe_float(pos.get("quantity")))
                        )
                        entry["qty"] = available_qty
                    else:
                        logger.info("Skip exit for %s: no matching position found", symbol)
                        summary_lines.append(
                            f"Skip exit for {symbol}; no matching position found."
                        )
                        continue
                pos = entry["position"]
                if available_qty <= 0:
                    logger.info("Skip exit for %s: no available quantity", symbol)
                    summary_lines.append(f"Skip exit for {symbol}; no available quantity.")
                    continue
                if qty <= 0:
                    qty = available_qty
                if qty > available_qty:
                    logger.info(
                        "Clamping exit qty for %s to %s from %s", symbol, available_qty, qty
                    )
                    qty = available_qty
                qty_int = int(math.floor(qty))
                if qty_int <= 0:
                    continue
                position_params = _autopilot_position_params.get(symbol.upper(), {})
                highest_price = safe_float(position_params.get("highest_price"), None)
                current_price_equity = _position_price(pos, "equity")
                avg_entry_price = safe_float(pos.get("avg_entry_price"), None)
                if current_price_equity and (highest_price is None or current_price_equity > highest_price):
                    highest_price = current_price_equity
                    position_params["highest_price"] = highest_price
                    with _autopilot_lock:
                        _autopilot_position_params[symbol.upper()] = position_params
                    _persist_autopilot_state()
                plpc = _position_unrealized_plpc(pos)
                opened_at = None
                opened_raw = position_params.get("opened_at") if isinstance(position_params, dict) else None
                if opened_raw:
                    try:
                        opened_at = datetime.fromisoformat(str(opened_raw))
                    except Exception:
                        opened_at = None
                # Growth stops
                hard_stop_pct = BALANCED_GROWTH_CONFIG.get("equity_stop_loss_pct", 0.12)
                trail_stop_pct = BALANCED_GROWTH_CONFIG.get("equity_trailing_stop_pct", 0.15)
                hard_stop_reason = None
                if current_price_equity and avg_entry_price and hard_stop_pct:
                    stop_price = avg_entry_price * (1 - hard_stop_pct)
                    if current_price_equity <= stop_price:
                        hard_stop_reason = f"equity_hard_stop {current_price_equity:.2f} <= {stop_price:.2f}"
                trail_stop_reason = None
                if current_price_equity and highest_price and trail_stop_pct:
                    trail_price = highest_price * (1 - trail_stop_pct)
                    if current_price_equity <= trail_price:
                        trail_stop_reason = f"equity_trailing_stop {current_price_equity:.2f} <= {trail_price:.2f}"
                aggressive_exit_reason: str | None = None
                if profile.name == "high" and plpc is not None:
                    if plpc <= profile.loser_cut_threshold_plpc:
                        aggressive_exit_reason = f"loss {plpc*100:.1f}% beyond aggressive stop"
                    elif opened_at and abs(plpc) < 0.01 and (now - opened_at) > timedelta(days=4):
                        aggressive_exit_reason = "stagnant trade freeing capital"
                stop_reason = hard_stop_reason or trail_stop_reason
                if stop_reason:
                    if _autopilot_order_blocked(symbol, open_orders):
                        summary_lines.append(
                            f"Exit pending for {symbol}; open order detected."
                        )
                        continue
                    qty_int = int(math.floor(qty))
                    if qty_int > 0:
                        try:
                            place_guarded_paper_order(symbol, qty_int, "sell", time_in_force="day")
                            summary_lines.append(
                                f"Exit {qty_int} {symbol} ({stop_reason})"
                            )
                            orders_placed += 1
                            continue
                        except AlpacaAPIError as exc:
                            message = (getattr(exc, "api_message", "") or str(exc)).lower()
                            if exc.status_code == 403 and "insufficient qty available for order" in message:
                                logger.warning(
                                    "Skip equity exit for %s: Alpaca insufficient qty (%s)", symbol, exc
                                )
                                summary_lines.append(
                                    f"Skip exit for {symbol}; insufficient qty (stale position)."
                                )
                                continue
                            logger.exception("Autopilot equity exit failed for %s", symbol)
                            errors.append(f"sell {symbol} failed: {exc}")
                        except Exception as exc:
                            logger.exception("Autopilot equity exit failed for %s", symbol)
                            errors.append(f"sell {symbol} failed: {exc}")
                if aggressive_exit_reason:
                    if _autopilot_order_blocked(symbol, open_orders):
                        summary_lines.append(
                            f"Exit pending for {symbol}; open order detected."
                        )
                        continue
                    qty_int = int(math.floor(qty))
                    if qty_int > 0:
                        try:
                            place_guarded_paper_order(symbol, qty_int, "sell", time_in_force="day")
                            summary_lines.append(
                                f"Exit {qty_int} {symbol} ({aggressive_exit_reason})"
                            )
                            orders_placed += 1
                            continue
                        except AlpacaAPIError as exc:
                            message = (getattr(exc, "api_message", "") or str(exc)).lower()
                            if exc.status_code == 403 and "insufficient qty available for order" in message:
                                logger.warning(
                                    "Skip equity exit for %s: Alpaca insufficient qty (%s)", symbol, exc
                                )
                                summary_lines.append(
                                    f"Skip exit for {symbol}; insufficient qty (stale position)."
                                )
                                continue
                            logger.exception("Autopilot equity exit failed for %s", symbol)
                            errors.append(f"sell {symbol} failed: {exc}")
                        except Exception as exc:
                            logger.exception("Autopilot equity exit failed for %s", symbol)
                            errors.append(f"sell {symbol} failed: {exc}")
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
                    except AlpacaAPIError as exc:
                        message = (getattr(exc, "api_message", "") or str(exc)).lower()
                        if exc.status_code == 403 and "insufficient qty available for order" in message:
                            logger.warning(
                                "Skip equity exit for %s: Alpaca insufficient qty (%s)", symbol, exc
                            )
                            summary_lines.append(
                                f"Skip exit for {symbol}; insufficient qty (stale position)."
                            )
                            continue
                        logger.exception("Autopilot equity exit failed for %s", symbol)
                        errors.append(f"sell {symbol} failed: {exc}")
                    except Exception as exc:
                        logger.exception("Autopilot equity exit failed for %s", symbol)
                        errors.append(f"sell {symbol} failed: {exc}")

            current_positions_option = {
                sym: entry for sym, entry in current_positions_option.items() if sym not in ZOMBIE_POSITIONS
            }
            held_option_underlyings = {
                entry["meta"].get("underlying")
                for entry in current_positions_option.values()
                if isinstance(entry.get("meta"), dict) and entry["meta"].get("underlying")
            }
            held_option_underlyings = {sym for sym in held_option_underlyings if sym}
            held_put_underlyings = {
                entry["meta"].get("underlying")
                for entry in current_positions_option.values()
                if isinstance(entry.get("meta"), dict)
                and entry["meta"].get("underlying")
                and entry["meta"].get("type") == "put"
            }
            current_positions = {}
            current_positions.update(current_positions_equity)
            current_positions.update(current_positions_option)

            max_positions = max(1, int(profile.max_open_positions))
            available_slots = max(0, max_positions - len(current_positions))
            gross_notional = gross_option_notional + gross_equity_notional

            allocation_warning_logged = False
            max_new_entries = max(1, int(profile.max_new_trades_per_run))
            entries_placed = 0
            cash_ratio = (cash_balance / equity) if equity else 1.0

            def _pyramid_for_position(symbol: str, entry: dict[str, Any], asset_class: str) -> None:
                nonlocal entries_placed, gross_notional, cash_balance, cash_ratio, orders_placed

                if entries_placed >= max_new_entries:
                    return
                if profile.max_pyramids_per_position <= 0:
                    return
                symbol_key = symbol.upper()
                pyramid_count = _autopilot_pyramid_counts.get(symbol_key, 0)
                if pyramid_count >= profile.max_pyramids_per_position:
                    return
                pos = entry.get("position", {})
                plpc = _position_unrealized_plpc(pos)
                if plpc is None:
                    return
                threshold = profile.winner_add_threshold_plpc
                if "option" in str(asset_class).lower():
                    threshold *= WINNER_OPTION_THRESHOLD_MULTIPLIER
                if plpc < threshold:
                    return
                open_buy_orders = paper_broker.list_open_orders_for_symbol(
                    symbol,
                    asset_class="option" if "option" in str(asset_class).lower() else "us_equity",
                    side="buy",
                    orders=open_orders,
                )
                if open_buy_orders:
                    return
                price = _position_price(pos, asset_class)
                if price is None or price <= 0:
                    return
                qty, target_notional = compute_position_qty(
                    symbol,
                    asset_class,
                    price,
                    equity,
                    profile,
                    position_multiplier=position_multiplier,
                    min_entry_notional=min_entry_notional,
                    scale=PYRAMID_POSITION_SCALE,
                )
                multiplier = OPTION_CONTRACT_MULTIPLIER if "option" in str(asset_class).lower() else 1
                order_notional = qty * price * multiplier
                if qty <= 0:
                    return
                if gross_notional + order_notional > equity * max_total_allocation:
                    return
                try:
                    place_guarded_paper_order(
                        symbol,
                        qty,
                        "buy",
                        asset_class="option" if "option" in str(asset_class).lower() else None,
                        time_in_force="day",
                        price_hint=price,
                        support_brackets=False,
                    )
                    gross_notional += order_notional
                    cash_balance = max(0.0, cash_balance - order_notional)
                    cash_ratio = (cash_balance / equity) if equity else cash_ratio
                    entries_placed += 1
                    orders_placed += 1
                    _autopilot_pyramid_counts[symbol_key] = pyramid_count + 1
                    summary_lines.append(
                        f"Add {qty} to {symbol} (pyramid, PL {plpc*100:.1f}% >= {threshold*100:.1f}%)"
                    )
                except Exception as exc:
                    logger.exception("Autopilot pyramid add failed for %s", symbol)
                    errors.append(f"pyramid {symbol} failed: {exc}")

            if profile.max_pyramids_per_position > 0 and profile.name == "high":
                for symbol, entry in current_positions_equity.items():
                    _pyramid_for_position(symbol, entry, "equity")
                    if entries_placed >= max_new_entries:
                        break
                for symbol, entry in current_positions_option.items():
                    if entries_placed >= max_new_entries:
                        break
                    _pyramid_for_position(symbol, entry, "option")

            held_equity_symbols = set(current_positions_equity.keys())
            held_underlyings = held_equity_symbols | held_option_underlyings

            held_and_pending = set(current_positions.keys())
            pending_underlyings: set[str] = set()
            active_statuses = {"new", "accepted", "partially_filled", "open", "pending_new"}
            for order in open_orders:
                try:
                    if str(order.get("side", "")).lower() != "buy":
                        continue
                    status = str(order.get("status", "")).lower()
                    if status and status not in active_statuses:
                        continue
                    order_symbol = str(order.get("symbol", "")).replace(" ", "").upper()
                    asset = str(order.get("asset_class", "")).lower()
                    held_and_pending.add(order_symbol)
                    if "option" in asset:
                        parsed = parse_option_symbol(order_symbol)
                        if parsed and parsed.get("underlying"):
                            pending_underlyings.add(parsed["underlying"])
                    else:
                        pending_underlyings.add(order_symbol)
                except Exception:
                    continue

            min_entry_score = profile.min_entry_score or hybrid.get("min_bullish_score", 3.0)
            bearish_min_score = safe_float(
                hybrid.get("min_bearish_score"), hybrid.get("min_bullish_score", 3.0)
            )
            logged_candidates: set[str] = set()

            sorted_candidates = sorted(
                candidate_entries,
                key=lambda item: safe_float(item.get("score"), 0.0),
                reverse=True,
            )

            if not sorted_candidates:
                summary_lines.append("No new symbols met entry criteria.")

            for candidate in sorted_candidates:
                if entries_placed >= max_new_entries:
                    summary_lines.append(
                        f"Entry cap reached ({entries_placed}/{max_new_entries}); skipping remaining candidates."
                    )
                    break
                symbol = candidate.get("symbol", "")
                score = candidate.get("score")
                bias = candidate.get("bias") or "bullish"
                if not symbol:
                    continue

                direction_label = "bullish" if bias == "bullish" else "bearish"
                reason_label = "pending evaluation"
                candidate_logged = False

                def log_candidate_outcome(
                    reason: str, direction_override: str | None = None
                ) -> None:
                    nonlocal reason_label, direction_label, candidate_logged
                    reason_label = reason
                    if direction_override:
                        direction_label = direction_override
                    score_fragment = (
                        f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                    )
                    logger.info(
                        "Candidate %s: direction=%s reason=%s score=%s bias=%s strategy=%s",
                        symbol,
                        direction_label,
                        reason_label,
                        score_fragment,
                        bias,
                        strategy_key,
                    )
                    candidate_logged = True
                    logged_candidates.add(symbol)

                if available_slots <= 0:
                    log_candidate_outcome("no entry, max positions reached", "none")
                    continue

                if score is None:
                    log_candidate_outcome(
                        "no entry, missing score", "none"
                    )
                    continue

                min_required = min_entry_score if bias == "bullish" else bearish_min_score
                if score < min_required:
                    log_candidate_outcome(
                        f"no entry, score {score:.2f} below min {min_required:.2f} (strategy={strategy_key})",
                        "none",
                    )
                    continue

                if symbol in pending_underlyings:
                    log_candidate_outcome(
                        "no entry, order already pending for underlying", "none"
                    )
                    continue

                try:
                    equity_price = fetch_latest_price(symbol)
                except Exception as exc:
                    logger.exception("Autopilot price fetch failed for %s", symbol)
                    errors.append(f"price {symbol} failed: {exc}")
                    log_candidate_outcome("price fetch failed", "none")
                    continue

                option_available = False
                selection: OptionSelection | None = None
                selection_diag: dict[str, Any] = {}
                put_choice: dict[str, Any] | None = None
                equity_growth_score: float | None = None
                option_score: float | None = None
                indicator_features: dict[str, Any] = {
                    "rsi": None,
                    "macd": None,
                    "volatility_20d": None,
                    "volume_rel_20d": None,
                }
                consider_option = bias == "bearish" or hybrid.get("option_bias") != "rare" or score >= (min_required + 0.5)

                if bias == "bearish":
                    if _option_on_cooldown(symbol, now):
                        summary_lines.append(
                            f"Candidate {symbol}: direction=none reason=cooldown in effect bias={bias}"
                        )
                        log_candidate_outcome("cooldown", "none")
                        continue
                    if symbol in held_put_underlyings:
                        summary_lines.append(
                            f"Skip bearish {symbol}; put position already open."
                        )
                        log_candidate_outcome(
                            "bearish entry blocked, already have open position",
                            "none",
                        )
                        continue
                    put_choice = choose_put_contract(symbol, now) if consider_option else None
                    option_available = bool(put_choice)
                else:
                    if consider_option:
                        try:
                            selection = _autopilot_select_option_contract(
                                symbol,
                                hybrid,
                                underlying_price=equity_price,
                                score=score,
                            )
                            selection_diag = selection.diagnostics or {}
                            option_available = bool(selection.contract and selection.premium and selection.premium > 0)
                            option_score = safe_float(selection.premium, None)
                        except PriceDataError as exc:
                            errors.append(f"options {symbol} chain failed: {exc}")
                            log_candidate_outcome(
                                "bullish entry blocked, option chain unavailable",
                                "none",
                            )
                            continue

                # Instrument decision: equity by default, options only if clearly superior and budget allows
                decision = "stock"
                decision_reason = "default equity path"
                equity_score_val = equity_growth_score if equity_growth_score is not None else safe_float(score, 0.0)
                option_score_val = option_score if option_score is not None else safe_float(score, 0.0)
                equity_threshold = max(min_required, 3.0)
                if bias == "bearish":
                    # Conservative bearish: only take options when available and budget allows, otherwise skip
                    if option_available and options_budget_remaining > 0:
                        decision = "option"
                        decision_reason = "bearish put available within budget"
                    else:
                        decision = "skip"
                        decision_reason = "bearish view but no suitable conservative options; skipping"
                else:
                    if (
                        option_available
                        and equity_score_val is not None
                        and equity_score_val >= equity_threshold
                        and option_score_val is not None
                        and option_score_val >= equity_score_val + 2.0
                        and options_budget_remaining > 0
                    ):
                        decision = "option"
                        decision_reason = "option path selected (score significantly higher and within options budget)"
                    else:
                        if equity_score_val is None or equity_score_val < equity_threshold:
                            decision = "skip"
                            decision_reason = f"equity score {equity_score_val} below threshold {equity_threshold}"
                        else:
                            decision = "stock"
                            decision_reason = "equity favored for growth profile (options not clearly superior or budget limited)"

                if decision == "skip":
                    log_candidate_outcome(
                        f"no entry, {decision_reason}",
                        "none",
                    )
                    continue

                mid_price: float | None = None
                contract_symbol: str | None = None
                if decision == "option":
                    if _option_on_cooldown(symbol, now):
                        summary_lines.append(
                            f"Candidate {symbol}: direction=none reason=cooldown in effect bias={bias}"
                        )
                        log_candidate_outcome("cooldown", "none")
                        continue

                    if bias == "bearish":
                        if not put_choice:
                            log_candidate_outcome(
                                "bearish entry blocked, no suitable put contract",
                                "none",
                            )
                            continue
                        contract_symbol = put_choice["option_symbol"].upper()
                        if contract_symbol in held_and_pending:
                            log_candidate_outcome(
                                "bearish entry blocked, order already pending",
                                "none",
                            )
                            continue
                        block_reason = _option_trade_block_reason(
                            contract_symbol,
                            positions,
                            open_orders,
                            closed_today_symbols,
                            now=now,
                        )
                        if block_reason:
                            reason_msg = (
                                f"Skip bearish {symbol}; {contract_symbol} blocked ({block_reason})."
                            )
                            logger.info(reason_msg)
                            summary_lines.append(reason_msg)
                            log_candidate_outcome(
                                "bearish entry blocked: " + block_reason,
                                "none",
                            )
                            continue
                        bid = safe_float(put_choice.get("bid"))
                        ask = safe_float(put_choice.get("ask"))
                        mid_price = safe_float(put_choice.get("mid"), 0.0)
                        if mid_price <= 0 and bid and ask:
                            mid_price = (bid + ask) / 2.0
                        if mid_price is None or mid_price <= 0:
                            summary_lines.append(
                                f"Skip bearish {symbol}; invalid quote for {contract_symbol}."
                            )
                            log_candidate_outcome(
                                "bearish entry blocked, invalid option quote",
                                "none",
                            )
                            continue
                        unit_cost = mid_price * OPTION_CONTRACT_MULTIPLIER
                        qty, target_notional = compute_position_qty(
                            contract_symbol,
                            "us_option",
                            mid_price,
                            equity,
                            profile,
                            position_multiplier=position_multiplier,
                            min_entry_notional=min_entry_notional,
                            contract_multiplier=OPTION_CONTRACT_MULTIPLIER,
                        )
                        if mid_price > 0 and equity > 0:
                            max_option_notional = equity * BALANCED_GROWTH_CONFIG["max_options_notional_pct"]
                            if options_budget_remaining:
                                max_option_notional = min(max_option_notional, options_budget_remaining)
                            cap_qty = int(max_option_notional // unit_cost) if unit_cost > 0 else 0
                            if cap_qty <= 0:
                                summary_lines.append(
                                    "Option position too small under conservative risk cap"
                                )
                                log_candidate_outcome(
                                    "option position too small under conservative risk cap",
                                    "none",
                                )
                                continue
                            cap_qty = min(cap_qty, MAX_OPTION_CONTRACTS_PER_TRADE)
                            qty = cap_qty if qty <= 0 else min(qty, cap_qty)
                        max_debit = safe_float(hybrid.get("max_position_debit"), None)
                        if max_debit:
                            target_notional = min(target_notional, max_debit)
                        if PAPER_MAX_POSITION_NOTIONAL:
                            target_notional = min(target_notional, PAPER_MAX_POSITION_NOTIONAL)
                        if unit_cost > 0:
                            qty = max(int(target_notional // unit_cost), qty)
                        max_contracts = int(safe_float(hybrid.get("max_contracts_per_trade"), 0))
                        if max_contracts:
                            qty = min(qty, max_contracts)
                        order_notional = qty * unit_cost
                        total_options_after = gross_option_notional + order_notional
                        if equity and total_options_after > equity * BALANCED_GROWTH_CONFIG["max_total_options_pct"]:
                            summary_lines.append(
                                "options budget exhausted under BalancedGrowth; skipping option entry"
                            )
                            log_candidate_outcome(
                                "options budget exhausted under BalancedGrowth",
                                "none",
                            )
                            break
                        if gross_notional + order_notional > equity * max_total_allocation:
                            if not allocation_warning_logged:
                                summary_lines.append(
                                    "Max portfolio allocation reached; skipping new entries."
                                )
                                allocation_warning_logged = True
                            log_candidate_outcome(
                                "bearish entry blocked by portfolio allocation",
                                "none",
                            )
                            break
                        if qty <= 0:
                            summary_lines.append(
                                f"Skip bearish {symbol}; notional ${target_notional:.2f} too small for put cost ${unit_cost:.2f}."
                            )
                            log_candidate_outcome(
                                "bearish entry blocked by sizing rules",
                                "none",
                            )
                            continue
                        stop_loss_pct = option_stop_default
                        take_profit_pct = option_profit_default
                        indicator_df = None
                        try:
                            indicator_df = _autopilot_prepare_dataframe(symbol, lookback)
                        except PriceDataError:
                            indicator_df = None
                        except Exception:
                            logger.debug("Indicator prep failed for %s during option log", symbol, exc_info=True)
                            indicator_df = None
                        indicator_features = _extract_indicator_features(indicator_df)
                    if bias == "bearish":
                        try:
                            logger.info(
                                "Placing bearish order: symbol=%s asset_class=option qty=%s side=buy",
                                contract_symbol,
                                qty,
                            )
                            log_ai_snapshot(
                                symbol=symbol,
                                asset_class="option",
                                strategy_key=strategy_key,
                                contract_symbol=contract_symbol,
                                direction="buy",
                                score=safe_float(score, 0.0) or 0.0,
                                spot_price=equity_price,
                                entry_price=mid_price * OPTION_CONTRACT_MULTIPLIER if mid_price else None,
                                rsi=indicator_features.get("rsi"),
                                macd=indicator_features.get("macd"),
                                volatility_20d=indicator_features.get("volatility_20d"),
                                volume_rel_20d=indicator_features.get("volume_rel_20d"),
                                sector_strength=None,
                                market_trend=None,
                                congress_score=None,
                                news_sentiment=None,
                            )
                            place_guarded_paper_order(
                                contract_symbol,
                                qty,
                                "buy",
                                order_type="limit",
                                limit_price=round(mid_price, 2),
                                stop_loss_pct=stop_loss_pct,
                                take_profit_pct=take_profit_pct,
                                time_in_force="day",
                                asset_class="option",
                                price_hint=mid_price,
                                support_brackets=False,
                                position_effect="open",
                            )
                            gross_notional += order_notional
                            cash_balance = max(0.0, cash_balance - order_notional)
                            cash_ratio = (cash_balance / equity) if equity else cash_ratio
                            available_slots -= 1
                            held_and_pending.add(contract_symbol)
                            pending_underlyings.add(symbol)
                            _remember_position_params(
                                contract_symbol,
                                asset_class="option",
                                stop_loss_pct=stop_loss_pct,
                                take_profit_pct=take_profit_pct,
                                strategy=strategy_key,
                                opened_at=now,
                            )
                            summary_lines.append(
                                f"Buy {qty} {contract_symbol} ({symbol} put {put_choice.get('strike', 0):.2f} exp {put_choice.get('expiration')}, limit ${mid_price:.2f})"
                            )
                            orders_placed += 1
                            entries_placed += 1
                            gross_option_notional += order_notional
                            options_budget_remaining = max(
                                0.0, options_budget_limit - gross_option_notional
                            )
                            log_candidate_outcome(
                                f"bearish entry placed via option ({decision_reason})"
                            )
                        except Exception as exc:
                            logger.exception("Autopilot put entry failed for %s", contract_symbol or symbol)
                            errors.append(f"buy {contract_symbol or symbol} failed: {exc}")
                            if not candidate_logged:
                                log_candidate_outcome(
                                    "bearish entry failed during order placement",
                                    "bearish",
                                )
                        continue
                    # Bullish option path
                    if not selection:
                        log_candidate_outcome(
                            "bullish entry blocked, option selection unavailable",
                            "none",
                        )
                        continue
                    diag = selection_diag or {}
                    contract_data = selection.contract
                    premium = selection.premium
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
                        log_candidate_outcome(
                            "bullish entry blocked, no suitable call contract",
                            "none",
                        )
                        continue
                    contract_symbol = (
                        str(contract_data.get("symbol", "")).replace(" ", "").upper()
                        if isinstance(contract_data, dict)
                        else ""
                    )
                    if not contract_symbol or contract_symbol in held_and_pending:
                        log_candidate_outcome(
                            "bullish entry blocked, order already pending",
                            "none",
                        )
                        continue
                    block_reason = _option_trade_block_reason(
                        contract_symbol,
                        positions,
                        open_orders,
                        closed_today_symbols,
                        now=now,
                    )
                    if block_reason:
                        reason_msg = (
                            f"Skip bullish {symbol}; {contract_symbol} blocked ({block_reason})."
                        )
                        logger.info(reason_msg)
                        summary_lines.append(reason_msg)
                        log_candidate_outcome(
                            "bullish entry blocked: " + block_reason,
                            "none",
                        )
                        continue
                    stop_loss_pct = option_stop_default
                    take_profit_pct = option_profit_default
                    qty, target_notional = compute_position_qty(
                        contract_symbol,
                        "us_option",
                        premium,
                        equity,
                        profile,
                        position_multiplier=position_multiplier,
                        min_entry_notional=min_entry_notional,
                        contract_multiplier=OPTION_CONTRACT_MULTIPLIER,
                    )
                    if premium > 0 and equity > 0:
                        unit_cost = premium * OPTION_CONTRACT_MULTIPLIER
                        max_option_notional = equity * BALANCED_GROWTH_CONFIG["max_options_notional_pct"]
                        if options_budget_remaining:
                            max_option_notional = min(max_option_notional, options_budget_remaining)
                        cap_qty = int(max_option_notional // unit_cost) if unit_cost > 0 else 0
                        if cap_qty <= 0:
                            summary_lines.append(
                                "Option position too small under conservative risk cap"
                            )
                            log_candidate_outcome(
                                "option position too small under conservative risk cap",
                                "none",
                            )
                            continue
                        cap_qty = min(cap_qty, MAX_OPTION_CONTRACTS_PER_TRADE)
                        qty = cap_qty if qty <= 0 else min(qty, cap_qty)
                    max_debit = safe_float(hybrid.get("max_position_debit"), None)
                    if max_debit:
                        target_notional = min(target_notional, max_debit)
                    if PAPER_MAX_POSITION_NOTIONAL:
                        target_notional = min(target_notional, PAPER_MAX_POSITION_NOTIONAL)
                    unit_cost = premium * OPTION_CONTRACT_MULTIPLIER
                    if unit_cost > 0:
                        qty = max(int(target_notional // unit_cost), qty)
                    max_contracts = int(safe_float(hybrid.get("max_contracts_per_trade"), 0))
                    if max_contracts:
                        qty = min(qty, max_contracts)
                    order_notional = qty * unit_cost
                    total_options_after = gross_option_notional + order_notional
                    if equity and total_options_after > equity * BALANCED_GROWTH_CONFIG["max_total_options_pct"]:
                        summary_lines.append(
                            "options budget exhausted under BalancedGrowth; skipping option entry"
                        )
                        log_candidate_outcome(
                            "options budget exhausted under BalancedGrowth",
                            "none",
                        )
                        break
                    if gross_notional + order_notional > equity * max_total_allocation:
                        if not allocation_warning_logged:
                            summary_lines.append(
                                "Max portfolio allocation reached; skipping new entries."
                            )
                            allocation_warning_logged = True
                        log_candidate_outcome(
                            "bullish entry blocked by portfolio allocation",
                            "none",
                        )
                        break
                    if qty <= 0:
                        summary_lines.append(
                            f"Skip bullish {symbol}; notional ${target_notional:.2f} too small for call cost ${unit_cost:.2f}."
                        )
                        log_candidate_outcome(
                            "bullish entry blocked by sizing rules",
                            "none",
                        )
                        continue
                    indicator_df = None
                    try:
                        indicator_df = _autopilot_prepare_dataframe(symbol, lookback)
                    except PriceDataError:
                        indicator_df = None
                    except Exception:
                        logger.debug("Indicator prep failed for %s during option log", symbol, exc_info=True)
                        indicator_df = None
                    indicator_features = _extract_indicator_features(indicator_df)
                    try:
                        logger.info(
                            "Placing bullish order: symbol=%s asset_class=option qty=%s side=buy",
                            contract_symbol,
                            qty,
                        )
                        log_ai_snapshot(
                            symbol=symbol,
                            asset_class="option",
                            strategy_key=strategy_key,
                            contract_symbol=contract_symbol,
                            direction="buy",
                            score=safe_float(score, 0.0) or 0.0,
                            spot_price=equity_price,
                            entry_price=premium * OPTION_CONTRACT_MULTIPLIER if premium else None,
                            rsi=indicator_features.get("rsi"),
                            macd=indicator_features.get("macd"),
                            volatility_20d=indicator_features.get("volatility_20d"),
                            volume_rel_20d=indicator_features.get("volume_rel_20d"),
                            sector_strength=None,
                            market_trend=None,
                            congress_score=None,
                            news_sentiment=None,
                        )
                        place_guarded_paper_order(
                            contract_symbol,
                            qty,
                            "buy",
                            order_type="market",
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            time_in_force="day",
                            asset_class="option",
                            price_hint=premium,
                        )
                        gross_notional += order_notional
                        cash_balance = max(0.0, cash_balance - order_notional)
                        cash_ratio = (cash_balance / equity) if equity else cash_ratio
                        available_slots -= 1
                        held_and_pending.add(contract_symbol)
                        pending_underlyings.add(symbol)
                        _remember_position_params(
                            contract_symbol,
                            asset_class="option",
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            strategy=strategy_key,
                            opened_at=now,
                        )
                        summary_lines.append(
                            f"Buy {qty} {contract_symbol} (score {score:.2f}, premium ${premium:.2f}, stop {stop_loss_pct*100:.1f}%, take {take_profit_pct*100:.1f}%)"
                        )
                        orders_placed += 1
                        entries_placed += 1
                        gross_option_notional += order_notional
                        options_budget_remaining = max(
                            0.0, options_budget_limit - gross_option_notional
                        )
                        log_candidate_outcome(
                            f"bullish entry placed via option ({decision_reason})"
                        )
                    except Exception as exc:
                        logger.exception(
                            "Autopilot option entry failed for %s", contract_symbol
                        )
                        errors.append(f"buy {contract_symbol} failed: {exc}")
                        if not candidate_logged:
                            log_candidate_outcome(
                                "bullish entry failed during order placement",
                                "bullish",
                            )
                    continue
                else:
                    try:
                        df = _autopilot_prepare_dataframe(symbol, lookback)
                    except PriceDataError as exc:
                        logger.exception("Autopilot price fetch failed for %s", symbol)
                        errors.append(f"data {symbol} failed: {exc}")
                        log_candidate_outcome(
                            "bullish entry blocked, historical data unavailable",
                            "none",
                        )
                        continue
                    if df is None:
                        log_candidate_outcome("bullish entry blocked, no price history", "none")
                        continue
                    recalced_score, _ = score_stock(df)
                    try:
                        equity_growth_score, _ = score_equity_growth(symbol, df, benchmark=benchmark_df)
                    except Exception:
                        equity_growth_score = None
                    if recalced_score < min_entry_score:
                        log_candidate_outcome(
                            f"no entry, refreshed score {recalced_score:.2f} below min {min_entry_score:.2f}",
                            "none",
                        )
                        continue
                    price = safe_float(df.iloc[-1].get("Close"), equity_price)
                    if price is None or price <= 0:
                        log_candidate_outcome(
                            "bullish entry blocked, invalid last close",
                            "none",
                        )
                        continue
                    stop_loss_pct = stock_stop_default
                    take_profit_pct = stock_take_default
                    indicators = _extract_indicator_features(df)
                    qty, target_notional = compute_position_qty(
                        symbol,
                        "equity",
                        price,
                        equity,
                        profile,
                        position_multiplier=position_multiplier,
                        min_entry_notional=min_entry_notional,
                    )
                    if equity > 0 and price > 0:
                        max_equity_dollar = equity * BALANCED_GROWTH_CONFIG["max_equity_position_pct"]
                        # Nudge sizing toward target number of positions
                        if current_equity_positions >= BALANCED_GROWTH_CONFIG["target_num_equity_positions"]:
                            max_equity_dollar *= 0.8
                        shares_cap = max(1, int(math.floor(max_equity_dollar / price)))
                        qty = min(qty, shares_cap) if qty > 0 else shares_cap
                    if PAPER_MAX_POSITION_NOTIONAL and price > 0:
                        paper_cap = int(PAPER_MAX_POSITION_NOTIONAL // price)
                        if paper_cap > 0:
                            qty = min(qty, paper_cap)
                    if gross_notional + (qty * price) > equity * max_total_allocation:
                        if not allocation_warning_logged:
                            summary_lines.append(
                                "Max portfolio allocation reached; skipping new entries."
                            )
                            allocation_warning_logged = True
                        log_candidate_outcome(
                            "bullish entry blocked by portfolio allocation",
                            "none",
                        )
                        break
                    if qty <= 0:
                        summary_lines.append(
                            f"Skip {symbol}; notional ${target_notional:.2f} too small for price ${price:.2f}."
                        )
                        log_candidate_outcome(
                            "bullish entry blocked by sizing rules",
                            "none",
                        )
                        continue
                    try:
                        logger.info(
                            "Placing bullish order: symbol=%s asset_class=stock qty=%s side=buy growth_score=%s",
                            symbol,
                            qty,
                            equity_growth_score,
                        )
                        log_ai_snapshot(
                            symbol=symbol,
                            asset_class="equity",
                            strategy_key=strategy_key,
                            contract_symbol="",
                            direction="buy",
                            score=safe_float(score, 0.0) or 0.0,
                            spot_price=price,
                            entry_price=price,
                            rsi=indicators.get("rsi"),
                            macd=indicators.get("macd"),
                            volatility_20d=indicators.get("volatility_20d"),
                            volume_rel_20d=indicators.get("volume_rel_20d"),
                            sector_strength=None,
                            market_trend=None,
                            congress_score=None,
                            news_sentiment=None,
                        )
                        place_guarded_paper_order(
                            symbol,
                            qty,
                            "buy",
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            time_in_force="gtc",
                        )
                        gross_notional += qty * price
                        cash_balance = max(0.0, cash_balance - (qty * price))
                        cash_ratio = (cash_balance / equity) if equity else cash_ratio
                        available_slots -= 1
                        held_and_pending.add(symbol)
                        pending_underlyings.add(symbol)
                        _remember_position_params(
                            symbol,
                            asset_class="equity",
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            strategy=strategy_key,
                            opened_at=now,
                            highest_price=price,
                        )
                        summary_lines.append(
                            f"Buy {qty} {symbol} (score {score:.2f}, stop {stop_loss_pct*100:.1f}%, take {take_profit_pct*100:.1f}%)"
                        )
                        orders_placed += 1
                        entries_placed += 1
                        log_candidate_outcome(
                            f"bullish entry placed via stock ({decision_reason})"
                        )
                    except Exception as exc:
                        logger.exception("Autopilot equity entry failed for %s", symbol)
                        errors.append(f"buy {symbol} failed: {exc}")
                        if not candidate_logged:
                            log_candidate_outcome(
                                "bullish entry failed during order placement",
                                "bullish",
                            )

                if not candidate_logged:
                    log_candidate_outcome("no entry, evaluation complete", "none")

            for rec in recs_snapshot:
                symbol = str(rec.get("Symbol", "")).upper()
                if not symbol or symbol in logged_candidates:
                    continue
                bearish_flag = symbol in bearish_symbols
                score_key = "BearishScore" if bearish_flag else "Score"
                score = safe_float(rec.get(score_key))
                score_fragment = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
                reason = "no entry, filtered before evaluation"
                logger.info(
                    "Candidate %s: direction=%s reason=%s score=%s bullish_considered=%s bearish_considered=%s bias=%s",
                    symbol,
                    "none",
                    reason,
                    score_fragment,
                    not bearish_flag,
                    bearish_flag,
                    "bearish" if bearish_flag else "bullish",
                )
                logged_candidates.add(symbol)

            if not summary_lines:
                summary_lines.append("Cycle complete with no trades.")
    except Exception as exc:
        logger.exception("Autopilot cycle failed")
        errors.append(str(exc))
    finally:
        with _autopilot_lock:
            _autopilot_state["last_run"] = now
            _autopilot_state["last_actions"] = summary_lines or ["Cycle complete with no trades."]
            _autopilot_state["last_error"] = "; ".join(errors) if errors else None
        _autopilot_runtime_lock.release()

        summary_text = " | ".join(summary_lines) if summary_lines else "Cycle complete with no trades."
        error_text = "; ".join(errors) if errors else None
        _record_autopilot_run(not errors, summary_text, error_text)
        logger.info(
            "Autopilot cycle finished (%s); candidates=%d, orders=%d, errors=%d",
            "forced" if force else "scheduled",
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
    prev_close = last_value(df["Close"].shift(1))
    latest_rsi = last_value(df["RSI"])
    latest_sma50 = last_value(df["SMA_50"])
    latest_sma200 = last_value(df["SMA_200"])
    company_name = None
    if yf is not None:
        try:
            info_fast = yf.Ticker(ticker).fast_info  # type: ignore[attr-defined]
            company_name = getattr(info_fast, "long_name", None) or getattr(info_fast, "short_name", None)
        except Exception:
            company_name = None
        if not company_name:
            try:
                info_full = yf.Ticker(ticker).info  # type: ignore[attr-defined]
                if isinstance(info_full, dict):
                    company_name = info_full.get("longName") or info_full.get("shortName")
            except Exception:
                company_name = None

    sentiment = fetch_news_sentiment(ticker) + fetch_reddit_sentiment(ticker)
    if not company_name:
        company_name = ""
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
        "Company": company_name,
        "Close": round(latest_close, 2) if latest_close is not None else None,
        "PrevClose": round(prev_close, 2) if prev_close is not None else None,
        "Change": round(latest_close - prev_close, 2) if latest_close is not None and prev_close is not None else None,
        "ChangePct": round(((latest_close - prev_close) / prev_close) * 100, 2) if latest_close is not None and prev_close else None,
        "RSI": round(latest_rsi, 2) if latest_rsi is not None else None,
        "50-day MA": round(latest_sma50, 2) if latest_sma50 is not None else None,
        "200-day MA": round(latest_sma200, 2) if latest_sma200 is not None else None,
        "Recommendation": rec,
        "Analysis Summary": analysis_summary,
        "Chart Data": chart_data,
        "Updated": datetime.now(timezone.utc).isoformat(),
        "Market Status": "Market open" if is_regular_equity_trading_hours() else "Market closed",
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
                if not _recommendations.get("bullish"):
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
                bullish_score = max(score, 0.0)
                bearish_score, bearish_reasons = score_bearish(df)
                bearish_score = max(bearish_score, 0.0)
                latest = df.iloc[-1]
                bearish_meta = {
                    "below_sma50": bool(
                        pd.notna(latest.get("SMA_50")) and latest["Close"] < latest["SMA_50"]
                    ),
                    "below_sma200": bool(
                        pd.notna(latest.get("SMA_200")) and latest["Close"] < latest["SMA_200"]
                    ),
                    "rsi": float(latest.get("RSI")) if pd.notna(latest.get("RSI")) else None,
                    "atr_pct": float(latest.get("ATR_pct"))
                    if pd.notna(latest.get("ATR_pct"))
                    else None,
                }
                rec = "BUY" if score >= 3.5 else "HOLD"
                return {
                    "Symbol": sym,
                    "Recommendation": rec,
                    "Score": bullish_score,
                    "BearishScore": bearish_score,
                    "Why": reasons,
                    "WhyBearish": bearish_reasons,
                    "Trend": "down" if bearish_score > bullish_score else "up",
                    "BearishMeta": bearish_meta,
                }
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
        bearish_candidates: list[dict[str, Any]] = []
        for entry in results:
            bearish_score = safe_float(entry.get("BearishScore"))
            meta = entry.get("BearishMeta") or {}
            if bearish_score is None or bearish_score <= 0:
                continue
            if not (meta.get("below_sma50") and meta.get("below_sma200")):
                continue
            bearish_candidates.append(entry)

        bearish_candidates.sort(
            key=lambda x: (-safe_float(x.get("BearishScore"), 0.0), x.get("Symbol", ""))
        )

        with _lock:
            if results:
                bullish_top = results[:5]
                payload = {
                    "timestamp": completed_at.isoformat(),
                    "bullish": bullish_top,
                    "bearish": bearish_candidates[:20],
                    "top": bullish_top,
                }
                _recommendations = payload
                _rec_state["last_completed"] = completed_at
                _rec_state["last_error"] = None
            elif not _recommendations.get("bullish"):
                _recommendations = _fallback_recommendations(
                    "Unable to compute recommendations; using placeholders"
                )
        if results:
            _record_recommendations_snapshot(payload)
    except Exception as exc:
        logger.exception("Recommendation refresh failed")
        with _lock:
            _rec_state["last_error"] = str(exc)
        with _lock:
            if not _recommendations.get("bullish"):
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
        logger.info("Starting background scheduler")
        # Run both routines immediately so the UI has fresh data without waiting
        threading.Thread(target=seek_recommendations, daemon=True).start()
        trigger_autopilot_run()
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            seek_recommendations,
            "interval",
            hours=1,
            # First run is triggered manually above; schedule subsequent runs for the future
            next_run_time=datetime.now() + timedelta(hours=1),
            id="seek_recommendations",
            replace_existing=True,
        )
        _scheduler.add_job(
            run_autopilot_cycle,
            "interval",
            minutes=5,
            # Avoid double-run at startup by spacing the first scheduled cycle
            next_run_time=datetime.now() + timedelta(minutes=5),
            id="run_autopilot_cycle",
            replace_existing=True,
        )
        _scheduler.start()
        _background_jobs_started = True
    logger.info("Background jobs started")
    job_seek = _scheduler.get_job("seek_recommendations")
    job_auto = _scheduler.get_job("run_autopilot_cycle")
    if job_seek:
        logger.info(
            "Job 'seek_recommendations' scheduled; next run at %s",
            job_seek.next_run_time,
        )
    else:
        logger.warning("Job 'seek_recommendations' not found after scheduler start")

    if job_auto:
        logger.info(
            "Job 'run_autopilot_cycle' scheduled; next run at %s",
            job_auto.next_run_time,
        )
    else:
        logger.warning("Job 'run_autopilot_cycle' not found after scheduler start")

def _ensure_background_jobs() -> None:
    if not _background_jobs_started:
        start_background_jobs()


app.before_request(_ensure_background_jobs)

# Start background tasks as soon as the module is imported (e.g., via `flask run`)
start_background_jobs()


@app.route("/scheduler-status")
def scheduler_status():
    global _scheduler
    if _scheduler is None:
        return {"scheduler": "not_started"}, 200

    jobs = _scheduler.get_jobs()
    data = {
        "scheduler": "running",
        "jobs": [
            {
                "id": job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in jobs
        ],
    }
    return data, 200


# -----------------------------
# CSV dashboard helpers
# -----------------------------


def classify_dataset(df: pd.DataFrame) -> str:
    """Classify the uploaded dataset so we can toggle trading-aware views."""

    cols = {c.lower() for c in df.columns}

    trading_core = {
        "row_type",
        "symbol",
        "side",
        "order_type",
        "status",
        "order_id",
        "mode_or_strategy",
        "strategy_name",
    }
    trading_optional = {
        "realized_pl",
        "realized_plpc",
        "unrealized_pl",
        "unrealized_plpc",
        "portfolio_value",
        "equity",
    }

    if trading_core.issubset(cols):
        return "trading_paper_trades"
    if "row_type" in cols and "symbol" in cols and ("realized_pl" in cols or "unrealized_pl" in cols):
        return "trading_like"

    return "generic"


def _build_trading_charts(df: pd.DataFrame) -> dict[str, str]:
    """Build trading-specific charts when the CSV comes from the stock app."""

    charts: dict[str, str] = {}
    if px is None:
        return charts

    cols = {c.lower(): c for c in df.columns}

    def col(name: str) -> str | None:
        return cols.get(name.lower())

    row_type_col = col("row_type")
    if not row_type_col:
        return charts

    account_df = df[df[row_type_col] == "account_summary"].copy()
    positions_df = df[df[row_type_col] == "position"].copy()
    trades_df = df[df[row_type_col] == "trade"].copy()

    ts_col = col("timestamp")
    equity_col = col("equity") or col("portfolio_value")
    if not account_df.empty and ts_col and equity_col:
        try:
            account_df[ts_col] = pd.to_datetime(account_df[ts_col], errors="coerce")
            account_df = account_df.dropna(subset=[ts_col])
            account_df = account_df.sort_values(by=ts_col)

            fig_equity = px.line(
                account_df,
                x=ts_col,
                y=equity_col,
                title="Equity over time",
                labels={ts_col: "Time", equity_col: "Equity"},
            )
            charts["equity_curve"] = fig_equity.to_html(full_html=False)
        except Exception:
            pass

    realized_col = col("realized_pl")
    mode_col = col("mode_or_strategy")
    strat_col = col("strategy_name")

    trade_pl = trades_df.copy()
    if realized_col and realized_col in trade_pl.columns:
        trade_pl = trade_pl[pd.to_numeric(trade_pl[realized_col], errors="coerce").notna()]
        if not trade_pl.empty:
            trade_pl[realized_col] = pd.to_numeric(trade_pl[realized_col], errors="coerce")

            strategy_key = strat_col or mode_col
            if strategy_key and strategy_key in trade_pl.columns:
                group = (
                    trade_pl.groupby(strategy_key)[realized_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=realized_col, ascending=False)
                )
                if not group.empty:
                    fig_strat = px.bar(
                        group,
                        x=strategy_key,
                        y=realized_col,
                        title="Realized P/L by strategy",
                        labels={strategy_key: "Strategy", realized_col: "Realized P/L"},
                    )
                    charts["strategy_performance"] = fig_strat.to_html(full_html=False)

            sym_col = col("symbol")
            if sym_col and sym_col in trade_pl.columns:
                sym_group = (
                    trade_pl.groupby(sym_col)[realized_col]
                    .sum()
                    .reset_index()
                    .sort_values(by=realized_col, ascending=False)
                )
                sym_group["abs_pl"] = sym_group[realized_col].abs()
                sym_group = sym_group.sort_values(by="abs_pl", ascending=False).head(20)

                if not sym_group.empty:
                    fig_sym = px.bar(
                        sym_group,
                        x=sym_col,
                        y=realized_col,
                        title="Realized P/L by symbol (top 20)",
                        labels={sym_col: "Symbol", realized_col: "Realized P/L"},
                    )
                    charts["symbol_performance"] = fig_sym.to_html(full_html=False)

    if not trades_df.empty and ts_col:
        price_col = col("filled_avg_price") or col("avg_entry_price") or col("current_price")
        side_col = col("side")
        sym_col = col("symbol")

        if price_col and price_col in trades_df.columns:
            try:
                trades_copy = trades_df.copy()
                trades_copy[ts_col] = pd.to_datetime(trades_copy[ts_col], errors="coerce")
                trades_copy = trades_copy.dropna(subset=[ts_col])
                trades_copy[price_col] = pd.to_numeric(trades_copy[price_col], errors="coerce")
                trades_copy = trades_copy.dropna(subset=[price_col])

                fig_timeline = px.scatter(
                    trades_copy,
                    x=ts_col,
                    y=price_col,
                    color=side_col if side_col in trades_copy.columns else None,
                    hover_data=[sym_col] if sym_col in trades_copy.columns else None,
                    title="Trade timeline",
                    labels={ts_col: "Time", price_col: "Price"},
                )
                charts["trade_timeline"] = fig_timeline.to_html(full_html=False)
            except Exception:
                pass

    unreal_col = col("unrealized_pl")
    sym_col = col("symbol")
    if unreal_col and sym_col and not positions_df.empty:
        try:
            pos_copy = positions_df.copy()
            pos_copy[unreal_col] = pd.to_numeric(pos_copy[unreal_col], errors="coerce")
            pos_copy = pos_copy.dropna(subset=[unreal_col])

            if not pos_copy.empty:
                fig_unreal = px.bar(
                    pos_copy.sort_values(by=unreal_col, ascending=False),
                    x=sym_col,
                    y=unreal_col,
                    title="Unrealized P/L by open position",
                )
                charts["unrealized_positions"] = fig_unreal.to_html(full_html=False)
        except Exception:
            pass

    return charts


def _render_csv_preview(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Return a small HTML preview table for the uploaded CSV."""

    try:
        return (
            df.head(max_rows)
            .to_html(classes="table table-sm table-striped mb-0", index=False, border=0)
        )
    except Exception:
        return ""


def _build_paper_dashboard_charts(
    account: dict[str, Any] | None,
    positions: list[dict[str, Any]] | None,
    portfolio_history: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Visual summaries for the paper trading desk."""

    charts: dict[str, str] = {}
    if px is None:
        return charts

    positions = positions or []
    positions_df = pd.DataFrame(positions)

    cash = safe_float(account.get("cash")) if account else None
    equity = safe_float(account.get("equity")) if account else None
    portfolio_value = safe_float(account.get("portfolio_value")) if account else None
    buying_power = safe_float(account.get("buying_power")) if account else None
    last_equity = safe_float(account.get("last_equity")) if account else None

    # Allocation pie
    positions_value = 0.0
    if not positions_df.empty and "market_value" in positions_df.columns:
        positions_df["market_value"] = pd.to_numeric(positions_df["market_value"], errors="coerce")
        positions_value = positions_df["market_value"].dropna().sum()

    allocation_parts = []
    if cash is not None and cash >= 0:
        allocation_parts.append(("Cash", cash))
    if positions_value and positions_value > 0:
        allocation_parts.append(("Positions", positions_value))
    if buying_power is not None and buying_power > 0:
        allocation_parts.append(("Buying Power", buying_power))

    if allocation_parts:
        try:
            alloc_df = pd.DataFrame(allocation_parts, columns=["label", "value"])
            fig_alloc = px.pie(
                alloc_df,
                names="label",
                values="value",
                title="Account allocation",
                hole=0.35,
            )
            charts["allocation"] = fig_alloc.to_html(full_html=False)
        except Exception:
            pass

    # Market value by symbol
    if not positions_df.empty and "symbol" in positions_df.columns and "market_value" in positions_df.columns:
        try:
            positions_df["market_value"] = pd.to_numeric(positions_df["market_value"], errors="coerce")
            mv_df = positions_df.dropna(subset=["market_value"])
            mv_df = mv_df.sort_values(by="market_value", ascending=False)
            if not mv_df.empty:
                fig_mv = px.bar(
                    mv_df.head(25),
                    x="symbol",
                    y="market_value",
                    title="Market value by position",
                    labels={"symbol": "Symbol", "market_value": "Market Value"},
                )
                charts["positions_value"] = fig_mv.to_html(full_html=False)
        except Exception:
            pass

    # Unrealized P/L by symbol
    if not positions_df.empty and "unrealized_pl" in positions_df.columns and "symbol" in positions_df.columns:
        try:
            positions_df["unrealized_pl"] = pd.to_numeric(positions_df["unrealized_pl"], errors="coerce")
            pl_df = positions_df.dropna(subset=["unrealized_pl"])
            pl_df = pl_df.sort_values(by="unrealized_pl", ascending=False)
            if not pl_df.empty:
                fig_unreal = px.bar(
                    pl_df.head(25),
                    x="symbol",
                    y="unrealized_pl",
                    title="Unrealized P/L by position",
                    labels={"symbol": "Symbol", "unrealized_pl": "Unrealized P/L"},
                    color="unrealized_pl",
                    color_continuous_scale="RdYlGn",
                )
                charts["unrealized_positions"] = fig_unreal.to_html(full_html=False)
        except Exception:
            pass

    # Intraday equity curve if available
    if portfolio_history and isinstance(portfolio_history, dict):
        timestamps = portfolio_history.get("timestamp") or []
        equity_points = portfolio_history.get("equity") or portfolio_history.get("equity") or []
        try:
            if timestamps and equity_points and len(timestamps) == len(equity_points):
                hist_df = pd.DataFrame({"timestamp": timestamps, "equity": equity_points})
                hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], unit="s", errors="coerce")
                hist_df = hist_df.dropna(subset=["timestamp", "equity"])
                if not hist_df.empty:
                    fig_equity = px.area(
                        hist_df,
                        x="timestamp",
                        y="equity",
                        title="Equity (intraday)",
                        labels={"timestamp": "Time", "equity": "Equity"},
                    )
                    fig_equity.update_traces(line_color="#f0b400", fill="tozeroy")
                    charts["todays_pl"] = fig_equity.to_html(full_html=False)
        except Exception:
            pass
    # Fallback: today's P/L as a bar if history unavailable
    if "todays_pl" not in charts and equity is not None and last_equity is not None:
        try:
            todays_pl = equity - last_equity
            fig_day = px.bar(
                pd.DataFrame({"metric": ["Today's P/L"], "value": [todays_pl]}),
                x="metric",
                y="value",
                title="Today's P/L",
                labels={"value": "P/L"},
                color="value",
                color_continuous_scale="RdYlGn",
            )
            charts["todays_pl"] = fig_day.to_html(full_html=False)
        except Exception:
            pass

    return charts


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
        rec_payload = _normalize_recommendations_payload(_recommendations)
        recs = [dict(r) for r in rec_payload.get("bullish", [])]
        rec_state = dict(_rec_state)

    if not recs:
        try:
            if not rec_state.get("refreshing"):
                threading.Thread(target=seek_recommendations, daemon=True).start()
                with _lock:
                    _rec_state["refreshing"] = True
                    rec_state["refreshing"] = True
            rec_payload = _fallback_recommendations(
                "No recommendations yet; first-time scan can take a few minutes."
            )
            recs = rec_payload.get("bullish", [])
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
        payload = _normalize_recommendations_payload(_recommendations)
        state = dict(_rec_state)
    if not payload.get("bullish"):
        payload = _fallback_recommendations("No recommendations available; using placeholders")
    status = _format_recommendation_status(payload.get("bullish", []), state)
    response_payload = {
        "bullish": [dict(r) for r in payload.get("bullish", [])],
        "bearish": [dict(r) for r in payload.get("bearish", [])],
        "top": [dict(r) for r in payload.get("top", payload.get("bullish", []))],
        "timestamp": payload.get("timestamp"),
    }
    return jsonify(
        {
            "ok": True,
            "data": response_payload["bullish"],
            "bullish": response_payload["bullish"],
            "bearish": response_payload["bearish"],
            "top": response_payload["top"],
            "timestamp": response_payload.get("timestamp"),
            "status": status,
        }
    )


@app.route("/api/recommendations/refresh", methods=["POST"])
def api_refresh_recommendations():
    try:
        seek_recommendations()
        with _lock:
            payload = _normalize_recommendations_payload(_recommendations)
            state = dict(_rec_state)
        if not payload.get("bullish"):
            payload = _fallback_recommendations(
                "No recommendations available; using placeholders"
            )
        status = _format_recommendation_status(payload.get("bullish", []), state)
        response_payload = {
            "bullish": [dict(r) for r in payload.get("bullish", [])],
            "bearish": [dict(r) for r in payload.get("bearish", [])],
            "top": [dict(r) for r in payload.get("top", payload.get("bullish", []))],
            "timestamp": payload.get("timestamp"),
        }
        return jsonify(
            {
                "ok": True,
                "data": response_payload["bullish"],
                "bullish": response_payload["bullish"],
                "bearish": response_payload["bearish"],
                "top": response_payload["top"],
                "timestamp": response_payload.get("timestamp"),
                "status": status,
            }
        )
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


@app.route("/paper", methods=["GET", "POST"])
def paper_dashboard():
    account = None
    positions = []
    orders = []
    broker_error = None
    todays_pl = None
    dashboard_charts: dict[str, str] = {}
    autopilot_status = get_autopilot_status()
    portfolio_history: dict[str, Any] | None = None
    if not paper_broker.enabled:
        broker_error = "Set ALPACA_PAPER_KEY_ID and ALPACA_PAPER_SECRET_KEY to enable paper trading."
    else:
        try:
            account = paper_broker.get_account()
            positions = list(paper_broker.get_positions())
            orders = list(paper_broker.list_orders(status="all", limit=25))
            try:
                portfolio_history = paper_broker.get_portfolio_history(period="1D", timeframe="5Min")
            except Exception:
                logger.warning("Portfolio history unavailable for intraday chart")
            if account:
                try:
                    equity = float(account.get("equity") or 0.0)
                    last_equity = float(account.get("last_equity") or equity)
                    todays_pl = equity - last_equity
                except Exception:
                    todays_pl = None
            dashboard_charts = _build_paper_dashboard_charts(account, positions, portfolio_history)
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
        dashboard_charts=dashboard_charts,
    )


@app.route("/paper/export_csv")
def export_paper_csv():
    if not paper_broker.enabled:
        flash("Enable paper trading to export a CSV snapshot.")
        return redirect(url_for("paper_dashboard"))

    tzinfo = datetime.now().astimezone().tzinfo or timezone.utc
    date_param = request.args.get("date")
    target_date = datetime.now(tzinfo).date()
    if date_param:
        try:
            target_date = datetime.strptime(date_param, "%Y-%m-%d").date()
        except ValueError:
            flash("Invalid date format, expected YYYY-MM-DD. Exporting today's data.")

    autopilot_status = get_autopilot_status()
    autopilot_mode = autopilot_status.get("risk") or autopilot_status.get("strategy")
    autopilot_strategy = autopilot_status.get("strategy")

    try:
        snapshot = get_daily_account_snapshot(paper_broker, target_date, tz=tzinfo)
    except Exception:
        logger.exception("Failed to build paper CSV for %s", target_date)
        snapshot = {
            "date": target_date,
            "timezone": tzinfo,
            "as_of": datetime.now(tzinfo),
            "account": {},
            "positions": [],
            "orders": [],
        }

    df = build_paper_trades_export(
        snapshot,
        mode_or_strategy=autopilot_mode,
        strategy_name=autopilot_strategy,
    )
    df = df.reindex(columns=PAPER_TRADES_COLUMNS)
    csv_data = df.to_csv(index=False)

    logger.info("Exporting CSV for paper account date=%s rows=%d", target_date, len(df))
    filename = f'paper_trades_{target_date.isoformat()}.csv'
    response = Response(csv_data, mimetype="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


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
