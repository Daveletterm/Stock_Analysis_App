import os
import re
import json
import random
import threading
import logging
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from apscheduler.schedulers.background import BackgroundScheduler

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

# -----------------------------
# Globals & caches
# -----------------------------
_lock = threading.Lock()
_sp500 = {"tickers": [], "updated": datetime.min}
_recommendations = []
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9\.\-]{0,9}$")

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


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Close"] = df["Close"].astype(float)
    df["SMA_50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["RSI"] = compute_rsi(df["Close"])  # 14
    df["ATR14"] = compute_atr(df, 14)
    df["ATR_pct"] = (df["ATR14"] / df["Close"]).clip(lower=0)
    df["HH_20"] = df["High"].rolling(20, min_periods=1).max().astype(float)
    df["LL_20"] = df["Low"].rolling(20, min_periods=1).min().astype(float)
    df["HH_252"] = df["High"].rolling(252, min_periods=1).max().astype(float)
    df["pct_from_52w_high"] = (df["Close"] / df["HH_252"] - 1.0).replace([np.inf, -np.inf], np.nan)
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


# -----------------------------
# Data sources
# -----------------------------

def update_sp500() -> None:
    logger.info("Updating S&P 500 list if stale…")
    if datetime.now() - _sp500["updated"] > timedelta(hours=24):
        tbl = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        with _lock:
            _sp500["tickers"] = tbl["Symbol"].tolist()
            _sp500["updated"] = datetime.now()
        logger.info("Cached %d tickers", len(_sp500["tickers"]))


# -----------------------------
# Analysis
# -----------------------------

def analyze_stock(ticker: str) -> dict:
    """Compute metrics and chart data for a single ticker (flattening any MultiIndex)."""
    logger.info("Analyze %s", ticker)
    # Download – retry once if Yahoo is flaky
    hist = None
    for _ in range(2):
        try:
            hist = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if hist is not None and not hist.empty:
                break
        except Exception:
            pass
    if hist is None or hist.empty:
        logger.warning("No price data for %s", ticker)
        raise ValueError(f"No price data for {ticker}")

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
        info = yf.Ticker(ticker).info
        return (info or {}).get("averageVolume", 0) > 1_000_000
    except Exception:
        return False


def seek_recommendations() -> None:
    logger.info("Refreshing recommendations…")
    update_sp500()
    with _lock:
        tickers = _sp500["tickers"][:]

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

    hist_all = yf.download(
        filtered,
        period="1y",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    def worker(sym: str) -> dict:
        try:
            df = None
            if isinstance(hist_all.columns, pd.MultiIndex):
                try:
                    df = hist_all[sym]
                except Exception:
                    df = None
            if df is None or df.empty:
                df = yf.download(sym, period="1y", auto_adjust=True, progress=False)
            if df is None or df.empty:
                logger.warning("No hist for %s in batch", sym)
                return {"Symbol": sym, "Recommendation": "HOLD", "Score": 0.0, "Why": ["no data"]}

            # Ensure we have needed cols
            if isinstance(df["Close"], pd.DataFrame):
                if sym in df["Close"].columns:
                    df["Close"] = df["Close"][sym]
                else:
                    df["Close"] = df["Close"].iloc[:, 0]
            df = df[["High", "Low", "Close"]].dropna().copy()
            df = compute_indicators(df)
            score, reasons = score_stock(df)
            rec = "BUY" if score >= 3.5 else "HOLD"
            return {"Symbol": sym, "Recommendation": rec, "Score": score, "Why": reasons}
        except Exception as e:
            logger.exception("Worker failed for %s", sym)
            return {"Symbol": sym, "Recommendation": "HOLD", "Score": 0.0, "Why": [f"error: {e}"]}

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(worker, filtered))

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
    hist = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
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
# Data sources
# -----------------------------

def update_sp500() -> None:
    logger.info("Updating S&P 500 list if stale…")
    if datetime.now() - _sp500["updated"] > timedelta(hours=24):
        tbl = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        with _lock:
            _sp500["tickers"] = tbl["Symbol"].tolist()
            _sp500["updated"] = datetime.now()
        logger.info("Cached %d tickers", len(_sp500["tickers"]))


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
    seek_recommendations()
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


@app.route("/api/health")
def api_health():
    with _lock:
        return jsonify({"tickers_cached": len(_sp500["tickers"]), "recs": _recommendations})


if __name__ == "__main__":
    start_background_jobs()
    app.run(debug=True)
