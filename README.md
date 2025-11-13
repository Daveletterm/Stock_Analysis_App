# Stock Analysis App

## Overview

This Flask application provides interactive analysis tools for publicly traded stocks.  It fetches
historical price data with [yfinance](https://pypi.org/project/yfinance/), computes technical
indicators (RSI, moving averages, ATR), surfaces top trade ideas, and exposes JSON APIs for
programmatic access.  Background jobs continuously refresh cached S&P 500 constituents and model
recommendations so that the UI can load quickly.

## Key Features

- **Ticker analysis form** – analyze an arbitrary symbol and view key metrics plus price/RSI charts.
- **Top five ideas** – cached recommendations driven by quantitative screens and heuristics.
- **Backtesting endpoint** – evaluate a ticker over a configurable lookback window to produce
  performance metrics and an equity curve suitable for visualization.
- **Paper trading desk** – submit bracketed orders against Alpaca's paper API with position sizing
  guardrails and review account balances, open positions, and recent orders.
- **JSON APIs** – `/api/analyze/<ticker>`, `/api/recommendations`, and `/api/backtest/<ticker>`
  make it simple to integrate the service with other tools.

## Local Setup (Python 3.10)

1. **Create and activate the virtual environment**

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # "./venv/Scripts/activate" on Windows PowerShell
   ```

2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Alpaca paper credentials** – either export environment variables or create a
   `.env` file (loaded automatically via [python-dotenv](https://pypi.org/project/python-dotenv/))
   alongside `app.py`:

   ```bash
   export ALPACA_PAPER_KEY_ID="your-key-id"
   export ALPACA_PAPER_SECRET_KEY="your-secret-key"
   export ALPACA_PAPER_BASE_URL="https://paper-api.alpaca.markets/v2"  # optional
   export FLASK_APP=app.py
   ```

   or create `.env` with:

   ```dotenv
   ALPACA_PAPER_KEY_ID=your-key-id
   ALPACA_PAPER_SECRET_KEY=your-secret-key
   # ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets/v2
   FLASK_APP=app.py
   ```

4. **Run the application**

   ```bash
   flask run --debug
   ```

The app serves `http://127.0.0.1:5000/` with the main analysis dashboard, `/paper` for the
paper-trading console, and `/api/...` routes for JSON integrations. Background jobs that refresh
recommendations start automatically when the Flask app boots.

> **Optional**: Additional technical indicators are enabled when
> [`pandas-ta`](https://github.com/twopirllc/pandas-ta) is installed. Install directly from GitHub if
> desired:
>
> ```bash
> pip install "pandas-ta @ git+https://github.com/twopirllc/pandas-ta@main"
> ```

### Paper trading guardrails

- Set `PAPER_MAX_POSITION_PCT` (default `0.10`) to cap each order at a percentage of account equity.
- Set `PAPER_MAX_POSITION_NOTIONAL` (default `5000`) to limit the absolute notional per order.
- Adjust `PAPER_STOP_LOSS_PCT` and `PAPER_TAKE_PROFIT_PCT` (defaults `0.05` and `0.10`) to control
  the protective bracket that wraps buy orders.
- API endpoints under `/api/paper/*` expose account, positions, orders, and allow programmatic order
  submission.

## Recommendations for Next Steps

1. **Add paper trading support**
   - Integrate with a brokerage that offers a paper-trading API (e.g., Alpaca, Interactive Brokers
     TWS demo, or Tradier Sandbox).
   - Create credentials configuration separate from production secrets.
   - Build a trade-execution module that can place simulated orders, track fills, and reconcile
     open positions.
   - Expose a UI page and corresponding APIs to submit orders from analyzed tickers and to review
     the simulated portfolio and its P&L.
   - Add guardrails (position sizing rules, stop-loss, take-profit) so the strategy can be exercised
     safely before touching real capital.

2. **Persist analytics results**
   - Store analysis snapshots, backtest results, and recommendation history in a lightweight
     database (e.g., SQLite or PostgreSQL).
   - Use the stored data to display performance trends and to evaluate how the recommendation engine
     performs over time.

3. **Enhance data coverage**
   - Enrich metrics with fundamentals (earnings surprise, analyst revisions) via a fundamentals API.
   - Incorporate real sentiment sources (news, Reddit, X/Twitter) to replace the placeholders in
     `app.py` and improve the scoring engine.

4. **User management & alerts**
   - Add authentication so multiple users can track personalized watchlists.
   - Provide email or push notifications when recommendation thresholds are met or when portfolio
     rules are triggered.

5. **Deployment readiness**
   - Containerize the app with Docker, configure logging/monitoring, and add automated tests for
     the data pipelines and API endpoints to support continuous deployment.

These enhancements will keep the repository focused on stock analysis while building a clear path
from research tooling to a production-ready, capital-trading workflow.
