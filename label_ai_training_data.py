from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional, List

import pandas as pd

from paper_trading import AlpacaPaperBroker
from market_data import get_price_history, PriceDataError


AI_LOG_PATH = Path("data/ai_training_log.csv")
GOOD_TRADE_THRESHOLD = 0.05  # 5 percent

_broker = AlpacaPaperBroker()


def load_trading_calendar(start_date: date, end_date: date) -> List[date]:
    """Return sorted trading dates between start_date and end_date inclusive."""

    try:
        data = _broker._request(  # type: ignore[attr-defined]
            "GET",
            "/calendar",
            params={"start": start_date.isoformat(), "end": end_date.isoformat()},
        )
    except Exception:
        return []

    days: list[date] = []
    for entry in data or []:
        try:
            day = datetime.fromisoformat(str(entry.get("date"))).date()
            days.append(day)
        except Exception:
            continue
    days = sorted(set(days))
    return days


def get_label_date(entry_dt: datetime, calendar_days: List[date]) -> Optional[date]:
    """
    Given an entry datetime and a sorted list of trading dates, return the date
    that is 5 trading days after the entry date. If there are not enough dates
    available yet, return None.
    """

    if not calendar_days:
        return None
    entry_date = entry_dt.date()
    try:
        start_index = next(i for i, d in enumerate(calendar_days) if d >= entry_date)
    except StopIteration:
        return None
    target_index = start_index + 5
    if target_index >= len(calendar_days):
        return None
    return calendar_days[target_index]


def get_underlying_close_on(symbol: str, label_date: date) -> Optional[float]:
    """
    Fetch the daily bar for `symbol` on `label_date` and return the close price.
    Return None if no bar is available.
    """

    start = datetime.combine(label_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    try:
        df = get_price_history(symbol, start, end, interval="1d")
    except PriceDataError:
        return None
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.sort_index()
    # Index may be tz-aware; normalize to date
    for idx, row in df.iterrows():
        try:
            idx_date = idx.date() if hasattr(idx, "date") else None
        except Exception:
            idx_date = None
        if idx_date == label_date:
            close_val = row.get("Close") or row.get("close") or row.get("Adj Close")
            try:
                return float(close_val)
            except Exception:
                return None
    return None


def update_ai_labels() -> None:
    if not AI_LOG_PATH.exists():
        print("No ai_training_log.csv found. Nothing to do.")
        return

    df = pd.read_csv(AI_LOG_PATH)

    if df.empty:
        print("ai_training_log.csv is empty. Nothing to do.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    min_entry_date = df["timestamp"].min().date()
    max_entry_date = df["timestamp"].max().date()
    calendar_start = min_entry_date
    calendar_end = max_entry_date + timedelta(days=20)

    calendar_days = load_trading_calendar(calendar_start, calendar_end)
    if not calendar_days:
        print("No trading calendar days found. Cannot update labels.")
        return

    updated_count = 0
    today_utc = datetime.now(timezone.utc).date()

    for idx, row in df.iterrows():
        label_val = row.get("label_good_trade")
        if pd.notna(label_val) and label_val != "":
            continue

        ts = row.get("timestamp")
        if pd.isna(ts):
            continue

        entry_dt: datetime = ts.to_pydatetime()
        label_date = get_label_date(entry_dt, calendar_days)
        if label_date is None:
            continue
        if label_date > today_utc:
            continue

        symbol = str(row.get("symbol"))
        spot_price = row.get("spot_price")
        if pd.isna(spot_price) or spot_price is None or spot_price == 0:
            continue

        close_price = get_underlying_close_on(symbol, label_date)
        if close_price is None or close_price <= 0:
            continue

        realized_return = (close_price - float(spot_price)) / float(spot_price)
        df.at[idx, "realized_return_5d"] = realized_return
        label_good = 1 if realized_return >= GOOD_TRADE_THRESHOLD else 0
        df.at[idx, "label_good_trade"] = label_good
        updated_count += 1

    df.to_csv(AI_LOG_PATH, index=False)
    print(f"Updated labels for {updated_count} rows.")


if __name__ == "__main__":
    update_ai_labels()
