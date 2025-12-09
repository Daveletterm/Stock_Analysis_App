from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from market_data import get_price_history, PriceDataError
except Exception:
    # Fallback to yfinance if project helpers are unavailable
    get_price_history = None  # type: ignore
    PriceDataError = Exception  # type: ignore
    import yfinance as yf  # type: ignore


AI_LOG_PATH = Path("data/ai_training_log.csv")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill 5-day outcomes in AI training log.")
    parser.add_argument("--as-of", dest="as_of", help="ISO date/datetime for evaluation (default: now UTC)")
    parser.add_argument("--days", dest="days", type=int, default=5, help="Trading day horizon (default: 5)")
    parser.add_argument(
        "--good-threshold",
        dest="good_threshold",
        type=float,
        default=0.0,
        help="Return threshold to label a trade as good (default: 0.0)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args()


def _load_history(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    if get_price_history is not None:
        try:
            return get_price_history(symbol, start, end, interval="1d")
        except PriceDataError:
            return None
        except Exception:
            return None
    # yfinance fallback
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=0, axis=1)
        df = df.rename(columns=str.title)
        return df
    except Exception:
        return None


def _get_close_on(df: pd.DataFrame, target_date: date) -> Optional[float]:
    try:
        for idx, row in df.sort_index().iterrows():
            idx_date = idx.date() if hasattr(idx, "date") else None
            if idx_date == target_date:
                close_val = row.get("Close") or row.get("Adj Close") or row.get("close")
                return float(close_val)
    except Exception:
        return None
    return None


def run_fill_outcomes(
    as_of: datetime | None = None, days: int = 5, good_threshold: float = 0.0, verbose: bool = True
) -> dict[str, int]:
    as_of_dt = as_of or datetime.now(timezone.utc)

    if not AI_LOG_PATH.exists():
        if verbose:
            print("ai_training_log.csv not found; nothing to do.")
        return {"updated": 0, "skipped_days": 0, "skipped_data": 0, "candidates": 0}

    df = pd.read_csv(AI_LOG_PATH)
    if df.empty:
        if verbose:
            print("ai_training_log.csv is empty; nothing to do.")
        return {"updated": 0, "skipped_days": 0, "skipped_data": 0, "candidates": 0}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    updated = 0
    skipped_days = 0
    skipped_data = 0
    total_candidates = 0
    total_rows = len(df.index)
    processed = 0

    for idx, row in df.iterrows():
        label_val = row.get("label_good_trade")
        rr_val = row.get("realized_return_5d")
        if not (pd.isna(label_val) and pd.isna(rr_val)):
            continue

        ts = row.get("timestamp")
        if pd.isna(ts):
            continue
        entry_dt: datetime = ts.to_pydatetime()
        if entry_dt > as_of_dt:
            continue

        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue

        total_candidates += 1
        processed += 1
        if verbose and processed % 25 == 0:
            print(f"Processing {processed}/{total_rows} rows...", flush=True)

        start_window = entry_dt - timedelta(days=10)
        end_window = as_of_dt + timedelta(days=days * 2 + 5)
        history = _load_history(symbol, start_window, end_window)
        if history is None or history.empty:
            skipped_data += 1
            continue

        history = history.sort_index()
        trading_days = [d.date() if hasattr(d, "date") else None for d in history.index]
        if entry_dt.date() not in trading_days:
            try:
                entry_idx = next(i for i, d in enumerate(trading_days) if d and d >= entry_dt.date())
            except StopIteration:
                skipped_data += 1
                continue
        else:
            entry_idx = trading_days.index(entry_dt.date())

        target_idx = entry_idx + days
        if target_idx >= len(trading_days):
            skipped_days += 1
            continue

        entry_date = trading_days[entry_idx]
        target_date = trading_days[target_idx]
        if entry_date is None or target_date is None:
            skipped_data += 1
            continue

        direction = str(row.get("direction", "buy")).lower()
        entry_price = row.get("entry_price")
        if pd.isna(entry_price) or entry_price is None or entry_price <= 0:
            entry_price = _get_close_on(history, entry_date)
        if entry_price is None or entry_price <= 0:
            skipped_data += 1
            continue

        close_price = _get_close_on(history, target_date)
        if close_price is None or close_price <= 0:
            skipped_data += 1
            continue

        if direction == "sell":
            realized_return = (entry_price - close_price) / entry_price
        else:
            realized_return = (close_price - entry_price) / entry_price

        df.at[idx, "realized_return_5d"] = realized_return
        df.at[idx, "label_good_trade"] = 1 if realized_return >= good_threshold else 0
        updated += 1

    df.to_csv(AI_LOG_PATH, index=False)
    if verbose:
        print(f"Processed {total_candidates} unlabeled rows; updated {updated}.")
        print(f"Skipped (insufficient trading days): {skipped_days}")
        print(f"Skipped (missing price data): {skipped_data}")
    return {
        "updated": updated,
        "skipped_days": skipped_days,
        "skipped_data": skipped_data,
        "candidates": total_candidates,
    }


def main() -> None:
    args = _parse_args()
    as_of_dt = datetime.now(timezone.utc)
    if args.as_of:
        as_of_dt = pd.to_datetime(args.as_of, utc=True).to_pydatetime()
    run_fill_outcomes(as_of=as_of_dt, days=args.days, good_threshold=args.good_threshold, verbose=not args.quiet)


if __name__ == "__main__":
    main()
