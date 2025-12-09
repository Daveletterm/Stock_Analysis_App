import os
from pathlib import Path

import numpy as np
import pandas as pd

from train_xgboost import run_training


def test_run_training_with_synthetic_data(tmp_path: Path, monkeypatch):
    data_path = tmp_path / "ai_training_log.csv"
    ts = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["TEST"] * len(ts),
            "asset_class": ["equity"] * len(ts),
            "strategy_key": ["strat"] * len(ts),
            "contract_symbol": ["" for _ in range(len(ts))],
            "direction": ["buy"] * len(ts),
            "score": np.linspace(0, 10, len(ts)),
            "spot_price": np.linspace(10, 20, len(ts)),
            "entry_price": np.linspace(10, 20, len(ts)),
            "rsi": np.linspace(30, 70, len(ts)),
            "macd": np.linspace(-1, 1, len(ts)),
            "volatility_20d": np.linspace(0.1, 0.2, len(ts)),
            "volume_rel_20d": np.linspace(0.5, 1.5, len(ts)),
            "sector_strength": [None] * len(ts),
            "market_trend": ["up"] * len(ts),
            "congress_score": [0.0] * len(ts),
            "news_sentiment": [0.0] * len(ts),
            "decision": ["enter"] * len(ts),
            "label_good_trade": [0, 1] * (len(ts) // 2),
            "realized_return_5d": np.linspace(-0.1, 0.1, len(ts)),
        }
    )
    df.to_csv(data_path, index=False)
    result = run_training(data_path=data_path, test_size=0.2)
    assert result["status"] == "ok"
    assert result["rows_train"] > 0
    assert result["rows_test"] > 0
