from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

AI_LOG_PATH = Path("data/ai_training_log.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_ai.pkl"
META_PATH = MODEL_DIR / "xgb_ai_meta.json"


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")

    numeric_features = [
        "score",
        "spot_price",
        "entry_price",
        "rsi",
        "macd",
        "volatility_20d",
        "volume_rel_20d",
        "congress_score",
        "news_sentiment",
    ]
    categorical_features = ["asset_class", "direction", "strategy_key", "market_trend"]

    X = df[numeric_features + categorical_features]
    y = df["label_good_trade"].astype(int)
    return X, y, numeric_features, categorical_features


def run_training(
    data_path: Path = AI_LOG_PATH, test_size: float = 0.2, verbose: bool = True
) -> dict:
    if not data_path.exists():
        if verbose:
            print(f"{data_path} not found; aborting training.")
        return {"status": "no_data"}

    df = pd.read_csv(data_path)
    df["label_good_trade"] = df["label_good_trade"].where(pd.notna(df["label_good_trade"]), np.nan)
    df = df.dropna(subset=["label_good_trade", "realized_return_5d", "timestamp"])
    if len(df) < 20:
        if verbose:
            print(f"Not enough labeled rows ({len(df)}); aborting training.")
        return {"status": "insufficient_rows", "rows": len(df)}

    X, y, num_feats, cat_feats = _build_features(df)
    # Simple time-based split
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    numeric_transformer = Pipeline(steps=[("imputer", "passthrough")])
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_feats),
            ("cat", categorical_transformer, cat_feats),
        ]
    )

    clf = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=42,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, MODEL_PATH)
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "model_path": str(MODEL_PATH),
        "metrics": metrics,
        "params": clf.get_params(),
        "features": num_feats + cat_feats,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"Training complete. AUC={metrics['auc']:.3f}, accuracy={metrics['accuracy']:.3f}")
        print(f"Model saved to {MODEL_PATH}")
    return {"status": "ok", **metrics}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost model on AI training log.")
    parser.add_argument("--data", type=Path, default=AI_LOG_PATH, help="Path to ai_training_log.csv")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (time-based split)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_training(data_path=args.data, test_size=args.test_size)


if __name__ == "__main__":
    main()
