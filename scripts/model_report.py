from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_xgboost import _build_features  # reuse the same feature builder

try:
    import joblib
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(f"joblib is required to load the model: {exc}")


def _top_features(model, top_n: int = 15) -> list[tuple[str, float]]:
    """Return top feature importances from the fitted pipeline."""
    try:
        preprocess = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]
    except Exception:
        return []
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(clf.feature_importances_))]
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return []
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def generate_report(
    data_path: Path,
    model_path: Path,
    test_size: float = 0.2,
) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")

    df = pd.read_csv(data_path)
    df["label_good_trade"] = df["label_good_trade"].where(pd.notna(df["label_good_trade"]), np.nan)
    df = df.dropna(subset=["label_good_trade", "realized_return_5d", "timestamp"])
    if len(df) < 20:
        raise ValueError(f"Not enough labeled rows for report: {len(df)}")

    X, y, _, _ = _build_features(df)
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    top_feats = _top_features(model, top_n=15)

    summary = {
        "rows": len(df),
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "auc": auc,
        "accuracy": acc,
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "confusion_matrix": cm,
        "top_features": top_feats,
        "classification_report": report,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trained XGBoost model and holdout metrics.")
    parser.add_argument("--data", type=Path, default=Path("data/ai_training_log.csv"), help="Path to ai_training_log.csv")
    parser.add_argument("--model", type=Path, default=Path("models/xgb_ai.pkl"), help="Path to saved model pickle")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for time-based test split")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write JSON report")
    args = parser.parse_args()

    summary = generate_report(args.data, args.model, test_size=args.test_size)

    print(f"Rows total: {summary['rows']} (train {summary['rows_train']}, test {summary['rows_test']})")
    print(f"AUC: {summary['auc']:.3f}  Accuracy: {summary['accuracy']:.3f}  Precision: {summary['precision']:.3f}  Recall: {summary['recall']:.3f}  F1: {summary['f1']:.3f}")
    print("Confusion matrix [[tn, fp], [fn, tp]]:", summary["confusion_matrix"])
    print("Top features:")
    for name, val in summary["top_features"]:
        print(f"  {name}: {val:.4f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote report to {args.json}")


if __name__ == "__main__":
    main()
