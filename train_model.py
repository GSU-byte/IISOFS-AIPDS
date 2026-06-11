"""Train a phishing detection model and save metrics/artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/phishing_sites_synthetic.csv")
MODEL_PATH = Path("models/phishing_detector.joblib")
METRICS_PATH = Path("models/metrics.json")
FEATURES_PATH = Path("models/features.json")

FEATURE_COLUMNS = [
    "url_length", "num_dots", "num_hyphens", "num_digits", "has_ip",
    "has_at_symbol", "uses_https", "suspicious_words_count", "brand_in_subdomain",
    "domain_age_days", "external_links_ratio", "forms_count", "popup_count",
    "ssl_valid", "redirect_count",
]


def train() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_dataset.py")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=260,
            max_depth=9,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, target_names=["legit", "phishing"], output_dict=True),
        "rows": int(len(df)),
        "features": FEATURE_COLUMNS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    FEATURES_PATH.write_text(json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Model saved to", MODEL_PATH)
    print(json.dumps({k: metrics[k] for k in ["accuracy", "f1", "roc_auc", "confusion_matrix"]}, indent=2))
    return metrics


if __name__ == "__main__":
    train()
