"""
ML Step: Train a multi-class risk classifier on the cleaned CKD data.

Input:  data/clean_kidney_disease.csv
Output: outputs/model_metrics.json
        outputs/feature_importance.csv
        outputs/confusion_matrix.csv
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLEAN_PATH = "../data/clean_kidney_disease.csv"
OUT_DIR = "../outputs"

RISK_LABELS = ["no_disease", "low_risk", "moderate_risk", "high_risk", "severe_disease"]


def main():
    df = pd.read_csv(CLEAN_PATH)

    # Target: multi-class risk level (0-4), already encoded in ETL step
    y = df["risk_level"]
    X = df.drop(columns=["target", "risk_level"])

    # Encode categorical (yes/no, good/poor, etc.) columns
    cat_cols = X.select_dtypes(include="object").columns
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",  # dataset is imbalanced (16k vs 400 severe cases)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")

    print(f"\nAccuracy:          {acc:.4f}")
    print(f"F1 (macro avg):    {f1_macro:.4f}")
    print(f"F1 (weighted avg): {f1_weighted:.4f}\n")
    report = classification_report(y_test, preds, target_names=RISK_LABELS, output_dict=True)
    print(classification_report(y_test, preds, target_names=RISK_LABELS))

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
    print("Top 10 most important features:")
    print(importance.head(10).to_string(index=False))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=RISK_LABELS, columns=RISK_LABELS)
    cm_df.to_csv(f"{OUT_DIR}/confusion_matrix.csv")

    # Save metrics summary
    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(f"{OUT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {OUT_DIR}/model_metrics.json")
    print(f"Feature importance saved to {OUT_DIR}/feature_importance.csv")
    print(f"Confusion matrix saved to {OUT_DIR}/confusion_matrix.csv")


if __name__ == "__main__":
    main()
