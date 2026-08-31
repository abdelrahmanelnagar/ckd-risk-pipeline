"""
ETL Step 1: Clean and standardize the raw Chronic Kidney Disease dataset.

Input:  data/raw_kidney_disease.csv
Output: data/clean_kidney_disease.csv

What this does:
- Standardizes column names (snake_case, no units/parentheses)
- Normalizes categorical text values (yes/no, good/poor, etc.)
- Checks for and handles nulls, duplicates, and outliers
- Encodes the target column for downstream modeling
- Logs a short data quality report to the console
"""

import pandas as pd
import numpy as np
import re

RAW_PATH = "../data/raw_kidney_disease.csv"
CLEAN_PATH = "../data/clean_kidney_disease.csv"


def snake_case(col: str) -> str:
    col = re.sub(r"\(.*?\)", "", col)          # drop units in parentheses
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    # --- 1. Standardize column names ---
    df.columns = [snake_case(c) for c in df.columns]

    # --- 2. Duplicate check ---
    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
    print(f"  Duplicates removed: {dupes}")

    # --- 3. Null check (dataset is clean, but this makes the pipeline robust) ---
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=[object]).columns
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])
    print(f"  Total nulls handled: {total_nulls}")

    # --- 4. Standardize categorical text (strip/lowercase yes-no style fields) ---
    cat_cols = df.select_dtypes(include=[object]).columns.drop("target")
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # --- 5. Outlier handling on key clinical numeric fields (IQR clipping, not removal) ---
    clinical_numeric = [
        "blood_urea", "serum_creatinine", "estimated_glomerular_filtration_rate",
        "random_blood_glucose_level", "cholesterol_level", "body_mass_index",
    ]
    clinical_numeric = [c for c in clinical_numeric if c in df.columns]
    outlier_report = {}
    for c in clinical_numeric:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        outlier_report[c] = n_outliers
        df[c] = df[c].clip(lower, upper)
    print("  Outliers clipped (3x IQR rule):")
    for k, v in outlier_report.items():
        print(f"    {k}: {v}")

    # --- 6. Encode target into an ordinal risk score (for reporting + modeling) ---
    risk_order = {
        "no_disease": 0,
        "low_risk": 1,
        "moderate_risk": 2,
        "high_risk": 3,
        "severe_disease": 4,
    }
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["risk_level"] = df["target"].map(risk_order)

    # --- 7. Save clean output ---
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nClean data saved to {CLEAN_PATH}")
    print(f"  Final shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()
