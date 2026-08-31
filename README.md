# CKD Risk Pipeline: Clinical Data Engineering + ML Risk Classification

An end-to-end data pipeline that cleans clinical data, loads it into a
relational database, trains a multi-class kidney disease risk classifier,
and exports dashboard-ready tables for Power BI reporting.

## What this project demonstrates

- **Data engineering**: ETL pipeline (cleaning, validation, outlier handling)
  from raw CSV to a query-ready SQL database
- **Database design**: normalized schema, indexes, and a reporting view
  (SQLite for portability; built with SQLAlchemy so swapping to
  PostgreSQL/MySQL in production is a one-line config change)
- **Machine learning**: a Random Forest classifier predicting CKD risk stage
  (5-class: No Disease → Low → Moderate → High → Severe) from 40+ clinical
  and lab features, achieving 99.5% accuracy with correct medical reasoning
  behind the top predictive features
- **Data validation judgment**: identifying and rejecting unreliable source
  data rather than building on top of it blindly (see below)

## Pipeline

```
raw_kidney_disease.csv
       │
       ▼  01_etl_clean.py     (clean, validate, encode)
clean_kidney_disease.csv
       │
       ▼  02_load_to_db.py    (load into SQL database)
ckd_pipeline.db
       │
       ▼  03_train_model.py   (train + evaluate risk classifier)
model_metrics.json, feature_importance.csv, confusion_matrix.csv
       │
       ▼  04_export_dashboard_data.py
dashboard-ready CSVs for Power BI
```

## A note on the dataset

This project went through two public "kidney disease" datasets (a Kaggle
dataset and a peer-reviewed clinical dataset published via Harvard Dataverse)
before building the version used here. Both were rejected after validation:
core clinical markers that should correlate strongly in real kidney patients
(e.g. eGFR and serum creatinine, which are causally linked) showed
near-zero correlation in both source files — a sign the lab values weren't
generated from real physiological relationships, despite one of them
carrying institutional ethics documentation.

Rather than train a model on data that fails basic clinical sanity checks,
this project uses a **synthetic dataset built to match published nephrology
relationships** (eGFR staging per KDIGO guidelines; correlations between
eGFR, creatinine, cystatin C, hemoglobin, and blood pressure drawn from
standard clinical literature). This is disclosed here transparently, and
`00_generate_synthetic_data.py` is included in the repo so the entire
generation logic is auditable.

**Why this matters more than it might seem:** validating a data source
before building on it is a core data engineering responsibility, not an
afterthought. A pipeline that blindly trusts bad input data produces
confidently wrong results.

## Results

- **Accuracy**: 99.5% | **F1 (macro)**: 0.989
- **Top predictive features**: estimated glomerular filtration rate (eGFR),
  serum creatinine, urine protein-to-creatinine ratio, cystatin C, blood urea
  — all clinically expected top predictors of kidney function, which is
  itself a validation that the model learned real signal rather than noise
- Note: because eGFR directly defines the risk stage boundaries (per KDIGO
  clinical guidelines), very high accuracy is expected here — this reflects
  correct staging logic, not overfitting to noise

## Why this design transfers beyond healthcare

The same pattern — rare-event / imbalanced-class risk scoring from
structured, multi-source data — applies directly to:
- Fraud detection (rare fraudulent transactions vs. many normal ones)
- Credit risk scoring
- Customer churn prediction
- Any early-warning system built on structured operational data

## Tech stack

Python (pandas, numpy, scikit-learn, SQLAlchemy) · SQLite/PostgreSQL/MySQL ·
Power BI

## Running it

```bash
cd scripts
python3 00_generate_synthetic_data.py   # optional: regenerate the data
python3 01_etl_clean.py
python3 02_load_to_db.py
python3 03_train_model.py
python3 04_export_dashboard_data.py
```
