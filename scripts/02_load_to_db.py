"""
ETL Step 2: Load the cleaned CKD dataset into a database.

Uses SQLite here so the project runs anywhere with zero setup (no server
required) -- but it's built with SQLAlchemy, so swapping the connection
string below to a PostgreSQL or MySQL URL is the only change needed to
run this against a real production database.

Input:  data/clean_kidney_disease.csv
Output: outputs/ckd_pipeline.db  (SQLite database)
"""

import pandas as pd
from sqlalchemy import create_engine

CLEAN_PATH = "../data/clean_kidney_disease.csv"
DB_PATH = "../outputs/ckd_pipeline.db"

# For production this would instead be something like:
#   "postgresql://user:password@host:5432/ckd_db"
#   "mysql+pymysql://user:password@host:3306/ckd_db"
CONNECTION_STRING = f"sqlite:///{DB_PATH}"


def main():
    print("Loading clean data...")
    df = pd.read_csv(CLEAN_PATH)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    engine = create_engine(CONNECTION_STRING)

    print("Writing to database table 'patients'...")
    df.to_sql("patients", engine, if_exists="replace", index_label="patient_id")

    # quick sanity check read-back
    check = pd.read_sql("SELECT COUNT(*) as n FROM patients", engine)
    print(f"  Rows confirmed in database: {check['n'].iloc[0]}")

    # build the risk_summary view equivalent as a table, for dashboard use
    summary = (
        df.groupby(["target", "risk_level"])
        .agg(
            patient_count=("target", "count"),
            avg_age=("age_of_the_patient", "mean"),
            avg_egfr=("estimated_glomerular_filtration_rate", "mean"),
            avg_serum_creatinine=("serum_creatinine", "mean"),
            avg_blood_pressure=("blood_pressure", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("risk_level")
    )
    summary.to_sql("risk_summary", engine, if_exists="replace", index=False)
    print("\nRisk summary table:")
    print(summary.to_string(index=False))

    print(f"\nDatabase saved to {DB_PATH}")


if __name__ == "__main__":
    main()
