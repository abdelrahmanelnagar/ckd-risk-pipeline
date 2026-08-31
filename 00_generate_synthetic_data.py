"""
Generate a synthetic Chronic Kidney Disease dataset with realistic clinical
relationships, for portfolio/demonstration purposes.

IMPORTANT: This is SYNTHETIC data. It is NOT real patient data. It is built
so that the relationships between variables (e.g. eGFR vs. serum creatinine,
hemoglobin vs. kidney function) follow real, published physiological patterns,
so that a model trained on it demonstrates genuine, medically sound signal --
unlike the two public datasets initially tried for this project, which failed
basic clinical sanity checks (see README for details).

Output: data/raw_kidney_disease.csv (same schema as before, so the rest of
the pipeline -- 01_etl_clean.py, 02_load_to_db.py, 03_train_model.py -- runs
unchanged).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000


def stage_from_egfr(egfr):
    if egfr >= 90:
        return "no_disease", 0
    elif egfr >= 60:
        return "low_risk", 1
    elif egfr >= 30:
        return "moderate_risk", 2
    elif egfr >= 15:
        return "high_risk", 3
    else:
        return "severe_disease", 4


def main():
    age = np.clip(np.random.normal(55, 15, N), 18, 95).round().astype(int)

    # True underlying "kidney health" latent variable driving everything else
    # (higher = healthier kidneys). Age reduces this on average.
    kidney_health = np.clip(
        np.random.normal(70, 25, N) - (age - 40) * 0.3, 2, 130
    )

    # eGFR IS the kidney health marker (with small measurement noise)
    egfr = np.clip(kidney_health + np.random.normal(0, 4, N), 2, 130)

    # Serum creatinine: real inverse relationship with eGFR
    # (approximation of the clinical Cockcroft-Gault / MDRD relationship)
    serum_creatinine = np.clip(140 / (egfr + 10) + np.random.normal(0, 0.3, N), 0.4, 15)

    # Blood urea: correlates positively with creatinine
    blood_urea = np.clip(
        15 + serum_creatinine * 9 + np.random.normal(0, 6, N), 5, 200
    )

    # Cystatin C: correlates inversely with eGFR (independent kidney marker)
    cystatin_c_level = np.clip(3.5 - (egfr / 130) * 2.5 + np.random.normal(0, 0.2, N), 0.5, 4.5)

    # Hemoglobin: anemia worsens as kidney function declines
    hemoglobin_level = np.clip(
        15.5 - (130 - egfr) / 130 * 6 + np.random.normal(0, 1.0, N), 5, 18
    )

    # Blood pressure: tends to rise with worse kidney function
    blood_pressure = np.clip(
        115 + (130 - egfr) / 130 * 35 + np.random.normal(0, 10, N), 90, 200
    ).round().astype(int)

    # Comorbidities more likely with worse kidney function
    ckd_severity = (130 - egfr) / 130
    hypertension = (np.random.random(N) < (0.2 + ckd_severity * 0.6)).astype(int)
    diabetes_mellitus = (np.random.random(N) < (0.15 + ckd_severity * 0.5)).astype(int)
    coronary_artery_disease = (np.random.random(N) < (0.05 + ckd_severity * 0.3)).astype(int)
    anemia = (hemoglobin_level < 11).astype(int)
    pedal_edema = (np.random.random(N) < (0.05 + ckd_severity * 0.4)).astype(int)
    family_history = (np.random.random(N) < 0.15).astype(int)
    smoking = (np.random.random(N) < 0.2).astype(int)

    # Urine markers worsen with disease
    albumin_in_urine = np.clip((ckd_severity * 5 + np.random.normal(0, 0.8, N)), 0, 5).round().astype(int)
    sugar_in_urine = np.clip(
        (diabetes_mellitus * 2 + np.random.normal(0, 0.6, N)), 0, 5
    ).round().astype(int)
    specific_gravity_of_urine = np.clip(
        1.025 - ckd_severity * 0.015 + np.random.normal(0, 0.003, N), 1.005, 1.03
    ).round(3)
    urine_output = np.clip(
        1800 - ckd_severity * 1200 + np.random.normal(0, 200, N), 100, 3000
    ).round().astype(int)
    urine_protein_to_creatinine_ratio = np.clip(
        ckd_severity * 4 + np.random.normal(0, 0.3, N), 0, 6
    ).round(2)

    random_blood_glucose_level = np.clip(
        100 + diabetes_mellitus * 60 + np.random.normal(0, 25, N), 70, 400
    ).round().astype(int)
    cholesterol_level = np.clip(
        180 + ckd_severity * 40 + np.random.normal(0, 30, N), 100, 350
    ).round().astype(int)
    body_mass_index = np.clip(np.random.normal(26, 5, N), 15, 45).round(1)

    sodium_level = np.clip(140 - ckd_severity * 5 + np.random.normal(0, 2, N), 125, 148).round(1)
    potassium_level = np.clip(4.2 + ckd_severity * 1.3 + np.random.normal(0, 0.3, N), 2.5, 7.5).round(2)
    packed_cell_volume = np.clip(40 - ckd_severity * 12 + np.random.normal(0, 3, N), 15, 55).round().astype(int)
    white_blood_cell_count = np.clip(
        7500 + np.random.normal(0, 1500, N), 3000, 15000
    ).round().astype(int)
    red_blood_cell_count = np.clip(5.0 - ckd_severity * 1.5 + np.random.normal(0, 0.4, N), 2.0, 6.5).round(2)
    serum_albumin_level = np.clip(4.2 - ckd_severity * 1.0 + np.random.normal(0, 0.3, N), 1.5, 5.0).round(2)
    parathyroid_hormone_level = np.clip(
        50 + ckd_severity * 350 + np.random.normal(0, 40, N), 10, 600
    ).round().astype(int)
    serum_calcium_level = np.clip(9.5 - ckd_severity * 1.5 + np.random.normal(0, 0.4, N), 6, 11).round(2)
    serum_phosphate_level = np.clip(3.5 + ckd_severity * 2.5 + np.random.normal(0, 0.4, N), 2, 9).round(2)
    c_reactive_protein_level = np.clip(
        np.random.exponential(2 + ckd_severity * 6, N), 0.1, 50
    ).round(2)
    interleukin_6_level = np.clip(
        np.random.exponential(3 + ckd_severity * 8, N), 0.1, 80
    ).round(2)
    duration_of_diabetes_mellitus = np.where(
        diabetes_mellitus == 1, np.random.randint(0, 25, N), 0
    )
    duration_of_hypertension = np.where(
        hypertension == 1, np.random.randint(0, 30, N), 0
    )

    def yn(arr):
        return np.where(arr == 1, "yes", "no")

    stages = [stage_from_egfr(e) for e in egfr]
    target = [s[0] for s in stages]
    risk_level = [s[1] for s in stages]

    df = pd.DataFrame({
        "age_of_the_patient": age,
        "blood_pressure": blood_pressure,
        "specific_gravity_of_urine": specific_gravity_of_urine,
        "albumin_in_urine": albumin_in_urine,
        "sugar_in_urine": sugar_in_urine,
        "red_blood_cells_in_urine": np.random.choice(["normal", "abnormal"], N, p=[0.85, 0.15]),
        "pus_cells_in_urine": np.random.choice(["normal", "abnormal"], N, p=[0.8, 0.2]),
        "pus_cell_clumps_in_urine": np.random.choice(["present", "not present"], N, p=[0.15, 0.85]),
        "bacteria_in_urine": np.random.choice(["present", "not present"], N, p=[0.1, 0.9]),
        "random_blood_glucose_level": random_blood_glucose_level,
        "blood_urea": blood_urea.round(2),
        "serum_creatinine": serum_creatinine.round(2),
        "sodium_level": sodium_level,
        "potassium_level": potassium_level,
        "hemoglobin_level": hemoglobin_level.round(2),
        "packed_cell_volume": packed_cell_volume,
        "white_blood_cell_count": white_blood_cell_count,
        "red_blood_cell_count": red_blood_cell_count,
        "hypertension": yn(hypertension),
        "diabetes_mellitus": yn(diabetes_mellitus),
        "coronary_artery_disease": yn(coronary_artery_disease),
        "appetite": np.random.choice(["good", "poor"], N, p=[0.75, 0.25]),
        "pedal_edema": yn(pedal_edema),
        "anemia": yn(anemia),
        "estimated_glomerular_filtration_rate": egfr.round(2),
        "urine_protein_to_creatinine_ratio": urine_protein_to_creatinine_ratio,
        "urine_output": urine_output,
        "serum_albumin_level": serum_albumin_level,
        "cholesterol_level": cholesterol_level,
        "parathyroid_hormone_level": parathyroid_hormone_level,
        "serum_calcium_level": serum_calcium_level,
        "serum_phosphate_level": serum_phosphate_level,
        "family_history_of_chronic_kidney_disease": yn(family_history),
        "smoking_status": yn(smoking),
        "body_mass_index": body_mass_index,
        "physical_activity_level": np.random.choice(["low", "moderate", "high"], N, p=[0.35, 0.45, 0.2]),
        "duration_of_diabetes_mellitus": duration_of_diabetes_mellitus,
        "duration_of_hypertension": duration_of_hypertension,
        "cystatin_c_level": cystatin_c_level.round(2),
        "urinary_sediment_microscopy_results": np.random.choice(["normal", "abnormal"], N, p=[0.7, 0.3]),
        "c_reactive_protein_level": c_reactive_protein_level,
        "interleukin_6_level": interleukin_6_level,
        "Target": [t.replace("_", " ").title().replace(" ", "_") for t in target],
    })

    # Match the Target casing style used previously
    label_map = {
        "no_disease": "No_Disease",
        "low_risk": "Low_Risk",
        "moderate_risk": "Moderate_Risk",
        "high_risk": "High_Risk",
        "severe_disease": "Severe_Disease",
    }
    df["Target"] = [label_map[t] for t in target]

    out_path = "../data/raw_kidney_disease.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(df["Target"].value_counts())
    print()
    print("Sanity check -- correlations (should be strong, unlike prior datasets):")
    print("eGFR vs serum_creatinine:", np.corrcoef(egfr, serum_creatinine)[0, 1].round(3))
    print("eGFR vs blood_urea:", np.corrcoef(egfr, blood_urea)[0, 1].round(3))
    print("eGFR vs hemoglobin:", np.corrcoef(egfr, hemoglobin_level)[0, 1].round(3))
    print("eGFR vs cystatin_c:", np.corrcoef(egfr, cystatin_c_level)[0, 1].round(3))


if __name__ == "__main__":
    main()
