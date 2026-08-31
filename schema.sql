-- Chronic Kidney Disease (CKD) clinical database schema
-- Designed for PostgreSQL / MySQL compatibility (minor type tweaks noted below)

CREATE TABLE patients (
    patient_id                              SERIAL PRIMARY KEY,
    age_of_the_patient                      INTEGER,
    blood_pressure                          INTEGER,
    specific_gravity_of_urine               NUMERIC(5,3),
    albumin_in_urine                        INTEGER,
    sugar_in_urine                          INTEGER,
    red_blood_cells_in_urine                VARCHAR(20),
    pus_cells_in_urine                      VARCHAR(20),
    pus_cell_clumps_in_urine                VARCHAR(20),
    bacteria_in_urine                       VARCHAR(20),
    random_blood_glucose_level              INTEGER,
    blood_urea                              NUMERIC(8,2),
    serum_creatinine                        NUMERIC(6,2),
    sodium_level                            NUMERIC(8,2),
    potassium_level                         NUMERIC(6,2),
    hemoglobin_level                        NUMERIC(5,2),
    packed_cell_volume                      INTEGER,
    white_blood_cell_count                  INTEGER,
    red_blood_cell_count                    NUMERIC(5,2),
    hypertension                            VARCHAR(5),
    diabetes_mellitus                       VARCHAR(5),
    coronary_artery_disease                 VARCHAR(5),
    appetite                                VARCHAR(10),
    pedal_edema                             VARCHAR(5),
    anemia                                  VARCHAR(5),
    estimated_glomerular_filtration_rate    NUMERIC(6,2),
    urine_protein_to_creatinine_ratio       NUMERIC(6,2),
    urine_output                            INTEGER,
    serum_albumin_level                     NUMERIC(5,2),
    cholesterol_level                       INTEGER,
    parathyroid_hormone_level               NUMERIC(8,2),
    serum_calcium_level                     NUMERIC(5,2),
    serum_phosphate_level                   NUMERIC(5,2),
    family_history_of_chronic_kidney_disease VARCHAR(5),
    smoking_status                          VARCHAR(5),
    body_mass_index                         NUMERIC(5,2),
    physical_activity_level                 VARCHAR(10),
    duration_of_diabetes_mellitus           INTEGER,
    duration_of_hypertension                INTEGER,
    cystatin_c_level                        NUMERIC(5,2),
    urinary_sediment_microscopy_results     VARCHAR(20),
    c_reactive_protein_level                NUMERIC(6,2),
    interleukin_6_level                     NUMERIC(6,2),
    target                                  VARCHAR(20),
    risk_level                              INTEGER
);

-- Indexes to support the kind of queries a dashboard / risk-monitoring
-- system would run most often
CREATE INDEX idx_patients_risk_level ON patients (risk_level);
CREATE INDEX idx_patients_target ON patients (target);
CREATE INDEX idx_patients_egfr ON patients (estimated_glomerular_filtration_rate);

-- Example view: quick risk summary for a reporting dashboard
CREATE VIEW risk_summary AS
SELECT
    target,
    risk_level,
    COUNT(*)                                   AS patient_count,
    ROUND(AVG(age_of_the_patient), 1)          AS avg_age,
    ROUND(AVG(estimated_glomerular_filtration_rate), 2) AS avg_egfr,
    ROUND(AVG(serum_creatinine), 2)            AS avg_serum_creatinine,
    ROUND(AVG(blood_pressure), 1)              AS avg_blood_pressure
FROM patients
GROUP BY target, risk_level
ORDER BY risk_level;
