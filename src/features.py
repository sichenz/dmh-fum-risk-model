"""
features.py
-----------
Feature engineering pipeline for the FUM (Follow-Up After ED Visit for
Mental Illness) risk stratification model.

Derives clinically meaningful predictors from the de-identified patient
dataset. Feature design is grounded in:

  1. HEDIS FUM measure specifications (NCQA)
  2. LA County barrier analysis documentation (QAPI Work Plan 2025-2026)
  3. Mental health readmission prediction literature
     (e.g., MIMIC-IV cohort studies, HCUP NRD analyses)

Feature groups:
  A. Clinical severity features
  B. Utilization history features
  C. Discharge planning quality features
  D. Social determinants of health (SDoH) features
  E. Geographic / access features
  F. Equity-sensitive features (used in fairness audit, not directly in model)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="[features] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Feature group definitions
# ──────────────────────────────────────────────────────────────────────────────

CLINICAL_FEATURES = [
    "diagnosis_severity",
    "sud_comorbidity",
    "has_secondary_diagnosis",
    "ed_los_hours_log",
    "prior_ed_visits_12mo",
    "prior_inpatient_admits_12mo",
    "prior_ed_visits_capped",       # capped at 10 to reduce outlier leverage
    "care_gap_days_log",            # log-transformed days since last outpatient
    "high_utilizer_flag",           # prior ED visits > 3
    "chronic_high_utilizer_flag",   # prior ED visits > 6
]

DISCHARGE_FEATURES = [
    "medication_prescribed_at_discharge",
    "appointment_scheduled_at_discharge",
    "appointment_wait_days_clean",   # 99 (not scheduled) → imputed separately
    "wait_exceeds_7days",            # binary: wait > 7 days (FUM window)
    "no_appointment_no_rx",          # compound risk: both absent
    "discharge_quality_score",       # composite score
]

SDOH_FEATURES = [
    "housing_instability",
    "transportation_barrier",
    "homelessness_flag",
    "limited_english_proficiency",
    "sdoh_burden_score",             # count of active SDoH flags
]

GEO_FEATURES = [
    "distance_to_provider_miles",
    "distance_over_5mi",
    "spa_region_encoded",
    "rural_spa_flag",
]

DEMOGRAPHIC_FEATURES = [
    "age_midpoint",                  # midpoint of age band
    "gender_encoded",
    "insurance_encoded",
]

# Sensitive attributes — used in fairness audit ONLY (not model input)
SENSITIVE_ATTRIBUTES = [
    "race_ethnicity",
    "preferred_language",
    "insurance_type",
    "spa_region",
]

ALL_MODEL_FEATURES = (
    CLINICAL_FEATURES
    + DISCHARGE_FEATURES
    + SDOH_FEATURES
    + GEO_FEATURES
    + DEMOGRAPHIC_FEATURES
)

TARGET = "failed_7day_followup"


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering functions
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the de-identified dataset.

    Returns a DataFrame with derived features appended.
    Original columns are retained for interpretability.
    """
    df = df.copy()

    # ── A. Clinical severity features ─────────────────────────────────────

    # Binary: has secondary diagnosis
    df["has_secondary_diagnosis"] = (
        df["secondary_diagnosis_code"].notna()
        & (df["secondary_diagnosis_code"] != "")
    ).astype(int)

    # Log-transform ED length of stay (right-skewed)
    df["ed_los_hours_log"] = np.log1p(df["ed_los_hours"])

    # Cap prior ED visits (extreme outliers distort tree splits)
    df["prior_ed_visits_capped"] = df["prior_ed_visits_12mo"].clip(upper=10)

    # Log-transform care gap (days since last outpatient visit)
    df["care_gap_days_log"] = np.log1p(df["days_since_last_outpatient_visit"])

    # High utilizer flags (threshold-based risk stratifiers)
    df["high_utilizer_flag"] = (df["prior_ed_visits_12mo"] > 3).astype(int)
    df["chronic_high_utilizer_flag"] = (df["prior_ed_visits_12mo"] > 6).astype(int)

    # ── B. Discharge planning quality features ────────────────────────────

    # Clean appointment wait days: when no appointment scheduled, set to 14
    # (a conservative assumption that they'd wait > the FUM window)
    df["appointment_wait_days_clean"] = np.where(
        df["appointment_scheduled_at_discharge"] == 0,
        14,
        df["appointment_wait_days"].clip(upper=30),
    )

    # Binary: wait exceeds the 7-day HEDIS FUM window
    df["wait_exceeds_7days"] = (df["appointment_wait_days_clean"] > 7).astype(int)

    # Compound risk: neither appointment nor medication
    df["no_appointment_no_rx"] = (
        (df["appointment_scheduled_at_discharge"] == 0)
        & (df["medication_prescribed_at_discharge"] == 0)
    ).astype(int)

    # Composite discharge quality score (0–3, higher = better)
    df["discharge_quality_score"] = (
        df["medication_prescribed_at_discharge"]
        + df["appointment_scheduled_at_discharge"]
        + (1 - df["wait_exceeds_7days"])
    )

    # ── C. SDoH features ──────────────────────────────────────────────────

    # Limited English Proficiency flag (LEP)
    df["limited_english_proficiency"] = (
        ~df["preferred_language"].isin(["English"])
    ).astype(int)

    # SDoH burden score: count of active social risk factors
    df["sdoh_burden_score"] = (
        df["housing_instability"]
        + df["transportation_barrier"]
        + df["homelessness_flag"]
        + df["limited_english_proficiency"]
    )

    # ── D. Geographic / access features ──────────────────────────────────

    # Distance binary threshold
    df["distance_over_5mi"] = (df["distance_to_provider_miles"] > 5.0).astype(int)

    # Rural SPA flag (Antelope Valley = SPA 1)
    df["rural_spa_flag"] = df["spa_region"].str.contains(
        "Antelope Valley", na=False
    ).astype(int)

    # SPA label encoding
    spa_encoder = LabelEncoder()
    df["spa_region_encoded"] = spa_encoder.fit_transform(
        df["spa_region"].fillna("Unknown")
    )

    # ── E. Demographic features ───────────────────────────────────────────

    # Age midpoint from age band (e.g., "35-39" → 37)
    df["age_midpoint"] = df["age_band"].apply(_age_band_to_midpoint)

    # Gender encoding
    gender_map = {"Male": 0, "Female": 1, "Non-binary/Other": 2}
    df["gender_encoded"] = df["gender"].map(gender_map).fillna(2)

    # Insurance tier encoding (ordinal by access quality)
    insurance_order = {
        "Commercial": 0,
        "Medicare Only": 1,
        "Medi-Cal/Medicare Dual": 2,
        "Medi-Cal": 3,
        "County-funded (indigent)": 4,
        "Uninsured": 5,
    }
    df["insurance_encoded"] = df["insurance_type"].map(insurance_order).fillna(3)

    logger.info(f"Feature engineering complete. Derived {len(ALL_MODEL_FEATURES)} model features.")
    logger.info(f"Dataset shape after engineering: {df.shape}")

    return df


def _age_band_to_midpoint(band: str) -> float:
    """Convert '35-39' → 37.0, '90+' → 92.0, etc."""
    if pd.isna(band) or band == "Unknown":
        return 45.0  # population median fallback
    if band == "90+":
        return 92.0
    try:
        parts = band.split("-")
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) / 2
    except (ValueError, AttributeError):
        pass
    return 45.0


# ──────────────────────────────────────────────────────────────────────────────
# Train/test split
# ──────────────────────────────────────────────────────────────────────────────

def prepare_model_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Split engineered data into train/test sets.

    Returns a dictionary containing:
        X_train, X_test, y_train, y_test,
        sensitive_train, sensitive_test,
        feature_names
    """
    df_eng = engineer_features(df)

    # Drop any rows missing the target
    df_eng = df_eng.dropna(subset=[TARGET])

    available_features = [f for f in ALL_MODEL_FEATURES if f in df_eng.columns]
    missing = set(ALL_MODEL_FEATURES) - set(available_features)
    if missing:
        logger.warning(f"Missing expected features (will be skipped): {missing}")

    X = df_eng[available_features].fillna(0)
    y = df_eng[TARGET].astype(int)

    # Retain sensitive attributes for fairness audit (not in X)
    sensitive_cols = [c for c in SENSITIVE_ATTRIBUTES if c in df_eng.columns]
    sensitive = df_eng[sensitive_cols]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    logger.info(f"Train failure rate: {y_train.mean():.1%} | Test failure rate: {y_test.mean():.1%}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_train": s_train,
        "sensitive_test": s_test,
        "feature_names": available_features,
        "df_engineered": df_eng,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/processed/patients_deidentified.csv")
    data = prepare_model_data(df)
    data["X_train"].to_csv("data/interim/X_train.csv", index=False)
    data["X_test"].to_csv("data/interim/X_test.csv", index=False)
    data["y_train"].to_csv("data/interim/y_train.csv", index=False)
    data["y_test"].to_csv("data/interim/y_test.csv", index=False)
    logger.info("Saved train/test splits to data/interim/")
