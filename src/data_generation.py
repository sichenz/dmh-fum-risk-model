"""
data_generation.py
------------------
Generates a synthetic behavioral health patient cohort modeled after the
clinical population served by LA County.

The synthetic data is designed to reflect:
  - ICD-10 F-code mental health diagnoses (F20–F99)
  - Emergency Department (ED) visit encounters
  - Follow-up appointment scheduling and completion
  - Demographics consistent with LA County's Medi-Cal population

This generator produces data *before* de-identification — intentionally
including quasi-identifiers and synthetic PHI — so that the downstream
de-identification pipeline (deidentification.py) has realistic input to work on.

Reference: This approach mirrors the Synthea™ open-source patient generator
methodology, adapted to Python for portability and customization to
LA County's behavioral health population profile.

HIPAA NOTE: All records generated here are entirely synthetic. No real
patient data was used. Records should still be treated as if they contain
PHI during pipeline development to practice correct data governance.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import string
import hashlib

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — LA County population parameters
# Distributions drawn from LA County published demographics reports and
# LA County American Human Development Index (2026 Portrait report)
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
fake = Faker("en_US")
Faker.seed(SEED)

# ICD-10 F-code mental health diagnoses used in LA County / HEDIS FUM measure
DIAGNOSIS_CODES = {
    "Schizophrenia Spectrum": {
        "codes": ["F20.0", "F20.1", "F20.2", "F20.3", "F20.5", "F20.9",
                  "F25.0", "F25.1", "F31.0", "F31.1"],
        "weight": 0.18,
        "severity": 3,
    },
    "Major Depressive Disorder": {
        "codes": ["F32.0", "F32.1", "F32.2", "F32.3", "F32.9",
                  "F33.0", "F33.1", "F33.2", "F33.3"],
        "weight": 0.28,
        "severity": 2,
    },
    "Bipolar Disorder": {
        "codes": ["F31.10", "F31.11", "F31.12", "F31.30", "F31.31",
                  "F31.60", "F31.61", "F31.9"],
        "weight": 0.15,
        "severity": 3,
    },
    "Anxiety Disorders": {
        "codes": ["F40.00", "F40.10", "F41.0", "F41.1", "F41.9",
                  "F42.0", "F43.10", "F43.11", "F43.12"],
        "weight": 0.22,
        "severity": 1,
    },
    "Substance Use Disorder with MH Comorbidity": {
        "codes": ["F10.10", "F10.20", "F11.10", "F11.20", "F12.10",
                  "F14.10", "F15.10", "F19.10"],
        "weight": 0.17,
        "severity": 2,
    },
}

# LA County Medi-Cal demographic distribution (approximate)
RACE_ETHNICITY = {
    "Hispanic/Latino": 0.47,
    "Black/African American": 0.18,
    "White (Non-Hispanic)": 0.16,
    "Asian/Pacific Islander": 0.11,
    "Multiracial": 0.05,
    "Other/Unknown": 0.03,
}

LANGUAGE_PREFERENCE = {
    "English": 0.55,
    "Spanish": 0.32,
    "Mandarin/Cantonese": 0.05,
    "Korean": 0.03,
    "Armenian": 0.02,
    "Other": 0.03,
}

INSURANCE_TYPE = {
    "Medi-Cal": 0.61,
    "Medi-Cal/Medicare Dual": 0.14,
    "Medicare Only": 0.08,
    "County-funded (indigent)": 0.12,
    "Commercial": 0.03,
    "Uninsured": 0.02,
}

# LA County SPA (Service Planning Areas) and their ZIP code ranges
# Using representative ZIP codes for geospatial feature engineering
SPA_ZIPS = {
    "SPA 1 - Antelope Valley": ["93534", "93535", "93536", "93550", "93551"],
    "SPA 2 - San Fernando Valley": ["91301", "91304", "91311", "91316", "91342", "91401", "91405"],
    "SPA 3 - San Gabriel Valley": ["91001", "91007", "91030", "91101", "91106", "91702", "91711"],
    "SPA 4 - Metro LA": ["90001", "90007", "90011", "90015", "90019", "90028", "90036"],
    "SPA 5 - West": ["90025", "90064", "90049", "90272", "90292", "90405"],
    "SPA 6 - South": ["90059", "90061", "90247", "90250", "90262", "90280"],
    "SPA 7 - East LA": ["90022", "90023", "90063", "90201", "90255", "90270"],
    "SPA 8 - South Bay/Harbor": ["90501", "90502", "90503", "90710", "90731", "90744"],
}

# Distance to nearest mental health outpatient provider (miles)
# Higher in SPA 1 (Antelope Valley = rural), lower in Metro
SPA_PROVIDER_DISTANCE = {
    "SPA 1 - Antelope Valley": (8.0, 4.0),   # (mean, std)
    "SPA 2 - San Fernando Valley": (3.5, 1.5),
    "SPA 3 - San Gabriel Valley": (3.0, 1.2),
    "SPA 4 - Metro LA": (1.8, 0.8),
    "SPA 5 - West": (2.2, 1.0),
    "SPA 6 - South": (2.5, 1.1),
    "SPA 7 - East LA": (2.1, 0.9),
    "SPA 8 - South Bay/Harbor": (3.0, 1.3),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_choice(choices: dict) -> str:
    keys = list(choices.keys())
    weights = list(choices.values())
    return np.random.choice(keys, p=weights)


def _generate_mrn() -> str:
    """Generate a synthetic Medical Record Number (MRN)."""
    return "MRN-" + "".join(random.choices(string.digits, k=9))


def _generate_ed_visit_date(base_date: datetime) -> datetime:
    """Generate a random ED visit date within the lookback window."""
    days_back = np.random.randint(1, 365)
    return base_date - timedelta(days=int(days_back))


def _compute_followup_failure(
    race_ethnicity: str,
    language: str,
    insurance: str,
    spa: str,
    diagnosis_category: str,
    diagnosis_severity: int,
    age: int,
    prior_ed_visits: int,
    prior_outpatient_gap_days: int,
    distance_miles: float,
    housing_instability: bool,
    medication_prescribed: bool,
    appointment_scheduled: bool,
    appointment_wait_days: int,
) -> int:
    """
    Compute synthetic follow-up failure label using a logistic-style
    probability function based on clinically plausible risk factors.

    Returns 1 if patient FAILED to complete 7-day follow-up, 0 if completed.

    This encodes known clinical and social determinants of health (SDoH)
    risk factors from the mental health readmission literature and
    LA County's own barrier analyses.
    """
    log_odds = -1.2  # baseline (30% base failure rate)

    # ── Clinical risk factors ──────────────────────────────────────────────
    if diagnosis_severity >= 3:
        log_odds += 0.4   # severe diagnoses → harder to follow up
    if prior_ed_visits > 2:
        log_odds += 0.6   # chronic high utilizers often miss follow-up
    if prior_outpatient_gap_days > 90:
        log_odds += 0.5   # long gap from outpatient care
    if not medication_prescribed:
        log_odds += 0.3   # no discharge Rx → less clinical anchoring
    if not appointment_scheduled:
        log_odds += 1.2   # no appointment at discharge → strongest predictor

    # ── Appointment accessibility ──────────────────────────────────────────
    if appointment_wait_days > 7:
        log_odds += 0.8   # wait > 7 days means FUM window already missed
    elif appointment_wait_days > 3:
        log_odds += 0.3

    # ── Social determinants of health (SDoH) ──────────────────────────────
    if housing_instability:
        log_odds += 0.7
    if distance_miles > 5.0:
        log_odds += 0.4
    if language not in ("English", "Spanish"):
        log_odds += 0.3   # limited-English proficiency with less coverage

    # ── Insurance access ──────────────────────────────────────────────────
    if insurance == "Uninsured":
        log_odds += 0.9
    elif insurance == "County-funded (indigent)":
        log_odds += 0.3

    # ── Geographic / equity factors ───────────────────────────────────────
    if "Antelope Valley" in spa:
        log_odds += 0.5   # rural access gap

    # ── Age factors ───────────────────────────────────────────────────────
    if age < 25:
        log_odds += 0.2   # young adults less engaged in outpatient care
    elif age > 65:
        log_odds += 0.15  # older adults with mobility / transportation barriers

    # ── Race/ethnicity equity signal ─────────────────────────────────────
    # NOTE: This encodes systemic inequities documented in LA County outcomes
    # data. Model fairness audits will specifically test and flag these.
    if race_ethnicity == "Black/African American":
        log_odds += 0.25  # documented disparity in follow-up completion
    elif race_ethnicity == "Hispanic/Latino" and language == "Spanish":
        log_odds += 0.15

    # ── Add noise ─────────────────────────────────────────────────────────
    log_odds += np.random.normal(0, 0.4)

    prob_failure = 1 / (1 + np.exp(-log_odds))
    return int(np.random.binomial(1, prob_failure))


# ──────────────────────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_patient_cohort(n_patients: int = 5000, as_of_date: str = "2026-01-01") -> pd.DataFrame:
    """
    Generate a synthetic behavioral health patient cohort.

    Parameters
    ----------
    n_patients : int
        Number of synthetic patient records to generate.
    as_of_date : str
        Reference date for the analysis window (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Raw synthetic EHR-style dataset with PHI fields included.
        Must be passed through deidentification.py before use in modeling.
    """
    base_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    records = []

    diag_categories = list(DIAGNOSIS_CODES.keys())
    diag_weights = [DIAGNOSIS_CODES[d]["weight"] for d in diag_categories]

    # ── Pre-generate PHI-equivalent field pools ────────────────────────────
    # These fields (names, phones, addresses, DOBs) are removed or
    # transformed a few steps downstream by the de-identification pipeline,
    # so they don't need to be individually unique — they just need to look
    # realistic. Calling Faker's providers thousands of times in a tight
    # loop is by far the slowest part of cohort generation (each call does
    # regex-based template parsing), so we draw from a smaller pre-built
    # pool instead and sample from it. This cuts generation time roughly
    # 5-10x with no change to the statistical properties of the dataset.
    pool_size = min(n_patients, 2000)
    first_name_pool = [fake.first_name() for _ in range(pool_size)]
    last_name_pool = [fake.last_name() for _ in range(pool_size)]
    phone_pool = [fake.phone_number() for _ in range(pool_size)]
    address_pool = [fake.street_address() for _ in range(pool_size)]
    dob_pool = [fake.date_of_birth(minimum_age=18, maximum_age=85) for _ in range(pool_size)]

    pool_idx = np.random.randint(0, pool_size, size=n_patients)

    for i in range(n_patients):
        # ── PHI fields (will be removed/transformed in de-id pipeline) ────
        p = pool_idx[i]
        mrn = _generate_mrn()
        first_name = first_name_pool[p]
        last_name = last_name_pool[p]
        dob = dob_pool[p]
        ssn_last4 = "".join(random.choices(string.digits, k=4))
        phone = phone_pool[p]
        address = address_pool[p]

        # ── Demographics ───────────────────────────────────────────────────
        age = (base_date.date() - dob).days // 365
        gender = np.random.choice(["Male", "Female", "Non-binary/Other"],
                                   p=[0.48, 0.49, 0.03])
        race_ethnicity = _weighted_choice(RACE_ETHNICITY)
        language = _weighted_choice(LANGUAGE_PREFERENCE)
        insurance = _weighted_choice(INSURANCE_TYPE)

        # ── Geography ──────────────────────────────────────────────────────
        spa = random.choice(list(SPA_ZIPS.keys()))
        zip_code = random.choice(SPA_ZIPS[spa])
        dist_mean, dist_std = SPA_PROVIDER_DISTANCE[spa]
        distance_to_provider = max(0.3, np.random.normal(dist_mean, dist_std))

        # ── Clinical ───────────────────────────────────────────────────────
        diag_cat = np.random.choice(diag_categories, p=diag_weights)
        diag_info = DIAGNOSIS_CODES[diag_cat]
        primary_dx = random.choice(diag_info["codes"])
        severity = diag_info["severity"]

        # Secondary diagnosis (comorbidity, ~35% of patients)
        has_comorbidity = np.random.binomial(1, 0.35)
        secondary_dx = ""
        if has_comorbidity:
            other_cats = [c for c in diag_categories if c != diag_cat]
            secondary_dx = random.choice(
                DIAGNOSIS_CODES[random.choice(other_cats)]["codes"]
            )

        # SUD comorbidity (separate flag — important LA County BHSA metric)
        sud_comorbidity = np.random.binomial(
            1, 0.38 if diag_cat != "Substance Use Disorder with MH Comorbidity" else 1.0
        )

        # ED visit
        ed_visit_date = _generate_ed_visit_date(base_date)
        ed_visit_date_str = ed_visit_date.strftime("%Y-%m-%d")
        los_hours = max(2, np.random.lognormal(mean=2.5, sigma=0.6))  # length of stay

        # Prior utilization (12-month lookback)
        prior_ed_visits_12mo = np.random.negative_binomial(1, 0.4)
        prior_inpatient_admits_12mo = np.random.negative_binomial(1, 0.7)
        days_since_last_outpatient = int(np.random.exponential(scale=75))

        # Discharge planning
        medication_prescribed_at_discharge = bool(np.random.binomial(1, 0.72))
        appointment_scheduled_at_discharge = bool(np.random.binomial(1, 0.64))
        appointment_wait_days = int(np.random.lognormal(mean=1.8, sigma=0.7)) if appointment_scheduled_at_discharge else 99

        # Social determinants
        housing_instability = bool(np.random.binomial(
            1, 0.28 + (0.20 if "Antelope Valley" in spa or race_ethnicity == "Black/African American" else 0)
        ))
        transportation_barrier = bool(np.random.binomial(1, 0.22 + (0.15 if distance_to_provider > 5 else 0)))
        homelessness_flag = bool(np.random.binomial(1, 0.15 if housing_instability else 0.04))

        # ── Target variable ────────────────────────────────────────────────
        failed_7day_followup = _compute_followup_failure(
            race_ethnicity=race_ethnicity,
            language=language,
            insurance=insurance,
            spa=spa,
            diagnosis_category=diag_cat,
            diagnosis_severity=severity,
            age=age,
            prior_ed_visits=prior_ed_visits_12mo,
            prior_outpatient_gap_days=days_since_last_outpatient,
            distance_miles=distance_to_provider,
            housing_instability=housing_instability,
            medication_prescribed=medication_prescribed_at_discharge,
            appointment_scheduled=appointment_scheduled_at_discharge,
            appointment_wait_days=appointment_wait_days,
        )

        records.append({
            # ── PHI (Safe Harbor removal targets) ──────────────────────
            "mrn": mrn,
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": dob.strftime("%Y-%m-%d"),
            "ssn_last4": ssn_last4,
            "phone_number": phone,
            "street_address": address,
            "zip_code": zip_code,
            # ──────────────────────────────────────────────────────────
            "age": age,
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "preferred_language": language,
            "insurance_type": insurance,
            "spa_region": spa,
            "distance_to_provider_miles": round(distance_to_provider, 2),
            "primary_diagnosis_code": primary_dx,
            "primary_diagnosis_category": diag_cat,
            "diagnosis_severity": severity,
            "secondary_diagnosis_code": secondary_dx,
            "sud_comorbidity": int(sud_comorbidity),
            "ed_visit_date": ed_visit_date_str,
            "ed_los_hours": round(los_hours, 1),
            "prior_ed_visits_12mo": int(prior_ed_visits_12mo),
            "prior_inpatient_admits_12mo": int(prior_inpatient_admits_12mo),
            "days_since_last_outpatient_visit": days_since_last_outpatient,
            "medication_prescribed_at_discharge": int(medication_prescribed_at_discharge),
            "appointment_scheduled_at_discharge": int(appointment_scheduled_at_discharge),
            "appointment_wait_days": appointment_wait_days,
            "housing_instability": int(housing_instability),
            "transportation_barrier": int(transportation_barrier),
            "homelessness_flag": int(homelessness_flag),
            # ── Target ────────────────────────────────────────────────
            "failed_7day_followup": failed_7day_followup,
        })

    df = pd.DataFrame(records)
    print(f"[data_generation] Generated {len(df):,} synthetic patient records.")
    print(f"[data_generation] Follow-up failure rate: {df['failed_7day_followup'].mean():.1%}")
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    df = generate_patient_cohort(n_patients=5000)
    df.to_csv("data/raw/synthetic_patients_raw.csv", index=False)
    print("[data_generation] Saved → data/raw/synthetic_patients_raw.csv")
    print(f"[data_generation] Shape: {df.shape}")
    print(df.head(3).to_string())