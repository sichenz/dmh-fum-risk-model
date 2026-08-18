"""
deidentification.py
-------------------
Implements a HIPAA Safe Harbor de-identification pipeline per
45 CFR §164.514(b)(2) — the "Safe Harbor" method for de-identification
of Protected Health Information (PHI).

The 18 PHI identifier categories that must be removed or transformed:
  1.  Names
  2.  Geographic data (street addresses; ZIPs → first 3 digits if pop > 20,000)
  3.  Dates (except year; ages ≥ 90 → "90+")
  4.  Phone numbers
  5.  Fax numbers
  6.  Email addresses
  7.  Social Security numbers
  8.  Medical record numbers → replaced with token
  9.  Health plan beneficiary numbers
  10. Account numbers
  11. Certificate/license numbers
  12. Vehicle identifiers
  13. Device identifiers
  14. Web URLs
  15. IP addresses
  16. Biometric identifiers (fingerprints, voice)
  17. Full-face photographs
  18. Any other unique identifying number

Reference: HHS Office for Civil Rights — Guidance Regarding Methods for
De-identification of PHI in Accordance with HIPAA Privacy Rule (2012)
https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/

LACDMH Context: This pipeline mirrors the data governance practices
required for IBHIS (Integrated Behavioral Health Information System)
data used in cross-departmental analytics, as described in LACDMH's
Chief Information Office Bureau data management policies.
"""

import hashlib
import re
import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="[deidentification] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Safe Harbor de-identification rules (45 CFR §164.514(b)(2))
# ──────────────────────────────────────────────────────────────────────────────

# PHI columns that must be REMOVED outright
PHI_REMOVE_COLS = [
    "first_name",
    "last_name",
    "phone_number",
    "street_address",
    "ssn_last4",
]

# PHI columns that require TRANSFORMATION (not simple removal)
PHI_TRANSFORM_COLS = {
    "mrn": "token",          # → one-way hash token (preserves linkage potential)
    "date_of_birth": "year_only",  # → birth year only
    "zip_code": "truncate_3",      # → first 3 digits (per Safe Harbor rule)
    "age": "bin_90plus",           # → ages 90+ collapsed to "90+"
    "ed_visit_date": "year_only",  # → year only (removes month/day)
}


def _hash_token(value: str, salt: str = "LACDMH-DEIDENT-2026") -> str:
    """
    One-way SHA-256 hash of an identifier for pseudonymization.
    Allows within-dataset linkage without exposing the original value.

    NOTE: This is pseudonymization, not full anonymization. Tokens should
    still be treated as sensitive. For public release, apply additional
    k-anonymity checks.
    """
    raw = f"{salt}:{value}"
    return "TOK-" + hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


def _truncate_zip(zip_code: str) -> str:
    """
    Truncate ZIP code to first 3 digits per Safe Harbor.
    Exception: If the 3-digit prefix corresponds to a geographic area with
    population < 20,000, replace with '000' per the rule.

    For LA County, all SPAs have population >> 20,000, so we retain first 3.
    """
    if pd.isna(zip_code):
        return "000"
    zc = str(zip_code).strip()[:5]
    if len(zc) >= 3:
        return zc[:3] + "XX"
    return "000"


def _extract_year(date_str: str) -> Optional[int]:
    """Extract year from a YYYY-MM-DD date string."""
    if pd.isna(date_str) or str(date_str).strip() == "":
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return None


def _bin_age(age: int) -> str:
    """
    Convert age to Safe Harbor-compliant age band.
    Ages ≥ 90 are collapsed to '90+' per 45 CFR §164.514(b)(2)(i)(C).
    Ages < 90 are binned into 5-year intervals for analysis utility.
    """
    if pd.isna(age):
        return "Unknown"
    age = int(age)
    if age >= 90:
        return "90+"
    lower = (age // 5) * 5
    return f"{lower}-{lower + 4}"


def _compute_birth_year_age(birth_year: Optional[int], reference_year: int = 2026) -> Optional[int]:
    """Derive approximate age from birth year (after DOB has been year-truncated)."""
    if birth_year is None:
        return None
    return reference_year - birth_year


# ──────────────────────────────────────────────────────────────────────────────
# Audit trail
# ──────────────────────────────────────────────────────────────────────────────

class DeidentificationAudit:
    """
    Tracks what transformations were applied for documentation purposes.
    In production, this log would be stored in a secure audit database
    accessible only to the Privacy Officer.
    """
    def __init__(self):
        self.actions = []

    def log(self, field: str, action: str, n_records: int):
        self.actions.append({
            "field": field,
            "action": action,
            "n_records_affected": n_records,
        })

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.actions)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_safe_harbor_deidentification(
    df: pd.DataFrame,
    reference_year: int = 2026,
    verbose: bool = True,
) -> tuple[pd.DataFrame, DeidentificationAudit]:
    """
    Apply HIPAA Safe Harbor de-identification to a raw patient DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing PHI fields (as generated by data_generation.py).
    reference_year : int
        The analysis reference year for age derivation.
    verbose : bool
        Log transformation steps.

    Returns
    -------
    df_deident : pd.DataFrame
        De-identified dataset safe for analytical use.
    audit : DeidentificationAudit
        Log of all transformations applied.
    """
    audit = DeidentificationAudit()
    df = df.copy()

    # ── Step 1: Remove direct PHI identifiers ─────────────────────────────
    for col in PHI_REMOVE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            audit.log(col, "REMOVED (direct PHI identifier)", len(df))
            if verbose:
                logger.info(f"REMOVED column: {col}")

    # ── Step 2: Pseudonymize MRN ─────────────────────────────────────────
    if "mrn" in df.columns:
        df["patient_token"] = df["mrn"].apply(_hash_token)
        df = df.drop(columns=["mrn"])
        audit.log("mrn", "PSEUDONYMIZED → patient_token (SHA-256)", len(df))
        if verbose:
            logger.info("PSEUDONYMIZED: mrn → patient_token")

    # ── Step 3: Date of birth → birth year only ───────────────────────────
    if "date_of_birth" in df.columns:
        df["birth_year"] = df["date_of_birth"].apply(_extract_year)
        df = df.drop(columns=["date_of_birth"])
        audit.log("date_of_birth", "TRUNCATED to birth_year (year only)", len(df))
        if verbose:
            logger.info("TRUNCATED: date_of_birth → birth_year")

    # ── Step 4: Age → Safe Harbor age band ───────────────────────────────
    if "age" in df.columns:
        df["age_band"] = df["age"].apply(_bin_age)
        df = df.drop(columns=["age"])
        audit.log("age", "BINNED to 5-year bands; 90+ collapsed", len(df))
        if verbose:
            logger.info("BINNED: age → age_band")

    # ── Step 5: ZIP code → truncated 3-digit ─────────────────────────────
    if "zip_code" in df.columns:
        df["zip3"] = df["zip_code"].apply(_truncate_zip)
        df = df.drop(columns=["zip_code"])
        audit.log("zip_code", "TRUNCATED to 3-digit prefix (zip3)", len(df))
        if verbose:
            logger.info("TRUNCATED: zip_code → zip3")

    # ── Step 6: ED visit date → year only ────────────────────────────────
    if "ed_visit_date" in df.columns:
        df["ed_visit_year"] = df["ed_visit_date"].apply(_extract_year)
        df = df.drop(columns=["ed_visit_date"])
        audit.log("ed_visit_date", "TRUNCATED to year (ed_visit_year)", len(df))
        if verbose:
            logger.info("TRUNCATED: ed_visit_date → ed_visit_year")

    # ── Step 7: Validate — no residual PHI patterns ───────────────────────
    _validate_no_residual_phi(df, verbose)

    logger.info(f"De-identification complete. Output shape: {df.shape}")
    return df, audit


def _validate_no_residual_phi(df: pd.DataFrame, verbose: bool = True):
    """
    Scan string columns for patterns that resemble PHI.
    Raises a warning (not error) to flag potential residual identifiers.

    Checks:
    - SSN-like patterns (NNN-NN-NNNN)
    - Phone-like patterns
    - Full dates (YYYY-MM-DD)
    - MRN patterns
    """
    phi_patterns = {
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "Phone": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "Full Date": r"\b\d{4}-\d{2}-\d{2}\b",
        "MRN Pattern": r"\bMRN-\d+\b",
    }
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        col_str = df[col].dropna().astype(str).str.cat(sep=" ")
        for phi_type, pattern in phi_patterns.items():
            if re.search(pattern, col_str):
                logger.warning(
                    f"POTENTIAL RESIDUAL PHI detected in column '{col}': "
                    f"matches pattern for {phi_type}. Manual review required."
                )
    if verbose:
        logger.info("PHI validation scan complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)

    logger.info("Loading raw synthetic data...")
    raw = pd.read_csv("data/raw/synthetic_patients_raw.csv")
    logger.info(f"Raw data shape: {raw.shape}")

    deident, audit = run_safe_harbor_deidentification(raw, verbose=True)

    deident.to_csv("data/processed/patients_deidentified.csv", index=False)
    audit.report().to_csv("data/processed/deidentification_audit_log.csv", index=False)

    logger.info("Saved → data/processed/patients_deidentified.csv")
    logger.info("Saved → data/processed/deidentification_audit_log.csv")
    logger.info(f"De-identified shape: {deident.shape}")
    logger.info(f"\nAudit log:\n{audit.report().to_string(index=False)}")
