"""
run_pipeline.py
---------------
Runs all stages end-to-end:
  1. Synthetic data generation
  2. HIPAA Safe Harbor de-identification
  3. Feature engineering
  4. Model training & evaluation
  5. Fairness / equity audit
  6. Risk score output

Usage:
    python run_pipeline.py
    python run_pipeline.py --n-patients 10000
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generation import generate_patient_cohort
from src.deidentification import run_safe_harbor_deidentification
from src.features import prepare_model_data
from src.model import run_full_pipeline
from src.fairness import run_fairness_audit


def make_dirs():
    for d in ["data/raw", "data/processed", "data/interim", "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)


def main(n_patients: int = 5000, skip_generation: bool = False):
    make_dirs()
    start = time.time()

    # ── Stage 1: Data generation ──────────────────────────────────────────
    raw_path = "data/raw/synthetic_patients_raw.csv"
    if skip_generation and os.path.exists(raw_path):
        logger.info("Skipping data generation (--skip-generation flag).")
        raw_df = pd.read_csv(raw_path)
    else:
        logger.info(f"Stage 1: Generating {n_patients:,} synthetic patients...")
        raw_df = generate_patient_cohort(n_patients=n_patients)
        raw_df.to_csv(raw_path, index=False)
        logger.info(f"Raw data saved → {raw_path}")

    # ── Stage 2: HIPAA de-identification ──────────────────────────────────
    deident_path = "data/processed/patients_deidentified.csv"
    logger.info("Stage 2: Running HIPAA Safe Harbor de-identification...")
    deident_df, audit = run_safe_harbor_deidentification(raw_df, verbose=True)
    deident_df.to_csv(deident_path, index=False)
    audit.report().to_csv("data/processed/deidentification_audit_log.csv", index=False)
    logger.info(f"De-identified data saved → {deident_path}")

    # ── Stage 3: Feature engineering ──────────────────────────────────────
    logger.info("Stage 3: Engineering features...")
    data = prepare_model_data(deident_df)

    # ── Stage 4: Model training ────────────────────────────────────────────
    logger.info("Stage 4: Training models...")
    results = run_full_pipeline(data)

    # ── Stage 5: Fairness audit ────────────────────────────────────────────
    logger.info("Stage 5: Running fairness / equity audit...")
    y_prob = results["xgb_model"].predict_proba(data["X_test"])[:, 1]
    group_metrics, summaries = run_fairness_audit(
        data["y_test"], y_prob, data["sensitive_test"]
    )

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"XGBoost ROC-AUC: {results['metrics'][results['metrics']['model']=='XGBoost']['roc_auc'].values[0]:.4f}")
    logger.info(f"XGBoost PR-AUC:  {results['metrics'][results['metrics']['model']=='XGBoost']['pr_auc'].values[0]:.4f}")
    logger.info(f"\nArtifacts:")
    logger.info(f"  data/processed/patients_deidentified.csv")
    logger.info(f"  data/processed/patient_risk_scores.csv")
    logger.info(f"  models/xgboost_fum.json")
    logger.info(f"  reports/figures/*.png")
    logger.info(f"  reports/fairness_*.csv")
    logger.info(f"\nTo launch the dashboard:")
    logger.info(f"  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUM Follow-Up Failure Risk Pipeline")
    parser.add_argument("--n-patients", type=int, default=5000,
                        help="Number of synthetic patients to generate (default: 5000)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip data generation if raw file already exists")
    args = parser.parse_args()
    main(n_patients=args.n_patients, skip_generation=args.skip_generation)
