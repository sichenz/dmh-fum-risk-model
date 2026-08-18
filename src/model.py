"""
model.py
--------
XGBoost binary classifier for predicting 7-day post-ED follow-up failure
in mental health patients, with SHAP-based explainability.

The model predicts the HEDIS FUM (Follow-Up After Emergency Department
Visit for Mental Illness) outcome — whether a patient will fail to
complete a 7-day follow-up appointment after an ED visit for a
psychiatric crisis.

Clinical interpretation:
  - Output score = probability of follow-up FAILURE
  - High-risk patients (score ≥ threshold) are candidates for proactive
    outreach by care coordinators
  - SHAP values provide per-patient explanations for clinical decision-making

Evaluation metrics:
  - AUC-ROC: overall discrimination ability
  - Precision-Recall AUC: accounts for class imbalance
  - Calibration: ensures scores are interpretable as probabilities
  - Brier score: proper scoring rule for probabilistic classifiers
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
import xgboost as xgb
import shap

logging.basicConfig(level=logging.INFO, format="[model] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "reports/figures"
MODEL_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline model
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train) -> LogisticRegression:
    """
    Logistic regression baseline — interpretable and deployable as a
    simple clinical scoring rule.
    """
    logger.info("Training logistic regression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train, y_train)
    logger.info("Baseline trained.")
    return lr


# ──────────────────────────────────────────────────────────────────────────────
# Primary XGBoost model
# ──────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    Train the primary XGBoost classifier with class-imbalance handling.

    Hyperparameters are selected to balance predictive performance with
    interpretability (shallow trees → more auditable SHAP values).
    """
    logger.info("Training XGBoost classifier...")

    # Compute class weight for imbalanced data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info(f"Class balance → negatives: {neg}, positives: {pos}, scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,                    # shallow for interpretability
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",            # PR AUC — better for imbalanced data
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    eval_set = [(X_train, y_train)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    logger.info(f"XGBoost trained. Best iteration: {model.best_iteration}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "XGBoost") -> dict:
    """
    Compute standard classification metrics with clinical context.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model": model_name,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "n_test": len(y_test),
        "positive_rate_actual": float(y_test.mean()),
        "positive_rate_predicted": float(y_pred.mean()),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:     {metrics['pr_auc']:.4f}")
    logger.info(f"Brier:      {metrics['brier_score']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Completed FU', 'Missed FU'])}")

    return metrics, y_prob


def plot_roc_pr_curves(results: list, y_test, save_path: str = None):
    """
    Plot ROC and Precision-Recall curves for all models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("FUM Follow-Up Failure Prediction — Model Comparison", fontsize=14, fontweight="bold")

    colors = ["#2563EB", "#DC2626", "#059669"]

    for i, (name, y_prob) in enumerate(results):
        color = colors[i % len(colors)]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

        # PR curve
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        prauc = average_precision_score(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AP = {prauc:.3f})")

    # ROC formatting
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.grid(alpha=0.3)

    # PR formatting
    baseline = y_test.mean()
    ax2.axhline(y=baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (prevalence = {baseline:.2f})")
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved ROC/PR plot → {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# SHAP explainability
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> shap.Explainer:
    """
    Compute SHAP values using TreeExplainer (exact, fast for tree models).

    SHAP (SHapley Additive exPlanations) provides:
    - Global feature importance: which features drive predictions overall
    - Local explanations: why any individual patient received a high/low score

    Clinical value: Care coordinators can see *why* a patient is high-risk
    (e.g., "no appointment scheduled + 5+ prior ED visits") rather than
    just a score — enabling targeted intervention.
    """
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    logger.info("SHAP values computed.")
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test: pd.DataFrame, save_path: str = None):
    """Global SHAP summary — beeswarm plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Feature Impact on Follow-Up Failure Prediction", fontsize=13, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved SHAP summary → {save_path}")
    plt.close()


def plot_shap_bar(shap_values, X_test: pd.DataFrame, save_path: str = None):
    """Global SHAP bar chart — mean absolute impact."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP Value| — Feature Importance for FUM Failure", fontsize=13, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved SHAP bar chart → {save_path}")
    plt.close()


def plot_calibration(model, X_test, y_test, model_name: str, save_path: str = None):
    """Calibration curve — how well predicted probabilities match observed rates."""
    y_prob = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#2563EB", lw=2, label=model_name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives (actual)", fontsize=11)
    ax.set_title(f"Calibration Curve — {model_name}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved calibration plot → {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Full training pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(data: dict) -> dict:
    """
    End-to-end model training, evaluation, and explainability pipeline.
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    all_metrics = []

    # ── Baseline ─────────────────────────────────────────────────────────
    baseline = train_baseline(X_train, y_train)
    bm, b_prob = evaluate_model(baseline, X_test, y_test, "Logistic Regression")
    all_metrics.append(bm)
    joblib.dump(baseline, f"{MODEL_DIR}/baseline_logreg.pkl")

    # ── XGBoost ──────────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train, y_train)
    xm, x_prob = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    all_metrics.append(xm)
    xgb_model.save_model(f"{MODEL_DIR}/xgboost_fum.json")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_roc_pr_curves(
        [("Logistic Regression", b_prob), ("XGBoost", x_prob)],
        y_test,
        save_path=f"{RESULTS_DIR}/roc_pr_curves.png",
    )

    plot_calibration(
        xgb_model, X_test, y_test, "XGBoost",
        save_path=f"{RESULTS_DIR}/calibration_curve.png",
    )

    # ── SHAP ──────────────────────────────────────────────────────────────
    explainer, shap_values = compute_shap_values(xgb_model, X_test)
    plot_shap_summary(shap_values, X_test, save_path=f"{RESULTS_DIR}/shap_summary.png")
    plot_shap_bar(shap_values, X_test, save_path=f"{RESULTS_DIR}/shap_bar.png")

    # ── Risk score output ─────────────────────────────────────────────────
    risk_df = X_test.copy()
    risk_df["risk_score"] = x_prob
    risk_df["risk_tier"] = pd.cut(
        x_prob,
        bins=[0, 0.33, 0.60, 1.0],
        labels=["Low", "Moderate", "High"],
    )
    risk_df["true_label"] = y_test.values
    risk_df.to_csv("data/processed/patient_risk_scores.csv", index=False)
    logger.info("Saved patient risk scores → data/processed/patient_risk_scores.csv")

    # ── Metrics summary ────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("reports/model_metrics.csv", index=False)
    logger.info(f"\nMetrics summary:\n{metrics_df.to_string(index=False)}")

    return {
        "baseline": baseline,
        "xgb_model": xgb_model,
        "shap_values": shap_values,
        "risk_df": risk_df,
        "metrics": metrics_df,
    }


if __name__ == "__main__":
    from src.features import prepare_model_data
    import os
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv("data/processed/patients_deidentified.csv")
    data = prepare_model_data(df)
    results = run_full_pipeline(data)