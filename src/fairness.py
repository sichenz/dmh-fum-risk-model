"""
fairness.py
-----------
Demographic equity audit for the FUM Follow-Up Failure prediction model.

LA County is explicitly committed to Anti-Racism, Diversity, and Inclusion
(ARDI) in its data science work. This audit tests whether the predictive
model performs equitably across demographic subgroups defined by:
  - Race/ethnicity
  - Preferred language
  - Insurance type (as a proxy for socioeconomic status)
  - Geographic service area (SPA)

Metrics computed per group:
  - Prevalence (actual positive rate)
  - Predicted positive rate
  - ROC-AUC
  - Equalized Odds: TPR and FPR parity across groups
  - Predictive Parity: precision parity across groups
  - Demographic Parity Difference: max gap in positive prediction rates

Fairlearn reference: https://fairlearn.org/
ARDI context: LA County Strategic Plan 2020-2030, Section 4 (Equity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix
import logging
import os

logging.basicConfig(level=logging.INFO, format="[fairness] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "reports/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

SENSITIVE_COLS = ["race_ethnicity", "preferred_language", "insurance_type", "spa_region"]


# ──────────────────────────────────────────────────────────────────────────────
# Per-group metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_group_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    sensitive: pd.Series,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """
    Compute classification metrics disaggregated by a sensitive attribute.

    Parameters
    ----------
    y_true : pd.Series
        True labels (0 = completed follow-up, 1 = failed follow-up).
    y_prob : np.ndarray
        Predicted probabilities of failure.
    sensitive : pd.Series
        Categorical sensitive attribute (e.g., race_ethnicity).
    threshold : float
        Decision threshold for binary predictions.
    min_group_size : int
        Minimum group size to report (smaller groups are unstable).

    Returns
    -------
    pd.DataFrame with per-group metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    rows = []

    for group in sensitive.unique():
        mask = sensitive == group
        n = mask.sum()
        if n < min_group_size:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_prob[mask]

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # sensitivity / recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan  # fall-out
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        pred_pos_rate = yp.mean()
        actual_pos_rate = float(yt.mean())

        try:
            auc = roc_auc_score(yt, yprob) if len(yt.unique()) > 1 else np.nan
        except Exception:
            auc = np.nan

        rows.append({
            "group": group,
            "n": n,
            "actual_failure_rate": round(actual_pos_rate, 3),
            "predicted_failure_rate": round(pred_pos_rate, 3),
            "tpr_sensitivity": round(tpr, 3) if not np.isnan(tpr) else None,
            "fpr_fallout": round(fpr, 3) if not np.isnan(fpr) else None,
            "precision_ppv": round(ppv, 3) if not np.isnan(ppv) else None,
            "npv": round(npv, 3) if not np.isnan(npv) else None,
            "roc_auc": round(auc, 3) if not np.isnan(auc) else None,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "group", "n", "actual_failure_rate", "predicted_failure_rate",
            "tpr_sensitivity", "fpr_fallout", "precision_ppv", "npv", "roc_auc",
            "tp", "fp", "fn", "tn",
        ])
    return pd.DataFrame(rows).sort_values("actual_failure_rate", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# Fairness summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_fairness_summary(group_metrics: pd.DataFrame) -> dict:
    """
    Compute fairness gap statistics across groups.

    Metrics:
    - Demographic Parity Difference: max - min predicted positive rate
    - Equalized Odds Difference: max TPR gap + max FPR gap
    - AUC Disparity: max - min ROC-AUC
    """
    valid = group_metrics.dropna(subset=["tpr_sensitivity", "fpr_fallout", "roc_auc"])
    if valid.empty:
        return {}

    pred_rates = group_metrics["predicted_failure_rate"].dropna()
    tprs = valid["tpr_sensitivity"].dropna()
    fprs = valid["fpr_fallout"].dropna()
    aucs = valid["roc_auc"].dropna()

    summary = {
        "demographic_parity_difference": round(pred_rates.max() - pred_rates.min(), 3),
        "tpr_max_gap": round(tprs.max() - tprs.min(), 3),
        "fpr_max_gap": round(fprs.max() - fprs.min(), 3),
        "equalized_odds_difference": round((tprs.max() - tprs.min()) + (fprs.max() - fprs.min()), 3),
        "auc_disparity": round(aucs.max() - aucs.min(), 3),
        "n_groups": len(group_metrics),
    }

    # Fairlearn-style flags (thresholds from literature)
    summary["passes_demographic_parity"] = summary["demographic_parity_difference"] < 0.10
    summary["passes_equalized_odds"] = summary["equalized_odds_difference"] < 0.10
    summary["passes_auc_parity"] = summary["auc_disparity"] < 0.05

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_group_performance(
    group_metrics: pd.DataFrame,
    attribute_name: str,
    save_path: str = None,
):
    """
    Bar chart comparing TPR, FPR, and Precision across demographic groups.
    Flags groups with concerning gaps (> 10pp from mean).
    """
    df = group_metrics.dropna(subset=["tpr_sensitivity"]).copy()
    if df.empty or len(df) < 2:
        return

    metrics_to_plot = ["actual_failure_rate", "tpr_sensitivity", "fpr_fallout", "precision_ppv"]
    metric_labels = ["Actual Failure Rate", "Sensitivity (TPR)", "False Alarm Rate (FPR)", "Precision (PPV)"]
    colors = ["#1E40AF", "#059669", "#DC2626", "#7C3AED"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Model Equity Audit — Disaggregated by {attribute_name}\n"
        f"(LA County ARDI Framework — Anti-Racism, Diversity, and Inclusion)",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, metric, label, color in zip(axes.flat, metrics_to_plot, metric_labels, colors):
        valid = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
        bars = ax.barh(valid["group"], valid[metric], color=color, alpha=0.75, edgecolor="white")
        mean_val = valid[metric].mean()
        ax.axvline(mean_val, color="black", linestyle="--", lw=1.2, label=f"Mean: {mean_val:.3f}")
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(0, min(1.05, valid[metric].max() * 1.3))

        # Annotate values
        for bar, val in zip(bars, valid[metric]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved equity plot → {save_path}")
    plt.close()


def plot_fairness_radar(summaries: dict, save_path: str = None):
    """
    Summary table of fairness metrics across all sensitive attributes.
    """
    rows = []
    for attr, summary in summaries.items():
        rows.append({
            "Sensitive Attribute": attr,
            "Demographic Parity Gap": summary.get("demographic_parity_difference", np.nan),
            "Equalized Odds Gap": summary.get("equalized_odds_difference", np.nan),
            "AUC Disparity": summary.get("auc_disparity", np.nan),
            "Passes Dem. Parity (<0.10)": "✓" if summary.get("passes_demographic_parity") else "✗",
            "Passes Eq. Odds (<0.10)": "✓" if summary.get("passes_equalized_odds") else "✗",
            "Passes AUC Parity (<0.05)": "✓" if summary.get("passes_auc_parity") else "✗",
        })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Color pass/fail cells
    col_names = list(df.columns)
    pass_cols = [i for i, c in enumerate(col_names) if "Passes" in c]
    for (row, col), cell in table._cells.items():
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif col in pass_cols:
            text = cell.get_text().get_text()
            cell.set_facecolor("#D1FAE5" if text == "✓" else "#FEE2E2")

    plt.title("Fairness Audit Summary — FUM Model\n(LA County ARDI Equity Evaluation)",
               fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved fairness summary table → {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Full fairness audit pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_fairness_audit(
    y_test: pd.Series,
    y_prob: np.ndarray,
    sensitive_test: pd.DataFrame,
) -> dict:
    """
    Run the full fairness audit across all sensitive attributes.
    Saves disaggregated metrics and visualizations.

    Returns a dictionary of per-attribute group metrics DataFrames.
    """
    all_group_metrics = {}
    all_summaries = {}

    for attr in SENSITIVE_COLS:
        if attr not in sensitive_test.columns:
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"Fairness audit: {attr}")

        gm = compute_group_metrics(y_test, y_prob, sensitive_test[attr])
        summary = compute_fairness_summary(gm)

        logger.info(f"\n{gm.to_string(index=False)}")
        logger.info(f"\nFairness summary: {summary}")

        # Save CSVs
        gm.to_csv(f"reports/fairness_{attr}.csv", index=False)

        # Plot
        plot_group_performance(
            gm, attribute_name=attr,
            save_path=f"{RESULTS_DIR}/fairness_{attr}.png",
        )

        all_group_metrics[attr] = gm
        all_summaries[attr] = summary

    # Summary table plot
    plot_fairness_radar(all_summaries, save_path=f"{RESULTS_DIR}/fairness_summary_table.png")

    return all_group_metrics, all_summaries


if __name__ == "__main__":
    import joblib, xgboost as xgb
    os.makedirs("reports", exist_ok=True)

    from src.features import prepare_model_data
    df = pd.read_csv("data/processed/patients_deidentified.csv")
    data = prepare_model_data(df)

    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_fum.json")

    y_prob = model.predict_proba(data["X_test"])[:, 1]
    run_fairness_audit(data["y_test"], y_prob, data["sensitive_test"])