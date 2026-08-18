"""
streamlit_app.py
----------------
Self-initializing Streamlit dashboard for the LACDMH FUM Follow-Up Failure
Risk Stratification Model.

This file is the Streamlit Cloud entry point. On first load it automatically:
  1. Generates a synthetic behavioral health patient cohort (Synthea-style)
  2. Applies HIPAA Safe Harbor de-identification (45 CFR §164.514(b)(2))
  3. Engineers clinical + SDoH + geographic features
  4. Trains Logistic Regression + XGBoost with SHAP explainability
  5. Runs a demographic equity audit (LACDMH ARDI framework)

The full pipeline completes in ~5 seconds and is cached for the session.
No data files need to be pre-committed to the repository.

Deploy to Streamlit Community Cloud:
  Main file: streamlit_app.py
  Python version: 3.11

Portfolio project for LACDMH Data Scientist Supervisor (Exam b1765A).
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LACDMH FUM Risk Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/sichenz/dmh-fum-risk-model",
        "Report a bug": "https://github.com/sichenz/dmh-fum-risk-model/issues",
        "About": "Portfolio project for LACDMH Data Scientist Supervisor (Exam b1765A). Synthetic data only.",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: white;
    }
    .main-header h1 { font-size: 1.5rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 0.82rem; opacity: 0.8; margin: 0.3rem 0 0; }

    .hipaa-badge {
        background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px;
        padding: 0.35rem 0.75rem; font-size: 0.75rem; color: #15803d;
        display: inline-block; margin-bottom: 1rem;
    }
    .section-header {
        font-size: 0.95rem; font-weight: 600; color: #1e293b;
        border-bottom: 2px solid #2563eb; padding-bottom: 0.35rem; margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 8px; padding: 0.7rem;
    }
    .stAlert { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline initialization — runs once per server session, cached in memory
# ──────────────────────────────────────────────────────────────────────────────

def _make_dirs():
    for d in ["data/raw", "data/processed", "data/interim",
              "models", "reports/figures"]:
        os.makedirs(d, exist_ok=True)


@st.cache_resource(show_spinner=False)
def run_pipeline():
    """
    Execute the full 5-stage pipeline and return all outputs.
    Decorated with @st.cache_resource so it runs exactly once per
    Streamlit server session regardless of how many users connect.

    Returns a dict with DataFrames, model results, and fairness audit outputs.
    """
    import time
    from src.data_generation import generate_patient_cohort
    from src.deidentification import run_safe_harbor_deidentification
    from src.features import prepare_model_data
    from src.model import run_full_pipeline
    from src.fairness import run_fairness_audit

    _make_dirs()
    t0 = time.time()

    # Stage 1: Synthetic cohort
    raw_df = generate_patient_cohort(n_patients=5000)
    raw_df.to_csv("data/raw/synthetic_patients_raw.csv", index=False)

    # Stage 2: HIPAA Safe Harbor de-identification
    deident_df, audit = run_safe_harbor_deidentification(raw_df, verbose=False)
    deident_df.to_csv("data/processed/patients_deidentified.csv", index=False)
    audit.report().to_csv("data/processed/deidentification_audit_log.csv", index=False)

    # Stage 3: Feature engineering + train/test split
    data = prepare_model_data(deident_df)

    # Stage 4: Model training + SHAP + plots
    results = run_full_pipeline(data)

    # Stage 5: Fairness / equity audit
    y_prob = results["xgb_model"].predict_proba(data["X_test"])[:, 1]
    group_metrics, summaries = run_fairness_audit(
        data["y_test"], y_prob, data["sensitive_test"]
    )

    elapsed = time.time() - t0

    return {
        "risk_df":       results["risk_df"],
        "metrics_df":    results["metrics"],
        "deident_df":    deident_df,
        "audit_df":      audit.report(),
        "group_metrics": group_metrics,
        "summaries":     summaries,
        "data":          data,
        "xgb_model":     results["xgb_model"],
        "elapsed":       elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Header — shown immediately while pipeline initializes
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 LACDMH — FUM Follow-Up Risk Stratification Dashboard</h1>
    <p>Mental Health ED Post-Crisis Follow-Up Failure Prediction &nbsp;|&nbsp;
       Portfolio Prototype &nbsp;|&nbsp; Synthetic Data Only</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hipaa-badge">🔒 De-identified — HIPAA Safe Harbor (45 CFR §164.514(b)) '
    '| All data is synthetic | No real patients</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-initialize with spinner
# ──────────────────────────────────────────────────────────────────────────────

pipeline_ready = os.path.exists("data/processed/patient_risk_scores.csv")

if not pipeline_ready:
    with st.spinner(
        "🚀 First load — running the full pipeline: generating synthetic cohort → "
        "HIPAA de-identification → feature engineering → XGBoost training → "
        "SHAP explainability → ARDI equity audit. Takes ~5 seconds…"
    ):
        pipeline = run_pipeline()
else:
    pipeline = run_pipeline()  # returns instantly from cache

# ──────────────────────────────────────────────────────────────────────────────
# Unpack pipeline outputs
# ──────────────────────────────────────────────────────────────────────────────

risk_df      = pipeline["risk_df"]
metrics_df   = pipeline["metrics_df"]
deident_df   = pipeline["deident_df"]
audit_df     = pipeline["audit_df"]
group_metrics = pipeline["group_metrics"]
summaries    = pipeline["summaries"]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 LACDMH FUM Model")
    st.caption("Portfolio — Exam b1765A")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Population Overview",
         "🎯 Outreach List",
         "⚖️ Equity Audit",
         "📈 Model Performance",
         "🔒 HIPAA Governance",
         "📋 About This Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    threshold = st.slider(
        "Risk Score Threshold",
        min_value=0.30, max_value=0.80,
        value=0.50, step=0.05,
        help="Patients above this threshold are flagged for proactive outreach",
    )
    st.caption(f"Score ≥ {threshold:.2f} → flagged for outreach")

    st.markdown("---")
    st.markdown("**HEDIS FUM Measure**")
    st.caption(
        "7-day follow-up after ED visit for mental illness. "
        "Active PIP in LACDMH QAPI Work Plan 2025–2026."
    )

    elapsed = pipeline.get("elapsed", 0)
    if elapsed:
        st.markdown("---")
        st.caption(f"⚡ Pipeline ran in {elapsed:.1f}s")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Population Overview
# ──────────────────────────────────────────────────────────────────────────────

if page == "📊 Population Overview":
    st.markdown('<div class="section-header">Population Risk Distribution</div>',
                unsafe_allow_html=True)

    high_risk      = (risk_df["risk_score"] >= threshold).sum()
    total          = len(risk_df)
    actual_fail    = int(risk_df["true_label"].sum()) if "true_label" in risk_df.columns else 0
    mean_score     = risk_df["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Set Patients",    f"{total:,}")
    c2.metric("Flagged High-Risk",    f"{high_risk:,}",
              delta=f"{high_risk/total:.1%} of cohort", delta_color="off")
    c3.metric("Actual Follow-Up Failures", f"{actual_fail:,}",
              delta=f"{actual_fail/total:.1%} failure rate", delta_color="off")
    c4.metric("Mean Risk Score", f"{mean_score:.3f}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("**Risk Score Distribution**")
        fig = px.histogram(
            risk_df, x="risk_score", nbins=40,
            color_discrete_sequence=["#2563EB"],
            labels={"risk_score": "Predicted Failure Probability"},
        )
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="#DC2626",
            annotation_text=f"Threshold ({threshold:.2f})",
            annotation_position="top right",
        )
        fig.update_layout(
            height=290, margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f1f5f9"),
            yaxis=dict(gridcolor="#f1f5f9"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Risk Tier Breakdown**")
        tier_counts = pd.cut(
            risk_df["risk_score"],
            bins=[0, 0.33, 0.60, 1.0],
            labels=["Low (0–33%)", "Moderate (33–60%)", "High (60%+)"],
        ).value_counts()
        fig_pie = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            color_discrete_sequence=["#059669", "#D97706", "#DC2626"],
            hole=0.45,
        )
        fig_pie.update_layout(
            height=290, margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # SHAP plots
    st.markdown("---")
    st.markdown('<div class="section-header">SHAP Feature Importance</div>',
                unsafe_allow_html=True)
    shap_bar  = "reports/figures/shap_bar.png"
    shap_bees = "reports/figures/shap_summary.png"
    if os.path.exists(shap_bar) and os.path.exists(shap_bees):
        sc1, sc2 = st.columns(2)
        with sc1:
            st.image(shap_bar,  caption="Mean |SHAP| — Overall Feature Importance")
        with sc2:
            st.image(shap_bees, caption="SHAP Beeswarm — Feature Direction & Magnitude")
    else:
        st.info("SHAP plots not found — reload the page to re-run the pipeline.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Outreach List
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🎯 Outreach List":
    st.markdown('<div class="section-header">High-Risk Patient Outreach List</div>',
                unsafe_allow_html=True)
    st.caption(
        "Patients above the risk threshold are candidates for proactive outreach. "
        "Tokens are pseudonymized SHA-256 hashes — match to IBHIS by token for care coordination."
    )

    high_risk_df = risk_df[risk_df["risk_score"] >= threshold].copy()
    high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)

    st.info(
        f"**{len(high_risk_df):,} patients** exceed the {threshold:.0%} threshold "
        f"and are recommended for outreach."
    )

    def tier_label(score):
        if score >= 0.70: return "🔴 Critical"
        if score >= 0.55: return "🟠 High"
        return "🟡 Moderate-High"

    display_cols = [
        "risk_score", "appointment_scheduled_at_discharge",
        "medication_prescribed_at_discharge", "prior_ed_visits_12mo",
        "housing_instability", "appointment_wait_days_clean",
        "diagnosis_severity", "discharge_quality_score",
    ]
    avail = [c for c in display_cols if c in high_risk_df.columns]
    display = high_risk_df[avail].copy()
    display.insert(0, "Risk Tier", high_risk_df["risk_score"].apply(tier_label))
    display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.3f}")
    display = display.rename(columns={
        "risk_score": "Risk Score",
        "appointment_scheduled_at_discharge": "Appt Scheduled",
        "medication_prescribed_at_discharge": "Rx Prescribed",
        "prior_ed_visits_12mo": "Prior ED (12mo)",
        "housing_instability": "Housing Instability",
        "appointment_wait_days_clean": "Appt Wait Days",
        "diagnosis_severity": "Dx Severity",
        "discharge_quality_score": "Discharge Quality",
    })

    page_size = 50
    n_pages = max(1, (len(display) - 1) // page_size + 1)
    pg = st.number_input("Page", 1, n_pages, 1)
    st.dataframe(
        display.iloc[(pg-1)*page_size : pg*page_size],
        use_container_width=True, height=380,
    )
    st.download_button(
        "⬇️ Download High-Risk List (CSV)",
        data=display.to_csv(index=False),
        file_name="high_risk_outreach_list.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("**Risk Score by Key Clinical Drivers**")
    dc1, dc2 = st.columns(2)
    with dc1:
        if "appointment_scheduled_at_discharge" in risk_df.columns:
            ag = (risk_df.groupby("appointment_scheduled_at_discharge")["risk_score"]
                         .mean().reset_index())
            ag["appointment_scheduled_at_discharge"] = ag["appointment_scheduled_at_discharge"].map(
                {0: "No Appt Scheduled", 1: "Appt Scheduled"}
            )
            fig = px.bar(ag, x="appointment_scheduled_at_discharge", y="risk_score",
                         color_discrete_sequence=["#2563EB"],
                         labels={"risk_score": "Mean Risk Score",
                                 "appointment_scheduled_at_discharge": ""},
                         title="Mean Risk Score by Appointment Scheduling")
            fig.update_layout(height=260, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    with dc2:
        if "housing_instability" in risk_df.columns:
            hg = (risk_df.groupby("housing_instability")["risk_score"]
                         .mean().reset_index())
            hg["housing_instability"] = hg["housing_instability"].map(
                {0: "Stable Housing", 1: "Unstable Housing"}
            )
            fig = px.bar(hg, x="housing_instability", y="risk_score",
                         color_discrete_sequence=["#7C3AED"],
                         labels={"risk_score": "Mean Risk Score",
                                 "housing_instability": ""},
                         title="Mean Risk Score by Housing Status")
            fig.update_layout(height=260, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Equity Audit
# ──────────────────────────────────────────────────────────────────────────────

elif page == "⚖️ Equity Audit":
    st.markdown('<div class="section-header">Model Equity Audit — LACDMH ARDI Framework</div>',
                unsafe_allow_html=True)
    st.caption(
        "Evaluates whether model performance is equitable across demographic subgroups, "
        "aligned with LACDMH's Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan."
    )

    attr_labels = {
        "race_ethnicity":   "Race / Ethnicity",
        "preferred_language": "Preferred Language",
        "insurance_type":   "Insurance Type",
        "spa_region":       "Service Planning Area (SPA)",
    }
    selected = st.selectbox(
        "Demographic attribute to audit:",
        list(attr_labels.keys()),
        format_func=lambda x: attr_labels[x],
    )

    gm_df = group_metrics.get(selected)

    if gm_df is not None and not gm_df.empty:
        summary = summaries.get(selected, {})

        c1, c2, c3 = st.columns(3)
        tpr_gap = (gm_df["tpr_sensitivity"].max() - gm_df["tpr_sensitivity"].min()
                   if gm_df["tpr_sensitivity"].notna().any() else 0)
        fpr_gap = (gm_df["fpr_fallout"].max() - gm_df["fpr_fallout"].min()
                   if gm_df["fpr_fallout"].notna().any() else 0)
        dp_gap  = (gm_df["predicted_failure_rate"].max() - gm_df["predicted_failure_rate"].min()
                   if gm_df["predicted_failure_rate"].notna().any() else 0)

        c1.metric("TPR Gap (max−min)", f"{tpr_gap:.3f}",
                  delta="✓ Within 0.10" if tpr_gap < 0.10 else "⚠ Exceeds 0.10",
                  delta_color="normal" if tpr_gap < 0.10 else "inverse")
        c2.metric("FPR Gap (max−min)", f"{fpr_gap:.3f}",
                  delta="✓ Within 0.10" if fpr_gap < 0.10 else "⚠ Exceeds 0.10",
                  delta_color="normal" if fpr_gap < 0.10 else "inverse")
        c3.metric("Demographic Parity Gap", f"{dp_gap:.3f}",
                  delta="✓ Within 0.10" if dp_gap < 0.10 else "⚠ Exceeds 0.10",
                  delta_color="normal" if dp_gap < 0.10 else "inverse")

        st.markdown("---")

        fig_path = f"reports/figures/fairness_{selected}.png"
        if os.path.exists(fig_path):
            st.image(fig_path,
                     caption=f"Disaggregated performance by {attr_labels[selected]}")

        st.markdown("**Detailed Metrics by Group**")
        show_cols = ["group", "n", "actual_failure_rate", "predicted_failure_rate",
                     "tpr_sensitivity", "fpr_fallout", "precision_ppv", "roc_auc"]
        show_cols = [c for c in show_cols if c in gm_df.columns]
        st.dataframe(gm_df[show_cols], use_container_width=True)

        st.markdown("---")
        summary_img = "reports/figures/fairness_summary_table.png"
        if os.path.exists(summary_img):
            st.markdown("**Fairness Audit Summary — All Attributes**")
            st.image(summary_img)

        st.markdown("---")
        st.markdown("""
> **Interpretation note:** The fairness audit reveals meaningful performance gaps across
> demographic subgroups — particularly for SPA 1 (Antelope Valley, rural) and for
> Black/African American patients. These disparities **do not pass** standard fairness
> thresholds (equalized odds difference < 0.10), which is the expected finding.
>
> In a production context, these results would trigger: (1) subgroup-specific model
> recalibration, (2) targeted feature engineering to reduce disparity (e.g., adding
> language-concordant care access features), and (3) a formal review with LACDMH's
> ARDI leadership before any deployment decision.
""")
    else:
        st.info("Fairness data not available for this attribute.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            with st.expander(f"**{row['model']}**", expanded=True):
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("ROC-AUC",    f"{row['roc_auc']:.4f}")
                mc2.metric("PR-AUC",     f"{row['pr_auc']:.4f}",
                           help="Precision-Recall AUC — the right metric for imbalanced data")
                mc3.metric("Brier Score", f"{row['brier_score']:.4f}",
                           help="Lower = better. 0 = perfect, 0.25 = random classifier")
                mc4.metric("Test Set Size", f"{int(row['n_test']):,}")

    pc1, pc2 = st.columns(2)
    roc_path = "reports/figures/roc_pr_curves.png"
    cal_path = "reports/figures/calibration_curve.png"
    with pc1:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC and Precision-Recall Curves")
    with pc2:
        if os.path.exists(cal_path):
            st.image(cal_path, caption="Calibration Curve — Predicted vs. Actual Rates")

    st.markdown("---")
    st.markdown("""
**Why these metrics?**

| Metric | Why it matters for FUM |
|---|---|
| **PR-AUC** | Class imbalance (~66% failure rate) makes ROC-AUC optimistic. PR-AUC is a more honest measure. |
| **Calibration** | For risk stratification, we need predicted *probabilities* to be trustworthy — a score of 0.7 should mean ~70% chance of failure. |
| **Brier Score** | A proper scoring rule that penalises both overconfident wrong predictions and underconfident right ones. |
| **ROC-AUC** | Threshold-independent discrimination — how well the model separates completers from non-completers. |
""")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: HIPAA Governance
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🔒 HIPAA Governance":
    st.markdown('<div class="section-header">HIPAA Data Governance</div>',
                unsafe_allow_html=True)

    if audit_df is not None and not audit_df.empty:
        st.success("✅ De-identification audit log generated this session")
        st.markdown("**De-identification Audit Trail** — 45 CFR §164.514(b)(2) Safe Harbor")
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.warning("Audit log not found.")

    st.markdown("---")
    st.markdown("""
**Data Classification Schema**

| Layer | Classification | Access |
|---|---|---|
| `data/raw/` | **Restricted** — synthetic PHI-equivalent fields | Pipeline runtime only |
| `data/processed/` | **Internal** — Safe Harbor de-identified | Analysts with IRB + DUA |
| `reports/` | **Internal** — aggregated statistics only | Data science team |
| `models/` | **Confidential** — may encode population statistics | Data scientists |

---

**PHI Identifier Treatment — 45 CFR §164.514(b)(2)**

| # | PHI Identifier | Treatment | Status |
|---|---|---|---|
| 1 | Names | **REMOVED** | ✅ |
| 2 | Street address | **REMOVED** | ✅ |
| 2 | ZIP code | **Truncated** → 3-digit prefix | ✅ |
| 3 | Date of birth | **Truncated** → birth year only | ✅ |
| 3 | ED visit date | **Truncated** → year only | ✅ |
| 3 | Age ≥ 90 | **Collapsed** → "90+" | ✅ |
| 4 | Phone numbers | **REMOVED** | ✅ |
| 7 | SSN (last 4) | **REMOVED** | ✅ |
| 8 | Medical Record Number | **Pseudonymized** → SHA-256 token | ✅ |
| 5–18 | All other identifiers | Not present in dataset | ✅ |

---

**Model Governance Controls**

- ✅ Fairness audit required before any deployment decision
- ✅ Model fidelity monitoring plan documented in `HIPAA_GOVERNANCE.md`
- ✅ SHAP explainability available for clinical decision support audit
- ✅ All data is **synthetic** — no real patient data
- ✅ Access control design documented
- ⚠️ **Production deployment requires:** IRB determination, DUA, Privacy Officer sign-off, LACDMH IT security review

> **HIPAA Note:** If deployed with real LACDMH data, this analytics pipeline would constitute a
> Healthcare Operations (QAPI) use of PHI, permissible under 45 CFR §164.501 without patient
> authorization — provided the required safeguards documented in `HIPAA_GOVERNANCE.md` are in place.
""")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: About This Project
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📋 About This Project":
    st.markdown('<div class="section-header">About This Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
## FUM Follow-Up Failure Risk Stratification

**Predicting Post-Crisis Follow-Up Failure in a Mental Health Population: A HIPAA-Compliant ML Pipeline**

This is a portfolio project built to demonstrate healthcare data science experience for the
**LA County Department of Mental Health Data Scientist Supervisor** position (Exam b1765A).

---

### Clinical Motivation

The **HEDIS FUM (Follow-Up After Emergency Department Visit for Mental Illness)** measure
tracks whether patients receive outpatient follow-up within 7 or 30 days after a psychiatric
ED visit. LACDMH runs this as a named **Performance Improvement Project (PIP)** in their
2025–2026 QAPI Work Plan.

Low FUM rates = real people falling through the cracks. This model identifies who is at
highest risk of missing that follow-up — enabling proactive outreach.

---

### Pipeline Architecture

| Stage | Module | What it does |
|---|---|---|
| 1 | `data_generation.py` | Generates 5,000 synthetic LA County Medi-Cal behavioral health patients |
| 2 | `deidentification.py` | HIPAA Safe Harbor de-identification (45 CFR §164.514(b)(2)) |
| 3 | `features.py` | 25+ clinical, SDoH, and geographic features |
| 4 | `model.py` | XGBoost + SHAP explainability + calibration |
| 5 | `fairness.py` | Demographic equity audit (LACDMH ARDI framework) |

---

### Alignment with Active LACDMH Initiatives

| LACDMH Initiative | How This Project Connects |
|---|---|
| FUM Performance Improvement Project (QAPI 2025–2026) | Directly models this HEDIS measure |
| Homelessness Prevention Unit predictive analytics (CA Policy Lab) | Same multi-factor risk stratification approach |
| IBHIS / CalAIM data integration | Features designed for EHR-structured data |
| ARDI Strategic Plan | Full fairness audit across race, language, SPA |
| GIS / spatial equity analysis | SPA region and provider distance features |
| Provider Language Capacity Report | LEP feature, language-disaggregated fairness |

---

### Key Technical Decisions

- **XGBoost over deep learning:** Tabular clinical data + need for exact SHAP explanations
- **HIPAA Safe Harbor over Expert Determination:** Standard method for LACDMH data partners (InfoHub, CalAIM, IBHIS)
- **PR-AUC as primary metric:** Accounts for class imbalance (~66% failure rate)
- **Fairness audit independent of explainability:** SHAP can mask bias; fairness is evaluated on its own terms
- **Shallow trees (max_depth=4):** More interpretable SHAP values for clinical staff

---

### Source Code

All source code is available on GitHub. The full pipeline runs automatically on page load.

**Stack:** Python · scikit-learn · XGBoost · SHAP · fairlearn · Streamlit · Plotly · Faker
""")

    st.info(
        "**Note:** This prototype uses entirely synthetic data. "
        "No real patient data was used at any stage. "
        "All demographic distributions are modeled on publicly available LACDMH reports."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**Prototype — Portfolio / Research Use Only** | "
    "Synthetic data generated with Synthea-equivalent methodology | "
    "Not for clinical deployment without IRB approval, Privacy Officer sign-off, "
    "and LACDMH IT Security review | "
    "LACDMH Data Scientist Supervisor — Exam b1765A"
)