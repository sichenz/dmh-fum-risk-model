"""
app.py — Streamlit Dashboard
----------------------------
Interactive operational dashboard for the LA County FUM Follow-Up Failure
Risk Stratification Model.

Features:
  - Population-level risk distribution
  - High-risk patient outreach list (with SHAP-derived explanations)
  - Equity audit panel: model performance disaggregated by demographics
  - Model performance metrics
  - HIPAA data governance summary

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LA County FUM Risk Model — Outreach Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --accent: #0071E3;
        --accent-dark: #0a58c2;
        --accent-tint: #eaf3ff;
        --ink: #10182c;
        --ink-soft: #5b6478;
        --line: #e7eaf2;
        --canvas: #f6f7fb;
        --success: #0f9d58;
        --warn: #b45309;
        --danger: #d93025;
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--ink);
    }

    [data-testid="stAppViewContainer"] > .main { background: var(--canvas); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

    h1, h2, h3, h4 { font-weight: 700 !important; letter-spacing: -0.01em; color: var(--ink); }

    .hero {
        position: relative;
        overflow: hidden;
        background: radial-gradient(120% 160% at 0% 0%, #16305e 0%, #0b1f42 45%, #071634 100%);
        padding: 2.6rem 2.8rem;
        border-radius: 22px;
        margin-bottom: 1.1rem;
        color: white;
        box-shadow: 0 18px 40px -18px rgba(11, 31, 66, 0.55);
    }
    .hero::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(50% 90% at 100% 0%, rgba(0,113,227,0.35), transparent 60%);
        pointer-events: none;
    }
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: #bcd4ff; background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.16);
        padding: 0.28rem 0.7rem; border-radius: 999px; margin-bottom: 0.9rem;
    }
    .hero h1 {
        font-size: 2.05rem; font-weight: 800; margin: 0; color: white !important;
        letter-spacing: -0.02em; line-height: 1.15; max-width: 40rem;
    }
    .hero p {
        font-size: 0.98rem; color: rgba(255,255,255,0.72); margin: 0.65rem 0 0;
        max-width: 34rem; line-height: 1.5;
    }

    .badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.3rem 0 1.4rem; }
    .pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.76rem; font-weight: 600;
        padding: 0.36rem 0.8rem; border-radius: 999px;
        border: 1px solid var(--line); background: white; color: var(--ink-soft);
    }
    .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); }
    .pill.pill-accent { background: var(--accent-tint); border-color: #cfe3ff; color: var(--accent-dark); }

    .metric-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(16, 24, 44, 0.04);
    }
    .metric-card .value { font-size: 2rem; font-weight: 800; color: var(--ink); }
    .metric-card .label {
        font-size: 0.75rem; color: var(--ink-soft);
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem;
    }

    .risk-high { color: var(--danger); font-weight: 600; }
    .risk-moderate { color: var(--warn); font-weight: 600; }
    .risk-low { color: var(--success); font-weight: 600; }

    .section-header {
        font-size: 1.02rem; font-weight: 700; color: var(--ink);
        padding-bottom: 0.6rem; margin: 0.4rem 0 1.1rem;
        border-bottom: 1px solid var(--line);
    }

    div[data-testid="stMetric"] {
        background-color: white; border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 44, 0.04);
    }
    div[data-testid="stMetric"] label { color: var(--ink-soft) !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; }

    section[data-testid="stSidebar"] { background: white; border-right: 1px solid var(--line); }
    .stAlert { font-size: 0.85rem; border-radius: 12px; }
    div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: white; }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
    .stDownloadButton button, .stButton button {
        border-radius: 999px !important; font-weight: 600 !important;
        border: 1px solid var(--accent) !important;
    }
    .stDownloadButton button { background: var(--accent) !important; color: white !important; }
    hr { border-color: var(--line) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_risk_scores():
    path = "data/processed/patient_risk_scores.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_deidentified():
    path = "data/processed/patients_deidentified.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics():
    path = "reports/model_metrics.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_fairness(attr):
    path = f"reports/fairness_{attr}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def check_pipeline_run():
    """Check if pipeline has been run. Show setup instructions if not."""
    return os.path.exists("data/processed/patient_risk_scores.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">🧠 &nbsp;LA County · Behavioral Health Analytics</span>
    <h1>FUM Follow-Up Risk Stratification Dashboard</h1>
    <p>Mental health ED post-crisis follow-up failure prediction —
       operational outreach dashboard. Prototype, synthetic data only.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <span class="pill pill-accent"><span class="pill-dot"></span> HIPAA Safe Harbor de-identified</span>
    <span class="pill">🔒 45 CFR §164.514(b)</span>
    <span class="pill">🧪 Synthetic population only</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Gate: check if pipeline has been run
# ──────────────────────────────────────────────────────────────────────────────

if not check_pipeline_run():
    st.warning("⚠️ Pipeline output not found. Run the pipeline first:")
    st.code("python run_pipeline.py", language="bash")
    st.info("This will generate synthetic data, de-identify it, train the model, and produce all outputs for this dashboard.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://dmh.lacounty.gov/wp-content/uploads/2021/03/LA-County-DMH-Color-Logo.png",
             width=180, use_column_width=False)
    st.caption("Prototype — Research Use Only")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Population Overview", "🎯 Outreach List", "⚖️ Equity Audit", "📈 Model Performance", "🔒 HIPAA Governance"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**HEDIS FUM Measure**")
    st.caption("7-day follow-up after ED visit for mental illness. Low FUM rates = care gaps.")
    st.markdown("---")

    threshold = st.slider("Risk Score Threshold", 0.30, 0.80, 0.50, 0.05,
                           help="Patients above this threshold are flagged as high-risk for outreach")
    st.caption(f"Patients with risk score ≥ {threshold:.2f} flagged for outreach")


# ──────────────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────────────

risk_df = load_risk_scores()
deident_df = load_deidentified()
metrics_df = load_metrics()

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Population Overview
# ──────────────────────────────────────────────────────────────────────────────

if page == "📊 Population Overview":
    st.markdown('<div class="section-header">Population Risk Distribution</div>', unsafe_allow_html=True)

    if risk_df is not None:
        high_risk = (risk_df["risk_score"] >= threshold).sum()
        total = len(risk_df)
        actual_failures = risk_df["true_label"].sum() if "true_label" in risk_df.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients (Test Set)", f"{total:,}")
        with col2:
            st.metric("Flagged High-Risk", f"{high_risk:,}", delta=f"{high_risk/total:.1%} of cohort", delta_color="off")
        with col3:
            st.metric("Actual Follow-Up Failures", f"{int(actual_failures):,}", delta=f"{actual_failures/total:.1%} failure rate", delta_color="off")
        with col4:
            avg_score = risk_df["risk_score"].mean()
            st.metric("Mean Risk Score", f"{avg_score:.3f}")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**Risk Score Distribution**")
            fig = px.histogram(
                risk_df, x="risk_score", nbins=40,
                color_discrete_sequence=["#2563EB"],
                labels={"risk_score": "Predicted Failure Probability"},
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#DC2626",
                         annotation_text=f"Threshold ({threshold:.2f})", annotation_position="top right")
            fig.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f1f5f9"),
                yaxis=dict(gridcolor="#f1f5f9"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("**Risk Tier Breakdown**")
            tier_counts = pd.cut(
                risk_df["risk_score"],
                bins=[0, 0.33, 0.60, 1.0],
                labels=["Low (0-33%)", "Moderate (33-60%)", "High (60%+)"]
            ).value_counts()

            fig_pie = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                color_discrete_sequence=["#059669", "#D97706", "#DC2626"],
                hole=0.45,
            )
            fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # SHAP plots
        st.markdown("---")
        st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)
        shap_bar_path = "reports/figures/shap_bar.png"
        shap_summary_path = "reports/figures/shap_summary.png"
        if os.path.exists(shap_bar_path) and os.path.exists(shap_summary_path):
            col_shap1, col_shap2 = st.columns(2)
            with col_shap1:
                st.image(shap_bar_path, caption="Mean |SHAP| — Overall Feature Importance")
            with col_shap2:
                st.image(shap_summary_path, caption="SHAP Beeswarm — Feature Impact Direction")
        else:
            st.info("Run the pipeline to generate SHAP visualizations.")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Outreach List
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🎯 Outreach List":
    st.markdown('<div class="section-header">High-Risk Patient Outreach List</div>', unsafe_allow_html=True)
    st.caption("Patients above the risk threshold are candidates for proactive outreach by care coordinators. "
               "Patient tokens are pseudonymized — match to IBHIS by token for care coordination.")

    if risk_df is not None:
        high_risk_df = risk_df[risk_df["risk_score"] >= threshold].copy()
        high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)

        st.info(f"**{len(high_risk_df):,} patients** exceed the {threshold:.0%} risk threshold and are recommended for outreach.")

        # Risk tier labels
        def tier_label(score):
            if score >= 0.70:
                return "🔴 Critical"
            elif score >= 0.55:
                return "🟠 High"
            else:
                return "🟡 Moderate-High"

        # Build display table
        display_cols = [
            "risk_score",
            "appointment_scheduled_at_discharge",
            "medication_prescribed_at_discharge",
            "prior_ed_visits_12mo",
            "housing_instability",
            "appointment_wait_days_clean",
            "diagnosis_severity",
            "discharge_quality_score",
        ]
        available = [c for c in display_cols if c in high_risk_df.columns]
        display = high_risk_df[available].copy()
        display.insert(0, "Risk Tier", high_risk_df["risk_score"].apply(tier_label))
        display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.3f}")

        rename_map = {
            "risk_score": "Risk Score",
            "appointment_scheduled_at_discharge": "Appt Scheduled",
            "medication_prescribed_at_discharge": "Rx Prescribed",
            "prior_ed_visits_12mo": "Prior ED (12mo)",
            "housing_instability": "Housing Instability",
            "appointment_wait_days_clean": "Wait Days",
            "diagnosis_severity": "Dx Severity",
            "discharge_quality_score": "Discharge Quality",
        }
        display = display.rename(columns=rename_map)

        # Paginate
        page_size = 50
        n_pages = max(1, (len(display) - 1) // page_size + 1)
        pg = st.number_input("Page", 1, n_pages, 1)
        start_idx = (pg - 1) * page_size
        end_idx = start_idx + page_size

        st.dataframe(
            display.iloc[start_idx:end_idx],
            use_container_width=True,
            height=400,
        )

        st.download_button(
            "⬇️ Download High-Risk List (CSV)",
            data=display.to_csv(index=False),
            file_name="high_risk_outreach_list.csv",
            mime="text/csv",
        )

        # Risk score by clinical drivers
        st.markdown("---")
        st.markdown("**Risk Score by Key Clinical Drivers**")
        col1, col2 = st.columns(2)
        with col1:
            if "appointment_scheduled_at_discharge" in risk_df.columns:
                appt_group = risk_df.groupby("appointment_scheduled_at_discharge")["risk_score"].mean().reset_index()
                appt_group["appointment_scheduled_at_discharge"] = appt_group["appointment_scheduled_at_discharge"].map({0: "No Appt Scheduled", 1: "Appt Scheduled"})
                fig = px.bar(appt_group, x="appointment_scheduled_at_discharge", y="risk_score",
                            color_discrete_sequence=["#2563EB"],
                            labels={"risk_score": "Mean Risk Score", "appointment_scheduled_at_discharge": ""},
                            title="Mean Risk Score by Appointment Scheduling")
                fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "housing_instability" in risk_df.columns:
                housing_group = risk_df.groupby("housing_instability")["risk_score"].mean().reset_index()
                housing_group["housing_instability"] = housing_group["housing_instability"].map({0: "Stable Housing", 1: "Unstable Housing"})
                fig = px.bar(housing_group, x="housing_instability", y="risk_score",
                            color_discrete_sequence=["#7C3AED"],
                            labels={"risk_score": "Mean Risk Score", "housing_instability": ""},
                            title="Mean Risk Score by Housing Status")
                fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Equity Audit
# ──────────────────────────────────────────────────────────────────────────────

elif page == "⚖️ Equity Audit":
    st.markdown('<div class="section-header">Model Equity Audit — LA County ARDI Framework</div>', unsafe_allow_html=True)
    st.caption("Evaluates whether model performance is equitable across demographic subgroups, "
               "aligned with LA County's Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan.")

    attr_labels = {
        "race_ethnicity": "Race / Ethnicity",
        "preferred_language": "Preferred Language",
        "insurance_type": "Insurance Type",
        "spa_region": "Service Planning Area (SPA)",
    }

    selected_attr = st.selectbox(
        "Select demographic attribute to audit:",
        list(attr_labels.keys()),
        format_func=lambda x: attr_labels[x],
    )

    fairness_df = load_fairness(selected_attr)

    if fairness_df is not None:
        col1, col2, col3 = st.columns(3)

        tprs = fairness_df["tpr_sensitivity"].dropna()
        fprs = fairness_df["fpr_fallout"].dropna()
        preds = fairness_df["predicted_failure_rate"].dropna()

        tpr_gap = tprs.max() - tprs.min() if len(tprs) > 1 else 0
        fpr_gap = fprs.max() - fprs.min() if len(fprs) > 1 else 0
        dp_gap = preds.max() - preds.min() if len(preds) > 1 else 0

        with col1:
            color = "normal" if tpr_gap < 0.10 else "inverse"
            st.metric("TPR Gap (max − min)", f"{tpr_gap:.3f}",
                     delta="✓ Within threshold" if tpr_gap < 0.10 else "⚠ Exceeds 0.10 threshold",
                     delta_color=color)
        with col2:
            color = "normal" if fpr_gap < 0.10 else "inverse"
            st.metric("FPR Gap (max − min)", f"{fpr_gap:.3f}",
                     delta="✓ Within threshold" if fpr_gap < 0.10 else "⚠ Exceeds 0.10 threshold",
                     delta_color=color)
        with col3:
            color = "normal" if dp_gap < 0.10 else "inverse"
            st.metric("Demographic Parity Gap", f"{dp_gap:.3f}",
                     delta="✓ Within threshold" if dp_gap < 0.10 else "⚠ Exceeds 0.10 threshold",
                     delta_color=color)

        st.markdown("---")

        # Plot equity figure
        equity_fig_path = f"reports/figures/fairness_{selected_attr}.png"
        if os.path.exists(equity_fig_path):
            st.image(equity_fig_path, caption=f"Disaggregated performance by {attr_labels[selected_attr]}")

        st.markdown("**Detailed Metrics by Group**")
        display_cols = ["group", "n", "actual_failure_rate", "predicted_failure_rate",
                        "tpr_sensitivity", "fpr_fallout", "precision_ppv", "roc_auc"]
        display_cols = [c for c in display_cols if c in fairness_df.columns]
        st.dataframe(fairness_df[display_cols], use_container_width=True)

        st.markdown("---")
        # Summary table
        summary_img = "reports/figures/fairness_summary_table.png"
        if os.path.exists(summary_img):
            st.markdown("**Fairness Audit Summary — All Attributes**")
            st.image(summary_img)
    else:
        st.info("Run the full pipeline to generate fairness audit results.")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            with st.expander(f"**{row['model']}**", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
                with col2:
                    st.metric("PR-AUC", f"{row['pr_auc']:.4f}")
                with col3:
                    st.metric("Brier Score", f"{row['brier_score']:.4f}",
                             help="Lower is better. 0 = perfect, 0.25 = random")
                with col4:
                    st.metric("Test Set Size", f"{int(row['n_test']):,}")

    col_left, col_right = st.columns(2)
    with col_left:
        roc_path = "reports/figures/roc_pr_curves.png"
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC and Precision-Recall Curves")
    with col_right:
        cal_path = "reports/figures/calibration_curve.png"
        if os.path.exists(cal_path):
            st.image(cal_path, caption="Calibration Curve — Predicted vs. Actual Rates")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: HIPAA Governance
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🔒 HIPAA Governance":
    st.markdown('<div class="section-header">HIPAA Data Governance Summary</div>', unsafe_allow_html=True)

    # Load audit log
    audit_path = "data/processed/deidentification_audit_log.csv"
    if os.path.exists(audit_path):
        audit_df = pd.read_csv(audit_path)
        st.success("✅ De-identification audit log present")
        st.markdown("**De-identification Audit Trail** — 45 CFR §164.514(b)(2) Safe Harbor")
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.warning("No audit log found. Run the pipeline first.")

    st.markdown("---")
    st.markdown("""
**Data Classification**

| Data Layer | Classification | Access |
|---|---|---|
| `data/raw/` | **Restricted** — contains synthetic PHI fields | Pipeline only |
| `data/processed/` | **Internal** — de-identified per Safe Harbor | Analysts with IRB |
| `reports/figures/` | **Internal** — aggregated analytics | Data science team |
| `models/` | **Confidential** — may encode population statistics | Data scientists |

**PHI Identifier Treatment (Safe Harbor — 45 CFR §164.514(b))**

| Identifier | Treatment Applied |
|---|---|
| Name (first, last) | **REMOVED** |
| Date of birth | **Truncated** → birth year only |
| Street address | **REMOVED** |
| ZIP code | **Truncated** → 3-digit prefix |
| Phone number | **REMOVED** |
| SSN last 4 | **REMOVED** |
| Medical Record Number | **Pseudonymized** → SHA-256 token |
| ED visit date | **Truncated** → year only |
| Age ≥ 90 | **Collapsed** → "90+" age band |

**Model Governance Controls**

- ✅ Fairness audit required before any deployment decision
- ✅ Model fidelity monitoring plan documented in `HIPAA_GOVERNANCE.md`
- ✅ SHAP explainability output available for clinical decision support audit
- ✅ All data is synthetic — no real patient data used in this prototype
- ✅ Access control design documented
- ⚠️ Production deployment would require: IRB approval, DUA, Privacy Officer sign-off
""")


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**Prototype — Research and Portfolio Use Only** | "
    "Synthetic data generated with Synthea-equivalent methodology | "
    "Not for clinical deployment without IRB approval, Privacy Officer sign-off, and LA County IT security review. | "
    "Built to demonstrate HIPAA-aware data science for LA County behavioral health operations."
)