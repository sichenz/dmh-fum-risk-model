"""
streamlit_app.py
----------------
Self-initializing Streamlit dashboard for the LA County FUM Follow-Up Failure
Risk Stratification Model.

This file is the Streamlit Cloud entry point. On first load it automatically:
  1. Generates a synthetic behavioral health patient cohort (Synthea-style)
  2. Applies HIPAA Safe Harbor de-identification (45 CFR §164.514(b)(2))
  3. Engineers clinical + SDoH + geographic features
  4. Trains Logistic Regression + XGBoost with SHAP explainability
  5. Runs a demographic equity audit (LA County ARDI framework)

The full pipeline completes in ~5 seconds and is cached for the session.
No data files need to be pre-committed to the repository.

Deploy to Streamlit Community Cloud:
  Main file: streamlit_app.py
  Python version: 3.11

Portfolio project for LA County behavioral health analytics.
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
    page_title="LA County FUM Risk Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/sichenz/dmh-fum-risk-model",
        "Report a bug": "https://github.com/sichenz/dmh-fum-risk-model/issues",
        "About": "Portfolio project for LA County behavioral health analytics. Synthetic data only.",
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
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

    /* ---------- Hero ---------- */
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

    /* ---------- Pill badges ---------- */
    .badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.3rem 0 1.4rem; }
    .pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-size: 0.76rem; font-weight: 600;
        padding: 0.36rem 0.8rem; border-radius: 999px;
        border: 1px solid var(--line); background: white; color: var(--ink-soft);
    }
    .pill-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success); }
    .pill.pill-accent { background: var(--accent-tint); border-color: #cfe3ff; color: var(--accent-dark); }

    /* ---------- Section headers ---------- */
    .section-header {
        font-size: 1.02rem; font-weight: 700; color: var(--ink);
        padding-bottom: 0.6rem; margin: 0.4rem 0 1.1rem;
        border-bottom: 1px solid var(--line);
    }
    .section-eyebrow {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 0.25rem; display: block;
    }

    /* ---------- Cards / metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: white; border: 1px solid var(--line);
        border-radius: 14px; padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 44, 0.04);
    }
    div[data-testid="stMetric"] label { color: var(--ink-soft) !important; font-weight: 600 !important; }
    div[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800 !important; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: white; border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.88rem;
    }
    .sidebar-brand {
        display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.1rem;
    }
    .sidebar-brand .mark {
        width: 34px; height: 34px; border-radius: 10px;
        background: linear-gradient(135deg, #0071E3, #16305e);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem; flex-shrink: 0;
    }
    .sidebar-brand .name { font-weight: 800; font-size: 0.98rem; line-height: 1.15; color: var(--ink); }
    .sidebar-brand .sub { font-size: 0.72rem; color: var(--ink-soft); }

    /* ---------- General polish ---------- */
    .stAlert { font-size: 0.85rem; border-radius: 12px; }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line); border-radius: 14px; background: white;
    }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }
    .stDownloadButton button, .stButton button {
        border-radius: 999px !important; font-weight: 600 !important;
        border: 1px solid var(--accent) !important;
    }
    .stDownloadButton button {
        background: var(--accent) !important; color: white !important;
    }
    hr { border-color: var(--line) !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Keep native controls usable when Streamlit changes its internal layout.
_control_css_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "controls.css"
)
with open(_control_css_path, encoding="utf-8") as _control_css:
    st.markdown(f"<style>{_control_css.read()}</style>", unsafe_allow_html=True)

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
<div class="hero">
    <span class="hero-eyebrow">🧠 &nbsp;LA County · Behavioral Health Analytics</span>
    <h1>FUM Follow-Up Risk Stratification Dashboard</h1>
    <p>Predicting which patients are likely to miss their post-crisis follow-up
       after a mental-health emergency department visit — so outreach teams can
       reach them first.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <span class="pill pill-accent"><span class="pill-dot"></span> HIPAA Safe Harbor de-identified</span>
    <span class="pill">🔒 45 CFR §164.514(b)</span>
    <span class="pill">🧪 Synthetic data only — no real patients</span>
    <span class="pill">Portfolio prototype</span>
</div>
""", unsafe_allow_html=True)

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
    st.markdown("""
    <div class="sidebar-brand">
        <div class="mark">🧠</div>
        <div>
            <div class="name">LA County FUM Model</div>
            <div class="sub">Portfolio Prototype</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
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
        "Active PIP in LA County QAPI Work Plan 2025–2026."
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
    st.markdown('<div class="section-header">Model Equity Audit — LA County ARDI Framework</div>',
                unsafe_allow_html=True)
    st.caption(
        "Evaluates whether model performance is equitable across demographic subgroups, "
        "aligned with LA County's Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan."
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
> language-concordant care access features), and (3) a formal review with LA County's
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
- ⚠️ **Production deployment requires:** IRB determination, DUA, Privacy Officer sign-off, LA County IT security review

> **HIPAA Note:** If deployed with real LA County data, this analytics pipeline would constitute a
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
**LA County** behavioral health data science initiative.

---

### Clinical Motivation

The **HEDIS FUM (Follow-Up After Emergency Department Visit for Mental Illness)** measure
tracks whether patients receive outpatient follow-up within 7 or 30 days after a psychiatric
ED visit. LA County runs this as a named **Performance Improvement Project (PIP)** in their
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
| 5 | `fairness.py` | Demographic equity audit (LA County ARDI framework) |

---

### Alignment with Active LA County Initiatives

| LA County Initiative | How This Project Connects |
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
- **HIPAA Safe Harbor over Expert Determination:** Standard method for LA County data partners (InfoHub, CalAIM, IBHIS)
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
        "All demographic distributions are modeled on publicly available LA County reports."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "**Prototype — Portfolio / Research Use Only** | "
    "Synthetic data generated with Synthea-equivalent methodology | "
    "Not for clinical deployment without IRB approval, Privacy Officer sign-off, "
    "and LA County IT Security review | "
    "LA County Behavioral Health Analytics"
)