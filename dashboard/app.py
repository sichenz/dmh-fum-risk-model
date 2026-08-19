"""
app.py — Streamlit Dashboard
----------------------------
Interactive operational dashboard for the LA County FUM Follow-Up Failure
Risk Stratification Model.

Run with:
    streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ui_theme import (
    BRAND,
    CHARCOAL,
    DANGER,
    LOW,
    MID,
    banner,
    configure_page,
    gap_delta,
    hero,
    inject_theme,
    metric_grid,
    page_header,
    plot,
    render_brand,
    risk_tier_label,
    sidebar_note,
    site_footer,
)

configure_page("FUM Risk Model")
inject_theme()


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


PAGES = [
    "Overview",
    "Outreach",
    "Equity audit",
    "Performance",
    "Governance",
]

with st.sidebar:
    render_brand()
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    threshold = st.slider(
        "Risk threshold",
        0.30, 0.80, 0.50, 0.05,
        help="Patients at or above this score are flagged for proactive outreach.",
    )
    st.caption(f"Score {threshold:.2f} and above is flagged for outreach.")
    st.markdown("---")
    sidebar_note(
        "HEDIS FUM",
        "7-day follow-up after an ED visit for mental illness. Low FUM rates are care gaps.",
    )
    
hero(
    "LA County · Behavioral health analytics",
    "Turn missed follow-ups into closed care gaps.",
    "Predict which patients will miss their 7-day visit after a mental-health "
    "emergency discharge — so outreach teams can reach them first.",
    [
        "HIPAA Safe Harbor",
        "45 CFR §164.514(b)",
        "Synthetic cohort only",
        "HEDIS FUM",
    ],
)

if not os.path.exists("data/processed/patient_risk_scores.csv"):
    banner(
        "Pipeline output was not found. Generate the cohort and scores first:",
        "warn",
    )
    st.code("python run_pipeline.py", language="bash")
    st.stop()

risk_df = load_risk_scores()
metrics_df = load_metrics()


if page == "Overview":
    page_header(
        "01 — Overview",
        "See who is about to fall through the crack.",
        "The test cohort, scored for follow-up failure after a psychiatric "
        "ED visit. Move the threshold in the sidebar to change who is flagged.",
    )

    if risk_df is not None:
        high_risk = (risk_df["risk_score"] >= threshold).sum()
        total = len(risk_df)
        actual_failures = risk_df["true_label"].sum() if "true_label" in risk_df.columns else 0
        avg_score = risk_df["risk_score"].mean()

        metric_grid([
            {"label": "Test-set patients", "value": f"{total:,}"},
            {"label": "Flagged high-risk", "value": f"{high_risk:,}",
             "hint": f"{high_risk / total:.1%} of the cohort"},
            {"label": "Actual failures", "value": f"{int(actual_failures):,}",
             "hint": f"{actual_failures / total:.1%} observed failure rate"},
            {"label": "Mean risk score", "value": f"{avg_score:.3f}"},
        ])

        col_left, col_right = st.columns([3, 2], gap="medium")

        with col_left:
            with st.container(border=True):
                st.markdown('<p class="panel-title">Risk score distribution</p>', unsafe_allow_html=True)
                st.markdown('<p class="panel-sub">Predicted probability of missing the 7-day follow-up.</p>', unsafe_allow_html=True)
                fig = px.histogram(
                    risk_df, x="risk_score", nbins=40,
                    color_discrete_sequence=[CHARCOAL],
                    labels={"risk_score": "Predicted failure probability"},
                )
                fig.add_vline(
                    x=threshold, line_dash="dash", line_color=DANGER,
                    annotation_text=f"Threshold {threshold:.2f}",
                    annotation_position="top right",
                    annotation_font_color=DANGER,
                )
                plot(fig, height=300)

        with col_right:
            with st.container(border=True):
                st.markdown('<p class="panel-title">Risk tiers</p>', unsafe_allow_html=True)
                st.markdown('<p class="panel-sub">Low, moderate, and high bands on the same scores.</p>', unsafe_allow_html=True)
                tier_counts = pd.cut(
                    risk_df["risk_score"],
                    bins=[0, 0.33, 0.60, 1.0],
                    labels=["Low", "Moderate", "High"],
                ).value_counts()
                fig_pie = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    color=tier_counts.index,
                    color_discrete_map={"Low": LOW, "Moderate": MID, "High": DANGER},
                    hole=0.62,
                )
                fig_pie.update_traces(textposition="outside", textinfo="percent+label")
                plot(fig_pie, height=300)

        page_header(
            "How the model decides",
            "The signals that move a risk score.",
            "SHAP values show which features raise or lower the chance of a missed follow-up.",
        )
        shap_bar_path = "reports/figures/shap_bar.png"
        shap_summary_path = "reports/figures/shap_summary.png"
        if os.path.exists(shap_bar_path) and os.path.exists(shap_summary_path):
            col_shap1, col_shap2 = st.columns(2, gap="medium")
            with col_shap1:
                with st.container(border=True):
                    st.markdown('<p class="panel-title">Mean absolute SHAP</p>', unsafe_allow_html=True)
                    st.image(shap_bar_path)
            with col_shap2:
                with st.container(border=True):
                    st.markdown('<p class="panel-title">SHAP beeswarm</p>', unsafe_allow_html=True)
                    st.image(shap_summary_path)
        else:
            banner("Run the pipeline to generate SHAP visualizations.", "warn")


elif page == "Outreach":
    page_header(
        "02 — Outreach",
        "The list coordinators should work first.",
        "Patients above the current threshold, ranked by predicted failure "
        "probability. Tokens are SHA-256 hashes — match to IBHIS by token.",
    )

    if risk_df is not None:
        high_risk_df = risk_df[risk_df["risk_score"] >= threshold].copy()
        high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)

        banner(
            f"<strong>{len(high_risk_df):,} patients</strong> sit at or above "
            f"the {threshold:.0%} threshold and are recommended for outreach.",
            "info",
        )

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
        display.insert(0, "Risk tier", high_risk_df["risk_score"].apply(risk_tier_label))
        display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.3f}")
        display = display.rename(columns={
            "risk_score": "Risk score",
            "appointment_scheduled_at_discharge": "Appt scheduled",
            "medication_prescribed_at_discharge": "Rx prescribed",
            "prior_ed_visits_12mo": "Prior ED (12mo)",
            "housing_instability": "Housing instability",
            "appointment_wait_days_clean": "Wait days",
            "diagnosis_severity": "Dx severity",
            "discharge_quality_score": "Discharge quality",
        })

        page_size = 50
        n_pages = max(1, (len(display) - 1) // page_size + 1)
        pg = st.number_input("Page", 1, n_pages, 1)
        start_idx = (pg - 1) * page_size
        st.dataframe(
            display.iloc[start_idx:start_idx + page_size],
            use_container_width=True,
            height=400,
        )
        st.download_button(
            "Download outreach list",
            data=display.to_csv(index=False),
            file_name="high_risk_outreach_list.csv",
            mime="text/csv",
        )

        page_header(
            "Clinical drivers",
            "Where risk concentrates.",
            "Average predicted failure by two of the strongest operational levers.",
        )
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if "appointment_scheduled_at_discharge" in risk_df.columns:
                with st.container(border=True):
                    appt_group = risk_df.groupby("appointment_scheduled_at_discharge")["risk_score"].mean().reset_index()
                    appt_group["appointment_scheduled_at_discharge"] = appt_group["appointment_scheduled_at_discharge"].map(
                        {0: "No appointment", 1: "Appointment scheduled"}
                    )
                    fig = px.bar(
                        appt_group,
                        x="appointment_scheduled_at_discharge",
                        y="risk_score",
                        color_discrete_sequence=[CHARCOAL],
                        labels={"risk_score": "Mean risk score", "appointment_scheduled_at_discharge": ""},
                    )
                    plot(fig, height=280, title="Mean risk by appointment scheduling")
        with col2:
            if "housing_instability" in risk_df.columns:
                with st.container(border=True):
                    housing_group = risk_df.groupby("housing_instability")["risk_score"].mean().reset_index()
                    housing_group["housing_instability"] = housing_group["housing_instability"].map(
                        {0: "Stable housing", 1: "Unstable housing"}
                    )
                    fig = px.bar(
                        housing_group,
                        x="housing_instability",
                        y="risk_score",
                        color_discrete_sequence=[BRAND],
                        labels={"risk_score": "Mean risk score", "housing_instability": ""},
                    )
                    plot(fig, height=280, title="Mean risk by housing status")


elif page == "Equity audit":
    page_header(
        "03 — Equity audit",
        "Does the model work equally well for everyone?",
        "Disaggregated performance aligned with LA County's Anti-Racism, "
        "Diversity, and Inclusion (ARDI) Strategic Plan.",
    )

    attr_labels = {
        "race_ethnicity": "Race / ethnicity",
        "preferred_language": "Preferred language",
        "insurance_type": "Insurance type",
        "spa_region": "Service Planning Area",
    }
    selected_attr = st.selectbox(
        "Demographic attribute",
        list(attr_labels.keys()),
        format_func=lambda x: attr_labels[x],
    )
    fairness_df = load_fairness(selected_attr)

    if fairness_df is not None:
        tprs = fairness_df["tpr_sensitivity"].dropna()
        fprs = fairness_df["fpr_fallout"].dropna()
        preds = fairness_df["predicted_failure_rate"].dropna()
        tpr_gap = tprs.max() - tprs.min() if len(tprs) > 1 else 0
        fpr_gap = fprs.max() - fprs.min() if len(fprs) > 1 else 0
        dp_gap = preds.max() - preds.min() if len(preds) > 1 else 0

        tpr_delta, tpr_color = gap_delta(tpr_gap)
        fpr_delta, fpr_color = gap_delta(fpr_gap)
        dp_delta, dp_color = gap_delta(dp_gap)

        col1, col2, col3 = st.columns(3)
        col1.metric("TPR gap", f"{tpr_gap:.3f}", delta=tpr_delta, delta_color=tpr_color)
        col2.metric("FPR gap", f"{fpr_gap:.3f}", delta=fpr_delta, delta_color=fpr_color)
        col3.metric("Demographic parity gap", f"{dp_gap:.3f}", delta=dp_delta, delta_color=dp_color)

        equity_fig_path = f"reports/figures/fairness_{selected_attr}.png"
        if os.path.exists(equity_fig_path):
            with st.container(border=True):
                st.markdown(
                    f'<p class="panel-title">Performance by {attr_labels[selected_attr].lower()}</p>',
                    unsafe_allow_html=True,
                )
                st.image(equity_fig_path)

        st.markdown("**Detailed metrics by group**")
        display_cols = ["group", "n", "actual_failure_rate", "predicted_failure_rate",
                        "tpr_sensitivity", "fpr_fallout", "precision_ppv", "roc_auc"]
        display_cols = [c for c in display_cols if c in fairness_df.columns]
        st.dataframe(fairness_df[display_cols], use_container_width=True)

        summary_img = "reports/figures/fairness_summary_table.png"
        if os.path.exists(summary_img):
            with st.container(border=True):
                st.markdown('<p class="panel-title">Fairness summary — all attributes</p>', unsafe_allow_html=True)
                st.image(summary_img)
    else:
        banner("Run the full pipeline to generate fairness audit results.", "warn")


elif page == "Performance":
    page_header(
        "04 — Performance",
        "Discrimination, precision, and calibration.",
        "The metrics that matter when the outcome is imbalanced and the "
        "score has to be a trustworthy probability.",
    )

    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            with st.expander(f"{row['model']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
                col2.metric("PR-AUC", f"{row['pr_auc']:.4f}")
                col3.metric("Brier score", f"{row['brier_score']:.4f}",
                            help="Lower is better. 0 is perfect; 0.25 is a coin flip.")
                col4.metric("Test-set size", f"{int(row['n_test']):,}")

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        roc_path = "reports/figures/roc_pr_curves.png"
        if os.path.exists(roc_path):
            with st.container(border=True):
                st.markdown('<p class="panel-title">ROC and precision-recall</p>', unsafe_allow_html=True)
                st.image(roc_path)
    with col_right:
        cal_path = "reports/figures/calibration_curve.png"
        if os.path.exists(cal_path):
            with st.container(border=True):
                st.markdown('<p class="panel-title">Calibration</p>', unsafe_allow_html=True)
                st.image(cal_path)


elif page == "Governance":
    page_header(
        "05 — Governance",
        "Safe Harbor, end to end.",
        "Every identifier treatment is logged. This session used a synthetic "
        "cohort; the same controls would apply to real LA County data.",
    )

    audit_path = "data/processed/deidentification_audit_log.csv"
    if os.path.exists(audit_path):
        audit_df = pd.read_csv(audit_path)
        banner("De-identification audit log is present.", "ok")
        st.markdown("**Audit trail** — 45 CFR §164.514(b)(2) Safe Harbor")
        st.dataframe(audit_df, use_container_width=True)
    else:
        banner("No audit log found. Run the pipeline first.", "warn")

    st.markdown("""
**Data classification**

| Data layer | Classification | Access |
|---|---|---|
| `data/raw/` | Restricted — synthetic PHI fields | Pipeline only |
| `data/processed/` | Internal — de-identified per Safe Harbor | Analysts with IRB |
| `reports/figures/` | Internal — aggregated analytics | Data science team |
| `models/` | Confidential — may encode population statistics | Data scientists |

**PHI identifier treatment (Safe Harbor — 45 CFR §164.514(b))**

| Identifier | Treatment |
|---|---|
| Name (first, last) | Removed |
| Date of birth | Truncated to birth year |
| Street address | Removed |
| ZIP code | Truncated to 3-digit prefix |
| Phone number | Removed |
| SSN last 4 | Removed |
| Medical record number | Pseudonymized SHA-256 token |
| ED visit date | Truncated to year |
| Age 90 and over | Collapsed to 90+ |

**Model governance**

- Fairness audit required before any deployment decision
- Model fidelity monitoring plan documented in `HIPAA_GOVERNANCE.md`
- SHAP explanations available for clinical decision-support review
- All data is synthetic — no real patient data in this prototype
- Access-control design is documented
- Production would still require IRB approval, a DUA, and Privacy Officer sign-off
""")

site_footer()
