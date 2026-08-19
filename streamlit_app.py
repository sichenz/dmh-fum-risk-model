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
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui_theme import (
    banner,
    render_brand,
    callout,
    configure_page,
    gap_delta,
    hero,
    inject_theme,
    metric_grid,
    page_header,
    plot,
    risk_tier_label,
    sidebar_note,
    site_footer,
    steps,
    BRAND,
    CHARCOAL,
    DANGER,
    LOW,
    MID,
)

configure_page("FUM Risk Model")
inject_theme()


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
    """
    import time
    from src.data_generation import generate_patient_cohort
    from src.deidentification import run_safe_harbor_deidentification
    from src.features import prepare_model_data
    from src.model import run_full_pipeline
    from src.fairness import run_fairness_audit

    _make_dirs()
    t0 = time.time()

    raw_df = generate_patient_cohort(n_patients=5000)
    raw_df.to_csv("data/raw/synthetic_patients_raw.csv", index=False)

    deident_df, audit = run_safe_harbor_deidentification(raw_df, verbose=False)
    deident_df.to_csv("data/processed/patients_deidentified.csv", index=False)
    audit.report().to_csv("data/processed/deidentification_audit_log.csv", index=False)

    data = prepare_model_data(deident_df)
    results = run_full_pipeline(data)

    y_prob = results["xgb_model"].predict_proba(data["X_test"])[:, 1]
    group_metrics, summaries = run_fairness_audit(
        data["y_test"], y_prob, data["sensitive_test"]
    )

    return {
        "risk_df":       results["risk_df"],
        "metrics_df":    results["metrics"],
        "deident_df":    deident_df,
        "audit_df":      audit.report(),
        "group_metrics": group_metrics,
        "summaries":     summaries,
        "data":          data,
        "xgb_model":     results["xgb_model"],
        "elapsed":       time.time() - t0,
    }


PAGES = [
    "Overview",
    "Outreach",
    "Equity audit",
    "Performance",
    "Governance",
    "About",
]

# Chrome first so the logo, nav, and threshold are visible while the
# first-load pipeline is still generating / training.
with st.sidebar:
    render_brand()
    st.markdown("---")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    threshold = st.slider(
        "Risk threshold",
        min_value=0.30, max_value=0.80,
        value=0.50, step=0.05,
        help="Patients at or above this score are flagged for proactive outreach.",
    )
    st.caption(f"Score {threshold:.2f} and above is flagged for outreach.")
    st.markdown("---")
    sidebar_note(
        "HEDIS FUM",
        "7-day follow-up after an ED visit for mental illness. "
        "Active PIP in the LA County QAPI Work Plan 2025–2026.",
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

pipeline_ready = os.path.exists("data/processed/patient_risk_scores.csv")
if not pipeline_ready:
    with st.spinner(
        "Initializing the model — generating a synthetic cohort, applying "
        "Safe Harbor de-identification, training XGBoost, and running the "
        "equity audit. This usually takes a few seconds."
    ):
        pipeline = run_pipeline()
else:
    pipeline = run_pipeline()

risk_df = pipeline["risk_df"]
metrics_df = pipeline["metrics_df"]
audit_df = pipeline["audit_df"]
group_metrics = pipeline["group_metrics"]
summaries = pipeline["summaries"]

elapsed = pipeline.get("elapsed", 0)
if elapsed:
    with st.sidebar:
        st.caption(f"Pipeline ran in {elapsed:.1f}s")


if page == "Overview":
    page_header(
        "01 — Overview",
        "See who is about to fall through the crack.",
        "The test cohort, scored for follow-up failure after a psychiatric "
        "ED visit. Move the threshold in the sidebar to change who is flagged.",
    )

    steps([
        ("01", "Score the discharge",
         "Clinical, social, and geographic signals become a failure probability."),
        ("02", "Rank the list",
         "Care coordinators see the highest-risk patients first, not a buried inbox."),
        ("03", "Check for equity",
         "Performance is audited across race, language, insurance, and SPA."),
        ("04", "Close the 7-day window",
         "Outreach happens before the HEDIS follow-up clock runs out."),
    ])

    high_risk = (risk_df["risk_score"] >= threshold).sum()
    total = len(risk_df)
    actual_fail = int(risk_df["true_label"].sum()) if "true_label" in risk_df.columns else 0
    mean_score = risk_df["risk_score"].mean()

    metric_grid([
        {"label": "Test-set patients", "value": f"{total:,}"},
        {"label": "Flagged high-risk", "value": f"{high_risk:,}",
         "hint": f"{high_risk / total:.1%} of the cohort"},
        {"label": "Actual failures", "value": f"{actual_fail:,}",
         "hint": f"{actual_fail / total:.1%} observed failure rate"},
        {"label": "Mean risk score", "value": f"{mean_score:.3f}"},
    ])

    col_l, col_r = st.columns([3, 2], gap="medium")

    with col_l:
        with st.container(border=True):
            st.markdown('<p class="panel-title">Risk score distribution</p>', unsafe_allow_html=True)
            st.markdown('<p class="panel-sub">Predicted probability of missing the 7-day follow-up.</p>', unsafe_allow_html=True)
            fig = px.histogram(
                risk_df, x="risk_score", nbins=40,
                color_discrete_sequence=[CHARCOAL],
                labels={"risk_score": "Predicted failure probability", "count": "Patients"},
            )
            fig.add_vline(
                x=threshold, line_dash="dash", line_color=DANGER,
                annotation_text=f"Threshold {threshold:.2f}",
                annotation_position="top right",
                annotation_font_color=DANGER,
            )
            plot(fig, height=300)

    with col_r:
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
    shap_bar = "reports/figures/shap_bar.png"
    shap_bees = "reports/figures/shap_summary.png"
    if os.path.exists(shap_bar) and os.path.exists(shap_bees):
        sc1, sc2 = st.columns(2, gap="medium")
        with sc1:
            with st.container(border=True):
                st.markdown('<p class="panel-title">Mean absolute SHAP</p>', unsafe_allow_html=True)
                st.markdown('<p class="panel-sub">Overall feature importance.</p>', unsafe_allow_html=True)
                st.image(shap_bar)
        with sc2:
            with st.container(border=True):
                st.markdown('<p class="panel-title">SHAP beeswarm</p>', unsafe_allow_html=True)
                st.markdown('<p class="panel-sub">Direction and magnitude of each feature.</p>', unsafe_allow_html=True)
                st.image(shap_bees)
    else:
        banner("SHAP plots are not available yet. Reload the page to re-run the pipeline.", "warn")


elif page == "Outreach":
    page_header(
        "02 — Outreach",
        "The list coordinators should work first.",
        "Patients above the current threshold, ranked by predicted failure "
        "probability. Tokens are SHA-256 hashes — match to IBHIS by token.",
    )

    high_risk_df = risk_df[risk_df["risk_score"] >= threshold].copy()
    high_risk_df = high_risk_df.sort_values("risk_score", ascending=False)

    banner(
        f"<strong>{len(high_risk_df):,} patients</strong> sit at or above "
        f"the {threshold:.0%} threshold and are recommended for outreach.",
        "info",
    )

    display_cols = [
        "risk_score", "appointment_scheduled_at_discharge",
        "medication_prescribed_at_discharge", "prior_ed_visits_12mo",
        "housing_instability", "appointment_wait_days_clean",
        "diagnosis_severity", "discharge_quality_score",
    ]
    avail = [c for c in display_cols if c in high_risk_df.columns]
    display = high_risk_df[avail].copy()
    display.insert(0, "Risk tier", high_risk_df["risk_score"].apply(risk_tier_label))
    display["risk_score"] = display["risk_score"].apply(lambda x: f"{x:.3f}")
    display = display.rename(columns={
        "risk_score": "Risk score",
        "appointment_scheduled_at_discharge": "Appt scheduled",
        "medication_prescribed_at_discharge": "Rx prescribed",
        "prior_ed_visits_12mo": "Prior ED (12mo)",
        "housing_instability": "Housing instability",
        "appointment_wait_days_clean": "Appt wait days",
        "diagnosis_severity": "Dx severity",
        "discharge_quality_score": "Discharge quality",
    })

    page_size = 50
    n_pages = max(1, (len(display) - 1) // page_size + 1)
    pg = st.number_input("Page", 1, n_pages, 1)
    st.dataframe(
        display.iloc[(pg - 1) * page_size: pg * page_size],
        use_container_width=True, height=380,
    )
    st.download_button(
        "Download outreach list",
        data=display.to_csv(index=False),
        file_name="high_risk_outreach_list.csv",
        mime="text/csv",
    )

    st.markdown("")
    page_header(
        "Clinical drivers",
        "Where risk concentrates.",
        "Average predicted failure by two of the strongest operational levers.",
    )
    dc1, dc2 = st.columns(2, gap="medium")
    with dc1:
        if "appointment_scheduled_at_discharge" in risk_df.columns:
            with st.container(border=True):
                ag = (risk_df.groupby("appointment_scheduled_at_discharge")["risk_score"]
                             .mean().reset_index())
                ag["appointment_scheduled_at_discharge"] = ag["appointment_scheduled_at_discharge"].map(
                    {0: "No appointment", 1: "Appointment scheduled"}
                )
                fig = px.bar(
                    ag, x="appointment_scheduled_at_discharge", y="risk_score",
                    color_discrete_sequence=[CHARCOAL],
                    labels={"risk_score": "Mean risk score",
                            "appointment_scheduled_at_discharge": ""},
                )
                plot(fig, height=260, title="Mean risk by appointment scheduling")
    with dc2:
        if "housing_instability" in risk_df.columns:
            with st.container(border=True):
                hg = (risk_df.groupby("housing_instability")["risk_score"]
                             .mean().reset_index())
                hg["housing_instability"] = hg["housing_instability"].map(
                    {0: "Stable housing", 1: "Unstable housing"}
                )
                fig = px.bar(
                    hg, x="housing_instability", y="risk_score",
                    color_discrete_sequence=[BRAND],
                    labels={"risk_score": "Mean risk score",
                            "housing_instability": ""},
                )
                plot(fig, height=260, title="Mean risk by housing status")


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
    selected = st.selectbox(
        "Demographic attribute",
        list(attr_labels.keys()),
        format_func=lambda x: attr_labels[x],
    )

    gm_df = group_metrics.get(selected)

    if gm_df is not None and not gm_df.empty:
        tpr_gap = (gm_df["tpr_sensitivity"].max() - gm_df["tpr_sensitivity"].min()
                   if gm_df["tpr_sensitivity"].notna().any() else 0)
        fpr_gap = (gm_df["fpr_fallout"].max() - gm_df["fpr_fallout"].min()
                   if gm_df["fpr_fallout"].notna().any() else 0)
        dp_gap = (gm_df["predicted_failure_rate"].max() - gm_df["predicted_failure_rate"].min()
                  if gm_df["predicted_failure_rate"].notna().any() else 0)

        tpr_delta, tpr_color = gap_delta(tpr_gap)
        fpr_delta, fpr_color = gap_delta(fpr_gap)
        dp_delta, dp_color = gap_delta(dp_gap)

        c1, c2, c3 = st.columns(3)
        c1.metric("TPR gap", f"{tpr_gap:.3f}", delta=tpr_delta, delta_color=tpr_color)
        c2.metric("FPR gap", f"{fpr_gap:.3f}", delta=fpr_delta, delta_color=fpr_color)
        c3.metric("Demographic parity gap", f"{dp_gap:.3f}", delta=dp_delta, delta_color=dp_color)

        fig_path = f"reports/figures/fairness_{selected}.png"
        if os.path.exists(fig_path):
            with st.container(border=True):
                st.markdown(
                    f'<p class="panel-title">Performance by {attr_labels[selected].lower()}</p>',
                    unsafe_allow_html=True,
                )
                st.image(fig_path)

        st.markdown("")
        st.markdown("**Detailed metrics by group**")
        show_cols = ["group", "n", "actual_failure_rate", "predicted_failure_rate",
                     "tpr_sensitivity", "fpr_fallout", "precision_ppv", "roc_auc"]
        show_cols = [c for c in show_cols if c in gm_df.columns]
        st.dataframe(gm_df[show_cols], use_container_width=True)

        summary_img = "reports/figures/fairness_summary_table.png"
        if os.path.exists(summary_img):
            st.markdown("")
            with st.container(border=True):
                st.markdown('<p class="panel-title">Fairness summary — all attributes</p>', unsafe_allow_html=True)
                st.image(summary_img)

        st.markdown("")
        callout(
            "<strong>How to read this.</strong> The audit is expected to find "
            "real gaps — particularly for SPA 1 (Antelope Valley) and for "
            "Black / African American patients. Those gaps do not pass a "
            "standard equalized-odds threshold of 0.10. In production this "
            "would trigger subgroup recalibration, targeted feature work "
            "(for example language-concordant access), and a formal review "
            "with ARDI leadership before any deployment decision."
        )
    else:
        banner("Fairness data is not available for this attribute.", "warn")


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
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("ROC-AUC", f"{row['roc_auc']:.4f}")
                mc2.metric("PR-AUC", f"{row['pr_auc']:.4f}",
                           help="Precision-Recall AUC — the honest metric under class imbalance.")
                mc3.metric("Brier score", f"{row['brier_score']:.4f}",
                           help="Lower is better. 0 is perfect; 0.25 is a coin flip.")
                mc4.metric("Test-set size", f"{int(row['n_test']):,}")

    pc1, pc2 = st.columns(2, gap="medium")
    roc_path = "reports/figures/roc_pr_curves.png"
    cal_path = "reports/figures/calibration_curve.png"
    with pc1:
        if os.path.exists(roc_path):
            with st.container(border=True):
                st.markdown('<p class="panel-title">ROC and precision-recall</p>', unsafe_allow_html=True)
                st.image(roc_path)
    with pc2:
        if os.path.exists(cal_path):
            with st.container(border=True):
                st.markdown('<p class="panel-title">Calibration</p>', unsafe_allow_html=True)
                st.markdown('<p class="panel-sub">Predicted probability versus observed failure rate.</p>', unsafe_allow_html=True)
                st.image(cal_path)

    st.markdown("")
    st.markdown("""
| Metric | Why it matters for FUM |
|---|---|
| **PR-AUC** | A ~66% failure rate makes ROC-AUC look optimistic. PR-AUC is the more honest ranking metric. |
| **Calibration** | A score of 0.70 should mean about a 70% chance of a missed visit — otherwise triage is guesswork. |
| **Brier score** | A proper scoring rule that penalizes both overconfident misses and underconfident hits. |
| **ROC-AUC** | Threshold-free discrimination between completers and non-completers. |
""")


elif page == "Governance":
    page_header(
        "05 — Governance",
        "Safe Harbor, end to end.",
        "Every identifier treatment is logged. This session used a synthetic "
        "cohort; the same controls would apply to real LA County data.",
    )

    if audit_df is not None and not audit_df.empty:
        banner("De-identification audit log generated for this session.", "ok")
        st.markdown("**Audit trail** — 45 CFR §164.514(b)(2) Safe Harbor")
        st.dataframe(audit_df, use_container_width=True)
    else:
        banner("Audit log not found.", "warn")

    st.markdown("")
    st.markdown("""
**Data classification**

| Layer | Classification | Access |
|---|---|---|
| `data/raw/` | Restricted — synthetic PHI-equivalent fields | Pipeline runtime only |
| `data/processed/` | Internal — Safe Harbor de-identified | Analysts with IRB and DUA |
| `reports/` | Internal — aggregated statistics | Data science team |
| `models/` | Confidential — may encode population statistics | Data scientists |

**PHI identifier treatment — 45 CFR §164.514(b)(2)**

| # | Identifier | Treatment | Status |
|---|---|---|---|
| 1 | Names | Removed | Complete |
| 2 | Street address | Removed | Complete |
| 2 | ZIP code | Truncated to 3-digit prefix | Complete |
| 3 | Date of birth | Truncated to birth year | Complete |
| 3 | ED visit date | Truncated to year | Complete |
| 3 | Age 90 and over | Collapsed to 90+ | Complete |
| 4 | Phone numbers | Removed | Complete |
| 7 | SSN (last 4) | Removed | Complete |
| 8 | Medical record number | Pseudonymized SHA-256 token | Complete |
| 5–18 | All other identifiers | Not present | Complete |

**Model governance**

- Fairness audit required before any deployment decision
- Model fidelity monitoring plan documented in `HIPAA_GOVERNANCE.md`
- SHAP explanations available for clinical decision-support review
- All data in this prototype is synthetic
- Access-control design is documented
- Production would still require IRB determination, a DUA, Privacy Officer sign-off, and LA County IT security review
""")
    callout(
        "<strong>HIPAA note.</strong> With real LA County data this pipeline "
        "would be a Healthcare Operations (QAPI) use of PHI under "
        "45 CFR §164.501 — permissible without patient authorization only if "
        "the safeguards in <code>HIPAA_GOVERNANCE.md</code> are in place."
    )


elif page == "About":
    page_header(
        "06 — About",
        "A HIPAA-aware risk model for the 7-day FUM gap.",
        "Portfolio work for LA County behavioral health analytics. "
        "The full pipeline runs on first load. Nothing here is real PHI.",
    )

    steps([
        ("01", "Generate",
         "5,000 synthetic LA County Medi-Cal behavioral health patients."),
        ("02", "De-identify",
         "Safe Harbor treatment of the 18 HIPAA identifiers."),
        ("03", "Engineer",
         "25+ clinical, SDoH, and geographic features."),
        ("04", "Score and audit",
         "XGBoost + SHAP, then an ARDI-aligned equity review."),
    ])

    st.markdown("""
### Why this exists

The **HEDIS FUM** measure asks whether a patient seen in the ED for mental
illness receives outpatient follow-up within 7 or 30 days. LA County runs
this as a named Performance Improvement Project in the 2025–2026 QAPI Work
Plan. Low FUM rates are people falling through the crack. This model ranks
who is most likely to miss that visit so limited outreach capacity is
pointed at the right discharges.

### How it lines up with active County work

| LA County initiative | Connection |
|---|---|
| FUM Performance Improvement Project (QAPI 2025–2026) | Directly models this HEDIS measure |
| Homelessness Prevention Unit predictive analytics | Same multi-factor risk-stratification approach |
| IBHIS / CalAIM data integration | Features designed for EHR-structured data |
| ARDI Strategic Plan | Fairness audit across race, language, and SPA |
| GIS / spatial equity analysis | SPA region and provider-distance features |
| Provider Language Capacity Report | LEP feature and language-disaggregated fairness |

### Technical choices

- **XGBoost over deep learning** — tabular clinical data and exact SHAP values
- **Safe Harbor over Expert Determination** — the method County data partners already use
- **PR-AUC as the primary metric** — accounts for a ~66% failure rate
- **Fairness kept independent of explainability** — SHAP can look clean while a subgroup is underserved
- **Shallow trees (max depth 4)** — more readable explanations for clinical staff

**Stack:** Python, scikit-learn, XGBoost, SHAP, fairlearn, Streamlit, Plotly, Faker
""")
    banner(
        "<strong>Synthetic data only.</strong> No real patient records were "
        "used at any stage. Demographic distributions are modeled on publicly "
        "available LA County reports.",
        "ok",
    )

site_footer()
