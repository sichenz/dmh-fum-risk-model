# FUM Follow-Up Failure Risk Stratification Model

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/YOUR_USERNAME/dmh-fum-risk-model/main/streamlit_app.py)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/data-synthetic%20only-orange)

**Predicting Post-Crisis Follow-Up Failure in a Mental Health Population: A HIPAA-Compliant ML Pipeline**

> A reproducible, HIPAA-aware data science pipeline that predicts which mental health patients are unlikely to complete their 7-day post-ED follow-up visit — enabling proactive outreach to prevent psychiatric readmission.
>
> Portfolio project for the **LA County Department of Mental Health Data Scientist Supervisor** role (Exam b1765A).

---

## 🚀 Live Demo

➡️ **[Launch the dashboard on Streamlit Cloud](https://share.streamlit.io/YOUR_USERNAME/dmh-fum-risk-model/main/streamlit_app.py)**

The app self-initializes on first load (~5 seconds). No setup required.

---

## 📦 Deploy to GitHub + Streamlit Cloud

### Step 1 — Push to GitHub

```bash
cd dmh-fum-risk-model
git init
git add .
git commit -m "Initial commit — FUM risk stratification pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dmh-fum-risk-model.git
git push -u origin main
```

### Step 2 — Deploy to Streamlit Community Cloud (free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **"New app"**
3. Set:
   - **Repository:** `YOUR_USERNAME/dmh-fum-risk-model`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Click **"Deploy"**

The app deploys in ~2 minutes. The pipeline runs automatically on first page load.

### Step 3 — Update your README badge

Replace `YOUR_USERNAME` in the badge URLs above with your GitHub username.

---

The **HEDIS FUM (Follow-Up After Emergency Department Visit for Mental Illness)** measure tracks whether patients who visit an emergency department for a psychiatric crisis receive a follow-up outpatient visit within **7 or 30 days**. The LA County Department of Mental Health (LACDMH) actively runs this as a named **Performance Improvement Project (PIP)** in their 2025–2026 QAPI Work Plan.

Low FUM rates represent real care gaps: patients discharged after a mental health crisis who never receive follow-up care are at dramatically elevated risk of readmission, psychiatric deterioration, and homelessness.

This project builds a machine learning system to **proactively identify** which patients are at highest risk of missing that follow-up appointment — enabling targeted outreach by care coordinators before the window closes.

---

## Project Architecture

```
dmh-fum-risk-model/
├── run_pipeline.py              ← Master orchestrator (run this first)
├── requirements.txt
├── HIPAA_GOVERNANCE.md          ← Full HIPAA compliance documentation
│
├── src/
│   ├── data_generation.py       ← Synthetic EHR patient cohort generator
│   ├── deidentification.py      ← HIPAA Safe Harbor pipeline (45 CFR §164.514(b))
│   ├── features.py              ← Clinical + SDoH feature engineering
│   ├── model.py                 ← XGBoost + SHAP explainability
│   └── fairness.py              ← Demographic equity audit (ARDI framework)
│
├── dashboard/
│   └── app.py                   ← Streamlit operational dashboard
│
├── data/
│   ├── raw/                     ← Synthetic EHR with PHI-equivalent fields
│   ├── processed/               ← De-identified data + risk scores
│   └── interim/                 ← Train/test splits
│
├── models/                      ← Trained model artifacts
└── reports/
    ├── figures/                 ← All visualizations
    └── fairness_*.csv           ← Disaggregated equity metrics
```

---

## Quickstart

### 1. Install dependencies

```bash
cd dmh-fum-risk-model
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_pipeline.py
```

This runs all 5 stages end-to-end (~2–5 minutes):
1. **Generate** 5,000 synthetic behavioral health patients
2. **De-identify** using HIPAA Safe Harbor (removes/transforms all 18 PHI identifiers)
3. **Engineer** clinical, SDoH, and geographic features
4. **Train** Logistic Regression baseline + XGBoost model with SHAP explanations
5. **Audit** model equity across race/ethnicity, language, insurance, and SPA

### 3. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Pipeline Stages

### Stage 1 — Synthetic Data Generation (`src/data_generation.py`)

Generates a realistic LA County Medi-Cal behavioral health patient cohort using:
- ICD-10 F-code mental health diagnoses (schizophrenia, MDD, bipolar, anxiety, SUD)
- LA County demographic distributions (race, language, insurance, SPA region)
- ED visit history, discharge planning quality, social determinants of health
- A logistic-style label generation function encoding clinically documented risk factors

The generator intentionally produces PHI-equivalent fields (names, MRNs, DOBs, addresses) so the downstream de-identification pipeline has realistic input to process.

### Stage 2 — HIPAA Safe Harbor De-identification (`src/deidentification.py`)

Implements the 18-identifier Safe Harbor method per **45 CFR §164.514(b)(2)**:

| Treatment | Fields |
|---|---|
| **REMOVED** | `first_name`, `last_name`, `phone_number`, `street_address`, `ssn_last4` |
| **Pseudonymized** | `mrn` → one-way SHA-256 token |
| **Truncated** | `date_of_birth` → year only; `zip_code` → 3-digit prefix; `ed_visit_date` → year |
| **Binned** | `age` → 5-year bands; ages ≥ 90 collapsed to "90+" |

Includes automated PHI validation scan to detect residual identifier patterns, and produces an immutable audit trail CSV.

### Stage 3 — Feature Engineering (`src/features.py`)

Derives 25+ clinically meaningful predictors across 5 groups:

| Group | Example Features |
|---|---|
| Clinical severity | `diagnosis_severity`, `prior_ed_visits_capped`, `care_gap_days_log` |
| Discharge quality | `appointment_scheduled_at_discharge`, `wait_exceeds_7days`, `discharge_quality_score` |
| Social determinants | `housing_instability`, `homelessness_flag`, `sdoh_burden_score` |
| Geographic access | `distance_to_provider_miles`, `rural_spa_flag` |
| Demographic | `age_midpoint`, `insurance_encoded` |

### Stage 4 — Modeling (`src/model.py`)

**Logistic Regression** (interpretable baseline) and **XGBoost** (primary) with:
- Class-imbalance weighting (`scale_pos_weight`)
- Early stopping on PR-AUC (appropriate for imbalanced clinical data)
- Evaluation: ROC-AUC, PR-AUC, Brier score, calibration curve
- **SHAP TreeExplainer** for both global feature importance and per-patient explanations
- Risk tier output: Low / Moderate / High for care coordinator triage

### Stage 5 — Fairness Audit (`src/fairness.py`)

Disaggregated performance evaluation across:
- Race/Ethnicity
- Preferred Language
- Insurance Type
- Service Planning Area (SPA)

Metrics: True Positive Rate, False Positive Rate, Precision, ROC-AUC, Demographic Parity Difference, Equalized Odds Difference — aligned with LACDMH's **Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan**.

---

## Key Design Decisions

### Why XGBoost over deep learning?

Tabular clinical data with structured features responds better to gradient-boosted trees than deep learning. XGBoost also supports exact SHAP computation, which is critical for clinical explainability requirements. A black-box model cannot be deployed in clinical workflows.

### Why HIPAA Safe Harbor over Expert Determination?

Safe Harbor is the standard method used by LACDMH and its data partners (InfoHub, CalAIM, IBHIS integrations). It is deterministic, auditable, and doesn't require a statistical expert attestation for each dataset version.

### Why fairness auditing before SHAP?

Model explanations can mask systemic bias. A model can produce "good" SHAP explanations while performing poorly for minority subgroups. The fairness audit runs independently of explainability to ensure equity is evaluated on its own terms.

---

## HIPAA & Data Governance

See [`HIPAA_GOVERNANCE.md`](./HIPAA_GOVERNANCE.md) for the full governance documentation, including:
- Data classification schema
- Role-based access control matrix
- PHI identifier treatment log
- Model risk classification
- Drift monitoring plan
- Bias risk assessment
- Incident response procedures

---

## Alignment with LACDMH Priorities

| LACDMH Initiative | How This Project Connects |
|---|---|
| FUM Performance Improvement Project (QAPI 2025–2026) | Directly models this HEDIS measure |
| Homelessness Prevention Unit predictive analytics | Same multi-factor risk stratification methodology |
| IBHIS / EHR data warehouse redesign | Feature engineering designed for EHR-structured data |
| CalAIM / Medi-Cal managed care integration | Insurance type and care gap features reflect Medi-Cal population |
| ARDI Strategic Plan | Full fairness audit across race, language, SPA |
| GIS / spatial equity analysis | `spa_region`, `distance_to_provider_miles` features |
| Provider Language Capacity report | `preferred_language` and `limited_english_proficiency` features |

---

## Technologies

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing, baseline model, evaluation |
| XGBoost | Primary classifier |
| SHAP | Model explainability |
| fairlearn | Equity audit framework |
| Streamlit | Operational dashboard |
| Plotly | Interactive visualizations |
| Faker | Synthetic PHI generation |
| hashlib | SHA-256 pseudonymization |

---

## Candidate Note

This project was designed to directly demonstrate HIPAA-aware healthcare data science experience for the LACDMH Data Scientist Supervisor position. Every design decision — from the choice of HEDIS measure, to the de-identification method, to the ARDI fairness framework — was grounded in publicly available LACDMH documentation and mirrors the actual work of the department's data science section.
