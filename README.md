# FUM Follow-Up Failure Risk Prediction Model

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dmh-fum-risk-model-hme9rttxfrvndwicmq6byz.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/data-synthetic%20only-orange)

**Predicting post-crisis follow-up failure in LA County's mental health population**

> This is a reproducible pipeline that flags which mental health patients are unlikely to show up for their 7-day follow-up visit after an ED crisis — so care coordinators can reach out before that window closes instead of after.

---

## Live demo

➡️ **[Open the dashboard](https://dmh-fum-risk-model-hme9rttxfrvndwicmq6byz.streamlit.app/)**

The first visit after the app wakes up takes a few seconds — it's generating a fresh synthetic cohort and training the model from scratch, not loading pre-made results. After that first run it's cached for the session, so there's nothing to install and nothing to configure.

---

## Why this project exists

The **HEDIS FUM (Follow-Up After Emergency Department Visit for Mental Illness)** measure tracks whether someone who lands in the ED during a psychiatric crisis actually gets seen for outpatient follow-up within 7 or 30 days. LA County runs this as a named Performance Improvement Project in its 2025–2026 QAPI Work Plan, and for good reason: patients who fall through that gap are at a much higher risk of readmission, deterioration, and homelessness.

Most FUM improvement work focuses on the process for everyone. This project takes a more targeted angle — instead of treating every discharge the same way, it tries to predict *in advance* which patients are most likely to miss that follow-up, so limited outreach capacity gets pointed at the people who need it most.

---

## What's in here

```
fum-risk-model/
├── run_pipeline.py              ← run this to execute everything locally
├── streamlit_app.py             ← entry point for the live dashboard
├── requirements.txt
├── HIPAA_GOVERNANCE.md          ← full compliance write-up
│
├── src/
│   ├── data_generation.py       ← builds a synthetic EHR-style patient cohort
│   ├── deidentification.py      ← HIPAA Safe Harbor de-identification (45 CFR §164.514(b))
│   ├── features.py              ← clinical + SDoH + geographic feature engineering
│   ├── model.py                 ← XGBoost model with SHAP explainability
│   └── fairness.py              ← demographic equity audit (ARDI framework)
│
├── dashboard/
│   └── app.py                   ← local Streamlit dashboard (mirrors streamlit_app.py, for `streamlit run`)
│
├── data/                        ← raw / de-identified / train-test data, generated at runtime
├── models/                      ← trained model artifacts
└── reports/
    ├── figures/                 ← charts and plots
    └── fairness_*.csv           ← disaggregated equity metrics
```

Everything here is generated on the fly — there's no PHI, real or synthetic, actually physically committed in this repo.

---

## Running it yourself

```bash
git clone https://github.com/sichenz/fum-risk-model.git
cd fum-risk-model
pip install -r requirements.txt
python run_pipeline.py
```

That kicks off all five stages end to end, usually in under 10 seconds:

1. Generate 5,000 synthetic behavioral health patients
2. De-identify them using HIPAA Safe Harbor (strips or transforms all 18 PHI identifiers)
3. Engineer clinical, SDoH, and geographic features
4. Train a logistic regression baseline plus an XGBoost model with SHAP explanations
5. Audit the model's equity across race/ethnicity, language, insurance, and Service Planning Area

Then, to look through the results yourself:

```bash
streamlit run dashboard/app.py
```

---

## Walking through the pipeline

### 1. Synthetic data generation (`src/data_generation.py`)

This builds a fake but plausible LA County Medi-Cal behavioral health cohort — ICD-10 F-code diagnoses, ED visit history, discharge planning details, social determinants of health, all drawn from distributions roughly matched to LA County's published demographics. It deliberately includes PHI-equivalent fields (names, MRNs, dates of birth, addresses) so there's something realistic for the de-identification step to actually work on.

### 2. HIPAA Safe Harbor de-identification (`src/deidentification.py`)

This applies the 18-identifier Safe Harbor method from **45 CFR §164.514(b)(2)**:

| Treatment | Fields |
|---|---|
| Removed entirely | `first_name`, `last_name`, `phone_number`, `street_address`, `ssn_last4` |
| Pseudonymized | `mrn` → one-way SHA-256 token |
| Truncated | `date_of_birth` → year only, `zip_code` → 3-digit prefix, `ed_visit_date` → year |
| Binned | `age` → 5-year bands, with anything 90+ collapsed into one bucket |

It also runs an automated scan afterward to check for residual identifier patterns that might have slipped through, and writes out an immutable audit log.

### 3. Feature engineering (`src/features.py`)

Turns the de-identified data into 25+ predictors grouped into five buckets: clinical severity, discharge quality, social determinants, geographic access, and demographics — things like `care_gap_days_log`, `discharge_quality_score`, `homelessness_flag`, and `distance_to_provider_miles`.

### 4. Modeling (`src/model.py`)

A logistic regression baseline for interpretability, and XGBoost as the primary model, both handling class imbalance directly. Evaluation covers ROC-AUC, PR-AUC, Brier score, and calibration. SHAP's TreeExplainer provides both global feature importance and per-patient explanations, and each patient gets bucketed into a Low / Moderate / High risk tier for care coordinator triage.

### 5. Fairness audit (`src/fairness.py`)

Performance broken out by race/ethnicity, preferred language, insurance type, and Service Planning Area — true positive rate, false positive rate, precision, ROC-AUC, demographic parity difference, and equalized odds difference. This is meant to align with LA County's Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan.

---

## A few decisions worth explaining

**Why XGBoost instead of a neural net?** This is tabular clinical data with a fairly small number of structured features — gradient-boosted trees handle that better than deep learning does, and XGBoost supports exact SHAP computation, which matters a lot when a clinical team needs to understand *why* a model flagged someone. A model nobody can explain doesn't really belong in this kind of workflow.

**Why Safe Harbor instead of Expert Determination?** Safe Harbor is what LA County and its data partners (InfoHub, CalAIM, IBHIS) already use. It's deterministic and auditable, and it doesn't require a statistician to sign off on every new data pull.

**Why keep the fairness audit separate from the SHAP analysis, instead of folding them together?** Because a model can produce SHAP explanations that look perfectly reasonable while still performing worse for certain subgroups. Keeping the equity audit as its own independent step means it can't quietly get absorbed into the explainability story.

---

## HIPAA and data governance

The full governance write-up lives in [`HIPAA_GOVERNANCE.md`](./HIPAA_GOVERNANCE.md) — data classification, role-based access control, the PHI identifier treatment log, model risk classification, drift monitoring, bias risk assessment, and incident response.

---

## How this connects to LA County's actual priorities

| LA County initiative | How this project relates |
|---|---|
| FUM Performance Improvement Project (QAPI 2025–2026) | Directly models this HEDIS measure |
| Homelessness Prevention Unit predictive analytics | Same multi-factor risk stratification approach |
| IBHIS / EHR data warehouse redesign | Feature engineering built around EHR-structured data |
| CalAIM / Medi-Cal managed care integration | Insurance type and care-gap features reflect the Medi-Cal population |
| ARDI Strategic Plan | Full fairness audit across race, language, and SPA |
| GIS / spatial equity analysis | `spa_region` and `distance_to_provider_miles` features |
| Provider Language Capacity report | `preferred_language` and limited-English-proficiency features |

---

## Stack

Python 3.11, pandas/numpy for data work, scikit-learn for the baseline model and evaluation, XGBoost as the primary classifier, SHAP for explainability, fairlearn-style metrics for the equity audit, Streamlit for the dashboard, Plotly for the interactive charts, and Faker for generating the synthetic PHI-equivalent fields.

---

## A note on why I built it this way

I wanted this to actually demonstrate HIPAA-aware healthcare data science, not just a generic classification demo with a health-sounding name attached to it. Every choice here — which HEDIS measure to model, which de-identification method to use, building the fairness audit around ARDI specifically — was grounded in publicly available LA County documentation, and is meant to mirror how the department's data science section would actually approach this problem.
