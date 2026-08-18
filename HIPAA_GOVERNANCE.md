# HIPAA Data Governance Documentation

**Project:** FUM Follow-Up Failure Risk Stratification Model  
**Organization:** LA County Department of Mental Health (LACDMH) — Portfolio Prototype  
**Data Classification:** Synthetic / De-identified  
**Last Updated:** 2026-08-18  
**Author:** Portfolio Project — Data Scientist Supervisor Candidate

---

## 1. Purpose and Scope

This document defines the data governance policies, HIPAA compliance controls, and model oversight framework for the FUM (Follow-Up After Emergency Department Visit for Mental Illness) risk stratification model.

This is a **portfolio prototype** using entirely synthetic data. It is designed to demonstrate governance-first data science methodology appropriate for production deployment of ML models involving behavioral health Protected Health Information (PHI).

> **HIPAA Applicability:** If deployed with real LACDMH patient data, this project would constitute a use of PHI for Quality Assessment and Performance Improvement (QAPI) activities, which is permissible under 45 CFR §164.501 as a Healthcare Operations activity — **without requiring patient authorization** — provided appropriate safeguards are in place.

---

## 2. Data Classification Schema

| Layer | Path | Classification | Description |
|---|---|---|---|
| Raw synthetic EHR | `data/raw/` | **RESTRICTED** | Contains synthetic PHI-equivalent fields. Treated as if it were real PHI for practice purposes. Access limited to pipeline runtime only. |
| De-identified | `data/processed/patients_deidentified.csv` | **INTERNAL — SENSITIVE** | HIPAA Safe Harbor de-identified. Appropriate for analyst access under IRB and DUA. |
| Risk scores | `data/processed/patient_risk_scores.csv` | **INTERNAL — SENSITIVE** | Contains patient tokens + risk scores. Requires same controls as de-identified data. |
| Aggregate reports | `reports/` | **INTERNAL** | Aggregated statistics. No patient-level data. |
| Models | `models/` | **CONFIDENTIAL** | ML model files. May encode population statistics; restricted to authorized data scientists. |

---

## 3. HIPAA Safe Harbor De-identification — 45 CFR §164.514(b)(2)

The de-identification pipeline in `src/deidentification.py` removes or transforms all 18 PHI identifier categories specified in the HIPAA Privacy Rule Safe Harbor method.

### 3.1 PHI Identifier Treatment Log

| # | PHI Category | Field in Data | Treatment Applied | Rule Citation |
|---|---|---|---|---|
| 1 | Names | `first_name`, `last_name` | **REMOVED** | §164.514(b)(2)(i)(A) |
| 2 | Geographic subdivisions smaller than state | `street_address`, `zip_code` | Street address **REMOVED**; ZIP → first 3 digits | §164.514(b)(2)(i)(B) |
| 3 | Dates (except year) | `date_of_birth`, `ed_visit_date` | Year retained; month/day **REMOVED** | §164.514(b)(2)(i)(C) |
| 4 | Phone numbers | `phone_number` | **REMOVED** | §164.514(b)(2)(i)(D) |
| 5 | Fax numbers | N/A in dataset | N/A | §164.514(b)(2)(i)(E) |
| 6 | Email addresses | N/A in dataset | N/A | §164.514(b)(2)(i)(F) |
| 7 | Social security numbers | `ssn_last4` | **REMOVED** | §164.514(b)(2)(i)(G) |
| 8 | Medical record numbers | `mrn` | **PSEUDONYMIZED** → SHA-256 token | §164.514(b)(2)(i)(H) |
| 9 | Health plan beneficiary numbers | N/A | N/A | §164.514(b)(2)(i)(I) |
| 10 | Account numbers | N/A | N/A | §164.514(b)(2)(i)(J) |
| 11–18 | Other identifiers | N/A | N/A | §164.514(b)(2)(i)(K–R) |
| Special | Ages ≥ 90 | `age` | **Collapsed** → "90+" age band | §164.514(b)(2)(i)(C) |

### 3.2 Pseudonymization Note

The MRN is pseudonymized (not removed) to preserve within-dataset linkage ability for longitudinal analysis. The one-way SHA-256 hash with a project-specific salt means:
- The original MRN **cannot** be recovered from the token
- Two records with the same MRN will produce the same token (enabling patient-level joins)
- The salt must be stored securely and **not** committed to version control in production

> ⚠️ **Production Note:** Pseudonymized data is **not** fully de-identified under HIPAA. It remains PHI and requires the same access controls as the original data. It is appropriate for internal analytics but not public release.

---

## 4. Access Control Design

### 4.1 Role-Based Access Matrix

| Role | Raw Data | De-identified | Risk Scores | Models | Reports |
|---|---|---|---|---|---|
| Data Pipeline (automated) | ✅ Read/Write | ✅ Write | ✅ Write | ✅ Write | ✅ Write |
| Data Scientist (approved) | ❌ | ✅ Read | ✅ Read | ✅ Read/Write | ✅ Read/Write |
| Clinical Analyst | ❌ | ✅ Read (aggregated) | ✅ Read (token-level) | ❌ | ✅ Read |
| Care Coordinator | ❌ | ❌ | ✅ Read (via dashboard — token only) | ❌ | ✅ Read |
| Program Manager | ❌ | ❌ | ❌ | ❌ | ✅ Read |

### 4.2 Technical Controls (Production Requirements)

- [ ] Data stored in encrypted storage (AES-256 at rest)
- [ ] All data access logged to immutable audit trail
- [ ] Network-level access controls (VPN required for data access)
- [ ] Database-level row-level security for multi-department access
- [ ] Model inference API behind authentication layer
- [ ] Dashboard requires LACDMH SSO authentication

---

## 5. Model Governance and Fidelity Monitoring

### 5.1 Model Risk Classification

This model is classified as **Moderate Risk** under AI governance frameworks because:
- Output is used for clinical workflow decisions (outreach prioritization)
- Errors may result in missed outreach for high-risk patients
- Model operates on sensitive behavioral health data

### 5.2 Pre-Deployment Requirements

- [ ] **Fairness audit** passed for all demographic groups (see `src/fairness.py`)
- [ ] **Clinical validation** by LACDMH clinical leadership
- [ ] **IRB determination** (QAPI exception likely applicable, but requires review)
- [ ] **Privacy Officer sign-off** on de-identification methodology
- [ ] **IT Security review** and approval
- [ ] **Data Use Agreement** if integrating cross-departmental data

### 5.3 Model Drift Monitoring Plan

After deployment, the following metrics must be monitored quarterly:

| Metric | Monitoring Frequency | Action Threshold | Response |
|---|---|---|---|
| ROC-AUC | Monthly | Drop > 0.05 from baseline | Retrain model |
| Positive rate (predicted) | Weekly | Drift > 20% from baseline | Investigate data pipeline |
| Demographic parity gap | Quarterly | Any subgroup gap > 0.10 | Fairness review + possible retraining |
| Calibration error | Quarterly | Brier score increase > 0.02 | Recalibrate model |
| Feature distribution shift | Monthly | KS statistic > 0.15 for key features | Data quality review |

### 5.4 Model Retirement Criteria

The model must be retired or replaced if:
- AUC falls below 0.65 and cannot be restored by retraining
- Regulatory changes alter the FUM measure definition
- Patient population characteristics shift substantially (e.g., new Medi-Cal policy)
- Fairness audit reveals persistent, uncorrectable bias

---

## 6. Bias Risk Assessment

### 6.1 Known Sources of Bias in Training Data

| Bias Type | Description | Mitigation |
|---|---|---|
| Historical bias | Systemic disparities in care access are encoded in training data | Fairness audit; TPR parity monitoring |
| Measurement bias | Documented care gaps for non-English speakers may cause feature noise | LEP flag as explicit feature; subgroup performance monitoring |
| Label bias | "Follow-up failure" may partly reflect documentation gaps vs. actual no-shows | Clinical validation of labeling logic |
| Feedback loop risk | If high-risk flags drive outreach, future data reflects intervention not baseline | Monitor separately for patients who received outreach |

### 6.2 Fairness Thresholds

Following established fairness literature and LACDMH's ARDI commitment:

| Metric | Acceptable Threshold |
|---|---|
| Demographic Parity Difference | < 0.10 |
| Equalized Odds Difference | < 0.10 |
| AUC Disparity across subgroups | < 0.05 |

---

## 7. Incident Response

If a data breach or unauthorized access is detected:

1. **Immediately** notify the LACDMH Privacy Officer and IT Security
2. Document the nature, scope, and timing of the incident
3. If real PHI is involved, initiate HIPAA Breach Notification Rule (45 CFR §164.400) process:
   - Affected individuals notified within 60 days
   - HHS notified annually (or immediately if > 500 individuals affected)
   - Media notification if > 500 individuals in a jurisdiction affected
4. Suspend model access pending investigation
5. Root cause analysis and remediation before re-deployment

---

## 8. Change Control

All changes to the model, data pipeline, or de-identification logic must be:
- Documented with version history in Git
- Reviewed by the Data Scientist Supervisor before merging
- Re-evaluated for fairness impact if feature engineering changes
- Approved by Privacy Officer if de-identification logic changes

---

## 9. References

- HHS. *Guidance Regarding Methods for De-identification of PHI in Accordance with HIPAA.* (2012). https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- NCQA. *HEDIS FUM Measure Specification.* (2025).
- LACDMH. *QAPI Work Plan 2025-2026.* LA County Department of Mental Health.
- LACDMH. *Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan.* LA County Department of Mental Health.
- Fairlearn. *Fairness Assessment for Machine Learning Systems.* https://fairlearn.org/
- Bird et al. *Fairlearn: A toolkit for assessing and improving fairness in AI.* (2020). Microsoft Research.
