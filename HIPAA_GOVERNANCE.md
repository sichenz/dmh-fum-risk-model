# HIPAA Data Governance Documentation

**Project:** FUM Follow-Up Failure Risk Stratification Model
**Organization:** LA County Department of Mental Health (LACDMH) — Prototype
**Data Classification:** Synthetic / De-identified
**Last Updated:** 2026-08-18
**Author:** Sichen Zhong

---

## 1. Purpose and scope

This document lays out the data governance policies, HIPAA compliance controls, and model oversight approach I'd want in place for the FUM (Follow-Up After Emergency Department Visit for Mental Illness) risk stratification model.

To be clear up front: this is a **prototype** built entirely on synthetic data. Nothing here has touched a real patient record. I wrote it this way anyway because I wanted to show how I'd actually think through governance before a model like this ever got near production — not bolt it on afterward.

> **HIPAA applicability:** If this were deployed with real LACDMH patient data, using it this way would fall under Quality Assessment and Performance Improvement (QAPI) activities, which 45 CFR §164.501 treats as a permitted Healthcare Operations use — meaning it wouldn't require individual patient authorization, as long as the right safeguards are in place.

---

## 2. Data classification

Here's how I'd tier the different outputs the pipeline produces, from most to least sensitive:

| Layer | Path | Classification | Description |
|---|---|---|---|
| Raw synthetic EHR | `data/raw/` | **RESTRICTED** | Contains synthetic PHI-equivalent fields. Treated as if it were real PHI, purely for practice. Access limited to pipeline runtime only. |
| De-identified | `data/processed/patients_deidentified.csv` | **INTERNAL — SENSITIVE** | Run through HIPAA Safe Harbor. Fine for analyst access under an IRB and DUA. |
| Risk scores | `data/processed/patient_risk_scores.csv` | **INTERNAL — SENSITIVE** | Patient tokens plus risk scores. Needs the same controls as the de-identified data. |
| Aggregate reports | `reports/` | **INTERNAL** | Aggregated statistics only, nothing patient-level. |
| Models | `models/` | **CONFIDENTIAL** | Model files can encode population-level statistics, so access stays limited to authorized data scientists. |

---

## 3. HIPAA Safe Harbor de-identification — 45 CFR §164.514(b)(2)

The de-identification step in `src/deidentification.py` walks through all 18 identifier categories in the Safe Harbor method and either removes or transforms each one that shows up in this dataset.

### 3.1 What happens to each identifier

| # | PHI category | Field in data | Treatment | Rule citation |
|---|---|---|---|---|
| 1 | Names | `first_name`, `last_name` | **Removed** | §164.514(b)(2)(i)(A) |
| 2 | Geographic subdivisions smaller than state | `street_address`, `zip_code` | Street address **removed**; ZIP truncated to first 3 digits | §164.514(b)(2)(i)(B) |
| 3 | Dates (other than year) | `date_of_birth`, `ed_visit_date` | Year kept, month/day **removed** | §164.514(b)(2)(i)(C) |
| 4 | Phone numbers | `phone_number` | **Removed** | §164.514(b)(2)(i)(D) |
| 5 | Fax numbers | not present in this dataset | — | §164.514(b)(2)(i)(E) |
| 6 | Email addresses | not present in this dataset | — | §164.514(b)(2)(i)(F) |
| 7 | Social security numbers | `ssn_last4` | **Removed** | §164.514(b)(2)(i)(G) |
| 8 | Medical record numbers | `mrn` | **Pseudonymized** → SHA-256 token | §164.514(b)(2)(i)(H) |
| 9 | Health plan beneficiary numbers | not present | — | §164.514(b)(2)(i)(I) |
| 10 | Account numbers | not present | — | §164.514(b)(2)(i)(J) |
| 11–18 | Remaining identifier categories | not present | — | §164.514(b)(2)(i)(K–R) |
| Special case | Ages 90 and up | `age` | **Collapsed** into a single "90+" band | §164.514(b)(2)(i)(C) |

### 3.2 A note on pseudonymizing the MRN

I chose to pseudonymize the MRN rather than drop it outright, because you need *some* stable key to link records together for longitudinal analysis — otherwise you can't track a patient across visits at all. It's a one-way SHA-256 hash with a project-specific salt, which means:

- You can't recover the original MRN from the token. The hash only runs one direction.
- The same MRN always produces the same token, so patient-level joins still work.
- The salt itself needs to live somewhere secure and should **never** end up in version control in a real deployment.

> ⚠️ **Worth remembering in production:** pseudonymized data is still PHI under HIPAA — it's not the same as fully de-identified data, even though it's been through a one-way hash. It's fine for internal analytics, but it should never go out the door for public release.

---

## 4. Access control

### 4.1 Who should be able to see what

| Role | Raw data | De-identified | Risk scores | Models | Reports |
|---|---|---|---|---|---|
| Data pipeline (automated) | ✅ Read/Write | ✅ Write | ✅ Write | ✅ Write | ✅ Write |
| Data scientist (approved) | ❌ | ✅ Read | ✅ Read | ✅ Read/Write | ✅ Read/Write |
| Clinical analyst | ❌ | ✅ Read (aggregated) | ✅ Read (token-level) | ❌ | ✅ Read |
| Care coordinator | ❌ | ❌ | ✅ Read (dashboard, token only) | ❌ | ✅ Read |
| Program manager | ❌ | ❌ | ❌ | ❌ | ✅ Read |

The general principle: nobody outside the automated pipeline itself ever needs to touch the raw data, and most people should only be interacting with tokens, not anything that could re-identify a patient directly.

### 4.2 Technical controls a real deployment would need

None of these are implemented in this prototype — they're a checklist of what I'd expect before this touched real patients:

- [ ] Data encrypted at rest (AES-256)
- [ ] Every data access logged to an immutable audit trail
- [ ] Network-level controls — VPN required for data access
- [ ] Row-level security at the database layer for multi-department access
- [ ] Model inference API sitting behind an authentication layer
- [ ] Dashboard gated behind LACDMH SSO

---

## 5. Model governance and monitoring

### 5.1 How risky is this model, really?

I'd classify this as **Moderate Risk** rather than low or high, for a few reasons:

- Its output feeds directly into a clinical workflow decision — who gets prioritized for outreach.
- Getting it wrong means a high-risk patient might not get the outreach they needed.
- It's working with sensitive behavioral health data the whole way through.

It's not a black-box model making autonomous clinical calls, but it's also not a low-stakes internal reporting tool — it sits in the middle, and the governance should reflect that.

### 5.2 What I'd want checked before this goes live

- [ ] **Fairness audit** passes across all demographic groups (`src/fairness.py`)
- [ ] **Clinical validation** from LACDMH clinical leadership
- [ ] **IRB determination** — the QAPI exception probably applies, but that's not a call I'd make unilaterally
- [ ] **Privacy Officer sign-off** on the de-identification approach
- [ ] **IT Security review**
- [ ] **Data Use Agreement** in place if this pulls in data from other departments

### 5.3 Keeping an eye on drift after deployment

Models don't stay accurate forever, especially ones trained on population data that shifts over time. Here's what I'd track and how often:

| Metric | How often | When to act | What to do |
|---|---|---|---|
| ROC-AUC | Monthly | Drops more than 0.05 from baseline | Retrain |
| Predicted positive rate | Weekly | Drifts more than 20% from baseline | Look into the data pipeline |
| Demographic parity gap | Quarterly | Any subgroup gap exceeds 0.10 | Fairness review, possible retrain |
| Calibration error | Quarterly | Brier score rises more than 0.02 | Recalibrate |
| Feature distribution shift | Monthly | KS statistic over 0.15 on key features | Data quality review |

### 5.4 When to pull the plug

A model like this shouldn't run forever unquestioned. I'd retire or replace it if:

- AUC drops below 0.65 and retraining doesn't bring it back
- The FUM measure definition itself changes at the regulatory level
- The underlying patient population shifts substantially — a new Medi-Cal policy, for instance
- The fairness audit turns up bias that retraining and feature changes can't fix

---

## 6. Bias risk assessment

### 6.1 Where bias could realistically creep in

| Bias type | What it looks like here | How I'd mitigate it |
|---|---|---|
| Historical bias | Existing disparities in care access get baked into the training data | Fairness audit, ongoing TPR parity monitoring |
| Measurement bias | Documented care gaps for non-English speakers could show up as feature noise rather than signal | Explicit LEP flag as a feature; watch subgroup performance directly |
| Label bias | "Follow-up failure" might sometimes reflect a documentation gap rather than an actual no-show | Clinical review of how the label is actually generated |
| Feedback loop risk | If high-risk flags drive outreach, future training data reflects the intervention, not the true baseline | Track outcomes separately for patients who did and didn't receive outreach |

None of these are hypothetical concerns unique to this project — they're the standard failure modes for any risk model built on top of real-world care data, and I'd expect all of them to need active monitoring, not just a one-time check at launch.

### 6.2 Fairness thresholds

Based on the broader fairness literature and LACDMH's own ARDI commitments, here's where I'd set the bar:

| Metric | Acceptable threshold |
|---|---|
| Demographic Parity Difference | < 0.10 |
| Equalized Odds Difference | < 0.10 |
| AUC disparity across subgroups | < 0.05 |

Worth noting: the synthetic data in this prototype is intentionally built with some demographic disparities baked in (so the fairness audit actually has something real to catch), and the current metrics don't clear these thresholds. That's expected for a demo dataset — the audit is doing its job by flagging it.

---

## 7. Incident response

If a breach or unauthorized access ever happened, here's the sequence I'd follow:

1. Notify the LACDMH Privacy Officer and IT Security **immediately** — not after triage, not after confirming severity.
2. Document what happened: the nature, scope, and timing of the incident.
3. If real PHI turns out to be involved, this triggers the HIPAA Breach Notification Rule (45 CFR §164.400):
   - Affected individuals notified within 60 days
   - HHS notified annually, or immediately if more than 500 people are affected
   - Media notification if more than 500 people in a single jurisdiction are affected
4. Suspend access to the model while the investigation is underway.
5. Do a root cause analysis and fix the underlying issue before anything gets redeployed.

---

## 8. Change control

Any change to the model, the data pipeline, or the de-identification logic should go through the same basic discipline:

- Tracked with version history in Git
- Reviewed by the Data Scientist Supervisor before merging
- Re-checked for fairness impact if it touches feature engineering
- Signed off by the Privacy Officer if it touches de-identification logic specifically

---

## 9. References

- HHS. *Guidance Regarding Methods for De-identification of PHI in Accordance with HIPAA.* (2012). https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- NCQA. *HEDIS FUM Measure Specification.* (2025).
- LACDMH. *QAPI Work Plan 2025-2026.* LA County Department of Mental Health.
- LACDMH. *Anti-Racism, Diversity, and Inclusion (ARDI) Strategic Plan.* LA County Department of Mental Health.
- Fairlearn. *Fairness Assessment for Machine Learning Systems.* https://fairlearn.org/
- Bird et al. *Fairlearn: A toolkit for assessing and improving fairness in AI.* (2020). Microsoft Research.
