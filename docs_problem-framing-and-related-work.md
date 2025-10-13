# Problem Framing and Related Work — Vision Status Prediction from Device Usage

## 1) Summary
We aim to predict vision status from self-reported device usage, ergonomics, and health indicators. The project explores whether everyday behavior signals (screen time, session length, viewing distance, lighting, sleep quality, etc.) are predictive of digital eye strain and longer-term risk of vision problems. Two complementary tasks are considered:
- Binary classification: Normal vs. Impaired
- Multi-class classification: Normal, Mild, Moderate, Severe

Current labels are proxy labels derived from risk heuristics; the project should be treated as an exploratory feasibility study until clinically validated labels are available.

Repo context:
- Data: df_new.csv (n=81, 18 columns)
- Notebook: HeavyVersion.ipynb
- Baselines: Logistic Regression; Main model: XGBoost
- Explainability: SHAP, partial dependence

---

## 2) Business and User Problem
- Who benefits: Students, knowledge workers, and organizations aiming to reduce digital eye strain, improve ergonomics, and promote proactive eye care.
- Why now: Screen exposure continues to grow; early detection and behavior feedback could reduce discomfort, improve productivity, and contribute to longer-term eye health.
- Outcome: A risk score and interpretable recommendations (e.g., increase breaks, adjust viewing distance, improve lighting, increase time outdoors).

---

## 3) Prediction Tasks
- Task A (binary): Predict whether a person is likely “impaired” vs “normal” (impaired includes mild–severe risk).
- Task B (multi-class): Predict severity: normal, mild, moderate, severe.

These tasks support:
- Screening and prioritization (binary)
- Tailored advice and triage intensity (multi-class)

---

## 4) Features and Signals
From df_new.csv:
- Device usage: daily_hours, session_length, breaks, dark_mode, font_size, brightness
- Ergonomics: viewing_distance, screen_height, lighting
- Behavior/health: sleep_quality, headache_freq, eyestrain_freq, outdoor_time
- Demographics: age, gender
- Device type: device_type (mobile, laptop, tablet, tv)

Notes:
- “vision_label” is empty in the raw data; current labels are engineered in the notebook via a risk scoring function.
- Class imbalance is likely (e.g., derived “normal” small vs. “impaired” large).

---

## 5) Labels and Ground Truth
- Current approach: Weak/proxy labels via a hand-crafted risk score combining screen time, session length without breaks, poor viewing distance, and health indicators.
- Implications:
  - Labels reflect assumptions from literature and ergonomics guidelines but are not clinical ground truth.
  - Treat results as hypothesis-generating, not diagnostic.
- Near-term improvement plan:
  - Collect self-reported outcomes (e.g., validated Computer Vision Syndrome questionnaire scores).
  - If feasible, include optometrist/ophthalmologist screening results or visual acuity/refraction measures.
  - Sensitivity analysis of the heuristic weights and thresholds; report robustness.

---

## 6) Constraints and Risks
- Small sample size (n=81): risk of overfitting; prefer simple models, strong regularization, and conservative claims.
- Label noise from proxy labels: emphasize uncertainty and avoid clinical claims.
- Class imbalance: select metrics robust to skew (PR-AUC, F1, class-weighting).
- Self-report bias in features (e.g., screen time, symptoms).
- Generalizability: Data from a limited population; may not transfer to broader demographics.

---

## 7) Modeling Approach
- Baseline: Logistic Regression (interpretable, strong regularization)
- Main model: XGBoost (handles mixed features, nonlinearity, interactions)
- Preprocessing:
  - One-hot encode categorical features
  - Standardize numeric features for linear models
  - Consider monotonic constraints in XGBoost for features with known directionality (optional)
- Validation:
  - Stratified train/validation splits
  - Stratified K-fold cross-validation given small n
  - Bootstrapped confidence intervals for key metrics

---

## 8) Evaluation Plan
Given imbalance and proxy labels:
- Binary:
  - Primary: PR-AUC (positive class: impaired), F1 (macro or binary for impaired), recall@fixed-precision
  - Secondary: ROC-AUC, balanced accuracy, calibration (Brier score; reliability curves)
- Multi-class:
  - Macro F1, macro recall, confusion matrix
  - Class-wise precision/recall to ensure smaller classes are not ignored
- Report:
  - Cross-validated metrics with mean ± std or 95% CI
  - Calibration plots, confusion matrices
  - SHAP summaries and partial dependence for interpretability

---

## 9) Interpretability and Recommendations
- SHAP values to rank important features (e.g., daily_hours, session_length, breaks, viewing_distance, sleep_quality).
- Partial dependence/ICE to visualize effect of single features.
- Translate top features into actionable guidance:
  - Increase micro-breaks and the 20-20-20 rule
  - Adjust viewing distance and screen height
  - Improve ambient lighting; reduce glare
  - Moderate brightness; consider dark mode contexts
  - Increase outdoor time where appropriate
  - Address sleep hygiene (indirectly linked via fatigue and symptom perception)

---

## 10) Ethics, Fairness, and Safety
- Non-clinical: Communicate that the tool is not a diagnosis. Encourage professional evaluation for symptoms.
- Demographic fairness:
  - Monitor for differential performance across age and gender.
  - Avoid reinforcing stereotypes; ensure recommendations are universally accessible.
- Privacy:
  - Anonymize data; minimize PII.
- Transparency:
  - Document assumptions in label construction and modeling choices.
- Risk of harm:
  - Provide conservative, general wellness guidance; include “when to seek care” disclaimers.

---

## 11) Deployment Considerations
- Intended use: Educational and wellness support; early signal for eye strain risk.
- Feedback loop: Provide simple dashboards and tips; collect follow-up outcomes to refine labels and model.
- Model monitoring:
  - Data drift (usage patterns change with season/school/workload)
  - Performance drift (retrain schedule)
  - Calibration maintenance

---

## 12) Success Criteria
- Technical:
  - PR-AUC and macro F1 substantially above baselines in cross-validation
  - Stable performance across folds; well-calibrated probabilities
- Product:
  - Users find recommendations understandable and actionable
  - Measurable improvement in self-reported symptoms over time (A/B or pre-post)
- Scientific:
  - Alignment with known risk factors (face-valid SHAP/PDP)
  - Robustness to label heuristic variations

---

## 13) Related Work (Conceptual Overview)
This project stands on several established lines of evidence and practice:

- Digital Eye Strain (Computer Vision Syndrome)
  - Prolonged near work and suboptimal ergonomics are associated with ocular discomfort: dryness, headaches, blurred vision, and fatigue.
  - Ergonomic factors such as viewing distance, screen height, lighting, glare, and break frequency are modifiable risk factors highlighted in clinical guidance.

- Near Work, Screen Time, and Myopia Risk
  - Observational and meta-analytic evidence links intensive near work with higher myopia prevalence/progression in youth.
  - Mechanisms are multifactorial; sustained accommodation and short viewing distances are frequently discussed.

- Time Outdoors as a Protective Factor
  - Multiple studies report an association between increased outdoor time and reduced myopia onset/progression in children and adolescents, potentially via higher ambient light exposure.

- Sleep and Visual Comfort
  - Evening screen exposure and high brightness can disrupt sleep; poorer sleep quality is associated with worsened symptom perception and daytime fatigue, indirectly affecting visual comfort.

- Visual Ergonomics
  - Occupational health and optometry literature recommend:
    - Appropriate viewing distance (e.g., ~50–70 cm for desktop use)
    - Screen at or slightly below eye level
    - Ambient lighting that reduces glare; appropriate contrast/brightness
    - Regular micro-breaks (e.g., 20-20-20 rule)

- ML in Vision Health
  - While much ML work targets imaging (e.g., diabetic retinopathy screening), behavior-based risk modeling is emerging for wellness screening and personalized recommendations.
  - Interpretability (e.g., SHAP) is crucial for trustworthy health-facing tools.

Notes on citing:
- For a formal write-up, consider citing exemplar sources such as:
  - Reviews on digital eye strain and visual ergonomics (e.g., peer-reviewed reviews in ophthalmology/optometry journals).
  - Meta-analyses on near work and myopia risk in children/adolescents.
  - Studies on outdoor time and myopia onset/progression.
  - Clinical/association guidance (e.g., optometry associations) on computer vision syndrome and ergonomic recommendations.

---

## 14) Next Steps
- Data:
  - Add validated symptom scales (e.g., CVS questionnaires) or screening outcomes to replace proxy labels.
  - Expand sample size and diversity; ensure balanced representation.
- Modeling:
  - Systematic sensitivity analysis of the current heuristic label weights.
  - Compare additional models (regularized linear models, calibrated gradient boosting).
  - Use stratified K-fold CV with bootstrapped CIs; add PR curves.
- Product:
  - Build an insights report translating model outputs into ergonomic tips.
  - Add “seek care” criteria when severe symptoms are reported.

---

## 15) Glossary
- Digital Eye Strain (Computer Vision Syndrome): Cluster of symptoms from prolonged screen use and suboptimal ergonomics.
- Calibration: Agreement between predicted probabilities and observed frequencies.
- PR-AUC: Area under precision–recall curve; robust to class imbalance.

```References placeholders for your report (fill with exact citations you select)
- Review: Digital eye strain prevalence, measurement, and mitigation.
- Meta-analysis: Near work and myopia in children.
- Observational studies: Outdoor time as a protective factor for myopia.
- Clinical guidance: Computer Vision Syndrome (optometry association).
```