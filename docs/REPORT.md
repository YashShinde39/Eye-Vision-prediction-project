# Vision Risk Prediction — Report (Concise)

Abstract
We predict vision risk from device usage, ergonomics, and symptoms. We compare Logistic Regression (baseline) and XGBoost (improved) under stratified cross-validation, emphasizing PR-AUC, calibration, and subgroup fairness.

Data
- Self-reported features: usage (daily_hours, session_length, breaks), ergonomics (viewing_distance, screen_height, lighting), health (sleep_quality, headache/eyestrain), demographics, device_type.
- Labels: proxy severity from heuristics when clinical labels are absent (exploratory).

Methods
- Preprocessing: cleaning, ordinal/one-hot encoding, engineered features (sessions_per_day, breaks_per_hour, long_session, short_distance, outdoor_ratio).
- Models: Logistic Regression (L2, class_weight=balanced) and XGBoost (shallow trees, regularization, class imbalance handling). Post-hoc calibration.
- Validation: stratified K-fold; OOF metrics; bootstrapped CIs.

Results (example placeholders)
- Binary:
  - PR-AUC: Baseline 0.X (CI), XGB 0.Y (CI)
  - F1/Recall at P≥0.7: ...
  - Brier (calibration): ...
- Multi-class: Macro F1, confusion matrix patterns
- Subgroups: gaps in recall ≤ 0.15 across age/gender/device_type
- Visuals: see reports/run_*/ for PR curve, calibration, confusion matrix, SHAP

Interpretation
- Drivers: screen time, long sessions, low breaks, short viewing distance, high symptom frequency. SHAP aligns with ergonomics guidance.

Limitations
- Small n, proxy labels, self-report biases; results are exploratory. See docs/LIMITATIONS_AND_ETHICS.md.

Ethics
- Non-diagnostic; fairness checks; privacy-aware processing; transparent assumptions.

Future work
- Replace proxy labels with validated outcomes; expand/stratify dataset; monotonic constraints; group-aware calibration; deploy A/B for recommendation efficacy.
