# Vision Status Prediction — Problem Framing (Short)

Goal
- Predict vision status from device usage, ergonomics, and health signals to surface risk and actionable tips.

Tasks
- Binary: normal vs impaired
- Multi-class: normal, mild, moderate, severe

Data (df_new.csv; n=81)
- Usage: daily_hours, session_length, breaks, dark_mode, brightness, font_size
- Ergonomics: viewing_distance, screen_height, lighting
- Health/behavior: sleep_quality, headache_freq, eyestrain_freq, outdoor_time
- Demographics: age, gender; device_type
- Note: vision_label is empty; labels are proxy heuristics in the notebook.

Modeling
- Baseline: Logistic Regression
- Main: XGBoost
- Preprocessing: one-hot for categoricals; scaling for linear models
- Validation: stratified K-fold; report mean ± std

Metrics
- Binary: PR-AUC (impaired), F1, ROC-AUC, calibration (Brier)
- Multi-class: macro F1, macro recall, confusion matrix

Key Risks
- Small sample → overfitting
- Proxy labels → label noise; non-clinical
- Class imbalance; self-report bias; limited generalizability

Interpretability → Actions
- SHAP/PDP to explain drivers (e.g., screen time, long sessions, few breaks, short distance)
- Tips: 20-20-20 breaks, 50–70 cm viewing distance, screen slightly below eye level, reduce glare, moderate brightness, increase outdoor time, improve sleep hygiene

Ethics & Safety
- Not diagnostic; include “seek care” guidance
- Monitor fairness across age/gender
- Protect privacy; document assumptions

Related Work (brief pointers)
- Digital eye strain/computer vision syndrome (ergonomics, breaks, lighting)
- Near work and myopia risk; outdoor time as protective factor
- Importance of interpretability for health-facing ML

Next Steps
- Replace proxy labels with validated symptom scales or clinical screens
- Grow and balance the dataset
- Sensitivity analysis of label heuristics; bootstrapped CIs
- Build a simple insights report with personalized recommendations