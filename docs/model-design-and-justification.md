# Model Design & Justification (Baseline + Improved, Confounder Handling)

Objectives and constraints
- Predict vision status (binary + severity) from device usage, ergonomics, and symptoms.
- Very small n (~81), noisy proxy labels → favor simple, regularized, interpretable models; strong validation discipline.
- Class imbalance (few “normal”) → use class weighting, PR-focused metrics, threshold tuning, and calibration.

Baseline model: Regularized Logistic Regression
- Why
  - Strong bias-variance tradeoff with L2; stable on small datasets.
  - Interpretable coefficients; easy to calibrate; fast nested CV.
- Setup
  - Features: numeric + engineered + ordinal codes; one-hot for nominal; scale numeric.
  - Class imbalance: class_weight="balanced".
  - Hyperparams (searched via nested CV): penalty=L2, C ∈ {0.05, 0.1, 0.5, 1, 2}, solver={liblinear,saga}, max_iter=2000.
  - Calibration: CalibratedClassifierCV with sigmoid (or isotonic if enough data), fit within inner CV only.
- Metrics
  - Binary: PR-AUC (impaired), F1 (impaired), ROC-AUC, Brier score, calibration curve.
  - Multi-class: macro F1, macro recall, confusion matrix.

Improved model: Gradient Boosted Trees (XGBoost)
- Why
  - Captures nonlinearity and interactions (e.g., screen time × breaks; viewing distance × screen height).
  - Robust to mixed features; often superior recall at fixed precision on tabular data.
- Setup
  - Shallow trees to prevent overfitting on small n: max_depth ∈ {2,3,4}, learning_rate ∈ {0.03,0.05,0.1}, n_estimators with early stopping (50–500), min_child_weight ∈ {1,3,5}, subsample ∈ {0.6,0.8,1.0}, colsample_bytree ∈ {0.6,0.8,1.0}, reg_lambda ∈ {1,3,10}, reg_alpha ∈ {0,1}.
  - Class imbalance: scale_pos_weight ≈ (neg/pos) for binary; objective=“binary:logistic”. For multi-class: objective=“multi:softprob”.
  - Optional monotonic constraints to encode domain directionality (if validated): daily_hours (+), session_length (+), breaks (−), viewing_distance (− for short distances). Use cautiously; validate via sensitivity analysis.
  - Probability calibration: post-hoc Platt/isotonic on out-of-fold predictions; or XGB + CalibratedClassifierCV.
- Interpretability
  - SHAP for global/local drivers; PDP/ICE for key features (screen time, breaks, viewing distance, symptoms).
  - Sanity checks: direction of effects matches ergonomics guidance.

Validation and selection
- Use stratified nested CV (outer k=5–10; inner k=3–5) due to hyperparameter tuning on small n.
- Report mean ± std (or 95% CI via bootstrapping) for primary metrics.
- Threshold selection: pick operating point maximizing F1 or subject to recall ≥ target (e.g., ≥0.8) given wellness screening use case.
- Tie-breaker: if XGBoost PR-AUC/F1 improvements are marginal and calibration/interpretability comparable, prefer Logistic Regression (simpler, more stable).

Confounder handling
- Likely confounders: age (associated with symptoms and behavior), gender, device_type/usage context; symptoms may act as mediators for “risk” labels.
- Predictive vs causal
  - The goal is prediction, not causal attribution; however, confounding can distort perceived feature importance and group performance.
  - Current proxy labels are constructed from features → risk of label leakage/tautology. Treat results as exploratory until true outcome labels are available.
- Strategies
  - Include confounders as features (age, gender, device_type) to absorb baseline differences.
  - Fairness and slicing: report metrics by age bands (e.g., <20, 20–35, >35), gender, and device_type; monitor disparities in recall/precision.
  - Sensitivity checks:
    - Drop-confounder analysis: retrain excluding age/gender/device_type; observe performance/feature-importance shifts.
    - Leave-one-group-out CV: train on all but one demographic/device group, test on the held-out group to gauge generalization.
    - Matched subsamples: compare models on age/gender-balanced subsets; check stability of effects.
  - Target leakage control with proxy labels:
    - If labels are built from a subset of features, avoid using the exact same features/thresholds exclusively for prediction; or run a secondary model excluding the most heavily-weighted label-construction inputs to estimate robustness.
    - Prefer replacing proxy labels with validated symptom scales or clinical screens ASAP.
- Calibration per group
  - Assess and, if necessary, recalibrate probabilities separately or use group-aware calibration to minimize systematic over/underestimation.

Practical choices and rationale
- Not using deep nets: dataset too small; risk of overfitting without substantive benefit on tabular data.
- Random Forest as an additional comparator is reasonable; expect similar or slightly worse calibrated probabilities than XGBoost; include if time permits.
- Final deliverable: baseline (LogReg) + improved (XGBoost), with nested CV results, calibration plots, SHAP/PDP, group-slice metrics, and threshold rationale aligned to the screening objective.

Operational guidance
- Refit the selected model on full data with optimal hyperparameters; export pipeline (preprocessing + model + calibrator).
- Log training artifacts: encoders, scaler parameters, feature lists, class priors, threshold, and calibration method.
- Plan for periodic re-calibration and re-training as new labeled data accumulates; monitor drift and group-wise performance.