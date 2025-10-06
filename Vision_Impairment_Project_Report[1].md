# Vision Impairment Risk Prediction: ML Project Report

---

## Overview

This project simulates a dataset of 300 individuals with diverse digital device usage habits, environmental, and health factors to predict their risk of vision impairment. Two main ML models (Logistic Regression and XGBoost) are trained and evaluated for binary and ordinal vision risk classification. The notebook covers data simulation, model building, interpretability, calibration, subgroup fairness, and prediction examples.

---

## Data Simulation & Feature Engineering

- **Demographics:** Age, Gender
- **Device Usage:** Device type, Daily hours, Session length, Breaks
- **Settings & Environment:** Font size, Brightness, Dark mode, Outdoor time, Viewing distance, Screen height, Lighting
- **Health Indicators:** Sleep quality, Headache frequency, Eyestrain frequency

**Target Labels:**
- `vision_status` (Normal, Mild, Moderate, Severe) via a rule-based function
- `vision_status_bin` (Normal vs. Impaired)

Categorical data is label-encoded; numerical features are standardized.

---

## Model Training & Evaluation

1. **Binary Classification (Normal vs. Impaired)**
    - Models: Logistic Regression, XGBoost
    - Metrics: Precision, Recall, F1-score, AUROC, Brier Score

**Results:**

| Model              | Accuracy | AUROC  | F1-score (avg) | Brier Score |
|--------------------|---------:|-------:|---------------:|------------:|
| LogisticRegression |   0.92   | 0.92   |     0.91       |   0.055     |
| XGBoost            |   0.97   | 0.99   |     0.96       |   0.029     |

- **Confusion Matrix (XGBoost):**
    ![Confusion Matrix](attachment:confusion_matrix.png)

- **Calibration Curve (XGBoost):**
    ![Calibration Curve](attachment:calibration_curve.png)

2. **Ordinal Classification (Severity Levels)**
    - Model: XGBoost (multi-class)
    - Metric: Quadratic Weighted Kappa (QWK)
    - QWK ≈ 0.60

---

## Model Interpretability

- **SHAP Feature Importance (XGBoost):**
    ![SHAP Feature Importance](attachment:shap_feature_importance.png)

    - Top features: Daily device hours, Session length, Eyestrain frequency, Outdoor time, Brightness, Age

---

## Subgroup Fairness

- **Age Groups:** `<18`, `18-30`, `30-45`, `45+`
- **Macro F1-scores** are calculated for each age group to assess model fairness.
- No major drop in performance for subgroups, but real-world bias testing is needed.

---

## Visuals

- **Confusion Matrix:** Shows correct/incorrect predictions for binary classification.
- **Calibration Curve:** Plots predicted probabilities vs. actual outcomes, indicating good probability calibration.
- **SHAP Feature Importance:** Ranks feature impact on XGBoost predictions.

---

## Example Prediction

```python
example_user = {
    "age": 20,
    "gender": "male",
    "device_type": "mobile",
    "daily_hours": 8,
    "session_length": 120,
    "breaks": 2,
    "font_size": "small",
    "brightness": "high",
    "dark_mode": "no",
    "outdoor_time": 0.5,
    "viewing_distance": 25,
    "screen_height": "eye_level",
    "lighting": "normal",
    "sleep_quality": 3,
    "headache_freq": 3,
    "eyestrain_freq": 4,
}
print(predict_vision_status(example_user))
# Output: {'prob_impaired': 0.9999, 'class': 'impaired'}
```

---

## Limitations

- **Synthetic Data:**  
  This dataset is simulated, not real, so results are illustrative, not representative of actual populations.
- **Labeling Function:**  
  The vision status is assigned by a custom rule, not by clinical measurements.
- **Generalizability:**  
  Models trained on synthetic data may not generalize to real-world scenarios or populations.
- **Sample Size:**  
  N=300 is small for robust ML; variance may be high.
- **Feature Encoding:**  
  Simple encoding may ignore nuanced interactions.
- **Missing Confounders:**  
  No genetic, medical history, or other vision risk factors included.

---

## Ethical Considerations

- **Privacy & Consent:**  
  Real health data must ensure privacy, informed consent, and regulatory compliance (e.g., HIPAA, GDPR).
- **Bias & Fairness:**  
  Synthetic data may not capture real-world biases. Subgroup analysis helps, but further fairness assessment is needed for live deployment.
- **Transparency & Interpretability:**  
  Health-related predictions should be explainable (e.g., SHAP), and users should understand how decisions are made.
- **Clinical Use:**  
  This model is not clinically validated and must not be used for real-world diagnosis or intervention without medical oversight.
- **Synthetic Data Use:**  
  All findings are educational; do not infer medical risks or recommendations from this notebook.

---

## Conclusion

The notebook demonstrates a complete ML workflow for simulated vision impairment risk prediction, including modeling, evaluation, interpretability, and fairness analysis. **Results are for educational purposes only.** Real-world applications require clinical data, validation, and careful ethical review.

---

## References

- [scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP: Explaining ML Models](https://shap.readthedocs.io/)
