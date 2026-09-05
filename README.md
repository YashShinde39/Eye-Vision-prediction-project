# Eye Vision Prediction Project

An academic machine-learning prototype that explores whether device usage, visual ergonomics, lifestyle, and demographic data can be used to estimate vision-risk severity.

> **Important:** This project is for educational and exploratory use only. It is not a medical device, does not provide a diagnosis, and must not replace an examination by a qualified eye-care professional.

## Overview

The project is implemented in the [`EyeVisionProject.ipynb`](EyeVisionProject.ipynb) notebook. The notebook:

- Loads and inspects the supplied survey dataset.
- Encodes categorical features and standardizes model inputs.
- Creates exploratory vision-risk targets from usage and health-related heuristics.
- Trains and evaluates Logistic Regression and XGBoost classifiers.
- Supports both:
  - **Binary classification:** `normal` vs. `impaired`
  - **Ordinal classification:** `normal`, `mild`, `moderate`, and `severe`
- Produces confusion matrices, ROC and calibration plots, cross-validation metrics, SHAP feature importance, and partial-dependence/ICE plots.
- Performs subgroup and symptom-feature leakage checks.
- Defines a notebook prediction function for trying a custom user profile.

## Repository contents

| File | Description |
| --- | --- |
| [`EyeVisionProject.ipynb`](EyeVisionProject.ipynb) | Main analysis, preprocessing, modeling, evaluation, and explainability workflow |
| [`df_new.csv`](df_new.csv) | Input dataset with 81 rows and 18 columns |
| [`docs_REPORT.md`](docs_REPORT.md) | Concise project report and interpretation notes |
| [`docs_model-design-and-justification.md`](docs_model-design-and-justification.md) | Model-selection, validation, and confounder-handling rationale |
| [`docs_data-acquisition-and-preprocessing.md`](docs_data-acquisition-and-preprocessing.md) | Dataset schema, cleaning, encoding, and feature-engineering guidance |

## Dataset

The CSV contains the following fields:

- **Demographics:** `age`, `gender`
- **Device usage:** `device_type`, `daily_hours`, `session_length`, `breaks`
- **Display and ergonomics:** `font_size`, `brightness`, `dark_mode`, `viewing_distance`, `screen_height`, `lighting`
- **Lifestyle and symptoms:** `outdoor_time`, `sleep_quality`, `headache_freq`, `eyestrain_freq`, `milk_consumption_ml`
- **Recorded label:** `vision_label`

The notebook uses 16 predictors and excludes `headache_freq` and `eyestrain_freq` from its primary model features. It then adds those symptom fields in a separate leakage-analysis comparison.

### Target construction

Because the dataset does not provide a validated clinical outcome, the notebook constructs an exploratory `vision_status` target using a heuristic risk score based on:

- Daily screen hours and session length
- Number of breaks
- Outdoor time and viewing distance
- Sleep quality
- Headache and eyestrain frequency

The resulting severity levels are mapped to binary and ordinal targets. These labels are **proxy labels**, not clinical measurements.

## Getting started

### Requirements

- Python 3.8 or newer
- Jupyter Notebook or JupyterLab, or Visual Studio Code with the Jupyter extension
- The Python packages imported by the notebook:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

### Installation

```bash
git clone https://github.com/YashShinde39/Eye-Vision-prediction-project.git
cd Eye-Vision-prediction-project

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
```

### Run the notebook

```bash
jupyter notebook EyeVisionProject.ipynb
```

Then run the cells from top to bottom. The notebook expects `df_new.csv` to remain in the project root.

The main stages are:

1. Load and explore the dataset.
2. Construct binary and ordinal targets.
3. Encode categorical values and scale features.
4. Train/test split and model training.
5. Evaluation and five-fold stratified cross-validation.
6. SHAP, partial-dependence, and ICE analysis.
7. Subgroup robustness and potential leakage checks.
8. Interactive custom-profile prediction.

## Models and evaluation

The notebook compares:

- **Logistic Regression** as a simple baseline.
- **XGBoost** as a nonlinear tree-based model.

For binary classification it reports accuracy, precision, recall, F1, ROC-AUC, Brier score, and confusion matrices. For ordinal classification it reports classification metrics, a confusion matrix, macro F1, and quadratic weighted kappa.

The notebook also uses stratified five-fold cross-validation. With only 81 observations and seven `normal` proxy labels, the recorded metrics should be treated as illustrative rather than as evidence of clinical or real-world performance.

## Example input

The prediction section accepts a dictionary containing the fields expected by the notebook:

```python
custom_user = {
    "age": 30,
    "gender": "female",
    "device_type": "mobile",
    "daily_hours": 5.0,
    "session_length": 60.0,
    "breaks": 2,
    "font_size": "medium",
    "brightness": "high",
    "dark_mode": "yes",
    "outdoor_time": 1.0,
    "viewing_distance": 35,
    "screen_height": "eye_level",
    "lighting": "normal",
    "sleep_quality": 4,
    "headache_freq": 1,
    "eyestrain_freq": 2,
    "milk_consumption_ml": 200,
    "vision_label": "mild",
}

prediction = predict_vision_status_severity(custom_user)
print(prediction["predicted_vision_severity"])
```

Run this after the notebook has trained the encoders, scaler, and ordinal XGBoost model. Input category spelling must match the categories present in the dataset.

## Limitations and responsible use

- The dataset is very small and may not represent the broader population.
- The target is heuristically generated rather than clinically validated.
- Several predictors are self-reported and may contain recall or measurement bias.
- The random train/test split can produce unstable results with so few normal examples.
- Symptom features can be closely related to the proxy target, creating potential leakage or circularity.
- Correlation in this analysis does not establish that device use causes vision impairment.
- Predictions should not be used for diagnosis, treatment, screening decisions, or medical advice.

For a more reliable study, replace the proxy target with validated clinical or symptom-scale outcomes, collect a larger and more diverse dataset, pre-register severity thresholds, and use nested cross-validation with calibrated, group-aware evaluation.

## Project team

- Arpit Raj — BTECH/10780/24
- Ayush Marvin Bilung — BTECH/10579/24
- Palash Siddharth Mendhe — BTECH/10536/24
- Pogula Raja Vardhan Reddy — BTECH/10985/24
- Yash Abasaheb Shinde — BTECH/10780/24

## License

This project is provided for academic and educational purposes. No separate license file is currently included in the repository.
