# Eye Vision Prediction Project

## Overview
Predicts vision impairment risk based on user device/lifestyle data using ML models.

## How to Run
1. Install requirements: `pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn`
2. Run the notebook: `test_split.ipynb`

## Features
- Synthetic data generation
- Binary and ordinal classification
- Model evaluation metrics
- SHAP explainability
- Example user prediction

## Usage
- Modify `example_user` in the notebook for custom predictions.

## Limitations
- Uses synthetic data, not real-world validated.
- Model may not generalize to real populations.
- Feature selection is illustrative.

## Ethics
- Do not use for medical diagnosis.
- Ensure user privacy if using real data.
- Explain limitations to users.

## Visuals
- SHAP summary plot
- Confusion matrix
- Calibration curve
