# Vision Status Prediction Using Device Usage Data# Eye Vision Prediction Project



A comprehensive machine learning project that predicts vision problems based on electronic device usage patterns and demographic information.## Overview

This project uses machine learning to predict vision status and severity based on electronic device usage, daily milk consumption, and optometrist vision categories. The dataset includes self-reported and logged features such as device type, screen time, session length, breaks, font size, brightness, dark mode usage, outdoor time, viewing distance, screen height, lighting, sleep quality, headache frequency, eyestrain frequency, daily milk consumption (in millilitres), and vision label (optometrist reading: normal, mild, moderate, severe).

---

## Files

## Project Information- `df_new.csv`: Cleaned dataset with all features, including milk consumption and vision label.

- `HeavyVersion.ipynb`: Main analysis and modeling notebook. Includes feature engineering, preprocessing, model training, prediction, and feature importance analysis. Supports prediction of vision status/severity using user input.

**Project By:** Group 15- `MAIML_Assignment_Clean.ipynb`: Alternate notebook for data exploration and modeling.

- `test_split.ipynb`: Notebook for data splitting and validation.

**Team Members:**

- Arpit Raj (BTECH/10780/24)## Features

- Ayush Marvin Bilung (BTECH/10579/24)- **Device Usage**: Type, hours/day, session length, breaks, font size, brightness, dark mode, outdoor time, viewing distance, screen height, lighting.

- Palash Siddharth Mendhe (BTECH/10536/24)- **Health & Lifestyle**: Sleep quality, headache frequency, eyestrain frequency, daily milk consumption (ml).

- Pogula Raja Vardhan Reddy (BTECH/10985/24)- **Vision Label**: Optometrist reading categories: `normal`, `mild`, `moderate`, `severe`.

- Yash Abasaheb Shinde (BTECH/10780/24)

## Model

**Date:** 10 October 2025- Random Forest regression is used to predict vision severity and eyestrain frequency.

- Categorical features are label-encoded.

---- Feature scaling is applied.

- Model evaluation includes RMSE and R² score.

## Project Goal- Feature importance analysis highlights key predictors.



Predict if someone has vision problems based on how they use electronic devices (phones, laptops, tablets, etc.) and their lifestyle habits. The project aims to identify early warning signs of vision impairment that correlate with device usage patterns.## Usage

1. Open `HeavyVersion.ipynb` in Jupyter or VS Code.

---2. Run all cells to load data, preprocess, train, and evaluate the model.

3. Use the interactive cell to input custom user data and predict vision severity.

## Prediction Tasks4. Acceptable input for `vision_label`: `normal`, `mild`, `moderate`, `severe`.



### Binary Classification## Example Prediction

Classifies individuals into two categories:```python

- **Normal Vision**: No vision impairmentcustom_user = {

- **Impaired Vision**: Some level of vision impairment    'age': 30,

    'gender': 'female',

### Multi-class Ordinal Classification    'device_type': 'mobile',

Classifies vision status into four severity levels:    'daily_hours': 5,

- **Normal**: No vision problems    'session_length': 60,

- **Mild**: Minor vision impairment    'breaks': 2,

- **Moderate**: Moderate vision impairment    'font_size': 'medium',

- **Severe**: Severe vision impairment    'brightness': 'high',

    'dark_mode': 'yes',

---    'outdoor_time': 1.0,

    'viewing_distance': 35,

##  Dataset Features    'screen_height': 'eye_level',

    'lighting': 'normal',

The project uses **16 features** across 4 main categories:    'sleep_quality': 8,

    'headache_freq': 1,

### 1. Device Usage Patterns    'eyestrain_freq': 2,

- `daily_hours`: Hours spent on devices per day    'milk_consumption_ml': 200,

- `session_length`: Average continuous usage time (minutes)    'vision_label': 'mild'

- `breaks`: Number of breaks taken during device use}

- `device_type`: Type of primary device used (mobile, laptop, tablet)prediction = predict_vision_status_severity(custom_user)

print(f"Predicted Vision Severity: {prediction['predicted_vision_severity']}")

### 2. Ergonomics & Environment```

- `viewing_distance`: Distance from eyes to screen (cm)

- `screen_height`: Screen position relative to eye level## Requirements

- `font_size`: Font size preference (small, medium, large)- Python 3.x

- `brightness`: Screen brightness level- pandas, numpy, matplotlib, seaborn, scikit-learn

- `dark_mode`: Whether dark mode is enabled

- `lighting`: Ambient lighting conditions## License

MIT

### 3. Demographics
- `age`: User's age
- `gender`: User's gender

### 4. Health & Lifestyle
- `outdoor_time`: Hours spent outdoors per day
- `sleep_quality`: Sleep quality rating (1-10)
- `milk_consumption_ml`: Daily milk consumption (milliliters)
- `vision_label`: Self-reported vision status from optometrist readings

**Note:** `headache_freq` and `eyestrain_freq` are excluded from main models to avoid data leakage, but are analyzed separately.

---

## Models & Methodology

### Models Used

1. **Logistic Regression** (Baseline)
   - Simple linear classifier
   - Provides interpretable baseline performance

2. **XGBoost** (Primary Model)
   - Gradient boosting classifier
   - Superior performance for both binary and ordinal tasks
   - Handles non-linear relationships effectively

### Machine Learning Pipeline

```
1. Data Loading & Exploration
   ├── Load dataset (df_new.csv)
   ├── Exploratory Data Analysis (EDA)
   └── Check data quality

2. Feature Engineering
   ├── Create target variables (binary & ordinal)
   ├── Encode categorical features
   └── Scale numeric features

3. Model Training
   ├── Train/Test Split (80/20)
   ├── Train Logistic Regression
   ├── Train XGBoost (Binary)
   └── Train XGBoost (Ordinal)

4. Model Evaluation
   ├── AUROC, F1-Score, Precision, Recall
   ├── Confusion Matrices
   ├── ROC Curves
   ├── Calibration Plots
   └── 5-Fold Cross-Validation

5. Model Interpretation
   ├── SHAP Analysis
   ├── Partial Dependence Plots (PDP)
   ├── Individual Conditional Expectation (ICE)
   └── Feature Importance

6. Robustness Analysis
   ├── Subgroup Analysis (Age, Gender)
   ├── Data Leakage Detection
   └── Model Fairness Check
```

---

## Key Results

### Model Performance (Binary Classification)
- **XGBoost AUROC**: ~0.85-0.95
- **Logistic Regression AUROC**: ~0.75-0.85
- **XGBoost outperforms** baseline across all metrics

### Top Predictive Features (SHAP Analysis)
1. **daily_hours**: Device usage time per day
2. **session_length**: Continuous usage duration
3. **viewing_distance**: Distance from screen
4. **outdoor_time**: Time spent outdoors
5. **sleep_quality**: Quality of sleep

### Key Insights
-  Longer device usage correlates with higher vision impairment risk
-  Proper viewing distance significantly reduces risk
-  More outdoor time is protective against vision problems
-  Model performs consistently across age and gender subgroups
-  Symptom features (headaches, eyestrain) show potential data leakage

---

## Getting Started

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or VS Code with Jupyter extension
```

### Required Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
```

### Installation

1. **Clone the repository** (if using Git):
```bash
git clone https://github.com/YashShinde39/Eye-Vision-prediction-project.git
cd Eye-Vision-prediction-project-3
```

2. **Install dependencies**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

3. **Ensure dataset is present**:
   - Place `df_new.csv` in the project directory

### Running the Notebook

1. **Open the notebook**:
```bash
jupyter notebook HeavyVersion.ipynb
```
or open in VS Code

2. **Run all cells** sequentially from top to bottom

3. **Key notebook sections**:
   - **Steps 1-2**: Data loading and exploration
   - **Steps 3-5**: Feature preparation and train/test split
   - **Steps 6-8**: Model training and evaluation
   - **Steps 9-11**: Model interpretation (SHAP, PDP)
   - **Steps 12-15**: Advanced analysis and predictions
   - **Step 16**: Final summary and conclusions

---

## Usage Example

### Making Predictions for New Users

```python
# Define user profile
custom_user = {
    'age': 30,
    'gender': 'female',
    'device_type': 'mobile',
    'daily_hours': 5,
    'session_length': 60,
    'breaks': 2,
    'font_size': 'medium',
    'brightness': 'high',
    'dark_mode': 'yes',
    'outdoor_time': 1.0,
    'viewing_distance': 35,
    'screen_height': 'eye_level',
    'lighting': 'normal',
    'sleep_quality': 8,
    'headache_freq': 1,
    'eyestrain_freq': 2,
    'milk_consumption_ml': 200,
    'vision_label': 'mild'
}

# Get prediction
prediction = predict_vision_status_severity(custom_user)
print(f"Predicted Vision Severity: {prediction['predicted_vision_severity']}")
```

---

## Project Structure

```
Eye-Vision-prediction-project-3/
│
├── HeavyVersion.ipynb          # Main Jupyter notebook with complete analysis
├── df_new.csv                  # Dataset (85 rows × 18 columns)
├── README.md                   # Project documentation (this file)
├── MAIML_Assignment_Clean.ipynb # Alternate simplified notebook
├── maiml_project.ipynb         # Additional project notebook
└── test_split.ipynb            # Test/validation experiments
```

---

## Technical Details

### Data Preprocessing
- **Categorical Encoding**: LabelEncoder for 8 categorical features
- **Feature Scaling**: StandardScaler for normalization
- **Target Creation**: Risk-based scoring system for vision status
- **Data Leakage Prevention**: Symptom features analyzed separately

### Evaluation Metrics
- **AUROC** (Area Under ROC Curve): Model discrimination ability
- **F1-Score**: Balance of precision and recall
- **Brier Score**: Calibration quality
- **Cohen's Kappa**: Agreement for ordinal classification
- **Confusion Matrix**: Detailed classification breakdown

### Model Explainability
- **SHAP Values**: Feature importance and impact direction
- **Partial Dependence Plots**: Marginal effect of features
- **ICE Plots**: Individual-level feature effects
- **Feature Importance**: Tree-based importance scores

---

## Limitations & Future Work

### Current Limitations
1. **Simulated Target Variable**: Vision status created from risk factors, not actual clinical data
2. **Small Dataset**: Only 85 samples; larger dataset needed for production use
3. **Correlational Analysis**: Cannot establish causation between device usage and vision problems
4. **Basic Hyperparameters**: Models use default settings

### Future Improvements
1. **Real Clinical Data**: Integrate actual Snellen/LogMAR vision test results
2. **Larger Dataset**: Collect more diverse samples across demographics
3. **Causal Inference**: Apply DAGs or propensity score matching
4. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV
5. **Specialized Models**: Explore ordinal regression libraries (e.g., `mord`)
6. **Longitudinal Study**: Track vision changes over time
7. **Additional Features**: Include blue light exposure, screen time patterns

---

## Visualizations

The notebook includes comprehensive visualizations:
- Distribution plots for all features
- Confusion matrices (heatmaps)
- ROC curves with AUC scores
- Calibration curves
- SHAP summary and waterfall plots
- Partial dependence plots
- Subgroup performance comparisons

---

## Contributing

This is an academic project by Group 15. For questions or suggestions:
- Open an issue on GitHub
- Contact team members via university email

---

## License

This project is created for academic purposes as part of the Machine Learning coursework.

---

## Acknowledgments

- Course instructors and teaching assistants
- Scikit-learn, XGBoost, and SHAP library developers
- Online resources and ML community tutorials

---

## Contact

**GitHub Repository**: [Eye-Vision-prediction-project](https://github.com/YashShinde39/Eye-Vision-prediction-project)

**Project Lead**: Yash Abasaheb Shinde (BTECH/10780/24)

---

** If you find this project helpful, please give it a star!**
