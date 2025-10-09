# Eye Vision Prediction Project

## Overview
This project uses machine learning to predict vision status and severity based on electronic device usage, daily milk consumption, and optometrist vision categories. The dataset includes self-reported and logged features such as device type, screen time, session length, breaks, font size, brightness, dark mode usage, outdoor time, viewing distance, screen height, lighting, sleep quality, headache frequency, eyestrain frequency, daily milk consumption (in millilitres), and vision label (optometrist reading: normal, mild, moderate, severe).

## Files
- `df_new.csv`: Cleaned dataset with all features, including milk consumption and vision label.
- `HeavyVersion.ipynb`: Main analysis and modeling notebook. Includes feature engineering, preprocessing, model training, prediction, and feature importance analysis. Supports prediction of vision status/severity using user input.
- `MAIML_Assignment_Clean.ipynb`: Alternate notebook for data exploration and modeling.
- `test_split.ipynb`: Notebook for data splitting and validation.

## Features
- **Device Usage**: Type, hours/day, session length, breaks, font size, brightness, dark mode, outdoor time, viewing distance, screen height, lighting.
- **Health & Lifestyle**: Sleep quality, headache frequency, eyestrain frequency, daily milk consumption (ml).
- **Vision Label**: Optometrist reading categories: `normal`, `mild`, `moderate`, `severe`.

## Model
- Random Forest regression is used to predict vision severity and eyestrain frequency.
- Categorical features are label-encoded.
- Feature scaling is applied.
- Model evaluation includes RMSE and R² score.
- Feature importance analysis highlights key predictors.

## Usage
1. Open `HeavyVersion.ipynb` in Jupyter or VS Code.
2. Run all cells to load data, preprocess, train, and evaluate the model.
3. Use the interactive cell to input custom user data and predict vision severity.
4. Acceptable input for `vision_label`: `normal`, `mild`, `moderate`, `severe`.

## Example Prediction
```python
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
prediction = predict_vision_status_severity(custom_user)
print(f"Predicted Vision Severity: {prediction['predicted_vision_severity']}")
```

## Requirements
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scikit-learn

## License
MIT
