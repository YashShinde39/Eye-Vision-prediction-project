# Eye Vision Prediction Project

This repository contains the code and data for predicting and analyzing factors affecting digital device-related symptoms using survey data. The project explores the relationships between digital device usage habits and symptoms such as headache frequency, eyestrain, and sleep quality.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Notebook Structure](#notebook-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Results and Visualizations](#results-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

**Objective:**  
Analyze and model survey data to predict and understand the factors influencing digital device-related symptoms (e.g., headache frequency, eyestrain, sleep quality).

The notebook covers:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering
- Predictive modeling (Random Forest Regression)
- Visualization of findings

## Dataset

The dataset (`df_new.csv`) is a cleaned survey of 49 entries with the following features:

- **age**: Age of the respondent
- **gender**: Gender (male/female)
- **device_type**: Device used (mobile/laptop/tablet/tv)
- **daily_hours**: Hours spent daily on the device
- **session_length**: Average session length (in minutes)
- **breaks**: Number of breaks per session
- **font_size**: Font size setting (small/medium/large)
- **brightness**: Screen brightness (low/medium/high)
- **dark_mode**: Use of dark mode (yes/no)
- **outdoor_time**: Time spent outdoors per day (in hours)
- **viewing_distance**: Viewing distance (in cm)
- **screen_height**: Screen position relative to eyes (below_eye/eye_level/above_eye)
- **lighting**: Ambient lighting (bright/normal/dim)
- **sleep_quality**: Self-reported sleep quality (1-5)
- **headache_freq**: Headache frequency (1-5)
- **eyestrain_freq**: Eyestrain frequency (1-5)

## Notebook Structure

The main notebook is [`MAIML_Assignment.ipynb`](MAIML_Assignment.ipynb). It includes:

1. **Introduction and Objective**
2. **Importing Libraries**
3. **Data Loading**
4. **Basic Exploration** (`df.info()`, `df.describe()`, missing value check)
5. **Exploratory Data Analysis (EDA)**  
   - Visualizations (histograms, count plots, etc.)
6. **Feature Engineering & Preprocessing**
7. **Predictive Modeling**  
   - Random Forest Regressor
   - Model evaluation (MSE, R²)
8. **Interpretation and Visualizations**  
   - Feature importance plots, etc.

## How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YashShinde39/Eye-Vision-prediction-project.git
    cd Eye-Vision-prediction-project
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start Jupyter Notebook and open `MAIML_Assignment.ipynb`:**
    ```bash
    jupyter notebook
    ```

4. **Run the cells in order. The notebook loads `df_new.csv` and generates all results and visualizations.**

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter notebook

*Install with:*
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Results and Visualizations

The notebook produces:
- Distribution plots of survey features
- Correlation heatmaps
- Regression model performance metrics
- Feature importance visualizations

These help to understand the factors most associated with headache and eyestrain frequency.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).
