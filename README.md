# 🏡 Smart Housing Price Analytics

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Enabled-brightgreen" alt="XGBoost">
  <img src="https://img.shields.io/github/last-commit/MeghaKA/Smart-Housing-Price-Analytics" alt="Last Commit">
  <img src="https://img.shields.io/github/stars/MeghaKA/Smart-Housing-Price-Analytics?style=social" alt="Stars">
</p>

<p align="center">
  <b>End-to-end machine learning pipeline for predicting residential property prices</b><br>
  using the King County Housing Dataset — from raw data to a deployable prediction model.
</p>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Demo / Screenshots](#-demo--screenshots)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Tech Stack](#️-tech-stack)
- [Getting Started](#-getting-started)
- [Future Work](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## 📌 Overview

This project builds and compares machine learning models to predict residential
property prices in King County, WA, using structural, locational, and quality
features of each home. It covers the full applied-ML lifecycle: exploratory
data analysis, data cleaning, feature engineering, model training,
hyperparameter tuning, and evaluation — with an emphasis on both **predictive
accuracy** and **interpretability** (which features actually drive price).

**Objectives**
- Perform in-depth Exploratory Data Analysis (EDA)
- Clean and preprocess real-world housing data
- Engineer meaningful predictive features (e.g. house age, renovation flag, price per sqft)
- Train and compare multiple regression models
- Identify the strongest price-driving factors
- Package a reproducible, end-to-end prediction pipeline

---

## 🎥 Demo / Screenshots

> _Add a screenshot or GIF of your `app/` prediction interface here, e.g.:_
> `![App demo](figures/app_demo.gif)`

| EDA | Model Performance |
|---|---|
| ![Price distribution](figures/price_distribution.png) | ![Predicted vs Actual](figures/predicted_vs_actual.png) |

---

## 📂 Dataset

**Source:** [King County Housing Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (Kaggle)

~21,600 home sales records (2014–2015) with features including:

| Category | Features |
|---|---|
| Structural | Bedrooms, Bathrooms, Living Area (sqft), Lot Size, Floors |
| Quality | Grade, Condition, Waterfront, View |
| Temporal | Year Built, Year Renovated |
| Location | Zip code, Latitude, Longitude |
| Target | Sale Price |

Raw data lives in `dataset/raw/`; the cleaned, feature-engineered version used
for modeling is in `dataset/processed/`.

---

## 📁 Repository Structure

```
Smart-Housing-Price-Analytics/
│
├── dataset/
│   ├── raw/                  # Original King County dataset
│   └── processed/            # Cleaned + feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── app/                      # Prediction app / interactive demo
├── figures/                  # All EDA & model-result visualizations
├── models/                   # Saved trained models (.pkl / .joblib)
├── results/                  # metrics.csv, model_comparison.csv
├── reports/                  # Written summary / findings report
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🔍 Methodology

```
King County Dataset
      ↓
Data Cleaning (missing values, outliers, dtype fixes)
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering (house age, renovation flag, price/sqft, location clusters)
      ↓
Train/Test Split + Cross-Validation
      ↓
Model Training (Linear Regression → Random Forest → XGBoost)
      ↓
Hyperparameter Tuning
      ↓
Evaluation (RMSE, MAE, R²) + Feature Importance
      ↓
Final Model Selection
```

---

## 📊 Results

> _Fill in with your actual numbers from `results/model_comparison.csv`._

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost (tuned) | — | — | — |

**Best model:** _[model name]_ — selected for the best balance of accuracy and
generalization on held-out data (see `reports/` for the full write-up).

![Feature Importance](figures/feature_importance.png)

---

## 💡 Key Insights

> _Replace with your actual findings, e.g.:_
- Living area (sqft) and grade were the strongest predictors of price.
- Waterfront properties commanded a significant price premium even after controlling for size.
- Renovation recency had a measurable but secondary effect on price.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data handling:** Pandas, NumPy
- **Modeling:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Dev Containers
- **Version control:** Git & GitHub

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/MeghaKA/Smart-Housing-Price-Analytics.git
cd Smart-Housing-Price-Analytics

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📊 Future Improvements

- [ ] Hyperparameter optimization (Optuna / GridSearchCV)
- [ ] Advanced ensemble / stacking models
- [ ] Geographic feature engineering (distance to city center, neighborhood clustering)
- [ ] SHAP-based model explainability
- [ ] Interactive Streamlit dashboard for live predictions
- [ ] Deployment (Docker + cloud hosting)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## 👩‍💻 Author

**Megha K A**
[GitHub](https://github.com/MeghaKA) · Feel free to ⭐ this repo if you found it useful!
