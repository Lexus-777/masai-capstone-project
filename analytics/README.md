# Module 2: Analytics Pipeline

## Overview
This module profiles, cleans, explores, and models the Titanic dataset using Seaborn, Scikit-Learn, and Imbalanced-Learn. It predicts passenger survival (classification) and ticket fare (regression) using preprocessing pipelines that prevent data leakage.

## Dataset Profiling & Cleaning Decisions
- **Missing Value Rates:**
  - `embarked` / `embark_town`: < 5% missing (0.22%) -> **Rows dropped**.
  - `age`: 19.8% missing (5%–30% range) -> **Imputed using training median**.
  - `deck`: 77.2% missing (> 30% threshold) -> **Column dropped** due to severe unreliability for imputation.
- **Offline Fallback:** Raw dataset saved locally as `titanic.csv` upon initial load.

## Exploratory Data Analysis Summary
- **Univariate Analysis:**
  - `age` IQR Outliers: 9 rows.
  - `fare` IQR Outliers: 116 rows.
  - `fare` Distribution: **Right-skewed** (Mean $32.20 > Median $14.45 > Mode $8.05).
- **Bivariate Analysis:**
  - Female Survival Rate: ~74.2% | Male Survival Rate: ~18.9%
  - Pclass 1 Survival: ~63.0% | Pclass 2: ~47.3% | Pclass 3: ~24.2%
  - **Top 2 Strongest Correlations:** `pclass` & `fare` (-0.55), `survived` & `pclass` (-0.34).
- **Multivariate Story:**
  1. *Sex:* Females had dramatically higher survival rates across all passenger classes.
  2. *Class & Age:* First-class passengers were older on average and enjoyed the highest survival rate, whereas third-class younger passengers had low survival rates.
  3. *Fare:* Higher fares strongly correlated with increased survival odds.
  4. *Port of Embarkation:* Passengers embarking at Cherbourg (C) had higher survival rates, largely driven by a higher proportion of first-class tickets.
- **Standardization Check:** Z-score calculation on `age` and `fare` confirmed scaling yields Mean ~0.0 and Std ~1.0.

## Machine Learning & Model Performance

### Classifier Performance Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.803 | 0.762 | 0.716 | 0.738 | 0.852 |
| **Decision Tree** | 0.809 | 0.811 | 0.642 | 0.717 | 0.824 |
| **Random Forest** | 0.820 | 0.794 | 0.716 | 0.753 | 0.865 |

### Imbalance Handling Strategy Comparison
- **Baseline RF F1:** 0.753
- **Class-Weighted RF F1:** 0.748
- **SMOTE RF F1:** 0.745
- *Conclusion:* Baseline Random Forest produced the strongest F1 score. Because the class balance (~38% survived) is moderately balanced, aggressive oversampling via SMOTE introduced slight noise.

### Hyperparameter Tuning & Out-of-Bag Score
- **Best Parameters:** `{'model__max_depth': 6, 'model__max_features': 'sqrt', 'model__n_estimators': 100}`
- **Out-of-Bag (OOB) Score:** 0.814

### Regression Side-Task (Fare Prediction)
- **MAE:** 13.85 | **RMSE:** 26.14 | **R²:** 0.718 | **Adjusted R²:** 0.702
- **Heteroscedasticity Analysis:** The residual plot displays a clear expansion pattern (cone shape) as predicted fares increase, confirming heteroscedasticity due to extreme luxury ticket price variance.

## Combined Summary Table & Deployment Recommendation

| Model Type | Model Name | Metric 1 | Metric 2 | Metric 3 | Metric 4 | Metric 5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classifier** | Logistic Regression | Acc: 0.803 | Prec: 0.762 | Rec: 0.716 | F1: 0.738 | AUC: 0.852 |
| **Classifier** | Decision Tree | Acc: 0.809 | Prec: 0.811 | Rec: 0.642 | F1: 0.717 | AUC: 0.824 |
| **Classifier** | **Random Forest** | **Acc: 0.820** | **Prec: 0.794** | **Rec: 0.716** | **F1: 0.753** | **AUC: 0.865** |
| **Regressor** | Linear Regression | MAE: 13.85 | RMSE: 26.14 | R²: 0.718 | Adj R²: 0.702 | N/A |

### Deployment Recommendation
I recommend deploying the **Random Forest Classifier**. It achieved the highest overall Accuracy (0.820), F1 Score (0.753), and ROC AUC (0.865). Its ensemble tree architecture effectively handles non-linear interactions (e.g., sex combined with pclass) better than Logistic Regression while resisting the overfitting tendencies of single Decision Trees.