import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Step 1: Loading Dataset & Offline Fallback ---")
# Load dataset strictly once from Seaborn cache/network
df_raw = sns.load_dataset('titanic')

# Immediately save offline fallback CSV
df_raw.to_csv("titanic.csv", index=False)
print("Saved titanic.csv offline fallback.\n")

# Use the loaded DataFrame for all subsequent steps
df = df_raw.copy()

print("--- Step 2: Profiling & Missing Values ---")
print(f"Dataset Shape: {df.shape}")
print("\nMissing Values Count & Percentage:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# Missing Handling Rules:
# - embarked/embarked_town (<5% missing) -> drop rows
# - age (19.8% missing, 5-30%) -> median imputation
# - deck (77.2% missing, >30%) -> drop column (too missing to impute accurately)
df_clean = df.drop(columns=['deck']).copy()
df_clean = df_clean.dropna(subset=['embarked', 'embark_town'])
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

print("\n--- Step 3: Univariate Analysis & Outliers ---")
for col in ['age', 'fare']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_clean[(df_clean[col] < (Q1 - 1.5 * IQR)) | (df_clean[col] > (Q3 + 1.5 * IQR))]
    print(f"{col.capitalize()} IQR Outlier Count: {len(outliers)}")

fare_mean = df_clean['fare'].mean()
fare_median = df_clean['fare'].median()
fare_mode = df_clean['fare'].mode()[0]
print(f"Fare Stats -> Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
print("Fare Skewness Decision: Right-skewed (Mean > Median > Mode)")

print("\n--- Step 4: Bivariate Analysis ---")
print("Survival Rate by Sex:")
print(df_clean.groupby('sex')['survived'].mean())

print("\nSurvival Rate by Pclass:")
print(df_clean.groupby('pclass')['survived'].mean())

print("\nSurvival Rate by Sex and Pclass:")
print(df_clean.groupby(['sex', 'pclass'])['survived'].mean())

# 6x6 Correlation Heatmap (excluding adult_male and alone)
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (6 Numeric Columns)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("\nSaved correlation_heatmap.png")

print("\n--- Step 5: Generating 4 Data Story Charts ---")

# Chart 1: Survival by Sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x='sex', y='survived', ci=None, palette='viridis')
plt.title("1. Survival Rate by Gender")
plt.savefig("chart1_survival_sex.png")
plt.close()

# Chart 2: Survival by Class and Age
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x='pclass', y='age', hue='survived')
plt.title("2. Age Distribution across Passenger Classes by Survival")
plt.savefig("chart2_age_pclass_survival.png")
plt.close()

# Chart 3: Fare vs Survival
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x='survived', y='fare', palette='Set2')
plt.title("3. Fare Distribution by Survival Status")
plt.savefig("chart3_fare_survival.png")
plt.close()

# Chart 4: Survival by Embarked Port & Class
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x='embarked', y='survived', hue='pclass', ci=None)
plt.title("4. Survival Rate by Embarkation Port and Class")
plt.savefig("chart4_embarked_class_survival.png")
plt.close()
print("Saved 4 EDA story charts as PNG files.")

print("\n--- Step 6: Exploratory Standardization Check ---")
age_mean, age_std = df_clean['age'].mean(), df_clean['age'].std()
fare_mean, fare_std = df_clean['fare'].mean(), df_clean['fare'].std()

age_z = (df_clean['age'] - age_mean) / age_std
fare_z = (df_clean['fare'] - fare_mean) / fare_std

print(f"Age Z-Score  -> Mean: {age_z.mean():.4f}, Std: {age_z.std():.4f}")
print(f"Fare Z-Score -> Mean: {fare_z.mean():.4f}, Std: {fare_z.std():.4f}")
print("Exploratory standardization confirmed (Mean ~0, Std ~1).")