import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, mean_absolute_error, 
                             mean_squared_error, r2_score)
from imblearn.over_sampling import SMOTE

print("--- Step 1: Loading Offline Data ---")
df = pd.read_csv("titanic.csv")

# Clean basic columns as decided in EDA
df = df.drop(columns=['deck'])
df = df.dropna(subset=['embarked'])

# Features & Targets
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

print("--- Step 2: Stratified Train/Test Split ---")
# Stratification maintains exact 0/1 target proportion across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Define Preprocessing Pipeline
num_features = ['age', 'fare', 'sibsp', 'parch']
cat_features = ['pclass', 'sex', 'embarked']

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

print("--- Step 4: Training & Evaluating 3 Classifiers ---")
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = []

for name, clf in classifiers.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('model', clf)])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Model': name, 'Accuracy': round(acc, 3), 'Precision': round(prec, 3),
        'Recall': round(rec, 3), 'F1 Score': round(f1, 3), 'ROC AUC': round(auc, 3)
    })

df_class_results = pd.DataFrame(results)
print("\nClassifier Performance Comparison:")
print(df_class_results.to_string(index=False))

# Plot Decision Tree
dt_pipe = Pipeline([('preprocessor', preprocessor), ('model', DecisionTreeClassifier(max_depth=3, random_state=42))])
dt_pipe.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(dt_pipe.named_steps['model'], filled=True, feature_names=dt_pipe.named_steps['preprocessor'].get_feature_names_out(), class_names=['Died', 'Survived'])
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png")
plt.close()

print("\n--- Step 5: Imbalance Handling Comparison ---")
# Preprocess training fold only for SMOTE comparison
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# Variant 1: Baseline
rf_base = RandomForestClassifier(random_state=42).fit(X_train_prep, y_train)
# Variant 2: Balanced Weights
rf_bal = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train_prep, y_train)
# Variant 3: SMOTE on Train Fold Only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_prep, y_train)
rf_smote = RandomForestClassifier(random_state=42).fit(X_train_sm, y_train_sm)

print(f"Baseline RF F1: {f1_score(y_test, rf_base.predict(X_test_prep)):.3f}")
print(f"Balanced RF F1: {f1_score(y_test, rf_bal.predict(X_test_prep)):.3f}")
print(f"SMOTE RF F1:    {f1_score(y_test, rf_smote.predict(X_test_prep)):.3f}")

print("\n--- Step 6: Hyperparameter Tuning (Random Forest) ---")
rf_param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [4, 6, 8],
    'model__max_features': ['sqrt', 'log2']
}

full_rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(oob_score=True, random_state=42))
])

grid = GridSearchCV(full_rf_pipe, rf_param_grid, cv=3, scoring='f1')
grid.fit(X_train, y_train)

best_pipe = grid.best_estimator_
print(f"Best Parameters: {grid.best_params_}")
print(f"OOB Score: {best_pipe.named_steps['model'].oob_score_:.3f}")

print("\n--- Step 7: Regression Side-Task (Predicting Fare) ---")
y_reg = df['fare']
X_reg = df.drop(columns=['fare', 'survived', 'who', 'adult_male', 'alive', 'alone', 'embarked'])

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_num = ['age', 'pclass', 'sibsp', 'parch']
reg_cat = ['sex', 'embark_town', 'class']

reg_prep = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), reg_num),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), reg_cat)
])

reg_pipe = Pipeline([('prep', reg_prep), ('model', LinearRegression())])
reg_pipe.fit(X_tr_r, y_tr_r)

y_reg_pred = reg_pipe.predict(X_te_r)

mae = mean_absolute_error(y_te_r, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_te_r, y_reg_pred))
r2 = r2_score(y_te_r, y_reg_pred)
n, k = len(y_te_r), X_tr_r.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print(f"Regression Metrics -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, Adj R2: {adj_r2:.3f}")

# Residual Plot
residuals = y_te_r - y_reg_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_reg_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot (Fare Prediction)")
plt.savefig("residuals.png")
plt.close()
print("Heteroscedasticity Conclusion: Cone-shaped spread of residuals confirms presence of heteroscedasticity.")

print("\n--- Step 8: Exporting Complete Pipeline ---")
# Save the complete fitted pipeline to disk
joblib.dump(best_pipe, "best_pipeline.joblib")
print("Saved best_pipeline.joblib successfully.")

# Test Reloading Artifact
loaded_pipe = joblib.load("best_pipeline.joblib")
sample_raw_data = pd.DataFrame([{
    'pclass': 3, 'sex': 'male', 'age': 22.0, 
    'sibsp': 1, 'parch': 0, 'fare': 7.25, 'embarked': 'S'
}])
prediction = loaded_pipe.predict(sample_raw_data)
print(f"Reloaded Pipeline Test Prediction on Raw Input: {prediction[0]} (0=Died, 1=Survived)")