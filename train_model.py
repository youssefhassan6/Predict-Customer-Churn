import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import (classification_report, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix)

from xgboost import XGBClassifier
from custom_transformers import IQRCapper

# 1. Load Data
print("Loading data...")
df = pd.read_csv("customer_churn.csv")

# 2. Basic Cleaning
print("Cleaning data...")
df = df.drop_duplicates().reset_index(drop=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Target variable
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop(columns=['customerID', 'Churn'])

# 3. Train/Test Split (Stratified)
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate scale_pos_weight for imbalance
# total negative / total positive
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# 4. Define Column Groups
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
multi_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']

# 5. Build Preprocessing ColumnTransformer
print("Building preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('capping', IQRCapper()),
    ('log_transform', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', StandardScaler())
])

binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder())
])

multi_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('bin', binary_transformer, binary_features),
        ('multi', multi_transformer, multi_features)
    ])

# 6. Model Pipeline without SMOTE (using scale_pos_weight instead for native XGBoost handling)
print("Building model pipeline...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

# 7. Hyperparameter Tuning
print("Tuning hyperparameters...")
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 1.5],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# 8. Threshold Optimization via Out-Of-Fold predictions
print("Optimizing classification threshold on cross-validation data...")
y_train_cv_probs = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

thresholds = np.arange(0.1, 0.9, 0.02)
best_f1 = 0
best_threshold = 0.5

for thresh in thresholds:
    y_train_cv_pred = (y_train_cv_probs >= thresh).astype(int)
    current_f1 = f1_score(y_train, y_train_cv_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = thresh

print(f"Optimized Threshold: {best_threshold:.2f} (CV F1: {best_f1:.4f})")

# 9. Evaluation on Untouched Test Set
print("Evaluating final model on test set...")
y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= best_threshold).astype(int)

acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred)
rec = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_proba)
pr_auc = average_precision_score(y_test, y_test_proba)
conf_matrix = confusion_matrix(y_test, y_test_pred)

print("Test Set Performance:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Extract feature names for SHAP
# Fit the preprocessor on train data to extract feature names
best_pipeline.named_steps['preprocessor'].fit(X_train)
cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['multi'].named_steps['onehot']
cat_features = cat_encoder.get_feature_names_out(multi_features).tolist()
all_feature_names = numeric_features + binary_features + cat_features

# 10. Save the robust artifact dictionary
print("Saving model artifacts...")
artifacts = {
    "pipeline": best_pipeline,
    "optimal_threshold": best_threshold,
    "feature_names": all_feature_names,
    "metrics": {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
}
joblib.dump(artifacts, "churn_artifacts.pkl")
print("Artifacts saved to 'churn_artifacts.pkl'.")
