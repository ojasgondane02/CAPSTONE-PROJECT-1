import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Adjust matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
# Load the data
df = pd.read_csv(r"C:\Users\ojasp\ML codes\CAPSTONE PROJECT #1\loan-default-mlops\data\Loan_Default.csv")

print("Shape of dataset:", df.shape)
display(df.head())

df.info()
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
import sys
sys.path.append('../')
from src.preprocessing import handle_missing_values, remove_outliers_iqr

# 1. Drop duplicates
df = df.drop_duplicates()

# Fix Data Leakage
columns_to_drop = ['ID', 'year', 'Interest_rate_spread', 'rate_of_interest', 'Upfront_charges']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# 2. Handle missing values
df_clean = handle_missing_values(df)
print("Missing values after cleaning:\n", df_clean.isnull().sum())

# 3. Remove outliers (applying only to numerical columns)
numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.drop('Status', errors='ignore')
df_clean = remove_outliers_iqr(df_clean, numerical_cols)
print("\nShape after outlier removal:", df_clean.shape)
# Target Variable Distribution
plt.figure(figsize=(5, 4))
sns.countplot(data=df_clean, x='Status', palette='pastel')
plt.title("Target Variable Distribution (Default)")

print("Insight: We can see the class distribution. If there are far more 0s than 1s, we have an imbalanced dataset.")
# Numerical Features Histogram
df_clean[numerical_cols].hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')

print("Insight: These histograms help us understand the distribution (normal, skewed, etc.) of each numerical feature.")
# Count Plot for Categorical columns
categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df_clean, x=col, palette='Set2')
    plt.title(f"Count Plot - {col}")
print("Insight: Visualize the frequencies of different categories in our dataset.")
# Boxplots for Outlier Analysis
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[numerical_cols], orient='h', palette='Set3')
plt.title("Boxplots of Numerical Features (Post-outlier treatment)")
print("Insight: Helps to visually verify that severe outliers were handled appropriately.")
# Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df_clean[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")

print("Insight: Checks for multicollinearity. Highly correlated features might need to be dropped.")
from src.preprocessing import encode_categorical_features, scale_features
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE

# 1. Encoding
df_encoded = encode_categorical_features(df_clean)

# 2. Train-Test Split
X = df_encoded.drop('Status', axis=1)
y = df_encoded['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# 3. Scaling
# Store column names to recreate DataFrame after scaling
columns = X_train.columns
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Convert back to DataFrame for easier handling
X_train = pd.DataFrame(X_train_scaled, columns=columns)
X_test = pd.DataFrame(X_test_scaled, columns=columns)

from src.preprocessing import apply_smote

print("Before SMOTE - y_train counts:\n", y_train.value_counts())
X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=RANDOM_STATE)
print("\nAfter SMOTE - y_train counts:\n", y_train_resampled.value_counts())
from src.train_utils import get_models, tune_hyperparameters
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.config import PARAM_GRIDS

models = get_models(random_state=RANDOM_STATE)
results = []
best_model_obj = None
best_f1 = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # Hyperparameter tuning if params exist
    if name in PARAM_GRIDS:
        print(f"Tuning hyperparameters for {name}...")
        model = tune_hyperparameters(model, PARAM_GRIDS[name], X_train_resampled, y_train_resampled)
    else:
        model.fit(X_train_resampled, y_train_resampled)
        
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Evaluation
    metrics = evaluate_model(y_test, y_pred, y_prob)
    metrics['Model'] = name
    results.append(metrics)
    
    print(f"Metrics: {metrics}")
    
    # Plotting
    plot_confusion_matrix(y_test, y_pred, name)
    if y_prob is not None:
        plot_roc_curve(y_test, y_prob, name)
        
    # Keep track of the best model (using F1-Score)
    if metrics['F1-Score'] > best_f1:
        best_f1 = metrics['F1-Score']
        best_model_obj = model
        best_model_name = name
results_df = pd.DataFrame(results).set_index('Model')
display(results_df.sort_values(by='F1-Score', ascending=False))
if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.title(f"Feature Importances ({best_model_name})")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=45)
    plt.tight_layout()
    print("Insight: These are the top features influencing whether a person defaults on a loan.")
else:
    print(f"{best_model_name} does not support feature_importances_.")
import os
from src.train_utils import save_model

# Ensure models directory exists
os.makedirs('../models', exist_ok=True)

model_path = '../models/best_model.pkl'
save_model(best_model_obj, model_path)

print(f"Best model ({best_model_name}) saved successfully!")

# Test reloading the model
loaded_model = joblib.load(model_path)
sample_pred = loaded_model.predict(X_test.iloc[[0]])
print(f"Sample prediction from loaded model: {'Default' if sample_pred[0] == 1 else 'No Default'}")
