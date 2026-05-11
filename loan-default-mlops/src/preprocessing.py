import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def handle_missing_values(df):
    """
    Fills numerical missing values with median and categorical with mode.
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    return df_cleaned

def remove_outliers_iqr(df, columns):
    """
    Removes outliers from specified columns using the Interquartile Range (IQR) method.
    """
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out = df_out[(df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)]
    return df_out

def encode_categorical_features(df):
    """
    Encodes categorical features using Label Encoding.
    """
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded

def scale_features(X_train, X_test):
    """
    Scales numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(X_train, y_train, random_state=42):
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset.
    """
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled
