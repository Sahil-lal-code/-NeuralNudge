import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    """Load and clean the Telco Churn dataset"""
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Binary encoding
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Handle special columns
    df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 0})
    
    # Internet features
    internet_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
    for feature in internet_features:
        df[feature] = df[feature].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    
    # One-hot encoding
    categorical_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Drop customerID
    df_clean = df_encoded.drop('customerID', axis=1)
    
    return df_clean

def prepare_training_data(df_clean):
    """Split data into training and test sets"""
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test