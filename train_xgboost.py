import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data():
    """Load the pre-cleaned data"""
    print("📁 Loading cleaned data...")
    df_clean = pd.read_csv('telco_churn_cleaned.csv')
    print(f"✅ Data loaded: {df_clean.shape}")
    return df_clean

def prepare_features_target(df_clean):
    """Split data into features (X) and target (y)"""
    print("\n🎯 Preparing features and target...")
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Churn rate: {y.mean():.2%}")
    
    return X, y

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    print("\n🚀 Training XGBoost Model...")
    
    # Initialize XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    print("✅ Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n📊 Evaluating Model Performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.2%}")
    
    # Detailed classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Confusion matrix
    print("🔄 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return y_pred, y_pred_proba

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    print(f"\n📊 Plotting Top {top_n} Feature Importances...")
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features for Churn Prediction')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📷 Feature importance plot saved as 'feature_importance.png'")
    
    return importance_df

def make_single_prediction(model, feature_names):
    """Make prediction on a single example"""
    print("\n🔮 Making Single Prediction Example...")
    
    # Create a sample customer (you can modify these values)
    sample_customer = {
        'tenure': 12,
        'MonthlyCharges': 70.50,
        'TotalCharges': 850.75,
        'SeniorCitizen': 0,
        'Partner': 1,
        'Dependents': 0,
        'PhoneService': 1,
        'MultipleLines': 1,
        'OnlineSecurity': 0,
        'OnlineBackup': 1,
        'DeviceProtection': 0,
        'TechSupport': 0,
        'StreamingTV': 1,
        'StreamingMovies': 1,
        'PaperlessBilling': 1,
        'gender_Male': 1,
        'InternetService_Fiber optic': 1,
        'InternetService_No': 0,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 1,
        'PaymentMethod_Mailed check': 0
    }
    
    # Convert to DataFrame
    sample_df = pd.DataFrame([sample_customer])
    
    # Ensure all columns are present and in correct order
    for col in feature_names:
        if col not in sample_df.columns:
            sample_df[col] = 0
    
    sample_df = sample_df[feature_names]
    
    # Make prediction
    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0][1]
    
    print(f"📋 Sample Customer Prediction:")
    print(f"   - Will Churn: {'YES' if prediction == 1 else 'NO'}")
    print(f"   - Churn Probability: {probability:.2%}")
    
    return prediction, probability

def main():
    """Main function to run the entire XGBoost training pipeline"""
    print("=" * 60)
    print("          XGBOOST CHURN PREDICTION MODEL")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        df_clean = load_cleaned_data()
        
        # Step 2: Prepare features and target
        X, y = prepare_features_target(df_clean)
        
        # Step 3: Split data
        print("\n✂️  Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 4: Train XGBoost model
        model = train_xgboost_model(X_train, y_train, X_test, y_test)
        
        # Step 5: Evaluate model
        y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
        
        # Step 6: Feature importance
        feature_importance_df = plot_feature_importance(model, X.columns.tolist())
        
        # Step 7: Single prediction example
        make_single_prediction(model, X.columns.tolist())
        
        # Step 8: Save model
        import joblib
        joblib.dump(model, 'xgboost_churn_model.pkl')
        print(f"\n💾 Model saved as 'xgboost_churn_model.pkl'")
        
        print("\n" + "=" * 60)
        print("✅ XGBOOST TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run main.py first to create the cleaned data file!")

if __name__ == "__main__":
    main()