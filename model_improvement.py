import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

def improved_xgboost_training():
    """Enhanced XGBoost training with hyperparameter tuning"""
    print("🔄 Loading data for improved model...")
    df_clean = pd.read_csv('telco_churn_cleaned.csv')
    
    X = df_clean.drop('Churn', axis=1)
    y = df_clean['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("⚙️  Training improved XGBoost with class weights...")
    
    # Calculate class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    
    # Create weight array for training
    sample_weights = np.where(y_train == 1, class_weights[1], class_weights[0])
    
    # Improved XGBoost with better parameters
    improved_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_weights[1]/class_weights[0],  # Handle class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train with sample weights
    improved_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_pred_improved = improved_model.predict(X_test)
    y_pred_proba_improved = improved_model.predict_proba(X_test)[:, 1]
    
    accuracy_improved = accuracy_score(y_test, y_pred_improved)
    auc_improved = roc_auc_score(y_test, y_pred_proba_improved)
    
    print(f"\n🎯 Improved Model Results:")
    print(f"Accuracy: {accuracy_improved:.2%}")
    print(f"AUC Score: {auc_improved:.2%}")
    
    print("\n📊 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_improved, target_names=['No Churn', 'Churn']))
    
    # Compare with original model
    print("\n📈 Comparison with Original Model:")
    print(f"Original Accuracy: 80.34%")
    print(f"Improved Accuracy: {accuracy_improved:.2%}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    cm_original = np.array([[926, 109], [168, 206]])
    sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues')
    plt.title('Original Model\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.subplot(1, 2, 2)
    cm_improved = confusion_matrix(y_test, y_pred_improved)
    sns.heatmap(cm_improved, annot=True, fmt='d', cmap='Blues')
    plt.title('Improved Model\nConfusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return improved_model

def predict_new_customers(model):
    """Make predictions on new customer data"""
    print("\n🔮 Predicting New Customers...")
    
    # Example new customers (you can modify these)
    new_customers = [
        {
            'name': 'High Risk Customer',
            'tenure': 1,
            'MonthlyCharges': 99.99,
            'TotalCharges': 99.99,
            'SeniorCitizen': 0,
            'Partner': 0,
            'Dependents': 0,
            'PhoneService': 1,
            'MultipleLines': 1,
            'OnlineSecurity': 0,
            'OnlineBackup': 0,
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
        },
        {
            'name': 'Low Risk Customer', 
            'tenure': 60,
            'MonthlyCharges': 45.50,
            'TotalCharges': 2730.00,
            'SeniorCitizen': 0,
            'Partner': 1,
            'Dependents': 1,
            'PhoneService': 1,
            'MultipleLines': 0,
            'OnlineSecurity': 1,
            'OnlineBackup': 1,
            'DeviceProtection': 1,
            'TechSupport': 1,
            'StreamingTV': 0,
            'StreamingMovies': 0,
            'PaperlessBilling': 0,
            'gender_Male': 0,
            'InternetService_Fiber optic': 0,
            'InternetService_No': 0,
            'Contract_One year': 0,
            'Contract_Two year': 1,
            'PaymentMethod_Credit card (automatic)': 1,
            'PaymentMethod_Electronic check': 0,
            'PaymentMethod_Mailed check': 0
        }
    ]
    
    df_clean = pd.read_csv('telco_churn_cleaned.csv')
    feature_columns = df_clean.drop('Churn', axis=1).columns.tolist()
    
    for customer in new_customers:
        # Prepare customer data
        customer_data = {col: customer.get(col, 0) for col in feature_columns}
        customer_df = pd.DataFrame([customer_data])[feature_columns]
        
        # Make prediction
        prediction = model.predict(customer_df)[0]
        probability = model.predict_proba(customer_df)[0][1]
        
        risk_level = "HIGH RISK" if probability > 0.7 else "MEDIUM RISK" if probability > 0.3 else "LOW RISK"
        
        print(f"\n📋 {customer['name']}:")
        print(f"   - Churn Probability: {probability:.2%}")
        print(f"   - Prediction: {'WILL CHURN' if prediction == 1 else 'WILL STAY'}")
        print(f"   - Risk Level: {risk_level}")

if __name__ == "__main__":
    print("=" * 60)
    print("          MODEL IMPROVEMENT & PREDICTION")
    print("=" * 60)
    
    # Train improved model
    improved_model = improved_xgboost_training()
    
    # Make predictions on new customers
    predict_new_customers(improved_model)
    
    print("\n" + "=" * 60)
    print("✅ MODEL IMPROVEMENT COMPLETED!")
    print("=" * 60)