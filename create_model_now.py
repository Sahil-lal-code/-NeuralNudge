import pandas as pd
import xgboost as xgb
import joblib
import os

print("🚀 CREATING PRODUCTION MODEL FILES...")

# Check if cleaned data exists
if not os.path.exists('telco_churn_cleaned.csv'):
    print("❌ telco_churn_cleaned.csv not found! Please run main.py first.")
    exit()

# Load cleaned data
df_clean = pd.read_csv('telco_churn_cleaned.csv')
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

print(f"📊 Data loaded: {X.shape}")

# Create and train model with optimal parameters
print("🎯 Training production model...")
model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,  # Optimal weight from our testing
    random_state=42,
    eval_metric='logloss'
)

model.fit(X, y)
print("✅ Model trained successfully!")

# Save the model
joblib.dump(model, 'production_churn_model.pkl')
print("✅ production_churn_model.pkl created!")

# Save feature information
feature_info = {
    'feature_names': X.columns.tolist(),
    'model_type': 'XGBoost',
    'version': '1.0',
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
}
joblib.dump(feature_info, 'model_features.pkl')
print("✅ model_features.pkl created!")

# Verify files were created
if os.path.exists('production_churn_model.pkl') and os.path.exists('model_features.pkl'):
    print("\n🎉 SUCCESS! Model files created and verified!")
    print("📁 Files created:")
    print("   - production_churn_model.pkl")
    print("   - model_features.pkl")
    print("\n🚀 Now you can run: python deploy_model.py")
else:
    print("❌ ERROR: Model files were not created properly!")