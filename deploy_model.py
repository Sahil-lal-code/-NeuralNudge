import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

class ChurnPredictor:
    """Production-ready churn prediction system"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.threshold = 0.5  # Default classification threshold
        
    def load_model(self, model_path='production_churn_model.pkl', 
                  features_path='model_features.pkl'):
        """Load the trained model and feature information"""
        try:
            self.model = joblib.load(model_path)
            feature_info = joblib.load(features_path)
            self.feature_names = feature_info['feature_names']
            print(f"✅ Model loaded successfully (v{feature_info.get('version', '1.0')})")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_churn(self, customer_data):
        """Predict churn probability for a single customer"""
        if self.model is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None
        
        # Create feature vector
        feature_vector = self._prepare_features(customer_data)
        
        # Make prediction
        churn_probability = self.model.predict_proba(feature_vector)[0][1]
        prediction = 1 if churn_probability >= self.threshold else 0
        
        # Risk assessment
        risk_level = self._assess_risk(churn_probability)
        
        return {
            'churn_probability': float(churn_probability),
            'prediction': int(prediction),
            'risk_level': risk_level,
            'recommendation': self._get_recommendation(risk_level, customer_data)
        }
    
    def predict_batch(self, customers_data):
        """Predict churn for multiple customers"""
        results = []
        for i, customer in enumerate(customers_data):
            result = self.predict_churn(customer)
            result['customer_id'] = customer.get('customer_id', f'cust_{i+1}')
            results.append(result)
        return results
    
    def _prepare_features(self, customer_data):
        """Prepare features in the correct order for the model"""
        # Create a DataFrame with all features set to 0
        feature_dict = {feature: 0 for feature in self.feature_names}
        
        # Update with customer data
        for feature in self.feature_names:
            if feature in customer_data:
                feature_dict[feature] = customer_data[feature]
        
        return pd.DataFrame([feature_dict])[self.feature_names]
    
    def _assess_risk(self, probability):
        """Assess risk level based on probability"""
        if probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommendation(self, risk_level, customer_data):
        """Get business recommendation based on risk level"""
        tenure = customer_data.get('tenure', 0)
        monthly_charges = customer_data.get('MonthlyCharges', 0)
        
        recommendations = {
            "HIGH": [
                "Immediate retention call required",
                "Offer personalized discount (15-20%)",
                "Assign to dedicated retention specialist",
                "Schedule follow-up within 24 hours"
            ],
            "MEDIUM": [
                "Proactive outreach email",
                "Offer value-added service",
                "Customer satisfaction survey",
                "Monitor closely for 2 weeks"
            ],
            "LOW": [
                "Regular communication",
                "Upsell additional services", 
                "Loyalty program promotion",
                "Quarterly check-in"
            ]
        }
        
        # Add tenure-based insights
        if tenure < 6:
            recommendations[risk_level].append("New customer - focus on onboarding")
        elif tenure > 36:
            recommendations[risk_level].append("Long-term customer - high value")
        
        # Add spending-based insights
        if monthly_charges > 80:
            recommendations[risk_level].append("High-value customer - prioritize retention")
        
        return recommendations[risk_level]
    
    def set_threshold(self, threshold):
        """Set custom classification threshold"""
        self.threshold = threshold
        print(f"✅ Classification threshold set to {threshold}")

def create_sample_customers():
    """Create sample customers for testing"""
    return [
        {
            'customer_id': 'CUST_001',
            'tenure': 2,
            'MonthlyCharges': 95.50,
            'TotalCharges': 191.00,
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
            'customer_id': 'CUST_002',
            'tenure': 48,
            'MonthlyCharges': 45.00,
            'TotalCharges': 2160.00,
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
        },
        {
            'customer_id': 'CUST_003', 
            'tenure': 15,
            'MonthlyCharges': 75.25,
            'TotalCharges': 1128.75,
            'SeniorCitizen': 0,
            'Partner': 0,
            'Dependents': 0,
            'PhoneService': 1,
            'MultipleLines': 1,
            'OnlineSecurity': 0,
            'OnlineBackup': 1,
            'DeviceProtection': 0,
            'TechSupport': 0,
            'StreamingTV': 1,
            'StreamingMovies': 0,
            'PaperlessBilling': 1,
            'gender_Male': 1,
            'InternetService_Fiber optic': 1,
            'InternetService_No': 0,
            'Contract_One year': 1,
            'Contract_Two year': 0,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 0,
            'PaymentMethod_Mailed check': 1
        }
    ]

def main():
    """Main deployment function"""
    print("=" * 70)
    print("              CHURN PREDICTION SYSTEM - PRODUCTION DEPLOYMENT")
    print("=" * 70)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load model
    if not predictor.load_model():
        print("❌ Failed to load model. Please run optimized_model.py first.")
        return
    
    print("\n🔮 Making Predictions on Sample Customers...")
    print("-" * 50)
    
    # Get sample customers
    sample_customers = create_sample_customers()
    
    # Make predictions
    predictions = predictor.predict_batch(sample_customers)
    
    # Display results
    for pred in predictions:
        print(f"\n📋 Customer: {pred['customer_id']}")
        print(f"   Churn Probability: {pred['churn_probability']:.2%}")
        print(f"   Prediction: {'🚨 WILL CHURN' if pred['prediction'] == 1 else '✅ WILL STAY'}")
        print(f"   Risk Level: {pred['risk_level']}")
        print(f"   Recommendations:")
        for i, rec in enumerate(pred['recommendation'], 1):
            print(f"     {i}. {rec}")
    
    # Save predictions to JSON
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model_version': '1.0',
        'predictions': predictions
    }
    
    with open('churn_predictions.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Predictions saved to 'churn_predictions.json'")
    
    # Business summary
    print("\n📊 BUSINESS SUMMARY")
    print("-" * 30)
    high_risk = sum(1 for p in predictions if p['risk_level'] == 'HIGH')
    medium_risk = sum(1 for p in predictions if p['risk_level'] == 'MEDIUM')
    
    print(f"High Risk Customers: {high_risk}")
    print(f"Medium Risk Customers: {medium_risk}")
    print(f"Total Customers Analyzed: {len(predictions)}")
    
    if high_risk > 0:
        print(f"🚨 ACTION REQUIRED: {high_risk} customers need immediate attention!")
    
    print("\n" + "=" * 70)
    print("✅ PRODUCTION SYSTEM READY!")
    print("=" * 70)

if __name__ == "__main__":
    main()