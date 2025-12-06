from deploy_model import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor()
predictor.load_model()

# Test with a real customer from your business
real_customer = {
    'customer_id': 'REAL_CUST_001',
    'tenure': 18,  # months
    'MonthlyCharges': 89.99,  # dollars
    'TotalCharges': 1619.82,  # dollars
    'Contract_One year': 0,
    'Contract_Two year': 0,
    'PaperlessBilling': 1,
    'PaymentMethod_Electronic check': 1,
    'InternetService_Fiber optic': 1,
    # Add other features as needed...
}

result = predictor.predict_churn(real_customer)
print(f"Churn Probability: {result['churn_probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")