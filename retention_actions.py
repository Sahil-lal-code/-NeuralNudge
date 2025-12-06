from deploy_model import ChurnPredictor
import pandas as pd
from datetime import datetime

class CustomerRetentionSystem:
    def __init__(self):
        self.predictor = ChurnPredictor()
        self.predictor.load_model()
    
    def analyze_customer(self, customer_data):
        """Comprehensive customer analysis with retention plan"""
        prediction = self.predictor.predict_churn(customer_data)
        
        print("=" * 60)
        print("🔄 CUSTOMER RETENTION ANALYSIS")
        print("=" * 60)
        
        print(f"📋 Customer ID: {customer_data.get('customer_id', 'Unknown')}")
        print(f"🎯 Churn Probability: {prediction['churn_probability']:.2%}")
        print(f"🚨 Risk Level: {prediction['risk_level']}")
        
        # Generate detailed retention plan
        retention_plan = self._generate_retention_plan(prediction, customer_data)
        
        print(f"\n📋 RETENTION ACTION PLAN:")
        for i, action in enumerate(retention_plan['immediate_actions'], 1):
            print(f"   {i}. {action}")
        
        print(f"\n💡 PROACTIVE STRATEGIES:")
        for i, strategy in enumerate(retention_plan['proactive_strategies'], 1):
            print(f"   {i}. {strategy}")
        
        print(f"\n💰 ESTIMATED SAVINGS: ${retention_plan['estimated_savings']:,.2f}")
        
        return retention_plan
    
    def _generate_retention_plan(self, prediction, customer_data):
        """Generate tailored retention plan based on risk level"""
        monthly_charges = customer_data.get('MonthlyCharges', 50)
        tenure = customer_data.get('tenure', 0)
        
        if prediction['risk_level'] == 'HIGH':
            return {
                'immediate_actions': [
                    "CALL WITHIN 24 HOURS: Personal retention call from manager",
                    "OFFER: 20% discount for 6 months",
                    "UPGRADE: Free premium service for 3 months", 
                    "SURVEY: Immediate customer satisfaction interview",
                    "ESCALATE: Assign to dedicated retention specialist"
                ],
                'proactive_strategies': [
                    "Weekly check-ins for first month",
                    "Personalized service review after 30 days",
                    "Loyalty program enrollment",
                    "Referral program invitation"
                ],
                'estimated_savings': monthly_charges * 24  # 2 years of revenue
            }
        
        elif prediction['risk_level'] == 'MEDIUM':
            return {
                'immediate_actions': [
                    "EMAIL: Personalized check-in within 48 hours",
                    "OFFER: 15% discount for 3 months",
                    "SURVEY: Customer feedback request",
                    "REVIEW: Service optimization suggestions"
                ],
                'proactive_strategies': [
                    "Monthly satisfaction check-ins",
                    "Educational content about service features",
                    "Cross-selling opportunities"
                ],
                'estimated_savings': monthly_charges * 12  # 1 year of revenue
            }
        
        else:  # LOW risk
            return {
                'immediate_actions': [
                    "EMAIL: Quarterly satisfaction survey",
                    "NEWSLETTER: Feature updates and tips"
                ],
                'proactive_strategies': [
                    "Upsell additional services",
                    "Invite to customer community",
                    "Request testimonials/referrals"
                ],
                'estimated_savings': monthly_charges * 6  # 6 months revenue
            }
    
    def save_retention_case(self, customer_data, retention_plan):
        """Save retention case for tracking"""
        case_data = {
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'churn_probability': retention_plan.get('churn_probability', 0),
            'risk_level': retention_plan.get('risk_level', 'UNKNOWN'),
            'monthly_charges': customer_data.get('MonthlyCharges', 0),
            'estimated_savings': retention_plan.get('estimated_savings', 0),
            'actions_taken': [],
            'status': 'OPEN'
        }
        
        # Save to CSV for tracking
        import csv
        with open('retention_cases.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=case_data.keys())
            if f.tell() == 0:  # Write header if file is empty
                writer.writeheader()
            writer.writerow(case_data)
        
        print(f"💾 Retention case saved to 'retention_cases.csv'")

def main():
    retention_system = CustomerRetentionSystem()
    
    # Your high-risk customer
    high_risk_customer = {
        'customer_id': 'REAL_CUST_001',
        'tenure': 18,
        'MonthlyCharges': 89.99,
        'TotalCharges': 1619.82,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaperlessBilling': 1,
        'PaymentMethod_Electronic check': 1,
        'InternetService_Fiber optic': 1,
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
        'gender_Male': 1,
        'InternetService_No': 0,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Mailed check': 0
    }
    
    # Analyze and create retention plan
    retention_plan = retention_system.analyze_customer(high_risk_customer)
    
    # Save the case
    retention_system.save_retention_case(high_risk_customer, retention_plan)
    
    print(f"\n🎯 BUSINESS IMPACT:")
    monthly_revenue = high_risk_customer['MonthlyCharges']
    print(f"   Monthly Revenue at risk: ${monthly_revenue:.2f}")
    print(f"   Annual Revenue at risk: ${monthly_revenue * 12:.2f}")
    print(f"   Potential savings from retention: ${retention_plan['estimated_savings']:.2f}")

if __name__ == "__main__":
    main()