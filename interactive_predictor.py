import pandas as pd
from deploy_model import ChurnPredictor
from retention_actions import CustomerRetentionSystem

class InteractiveChurnPredictor:
    def __init__(self):
        self.predictor = ChurnPredictor()
        self.predictor.load_model()
        self.retention_system = CustomerRetentionSystem()
    
    def collect_customer_data(self):
        """Collect all required customer data through interactive input"""
        print("🎯 CUSTOMER CHURN PREDICTION - INTERACTIVE MODE")
        print("=" * 50)
        print("Please enter the customer details below:\n")
        
        customer_data = {}
        
        # Basic Information
        customer_data['customer_id'] = input("Customer ID: ").strip() or "CUSTOMER_001"
        customer_data['tenure'] = int(input("Tenure (months): ") or "12")
        customer_data['MonthlyCharges'] = float(input("Monthly Charges ($): ") or "70.00")
        customer_data['TotalCharges'] = float(input("Total Charges ($): ") or "840.00")
        
        # Demographic Information
        print("\n👤 DEMOGRAPHIC INFORMATION:")
        customer_data['SeniorCitizen'] = self._get_yes_no_input("Senior Citizen (Yes/No): ")
        customer_data['Partner'] = self._get_yes_no_input("Has Partner (Yes/No): ")
        customer_data['Dependents'] = self._get_yes_no_input("Has Dependents (Yes/No): ")
        customer_data['gender_Male'] = self._get_yes_no_input("Gender is Male (Yes/No): ")
        
        # Services Information
        print("\n📞 SERVICES INFORMATION:")
        customer_data['PhoneService'] = self._get_yes_no_input("Has Phone Service (Yes/No): ")
        
        if customer_data['PhoneService']:
            customer_data['MultipleLines'] = self._get_yes_no_input("Has Multiple Lines (Yes/No): ")
        else:
            customer_data['MultipleLines'] = 0
        
        # Internet Service
        print("\n🌐 INTERNET SERVICE:")
        internet_service = self._get_choice_input(
            "Internet Service Type: ",
            ["DSL", "Fiber optic", "No"]
        )
        customer_data['InternetService_DSL'] = 1 if internet_service == "DSL" else 0
        customer_data['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
        customer_data['InternetService_No'] = 1 if internet_service == "No" else 0
        
        # Internet Add-ons (only if they have internet)
        if internet_service != "No":
            print("\n🔒 INTERNET ADD-ONS:")
            customer_data['OnlineSecurity'] = self._get_yes_no_input("Online Security (Yes/No): ")
            customer_data['OnlineBackup'] = self._get_yes_no_input("Online Backup (Yes/No): ")
            customer_data['DeviceProtection'] = self._get_yes_no_input("Device Protection (Yes/No): ")
            customer_data['TechSupport'] = self._get_yes_no_input("Tech Support (Yes/No): ")
            customer_data['StreamingTV'] = self._get_yes_no_input("Streaming TV (Yes/No): ")
            customer_data['StreamingMovies'] = self._get_yes_no_input("Streaming Movies (Yes/No): ")
        else:
            # No internet service means no add-ons
            customer_data['OnlineSecurity'] = 0
            customer_data['OnlineBackup'] = 0
            customer_data['DeviceProtection'] = 0
            customer_data['TechSupport'] = 0
            customer_data['StreamingTV'] = 0
            customer_data['StreamingMovies'] = 0
        
        # Contract and Billing
        print("\n📝 CONTRACT & BILLING:")
        contract_type = self._get_choice_input(
            "Contract Type: ",
            ["Month-to-month", "One year", "Two year"]
        )
        customer_data['Contract_One year'] = 1 if contract_type == "One year" else 0
        customer_data['Contract_Two year'] = 1 if contract_type == "Two year" else 0
        
        customer_data['PaperlessBilling'] = self._get_yes_no_input("Paperless Billing (Yes/No): ")
        
        payment_method = self._get_choice_input(
            "Payment Method: ",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
        )
        customer_data['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
        customer_data['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0
        customer_data['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == "Bank transfer" else 0
        customer_data['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card" else 0
        
        return customer_data
    
    def _get_yes_no_input(self, prompt):
        """Get Yes/No input and convert to 1/0"""
        while True:
            response = input(prompt).strip().lower()
            if response in ['yes', 'y', '1']:
                return 1
            elif response in ['no', 'n', '0', '']:
                return 0
            else:
                print("Please enter Yes or No")
    
    def _get_choice_input(self, prompt, options):
        """Get choice from multiple options"""
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            try:
                choice = int(input(f"Enter choice (1-{len(options)}): ") or "1")
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
    
    def predict_and_analyze(self, customer_data):
        """Make prediction and provide comprehensive analysis"""
        print("\n" + "=" * 60)
        print("🔮 PREDICTION RESULTS")
        print("=" * 60)
        
        # Make prediction
        prediction = self.predictor.predict_churn(customer_data)
        
        # Display results
        print(f"📋 Customer: {customer_data.get('customer_id', 'Unknown')}")
        print(f"🎯 Churn Probability: {prediction['churn_probability']:.2%}")
        print(f"🚨 Risk Level: {prediction['risk_level']}")
        
        # Generate retention plan
        retention_plan = self.retention_system._generate_retention_plan(prediction, customer_data)
        
        print(f"\n📋 RETENTION ACTION PLAN:")
        for i, action in enumerate(retention_plan['immediate_actions'], 1):
            print(f"   {i}. {action}")
        
        print(f"\n💡 PROACTIVE STRATEGIES:")
        for i, strategy in enumerate(retention_plan['proactive_strategies'], 1):
            print(f"   {i}. {strategy}")
        
        print(f"\n💰 BUSINESS IMPACT:")
        monthly_revenue = customer_data.get('MonthlyCharges', 0)
        print(f"   Monthly Revenue: ${monthly_revenue:.2f}")
        print(f"   Annual Revenue at Risk: ${monthly_revenue * 12:.2f}")
        print(f"   Potential Savings: ${retention_plan['estimated_savings']:.2f}")
        
        return prediction, retention_plan
    
    def run_interactive_mode(self):
        """Run the complete interactive prediction system"""
        try:
            # Collect customer data
            customer_data = self.collect_customer_data()
            
            # Make prediction and analysis
            prediction, retention_plan = self.predict_and_analyze(customer_data)
            
            # Save the case
            self.retention_system.save_retention_case(customer_data, {
                'churn_probability': prediction['churn_probability'],
                'risk_level': prediction['risk_level'],
                'estimated_savings': retention_plan['estimated_savings']
            })
            
            print(f"\n✅ Analysis completed successfully!")
            print(f"💾 Results saved to retention_cases.csv")
            
            return customer_data, prediction, retention_plan
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None

def main():
    interactive_predictor = InteractiveChurnPredictor()
    
    print("🚀 INTERACTIVE CHURN PREDICTION SYSTEM")
    print("=" * 50)
    print("This system will predict customer churn risk based on")
    print("the information you provide. Perfect for individual")
    print("customer analysis and portfolio demonstrations!\n")
    
    # Run interactive prediction
    customer_data, prediction, retention_plan = interactive_predictor.run_interactive_mode()
    
    if prediction:
        print(f"\n🎉 PREDICTION COMPLETE!")
        print(f"📊 Churn Risk: {prediction['churn_probability']:.2%}")
        print(f"🚨 Immediate action required: {prediction['risk_level'] == 'HIGH'}")
        
        # Option to analyze another customer
        another = input("\n🔍 Analyze another customer? (yes/no): ").strip().lower()
        if another in ['yes', 'y']:
            main()  # Restart

if __name__ == "__main__":
    main()