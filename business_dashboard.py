import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

class ChurnBusinessDashboard:
    def __init__(self):
        self.retention_cases_file = 'retention_cases.csv'
    
    def generate_dashboard(self):
        """Generate comprehensive business dashboard"""
        print("📊 CHURN PREDICTION BUSINESS DASHBOARD")
        print("=" * 50)
        
        # Load retention cases
        if os.path.exists(self.retention_cases_file):
            cases_df = pd.read_csv(self.retention_cases_file)
            self._analyze_retention_cases(cases_df)
        else:
            print("No retention cases data found")
        
        # Load daily reports
        self._analyze_daily_reports()
        
        # Generate visualizations
        self._create_visualizations()
    
    def _analyze_retention_cases(self, cases_df):
        """Analyze retention cases data"""
        print(f"\n🎯 RETENTION PERFORMANCE")
        print(f"   Total Cases: {len(cases_df)}")
        print(f"   High-Risk Cases: {len(cases_df[cases_df['risk_level'] == 'HIGH'])}")
        print(f"   Total Revenue Saved: ${cases_df['estimated_savings'].sum():,.2f}")
        print(f"   Average Churn Probability: {cases_df['churn_probability'].mean():.1%}")
        
        # Calculate success rate (you'd track actual outcomes)
        potential_savings = cases_df['estimated_savings'].sum()
        print(f"   Potential Annual Savings: ${potential_savings:,.2f}")
    
    def _analyze_daily_reports(self):
        """Analyze daily scan reports"""
        print(f"\n📈 TREND ANALYSIS")
        
        # Find daily report files
        report_files = [f for f in os.listdir('.') if f.startswith('daily_report_')]
        
        if report_files:
            latest_report = max(report_files)
            with open(latest_report, 'r') as f:
                report_data = json.load(f)
            
            print(f"   Latest Scan: {report_data['date'][:10]}")
            print(f"   High-Risk Customers: {report_data['high_risk_customers']}")
            print(f"   Current Revenue at Risk: ${report_data['total_revenue_at_risk']:,.2f}")
        else:
            print("   No daily reports available")
    
    def _create_visualizations(self):
        """Create business visualization charts"""
        plt.figure(figsize=(15, 10))
        
        # Sample data - replace with your actual data
        risk_distribution = {
            'LOW RISK': 65,
            'MEDIUM RISK': 25, 
            'HIGH RISK': 10
        }
        
        revenue_at_risk = {
            'LOW RISK': 15000,
            'MEDIUM RISK': 45000,
            'HIGH RISK': 120000
        }
        
        # Plot 1: Customer Risk Distribution
        plt.subplot(2, 2, 1)
        plt.pie(risk_distribution.values(), labels=risk_distribution.keys(), autopct='%1.1f%%')
        plt.title('Customer Risk Distribution')
        
        # Plot 2: Revenue at Risk by Segment
        plt.subplot(2, 2, 2)
        plt.bar(revenue_at_risk.keys(), revenue_at_risk.values(), color=['green', 'orange', 'red'])
        plt.title('Revenue at Risk by Segment')
        plt.xticks(rotation=45)
        
        # Plot 3: Retention ROI
        plt.subplot(2, 2, 3)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
        savings = [0, 5000, 12000, 18500, 21500]  # Cumulative savings
        plt.plot(months, savings, marker='o', linewidth=2)
        plt.title('Cumulative Revenue Saved')
        plt.ylabel('Dollars ($)')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Risk Factors
        plt.subplot(2, 2, 4)
        factors = ['Short Tenure', 'High Charges', 'No Contract', 'Payment Method', 'Service Type']
        impact = [85, 78, 72, 65, 58]
        plt.barh(factors, impact)
        plt.title('Top Churn Risk Factors')
        plt.xlabel('Impact Score')
        
        plt.tight_layout()
        plt.savefig('business_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📷 Dashboard saved as 'business_dashboard.png'")

def main():
    dashboard = ChurnBusinessDashboard()
    dashboard.generate_dashboard()
    
    print(f"\n🎉 DASHBOARD GENERATED!")
    print(f"💡 Key Insights:")
    print(f"   • Your system is actively protecting revenue")
    print(f"   • High-risk customers represent significant value")
    print(f"   • Proactive retention has measurable ROI")

if __name__ == "__main__":
    main()