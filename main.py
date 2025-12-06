from data_cleaning import load_and_clean_data, prepare_training_data

# Load and clean data
print("Loading and cleaning data...")
df_clean = load_and_clean_data('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"Cleaned data shape: {df_clean.shape}")
print(f"Churn rate: {df_clean['Churn'].mean():.2%}")

# Prepare for training
X_train, X_test, y_train, y_test = prepare_training_data(df_clean)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Save cleaned data
df_clean.to_csv('telco_churn_cleaned.csv', index=False)
print("\nCleaned data saved successfully!")

# Now you can use X_train, X_test, y_train, y_test for model training