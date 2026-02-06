import pandas as pd
import numpy as np
import random

def generate_data(num_samples=1000):
    np.random.seed(42)
    random.seed(42)

    data = {
        'Income': np.random.normal(50000, 15000, num_samples).astype(int),
        'Age': np.random.randint(18, 70, num_samples),
        'Loan_Amount': np.random.normal(10000, 5000, num_samples).astype(int),
        'Months_Employed': np.random.randint(0, 120, num_samples),
        'Num_Credit_Lines': np.random.randint(0, 15, num_samples),
        'Interest_Rate': np.random.uniform(5, 25, num_samples).round(2),
        'Loan_Term_Months': np.random.choice([12, 24, 36, 48, 60], num_samples),
        'DTi_Ratio': np.random.uniform(0.1, 0.6, num_samples).round(2), # Debt to Income
        'Payment_History': np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.7, 0.15, 0.1, 0.05]) # 0=On-time, >0=Late counts
    }
    
    df = pd.DataFrame(data)
    
    # Ensure no negative values
    df['Income'] = df['Income'].apply(lambda x: max(x, 10000))
    df['Loan_Amount'] = df['Loan_Amount'].apply(lambda x: max(x, 1000))
    
    # Generate Target Variable: Default (0 = Repay, 1 = Default)
    # Logic: High DTI, High Loan, Low Income, High Interest, Poor Payment History -> Higher risk
    
    def calculate_default_prob(row):
        score = 0
        score += row['DTi_Ratio'] * 100  # Max ~60
        score += (row['Loan_Amount'] / row['Income']) * 100 # Max ~50
        score += row['Interest_Rate'] * 2 # Max ~50
        score += row['Payment_History'] * 25 # High penalty for late payments (Max ~75)
        score -= (row['Months_Employed'] / 12) * 5 # Deduction for stability
        
        # Normalize and add noise
        prob = (score) / 250 # Increased denominator due to new term
        prob += np.random.normal(0, 0.1)
        return 1 if prob > 0.5 else 0

    df['Default'] = df.apply(calculate_default_prob, axis=1)
    
    # Rename target to be clearer: 1 = Creditworthy, 0 = Risky? 
    # Usually "Default" means bad. Let's stick to standard: 0 = Good, 1 = Bad (Default).
    # Or to fit user request "Creditworthiness": 1 = Worthy, 0 = Not.
    # Let's invert 'Default' to 'Creditworthy'.
    
    df['Creditworthy'] = 1 - df['Default']
    df.drop('Default', axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    df = generate_data(2000)
    output_path = 'credit_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data generated and saved to {output_path}")
    print(df['Creditworthy'].value_counts())
    print(df.head())
