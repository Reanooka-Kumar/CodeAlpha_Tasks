import data_loader
import preprocessing
from models import ModelTrainer
import pandas as pd

def main():
    print("Initializing Disease Prediction System...")
    
    # 1. Load Data
    datasets = {
        'Breast Cancer': data_loader.load_breast_cancer_data(),
        'Diabetes': data_loader.load_diabetes_data(),
        'Heart Disease': data_loader.load_heart_disease_data()
    }
    
    trainer = ModelTrainer()
    
    # 2. Process and Train
    for name, df in datasets.items():
        if df is None:
            print(f"Skipping {name} due to load error.")
            continue
            
        print(f"\nProcessing {name} (Shape: {df.shape})...")
        
        # Checking balance
        print("Target distribution:\n", df['target'].value_counts())
        
        # Preprocessing
        X_train, X_test, y_train, y_test = preprocessing.preprocess_and_split(df)
        
        # Training
        trainer.train_and_evaluate(X_train, X_test, y_train, y_test, name)

    # 3. Summary
    trainer.print_summary()

if __name__ == "__main__":
    main()
