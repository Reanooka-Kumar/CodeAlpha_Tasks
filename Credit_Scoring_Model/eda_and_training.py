import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def run_pipeline():
    # 1. Load Data
    try:
        df = pd.read_csv('credit_data.csv')
    except FileNotFoundError:
        print("Error: credit_data.csv not found. Run data_generator.py first.")
        return

    print("--- Data Info ---")
    print(df.info())
    print("\n--- Class Distribution ---")
    print(df['Creditworthy'].value_counts())

    # 2. EDA (Simple visualizations)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    print("\nSaved correlation_matrix.png")

    # 3. Preprocessing
    X = df.drop('Creditworthy', axis=1)
    y = df['Creditworthy']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Modeling
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc
        }

        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, y_pred))

    # 5. Best Model
    best_model_name = max(results, key=lambda x: results[x]['F1 Score'])
    print(f"\nBest Model by F1 Score: {best_model_name}")
    print("Performance:", results[best_model_name])

if __name__ == "__main__":
    run_pipeline()
