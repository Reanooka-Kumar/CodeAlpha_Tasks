from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }
        self.results = {}

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, dataset_name):
        print(f"\n--- Training Models for {dataset_name} ---")
        dataset_results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                dataset_results[name] = {
                    'accuracy': acc,
                    'report': report,
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }
                print(f"  Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        self.results[dataset_name] = dataset_results
        return dataset_results

    def print_summary(self):
        print("\n\n=== Final Summary ===")
        for dataset, models in self.results.items():
            print(f"\nDataset: {dataset}")
            # Find best model
            best_model_name = max(models, key=lambda k: models[k]['accuracy'])
            best_acc = models[best_model_name]['accuracy']
            print(f"  Best Model: {best_model_name} (Accuracy: {best_acc:.4f})")
            for name, metrics in models.items():
                print(f"    {name}: {metrics['accuracy']:.4f}")
