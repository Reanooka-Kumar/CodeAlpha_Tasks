import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import ssl

# Fix for SSL certificate verify failed issues when fetching from URLs
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def load_breast_cancer_data():
    """
    Loads Breast Cancer Wisconsin (Diagnostic) dataset.
    Target: 0 = malignant, 1 = benign (Note: sklearn standard)
    But typically for disease prediction we want 1=Disease.
    We will invert if necessary, but let's keep it standard for now.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    # In sklearn breast cancer: 0 = Malignant (Disease), 1 = Benign (No Disease).
    # Let's map so 1 = Disease for consistency with other datasets.
    # New mapping: 1 = Malignant, 0 = Benign
    df['target'] = df['target'].apply(lambda x: 1 if x == 0 else 0)
    print("Breast Cancer Data Loaded. Shape:", df.shape)
    return df

def load_diabetes_data():
    """
    Loads Pima Indians Diabetes Database.
    Source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    Cols: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    try:
        df = pd.read_csv(url, names=columns)
        df.rename(columns={'Outcome': 'target'}, inplace=True)
        print("Diabetes Data Loaded. Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading Diabetes data: {e}")
        return None

def load_heart_disease_data():
    """
    Loads Heart Disease UCI dataset (Cleveland).
    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        df = pd.read_csv(url, names=columns, na_values="?")
        # Target usually 0=No Disease, 1,2,3,4 = Disease. Convert >0 to 1.
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        print("Heart Disease Data Loaded. Shape:", df.shape)
        return df
    except Exception as e:
        print(f"Error loading Heart Disease data: {e}")
        return None
