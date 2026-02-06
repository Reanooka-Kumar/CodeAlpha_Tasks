import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def clean_data(df):
    """
    Handles missing values and duplicates.
    """
    if df is None:
        return None
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Impute missing values
    # For numerical columns, use median (robust to outliers)
    # For categorical, use mode? 
    # For simplicity in this pipeline, we'll use SimpleImputer on all feature columns.
    
    return df

def preprocess_and_split(df, target_col='target', test_size=0.2, random_state=42):
    """
    Splits features/target, imputes missing values, scales features, and splits train/test.
    """
    if df is None:
        return None, None, None, None
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train test split first to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    
    # Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train, X_test, y_train, y_test
