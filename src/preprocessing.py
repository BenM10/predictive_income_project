import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class CountryBinner(BaseEstimator, TransformerMixin):
    """
    Bins native-country into 'United-States' and 'Other'.
    Expects input where missing values are already handled (e.g., as 'Unknown').
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.asarray(X).reshape(-1)
        binned = np.where(X == 'United-States', 'United-States', 'Other')
        return binned.reshape(-1, 1)

def get_preprocessing_pipeline(numeric_features, categorical_features, scale=False):
    """
    Creates a scikit-learn preprocessing pipeline based on approved plan.
    
    Args:
        numeric_features (list): List of numeric column names.
        categorical_features (list): List of categorical column names.
        scale (bool): Whether to apply StandardScaler to numeric features.
        
    Returns:
        sklearn.compose.ColumnTransformer: The complete preprocessing pipeline.
    """
    
    # Define transformers for numeric features
    # log1p transformation for capital features specifically
    capital_features = [f for f in numeric_features if 'capital' in f]
    other_numeric = [f for f in numeric_features if f not in capital_features]
    
    numeric_transformer_steps = []
    if scale:
        numeric_transformer_steps.append(('scaler', StandardScaler()))
    
    numeric_transformer = Pipeline(steps=numeric_transformer_steps) if numeric_transformer_steps else 'passthrough'

    # Log1p transformer
    log_transformer = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p))
    ])
    if scale:
        log_transformer.steps.append(('scaler', StandardScaler()))

    # Categorical transformer
    # 1. Impute missing values with 'Unknown'
    # 2. Treat native-country with a custom binner
    # 3. OneHotEncode
    
    def get_cat_pipe(feature_name):
        steps = [('imputer', SimpleImputer(strategy='constant', fill_value='Unknown'))]
        if feature_name == 'native-country':
            steps.append(('binner', CountryBinner()))
        steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        return Pipeline(steps=steps)

    # Combine all into a ColumnTransformer
    transformers = []
    
    # Add capital features with log1p
    if capital_features:
        transformers.append(('num_log', log_transformer, capital_features))
    
    # Add other numeric features
    if other_numeric:
        transformers.append(('num', numeric_transformer, other_numeric))
        
    # Add categorical features (each gets its own pipe to handle native-country specifically)
    for cat in categorical_features:
        transformers.append((f'cat_{cat}', get_cat_pipe(cat), [cat]))

    return ColumnTransformer(transformers=transformers, remainder='drop')

def load_and_split_data(filepath, target_col='income', test_size=0.1, val_size=0.1, random_state=42):
    """
    Loads cleaned data and performs stratified split into train, validation, and test sets.
    """
    df = pd.read_csv(filepath)
    
    # Stratified split to handle target imbalance (76/24)
    # First split into train and holdout (test + val)
    holdout_size = test_size + val_size
    df_train, df_holdout = train_test_split(
        df, 
        test_size=holdout_size, 
        stratify=df[target_col], 
        random_state=random_state
    )
    
    # Split holdout into validation and test sets
    relative_val_size = val_size / holdout_size
    df_val, df_test = train_test_split(
        df_holdout, 
        test_size=1-relative_val_size, 
        stratify=df_holdout[target_col], 
        random_state=random_state
    )
    
    return df_train, df_val, df_test

