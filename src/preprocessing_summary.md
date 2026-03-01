# Preprocessing Summary

This document provides a factual summary of the preprocessing logic implemented in `src/preprocessing.py` and verified in `03_preprocessing.ipynb`.

## 1. Preprocessing Logic (`src/preprocessing.py`)

The preprocessing stage utilizes a scikit-learn `ColumnTransformer` to ensure a reproducible and leak-safe pipeline.

### Categorical Features
- **Missing Values**: Handled by a `SimpleImputer` using a constant 'Unknown' strategy.
- **Native Country**: Processed via a custom `CountryBinner` that groups values into 'United-States' and 'Other'.
- **Encoding**: All categorical features are converted to binary vectors using `OneHotEncoder`.

### Numeric Features
- **Capital Gains/Losses**: Applied a `log1p` (log(1+x)) transformation to mitigate the extreme right-skewness identified during EDA.
- **Scaling**: A `StandardScaler` is applied to all numeric features (optional, but enabled by default in the pipeline verification).

### Data Splitting
- **Stratification**: Data is split into Training (80%), Validation (10%), and Test (10%) sets. 
- **Consistency**: Stratification is based on the `income` target to maintain the ~76/24 class distribution across all subsets.
- **Reproducibility**: All splits use `random_state=42`.

## 2. Pipeline Verification (`03_preprocessing.ipynb`)

Numerical and distributional checks were performed to validate the pipeline's integrity.

- **Leakage Prevention**: The pipeline is strictly fitted on the training data only. Validation and test sets are transformed using the parameters (means, variances, categories) derived from the training set.
- **Dimensionality**: The final processed dataset contains 67 features after one-hot encoding.
- **Data Quality**: Post-processing checks confirm 0 missing values across all splits.
- **Distributional Stability**: Visualization of age density and income proportions confirms that the stratified splitting successfully preserved the underlying data structure across train, validation, and test sets.
