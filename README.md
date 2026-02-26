# Predictive Income Analysis

This project implements a binary classification pipeline to predict whether an individual's annual income exceeds $50,000 using the UCI Adult Census dataset. The analysis explores feature engineering, baseline modelling, and advanced ensemble methods to achieve high predictive performance. A final exploratory phase investigates the underlying data structure through dimensionality reduction and clustering.

## Dataset
The UCI Adult dataset (the "Census Income" dataset) contains demographic and employment data for over 48,000 individuals. The target variable is `income`, a binary feature indicating whether an individual earns more than $50K per year.

## Model Performance
The final selected model is a **HistGradientBoostingClassifier**, tuned via grid search on the training set.

| Metric | Validation Score |
| :--- | :--- |
| **ROC AUC** | 0.9325 |
| **F1-Score (>50K)** | 0.72 |
| **Precision (>50K)** | 0.80 |
| **Recall (>50K)** | 0.66 |

## Workflow Summary
The project follows a structured modelling lifecycle:
1. **EDA**: Initial exploratory data analysis and visualization.
2. **Preprocessing**: Handling missing values, custom binning for high-cardinality features, log transformations for skewed variables, and feature scaling.
3. **Baselines**: Establishing performance benchmarks using Dummy classifiers and Logistic Regression.
4. **Ensembles**: Implementing and tuning Random Forest and Gradient Boosting models.
5. **Structure Analysis**: Unsupervised exploration using PCA and K-Means to validate feature redundancy and class separation.

## Project Structure
- `data/`: Raw and processed CSV files.
- `notebooks/`: Sequential analysis from data loading to structural exploration.
- `src/`: Reusable preprocessing and utility modules.
- `logs/`: High-level decision logs (`agent_decisions.md`).
- `outputs/figures/`: Visualizations generated during the analysis.

## Reproducibility
To reproduce the results, install the dependencies and execute the notebooks in numerical order. Experimental consistency is maintained throughout using a fixed seed (`random_state=42`).

1. `01_load_data.ipynb`
2. `02_eda.ipynb`
3. `03_preprocessing.ipynb`
4. `04_models_baselines.ipynb`
5. `05_ensembles_and_advanced.ipynb`
6. `06_exploratory_structure.ipynb`

Full technical logs and modelling choices are documented in the notebooks and `logs/agent_decisions.md`.
