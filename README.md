# Predictive Income Analysis

This project implements a binary classification pipeline to predict whether an individual's annual income exceeds $50,000 using the UCI Adult Census dataset. The analysis explores feature engineering, baseline modelling, and advanced ensemble methods to achieve high predictive performance. A final exploratory phase investigates the underlying data structure through dimensionality reduction and clustering.

## Dataset
The UCI Adult dataset (the "Census Income" dataset) contains demographic and employment data for over 48,000 individuals. The target variable is `income`, a binary feature indicating whether an individual earns more than $50K per year.

Source: https://archive.ics.uci.edu/ml/datasets/adult

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
- `logs/`: High-level decision logs (`agent_decisions.md`).
- `notebooks/`: Sequential analysis from data loading to structural exploration.
- `outputs/figures/`: Visualizations generated during the analysis.
- `report/`: Final report and presentation.
- `src/`: Reusable preprocessing and utility modules.

## Dependencies
Key Python Libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

Install via:
pip install -r requirements.txt

## Reproducibility
To reproduce results:
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place the Adult dataset files in 'data/raw'
4. Run the notebooks in order:
    - `01_load_data.ipynb`
    - `02_eda.ipynb`
    - `03_preprocessing.ipynb`
    - `04_models_baselines.ipynb`
    - `05_ensembles_and_advanced.ipynb`
    - `06_exploratory_structure.ipynb`

All experiments run with a fixed 'random_state=42' for reproducibility.

## Agent Collaboration

This project was developed using an agent-based workflow (Antigravity).
Key modelling decisions and agent verification steps are documented in 'logs/agent_decisions.md'.

Final results and conclusions reflect manual verification and critical evaluation.