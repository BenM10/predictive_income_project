# MSIN0097: Predictive Income Classification Project 

MSIN0097 Predictive Analytics
Candidate Number: XDHH9
Final Word Count: 1997

## Executive Summary

This project examines binary income classification using the UCI Adult (1994 US Census) dataset. The objective is to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes. Such classification problems are common in socioeconomic research and financial risk modelling, where income level serves as a proxy for economic stability.

The workflow combined exploratory data analysis with a structured preprocessing pipeline, including targeted log transformations and categorical encoding. A range of models were evaluated, spanning logistic regression, decision trees, ensemble methods, and a neural network baseline. Performance was primarily assessed using ROC AUC and the F1-score for the minority (>50K) class to ensure robustness under class imbalance.

HistGradientBoostingClassifier achieved the strongest validation performance, with a ROC AUC of 0.9325 and an F1-score of 0.724. Ensemble methods consistently outperformed linear and single-tree approaches, indicating the importance of modelling nonlinear interactions. Exploratory Principal Component Analysis further suggested moderate high-dimensional structure within the feature space. Nevertheless, the age of the dataset and the fixed binary income threshold limit direct contemporary generalisation.

## 1. Obtain a Dataset and Frame the Predictive Problem

This project uses the UCI Adult dataset, derived from the 1994 US Census, to address a supervised binary income classification task. The objective is to predict whether an individual’s annual income exceeds $50,000 using demographic and employment-related attributes. This corresponds to roughly $110,000 in 2026 dollars, representing a comparatively high real income level. Such classification problems are common in applied socioeconomic analysis and financial risk modelling, where income acts as a proxy for economic stability.
The dataset contains fourteen demographic and employment-related attributes, including age, education level, marital status, occupation, working hours, and capital gains/losses.

The dataset is imbalanced, with approximately 76% of individuals earning ≤$50K and 24% earning above this threshold. For this reason, overall accuracy is not an informative standalone metric; a model predicting only the majority class would appear strong while failing to identify higher earners. The primary evaluation metric is therefore ROC AUC, which assesses how effectively the model ranks higher-income individuals above lower-income individuals across all possible classification thresholds. This is complemented by the F1-score for the >50K class to ensure minority-class performance is not obscured.

Several limitations shape the scope of the analysis. The data reflects labour market conditions from 1994 and has limited direct applicability to contemporary contexts. The binary threshold constrains modelling flexibility and prevents regression-based approaches. As Census data is self-reported, measurement error is possible, and the cross-sectional design does not permit causal interpretation.

The workflow was designed around the Google Antigravity agent as a collaborator. The analysis was modularised into six notebooks allowing discrete components to be generated, reviewed, and integrated with minimal modification. Prompts specified evaluation criteria and modelling constraints, and outputs were manually verified through distribution checks and explicit leakage prevention (preprocessing fitted only on training data). Reproducibility was maintained using random_state = 42.

## 2. Data Exploration and Insights

Exploratory analysis on the cleaned dataset of 48,842 observations and fourteen primary features to identify structural patterns and modelling risks. Figure 1 illustrates the distribution of the income target, with approximately 76% of individuals earning ≤$50K and 24% earning above this threshold.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/eda_target_distribution.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 1:</strong> Class Imbalance
    </p>
</div>

Several features exhibit distributional characteristics with direct modelling implications. The capital-gain variable is extremely right-skewed (Figure 2): most individuals report zero gains, whilst a small minority exhibit very large values. Without transformation, such heavy tails would disproportionately influence model fitting. This motivated the use of a log(1+x) transformation during preprocessing.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/eda_capital_gain_distribution.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 2:</strong> Capital-Gain Distribution
    </p>
</div>

Education displays a clear monotonic association with income (Figure 3). The proportion of individuals earning >$50K increases steadily across education levels, suggesting strong predictive signal in both ordinal and categorical representations of education.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/eda_education_vs_income.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 3:</strong> Education vs. Income
    </p>
</div>

The distribution of hours-per-week (Figure 4) is concentrated around the standard 40-hour mark but exhibits meaningful dispersion and extreme values, indicating work intensity may contribute to income differentiation.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/eda_work_hours_worked.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 4:</strong> Hours Worked per Week Distribution
    </p>
</div>

Missing values were present in workclass, occupation, and native-country, affecting roughly 7% of instances. Their structured appearance suggested that imputation, rather than deletion, would preserve information. Although the agent generated the initial visualisations, numerical summaries and class proportions were manually verified to ensure interpretations accurately reflected the underlying data prior to finalising preprocessing decisions.

## 3. Prepare the Data

Preprocessing was implemented using a scikit-learn ColumnTransformer, enabling a modular and fully reproducible transformation pipeline. The dataset was partitioned into training (80%), validation (10%), and test (10%) subsets using stratified sampling on the income target. This preserved the approximately 76/24 class imbalance across splits. All procedures were executed with random_state=42 to ensure consistent experimental replication. The pipeline design ensures that identical transormations can be re-applied in deployment without refitting on unseen data.

Categorical features were processed through a consistent pipeline comprising imputation followed by one-hot encoding. Missing values in workclass, occupation, and native-country were replaced with a constant "Unknown" category, preserving sample size while retaining potential signal from structured absence. Given the dominance of United States observations, native-country was further simplified into a binary distinction between "United-States" and "Other" to reduce sparsity and mitigate overfitting risk.

Numeric variables were treated separately. The heavily skewed capital-gain and capital-loss features were transformed using log(1+x), as motivated by their extreme right-tailed distributions observed during EDA. Remaining numeric features were standardised to ensure comparability for gradient-based models such as logistic regression and neural networks.

To prevent leakage, all preprocessing steps were fitted exclusively on the training data, with learned parameters applied unchanged to validation and test sets. Post-transformation checks confirmed zero missing values and a stable 67-dimensional feature space. Figures 5 and 6 demonstrate class proportions and age distributions remain consistent across splits, indicating stratification preserved the underlying data structure.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/preprocessing_income_proportions.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 5:</strong> Income Distribution Across Splits
    </p>
</div>


<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/preprocessing_age_distribution.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 6:</strong> Age Distribution Across Splits
    </p>
</div>

The overall preprocessing design, including the modular pipeline architecture and log transformation of skewed capital variables, was initially suggested by the agent. Each component was reviewed prior to execution and validated through dimensionality checks, inspection of transformed outputs, and confirmation that fitting was restricted to the training set.

## 4. Model Exploration and Shortlisting

The modelling stage began with the establishment of clear performance baselines. A dummy classifier, which predicts only the majority class, defined the minimum acceptable benchmark, whilst logistic regression provided a regularised linear reference model. With L2 regularisation applied, logistic regression achieved a validation ROC AUC of approximately 0.90. This indicated a meaningful proportion of the predictive signal could be captured through linear relationships alone and provided a credible benchmark for more flexible models.

To capture potential non-linear interactions, a decision tree was introduced with structural constraints applied via hyperparameter tuning (notably min_samples_leaf). Whilst increasing modelling flexibility, ensemble methods offered superior generalisation. Random Forest and HistGradientBoosting were evaluated using validation-based comparison. Among all candidates, HistGradientBoosting achieved the strongest performance, with a validation ROC AUC of 0.9325. As illustrated in Figure 7, the gradient boosting model consistently outperformed alternative approaches.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/advanced_roc_comparison.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 7:</strong> Advanced Model ROC Curve Comparison
    </p>
</div>

Importantly, the training ROC AUC (≈0.94) exceeded validation performance only modestly (≈0.93), suggesting controlled model complexity rather than substantial overfitting. This behaviour is further supported by the learning curve shown in Figure 8, where training and cross-validation scores converge smoothly as sample size increases. The iterative nature of boosting, sequentially correcting residual errors from earlier trees, likely explains its improved discriminative ability relative to single-tree and linear models.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/advanced_learning_curve.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 8:</strong> HGB Learning Curve
    </p>
</div>

A multi-layer perceptron (MLP) was also evaluated, incorporating L2 regularisation through the alpha parameter. Although the neural network achieved a respectable ROC AUC of approximately 0.91, it did not surpass the ensemble methods. In this structured tabular setting, tree-based ensembles appeared better suited to modelling feature interactions.

Model shortlisting was based primarily on validation ROC AUC and minority-class F1-score. The test set was reserved strictly for final evaluation and remained untouched during model comparison. Feature scaling was applied consistently across all models for pipeline coherence; whilst tree-based methods do not require scaling, this unified approach simplified experimentation and did not adversely affect performance.

Model families were selected based on methodological considerations, while the agent assisted in proposing structured hyperparameter search spaces that were subsequently refined and validated.

## 5. Fine-Tune and Evaluate

The Histogram-based Gradient Boosting (HGB) model was tuned using GridSearchCV with three-fold cross-validation applied exclusively to the training partition. Hyperparameter selection was guided by maximising ROC AUC. Cross-validation was performed entirely within the training data, ensuring validation and test partitions remained unseen. The validation set was subsequently used to confirm the relative ranking of shortlisted models, whilst the test set was reserved for final performance estimation. This approach ensured performance estimates reflect genuine generalisation rather than accidental data leakage.

Final evaluation on the held-out test data demonstrates strong discriminative ability, illustrated by the confusion matrix (Figure 9). It shows that the model correctly identifies a substantial proportion of high-income individuals whilst maintaining a low false-positive rate, though some high earners remain misclassified due to overlapping demographic characteristics, reflecting the inherent difficulty of identifying all high-income individuals from cross-sectional demographic variables alone. This trade-off is consistent with the structure of the dataset and the limits of observable features. The relatively small gap between training and cross-validation ROC AUC (≈0.94 vs ≈0.93), shown previously in Section 4 (Figure 8), further indicates model complexity is controlled and performance gains are unlikely to be driven by overfitting.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/advanced_best_confusion_matrix.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 9:</strong> HGB Confusion Matrix
    </p>
</div>

To better understand the structure of the feature space, Principal Component Analysis (PCA) was conducted on the transformed dataset. The cumulative explained variance plot (Figure 10) shows the first two principal components account for approximately 28% of total variance, while around ten components are required to reach roughly 75%. This dispersion suggests predictive information is distributed across multiple interacting dimensions rather than concentrated in a small number of dominant features. In parallel, exploratory KMeans clustering did not produce clean separation aligned with income labels, reinforcing that the classification boundary is not naturally clustered in low-dimensional space. These diagnostics provide further support for the use of supervised ensemble methods capable of modelling complex interactions.

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/structure_pca_variance.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure 10:</strong> PCA Cumulative Explained Variance
    </p>
</div>

The agent-assisted workflow required active verification and correction. The handling of native-country was deliberately revised to a binary “United-States” versus “Other” representation after reviewing category sparsity and overfitting risk from agents' suggestion of having the top 3 countries and "Other" as separate categories. Additionally, limitations in automated notebook editing led to a modular structure in which agent-generated components were carefully reviewed and further modifications integrated as required. Whilst this introduced additional oversight, it strengthened traceability and ensured explicit validation at each stage of the modelling process.

## 6. Final Conclusions

This project examines binary income classification using the UCI Adult (1994 US Census) dataset. The objective is to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes. Such classification problems are common in socioeconomic research and financial risk modelling, where income level serves as a proxy for economic stability.

The workflow combined exploratory data analysis with a structured preprocessing pipeline, including targeted log transformations and categorical encoding. A range of models were evaluated, spanning logistic regression, decision trees, ensemble methods, and a neural network baseline. Performance was assessed primarily using ROC AUC and the F1-score for the minority (>50K) class to ensure robustness under class imbalance.

HistGradientBoostingClassifier achieved the strongest validation performance, with a ROC AUC of 0.9325 and an F1-score of 0.724. Ensemble methods consistently outperformed linear and single-tree approaches, indicating the importance of modelling nonlinear interactions. Exploratory Principal Component Analysis further suggested moderate high-dimensional structure within the feature space. Nevertheless, the age of the dataset and the fixed binary income threshold limit direct contemporary generalisation.

### Model Card Summary: HGB (Histogram-based Gradient Boosting)

#### Intended Use: 
Binary income classification for structured datasets similar to the UCI Adult sample; suitable for benchmarking and academic modelling.

#### Not Intended For: 
Contemporary credit, hiring, or policy decisions without retraining on updated and context-specific data.

#### Data Provenance: 
Contemporary credit, hiring, or policy decisions without retraining on updated and context-specific data.

#### Evaluation Summary:
Test ROC AUC ≈ 0.93; high precision and moderate recall for >50K class; controlled generalisation gap.

#### Key Caveats
Test ROC AUC ≈ 0.93; high precision and moderate recall for >50K class; controlled generalisation gap.

## Appendix

### Appendix A - Agent Usage and Decision Log

The Antigravity agent was used throughout the project as a structured collaborator rather than a passive code generator. Prompts were planned in advance for each stage of the workflow (EDA, preprocessing, modelling, and evaluation), with clear expectations about outputs and verification steps. A decision log was automatically generated and updated during development to record proposed transformations, modelling choices, hyperparameter ranges, and any subsequent revisions. This log was later reviewed and summarised to ensure completeness and provide a transparent audit trail.

All agent-generated plans and code were manually inspected before integration. Numerical outputs, class distributions, dimensionality checks, and evaluation metrics were cross-validated against independent summaries to confirm correctness. Where inconsistencies arose, adjustments were made and documented. The project’s modular notebook structure supported this process by isolating each analytical stage, making verification more manageable and reducing the risk of compounding errors.

| Stage | Agent Proposal | Our Decision (Accept / Modify / Reject) | Verification Performed | Final Implementation | Evidence Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EDA** | Set `na_values='?'` and `skipinitialspace=True` in loader. | Accept | Visual inspection of raw CSV files for leading spaces. | Automated missing value handling during ingestion. | 02_eda.ipynb |
| **EDA** | Skip first row of `adult.test` file. | Accept | Parsing error logs identified informal footer/header text. | `skiprows=1` applied to test set reader. | 02_eda.ipynb |
| **EDA** | Use manually defined column names from documentation. | Accept | Cross-referenced missing CSV headers with UCI repository metadata. | Standardized schema applied across all dataset splits. | 02_eda.ipynb |
| **EDA** | Include semantic column explanations in notebook. | Reject | User verification required prior to documenting feature meanings. | Removed unverified descriptions from early exploratory notebooks. | 02_eda.ipynb |
| **Repo-Workflow** | Direct in-place editing of notebook cells. | Reject | Verification showed agent claimed edits failed to persist. | Shifted to full-notebook generation to ensure consistency. | 02_eda.ipynb |
| **Preprocessing** | Normalize income labels by removing trailing periods. | Accept | Found label mismatch ('>50K.' vs '>50K') between splits. | Standardized binary target encoding for classification. | 03_preprocessing.ipynb |
| **Preprocessing** | Drop `fnlwgt` column from feature set. | Accept | Analyzed feature as census weight rather than predictive attribute. | Excluded non-predictive sampling weights to prevent distortion. | 03_preprocessing.ipynb |
| **Preprocessing** | Encode categorical missing values as "Unknown". | Accept | Frequency analysis showed missingness was non-random in attributes. | Preserves predictive signal of nulls without imputation bias. | 03_preprocessing.ipynb |
| **Preprocessing** | Retain both `education` and `education-num` columns. | Accept | Planned ablation study to identify optimal feature representation. | Both features preserved for empirical evidence testing. | 03_preprocessing.ipynb |
| **Preprocessing** | Apply `log1p` transformation to skewed capital features. | Accept | Distribution analysis confirmed extreme right skew in capital data. | Logarithmic scaling applied to gain and loss features. | 03_preprocessing.ipynb |
| **Preprocessing** | Defer robust scaling and winsorization steps. | Accept | Baseline model performance monitored for outlier sensitivity. | Minimized complexity until justified by model requirements. | 03_preprocessing.ipynb |
| **Preprocessing** | Binary binning of `native-country` (US vs Other). | Modify | Cardinality check found extreme imbalance (>90% US-based). | Initial failure refactored into robust `CountryBinner` class. | 03_preprocessing.ipynb |
| **Modelling** | Apply model-dependent scaling strategy. | Accept | Theoretical check on scale-invariance of tree-based vs. linear. | Scaling only applied to distance/linear model pipelines. | 04_models_baselines.ipynb |
| **EDA** | High-resolution figure saving with tight bounding. | Accept | Verified publication quality of generated plots in `outputs/`. | Applied `dpi=300` and `bbox_inches='tight'` globally. | 02_eda.ipynb |
| **EDA** | Create `save_and_show` standardization helper. | Accept | Code review for DRY (Don't Repeat Yourself) compliance. | Centralized plot management for cross-notebook consistency. | 02_eda.ipynb |
| **Repo-Workflow** | Remove auto-execution logic from `preprocessing.py`. | Accept | Checked module import behavior in downstream notebooks. | Converted script to definition-only module for cleaner imports. | preprocessing.py |
| **Preprocessing** | Develop dedicated 03_preprocessing.ipynb notebook. | Accept | Verified transform logic and split consistency in isolation. | Separated verification from analysis for better modularity. | 03_preprocessing.ipynb |
| **Evaluation** | Prioritize F1 and ROC AUC over Accuracy. | Accept | Analyzed target imbalance (24% minority class frequency). | Selected metrics focused on imbalanced discrimination ability. | 04_models_baselines.ipynb |
| **Modelling** | Use GridSearchCV for Decision Tree pre-pruning. | Accept | Compared cross-validation scores against baseline holdout. | Optimized tree depth and splits via systematic search. | 04_models_baselines.ipynb |
| **Modelling** | Select HistGradientBoosting for primary ensemble test. | Accept | Benchmarked against XGBoost for speed and null handling. | Implemented histogram-based gradients for final production. | 05_ensembles_and_advanced.ipynb |
| **Modelling** | Implement Grid Search with `cv=3` and `roc_auc`. | Accept | Computational resource check vs. scoring reliability. | Secured robust tuning for imbalanced class distribution. | 05_ensembles_and_advanced.ipynb |
| **Diagnostics** | Include learning curves for bias/variance check. | Accept | Examined gap between train/validation error as data scaled. | Confirmed model generalization and sufficiency of training size. | 06_exploratory_structure.ipynb |
| **Diagnostics** | Conduct PCA and KMeans exploratory analysis. | Accept | Analyzed explained variance ratios and cluster silhouette scores. | Explored unsupervised data structure for feature insight. | 06_exploratory_structure.ipynb |

### Evidence Screenshots

<div style="text-align:center; width:50%; margin: 0 auto;">
    <img src="../outputs/appendix_figures/01_project_initialisation_agent_created_structure.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure A1:</strong> Structural Project Initialisation
    </p>
</div>

<div style="text-align:center; width:50%; margin: 0 auto;">
    <img src="../outputs/appendix_figures/03_agent_struggling_with_notebook_mods_paste_in_big_block.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure A2:</strong> In-Place Notebook Editing Failure
    </p>
</div>

<div style="text-align:center; width:50%; margin: 0 auto;">
    <img src="../outputs/appendix_figures/05_asked_for_implementation_plan_agent_created_preprocessing_plan.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure A3:</strong> Agent-Generated Preprocessing Plan
    </p>
</div>

<div style="text-align:center; width:50%; margin: 0 auto;">
    <img src="../outputs/appendix_figures/09_agent_created_advanced_notebook.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure A4:</strong> Agent Created Advanced Modelling Notebook
    </p>
</div>

### Appendix B - Additional Figures

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/eda_missing_values.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure B1:</strong> Structure of Missing Values
    </p>
</div>

<div style="text-align:center; width:60%; margin: 0 auto;">
    <img src="../outputs/figures/structure_pca_scatter.png" style="width:100%;">
    <p style="text-align:left; margin-top:4px;">
        <strong>Figure B2:</strong> Scatter of Data in 2D PCA Subspace
    </p>
</div>

#### Dataset Variable Summary

| Variable        | Type        | Description |
|----------------|------------|-------------|
| age            | Numeric    | Age of the individual in years |
| workclass      | Categorical | Employment type (e.g., Private, Self-emp, Government) |
| education      | Categorical | Highest level of education achieved |
| education-num  | Numeric    | Ordinal encoding of education level |
| marital-status | Categorical | Marital status category |
| occupation     | Categorical | Occupational category |
| relationship   | Categorical | Relationship status within household |
| race           | Categorical | Self-reported race category |
| sex            | Categorical | Gender of the individual |
| capital-gain   | Numeric    | Annual capital gains |
| capital-loss   | Numeric    | Annual capital losses |
| hours-per-week | Numeric    | Number of hours worked per week |
| native-country | Categorical | Country of origin |
| income         | Binary Target | Annual income classification (>50K / ≤50K) |

## Bibliography

## References

Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science. Available at: https://archive.ics.uci.edu/dataset/2/adult.

Google Antigravity Agent Tool (2026). Agentic IDE using Gemini 3 Pro for execution. Available at: https://www.antigravity.google.

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.