# Predictive Income Analysis

## Executive Summary

This project examines binary income classification using the UCI Adult (1994 US Census) dataset. The objective is to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes. Such classification problems are common in socioeconomic research and financial risk modelling, where income level serves as a proxy for economic stability.

The workflow combined exploratory data analysis with a structured preprocessing pipeline, including targeted log transformations and categorical encoding. A range of models were evaluated, spanning logistic regression, decision trees, ensemble methods, and a neural network baseline. Performance was assessed primarily using ROC AUC and the F1-score for the minority (>50K) class to ensure robustness under class imbalance.

HistGradientBoostingClassifier achieved the strongest validation performance, with a ROC AUC of 0.9325 and an F1-score of 0.724. Ensemble methods consistently outperformed linear and single-tree approaches, indicating the importance of modelling nonlinear interactions. Exploratory Principal Component Analysis further suggested moderate high-dimensional structure within the feature space. Nevertheless, the age of the dataset and the fixed binary income threshold limit direct contemporary generalisation.

## 1. Obtain a Dataset and Frame the Predictive Problem

This project uses the UCI Adult dataset, derived from the 1994 US Census, to address a supervised binary income classification task. The objective is to predict whether an individual’s annual income exceeds $50,000 using demographic and employment-related attributes. In 1994 terms, this threshold corresponds to roughly $110,000 in 2026 dollars, meaning the classification boundary represents a comparatively high real income level. Such classification problems are common in applied socioeconomic analysis and financial risk modelling, where income acts as a proxy for economic stability.

The dataset is imbalanced, with approximately 76% of individuals earning ≤$50K and 24% earning above this threshold. For this reason, overall accuracy is not an informative standalone metric; a model predicting only the majority class would appear strong while failing to identify higher earners. The primary evaluation metric is therefore ROC AUC, which assesses how effectively the model ranks higher-income individuals above lower-income individuals across possible classification thresholds. This is complemented by the F1-score for the >50K class to ensure minority-class performance is not obscured.

Several limitations shape the scope of the analysis. The data reflects labour market conditions from 1994 and has limited direct applicability to contemporary contexts. The binary threshold constrains modelling flexibility and prevents regression-based approaches. As Census data is self-reported, measurement error is possible, and the cross-sectional design does not permit causal interpretation.

The workflow was designed around the Antigravity agent as a structured collaborator. The analysis was modularised into six notebooks to allow discrete components to be generated, reviewed, and integrated with minimal modification. Prompts specified evaluation criteria and modelling constraints in advance, and outputs were manually verified through distribution checks and explicit leakage prevention (preprocessing fitted only on training data). Reproducibility was maintained using random_state = 42.

## 2. Data Exploration and Insights

Exploratory analysis was conducted on the cleaned dataset of 48,842 observations and fourteen primary features to identify structural patterns and modelling risks. Figure 1 illustrates the distribution of the income target, with approximately 76% of individuals earning ≤$50K and 24% earning above this threshold.

![Figure 1: Class Imbalance](outputs/figures/eda_target_distribution.png)

Several features exhibit distributional characteristics with direct modelling implications. The capital-gain variable is extremely right-skewed (Figure 2): most individuals report zero gains, while a small minority exhibit very large values. Without transformation, such heavy tails would disproportionately influence model fitting. This motivated the use of a log(1+x) transformation during preprocessing.

![Figure 2: Capital-gain Distribution](outputs/figures/eda_capital_gain_distribution.png)

Education displays a clear monotonic association with income (Figure 3). The proportion of individuals earning >$50K increases steadily across education levels, suggesting strong predictive signal in both ordinal and categorical representations of education.

![Figure 3: Education Level vs Income](outputs/figures/eda_education_vs_income.png)

The distribution of hours-per-week (Figure 4) is concentrated around the standard 40-hour mark but exhibits meaningful dispersion and extreme values, indicating that work intensity may contribute to income differentiation.

![Figure 4: Hours-per-week Distribution](outputs/figures/eda_work_hours_worked.png)

Missing values were present in workclass, occupation, and native-country, affecting roughly 7% of instances. Their structured appearance suggested that imputation, rather than deletion, would preserve information. Although the agent generated the initial visualisations, numerical summaries and class proportions were manually verified to ensure that interpretations accurately reflected the underlying data prior to finalising preprocessing decisions.

## 3. Prepare the Data

Preprocessing was implemented using a scikit-learn ColumnTransformer, enabling a modular and fully reproducible transformation pipeline. The dataset was partitioned into training (80%), validation (10%), and test (10%) subsets using stratified sampling on the income target. This preserved the approximately 76/24 class imbalance across splits. All procedures were executed with random_state=42 to ensure consistent experimental replication. The pipeline design ensures that identical transormations can be re-applied in deployment without refitting on unseen data.

Categorical features were processed through a consistent pipeline comprising imputation followed by one-hot encoding. Missing values in workclass, occupation, and native-country were replaced with a constant "Unknown" category, preserving sample size while retaining potential signal from structured absence. Given the dominance of United States observations, native-country was further simplified into a binary distinction between "United-States" and "Other" to reduce sparsity and mitigate overfitting risk.

Numeric variables were treated separately. The heavily skewed capital-gain and capital-loss features were transformed using log(1+x), as motivated by their extreme right-tailed distributions observed during EDA. Remaining numeric features were standardised to ensure comparability for gradient-based models such as logistic regression and neural networks.

To prevent leakage, all preprocessing steps were fitted exclusively on the training data, with learned parameters applied unchanged to validation and test sets. Post-transformation checks confirmed zero missing values and a stable 67-dimensional feature space. Figures 5 and 6 demonstrate that class proportions and age distributions remain consistent across splits, indicating that stratification preserved the underlying data structure.

![Figure 5: Income Distribution Across Splits](outputs/figures/preprocessing_income_proportions.png)

![Figure 6: Age Distribution Across Splits](outputs/figures/preprocessing_age_distribution.png)

The overall preprocessing design, including the modular pipeline architecture and log transformation of skewed capital variables, was initially suggested by the agent. Each component was reviewed prior to execution and validated through dimensionality checks, inspection of transformed outputs, and confirmation that fitting was restricted to the training set.

## 4. Model Exploration and Shortlisting

The modelling stage began with the establishment of clear performance baselines. A dummy classifier, which predicts only the majority class, defined the minimum acceptable benchmark, while logistic regression provided a regularised linear reference model. With L2 regularisation applied to constrain coefficient magnitudes, logistic regression achieved a validation ROC AUC of approximately 0.90. This indicated that a meaningful proportion of the predictive signal could be captured through linear relationships alone and provided a credible benchmark for more flexible models.

To capture potential non-linear interactions, a decision tree was introduced with structural constraints applied via hyperparameter tuning (notably min_samples_leaf). While the single tree increased modelling flexibility, ensemble methods offered superior generalisation. Random Forest and HistGradientBoosting were evaluated using validation-based comparison. Among all candidates, HistGradientBoosting achieved the strongest performance, with a validation ROC AUC of 0.9325. As illustrated in Figure 7, the gradient boosting model consistently dominates alternative approaches across the ROC space.

Importantly, the training ROC AUC (≈0.94) exceeded validation performance only modestly (≈0.93), suggesting controlled model complexity rather than substantial overfitting. This behaviour is further supported by the learning curve shown in Figure 8, where training and cross-validation scores converge smoothly as sample size increases. The iterative nature of boosting, which sequentially corrects residual errors from earlier trees, likely explains its improved discriminative ability relative to single-tree and linear models.

A multi-layer perceptron (MLP) was also evaluated, incorporating L2 regularisation through the alpha parameter. Although the neural network achieved a respectable ROC AUC of approximately 0.91, it did not surpass the ensemble methods. In this structured tabular setting, tree-based ensembles appeared better suited to modelling feature interactions.

Model shortlisting was based primarily on validation ROC AUC and minority-class F1-score. The test set was reserved strictly for final evaluation and remained untouched during model comparison. Feature scaling was applied consistently across all models for pipeline coherence; while tree-based methods do not require scaling, this unified approach simplified experimentation and did not adversely affect performance.

The initial selection of model families and hyperparameter ranges was proposed by the agent tool. These suggestions were reviewed and deliberately constrained before execution, and final selection was determined solely by empirical validation results.

![Figure 7: Advanced Model ROC Curve Comparison](outputs/figures/advanced_roc_comparison.png)
![Figure 8: HGB Learning Curve](outputs/figures/advanced_learning_curve.png)

## 5. Fine-Tune and Evaluate

The Histogram-based Gradient Boosting (HGB) model was tuned using GridSearchCV with three-fold cross-validation applied exclusively to the training partition. Hyperparameter selection was guided by maximising ROC AUC. Cross-validation was performed entirely within the training data, ensuring that validation and test partitions remained unseen during optimisation. The validation set was subsequently used to confirm the relative ranking of shortlisted models, while the test set was reserved strictly for final performance estimation. This structured approach ensured that performance estimates reflect genuine generalisation rather than accidental data leakage.

Final evaluation on the held-out test data demonstrates strong discriminative ability, as illustrated by the confusion matrix (Figure 9). Performance on the minority (>50K) class reflects a deliberate balance between precision and recall in the context of class imbalance. The model achieves high precision, indicating that predicted high-income classifications are typically correct, while recall remains moderate, reflecting the inherent difficulty of identifying all high-income individuals from cross-sectional demographic variables alone. This trade-off is consistent with the structure of the dataset and the limits of observable features. The relatively small gap between training and cross-validation ROC AUC (≈0.94 vs ≈0.93), shown previously in Section 4 (Figure 8), further indicates that model complexity is controlled and that performance gains are unlikely to be driven by overfitting.

![Figure 9: HGB Confusion Matrix](outputs/figures/advanced_best_confusion_matrix.png)

To better understand the structure of the feature space, Principal Component Analysis (PCA) was conducted on the transformed dataset. The cumulative explained variance plot (Figure 10) shows that the first two principal components account for approximately 28% of total variance, while around ten components are required to reach roughly 75%. This dispersion suggests that predictive information is distributed across multiple interacting dimensions rather than concentrated in a small number of dominant features. In parallel, exploratory KMeans clustering did not produce clean separation aligned with income labels, reinforcing that the classification boundary is not naturally clustered in low-dimensional space. These diagnostics provide further support for the use of supervised ensemble methods capable of modelling complex interactions.

![Figure 10: PCA Cumulative Explained Variance](outputs/figures/structure_pca_variance.png)

The agent-assisted workflow required active verification and correction. The handling of native-country was deliberately revised to a binary “United-States” versus “Other” representation after reviewing category sparsity and overfitting risk from agents' suggestion of having top 3 countries and "Other" as separate categories. Additionally, limitations in automated notebook editing led to the adoption of a modular structure in which agent-generated components were carefully reviewed and further modifications integrated as required. While this introduced additional oversight, it strengthened traceability and ensured explicit validation at each stage of the modelling process.

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