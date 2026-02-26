# Executive Summary

This project examines binary income classification using the UCI Adult (1994 US Census) dataset. The objective is to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes. Such classification problems are common in socioeconomic research and financial risk modelling, where income level serves as a proxy for economic stability.

The workflow combined exploratory data analysis with a structured preprocessing pipeline, including targeted log transformations and categorical encoding. A range of models were evaluated, spanning logistic regression, decision trees, ensemble methods, and a neural network baseline. Performance was assessed primarily using ROC AUC and the F1-score for the minority (>50K) class to ensure robustness under class imbalance.

HistGradientBoostingClassifier achieved the strongest validation performance, with a ROC AUC of 0.9325 and an F1-score of 0.724. Ensemble methods consistently outperformed linear and single-tree approaches, indicating the importance of modelling nonlinear interactions. Exploratory Principal Component Analysis further suggested moderate high-dimensional structure within the feature space. Nevertheless, the age of the dataset and the fixed binary income threshold limit direct contemporary generalisation.

## 1. Obtain a Dataset and Frame the Predictive Problem

### 1.1 Dataset Selection and Problem Definition
This project uses the UCI Adult dataset, derived from the 1994 US Census, to address a supervised binary income classification task. The objective is to predict whether an individual’s annual income exceeds $50,000 using demographic and employment-related attributes. In 1994 terms, this threshold equates to roughly $110,000 in 2026 dollars, indicating that the classification boundary represents a comparatively high real income level. Income classification problems of this type are common in socioeconomic analysis and risk modelling, where income serves as a proxy for economic stability.

### 1.2 Target Variable and Success Metrics
The problem is formulated as supervised binary classification. Given the class imbalance in the dataset (approximately 76% ≤50K vs 24% >50K), overall accuracy alone is not an informative measure of performance. The primary evaluation metric is ROC AUC, which measures how well the model distinguishes between the two classes across all possible classification thresholds rather than relying on a single cut-off. This is complemented by the F1-score for the >50K class to balance precision and recall and ensure minority-class performance is adequately captured.

### 1.3 Assumptions and Limitations
Several limitations are inherent in the dataset. First, the data reflects labour market conditions from 1994 and therefore has limited direct applicability to contemporary economic environments. Second, the fixed binary income threshold restricts modelling flexibility and prevents more granular regression-based analysis. Third, as Census data is self-reported, measurement error is possible. The dataset provides only a cross-sectional snapshot and does not include information on income or employment trajectories over time. Finally, the observational structure of the data prevents causal interpretation.

### 1.4 Agent Planning and Verification Strategy
The project was conducted using the Antigravity agent tool as a collaborative assistant. The agent was unable to reliably edit .ipynb files in place, requiring code to be generated separately and inserted manually. As a result, the workflow was modularised into six structured notebooks, each corresponding to a defined analytical stage. Prompts were rigorously planned in advance, with explicit specification of model structure, evaluation metrics, and hyperparameter boundaries. Sanity checks included verifying class distributions, confirming skew in capital-gain variables prior to transformation, and ensuring no unexpected missing values remained after preprocessing. Outputs were manually validated through EDA sanity checks, metric verification, and explicit prevention of data leakage by fitting preprocessing exclusively on training data. Reproducibility was maintained through a fixed random_state = 42 across all experiments.

## 2. Explore the Data to Gain Insights

Exploratory data analysis was conducted on the cleaned dataset comprising 48,842 observations and fourteen primary features. The aim was to identify structural characteristics, predictive signals, and potential modelling risks prior to formal preprocessing.

### 2.1 Class Imbalance and Target Distribution
Figure 1 displays the distribution of the income target variable. Approximately 76% of individuals earn ≤$50,000, while 24% exceed this threshold. This imbalance presents a clear modelling risk: a classifier predicting only the majority class would achieve superficially high accuracy while failing to meaningfully identify high-income individuals. The imbalance therefore motivated the prioritisation of ROC AUC and minority-class F1-score in subsequent evaluation.

![Figure 1: Class Imbalance](outputs/figures/eda_target_distribution.png)

### 2.2 Distributional Characteristics and Feature Signals
The capital-gain variable exhibited extreme right-skewness (Figure 2). The majority of individuals reported zero capital gains, while a small subset displayed very large positive values. This heavy-tailed structure indicated that raw-scale modelling would be dominated by outliers, motivating the use of a log(1+x) transformation during preprocessing.

![Figure 2: Capital-gain Distribution](outputs/figures/eda_capital_gain_distribution.png)

Figure 3 demonstrates a clear monotonic relationship between education level and the proportion of individuals earning >$50K. As education increases, so does the relative frequency of high-income outcomes. This pattern suggests substantial predictive signal within education-related features.

![Figure 3: Education Level vs Income](outputs/figures/eda_education_vs_income.png)

The distribution of hours-per-week (Figure 4) shows strong concentration around the standard 40-hour work week, alongside meaningful dispersion and extreme values. This variation suggests that working hours may contribute to income differentiation but requires careful scaling treatment.

![Figure 4: Hours-per-week Distribution](outputs/figures/eda_work_hours_worked.png)

### 2.3 Data Quality and Modelling Considerations
Missing values were observed in workclass, occupation, and native-country, affecting approximately 7% of records. These appeared systematic rather than random but were not sufficiently extensive to justify row deletion.

While the agent generated the initial EDA visualisations, numerical summaries and class proportions were manually cross-checked to ensure that graphical interpretations accurately reflected the underlying data before informing preprocessing decisions.

