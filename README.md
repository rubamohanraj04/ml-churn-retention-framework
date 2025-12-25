# Integrating Machine Learning with Customer Retention Strategies: A Risk-Based Approach to E-Commerce Churn Prediction

This repository contains the code and dissertation materials for an MSc Business Analytics project on integrating machine-learning-based churn prediction with risk-based segmentation and value-driven retention strategy in an e-commerce context.

## Project overview

The project builds an end-to-end analytics pipeline that:

- Uses supervised machine learning models (Logistic Regression, Random Forest, Gradient Boosting) to predict customer churn.
- Applies a consistent preprocessing pipeline (missing-value imputation, scaling, one-hot encoding).
- Converts churn probabilities into interpretable risk bands (Low / Medium / High).
- Builds behavioural and RFM-style clusters using K-Means.
- Constructs a simple value proxy (`ValueProxy`) to approximate customer lifetime value (CLV) from order frequency, cashback and spend growth.
- Combines risk, behaviour and value into actionable segments to support targeted retention decisions.

The work supports the dissertation:  
**“Integrating Machine Learning with Customer Retention Strategies: A Risk-Based Approach to E-Commerce Churn Prediction.”**

## Repository structure

- `notebooks/` – Jupyter notebook(s) implementing the full modelling pipeline:
  - data loading and cleaning  
  - feature engineering and preprocessing  
  - model training, evaluation and comparison  
  - risk-band creation, clustering and value proxy construction  
  - generation of tables and figures used in the dissertation
- `data/` – Location for the churn dataset (not committed if confidential).
- `reports/` – Dissertation PDF and any supporting technical documentation.
- `figures/` – Exported plots (confusion matrix, feature importance, elbow plot, cluster profiles, risk distributions, etc.).

## Data

The project uses an e-commerce customer churn dataset with 5,630 customers and 20 attributes, including behavioural, transactional, experiential and demographic variables (e.g. `Tenure`, `CityTier`, `OrderCount`, `CouponUsed`, `HourSpendOnApp`, `SatisfactionScore`).

Source dataset:

> hashexplaindata (2025) *Customer Churn Dataset* [online]. Available at:  
> GitHub repository `hashexplaindata/e-commerce_customer_churn_analysis` 

Save the CSV as, for example:

- `data/customer_churn_data.csv`

and update the file path in the notebook if needed.

## Methods

1. **Preprocessing**
   - Numeric features: median imputation + standardisation.
   - Categorical features: most-frequent imputation + one-hot encoding.
   - Implemented via `ColumnTransformer` and `Pipeline` in scikit-learn.

2. **Modelling**
   - Train/test split (80/20, stratified on churn).
   - Models: Logistic Regression, Random Forest, Gradient Boosting.
   - Metrics: Accuracy, Precision, Recall, F1-score, ROC AUC.

3. **Risk-based segmentation**
   - Convert predicted churn probabilities into Low / Medium / High risk bands.
   - Apply K-Means clustering on RFM-style and behavioural features.
   - Engineer `ValueProxy` and define Low / Mid / High value tiers.
   - Cross risk, clusters and value to derive priority segments (e.g. high-value & high-risk).

## Reproducibility

- All random seeds are fixed (`random_state=42`) for model training and clustering.
- The full pipeline is implemented using scikit-learn, NumPy and pandas.
- See the main notebook in `notebooks/` for step-by-step execution.

## Citation

- Mohanraj, R. (2025) *Integrating Machine Learning with Customer Retention Strategies: A Risk-Based Approach to E-Commerce Churn Prediction*. MSc dissertation, University of Greenwich.
- hashexplaindata (2025) *Customer Churn Dataset* [online]. GitHub: `hashexplaindata/e-commerce_customer_churn_analysis`.
