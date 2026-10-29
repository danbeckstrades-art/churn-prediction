# Customer Churn Prediction

Predicting which telecom customers are likely to churn, with an emphasis on honest evaluation: leakage-safe preprocessing, paired model comparison, a cost-based decision threshold, and a test set used exactly once.

## Problem
Retention offers cost money, so they should go to the customers most likely to leave. The model ranks customers by churn risk; a threshold chosen from an explicit cost assumption turns that ranking into an outreach list.

## Data
[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle, BlastChar): 7,043 customers, 21 columns, 26.5% churn rate. The CSV is not included; download it and save it as `data/telco_churn.csv`.

## Method
- **Cleaning:** `TotalCharges` was stored as text. 11 blank values, all with `tenure = 0` (new customers not yet billed), were set to 0. `customerID` was dropped.
- **Split:** 80/20 stratified train/test split made before any modeling. The test set was used once, at the end.
- **Metrics:** ROC-AUC for ranking, precision and recall for the outreach list. Accuracy is reported but not used for decisions, because always predicting "no churn" scores 73.5% while catching no one.
- **Preprocessing:** scaling and one-hot encoding live inside a scikit-learn `Pipeline`, so they are fit within each cross-validation fold (no leakage).
- **Model comparison:** 5-fold stratified CV on the training data with identical folds, plus paired per-fold differences.
- **Threshold:** chosen from out-of-fold predictions by maximizing net value under cost assumptions.

## Results

| Model (5-fold CV, training data) | ROC-AUC |
|---|---|
| Always predict "no churn" | 0.500 |
| Logistic regression | 0.846 +/- 0.014 |
| Random forest | 0.844 +/- 0.010 |
| Gradient boosting | 0.848 +/- 0.012 |

Paired fold differences against logistic regression flip sign and are smaller than the fold-to-fold noise, so the three models are indistinguishable on this data. I chose logistic regression for simplicity and interpretability.

**Held-out test set (used once), threshold 0.35:**
- ROC-AUC 0.842 (95% bootstrap CI 0.818 to 0.863), consistent with the CV estimate of 0.846
- Precision 0.544, recall 0.706; 485 of 1,409 customers flagged (34.4%)

**Drivers (logistic regression):** longer tenure and two-year contracts reduce churn risk; month-to-month contracts, fiber optic internet, and paperless billing increase it; customers without internet service churn least.

## Limitations
- The cost figures used to pick the threshold (offer cost 20, save value 200, 30% acceptance) are illustrative placeholders. The break-even formula `offer_cost / (accept_rate * save_value)` carries over; the dollar results do not.
- Coefficients are associations in observational data, not causal effects. `tenure`, `MonthlyCharges`, and `TotalCharges` overlap heavily, so those individual coefficients should not be interpreted in isolation.
- The data is a single static snapshot with no time dimension, so I could not test how the model holds up as customer behavior changes.
- The bootstrap interval reflects test-sample variability only, not variation from retraining.

## Reproduce
Place the CSV at `data/telco_churn.csv`, then run `notebooks/01_eda.ipynb` followed by `notebooks/02_baselines.ipynb`. Shared cleaning code lives in `src/data.py`. Random seeds are fixed at 42.
