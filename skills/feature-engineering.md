---
name: feature-engineering
description: Engineer 3 new features from encoded data, split into train/val/test, fit scaler
---

# Feature Engineering

## When to use

After preprocess-data. Creates derived features that capture known fraud patterns, then splits the data.

## How to execute

1. Create `claim_delay` = MonthClaimed - Month (clipped at 0). Longer delays correlate with fraud (3+ months: ~10% vs 0 months: 5.6%).
2. Create `nonstandard_deductible` = binary flag for Deductible != 400. Non-standard deductibles have 11.5% fraud vs 5.8%.
3. Create `fault_x_basepolicy` = Fault * 10 + BasePolicy. Encodes the interaction: Policy Holder + All Perils = 15.6% fraud, Third Party + anything = ~1%.
4. Define feature columns (28 total: 25 from preprocessing + 3 engineered).
5. Stratified 70/15/15 train/val/test split preserving 6% fraud rate.
6. Fit StandardScaler on training data for Logistic Regression.
7. Compute per-feature training means for feature contribution explanations at inference.

## Inputs from agent state

- `df` (DataFrame): encoded data from preprocess-data
- `label_encoders` (dict): from preprocess-data

## Outputs to agent state

- `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` (numpy arrays)
- `X_train_scaled`, `X_val_scaled`, `X_test_scaled` (numpy arrays)
- `feature_cols` (list of str): 28 feature names in order
- `scaler` (StandardScaler)
- `train_feature_means` (dict): per-feature mean from training set

## Output format

Numpy arrays for X (float64, shape [n, 28]) and y (int, shape [n]).

## Notes

- The interaction feature `fault_x_basepolicy` helps trees that might not discover this 2-way split efficiently on their own
- `claim_delay` captures a real behavioral signal: fraudsters tend to file later
- All 3 engineered features appear in the top 10 correlations with the target
