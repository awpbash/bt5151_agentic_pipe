---
name: train-models
description: Train XGBoost, LightGBM, and Logistic Regression with class weight tuning
---

# Train Models

## When to use

After feature-engineering. Trains three candidate models for comparison.

## How to execute

1. Sweep `scale_pos_weight` in [1, 3, 5, 10, 15] using quick XGBoost (100 trees) on validation F1
2. Select the weight that maximises validation F1
3. Train XGBoost: 500 trees, max_depth=6, lr=0.05, early stopping on AUCPR (patience=20)
4. Train LightGBM: 500 trees, max_depth=6, lr=0.05, num_leaves=31, early stopping on binary_logloss (patience=20)
5. Train Logistic Regression: saga solver, max_iter=200, class_weight applied

## Inputs from agent state

- `X_train`, `y_train`, `X_val`, `y_val` (numpy arrays)
- `X_train_scaled` (numpy array): for Logistic Regression

## Outputs to agent state

- `models` (dict): {'XGBoost': model, 'LightGBM': model, 'Logistic Regression': model}
- `best_weight` (int): selected scale_pos_weight
- `messages` (list of str): updated processing log

## Output format

A dict mapping model name strings to fitted sklearn-compatible objects with `predict_proba()`.

## Notes

- All three models use the same class weight for fair comparison
- XGBoost and LightGBM use unscaled features; LR uses scaled
- Early stopping prevents overfitting and reports optimal tree count
