---
name: evaluate-models
description: Compute all metrics across multiple thresholds for each model on held-out test data
---

# Evaluate Models

## When to use

After train-models. Evaluates all three candidates at multiple operating points.

## How to execute

1. For each model, predict probabilities on validation and test sets
2. Compute F1-optimal threshold from validation precision-recall curve
3. Evaluate on test set at that threshold: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC
4. Compute precision, recall, F1, and flag count at 7 thresholds per model for the select-model agent to reason over
5. Generate visualisations: ROC curves, PR curves, confusion matrix, feature importances, threshold sensitivity

## Inputs from agent state

- `models` (dict): three trained model objects
- `X_val`, `y_val`, `X_test`, `y_test` (numpy arrays)
- `X_val_scaled`, `X_test_scaled` (numpy arrays)
- `feature_cols` (list of str)

## Outputs to agent state

- `results` (dict): per-model metrics at F1-optimal threshold
- `results_df` (DataFrame): formatted comparison table
- `threshold_analysis` (str): formatted summary of each model's precision, recall, F1 at multiple thresholds
- `test_probs_all` (dict): model name → test set probabilities
- `messages` (list of str): updated processing log

## Output format

Results as a dict of dicts (model name → metric name → float). The threshold_analysis is a formatted string table designed to be passed directly to the select-model LLM agent.

## Notes

- Threshold tuning is done on validation set to avoid test-set leakage
- PR-AUC is more informative than AUC-ROC for imbalanced datasets
- The threshold_analysis shows how each model's precision and recall shift across operating points, giving the select-model agent the full picture beyond a single F1-optimal number
