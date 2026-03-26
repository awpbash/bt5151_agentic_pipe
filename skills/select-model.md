---
name: select-model
description: LLM agent selects the best model and threshold by weighing statistical performance and precision-recall tradeoffs
---

# Select Model

## When to use

After evaluate-models. This is an LLM-driven node that reasons about which model to deploy based on statistical performance, not just argmax(F1).

## How to execute

1. Receive the full metrics table and multi-threshold analysis from evaluate-models
2. Construct a prompt with:
   - Per-model metrics: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC at their F1-optimal thresholds
   - Multi-threshold breakdown: precision, recall, F1, and flag volume at 7 operating points per model
   - The SKILL.md body as the LLM's instruction set
3. Ask the LLM to select a model and threshold, considering:
   - Which model generalises best (AUC-ROC, PR-AUC as threshold-independent measures, F1 as key indicator)
   - The precision-recall tradeoff at different thresholds (higher recall catches more fraud but increases false positives)
   - Whether the flagged volume is practical (flagging 50% of all claims is not useful)
   - Early stopping behaviour as a signal of model complexity and overfitting risk
4. Parse the response for: model name, threshold, written justification
5. Fall back to F1-optimal if the LLM response cannot be parsed

## Inputs from agent state

- `results` (dict): per-model metrics at F1-optimal threshold
- `results_df` (DataFrame): formatted comparison table
- `threshold_analysis` (str): multi-threshold precision/recall/F1 breakdown
- `models` (dict): trained model objects

## Outputs to agent state

- `selected_model_name` (str): name of the chosen model
- `selected_model` (model object): the chosen model for inference
- `best_threshold` (float): the chosen operating threshold
- `selection_justification` (str): LLM-generated reasoning for the choice
- `messages` (list of str): updated processing log

## Output format

Model name as string, threshold as float, justification as 3-5 sentences explaining the statistical reasoning behind the choice.

## Notes

- The LLM should not just pick the highest F1. F1 weights precision and recall equally, but in fraud detection recall is typically more important because missing fraud has greater consequences than a false investigation.
- PR-AUC is preferred over AUC-ROC as the primary ranking metric because it is more sensitive to performance on the minority (fraud) class.
- A model that stops early (few trees) is not necessarily worse -- it may indicate the signal is learnable with a simpler model, reducing overfitting risk.
- A deterministic fallback (F1-argmax) ensures the pipeline completes even if the LLM call fails.
