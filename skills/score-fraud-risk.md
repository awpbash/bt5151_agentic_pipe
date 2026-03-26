---
name: score-fraud-risk
description: Score a single claim through the selected model and extract top feature contributions
---

# Score Fraud Risk

## When to use

At inference time when a new claim arrives via Gradio. This is the deterministic scoring step.

## How to execute

1. Receive a raw claim dict with original categorical values (e.g. 'Honda', 'Male', 'Urban')
2. Encode each field using the ordinal maps, binary maps, and label encoders fitted during preprocessing
3. Compute 3 engineered features (claim_delay, nonstandard_deductible, fault_x_basepolicy)
4. Build a 28-feature vector in the same column order as training
5. Run `selected_model.predict_proba()` to get fraud probability
6. Compare against the threshold chosen by select-model
7. Extract top 5 feature contributions by model importance, showing each feature's value and how it compares to training mean

## Inputs from agent state

- `claim` (dict): raw claim fields with original categorical string values

## Outputs to agent state

- `probability` (float): fraud probability from the model
- `prediction` (int): 0 or 1 based on the selected threshold
- `feature_contributions` (dict): top 5 features with values and deviation from training mean
- `messages` (list of str): updated processing log

## Output format

Probability as float in [0, 1]. Prediction as 0 or 1. Feature contributions as dict mapping feature name → string like "3.00 (1.5x avg)".

## Notes

- Unknown categories not seen during training are mapped to 0 as a safe fallback
- The label encoders are the same ones fitted in preprocess-data -- never re-fitted
- If the selected model is Logistic Regression, the feature vector is scaled using the StandardScaler from feature-engineering before prediction. Tree models use unscaled features.
- This node is deterministic -- no LLM call, pure model scoring
