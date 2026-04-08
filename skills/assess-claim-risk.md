---
name: assess-claim-risk
description: LLM agent explains why the claim was flagged using model output and claim details
---

# Assess Claim Risk

## When to use

After score-fraud-risk. The LLM analyses the claim details alongside the model's prediction to produce a human-readable risk explanation.

## How to execute

1. Map probability to risk level: LOW (below threshold), MEDIUM (threshold to 0.5), HIGH (0.5 to 0.8), CRITICAL (above 0.8)
2. Construct a prompt with:
   - All claim fields in readable format
   - Model confidence and threshold
   - Top 5 feature contributions from score-fraud-risk
   - Instructions to explain the risk in plain English for a claims adjuster
3. Send to GPT-4o-mini (temperature=0.2, max_tokens=250)
4. The explanation should identify which specific claim characteristics contributed to the risk score

## Inputs from agent state

- `probability` (float): from score-fraud-risk
- `prediction` (int): from score-fraud-risk
- `claim` (dict): original claim fields
- `feature_contributions` (dict): top features from score-fraud-risk

## Outputs to agent state

- `risk_level` (str): LOW, MEDIUM, HIGH, or CRITICAL
- `risk_explanation` (str): 2-4 sentence explanation for a claims adjuster
- `messages` (list of str): updated processing log

## Output format

Risk level as one of four string categories. Explanation as natural language suitable for display in the Gradio interface. The explanation should reference specific claim details (e.g. "no police report was filed" rather than "PoliceReportFiled=0"). A business user should be able to understand the explanation and take actions on it, do not include model related technical terms. 

## Notes

- The LLM translates encoded feature values back into business language
- If the API is unavailable, returns a fallback with the raw probability
- Temperature is low (0.2) for consistent, factual explanations
