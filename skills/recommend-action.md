---
name: recommend-action
description: LLM agent prescribes the next step for the claims team based on risk level and claim details
---

# Recommend Action

## When to use

After assess-claim-risk. This is the final downstream node -- it tells the claims team what to DO, not just what the risk is.

## How to execute

1. Check the risk level from assess-claim-risk
2. Construct a prompt with the claim details, risk level, risk explanation, and feature contributions
3. Ask GPT-4o-mini to recommend a specific action based on the risk tier:
   - **LOW**: auto-approve the claim, no further review needed
   - **MEDIUM**: flag for senior adjuster review, specify which aspects to verify
   - **HIGH**: assign to Special Investigations Unit (SIU), outline specific investigation steps
   - **CRITICAL**: immediate escalation, freeze claim payment, recommend field investigation
4. The recommendation should be specific to THIS claim, not generic

## Inputs from agent state

- `risk_level` (str): from assess-claim-risk
- `risk_explanation` (str): from assess-claim-risk
- `claim` (dict): original claim fields
- `probability` (float): from score-fraud-risk
- `feature_contributions` (dict): from score-fraud-risk

## Outputs to agent state

- `recommended_action` (str): specific action recommendation with steps
- `messages` (list of str): updated processing log

## Output format

A structured recommendation (3-6 sentences) with a clear action verb at the start (e.g. "Approve this claim..." or "Escalate to SIU..."). Should include specific investigation steps for HIGH/CRITICAL claims referencing the actual claim details.

## Notes

- LOW claims get a short auto-approve message (saves LLM tokens for the common case)
- HIGH/CRITICAL recommendations reference specific red flags from the claim (e.g. "verify the Ferrari was at the insured address" rather than generic "investigate further")
- Temperature is 0.1 for HIGH/CRITICAL (formal, consistent) and 0.3 for MEDIUM (slightly more flexible)
- If the API fails, returns a fallback recommending manual review
