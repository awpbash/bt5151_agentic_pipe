# Insurance Claim Fraud Detection Pipeline

BT5151 Advanced Analytics and Machine Learning -- Group Project

## What this is

A LangGraph multi-agent pipeline that detects fraudulent insurance claims. The pipeline trains 3 ML models (XGBoost, LightGBM, Logistic Regression), has an LLM agent select the best model by reasoning about precision-recall tradeoffs, then scores incoming claims with LLM-generated risk explanations and recommended actions via a Gradio dashboard.

## Dataset

Vehicle Insurance Fraud Oracle (Kaggle) -- 15,420 claims, 33 features, 6% fraud rate.

## Pipeline

```
Training (runs once):
  preprocess-data -> feature-engineering -> train-models -> evaluate-models -> select-model (LLM)

Inference (per claim via Gradio):
  score-fraud-risk -> assess-claim-risk (LLM) -> [conditional] -> recommend-action (LLM)
```

- 5 deterministic nodes (data prep, feature engineering, training, evaluation, scoring)
- 3 LLM nodes (model selection, risk assessment, action recommendation)
- 1 conditional edge (skip recommend-action for LOW risk claims)

## How to run

1. Open `notebook.ipynb` in Google Colab
2. Add your OpenAI API key to Colab Secrets as `OPENAI_API_KEY`
3. Run All -- the training pipeline runs via `training_app.invoke({})`, then Gradio launches with `share=True`

## Project structure

```
notebook.ipynb          -- single notebook, runs end-to-end in Colab
report.md               -- technical report (2,500-3,000 words)
skills/                 -- 8 SKILL.md files (one per pipeline node)
  preprocess-data.md
  feature-engineering.md
  train-models.md
  evaluate-models.md
  select-model.md
  score-fraud-risk.md
  assess-claim-risk.md
  recommend-action.md
data/
  fraud_oracle.csv      -- dataset
```

## Key design decisions

- **Feature curation over blind usage** -- dropped 7 redundant/noisy columns, engineered 3 new features (claim_delay, nonstandard_deductible, fault_x_basepolicy). Final: 28 features.
- **Three-tier encoding** -- ordinal for ordered categoricals (preserves tree split logic), binary for yes/no columns (consistent fraud direction), label encoding for true nominals.
- **LLM model selection** -- the select-model agent reasons about PR-AUC, precision-recall tradeoffs, flag volume, and overfitting risk across multiple operating points, not just argmax(F1).
- **SKILL.md as prompt specs** -- YAML frontmatter parsed at runtime, body injected into LLM prompts. Skills are not just documentation.
- **Conditional routing** -- LOW risk claims skip the recommend-action LLM call, saving API cost on the 94% of claims that are legitimate.
