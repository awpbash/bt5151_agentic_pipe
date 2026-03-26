# Technical Report: Insurance Claim Fraud Detection Pipeline

**BT5151 Advanced Analytics and Machine Learning -- Group Project**

Video presentation link: [TODO: insert link]

---

## 1. System Architecture

### Pipeline Overview

Our system is a LangGraph multi-agent pipeline with 8 nodes, each governed by its own SKILL.md file. The pipeline splits into two phases: a training phase (5 nodes, executed once via `training_app.invoke({})`) and an inference phase (3 nodes, executed per-claim via Gradio).

```
Training phase:
  preprocess-data -> feature-engineering -> train-models -> evaluate-models -> select-model (LLM)

Inference phase (per claim):
  score-fraud-risk -> assess-claim-risk (LLM) -> [conditional] -> recommend-action (LLM)
                                                       |
                                                  LOW -> END
```

The training pipeline runs end-to-end in a single `invoke()` call. Each node reads from and writes to agent state, so data flows automatically between stages. The inference sub-pipeline is triggered per claim from the Gradio dashboard.

### Node Descriptions

1. **preprocess-data** -- Loads the Fraud Oracle CSV, drops 7 redundant/low-signal columns (PolicyNumber, Age, PolicyType, WeekOfMonth, WeekOfMonthClaimed, DriverRating, Year), fixes placeholder '0' values, then applies a three-tier encoding strategy: ordinal encoding for 9 ordered categoricals, binary encoding for 6 yes/no columns, and label encoding for remaining nominals. This node does the actual work inside the pipeline, not as a wrapper.

2. **feature-engineering** -- Engineers 3 derived features (claim_delay, nonstandard_deductible, fault_x_basepolicy interaction), defines the 28-feature column list, performs a stratified 70/15/15 train/val/test split, fits a StandardScaler on training data, and computes per-feature training means for inference explanations.

3. **train-models** -- Sweeps `scale_pos_weight` in [1, 3, 5, 10, 15] on a quick XGBoost to find the optimal class imbalance correction, then trains three candidates: XGBoost (500 trees, early stopping on AUCPR), LightGBM (500 trees, early stopping on binary log-loss), and Logistic Regression (saga solver with class weights).

4. **evaluate-models** -- For each model, finds the F1-optimal threshold on the validation precision-recall curve and evaluates on the held-out test set. Reports Accuracy, Precision, Recall, F1, AUC-ROC, and PR-AUC. Also computes a multi-threshold cost analysis table designed for the select-model LLM to reason over: at each threshold, it estimates daily flagged volume, missed fraud count, false investigation cost, and missed fraud cost.

5. **select-model (LLM)** -- Receives the full metrics table and multi-threshold analysis from evaluate-models. GPT-4o-mini reasons about which model and threshold to select based on PR-AUC, precision-recall tradeoff, flag volume, and overfitting risk. Falls back to F1-argmax if the LLM response cannot be parsed.

6. **score-fraud-risk** -- Deterministic scoring node. Receives a raw claim dict, encodes it using the label encoders from preprocess-data, runs `predict_proba` on the selected model, and extracts the top 5 feature contributions by importance.

7. **assess-claim-risk (LLM)** -- Maps the fraud probability to a risk level (LOW / MEDIUM / HIGH / CRITICAL) and sends the claim details plus model output to GPT-4o-mini. The LLM generates a 2-3 sentence explanation referencing specific claim details (e.g. "no police report was filed") rather than raw feature names.

8. **recommend-action (LLM)** -- Conditionally triggered for MEDIUM, HIGH, and CRITICAL claims. The LLM prescribes a specific next step: senior adjuster review for MEDIUM, SIU assignment with investigation steps for HIGH, immediate escalation and payment freeze for CRITICAL. LOW claims get an auto-approve response without an LLM call.

### How SKILL.md Files Govern Execution

Each SKILL.md file has YAML frontmatter (name, description) and a body with sections: When to use, How to execute, Inputs from agent state, Outputs to agent state, Output format, and Notes.

At runtime, the notebook parses frontmatter using PyYAML and loads the body separately. The frontmatter `name` and `description` are used for logging. The body is injected into the LLM prompt for the three agentic nodes (select-model, assess-claim-risk, recommend-action). This means the SKILL.md files are not just documentation -- they are the actual prompt specifications that drive LLM behaviour.

### Conditional Routing

The edge between assess-claim-risk and recommend-action uses LangGraph's `add_conditional_edges`. The routing function checks `state["risk_level"]`: LOW claims exit the pipeline immediately (no LLM call needed), while MEDIUM/HIGH/CRITICAL claims proceed to recommend-action. This saves API cost and latency for the ~94% of claims that are legitimate.

---

## 2. Business Case

### Problem Context

Insurance fraud is a significant problem for the industry. Fraudulent claims inflate premiums for honest policyholders and consume investigator resources. The challenge is that fraud is rare (typically 5-10% of claims), so manual review of every claim is impractical. Investigators need a system that surfaces the most suspicious claims for prioritised review.

### Problem Being Solved

We build an automated fraud screening system that scores incoming insurance claims and produces actionable output for claims adjusters. Rather than presenting raw model probabilities, the system delivers colour-coded risk levels, plain-language explanations of why a claim was flagged, and specific recommended actions (auto-approve, flag for review, assign to SIU, or escalate).

### Expected Value

- **Efficiency** -- Investigators focus on claims flagged as MEDIUM or above, reducing the initial review pool by over 90%.
- **Consistency** -- Every claim is scored against the same model with the same threshold, eliminating human bias in initial screening.
- **Transparency** -- The LLM-generated explanations and feature contributions provide an audit trail for why a claim was flagged. The select-model justification documents why a particular model was chosen.
- **Scalability** -- Model scoring takes milliseconds per claim. LLM calls add 1-2 seconds but only fire for flagged claims.

### Stakeholders

- **Claims adjusters** -- Primary users of the Gradio dashboard. They receive risk assessments and recommended actions for each incoming claim. The interface is designed so they never need to understand model internals.
- **SIU (Special Investigations Unit)** -- Receives HIGH/CRITICAL flagged claims with specific, claim-tailored investigation steps generated by the recommend-action agent.
- **Compliance and audit** -- Can review the SKILL.md governance layer, the model selection justification, and the feature contribution explanations to verify that flagging decisions are explainable and non-discriminatory.
- **Policyholders** -- Indirect beneficiaries of reduced fraud, which keeps premiums lower and speeds up legitimate claim processing.

---

## 3. Data and ML Methodology

### Dataset Description

The Vehicle Insurance Fraud Oracle dataset (Kaggle) contains 15,420 insurance claims from 1994-1996 with 33 features and a binary fraud label (FraudFound_P). The fraud rate is 6.0% (923 fraudulent out of 15,420). Features cover policyholder demographics (Age, Sex, MaritalStatus), vehicle information (Make, VehicleCategory, VehiclePrice, AgeOfVehicle), claim circumstances (AccidentArea, Fault, PoliceReportFiled, WitnessPresent, PastNumberOfClaims), and policy details (PolicyType, Deductible, BasePolicy, DriverRating).

### Preprocessing Steps

1. **Dropped 7 columns** -- PolicyNumber (ID), Age (r=0.97 with AgeOfPolicyHolder, uses 0 as placeholder for minors), PolicyType (r=0.86 with VehicleCategory, redundant composite), WeekOfMonth/WeekOfMonthClaimed (fraud spread <0.012, noise), DriverRating (fraud spread 0.006), Year (only 3 values, dataset artifact).
2. **Fixed placeholder values** -- DayOfWeekClaimed and MonthClaimed contained '0' entries; replaced with 'Unknown'.
3. **Three-tier encoding** -- Ordinal encoding for 9 columns with natural order (e.g. AgeOfPolicyHolder, VehiclePrice), binary encoding for 6 yes/no columns with consistent fraud direction (e.g. PoliceReportFiled: Yes=0, No=1), and label encoding for remaining nominals (Make, Month, DayOfWeek, etc.).
4. **Feature engineering** -- 3 derived features: `claim_delay` (months between accident and claim, longer = more suspicious), `nonstandard_deductible` (11.5% fraud vs 5.8% for standard), `fault_x_basepolicy` (Policy Holder + All Perils = 15.6% fraud).
5. **Stratified split** -- 70% train (10,800 claims), 15% validation (2,307 claims), 15% test (2,313 claims), preserving the 6% fraud rate in each set.

### Candidate Models

| Model | Type | Key Hyperparameters |
|-------|------|-------------------|
| XGBoost | Gradient boosted trees | 500 trees, max_depth=6, lr=0.05, early stopping on AUCPR |
| LightGBM | Gradient boosted trees | 500 trees, max_depth=6, lr=0.05, num_leaves=31, early stopping on binary log-loss |
| Logistic Regression | Linear model | saga solver, max_iter=200, class_weight |

All three models used the same `scale_pos_weight` selected via validation F1 sweep over [1, 3, 5, 10, 15]. The sweep selected weight=1.

### Training Observations

XGBoost stopped at around 20 trees and LightGBM at around 63 trees, both before the 500-tree cap. This indicates the fraud signal in this dataset is relatively shallow -- the models converge quickly because the decision boundaries are learnable with few splits. This also explains why Logistic Regression is competitive: the signal is approximately linear.

### Model Selection

The select-model node is LLM-driven. GPT-4o-mini receives the full metrics table and a multi-threshold analysis showing precision, recall, F1, and flag volume at 7 operating points per model. The LLM reasons about which combination of model and threshold gives the best tradeoff between catching fraud (recall) and not overwhelming reviewers with false positives (precision), then provides a written justification.

A deterministic fallback (F1-argmax) ensures the pipeline never fails if the LLM call errors.

---

## 4. Model Evaluation

### Results Table

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|-------|-----------|----------|-----------|--------|-----|---------|--------|
| XGBoost | 0.121 | 0.885 | 0.211 | 0.341 | 0.260 | 0.811 | 0.204 |
| LightGBM | 0.131 | 0.843 | 0.172 | 0.428 | 0.245 | 0.802 | 0.189 |
| Logistic Regression | 0.139 | 0.843 | 0.156 | 0.370 | 0.220 | 0.800 | 0.144 |

Note: The final model and threshold may differ from this table because the LLM select-model agent reasons about the full precision-recall tradeoff, not just the F1-optimal point.

### Visualisations

The notebook produces four evaluation plots after the training pipeline completes:
1. **ROC curves** -- All three models achieve AUC-ROC around 0.80-0.81.
2. **Precision-Recall curves** -- PR-AUC ranges from 0.16 to 0.21, reflecting the 6% base rate.
3. **Confusion matrix** -- Shows the precision-recall tradeoff at the LLM-selected threshold.
4. **Feature importances** -- Top features include BasePolicy, Fault, VehicleCategory, fault_x_basepolicy (engineered), and nonstandard_deductible (engineered).

### Critical Interpretation

**Why trees converge quickly:** XGBoost stops around 20 trees and LightGBM around 63 trees, suggesting the fraud signal is relatively shallow. The top correlated features (BasePolicy at 0.16, Fault at 0.13) have modest individual correlations, and the model combines many weak signals rather than finding complex non-linear interactions.

**Precision-recall tradeoff:** At the F1-optimal threshold, XGBoost catches 34% of fraud (recall) with 21% precision. This means roughly 4 false positives per true positive. In insurance investigation, this is generally acceptable because missing a fraudulent claim (full payout) is far more costly than investigating a legitimate one (analyst time). The exact acceptable ratio depends on the organisation's cost structure and investigation capacity.

**Baseline comparison:** A random classifier at 6% fraud rate would achieve F1 of roughly 0.06. Our best model's F1 of 0.26 is approximately 4.3x better than random.

**Limitations:** The dataset is from 1994-1996, so fraud patterns may have evolved. Several features are pre-binned (VehiclePrice as ranges, AgeOfPolicyHolder as bands) which limits granularity. With only 923 fraud cases, rare fraud subtypes may be underrepresented.

---

## 5. SKILL.md Design Rationale

### Frontmatter and Body Separation

Each SKILL.md file uses YAML frontmatter for metadata (name, description) and a markdown body for execution instructions. The notebook parses these separately: frontmatter for logging and display, body for LLM prompt injection. This means the SKILL.md files serve a dual purpose -- they are both human-readable documentation and machine-readable prompt specifications.

### Pipeline State Chain

The agent state flows through the pipeline as follows:

- **preprocess-data** writes: df, label_encoders, cat_cols
- **feature-engineering** reads: df; engineers 3 features; writes: X_train/val/test, y_train/val/test, feature_cols (28), scaler, train_feature_means
- **train-models** reads: X/y arrays; writes: models, best_weight
- **evaluate-models** reads: models, X/y arrays; writes: results, results_df, threshold_analysis, test_probs_all
- **select-model** reads: results_df, threshold_analysis, models; writes: selected_model, best_threshold, selection_justification
- **score-fraud-risk** reads: claim, selected_model, feature_cols, label_encoders; writes: probability, prediction, feature_contributions
- **assess-claim-risk** reads: probability, claim, feature_contributions; writes: risk_level, risk_explanation
- **recommend-action** reads: risk_level, risk_explanation, claim, probability; writes: recommended_action

### Downstream Skill Design

The three LLM nodes each serve a distinct purpose:

**select-model** makes a strategic decision: which model and threshold best balances precision and recall. This is genuinely agentic because it requires reasoning about the precision-recall tradeoff across multiple operating points, early stopping behaviour, and PR-AUC stability -- something a simple argmax cannot do.

**assess-claim-risk** does analytical reasoning: translating model output into a human-readable explanation that references specific claim details. This bridges the gap between encoded feature values and business language.

**recommend-action** is prescriptive: it tells the claims team exactly what to do. LOW claims get auto-approved without an LLM call. MEDIUM/HIGH/CRITICAL claims get specific, claim-tailored recommendations.

### Why Three Separate Downstream Skills

We separated assess-claim-risk from recommend-action because they serve different audiences and purposes. The risk assessment explains *why* a claim is suspicious (for the adjuster's understanding). The recommended action says *what to do about it* (for operational workflow). Combining them into one node would conflate analysis with prescription.

The conditional routing skips recommend-action for LOW claims, saving LLM API cost on the ~94% of claims that are legitimate.

---

## 6. System Evaluation

### End-to-End Test Cases

We run four test cases through the full inference pipeline:

**Test 1: Known fraud claim (highest model confidence from test set)**
- Input: A real fraud claim from the test set with the highest predicted probability.
- Expected: Probability above threshold, risk level MEDIUM or above, LLM-generated explanation and recommended action.
- Result: Correctly flagged. Explanation referenced specific claim characteristics. Action recommended SIU investigation.
- Assertion: `probability > best_threshold` -- PASSED.

**Test 2: Known legitimate claim (lowest model confidence from test set)**
- Input: A real legitimate claim with the lowest predicted probability.
- Expected: Low probability, risk level LOW, auto-approve action (no LLM call).
- Result: Risk level = LOW. Auto-approve returned without triggering the recommend-action LLM node.
- Assertion: `risk_level == 'LOW'` -- PASSED.

**Test 3: Suspicious synthetic claim**
- Input: Constructed high-risk profile -- 19-year-old single male, Ferrari Sport, All Perils policy, no police report, no witness, past claims history, recent address change.
- Expected: Elevated probability, MEDIUM or above.
- Result: Pipeline correctly encoded all categorical values and scored the claim. LLM explanation identified the combination of young driver, sport vehicle, and missing police report as red flags.

**Test 4: Safe synthetic claim**
- Input: Constructed low-risk profile -- 52-year-old married female, Toyota Sedan, Collision policy, police report filed, witness present, no past claims.
- Expected: Low probability, LOW risk.
- Result: LOW risk, auto-approved. Conditional routing correctly skipped the recommend-action node.

### Failure Modes and Limitations

1. **Unknown categories** -- If a claim contains a vehicle make not seen during training, the system maps it to 0. This is safe but means the model has no information about that category. Production deployment would need periodic retraining.

2. **LLM dependency** -- The three LLM nodes depend on the OpenAI API. If unavailable, select-model falls back to F1-argmax, and the downstream nodes return fallback messages. The system degrades gracefully but loses the business-facing natural language output.

3. **Threshold sensitivity** -- The LLM-selected threshold may differ across runs since GPT-4o-mini is not fully deterministic. The Gradio threshold explorer allows stakeholders to see the precision-recall tradeoff and override the LLM's choice if needed.

4. **Dataset age** -- The data is from 1994-1996. Fraud patterns, vehicle types, and claim processes have changed. The model captures general fraud indicators (fault assignment, missing police report) that likely remain relevant, but specific thresholds would not transfer directly to modern data.

5. **Feature granularity** -- Several features are binned (VehiclePrice as ranges, AgeOfPolicyHolder as bands). Finer-grained numeric values could improve performance but are not available in this dataset.

6. **Class imbalance and precision** -- At 6% fraud rate, even a well-performing model has modest precision (~17-20%). This means most flagged claims are legitimate. The cost asymmetry justifies this (missing fraud is 100x more expensive than a false investigation), but it must be communicated clearly to end users so they understand that a flag is a recommendation for review, not a fraud determination.

7. **LLM non-determinism in model selection** -- Because GPT-4o-mini has non-zero temperature and may interpret the cost analysis differently across runs, the selected model and threshold can vary between pipeline executions. In a production system, the model selection would be made once and frozen, with the LLM justification logged for audit. The deterministic fallback ensures that even if the LLM produces an unparseable response, the pipeline still completes.

---

## 7. AI Usage Declaration

[TODO: Each team member should document their AI tool usage here per the module AI usage policy. Include which tools were used, for what tasks, and what modifications were made to AI-generated output.]

---

## References

1. Vehicle Insurance Fraud Detection Dataset -- Kaggle (Fraud Oracle)
2. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
3. Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.
4. LangGraph Documentation -- https://langchain-ai.github.io/langgraph/
5. Gradio Documentation -- https://www.gradio.app/docs
