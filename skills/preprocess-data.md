---
name: preprocess-data
description: Load CSV, drop redundant columns, encode categoricals with proper ordinal/binary/label strategy
---

# Preprocess Data

## When to use

Entry node. Loads raw data, cleans it, and encodes all columns for model consumption.

## How to execute

1. Load `data/fraud_oracle.csv` (15,420 claims, 33 columns)
2. Drop 7 useless/redundant columns: PolicyNumber (ID), Age (r=0.97 with AgeOfPolicyHolder), PolicyType (r=0.86 with VehicleCategory), WeekOfMonth/WeekOfMonthClaimed (noise), DriverRating (no fraud signal), Year (artifact)
3. Fix placeholder '0' in DayOfWeekClaimed/MonthClaimed with 'Unknown'
4. Ordinal encode 9 columns that have a natural order (e.g. AgeOfPolicyHolder: '16 to 17' -> 0, ..., 'over 65' -> 8)
5. Binary encode 6 columns (Fault, Sex, AccidentArea, PoliceReportFiled, WitnessPresent, AgentType)
6. Label encode remaining nominal categoricals (Make, Month, DayOfWeek, etc.)

## Inputs from agent state

None -- entry node, reads from disk.

## Outputs to agent state

- `df` (DataFrame): fully encoded, 26 columns (25 features + target)
- `label_encoders` (dict): column -> LabelEncoder, plus `_ordinal_maps` and `_binary_maps` for inference
- `cat_cols` (list): names of label-encoded columns
- `messages` (list)

## Output format

DataFrame with all numeric columns. Encoders stored as a dict for downstream inference decoding.

## Notes

- Ordinal encoding preserves natural order so trees can split meaningfully (e.g. "new" car < "7 years" < "more than 7")
- Binary encoding ensures consistent direction (e.g. PoliceReportFiled: Yes=0, No=1, so higher = more suspicious)
- The 7 dropped columns were identified via correlation analysis (redundancy) and fraud-rate spread analysis (no signal)
