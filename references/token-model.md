# Token Model

Estimate total agent token use as a range:

```text
total =
  user_prompt_tokens
+ existing_context_tokens
+ repository_discovery_tokens
+ file_read_tokens
+ command_output_tokens
+ failure_retry_tokens
+ intermediate_analysis_tokens
+ final_response_tokens
```

Use ranges because agent work has high variance. A precise single value is misleading.

## Component Heuristics

| Component | Low | High |
|---|---:|---:|
| user_prompt_tokens | measured or chars / 4 | measured or chars / 3 |
| repository_discovery_tokens | searches * 250 | searches * 900 |
| file_read_tokens | files * 700 | files * 2500 |
| command_output_tokens | commands * 400 | commands * 2500 |
| failure_retry_tokens | failures * 800 | failures * 6000 |
| intermediate_analysis_tokens | complexity * 700 | complexity * 2200 |
| final_response_tokens | 300 | 2200 |

## Expected Workload Defaults

| Task type | Searches | Files read | Commands | Failure loops |
|---|---:|---:|---:|---:|
| simple_question | 0-1 | 0-1 | 0 | 0 |
| code_explanation | 2-6 | 2-8 | 0-1 | 0 |
| small_code_change | 3-8 | 3-10 | 1-3 | 0-1 |
| debugging | 5-14 | 6-20 | 2-6 | 1-4 |
| test_repair | 4-12 | 5-18 | 3-8 | 1-5 |
| feature_development | 6-16 | 8-24 | 2-6 | 1-3 |
| frontend_build | 6-18 | 8-28 | 4-10 | 1-4 |
| git_publish | 1-4 | 2-8 | 4-10 | 0-2 |
| refactor | 10-28 | 18-60 | 4-12 | 2-6 |
| research_or_docs | 6-18 | 0-8 | 0-3 | 0-2 |
| large_project_task | 18-50 | 30-100 | 8-20 | 3-10 |

## Calibration

Store prediction records as JSONL. For each task type, compute actual / predicted_midpoint ratios. Apply the median ratio to future estimates for that type. Keep the lower bound conservative and widen the range when fewer than five samples exist.

## Context Fit

When `context_window` is configured, compare the upper token estimate to the window:

- `fits`: upper estimate is at most 75% of the window.
- `tight`: upper estimate is within the full window but above 75%.
- `likely_exceeds`: upper estimate is greater than the window.

## Cost Estimate

Cost estimates require explicit prices. Do not hard-code current model prices in the skill. Use:

- `input_price_per_million`
- `output_price_per_million`
- `currency`

Estimate low/high cost from the input-token and output-token ranges.
