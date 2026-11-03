# Token Budget Estimator

A pure-preflight Codex skill and Python CLI for estimating total agent input + output tokens **before the solver starts**. Python 3.10+, standard library only.

Inputs: task text, lightweight local repository counts, declared execution configuration, and prior usage records. No solver execution, LLM calls, or running-task trajectory inputs. Post-run recording improves future forecasts only.

[Website](https://yuk1207.github.io/token-budget-estimator/) · [API concept](https://yuk1207.github.io/token-budget-estimator/api.html) · [Method and papers](references/preflight-method.md)

The website's HTTP endpoints and paid plans are design concepts, not implemented services.

## Changes

- English word boundaries prevent `callback` and `small` from matching `all`.
- History matches task type, model/version, agent/version, execution configuration and raw baseline version.
- P50/P90 come from historical ratios, not the arithmetic midpoint of a wide interval.
- Independent fit/calibration tasks support split-conformal interval expansion.
- Insufficient data explicitly falls back to rough rules. Sample count alone never establishes confidence.
- Peak single-request context is separate from cumulative tokens.
- Prompt suggestions no longer apply fixed 0.45/0.65/0.82 savings multipliers.

This baseline is inspired by quantile calibration and task-start forecasting. It is not a reproduction of TokenCast or learned CQR. Predictive accuracy and token savings remain unvalidated on a held-out benchmark.

## Install

Copy the repository into `~/.codex/skills/token-budget-estimator`, then start a new Codex chat:

```text
$token-budget-estimator Estimate the budget before fixing the failing login tests.
```

Keep the whole `scripts/` directory: `estimate.py` imports `history_model.py`.

## CLI

From this repository root:

```bash
python scripts/estimate.py estimate --task "Fix failing login tests" --cwd /path/to/target-repo --budget 50000 --json
```

For history matching, use stable exact identifiers. The execution configuration should identify reasoning effort, context/compaction policy, prompt/tool versions and termination limits.

```bash
python scripts/estimate.py estimate --task "Fix failing login tests" --cwd /path/to/target-repo --model model-version --agent harness-version --execution-config reasoning-medium-context-v1 --instance-id repo-issue-123 --history /path/to/private-history.jsonl --json > prediction.json
```

Save this prediction **before** solving. After the run, record provider-accounted usage from the same scope:

```bash
python scripts/estimate.py record --prediction-file prediction.json --actual-tokens 52000 --instance-id repo-issue-123 --outcome completed --history /path/to/private-history.jsonl
```

The frozen prediction preserves the raw baseline, configuration and history digest. Recording does not update an already issued forecast. Do not publish private tasks or histories.

## Output

| Field | Meaning |
|---|---|
| `low_tokens`, `high_tokens` | Interval; inspect `interval_method` before interpreting coverage |
| `midpoint_tokens` | Arithmetic interval midpoint retained for compatibility; **not P50** |
| `p50_tokens`, `p90_tokens` | Empirical historical quantiles, or `null` |
| `prediction_method` | `heuristic`, `historical_quantiles`, or `historical_split_conformal` |
| `interval_method` | `heuristic_uncalibrated` or `split_conformal_90` |
| `interval_target_coverage` | `0.9` only for calibrated intervals; a marginal target under exchangeability assumptions, not measured accuracy |
| `fit_samples`, `calibration_samples` | Distinct eligible task counts |
| `confidence` | `low` for fallback; `unvalidated` for historical predictions |
| `baseline_midpoint_tokens` | Versioned, history-free scale for ratio fitting |
| `history_digest` | Fingerprint of selected task identities, splits and usage |
| `budget_status` | `within_budget`, `near_budget`, or `over_budget` |

Five matching **fit** tasks enable empirical P50/P90. Ten fit tasks and nineteen independent calibration tasks enable the current 90% split-conformal interval. These are operational minima, not evidence of sufficient sample size or accuracy. Calibration values never train the quantiles.

With insufficient calibration, the interval remains a scaled heuristic, explicitly uncalibrated. Total-only history cannot identify input/output composition or cache pricing: component fields and cost become `null`. Heuristic cost scenarios accept manual input/output prices and omit cache discounts.

`--budget` is advisory and exits successfully on `over_budget`. External launchers must parse JSON to enforce policy. Contracts are suggestions, not runtime enforcement.

## Context

`--context-tokens` is known initial context. `--context-window` is a single-request limit. Without a separately supplied peak estimate, context fit is `unknown`.

```bash
python scripts/estimate.py estimate --task "Fix a bug" --context-window 128000 --peak-context-tokens 64000
```

`fits` (at most 75%), `tight` (within the window), and `likely_exceeds` use only that supplied peak. The tool does not learn peak context from total usage.

## Other commands

```bash
python scripts/estimate.py compare --task "Explain this function" --task "Refactor the entire project" --json
python scripts/estimate.py optimize --task "Fix failing tests" --mode cheap --json
python scripts/estimate.py control --task "Fix failing tests" --budget 50000 --mode balanced --json
```

All accept model/agent/configuration/history options. `optimize` suggests a task contract and reports **savings not measured**; selecting a cheaper mode does not numerically reduce the forecast. `control` adds heuristic ROI, contract and context-diet suggestions. Its score is not measured economic return or success probability.

Project defaults: [.token-budget.json configuration](references/project-config.md). CLI flags take precedence.

## History and evaluation

- `--split auto` partitions canonical instance IDs deterministically into fit/calibration. Explicit `fit`, `calibration` and `test` are supported. Repeated IDs or normalized task text are grouped; no task can cross partitions.
- Use the first eligible run per task. Test groups are excluded from both fit and calibration. Keep a separate frozen training history for strict benchmark evaluation.
- Always pass `--instance-id` when predicting. Current ID and matching task text are excluded. Canonical IDs must remain consistent across prompt variants.
- Natural solver success (`completed`) and failure (`failed`) are eligible. `capped` and `environment_error` runs are recorded but excluded: their observed tokens are not natural completion costs.
- Target provider input + output across one solver run. Count cache-read input once; do not add reasoning tokens twice if included in output. Track setup, evaluator and environment-retry costs separately.
- Legacy `record --task ... --task-type ... --predicted-low ... --predicted-high ... --predicted-midpoint ... --actual-tokens ...` remains accepted as schema v1, but cannot calibrate v2 forecasts without configuration/raw baseline.
- On untouched tasks report point MAE/WAPE and bias, interval coverage **and width**, and solver success/cost for any proposed policy. Compare against raw heuristics and a same-configuration historical median.
- Three runs in one repository do not establish generalization. Interval hits and unit tests do not establish predictive accuracy.

## Tests

```bash
python -m unittest discover -s scripts -v
```

Synthetic tests verify calculations, CLI behavior, task isolation and regressions. They do not measure forecast quality.
