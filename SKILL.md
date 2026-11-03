---
name: token-budget-estimator
description: Estimate total token usage before launching Codex or coding-agent tasks. Use when a user asks to predict token cost, budget a command, decide whether to run a task, split a high-cost task, compare task sizes, or optimize an agent prompt before execution. Also use as a preflight gate when project instructions require token budgeting before task execution.
---

# Token Budget Estimator

Forecast whole-task tokens before the solver starts. Inputs are task text, lightweight repository signals, declared configuration and prior usage. Never launch a solver or add LLM calls to produce a forecast. Post-run recording improves future tasks only.

## Workflow

1. Summarize the task and identify the target repository.
2. Run `scripts/estimate.py estimate` before execution. Keep `history_model.py` beside it.
3. Supply exact model/version, harness/version and an execution configuration ID when known. Never invent identifiers to force history matching.
4. Pass a stable canonical instance ID to exclude the current task from history.
5. Save JSON before solving if actual usage will be recorded later.
6. Report method, evidence limits, interval, empirical P50/P90 when available, risk and budget status. Respect existing user authorization; checks and contracts are advisory.
7. After execution, record actual usage from the frozen prediction, with the correct outcome and partition.

```bash
python <skill-dir>/scripts/estimate.py estimate --task "<task>" --cwd "<target-repo>" --budget 50000 --json
```

Optional flags:

```text
--model <exact-model-version>
--agent <harness-version>
--execution-config <reasoning-context-tools-config-version>
--instance-id <canonical-task-id>
--history <private-history.jsonl>
--context-tokens <known-initial-context>
--context-window <single-request-window>
--peak-context-tokens <separately-estimated-peak>
--input-price-per-million <price>
--output-price-per-million <price>
--currency <currency>
```

If the script cannot run, use `references/token-model.md` as an explicitly uncalibrated fallback. Do not manufacture historical quantiles or calibrated coverage manually.

## Interpretation

- `midpoint_tokens` is the arithmetic interval midpoint, not P50.
- P50/P90 require five matching independent fit tasks and remain empirical, unvalidated estimates.
- Ten fit and nineteen independent calibration tasks enable the 90% marginal coverage target, assuming exchangeable tasks and consistent configuration. This is not measured accuracy.
- Confidence remains `low` or `unvalidated`; sample count alone does not demonstrate quality.
- Total-only calibration leaves component token splits and cost unknown. Explain `null` values instead of inventing them.
- Cumulative usage does not measure peak context. Without a supplied peak, context fit is `unknown`.
- `--budget` reports a status, not an enforced runtime stop or nonzero exit on over-budget forecasts.

## Prompt and policy suggestions

`optimize` suggests a task contract. `control` adds a heuristic ROI, contract and context-diet report. Both remain preflight-only; modes are `cheap`, `balanced`, and `thorough`.

Prompt suggestions do not change the forecast automatically. Report "savings not measured"; do not claim fixed percentage discounts as predicted or achieved savings. ROI scores are policy heuristics, not validated economic returns.

## Record

```bash
python <skill-dir>/scripts/estimate.py record --prediction-file <frozen-estimate.json> --instance-id <task-id> --actual-tokens <usage> --history <private-history.jsonl> --outcome completed
```

Use `failed` for natural solver failure, `capped` for interrupted runs, or `environment_error` for infrastructure failure. `--split auto` separates fit/calibration by task identity; `--split test` reserves outcomes from fitting. Keep benchmark history frozen for strict evaluation.

Read [method and papers](references/preflight-method.md) for architecture and assumptions, [configuration](references/project-config.md) for defaults, [prompt suggestions](references/prompt-optimization.md) and [policy report](references/preflight-controller.md) as needed.
