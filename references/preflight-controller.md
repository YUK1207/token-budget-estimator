# Preflight Controller

The preflight controller turns token estimation into an pre-execution advisory report.

Use it when the user wants Codex to:

- predict token cost before execution,
- decide whether the task is worth executing now,
- generate an explicit budget contract,
- plan the smallest useful context set,
- and suggest a bounded first pass.

ROI scores are hand-set policy heuristics, not measured economic returns. Contracts do not enforce runtime limits. Forecast interpretation follows [the preflight method](preflight-method.md).

## Modules

| Module | Purpose |
|---|---|
| Token Forecast | Predict the total task token range and risk tier |
| ROI Assessor | Decide whether to execute now, split first, run discovery first, or defer |
| Budget Contract Generator | Set limits for files, commands, log lines, denied actions, and stop-loss conditions |
| Context Diet Planner | Decide what to read first, what to read second, and what to avoid initially |

## Decisions

ROI decisions:

- `execute_now`: cost and risk are acceptable.
- `split_first`: the task has value but should be split to fit budget.
- `discovery_first`: uncertainty is high; inspect first, do not edit yet.
- `defer`: cost pressure is high and expected value is low.

## Budget Contract

A budget contract must include:

- max files before first report,
- max commands before reassessment,
- max log lines,
- allowed actions,
- denied actions,
- stop-loss checkpoints.

## Context Diet

Context diet plans should name:

- read first,
- read second,
- avoid initially,
- escalation rule.

The goal is to avoid wasting context on unrelated files, full logs, generated artifacts, and broad repository scans.
