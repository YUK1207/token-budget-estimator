---
name: token-budget-estimator
description: Estimate total token usage before launching Codex or coding-agent tasks. Use when a user asks to predict token cost, budget a command, decide whether to run a task, split a high-cost task, compare task sizes, or optimize an agent prompt before execution. Also use as a preflight gate when project instructions require token budgeting before task execution.
---

# Token Budget Estimator

Estimate the full agent-task token budget before execution. Predict the total token range created by user input, repository discovery, code reading, command output, failed tests, retries, intermediate analysis, and final response. Do not present a precise single number.

## Preflight Workflow

1. Build a compact task summary from the user's request, current working directory, known constraints, and whether the task likely modifies code.
2. Run `scripts/estimate.py` with the task text and current working directory when a shell is available.
3. If the script cannot run, apply the model in `references/token-model.md` manually.
4. Render the budget report before starting the requested task.
5. For high or extreme risk, recommend a smaller first task. Wait for confirmation only when project instructions or the user require it.
6. After execution, if actual usage is available, record the prediction and actual token usage with `scripts/estimate.py record` so future predictions can calibrate.

## Command

Use:

```bash
python <skill-dir>/scripts/estimate.py estimate --task "<user task>" --cwd "<project cwd>"
```

Optional flags:

```bash
--context-tokens 12000
--budget 50000
--context-window 128000
--input-price-per-million 1.25
--output-price-per-million 10.0
--currency USD
--actual-tokens 42000
--history <path-to-jsonl>
--json
```

Use `--json` when another tool or script will consume the result. Use the default Markdown output for user-facing preflight reports.

Optimize a prompt under a token budget:

```bash
python <skill-dir>/scripts/estimate.py optimize --task "<rough task>" --budget 50000 --mode cheap
```

Use `optimize` when the user wants a cheaper, safer Codex-ready task contract. It produces a compressed task, scope guard, output caps, auto-split plan, and before/after estimates. Do not produce a prompt diff unless the user explicitly asks for one.

Compare multiple tasks:

```bash
python <skill-dir>/scripts/estimate.py compare --task "Fix failing tests" --task "Publish this skill to GitHub" --cwd "<project cwd>"
```

Use a project-level `.token-budget.json` to avoid repeating common defaults. See `references/project-config.md`.

## Report Requirements

Always include:

- Estimated total token range.
- Risk level: low, medium, high, or extreme.
- Confidence: low, medium, or high.
- Task type and complexity.
- Main token drivers.
- Budget check when a user provides a token ceiling.
- Context fit when a context window is configured.
- Cost estimate when input/output prices are configured.
- Recommended execution shape.

Use this concise format:

```markdown
**Token Preflight**
- Estimate: 18k-42k tokens
- Risk: medium
- Confidence: medium
- Task type: debugging + tests
- Main drivers: repository search, reading 8-18 files, 2-5 test runs, possible failure logs
- Budget check: within_budget against 50k tokens
- Context fit: fits against 128000 tokens
- Cost estimate: USD 0.02-0.18
- Recommendation: continue, but cap test output and fix one failure chain first
```

## Risk Policy

Use `references/risk-policy.md` for thresholds and recommended behavior.
Use `references/agents-preflight.md` when the user wants this skill to run before every agent task.

Default behavior:

- Low: continue after a compact report.
- Medium: continue after naming the likely token drivers.
- High: suggest splitting and ask for confirmation if the task is broad or destructive.
- Extreme: recommend a discovery-only pass first.

## References

- Read `references/task-taxonomy.md` when the task type is ambiguous.
- Read `references/token-model.md` when adjusting formulas or explaining the prediction model.
- Read `references/risk-policy.md` when deciding whether to continue, ask for confirmation, or split.
- Read `references/project-config.md` when configuring default budgets, context windows, or prices.
- Read `references/prompt-optimization.md` when optimizing prompts or explaining budget modes.
- Read `references/agents-preflight.md` when installing a project-level or global preflight rule.
