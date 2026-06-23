# AGENTS.md Preflight Rule

Skill installation allows implicit invocation, but it does not create a hard system hook before every command. To make token budgeting run by default, add this rule to a project-level or global `AGENTS.md`.

```markdown
# Token Budget Preflight

Before executing a non-trivial user task, use `$token-budget-estimator` to estimate the full agent-task token range. The estimate must account for code discovery, file reads, command output, failed tests, retry loops, intermediate reasoning, and final response.

For low-risk tasks, provide a compact budget and continue.
For medium-risk tasks, name the main token drivers and continue.
For high-risk tasks, recommend a split and ask for confirmation if the task is broad.
For extreme-risk tasks, recommend a discovery-only pass before implementation.
```

Use a "non-trivial" threshold to avoid noise for direct one-line answers, but always run preflight for debugging, code modification, tests, broad repository work, research, frontend builds, and refactors.
