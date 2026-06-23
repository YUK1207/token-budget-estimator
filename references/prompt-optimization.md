# Prompt Optimization

Use prompt optimization when a task is broad, vague, expensive, over budget, or likely to make the agent explore too much of the repository.

The optimizer must:

- Compress repeated or vague wording.
- Preserve explicit user intent.
- Convert the request into a Codex task contract.
- Add scope guards to prevent unrelated file reads.
- Add output caps to reduce command-log token waste.
- Add auto-split stages when the task is high risk.
- Provide before/after token estimates.

Do not output a prompt diff unless the user explicitly asks for one.

## Budget Modes

| Mode | Use when | Behavior |
|---|---|---|
| cheap | User has a tight budget or the task is high/extreme risk | Discovery-first, strict scope, few files, few commands |
| balanced | Default for medium-risk work | Scoped implementation with focused validation |
| thorough | User values completeness over token savings | Broader checks while still avoiding unrelated exploration |

## Contract Sections

An optimized prompt should include:

- Goal
- Original request
- Budget mode
- Scope
- Execution plan
- Output caps
- Validation

The contract should be directly usable as a follow-up Codex task.
