# Risk Policy

| Risk | Estimated total | Default behavior |
|---|---:|---|
| low | < 15k | Continue after compact report |
| medium | 15k-50k | Continue, name main token drivers |
| high | 50k-120k | Recommend splitting; ask confirmation for broad tasks |
| extreme | > 120k | Recommend discovery-only pass first |

## Split Recommendations

For high or extreme risk, recommend one of these shapes:

- Discovery pass: locate relevant files, tests, and commands only.
- Git publish pass: inspect target files, make one focused commit, then create/configure the remote and push.
- One failure chain: fix the first reproducible failure before broad cleanup.
- One module: limit implementation to a named package or component.
- Output cap: run commands with limited logs where possible.
- Two-turn plan: first inspect and plan, second implement.

## Continuation Guidance

Do not block trivial or medium tasks with excessive ceremony. The report should help the user make a decision, not become the main task.
