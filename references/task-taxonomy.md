# Task Taxonomy

Classify the request by the most expensive likely execution path, not by the shortest interpretation.

| Task type | Triggers | Base complexity | Typical drivers |
|---|---|---:|---|
| simple_question | Direct explanation, no repo changes | 1 | final answer |
| code_explanation | Explain local code, trace behavior | 2 | code search, file reads |
| small_code_change | One narrow change, known files | 3 | file reads, patch, focused tests |
| debugging | Bug, failing behavior, unknown root cause | 4 | search, repeated tests, logs, retries |
| test_repair | Fix failing tests or CI | 4 | test output, multiple runs |
| feature_development | Add behavior or module | 5 | discovery, implementation, tests |
| frontend_build | Build UI, app, visualization, game | 6 | asset work, browser verification, screenshots |
| git_publish | Publish local work to GitHub or configure git remote/push | 3 | file inspection, git commands, repo creation, push output |
| refactor | Multi-file structural change | 7 | broad reads, test suite, regressions |
| research_or_docs | Internet/doc reading or generated report | 6 | source collection, summarization |
| large_project_task | Broad scope across unknown system | 10 | large discovery, many files, many commands |

Complexity modifiers:

- Add 1 if the task mentions "entire project", "all", "full", "complete", "production", or "end to end".
- Add 1 if tests, CI, browser verification, or screenshots are required.
- Add 1 if the repo is unfamiliar or large.
- Add 1 if the task likely needs network access or external docs.
- Subtract 1 if exact files, exact functions, and expected behavior are supplied.

Clamp complexity to 1-10.
