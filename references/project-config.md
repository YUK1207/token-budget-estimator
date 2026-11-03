# Project Configuration

Place `.token-budget.json` in a project directory to provide defaults for estimates and comparisons. The estimator searches from the current working directory upward.

```json
{
  "task_kind": "auto",
  "predictor": "adaptive",
  "point_model": "auto",
  "model": "model-version",
  "agent": "harness-version",
  "execution_config": "reasoning-medium-context-v1",
  "budget_tokens": 50000,
  "context_tokens": 8000,
  "context_window": 128000,
  "input_price_per_million": 1.25,
  "output_price_per_million": 10.0,
  "currency": "USD"
}
```

CLI flags override project config values.

Fields:

- `budget_tokens`: upper token budget for budget checks.
- `context_tokens`: known existing conversation/context tokens.
- `context_window`: single-request context limit; fit is unknown without a supplied peak.
- `peak_context_tokens`: separately supplied peak request scenario, not cumulative usage.
- `model`, `agent`, `execution_config`: exact identifiers for v2 history matching.
- `input_price_per_million`: input-token price for cost estimates.
- `output_price_per_million`: output-token price for cost estimates.
- `currency`: display currency label.

Do not commit private pricing or account-specific budget assumptions unless the project owner wants them versioned.

`task_kind` may be `auto` or explicitly `code_repair`. `predictor` is `adaptive` or `legacy`. `point_model` is `auto`, `median`, `ridge`, `repository`, or `retrieval`; select on fitting/validation data only. Optional `repository` supplies a known-start `owner/repo` identity; otherwise the local git origin is used when available.
