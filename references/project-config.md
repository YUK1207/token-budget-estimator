# Project Configuration

Place `.token-budget.json` in a project directory to provide defaults for estimates and comparisons. The estimator searches from the current working directory upward.

```json
{
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
- `context_window`: model context window used for fit checks.
- `input_price_per_million`: input-token price for cost estimates.
- `output_price_per_million`: output-token price for cost estimates.
- `currency`: display currency label.

Do not commit private pricing or account-specific budget assumptions unless the project owner wants them versioned.
