import json
import tempfile
import unittest
from pathlib import Path

import estimate as estimator


class TokenBudgetEstimatorTests(unittest.TestCase):
    def test_loads_project_config_from_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".token-budget.json").write_text(
                json.dumps(
                    {
                        "budget_tokens": 25000,
                        "context_window": 128000,
                        "input_price_per_million": 1.25,
                        "output_price_per_million": 10.0,
                        "currency": "USD",
                    }
                ),
                encoding="utf-8",
            )

            config = estimator.load_project_config(root)

        self.assertEqual(config.budget_tokens, 25000)
        self.assertEqual(config.context_window, 128000)
        self.assertEqual(config.input_price_per_million, 1.25)
        self.assertEqual(config.output_price_per_million, 10.0)
        self.assertEqual(config.currency, "USD")

    def test_estimate_reports_context_fit_and_cost(self):
        result = estimator.estimate(
            "Fix a focused bug in this repo",
            "C:\\does-not-exist",
            context_tokens=0,
            history_path=Path("missing-history.jsonl"),
            budget_tokens=50000,
            context_window=128000,
            input_price_per_million=1.0,
            output_price_per_million=5.0,
            currency="USD",
        )

        self.assertIn(result.context_fit, {"fits", "tight", "likely_exceeds"})
        self.assertIsNotNone(result.cost)
        self.assertGreater(result.cost.high, 0)
        self.assertEqual(result.cost.currency, "USD")

    def test_compare_tasks_sorts_by_midpoint_descending(self):
        results = estimator.compare_tasks(
            [
                "Explain this function",
                "Refactor the entire project and run all tests",
            ],
            "C:\\does-not-exist",
            context_tokens=0,
            history_path=Path("missing-history.jsonl"),
            budget_tokens=None,
            context_window=None,
            input_price_per_million=None,
            output_price_per_million=None,
            currency="USD",
        )

        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].midpoint_tokens, results[1].midpoint_tokens)
        self.assertEqual(results[0].rank, 1)

    def test_optimize_task_generates_budget_aware_contract_without_diff(self):
        result = estimator.optimize_task(
            "Refactor the entire project and run all tests",
            "C:\\does-not-exist",
            context_tokens=0,
            history_path=Path("missing-history.jsonl"),
            budget_tokens=50000,
            context_window=128000,
            input_price_per_million=None,
            output_price_per_million=None,
            currency="USD",
            mode="cheap",
        )

        self.assertEqual(result.mode, "cheap")
        self.assertLess(result.optimized_estimate.high_tokens, result.original_estimate.high_tokens)
        self.assertIn("Goal", result.optimized_prompt)
        self.assertIn("Scope", result.optimized_prompt)
        self.assertIn("Validation", result.optimized_prompt)
        self.assertTrue(result.auto_split)
        self.assertTrue(result.scope_guard)
        self.assertTrue(result.output_caps)
        self.assertTrue(result.savings_summary)
        self.assertFalse(hasattr(result, "prompt_diff"))

    def test_preflight_controller_builds_forecast_roi_contract_and_diet(self):
        result = estimator.build_preflight_controller(
            "Fix the failing login tests and run focused pytest targets",
            "C:\\does-not-exist",
            context_tokens=0,
            history_path=Path("missing-history.jsonl"),
            budget_tokens=50000,
            context_window=128000,
            input_price_per_million=None,
            output_price_per_million=None,
            currency="USD",
            mode="cheap",
        )

        self.assertIsNotNone(result.forecast)
        self.assertIn(result.roi.decision, {"execute_now", "split_first", "discovery_first", "defer"})
        self.assertGreater(result.roi.score, 0)
        self.assertGreater(result.contract.max_files, 0)
        self.assertGreater(result.contract.max_commands, 0)
        self.assertTrue(result.contract.stop_loss)
        self.assertTrue(result.diet.read_first)
        self.assertTrue(result.diet.avoid_initially)
        self.assertTrue(result.recommended_first_pass_prompt)


if __name__ == "__main__":
    unittest.main()
