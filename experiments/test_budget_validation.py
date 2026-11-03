"""Engineering checks for the experiment's scoring and calibration boundaries."""
import unittest
from budget_validation import calibrated_upper, point_metrics, wilson


class BudgetValidationTest(unittest.TestCase):
    def test_rank_is_finite_sample_and_abstains_if_unavailable(self):
        self.assertIsNone(calibrated_upper([10] * 4, [20] * 4, [10], .9))
        self.assertEqual(calibrated_upper([10] * 19, [20] * 19, [10], .95), [20])
        # The 90% rank for 19 calibration values is 18, not 17.
        upper = calibrated_upper([10] * 19, list(range(1, 20)), [10], .9)
        self.assertEqual(upper, [18])

    def test_scoring_direction_and_coverage_interval(self):
        metrics = point_metrics([10, 30], [20, 20])
        self.assertEqual(metrics['mae'], 10)
        self.assertEqual(metrics['wape'], .5)
        self.assertEqual(metrics['signed_bias'], 0)
        lo, hi = wilson(90, 100)
        self.assertLess(lo, .9)
        self.assertGreater(hi, .9)


if __name__ == '__main__':
    unittest.main()
