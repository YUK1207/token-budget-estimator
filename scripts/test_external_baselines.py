import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import external_baselines as comparison


class ExternalProtocolTests(unittest.TestCase):
    def test_common_budget_calibration_uses_only_calibration_residuals(self):
        bounds = comparison.upper_bounds([10] * 19, [30] * 19, [20, 40])
        self.assertAlmostEqual(bounds[0], 60, delta=1)
        self.assertAlmostEqual(bounds[1], 120, delta=1)

    def test_insufficient_and_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):
            comparison.upper_bounds([1] * 5, [2] * 5, [1])
        with self.assertRaises(ValueError):
            comparison.upper_bounds([0] * 19, [2] * 19, [1])
        with self.assertRaises(ValueError):
            comparison.upper_bounds([1] * 19, [float('nan')] * 19, [1])

    def test_partition_overlap_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            comparison.validate_partitions(['a'], ['a'], ['c'])
        with self.assertRaises(ValueError):
            comparison.validate_partitions(['a', 'a'], ['b'], ['c'])
        comparison.validate_partitions(['a'], ['b'], ['c'])

    def test_official_input_is_registered_prelaunch_vector_only(self):
        values = {name: 1.0 for name in comparison.FEATURE_NAMES}
        snapshot = {'version': 'preflight-counts-v1', 'values': {**values, 'actual_tokens': 100000}}
        self.assertEqual(comparison.vector(snapshot), [1.0] * 8)
        self.assertIsInstance(comparison.official_features([1.0] * 8)['task_embedding'], tuple)
        with self.assertRaises(ValueError):
            comparison.vector({'version': 'unknown', 'values': values})
