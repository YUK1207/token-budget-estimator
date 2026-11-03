import json
import tempfile
import unittest
from pathlib import Path

import history_model as history


PROFILE = {'model': 'm1', 'agent': 'a1', 'execution_config': 'default-v1'}


def sample(index, ratio=2, split='fit', **updates):
    row = dict(schema_version=2, baseline_version=history.BASELINE_VERSION,
               instance_id=f'issue-{index}', task=f'Fix bug {index}', task_type='debugging',
               execution_profile=PROFILE, baseline_midpoint_tokens=10000,
               actual_tokens=ratio * 10000, split=split, outcome='completed')
    row.update(updates)
    return row


class HistoryTests(unittest.TestCase):
    def test_identical_text_different_ids_cannot_cross_partitions(self):
        rows = [sample(i, task='Same task', split='fit') for i in range(10)]
        rows += [sample(100+i, task='Same   task', split='calibration') for i in range(19)]
        result = self.forecast(rows)
        self.assertEqual(result.fit_samples, 1)
        self.assertEqual(result.calibration_samples, 0)
        rows.append(sample(200, task='Same task', split='test'))
        self.assertEqual(self.forecast(rows).fit_samples, 0)

    def test_task_alias_exclusion_is_transitive(self):
        rows = [sample(1, task='Original'), sample(1, task='Paraphrase'),
                sample(2, task='Paraphrase', split='test')]
        self.assertEqual(self.forecast(rows).fit_samples, 0)
        rows[-1]['split'] = 'fit'
        self.assertEqual(self.forecast(rows, instance_id='issue-2').fit_samples, 0)

    def test_malformed_collection_fields_are_skipped(self):
        rows = [sample(1, split=[]), sample(2, split={}),
                sample(3, outcome=[]), sample(4, outcome={})]
        self.assertEqual(self.forecast(rows).fit_samples, 0)

    def forecast(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            return history.predict(path, 'debugging', PROFILE, 20000, **kwargs)

    def test_empirical_median_is_not_interval_midpoint(self):
        result = self.forecast([sample(i, r) for i, r in enumerate([1, 1, 1, 2, 10])])
        self.assertEqual(result.p50_tokens, 20000)
        self.assertEqual(result.p90_tokens, 200000)
        self.assertFalse(result.calibrated)

    def test_split_conformal_calibration_uses_independent_residuals(self):
        rows = [sample(i, 1) for i in range(10)]
        rows += [sample(100+i, 3, 'calibration') for i in range(19)]
        result = self.forecast(rows)
        self.assertTrue(result.calibrated)
        self.assertEqual(result.high_tokens, 60000)
        self.assertEqual(result.low_tokens, 0)
        self.assertEqual(result.p50_tokens, 20000)
        self.assertEqual(result.fit_samples, 10)
        self.assertEqual(result.calibration_samples, 19)

    def test_configuration_version_and_outcome_are_isolated(self):
        rows = [sample(i) for i in range(5)]
        rows += [sample(10, 100, execution_profile={**PROFILE, 'model': 'other'}),
                 sample(11, 100, baseline_version='old'),
                 sample(12, 100, outcome='capped'),
                 sample(13, 100, outcome='environment_error'),
                 sample(14, 100, split='test'), {'actual_tokens': 1}]
        result = self.forecast(rows)
        self.assertEqual(result.fit_samples, 5)
        self.assertEqual(result.p50_tokens, 40000)

    def test_duplicate_and_current_tasks_do_not_leak(self):
        rows = [sample(i) for i in range(8)]
        rows += [sample(0, 100, 'calibration'), sample(1, 100, 'test')]
        result = self.forecast(rows, instance_id='issue-2', task='Fix bug 3')
        # Test-tagged issue-1 is excluded entirely; issue-0 stays in fit once.
        self.assertEqual(result.fit_samples, 5)
        self.assertEqual(result.calibration_samples, 0)

    def test_sparse_invalid_and_missing_histories_fall_back(self):
        result = self.forecast([sample(1), sample(2, float('nan')),
                                sample(3, -1), sample(4, float('inf'))])
        self.assertIsNone(result.p50_tokens)
        self.assertEqual(result.fit_samples, 1)
        result = history.predict(Path('does-not-exist'), 'debugging', PROFILE, 10000)
        self.assertEqual(result.method, 'heuristic')

    def test_unknown_profile_never_pools_legacy_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history'
            path.write_text(json.dumps(sample(0)), encoding='utf-8')
            result = history.predict(path, 'debugging', {}, 10000)
            self.assertEqual(result.method, 'heuristic')

    def test_auto_partition_is_stable_for_task_identity(self):
        rows = [sample(i, split='auto') for i in range(150)]
        a = self.forecast(rows)
        b = self.forecast(list(reversed(rows)))
        self.assertEqual(a, b)
        self.assertTrue(a.calibrated)

    def test_finite_sample_rank_rejects_too_small_calibration_set(self):
        self.assertIsNone(history.conformal_radius([1] * 5, 0.9))
        self.assertEqual(history.conformal_radius(list(range(19)), 0.9), 17)
