import math
import unittest

import task_neighbors as neighbors


class NeighborTests(unittest.TestCase):
    def test_query_does_not_change_fit_vocabulary(self):
        rows = [dict(instance_id='a', task='parser syntax failure', actual_tokens=100),
                dict(instance_id='b', task='network timeout retry', actual_tokens=1000)]
        index = neighbors.TextIndex(rows)
        before = dict(index.idf)
        self.assertAlmostEqual(index.predict_log('parser syntax failure', math.log(500))[0],
                               index.predict_log('parser syntax failure', math.log(500))[0])
        index.predict_log('unseenword totallynovel', math.log(500))
        self.assertEqual(index.idf, before)
        self.assertNotIn('unseenword', index.idf)

    def test_similar_task_moves_point_toward_relevant_fit_cost(self):
        rows = [dict(instance_id='a', task='parser syntax failure', actual_tokens=100),
                dict(instance_id='b', task='network timeout retry', actual_tokens=1000)]
        value, support = neighbors.TextIndex(rows).predict_log('parser syntax failure', math.log(500))
        self.assertLess(value, math.log(500))
        self.assertGreater(value, math.log(100))
        self.assertEqual(support['neighbor_count'], 1)

    def test_empty_text_and_unknown_words_retain_prior(self):
        index = neighbors.TextIndex([dict(instance_id='a', task='parser syntax', actual_tokens=100)])
        self.assertEqual(index.predict_log('', 5)[0], 5)
        self.assertEqual(index.predict_log('unseenword', 5)[0], 5)

    def test_repository_shrinkage_and_unknown_repository(self):
        rows = [dict(actual_tokens=100, feature_snapshot={'repository': 'a/repo'}) for _ in range(10)]
        self.assertAlmostEqual(neighbors.repository_prior(rows, 'a/repo', math.log(1000))[0], math.log(math.sqrt(100000)))
        self.assertEqual(neighbors.repository_prior(rows, 'other/repo', math.log(1000))[0], math.log(1000))

    def test_weighted_median_rejects_invalid_support(self):
        self.assertEqual(neighbors.weighted_median([(1, 3), (10, 1)]), 1)
        with self.assertRaises(ValueError):
            neighbors.weighted_median([(1, 0)])
