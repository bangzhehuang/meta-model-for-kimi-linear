"""Checks for the semantic deduplication used by the conditional random lane."""
import unittest

from graph import analyze, canonical, edge, graph_id, kda
from lower import classify
from structured import polynomial, sample, semantic_id


def outer(left='v', right='k', scaled=False, scale_output=False):
    nodes = []
    if scaled:
        nodes.append({'name': 'scaled', 'operation': 'ScalarScale',
                      'inputs': [edge('beta'), edge(left)]})
        left = 'scaled'
    nodes.append({'name': 'write', 'operation': 'MatMul',
                  'inputs': [edge(left, 'column'), edge(right, 'row')]})
    write = 'write'
    if scale_output:
        nodes.append({'name': 'scaled', 'operation': 'ScalarScale',
                      'inputs': [edge('beta'), edge(write)]})
        write = 'scaled'
    return {'nodes': nodes, 'output': [edge('Sprev'), edge(write)]}


class StructuredPopulationTests(unittest.TestCase):
    def test_scalar_movement_has_same_semantic_id(self):
        left, right = outer(scaled=True), outer(scale_output=True)
        self.assertNotEqual(graph_id(left), graph_id(right))
        self.assertEqual(semantic_id(left), semantic_id(right))

    def test_outer_orientation_is_not_quotiented(self):
        self.assertNotEqual(semantic_id(outer('v', 'k')), semantic_id(outer('k', 'v')))

    def test_transpose_equivalence_is_detected(self):
        left, right = outer('v', 'k'), outer('k', 'v')
        right['output'][1]['view'] = 'transpose'
        self.assertEqual(semantic_id(left), semantic_id(right))

    def test_factored_vector_sum_matches_expanded_writes(self):
        factored = {'nodes': [
            {'name': 'sum', 'operation': 'SignedSum',
             'inputs': [{**edge('v'), 'sign': 1}, {**edge('q'), 'sign': 1}]},
            {'name': 'write', 'operation': 'MatMul',
             'inputs': [edge('sum', 'column'), edge('k', 'row')]}],
            'output': [edge('Sprev'), edge('write')]}
        expanded = {'nodes': [
            {'name': 'a', 'operation': 'MatMul',
             'inputs': [edge('v', 'column'), edge('k', 'row')]},
            {'name': 'b', 'operation': 'MatMul',
             'inputs': [edge('q', 'column'), edge('k', 'row')]}],
            'output': [edge('Sprev'), edge('a'), edge('b')]}
        self.assertEqual(polynomial(factored), polynomial(expanded))

    def test_kda_canonicalization_preserves_semantics(self):
        self.assertEqual(semantic_id(kda()), semantic_id(canonical(kda())))

    def test_conditional_sampling_is_reproducible_and_valid(self):
        args = {'count': 12, 'proposals': 4000, 'seed': 2026102002}
        first, report, _ = sample(**args)
        second, repeated_report, _ = sample(**args)
        self.assertEqual(first, second)
        self.assertEqual(report, repeated_report)
        self.assertEqual(len({x['id'] for x in first}), 12)
        self.assertEqual(len({x['semantic_id'] for x in first}), 12)
        for entry in first:
            analyze(entry['graph'])
            self.assertEqual(classify(entry['graph'])['backend'], 'dplr')
            self.assertEqual(entry['graph'], canonical(entry['graph']))
            self.assertNotEqual(entry['semantic_id'], semantic_id(kda()))

    def test_insufficient_pool_is_reported_without_padding(self):
        with self.assertRaisesRegex(RuntimeError, 'No padding performed'):
            sample(count=500, proposals=1)


if __name__ == '__main__':
    unittest.main()
