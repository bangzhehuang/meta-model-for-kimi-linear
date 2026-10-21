"""Small regression fixtures for numerical tensor-polynomial correspondence."""
import unittest

import torch

from graph import edge
from influence_gate import checker_polynomial, check_graph, fixtures


class InfluenceTests(unittest.TestCase):
    def test_fp64_outputs_gradients_and_influence(self):
        torch.manual_seed(1837)
        for name, graph in fixtures():
            with self.subTest(name=name):
                check_graph(graph)

    def test_canceled_write_retains_only_state_monomial(self):
        graph = dict(fixtures())['write_cancels']
        summary, polynomial = checker_polynomial(graph)
        self.assertEqual(summary, {'previous_state': True, 'current_input': False,
                                   'zero_state_write': False, 'terms': 1})
        self.assertEqual(polynomial, { (('Sprev', ('f0', 'f1')),): 1 })

    def test_canceled_state_retains_only_feature_monomial(self):
        graph = dict(fixtures())['state_cancels']
        summary, _ = checker_polynomial(graph)
        self.assertFalse(summary['previous_state'])
        self.assertTrue(summary['current_input'])
        self.assertTrue(summary['zero_state_write'])

    def test_transposed_outer_products_cancel(self):
        graph = {'nodes': [
            {'name': 'a', 'operation': 'MatMul',
             'inputs': [edge('v', 'column'), edge('k', 'row')]},
            {'name': 'b', 'operation': 'MatMul',
             'inputs': [edge('k', 'column'), edge('v', 'row')]},
            {'name': 'cancel', 'operation': 'SignedSum',
             'inputs': [{**edge('a', 'transpose'), 'sign': -1},
                        {**edge('b'), 'sign': 1}]}],
            'output': [edge('Sprev'), edge('cancel')]}
        result = check_graph(graph)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['influence']['zero_state_write'])

    def test_vector_dot_controls_state_coefficient(self):
        graph = {'nodes': [
            {'name': 'dot', 'operation': 'MatMul', 'inputs': [edge('q'), edge('k')]},
            {'name': 'state', 'operation': 'ScalarScale', 'inputs': [edge('dot'), edge('Sprev')]},
            {'name': 'write', 'operation': 'MatMul',
             'inputs': [edge('v', 'column'), edge('k', 'row')]}],
            'output': [edge('state'), edge('write')]}
        self.assertTrue(check_graph(graph)['accepted'])


if __name__ == '__main__':
    unittest.main()
