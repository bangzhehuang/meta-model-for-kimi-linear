"""CPU algebra checks before exercising the CUDA implementation."""
import random
import unittest

from graph import canonical, edge, kda
from lower import Matrix, ONE, ZERO, classify, normalize
from sample import propose


class LoweringTests(unittest.TestCase):
    def test_exact_kda_contained(self):
        info = classify(kda())
        self.assertEqual(info['backend'], 'kda')
        plan = info['plan']
        self.assertEqual(plan.diagonal, ('input', 'alpha'))
        self.assertEqual(len(plan.ranks), 1)
        self.assertEqual(len(plan.offsets), 1)

    def test_homogeneous_transpose_is_rejected(self):
        graph = {'nodes': [{'name': 'memory', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]}],
                 'output': [edge('memory', 'transpose')]}
        with self.assertRaisesRegex(ValueError, 'Zero-state write required'): classify(graph)

    def test_column_view_kda_uses_official_mapping(self):
        graph = kda()
        graph['nodes'][1]['inputs'][1] = edge('k', 'column')
        graph['nodes'][2]['inputs'][0] = {**edge('v', 'column'), 'sign': 1}
        graph['nodes'][-1]['inputs'][0] = edge('w')
        self.assertEqual(classify(graph)['backend'], 'kda')

    def test_unsupported_is_valid(self):
        graph = {'nodes': [{'name': 'O', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]}],
                 'output': [edge('O'), edge('Sprev', 'transpose')]}
        self.assertEqual(classify(graph)['backend'], 'recurrent')
        canonical(graph)

    def test_proposals_do_not_crash_classifier(self):
        rng = random.Random(532)
        for _ in range(1000):
            graph = propose(rng, (2, 1, 1, 1))
            if graph is not None: self.assertIn(classify(graph)['backend'], ('kda', 'dplr', 'recurrent'))


if __name__ == '__main__': unittest.main()
