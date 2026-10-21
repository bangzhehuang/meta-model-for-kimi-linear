import unittest

from affinity import AffinityError, Node, affine_dependencies, kda_graph


class AffinityTests(unittest.TestCase):
    def test_kda_and_shared_state_branch(self):
        result = affine_dependencies(kda_graph())
        self.assertTrue(all(result[k] for k in ('F', 'r', 'e', 'w', 'U', 'St')))
        self.assertTrue(all(not result[k] for k in ('q', 'k', 'v', 'alpha', 'beta')))

    def test_input_only_outer_product(self):
        self.assertFalse(affine_dependencies([Node('U', 'MatMul', ('v', 'k'))])['U'])

    def test_state_square_rejected(self):
        with self.assertRaises(AffinityError):
            affine_dependencies([Node('square', 'MatMul', ('Sprev', 'Sprev'))])

    def test_all_bilinear_nodes_reject_two_dependent_operands(self):
        for op in ('MatMul', 'ScalarScale', 'ChannelScale'):
            with self.subTest(op=op), self.assertRaises(AffinityError):
                affine_dependencies([Node('a', 'Transpose', ('Sprev',)),
                                     Node('bad', op, ('Sprev', 'a'))])

    def test_later_cancellation_does_not_rescue_bad_product(self):
        with self.assertRaises(AffinityError):
            affine_dependencies([Node('bad', 'MatMul', ('Sprev', 'Sprev')),
                                 Node('zero', 'SignedSum', ('bad', 'bad'))])

    def test_state_nonlinearity_and_cycles_rejected(self):
        for nodes in ([Node('bad', 'RMSNorm', ('Sprev',))],
                      [Node('loop', 'OutputSum', ('loop',))]):
            with self.assertRaises(AffinityError):
                affine_dependencies(nodes)


if __name__ == '__main__':
    unittest.main()
