import unittest
from graph import analyze, canonical, edge, graph_id, kda


class GraphTests(unittest.TestCase):
    def test_exact_kda_budget(self):
        shapes, dependent, counts = analyze(kda())
        self.assertEqual(shapes['U'], ('d', 'd'))
        self.assertTrue(dependent['U'])
        self.assertEqual(dict(counts), {'ChannelScale': 1, 'MatMul': 2, 'SignedSum': 1, 'ScalarScale': 1})

    def test_dead_component_is_not_used(self):
        base = {'nodes': [{'name': 'write', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]}],
                'output': [edge('Sprev'), edge('write')]}
        dead = {**base, 'nodes': [*base['nodes'], {'name': 'unused', 'operation': 'MatMul', 'inputs': [edge('k'), edge('q')]}]}
        self.assertEqual(canonical(dead), canonical(base))
        self.assertEqual(graph_id(dead), graph_id(base))

    def test_stateless_injection_rejected(self):
        graph = {'nodes': [{'name': 'write', 'operation': 'MatMul',
                            'inputs': [edge('v', 'column'), edge('k', 'row')]}],
                 'output': [edge('write')]}
        with self.assertRaisesRegex(ValueError, 'Sprev must contribute'): analyze(graph)

    def test_previous_state_in_dead_branch_is_not_enough(self):
        graph = {'nodes': [
            {'name': 'dead_memory', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]},
            {'name': 'write', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]},
        ], 'output': [edge('write')]}
        with self.assertRaisesRegex(ValueError, 'Sprev must contribute'): canonical(graph)

    def test_indirect_previous_state_path_accepted(self):
        graph = {'nodes': [
            {'name': 'memory', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]},
            {'name': 'write', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]},
        ], 'output': [edge('memory'), edge('write')]}
        self.assertTrue(analyze(graph)[1]['memory'])

    def test_current_input_required_inside_state_update(self):
        with self.assertRaisesRegex(ValueError, 'Current-input features'):
            analyze({'nodes': [], 'output': [edge('Sprev')]})
        graph = {'nodes': [{'name': 'unused', 'operation': 'MatMul', 'inputs': [edge('q'), edge('k')]}],
                 'output': [edge('Sprev')]}
        with self.assertRaisesRegex(ValueError, 'Current-input features'): analyze(graph)

    def test_homogeneous_input_gating_rejected(self):
        for operation, inputs in [('ChannelScale', [edge('Sprev'), edge('alpha')]),
                                  ('ScalarScale', [edge('beta'), edge('Sprev')])]:
            graph = {'nodes': [{'name': 'memory', 'operation': operation, 'inputs': inputs}], 'output': [edge('memory')]}
            with self.assertRaisesRegex(ValueError, 'Zero-state write required'): analyze(graph)

    def test_canceled_previous_state_rejected(self):
        graph = {'nodes': [
            {'name': 'write', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]},
            {'name': 'negative', 'operation': 'SignedSum', 'inputs': [{**edge('Sprev', 'transpose'), 'sign': -1}]},
        ], 'output': [edge('Sprev'), edge('negative', 'transpose'), edge('write')]}
        with self.assertRaisesRegex(ValueError, 'Sprev contribution cancels'): analyze(graph)

    def test_canceled_write_rejected(self):
        graph = {'nodes': [
            {'name': 'memory', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]},
            {'name': 'write', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]},
            {'name': 'negative', 'operation': 'SignedSum', 'inputs': [{**edge('write'), 'sign': -1}]},
        ], 'output': [edge('memory'), edge('write'), edge('negative')]}
        with self.assertRaisesRegex(ValueError, 'Zero-state write required'): analyze(graph)

    def test_transposed_equivalent_write_cancellation_rejected(self):
        graph = {'nodes': [
            {'name': 'memory', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]},
            {'name': 'a', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('k', 'row')]},
            {'name': 'b', 'operation': 'MatMul', 'inputs': [edge('k', 'column'), edge('v', 'row')]},
            {'name': 'negative', 'operation': 'SignedSum', 'inputs': [{**edge('b', 'transpose'), 'sign': -1}]},
        ], 'output': [edge('memory'), edge('a'), edge('negative')]}
        with self.assertRaisesRegex(ValueError, 'Zero-state write required'): analyze(graph)

    def test_relabel_and_topological_order_deduplicate(self):
        graph = kda()
        renamed = {'nodes': [], 'output': []}
        names = {n['name']: 'x' + str(i) for i, n in enumerate(graph['nodes'])}
        for n in graph['nodes']:
            renamed['nodes'].append({**n, 'name': names[n['name']], 'inputs': [{**a, 'source': names.get(a['source'], a['source'])} for a in n['inputs']]})
        renamed['output'] = [{**a, 'source': names.get(a['source'], a['source'])} for a in reversed(graph['output'])]
        self.assertEqual(graph_id(graph), graph_id(renamed))

    def test_nonlinear_state_rejected(self):
        graph = {'nodes': [{'name': 'bad', 'operation': 'MatMul', 'inputs': [edge('Sprev'), edge('Sprev')]}], 'output': [edge('bad')]}
        with self.assertRaises(ValueError): analyze(graph)

    def test_repeated_transpose_sum_source_rejected(self):
        graph = {'nodes': [], 'output': [edge('Sprev'), edge('Sprev', 'transpose')]}
        with self.assertRaises(ValueError): analyze(graph)

    def test_wrong_output_shape_rejected(self):
        with self.assertRaises(ValueError): analyze({'nodes': [], 'output': [edge('q', 'row')]})

    def test_forward_reference_rejected(self):
        graph = {'nodes': [{'name': 'a', 'operation': 'ScalarScale', 'inputs': [edge('beta'), edge('b')]}], 'output': [edge('a')]}
        with self.assertRaises(ValueError): analyze(graph)


if __name__ == '__main__': unittest.main()
