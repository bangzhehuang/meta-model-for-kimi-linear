"""Independent FP64 CPU evaluation of the actual influence-checker polynomial."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from unittest.mock import patch

import torch

import influence
import sample as proposal
from graph import LIMITS, PORTS, InactiveInputError, analyze, edge, evaluate, kda, product, viewed


def infer_shapes(graph):
    """Infer typed shapes without calling or filtering on influence checks."""
    shapes = dict(PORTS)
    for node in graph['nodes']:
        sizes = [viewed(shapes[a['source']], a.get('view', 'none')) for a in node['inputs']]
        op = node['operation']
        shapes[node['name']] = product(*sizes) if op == 'MatMul' else sizes[1] if op == 'ScalarScale' else sizes[0]
    return shapes


def checker_polynomial(graph):
    """Capture inspect_influence's final plus result, not a reimplementation."""
    recorded = []
    original_plus = influence.plus

    def capture(values):
        result = original_plus(values)
        recorded.append(result)
        return result

    with patch.object(influence, 'plus', capture):
        summary = influence.inspect_influence(graph, infer_shapes(graph))
    return summary, recorded[-1]


def evaluate_polynomial(polynomial, ports):
    result = ports['Sprev'] * 0
    for term, coefficient in polynomial.items():
        indices = sorted({i for _, labels in term for i in labels})
        mapping = {i: n for n, i in enumerate(indices)}
        operands = []
        for port, labels in term:
            operands.extend((ports[port], [mapping[i] for i in labels]))
        operands.append([mapping['f0'], mapping['f1']])
        result = result + coefficient * torch.einsum(*operands)
    return result


def fixtures():
    write = {'name': 'w', 'operation': 'MatMul',
             'inputs': [edge('v', 'column'), edge('k', 'row')]}
    decay = {'name': 'd', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]}
    yield 'exact_kda', kda()
    yield 'state_only', {'nodes': [], 'output': [edge('Sprev')]}
    yield 'homogeneous_decay', {'nodes': [decay], 'output': [edge('d')]}
    yield 'write_cancels', {'nodes': [write, {'name': 'r', 'operation': 'SignedSum',
        'inputs': [{**edge('Sprev'), 'sign': 1}, {**edge('w'), 'sign': -1}]}],
        'output': [edge('w'), edge('r')]}
    yield 'state_cancels', {'nodes': [write, {'name': 'r', 'operation': 'SignedSum',
        'inputs': [{**edge('Sprev'), 'sign': -1}, {**edge('w'), 'sign': 1}]}],
        'output': [edge('Sprev'), edge('r')]}
    yield 'transposed_decay_write', {'nodes': [decay, write],
        'output': [edge('d', 'transpose'), edge('w', 'transpose')]}
    yield 'singleton_dot_then_outer', {'nodes': [
        {'name': 'r', 'operation': 'MatMul', 'inputs': [edge('q', 'row'), edge('k', 'column')]},
        {'name': 'vrow', 'operation': 'MatMul', 'inputs': [edge('r'), edge('v', 'row')]},
        {'name': 'write', 'operation': 'ScalarScale', 'inputs': [edge('beta'), edge('Sprev')]}],
        'output': [edge('write')]}
    yield 'singleton_transpose_outer', {'nodes': [
        {'name': 'row', 'operation': 'SignedSum', 'inputs': [{**edge('v', 'row'), 'sign': -1}]},
        {'name': 'w', 'operation': 'MatMul', 'inputs': [edge('row', 'transpose'), edge('k', 'row')]}],
        'output': [edge('Sprev'), edge('w')]}
    yield 'row_channel_then_outer', {'nodes': [
        {'name': 'row', 'operation': 'ChannelScale', 'inputs': [edge('v', 'row'), edge('alpha')]},
        {'name': 'w', 'operation': 'MatMul', 'inputs': [edge('q', 'column'), edge('row')]}],
        'output': [edge('Sprev'), edge('w')]}
    yield 'left_state_read_then_write', {'nodes': [
        {'name': 'r', 'operation': 'MatMul', 'inputs': [edge('q', 'row'), edge('Sprev')]},
        {'name': 'w', 'operation': 'MatMul', 'inputs': [edge('v', 'column'), edge('r')]}],
        'output': [edge('Sprev'), edge('w')]}


def check_graph(graph, dimensions=(2, 3), draws=3):
    summary, polynomial = checker_polynomial(graph)
    observed = dict.fromkeys(('previous_state', 'current_input', 'zero_state_write'), False)
    maximum_output_error, maximum_gradient_error = 0., 0.
    names = tuple(PORTS)
    for dimension in dimensions:
        for _ in range(draws):
            ports = {p: torch.randn(tuple(dimension if s == 'd' else s for s in shape),
                                    dtype=torch.float64, requires_grad=True)
                     for p, shape in PORTS.items()}
            direct = evaluate(graph, ports)
            expanded = evaluate_polynomial(polynomial, ports)
            error = (direct - expanded).abs().max().item()
            maximum_output_error = max(maximum_output_error, error)
            torch.testing.assert_close(expanded, direct, atol=1e-9, rtol=1e-10)
            weights = torch.randn_like(direct)
            arguments = [ports[p] for p in names]
            direct_gradients = torch.autograd.grad((direct * weights).sum(), arguments,
                                                   allow_unused=True, retain_graph=True)
            expanded_gradients = torch.autograd.grad((expanded * weights).sum(), arguments,
                                                     allow_unused=True)
            norms = {}
            for name, port, a, b in zip(names, arguments, direct_gradients, expanded_gradients):
                a = torch.zeros_like(port) if a is None else a
                b = torch.zeros_like(port) if b is None else b
                maximum_gradient_error = max(maximum_gradient_error, (a-b).abs().max().item())
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-10)
                norms[name] = a.abs().max().item()
            observed['previous_state'] |= norms['Sprev'] > 1e-8
            observed['current_input'] |= any(norms[p] > 1e-8 for p in names if p != 'Sprev')
            zero_ports = {**ports, 'Sprev': torch.zeros_like(ports['Sprev'])}
            observed['zero_state_write'] |= evaluate(graph, zero_ports).abs().max().item() > 1e-8
    expected = {k: summary[k] for k in observed}
    if observed != expected:
        raise AssertionError(f'Influence disagreement: symbolic={expected}, numerical={observed}')
    try:
        analyze(graph)
        accepted = True
    except InactiveInputError:
        accepted = False
    if accepted != all(expected.values()):
        raise AssertionError(f'Validity disagreement: accepted={accepted}, influence={expected}')
    return {'accepted': accepted, 'maximum_output_absolute_error': maximum_output_error,
            'maximum_gradient_absolute_error': maximum_gradient_error, 'influence': expected}


def run(proposals=20000, seed=2026102051, draws=3):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    budgets = list(itertools.product(*(range(n + 1) for n in LIMITS.values())))
    counts, views, operations, flags = Counter(), Counter(), Counter(), Counter()
    maximum_output_error, maximum_gradient_error = 0., 0.
    failures = []

    def inspect(label, graph):
        nonlocal maximum_output_error, maximum_gradient_error
        counts['graphs_checked'] += 1
        views.update(a.get('view', 'none') for n in graph['nodes'] for a in n['inputs'])
        operations.update(n['operation'] for n in graph['nodes'])
        try:
            result = check_graph(graph, draws=draws)
            counts['accepted' if result['accepted'] else 'influence_rejected'] += 1
            flags.update(k for k, v in result['influence'].items() if v)
            maximum_output_error = max(maximum_output_error, result['maximum_output_absolute_error'])
            maximum_gradient_error = max(maximum_gradient_error, result['maximum_gradient_absolute_error'])
        except Exception as error:
            failures.append({'label': label, 'error': str(error)[:2000], 'graph': graph})

    for label, graph in fixtures():
        inspect(label, graph)
    # Keep typed/affine proposal construction but bypass its influence-based
    # canonical admission, so rejected graphs also reach this independent gate.
    with patch.object(proposal, 'canonical', lambda graph: graph):
        for index in range(proposals):
            graph = proposal.propose(rng, budgets[index % len(budgets)])
            if graph is None:
                counts['no_state_path_or_unconstructible_proposals'] += 1
            else:
                inspect(f'proposal_{index}', graph)
    directory = Path(__file__).resolve().parent
    return {'passed': not failures, 'seed': seed, 'proposals': proposals,
            'dimensions': [2, 3], 'draws_per_dimension': draws, 'dtype': 'torch.float64',
            'device': 'cpu', 'torch_version': torch.__version__, 'counts': dict(counts),
            'input_view_coverage': dict(views), 'operation_coverage': dict(operations),
            'positive_influence_counts': dict(flags),
            'maximum_output_absolute_error': maximum_output_error,
            'maximum_gradient_absolute_error': maximum_gradient_error,
            'failures': failures, 'source_sha256': {
                name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                for name in ('influence_gate.py', 'influence.py', 'graph.py', 'sample.py')}}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--proposals', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=2026102051)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = run(args.proposals, args.seed, args.draws)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'failures'}))
    print(json.dumps({'failure_count': len(report['failures']), 'first_failures': report['failures'][:2]}))
    raise SystemExit(0 if report['passed'] else 1)
