"""Immutable CUDA correctness/throughput evidence; dispatches no training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import torch

from backend import recurrent
from fast import PROTOCOL, compile_cell
from graph import canonical, edge, graph_id, kda
from lower import classify
from prototype import gradients, inputs, reference
from sample import propose


def errors(actual, expected):
    a, e = actual.double(), expected.double()
    return {'relative_l2': ((a-e).norm() / e.norm().clamp_min(1e-7)).item(),
            'maximum_absolute': (a-e).abs().max().item(),
            'finite': bool(torch.isfinite(a).all() & torch.isfinite(e).all())}


def check(graph, directory, label, factor_dtype=None):
    values = inputs(1, 65, 1, 32, torch.bfloat16)
    # The model passes FP32 exp(log_decay); preserve this precision contract.
    values[3] = values[3].detach().float().requires_grad_()
    expected_inputs = [v.detach().float().requires_grad_() for v in values]
    generic_inputs = [v.detach().clone().requires_grad_() for v in values]
    expected = reference(graph, *expected_inputs)
    actual, info = compile_cell(graph, directory, require_fast=True)
    if factor_dtype is not None:
        original = actual
        def actual(q, k, v, alpha, beta):
            return original(q.to(factor_dtype), k.to(factor_dtype), v.to(factor_dtype),
                            alpha, beta.to(factor_dtype)).to(v.dtype)
    output = actual(*values)
    if output.dtype != values[2].dtype: raise AssertionError('Output dtype changed')
    generic = recurrent(graph, directory)(*generic_inputs)
    dy = torch.randn_like(output)
    want = gradients(expected, expected_inputs, dy.float())
    raw_got = torch.autograd.grad((output * dy).sum(), values, allow_unused=True)
    if any(g is None for g in raw_got):
        raise AssertionError('Fast backend must materialize zero gradients like the generic backend')
    got = list(raw_got)
    baseline = gradients(generic, generic_inputs, dy)
    result = {'label': label, 'candidate_id': graph_id(graph),
              'backend': info['backend'], 'implementation': info['implementation'],
              'factor_dtype_override': str(factor_dtype),
              'forward': errors(output, expected),
              'gradients': [errors(a, b) for a, b in zip(got, want)],
              'generic_forward': errors(generic, expected),
              'generic_gradients': [errors(a, b) for a, b in zip(baseline, want)]}
    if info['backend'] == 'kda':
        from fla.ops.kda import chunk_kda
        other = [v.detach().clone().requires_grad_() for v in values]
        official = chunk_kda(*other[:3], other[3].log(), other[4],
                             use_qk_l2norm_in_kernel=False)[0]
        official_grads = gradients(official, other, dy)
        result['official_forward'] = errors(output, official)
        result['official_gradients'] = [errors(a, b) for a, b in zip(got, official_grads)]
    checks = [(result['forward'], .03), *[(e, .10) for e in result['gradients']]]
    result['passed'] = all(e['finite'] and (e['relative_l2'] <= tol or e['maximum_absolute'] <= 1e-5)
                           for e, tol in checks)
    return result


def examples(count):
    def node(name, op, *args): return {'name': name, 'operation': op, 'inputs': list(args)}
    outer = node('O', 'MatMul', edge('v', 'column'), edge('k', 'row'))
    decay = node('F', 'ChannelScale', edge('Sprev'), edge('alpha'))
    graphs = [('kda', canonical(kda())),
              ('diagonal_write', canonical({'nodes': [outer, decay], 'output': [edge('O'), edge('F')]})),
              ('signed_diagonal_write', canonical({'nodes': [outer, decay,
                  node('N', 'SignedSum', {**edge('O'), 'sign': 1}, {**edge('F'), 'sign': -1})],
                  'output': [edge('N')]})),
              ('two_writes', canonical({'nodes': [outer, decay,
                  node('P', 'MatMul', edge('k', 'column'), edge('v', 'row'))],
                  'output': [edge('O'), edge('P'), edge('F')]}))]
    rng = random.Random(2026102017)
    seen = {graph_id(g) for _, g in graphs}
    attempts = 0
    while len(graphs) < count and attempts < 100000:
        attempts += 1
        graph = propose(rng, (2, 1, 1, 1))
        if graph is None or classify(graph)['backend'] != 'dplr': continue
        key = graph_id(graph)
        if key not in seen:
            seen.add(key)
            graphs.append(('proposal_' + str(attempts), graph))
    return graphs


def timing(call, repeats):
    values = []
    for i in range(repeats + 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        if i >= 2: values.append(time.perf_counter()-start)
    return statistics.median(values)


def benchmark(graph, directory, repeats):
    values = inputs(32, 2048, 2, 128, torch.bfloat16)
    values[3] = values[3].detach().float().requires_grad_()
    cell, info = compile_cell(graph, directory, require_fast=True)
    def run(fn):
        fn(*values).float().square().mean().backward()
        for x in values: x.grad = None
    torch.cuda.reset_peak_memory_stats()
    seconds = timing(lambda: run(cell), repeats)
    peak = torch.cuda.max_memory_allocated()/2**20
    result = {'backend': info['backend'], 'shape': [32, 2048, 2, 128],
              'forward_backward_seconds': seconds, 'peak_memory_mib': peak}
    if info['backend'] == 'kda':
        from fla.ops.kda import chunk_kda
        official = lambda q, k, v, a, b: chunk_kda(q, k, v, a.log(), b,
                                                  use_qk_l2norm_in_kernel=False)[0]
        result['official_seconds'] = timing(lambda: run(official), repeats)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--checks', type=int, default=12)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    report_path = Path(args.report)
    if report_path.exists(): raise FileExistsError(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open('/tmp/kimi_synthetic_gpu_' + os.environ.get('CUDA_VISIBLE_DEVICES', '0') + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(94123)
    from common import SCRATCH
    directory = SCRATCH / 'fast_gate_kernels'
    report = {'protocol': PROTOCOL, 'started_unix': time.time(), 'checks': [],
              'output_relative_l2_limit': .03, 'gradient_relative_l2_limit': .10,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'source_sha256': {p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                for p in ('fast.py', 'lower.py', 'graph.py', 'influence.py', 'backend.py', 'fast_gate.py')}}
    try:
        graphs = examples(args.checks)
        for label, graph in graphs:
            try:
                checked = check(graph, directory, label)
            except Exception:
                checked = {'label': label, 'candidate_id': graph_id(graph),
                           'backend': classify(graph)['backend'], 'passed': False,
                           'exception': traceback.format_exc()[-6000:]}
            report['checks'].append(checked)
            print(json.dumps({key: checked[key] for key in ('label', 'backend', 'passed')}
                             | ({'forward': checked['forward'],
                                 'maximum_gradient_relative_l2': max(x['relative_l2'] for x in checked['gradients'])}
                                if 'forward' in checked else {'exception': checked['exception'][-500:]})), flush=True)
        report['passed'] = all(x['passed'] for x in report['checks'])
        if args.benchmark:
            report['benchmarks'] = [benchmark(graph, directory, args.repeats)
                for label, graph in graphs[:2]
                if any(x['label'] == label and x['passed'] for x in report['checks'])]
        report['status'] = 'passed' if report['passed'] else 'failed'
    except Exception:
        report['status'] = 'failed'
        report['exception'] = traceback.format_exc()[-6000:]
    finally:
        report['finished_unix'] = time.time()
        with report_path.open('x') as stream: json.dump(report, stream, indent=2)
    print(json.dumps({'report': str(report_path), 'status': report['status']}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__': sys.exit(main())
