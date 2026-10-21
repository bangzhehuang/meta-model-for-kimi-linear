"""Focused CUDA correctness and throughput gate; this dispatches no search jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DURABLE, SCRATCH, atomic_json
from graph import canonical, edge, evaluate, kda
from backend import recurrent
from sample import propose


def reference(graph, q, k, v, alpha, beta):
    batches = []
    for b in range(q.shape[0]):
        heads = []
        for h in range(q.shape[2]):
            state = torch.zeros((q.shape[-1], q.shape[-1]), device=q.device)
            result = []
            for t in range(q.shape[1]):
                state = evaluate(graph, {'Sprev': state, 'q': q[b,t,h].float(), 'k': k[b,t,h].float(),
                                        'v': v[b,t,h].float(), 'alpha': alpha[b,t,h].float(), 'beta': beta[b,t,h].float()})
                result.append((state @ q[b,t,h].float()) / q.shape[-1] ** .5)
            heads.append(torch.stack(result))
        batches.append(torch.stack(heads, dim=1))
    return torch.stack(batches).to(q.dtype)


def inputs(b, t, h, d, dtype=torch.float32):
    values = [F.normalize(torch.randn(b,t,h,d,device='cuda'), dim=-1),
              F.normalize(torch.randn(b,t,h,d,device='cuda'), dim=-1),
              torch.randn(b,t,h,d,device='cuda') * .1,
              torch.rand(b,t,h,d,device='cuda') * .1 + .85,
              torch.rand(b,t,h,device='cuda') * .5]
    return [v.to(dtype).detach().requires_grad_() for v in values]


def gradients(y, values, dy):
    if not y.requires_grad: return [torch.zeros_like(v) for v in values]
    return [torch.zeros_like(v) if g is None else g for v, g in zip(values, torch.autograd.grad((y * dy).sum(), values, allow_unused=True))]


def check(graph, directory, index):
    values = inputs(1, 17, 2, 32)
    other = [v.detach().clone().requires_grad_() for v in values]
    expected = reference(graph, *values)
    actual = recurrent(graph, directory)(*other)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    dy = torch.randn_like(expected)
    want, got = gradients(expected, values, dy), gradients(actual, other, dy)
    for p, (a, b) in enumerate(zip(want, got)):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3, msg=f'graph{index} input{p}')
    return {'graph': index, 'maximum_forward_error': (actual-expected).abs().max().item()}


def time_call(call, repetitions):
    times = []
    for i in range(repetitions + 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        if i >= 2: times.append(time.perf_counter()-start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', type=int, default=12)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(94123)
    lock = open('/tmp/kimi_synthetic_gpu_' + os.environ.get('CUDA_VISIBLE_DEVICES','0') + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    directory = SCRATCH / 'search_prototype' / 'kernels'
    graphs = [canonical(kda())]
    def node(name, op, *args): return {'name': name, 'operation': op, 'inputs': list(args)}
    outer = node('O', 'MatMul', edge('v', 'column'), edge('k', 'row'))
    graphs += [canonical({'nodes': [outer], 'output': [edge('O'), edge('Sprev', 'transpose')]}),
               canonical({'nodes': [outer, node('M', 'MatMul', edge('Sprev'), edge('O'))], 'output': [edge('O'), edge('M')]}),
               canonical({'nodes': [outer, node('M', 'MatMul', edge('O'), edge('Sprev'))], 'output': [edge('O'), edge('M')]}),
               canonical({'nodes': [outer, node('a', 'MatMul', edge('q'), edge('k')), node('M', 'ScalarScale', edge('a'), edge('Sprev'))], 'output': [edge('O'), edge('M')]})]
    column_kda = kda()
    column_kda['nodes'][1]['inputs'][1] = edge('k', 'column')
    column_kda['nodes'][2]['inputs'][0] = {**edge('v', 'column'), 'sign': 1}
    column_kda['nodes'][-1]['inputs'][0] = edge('w')
    graphs.append(canonical(column_kda))
    rng = random.Random(523)
    while len(graphs) < args.checks:
        graph = propose(rng, (2, 1, 1, 1))
        if graph and graph['nodes']: graphs.append(graph)
    checked = []
    for i, graph in enumerate(graphs):
        checked.append(check(graph, directory, i))
        print(json.dumps({'check': i, 'status': 'passed', **checked[-1]}), flush=True)
    report = {'status': 'passed', 'checks': checked, 'time_unix': time.time(), 'time_parallel_scan': False}
    if args.benchmark:
        from fla.ops.kda import chunk_kda
        from fla.modules.l2norm import l2_norm
        q,k,v,alpha,beta = inputs(32,2048,2,128,torch.bfloat16)
        cell = recurrent(kda(), directory)
        def custom():
            cell(q,k,v,alpha.float(),beta).float().square().mean().backward()
            for x in (q,k,v,alpha,beta): x.grad = None
        def official():
            chunk_kda(q,k,v,alpha.float().log(),beta,use_qk_l2norm_in_kernel=False)[0].float().square().mean().backward()
            for x in (q,k,v,alpha,beta): x.grad = None
        report['cell_forward_backward_seconds'] = time_call(custom, 3)
        report['official_cell_forward_backward_seconds'] = time_call(official, 3)
        report['peak_memory_mib'] = torch.cuda.max_memory_allocated()/2**20
    atomic_json(DURABLE / 'search_1000_v1' / 'prototype' / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
