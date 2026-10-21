"""Generate fused recurrent forward and reverse-mode kernels for a typed DAG.

Time is sequential inside each head; independent sequences/heads run in parallel.
The complete state tape is FP32. This is a general backend, not a claim of a
cheap associative prefix implementation for every affine map.
"""
import importlib.util
import hashlib
import math
import os
from pathlib import Path
import sys

from graph import PORTS, analyze, canonical, graph_id, viewed


def total(expr, shape):
    for _ in shape: expr = f'tl.sum({expr}, 0)'
    return expr


def matmul(a, b, ashape, bshape):
    if len(ashape) == len(bshape) == 1: return f'tl.sum({a} * {b}, 0)'
    if len(bshape) == 1: return f'tl.sum({a} * {b}[None, :], 1)'
    if len(ashape) == 1: return f'tl.sum({a}[:, None] * {b}, 0)'
    m, n = ashape
    _, p = bshape
    if n == 1: return f'({a} * {b})'
    if m == p == 1: return f'({total(f"({a}.T * {b})", (n, 1))})[None, None]'
    if p == 1: return f'tl.sum({a} * {b}.T, 1)[:, None]'
    if m == 1: return f'tl.sum({a}.T * {b}, 0)[None, :]'
    return f'tl.dot({a}, {b}, input_precision="ieee")'


def source(graph):
    shapes, dependencies, _ = analyze(graph)
    stateful = any(dependencies[a['source']] for a in graph['output'])
    def shape_expr(shape): return '(' + ', '.join('D' if d == 'd' else str(d) for d in shape) + (',' if len(shape) == 1 else '') + ')'
    def val(a):
        name, view = a['source'], a.get('view', 'none')
        return name + ('.T' if view == 'transpose' else '[None, :]' if view == 'row' else '[:, None]' if view == 'column' else '')
    def size(a): return viewed(shapes[a['source']], a.get('view', 'none'))
    def add(a, gradient):
        view = a.get('view', 'none')
        gradient = f'({gradient})'
        if view == 'transpose': gradient += '.T'
        elif view == 'row': gradient = f'tl.sum({gradient}, 0)'
        elif view == 'column': gradient = f'tl.sum({gradient}, 1)'
        return f'd_{a["source"]} = d_{a["source"]} + {gradient}'
    forward = []
    for node in graph['nodes']:
        op, args = node['operation'], node['inputs']
        a = [val(x) for x in args]
        if op == 'MatMul': expr = matmul(*a, *(size(x) for x in args))
        elif op == 'ScalarScale': expr = f'{a[0]} * {a[1]}'
        elif op == 'ChannelScale': expr = f'{a[0]} * {a[1]}[None, :]'
        else: expr = ' + '.join(('' if x['sign'] == 1 else '-') + y for x, y in zip(args, a))
        forward.append(f'{node["name"]} = {expr}')
    forward.append('St = ' + ' + '.join(val(a) for a in graph['output']))
    backward = [f'd_{name} = tl.full({shape_expr(shape)}, 0, tl.float32)' for name, shape in shapes.items()]
    backward += [add(a, 'ds') for a in graph['output']]
    for node in reversed(graph['nodes']):
        args, op, g = node['inputs'], node['operation'], 'd_' + node['name']
        a = [val(x) for x in args]
        sizes = [size(x) for x in args]
        if op == 'SignedSum': derivatives = [g if x['sign'] == 1 else '-' + g for x in args]
        elif op == 'ScalarScale': derivatives = [total(f'({g} * {a[1]})', sizes[1]), f'{g} * {a[0]}']
        elif op == 'ChannelScale': derivatives = [f'{g} * {a[1]}[None, :]', f'tl.sum({g} * {a[0]}, 0)']
        elif len(sizes[0]) == len(sizes[1]) == 1: derivatives = [f'{g} * {a[1]}', f'{g} * {a[0]}']
        elif len(sizes[1]) == 1:
            derivatives = [f'{g}[:, None] * {a[1]}[None, :]', matmul(f'{a[0]}.T', g, sizes[0][::-1], shapes[node['name']])]
        elif len(sizes[0]) == 1:
            derivatives = [matmul(g, f'{a[1]}.T', shapes[node['name']], sizes[1][::-1]), f'{a[0]}[:, None] * {g}[None, :]']
        else:
            derivatives = [matmul(g, f'{a[1]}.T', shapes[node['name']], sizes[1][::-1]),
                           matmul(f'{a[0]}.T', g, sizes[0][::-1], shapes[node['name']])]
        backward += [add(arg, value) for arg, value in zip(args, derivatives)]
    # Every token has its own feature-gradient address, so no atomics are needed.
    prelude = '''import triton
import triton.language as tl

'''
    signature = 'Q, K, V, A, B, Y, Tape, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr'
    indexing = ['bh = tl.program_id(0)', 'batch = bh // H', 'head = bh % H', 'd = tl.arange(0, D)',
                'ij = d[:, None] * D + d[None, :]']
    loads = ['offset = ((batch * T + t) * H + head) * D',
             'q = tl.load(Q + offset + d).to(tl.float32)', 'k = tl.load(K + offset + d).to(tl.float32)',
             'v = tl.load(V + offset + d).to(tl.float32)', 'alpha = tl.load(A + offset + d).to(tl.float32)',
             'beta = tl.load(B + (batch * T + t) * H + head).to(tl.float32)',
             'tape_offset = ((batch * T + t) * H + head) * D * D']
    def function(name, sig, lines): return '@triton.jit\ndef ' + name + '(' + sig + '):\n' + '\n'.join('    ' + line for line in lines) + '\n\n'
    if not stateful:
        indexing = ['task = tl.program_id(0)', 't = task % T', 'bh = task // T'] + indexing[1:]
    fbody = indexing + ['Sprev = tl.full((D, D), 0, tl.float32)']
    if stateful: fbody += ['for t in range(T):']
    loop = loads + (['tl.store(Tape + tape_offset + ij, Sprev)'] if stateful else []) + forward
    loop += ['y = tl.sum(St * q[None, :], 1) / tl.sqrt(float(D))', 'tl.store(Y + offset + d, y)', 'Sprev = St']
    fbody += [('    ' if stateful else '') + line for line in loop]
    bbody = indexing + ['carry = tl.full((D, D), 0, tl.float32)']
    if stateful: bbody += ['for reverse_t in range(T):']
    loop = (['t = T - 1 - reverse_t'] if stateful else []) + loads + ['Sprev = ' + ('tl.load(Tape + tape_offset + ij)' if stateful else 'tl.full((D, D), 0, tl.float32)')] + forward
    loop += ['dy = tl.load(DY + offset + d).to(tl.float32) / tl.sqrt(float(D))', 'ds = carry + dy[:, None] * q[None, :]'] + backward
    loop += ['d_q = d_q + tl.sum(St * dy[:, None], 0)', 'carry = d_Sprev']
    loop += [f'tl.store({pointer} + offset + d, d_{port})' for pointer, port in [('DQ', 'q'), ('DK', 'k'), ('DV', 'v'), ('DA', 'alpha')]]
    loop += ['tl.store(DB + (batch * T + t) * H + head, d_beta)']
    bbody += [('    ' if stateful else '') + line for line in loop]
    return prelude + f'RECURRENT = {stateful}\n\n' + function('forward', signature, fbody) + function('backward', signature + ', DY, DQ, DK, DV, DA, DB', bbody)


def load_kernels(graph, directory):
    graph = canonical(graph)
    digest = graph_id(graph, canonicalized=True)
    code = source(graph)
    version = hashlib.sha256(code.encode()).hexdigest()[:16]
    path = Path(directory) / ('g_' + digest + '_' + version + '.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != code: raise RuntimeError('Generated source identity mismatch')
    else:
        temporary = path.with_suffix('.' + str(os.getpid()) + '.tmp')
        temporary.write_text(code)
        os.replace(temporary, path)
    spec = importlib.util.spec_from_file_location('memory_graph_' + digest + '_' + version, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recurrent(graph, directory):
    import torch
    kernels = load_kernels(graph, directory)
    class Memory(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, alpha, beta):
            q, k, v, alpha, beta = [x.contiguous() for x in (q, k, v, alpha, beta)]
            b, t, h, d = q.shape
            tape = torch.empty((b, t, h, d, d) if kernels.RECURRENT else (1,), device=q.device, dtype=torch.float32)
            y = torch.empty_like(v)
            kernels.forward[(b*h if kernels.RECURRENT else b*t*h,)](q, k, v, alpha, beta, y, tape, t, h, d, num_warps=8, enable_fp_fusion=False)
            ctx.save_for_backward(q, k, v, alpha, beta, tape)
            return y
        @staticmethod
        def backward(ctx, dy):
            q, k, v, alpha, beta, tape = ctx.saved_tensors
            b, t, h, d = q.shape
            gradients = [torch.empty_like(x) for x in (q, k, v, alpha, beta)]
            kernels.backward[(b*h if kernels.RECURRENT else b*t*h,)](q, k, v, alpha, beta, dy, tape, t, h, d,
                                    dy.contiguous(), *gradients, num_warps=8, enable_fp_fusion=False)
            return tuple(gradients)
    return Memory.apply
