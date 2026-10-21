"""Typed, budgeted and conditionally affine memory graphs; no implicit views."""
from collections import Counter
import hashlib
import itertools
import json
from influence import inspect_influence

PORTS = {'Sprev': ('d', 'd'), 'alpha': ('d',), 'beta': (),
         'q': ('d',), 'k': ('d',), 'v': ('d',)}
LIMITS = {'MatMul': 2, 'ScalarScale': 1, 'SignedSum': 1, 'ChannelScale': 1}
CONSTRAINT_VERSION = 'state_input_and_write_required_v4'


class InactiveInputError(ValueError):
    pass


def edge(source, view='none'):
    return {'source': source, 'view': view}


def viewed(shape, view):
    if view == 'none': return shape
    if view == 'transpose' and len(shape) == 2: return shape[::-1]
    if view == 'row' and len(shape) == 1: return (1, shape[0])
    if view == 'column' and len(shape) == 1: return (shape[0], 1)
    raise ValueError('Invalid explicit view')


def product(a, b):
    if len(a) not in (1, 2) or len(b) not in (1, 2):
        raise ValueError('MatMul needs vectors/matrices')
    if a[-1] != (b[0] if len(b) == 1 else b[-2]):
        raise ValueError('Incompatible MatMul dimensions')
    return (() if len(a) == 1 else a[:-1]) + (() if len(b) == 1 else b[-1:])


def analyze(graph):
    shapes, dependencies, counts = dict(PORTS), {k: k == 'Sprev' for k in PORTS}, Counter()
    for node in graph['nodes']:
        name, op, args = node['name'], node['operation'], node['inputs']
        if name in shapes or op not in LIMITS: raise ValueError('Duplicate name or invalid operation')
        if not args or any(a['source'] not in shapes for a in args): raise ValueError('Empty/forward/cyclic input')
        sizes = [viewed(shapes[a['source']], a.get('view', 'none')) for a in args]
        flags = [dependencies[a['source']] for a in args]
        if op == 'SignedSum':
            if len({a['source'] for a in args}) != len(args): raise ValueError('Repeated sum source')
            if any(s != sizes[0] for s in sizes): raise ValueError('Sum shape mismatch')
            if any(a.get('sign') not in (-1, 1) for a in args): raise ValueError('Invalid sign')
            shape = sizes[0]
        else:
            if len(args) != 2 or sum(flags) > 1: raise ValueError('Non-affine bilinear product')
            if op == 'MatMul': shape = product(*sizes)
            elif op == 'ScalarScale':
                if sizes[0] != (): raise ValueError('First scale operand must be scalar')
                shape = sizes[1]
            else:
                if len(sizes[0]) != 2 or sizes[1] != (sizes[0][-1],): raise ValueError('ChannelScale shape mismatch')
                shape = sizes[0]
        counts[op] += 1
        if counts[op] > LIMITS[op]: raise ValueError('Operation budget exceeded')
        shapes[name], dependencies[name] = shape, any(flags)
    output = graph['output']
    if not output or len({a['source'] for a in output}) != len(output): raise ValueError('Empty/repeated output')
    if any(a['source'] not in shapes or viewed(shapes[a['source']], a.get('view', 'none')) != ('d', 'd') for a in output):
        raise ValueError('Output must have exact state shape')
    if not any(dependencies[a['source']] for a in output):
        raise InactiveInputError('Sprev must contribute to the final state output')
    influence = inspect_influence(graph, shapes)
    if not influence['previous_state']:
        raise InactiveInputError('Sprev contribution cancels from the final state output')
    if not influence['current_input']:
        raise InactiveInputError('Current-input features must contribute to the final state output')
    if not influence['zero_state_write']:
        raise InactiveInputError('Zero-state write required: the update stays zero from zero initialization')
    return shapes, dependencies, counts


def canonical(graph):
    analyze(graph)
    by_name = {n['name']: n for n in graph['nodes']}
    used = set()
    def visit(name):
        if name in by_name and name not in used:
            used.add(name)
            for a in by_name[name]['inputs']: visit(a['source'])
    for a in graph['output']: visit(a['source'])
    nodes = [n for n in graph['nodes'] if n['name'] in used]
    encodings = []
    for order in itertools.permutations(nodes):
        names = {p: p for p in PORTS}
        result = []
        for node in order:
            if any(a['source'] not in names for a in node['inputs']): break
            args = [{**a, 'source': names[a['source']], 'view': a.get('view', 'none')} for a in node['inputs']]
            if node['operation'] == 'SignedSum': args.sort(key=lambda a: json.dumps(a, sort_keys=True))
            name = 'n' + str(len(result))
            names[node['name']] = name
            result.append({'name': name, 'operation': node['operation'], 'inputs': args})
        else:
            output = sorted([{**a, 'source': names[a['source']], 'view': a.get('view', 'none')} for a in graph['output']],
                            key=lambda a: json.dumps(a, sort_keys=True))
            encodings.append(json.dumps({'nodes': result, 'output': output}, sort_keys=True, separators=(',', ':')))
    return json.loads(min(encodings))


def graph_id(graph, canonicalized=False):
    return hashlib.sha256(json.dumps(graph if canonicalized else canonical(graph), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def stratum(graph):
    _, _, counts = analyze(graph)
    depths, consumers = {p: 0 for p in PORTS}, Counter()
    for n in graph['nodes']:
        depths[n['name']] = 1 + max(depths[a['source']] for a in n['inputs'])
        consumers.update({a['source'] for a in n['inputs']})
    consumers.update({a['source'] for a in graph['output']})
    depth = max(depths[a['source']] for a in graph['output'])
    shared = any(v > 1 for k, v in consumers.items() if k not in PORTS)
    return '_'.join(str(counts[k]) for k in LIMITS) + f'_depth{depth}_shared{int(shared)}'


def kda():
    return {'nodes': [
        {'name': 'F', 'operation': 'ChannelScale', 'inputs': [edge('Sprev'), edge('alpha')]},
        {'name': 'r', 'operation': 'MatMul', 'inputs': [edge('F'), edge('k')]},
        {'name': 'e', 'operation': 'SignedSum', 'inputs': [{**edge('v'), 'sign': 1}, {**edge('r'), 'sign': -1}]},
        {'name': 'w', 'operation': 'ScalarScale', 'inputs': [edge('beta'), edge('e')]},
        {'name': 'U', 'operation': 'MatMul', 'inputs': [edge('w', 'column'), edge('k', 'row')]},
    ], 'output': [edge('F'), edge('U')]}


def evaluate(graph, ports):
    values = dict(ports)
    def get(a):
        x, view = values[a['source']], a.get('view', 'none')
        return x.transpose(-1, -2) if view == 'transpose' else x.unsqueeze(-2) if view == 'row' else x.unsqueeze(-1) if view == 'column' else x
    for node in graph['nodes']:
        args = [get(a) for a in node['inputs']]
        op = node['operation']
        if op == 'SignedSum': value = sum(a.get('sign', 1) * x for a, x in zip(node['inputs'], args))
        elif op == 'MatMul': value = args[0] @ args[1]
        elif op == 'ChannelScale': value = args[0] * args[1].unsqueeze(-2)
        else: value = args[0] * args[1]
        values[node['name']] = value
    return sum(get(a) for a in graph['output'])
