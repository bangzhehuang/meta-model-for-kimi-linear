"""Conservative symbolic lowering; failure here never invalidates a DSL graph.

Recognize S' = S diag(d) + sum((S a) b.T) + sum(v k.T).
Feature expressions contain no state and are evaluated over all time steps.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from graph import PORTS, analyze, canonical, kda, product, viewed


ZERO = ('number', 0)
ONE = ('number', 1)


def mul(*items):
    flat, number = [], 1
    for item in items:
        values = item[1:] if item[0] == 'mul' else (item,)
        for value in values:
            if value[0] == 'number': number *= value[1]
            else: flat.append(value)
    if number == 0: return ZERO
    if number != 1 or not flat: flat.append(('number', number))
    return flat[0] if len(flat) == 1 else ('mul', *sorted(flat, key=repr))


def add(*items):
    counts = Counter()
    for item in items:
        for value in item[1:] if item[0] == 'add' else (item,):
            coefficient = 1
            if value[0] == 'mul':
                factors = []
                for factor in value[1:]:
                    if factor[0] == 'number': coefficient *= factor[1]
                    else: factors.append(factor)
                value = mul(*factors)
            if value != ZERO: counts[value] += coefficient
    values = [mul(('number', c), value) for value, c in sorted(counts.items(), key=lambda x: repr(x[0])) if c]
    return ZERO if not values else values[0] if len(values) == 1 else ('add', *values)


def dot(a, b):
    if a == ZERO or b == ZERO: return ZERO
    return ('dot', *sorted((a, b), key=repr))


def merge_terms(terms):
    terms = [(a, b) for a, b in terms if a != ZERO and b != ZERO]
    for side in (0, 1):
        groups = defaultdict(list)
        for a, b in terms: groups[(a, b)[side]].append((a, b)[1-side])
        terms = [(key, add(*values)) if side == 0 else (add(*values), key) for key, values in groups.items()]
        terms = [(a, b) for a, b in terms if a != ZERO and b != ZERO]
    return tuple(terms)


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Vector:
    constant: tuple = ZERO
    reads: tuple = ()  # (u, scalar): (S @ u) * scalar


@dataclass(frozen=True)
class Matrix:
    diagonal: tuple = ZERO
    ranks: tuple = ()  # (a, b): (S @ a) @ b.T
    offsets: tuple = ()  # (v, k): v @ k.T


def stateful(value):
    return isinstance(value, Vector) and bool(value.reads) or isinstance(value, Matrix) and (value.diagonal != ZERO or bool(value.ranks))


def scale(value, scalar):
    if isinstance(value, Matrix):
        return Matrix(mul(value.diagonal, scalar),
                      tuple((a, mul(b, scalar)) for a, b in value.ranks),
                      tuple((mul(v, scalar), k) for v, k in value.offsets))
    if isinstance(value, Vector):
        return Vector(mul(value.constant, scalar), tuple((u, mul(c, scalar)) for u, c in value.reads))
    return mul(value, scalar)


def plus(values):
    if isinstance(values[0], Matrix):
        return Matrix(add(*(v.diagonal for v in values)),
                      merge_terms([term for v in values for term in v.ranks]),
                      merge_terms([term for v in values for term in v.offsets]))
    if isinstance(values[0], Vector):
        return Vector(add(*(v.constant for v in values)), merge_terms([term for v in values for term in v.reads]))
    return add(*values)


def read(matrix, vector):
    if vector.reads: raise Unsupported('State-dependent right operand')
    u = vector.constant
    reads = list(matrix.ranks)
    reads = [(a, dot(b, u)) for a, b in reads]
    if matrix.diagonal != ZERO: reads.append((mul(matrix.diagonal, u), ONE))
    return Vector(add(*(mul(v, dot(k, u)) for v, k in matrix.offsets)), merge_terms(reads))


def outer(left, right):
    if right.reads: raise Unsupported('State-dependent outer-product right operand')
    return Matrix(ranks=merge_terms([(a, mul(c, right.constant)) for a, c in left.reads]),
                  offsets=merge_terms([(left.constant, right.constant)]))


def channel(matrix, vector):
    if vector.reads: raise Unsupported('State-dependent channel weights')
    x = vector.constant
    if isinstance(matrix, Vector):
        if matrix.reads: raise Unsupported('Channel scaling a state read needs a left transform')
        return Vector(mul(matrix.constant, x))
    if not isinstance(matrix, Matrix): raise Unsupported('Singleton channel-scale view')
    return Matrix(mul(matrix.diagonal, x), tuple((a, mul(b, x)) for a, b in matrix.ranks),
                  tuple((v, mul(k, x)) for v, k in matrix.offsets))


def transpose(matrix):
    if stateful(matrix): raise Unsupported('State transpose needs a different backend')
    return Matrix(offsets=tuple((k, v) for v, k in matrix.offsets))


def matmul(left, right, ls, rs):
    if ls == ('d', 'd') and rs in (('d',), ('d', 1)):
        return read(left, right)
    if ls == ('d', 1) and rs == (1, 'd'):
        return outer(left, right)
    if ls == ('d',) and rs == ('d',) or ls == (1, 'd') and rs == ('d', 1):
        if left.reads or right.reads: raise Unsupported('State-dependent dot product')
        return dot(left.constant, right.constant)
    if ls == ('d', 'd') and rs == ('d', 'd'):
        if stateful(right): raise Unsupported('Left multiplication of state')
        values = [outer(read(left, Vector(v)), Vector(k)) for v, k in right.offsets]
        return plus(values) if values else Matrix()
    if ls in (('d',), (1, 'd')) and rs == ('d', 'd'):
        if left.reads or stateful(right): raise Unsupported('Left read of state')
        return Vector(add(*(mul(k, dot(left.constant, v)) for v, k in right.offsets)))
    raise Unsupported('Unrecognized matrix-product view')


def has_forcing(graph):
    nonzero = {p: p != 'Sprev' for p in PORTS}
    for node in graph['nodes']:
        flags = [nonzero[a['source']] for a in node['inputs']]
        nonzero[node['name']] = any(flags) if node['operation'] == 'SignedSum' else all(flags)
    return any(nonzero[a['source']] for a in graph['output'])


def normalize(graph):
    graph = canonical(graph)
    shapes, _, _ = analyze(graph)
    values = {p: Vector(('input', p)) for p in ('alpha', 'q', 'k', 'v')}
    values.update(Sprev=Matrix(ONE), beta=('input', 'beta'))

    def get(arg):
        value = values[arg['source']]
        shape = viewed(shapes[arg['source']], arg.get('view', 'none'))
        if arg.get('view') == 'transpose':
            if isinstance(value, Matrix): value = transpose(value)
        return value, shape

    for node in graph['nodes']:
        args = [get(a) for a in node['inputs']]
        op = node['operation']
        if op == 'SignedSum': value = plus([scale(v, ('number', a['sign'])) for a, (v, _) in zip(node['inputs'], args)])
        elif op == 'ScalarScale':
            if not isinstance(args[0][0], tuple): raise Unsupported('State-dependent scalar')
            value = scale(args[1][0], args[0][0])
        elif op == 'ChannelScale': value = channel(args[0][0], args[1][0])
        else: value = matmul(args[0][0], args[1][0], args[0][1], args[1][1])
        values[node['name']] = value
    result = plus([get(a)[0] for a in graph['output']])
    if not isinstance(result, Matrix): raise Unsupported('Non-matrix state')
    return result


def classify(graph):
    graph = canonical(graph)
    _, dependencies, _ = analyze(graph)
    if not any(dependencies[a['source']] for a in graph['output']):
        return {'backend': 'stateless', 'reason': 'No previous-state dependency'}
    if not has_forcing(graph):
        return {'backend': 'zero', 'reason': 'Homogeneous update with zero initial state'}
    try:
        plan = normalize(graph)
        if plan.diagonal == ZERO: raise Unsupported('Zero diagonal: logarithmic DPLR is not applicable')
        if len(plan.ranks) > 1: raise Unsupported('More than one irreducible rank-one transition')
        if plan == normalize(kda()):
            return {'backend': 'kda', 'reason': 'Exact normalized official KDA recurrence', 'plan': plan}
        return {'backend': 'dplr', 'reason': 'Right diagonal plus rank-one transition', 'plan': plan}
    except Unsupported as exc:
        return {'backend': 'recurrent', 'reason': str(exc)}


def evaluate_expression(expression, features, cache):
    if expression in cache: return cache[expression]
    op, *args = expression
    if op == 'input': result = features[args[0]]
    elif op == 'number': result = args[0]
    elif op in ('add', 'mul'):
        result = 0 if op == 'add' else 1
        for a in args:
            item = evaluate_expression(a, features, cache)
            if hasattr(item, 'ndim') and hasattr(result, 'ndim'):
                if item.ndim == result.ndim - 1: item = item.unsqueeze(-1)
                elif result.ndim == item.ndim - 1: result = result.unsqueeze(-1)
            result = result * item if op == 'mul' else result + item
    elif op == 'dot':
        result = (evaluate_expression(args[0], features, cache) * evaluate_expression(args[1], features, cache)).sum(-1)
    else: raise ValueError(op)
    cache[expression] = result
    return result
