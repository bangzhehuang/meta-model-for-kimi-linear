"""Exact polynomial influence checks for the DSL's algebraic operations.

Monomials are scalar tensor contractions with named free indices. Renaming
summation indices does not change a monomial. No random numerical probe or
learned feature values are used to decide whether a contribution is zero.
"""
from collections import defaultdict
from functools import lru_cache
import itertools


@lru_cache(maxsize=8192)
def monomial(factors):
    bound = sorted({i for _, indices in factors for i in indices if not i.startswith('f')})
    if not bound: return tuple(sorted(factors))
    # Refine contraction-index classes before enumerating their permutations.
    colors = dict.fromkeys(bound, 0)
    for _ in range(len(bound) + 1):
        signatures = {}
        for index in bound:
            signatures[index] = tuple(sorted(
                (name, axis, tuple(('free', i) if i.startswith('f') else
                                   ('self', '') if i == index else ('bound', str(colors[i])) for i in indices))
                for name, indices in factors for axis, i in enumerate(indices) if i == index))
        palette = {s: n for n, s in enumerate(sorted(set(signatures.values())))}
        refined = {i: palette[signatures[i]] for i in bound}
        if refined == colors: break
        colors = refined
    groups = [[i for i in bound if colors[i] == c] for c in sorted(set(colors.values()))]
    best = None
    for parts in itertools.product(*(itertools.permutations(g) for g in groups)):
        rename = {i: 'd' + str(n) for n, i in enumerate(itertools.chain.from_iterable(parts))}
        candidate = tuple(sorted((name, tuple(rename.get(i, i) for i in indices)) for name, indices in factors))
        if best is None or candidate < best: best = candidate
    return best


def rename(term, free, prefix):
    return tuple((name, tuple(free[i] if i.startswith('f') else prefix + i for i in indices))
                 for name, indices in term)


def plus(polynomials):
    result = defaultdict(int)
    for sign, polynomial in polynomials:
        for term, coefficient in polynomial.items(): result[term] += sign * coefficient
    return {t: c for t, c in result.items() if c}


def multiply(left, right, left_free, right_free):
    result = defaultdict(int)
    for a, ca in left.items():
        aa = rename(a, left_free, 'a')
        for b, cb in right.items():
            term = monomial(aa + rename(b, right_free, 'b'))
            result[term] += ca * cb
    return {t: c for t, c in result.items() if c}


def axes(shape):
    next_index = 0
    result = []
    for size in shape:
        result.append('f' + str(next_index) if size == 'd' else None)
        next_index += size == 'd'
    return result


def identity(shape):
    return {i: i for i in axes(shape) if i is not None}


def inspect_influence(graph, shapes):
    values = {name: {((name, tuple(i for i in axes(shape) if i is not None)),): 1}
              for name, shape in shapes.items() if name in ('Sprev', 'alpha', 'beta', 'q', 'k', 'v')}

    def get(arg):
        value, shape = values[arg['source']], shapes[arg['source']]
        view = arg.get('view', 'none')
        if view == 'transpose':
            changed = shape[::-1]
            mapping = {old: new for old, new in zip(axes(shape), axes(changed)[::-1]) if old is not None}
            value = plus([(1, {monomial(rename(t, mapping, 't')): c for t, c in value.items()})])
            shape = changed
        elif view == 'row': shape = (1, *shape)
        elif view == 'column': shape = (*shape, 1)
        return value, shape

    for node in graph['nodes']:
        args = [get(a) for a in node['inputs']]
        op = node['operation']
        if op == 'SignedSum':
            value = plus([(a['sign'], v) for a, (v, _) in zip(node['inputs'], args)])
        else:
            (a, ashape), (b, bshape) = args
            if op == 'ScalarScale':
                amap, bmap = {}, identity(bshape)
            elif op == 'ChannelScale':
                amap = identity(ashape)
                bmap = {i: axes(ashape)[-1] for i in axes(bshape) if i is not None}
            else:
                aa, bb, oo = axes(ashape), axes(bshape), iter(axes(shapes[node['name']]))
                amap, bmap = {}, {}
                if len(ashape) == 2:
                    out = next(oo)
                    if aa[0] is not None: amap[aa[0]] = out
                if len(bshape) == 2:
                    out = next(oo)
                    if bb[-1] is not None: bmap[bb[-1]] = out
                if aa[-1] is not None: amap[aa[-1]] = 'contract'
                contraction = bb[0] if len(bshape) == 1 else bb[-2]
                if contraction is not None: bmap[contraction] = 'contract'
            value = multiply(a, b, amap, bmap)
        values[node['name']] = value
    output = plus([(1, get(a)[0]) for a in graph['output']])
    return {
        'previous_state': any(any(name == 'Sprev' for name, _ in term) for term in output),
        'current_input': any(any(name != 'Sprev' for name, _ in term) for term in output),
        'zero_state_write': any(all(name != 'Sprev' for name, _ in term) for term in output),
        'terms': len(output),
    }
