"""Exact polynomial recurrence identity using the existing DSL tensor algebra.

This distributes sums/products and canonicalizes contracted indices. It does
not identify floating-point evaluation orders or normalized-feature identities.
"""
from loop import BASE
from common import identity as digest
from graph import analyze, canonical
from influence import axes, identity, monomial, multiply, plus, rename


def signature(graph):
    graph = canonical(graph); shapes, _, _ = analyze(graph)
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
        if node['operation'] == 'SignedSum':
            value = plus([(a['sign'], v) for a, (v, _) in zip(node['inputs'], args)])
        else:
            (a, ashape), (b, bshape) = args
            if node['operation'] == 'ScalarScale': amap, bmap = {}, identity(bshape)
            elif node['operation'] == 'ChannelScale':
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
    return digest(sorted(output.items()))


def inspect_packet(packet, manifests):
    from common import read_json
    previous = read_json(BASE / 'search_chunkable500_v1/manifest.json')
    ids = {r['candidate_id'] for r in previous['runs']}
    fingerprints = {signature(read_json(BASE / 'search_chunkable500_v1/graphs' / (cid + '.json'))): cid for cid in ids}
    for manifest in manifests:
        for candidate in manifest['candidates']:
            fingerprints[signature(candidate['graph'])] = candidate['id']
    accepted, rejected = [], []
    for candidate in packet['proposals']:
        fingerprint = signature(candidate['graph'])
        if fingerprint in fingerprints:
            rejected.append({'mutation': candidate['mutation'], 'equivalent_to': fingerprints[fingerprint]})
        else:
            accepted.append(candidate)
            fingerprints[fingerprint] = candidate['mutation']
    return accepted, rejected
