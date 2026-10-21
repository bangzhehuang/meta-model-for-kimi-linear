"""Random proposal/rejection lane conditional on the existing DPLR backend.

The historical module name describes the backend's required structure; no
hand-selected graph families or family quotas enter the proposal distribution.
"""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random

from graph import CONSTRAINT_VERSION, LIMITS, analyze, canonical, graph_id, kda, stratum
from influence import axes, identity, monomial, multiply, plus, rename
from lower import classify
from sample import propose


def polynomial(graph):
    """Exact free-index tensor polynomial, including output and contractions.

    This reuses influence.py's dimension-generic monomial normalization. Its
    public influence summary intentionally discards coefficients and terms;
    retain them here to detect equivalent output functions across DAGs.
    """
    shapes, _, _ = analyze(graph)
    ports = ('Sprev', 'alpha', 'beta', 'q', 'k', 'v')
    values = {p: {((p, tuple(i for i in axes(shapes[p]) if i is not None)),): 1}
              for p in ports}

    def get(arg):
        value, shape = values[arg['source']], shapes[arg['source']]
        view = arg.get('view', 'none')
        if view == 'transpose':
            changed = shape[::-1]
            mapping = {old: new for old, new in zip(axes(shape), axes(changed)[::-1])
                       if old is not None}
            value = plus([(1, {monomial(rename(t, mapping, 't')): c
                               for t, c in value.items()})])
            shape = changed
        elif view == 'row':
            shape = (1, *shape)
        elif view == 'column':
            shape = (*shape, 1)
        return value, shape

    for node in graph['nodes']:
        args = [get(a) for a in node['inputs']]
        operation = node['operation']
        if operation == 'SignedSum':
            value = plus([(a['sign'], v) for a, (v, _) in zip(node['inputs'], args)])
        else:
            (left, ls), (right, rs) = args
            if operation == 'ScalarScale':
                lm, rm = {}, identity(rs)
            elif operation == 'ChannelScale':
                lm = identity(ls)
                rm = {i: axes(ls)[-1] for i in axes(rs) if i is not None}
            else:
                la, ra = axes(ls), axes(rs)
                output_axes = iter(axes(shapes[node['name']]))
                lm, rm = {}, {}
                if len(ls) == 2:
                    out = next(output_axes)
                    if la[0] is not None:
                        lm[la[0]] = out
                if len(rs) == 2:
                    out = next(output_axes)
                    if ra[-1] is not None:
                        rm[ra[-1]] = out
                if la[-1] is not None:
                    lm[la[-1]] = 'contract'
                contraction = ra[0] if len(rs) == 1 else ra[-2]
                if contraction is not None:
                    rm[contraction] = 'contract'
            value = multiply(left, right, lm, rm)
        values[node['name']] = value
    result = plus([(1, get(a)[0]) for a in graph['output']])
    return tuple(sorted(result.items()))


def semantic_id(graph):
    return hashlib.sha256(json.dumps(polynomial(graph), separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    directory = Path(__file__).resolve().parent
    paths = [directory / name for name in ('structured.py', 'sample.py', 'graph.py',
                                          'influence.py', 'lower.py')]
    paths.append(directory.parent / 'DSL_SPEC.md')
    return {str(p.relative_to(directory.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def sample(count=500, proposals=500000, seed=2026102002):
    """Balance strata after exact semantic deduplication of a fixed pool."""
    rng = random.Random(seed)
    budgets = list(itertools.product(*(range(limit + 1) for limit in LIMITS.values())))
    diagnostics = Counter()
    reasons = Counter()
    unique = {}
    classes = defaultdict(list)
    reference = canonical(kda())
    reference_semantic = semantic_id(reference)
    for index in range(proposals):
        budget = budgets[index % len(budgets)]
        graph = propose(rng, budget)
        if graph is None:
            diagnostics['invalid_or_inactive_proposals'] += 1
            continue
        diagnostics['valid_proposals'] += 1
        if any(n['operation'] == 'SignedSum' and len(n['inputs']) == 1
               and n['inputs'][0]['sign'] == 1 for n in graph['nodes']):
            diagnostics['nonfunctional_singleton_sum_rejections'] += 1
            continue
        classification = classify(graph)
        diagnostics['backend_' + classification['backend']] += 1
        if classification['backend'] != 'dplr':
            reasons[classification['reason']] += 1
            continue
        digest = graph_id(graph, canonicalized=True)
        if digest in unique:
            diagnostics['duplicate_canonical_dplr_occurrences'] += 1
            continue
        semantic = semantic_id(graph)
        record = {'id': digest, 'semantic_id': semantic, 'stratum': stratum(graph),
                  'graph': graph, 'backend': 'dplr',
                  'provenance': {'proposal_index': index, 'proposal_budget': list(budget)}}
        unique[digest] = record
        classes[semantic].append(digest)
    pools = defaultdict(dict)
    equivalences = []
    for semantic, ids in sorted(classes.items()):
        representative = min(ids)
        if len(ids) > 1:
            equivalences.append({'semantic_id': semantic, 'representative_id': representative,
                                 'canonical_graph_ids': sorted(ids)})
        if semantic == reference_semantic:
            continue
        record = unique[representative]
        record['provenance']['semantic_class_canonical_count'] = len(ids)
        pools[record['stratum']][representative] = record
    available_count = sum(map(len, pools.values()))
    if available_count < count:
        raise RuntimeError(f'Only {available_count} distinct eligible semantic classes '
                           f'from {proposals} proposals; requested {count}. No padding performed.')

    # A separate RNG makes selection independent of proposal RNG consumption.
    selection_rng = random.Random(seed + 1)
    buckets = {s: sorted(pool) for s, pool in sorted(pools.items())}
    for ids in buckets.values():
        selection_rng.shuffle(ids)
    selected, counts = [], Counter()
    while len(selected) < count:
        available = [s for s, ids in buckets.items() if ids]
        smallest = min(counts[s] for s in available)
        chosen = selection_rng.choice(sorted(s for s in available if counts[s] == smallest))
        digest = buckets[chosen].pop()
        selected.append(pools[chosen][digest])
        counts[chosen] += 1
    report = {'constraint_version': CONSTRAINT_VERSION, 'count': len(selected),
              'sampling_seed': seed, 'selection_seed': seed + 1, 'proposals': proposals,
              'proposal_budget_order': list(LIMITS), 'proposal_budget_combinations': len(budgets),
              'diagnostics': dict(sorted(diagnostics.items())),
              'backend_rejections': dict(sorted(reasons.items())),
              'unique_canonical_dplr': len(unique), 'unique_semantic_dplr': len(classes),
              'eligible_nonreference_semantic_pool': available_count,
              'equivalent_canonical_graphs_removed': len(unique) - len(classes),
              'selected_counts': dict(sorted(counts.items())),
              'pool_counts': {s: len(p) for s, p in sorted(pools.items())},
              'exact_kda_reference': {'id': graph_id(reference, canonicalized=True),
                                      'semantic_id': reference_semantic,
                                      'included_in_500': False, 'separate_reference_count': 1},
              'sampling': ('Unmodified sample.propose over cyclic exhaustive operation budgets; '
                           'automatic v4 validity and DPLR recognition filters; exact tensor '
                           'polynomial equivalence classes; reject singleton positive signed '
                           'sums that only spend budget on a free view; '
                           'representative=min canonical SHA256; '
                           'balanced water filling across operation-count/depth/shared strata, '
                           'uniform without replacement within each recorded stratum pool. '
                           'Conditional finite-pool sampling, not global uniform DAG sampling; '
                           'no hand-selected template families or family quotas.'),
              'semantic_scope': ('Exact integer tensor-contraction polynomial equality treating '
                                 'ports as independent variables and dimension as generic. '
                                 'Feature normalization identities and special-dimension identities '
                                 'are not quotient relations. No known polynomial duplicates remain.'),
              'training_launched': False, 'frozen': False, 'source_sha256': source_hashes()}
    return selected, report, equivalences


def write_population(output, count=500, proposals=500000, seed=2026102002):
    if output.exists():
        raise FileExistsError('Never overwrite an existing population directory')
    selected, report, equivalences = sample(count=count, proposals=proposals, seed=seed)
    output.mkdir(parents=True)
    with (output / 'candidates.jsonl').open('w') as stream:
        for candidate in selected:
            stream.write(json.dumps(candidate, sort_keys=True) + '\n')
    reference = canonical(kda())
    (output / 'kda_reference.json').write_text(json.dumps(
        {'id': graph_id(reference, canonicalized=True), 'semantic_id': semantic_id(reference),
         'graph': reference, 'backend': classify(reference)['backend'],
         'role': 'separate reference; not part of the 500 candidates'}, sort_keys=True) + '\n')
    with (output / 'equivalence_classes.jsonl').open('w') as stream:
        for entry in equivalences:
            stream.write(json.dumps(entry, sort_keys=True) + '\n')
    report['artifacts_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'sampling.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--proposals', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=2026102002)
    args = parser.parse_args()
    report = write_population(args.output, args.count, args.proposals, args.seed)
    print(json.dumps({k: report[k] for k in ('count', 'proposals', 'unique_canonical_dplr',
                                            'unique_semantic_dplr', 'eligible_nonreference_semantic_pool')}))
