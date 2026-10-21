"""Reproducible stratification over a recorded finite random proposal pool."""
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random

from graph import CONSTRAINT_VERSION, PORTS, LIMITS, InactiveInputError, analyze, canonical, edge, graph_id, product, stratum, viewed


def options(shapes):
    values = []
    for source, shape in shapes.items():
        views = ['none'] + (['row', 'column'] if len(shape) == 1 else ['transpose'] if len(shape) == 2 and shape != (1, 1) else [])
        values.extend((edge(source, v), viewed(shape, v)) for v in views)
    return values


def propose(rng, budget):
    operations = [op for op, count in zip(LIMITS, budget) for _ in range(count)]
    rng.shuffle(operations)
    graph = {'nodes': [], 'output': [edge('Sprev')]}
    shapes, dependencies = dict(PORTS), {k: k == 'Sprev' for k in PORTS}
    for i, op in enumerate(operations):
        available = options(shapes)
        if op == 'SignedSum':
            groups = defaultdict(list)
            for a, shape in available: groups[shape].append(a)
            group = rng.choice(list(groups.values()))
            by_source = defaultdict(list)
            for a in group: by_source[a['source']].append(a)
            sources = rng.sample(list(by_source), rng.randint(1, len(by_source)))
            args = [{**rng.choice(by_source[s]), 'sign': rng.choice((-1, 1))} for s in sources]
            shape = viewed(shapes[args[0]['source']], args[0]['view'])
        else:
            valid = []
            for (a, ashape), (b, bshape) in itertools.product(available, repeat=2):
                if dependencies[a['source']] and dependencies[b['source']]: continue
                try:
                    if op == 'MatMul': shape = product(ashape, bshape)
                    elif op == 'ScalarScale':
                        if ashape: continue
                        shape = bshape
                    else:
                        if len(ashape) != 2 or bshape != (ashape[-1],): continue
                        shape = ashape
                except ValueError: continue
                valid.append(([a, b], shape))
            if not valid: return None
            args, shape = rng.choice(valid)
        name = 'n' + str(i)
        graph['nodes'].append({'name': name, 'operation': op, 'inputs': args})
        shapes[name], dependencies[name] = shape, any(dependencies[a['source']] for a in args)
    outputs = defaultdict(list)
    for a, shape in options(shapes):
        if shape == ('d', 'd'): outputs[a['source']].append(a)
    sources = rng.sample(list(outputs), rng.randint(1, len(outputs)))
    graph['output'] = [rng.choice(outputs[s]) for s in sources]
    if not any(dependencies[a['source']] for a in graph['output']): return None
    try: return canonical(graph)
    except InactiveInputError: return None


def sample(count=1000, proposals=50000, seed=2026102001):
    rng = random.Random(seed)
    budgets = list(itertools.product(*(range(limit + 1) for limit in LIMITS.values())))
    pools = defaultdict(dict)
    valid = 0
    for index in range(proposals):
        graph = propose(rng, budgets[index % len(budgets)])
        if graph is None: continue
        valid += 1
        pools[stratum(graph)][graph_id(graph, canonicalized=True)] = graph
    if sum(map(len, pools.values())) < count: raise RuntimeError('Proposal pool too small')
    buckets = {s: list(sorted(pool)) for s, pool in pools.items()}
    for values in buckets.values(): rng.shuffle(values)
    selected, counts = [], defaultdict(int)
    while len(selected) < count:
        available = [s for s, values in buckets.items() if values]
        smallest = min(counts[s] for s in available)
        s = rng.choice(sorted(s for s in available if counts[s] == smallest))
        digest = buckets[s].pop()
        selected.append({'id': digest, 'stratum': s, 'graph': pools[s][digest]})
        counts[s] += 1
    report = {'constraint_version': CONSTRAINT_VERSION, 'previous_state_required': True,
              'current_input_required': True, 'zero_state_write_required': True,
              'sampling_seed': seed, 'proposals': proposals, 'valid_proposals': valid,
              'unique_pool': sum(map(len, pools.values())), 'count': count,
              'stratum_order': list(LIMITS), 'selected_counts': dict(sorted(counts.items())),
              'pool_counts': {s: len(p) for s, p in sorted(pools.items())},
              'sampling': 'Balanced water filling across operation-count strata; uniform without replacement within each recorded proposal pool. Not uniform over every valid DAG.'}
    return selected, report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--proposals', type=int, default=50000)
    args = parser.parse_args()
    selected, report = sample(proposals=args.proposals)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'candidates.jsonl'
    assert not path.exists(), 'Never overwrite a sampled population'
    with path.open('w') as stream:
        for value in selected: stream.write(json.dumps(value, sort_keys=True) + '\n')
    (args.output / 'sampling.json').write_text(json.dumps(report, sort_keys=True) + '\n')
    print(json.dumps(report))
