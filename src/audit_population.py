"""Bounded draft-population audit; this never dispatches training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from graph import CONSTRAINT_VERSION, InactiveInputError, analyze, canonical, graph_id
from influence import inspect_influence
from lower import classify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts, reasons, ids = Counter(), Counter(), set()
    total = 0
    with args.population.open() as stream:
        for total, line in enumerate(stream, 1):
            if total > 1000: raise ValueError('Expected a bounded 1000-candidate draft')
            row = json.loads(line)
            graph = row['graph']
            shapes, _, _ = analyze(graph)
            assert graph == canonical(graph)
            assert row['id'] == graph_id(graph, canonicalized=True)
            assert row['id'] not in ids
            ids.add(row['id'])
            influence = inspect_influence(graph, shapes)
            assert all(influence[k] for k in ('previous_state', 'current_input', 'zero_state_write'))
            info = classify(graph)
            counts[info['backend']] += 1
            reasons[info['reason']] += 1
    assert total == 1000
    report = {'constraint_version': CONSTRAINT_VERSION, 'status': 'passed',
              'population_status': 'draft_only_not_frozen_not_dispatched', 'count': total,
              'unique': len(ids), 'all_requirements_passed': total,
              'backend_classification_only_not_cuda_validation': dict(counts),
              'backend_reasons': dict(reasons), 'population_sha256': digest(args.population),
              'sampling_sha256': digest(args.population.with_name('sampling.json')),
              'population_path': str(args.population),
              'source_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                                ('graph.py', 'influence.py', 'sample.py', 'lower.py', 'audit_population.py')}}
    if args.previous:
        excluded, retained = Counter(), 0
        with args.previous.open() as stream:
            for count, line in enumerate(stream, 1):
                if count > 1000: raise ValueError('Previous draft exceeded audit limit')
                try:
                    analyze(json.loads(line)['graph'])
                    retained += 1
                except InactiveInputError as exc: excluded[str(exc)] += 1
        report['previous_draft'] = {'path': str(args.previous), 'retained': retained, 'excluded': dict(excluded)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise ValueError('Never overwrite an audit')
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__': main()
