"""Train one DSL candidate on MQAR with the protocol used for all reported runs.

Example:
    python src/run_candidate.py --pool data/pool500/candidates.jsonl --index 274 --seed-index 0
    python src/run_candidate.py --candidates results/mutation_loop/g000_candidates.jsonl --name g000m003 --seed-index 0

Seed index i uses model seed 239+i and data seed 20261020+i (i=0..2 for the
3-seed screen, i=3..9 for the 7 additional top-10 seeds).
Outputs go to $KIMI_DSL_DURABLE/<namespace>/<run_id>/ (default ./runs).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DURABLE, atomic_json, base_config, sha256
from graph import canonical, graph_id
import train
from model import build_model


def load_candidate(args):
    path = args.pool or args.candidates
    rows = [json.loads(line) for line in open(path)]
    if args.index is not None:
        return 'c%04d' % args.index, rows[args.index]
    row = next(r for r in rows if r.get('name') == args.name or r['id'].startswith(args.name))
    return row.get('name', row['id'][:12]), row


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool')
    source.add_argument('--candidates')
    parser.add_argument('--index', type=int)
    parser.add_argument('--name')
    parser.add_argument('--seed-index', type=int, default=0)
    parser.add_argument('--namespace', default='runs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    label, row = load_candidate(args)
    graph = canonical(row['graph'])
    candidate_id = graph_id(graph)
    if candidate_id != row['id']:
        raise RuntimeError('Candidate graph does not match its recorded id')
    graphs = DURABLE / 'graphs'
    graph_path = graphs / (candidate_id + '.json')
    if not graph_path.exists():
        atomic_json(graph_path, graph)
    config = base_config(2048, 1e-4)
    config.update({
        'architecture': 'memory_dsl',
        'model_seed': 239 + args.seed_index, 'data_seed': 20261020 + args.seed_index,
        'generator_sha256': sha256(Path(__file__).resolve().parent / 'vendor' / 'zoology_multiquery_ar.py'),
        'candidate_id': candidate_id, 'candidate_graph_path': str(graph_path),
        'candidate_graph_sha256': sha256(graph_path),
        'memory_backend': 'chunk_admitted_v1', 'expected_memory_backend': row.get('backend', 'dplr'),
    })
    run_id = label + '_' + candidate_id[:12] + '_s' + str(config['model_seed'])
    config_path = DURABLE / 'configs' / (run_id + '.json')
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Existing config differs; use a new namespace')
    atomic_json(config_path, config)
    train.build_model = build_model
    train.train(argparse.Namespace(config=str(config_path), run_id=run_id, namespace=args.namespace,
                                   resume=args.resume, stop_after=args.stop_after))


if __name__ == '__main__':
    main()
