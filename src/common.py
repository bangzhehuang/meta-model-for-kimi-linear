import hashlib
import json
import os
from pathlib import Path
import time

FROZEN_FLA = '7ab283715a50ecceec3a4982f275a5a7039d01c6'
# DURABLE holds checkpoints/results that must survive a worker restart; SCRATCH
# holds generated Triton kernels and local logs. Override with environment variables.
DURABLE = Path(os.environ.get('KIMI_DSL_DURABLE', 'runs'))
SCRATCH = Path(os.environ.get('KIMI_DSL_SCRATCH', 'scratch'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as target:
        json.dump(value, target, allow_nan=False, separators=(',', ':'))
        target.write('\n')
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        with open(path) as source:
            return json.load(source)
    except FileNotFoundError:
        return default


def start_token(pid):
    try:
        return Path('/proc/' + str(pid) + '/stat').read_text().split(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def base_config(length, lr):
    return {
        'schema': 1, 'task': 'mqar', 'architecture': 'kda', 'fla_commit': FROZEN_FLA,
        'sequence_length': length, 'num_kv_pairs': length // 8, 'vocab_size': 8192,
        'power_a': 0.01, 'num_passes': 1, 'random_non_queries': True,
        'batch_size': 32, 'gradient_accumulation': 1, 'max_steps': 20000,
        'learning_rate': lr, 'optimizer': 'AdamW', 'betas': [0.9, 0.999],
        'epsilon': 1e-8, 'weight_decay': 0.01, 'gradient_clip': 1.0,
        'scheduler': 'constant', 'model_seed': 239, 'data_seed': 20261020,
        'validation_seed': 9120261020, 'validation_examples': 1024,
        'validation_every': 500, 'checkpoint_every': 500,
        'status_every': 20, 'curve_window': 100, 'data_workers': 2,
        'hidden_size': 256, 'intermediate_size': 1024, 'layers': 2,
        'heads': 2, 'head_dim': 128, 'conv_size': 4, 'expand_v': 1.0,
        'tie_embeddings': False, 'allow_neg_eigval': False,
        'autocast': 'bfloat16', 'parameter_dtype': 'float32',
        'optimizer_dtype': 'float32', 'tf32': False, 'torch_compile': False,
        'metric': 'masked_query_token_accuracy', 'selection': 'peak_training_accuracy',
        'generator_commit': '1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb',
        'paper_alignment': 'reported_dimensions_and_budgets_matched; unpublished_details_assumed',
    }


def run_id(length, lr):
    return 'kda_mqar_l' + str(length) + '_lr' + format(lr, '.0e').replace('-', 'm').replace('+', 'p')


def compact_curve(rows, maximum=501):
    if len(rows) <= maximum:
        return rows
    indices = {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
    return [rows[i] for i in sorted(indices)]
