"""Dependency adapter for the unmodified, pinned Zoology MQAR generator."""
import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: dict


@lru_cache(maxsize=1)
def generator():
    source = Path(__file__).parent / 'vendor' / 'zoology_multiquery_ar.py'
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'multiquery_ar')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'np': np, 'torch': torch, 'DataSegment': DataSegment}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['multiquery_ar']


def batch(config, index, validation=False):
    base = config['validation_seed'] if validation else config['data_seed']
    seed = (base + index * 104729 + config['sequence_length'] * 1009) % (2 ** 32)
    torch.manual_seed(seed)
    data = generator()(
        vocab_size=config['vocab_size'], num_examples=config['batch_size'],
        input_seq_len=config['sequence_length'], seed=seed,
        power_a=config['power_a'], num_kv_pairs=config['num_kv_pairs'],
        num_passes=config['num_passes'], random_non_queries=config['random_non_queries'],
        include_slices=True)
    return {'input_ids': data.inputs, 'labels': data.labels, 'index': index}


class Batches(IterableDataset):
    def __init__(self, config, start, stop):
        self.config, self.start, self.stop = config, start, stop

    def __iter__(self):
        worker = get_worker_info()
        offset, stride = (worker.id, worker.num_workers) if worker else (0, 1)
        torch.set_num_threads(1)
        for index in range(self.start + offset, self.stop, stride):
            yield batch(self.config, index)
