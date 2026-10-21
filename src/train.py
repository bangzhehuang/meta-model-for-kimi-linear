"""Single-GPU official KDA baseline with exact-step deterministic MQAR batches."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import DURABLE, SCRATCH, atomic_json, compact_curve, identity, read_json, sha256, start_token
from mqar import Batches, batch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['cpu'])
    torch.cuda.set_rng_state_all(value['cuda'])


def build_model(config):
    from fla.models.kda import KDAConfig, KDAForCausalLM
    args = KDAConfig(
        hidden_size=config['hidden_size'], intermediate_size=config['intermediate_size'],
        num_hidden_layers=config['layers'], num_heads=config['heads'],
        num_v_heads=config['heads'], head_dim=config['head_dim'],
        expand_v=config['expand_v'], conv_size=config['conv_size'], use_short_conv=True,
        allow_neg_eigval=config['allow_neg_eigval'], attn_mode='chunk', attn=None,
        vocab_size=config['vocab_size'], tie_word_embeddings=config['tie_embeddings'],
        norm_eps=1e-6, fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=False,
        use_cache=False, max_position_embeddings=config['sequence_length'])
    return KDAForCausalLM(args).float().cuda()


def build_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        target = no_decay if parameter.ndim < 2 or getattr(parameter, '_no_weight_decay', False) else decay
        target.append(parameter)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': config['weight_decay']},
         {'params': no_decay, 'weight_decay': 0.0}], lr=config['learning_rate'],
        betas=tuple(config['betas']), eps=config['epsilon'], fused=True)


def objective(model, inputs, labels):
    # Zoology labels are already aligned to the query position; never LM-shift again.
    hidden = model.model(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
    mask = labels.ne(-100)
    logits = model.lm_head(hidden[mask])
    target = labels[mask]
    loss = F.cross_entropy(logits.float(), target)
    correct_masked = logits.argmax(-1).eq(target)
    correct = correct_masked.sum()
    per_sequence = torch.zeros_like(labels, dtype=torch.bool)
    per_sequence[mask] = correct_masked
    exact = (per_sequence | ~mask).all(-1).sum()
    return loss, correct, target.numel(), exact


class Publisher:
    def __init__(self, local, durable):
        self.local, self.durable = local, durable
        self.local.mkdir(parents=True, exist_ok=True)
        self.durable.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.last_checkpoint_step = -1
        self.last_checkpoint_unix = time.time()
        self.receipt = read_json(durable / 'mirror_receipt.json', {})

    def check(self):
        if self.pending is not None and self.pending.done():
            self.pending.result()
            self.pending = None
        if self.last_checkpoint_step >= 0 and time.time() - self.last_checkpoint_unix > 600:
            raise RuntimeError('Full-state durable publication stale for over 600 seconds')

    def write(self, status, curve, checkpoint=None):
        receipt = self.receipt
        if checkpoint is not None:
            tmp = self.local / '.last_checkpoint.pt.tmp'
            torch.save(checkpoint, tmp)
            os.replace(tmp, self.local / 'last_checkpoint.pt')
            # Serialize large shared writes across all hosts; keep training asynchronous.
            with (DURABLE / '.checkpoint_publish.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                import hashlib
                digest = hashlib.sha256()
                temporary = self.durable / ('.last.' + str(os.getpid()) + '.tmp')
                with (self.local / 'last_checkpoint.pt').open('rb') as source, temporary.open('wb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block)
                        digest.update(block)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, self.durable / 'last_checkpoint.pt')
                receipt = {'step': checkpoint['step'], 'sha256': digest.hexdigest(),
                           'bytes': (self.durable / 'last_checkpoint.pt').stat().st_size,
                           'published_unix': time.time(), 'host': socket.gethostname(),
                           'config_sha256': checkpoint['config_sha256']}
                atomic_json(self.durable / 'mirror_receipt.json', receipt)
                self.receipt = receipt
                self.last_checkpoint_step = checkpoint['step']
                self.last_checkpoint_unix = time.time()
        status['mirror'] = receipt
        atomic_json(self.local / 'status.json', status)
        atomic_json(self.durable / 'curve.json', compact_curve(curve))
        atomic_json(self.durable / 'status.json', status)

    def submit(self, status, curve, checkpoint=None, wait=False):
        self.check()
        if self.pending is not None:
            if not wait:
                return False
            self.pending.result()
            self.pending = None
        self.pending = self.pool.submit(self.write, copy.deepcopy(status), copy.deepcopy(curve), checkpoint)
        if wait:
            self.pending.result()
            self.pending = None
        return True

    def close(self):
        try:
            if self.pending is not None:
                self.pending.result()
        finally:
            self.pool.shutdown(wait=True)


@torch.no_grad()
def validate(model, config, batches):
    model.eval()
    correct = tokens = exact = sequences = 0
    total_loss = 0.0
    start = time.perf_counter()
    for value in batches:
        inputs = value['input_ids'].cuda(non_blocking=True)
        labels = value['labels'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, found, count, all_found = objective(model, inputs, labels)
        correct += found.item()
        tokens += count
        total_loss += loss.item() * count
        exact += all_found.item()
        sequences += inputs.shape[0]
    model.train()
    return {'accuracy': correct / tokens, 'loss': total_loss / tokens,
            'exact_accuracy': exact / sequences, 'examples': sequences,
            'target_tokens': tokens, 'seconds': time.perf_counter() - start}


def train(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(args.config)
    if sha256(Path(__file__).parent / 'vendor' / 'zoology_multiquery_ar.py') != config['generator_sha256']:
        raise RuntimeError('Pinned generator source identity mismatch')
    config_hash = identity(config)
    run = args.run_id or Path(args.config).stem
    durable = DURABLE / args.namespace / run
    local = SCRATCH / args.namespace / run
    durable.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    lock = open('/tmp/kimi_synthetic_gpu_' + gpu + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.getpid()
    status = {'run_id': run, 'config': config, 'config_sha256': config_hash,
              'status': 'initializing', 'pid': pid, 'start_token': start_token(pid),
              'host': socket.gethostname(), 'gpu': gpu, 'step': 0,
              'updated_unix': time.time(), 'started_unix': time.time(),
              'compiler': 'official Triton JIT; outer PyTorch eager'}
    atomic_json(durable / 'status.json', status)
    seed = config['model_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(config)
    optimizer = build_optimizer(model, config)
    status['parameters'] = sum(p.numel() for p in model.parameters())
    status['model_parameter_dtype'] = str(next(model.parameters()).dtype)
    status['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    initial_digest = __import__('hashlib').sha256()
    for name, value in model.state_dict().items():
        initial_digest.update(name.encode())
        initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    status['initial_model_sha256'] = initial_digest.hexdigest()
    step, history, curve, validations = 0, [], [], []
    best_accuracy, best_step, best_model = -1.0, 0, None
    best_window, best_window_step = -1.0, 0
    thresholds = {}
    window = deque(maxlen=config['curve_window'])
    checkpoint_path = durable / 'last_checkpoint.pt'
    if args.resume:
        receipt = read_json(durable / 'mirror_receipt.json')
        if not receipt or sha256(checkpoint_path) != receipt['sha256']:
            raise RuntimeError('Resume checkpoint receipt/SHA256 mismatch')
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['config_sha256'] != config_hash or saved['run_id'] != run:
            raise RuntimeError('Resume configuration/run identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step, history, curve, validations = saved['step'], saved['history'], saved['curve'], saved['validations']
        best_accuracy, best_step, best_model = saved['best_accuracy'], saved['best_step'], saved['best_model']
        best_window, best_window_step = saved['best_window'], saved['best_window_step']
        thresholds = saved.get('thresholds', {})
        window.extend(saved['window'])
        restore_rng(saved['rng'])
        status['resumed_from'] = {'step': step, 'sha256': receipt['sha256'],
                                  'restored': ['model', 'optimizer', 'step', 'history', 'best_model',
                                               'metrics', 'window', 'validation', 'python_rng', 'numpy_rng',
                                               'cpu_rng', 'cuda_rng', 'deterministic_step_data']}
    stop_requested = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop_requested.__setitem__(0, True))
    publisher = Publisher(local, durable)
    limit = min(config['max_steps'], args.stop_after or config['max_steps'])
    dataset = Batches(config, step, limit)
    loader = DataLoader(dataset, batch_size=None, num_workers=config['data_workers'], pin_memory=True,
                        persistent_workers=config['data_workers'] > 0,
                        generator=torch.Generator().manual_seed(config['data_seed']),
                        **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2}
                           if config['data_workers'] else {}))
    validation_batches = None
    total_seconds, recent = 0.0, deque(maxlen=100)

    def capture():
        return {'format': 'kimi_mqar_full_state_v1', 'run_id': run, 'config': config,
                'config_sha256': config_hash, 'step': step,
                'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
                'history': history.copy(), 'curve': curve.copy(), 'validations': validations.copy(),
                'best_accuracy': best_accuracy, 'best_step': best_step,
                'best_model': cpu_tree(best_model), 'best_window': best_window,
                'best_model_parameter_step': max(0, best_step - 1), 'thresholds': thresholds.copy(),
                'best_window_step': best_window_step, 'window': list(window), 'rng': rng_state(),
                'scheduler': None, 'gradient_scaler': None,
                'data_identity': config['generator_sha256'], 'published_after_update': True}

    model.train()
    log = (local / 'history.jsonl').open('a', buffering=1)
    try:
        waiting = time.perf_counter()
        for value in loader:
            started = time.perf_counter()
            data_seconds = started - waiting
            if value['index'] != step:
                raise RuntimeError('Deterministic loader returned the wrong update index')
            publisher.check()
            inputs = value['input_ids'].cuda(non_blocking=True)
            labels = value['labels'].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, found, count, all_found = objective(model, inputs, labels)
            accuracy = found.item() / count
            if accuracy > best_accuracy:
                best_accuracy, best_step = accuracy, step + 1
                best_model = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            loss_value, norm_value = loss.item(), norm.item()
            if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                raise FloatingPointError('Nonfinite loss or gradient; stopping without changing protocol')
            optimizer.step()
            end.record()
            end.synchronize()
            compute_seconds = begin.elapsed_time(end) / 1000
            step += 1
            seconds = time.perf_counter() - started + data_seconds
            total_seconds += seconds
            if step > 5:
                recent.append(seconds)
            window.append((found.item(), count, loss_value))
            window_accuracy = sum(x[0] for x in window) / sum(x[1] for x in window)
            if len(window) == config['curve_window'] and window_accuracy > best_window:
                best_window, best_window_step = window_accuracy, step
            if len(window) == config['curve_window']:
                for threshold in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
                    if window_accuracy >= threshold:
                        thresholds.setdefault(str(threshold), step)
            row = {'step': step, 'accuracy': accuracy, 'window_accuracy': window_accuracy,
                   'loss': loss_value, 'exact_accuracy': all_found.item() / config['batch_size'],
                   'gradient_norm': norm_value, 'seconds': seconds, 'compute_seconds': compute_seconds,
                   'data_seconds': data_seconds, 'updated_unix': time.time()}
            history.append(row)
            log.write(json.dumps(row, separators=(',', ':')) + '\n')
            if step == 1 or step % config['status_every'] == 0 or step == limit:
                curve.append({key: row[key] for key in ('step', 'accuracy', 'window_accuracy', 'loss')})
                speed = sum(recent) / len(recent) if recent else None
                status.update({'status': 'running', 'step': step, 'latest': row,
                               'peak_train_accuracy': best_accuracy, 'peak_train_step': best_step,
                               'peak_window_accuracy': max(0.0, best_window),
                               'peak_window_step': best_window_step,
                               'threshold_steps': thresholds.copy(),
                               'seconds_per_update': speed,
                               'tokens_per_second': config['batch_size'] * config['sequence_length'] / speed if speed else None,
                               'remaining_seconds': (config['max_steps'] - step) * speed if speed else None,
                               'peak_memory_mib': torch.cuda.max_memory_allocated() / 2**20,
                               'updated_unix': time.time(), 'validations': validations})
                checkpoint_due = (step == 1 or step - publisher.last_checkpoint_step >= config['checkpoint_every']
                                  or time.time() - publisher.last_checkpoint_unix >= 180)
                if step == 1:
                    publisher.submit(status, curve, capture(), wait=True)
                elif publisher.pending is None:
                    publisher.submit(status, curve, capture() if checkpoint_due else None)
            if step % config['validation_every'] == 0 or step == config['max_steps']:
                if validation_batches is None:
                    # Preserve training RNG while constructing the fixed auxiliary set.
                    state = rng_state()
                    validation_batches = [batch(config, index, validation=True)
                                          for index in range(config['validation_examples'] // config['batch_size'])]
                    restore_rng(state)
                status.update({'status': 'validating', 'step': step, 'updated_unix': time.time()})
                publisher.submit(status, curve, wait=True)
                result = validate(model, config, validation_batches)
                result['step'] = step
                validations.append(result)
                status.update({'status': 'running', 'validation': result, 'validations': validations,
                               'updated_unix': time.time()})
                publisher.submit(status, curve, capture(), wait=False)
            if step % 100 == 0 or step == 1:
                print(json.dumps({'run': run, 'step': step, 'loss': loss_value,
                                  'accuracy': accuracy, 'seconds_per_update': status.get('seconds_per_update')}), flush=True)
            if stop_requested[0]:
                break
            waiting = time.perf_counter()
        status.update({'status': 'completed' if step >= config['max_steps'] else 'paused',
                       'step': step, 'latest': history[-1] if history else None,
                       'updated_unix': time.time(), 'finished_unix': time.time()})
        publisher.submit(status, curve, capture(), wait=True)
        if status['status'] == 'completed':
            atomic_json(durable / 'result.json', {'run_id': run, 'config_sha256': config_hash,
                        'parameters': status['parameters'], 'final': history[-1],
                        'peak_train_accuracy': best_accuracy, 'peak_train_step': best_step,
                        'peak_window_accuracy': best_window, 'peak_window_step': best_window_step,
                        'validation': validations[-1] if validations else None,
                        'checkpoint': publisher.receipt, 'completed_unix': time.time()})
        print(json.dumps({'status': status['status'], 'step': step, 'checkpoint': publisher.receipt}), flush=True)
    except BaseException:
        failure = {'status': 'failed', 'step': step, 'updated_unix': time.time(),
                   'pid': pid, 'start_token': start_token(pid), 'traceback': traceback.format_exc()[-6000:]}
        status.update(failure)
        atomic_json(durable / ('failure_' + str(time.time_ns()) + '.json'), failure)
        publisher.submit(status, curve, wait=True)
        raise
    finally:
        log.close()
        publisher.close()
        del loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--namespace', default='runs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    train(parser.parse_args())
