"""Keep the pinned KDA model and replace only its per-head memory recurrence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCRATCH, read_json, sha256
from graph import canonical, graph_id
from backend import recurrent
from train import build_model as original_builder


def build_model(config):
    import fla.layers.kda as layer
    from fla.modules.l2norm import l2_norm
    graph = read_json(config['candidate_graph_path'])
    if sha256(config['candidate_graph_path']) != config['candidate_graph_sha256']:
        raise RuntimeError('Candidate graph source identity mismatch')
    if graph_id(graph) != config['candidate_id'] or canonical(graph) != graph:
        raise RuntimeError('Candidate is not canonical/active or has wrong identity')
    if config.get('memory_backend') == 'chunk_admitted_v1':
        from fast import compile_cell
        cell, info = compile_cell(graph, SCRATCH / 'search_kernels', require_fast=True)
        if info['backend'] != config['expected_memory_backend']:
            raise RuntimeError('Admitted memory backend differs from compiled backend')
    else:
        cell = recurrent(graph, SCRATCH / 'search_kernels')

    def memory(q, k, v, g, beta, initial_state=None, output_final_state=False,
               use_qk_l2norm_in_kernel=False, cu_seqlens=None, scale=None, **kwargs):
        if initial_state is not None or output_final_state or cu_seqlens is not None:
            raise ValueError('Only independent complete equal-length sequences are supported')
        if scale is not None and scale != q.shape[-1] ** -.5:
            raise ValueError('Unexpected query scale')
        if use_qk_l2norm_in_kernel:
            q, k = l2_norm(q), l2_norm(k)
        return cell(q, k, v, g.float().exp(), beta), None

    layer.chunk_kda = memory
    layer.fused_recurrent_kda = memory
    return original_builder(config)
