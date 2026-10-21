"""Select a proven structured backend without changing the legal DSL space."""
from graph import canonical
from lower import ZERO, classify, evaluate_expression

PROTOCOL = 'dsl_v4_fla7ab2837_bf16_boundary_fp32_features_zero_grad_v3'


def compile_cell(graph, directory, require_fast=False):
    import torch
    from backend import recurrent
    graph = canonical(graph)
    info = classify(graph)
    info = {**info, 'protocol': PROTOCOL, 'implementation': 'generic_recurrent'}
    if info['backend'] in ('recurrent', 'stateless'):
        if require_fast and info['backend'] == 'recurrent': raise ValueError(info['reason'])
        return recurrent(graph, directory), info
    if info['backend'] == 'zero':
        def zero(q, k, v, alpha, beta):
            return v * 0 + sum(x.sum() * 0 for x in (q, k, alpha, beta))
        return zero, info

    if info['backend'] == 'kda':
        from fla.ops.kda import chunk_kda
        info['implementation'] = 'official_chunk_kda'

        def kda_cell(q, k, v, alpha, beta):
            if not bool(torch.isfinite(alpha).all() & (alpha > 0).all()):
                raise ValueError('Official KDA requires finite positive alpha')
            return chunk_kda(q, k, v, alpha.float().log(), beta,
                             use_qk_l2norm_in_kernel=False)[0]
        return kda_cell, info

    from fla.ops.generalized_delta_rule import chunk_dplr_delta_rule
    info['implementation'] = 'official_chunk_dplr_signed_gauge'
    plan = info['plan']

    def cell(q, k, v, alpha, beta):
        features = {name: x.float() for name, x in zip(('q', 'k', 'v', 'alpha', 'beta'), (q, k, v, alpha, beta))}
        cache = {}
        def expr(x): return evaluate_expression(x, features, cache)
        def vector(x):
            value = expr(x)
            if hasattr(value, 'ndim') and value.ndim == q.ndim - 1: value = value.unsqueeze(-1)
            return torch.zeros_like(q, dtype=torch.float32) + value
        diagonal = vector(plan.diagonal)
        # Signed gauge is exact for nonzero diagonal entries. Never clamp zeros.
        if not bool(torch.isfinite(diagonal).all() & (diagonal != 0).all()):
            raise ValueError('DPLR requires finite nonzero diagonal; use the general backend for this candidate')
        signs = diagonal.detach().sign().cumprod(1)
        previous_signs = torch.cat((torch.ones_like(signs[:, :1]), signs[:, :-1]), dim=1)
        a, b = plan.ranks[0] if plan.ranks else (ZERO, ZERO)
        aa, bb = vector(a) * previous_signs, vector(b) * signs
        query = q * signs.to(q.dtype)
        decay = diagonal.abs().log()
        results = []
        for write_value, write_key in plan.offsets:
            # Evaluate expressions in FP32, then round each derived factor at
            # the explicit BF16 kernel boundary. FLA requires matching dot
            # operand dtypes; decay and recurrent state remain FP32.
            result, _ = chunk_dplr_delta_rule(query, (vector(write_key) * signs).to(q.dtype),
                                             vector(write_value).to(q.dtype),
                                             aa.to(q.dtype), bb.to(q.dtype), decay)
            results.append(result)
        if not results:
            return v * 0 + sum(x.sum() * 0 for x in (q, k, alpha, beta))
        result = sum(results).to(v.dtype)
        # Generic custom autograd materializes zeros even for unused ports.
        # Preserve optimizer weight-decay behavior for their projections.
        for name, value in features.items():
            if name != 'q' and ('input', name) not in cache:
                result = result + value.reshape(-1)[0] * 0
        return result
    return cell, info
