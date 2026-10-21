# Independent-Head Memory DAG DSL v4

Search space used for every candidate in this repository. The exact KDA
recurrence is a member of this space and is kept as a separate reference.

## Boundary And Fixed Computation

Each layer contains independent recurrent heads. A head cannot read another
head's previous state. Fixed input/output projections may mix feature dimensions.
Use column vectors and state layout S[value_dim, key_dim].

Six available input ports. Sprev and at least one x_t-derived feature must
contribute to St. Any subset of the five feature ports may be used, provided
the update can also write nonzero information from a zero previous state:

| Port | Shape | Fixed source |
| --- | --- | --- |
| alpha | [key_dim] | Input-dependent KDA channel decay from x_t |
| beta | scalar | sigmoid(W_beta x_t), one scalar per head |
| Sprev | [value_dim, key_dim] | Previous recurrent state |
| q | [key_dim] | x_t projection, causal short convolution, SiLU, L2 norm |
| k | [key_dim] | x_t projection, causal short convolution, SiLU, L2 norm |
| v | [value_dim] | x_t projection, causal short convolution, SiLU |

q/k/v originate from x_t, never from Sprev. Their preprocessing, gate
parameterization, state initialization, and model wrapper are fixed between
baseline and future candidates. No trainable parameters are introduced by DAG
operations. Initial recurrent state is zero for each independent sequence.

There is one output port St[value_dim, key_dim]. Its inputs must have exactly
that shape and are added with coefficient +1. At least one input is required.
This final sum is outside the internal signed-sum node budget.

After St, fixed readout computes y_t = (St @ q_t) / sqrt(key_dim), preserving
the official kernel's fixed query scale. This matrix-vector multiply is
outside the searchable budget. Then apply per-head RMSNorm, the input-generated
sigmoid output gate, concatenate heads, and apply output projection. Full model
blocks keep the same pre-normalization/residual/MLP wrapper as the KDA baseline;
the cell DAG has no additional residual or implicit projection.

## Typed Operations And Per-Cell Maximum Counts

| Operation | Maximum | Contract |
| --- | --- | --- |
| ScalarScale | 1 | scalar a and tensor X produce a * X, same shape as X |
| MatMul | 2 | Standard dimension-compatible matrix product; vectors may be used as explicit row/column operands, including dot and outer products |
| SignedSum | 1 | One or more exactly same-shaped inputs, coefficient +1 or -1 for each input |
| ChannelScale | 1 | Matrix M[m,n] and vector a[n] produce Y[i,j] = M[i,j] * a[j] |
| OutputSum | exactly 1 | One or more St-shaped inputs, all coefficients +1; not charged to SignedSum |

These are upper bounds, not required counts. Dimensions and count limits are
explicit versioned configuration fields and can be changed in a future version.
No FLOPs cap or additional matrix-matrix shape restriction is imposed. Products
must also satisfy the affine-dependency rule below. Measure actual FLOPs/time.
No broadcasting is allowed except ChannelScale's explicitly defined column
scaling and ScalarScale. MatMul operand transpose/row/column views are free and
explicit; other implicit reshapes, reductions, identities, constants, nonlinear
functions, projections, or gates are not searchable operations in this version.

## Connectivity And Validity

- A cell is a finite directed acyclic graph. Time recurrence through St is the
  declared state boundary, not an intra-cell cycle.
- Each available input or intermediate output can feed any compatible operation
  or the final output. Fan-out to distinct consumers is unrestricted.
- The same source can occupy both MatMul operands, e.g. k @ k.T.
- A source occurs at most once in each SignedSum or OutputSum, regardless of its
  sign; repeating U N times to obtain an unbounded free multiplier is forbidden.
- Transpose views do not create a second independent source for sum deduplication.
- Only nodes contributing to St count as the active graph; prune dead nodes.
- Sprev must have a directed dependency path to St. Its presence as an unused
  port or only in a dead branch is insufficient. Reject state-independent
  outputs, even when a dead intermediate reads Sprev.
- At least one x_t-derived input (alpha, beta, q, k or v) must contribute to St,
  through any legal operations. The fixed readout does not satisfy this rule.
- Exclude updates that stay identically zero from zero initialization. Require
  b(x_t) = F(0, x_t) not to be the zero function. It need not be nonzero on every
  example and need not use the KDA write formula. No new nodes are introduced.
- Output shape must be exactly [value_dim,key_dim]. Merely having equal numbers
  of elements is not sufficient. No hidden learnable alignment is permitted.

## Conditional Affine Memory Restriction

For each head and cell, conditional on its supplied input features:

```text
vec(St) = A(x_t) vec(Sprev) + b(x_t)
```

A and b may be generated nonlinearly from input features. There is no required
diagonal, low-rank, KDA-specific or other structure for A. The implementation
need not construct the vectorization or A. Existing node limits, shape checks,
head independence, sum deduplication and DAG rules continue to apply.

Track a conservative boolean `depends_on_previous_state` for every result:

| Source / operation | Dependency and validity |
| --- | --- |
| Sprev | true |
| alpha / beta / q / k / v | false; fixed features sourced from x_t |
| Transpose / explicit row or column view | Preserve operand dependency |
| SignedSum / OutputSum | OR of all operand dependencies |
| ScalarScale / MatMul / ChannelScale | At most one operand may depend on Sprev; output dependency is OR |

Require `depends_on_previous_state(St) == true`. Additionally normalize the
algebraic output into tensor-contraction polynomials to verify that old-state
and current-input contributions do not cancel and that a nonzero zero-state
write remains. `St = ChannelScale(Sprev, alpha)` and `St = beta * Sprev` are
now excluded despite their paths from both inputs. An offset that cancels
against another branch does not satisfy the write requirement. Exact KDA
retains a genuine previous-state contribution and a nonzero input write.
The symbolic check treats the supplied feature ports as independent variables;
it checks algebraic zero/cancellation, not data-specific empirical variance.

Thus Sprev @ k and outer(Sprev @ k, k) are allowed, but Sprev @ Sprev,
or products of two different state-dependent branches, are rejected. Unlimited
fan-out remains valid. Even nonlinear expressions that cancel later are
rejected at the first invalid product; cancellation is not used as a proof.
Nonlinear functions of memory are not admitted inside the state-update DAG.

The fixed readout's RMSNorm and output gate are outside this restriction. Their
outputs cannot feed back into this cell's recurrent memory. Affinity is local
to each cell, conditional on the fixed supplied feature stream, including the
causal-convolution input buffers. It does not assert joint global affinity of
all stacked layers and preprocessing buffers.

Affine transitions compose associatively in mathematics. This alone gives no
efficient parallel kernel: arbitrary composed A can be dense and expensive.
Candidates still require a separate implementation and timing analysis.

`src/affinity.py` implements this dependency filter; it is not a shape checker,
search engine or parallel execution backend. Node/shape/budget validity remains
a separate contract and must also pass before executing a future candidate.

## Exact KDA Member

```text
F = ChannelScale(Sprev, alpha)
r = MatMul(F, k)
e = SignedSum(+v, -r)
w = ScalarScale(beta, e)
U = MatMul(column(w), row(k))
St = OutputSum(F, U)
```

This is exactly St = Sprev Diag(alpha) +
beta (v - Sprev Diag(alpha) k) k.T, using 1/2/1/1 internal operations in the
table's order. q is unused by this state-update DAG but is used by fixed readout.
Decay precedes the delta error. Fan-out of F is required and allowed.

Equivalently, St = Sprev Diag(alpha) (I - beta k k.T) + beta v k.T.
The coefficient and offset depend only on supplied features, so KDA is fully
retained under the v2 restriction. F/r/e/w/U/St are state-dependent; q/k/v,
alpha and beta are not.
