# MQAR Training Protocol

Shared by the KDA baseline and every DSL candidate. Only the per-head memory
recurrence changes between candidates.

## Paper-Reported Settings: Matched

Source: Kimi Linear v2, section 5.1 and Figure 4,
https://arxiv.org/html/2510.26692v2 .

- Task: multi-query associative recall (MQAR).
- Two KDA layers, two heads per layer, key/value head dimension 128.
- Paper input lengths: 256, 512, 1024, 2048 tokens. All runs in this repository
  use length 2048 (256 key-value pairs).
- Paper learning rates: 0.00005, 0.0001, 0.0005, 0.001. All DSL candidates use
  1e-4, the best KDA learning rate at length 2048 in our baseline sweep.
- At most 20,000 optimizer updates per run. Every run uses exactly 20,000. No performance
  early stop or curriculum: the paper does not publish an early-stop criterion.
  Log threshold-crossing updates to expose convergence speed separately.
- Primary reported quantity: peak training accuracy; show length-1024 training
  accuracy against optimizer updates and best-over-LR peak accuracy by length.
- Preserve each LR curve, final accuracy and loss, not just selected curves.

## Explicit Reproduction Assumptions: Not Reported By Kimi

The inspected Kimi release contains no Figure-4 training/generator scripts.
The following are our frozen choices, not established paper-era settings.

- Generator: original Zoology MQAR function, pinned at repository commit
  1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb; parameterized as below. Import the
  unmodified function through a small dependency adapter rather than replacing
  its sampling distribution with a different hand-written task.
- Vocabulary size 8192, disjoint key/value domains as in that generator.
- Number of unique associations = sequence_length / 8: 32,64,128,256. One pass
  through associations; one query for each association; random distractors;
  power-law query-gap exponent 0.01. These choices make memory load grow with
  length. They are not claimed to be Kimi's undisclosed association counts.
- Batch 32 sequences/update, gradient accumulation 1, single GPU per run.
- Generate fresh deterministic batches from (data seed, optimizer step).
  The four LRs at one length see identical data and initial model parameters.
- Masked target-token accuracy over queried positions (labels != -100), measured
  on the training batch before its update. Report per-update peak and a
  100-update weighted window curve/peak; the paper does not specify aggregation
  or smoothing, so do not silently equate them. Log exact-answer accuracy too.
- Auxiliary validation: fixed independent 1024 examples per length, every 500
  updates and at the end; no validation-based LR/checkpoint selection. This
  separates generalization from a noisy training maximum.
- AdamW, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01, constant LR, gradient
  clipping 1.0; respect official no-weight-decay gate/norm/bias parameters.
- Model width 256, intermediate size 1024, untied input/output embeddings,
  official two-block KDA model with its pre-RMSNorm/residual/SwiGLU wrapper.
  Width is our explicit choice, not inferred to be paper-reported from 2x128.
- Official FLA source pinned to 7ab283715a50ecceec3a4982f275a5a7039d01c6
  (2025-10-31, publication-era KDA with no-weight-decay fix). No hybrid full
  attention layers, positional embeddings or negative-eigenvalue beta extension.
- Kernel chunk mode, conv size 4, expand_v 1, L2-normalized q/k, sigmoid output
  gate, official initialization and decay/delta order.

## Precision And Execution

FP32 model/master parameters, gradients and Adam states; BF16 autocast for
eligible projections and matmuls. Official kernels retain their FP32 sensitive
accumulation/decay behavior. Norm reductions, loss and logged metrics use FP32.
No BF16 parameter conversion or FP16 gradient scaler. TF32 is disabled for FP32
matmuls, so "other FP32" is not silently reduced to TF32.

Use the published Triton KDA kernels, which JIT compile for A100. Benchmark
PyTorch compilation separately before enabling an outer graph compiler; do not
claim torch.compile when only Triton kernels are compiled. No DDP communication
is necessary for independent LR/length runs. Real batch size remains fixed.

Checkpoint model, optimizer, step, metric history, best metric/model, Python,
NumPy, CPU/CUDA RNG and data identities atomically. Full resumable copies are published asynchronously to durable storage
(`KIMI_DSL_DURABLE`); training stops if publication goes stale.
