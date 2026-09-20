# Dynamic exact-source retrieval: four directions beyond the failed residual bridges

Date: 2026-09-20

## Scope and correction

The recent `query_regions`, `query_qspace`, `adaptive_topdown`, and
`evidence_slots` runs all underperformed the matched `direct_projection +
grounded_copy` control. They share a failure pattern: they alter or augment
source keys before the decoder query performs its actual read. A static bridge
cannot be genuinely decoder-query-dependent because the decoder state does not
exist when the bridge is computed.

The four proposals below therefore treat the bridge as a source-memory
interface plus a small cross-attention read module. In all cases the original
token K/V and H0 grounded-copy path remain available. The paper evidence is
mechanism-level evidence, not a prediction of PubMed ROUGE.

## Four candidates

### 1. Query-hierarchical exact read — first candidate

Use the Maruf et al. design: a decoder query first scores fixed source blocks
with sparsemax/entmax, then scores exact tokens inside each selected block.
Multiply the two distributions and read the original token values. Use a
bounded mixture with the existing full cross-attention and initialize the new
branch at zero. Fixed blocks can be 32–64 tokens; if sentence boundaries are
available, use sentence IDs as the coarse index.

Why it may raise R-2/R-L: the query gets a coherent local span while retaining
exact lexical values. It avoids the key-only residual mismatch and does not
force all information through a pooled vector. The paper’s caution about
residual connections is directly relevant: use an attention read and gate, not
another residual added to the source keys.

Training: token-level CE only. Sparsemax is piecewise differentiable; no oracle
span labels or auxiliary selector loss are needed. Keep a dense fallback to
avoid catastrophic evidence misses.

### 2. Landmark-gated exact block read — second candidate

For each fixed source block, create a lightweight landmark key/value used only
to choose a block. After the decoder query scores landmarks, grouped softmax
gates attention over the block’s original token K/V. Landmarks are indices/gates,
not replacement values; grounded copy never sees landmark entries.

Why it is different from slots/regions: each entry has a fixed source span and
preserves a direct route to every token in that span. There is no global pooled
residual and no competition among free latent slots. The NeurIPS paper reports
that block retrieval retains random-access flexibility and exact local context.

Training: standard CE/NLL with grouped attention; no auxiliary loss in the
original landmark training procedure. The practical risk is mismatch from
introducing landmark entries into a pretrained encoder-decoder interface, so
use a zero-initialized gate and compare token-only versus token-plus-landmark.

### 3. Posterior span/edit mixture

At each decoder step, compute a soft distribution over exact source tokens or
short spans conditioned on the decoder history. Each candidate contributes an
edit-aware token distribution derived from its H0 value plus a small
query-conditioned relation vector; marginalize candidates into the final token
probability. This extends the current copy idea toward latent alignment instead
of modifying all source keys.

Why it may raise R-2: a phrase candidate can explain a multi-token continuation
and the edit relation can handle inflection/paraphrase while staying grounded.
Use a differentiable block shortlist or low-rank all-token scoring; do not use a
gold-summary top-k shortlist.

Training: CE/NLL-only is a proposed adaptation. GPG and DYLE use top-k/oracle
approximations and, for DYLE, extra oracle/consistency objectives. Those parts
must be excluded or explicitly labeled as a separate experiment.

### 4. Coverage-aware novelty read

Accumulate the previous cross-attention distribution over original source
tokens. In later decoder layers, add a bounded learned novelty bias to the next
source read, e.g. `-beta * tanh(log1p(coverage))`, with a gate initialized to
zero. Keep a positive residual route for repeated biomedical terms by allowing
the query to override the bias and retain full attention.

Why it may raise R-L: it can reduce repeated phrases and make later summary
tokens move to new evidence. This is the riskiest direction for PubMed because
legitimate repeated entities and terminology are common.

Training: CE-only novelty bias is a new hypothesis. See et al. report that
coverage without an auxiliary coverage loss was ineffective, and that enabling
the coverage objective too early interfered with the main objective. Therefore
this should be a gated late-layer ablation, not the primary architecture.

## Ranking

1. Query-hierarchical exact read — best balance of direct summarization evidence,
   lexical preservation, and low implementation risk.
2. Landmark-gated exact block read — strongest distinction from failed pooled
   residuals and a clean CE-only story, but more pretraining/interface risk.
3. Posterior span/edit mixture — potentially highest R-2 upside, but expensive
   and easy to contaminate with train-time target-dependent candidate selection.
4. Coverage-aware novelty read — mechanism has historical negative CE-only
   evidence; use only as a controlled fourth experiment.

## Minimal fair experiment

Keep PPLX, Qwen, grounded copy, tokenizer normalization, generation settings,
epochs, seed, batch, and optimizer fixed. Run each candidate against the same
direct-projection control on validation. Log (a) new-branch gate, (b) attention
mass on original tokens, (c) block/span entropy, (d) fraction of target tokens
whose best source position is preserved, and (e) R-1/R-2/R-L. Only after a
validation win should the architecture be run once on the held-out test set.
