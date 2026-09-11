# GroundedSummary architecture and research audit

**Date:** 2026-09-11
**Scope:** `src/eviseq_new` (the primary baseline) and
`src/grounded_summary` (the current candidate). `src/eviseq_update` is retained
only as a historical negative control; it is not imported, configured, or used
by the current candidate. No other `src` architecture folder was inspected.
**Decision target:** ROUGE-1 > 49.626, ROUGE-2 > 21.953, and ROUGE-L > 45.895,
using the same ROUGE-1.5.5 protocol as the comparison runs.

## Executive decision

The current graph is internally coherent and its main gradient routes are valid.
Its architectural baseline is `eviseq_new`, which remains the primary anchor
because it has the best overall R1/ROUGE-L result. The candidate keeps that
graph's value-anchored AFMR, full-memory cross-attention, sparse
character-overlap alignment, contextual plus lexical copy key, FP32 likelihood
calculation, and one teacher-forced CE objective. It deliberately excludes
update_v2's semantic residual and evidence-contrastive path because the merged
experiment reduced the overall score.

The current runner is configured for the requested two-GPU execution with
`batch_size: 84` per GPU, `gradient_accumulation_steps: 1`, and
`save_best: false`. These are runtime choices, not evidence that the candidate
beats the `eviseq_new` score. A headline comparison must keep tokenization,
prompt, optimizer schedule, effective batch, seed, and decoding fixed across
the baseline and candidate.

No real PPLX/Qwen checkpoint was downloaded or evaluated during this audit. The
three target inequalities therefore remain hypotheses until a matched
ROUGE-1.5.5 test run is completed.

## Baseline identity and extension boundary

The answer to “is this based on `eviseq_new`?” is **yes for the computation
graph, but not yet as a literal score-control configuration**. A source comparison
shows that `grounded_summary/modeling/encoder.py` is identical to
`eviseq_new/eviseq_afmr/modeling/encoder.py`; the AFMR bridge and copied
cross-attention are also the same graph, apart from the namespace and the
explicit copy-cap validation. The current copy head also hardens empty/masked
candidate handling; those checks are correctness changes, while its cap and
mixture parameters are score-affecting changes. There is no import from
`src/eviseq_update`, no semantic-read branch, and no contrastive/candidate-
generation loss in the current runtime.

The current files nevertheless change several variables that matter to ROUGE:

| Reported run | ROUGE-1 | ROUGE-2 | ROUGE-L | Role in the experiment |
|---|---:|---:|---:|---|
| `eviseq_new` | 49.626 | 21.901 | 45.895 | Primary overall anchor |
| `eviseq_update` / v2 | 49.488 | 21.953 | 45.776 | Historical R2-positive, R1/RL-negative control |
| Required target | >49.626 | >21.953 | >45.895 | Must be exceeded by one frozen recipe |

| Variable | `eviseq_new` control | Current `grounded_summary` candidate | Consequence |
|---|---:|---:|---|
| AFMR controller / focus hidden | 256 | 384 | More capacity, so not a strict baseline |
| Depth rank / feature rank | 128 / 256 | 256 / 512 | More residual capacity, requires an ablation |
| Focus windows | 32, 128, 512 | 32, 128, 512, 1024 | Different long-range prior |
| Copy key dimension | 128 | 256 in PubMed | Different copy scorer |
| Copy cap | uncapped sigmoid mixture | `alpha_max: 0.20` in PubMed | Protects generation, changes copy behavior |
| Decoder prompt | concise `Summarize ...` prompt | longer biomedical instruction | Changes controller input and decoder context |
| Schedule / batch | 1 warm-up + 3 full in PubMed; historical score used 48 × 2 on one GPU | 0 warm-up + 4 full; 84 per GPU × 2, accumulation 1 | Not score-parity |

Therefore the scientific control should be a literal `eviseq_new` run with its
original prompt, dimensions, copy mixture and matched ROUGE wrapper. The present
candidate can be the first extension, but a gain cannot be attributed to a new
module while these baseline variables move together. The historical
`eviseq_update`/v2 path remains a negative control only; its semantic residual is
not part of the proposed architecture.

## Corpus and extraction status

The local corpus is the source code, YAML, tests and README files under
`src/eviseq_new` and `src/grounded_summary`; the historical `src/eviseq_update`
folder is mentioned only to verify that it is not on the current import path.
The code comparison is complete for the encoder, AFMR bridge, decoder,
grounded-copy head, collator, runtime and training engine. The external corpus
is the linked ACL/arXiv papers in the Sources section. Their abstracts, method
descriptions and ablation/results sections were available for inspection; no
claim here depends on an inaccessible appendix or a downloaded model.

The central decision is: **retain the `eviseq_new` graph as the only score
anchor, and add at most one zero-initialized source-side phrase/structure signal
at a time.** The purpose is to raise ROUGE-2 without allowing a new semantic
residual to rewrite the LM path that gave `eviseq_new` its R1/ROUGE-L advantage.

### Decision argument tree

- **Anchor claim — Source fact or data:** the current candidate has the same
  encoder/AFMR/cross-attention computation graph as `eviseq_new`, while its
  dimensions, copy cap, prompt and schedule differ.
  - **Implication — Reasoned inference:** call it an `eviseq_new`-derived
    candidate, not a matched baseline; freeze a literal control before claiming
    an architectural gain.
- **R2 mechanism — Author's stated position:** PROM and SpanCopy use phrase/span
  representations to improve copied multi-token units.
  - **Adaptation — Reasoned inference:** add a source-only phrase context to the
    existing copy keys, not PROM's reference-derived indicator or auxiliary loss.
- **R1/RL guard — Source fact or data:** v2's semantic residual changes the LM
  hidden state and coincided with lower R1/RL in the reported run.
  - **Decision — Reasoned inference:** keep new modules out of the LM hidden
    path, initialize their output to zero, and reject any validation R1/RL drop.
- **Protocol boundary — Source fact or data:** candidate generation, synthetic
  evidence and extra pre-training change the supervised comparison.
  - **Decision — Reasoned inference:** retain one teacher-forced CE objective and
    inference-only sampling controls.

## What was audited

The audit compared the source code, configs, tests, and README files in the two
allowed folders. The comparison was also checked against primary summarization
papers and reports. Every conclusion below is labelled as one of:

- **Source fact or data:** directly present in code, config, or a cited paper.
- **Reasoned inference:** an architectural implication that still needs an
  ablation.
- **Unverified:** cannot be established without the real checkpoint/data run.

The historical update path is `src/eviseq_update`; it is not part of the
current training or evaluation path.

## Graph comparison

### Current `grounded_summary`

```text
encoder final states H0 and depth taps
          │
          ├─ AFMR depth/feature residuals + multi-scale source prior → M
          │                                                           │
          └──────────────────────────── value anchor H0 ─────────────┤
                                                                      ▼
                Qwen cross-attention in every decoder layer → h_t
                                                                      │
                  contextual H0 + decoder-lexical copy keys           │
                                                                      ▼
                source copy distribution Pcopy + LM distribution PLM
                                                                      │
                     P = (1 − α) PLM + α Pcopy, α ≤ 0.20
```

`M` is used for cross-attention keys and `H0` is used for values. The bridge
and cross-attention parameters are zero/low-gated at initialization, so adding
the interface does not immediately overwrite the pretrained decoder behavior.
The copy head uses source/decoder character offsets, pools contextual source
states into decoder-token candidates, adds a decoder embedding feature, and
scatters duplicate token IDs into one vocabulary distribution.

The PubMed recipe uses `key_dim: 256`, initial copy mass 0.05, and
`alpha_max: 0.20` with a 0.05 LM reserve. Because 0.20 is already below
`1 − reserve = 0.95`, the reserve is currently redundant: the effective cap is
0.20. This is harmless but should be documented as a single cap to avoid
mistaking it for an additional constraint.

### Excluded update_v2 (`eviseq_update`)

The excluded variant keeps the same encoder, AFMR value anchor and per-layer
copied cross-attention, then adds a second native source attention from `H0`:

```text
h_t → independent source query over H0 → semantic context
    → gated residual (RMS(delta) ≤ 0.10 RMS(h_t)) → h'_t → LM logits
h_t → copy query over aligned token keys → Pcopy
```

The semantic branch is independent of decoder-token boundaries. It adds a
second source read and changes the hidden state seen by the LM head before the
copy/LM mixture. Its output projection starts at zero and its residual is
bounded, which makes the initialization safe, but after learning it can still
change token choice and word order. The v2 recipe uses copy key rank 128 and
semantic rank 128, cosine decay with a 5% warm-up, and explicit effective batch
96.

## Evidence ledger

| Claim | Evidence used | Confidence |
|---|---|---|
| The current graph is CE-only and does not generate training candidates. | `grounded_summary/modeling/model.py`, `training/engine.py`, and the current config contain one CE loss and greedy evaluation. | Source fact or data |
| The value anchor separates retrieval keys from source values. | `grounded_summary/modeling/afmr.py` and `modeling/decoder.py`; the value-anchor regression test keeps values unchanged when bridge retrieval weights change. | Source fact or data |
| The current copy route cannot exceed 20% mass on PubMed. | `configs/afmr_pubmed.yaml` sets `alpha_max: 0.20`; `GroundedCopyHead` computes `alpha_max * sigmoid(gate)`. | Source fact or data |
| Removing v2 semantic residual should help preserve R1/RL. | The residual is absent in the current graph and v2 changes the LM hidden state; the direction is an architectural inference, not a measured gain. | Reasoned inference |
| Phrase-context or hierarchical bias can lift R2. | PROM, HIBRIDS and HiStruct+ report gains in their own settings; transfer to this graph is unverified. | Reasoned inference |
| All three target thresholds are exceeded. | No real PPLX/Qwen test was run in this audit. | Unverified |

### Consequences for the observed scores

The reported update_v2 score was R1 49.488, R2 21.953, RL 45.776. It improved
the R2 direction but was below `eviseq_new` on R1 and RL. The current graph
therefore keeps `eviseq_new` as the anchor and does not combine the excluded
semantic branch with it. Any capacity or copy change is a separate ablation,
not an update_v2 component. The expected behavior is therefore:

1. **R1/RL protection (reasoned inference):** the pretrained LM hidden state is
   not rewritten by a second semantic residual, and copy cannot take more than
   20% probability mass.
2. **R2 upside (reasoned inference):** the 256-dimensional contextual/lexical
   key can bind biomedical terms and nearby context more accurately than the
   v2 rank-128 key, but this is not guaranteed.
3. **R2 risk (Source fact or data):** no component explicitly models source
   order or phrases. A token-level copy distribution can recover unigrams while
   still selecting adjacent words inconsistently.

## Gradient and update audit

The current implementation passes the relevant local gradient tests with tiny
random backbones. The path for a supervised token is:

```text
CE(P) → LM head and copy mixture
      → decoder hidden, copy query/gate, contextual and lexical keys
      → H0 value path and M/source-bias retrieval path
      → AFMR bridge and encoder (full-finetune stage)
```

During interface warm-up the encoder and pretrained decoder backbone are frozen;
the bridge, copied cross-attention, cross gates and grounded-copy head remain
trainable. During full fine-tuning, the encoder and decoder are unfrozen. No
`detach()` is used on the source keys or values. The value anchor intentionally
has zero direct derivative with respect to retrieval-only residual parameters,
but it still receives gradients through the value projection and encoder.

The copy likelihood is accumulated in FP32 and agrees with the dense-logit path
in regression tests. Masked/padded candidates are removed before lexical lookup;
an empty copy state returns the LM logits exactly. Duplicate source occurrences
are summed by `scatter_add`, which is the correct probability semantics for a
pointer distribution.

The remaining implementation caveats are small and observable:

- CUDA input-range validation is deferred to the collator to avoid a host sync;
  malformed GPU tensors would fail later, but normal data is validated before
  transfer.
- `bool(mask.any())` appears in a few empty-source guards. It is correct but
  can introduce a small synchronization cost on CUDA; it is not a correctness
  or ROUGE issue.
- The initial current runner was single-process. It now includes the verified
  DDP pattern: global length-bucket sharding, global-token loss normalization,
  rank-zero artifact writes, and per-rank RNG checkpoint state. DDP itself does
  not improve ROUGE. Effective batch, seed, tokenization, epochs, scheduler and
  decode settings must still be matched before attributing a score to
  architecture.

## Decoder prompt and token budgets

The prompt has three separate effects:

1. `max_source_length` is applied only while tokenizing
   `encoder_prefix + source`. The encoder prefix therefore consumes source
   positions, exactly as it does in the matched source-prompt baseline.
2. The decoder prompt and `decoder_prefix` are tokenized separately and placed
   before the target. They are masked with `-100`, so they do not add supervised
   CE tokens or truncate the target. The target is capped independently at
   `max_target_length` (including EOS).
3. The prompt embedding is pooled by `FocusController`; it can change AFMR
   depth/focus routing even though it does not remove source tokens. It also
   consumes decoder context positions and is present before all generated
   tokens.

Thus the current prompt does **not** directly make the encoder lose source
information. It does increase decoder sequence length and can indirectly alter
which source regions receive bias. The collator currently has no explicit check
that `prompt_tokens + max_target_length` and
`prompt_tokens + max_new_tokens` fit the decoder's context window. With Qwen's
usual context sizes the PubMed values are expected to fit, but a real run should
record the prompt token count and assert this inequality against the loaded
decoder config. This is a guard against a future long prompt, not a reason to
remove the current concise instruction.

For a clean ablation, run the current prompt and the exact `eviseq_new` prompt
with every other setting fixed. Do not call a prompt change an architectural
gain. If the goal is a literal baseline, use the `eviseq_new` prompt first and
only introduce the longer biomedical instruction in a separate prompt ablation.

## Research findings and applicability

### Argument reconstructed from the literature

The relevant papers support three separable mechanisms rather than one recipe:

1. **A token pointer needs local phrase information.** A unigram pointer can
   reproduce biomedical terms but has no state saying that two adjacent source
   tokens form one useful unit. This is the mechanism behind the n-gram branch in
   PROM and the span branch in Entity-based SpanCopy.
2. **A full-memory model still benefits from a soft structural prior.** HIBRIDS
   adds path/level relations as attention biases; HiStruct+ attributes much of its
   scientific-document gain to hierarchical position. Both support a bias on
   source retrieval, not replacing the memory with selected chunks.
3. **Local gains can hurt global order.** Structure-Aware Chunking reports a
   ROUGE-2 improvement together with a ROUGE-L/coherence drop. That is the same
   failure pattern seen in the v2 experiment, so any R2-oriented addition must be
   gated and evaluated on R1/RL at the same time.

These are mechanisms, not evidence that a component will improve this PubMed
checkpoint. The papers use different datasets, tokenizers and training budgets;
the proposed changes below are therefore reasoned hypotheses.

### Directly useful precedents

- **Phrase continuity:** [PROM (LREC-COLING 2024)](https://aclanthology.org/2024.lrec-main.1148/)
  enhances n-gram attention, predicts a source-copy indicator and reports gains
  in copied bi/tri/4/5-grams and ROUGE-2 (model and experiments sections). Its
  pseudo-labels come from source/reference overlap, and its auxiliary copy loss
  and two-stage pre-training are outside the current CE-only protocol. We keep
  only the source-only n-gram representation idea.
- **Span/entity evidence:** [Entity-based SpanCopy (CODI 2023)](https://aclanthology.org/2023.codi-1.9/)
  pools entity spans before copying and reports better entity factual
  consistency. Its NER and global-relevance labels are dataset/tooling
  dependencies; the transferable part is pooling adjacent source states before
  the copy score. Character offsets already available in this project provide a
  tokenizer-safe way to do that without NER.
- **Soft hierarchy:** [HIBRIDS (ACL 2022)](https://aclanthology.org/2022.acl-long.58/)
  adds a learnable bias indexed by path length and asymmetric hierarchy level to
  attention scores, then shows that removing either relation hurts. [HiStruct+
  (Findings 2022)](https://aclanthology.org/2022.findings-acl.102/) finds that
  explicit hierarchy/position helps scientific-document summarization, including
  PubMed and arXiv, but uses an extractive setting. The safe adaptation is a
  bounded, zero-initialized `source_bias` feature while retaining every source
  token.
- **Position diagnostic:** [On Context Utilization in Summarization with Large
  Language Models (ACL 2024)](https://aclanthology.org/2024.acl-long.153/)
  studies six LLMs over ten datasets and finds a U-shaped position preference;
  salient facts can be dispersed. This justifies measuring position-conditioned
  copy coverage and adding a learned prior only when it is source-structure
  aware, rather than assuming the first/last sentences are always best.
- **Copy/generate balance:** [Pointer-Generator Networks (ACL 2017)](https://aclanthology.org/P17-1099/)
  establishes the hybrid distribution; [Generalizing the Pointer Generator
  (EMNLP-IJCNLP 2019)](https://aclanthology.org/D19-1390/) warns that exact
  pointing can over-favor extraction. This supports preserving an LM reserve and
  treating the copy cap as an ablation, not silently raising it.

### Evidence that argues against tempting additions

- [Structure-Aware Chunking (JUST-NLP 2025)](https://aclanthology.org/2025.justnlp-main.19/)
  explicitly reports the R2-up/RL-down coherence gap. Chunking or sentence
  pruning is therefore not a safe default for this target.
- [LongT5](https://arxiv.org/abs/2112.07916) obtains long-context scaling from
  local plus transient-global attention. The current bridge already retains
  full memory and has multi-scale focus windows; adding a second global memory
  path would be redundant for four supervised epochs and would make attribution
  difficult.
- [Salient Information Prompting (EMNLP Industry 2024)](https://aclanthology.org/2024.emnlp-industry.4/)
  improves phrase-level ROUGE with an external keyphrase extractor, while
  faithfulness is not uniformly improved. A longer decoder prompt or extractor
  would change the protocol and consume context positions.
- [Unstructured Evidence Attribution (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.95/)
  uses synthetic evidence spans to train relevance/faithfulness. It is relevant
  to the later hallucination claim, but violates the current requirement of no
  generated training candidates and is not a first R1/R2/RL intervention.
- [GSum](https://aclanthology.org/2021.naacl-main.384/), [PRIMERA](https://aclanthology.org/2022.acl-long.360/)
  and [PEGASUS](https://arxiv.org/abs/1912.08777) add external guidance or
  extra pre-training; [SimCLS](https://aclanthology.org/2021.acl-short.135/) and
  [BRIO](https://aclanthology.org/2022.acl-long.207/) rerank generated
  candidates. None is a fair small change to this supervised comparison.
- [To Point or Not to Point](https://aclanthology.org/2021.findings-acl.298/)
  cautions that more pointing or coverage is not uniformly better. A coverage
  state should remain a later, isolated experiment rather than a default.

### External evidence ledger

| Source location | Author's stated position / result | What it supports here | Boundary |
|---|---|---|---|
| PROM, phrase-copy model and experiments sections | n-gram attention plus a copy indicator improves copied 2–5-grams and ROUGE, with the strongest R2 gain in a pre-training/two-stage variant | A phrase representation is a credible R2 mechanism | Reference-overlap pseudo-labels and auxiliary loss are excluded |
| HIBRIDS, hierarchical-bias model and ablations | Path length and asymmetric hierarchy-level biases improve long-document coverage; removing either relation hurts | A soft source retrieval bias is worth an isolated test | PubMed rhetorical structure is not guaranteed to match the source tree |
| HiStruct+, structure/position analysis | Hierarchical position is a major contributor on scientific document summarization, including PubMed/arXiv | Source sentence/section features can be useful | The reported system is extractive, so no score transfer is assumed |
| Ravaut et al., context-utilization study | Six LLMs and ten datasets show uneven, often U-shaped, use of long inputs | Measure position-conditioned coverage and avoid fixed lead bias | It is a diagnosis/benchmark, not a trainable AFMR module |
| Entity-based SpanCopy, model section and results | Span pooling improves entity-level factual consistency and saliency | Pool adjacent source states before a token copy score | Their NER/global-relevance supervision is not used |
| Structure-Aware Chunking, abstract/results | Rhetorical chunking raises local ROUGE-2 but can lower ROUGE-L/coherence | Reject chunking/pruning as a first change; protect RL | Legal documents and chunking setting differ from PubMed |
| Salient Information Prompting, abstract/analysis | Phrase keyphrases improve ROUGE recall/F1, but faithfulness is not uniformly improved | Phrase-level information is useful evidence | External extractor and longer prompt change protocol |
| LongT5, architecture section | Local plus transient-global attention scales long inputs | Confirms the value of local/global separation | Current AFMR already has full memory and multi-scale focus |
| Pang et al., top-down/bottom-up method and experiments | Coarse global representations correct local token states and improve long-document summarization | B1 top-down sentence memory is a principled non-copy direction | Do not replace the pretrained encoder or add token-wide quadratic attention |
| Planning with Learned Entity Prompts, planning objective | Ordered intermediate entity plans improve specificity and can reduce hallucinated entities | B2 internal plan slots are a protocol-preserving research lead | Published method generates plan tokens and changes target format |
| Semformer, architecture and fine-tuning sections | Latent semantic planning reduces teacher-forcing shortcuts | Motivates an internal global workspace | Requires an autoencoder/extra planning loss in the published form |
| DecoderLens, cross-layer analysis | Different encoder depths solve different subtasks | Supports depth-aware cross-gate diagnostics | It is interpretability evidence, not a score result |
| Z-Code++, architecture and pre-training sections | Disentangled content/position and hierarchical fusion improve summarization after extra pre-training | Supports source position as a soft feature | Full encoder replacement/continued pre-training is out of scope |
| Sparsity and Sentence Structure, analysis | Encoder-decoder attention can focus on a sparse set of sentences | Measure sentence coverage and profile efficiency | Hard top-k masking risks recall and ROUGE-L |
| Explanatory Summarization with Discourse-Driven Planning, abstract | Discourse plans improve scientific lay-summary quality and hallucination metrics | Future planning direction after the ROUGE claim | Explicit plan conditioning changes the current decoder protocol |
| DeepSeek-V3, MTP section | Multi-token prediction is reported as a beneficial denser training objective | B9 is a teacher-forced local-continuation ablation | Evidence is from large-scale pre-training, not this PubMed fine-tune |
| HMT, memory hierarchy and summarization experiments | Segment-level recurrence can preserve early tokens, pass compressed memory forward and recall relevant history | B11 is a conditional long-context direction if truncation is the bottleneck | It changes the encoder execution and its evidence is not PubMed-specific |
| AWESOME, abstract and memory mechanisms | External segment memory preserves global context while divide-and-conquer reduces memory | B11 provides a second memory design to compare with B1 | Their global salient-content augmentation and multi-stage protocol are excluded |
| LOCOST, model and complexity sections | A bidirectional state-space encoder provides long-range context with subquadratic complexity | A long-context backbone is a possible future replacement experiment | Replacing the pretrained encoder and adding pre-training would confound the score claim |
| Summarizing Long Regulatory Documents with a Multi-Step Pipeline, abstract | Extractive pre-selection can help decoder-only models but can hurt long-context encoder-decoder models | Keep full memory and test soft routing rather than hard source pruning | The domain and models differ from PubMed |

The ledger distinguishes an author's result from the architectural inference made
for this project. None of the external rows is a measured result for the local
checkpoint.

### Beyond copy: broader architecture candidates

Copy is only one route to the target. The current AFMR has three structural
properties that leave room for a second family of interventions: it pools the
document into one controller vector, it creates multi-scale source bias but does
not update source tokens from a sentence-level global state, and every decoder
layer receives a learned scalar cross gate rather than a content-conditioned
layer profile.

#### B1 — top-down global source memory (recommended second experiment)

[Long Document Summarization with Top-down and Bottom-up
Inference (Findings 2023)](https://aclanthology.org/2023.findings-eacl.94/) argues
that local token representations need a top-down correction from a coarser
document representation. The current AFMR already has local/full-memory access,
but its `FocusController` is a single pooled vector and its focus prior is a
source bias; it does not write a learned sentence-level global state back into
`M`.

A compatible adaptation is:

1. pool `H0` by sentence or reliable source line into at most 128 sentence
   vectors;
2. run one small self-attention/MLP over those vectors to obtain global context;
3. scatter the corrected sentence state back to its source tokens and add it to
   the retrieval memory `M` through a zero-initialized, bounded projection;
4. keep `H0` as the value memory and keep the copy path unchanged.

This targets missing cross-sentence facts and should be evaluated primarily on
R1/RL, with R2 as a secondary measure. It adds no decoder prompt, no generated
plan and no auxiliary loss. The implementation must avoid an \(S^2\) matrix over
all tokens; the sentence-level matrix is bounded by the number of sentences.
If sentence boundaries are unreliable, this branch must be disabled rather than
guessing. It is a stronger architectural change than P3's scalar source bias,
so test it only after protocol parity and the P2 phrase ablation.

The current collator exposes source text and token offsets but does not emit a
sentence-ID tensor. A real B1/P3 implementation therefore needs a deterministic
source-side span map (newline/list boundaries first, punctuation fallback) and a
test that truncation keeps the map aligned with encoder offsets. This metadata
does not change the target or the decoder prompt, but silently misaligned spans
would be worse than omitting the branch.

#### B2 — internal latent planner slots (research lead, not first default)

[Planning with Learned Entity Prompts (TACL 2021)](https://aclanthology.org/2021.tacl-1.88/)
and [Semformer (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.1039/)
show that an intermediate plan can reduce next-token shortcuts and improve
summary planning. Their published protocols generate or predict plan tokens and
use additional supervision, which is outside the current constraint. A safer
adaptation is to create four internal plan slots from source sentence pools,
append them only to the cross-attention memory, and initialize their value/output
projection to zero. The decoder then has a compact global workspace without
putting extra tokens in the user-visible decoder prefix.

This is a hypothesis for ROUGE-L and information coverage, not a claim that the
plan slots know the reference order. To keep it CE-only, the slots receive
gradient only through the ordinary target likelihood. A no-op initialization and
an explicit `max_plan_slots` cap are required; otherwise the extra memory can
compete with token-level evidence and reproduce the v2 regression.

#### B3 — layerwise cross-attention profile (small, but lower priority)

The decoder currently uses one learned scalar gate per cross-attention layer.
Replace that scalar with a controller-conditioned vector of bounded gates, with
the initial vector equal to the current scalar gate. Lower layers can then retain
lexical alignment while upper layers use a stronger global source signal. This
idea is consistent with [DecoderLens (Findings NAACL 2024)](https://aclanthology.org/2024.findings-naacl.296/),
which finds that different encoder depths support different subtasks, but that
paper is interpretability work rather than a summarization gain claim.

The branch changes the decoder hidden state, so it must be initialized to the
exact current gate and tested with a hidden/logit parity check. It is not a first
experiment: its likely gain is smaller than B1 and its failure mode is another
R1/RL drift.

#### B4 — content/position disentangling (diagnostic only)

[Z-Code++ (ACL 2023)](https://aclanthology.org/2023.acl-long.279/) combines
disentangled content/position representations with hierarchical fusion and
reports broad summarization gains after extra pre-training. The current encoder
is pretrained and frozen/unfrozen through an adapter; replacing its self-attention
or adding a second hierarchical encoder would invalidate the matched comparison.
The usable low-risk fragment is a source-position feature in `source_bias`,
already covered by P3. Do not implement a new encoder block before B1 has been
tested.

#### B5 — selective cross-attention (efficiency, not a score default)

[Sparsity and Sentence Structure in Encoder-Decoder Attention](https://aclanthology.org/2021.emnlp-main.739/)
shows that document summarizers often attend to a sparse set of sentences. A
top-k sentence mask could reduce memory cost, but it can delete a fact needed
for recall and would make a R2 gain/RL loss more likely. Keep full-memory
attention; if profiling requires a speed experiment, use a soft bias or a
training-only mask ablation and report coverage of the removed sentences.

#### B6 — training-only semantic planning (not compatible with the first claim)

[Explanatory Summarization with Discourse-Driven Planning (TACL 2025)](https://aclanthology.org/2025.tacl-1.53/)
reports that discourse plans can improve summary quality and reduce
hallucination on lay scientific summaries. Its plan is an explicit input/output
conditioning protocol. It is relevant to the later hallucination claim, but
using it now would change the decoder prompt, target format and evaluation
contract. Keep it as a future paper extension rather than mixing it into the
ROUGE-first architecture.

#### B7 — anchor-preserving Fisher regularization (training ablation)

[CSTRL (Findings ACL 2025)](https://aclanthology.org/2025.findings-acl.1360/)
uses Fisher-matrix regularization to reduce parameter decay and knowledge loss
when transferring a pretrained model to a specialized medical summarization
task. The domain is radiology rather than PubMed, and the paper does not prove a
gain for this graph. The transferable training idea is an anchor penalty on the
pretrained encoder/decoder only:

```text
L = L_CE + λ/2 · Σ_i F_i · (θ_i − θ_i^new-anchor)^2
```

`θ_new-anchor` is a frozen literal `eviseq_new` control checkpoint; bridge and
new route parameters are excluded from the penalty. This could protect the
lexical/fluency behavior responsible for R1/RL while the source-side module
learns. It requires an additional Fisher-estimation pass and can over-constrain
adaptation, so it is a later training ablation, never bundled with P2/B1.

#### B8 — causal-encoder backbone as a controlled ablation

The adapter can consume a causal encoder such as Qwen3 when the checkpoint
exposes `last_hidden_state`, its layer list and the requested depth taps. The
current implementation passes the ordinary attention mask and does not assume a
bidirectional encoder; `upper_bidirectional_layers` remains zero. This is a
backbone comparison, not a new module. If tested, keep the AFMR dimensions,
copy alignment, prompt, optimizer and decoding fixed, and compare the causal
encoder against the PPLX/embedding encoder on the same split. A score change
cannot be attributed to P2/B1 if the encoder family changes at the same time.

#### B9 — one-step-ahead multi-token auxiliary CE (training ablation)

The [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), building on
[Gloeckle et al. (ICML 2024)](https://proceedings.mlr.press/v235/gloeckle24a.html),
reports multi-token prediction as a denser training signal. A small supervised
version is compatible with the no-generation rule: at decoder position `t`,
use the teacher-forced hidden state to predict the gold token at `t+2`, reusing
the tied LM head, and add a small scheduled weight to the ordinary loss. No
sampled prefix, candidate summary or self-improvement loop is involved.

```text
L_train = L_CE(y_t | y_<t, x) + λ_mtp L_CE(y_{t+2} | y_≤t, x)
```

Start with `λ_mtp = 0.05` and ramp it from zero over the first 10% of updates;
mask positions without a valid `t+2` target. This gives a direct local
continuation signal that could lift ROUGE-2, while keeping the main one-token
distribution unchanged at inference. It is not a default because the evidence
comes from large-scale pre-training rather than PubMed fine-tuning and the extra
loss can trade R1/RL for local n-gram fit. Test it only after the literal control
and P2/B1, with an ablation at `λ_mtp = 0`.

#### B10 — query-conditioned hierarchical routing (non-copy, preferred if B1 is too costly)

The bridge currently computes one static `source_bias` vector per document, so
every decoder position receives the same focus prior. The cross-attention query
already changes the token-level scores, but there is no explicit low-cost route
for a decoder position to choose a different *coarse region* of the document.
This is the main non-copy gap after the value anchor is fixed.

Use the existing multi-scale windows to build a bounded set of region summaries
`G` from `M` (or from sentence pools when B1 metadata is available). At each
decoder layer, score the current query against those summaries with a small
low-rank router, overlap-add the region weights back to source-token positions,
and add the result as a four-dimensional attention bias:

```text
q_t = W_q(RMSNorm(h_t))
r_t = softmax(q_t G^T / sqrt(d_router))
b_{t,s} = beta * tanh(overlap_add(r_t) - mean_s(overlap_add(r_t)))
A_{t,s} = A_{t,s} + b_{t,s}
```

`beta` and the router output must start at zero. The existing `H0` value memory,
copy candidates, and full token memory remain untouched. The implementation can
cache `G` with the cross-attention key cache during generation, while computing
only `T x R` router scores instead of a second `T x S` semantic attention matrix.
The design is motivated by the local/global separation in [LongT5](https://arxiv.org/abs/2112.07916),
the hierarchical recall in [HMT (NAACL 2025)](https://aclanthology.org/2025.naacl-long.410/),
and the global-context memory in [AWESOME (NAACL 2024)](https://aclanthology.org/2024.naacl-long.330/),
but the query-conditioned bias is a new adaptation to this AFMR interface.

This branch is expected to affect R1/RL first because different generated
sentences can retrieve different source regions; R2 is a secondary measure. It
must pass step-zero hidden/logit parity, finite-gradient checks, and a generation
cache parity test. A small cap (for example `beta <= 0.10` initially) is safer
than allowing the router to replace the pretrained cross-attention. If it lowers
R1 or RL while improving R2, reject it as another local-coherence tradeoff.

The implemented first ablation is deliberately smaller than the full proposal:
`region_router` pools one non-overlapping 128-token partition over compact content
positions, uses rank 64 query/key projections, applies a masked region softmax,
and centers the gathered token route before a conservative `gate_max: 0.05`
tanh cap. The
scalar route gate is initialized to zero, and the route is installed only in the
upper decoder quarter (at least four layers when the decoder is deep enough).
The route is passed separately from the static `[B,S]` `source_bias`; `H0` stays
the value memory and the copy head receives the unchanged source bias and
candidate state. Region summaries are built once per source; projected region
keys are cached during generation while query-to-region scores are recomputed
for each decoder query. Finished-row compaction selects both cache tensors.
This one-scale, no-overlap version is the controlled experiment; multi-scale
overlap-add routing remains a later extension rather than an implicit claim.

#### B11 — segment-recurrent memory (conditional long-context direction)

If a length audit shows that many PubMed articles are truncated at
`max_source_length`, use a shared encoder over source segments and carry a small
set of compressed historical states into the next segment. [HMT](https://aclanthology.org/2025.naacl-long.410/)
uses segment-level recurrence and a hierarchy of token/memory states, while
[AWESOME](https://aclanthology.org/2024.naacl-long.330/) uses external memory to
retain global context during divide-and-conquer summarization. A protocol-safe
adaptation would carry only source-derived memory vectors into `M`, keep the
visible segment's `H0` values and copy offsets unchanged, and cap the historical
memory at a fixed small `K`.

This is not a first score experiment. It changes encoder execution, requires
segment-aware checkpointing and may introduce boundary artifacts. The 2024
regulatory-document study also reports that extractive pre-selection can hurt
long-context encoder-decoder models, so the branch must not prune source
sentences. Only test B11 when truncation, rather than token routing, is shown to
be the dominant error source. A full replacement with a state-space encoder such
as [LOCOST](https://aclanthology.org/2024.eacl-long.69/) is a separate backbone
study, not an extension of the `eviseq_new` score anchor.

#### B12 — bounded decoder coverage state (later R1/RL ablation)

The decoder has no explicit history of which source regions were already used.
A small recurrent coverage state over the same coarse regions as B10 could add a
negative, bounded bias to regions repeatedly attended by previous generated
tokens. This is an architecture/inference mechanism rather than a copy feature,
but it is sensitive to early decoding errors and can suppress a legitimately
repeated term. Keep it disabled for the main run; if tested, use greedy decoding,
reset the state on each example, cap the penalty, and compare source coverage and
repetition alongside all three ROUGE scores. The earlier coverage literature is
why this remains an isolated ablation rather than a bundled improvement.

#### Broad-priority decision

The order after the literal `eviseq_new` control is:

```text
P1 cap/prompt/rank controls
→ P2 source-only phrase context (R2)
→ B1 top-down sentence memory (R1/RL)
→ B10 query-conditioned region routing (R1/RL, lower metadata cost)
→ P3 soft section/position bias
→ B2 internal plan slots (R1/RL, higher risk)
→ P4 typed candidate prior
→ B7 Fisher anchor (training ablation)
→ B9 MTP auxiliary CE (R2-focused training ablation)
```

For a non-copy route, B10 is the preferred alternative to B1 when a dynamic
decoder-to-region decision is more useful than writing a static sentence state
back into `M`. B11 is conditional on truncation evidence and B12 remains a later
coverage ablation. Only one of P2, B1, B10 or P3 should be enabled in a run. A
module is retained only
when the validation movement is compatible with all three target metrics; a
single-metric R2 improvement is not sufficient.

### Deep-learning design checks

The capacity/optimization distinction from Goodfellow, Bengio and Courville is
useful here:

- **Task / performance / experience:** the task is teacher-forced biomedical
  abstract generation; the performance contract is the three ROUGE-1.5.5
  inequalities; the experience is one fixed PubMed split, tokenizer, prompt,
  optimizer schedule, seed and decoding protocol. A changed prompt or effective
  batch is a changed experience, not a free architecture improvement.
- **Train-error-first:** if train CE remains high, first fix schedule, effective
  supervised-token batch, source truncation and warm-up trainability. If train CE
  falls while validation R1/RL fall, suspect overfitting, copy overuse or
  protocol mismatch before adding capacity.
- **Architecture as prior:** phrase continuity and top-down sentence memory are
  useful only because biomedical abstracts contain local terminology and
  cross-sentence relations. They should be soft priors that preserve the
  pretrained LM and all source tokens, not hard selection rules.
- **Gradient conditioning:** zero output projections, bounded gates and FP32
  likelihood keep a new route near the known-good function at initialization.
  The first test for every module is hidden/logit parity at step zero, followed
  by a finite-gradient check and a one-variable validation ablation.

Increasing `controller_dim`, `depth_rank` or `feature_rank` without a matched
control is not evidence of better capacity: it changes both the hypothesis class
and the optimization problem. The current 1.37B backbone leaves parameter room,
but the evidence favors a low-rank route with a measurable inductive bias over a
wide undirected expansion.

## Recommended changes, in order

### P0 — protocol parity (implemented)

The current runner supports two-GPU DDP with 84 examples per GPU,
`gradient_accumulation_steps: 1`, four full-fine-tuning epochs, `max_grad_norm:
2.0`, and `last.pt` (`save_best: false`). This is a speed/runtime setting, not
the historical `eviseq_new` score protocol. For a fair headline comparison,
first reproduce the `eviseq_new` control with its original prompt, dimensions,
copy mixture, schedule and effective batch; then use the same two-GPU protocol
for every extension. Keep data normalization, seed, maximum lengths, greedy
decoding and the ROUGE-1.5.5 wrapper identical. The queue exposes
`AFMR_REGION_ROUTER=false` as a static-routing control and separates its output
directory from the enabled route run.

### P1 — low-risk score ablations

Run one variable at a time on validation, then repeat finalists with three seeds:

1. grounded copy off (LM-only value-anchor control);
2. current copy rank 256 vs rank 128;
3. copy cap matching `eviseq_new` (uncapped, `alpha_max: 1.0`) vs 0.20 and
   0.35;
4. current decoder prompt vs the `eviseq_new` decoder prompt.

Use the same `last.pt` policy and evaluate the held-out test only after the
configuration is frozen. Do not choose a test checkpoint by ROUGE after seeing
the test result.

### P2 — recommended architectural experiment for the R2 gap

Add an **order-aware phrase-context feature** inside the existing copy head,
without adding a second semantic residual or a new training objective. This is
the highest-value, lowest-disruption hypothesis from the literature:

1. From the existing character-overlap edges, identify adjacent source spans
   for n=2 and n=3. Do not use the reference to create spans, select spans, or
   assign labels. Keep the original token candidates and duplicate-ID
   `scatter_add` semantics.
2. Pool the value-anchor states `H0` over each span. Project the span vector with
   a low-rank branch and add it to the current contextual key before the final
   RMS normalization. Retain the unigram lexical feature and existing
   `source_bias`.
3. Make the branch **zero-initialized at the output**. At step zero,
   `CopyState.keys`, copy attention and mixed logits must match the control to
   numerical tolerance. Give it a bounded gate (start 0, cap 0.10–0.20) so a
   useful phrase signal can grow without rewriting the LM hidden state.
4. Keep the same token-level distribution, LM reserve and `alpha_max`; the
   decoder still emits one token at a time. The feature supplies local
   continuity, not a second decoder or a phrase-level generation vocabulary.
5. Train only with the existing gold-reference CE. No candidate generation,
   pseudo-reference, contrastive loss, auxiliary copy labels or teacher output
   is introduced.

For a candidate (i), the intended operation is

```text
r_i = U(P(mean(H0_span_i))),  U initialized to zero
g_i = g_max * tanh(gate_i),   gate_i initialized to zero
k_i = RMSNorm(k_context_i + k_lexical_i + g_i r_i)
```

The span index is built from source character offsets and source-token
adjacency. It is not built from target/reference overlap. The branch must never
feed `h_t` or the LM head; it only changes the key used by the existing copy
attention. This keeps the value anchor (`H0` as values, `M` as retrieval keys)
and makes the gradient route auditable: CE → copy score → phrase projection →
`H0` encoder states. Do not materialize a source-by-source attention matrix; a
bounded list of n=2/3 spans has linear memory in the candidate width.

For decoder hidden size 1024, copy key 256 and phrase rank 64, two projections
(`1024 × 64` and `64 × 256`) add about 82k weights including a small gate. This
is negligible relative to the 1.37B backbone and does not add a second
full-memory tensor. Start with rank 64; rank 128 is the only planned capacity
ablation. A phrase branch is accepted only if validation R1 and RL do not
fall outside seed noise while R2 improves; a R2-only gain is the known
coherence failure mode and is rejected.

The adaptation is novel at the system level because it uses existing
tokenizer-safe character alignments and the value anchor, while PROM uses
reference-derived indicators and a separate auxiliary objective. The literature
motivates the mechanism; it does not establish a gain on this dataset.

### P3 — structure-aware source prior

If P2 is neutral, add a separately gated, zero-initialized sentence/section
prior to the existing AFMR `source_bias`. Infer boundaries from source text and
retain all tokens; do not prune to selected sentences. The prior should be
bounded and consume the same H0/controller state. This follows the mechanism
idea in HIBRIDS/HiStruct+ while preserving the full-memory graph. Measure R1,
R2, RL and source coverage because a stronger prior can improve R2 while
dropping recall.

Use only deterministic source-side features that are available at inference:
sentence index/relative position, section-heading role when a reliable PubMed
heading is present, and a clipped distance to the current copy candidate. Start
with a single role/position table plus one controller projection, all zero at
initialization. If the heading parser is uncertain, emit zero bias for that
sentence instead of guessing. This makes the ablation a retrieval prior rather
than a new encoder or a hidden-state residual.

### P4 — optional typed candidate prior

If P2 and P3 are neutral, test a tiny source-only bias for candidate classes that
are common in PubMed (number, unit, percent, punctuation, gene/protein-like
token, and ordinary word). Compute the class from character spans and tokenizer
metadata; do not add an external NER model or reference labels. Add the bounded
zero-initialized bias only to copy scores. This is a low-cost exact-match
hypothesis for R1/R2, but its evidence is weaker than P2 and it should not be
bundled with P2 in the first run.

### Do not make these defaults

Do not add an independent semantic residual, coverage, or candidate-based
contrastive/KD/RL training to the main recipe. Each can change logits or the
training protocol enough to recreate the R1/RL regression. Do not add chunking,
sentence pruning, a second global memory, a new decoder prompt, or a new
pre-training phase to the first experiment. If any of these is tested later,
give it a separate run directory and keep the literal `eviseq_new` control as
the anchor.

## Training diagnosis lens

Use the capacity/optimization split from the deep-learning text when reading the
next run. If train CE stays high, first check effective batch, learning-rate
schedule, warm-up trainability and source truncation before adding modules. If
train CE falls while validation ROUGE falls, the likely causes are overfitting,
copy overuse, prompt mismatch or decoding mismatch; increasing rank or adding a
semantic branch is not a diagnosis. Record train/validation CE, supervised-token
count, mean copy mass, copy-source coverage and gradient norm together. Change
one variable per ablation so a R2 gain can be attributed to the mechanism that
produced it.

## Evaluation contract for the claim

The claim is supported only if one frozen recipe exceeds all three thresholds on
the same test split and metric implementation:

```text
ROUGE-1 > 49.626
ROUGE-2 > 21.953
ROUGE-L > 45.895
```

Report mean and standard deviation over at least three seeds for the final
candidate, plus the exact resolved config and decoder prompt token count. Run
Python ROUGE only as a diagnostic; use Perl ROUGE-1.5.5 with identical options
for the headline comparison. A real run is also required to verify that the
prompt-plus-target length fits the decoder context and that either the matched
historical one-GPU control or the current 84 × 1 × 2-GPU protocol fits available
memory. Record the effective supervised-token batch, not only examples per GPU.

## Recall questions

1. Why can a bounded semantic residual improve R2 while lowering R1 and RL?
2. Which path receives direct gradient when the target token is absent from the
   source copy candidates?
3. Why is `generate_reserve: 0.05` redundant when `alpha_max: 0.20`?
4. Does changing the decoder prompt alter `max_source_length`, decoder context,
   controller routing, or all three?
5. Which proposed P2 feature changes the initial logits, and why should it be
   zero-initialized?
6. Why is the current `grounded_summary` graph an `eviseq_new`-derived candidate
   rather than a literal score control?
7. Which metric pattern would reject a phrase or structure module even if its
   ROUGE-2 score rises?

## Sources

- [See, Liu & Manning, 2017 — Pointer-Generator Networks](https://aclanthology.org/P17-1099/)
- [Shen et al., 2019 — Generalizing the Pointer Generator](https://aclanthology.org/D19-1390/)
- [Xiao & Carenini, 2023 — Entity-based SpanCopy](https://aclanthology.org/2023.codi-1.9/)
- [Ma et al., 2024 — PROM](https://aclanthology.org/2024.lrec-main.1148/)
- [Cao & Wang, 2022 — HIBRIDS](https://aclanthology.org/2022.acl-long.58/)
- [Ruan et al., 2022 — HiStruct+](https://aclanthology.org/2022.findings-acl.102/)
- [Ravaut et al., 2024 — On Context Utilization in Summarization](https://aclanthology.org/2024.acl-long.153/)
- [Sonowal & Sadhu, 2025 — Structure-Aware Chunking](https://aclanthology.org/2025.justnlp-main.19/)
- [Ainslie et al., 2021 — LongT5](https://arxiv.org/abs/2112.07916)
- [Pang et al., 2023 — Long Document Summarization with Top-down and Bottom-up Inference](https://aclanthology.org/2023.findings-eacl.94/)
- [Narayan et al., 2021 — Planning with Learned Entity Prompts](https://aclanthology.org/2021.tacl-1.88/)
- [Yin et al., 2024 — Semformer](https://aclanthology.org/2024.emnlp-main.1039/)
- [Langedijk et al., 2024 — DecoderLens](https://aclanthology.org/2024.findings-naacl.296/)
- [He et al., 2023 — Z-Code++](https://aclanthology.org/2023.acl-long.279/)
- [Manakul & Gales, 2021 — Sparsity and Sentence Structure](https://aclanthology.org/2021.emnlp-main.739/)
- [Liu et al., 2025 — Explanatory Summarization with Discourse-Driven Planning](https://aclanthology.org/2025.tacl-1.53/)
- [Naznin et al., 2025 — CSTRL](https://aclanthology.org/2025.findings-acl.1360/)
- [He et al., 2025 — HMT: Hierarchical Memory Transformer](https://aclanthology.org/2025.naacl-long.410/)
- [Cao & Wang, 2024 — AWESOME: GPU Memory-constrained Long Document Summarization](https://aclanthology.org/2024.naacl-long.330/)
- [Le Bronnec et al., 2024 — LOCOST: State-Space Models for Long Document Abstractive Summarization](https://aclanthology.org/2024.eacl-long.69/)
- [Sie et al., 2024 — Summarizing Long Regulatory Documents with a Multi-Step Pipeline](https://aclanthology.org/2024.nllp-1.2/)
- [DeepSeek-AI, 2024 — DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Gloeckle et al., 2024 — Better & Faster Large Language Models via Multi-token Prediction](https://proceedings.mlr.press/v235/gloeckle24a.html)
- [Xu et al., 2024 — Salient Information Prompting](https://aclanthology.org/2024.emnlp-industry.4/)
- [Unstructured Evidence Attribution, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.95/)
- [Dou et al., 2021 — GSum](https://aclanthology.org/2021.naacl-main.384/)
- [Xiao et al., 2022 — PRIMERA](https://aclanthology.org/2022.acl-long.360/)
- [Zhang et al., 2020 — PEGASUS](https://arxiv.org/abs/1912.08777)
- [Liu & Liu, 2021 — SimCLS](https://aclanthology.org/2021.acl-short.135/)
- [Wilber et al., 2021 — To Point or Not to Point](https://aclanthology.org/2021.findings-acl.298/)
- [Kryscinski et al., 2020 — FactCC](https://aclanthology.org/2020.emnlp-main.750/)

## Execution additions implemented after the audit

The B10-lite query-conditioned region route described above is now wired through
`BridgeState`, the AFMR model, every wrapped decoder layer that falls in the
configured upper-quarter range, and the cached generation compaction path. Its
configuration is included in `architecture_spec`, so a checkpoint cannot be
resumed with a different region partition or route width by accident. Optional
route construction preserves same-seed initialization of all shared modules,
and projected region keys are cached during autoregressive evaluation while
query scores remain token-dependent.
The local contract tests cover disabled/zero-gate parity, content-only pooling,
bounded query-dependent bias, route gradients before and after a gate update,
copy/value-anchor isolation, BF16 forward, and cached versus uncached
generation. These tests use the tiny offline backbone and do not establish a
ROUGE gain; the PubMed run remains the score gate.

The runner now supports `torchrun` with two or more workers without changing
the AFMR computation graph or objective. A global length-bucket batch is
constructed once, then strided across ranks; an incomplete final batch uses a
masked placeholder on ranks that have no remaining example. During each
gradient-accumulation window, the local CE sum is multiplied by
`world_size / global_supervised_token_count` before DDP's gradient average. This
is the same token-normalized gradient as a single process over the complete
global window, including uneven last batches. Metrics and checkpoints are
written by rank zero; checkpoints additionally keep one Python/NumPy/torch RNG
state per rank for deterministic resume.

Evaluation uses a non-overlapping deterministic index sampler, gathers rows as
`(index, id, prediction, reference)`, sorts them on rank zero, and writes one
ordered JSONL. The distributed path intentionally requires a fresh output file
to avoid mixing a rank-sharded run with the single-process prefix-resume
format. The benchmark remains greedy by default.

Candidate generation is an inference-only branch. `temperature`, `top_k`, and
`top_p` are validated in the config and can be supplied on the CLI; filtering
is applied in that order and each rank uses a rank-offset seed. The training
engine never imports or calls a generation function, so no self-improvement or
reference-free candidate loop has been introduced. `model.attention_implementation`
accepts `sdpa` (default), `flash_attention_2`, or `eager` and is passed to both
pretrained backbones. The custom AFMR cross-attention continues to use
PyTorch's scaled-dot-product attention, which can select an available fused
kernel under the selected runtime.
