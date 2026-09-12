# ROUGE Regression Audit: grounded_summary versus eviseq_new

## Scope and assumptions

This audit explains the reported PubMed result:

```text
grounded_summary: ROUGE-1=49.299, ROUGE-2=21.499, ROUGE-L=45.533
```

The comparison anchor is the previously reported `eviseq_new` result:

```text
eviseq_new: ROUGE-1=49.626, ROUGE-2=21.901, ROUGE-L=45.895
```

The analysis assumes that both numbers use the same PubMed test split, PPLX
encoder, Qwen decoder, Perl ROUGE-1.5.5 wrapper, and greedy decoding. The
latest report also states that the current run used five epochs; the local YAML
still says four, so the remote resolved configuration is the authoritative
source for the final audit. These items need to be confirmed from the resolved
configurations and output metadata before the result is used as a paper claim.

## Finding

The new run is lower on all three metrics, by 0.327 ROUGE-1, 0.402 ROUGE-2,
and 0.362 ROUGE-L. It is therefore a real regression relative to the supplied
number, but it is not yet an architectural diagnosis. The current run changes
several optimization and protocol variables at the same time, so the first
experiment must restore protocol parity. Adding another architectural module
before that control would make the cause harder to identify.

## The strongest confound: optimizer-step budget

The historical anchor used one GPU with batch 48 and accumulation 2, giving an
effective batch of 96. The current distributed log reports two GPUs with batch
84 per GPU and accumulation 1, giving an effective batch of 168. The latest
run is reported as five epochs rather than the four epochs in the local YAML.

For 116,669 training examples, the approximate optimizer steps are:

```text
anchor:  ceil(116,669 / 96)  × 4 epochs = 4,864 steps
current: ceil(116,669 / 168) × 5 epochs = 3,475 steps
```

Thus the current run still receives 1,389 fewer parameter updates, or about
28.6% fewer, despite the extra epoch. Its scheduler also sees a different
number of steps and its gradient noise is different. Five epochs therefore
weakens the step-budget hypothesis but does not eliminate it; it cannot explain
the entire regression without an isolation run. This is not an argument that a
larger batch is always worse. The large-batch study by Shallue et al. shows that
quality differences depend strongly on optimizer settings and compute/step
budgets, rather than on batch size alone.^1

The practical consequence is that “four epochs” is not a matched training
budget here. Either use global batch 96, or compare runs at the same optimizer
step/token budget. With two GPUs, batch 48 per GPU and accumulation 1 gives the
same global batch of 96. If memory requires batch 24, accumulation 2 gives the
same result.

## Other variables that moved together

The current `grounded_summary` PubMed recipe differs from the `eviseq_new`
recipe in the following score-affecting ways:

- interface warm-up is 0 epochs instead of 1 warm-up plus 3 full-finetuning
  epochs;
- `max_grad_norm` is 2.0 instead of the anchor's 1.0;
- AFMR controller/focus hidden size is 384 instead of 256;
- depth and feature ranks are 256/512 instead of 128/256;
- a 1024-token focus window was added;
- grounded-copy key dimension is 256 instead of 128;
- PubMed copy mass is capped at 0.20, whereas the old recipe was uncapped;
- the decoder instruction is a different, longer biomedical prompt;
- the current runner enables the query-conditioned region router.

Any one of these can alter the learned distribution. The copy cap is especially
relevant to ROUGE: a lexical source token that was previously copied with high
probability can now receive at most the configured copy mass, forcing the LM
route to reproduce it. The cap may improve factuality, but its effect on ROUGE
must be measured rather than assumed.

The local code tests verify the DDP token-normalization path, checkpoint wiring,
and gradient finiteness on tiny models. They do not prove that the full PubMed
run has the same optimizer trajectory as the old single-GPU run. DDP is a
correctness/runtime mechanism here, not a source of expected ROUGE gains.

## Ordered hypotheses

### H1 — changed step budget and scheduler (high confidence)

The current run has fewer updates and a different linear decay schedule. This
can leave the model under-optimized even after five epochs, but the extra epoch
makes this a contributing hypothesis rather than a sufficient explanation. The
supplied observation that training loss is higher in the newer run is consistent
with it, although it does not prove it.

### H2 — warm-up and clipping changed (high confidence)

The anchor isolates the newly trainable interface for one epoch and clips the
global norm at 1.0. The current run starts full fine-tuning immediately and
allows twice the norm. These choices change how quickly the pretrained decoder
and encoder move away from their useful initialization.

### H3 — copy mixture changed (medium confidence)

The current `alpha_max: 0.20` protects an LM reserve but also limits extraction.
If PubMed references contain many source-identical terms, this can lower all
ROUGE metrics. The key dimension and copy scorer also changed at the same time.

### H4 — prompt and capacity changed (medium confidence)

The longer decoder instruction changes prompt embeddings, decoder positions,
and the controller input. Larger residual ranks and the extra long focus window
increase capacity but also change optimization and initialization statistics.

### H5 — router effect (low-to-medium confidence)

The region router is zero-gated and bounded, so it should be a small perturbation
at initialization. After training it can still change source bias, especially in
the upper decoder layers. It must be isolated with a route-off ablation.

### H6 — evaluation mismatch (must be ruled out)

Even a small difference in token normalization, prompt stripping, checkpoint
selection, or ROUGE command-line flags can move scores by several tenths. The
two runs must record the resolved config, checkpoint path, prediction count,
tokenizer identifiers, and exact ROUGE command.

## Required diagnostic matrix

Run these in order, using validation for selection and the test set only after a
recipe is frozen:

1. **Literal control:** run the old `eviseq_new` code/config with its original
   prompt, dimensions, copy settings, one warm-up plus three full epochs, global
   batch 96, clip 1.0, and greedy Perl ROUGE-1.5.5.
2. **Runner control:** run the current `grounded_summary` code with the same
   global batch, stage split, clip, prompt, dimensions, copy cap/rank, seed and
   scheduler as the literal control. This tests the refactor/DDP path.
3. **Architecture candidate:** change only the intended current module (for
   example, the region router) while keeping the runner control fixed.
4. **One-variable ablations:** route off; copy cap 1.0 versus 0.20; key rank
   128 versus 256; old versus current decoder prompt; old versus widened AFMR
   ranks.

If the five-epoch result is the only existing checkpoint, one decisive
additional check is to train the same current recipe for approximately 4,864
updates (about seven epochs at global batch 168) or to add an explicit
`max_steps=4864` mode. Recovery with the extra updates supports an optimization
explanation; no recovery shifts priority to copy, prompt, clipping, and router
ablations.

For each run, log total optimizer steps, global supervised tokens, validation
CE, validation ROUGE-1/2/L, copy mass, generated length, and the exact resolved
configuration. A candidate is retained only when its movement is acceptable on
all three ROUGE metrics. The current regression is small enough that one seed
is useful for screening but not sufficient for a paper claim; finalists should
use at least three seeds or paired bootstrap intervals.

## Research-backed next methods after parity

The regression should be repaired before introducing a new loss. Once the
current method has a matched control, the following sequence is defensible:

1. **Content/phrase-aware CE:** upweight biomedical content tokens and intact
   multi-token terms using source-only lexical properties. This is the lowest
   risk path for recovering ROUGE-1/2 because CE remains the only generation
   objective.
2. **Source-grounded contrastive loss:** align decoder states with their
   source evidence and repel hard negatives from the same article. Keep the
   loss small and teacher-forced; do not generate candidates. Contrastive
   summarization work supports the general mismatch motivation, but its reported
   gains do not transfer automatically to this architecture.^2
3. **Bounded coverage/planning signal:** add a source-region coverage state or a
   sentence-level salience auxiliary target. Structural coverage work motivates
   this direction, while recent long-document results warn that local ROUGE-2
   gains can accompany a ROUGE-L/coherence loss.^3,^4
4. **Entity/number/negation consistency:** apply an auxiliary loss only to
   evidence-sensitive spans. Biomedical guided continued pretraining shows the
   value of domain attributes such as PICO spans, but it is a separate,
   higher-cost experiment rather than part of the score-repair control.^5
5. **Sequence-level ranking:** BRIO and later sequence-level contrastive methods
   address the MLE-to-ROUGE mismatch, but they require multiple candidate
   summaries during training. Keep this as a later phase because it violates the
   current no-extra-candidate training contract.^6

ROUGE should remain the primary selection metric for the present objective, but
factuality must be measured separately. ROUGE is largely a topic/content
overlap signal and is not a factuality guarantee; QAEval-style question
answering and scientific-document factuality benchmarks are useful additions
for the later hallucination claim.^7,^8

## Decision

Do not modify the architecture again based on the 49.299/21.499/45.533 run.
First run the protocol-parity control. If the control returns near 49.626 /
21.901 / 45.895, the regression was primarily training/protocol drift. If the
candidate remains lower under parity, isolate copy cap, prompt, dimensions, and
region routing in that order. The five-epoch run makes a pure step-budget
explanation less likely, but it does not test it. Only after this audit should
an auxiliary content-aware or evidence-contrastive loss be evaluated.

## Sources

1. Christopher J. Shallue et al., “Measuring the Effects of Data Parallelism on
   Neural Network Training,” arXiv:1811.03600, 2018.
   <https://arxiv.org/abs/1811.03600>
2. Yixin Liu and Pengfei Liu, “SimCLS: A Simple Framework for Contrastive
   Learning of Abstractive Summarization,” ACL-IJCNLP 2021.
   <https://aclanthology.org/2021.acl-short.135/>
3. Wei Li et al., “Improving Neural Abstractive Document Summarization with
   Structural Regularization,” EMNLP 2018.
   <https://aclanthology.org/D18-1441/>
4. Himadri Sonowal et al., “Structure-Aware Chunking for Abstractive
   Summarization of Long Legal Documents,” JUST-NLP 2025.
   <https://aclanthology.org/2025.justnlp-main.19/>
5. Ahmed Elhady et al., “Improving Factuality in Clinical Abstractive
   Multi-Document Summarization by Guided Continued Pre-training,” NAACL 2024.
   <https://aclanthology.org/2024.naacl-short.66/>
6. Yixin Liu et al., “BRIO: Bringing Order to Abstractive Summarization,” ACL
   2022.
   <https://aclanthology.org/2022.acl-long.207/>
7. Daniel Deutsch and Dan Roth, “Understanding the Extent to which Content
   Quality Metrics Measure the Information Quality of Summaries,” CoNLL 2021.
   <https://aclanthology.org/2021.conll-1.24/>
8. Jennifer A. Bishop et al., “LongDocFACTScore: Evaluating the Factuality of
   Long Document Abstractive Summarisation,” LREC-COLING 2024.
   <https://aclanthology.org/2024.lrec-main.941/>
