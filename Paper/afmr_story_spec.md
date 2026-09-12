# Manuscript story specification

## Active artifact and evidence state

- **Active manuscript:** `Paper/afmr_question.tex`.
- **Architecture authority:** `src/eviseq_new` and `Technical_Report/AFMR_IMPLEMENTATION_PLAN.md` sections 1--10, 27--28.
- **Other artifacts:** Existing drafts and experiments are outside the scope of this manuscript. They must not define the proposed method and are not discussed in the paper.
- **Evidence state:** implementation and unit-test evidence are available; the four-dataset fine-tuning, prediction files, ROUGE-1.5.5, AlignScore, and BERTScore results are not available in this workspace. Every quality score in the manuscript is therefore a placeholder.
- **Lifecycle:** exploratory/complete internal draft, pending empirical completion. The manuscript must not use `FINAL`, `SUBMISSION_READY`, or an achieved-performance claim.

## Locked title and names

**Locked title:** *EviSeq: Source-Grounded Summarization with Composed Pretrained Encoders and Causal Decoders*

The title is descriptive and makes no performance claim. The empirical
comparison remains pending until the registered runs are complete.

The architecture is **Adaptive Full-Memory Residual (AFMR)**. The complete proposed system is called **EviSeq**. Version labels are excluded from the paper's method name.

All repeated model, dataset, and architecture names must be defined once with `\\newcommand` in the LaTeX preamble and used through those commands:

- architecture/system: `\\AFMR`, `\\AFMRFull`, `\\EviSeq`
- encoder and decoder: `\\PPLXEmbed`, `\\QwenEmbed`, `\\QwenCausal`, `\\NemotronEmbed`, `\\TfiveGemma`
- decoder-only controls: `\\QwenZeroSix`, `\\QwenFour`, `\\QwenEight`, `\\LlamaThreeTwoThree`, `\\LlamaThreeEight`, `\\NemotronThree`, `\\NemotronEight`
- datasets: `\\PubMed`, `\\ArXiv`, `\\BookSum`, `\\GovReport`
- metrics: `\\Rouge`, `\\AlignScore`, `\\BERTScore`

## Central argument

Independent pretrained encoders and causal decoders can be composed, but a width-matching projection does not decide which source representations, positions, and lexical forms should drive generation. EviSeq treats the interface as a constrained information-allocation problem. AFMR keeps one full retrieval tensor and one aligned final-state value tensor, then learns small, controller-conditioned adjustments along three axes: encoder depth, cross-space feature channels, and source-span priority. A final-state value anchor keeps content available to the decoder, while refined keys and a shared additive prior shape retrieval. A grounded copy route provides a path for copying source tokens and numbers. The whole graph is trained by the same gold-token cross-entropy used for generation. This design yields falsifiable questions about source-support signal, overlap quality, component contribution, and encoder portability.

The argument is deliberately conditional: the architecture is a hypothesis until the locked four-dataset experiments are complete. The paper can report the planned protocol and implementation evidence now; it cannot claim a ROUGE or factuality win before the runs.

## Research questions and bridges

1. **RQ1 (factual-support signal):** Does EviSeq reduce a source-unsupported-content signal relative to fine-tuned decoder-only LLMs on all four datasets? Operationalize the automatic comparison with native AlignScore consistency (higher is better) and report `H=1-C` only as a direction-transformed hallucination-oriented score (lower is better), never as a calibrated probability. A typed entity/number/relation error claim requires a separate adjudicated audit.
2. **RQ2 (quality):** How does EviSeq change ROUGE-1/2/L and BERTScore relative to the same decoder-only controls and fine-tuned T5Gemma2 under matched records, budgets, preprocessing, and decoding, with native input serialization documented for each model, on all four datasets? This guards against a factuality signal change produced by conservative or empty outputs.
3. **RQ3 (mechanism):** Which of the encoder, AFMR bridge, and grounded-copy route is necessary for the observed behavior? Run no-encoder, no-bridge, and no-grounded-copy controls on all four datasets with a fixed decoder and matched training budget.
4. **RQ4 (portability):** Does the same interface transfer across a bidirectional embedding encoder, a causal encoder, and a larger embedding encoder, and how do directionality and capacity affect quality and factuality? Run `PPLXEmbed` as the anchor plus `QwenCausal`, `QwenEmbed`, and `NemotronEmbed` variants on all four datasets.

The transitions are fixed: RQ1 tests whether source grounding changes factual support; RQ2 checks that any gain is not obtained by sacrificing summary coverage or lexical quality; RQ3 attributes an aggregate effect to the interface components; RQ4 tests whether that mechanism survives a change in encoder family and directionality.

## Method facts authorized by the code

- Encoder output: final state plus the last four hidden-state taps when `depth_taps=4`.
- Controller: masked source mean, masked decoder-prompt mean, and three inference-available budget features (`log(1+N)`, `log(1+K)`, `K/max(N,1)`) projected to dimension 256 and RMS-normalized.
- Depth route: token-wise softmax over four normalized taps, low-rank rank 128 residual around the final-state anchor; output factor zero-initialized; gate initialized at 0.02 and bounded by 0.15.
- Feature route: low-rank rank 256 encoder-to-decoder correction, feature-wise gate initialized at 0.02 and bounded by 0.20; base projection is identity when widths match or orthogonally initialized and trainable otherwise.
- Source focus: overlapping content-token windows `[32,128,512]` with 50% overlap in the base profile; controller-conditioned relative prior, bounded strength initialized at 0.10 and capped at 1.0; learned temperature initialized at 1.0 in `[0.5,2.0]`; padding and prefix positions are masked. Any task-specific capacity override is recorded in the resolved configuration.
- Decoder: Qwen causal decoder with a copied cross-attention block in every layer; cross-attention scalar gate initialized at 0.10 and bounded by 1.0; same source key/value memory is cached once per layer at generation.
- Grounded copy: offset-based source-to-decoder-token alignment, lexical/context keys plus the pooled source-prior bias, marginalization over repeated source token IDs, and a learned copy/LM mixture. It uses no reference or evidence labels.
- Objective: gold-token mixture cross-entropy only. No contrastive loss, R-Drop, NEFTune, candidate generation, reranker, knowledge distillation, or self-improvement in the main graph.
- Decoding: official comparison uses one beam, `do_sample=false`, `temperature=0`, `top_k=0`, `top_p=1`, the configured repetition and no-repeat constraints; sampling controls remain available only for separate candidate generation.
- Training: one interface warm-up epoch followed by full fine-tuning; FP32 parameters/gradients/AdamW states with CUDA BF16 autocast; global batch and step budgets are matched per dataset; final evaluation uses `last.pt`.

## Experimental contract

- All four datasets (`\\PubMed`, `\\ArXiv`, `\\BookSum`, `\\GovReport`) appear in RQ1--RQ4.
- Main baselines are fine-tuned, not prompt-only: decoder-only controls listed in the manuscript and `\\TfiveGemma` as an encoder--decoder reference.
- Use one fixed preprocessing and one fixed Perl ROUGE-1.5.5 wrapper for all primary ROUGE values. Python ROUGE is diagnostic only.
- Use identical example IDs within each dataset, fixed train/validation/test splits, equal source/target length budgets within a dataset, and no test-checkpoint selection. Record each model's native source and decoder serialization rather than treating prompts or chat templates as identical by assumption.
- Report per-example paired bootstrap intervals, dataset rows first, and a secondary macro statistic defined as the unweighted mean of the four dataset-level effects. Report prediction diagnostics (length, truncation, empty outputs, repetition, source-copy rate, latency, and peak memory); keep a macro value pending when any dataset is missing.
- Report encoder/decoder identifiers, local paths or immutable revision IDs, tokenizer IDs, prompt text, sampling settings, seeds, effective batch, optimizer schedule, precision, software versions, and failed/omitted runs.
- AlignScore is the primary factuality proxy. BERTScore is a semantic similarity metric and must not be described as a hallucination detector. Include metric limitations and, if available, a manually audited factual-error subset.

## Claim--evidence map

| Claim | Direct evidence required | Current status | Safe wording now |
|---|---|---|---|
| EviSeq can compose the selected encoder and decoder | smoke/integration run and checkpoint load | confirmed on fixtures; real backbones pending | “implements” |
| AFMR preserves a full source memory and separates keys from values | code, tensor-contract tests | confirmed | “uses/retains” |
| Grounded copy exposes source lexical forms | alignment and mixture code/tests | confirmed | “provides a copy route” |
| EviSeq reduces hallucination/factual errors | AlignScore plus audit on held-out test | unverified | “we test whether” |
| EviSeq improves ROUGE or BERTScore | scorer-locked predictions and paired intervals | unverified | “we will estimate” / placeholder |
| Each component is necessary | matched RQ3 ablations | unverified | “the ablation tests” |
| The interface transfers across encoder families | matched RQ4 runs | unverified | “we evaluate portability” |

## Paragraph contracts

- Introduction: establish the composition/interface gap, distinguish factuality from overlap, state the four-question chain, and limit claims to the planned evidence.
- Related work: position warm-start encoder--decoder composition, long-document source access, copying/coverage, factuality metrics, and T5Gemma2; state what AFMR borrows and what it does not claim.
- Method: define tensor path and equations directly from `src/eviseq_new`; explain what remains available at inference and where gradients flow.
- Experiments: make the four-dataset, baseline, metric, fairness, and statistical contracts reproducible; identify BookSum/GovReport preparation fields as pending if local files are absent.
- Results: leave every score cell blank, separate implementation validation from quality results, and provide fill-in sentences that answer RQ1--RQ4 after runs.
- Discussion/limitations: explain possible alternative causes, metric limits, long-context truncation, encoder-family confounding, and the fact that no quality result is available yet.
- Conclusion: restate the testable contribution and the evidence boundary; do not imply a win.
