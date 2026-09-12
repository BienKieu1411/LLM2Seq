# Title and abstract alternatives: Agent B

> **Candidate record only.** The coordinator later selected a shorter
> parenthesis-free abstract and the descriptive title recorded in
> `Paper/reviews/active_title_abstract.txt`; the alternatives below are kept
> for auditability.

## Scope and editorial position

This review revisits the title and abstract for the active manuscript. It uses
the `src/eviseq_new` implementation and the current evidence contract as its
authority. The four-dataset runs, held-out predictions, and quality and
source-support scores are still pending. The wording below therefore describes
the design and the registered study without implying a performance gain,
hallucination reduction, or completed empirical comparison.

The current title is technically accurate but reads like a grant question. It
also foregrounds two outcomes before the reader knows the interface that is
being tested. A descriptive title is cleaner at the pre-results stage and will
remain valid if the results are mixed.

## Title candidates

### A. Recommended

**EviSeq: Source-Grounded Full-Memory Composition of Pretrained Encoders and Causal Decoders**

This title names the system, the central interface idea, and the composition
setting without promising an improvement. It leaves the task and evaluation
protocol to the abstract, where they can be stated precisely. The title is
slightly method-oriented, which fits a paper whose main contribution is the
interface design.

### B. Task-forward

**EviSeq: A Source-Grounded Full-Memory Interface for Long-Document Summarization**

This is the most readable option for a broad NLP audience. It makes the task
immediately clear, but it hides the fact that the interface joins independently
pretrained encoders and causal decoders.

### C. Composition-forward

**Source-Grounded Full-Memory Interfaces for Composed Encoder--Decoder Summarization**

This title is compact and emphasizes the problem setting. It does not name
EviSeq or make the causal-decoder setting explicit, so it is less distinctive.

### D. Mechanism-forward

**EviSeq: Separating Retrieval Keys from Source Values in Pretrained Summarization Models**

This option highlights the key/value separation that gives the method its
clearest architectural distinction. It may overemphasize one interface choice
unless the paper presents that separation as the primary contribution.

### E. Plain descriptive

**Full-Memory Source Grounding for Pretrained Encoder--Decoder Summarization**

This is easy to scan and carries no result claim. It is less memorable because
it omits the system name and the causal-decoder composition setting.

## Recommended abstract

**Role: complete abstract, organized as task, gap, method, evaluation, and evidence boundary.**

Long-document summarization requires a model to preserve source-supported facts
while producing concise text. This is difficult when the source encoder and
causal decoder come from separate pretrained checkpoints: a width-matching
projection makes their tensors compatible but does not determine which source
information the decoder should use. We present EviSeq, a source-grounded
full-memory interface for this setting. EviSeq retains every visible source
position and separates retrieval keys from aligned final-state values. Its
interface learns bounded adjustments over encoder depth, feature channels, and
source spans, while copied cross-attention and an offset-based copy route keep
source words and numbers available to generation. The system is trained with
gold-token mixture cross-entropy and evaluated with deterministic decoding. We
compare it with fine-tuned decoder-only language models and T5Gemma2 across
PubMed, arXiv, BookSum, and GovReport. The study measures source support with
AlignScore consistency, summary quality with ROUGE and BERTScore, and component
and encoder effects through controlled ablations and encoder substitutions.
Implementation tests validate the executable design; held-out predictions and
quality results are still pending.

## Why this abstract is preferable

- It states the task and the interface gap before introducing the system.
- It keeps the architectural distinction that matters most: retrieval keys can
  adapt while aligned final-state values preserve a direct content route.
- It describes grounded lexical access without suggesting that copying proves
  factual correctness.
- It names the evaluation dimensions without listing metric subtypes, formulas,
  tensor widths, window sizes, gate values, or training-stage details.
- It uses no parenthetical explanation. The only numerical content is implicit
  in the dataset names and model names, and no score is reported.
- It ends with the actual evidence state rather than a generic promise of
  improvement.

## Risk notes and rejected directions

1. **Avoid a result-shaped title before the runs.** Titles containing
   “improves,” “outperforms,” “reduces hallucination,” or a direct ROUGE claim
   would overstate the current evidence. The recommended title remains valid
   under positive, null, or mixed results.
2. **Avoid making “factuality” the headline construct.** The registered
   automatic measure is AlignScore consistency, which is a source-support
   proxy. It is not a calibrated hallucination rate. The abstract uses “source
   support” for the same reason.
3. **Avoid a method inventory in the abstract.** Controller dimensions,
   depth-tap counts, window lengths, gate initialization, the transform
   $H=1-C$, and separate ROUGE variants belong in the method and evaluation
   sections. They interrupt the story without helping a first-time reader
   understand the contribution.
4. **Avoid presenting copied attention as the novelty by itself.** Copying and
   cross-attention are established mechanisms. The defensible contribution is
   their placement within a full-memory interface that separates retrieval
   adaptation from source values for composed pretrained components.
5. **Avoid saying “end to end” without qualification.** The training protocol
   includes an interface warm-up before full fine-tuning. “Trained with
   gold-token mixture cross-entropy” is accurate at the abstract level and
   avoids inviting a schedule objection.
6. **Avoid calling the encoder substitutions a general transfer result.** They
   are registered comparisons whose outcome is pending. “Encoder effects” and
   “encoder substitutions” state the design without claiming portability.

## Compact abstract outline

1. Long-document summarization needs source-supported, concise output.
2. Separate pretrained encoders and causal decoders lack a principled source
   information interface after width matching.
3. EviSeq retains full visible memory and separates retrieval keys from aligned
   source values.
4. Bounded depth, feature, and span adjustments plus grounded lexical access
   support generation under one gold-token objective.
5. Four datasets and matched decoder-only and T5Gemma2 comparisons measure
   source support, overlap, semantic similarity, ablations, and encoder effects.
6. Implementation evidence is available; empirical results remain pending.

## Self-review

- **Clarity:** The first two sentences identify the task, practical difficulty,
  and unresolved interface problem before introducing project terminology.
- **Flow:** Each sentence advances from problem to design to evaluation and then
  limits the conclusion to implementation evidence.
- **Terminology:** `EviSeq`, source-grounded, full-memory, causal decoder,
  retrieval keys, aligned final-state values, AlignScore, ROUGE, and BERTScore
  retain their registered meanings.
- **Unsupported claims:** No score, superiority, hallucination reduction,
  factuality guarantee, or portability result is asserted.
- **Missing evidence:** The abstract states that held-out predictions and
  quality results are pending. A future results revision must replace this
  sentence only after the registered predictions, metric outputs, and paired
  uncertainty are available.
- **Style:** The prose avoids parenthetical asides, formula-heavy detail,
  generic contribution language, and stacked metric qualifiers. Technical
  hyphens remain where they define established terms.

## Claim--evidence map

Claim: EviSeq is a source-grounded full-memory interface for composing a
pretrained source encoder with a causal decoder.  
Evidence: `src/eviseq_new` graph and the active architecture specification.  
Status: supported as an implementation description.

Claim: The interface retains visible source positions and separates retrieval
keys from aligned final-state values.  
Evidence: AFMR tensor contracts and implementation tests.  
Status: supported.

Claim: The grounded copy route keeps source words and numbers available to
generation.  
Evidence: offset alignment, copy mixture, and tensor-contract tests.  
Status: supported as a capability; factual correctness is not implied.

Claim: The study compares EviSeq with fine-tuned decoder-only models and
T5Gemma2 on four datasets using source-support and quality measures.  
Evidence: registered evaluation contract.  
Status: planned.

Claim: EviSeq improves ROUGE or reduces hallucination relative to the
baselines.  
Evidence: held-out predictions and locked metric outputs.  
Status: needs evidence; deliberately omitted from the abstract.

## Integration note

Candidate A is the recommended replacement for the current question-form
title. If adopted, propagate it to `Paper/afmr_question.tex`,
`Paper/afmr_story_spec.md`, the exact title lock in
`Paper/afmr_manuscript_state.json`, and review headers before running the exact
candidate audits. The recommended abstract can replace the active abstract
without changing the scientific contract, provided the method remains tied to
`src/eviseq_new` and the pending-result sentence is preserved until evaluation
finishes.

## Exact-candidate check

The recommended abstract was audited after the last wording change against the
active manuscript terminology and evidence profile. The checked abstract is
stored in the review file between the `Recommended abstract` heading and the
`Why this abstract is preferable` heading. Its SHA-256 is
`6aa8e802ccbabac1b24a01e5ec1d5ae21e621e44e9ee07407109c3078f7e1b37`.
The deterministic audit passed with no findings. The manual checks above are
the remaining scholarly review of claim scope, flow, and reader accessibility.
