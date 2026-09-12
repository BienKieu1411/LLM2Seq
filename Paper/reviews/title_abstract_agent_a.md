# Title and abstract alternatives

> **Candidate record only.** The coordinator later selected a shorter
> parenthesis-free abstract and the descriptive title recorded in
> `Paper/reviews/active_title_abstract.txt`; the alternatives below are kept
> for auditability.

## Scope and writing contract

These alternatives revise the paper's public framing only. The active method is
the `src/eviseq_new` graph configured as `afmr_value_anchor`. The current
manuscript has implementation and fixture-test evidence, but no completed
four-dataset predictions or quality measurements. The candidates therefore
describe the design and registered evaluation without claiming a ROUGE gain,
better factual support, or reduced hallucination.

The abstracts use a problem, gap, method, and evaluation sequence. They avoid
parenthetical asides, formulas, score values, and a long list of dataset names.
The technical terms that remain are load-bearing: full-memory source access,
separate retrieval and value paths, cross-attention, grounded copying, and the
gold-token training objective.

## Title candidates

### Candidate 1 — recommended

**EviSeq: Source-Grounded Summarization with Composed Pretrained Encoders and Causal Decoders**

This title is the clearest balance between method, task, and setting. It names
the system, explains what the source-grounded design is used for, and makes the
checkpoint-composition problem visible. It does not name a metric or imply an
empirical win.

### Candidate 2 — interface-focused

**Learning the Interface Between Pretrained Encoders and Causal Decoders for Source-Grounded Summarization**

This version foregrounds the paper's central argument: the interface is the
design problem left open by independent pretraining. It reads naturally for a
method paper and remains valid if the measured effects are mixed.

### Candidate 3 — architecture-focused

**Full-Memory Source Access for Summarization with Independently Pretrained Encoders and Causal Decoders**

This version highlights the main architectural commitment without relying on an
acronym. It is descriptive and cautious, although it gives less visibility to
the EviSeq name and the adaptive residual controller.

## Abstract candidate 1 — recommended

**Role: problem, interface gap, method, evaluation, evidence boundary.**

Pretrained encoders and causal decoders offer a practical way to build
summarization systems from reusable components. Yet matching their hidden widths
does not determine which source information should guide generation, especially
when the two models were trained independently. We introduce EviSeq, a
source-grounded encoder--decoder system built around an Adaptive Full-Memory
Residual interface. The interface keeps the complete visible source memory
available to the decoder while learning bounded changes in retrieval priority.
It separates retrieval features from final-state values, and combines
layer-wise cross-attention with a grounded copy route so that contextual
information and source wording remain accessible during generation. EviSeq is
trained end to end with token-level likelihood and does not require evidence
labels, auxiliary candidate generation, or a second training objective. We
evaluate it against fine-tuned decoder-only language models and a pretrained
encoder--decoder reference on long-document summarization tasks. The evaluation
covers overlap, semantic similarity, source consistency, component contribution,
and encoder portability. The empirical comparison is still pending.

**Why it works:** The first three sentences establish the practical setting and
the unresolved interface problem. The middle of the abstract gives one compact
description of the design and its intended advantage. The final sentences state
the comparison and evidence boundary without turning planned metrics into
results.

## Abstract candidate 2 — more narrative

**Role: task, composition challenge, design insight, evaluation, evidence boundary.**

Long-document summarization asks a model to shorten a source without losing the
entities, numbers, and relations that make it useful. The problem becomes harder
when an encoder and a causal decoder come from separate pretrained checkpoints.
A projection can align their widths, but it cannot by itself decide how source
representations should be selected, combined, and exposed to the generator.
EviSeq treats this connection as a learned interface. Its Adaptive Full-Memory
Residual bridge retains all visible source positions, adapts retrieval using
information from the encoder and decoding budget, and keeps final-state values
as a direct content path. Each decoder layer can attend to that memory, while a
grounded copy route preserves access to source tokens and numbers. The model is
optimized with teacher-forced token likelihood under a generation protocol. We
compare EviSeq with fine-tuned decoder-only models and an encoder--decoder
reference across long-document domains. We report source consistency alongside
reference overlap and semantic similarity, then use ablations and encoder
substitutions to examine interface behavior. Empirical results are pending.

**Why it works:** This version starts from the reader's document-level concern
and then narrows to the checkpoint-composition gap. It explains the full-memory
and value-path choices with less implementation inventory than Candidate 1. It
is a good fallback if the paper wants the abstract to sound less like a system
specification.

## Selection recommendation

Use Title Candidate 1 with Abstract Candidate 1 for the next manuscript pass.
Together they give the paper a recognizable name, a concrete task, and a clear
reason for the architecture without promising an outcome that has not been
measured. Use Title Candidate 2 with Abstract Candidate 2 if the authors want a
more problem-led framing and a lighter method description.

Do not put ROUGE values, AlignScore transformations, or parenthetical metric
definitions in the abstract while the evidence is pending. Those details belong
in the evaluation section. Once the runs finish, the abstract should be rebuilt
from the observed directions and uncertainty rather than edited by inserting
numbers into the current prose.

## Self-review

- **Clarity:** Each candidate states the task or setting before introducing the
  interface and uses direct subjects and verbs.
- **Flow:** Both abstracts move from the composition gap to the design response,
  then to the registered comparison and its current status.
- **Terminology:** `EviSeq`, `Adaptive Full-Memory Residual`, encoder--decoder,
  decoder-only, grounded copy, and source consistency retain their project
  meanings.
- **Claim scope:** The candidates describe implementation properties and planned
  analyses. They do not claim superiority, factuality improvement, or
  hallucination reduction.
- **Style:** The abstract text contains no parenthetical asides, formulas, or
  metric-score inventory. Technical compounds use standard hyphenation where
  needed for precision.
- **Evidence gap:** Held-out predictions, scorer-locked values, paired
  uncertainty, and completed ablations remain necessary before any result claim
  can be added.

## Claim--evidence map

| Claim | Evidence source | Status |
|---|---|---|
| Independent checkpoint composition leaves source-information routing underspecified after width matching. | `Paper/afmr_story_spec.md` central argument and `src/eviseq_new` interface design. | Supported as the paper's motivating problem. |
| EviSeq retains visible source memory, separates retrieval features from final-state values, and provides cross-attention and grounded-copy paths. | `src/eviseq_new` implementation and controlled tensor-contract tests. | Supported as an implementation claim. |
| The system is trained with one gold-token generation objective and does not use auxiliary candidate generation. | `Paper/afmr_story_spec.md` method facts and training contract. | Supported as a design claim. |
| The evaluation compares fine-tuned decoder-only models and an encoder--decoder reference using overlap, semantic similarity, source consistency, ablations, and encoder substitutions. | `Paper/afmr_story_spec.md` research questions and experimental contract. | Planned; wording remains prospective. |
| EviSeq improves ROUGE or reduces hallucination. | No completed held-out predictions or quality measurements are available. | Needs evidence; intentionally not claimed. |

## Candidate checks

The two abstracts are each about 160 words, within the requested concise range.
They were read against the active story specification and manuscript state. No
main manuscript, state file, bibliography, or implementation file was edited by
this worker.
