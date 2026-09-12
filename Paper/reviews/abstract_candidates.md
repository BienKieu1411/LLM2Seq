# Abstract and title candidates

> **Superseded round (2026-09-13).** Candidate A below was the previous active
> abstract. The current abstract is the shorter, parenthesis-free text in
> `Paper/reviews/active_title_abstract.txt` and `Paper/afmr_question.tex`.

## Scope and authority

These candidates are written for the active manuscript only. The proposed method
is the `src/eviseq_new` graph whose configured architecture is
`afmr_value_anchor`; no other implementation is used to motivate or describe the
method. At the time of this candidate round, the manuscript state locked the selected question-form title below, and the evidence
contract says that the four-dataset predictions and all
quality/factuality scores are still pending. Consequently, every candidate
uses prospective or evidence-bounded language and makes no performance claim.

## Title candidates

1. **Can a Source-Grounded Full-Memory Interface Improve Source Support and ROUGE in Pretrained Encoder--Decoder Compositions?**
   **Recommendation in the earlier round:** adopted for the active manuscript
   at that time. It stated the paper's central empirical question and did not
   presuppose a positive result.

2. **EviSeq: Learning a Source-Grounded Full-Memory Interface for Pretrained Encoder--Decoder Composition**
   A concise descriptive alternative if the question-form title is later
   unlocked. It names the system and the interface while avoiding an unsupported
   superiority claim.

3. **Full-Memory Source Grounding for Composed Encoder--Decoder Summarization**
   A shorter method-oriented alternative. It is less explicit about the
   factuality and ROUGE evaluation questions than the selected title.

The alternatives were recorded for discussion only; the first title was the
active lock after the earlier coordinator review and has since been superseded.

## Candidate A — adopted

Pretrained encoders and causal decoders are reusable building blocks, yet a
width-matching projection alone does not decide which source evidence should
reach the decoder. We present EviSeq, a source-grounded encoder--decoder
system with the Adaptive Full-Memory Residual (AFMR) interface. AFMR preserves
every visible source position in a retrieval tensor and pairs it with an
aligned final-state value tensor. A controller applies bounded adjustments
across encoder depth, feature channels, and overlapping multi-scale windows;
the value anchor maintains a direct content path while refined keys guide
retrieval. Copied cross-attention in every decoder layer and an offset-aligned
copy head retain access to source words and numbers. The graph is trained end
to end with gold-token mixture cross-entropy, and the primary comparison uses
deterministic decoding. We register four datasets---PubMed, arXiv, BookSum, and
GovReport---and compare with fine-tuned decoder-only LLMs and T5Gemma2. We
separate source support (AlignScore consistency and H=1-C), reference overlap
(ROUGE-1/2/L), semantic similarity (BERTScore-F1), component contribution, and
encoder portability. Implementation tests pass on controlled fixtures, but
held-out predictions and quality scores are pending; no improvement or
hallucination-reduction claim is made.

**Why this is adopted.** It follows the requested problem--method--
evaluation--evidence-boundary order, names the full-memory/key--value
distinction that defines AFMR, and states the training and comparison controls
without turning planned analyses into findings.

## Candidate B — evaluation-forward

Composing independently pretrained encoders and causal decoders offers a
practical route to sequence-to-sequence summarization, yet a simple projection
does not control which source evidence reaches the decoder. EviSeq addresses
this interface problem with Adaptive Full-Memory Residual (AFMR), which keeps
all visible source positions and separates retrieval adaptation from final-state
content values. Its controller selects bounded adjustments across encoder depth,
feature channels, and overlapping source windows. A copied cross-attention
block in each decoder layer reads the resulting source memory, while a grounded
copy route preserves access to aligned source words and numbers. EviSeq is
trained end to end with the gold-token mixture cross-entropy objective; the
primary comparison uses deterministic decoding and does not generate auxiliary
candidates. We register four research questions over PubMed, arXiv, BookSum,
and GovReport. Fine-tuned decoder-only language models provide the main
comparison, and T5Gemma2 supplies an encoder--decoder reference. We will measure
AlignScore consistency for source support, ROUGE-1/2/L and BERTScore-F1 for
summary quality, and matched ablations and encoder replacements for mechanism
and portability. The implementation and tensor-contract tests currently
support only the executable design; test predictions, scores, and any claim of
improved factuality or overlap remain pending.

**Strength.** This version foregrounds the comparison design and makes the
distinction between source support and summary quality easy to recover.

## Candidate C — method-forward

When a pretrained encoder and a causal decoder were trained independently,
their interface must allocate source information without discarding lexical
evidence. EviSeq introduces AFMR, a constrained full-memory bridge for this
setting. The bridge uses the encoder's final state as an anchor, combines depth
taps at each source position with a low-rank correction across encoder and
decoder spaces, and derives a bounded source prior from overlapping windows at
multiple scales. The decoder receives
the complete visible memory through copied cross-attention in every layer:
refined representations guide retrieval, while aligned final-state values
carry source content. A separate grounded copy head aligns source and decoder
token offsets and marginalizes repeated source token IDs. These routes are
optimized jointly by one gold-token mixture cross-entropy loss and are evaluated
with deterministic decoding. The planned study uses PubMed, arXiv, BookSum,
and GovReport, comparing EviSeq with fine-tuned decoder-only controls and
T5Gemma2. ROUGE-1/2/L and BERTScore-F1 assess overlap and semantic similarity;
AlignScore consistency provides the primary automatic source-support signal;
matched component ablations and encoder substitutions test the remaining
questions. The current draft reports implementation evidence only. Four-dataset
predictions and all empirical values are pending, so the design is presented as
a falsifiable hypothesis rather than a demonstrated improvement.

**Strength.** This version gives the clearest compact account of the tensor
paths and is suitable when the method section is the paper's main contribution.

## Selection and exact-candidate checks

Candidate A was adopted as the abstract for the active manuscript in the
earlier review round. It is approximately 180
words under a whitespace count (excluding its heading), within the requested
150--200-word range. Candidates B and C are retained as alternatives for a
later editorial choice; their word counts should be rechecked if edited.

The candidate set was checked against the active story specification and
manuscript state for:

- the `src/eviseq_new` authority and `afmr_value_anchor` graph;
- the locked AFMR/EviSeq terminology;
- the four datasets and the registered decoder-only/T5Gemma2 comparisons;
- the distinction between AlignScore consistency, ROUGE overlap,
  BERTScore semantic similarity, and the direction transform `H=1-C`;
- the gold-token cross-entropy-only training boundary; and
- the absence of invented scores, completed runs, or superiority claims.

No main-manuscript file was modified by this task.
