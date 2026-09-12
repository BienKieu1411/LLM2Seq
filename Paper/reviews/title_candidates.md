# Title candidates

## Scope and evidence boundary

These candidates are for the active manuscript only. They describe the proposed
system implemented in `src/eviseq_new`: an `Adaptive Full-Memory Residual`
(`AFMR`) interface within `EviSeq`, which composes a pretrained source encoder
with a causal decoder. The title must remain a question because the
four-dataset quality and source-support results are not available yet. It must
also avoid treating `AlignScore` as a direct hallucination rate or implying a
measured improvement before the registered runs.

## Candidates

1. **Can a Source-Grounded Full-Memory Interface Improve Source Support and ROUGE in Pretrained Encoder--Decoder Compositions?**

   Recommended. It preserves the question form and the paper's central
   interface contribution, while replacing the broader term “factuality” with
   the narrower source-support construct that the current evaluation can
   measure with `AlignScore`.

2. **Can an Adaptive Full-Memory Interface Improve Long-Document Summarization with Composed Pretrained Models?**

   Concise and method-centered. It names the adaptive interface and the task,
   but leaves the factual-support and overlap outcomes implicit.

3. **Can Adaptive Full-Memory Residuals Bridge Pretrained Encoders and Causal Decoders for Source-Grounded Summarization?**

   Most technically explicit. It identifies `AFMR`'s role as an interface and
   the composed model types, but it is less direct about the planned empirical
   comparison.

4. **Can Full-Memory Source Access Improve Quality and Source Support in Composed Encoder--Decoder Summarization?**

   Emphasizes the retained source memory and keeps both evaluation dimensions
   at a defensible level. “Quality” is intentionally broad and therefore less
   searchable than “ROUGE.”

5. **How Does an Adaptive Full-Memory Interface Shape Source-Grounded Generation from Pretrained Components?**

   Safest if the paper later reports mixed results. It frames the study as an
   investigation of behavior, but it does not foreground summarization or the
   primary metrics.

6. **Can Separating Retrieval Keys from Value Anchors Improve Source-Grounded Summarization?**

   Highlights the key/value design distinction that is specific to the active
   implementation. It should be used only if the method section keeps this
   separation as the main explanatory contribution rather than one component
   among several AFMR routes.

7. **Does an Adaptive Full-Memory Interface Transfer Across Encoders for Source-Grounded Summarization?**

   Best aligned with the portability question, but too narrow for the complete
   paper because the main study also evaluates source support, ROUGE, and
   component contribution.

8. **Can a Source-Grounded Interface Preserve Lexical Evidence in Long-Document Summarization with Pretrained Components?**

   Connects the grounded-copy route to the source-coverage motivation without
   claiming that copying guarantees factual correctness. It underrepresents
   the full-memory and multi-axis AFMR design.

## Selection recommendation

Use candidate **1** if the title is being rebuilt before results are filled:

> **Can a Source-Grounded Full-Memory Interface Improve Source Support and ROUGE in Pretrained Encoder--Decoder Compositions?**

This title gives the reader the task-level problem, the interface-level
contribution, and the two headline outcome families. “Source support” matches
the locked RQ1 construct and avoids claiming a calibrated hallucination
measure. The modal question (“Can”) keeps the result conditional. “Pretrained
Encoder--Decoder Compositions” identifies the setting directly without naming a
model family that may change in the registered comparison.

The prior locked title,
“Can a Source-Grounded Full-Memory Interface Improve Factuality and ROUGE When
Pretrained Models Are Composed?” was also defensible as a working title. It
was replaced by candidate 1 after coordinator review; the active abstract and
Results define source support through the narrower `AlignScore` signal and do
not turn the title into evidence for a hallucination-reduction claim.

## Screening criteria applied

- **Architecture fidelity:** every candidate refers to the full-memory,
  source-grounded interface implemented by `src/eviseq_new`; none introduces a
  loss, reranker, self-improvement stage, or unimplemented module.
- **Claim scope:** no candidate states that the system wins, reduces
  hallucinations, or improves a metric as an established result.
- **Evaluation alignment:** candidates 1 and 4 use “source support,” matching
  the planned native `AlignScore` consistency signal; ROUGE remains a
  reference-overlap outcome rather than a factuality measure.
- **Reader function:** the title should identify the problem setting and the
  contribution before the reader reaches the method section.
- **Cross-artifact stability:** the selected title is synchronized in
  `afmr_question.tex`, `afmr_story_spec.md`, and the exact title lock in
  `afmr_manuscript_state.json`.

## Functional-completeness note

This bounded title review covered the title lock, story specification, active
manuscript preamble/abstract, and the current method/evaluation framing. The
selected title and abstract are now integrated in the active manuscript; the
equations, experiments, and score tables remain unchanged.
