# Title and abstract integration review

REVIEW PROFILE: general scientific review with AI/LLM-computational and display/notation checks
PAPER: *Can a Source-Grounded Full-Memory Interface Improve Source Support and ROUGE in Pretrained Encoder--Decoder Compositions?*
MODE: authorized revision and integration
STAGE OR ROUND: integration
MODULES: academic-writing abstract/evidence gates; paper-review AI/LLM-computational checks; display/notation/provenance checks
SOURCE BASIS: `src/eviseq_new`, `Paper/afmr_story_spec.md`, `Paper/afmr_manuscript_state.json`, the active LaTeX manuscript, and the two agent candidate files. No quality run was used as evidence.
READINESS: title and abstract are integrated; empirical release remains blocked until the registered four-dataset evidence exists.

## Priority-ranked action items

1. **Title scope — resolved, S2, verified.** The title is a question and names the
   operational source-support construct alongside ROUGE. It does not state a
   measured gain or treat AlignScore as a hallucination probability. The exact
   title is synchronized in the manuscript, story specification, and state lock.

2. **Abstract evidence boundary — resolved, S2, verified.** The abstract follows
   problem, gap, method, evaluation, and evidence status. It states that held-out
   predictions and quality scores are pending, so the text makes no superiority
   or hallucination-reduction claim.

3. **Method fidelity — resolved, S2, verified.** The abstract names the full
   retrieval tensor, aligned final-state value tensor, bounded controller paths,
   copied cross-attention, grounded copy route, and gold-token mixture
   cross-entropy. These claims are implementation properties documented by
   `src/eviseq_new`; no auxiliary training objective or self-improvement step is
   introduced.

4. **Metric interpretation — resolved, S2, verified.** Source support is tied to
   native AlignScore consistency and the direction transform `H=1-C`; ROUGE is
   reference overlap and BERTScore is semantic similarity. The abstract does not
   call BERTScore a hallucination detector or call `H` a calibrated probability.

5. **Open empirical gate — S4, confirmed, blocking the empirical release only.**
   The abstract cannot be converted to a results claim until the four dataset
   manifests, scorer-locked predictions, paired uncertainty, and provenance
   records are available. Score cells remain blank by design.

## Writing and presentation flags

- `\newcommand` macros are used for repeated system, architecture, model,
  dataset, and metric names; `\xspace` preserves word boundaries in compiled
  prose.
- Candidate A is approximately 180 words before LaTeX macro expansion and fits the current
  abstract budget. Its final source is the exact abstract in
  `Paper/afmr_question.tex`.
- The title's “source support” wording narrows RQ1 to its registered automatic
  construct. Any later title change is a Class A change and must propagate to
  the state lock, story specification, abstract, review headers, and conclusion.

## Checks performed

- AgentHub produced independent title and abstract candidates; the coordinator
  selected the source-support title and Candidate A.
- Candidate text and terminology checks were rerun after integration.
- LaTeX source structure, bibliography references, and the full manuscript were
  compiled with Tectonic; representative pages were visually inspected.
- The local model-free test suite remains green (107 passed, 2 warnings).

## Functional-completeness retrospective

The requested lifecycle stage (title/abstract drafting and review) and companion
artifacts were covered. The active authority is only `src/eviseq_new`; locked
meaning and score placeholders were preserved. The title change was propagated
through the manuscript, story, state, and review headers. No quality evidence was
invented. The remaining state-audit blockers are empirical and reproducibility
gates, not unresolved title or abstract defects.
