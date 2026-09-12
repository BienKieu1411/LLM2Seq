# Prose-consistency review

REVIEW PROFILE: General scientific review with the academic-writing, writing-skill, AI/LLM computational, and display/notation/provenance overlays

PAPER: *Can a Source-Grounded Full-Memory Interface Improve Source Support and ROUGE in Pretrained Encoder--Decoder Compositions?*

MODE: Review only; no manuscript or story-spec edits authorized

STAGE OR ROUND: Developmental/integration review of the exploratory internal draft; no prior review round supplied

MODULES: `paper-review` with the `academic-writing-skills` integrity base; `writing-skill` for reverse outlining and claim–evidence prose; the AI/LLM computational module for proxy, transfer, and ablation wording; and the display/notation/provenance module for equations and derived scores.

SOURCE BASIS: `Paper/afmr_question.tex`, its three `\input` sections under `Paper/drafts/`, and `Paper/afmr_story_spec.md`. `Paper/afmr_question.bib` was read only to reconcile citation keys. The included draft files are treated as part of the active manuscript because `afmr_question.tex` loads them. No completed quality runs or venue-specific language rules were supplied.

READINESS: The narrative is conditionally suitable for another internal prose pass, but it is not ready for submission or empirical RQ1–RQ4 answers. The main blockers are the RQ1 construct/evidence boundary, causal-sounding RQ3/RQ4 bridge language, incomplete equation definitions, and the remaining macro/layout cleanup.

The four questions appear in the same order in the Introduction, Experiments, Results templates, Discussion, and story specification. The manuscript also keeps quality cells pending and explicitly distinguishes implementation checks from generation evidence. Those are strong structural choices. The issues below preserve those choices while tightening scope and reader-facing terminology.

## Priority-ranked action items

### PC-01 — “Factuality” and typed-error wording exceed the direct RQ1 evidence

Location: `Paper/afmr_question.tex:57,69`; `Paper/drafts/introduction_related_work.tex:28,35`; `Paper/drafts/results_discussion.tex:65-81`; `Paper/afmr_story_spec.md:13,33,61,65-71`.

Classification: scope = research-question/epistemic; severity = **S4**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the locked RQ1 estimand plus either a completed typed factual-error audit or an explicit proxy-only decision; affected artifacts = manuscript, story specification, Results templates, Discussion, and Conclusion; blocking stage = empirical RQ answers and submission.

The title and RQ1 ask about factuality and reductions in unsupported entities, numbers, and relations. The evidence contract retains native AlignScore consistency, which the manuscript itself describes as a learned proxy; the fill-in sentence also says that the value does not establish a reduction in factual errors. The optional phrase “where available” for a manual audit cannot support the typed-error wording when no such audit is available.

Choose one contract and propagate it through the locked files. A proxy-only version can use `To what extent does \EviSeq change native \AlignScore consistency ...?` and, if the title lock is reopened, `... improve source--summary consistency and \Rouge ...?`. If the current factuality wording is retained, make the entity/number/relation audit a required RQ1 output with its sampling, adjudication, denominator, and paired-comparison rules. Until then, keep `we test whether` and avoid “factuality improvement” or “fewer factual errors” in fill-in text.

### PC-02 — The RQ1→RQ4 bridge contains stronger relations than the measures and designs support

Location: `Paper/drafts/introduction_related_work.tex:30-35`; `Paper/afmr_story_spec.md:35-38`; qualifications in `Paper/drafts/results_discussion.tex:166-188`.

Classification: scope = main argument/cross-file; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the story-spec question and transition locks; affected artifacts = Introduction, story specification, and RQ result templates; blocking stage = integration of the next draft.

The bridge says RQ2 tests that a gain is not obtained by sacrificing “summary coverage or lexical quality,” RQ3 “attributes an aggregate effect,” and RQ4 tests whether “that mechanism survives” while asking how directionality and capacity “affect” quality. ROUGE/BERTScore measure reference overlap and semantic similarity, the proposed ablations support conditional component contribution, and the selected encoders confound directionality with family, tokenization, pretraining, width, and capacity. The later Discussion correctly states these limits, so the Introduction/spec bridge should not be stronger than it.

Use a content-bearing bridge such as: `RQ1 tests source-support consistency; RQ2 tests whether any source-support difference is accompanied by a change in reference overlap or semantic similarity; RQ3 tests which routes contribute to the observed benchmark behavior under the matched graph; and RQ4 tests whether that behavior persists across the registered encoder choices, whose directionality and capacity differ.` Qualify RQ3 as “needed under the matched protocol” or use “contribute,” and describe RQ4 as transfer across tested choices unless a factor-separated design is added.

### PC-03 — The protected macro contract is not applied consistently

Location: macro definitions at `Paper/afmr_question.tex:26-55`; raw or mixed uses at `Paper/afmr_question.tex:57,69,77,79,100,111`, `Paper/drafts/method_experiments.tex:27,92,180,187,245-258,277`, and `Paper/drafts/results_discussion.tex:70,78,156,217`.

Classification: scope = cross-file consistency/notation; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the macro rules in the story specification; affected artifacts = manuscript and story specification; blocking stage = exact-candidate release check.

The specification requires repeated architecture, selected-model, dataset, and metric names to use the preamble commands. The manuscript still writes `ROUGE` directly in the title, abstract, and metric Methods; writes `Adaptive Full-Memory Residual (AFMR)` instead of using `\AFMRFull`; and uses raw `AFMR`, `AlignScore`, and `BERTScore` in headings, prose, and displays. The same derived metric appears as `$1-\AlignScore$`, `$1-\text{AlignScore}$`, and `$1-\mathrm{AlignScore}$`; the first form expands to italic math letters rather than a named metric. `\verb|\\newcommand|` in the reproducibility paragraph also displays two backslashes.

Use `\Rouge`, `\AFMRFull{} (\AFMR)`, `\AFMR{}`, `\AlignScore`, and `\BERTScore` for the corresponding text occurrences. Define a math-only form such as `\newcommand{\AlignScoreMath}{\mathrm{AlignScore}}` (or use one consistent `\mathrm{...}` form) for `$1-...$`. In the gradient display use `\text{\AFMR}`. Correct the documentation to `\verb|\newcommand|`, then scan the exact compiled manuscript for raw selected names. Prior-work names such as T5, BART, and LLM2Vec can remain literal unless the project extends the macro registry to them.

### PC-04 — The source-prior and copy equations need a complete symbol/provenance ledger

Location: `Paper/drafts/method_experiments.tex:91-114,124-156`; related method facts in `Paper/afmr_story_spec.md:42-50,79`.

Classification: scope = method notation/reproducibility; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the architecture implementation or an approved formal definition; affected artifacts = Method, story specification, captions, and any later tables; blocking stage = reproducibility and notation release check.

The displayed source prior uses `s_i` in `r_i=\lambda(c)\tanh(s_i/\tau(c))` without defining the window score, overlap-add weights, scale mixing, or behavior when a scale is unavailable. `M` is passed to `\operatorname{MHA}` without being defined as a mask, `P` has no stated shape, and the denominator/empty-mask behavior of `\MaskedMean` is implicit. In the copy equation, `\operatorname{tok}_j` and the index set for eligible aligned source spans are ambiguous under two tokenizers, and `h_t` is not explicitly tied to the decoder state used by the vocabulary head. These gaps matter because all are on the retrieval/copy path rather than decorative symbols.

Define each quantity immediately before first use: the per-window score and its scale/overlap aggregation for `s_i`, `M` and its masking semantics, `P` and `M_p` dimensions, `\mathcal C=\{i:C_i=1\}`, the mapping from an eligible source span `j` to a decoder vocabulary ID, and the exact decoder state `h_t`. Replace “keeps the mean prior at a stable reference” with `sets the log-mean-exp of the content prior to zero` (equivalently, the mean of `\exp b_i` is one); log-mean-exp centering does not make the arithmetic mean of `b_i` zero.

### PC-05 — The abstract turns a copy route into an unmeasured preservation claim

Location: `Paper/afmr_question.tex:69`; compare the safer wording in `Paper/drafts/introduction_related_work.tex:19`, `Paper/drafts/results_discussion.tex:145-148`, and `Paper/afmr_story_spec.md:67-69`.

Classification: scope = claim/evidence; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = held-out copy/number evidence or the implementation-only claim contract; affected artifacts = Abstract, Introduction, Results, and Conclusion; blocking stage = summary-claim release.

“A grounded copy head preserves source lexical forms and numbers” describes an output behavior that is not available before the locked runs. The available contract supports that the implementation provides an aligned copy route. Replace it with `provides a route for copying source lexical forms and numbers` or `aligns source lexical forms and numbers for copying`; reserve “preserves” for a reported held-out diagnostic with a defined denominator.

### PC-06 — Pending-score notation is clear to a reader but inconsistent as a filling workflow

Location: `Paper/afmr_question.tex:69,111,117`; `Paper/drafts/results_discussion.tex:25-47,74-139`; `Paper/afmr_story_spec.md:8,29,70-73,81`.

Classification: scope = delivery/provenance; severity = **S2**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the score-filling convention; affected artifacts = Results, appendix, abstract, and story specification; blocking stage = score insertion/release audit.

The `--` cells, caption, and Results prose clearly say “not yet computed,” not zero or an imputation, and the fill-in templates visibly use `\tbd{...}`. The appendix checklist says only “Before replacing any `\tbd{}` cell,” although the actual result table uses `--`; the caption says score fields are “absent” even though a visible marker is present. The abstract’s “all quality values” can also be read as excluding AlignScore factual-consistency values.

Use one operational sentence, for example: `Before filling any numeric score cell or \tbd{} placeholder, verify ...`; call the table values `numeric score fields are intentionally unfilled`; and use `all quantitative outcome values` or name ROUGE, BERTScore, and AlignScore explicitly. Retain the explicit `--` explanation and never replace a failed or omitted run with zero.

### PC-07 — Citation-key resolution passes, but dataset attribution and duplicate-key hygiene remain open

Location: dataset descriptions at `Paper/drafts/method_experiments.tex:224-231`; cited literature at `Paper/drafts/introduction_related_work.tex:10,13,16,45,49,53,57`; bibliography reconciliation against `Paper/afmr_question.bib`.

Classification: scope = citation/reproducibility; severity = **S2**; evidence = **CONFIRMED** for the current key inventory; status = **OPEN**; authority needed = canonical dataset releases and bibliography ownership; affected artifacts = manuscript and bibliography; blocking stage = submission reference check.

All 18 distinct in-text citation keys resolve to entries in `afmr_question.bib`, and no undefined citation warning appears in the Tectonic build. The bibliography contains duplicate AlignScore records, `zha2023alignscore` and `zha-etal-2023-alignscore`, while only the latter is cited. The dataset paragraph makes corpus-composition claims without dataset citations; `kryscinski2022booksum` is present but unused, and no corresponding PubMed, arXiv, or GovReport dataset keys appear in the reviewed bibliography.

Choose one canonical AlignScore key and remove or mark the duplicate. Add authoritative dataset citations (or explicit manifest/release authority) for each corpus description before submission. Do not add a citation merely to silence a warning; tie each source to the exact dataset, split, or task-variant claim it supports.

## Section-linked comments and presentation flags

The reverse outline is functional: Introduction moves from task distinction → interface gap → AFMR idea → evidence boundary → RQ1–RQ4; Related Work moves from composed interfaces → long-document access → copying → metrics → gap; Method moves from setting → controller/residuals/prior → cross-attention/copy → loss and inference; Experiments follows comparisons → data → metrics → fairness; Results and Discussion preserve RQ order before limitations and the evidence-bounded Conclusion. The paragraph-role comments in the included drafts make this structure easy to audit.

The Introduction architecture paragraph (`introduction_related_work.tex:19`) carries a full route inventory in one dense paragraph. Split it after the three-axis summary or compress the inventory so the paragraph states one overview claim before decoder/copy detail. The bridge paragraph at line 35 also performs two functions, RQ transitions and metric definitions; split it after the RQ chain. In Results, some repetition is necessary because the evidence boundary belongs in Results, Discussion, Limitations, and Conclusion, but replace vague frames such as “The current result is an implementation boundary,” “ROUGE and BERTScore answer a different question,” “acceptable summary coverage,” and “Several negative-space checks govern...” with concrete subjects and measures. For example: `At this stage, the evidence is limited to implementation properties`; `ROUGE and BERTScore measure reference overlap and semantic similarity, respectively`; and `The final report must link each score row to its prediction manifest and paired uncertainty.`

The prose audit found no exact duplicated sentence. It flagged repeated fill-in-template openings and dense lexical hyphenation; those are mostly intentional template and technical terms (`decoder-only`, `cross-attention`, `controller-conditioned`, and `source-token`). Keep necessary technical compounds, but avoid rotating `factuality`, `source support`, `source consistency`, and `AlignScore consistency` without defining their relationship. “Observed behavior,” “that mechanism,” and “a different question” should identify the metric, component, or benchmark condition locally.

Tectonic compiles the current active manuscript and runs BibTeX successfully, with no fatal TeX error or unresolved citation/reference reported. The build still emits many underfull boxes and overfull boxes in the long equations at `method_experiments.tex:37,47,58,78,102,110,129,153,181` and in the long appendix caption at `afmr_question.tex:111`. Use `aligned` rows or concise surrounding prose where the overflow is visible, then perform a visual PDF check; the warnings alone do not establish whether the rendered page is acceptable. No visual check was performed in this review.

## Next-step instruction

First settle the locked RQ1 construct and the RQ3/RQ4 transition wording, then propagate the decision through the title (if the lock is reopened), abstract, RQs, templates, Discussion, Conclusion, and story specification. Complete the symbol/provenance definitions, canonicalize selected-name macros and math metric notation, align the score-filling checklist with both `--` and `\tbd{}`, and add dataset citations or manifest authority. Recompile and visually inspect the exact active manuscript after those edits; keep every score pending until the matched runs and provenance checks exist.

## Functional-completeness retrospective

Scope covered: review-only prose, argument flow, RQ1–RQ4 transitions, terminology/macros, equation notation, citation keys, LaTeX structure, and pending-score wording across `afmr_question.tex`, its included manuscript sections, and `afmr_story_spec.md`.

Authority and locks: the story specification’s title, names, evidence boundary, question chain, and score policy were consulted; no locked wording was changed. The RQ1 proxy decision and any title change remain author decisions.

Alignment: question order and paragraph functions are aligned; RQ1’s typed-factuality wording, RQ3’s necessity/attribution wording, and RQ4’s factor-effect wording remain affected links.

Adapters and overlays: paper-review, academic-writing, writing-skill, AI/LLM computational, display/notation/provenance, and red-team/release checks were applied. No venue rule, named reviewer overlay, or prior-round evidence was available.

Change propagation: no manuscript or story-spec file was edited. The requested review memo is the only artifact written; any future wording change must be propagated across the title, abstract, RQs, templates, summaries, equations, and specification.

Deterministic and visual checks: citation-key and label/reference inventories, the prose-pattern audit, and a Tectonic/BibTeX compile were run. The prose audit’s template-opening and technical-hyphen findings were inspected in context. No visual PDF check or completed quality-display check was run.

Open issues and unknowns: **1 S4**, **4 S3**, **2 S2**, and **2 S1** issues remain open. The absent matched quality evidence and pending BookSum/GovReport preparation are confirmed by the story specification; exact implementation definitions for the incomplete symbols and the appropriate dataset citation authority remain to be supplied.

Readiness: **not ready** for empirical RQ answers or `SUBMISSION_READY`; conditionally ready for an internal prose/notation integration pass after the S4 and S3 issues are resolved.
