# Evaluation-fairness review

REVIEW PROFILE: General scientific review with the AI/LLM computational and display/notation/provenance overlays

PAPER: *EviSeq: Source-Grounded Summarization with Composed Pretrained Encoders and Causal Decoders*

MODE: Review only; no manuscript or evaluator changes authorized

STAGE OR ROUND: Integration review of an exploratory/complete internal draft pending empirical completion

MODULES: `paper-review` with the `academic-writing-skills` integrity base; the AI/LLM computational module for model, prompt, sampling, provenance, and evaluation claims; the display/notation/provenance module for metric definitions and derived scores; and the red-team/release checks for negative space and dimension completeness.

SOURCE BASIS: `Paper/afmr_question.tex`, `Paper/afmr_story_spec.md`, `Technical_Report/FACTUALITY_EVALUATION.md`, and the evaluator/tests under `src/rouge155`. No prior review round, response letter, venue rule, or completed EviSeq run was supplied. The existing `src/rouge155/results` files are old LRSum/WikiLingua ROUGE artifacts and are not evidence for the four-dataset EviSeq contract.

READINESS: Not ready for a quality claim, an RQ1--RQ4 answer, or submission. The primary bottlenecks are the absent matched runs and BookSum/GovReport artifacts, the RQ1 construct gap, and an uncertainty/provenance protocol that currently covers only a subset of the promised metrics.

## Superseding title and abstract pass

The title and abstract were revised on 2026-09-13 after this review. The
current title is descriptive, and the current abstract removes formulas,
score values, and parenthetical metric definitions while retaining the
pre-results evidence boundary. Findings below that refer to the former
question-form title are historical.

The draft has several sound safeguards. It keeps all quality values pending, states that AlignScore is a proxy rather than a calibrated hallucination probability, distinguishes native AlignScore consistency from `1-C`, and labels BERTScore as semantic similarity. The local-only model loading and JSON output metadata are useful foundations. Those safeguards do not yet establish that the planned comparisons are fair or reproducible.

## Priority-ranked action items

### EVAL-01 — The locked four-dataset evidence set is absent

Location: `Paper/afmr_question.tex:69,77,115-117`; `Paper/afmr_story_spec.md:8-9,53-61,65-73`; `src/rouge155/results/`.

Classification: scope = validity/project; severity = **S4**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the completed experiment manifests, predictions, checkpoints, and scorer artifacts; affected gates = Results, cross-artifact integration, and release.

The story specification explicitly says that the four-dataset fine-tuning, predictions, and metric results are unavailable. The manuscript says that only PubMed and arXiv pipelines are prepared and that BookSum and GovReport manifests, recipes, and split-specific truncation statistics remain to be supplied. The scoped results directory contains only older LRSum/WikiLingua files; it contains no matched EviSeq outputs for PubMed, arXiv, BookSum, or GovReport and no per-example uncertainty artifacts for the proposed study. Consequently none of RQ1--RQ4 has a substantive answer.

Complete a dimension-completeness manifest before filling any score cell. For every dataset, record the fixed train/validation/test IDs, source and target fields, every main baseline, EviSeq, every RQ3 ablation, every RQ4 encoder, checkpoint, prediction JSONL, ROUGE headline/detail/raw files, BERTScore and AlignScore detail files, diagnostics, and uncertainty output. Record an explicit omission and reason for any missing cell. Keep all cells and prose conditional until every required comparison is present; do not reuse the old LRSum/WikiLingua artifacts.

### EVAL-02 — RQ1 asks for error types that the retained metric does not measure directly

Location: `Paper/afmr_story_spec.md:33,61,65-71`; `Technical_Report/FACTUALITY_EVALUATION.md:3-4,17-40,72-83`; `src/rouge155/evaluate_alignscore.py:267-284`.

Classification: scope = epistemic/validity; severity = **S4**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = an operational RQ1 definition and held-out factual-error evidence; affected gates = RQ1, Results, Discussion, and Conclusion.

RQ1 is phrased as reduction of source-unsupported entities, numbers, and relations. The retained headline evaluator is sentence-level AlignScore: source chunks are paired with prediction sentences, the best chunk support is retained for each sentence, and those supports are averaged. The report also explicitly says that the repository's lexical entity/number diagnostic is not the headline result. Without typed annotations, extraction rules with validation, or a human/independent entailment audit, a change in AlignScore cannot answer the entity/number/relation question and cannot be promoted to “fewer factual errors.”

Choose one bounded contract. Either narrow RQ1 and its claims to a change in `AlignScore-nli_sp` sentence-level factual-consistency signal, with `1-C` described only as a transformed proxy, or add a held-out entity/number/relation audit with adjudication rules, sampling, denominators, and paired comparisons. In either case report domain-shift and cross-sentence limitations, and use “automatic factual-consistency signal” rather than “hallucination rate” or “factual error reduction” unless the added audit supports that wording.

### EVAL-03 — Paired uncertainty is promised for all quality comparisons but implemented only for ROUGE

Location: `Paper/afmr_question.tex:79,117`; `Paper/afmr_story_spec.md:34,59-61`; `Technical_Report/FACTUALITY_EVALUATION.md:64-75`; `src/rouge155/paired_bootstrap.py:151-231`.

Classification: scope = method/reproducibility; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the analysis plan and metric-level uncertainty artifacts; affected gates = RQ1, RQ2, statistical Results, and release.

`paired_bootstrap.py` accepts only the Perl ROUGE detail schema and computes deltas for ROUGE-1/2/L. The BERTScore and AlignScore evaluators can emit per-example rows, but there is no paired comparison path or confidence interval for either metric. This conflicts with the report's request for an AlignScore paired interval and with the manuscript's broader promise of paired intervals over identical test IDs. The current `p_nonpositive` field is a fraction of bootstrap replicates at or below zero; it is not a calibrated null-test p-value. A fixed bootstrap seed also quantifies resampling variability, not variability from stochastic training or checkpoint selection.

Add a metric-agnostic paired analysis that verifies ordered IDs, references, and (for AlignScore) source IDs/hashes plus the complete scorer configuration. Compute candidate-minus-baseline intervals for ROUGE F1, BERTScore F1, native AlignScore consistency, and transformed `H=1-C`, with direction metadata and explicit scales. Rename `p_nonpositive` to a bootstrap-tail quantity or document its non-p-value status. Predeclare one primary endpoint per RQ, handle the dataset/metric/baseline multiplicity, and add repeated training seeds or a hierarchical analysis if the claim is intended to generalize beyond one checkpoint. Do not present the ROUGE-only intervals as uncertainty for the whole evaluation.

### EVAL-04 — AlignScore chunking, truncation, and checkpoint loading are not yet validated as the stated metric

Location: `src/rouge155/evaluate_alignscore.py:86-103,167-208,210-253,270-345`; `Technical_Report/FACTUALITY_EVALUATION.md:17-27,42-62,77-83`.

Classification: scope = metric validity/provenance; severity = **S3**; evidence = **CONFIRMED** for missing validation and instrumentation, **PROBABLE** for a chunk-boundary mismatch; status = **OPEN**; authority needed = a parity check against the released `nli_sp` procedure and a locked scorer manifest; affected gates = RQ1, metric Methods, reproducibility, and release.

The code claims approximately 350-word complete-sentence source chunks, but it sets `n_groups = word_count // chunk_words + 1` and then divides the number of sentences by that group count. A multi-sentence 350-word source therefore produces two groups, typically about 175 words each, while a 700-word source can produce three or more groups depending on sentence count. No test covers the 349/350/351 or 699/700 boundaries, punctuation, or long sentences, so parity with the released procedure is not established. The fallback tokenizer call changes from `truncation="only_first"` to generic `truncation=True`; which side is truncated is then backend-dependent. No per-pair source/claim token counts or truncation rates are emitted. `strict=False` loading can also hide missing base-model weights, and a checkpoint with a compatible three-way head but the wrong backbone can pass the current shape checks.

Add byte- or score-level parity tests against the released `nli_sp` path at the chunk boundaries and for sentence splitting. Lock one truncation policy, fail if the tokenizer cannot enforce it, or record exactly which side was truncated and how often. Emit source/claim token lengths, truncated rows, empty claims, and splitter-resource/version metadata. Require and report checkpoint/backbone configuration, SHA-256 digests, and missing/unexpected load keys; reject a checkpoint that lacks the expected fine-tuned base weights or `nli_sp` head. Until parity is demonstrated, describe this as a local implementation of the intended protocol rather than verified reproduction.

### EVAL-05 — Decoder-only and T5Gemma2 comparisons are under-specified for fairness

Location: `Paper/afmr_question.tex:69,77,98-111`; `Paper/afmr_story_spec.md:34,50-60,77-82`.

Classification: scope = design/cross-model validity; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = per-system training and generation manifests; affected gates = RQ2, baseline Methods, and Results.

“Fine-tuned” and “matched protocols” do not specify a fair comparison by themselves. Decoder-only controls need an exact source-to-prompt serialization, while EviSeq and T5Gemma2 consume encoder--decoder inputs. The scoped artifacts do not provide prompt text, model/revision IDs, tokenizer IDs, parameter or trainable-parameter counts, or per-system source coverage. Equal source/target budgets within a dataset can still expose different content because tokenizers differ. T5Gemma2 changes the decoder and encoder family at once, so it is a useful external reference but not an interface-isolated control.

Provide a per-system table or machine-readable manifest containing immutable model/checkpoint/tokenizer IDs, parameter counts, input serialization including system/user/BOS/EOS/stop handling, source and target limits in the relevant tokenizers, identical record IDs and split, effective batch and optimizer schedule, update/token/compute budget, precision, checkpoint rule, and post-processing. Lock `num_beams=1`, `do_sample=false`, `temperature=0`, `top_k=0`, `top_p=1`, maximum new tokens, EOS/stop behavior, and repetition/no-repeat constraints for every system. Keep sampled candidates out of the official comparison. Frame T5Gemma2 as an architecture-level reference unless a same-decoder control is added; do not attribute its difference to AFMR alone.

### EVAL-06 — RQ3 “necessity” is not identified by the proposed ablations

Location: `Paper/afmr_story_spec.md:35,38,48-50,72`; `Paper/afmr_question.tex:69,98-106`.

Classification: scope = design/causal interpretation; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = an ablation estimand and implementation specification; affected gates = RQ3, Results, and Discussion.

“No encoder,” “no bridge,” and “no grounded copy” are named, but their input path, retraining procedure, initialization, parameter count, and output behavior are not defined. A no-encoder run may be a source-free decoder baseline rather than a component ablation; a no-bridge run may change width matching; and removing the copy route changes the output mixture and possibly trainable capacity. A performance drop after removal is evidence for a conditional component contribution under that retraining protocol, not proof that the component is universally necessary.

Define each ablation tensor path and what source information remains. Retrain every arm from the same pretrained starting point with a declared seed set, data order, optimization and update/token budget, and checkpoint rule; report capacity/compute changes. Use paired per-example contrasts on every dataset and state the narrowest interpretation (“under the locked decoder and training protocol”). Treat the no-encoder arm separately if it removes the task's source input. Use “component contribution” unless a predeclared effect criterion and robustness across seeds justify “necessary.”

### EVAL-07 — RQ4 conflates portability with directionality, capacity, and encoder-specific factors

Location: `Paper/afmr_story_spec.md:36,38,42-47,73`; `Paper/afmr_question.tex:98-104`.

Classification: scope = design/claim scope; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = a portability estimand or a factor-separated encoder design; affected gates = RQ4, Results, and Discussion.

The proposed PPLXEmbed anchor, QwenCausal, QwenEmbed, and NemotronEmbed variants differ simultaneously in directionality, family, tokenizer, pretraining, capacity, and likely context coverage. The planned contrasts can show transfer across the selected encoders, but they cannot identify separate effects of directionality or capacity. Differences in source tokenization and truncation can also masquerade as portability effects.

Either narrow RQ4 to descriptive transfer across the four named encoders under fixed training conditions, or add matched capacity/family controls that isolate directionality and capacity. Report each encoder's tokenizer, source-token coverage, truncation, parameter count, and compute, with per-dataset paired effects. Do not write that directionality or capacity “affects” quality unless the design separates those factors.

### EVAL-08 — Checkpoint, model, prompt, seed, and runtime provenance is promised but not immutable

Location: `Paper/afmr_question.tex:77,83,111-117`; `Paper/afmr_story_spec.md:40-61`; `src/rouge155/evaluate_bertscore.py:63-90,144-160`; `src/rouge155/evaluate_alignscore.py:167-208,326-349`; `src/rouge155/evaluate_rouge.py:213-226`.

Classification: scope = reproducibility/submission integrity; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = an experiment/scorer manifest committed with the release; affected gates = Methods, supplements, Results, and release.

The documents promise immutable model and tokenizer identifiers, prompts, preprocessing, manifests, checkpoints, generation settings, package versions, seeds, and failed-run records. The evaluators currently write paths, a few settings, and partial package metadata. BERTScore does not persist the language, fast-tokenizer choice, tokenizer identity, or model digest. AlignScore records paths and chunk/max-length settings but no model/checkpoint digest, Transformers/Torch/NLTK version, Punkt resource identity, or checkpoint load report. ROUGE records a pyrouge version and protocol string but not hashes for the Perl runtime/script, input manifest, or preprocessing implementation. `last.pt` is a rule, not evidence that the selected checkpoint exists or was not changed.

Create one manifest per model/dataset/run with SHA-256 digests for predictions, references/source manifest, model and tokenizer files or immutable revisions, scorer checkpoint and config, exact prompt, preprocessing and truncation policy, training and generation seed(s), deterministic flags, package/hardware versions, checkpoint-selection rationale, and failed/omitted runs. Embed the manifest digest and a run ID in every metric JSON and uncertainty artifact. Preserve the declared `last.pt` rule and verify it from the manifest rather than from a filename.

### EVAL-09 — Metric names, primary statistics, and derived-score notation need one ledger

Location: `Paper/afmr_question.tex:46-48,79`; `Paper/afmr_story_spec.md:33-34,57,61`; `Technical_Report/FACTUALITY_EVALUATION.md:15-40`; `src/rouge155/evaluate_rouge.py:21-22,211-227`; `src/rouge155/evaluate_bertscore.py:138-160`.

Classification: scope = notation/display/interpretation; severity = **S2**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = the metric reporting contract; affected gates = metric Methods, tables/captions, and Results.

The direction statements are largely correct: the ROUGE wrapper extracts F-scores on a 0--100 scale; BERTScore returns precision, recall, and F1 on a 0--100 scale; native AlignScore `C` is high-is-better and `H=1-C` is low-is-better. The remaining ambiguity is which statistic is the headline for ROUGE and BERTScore, and the manuscript writes `$1-\text{AlignScore}$` without the `nli_sp` subscript or the `C/H` definitions used in the technical report. The ROUGE protocol also includes Perl's own `-c 95` and `-r 1000` resampling, while the paired-bootstrap script computes a separate interval with a different procedure and seed.

Add a notation/provenance ledger: `x` = source, `y` = generated summary, `C_i = AlignScore_{nli\_sp}(x_i,y_i)`, `H_i=1-C_i`, and explicitly state the per-example-to-dataset mean. Name ROUGE-1/2/L F1 and BERTScore-F1 as the primary statistics (or state another choice), with P/R as secondary diagnostics and all directions/scales in captions. Declare whether Perl's intervals are diagnostic or primary; if retained, record their randomness/protocol, and do not label them paired intervals. The transform trace should be visible as source chunks → claim sentences → max support per claim → mean `C` → `H`.

### EVAL-10 — Macro-averaging and multiplicity are not defined

Location: `Paper/afmr_question.tex:79`; `Paper/afmr_story_spec.md:55,59,70-73`.

Classification: scope = statistical/reporting; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = a prespecified analysis plan; affected gates = all RQs, summary Results, and release.

“Macro-average across the four datasets” does not state whether the statistic is the unweighted mean of four dataset means, a pooled example-level mean, or a mean of dataset effects. These estimands differ substantially when test-set sizes and domains differ. The four RQs also create many metric, baseline, ablation, encoder, and dataset contrasts, but no primary endpoint or multiplicity policy is specified.

Report the four dataset estimates first. Define the secondary macro statistic explicitly, preferably as an unweighted mean of the four predeclared dataset-level effects, and use a dataset-aware bootstrap or hierarchical interval. Predeclare one primary contrast/metric per RQ; label the remaining metrics and model contrasts exploratory or apply a declared multiplicity adjustment. Never fill a macro cell when one dataset is missing.

### EVAL-11 — ROUGE headline evaluation can bypass the ID integrity checks used by the other scorers

Location: `src/rouge155/evaluate_rouge.py:64-114`; `src/rouge155/metric_io.py:49-100`; `src/rouge155/paired_bootstrap.py:35-65,91-102`.

Classification: scope = reproducibility/cross-metric integrity; severity = **S3**; evidence = **CONFIRMED**; status = **OPEN**; authority needed = a shared prediction-manifest contract; affected gates = paired comparisons and release.

ROUGE's `_prepare_data` binds scores to row order and silently substitutes the row number when `id` is absent; it does not reject duplicate IDs. BERTScore and AlignScore use `metric_io.load_jsonl`, which rejects duplicate IDs, but ROUGE can therefore produce a headline artifact that is not eligible for a valid paired comparison. The later bootstrap script checks IDs and references, yet it cannot repair a headline that was produced without stable IDs or verify source alignment for factuality.

Make all scorers use one loader that requires a stable, unique, non-null example ID and records the ordered ID/reference/source digest. Require `--details` for any result intended for paired inference, and make the comparison tool verify the same manifest digest, not only the ordered IDs and references. Preserve canonical Unicode/reference handling across all metrics and report any excluded or empty row.

## Section-linked comments and presentation flags

The manuscript's conditional framing is appropriate: `afmr_question.tex:69` and the story specification's claim/evidence map correctly keep quality claims unverified. That wording should remain in the abstract, Results placeholders, and conclusion until the complete evidence matrix is available. The descriptive title now avoids an outcome claim; a filled Results sentence should still name the measured signal and comparison set rather than use “factuality” as an unqualified synonym for AlignScore.

The display contract should make the direction visible wherever a value appears: ROUGE and BERTScore F1 higher-is-better; native AlignScore consistency higher-is-better; `H=1-C` lower-is-better and not a probability. Captions should name the scorer checkpoint/version, reference policy, aggregation, uncertainty type, and whether the value is per-dataset or macro. The display/notation pass cannot verify table/figure consistency because the substantive sections are `\input` files outside this review scope and no completed quality displays exist.

## Next-step instruction

Before the next review, complete the BookSum and GovReport manifests/recipes and all four-dataset run cells; lock the decoder-only prompts and T5Gemma2 comparison boundary; define the RQ1 typed-error or narrowed proxy estimand, RQ3 ablation estimands, and RQ4 portability scope; add metric-level paired uncertainty and dataset-aware aggregation; validate AlignScore chunking/truncation/checkpoint parity; and attach immutable run/scorer manifests. Then regenerate the exact metric JSON, per-example details, raw scorer outputs, diagnostics, and captions from the same ordered test IDs. Keep any failed or omitted run visible and leave its score pending.

## Functional-completeness retrospective

Scope covered: integration-stage evaluation review across `afmr_question.tex`, `afmr_story_spec.md`, `FACTUALITY_EVALUATION.md`, and the evaluator/tests in `src/rouge155`; RQ1--RQ4, metric definitions, fairness, uncertainty, provenance, truncation, and missing-dataset evidence were all checked.

Authority and locks: passed for the locked conditional story and the stated AlignScore/BERTScore directions; limited because no completed experiment manifest, active release record, prior-round material, or venue requirements were supplied.

Alignment: RQ1--RQ4 were traced to their proposed metrics and controls; RQ1's typed-error claim, RQ2's fair-control contract, RQ3's necessity language, and RQ4's factor attribution remain affected links.

Adapters and overlays: AI/LLM computational, display/notation/provenance, computational-design, and red-team/release checks ran; no conflicting venue or reviewer overlay was available.

Change propagation: no manuscript or evaluator source was changed; this memo is the only new artifact. Downstream summary/table synchronization remains uncheckable until the `\input` sections and completed runs are in scope.

Deterministic and visual checks: source/line inventory, scoped artifact inventory, evaluator inspection, and `PYTHONPATH=src /Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q src/rouge155/tests` were run; the result was **13 passed**. No completed paper displays were available for visual verification, and the tests do not cover AlignScore boundary parity, truncation accounting, checkpoint integrity, or metric-level paired uncertainty.

Open issues and unknowns: **2 S4**, **8 S3**, and **1 S2** open issues; no prior-round status can be assigned. The missing BookSum/GovReport artifacts and absent matched quality runs are confirmed; scorer parity, model provenance, prompts, seeds, and completed four-dataset comparisons remain unavailable.

Readiness: **not ready** for empirical RQ answers, performance claims, or `SUBMISSION_READY`. The next named stage is a reproducible four-dataset evaluation release after EVAL-01 and EVAL-02 are resolved and the S3 protocol blockers are closed.
