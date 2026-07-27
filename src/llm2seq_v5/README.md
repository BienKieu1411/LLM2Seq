# LLM2Seq-v5 — Stateful Phrase-Continuation Bridge

LLM2Seq-v5 is a prospective, low-data summarization experiment. Its strict
research target is to outperform the locked T5Gemma2-1B-1B baseline, not merely
reach parameter-efficient parity with it. No V5 score is claimed until the
frozen full run and artifact-level audit pass.

The deployable graph is always:

```text
one Qwen3-Embedding encoder
            ↓
one identity-preserving, bidirectional summary bridge
            ↓
one pretrained causal Qwen3 decoder
            ↓
one summary
```

The phrase-continuation mechanism is an output head inside that single decoder.
It is not a second encoder, decoder, retriever, teacher, or inference-time
model. Training uses full fine-tuning after an interface warm-up; it is not
LoRA/PEFT. The project saves only `last.pt` and contains no Hub upload/push
path.

## Locked claim and score targets

The baseline and protocols are fixed before V5 is run:

| Role | Scorer | ROUGE-1 | ROUGE-2 | ROUGE-L | Rows |
|---|---|---:|---:|---:|---:|
| Diagnostic T5Gemma baseline | Python `rouge==1.0.0` | 39.3040 | 19.5308 | 39.2955 | 3,901 |
| Paper T5Gemma baseline | Perl ROUGE-1.5.5 via pyrouge | 62.013 | 32.654 | 58.143 | 3,901 |
| Strict V5 paper target | Perl ROUGE-1.5.5 via pyrouge | **> 62.013** | **> 32.654** | **> 58.143** | 3,901 |

Equality is not a win. `final_audit.py` permits the intended claim only if V5
strictly exceeds all three paper scores, uses the same locked test/protocol,
and has both fewer total and fewer deployable runtime parameters than the
actual T5Gemma checkpoint. The rounded `2B` model-card size is not accepted as
an exact parameter count.

These values are targets, not expected or guaranteed results. In particular,
the approximately five-point paper ROUGE-2 gap is large and the new mechanism
may fail to close it.

## Architecture

### 1. Identity-preserving bidirectional bridge

The main encoder is `Qwen/Qwen3-Embedding-0.6B`. Its transformer remains causal
internally. V5 does not relabel it as natively bidirectional.

The bridge starts from the raw final encoder state. Earlier encoder layers
contribute only through a zero-initialized gated residual, so the initial path
preserves pretrained coordinates. Six full self-attention refinement layers
then make the token memory bidirectional. `check-model` tests this behavior by
changing a future source token and checking whether an earlier bridge-memory
state changes.

The same bridge also keeps V4's source-only prospective summary path: 16
ordered slots and two compact planner layers produce a soft prefix for the
decoder while dense token memory remains available through copied
cross-attention at every decoder layer. The reference is never an input to
this path at validation or inference.

### 2. Decoder-integrated phrase continuation

At every target step, the single decoder mixes three distributions:

1. `generate`: the pretrained Qwen LM vocabulary distribution;
2. `new_span`: a low-rank pointer to a newly selected source position;
3. `continue_span`: a pointer biased to the position immediately following the
   source position responsible for the previous emitted token.

The state is a probability distribution over source positions, not a hard
alignment. After a token is emitted, Bayes responsibility marginalizes
repeated occurrences and subword ambiguity. The next step shifts that soft
responsibility one position to the right, but never crosses a source-unit
boundary. The recurrent responsibility is detached by default to bound memory;
the value itself is still carried from one generation step to the next.

The head starts with `p(generate)=0.98`, preserving normal pretrained decoding
at installation. It then learns whether to generate, begin a supported source
span, or continue one. The final token probability is one normalized mixture,
so training and greedy inference optimize/use the same output family.

### 3. Summary-specific supervision

Training creates exact tokenizer-level 2/3/4-gram labels for source positions
whose source phrase also occurs in the reference. Labels are valid only inside
one visible source unit. The pipeline first verifies an identical
encoder/decoder vocabulary mapping; otherwise exact token copying is rejected.

The default V5 output loss is the mixture negative log-likelihood plus:

```text
L = L_mixture
  + 0.10 L_new_source_alignment
  + 0.10 L_continuation
  + 0.05 L_phrase_start_2/3/4
  + 0.02 L_pointer_coverage
  + 0.10 L_salience
  + 0.05 L_source_swap
```

`response_alignment_weight` is zero in the V5 main config. This deliberately
removes V4's equal-token-chunk target, which does not supervise order inside a
chunk. Sentence evidence and phrase labels use the reference only to construct
training targets. Oracle evidence is annealed from 1 to 0 by epoch five and is
hard-reset to zero for validation/test.

The plan-only curriculum remains at 25% -> 10%: dense memory is hidden for a
subset of training rows while the source-derived prefix remains. This prevents
the decoder from completely bypassing the summary bridge. It does not add a
second model or a second decoder pass.

## Novelty boundary

The defensible hypothesis is not “the first pointer-generator,” “the first
learned query bridge,” “the first dynamic soft prompt,” or “the first
decoder-only-to-encoder-decoder conversion.” Those components have substantial
prior art.

The candidate contribution is their summarization-specific combination under
only 13,999 task pairs:

- identity-preserving causal-LLM-to-bidirectional token conversion;
- a source-only prospective summary prefix plus dense token memory;
- a decoder-integrated `generate/new/continue` distribution;
- Bayesian soft source responsibility carried across target steps;
- exact 2/3/4-token source/reference phrase-start supervision and coverage.

Relative to V4, V5 changes the actual output distribution rather than only
adding another latent alignment loss. Relative to a standard copy head, it
models whether the next copied token should continue the previously selected
source span. The four registered ablations are required to establish which
part, if any, creates the gain.

## Known limitations

- Exact copying currently requires identical encoder/decoder token-ID maps.
  The primary Qwen-family pair satisfies this invariant; arbitrary embedding
  encoders do not automatically satisfy it.
- The 2/3/4-gram labels are tokenizer-level proxies, not direct word-level
  ROUGE-2 optimization.
- The phrase head predicts phrase starts and one-step continuation; it has no
  explicit phrase-length variable.
- Copy/continuation can over-favor extractive summaries and can inflate overlap
  through excessive length. The promotion gate below explicitly rejects that
  behavior.
- WikiLingua Vietnamese is one language/domain with 13,999 training pairs.
  Cross-dataset and English generalization remain unproven.
- Full fine-tuning is compute-heavy, and a single seed is not a variance
  estimate. A paper-quality result should repeat the frozen selected config
  with predeclared seeds or disclose the limitation.
- Perl ROUGE-1.5.5 has imperfect Vietnamese tokenization. It is retained only
  because it is the locked comparison protocol; the Unicode-aware diagnostic
  score must also be reported separately.
- A better validation pilot does not imply that V5 beats T5Gemma. Only the
  frozen 3,901-row paper evaluation can answer that question.

## Validation-only architecture selection

Smoke and pilot runs must never evaluate the test split. The evaluator rejects
test use from any subset-training config and also rejects partial locked-test
evaluation. `pilot-ablation-all` trains the
main candidate, a matched V2-core control, and four component controls on the
same hashed 2,000-row training subset, evaluates the same 512 validation rows,
and writes paired 10,000-sample bootstrap reports with seed 1729.

The predeclared promotion gate for moving from pilot to the one-shot full test
is:

1. `paired_vs_v2_core_single_bank.json` and `paired_vs_v4_psb.json` both have
   `comparable: true`.
2. Against both V2 core and V4 PSB, the 95% paired-bootstrap lower bound for
   diagnostic ROUGE-2 is strictly positive:
   `paired_bootstrap.rouge.rouge2.ci95_low > 0`.
3. Against both controls, ROUGE-1 and ROUGE-L are non-inferior: each 95% lower
   bound is at least `-0.5` point.
4. Main V5 has a positive paired mean ROUGE-2 delta against each component
   control (`no_continuation`, `no_phrase_prior`, and `no_coverage`). These
   comparisons support attribution; they do not replace the V4 control.
5. On validation, generation quality satisfies all of the following:

   - `empty_prediction_rate == 0`;
   - `unique_prediction_rate >= 99`;
   - `repeated_trigram_rate_mean` is no more than the lower of V2 core and V4
     PSB + 0.5 percentage point;
   - `dominant_prefix_5gram_rate` is no more than the lower of V2 core and V4
     PSB + 1 percentage point;
   - `length_ratio_mean` is between 0.80 and 1.20;
   - `too_long_rate` is no more than the lower of V2 core and V4 PSB + 2
     percentage points.

`paired_compare` verifies artifact comparability and emits the statistics; the
five rules above are the predeclared research decision policy. In particular,
read the ROUGE-2 confidence interval directly rather than treating the helper's
generic “all ROUGE dimensions superior” Boolean as the only promotion rule.

If any gate fails, do not inspect or tune on test. Revise/reject the mechanism
using validation only. If it passes, freeze architecture, hyperparameters,
seed list, data fingerprints, prompts, generation settings, scorer versions,
and the decision rule before running `full`.

## Scorer separation

V5 intentionally uses two non-interchangeable ROUGE paths:

### Diagnostic and selection scorer

`llm2seq_v5.evaluate` uses Python `rouge==1.0.0` with the stored
NFC/lowercase/whitespace preprocessing. It produces prediction rows,
`*.metrics.json`, and generation-quality diagnostics; `paired_compare`
recomputes aligned per-example scores and the paired validation bootstrap. Use
this path for smoke checks and validation-only model selection.

### Paper scorer

`../rouge155/evaluate_rouge.py` uses Perl ROUGE-1.5.5 through pyrouge and writes
`*.rouge155.json`. It is the only scorer used for the locked T5Gemma paper
claim.

Never compare a Python score from one system with a Perl score from another,
never substitute one backend's threshold for the other, and never use the
diagnostic pilot to state paper-level superiority. `final-audit` binds the
candidate and baseline Perl score files to their actual metrics, checkpoint,
test fingerprint, parameter counts, and decoding protocol.

## Main configuration

The main config is
`configs/qwen3_embedding_0_6b_phrase_continuation.yaml`, which inherits
`configs/base.yaml`. Important defaults are:

```yaml
model:
  encoder_name: Qwen/Qwen3-Embedding-0.6B
  decoder_name: Qwen/Qwen3-0.6B

adapter:
  num_bidirectional_layers: 6
  num_summary_slots: 16
  summary_planner_layers: 2

phrase_pointer:
  rank: 128
  phrase_hidden_size: 256
  generate_probability_init: 0.98
  use_continuation: true

training:
  interface_warmup_epochs: 3
  full_finetune_epochs: 12
  batch_size: 64
  gradient_accumulation_steps: 1
  seed: 42

data:
  max_source_length: 3072
  max_target_length: 384
  phrase_orders: [2, 3, 4]

generation:
  min_new_tokens: 16
  max_new_tokens: 256
  repetition_penalty: 1.05
  no_repeat_ngram_size: 3
```

Model names may be replaced with already-downloaded local checkpoint paths in
the config. None of the commands below uploads or pushes a checkpoint.

## CNN/DailyMail profile

The CNN/DailyMail profile keeps the same six-layer V5 architecture and uses a
4096-token source window. Its six epochs are exactly one interface warm-up
epoch plus five full-fine-tuning epochs; warm-up is not added on top of six.
The source prompt and greedy decoding settings match the saved T5Gemma
CNN/DailyMail baseline protocol.

Prepare local JSONL-style `train.txt`, `val.txt`, and `test.txt`, smoke-test,
then run the complete train/evaluation pipeline:

```bash
bash run.sh cnndm-prepare /absolute/path/to/cnndm
bash run.sh cnndm-smoke --overwrite-output-dir
bash run.sh cnndm --overwrite-output-dir
```

The equivalent one-command full run is:

```bash
CNNDM_SOURCE_DIR=/absolute/path/to/cnndm \
  bash run.sh cnndm --overwrite-output-dir
```

CNN/DailyMail data is copied into `data/raw/cnndm/` and converted into
`data/cnndm/`; V5 never trains from T5Gemma's processed-data directory. The
preparer writes `data/cnndm/manifest.json` with exact canonical fingerprints.
The recorded T5Gemma2-1B-1B Perl result is `43.591 / 19.396 / 40.436`, but it
remains reference-only until its exact test count and fingerprint are bound to
the same manifest.

## Exact run order on B200

Run from `src/llm2seq_v5`.

`run.sh` never falls back to an unrelated system Python. On the B200 machine,
point it at that machine's prepared environment first (the local workstation
default is `/Users/kieugiangbien/bienkieu_env/bin/python`):

```bash
export PYTHON_BIN=/absolute/path/to/bienkieu_env/bin/python
```

### 1. Offline code checks

```bash
bash run.sh test
```

This mode forces the Hugging Face/Transformers offline flags.

### 2. Real-checkpoint architecture and parameter checks

```bash
bash run.sh check-model
bash run.sh count-params
```

Record the exact total, deployable, bridge, decoder, and phrase-pointer counts.
Do not infer them from model names.

### 3. Flow smoke test

```bash
bash run.sh smoke --overwrite-output-dir
```

Inspect
`runs/llm2seq_v5/smoke_phrase_continuation_100/last_validation_predictions.jsonl`,
its metrics, and `smoke_gate.json`. This is only a graph/loss/generation test.

### 4. Validation-only main/control pilot

```bash
bash run.sh pilot-ablation-all --overwrite-output-dir
```

The candidate reports are written under
`runs/llm2seq_v5/pilot_phrase_continuation_2000/`:

```text
paired_vs_v4_psb.json
paired_vs_v2_core_single_bank.json
paired_vs_no_continuation.json
paired_vs_no_phrase_prior.json
paired_vs_no_coverage.json
```

Apply the promotion gate above. Do not run `eval-test` during selection.

### 5. Freeze, then run the full candidate once

Only after the gate passes and the experiment definition is frozen:

```bash
export PYROUGE_HOME_DIR=/absolute/path/to/ROUGE-1.5.5
bash run.sh full --overwrite-output-dir
```

The main output directory is:

```text
runs/llm2seq_v5/qwen3_embedding_0_6b_phrase_continuation/
```

It contains `last.pt`, `resolved_config.yaml`,
`last_test_predictions.jsonl`, `last_test_predictions.metrics.json`, and—when
Perl ROUGE is configured—`last_test_predictions.rouge155.json`.

If training/evaluation completed before `PYROUGE_HOME_DIR` was set, score the
existing predictions without retraining:

```bash
export PYROUGE_HOME_DIR=/absolute/path/to/ROUGE-1.5.5
bash run.sh rouge155
```

### 6. Audit the strict T5Gemma claim

Use the real T5Gemma prediction-derived artifacts, not numbers copied by hand:

```bash
BASELINE_SCORES=/absolute/path/to/t5gemma_predictions.rouge155.json
BASELINE_METRICS=/absolute/path/to/t5gemma_predictions.metrics.json

bash run.sh final-audit \
  --config runs/llm2seq_v5/qwen3_embedding_0_6b_phrase_continuation/resolved_config.yaml \
  --candidate-scores runs/llm2seq_v5/qwen3_embedding_0_6b_phrase_continuation/last_test_predictions.rouge155.json \
  --candidate-metrics runs/llm2seq_v5/qwen3_embedding_0_6b_phrase_continuation/last_test_predictions.metrics.json \
  --baseline-scores "$BASELINE_SCORES" \
  --baseline-metrics "$BASELINE_METRICS" \
  --output runs/llm2seq_v5/qwen3_embedding_0_6b_phrase_continuation/final_claim_audit.json
```

The paper may say “strictly outperforms T5Gemma” only when
`final_claim_audit.json` reports `passed: true`. Otherwise report the measured
result and narrower supported conclusion.

### 7. Run frozen full ablations for the paper

After architecture selection is closed—and without using their test scores to
retune the main model—run:

```bash
bash run.sh ablation-all --overwrite-output-dir
```

With `PYROUGE_HOME_DIR` still exported, each full ablation is evaluated with
both scorers. Individual modes are also available:

```bash
bash run.sh ablation-v4-psb --overwrite-output-dir
bash run.sh ablation-no-continuation --overwrite-output-dir
bash run.sh ablation-no-phrase-prior --overwrite-output-dir
bash run.sh ablation-no-coverage --overwrite-output-dir
```

## Four registered ablations

| Ablation | Exact change | Question |
|---|---|---|
| `v4_psb_control` | Disable the phrase pointer and restore V4 response alignment (`0.15`) | Does the complete V5 output mechanism improve over the immediately preceding PSB architecture? |
| `no_continuation` | Keep `generate + new_span`, disable stateful continuation and its loss | Is carrying Bayesian source responsibility across decoding steps useful? |
| `no_phrase_prior` | Keep the pointer mixture, set phrase-start bias and 2/3/4-gram label loss to zero | Do summary-supported phrase-start labels improve where copying begins? |
| `no_coverage` | Set pointer coverage loss to zero | Does coverage reduce repeated/reused source spans without hurting recall? |

All four retain one encoder, one bridge, and one decoder. They differ only in
the registered mechanism, use the same subset/protocol in pilot, and use the
same locked protocol after freeze.

## Reporting checklist

- Report both ROUGE backends with explicit names; never merge their values.
- Report all three paper ROUGE scores, length/quality diagnostics, and exact
  runtime parameter counts.
- Report the 3,901-row test fingerprint and the locked greedy decoding recipe.
- Report the four validation paired comparisons and full ablations, including
  negative results.
- State that the encoder transformer is causal and bidirectionality is added by
  the bridge.
- State that phrase labels and oracle evidence are training-only.
- Do not infer superiority from smoke, train-set overfit, validation alone, or
  equality with T5Gemma.
