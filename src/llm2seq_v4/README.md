# LLM2Seq-v4 — Prospective Summary Bridge

V4 is an isolated research implementation for converting a pretrained text
encoder and a pretrained causal Qwen decoder into a summarizer without the
massive architecture-conversion corpus used by T5Gemma.  The deployed graph is
always:

```text
one encoder -> one prospective-summary bridge -> one causal decoder
```

No reference summary, teacher model, retrieval system, or extra decoder is used
at inference.  Training is full fine-tuning after an interface-only warm-up;
this is not LoRA/PEFT.  The code has no Hub upload or push command.

## Why V3 was not enough

V3 could prove source use while still learning the wrong thing for ROUGE-2.
Its global source/summary InfoNCE and source-swap objective distinguish the
correct document from a wrong document, but do not say which phrases should be
generated or in which order.  HiRoute also keeps three full-length memories and
can collapse to one route.  Increasing its adapter from four to six/eight
random blocks does not close that supervision gap on only 13,999 training pairs.

V4 therefore removes HiRoute from the main configuration and changes the
interface itself.

## Architecture

### 1. Identity-preserving source path

The raw final encoder state is the base representation.  Earlier-layer fusion
is only a gated residual, initialized at zero.  When encoder/decoder widths are
equal, the projection base is an exact identity; normalization is confined to
the learned residual branch.  This avoids destroying pretrained coordinates at
step zero, a weakness of the V3 `RMSNorm -> identity matrix -> RMSNorm` path.

Six full bidirectional refinement blocks produce dense token memory.  Native
bidirectional encoders such as PPLX use one refinement block.

The primary Qwen3-Embedding checkpoint still uses causal attention internally;
it is not mislabeled as a bidirectional encoder. `check-model` verifies that
the raw encoder is causal and then separately verifies that changing a future
source token changes an earlier **bridge-memory** state. Thus the bridge—not a
model-card flag—is the component that performs the bidirectional conversion.

### 2. Prospective summary latents

Sixteen learned ordered queries pass through two compact planner blocks:

```text
slot self-attention -> slot-to-source cross-attention -> SwiGLU
```

The slots cross-attend the complete dense memory, with a soft evidence bias;
there is no hard top-k pruning.  They form a fixed `[batch, 16, decoder_width]`
representation of the *potential summary*.

### 3. Dual conditioning without a second model

The decoder receives:

- full dense source memory through copied, independently normalized
  cross-attention for coverage and exact details;
- the 16 summary latents as a source-dependent soft prefix to the decoder's
  native causal self-attention path.

The prefix is inserted only on the first decoding call.  It remains in the KV
cache and is not inserted again.  Prefix hidden positions are removed before
the LM head, so teacher-forcing labels remain exactly aligned.

### 4. Summary-specific supervision

For every training reference, valid target tokens are split into 16 contiguous,
ordered chunks.  Each predicted slot is aligned to the detached mean decoder
embedding of its matching chunk with diagonal cosine plus in-example InfoNCE.
This trains content *and order*, unlike V3's global document identity loss.

On a scheduled 25% -> 10% of training examples, dense token memory is zeroed
while the source-derived prefix remains.  The decoder must reconstruct the
summary from the latent plan, preventing a token-memory shortcut.  This is a
single forward pass, not an additional decoder pass.

Existing sentence-evidence labels supervise salience.  During early training,
the planner bias mixes oracle and predicted evidence; the oracle contribution
anneals from 1 to 0 by epoch five.  Validation/test hard-reset the mix to zero,
so there is no reference leakage.

The default objective is:

```text
L = L_CE
  + 0.10 L_salience
  + 0.15 L_ordered_response_alignment
  + 0.05 L_source_swap
```

Label smoothing is zero.  V3's global prompt InfoNCE is disabled in the main
run.  Generation is greedy and locked to the saved WikiLingua T5Gemma recipe:
minimum 16 new tokens, repetition penalty 1.05, no-repeat trigram 3, no beam
search.  Both systems use the same source instruction and 3,072-token limit.
V4 additionally records sentence boundaries as `unit_ids` for its salience
loss; this architecture-specific annotation is disclosed and never changes the
locked raw source/reference rows.
Adam moments for adapter/cross-attention parameters are carried from warm-up
into full fine-tuning; newly unfrozen pretrained weights start with fresh state.

## Research basis and novelty boundary

- [T5Gemma](https://arxiv.org/abs/2504.06225) shows that decoder-only weights
  can initialize strong encoder-decoder models, but relies on large continued
  UL2/PrefixLM adaptation.  V4 tests whether a summary-specific bridge can do
  this from task pairs only.
- [LLM2Vec-Gen](https://arxiv.org/abs/2603.10913) motivates output-centric
  fixed-length representations of a potential response.  V4 makes those
  representations source-conditioned, ordered, and usable by a summarization
  decoder.
- [BLIP-2](https://proceedings.mlr.press/v202/li23q.html) establishes learned
  query bridges into a frozen language model.  V4 is text-to-text, retains full
  token memory, and adds ordered response alignment plus evidence curriculum.
- [Dynamic Soft Prompting](https://aclanthology.org/2024.emnlp-main.546/) supports
  input-dependent soft prompts.  The paper claim must not be “first dynamic
  prompt”; it is a low-data decoder-only-to-seq2seq summarization bridge.
- [FROST](https://aclanthology.org/2021.tacl-1.88/) and
  [Explicit Information Selection](https://aclanthology.org/D18-1205/) motivate
  ordered content plans and explicit salience for summarization.
- [GMSA](https://aclanthology.org/2026.acl-long.1324/) identifies the semantic
  gap between compressed high-level vectors and decoder input space.  V4's
  ordered target-embedding alignment directly addresses that gap.

The defensible novelty is the combination: **identity-preserving conversion +
source-conditioned prospective summary latents + ordered response-space
alignment + memory-path curriculum**, while retaining a single encoder and
single causal decoder.  Learned queries alone are not a novelty claim.

No architecture guarantees a five-point ROUGE-2 gain.  The code is designed so
this hypothesis can be rejected cleanly by held-out pilots before expensive
full runs.

## Configuration

Main files:

- `configs/qwen3_embedding_0_6b_psb.yaml`: primary score/paper model.
- `configs/qwen3_base_0_6b_psb.yaml`: causal Qwen encoder conversion control.
- `configs/pplx_embed_v1_0_6b_psb.yaml`: native-bidirectional encoder score
  candidate.
- `configs/smoke_qwen3_embedding_100.yaml`: 100 train / 20 validation-test.
- `configs/pilot_qwen3_embedding_2000.yaml`: 2,000 train / 512 held-out.
- `configs/cnndm_qwen3_embedding_0_6b_psb_4096.yaml`: CNN/DailyMail,
  4,096-token source, one interface warm-up plus five full-finetune epochs.
- `configs/smoke_cnndm_100.yaml`: 100/20/20 CNN/DailyMail flow check.

The actual model path or local checkpoint path is set in `model.encoder_name`
and `model.decoder_name`.  The code never uploads checkpoints.

Important parameters are in `configs/base.yaml`:

```yaml
adapter:
  num_bidirectional_layers: 6
  num_summary_slots: 16
  summary_planner_layers: 2

training:
  interface_warmup_epochs: 3
  full_finetune_epochs: 12
  batch_size: 64

objectives:
  response_alignment_weight: 0.15
  plan_only_probability_start: 0.25
  plan_only_probability_end: 0.10
  oracle_evidence_mix_start: 1.0
  oracle_evidence_mix_end: 0.0
```

## Run on B200

From `src/llm2seq_v4`:

```bash
# Offline structural/unit checks. This mode forbids Hub access.
bash run.sh test

# Real checkpoint invariant check on the B200 machine.
bash run.sh check-model

# Flow test; always inspect predictions and smoke_gate.json.
bash run.sh smoke --overwrite-output-dir

# Decisive held-out pilot.
bash run.sh pilot --overwrite-output-dir

# Full primary run, then last.pt test evaluation.
bash run.sh qwen --overwrite-output-dir

# Repeat every invariant on the actual trained weights.
bash run.sh check-model \
  --config runs/llm2seq_v4/qwen3_embedding_0_6b_psb/resolved_config.yaml \
  --checkpoint runs/llm2seq_v4/qwen3_embedding_0_6b_psb/last.pt

# Optional encoder controls/candidates.
bash run.sh qwen-base --overwrite-output-dir
bash run.sh pplx --overwrite-output-dir
```

### CNN/DailyMail

V4 owns its CNN/DailyMail copy and does not read V5 or T5Gemma runtime files.
The preparation step converts `article_text`/`abstract_text` and common aliases
to canonical JSONL, rejects duplicate/cross-split IDs, and writes a manifest.
The full profile uses the same English source instruction, 4,096/256
source/target limits, and greedy 8--192-token generation contract as T5Gemma.

From `src/llm2seq_v4`:

```bash
bash run.sh cnndm-prepare /absolute/path/to/cnndm
bash run.sh cnndm-smoke --overwrite-output-dir
bash run.sh cnndm --overwrite-output-dir

# Equivalent one-command prepare + full run:
CNNDM_SOURCE_DIR=/absolute/path/to/cnndm \
  bash run.sh cnndm --overwrite-output-dir
```

This run saves/evaluates only `last.pt`. The recorded T5Gemma CNN/DailyMail
score (43.591/19.396/40.436, Perl ROUGE-1.5.5) remains `reference_only` until
the exact baseline test count and fingerprint are bound to the local manifest;
the runner will not emit a false formal-comparability claim before then.

Only `last.pt` is saved and evaluated.  If Perl ROUGE is configured, each
pipeline also writes `last_test_predictions.rouge155.json`; otherwise scoring
is skipped without failing training and can be run later:

```bash
export PYROUGE_HOME_DIR=/absolute/path/to/ROUGE-1.5.5
bash run.sh rouge155 runs/llm2seq_v4/qwen3_embedding_0_6b_psb/last_test_predictions.jsonl
```

Before making the final superiority claim, compare the actual candidate and
T5Gemma artifacts—not only the score copied into a config:

```bash
bash run.sh final-audit \
  --config runs/llm2seq_v4/qwen3_embedding_0_6b_psb/resolved_config.yaml \
  --candidate-scores runs/llm2seq_v4/qwen3_embedding_0_6b_psb/last_test_predictions.rouge155.json \
  --candidate-metrics runs/llm2seq_v4/qwen3_embedding_0_6b_psb/last_test_predictions.metrics.json \
  --baseline-scores /path/to/t5gemma_predictions.rouge155.json \
  --baseline-metrics /path/to/t5gemma_predictions.metrics.json \
  --output runs/llm2seq_v4/qwen3_embedding_0_6b_psb/final_claim_audit.json
```

This audit fails unless the test fingerprint, exact T5Gemma checkpoint,
parameter counts, source/target limits, source instruction, greedy decoding
settings and Perl ROUGE backend all match. Older T5Gemma metric files that do
not contain `max_source_length`/`max_target_length` must be regenerated by the
evaluation script; no retraining is needed.

The paper comparison is the locked 3,901-example split with Perl
ROUGE-1.5.5.  `rouge==1.0.0` remains diagnostic only; scores from the two
backends must never be mixed.

## Ablations

The four decisive questions are:

1. `no_summary_prefix`: is the output-centric latent path necessary?
2. `no_ordered_alignment`: do the slots need response-space/order supervision?
3. `no_plan_only`: does forcing the decoder to use slots matter?
4. `no_oracle_evidence`: does train-only evidence curriculum help?

Run all five 2k pilots (main plus four controls) sequentially:

```bash
bash run.sh pilot-ablation-all --overwrite-output-dir
```

Additional analysis configs test salience, cross-gate initialization, label
smoothing, and 8/32 slots.  Full ablations should only be run after the main
pilot improves held-out ROUGE-2 by roughly one point without increased
repetition or source-swap failure.

## Required interpretation

- A 100-example overfit score only proves that the graph and loss connect.
- A 2k pilot decides whether to spend B200 time; it is not the paper result.
- The claim “beats T5Gemma” requires all 3,901 locked raw examples, the same
  3,072-token/source-instruction/greedy-decoding contract and Perl
  ROUGE-1.5.5, plus at least three seeds for the selected configuration.
- Report parameter count from the real graph on B200; do not infer it from model
  names or count a training-only teacher that is absent at inference.
