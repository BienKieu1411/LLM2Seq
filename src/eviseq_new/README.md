# EviSeq AFMR

AFMR (Adaptive Full-Memory Residual) is a full-source encoder–decoder architecture for general text summarization:

```text
pretrained encoder → value-anchored AFMR interface → cross-attention Qwen decoder → contextual copy/LM mixture → greedy
```

The bridge preserves every valid source token. A document/prompt/requested-budget controller conditions a low-rank depth residual in encoder space, a low-rank encoder-to-decoder feature residual, and one multi-scale source prior. Depth weights have shape `[batch, source_tokens, depth_taps]`: a shared learned scorer reads each normalized candidate token representation, plus a document-conditioned depth preference. Softmax is over depth, not tokens; selection runs once after the encoder. Residual output factors and the depth/focus scorers start at zero; gates and prior strength are bounded, but learned residual vector norms are not mathematically bounded. The prior is additive `[batch, source_tokens]` and is consumed by every decoder cross-attention layer. There are no banks, hard top-k pruning, per-layer routers, rerankers, beam search, KD, or test-time evidence labels.

Focus regions are overlapping **token windows**, not selected sentences. Decoder memory contains the entire source. The only objective is token-level cross-entropy (CE): depth, feature and focus routes all learn through it. There is no positive-sentence mining, allocation loss, or contrastive head. External evidence labels are ignored.

## Value-anchored retrieval adaptation

The current recipe uses `architecture.name: afmr_value_anchor`. Let `H0` be the final encoder state projected to decoder width, and `M` the existing AFMR-adapted states. For each decoder layer:

```text
K = key_norm(W_K memory_norm(M))
V = W_V memory_norm(H0)
attention = softmax(Q K^T / sqrt(head_dim) + source_bias + padding_mask)
output = W_O (attention V)
```

The previous `afmr_v1` uses `M` for both K and V. The new interface separates retrieval adaptation from the value content: depth/feature residuals and the focus prior learn where to retrieve, while the final encoder states supply the values. The encoder, value projection and shared width projection remain trainable in full fine-tuning. Nothing is detached. Gradients from CE reach both paths; for retrieval-only residual parameters, the direct derivative of V is zero. This restriction does not guarantee factuality or ROUGE improvement, and normalization of keys can still change addressing.

The residual outputs start at zero, so the two variants have identical initial outputs with identical weights and inputs. The new variant introduces no trainable parameters, candidate generation, extra Transformer pass or auxiliary loss. It retains another full-source hidden tensor and performs value-side normalization separately; therefore memory/latency are not claimed to be identical. Both variants use one cross-attention operation per layer and the same once-per-document K/V cache and finished-row compaction.

The conceptual precedent is [Key-Value Memory Networks](https://aclanthology.org/D16-1147/), which separates addressing and reading. That work is not evidence of a summarization gain for this model. The hypothesis here is that task-specific routing need not rewrite the value stream of an already pretrained encoder. Run the shared-memory control with the same FP32 updates, prompt, preprocessing, seed, epochs and decoding settings before attributing any gain to this interface.

## Contextual grounded output

The current base recipe enables `decoder.grounded_copy`. Audited predictions contained corrupted biomedical terms, incorrect numbers, swapped entities and reversed result directions. Merely changing hidden-state addressing does not provide an explicit path for preserving source tokens. The new output head complements value-anchored retrieval with a decoder-conditioned source distribution:

```text
k_j = RMSNorm(overlap_pool_j(W_context RMSNorm(H0)) + W_lexical RMSNorm(E_decoder[token_j]))
q_t = W_query RMSNorm(decoder_hidden_t)
a_t = softmax(q_t k^T / sqrt(rank) + overlap_pool(source_bias) + source_mask)
g_t = sigmoid(W_gate [q_t ; sum_j a_tj k_j] + gate_bias)
P(y_t=v) = (1-g_t) P_LM(v) + g_t sum_{j: token_j=v} a_tj
loss = -mean_t log P(reference_t)
```

`g_t` initially equals 0.05 and is subsequently learned without a fixed copy quota. Duplicate occurrences accumulate probability; there is no hard selection of an evidence sentence. A row with no eligible source tokens uses the LM logits exactly. Training remains **one CE objective**, not CE plus a separately weighted copy/contrastive loss. CE reaches the decoder query, copy gate, lexical projection, source context and source prior. Encoder parameters are frozen during warm-up as before and receive these gradients in full fine-tuning. Source keys are not detached during training. No gold evidence, candidate ranking, teacher or reference-derived input is required at inference.

This is an adaptation of [pointer-generator networks](https://aclanthology.org/P17-1099/), not a claim that copying is a new invention. The integration uses a pretrained LLM vocabulary and sparse cross-tokenizer alignment instead of assuming identical encoder/decoder token IDs. Copying cannot by itself fix incorrect role binding, omitted findings or evidence outside the encoder input: selecting a supplier's country is still wrong even if that country appears verbatim in the source. Exact-copy biases can also limit abstraction; see [Improving Latent Alignment in Text Summarization by Generalizing the Pointer Generator](https://aclanthology.org/D19-1390/). No ROUGE gain is guaranteed.

Both fast tokenizers supply character offsets. Only the visible source prefix is re-tokenized with the **decoder** tokenizer. Sparse overlap edges connect those tokens to encoder positions, excluding the encoder instruction, special/padding tokens, uncovered spans and a possibly partial token at a truncation boundary. Overlap weights sum to one per eligible decoder-source token. References never participate in this alignment, and the source length limit is unchanged. This supports different tokenizers without passing encoder IDs into the decoder vocabulary or reading text beyond truncation.

The head has **393,473 parameters** at hidden width 1024 and key rank 128. Context is projected before sparse pooling; no dense encoder-token × decoder-token alignment tensor is materialized. Keys and token IDs are prepared once per source and reused by greedy decoding, including finished-row compaction. Training checkpoints CE in time chunks, computes vocabulary logits only for supervised positions and aggregates mixture likelihood in FP32. It adds one output-side attention calculation and sparse pooling, not another encoder/decoder pass or cross-attention bank. Extra tokenization, attention and activations still cost time/VRAM; real B200 throughput must be measured.

For an architectural control set `decoder.grounded_copy.enabled: false` and use a separate run directory. The queue exposes `AFMR_GROUNDED_COPY=false`. Keep prompt, preprocessing, FP32 updates, seed, epochs and decoding fixed; legacy resolved configs without this section keep the LM-only graph. The new head does not change random initialization of shared modules at a fixed seed. An old checkpoint cannot be evaluated with copying simply enabled in YAML: the new head must be trained, and architecture checks reject this mismatch. Compare validation results before a final held-out test comparison; repeated test-guided development must be disclosed.

## Offline smoke test

The offline smoke uses the actual AFMR/copy graph with tiny randomly initialized Qwen backbones and no model downloads. It enables grounded copy explicitly and checks CE gradients, warm-up/full optimizer updates, dense/chunked CE parity, checkpoint round-trip, greedy evaluation and prediction resume. The legacy `afmr_smoke.yaml` fixture disables copy for backward-compatibility tests; the smoke command enables it:

```bash
cd /absolute/path/to/LLM2Seq
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  src/eviseq_new/scripts/run_afmr.sh smoke
```

Expected output is one JSON object with `"status": "ok"`. Local pytest is also model-free:

```bash
cd src/eviseq_new
PYTHONPATH=. /absolute/path/to/bienkieu_env/bin/pytest -q
```

To smoke the real checkpoints on an A100 without a long run, use a task config with local model/data paths and cap train, validation, and test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON=python3 \
  bash scripts/smoke_a100.sh configs/afmr_pubmed.yaml
```

This exercises the actual tokenizer, encoder, AFMR bridge, copied
cross-attention, backward pass, checkpoint writer, and greedy KV-cache
generation. Offline mode is enforced: configured models must already be available locally. Set `AFMR_SMOKE_EVAL_EXAMPLES` and
`AFMR_SMOKE_EVAL_BATCH_SIZE` to control the bounded evaluation; set
`AFMR_SMOKE_DEVICE=cpu` only when exercising the script locally without CUDA.
The script creates an isolated `runs/smoke/run_*` directory and never overwrites the task's full-training directory. It runs one warm-up and one full epoch with batch 2 × accumulation 2; task architecture and source/target limits are preserved. Set `AFMR_SMOKE_TRAIN_EXAMPLES` and `AFMR_SMOKE_VALIDATION_EXAMPLES` to change the default 100/20 caps.

## Preparing data

Each input record contains the configured source and target fields (strings or lists of strings). Training can read raw JSONL directly. For the same three-split preparation used by `eviseq_v2`, run `prepare-dataset`: PubMed/ArXiv expect `train.label.jsonl`, `val.label.jsonl`, and `test.label.jsonl`; CNNDM/WikiLingua accept the usual `train`, `val`/`validation`, and `test` JSONL/JSON/TXT names. The command copies raw files, writes canonical `id/text/summary` JSONL, emits `preparation_report.json`, and rejects duplicate IDs or exact source text across splits. Evidence labels are not consumed by AFMR.

PubMed, ArXiv and CNNDM preparation detokenizes punctuation, brackets, quotes and contractions using the T5Gemma sentence-level rules, preserving sentence newlines. Their recipes also enable `data.detokenize: true`, so already-prepared files receive the same idempotent normalization when read. No full-corpus cache or repeated copy is required. This applies to both source and target, including test references. Legacy resolved configs without this key retain their previous text handling; do not mix old partial predictions with newly normalized references. WikiLingua does not enable this English-oriented normalization.

```bash
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  bash scripts/prepare_afmr.sh --dataset pubmed \
  --input-dir /data/pubmed --output-dir datasets/pubmed \
  --raw-copy-dir data/raw/pubmed
```

The same command handles ArXiv, CNN/DailyMail, and WikiLingua by changing
`--dataset` and `--input-dir`:

```bash
bash scripts/prepare_afmr.sh --dataset arxiv --input-dir /data/arxiv --output-dir datasets/arxiv
bash scripts/prepare_afmr.sh --dataset cnndm --input-dir /data/cnndm --output-dir datasets/cnndm
bash scripts/prepare_afmr.sh --dataset wikilingua --input-dir /data/wikilingua --output-dir datasets/wikilingua
```

## A100 training/evaluation

Start from `configs/afmr_base.yaml`, copy it to a task recipe, and set only model locations, data files, lengths, batch resources, and output directory. The generic base uses one warm-up epoch and four full-finetuning epochs; benchmark recipes override this to match the corresponding T5Gemma total (PubMed 1+3, CNNDM/WikiLingua 1+5). Decoding is greedy (`num_beams: 1`, `do_sample: false`), and training is CE-only.

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml

PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh evaluate \
  configs/afmr_pubmed.yaml runs/afmr/pubmed_value_anchor_copy/last.pt \
runs/afmr/pubmed_value_anchor_copy/test_predictions.jsonl --split test
```

For a one-GPU PubMed queue that prepares data, trains the PPLX
encoder recipe, then trains a Qwen3-Embedding control and evaluates both
`last.pt` checkpoints, run:

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
CUDA_VISIBLE_DEVICES=0 \
PPLX_ENCODER=/path/to/pplx-embed-v1-0.6b \
QWEN_ENCODER=/path/to/Qwen3-Embedding-0.6B \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PUBMED_SOURCE_DIR=/path/to/pubmed \
bash scripts/run_pubmed_pair.sh
```

Preparation is skipped when the canonical PubMed files already exist. Set
`EVAL_BATCH_SIZE` to control evaluation memory. Set `ROUGE155_SCRIPT` to the
local `evaluate_rouge.py` wrapper to append the Perl ROUGE-1.5.5 audit.

The queue now writes to `runs/afmr/pubmed_pair_afmr_value_anchor_copy`, leaving earlier results untouched. Set `AFMR_GROUNDED_COPY=false` for the value-anchor LM-only control in `pubmed_pair_afmr_value_anchor_lm`. Additionally set `AFMR_ARCHITECTURE=afmr_v1` for the shared-memory LM-only control in `pubmed_pair_afmr_v1_lm`. Numerical/text fixes remain enabled. The queue still runs PPLX then Qwen3-Embedding on the same GPU. For a single experiment, use `run_afmr.sh train` with one task config instead.

The runtime loads models only for `train` or `evaluate`; importing AFMR and running tests does not download anything. Checkpoints are structurally guarded: changing batch size, generation batch size, data paths, or model folder location is allowed, while changing AFMR ranks, windows, depth taps, or cross-attention layout is rejected.

The benchmark recipes retain the exact T5Gemma encoder/source instruction. The Qwen decoder now receives a task instruction through its own native chat template with `enable_thinking=False`, an assistant generation prompt, and a short output prefix. That prefix is identical during training and inference and excluded from supervised labels and returned predictions. No reference is used to construct it. For chat prompts, repetition/n-gram constraints apply only to generated summary tokens, not the instruction. Greedy decoding retains the corresponding T5Gemma recipe's penalties and length limits. Report the different decoder conditioning in experiments; do not describe the full input protocol as identical. Legacy configs without `decoder_chat_template` retain their literal prompt/BOS behavior.

Train a new checkpoint for the new prompt and architecture. Do not edit an old `resolved_config.yaml` and evaluate old weights as if they had been trained with these inputs. Changing `afmr_v1` to `afmr_value_anchor` is rejected by checkpoint compatibility checks; their parameter shapes alone do not imply semantic compatibility.

The token-wise graph (`afmr_token_depth_lowrank_v3`) is intentionally incompatible with earlier document-wise AFMR checkpoints. Do not use `strict=False` to force-load those checkpoints. Train this graph from pretrained backbones in a separate output directory. New token-wise checkpoints resume normally.

Training stores parameters, gradients and AdamW moments in FP32; `model.compute_dtype: bfloat16` enables CUDA BF16 autocast for the heavy operations. CPU tests use FP32. This avoids directly accumulating tiny updates into BF16 parameters; see [Mixed Precision Training](https://arxiv.org/abs/1710.03740) for the FP32 accumulated-update principle. It is not all-FP32 matrix computation. CUDA evaluation loads the backbone/cross-attention in `compute_dtype`, keeping BF16 KV caches by default; legacy configs retain their configured inference dtype.

Non-reentrant backbone checkpointing, token-weighted gradient accumulation (including a partial final window), and per-stage linear LR decay remain enabled. Optimizer moments are carried from warm-up to full fine-tuning. LM-head CE is computed in checkpointed token chunks instead of retaining full `[B,T,V]` logits. Encoder KV caching is disabled; only the requested depth taps are captured. The runner currently supports one GPU/process and rejects multi-process launches. FP32 training storage requires more VRAM than direct BF16 updates; a B200 smoke/profile is necessary before reusing the maximum old batch size.

Training prints reusable, machine-readable progress lines with stage, epoch
percentage, epoch/total optimizer steps, token-weighted CE, gradient norm,
learning rates, total elapsed time, epoch/total training ETA, and sample/token throughput. For
example:

```text
[train] stage=full | epoch=2/4 | epoch_progress=[====>.............] 24.0% | step=120/500 | total_step=620/2000 | CE=1.23840 | grad=0.8123 | lr=bridge:3.00e-05,cross_attention:5.00e-05 | elapsed=03:17:42 | epoch_eta=01:48:09 | total_eta=06:32:45 | vram=67.42GiB | ex/s=7.42 | tok/s=30120
```

`total_eta` extrapolates the current epoch's average optimizer-step time over
all remaining training epochs. It excludes future validation, checkpoint saving,
and generation. The estimate adapts when the stage or throughput changes;
warm-up throughput may underestimate full-finetuning time. Numeric ETA is saved
as `total_eta_seconds` in the step records, including resumed runs.

The same step/epoch records are appended to `training_metrics.jsonl`, including
numeric progress and elapsed-time fields; elapsed time is also stored in each
checkpoint so a resumed run continues the total-time counter. Generation
writes every completed sample batch immediately to JSONL, prints progress/ETA,
resumes from a contiguous prefix, retries CUDA OOM by halving the active batch,
and stores final metrics in `<predictions>.metrics.json`.

Checkpoints are `epoch_001.pt`, ..., `last.pt`; `best.pt` is optional and off in the main recipe. Final evaluation uses `last.pt`. `generation.batch_size` controls decoding; `--batch-size` overrides it explicitly. An existing prediction file resumes only if its IDs and references match a contiguous prefix of the active split. Each new batch is flushed to JSONL with progress/ETA. Use a different output filename when comparing another checkpoint. Evaluation retries CUDA OOM by halving the active batch; if a single example still fails, it raises the original error.

All local verification uses tiny random backbones with network disabled, including BF16 attention backward, CE-only focus gradients, cache parity, accumulation and warm-up resume equivalence. A real PPLX/Qwen GPU smoke and throughput profile remain necessary; local tests do not establish B200 kernel efficiency or ROUGE improvement.

For bundled `configs/*.yaml`, relative data/output paths are rooted at `eviseq_new/`, independent of the calling directory or whether files already exist. An external YAML resolves relative paths against its own directory. At encoder/decoder hidden size 1024, the main AFMR bridge contains 1,981,446 parameters; this excludes both pretrained backbones and the copied decoder cross-attention. Token-wise depth selection adds 1,024 parameters over the previous document-wise graph and does not stack a second full-depth memory tensor.

## Performance controls

Training speed controls in the base recipe enable length bucketing, eight data
workers, persistent workers, CUDA fused AdamW, and TF32 FP32 matrix products.
Bucketing uses JSONL record byte lengths collected during the existing offset
scan as an inexpensive length proxy. It does not tokenize or cache the corpus
in RAM. Each example is visited once per epoch, including the final partial
batch; batches are reshuffled deterministically by seed and epoch. Changing
batch composition changes the training trajectory even though the objective is
unchanged. TF32 can be disabled with `training.tf32: false` for full FP32
matrix-product precision. BF16 autocast is independently controlled by `model.compute_dtype`.

The chunked LM head now processes only non-ignored labels. Its CE and gradients
match the full-logit path in regression tests. `decoder.ce_chunk_size: 1024`
reduces chunk/recomputation overhead relative to 256 while keeping a bounded
logit allocation. Set it to 256 if the larger chunk exceeds the memory budget.
These settings do not change checkpoint architecture compatibility.

Greedy evaluation applies repetition and n-gram constraints with tensor
operations on the model device. `generation.compact_finished: true` removes
finished rows from both self-attention and cross-attention KV caches, and
restores predictions to their original row order. Cache implementations without
batch selection automatically retain the full batch. Generation does not
tokenize references. JSONL append/resume and CUDA OOM splitting remain active.
Compaction and TF32 may change floating-point rounding on GPUs; use
`compact_finished: false` and `tf32: false` for a controlled parity comparison.

To use the data-loader improvements with an older external/resolved config,
add the following keys to its existing sections (do not duplicate sections):

```yaml
training:
  length_bucketing: true
  length_bucket_multiplier: 50
  num_workers: 8
  persistent_workers: true
  fused_optimizer: true
  tf32: true
decoder:
  ce_chunk_size: 1024
generation:
  compact_finished: true
```

An already running Python process needs a restart to use updated code. Existing
AFMR checkpoints can resume at the next epoch through `--resume-checkpoint`.
GPU speedups must be measured on the actual B200, model paths, and batch shapes;
offline tiny-model correctness tests are not hardware performance benchmarks.

## Package layout

```text
eviseq_new/
├── eviseq_afmr/       public AFMR namespace
│   ├── data/           schema, split preparation, dataset, collator
│   ├── modeling/      encoder, controller, bridge, decoder, model
│   ├── training/      optimizer, checkpoint, engine
│   └── evaluation/    greedy generation, metrics
├── configs/            AFMR base and task recipes
├── scripts/run_afmr.sh source-tree entry point
├── scripts/smoke_a100.sh bounded real-model smoke run
└── tests/              model-free contract and gradient tests
```
