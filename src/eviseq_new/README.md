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

For the bridge ablation, set `architecture.bridge_mode: direct_projection` in a
fresh config and output directory. This keeps the pretrained encoder and the
decoder cross-attention path, but maps the final encoder states through only a
width projection; AFMR's controller, depth/feature residuals, focus prior,
temperature and value anchor are absent. Grounded copy remains enabled unless
it is disabled separately, so the bridge contribution is isolated. The
checkpoint architecture spec records this mode and refuses to load it as a
full AFMR checkpoint. The ArXiv and PubMed runners expose the same control via
`AFMR_BRIDGE_MODE=direct_projection`; use `AFMR_GROUNDED_COPY=false` for the
independent no-copy condition.

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

Each input record contains the configured source and target fields (strings or lists of strings). Training can read raw JSONL directly. For the same three-split preparation used by `eviseq_v2`, run `prepare-dataset`: PubMed/ArXiv first look for `train.label.jsonl`, `val.label.jsonl`, and `test.label.jsonl`, then fall back to the usual split names; CNNDM, WikiLingua, and BookSum accept `train`, `val`/`validation`, and `test` JSONL/JSON/TXT names. GovReport accepts those names and the public release's `gao_{train,valid,test}` and `crs_{train,valid,test}` files, merging GAO and CRS records when both are present. BookSum aliases `chapter`/`chapter_text` and `summary_text`; GovReport also flattens nested `report`/`highlight` sections. The command copies raw files, writes canonical `id/text/summary` JSONL, emits `preparation_report.json`, and rejects duplicate IDs or exact source text across splits. Evidence labels and prompt fields are not consumed by AFMR.

The source instruction is part of `data.encoder_prefix`, so the same instruction is used for every row and travels through the encoder. The decoder-side `data.decoder_prompt` is a short generation instruction; it does not duplicate the source text. Prompt text is not read from dataset rows. `configs/8192_avg.yaml` includes the full summarization instruction in its encoder prefix and uses an 8,192-token source budget.

`decoder_chat_template: true` sends the fixed decoder instruction as a user message to the decoder tokenizer and appends its assistant generation marker. With `decoder_chat_template: false`, the decoder instruction is tokenized as literal text. When changing either prompt for a comparison, use a fresh prediction filename so an existing JSONL resume prefix cannot be mistaken for generations made with the new instruction.

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

To prepare the ArXiv, BookSum, and GovReport benchmark tree in one command, set one input root containing a subdirectory for each dataset (or pass the three `--*-input` paths explicitly):

```bash
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  bash scripts/prepare_benchmark_datasets.sh \
  --input-root /data/summarization \
  --output-root datasets \
  --raw-root data/raw
```

For BookSum mirrors whose source and target columns have different names, use the single-dataset wrapper with explicit fields:

```bash
bash scripts/prepare_afmr.sh --dataset booksum \
  --input-dir /data/booksum --output-dir datasets/booksum \
  --source-field chapter --target-field summary_text --id-field chapter_id
```

## A100 training/evaluation

Start from `configs/afmr_base.yaml`, copy it to a task recipe, and set only model locations, data files, lengths, batch resources, and output directory. The generic base uses one warm-up epoch and four full-finetuning epochs; benchmark recipes override this to match the corresponding T5Gemma total. WikiLingua uses six full-finetuning epochs with no warm-up. The benchmark defaults to greedy decoding (`num_beams: 1`, `do_sample: false`, `temperature: 0.0`, `top_k: 0`, `top_p: 1.0`), and training is CE-only.

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh train configs/afmr_pubmed.yaml

PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_afmr.sh evaluate \
  configs/afmr_pubmed.yaml runs/afmr/pubmed_value_anchor_copy/last.pt \
runs/afmr/pubmed_value_anchor_copy/test_predictions.jsonl --split test
```

To train WikiLingua directly from the local pretrained encoder and decoder,
save the checkpoint with the lowest validation CE, and evaluate that checkpoint
on the test split, run:

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_wikilingua.sh
```

Set `ENCODER_MODEL`, `DECODER_MODEL`, `WIKILINGUA_DATA_DIR`, or
`AFMR_OUTPUT_DIR` when the local paths differ. Use `CUDA_VISIBLE_DEVICES=0,1`
for DDP training; test evaluation runs on the first visible GPU.

To continue a completed train-only run on WikiLingua, use a fresh output
directory. The wrapper reads the checkpoint stage and extends that stage by
five WikiLingua epochs, then evaluates `last.pt` on the WikiLingua test split:

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/continue_wikilingua.sh \
  --config configs/afmr_wikilingua.yaml \
  --checkpoint runs/afmr/train_only/last.pt \
  --output-dir runs/afmr/wikilingua_from_train_only \
  --additional-epochs 5 \
  --device cuda:0 \
  --eval-batch-size 16 \
  --overwrite-output-dir
```

The active WikiLingua config must point to the same local encoder and decoder
backbones used to create the checkpoint. The wrapper rejects a missing
checkpoint or dataset split and never overwrites the checkpoint directory.

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

To run the same PubMed recipe with a local Nemotron embedding encoder, use the
dedicated wrapper. It runs one process on one GPU or launches DDP when two
GPUs are listed in `CUDA_VISIBLE_DEVICES`. It defaults to the server folder
`/workspace/storage-shared/nlp/dungdx4/BERT/Nemotron-3-Embed-1B-BF16`; set
`ENCODER_MODEL` to the actual local folder when the checkpoint is stored
elsewhere (the older `llama-nemotron-embed-1b-v2` folder is also accepted):

```bash
cd src/eviseq_new
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
CUDA_VISIBLE_DEVICES=0 \
ENCODER_MODEL=/path/to/Nemotron-3-Embed-1B-BF16 \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PUBMED_SOURCE_DIR=/path/to/pubmed \
TRAIN_BATCH_SIZE=16 \
GRADIENT_ACCUMULATION_STEPS=6 \
EVAL_BATCH_SIZE=16 \
bash scripts/run_pubmed_nemotron.sh
```

For two GPUs, change only the visible-device list; the wrapper changes the
default accumulation from 6 to 3 so the global effective batch remains 96:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ENCODER_MODEL=/path/to/Nemotron-3-Embed-1B-BF16 \
DECODER_MODEL=/path/to/Qwen3-0.6B \
PUBMED_SOURCE_DIR=/path/to/pubmed \
bash scripts/run_pubmed_nemotron.sh
```

The wrapper keeps the effective training batch at 96 while using a smaller
per-GPU micro-batch for the larger encoder, materializes a local-only config,
trains the configured warm-up/full stages, and evaluates `last.pt` on the
PubMed test split. It writes to `runs/afmr/pubmed_nemotron_embed` and logs to
`logs/afmr`; `OVERWRITE_OUTPUT_DIR=true` starts a fresh run and
`RESUME_CHECKPOINT=/path/to/last.pt` resumes a compatible run. No model is
downloaded by this wrapper.

For separate candidate generation, enable sampling explicitly with a fresh
output JSONL. Filtering is applied in the order temperature, top-k, then
nucleus top-p; this path is never called during training:

```bash
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  bash scripts/run_afmr.sh evaluate configs/afmr_pubmed.yaml \
  runs/afmr/pubmed_value_anchor_wide/last.pt \
  runs/afmr/pubmed_value_anchor_wide/test_candidates.jsonl \
  --split test --do-sample --temperature 0.7 --top-k 50 --top-p 0.9
```

The queue now writes to `runs/afmr/pubmed_pair_afmr_value_anchor_copy`, leaving earlier results untouched. Set `AFMR_GROUNDED_COPY=false` for the value-anchor LM-only control in `pubmed_pair_afmr_value_anchor_lm`. Additionally set `AFMR_ARCHITECTURE=afmr_v1` for the shared-memory LM-only control in `pubmed_pair_afmr_v1_lm`. Numerical/text fixes remain enabled. The queue still runs PPLX then Qwen3-Embedding on the same GPU. For a single experiment, use `run_afmr.sh train` with one task config instead.

For ArXiv, `scripts/run_arxiv.sh` prepares the canonical ArXiv JSONL tree
when it is missing, materializes a config with local model/data paths, trains
the PPLX-to-Qwen AFMR model, and evaluates `last.pt` on the test split. It
uses an 8,192-token source budget, the scientific source instruction used by
the decoder baseline, greedy decoding (`temperature: 0`, `top_k: 0`,
`top_p: 1`), and conservative long-context defaults of batch 8 per GPU.
With one visible GPU the default accumulation is 12; with two visible GPUs
the wrapper launches DDP and uses accumulation 6, preserving global effective
batch 96. Resource settings are environment overrides:

```bash
cd src/eviseq_new
PYTHON=python3 \
CUDA_VISIBLE_DEVICES=0 \
ARXIV_SOURCE_DIR=/data/arxiv \
ENCODER_MODEL=/models/pplx-embed-v1-0.6b \
DECODER_MODEL=/models/Qwen3-0.6B \
TRAIN_BATCH_SIZE=8 \
GRADIENT_ACCUMULATION_STEPS=12 \
EVAL_BATCH_SIZE=8 \
bash scripts/run_arxiv.sh
```

To use both cards, set `CUDA_VISIBLE_DEVICES=0,1`; training switches to DDP
automatically. Evaluation runs two independent shards concurrently and
concatenates them by dataset index.

Set `OVERWRITE_OUTPUT_DIR=true` only for an intentional restart. Set
`RESUME_CHECKPOINT=/path/to/last.pt` to continue a compatible run. Prepared
files go to `ARXIV_DATA_DIR` (default `datasets/arxiv`), the generated config
is kept under the run directory, and logs are written separately under
`logs/afmr` so progress output cannot corrupt prediction JSONL. Set
`ROUGE155_SCRIPT` to the local ROUGE wrapper to run Perl ROUGE after the
built-in evaluation. `ENCODER_MODEL` may point to a local Nemotron embedding
checkpoint; `PPLX_ENCODER` remains a backward-compatible alias.

The runtime loads models only for `train` or `evaluate`; importing AFMR and running tests does not download anything. Checkpoints are structurally guarded: changing batch size, generation batch size, data paths, or model folder location is allowed, while changing AFMR ranks, windows, depth taps, or cross-attention layout is rejected.

The benchmark recipes retain the exact T5Gemma encoder/source instruction. The Qwen decoder receives a fixed task instruction through its native chat template with `enable_thinking=False`, an assistant generation prompt, and a short output prefix. That prefix is identical during training and inference and excluded from supervised labels and returned predictions. No reference is used to construct it. For chat prompts, repetition/n-gram constraints apply only to generated summary tokens, not the instruction. Greedy decoding retains the corresponding T5Gemma recipe's penalties and length limits. Report the different decoder conditioning in experiments; do not describe the full input protocol as identical. Legacy configs without `decoder_chat_template` retain their literal prompt/BOS behavior.

Train a new checkpoint for the new prompt and architecture. Do not edit an old `resolved_config.yaml` and evaluate old weights as if they had been trained with these inputs. Changing `afmr_v1` to `afmr_value_anchor` is rejected by checkpoint compatibility checks; their parameter shapes alone do not imply semantic compatibility.

The token-wise graph (`afmr_token_depth_lowrank_v3`) is intentionally incompatible with earlier document-wise AFMR checkpoints. Do not use `strict=False` to force-load those checkpoints. Train this graph from pretrained backbones in a separate output directory. New token-wise checkpoints resume normally.

Training stores parameters, gradients and AdamW moments in FP32; `model.compute_dtype: bfloat16` enables CUDA BF16 autocast for the heavy operations. CPU tests use FP32. This avoids directly accumulating tiny updates into BF16 parameters; see [Mixed Precision Training](https://arxiv.org/abs/1710.03740) for the FP32 accumulated-update principle. It is not all-FP32 matrix computation. CUDA evaluation loads the backbone/cross-attention in `compute_dtype`, keeping BF16 KV caches by default; legacy configs retain their configured inference dtype.

Non-reentrant backbone checkpointing, token-weighted gradient accumulation (including a partial final window), and per-stage linear LR decay remain enabled. Optimizer moments are carried from warm-up to full fine-tuning. LM-head CE is computed in checkpointed token chunks instead of retaining full `[B,T,V]` logits. Encoder KV caching is disabled; only the requested depth taps are captured. The PubMed Nemotron and ArXiv wrappers support one or two GPUs with DDP; two-GPU evaluation uses independent shards and merges them by dataset index so the final prediction JSONL remains ordered. FP32 training storage requires more VRAM than direct BF16 updates; a B200 smoke/profile is necessary before reusing the maximum old batch size.

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

## JSONL length inspection and filtering

The repository includes two model-free utilities for datasets whose fields may
differ from the canonical `text`/`summary` names. Both scripts preserve the
complete JSON object when filtering; pass dotted paths when the source or
target is nested. For model-token lengths, point the scripts at local
tokenizers. Without `--tokenizer`, they use whitespace counts and label that
choice in the report.

Inspect all fields, sample counts, length percentiles, and a histogram:

```bash
python scripts/inspect_jsonl_lengths.py /data/train.jsonl \
  --source-field input.article --target-field output.summary \
  --source-tokenizer /models/pplx-embed-v1-0.6b \
  --target-tokenizer /models/Qwen3-0.6B \
  --output-dir /data/train_length_report
```

The report directory contains `length_report.json`, `lengths.csv`, and
`length_histogram.png`. The JSON report lists every observed field and its
types, plus complete field names in a few sample records. Use
`--sample-rows 0` when sample values are not needed.

Create a training subset with targets strictly shorter than 512 tokens,
sources shorter than 4,096 tokens, and a fixed number of examples:

```bash
python scripts/filter_jsonl_by_target_length.py /data/train.jsonl \
  /data/train_lt512_100k.jsonl \
  --source-field input.article --source-tokenizer /models/pplx-embed-v1-0.6b \
  --source-below 4096 \
  --target-field output.summary --target-tokenizer /models/Qwen3-0.6B \
  --target-below 512 --num-samples 100000 --selection random --seed 17 \
  --report /data/train_lt512_100k.report.json
```

Use `--selection first` to retain input order, `--allow-fewer` when the
eligible pool may be smaller than the requested count, `--source-below N` or
`--target-below N` for strict `< N` limits, and
`--max-source-tokens N`/`--max-target-tokens N` for inclusive `<= N` limits.
The output lines are copied verbatim, so additional metadata fields are
retained.

To balance the source-length distribution, cap every source-length bin rather
than keeping only the sparse tail. The following also enforces the 2,048-token
source and 512-token target limits; the exact written count is recorded per
bin in the report:

```bash
python scripts/filter_jsonl_by_target_length.py /data/train.jsonl \
  /data/train_balanced.jsonl \
  --source-field input --target-field output \
  --target-tokenizer /models/Qwen3-0.6B \
  --detokenize \
  --max-source-tokens 2048 \
  --max-target-tokens 512 \
  --balance-source-bins 50 --max-per-source-bin 20000 \
  --selection random --seed 17 \
  --report /data/train_balanced.report.json
```

The report contains the eligible and written count for every source-length
bin. `--selection first` keeps the earliest rows in each bin; `random` uses a
reproducible reservoir sample and then writes the selected rows in input order.
With `--detokenize`, only the configured source and target fields are
normalized; all other metadata fields are preserved.

The server defaults are also packaged as a one-command wrapper:

```bash
bash scripts/balance_train_jsonl.sh
```

Override its paths with `--input`, `--output`, and `--report` when using a
different dataset.

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
