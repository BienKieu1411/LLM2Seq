# LLM2Seq H200 Version

H200 version for long-context WikiLingua summarization.

This folder is self-contained for server upload. It includes the H200 training
entrypoints/configs and a bundled copy of the core `llm2seq` package under
`llm2seq_h200/llm2seq`. It implements the architecture discussed for reducing
encoder-to-decoder information loss:

- LLM2Vec bidirectional encoder.
- Token-wise layer fusion over selected encoder layers.
- Gated residual projection from encoder space to decoder memory space.
- Token-level salience gate for content selection.
- EncStack memory refinement.
- Learnable global memory tokens prepended to cross-attention memory.
- Bigger decoder for H200: 8 layers, hidden size 1024, 16 heads.
- Long context: source 4096 tokens, target 512 tokens.
- Three training phases.

## Architecture Overview

![LLM2Seq Architecture](/Users/kieugiangbien/Downloads/Project/Encoder-Decoder LLM/llm2seq_final/figures/image.png)

LLM2Seq is designed to minimize information loss when bridging a pre-trained bidirectional encoder with a causal decoder. Key components include:
- **LLM2Vec Encoder**: A powerful bidirectional encoder processing long contexts.
- **Layer Fusion**: Token-wise layer fusion aggregates representations across selected encoder layers to capture both shallow and deep semantic features.
- **Salience Gate**: A token-level gating mechanism that identifies and filters out irrelevant tokens, reducing noise for the decoder.
- **Gated Residual Adaptor**: Maps encoder representations into the decoder's dimension while maintaining a residual connection for stability.
- **EncStack**: Refines memory representations before cross-attention.
- **Global Memory Tokens**: Learnable tokens prepended to the context to capture document-level global representations.
- **MTP (Multi-Token Prediction) Heads**: Used in Phase 3 to accelerate inference via Speculative Decoding, allowing the model to draft multiple future tokens simultaneously.

## Standalone Folder Layout

Upload the whole `llm2seq_h200/` folder to the server. You do not need to
upload the parent repo or a separate `llm2seq/` folder.

Key files included:

```text
llm2seq_h200/requirements.txt
llm2seq_h200/llm2seq/                 # bundled core package
llm2seq_h200/configs/*.yaml
llm2seq_h200/scripts/*.sh
llm2seq_h200/scripts/*.py
llm2seq_h200/run_pipeline.sh
llm2seq_h200/install_deps.sh
llm2seq_h200/smoke_check.sh
```

Recommended server commands:

```bash
cd /path/containing/llm2seq_h200
bash llm2seq_h200/smoke_check.sh
bash llm2seq_h200/install_deps.sh
cp llm2seq_h200/env.example.txt llm2seq_h200/env.txt
nano llm2seq_h200/env.txt
bash llm2seq_h200/run_pipeline.sh
```

If the server only has `python3`, set this in `env.txt`:

```text
PYTHON_BIN=python3
```

The scripts prepend `llm2seq_h200/` to `PYTHONPATH`, so training/evaluation
uses the bundled package rather than requiring a package installed elsewhere.

For H200 servers with NVIDIA driver CUDA 12.4, `install_deps.sh` installs a
CUDA 12.4 PyTorch wheel by default:

```text
TORCH_VERSION=2.5.1
TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124
```

This avoids accidentally installing a newer PyTorch build such as CUDA 12.6 or
12.8, which can make `torch.cuda.is_available()` return `False` even though
`nvidia-smi` shows the GPU.

With a server reporting driver `550.x` and `nvidia-smi` CUDA `12.4`, keep the
default `torch==2.5.1` / `cu124` unless you also upgrade the NVIDIA driver.

Check the active PyTorch CUDA build:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

## Local Data Shape

The bundled WikiLingua folder is expected here:

```text
llm2seq_h200/wikilingua/train.json
llm2seq_h200/wikilingua/val.json
llm2seq_h200/wikilingua/test.json
```

These files are JSONL-style files: each line is one JSON object with `src` and
`tgt` lists. The preprocessing script joins `src` into `source` with newlines
between input sentences, and joins `tgt` into `target` with spaces so the
summary is trained as one natural paragraph. Training adds the configured task
prefix:

```text
Tóm tắt văn bản sau:
```

Processed JSONL files are generated at runtime under
`llm2seq_h200/data/processed/`. They are not required in the upload folder
because `run_pipeline.sh` runs preprocessing before Phase 1.

Current split sizes after preprocessing:

```text
train raw/processed: 13,999 / 13,999
val raw/processed:    1,680 / 1,680
test raw/processed:   3,902 / 3,901
```

One test row has an empty `src`, so preprocessing skips it.

## Three Phases

### Phase 1: Warmup Adapter + Decoder

Encoder is frozen. Train only:

- token-wise layer fusion
- gated residual adaptor
- token-level salience gate
- EncStack
- global memory tokens
- decoder
- LM head

Config:

```bash
llm2seq_h200/configs/phase1_warmup_4096.yaml
```

### Phase 2: LoRA Encoder Adaptation

Resume Phase 1 checkpoint, then train:

- LoRA adapters on the LLM2Vec encoder
- adaptor
- decoder
- LM head

This phase uses parameter-efficient LoRA adaptation instead of full encoder
fine-tuning. It is designed to run with much lower VRAM while still adapting
the encoder to WikiLingua summarization.

Config:

```bash
llm2seq_h200/configs/phase2_lora_encoder_4096.yaml
```

Phase 2 resets optimizer/global step on resume because it is a new training
phase with newly trainable LoRA/adaptor/decoder parameters.

### Phase 3: MTP-D Self-Distillation

Resume Phase 2 checkpoint, add cascaded MTP heads, then train only the MTP
blocks. The main encoder/adaptor/decoder/lm_head path is frozen so the main
summarizer learned in Phase 1 and Phase 2 is preserved.

This adapts the paper "Self-Distillation for Multi-Token Prediction" to the
third phase of this encoder-decoder project:

- MTP architecture: cascaded MTP, sharing decoder embedding and LM head.
- MTP CE coefficient `alpha`: `loss_weight: 0.6`.
- Head weights are discounted by depth: `[1.0, 0.8, 0.5, 0.25]`.
- Self-distillation: main head logits are detached and used as the teacher.
- KL curriculum: CE-only first, then ramp self-distillation from 20% to 50%
  of training.
- TopN logits: `self_distill_top_k: 2048`.
- KL direction: forward KL from main-head distribution to MTP-head distribution.
- Four-head KL coefficient `beta`: `self_distill_loss_weight: 0.2`.
- Phase 3 optimizer uses `mtp_lr`; encoder/adaptor/decoder LR are `0.0`
  because only MTP blocks are trainable.

The paper trains MTP-D during LLM pre-training and also studies frozen
continued-training extensions. Here Phase 3 is intentionally the frozen-main
variant: Phase 1 and Phase 2 first make the main summarizer usable, then Phase
3 trains only the MTP blocks.

Config:

```bash
llm2seq_h200/configs/phase3_mtp_self_distill_4096.yaml
```

Phase 3 instantiates the same LoRA encoder structure as Phase 2, loads the
learned LoRA adapter weights, then freezes the main encoder/adaptor/decoder
path. Phase 3 checkpoints are trainable-only deltas containing the MTP
block/head weights, so Phase 3 evaluation loads the Phase 2 checkpoint first
and then overlays the Phase 3 MTP delta.

During Phase 3 training, the frozen main path is kept in eval mode so dropout
does not make the detached teacher logits noisy. Only the MTP blocks stay in
train mode.

## Prepare WikiLingua JSON

Input folder should contain:

```text
train.json
val.json
test.json
```

Each sample can be a JSON object:

```json
{"src": ["sentence 1", "sentence 2"], "tgt": ["summary 1"]}
```

The script supports either a JSON list or multiple JSON objects in one file.

```bash
python llm2seq_h200/scripts/prepare_wikilingua_json.py \
  --input_dir llm2seq_h200/wikilingua \
  --output_dir llm2seq_h200/data/processed \
  --max_train -1 \
  --max_eval -1 \
  --source_joiner "\\n" \
  --target_joiner " "
```

## Train

Install dependencies:

```bash
bash llm2seq_h200/install_deps.sh
```

Create the local environment file:

```bash
cp llm2seq_h200/env.example.txt llm2seq_h200/env.txt
nano llm2seq_h200/env.txt
```

Fill `HF_TOKEN` and `HF_REPO_ID` in `llm2seq_h200/env.txt` if you want
automatic upload to Hugging Face. Do not share `env.txt` because it contains
secrets.

The H200 configs enable Hugging Face uploads by default. Training reads
`HF_TOKEN` and `HF_REPO_ID` from the environment or `env.txt`.
`HF_AUTO_DOWNLOAD_CHECKPOINTS=true` enables checkpoint fallback from HF when a
local checkpoint path is missing.

Phase 1:

```bash
bash llm2seq_h200/scripts/train_phase1.sh
```

Default for H200: 4 epochs, micro-batch size 12, gradient accumulation 4,
effective batch 48. Long-context 4096-token training is memory dominated, so
micro-batch is kept below the point that fills the whole H200.

Phase 2:

```bash
bash llm2seq_h200/scripts/train_phase2.sh runs/h200_llm2seq_phase1_warmup/best.pt
```

Default: 4 epochs, LoRA encoder adaptation, micro-batch size 2, gradient
accumulation 16, effective batch 32. This is much lighter than full encoder
fine-tuning and is suitable for H200/H100-class runs, though 4096-token context
still keeps micro-batch small.

Phase 3:

```bash
bash llm2seq_h200/scripts/train_phase3.sh runs/h200_llm2seq_phase2_lora_encoder/best.pt
```

Default for H200: 4 epochs, cascaded 4-head MTP-D, micro-batch size 8,
gradient accumulation 8, effective batch 64. Only MTP blocks are trainable.

Run all phases:

```bash
bash llm2seq_h200/scripts/train_all.sh
```

By default, `train_all.sh` evaluates the full test set after each phase. Set
`RUN_PHASE_EVAL=false` if you only want training.

Logs are written to:

```text
llm2seq_h200/logs/
```

During training, checkpoints are pushed to Hugging Face after every epoch:

```text
checkpoints/h200_phase1_warmup/epochs/epoch_XXX_step_YYY.pt
checkpoints/h200_phase2_lora_encoder/epochs/epoch_XXX_step_YYY.pt
checkpoints/h200_phase3_mtp_self_distill/epochs/epoch_XXX_step_YYY.pt
```

All `.pt` files pushed by the trainer store only the lightweight trained
weights for that phase. The original LLM2Vec/base encoder weights are never
saved or pushed to your HF repo; they are loaded directly from `encoder_name`.
Phase 2 stores encoder LoRA adapter tensors, plus adaptor/decoder weights.
Phase 3 stores only MTP delta weights and reloads the Phase 2 base first.
The current `best.pt`, `config.yaml`, and `train.log` are also updated during
epoch uploads. At the end of each phase, the trainer pushes final durable
artifacts:

```text
checkpoints/<phase>/best.pt
checkpoints/<phase>/final.pt
checkpoints/<phase>/config.yaml
checkpoints/<phase>/train.log
checkpoints/<phase>/checkpoint_manifest.json
```

Local epoch checkpoints are cleaned up after upload, keeping only the newest
one by default. The epoch history remains on Hugging Face.

Canonical HF checkpoint paths:

```text
checkpoints/h200_phase1_warmup/best.pt
checkpoints/h200_phase2_lora_encoder/best.pt
checkpoints/h200_phase3_mtp_self_distill/best.pt
```

Fallback behavior:

- `train_phase2.sh` first looks for local Phase 1 `best.pt`; if missing, it
  downloads `checkpoints/h200_phase1_warmup/best.pt` from HF.
- `train_phase3.sh` first looks for local Phase 2 `best.pt`; if missing, it
  downloads `checkpoints/h200_phase2_lora_encoder/best.pt` from HF.
- `evaluate_phase.sh` and `evaluate_full_test.py` also fallback to HF for the
  evaluated checkpoint. For Phase 3 they also fallback to the Phase 2 base
  checkpoint passed via `--base_checkpoint`.

Downloaded checkpoints are cached under `runs/hf_checkpoints` by default. You
can change this with `HF_CHECKPOINT_CACHE`.

## Full Server Pipeline

The dataset is expected at:

```text
llm2seq_h200/wikilingua/train.json
llm2seq_h200/wikilingua/val.json
llm2seq_h200/wikilingua/test.json
```

Run preparation, phase 1, phase 2, phase 3, full test evaluation after each
phase, and optional final HF upload of evaluation artifacts:

```bash
bash llm2seq_h200/scripts/run_h200_pipeline.sh
```

The script automatically loads `llm2seq_h200/env.txt` when that file exists.
You can also point to a different file with `ENV_FILE=/path/to/env.txt`.

Default evaluation folders:

```text
llm2seq_h200/eval_outputs/full_test_phase1_main
llm2seq_h200/eval_outputs/full_test_phase2_main
llm2seq_h200/eval_outputs/full_test_phase3_main
llm2seq_h200/eval_outputs/full_test_phase3_mtp_verified
llm2seq_h200/eval_outputs/phase3_speed_comparison
```

`EVAL_LIMIT=-1` means full test set. For a smoke run, set a smaller value,
for example `EVAL_LIMIT=32`.

## Full Test Evaluation

Evaluate a checkpoint on the full test set:

```bash
python llm2seq_h200/scripts/evaluate_full_test.py \
  --config llm2seq_h200/configs/phase3_mtp_self_distill_4096.yaml \
  --checkpoint runs/h200_llm2seq_phase3_mtp_self_distill/best.pt \
  --base_checkpoint runs/h200_llm2seq_phase2_lora_encoder/best.pt \
  --test_file llm2seq_h200/data/processed/test.jsonl \
  --output_dir llm2seq_h200/eval_outputs/full_test_phase3_main \
  --decode_mode autoregressive
```

Evaluate Phase 3 with main-head-constrained MTP decoding:

```bash
python llm2seq_h200/scripts/evaluate_full_test.py \
  --config llm2seq_h200/configs/phase3_mtp_self_distill_4096.yaml \
  --checkpoint runs/h200_llm2seq_phase3_mtp_self_distill/best.pt \
  --base_checkpoint runs/h200_llm2seq_phase2_lora_encoder/best.pt \
  --test_file llm2seq_h200/data/processed/test.jsonl \
  --output_dir llm2seq_h200/eval_outputs/full_test_phase3_mtp_verified \
  --decode_mode mtp_verified
```

Outputs:

```text
<eval_dir>/predictions.jsonl
<eval_dir>/metrics.json
<eval_dir>/eval_run_info.json
```

Metrics include:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- BLEU
- chrF
- output/reference length ratio
- source/prediction compression ratio
- empty, too-short, and too-long prediction rates
- repeated trigram rate
- wall-clock throughput statistics
- decode-only throughput statistics
- mean/median/p95 latency per sample
- generated tokens per second
- tokens per decode step
- peak GPU memory
- MTP acceptance rate, average accepted draft length, and average emitted
  tokens per verifier step for `decode_mode=mtp_verified`

`predictions.jsonl` stores source, reference, and prediction for every test
example, plus per-sample latency and generated-token count. For MTP evaluation
it also stores per-sample MTP acceptance metrics, so you can copy it back to
your machine even if the server is later stopped.

Evaluation uses deterministic generation by default:

```text
do_sample: false
temperature: 0.0
min_new_tokens: 32
max_new_tokens: 256
no_repeat_ngram_size: 3
repetition_penalty: 1.15
```

For optional semantic scoring, add `--compute_bertscore`. It is slower and uses
`xlm-roberta-large` by default.

Phase 1, Phase 2, and the main Phase 3 quality evaluation use autoregressive
main-head generation. Phase 3 also has a separate `mtp_verified` evaluation.
In that mode, MTP heads draft future tokens, but every draft is checked by the
main head before acceptance. The verifier applies the same deterministic
decoding constraints as autoregressive generation: `min_new_tokens`,
`repetition_penalty`, and `no_repeat_ngram_size`. This makes the MTP evaluation
appropriate for speed/acceptance measurement without silently changing the
summary model being evaluated.

For MTP, `speedup_vs_autoregressive` and `tokens_per_decode_step` measure the
reduction in verifier steps. The real end-to-end speed should be read from
`decode_generated_tokens_per_second`, `latency_seconds_mean`, and
`latency_seconds_p95`, because this verifier path is intentionally correctness
first and is not yet a fully KV-cache-optimized serving kernel.

Phase 3 evaluation enables an adaptive speed guard by default. After an early
MTP probe, if verified MTP is not emitting enough tokens per step to offset
draft and verifier overhead, decoding falls back to the main autoregressive
path for the rest of that sample. This preserves the same greedy main-head
output while avoiding the worst slowdown cases.

After Phase 3, the pipeline also writes:

```text
llm2seq_h200/eval_outputs/phase3_speed_comparison/speed_comparison.json
```

That file directly answers how much faster D-MTP is than the Phase 3 main-head
decoder. The most important fields are:

- `real_decode_tokens_per_second_speedup`: measured D-MTP token throughput
  divided by measured main-head token throughput.
- `latency_mean_speedup`: main-head mean latency divided by D-MTP mean latency.
- `latency_p95_speedup`: main-head p95 latency divided by D-MTP p95 latency.
- `mtp_theoretical_decode_step_speedup`: reduction in verifier decode steps.
- `quality_delta`: ROUGE/BLEU/chrF difference between main-head and
  `mtp_verified` outputs.

## Push Artifacts to Hugging Face

```bash
python llm2seq_h200/scripts/push_to_hf.py \
  --folder runs/h200_llm2seq_phase1_warmup \
  --folder runs/h200_llm2seq_phase2_lora_encoder \
  --folder runs/h200_llm2seq_phase3_mtp_self_distill \
  --folder llm2seq_h200/eval_outputs/full_test_phase1_main \
  --folder llm2seq_h200/eval_outputs/full_test_phase2_main \
  --folder llm2seq_h200/eval_outputs/full_test_phase3_main \
  --folder llm2seq_h200/eval_outputs/full_test_phase3_mtp_verified \
  --folder llm2seq_h200/eval_outputs/phase3_speed_comparison \
  --folder llm2seq_h200/logs \
  --commit_message "Upload H200 LLM2Seq run"
```

The manual push script skips local `checkpoint_*.pt` optimizer-resume files by
default, because trainer-uploaded `best.pt`, `final.pt`, and `epoch_*.pt` are
the trainable-only model-weight artifacts you normally want on Hugging Face.

## Notes

- Default encoder is `McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp` for compatibility.
- `encoder_torch_dtype: bfloat16` is intended for H200/A100/H100-class GPUs.
- The H200 configs intentionally keep VRAM headroom instead of maxing out the
  140GB device. A run that sits around 100GB is healthier than a run that dies
  during validation, checkpointing, or allocator fragmentation.
- Phase 2 uses LoRA encoder adaptation rather than full encoder fine-tuning.
- For 8B encoders or multi-GPU H200, the current trainer should eventually be
  upgraded to FSDP/DeepSpeed.

## Evaluation Results: LLM2Seq (Phase 2) vs T5Gemma-1B

The table below compares the performance of **LLM2Seq (Phase 2 - LoRA Encoder)** against **T5Gemma-1b-1b (LoRA)** on the WikiLingua summarization test set (3,901 examples).

| Metric | LLM2Seq (Phase 2) | T5Gemma (1B-1B) |
| :--- | :--- | :--- |
| **ROUGE-1** | **48.36** | 33.08 |
| **ROUGE-L** | **29.05** | 21.86 |
| **chrF** | 15.45 | **28.60** |
| **Mean Prediction Words** | **29.1** | 211.8 |
| **Too Short Rate (%)** | 33.56% | 0.15% |
| **Too Long Rate (%)** | **5.46%** | 95.46% |
| **Latency Mean (s)** | 0.69s | **0.33s** |
| **Peak VRAM** | **~6 GB** | ~23.5 GB |

*Note: The reference summaries have a mean length of 51.8 words.*

### Analysis & Observations
1. **Summary Quality (ROUGE)**: LLM2Seq significantly outperforms T5Gemma in capturing the core meaning, achieving a ROUGE-1 score of 48.36 compared to T5Gemma's 33.08.
2. **Length Control & Hallucination**: T5Gemma suffers from a critical issue where it fails to generate the EOS (End of Sequence) token, leading to a "Too Long Rate" of 95.46%. It generates an average of 212 words per summary, essentially rambling until it hits the `max_new_tokens` limit. In contrast, LLM2Seq learned the summarization task structure much better, successfully stopping generation and producing concise summaries (29.1 words on average).
3. **Hardware Efficiency**: LLM2Seq is exceptionally lightweight, peaking at just ~6.1 GB VRAM during evaluation, while T5Gemma demands over 23 GB VRAM for the same batch size.
4. **Conclusion**: LLM2Seq serves as a highly capable and efficient baseline, perfectly positioning it for the MTP Self-Distillation phase (Phase 3) to further accelerate inference without sacrificing summary quality.
