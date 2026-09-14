# Decoder-only baseline suite

This folder contains the full-finetuning controls used to compare EviSeq with
Qwen3, Llama 3 and Nemotron-Labs-Diffusion.  All pairs share the same prepared
JSONL splits, seed, causal prompt masking and greedy benchmark decoding.  The
source instruction is the T5Gemma instruction for each task; no chat template
is applied.  Every target token, including EOS, is supervised while all prompt
tokens are `-100` in the causal loss.

Nemotron-Labs-Diffusion is loaded through `AutoModel` and explicitly switched
to its autoregressive paradigm for fine-tuning. Evaluation uses the
OpenAI-compatible vLLM service with the Transformers backend; `--backend local`
remains available for a direct custom-AR check. This keeps its diffusion
denoising objective out of the decoder-only comparison.

## Sequential matrix with optional DDP training

The runner starts a new process for every model/dataset pair. Pairs remain
sequential, which releases model and optimizer memory before the next run. A
single visible GPU runs normal Trainer training; a comma-separated `GPU_ID`
launches one DDP worker per GPU. Evaluation uses the vLLM OpenAI-compatible
service for every model, with progress logs and incremental JSONL writes.
Nemotron-Labs-Diffusion selects vLLM's `transformers` backend because it
exposes the custom `NemotronLabsDiffusionModel` through `AutoModel`.

```bash
cd src/decoder_baselines
GPU_ID=0,1 \
QWEN3_0_6B_PATH=/models/Qwen3-0.6B \
QWEN3_8B_PATH=/models/Qwen3-8B \
QWEN3_4B_PATH=/models/Qwen3-4B \
LLAMA3_8B_PATH=/models/Llama-3.1-8B \
LLAMA3_3B_PATH=/models/Llama-3.2-3B-Instruct \
NEMOTRON_DIFFUSION_8B_PATH=/models/Nemotron-Labs-Diffusion-8B-Base \
NEMOTRON_DIFFUSION_3B_PATH=/models/Nemotron-Labs-Diffusion-3B \
bash scripts/run_suite.sh --models qwen3_0_6b,qwen3_8b,qwen3_4b,llama3_8b,llama3_3b,nemotron_diffusion_8b,nemotron_diffusion_3b --datasets pubmed,arxiv
```

For one GPU, use `GPU_ID=0`. For two GPUs, use `GPU_ID=0,1`; the script
invokes `torchrun --standalone --nproc_per_node=2` for each training run. The
training code binds rank `i` to visible device `i`, lets Transformers Trainer
own the DDP wrapper and gradient all-reduce, and writes checkpoints and
manifests only from rank zero.

The model paths are mandatory local directories.  The runner sets
`HF_HUB_OFFLINE=1` and never downloads or resolves a Hugging Face repository
ID.  Set each `*_PATH` variable (or set `MODEL_ROOT` once when its child
directory names match `local_dir`) before starting.  Add `cnndm`, `wikilingua`
or `lrsum` to `--datasets`
after their three prepared files exist under `src/eviseq_new/datasets/`, or
override `data_root` in a copied suite YAML.  Use `--dry-run` to materialize
and inspect all commands without loading a model.  Use `--continue-on-error`
only when a failed run should not stop the matrix.

Each JSONL row must contain the configured source and target fields.  An ID
field is optional: when `id_field` is absent from a row, the loader assigns a
stable file-local ID such as `row-00000001` for the prediction JSONL.

The arXiv recipe uses a `9216`-token total context so its `8096`-token source
budget can coexist with the `512`-token target and the instruction overhead.

Training and evaluation batch sizes are configured independently per model:
larger models use smaller evaluation batches and Qwen3-0.6B uses a larger
batch. With the default vLLM backend, `generation.batch_size` is the number of
tokenized prompts sent in one HTTP request; `VLLM_BATCH_SIZE` or
`--vllm-batch-size` overrides it without editing the YAML. vLLM may internally
schedule those requests in smaller GPU-safe groups. The local Nemotron
fallback still uses near-length, token-budgeted batches for its custom AR cache
loop.
Each run is written to `runs/decoder_baselines/<model>__<dataset>/` with its
resolved config, `final_model/`, `trainer_state.json`, predictions and metrics.
Training enables length-grouped sampling by default, so examples with similar
source/target lengths share a batch and dynamic padding does less work.  The
sampler uses a cheap character-length estimate; tokenization and labels are
unchanged. Persistent workers and prefetching keep the tokenizer pipeline warm
between epochs.

`per_device_train_batch_size` remains the batch on each GPU. With DDP, the
effective optimizer batch is
`per_device_train_batch_size × number_of_GPUs × gradient_accumulation_steps`;
the resolved run manifest records both the world size and this global batch.
The final evaluation uses `temperature: 0`, `top_k: 0`, `top_p: 1`; candidates
can be generated later by editing a copied run config and enabling sampling.
Evaluation runs automatically after training on the `test` split. By default,
the suite starts a local `vllm serve` process for each checkpoint,
waits for `/v1/models`, sends batched completion requests, and stops the service
after that evaluation. Set `VLLM_BASE_URL` to use an already-running service;
the service model must match the checkpoint being evaluated. Use
`VLLM_BATCH_SIZE=32` (or another value appropriate for GPU memory) to increase
the HTTP request batch size independently from the training batch size. The
Nemotron server is launched with `--model-impl transformers`,
`--enforce-eager` and the canonical model ID as `--served-model-name`.
`--enforce-eager` is required for this custom model because its `forward()`
does not expose the `inputs_embeds` argument expected by vLLM's compile wrapper.
If the installed vLLM release does not accept the backend flag, set
`VLLM_MODEL_IMPL=auto`; vLLM will select its Transformers backend from the
checkpoint's `auto_map`.

Every evaluation prints `[eval] batch ... ETA=...` and a final
`[eval] COMPLETE ...` line. It also writes `*.metrics.json` with
`started_at`, `finished_at`, `elapsed_seconds` and `examples_per_second`, while
predictions are flushed batch by batch so the output file remains observable
during a long run.

For evaluation only, the generated per-run YAML is optional. Read the model,
dataset and evaluation parameters directly from `suite.yaml`:

```bash
cd src/decoder_baselines
CUDA_VISIBLE_DEVICES=1 python3 evaluate.py \
  --suite configs/suite.yaml \
  --model qwen3_4b \
  --dataset pubmed \
  --checkpoint /models/Qwen3-4B \
  --output /runs/decoder_baselines/qwen3_4b__pubmed_base/test_predictions.jsonl \
  --split test \
  --backend vllm \
  --start-vllm-service
```

This mode uses the suite's local model path environment variable, canonical
EviSeq data files, prompt, context limits and per-model generation batch size;
it does not write a temporary config. For an existing service, replace
`--start-vllm-service` with `--no-start-vllm-service` and set
`VLLM_BASE_URL=http://host:8000/v1`. Use `--backend local` only when you need
the in-process Transformers evaluator or when the installed vLLM cannot load
the custom Nemotron checkpoint.

The seven model IDs in the bundled matrix are `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`,
`Qwen/Qwen3-4B`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B-Instruct`,
`nvidia/Nemotron-Labs-Diffusion-8B-Base` and
`nvidia/Nemotron-Labs-Diffusion-3B`. If a local directory uses another
name, set the corresponding `*_PATH` variable; the suite records both the
canonical model ID and the resolved local path.
