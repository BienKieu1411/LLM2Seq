# Decoder-only baseline suite

This folder contains the full-finetuning controls used to compare EviSeq with
Qwen3, Llama 3 and Nemotron-Labs-Diffusion.  All pairs share the same prepared
JSONL splits, seed, causal prompt masking and greedy benchmark decoding.  The
source instruction is the T5Gemma instruction for each task; no chat template
is applied.  Every target token, including EOS, is supervised while all prompt
tokens are `-100` in the causal loss.

Nemotron-Labs-Diffusion is loaded through `AutoModel` and explicitly switched
to its autoregressive paradigm.  This keeps its diffusion denoising objective
out of the decoder-only comparison; its custom AR cache loop is used only at
evaluation time.

## One GPU, sequential matrix

The runner exposes exactly one GPU and starts a new Python process for every
model/dataset pair.  The next run starts only after training and evaluation of
the previous run have exited, which releases model and optimizer memory.

```bash
cd src/decoder_baselines
GPU_ID=0 \
QWEN3_0_6B_PATH=/models/Qwen3-0.6B \
QWEN3_8B_PATH=/models/Qwen3-8B \
QWEN3_4B_PATH=/models/Qwen3-4B \
LLAMA3_8B_PATH=/models/Llama-3.1-8B \
LLAMA3_3B_PATH=/models/Llama-3.2-3B-Instruct \
NEMOTRON_DIFFUSION_8B_PATH=/models/Nemotron-Labs-Diffusion-8B-Base \
NEMOTRON_DIFFUSION_3B_PATH=/models/Nemotron-Labs-Diffusion-3B-Base \
bash scripts/run_suite.sh --models qwen3_0_6b,qwen3_8b,qwen3_4b,llama3_8b,llama3_3b,nemotron_diffusion_8b,nemotron_diffusion_3b --datasets pubmed,arxiv
```

The model paths are mandatory local directories.  The runner sets
`HF_HUB_OFFLINE=1` and never downloads or resolves a Hugging Face repository
ID.  Set each `*_PATH` variable (or set `MODEL_ROOT` once when its child
directory names match `local_dir`) before starting.  Add `cnndm`, `wikilingua`
or `lrsum` to `--datasets`
after their three prepared files exist under `src/eviseq_new/datasets/`, or
override `data_root` in a copied suite YAML.  Use `--dry-run` to materialize
and inspect all commands without loading a model.  Use `--continue-on-error`
only when a failed run should not stop the matrix.

The arXiv recipe uses a `9216`-token total context so its `8096`-token source
budget can coexist with the `512`-token target and the instruction overhead.

Training and evaluation batch sizes are configured independently per model:
larger models use smaller evaluation batches, Qwen3-0.6B uses a larger batch,
and Nemotron uses `batch_size: 1` for its autoregressive cache loop.  Each run
is written to `runs/decoder_baselines/<model>__<dataset>/` with its
resolved config, `final_model/`, `trainer_state.json`, predictions and metrics.
Training enables length-grouped sampling by default, so examples with similar
source/target lengths share a batch and dynamic padding does less work.  The
sampler uses a cheap character-length estimate; tokenization and labels are
unchanged. Persistent workers and prefetching keep the tokenizer pipeline warm
between epochs.
The final evaluation uses `temperature: 0`, `top_k: 0`, `top_p: 1`; candidates
can be generated later by editing a copied run config and enabling sampling.
Evaluation runs automatically after training on the `test` split.  To evaluate
an existing checkpoint without retraining, call `decoder_baselines.evaluate`
with its resolved config and `final_model/` path; the same decode controls are
used unless the copied config explicitly enables sampling.

The seven model IDs in the bundled matrix are `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`,
`Qwen/Qwen3-4B`, `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-3B-Instruct`,
`nvidia/Nemotron-Labs-Diffusion-8B-Base` and
`nvidia/Nemotron-Labs-Diffusion-3B-Base`.  If a local directory uses another
name, set the corresponding `*_PATH` variable; the suite records both the
canonical model ID and the resolved local path.
