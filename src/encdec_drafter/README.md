# T5Gemma2 inference drafter

This package is the first runtime implementation from the T5Gemma2 acceleration plan. It keeps the target checkpoint frozen and uses its native encoder, decoder, tokenizer, and output head. The first runtime supports one request at a time, greedy generation, full target source attention, a compact parallel drafter, and transactional target-cache rollback.

## Requirements

Use the project environment on Mac for code work. The target checkpoint and tokenizer must exist locally; remote model loading is disabled. The Mac does not need a copy of the target model to edit this package. Runtime expects PyTorch with the appropriate device build, Transformers 5.14.1, and PyYAML. Avoid allowing package installation to replace a working CUDA PyTorch build.

Install the package without replacing the environment's PyTorch or Transformers build:

```bash
python -m pip install --no-deps -e src/encdec_drafter
```

## Prepare a local config

The sample config follows the repository's current PubMed T5Gemma2 evaluation settings, including its prompt, detokenization, 4096-token source cap, 512-token output cap, minimum output length, repetition penalty, and trigram blocking. Point `target.checkpoint` at the same locally available fine-tuned checkpoint used by the baseline. For another dataset, copy the prompt, source/output limits, and greedy decoding settings from its matching file under `src/T5Gemma/configs/`. Data paths in YAML resolve relative to that YAML file; paths passed on the command line resolve relative to the current shell directory.

```bash
python -m encdec_drafter profile --config src/encdec_drafter/configs/t5gemma2.yaml
python -m encdec_drafter collect-traces --config src/encdec_drafter/configs/t5gemma2.yaml --output "$PWD/runs/encdec_drafter/traces.jsonl" --limit 128
python -m encdec_drafter train --config src/encdec_drafter/configs/t5gemma2.yaml --traces "$PWD/runs/encdec_drafter/traces.jsonl"
python -m encdec_drafter benchmark --config src/encdec_drafter/configs/t5gemma2.yaml --draft-checkpoint "$PWD/runs/encdec_drafter/last.pt" --limit 32 --repeats 3
```

On Mac, activate `~/bienkieu_env` before running Python. If the drafter is trained on one machine and evaluated against an identical copy of the target checkpoint on another, set the same explicit `target.identity` in both configs; otherwise the local checkpoint metadata fingerprint must match.

The target remains the verifier. The drafter proposes a block from the final source memory, the pending token, and only already-processed target features. Unsupported sampling and beam search fail explicitly. The initial cache stores full decoder history so that rejection can be rolled back safely even for local-attention layers; this uses more memory than the native sliding cache and is a correctness implementation, not a claim of speedup.

## Colab smoke pilot

`colab/t5gemma2_pubmed_l4_colab_cli.ipynb` runs the frozen base `google/t5gemma-2-1b-1b` checkpoint on an L4 through Colab CLI. It downloads weights only to the Colab VM and uses 8 train examples, 4 validation examples, a 4096-token source cap, 64 generated tokens, and 32 drafter updates. The saved run is in `runs/colab_cli_l4_20261021_230721/`.

This smoke run did not pass the correctness gate: target-only decoding matched native generation, but speculative generation had 50% exact sequence parity over four examples. The median raw speed ratio was 0.874x, and proposal acceptance was below 1% across the timed sample. Since token parity failed, the timing is diagnostic only and is not evidence of acceleration. The detailed token IDs and first mismatch positions are in the run's `benchmark/*/requests.jsonl`.

## Gated PubMed retraining

`diagnose-verifier` compares same-prefix target logits from full-prefix, token-by-token cached, and block-cached execution. It also checks that cached target-only generation matches the native Transformers generation. A non-passing gate exits with code 2, so `scripts/retrain_gated.sh` will stop before collecting traces or training.

The new `configs/t5gemma2_pubmed_margin.yaml` is for the base `google/t5gemma-2-1b-1b` target. Point `target.checkpoint` at a local copy on the GPU machine first. No target weights are downloaded by this package or the shell script. The script defaults to 1,000 PubMed training traces, 5,000 drafter steps, and 128 validation requests. Override `TRAIN_LIMIT`, `TRAIN_STEPS`, `BENCHMARK_LIMIT`, and `PYTHON` as needed.

The drafter uses 1,000 cross-entropy warm-up updates, then a sigmoid top-1 logit-margin surrogate for accepted prefix length, with a small cross-entropy term. The runtime proposes tokens with `argmax`, so this is an adaptation for greedy proposals, not the paper's probability-mass EAL objective and not a calibrated acceptance probability. It is not the sampling-based WTV objective. Run it from the repository root:

```bash
bash src/encdec_drafter/scripts/retrain_gated.sh \
  src/encdec_drafter/configs/t5gemma2_pubmed_margin.yaml \
  src/eviseq_new/datasets/pubmed/train.jsonl \
  src/eviseq_new/datasets/pubmed/validation.jsonl
```

The script saves the parity report, teacher traces, drafter checkpoint, and benchmark outputs under one timestamped run directory. Benchmarking fails if output IDs do not exactly match the native target. A speedup still must be assessed from the reported timing distribution; acceptance rate alone is not a performance claim.

## Current limits

The runtime is deliberately single-request and greedy. It does not yet include a custom merged-attention kernel, CUDA graphs, a serving scheduler, adaptive source budgets, sampling, or multi-GPU execution. No speedup claim is supported until speculative decoding achieves exact token parity and beats native cached `generate` on a representative validation set.
