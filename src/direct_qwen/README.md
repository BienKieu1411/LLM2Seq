# Direct Qwen3-0.6B control

This is the external decoder-only control for the paper, not an architectural
ablation. It full-fine-tunes all `Qwen/Qwen3-0.6B` parameters on the exact
WikiLingua files and prompt/generation contract referenced by the paper method.

The causal prompt is deterministic:

1. the exact source token segment (`source_prefix + cleaned source`, capped at
   3072 tokens, including its terminal EOS);
2. the exact decoder seed (Qwen chat template over `decoder_instruction`,
   thinking disabled, followed by `decoder_prefix`);
3. the reference summary, capped at 384 tokens including EOS.

Labels are `-100` over both prompt segments. Only reference-summary tokens
contribute to causal CE. There is no LoRA, auxiliary loss, best-checkpoint
selection, per-epoch checkpoint, Hub upload, or network fallback.

## Run

Activate `bienkieu_env`, then run from `src`:

```bash
export DIRECT_QWEN_MODEL_PATH=/absolute/local/path/to/Qwen3-0.6B  # optional cached/local path
bash direct_qwen/run.sh wiki --overwrite-output-dir
bash direct_qwen/run.sh paper-test
bash direct_qwen/run.sh rouge155 \
  runs/direct_qwen/qwen3_0_6b_full_wikilingua/last_test_predictions.jsonl
```

From the repository root, prepend `src/` to each script path, for example
`bash src/direct_qwen/run.sh wiki --overwrite-output-dir`.

`wiki` performs 14 full-finetuning epochs with physical batch 32,
accumulation 2, LR `1e-5`, FP32 master parameters and BF16 autocast. It then
evaluates validation. Test generation is a separate, explicitly locked step.

The runner sets both Transformers and Hub offline modes, and Python passes
`local_files_only=True`. A missing model fails instead of being downloaded.
