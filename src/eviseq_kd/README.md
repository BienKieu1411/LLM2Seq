# EviSeq-KD

`eviseq_kd` is a self-contained sibling package. It does not import or require the separate `src/eviseq` package. Its student graph is vendored under `eviseq_kd.student`, so both projects can be run independently.

The default paper configuration is online, gold-anchored full-objective KD
with a frozen Qwen3-4B teacher. For every supervised training batch, the
teacher generates its pseudo-summary and computes its soft targets inside the
training step; no teacher-generated text is written to disk or reused from an
offline cache:

```text
L = L_EviSeq
  + 0.30 * L_sequence_KD
  + 0.30 * (0.5 * L_gold_prefix_KL + 0.5 * L_pseudo_prefix_KL)
```

`L_sequence_KD` trains on the Qwen3-4B output as a hard sequence target,
following Kim & Rush (2016), [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/).
The two KL terms transfer the teacher's token distribution with
`T² * KL(teacher || student)` at temperature `T=2.0`. Cache schema v3 stores
the top-k logits plus one full-vocabulary log normalizer per token. Training
therefore computes an exact KL over `K + 1` buckets: each cached teacher token
is one bucket and all remaining vocabulary items form an `OTHER` bucket. This
preserves probability mass outside the top-k without storing a full
151k-vocabulary tensor. Gold evidence labels remain attached only to the gold
branch.

Gold and pseudo trajectories each use one student encoder-decoder forward.
The teacher is frozen, excluded from the optimizer and checkpoints, and is
run in inference mode in teacher micro-batches (`teacher_batch_size`). When
soft KD is enabled, the same student forwards return the logits needed by both
CE and KD; the student graph is not repeated merely to recover logits.

The KD configs retain the full EviSeq task/model/bridge/objective/training/data,
generation, checkpoint, benchmark, reporting, and limit settings. They add an
explicit `training.distillation` block for the teacher model, online teacher
micro-batch and context limits, sequence/logit KD weights, temperature,
gold/pseudo path mixing, and top-k width. The inherited EviSeq objective
settings are kept as configured: document-level InfoNCE is disabled by default
while evidence contrastive learning remains enabled.

## Run directly from source

No editable installation is required. Run the source launcher directly:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml
```

The `pyproject.toml` file remains optional packaging metadata only.

For a server upload, the bundled launcher resolves its own package root and
uses the active `bienkieu_env` (or `PYTHON_BIN`):

```bash
bash src/eviseq_kd/scripts/run.sh inspect pubmed
bash src/eviseq_kd/scripts/run.sh pubmed --overwrite-output-dir
bash src/eviseq_kd/scripts/run.sh evaluate-pubmed-test --batch-size 8
```

The queue wrapper trains, evaluates, and optionally runs Perl ROUGE-1.5.5:

```bash
PYROUGE_HOME_DIR=/absolute/path/to/ROUGE-1.5.5 \
CUDA_VISIBLE_DEVICES=0 \
bash src/eviseq_kd/scripts/gpu_0.sh
```

Set `KD_TASK=wiki` for WikiLingua. Set `CHECKPOINT_NAME=best.pt` to evaluate
the best checkpoint instead of `last.pt`.

If the contents of this directory were uploaded as `/content/eviseq_kd` in
Colab (so `configs/`, `datasets/`, and the inner `eviseq_kd/` are siblings),
run from `/content` instead:

```python
%cd /content
!python3 eviseq_kd/run.py \
  --config eviseq_kd/configs/smoke_a100.yaml \
  --overwrite-output-dir
```

The dataset paths in the KD configs are relative to `configs/`, so this
flattened Colab layout and the repository `src/eviseq_kd/` layout resolve to
the same `datasets/wikilingua/` files.

## Train

The normal training command is a single command. It loads the frozen Qwen3-4B
teacher lazily and runs it online during training:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml \
  --output-dir runs/eviseq_kd/wikilingua_qwen3_4b_online
```

The legacy cache builder remains available for ablations and reproducibility
checks with `training.distillation.mode: offline`, but it is not used by the
paper configs and is never silently invoked by online training.

## A100 smoke run

From the repository root, run this short real-GPU flow directly from source:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/smoke_a100.yaml \
  --overwrite-output-dir
test -s runs/eviseq_kd/smoke_a100/last.pt
```

This is a 100-example, one-warmup-epoch plus one-full-finetune-epoch flow; it is only a GPU/forward/backward/checkpoint smoke test, not a quality benchmark.

Evaluate the trained KD checkpoint on the held-out test split:

```bash
python3 src/eviseq_kd/evaluate.py \
  --config src/eviseq_kd/configs/smoke_a100.yaml \
  --checkpoint runs/eviseq_kd/smoke_a100/last.pt \
  --output runs/eviseq_kd/smoke_a100/test_predictions.jsonl \
  --split test
```

The command writes predictions and a sibling `.metrics.json` file.

To initialize from an older compatible EviSeq checkpoint, use `--init-checkpoint`. The KD loader can remap unprefixed student keys to the wrapper's internal `base.*` keys without importing the other package.

## Isolation guarantee

No KD import is added to the separate EviSeq package. The regression gate is:

```bash
git diff --exit-code HEAD -- src/eviseq
```
