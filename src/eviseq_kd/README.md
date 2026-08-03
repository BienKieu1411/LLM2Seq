# EviSeq-KD

`eviseq_kd` is a self-contained sibling package. It does not import or require the separate `src/eviseq` package. Its student graph is vendored under `eviseq_kd.student`, so both projects can be run independently.

The default implementation is offline, gold-anchored sequence-level knowledge distillation:

```text
L = L_gold_CE + salience + evidence_InfoNCE + 0.30 * L_pseudo_CE
```

`L_pseudo_CE` trains the EviSeq decoder on text generated offline by Qwen3-4B, then retokenizes that text with the student tokenizer. Gold evidence labels remain attached only to the gold branch. This follows Kim & Rush (2016), [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/). No tokenizer compatibility check is needed for this text-only KD path.

The KD configs retain the full EviSeq task/model/bridge/objective/training/data,
generation, checkpoint, benchmark, reporting, and limit settings. They add an
explicit `training.distillation` block for the teacher model, cache split and
limits, teacher generation controls, sequence-KD weight, optional logit-KD
weight, and temperature. The inherited EviSeq objective settings are kept as
configured: document-level InfoNCE is disabled by default while evidence
contrastive learning remains enabled.

## Run directly from source

No editable installation is required. Run the source launcher directly:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml
```

The `pyproject.toml` file remains optional packaging metadata only.

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

The trainer automatically builds the configured teacher cache when it is
missing, then reuses it on later runs. Thus the normal training command is a
single command:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml \
  --output-dir runs/eviseq_kd/wikilingua_qwen3_4b_teacher
```

No cache command or tokenizer check is needed for the normal run.

Training validates that every cached pseudo-target belongs to the current
source text and that the cache teacher/split match the configuration. This is
source-level cache validation, not a tokenizer fingerprint requirement. If a
cache was created from another dataset snapshot or an older teacher, rebuild
it explicitly:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml \
  --force-rebuild-cache
```

## A100 smoke run

From the repository root, run this short real-GPU flow directly from source:

```bash
export CUDA_VISIBLE_DEVICES=0
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/smoke_a100.yaml \
  --overwrite-output-dir
test -s runs/eviseq_kd/smoke_a100/last.pt
```

The launcher automatically builds the teacher cache if it is absent, then
reuses it on later runs. For the smoke config, all teacher generations are in
one file: `runs/eviseq_kd/cache/smoke_a100_train_qwen3_4b.jsonl`.

To inspect that reusable file:

```bash
python3 src/eviseq_kd/check_cache.py \
  --cache runs/eviseq_kd/cache/smoke_a100_train_qwen3_4b.jsonl
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
