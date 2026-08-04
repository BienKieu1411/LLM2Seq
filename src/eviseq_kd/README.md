# EviSeq-KD

`eviseq_kd` is a self-contained sibling package. It does not import or require the separate `src/eviseq` package. Its student graph is vendored under `eviseq_kd.student`, so both projects can be run independently.

The default implementation is offline, gold-anchored full-objective KD with a
top-k-plus-tail representation of the teacher vocabulary distribution:

```text
L = L_EviSeq
  + 0.30 * L_sequence_KD
  + 0.30 * (0.5 * L_gold_prefix_KL + 0.5 * L_pseudo_prefix_KL)
```

`L_sequence_KD` trains on the Qwen3-4B beam output as a hard sequence target,
following Kim & Rush (2016), [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/).
The two KL terms transfer the teacher's token distribution with
`T² * KL(teacher || student)` at temperature `T=2.0`. Cache schema v3 stores
the top-k logits plus one full-vocabulary log normalizer per token. Training
therefore computes an exact KL over `K + 1` buckets: each cached teacher token
is one bucket and all remaining vocabulary items form an `OTHER` bucket. This
preserves probability mass outside the top-k without storing a full
151k-vocabulary tensor. Gold evidence labels remain attached only to the gold
branch.

Gold and pseudo trajectories each use one encoder-decoder forward. When soft
KD is enabled, that same forward returns the full student logits needed by
both CE and KD; it does not repeat the complete model merely to recover
logits.

The KD configs retain the full EviSeq task/model/bridge/objective/training/data,
generation, checkpoint, benchmark, reporting, and limit settings. They add an
explicit `training.distillation` block for the teacher model, cache split and
limits, teacher generation controls, sequence/logit KD weights, temperature,
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

The cache builder stores the pseudo sequence, gold and pseudo top-k teacher
logits, full-vocabulary log normalizers, EOS-aligned token rows, source hash,
vocabulary metadata, and tokenizer fingerprint. Training refuses to consume
the cache if the decoder tokenizer, vocabulary identity, KD temperature, or
cache schema does not match.

Training validates that every cached target belongs to the current source
text and that the cache teacher/split/top-k settings match the configuration.
If a cache was created from another dataset snapshot, teacher, tokenizer, or
KD schema, rebuild it explicitly:

```bash
python3 src/eviseq_kd/run.py \
  --config src/eviseq_kd/configs/wikilingua_kd.yaml \
  --force-rebuild-cache
```

Version-1/2 caches remain inspectable, but logit KD intentionally requires a
version-3 rebuild so the loss cannot silently discard student probability mass
outside the teacher top-k.

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
