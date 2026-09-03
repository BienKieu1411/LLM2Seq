# LLM2Seq

LLM2Seq contains the maintained EviSeq text-to-text training pipeline. EviSeq
combines a pretrained source encoder, an evidence bridge and a pretrained
causal decoder in one trainable graph:

```text
source encoder -> evidence bridge -> causal decoder
```

The bridge maps encoder memory to decoder coordinates and adds a learned
source-unit attention prior. Evidence labels and contrastive losses are
training-only; the optional DualBridge prompt route is target-free and can be
reused at inference. Inference remains one encoder, one bridge and one decoder
with greedy generation.

## Project layout

```text
src/eviseq_v2/
├── core/           data, modeling, training and evaluation packages
├── configs/        model, task and reusable task-template YAML files
├── scripts/        source-tree launchers and data preparation commands
├── tests/          unit and integration tests
├── docs/           method notes
└── run.py          command-line entry point
```

The other top-level source directories are retained for reproducibility of
earlier experiments. New experiments should use `src/eviseq_v2`.

## Environment

Use the project environment and make model paths available locally before
training. The launchers do not issue model or dataset download commands.

```bash
source /absolute/path/to/bienkieu_env/bin/activate
cd src/eviseq_v2
python -m pip install -e .
```

## Configure and prepare data

Copy a template from `src/eviseq_v2/configs/templates/`, then set model names,
JSONL fields, paths, sequence lengths and training hyperparameters. Each input
record must provide a source field, a target field and an optional stable id.

For the built-in biomedical converters:

```bash
cd src/eviseq_v2
bash scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
bash scripts/run.sh prepare-arxiv /absolute/path/to/arxiv
bash scripts/run.sh prepare-cnndm /absolute/path/to/cnndm
```

The PubMed and ArXiv converters preserve supplied sentence-index labels. The
preparation step rejects duplicate ids or source texts across splits unless
`EVISEQ_ALLOW_CROSS_SPLIT_CONTENT=true` is explicitly set for debugging.

## Train

```bash
cd src/eviseq_v2
bash scripts/run.sh train configs/tasks/wikilingua.yaml --overwrite-output-dir
bash scripts/run.sh pceb-pubmed --overwrite-output-dir
```

Training writes `resolved_config.yaml`, `last.pt`, optional per-epoch
checkpoints and a validation-selected `best.pt` in the configured output
directory.

## Evaluate

```bash
python run.py evaluate \
  --config runs/eviseq/my_task/resolved_config.yaml \
  --checkpoint runs/eviseq/my_task/last.pt \
  --output runs/eviseq/my_task/test_predictions.jsonl \
  --split test --batch-size 96 --resume
```

Predictions are flushed after every completed batch, so an interrupted run can
resume from the existing JSONL. Built-in metrics are `rouge`, `exact_match`
and `token_f1`. Perl ROUGE-1.5.5 is available through the separate
`rouge155` command when `PYROUGE_HOME_DIR` is set.

## Continue training and optional KD

`--init-checkpoint` initializes a new run from an existing EviSeq model. The
optimizer and epoch counters start fresh; use the saved resolved configuration
to preserve the model and data protocol.

An optional online gold-prefix KD phase uses a local teacher with the same
tokenizer vocabulary. Set `online_kd.enabled: true` in the resolved config and
run:

```bash
bash scripts/run.sh kd \
  runs/eviseq/my_task/resolved_config.yaml \
  runs/eviseq/my_task/last.pt \
  runs/eviseq/my_task_kd \
  --teacher-model /absolute/path/to/Qwen3-4B \
  --epochs 1 --overwrite-output-dir
```

## Verification

```bash
bash scripts/run.sh test
```
