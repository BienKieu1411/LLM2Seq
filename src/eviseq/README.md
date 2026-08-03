# EviSeq

EviSeq builds a text-to-text encoder-decoder from two pretrained components:
an embedding-oriented LLM encoder, an evidence-aware bridge, and a causal LLM
decoder. The saved model has one encoder, one bridge, and one decoder.

The package is not tied to one dataset or to continual training. Summarization
is the research setup, but translation, question answering, data-to-text,
rewriting, instruction generation, and label-as-text classification use the
same pipeline. Continual fine-tuning is only one optional checkpoint
initialization mode.

## Install

Activate your environment, then install the local package:

```bash
cd src/eviseq
pip install -e .
```

No command in this project uploads a model or dataset.

## Project structure

```text
eviseq/
├── core/
│   ├── modeling/       # encoder, evidence bridge, decoder, attention
│   ├── training/       # trainer, objectives, checkpoints
│   ├── data/           # JSONL datasets, collators, preparation
│   ├── evaluation/     # greedy generation, metrics, evaluator
│   ├── cli.py
│   └── configuration.py
├── configs/
│   ├── tasks/          # WikiLingua, CNN/DM, PubMed, smoke
│   ├── models/         # encoder/model variants
│   ├── ablations/
│   └── templates/      # reusable task templates
├── datasets/
├── docs/
├── scripts/
├── tests/
└── pyproject.toml
```

The installed Python namespace is `eviseq`; `pyproject.toml` maps it to the
clearly named `core/` directory without repeating the project name.

## Configure a task

Start from one of:

```text
configs/templates/custom_summarization.yaml
configs/templates/custom_text2text.yaml
configs/templates/translation.yaml
configs/templates/question_answering.yaml
configs/templates/classification.yaml
```

EviSeq accepts JSONL with arbitrary field names. Map a single input field:

```yaml
data:
  source_field: article
  target_field: summary
  id_field: document_id
```

Or construct an input from several fields:

```yaml
data:
  source_template: "Question: {question}\nContext: {context}"
  target_field: answer
  id_field: uid
```

List-valued fields are joined with `data.list_separator`. All tasks are
converted internally to `source -> target` generation, so classification can
also be represented by generating the label text.

For a format that templates cannot express, expose
`map_record(row, data_config) -> {"id": ..., "source": ..., "target": ...}`:

```yaml
data:
  record_mapper: my_project.data:map_record
```

For summarization with sentence evidence, keep:

```yaml
data:
  supervise_evidence: true
objectives:
  salience_weight: 0.10
  use_evidence_contrastive: true
  evidence_contrastive_weight: 0.10
```

For a task without evidence annotations, use CE-only training:

```yaml
data:
  supervise_evidence: false
  precompute_evidence: false
objectives:
  salience_weight: 0.0
  use_evidence_contrastive: false
  evidence_contrastive_weight: 0.0
```

Built-in evaluation metrics are `rouge`, `exact_match`, and `token_f1`:

```yaml
task:
  format: text_to_text
  metrics: [exact_match, token_f1]
```

For another metric, expose a function
`score(predictions, references) -> dict[str, float]` and configure:

```yaml
task:
  metrics: []
  metric_callable: my_project.metrics:score
```

`train_file` and `validation_file` are required. `test_file` is optional, so a
new user can start experimentation before a held-out test set exists.

## Train and evaluate

Validate data mapping without loading a model:

```bash
eviseq validate-data --config configs/my_task.yaml
```

Train:

```bash
eviseq train \
  --config configs/my_task.yaml \
  --overwrite-output-dir
```

Evaluate:

```bash
eviseq evaluate \
  --config runs/eviseq/my_task/resolved_config.yaml \
  --checkpoint runs/eviseq/my_task/last.pt \
  --output runs/eviseq/my_task/test_predictions.jsonl \
  --split test
```

Training has two stages: interface warm-up and full fine-tuning. Set
`training.interface_warmup_epochs: 0` to start directly with full fine-tuning.
The canonical recipe saves `epoch_XXX.pt` after every epoch, selects `best.pt`
by minimum validation CE loss, and always writes `last.pt` at completion.
Evaluation accepts any of these complete checkpoints; test data is never used
for checkpoint selection.

## Continue from an existing EviSeq model

Use a previous checkpoint to initialize a new task or a later training run:

```bash
eviseq train \
  --config configs/new_task.yaml \
  --init-checkpoint runs/eviseq/old_task/last.pt \
  --output-dir runs/eviseq/new_task_from_old \
  --overwrite-output-dir
```

By default, the model graph must match exactly. If only optional task-training
heads differ, add `--allow-partial-init`; every matching tensor is loaded and
the skipped tensor count is logged. Optimizer state and epoch counters are
intentionally reset, so this is continual fine-tuning rather than recovery of
an interrupted optimizer.

## Existing research recipes

The research recipes are available from the repository `src` directory:

```bash
bash eviseq/scripts/run.sh smoke --overwrite-output-dir
bash eviseq/scripts/run.sh wiki --overwrite-output-dir
bash eviseq/scripts/run.sh cnndm --overwrite-output-dir
bash eviseq/scripts/run.sh pubmed --overwrite-output-dir
bash eviseq/scripts/run.sh test
```

The ready-to-run PubMed GPU queue is isolated in `scripts/gpu_0.sh` instead
of being mixed with the Python package.
