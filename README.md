# LLM2Seq

> **Evidence-aware text-to-text generation with pretrained encoder and causal-decoder language models.**

LLM2Seq is a research repository for building and evaluating **EviSeq**, a text-to-text generation pipeline that combines a pretrained source encoder, an evidence-aware bridge, and a pretrained causal decoder in a single trainable architecture.

```text
Input document
     │
     ▼
Pretrained Source Encoder
     │
     ▼
Evidence Bridge
     │
     ▼
Pretrained Causal Decoder
     │
     ▼
Generated text
```

The bridge projects encoder representations into the decoder space and learns a source-unit attention prior. Evidence supervision and contrastive objectives are used during training, while inference keeps the architecture simple: **one encoder + one bridge + one decoder** with autoregressive generation.

The actively maintained implementation lives in [`src/eviseq_v2`](src/eviseq_v2). Other source directories are retained for reproducibility of earlier experiments.

## Highlights

- **Pretrained encoder + causal LLM decoder** in a unified text-to-text model.
- **Evidence-aware bridge** for connecting source representations to decoder coordinates.
- Optional evidence supervision and contrastive training objectives.
- Optional **DualBridge** prompt route that remains target-free at inference time.
- Config-driven training and evaluation with reusable YAML task templates.
- Built-in data preparation utilities for **PubMed**, **ArXiv**, and **CNN/DailyMail**.
- Checkpointing with resolved experiment configurations for reproducibility.
- Resumable evaluation that continuously flushes predictions to JSONL.
- Optional online **knowledge distillation (KD)** from a local teacher model.
- Unit and integration tests included in the maintained implementation.

## Repository Structure

```text
LLM2Seq/
├── App/                    # Application/demo-related code
├── Paper/                  # Paper assets
├── Slide/                  # Presentation materials
├── Technical_Report/       # Technical reports
├── deploy/                 # Deployment-related files
├── src/
│   └── eviseq_v2/          # Maintained EviSeq implementation
│       ├── configs/        # Model, task and template YAML files
│       ├── core/           # Data, modeling, training and evaluation
│       ├── datasets/       # Dataset-related resources
│       ├── docs/           # Method and implementation notes
│       ├── scripts/        # Training/data-preparation launchers
│       ├── tests/          # Unit and integration tests
│       ├── run.py          # CLI entry point
│       ├── pyproject.toml  # Python package configuration
│       └── requirements.txt
├── Makefile
└── README.md
```

## Requirements

The maintained EviSeq package requires **Python 3.10+** and uses the following main dependencies:

- PyTorch `>=2.6`
- Transformers `>=5.2,<6`
- Accelerate `>=1.2`
- Safetensors `>=0.4`
- PyYAML `>=6.0`
- NumPy `>=1.24`
- Rouge `==1.0.0`

Development dependencies include `pytest` and `ruff`.

## Installation

Clone the repository and install the maintained package in editable mode:

```bash
git clone https://github.com/BienKieu1411/LLM2Seq.git
cd LLM2Seq/src/eviseq_v2

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

> Model checkpoints and datasets are expected to be available locally. The training launchers do not automatically download models or datasets.

## Configuration

Experiments are configured through YAML files under:

```text
src/eviseq_v2/configs/
```

A good starting point is a reusable template from:

```text
src/eviseq_v2/configs/templates/
```

Configure the experiment with the appropriate:

- encoder and decoder model paths/names,
- dataset JSONL paths,
- source and target fields,
- sequence-length limits,
- optimization settings,
- output directory,
- evidence/contrastive-loss options.

Each input record should contain a source field, a target field, and optionally a stable ID.

## Data Preparation

Built-in preparation commands are available for PubMed, ArXiv, and CNN/DailyMail:

```bash
cd src/eviseq_v2

bash scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
bash scripts/run.sh prepare-arxiv /absolute/path/to/arxiv
bash scripts/run.sh prepare-cnndm /absolute/path/to/cnndm
```

The PubMed and ArXiv converters preserve provided sentence-index evidence labels.

To reduce accidental evaluation leakage, the preparation pipeline rejects duplicate IDs or source texts across splits. For debugging only, this can be overridden with:

```bash
export EVISEQ_ALLOW_CROSS_SPLIT_CONTENT=true
```

## Training

Run a configured task with:

```bash
cd src/eviseq_v2
bash scripts/run.sh train configs/tasks/wikilingua.yaml --overwrite-output-dir
```

A built-in PubMed experiment can also be launched with:

```bash
bash scripts/run.sh pceb-pubmed --overwrite-output-dir
```

A training run writes artifacts such as:

```text
resolved_config.yaml
last.pt
best.pt
```

Optional per-epoch checkpoints may also be produced. `best.pt` is selected using validation performance.

## Evaluation

Evaluate a trained checkpoint with:

```bash
cd src/eviseq_v2

python run.py evaluate \
  --config runs/eviseq/my_task/resolved_config.yaml \
  --checkpoint runs/eviseq/my_task/last.pt \
  --output runs/eviseq/my_task/test_predictions.jsonl \
  --split test \
  --batch-size 96 \
  --resume
```

Predictions are flushed after each completed batch, allowing interrupted evaluations to resume from an existing JSONL output.

Built-in metrics include:

- ROUGE
- Exact Match
- Token F1

Perl **ROUGE-1.5.5** is also supported through the separate `rouge155` command when `PYROUGE_HOME_DIR` is configured.

## Continue Training

Use `--init-checkpoint` to initialize a new run from an existing EviSeq checkpoint.

The model weights are restored, while optimizer state and epoch counters start fresh. Reusing the saved `resolved_config.yaml` is recommended to preserve the original model and data protocol.

## Knowledge Distillation

EviSeq supports an optional online gold-prefix knowledge-distillation phase using a local teacher model with the same tokenizer vocabulary.

Enable KD in the resolved configuration:

```yaml
online_kd:
  enabled: true
```

Then run:

```bash
cd src/eviseq_v2

bash scripts/run.sh kd \
  runs/eviseq/my_task/resolved_config.yaml \
  runs/eviseq/my_task/last.pt \
  runs/eviseq/my_task_kd \
  --teacher-model /absolute/path/to/Qwen3-4B \
  --epochs 1 \
  --overwrite-output-dir
```

## Testing

Run the test suite with:

```bash
cd src/eviseq_v2
bash scripts/run.sh test
```

Or, after installing the development dependencies:

```bash
pytest
```

## CLI

Installing the project exposes the `eviseq` command-line entry point:

```bash
eviseq --help
```

You can also use the source entry point directly:

```bash
python run.py --help
```

## Research and Reproducibility

This repository contains paper, slide, technical-report, experiment, and implementation artifacts for the LLM2Seq/EviSeq research project. The maintained `eviseq_v2` pipeline should be used for new experiments, while older source directories are kept to help reproduce previous work.

When reporting experimental results, keep the generated `resolved_config.yaml` together with the corresponding checkpoint and prediction files so the complete experiment protocol can be reconstructed.

## Contributing

Contributions, bug reports, and experiment improvements are welcome. For code changes, please run the tests before submitting changes:

```bash
bash src/eviseq_v2/scripts/run.sh test
```

## License

No explicit license file is currently included in this repository. Unless a license is added, the repository should not be assumed to grant reuse, redistribution, or modification rights beyond those provided by GitHub's Terms of Service.
