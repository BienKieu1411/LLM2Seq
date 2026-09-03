# EviSeq v2

EviSeq v2 builds a reusable text-to-text encoder-decoder. It connects a
pretrained source encoder to a causal Qwen decoder through one evidence
bridge:

```text
source encoder -> evidence bridge -> causal decoder
```

The model, data mapping, objectives, training schedule and decoding protocol
are versioned in YAML recipes. The resolved recipe is copied into each run so
that checkpoints can be evaluated only with the same model and data contract.

## Layout

```text
eviseq_v2/
├── core/
│   ├── data/           PubMed preparation, readers and collators
│   ├── evaluation/     generation and metrics
│   ├── modeling/       encoder, bridge, attention and decoder
│   ├── training/       objectives, checkpoints and training loop
│   ├── cli.py
│   └── config.py
├── configs/           task, model and reusable YAML recipes
├── datasets/           prepared local JSONL data
├── scripts/             PubMed shell entry points
├── tests/               unit and integration tests
└── run.py               source-tree launcher
```

## Prepare PubMed

```bash
bash scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
bash scripts/run.sh validate-data configs/tasks/pubmed.yaml
```

The prepared files must be `datasets/pubmed/train.jsonl`,
`validation.jsonl` and `test.jsonl`.  The PCEB run expects zero-based external
sentence labels in the `label` field.

## Train and evaluate

```bash
bash scripts/run.sh train configs/models/pplx_pubmed_pceb_corrected.yaml --overwrite-output-dir
```

For multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run.sh pceb-pubmed-ddp --overwrite-output-dir
```

The Python entrypoint is also available directly:

```bash
python3 run.py train \
  --config configs/models/pplx_pubmed_pceb_corrected.yaml \
  --output-dir runs/eviseq/pubmed_pceb_corrected
python run.py evaluate \
  --config runs/eviseq/pubmed_pceb_corrected/resolved_config.yaml \
  --checkpoint runs/eviseq/pubmed_pceb_corrected/best.pt \
  --output runs/eviseq/pubmed_pceb_corrected/best_test_predictions.jsonl \
  --split test
```

All model and data hyperparameters come from the selected YAML recipe. The
resolved recipe is emitted next to the checkpoints, and evaluation fails
closed if the supplied recipe differs from the checkpoint contract.

## Built-in objective

The corrected PCEB recipe uses the evidence route, identity-initialized bridge
correction and a target-free attention-aligned evidence objective. Optional
source/prompt InfoNCE and hard in-batch source-swap ranking are available in
the strong research recipe; source-swap is training-only, so inference remains
one encoder, one bridge and one decoder.

## Benchmark status

The relevant PubMed comparison is:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| T5Gemma | 49.580 | 21.990 | 45.463 |
| EviSeq PCEB | 49.228 | 21.644 | 45.312 |

EviSeq is currently behind by `0.352`, `0.346` and `0.151` points.  The
source-swap/InfoNCE additions are intended to close this source-utilization
gap; an actual win still requires a fresh PubMed training run.

The architecture review and paper references are in `docs/t5gemma_benchmark.md`.

## Test

```bash
bash scripts/run.sh test
```
