# Text quality diagnostics

The scripts in this directory keep the existing Perl ROUGE-1.5.5 evaluator
separate from semantic and factuality diagnostics. All scripts consume JSONL
rows with `prediction` and `reference`; the hallucination diagnostic also
requires `source` by default.

## BERTScore with a local model

`evaluate_bertscore.py` requires an existing Hugging Face model directory and
sets Transformers/Hugging Face offline mode before loading it. It never
downloads a model implicitly.

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_bertscore.py \
  src/T5Gemma/results/lrsum/predictions.jsonl \
  --model-path /path/to/local/bert-score-encoder \
  --num-layers 12 \
  --batch-size 16 \
  --device cuda:0 \
  --details
```

If `--num-layers` is omitted, the script reads the layer count from the local
`config.json`. When a row has multiple references, BERTScore uses the best
reference for that example, as implemented by the `bert-score` package. The
headline precision, recall and F1 values are reported on a 0--100 scale.

The evaluator patches the very-large `model_max_length` sentinel emitted by
some Transformers tokenizer configs. This avoids
`OverflowError: int too big to convert` with bert-score 0.3.x; pass
`--max-length` only when an explicit truncation limit is required.

## Paper-grounded hallucination score: AlignScore

For the paper's primary factuality result, use **AlignScore-nli_sp** from
Zha et al. (ACL 2023). AlignScore evaluates whether each prediction sentence
is supported by a source chunk, takes the best source support for each sentence,
and averages those values. Its native score is factual consistency (higher is
better); this repository also reports
`hallucination_score = 1 - alignscore_consistency`, so lower is better.

The evaluator in `evaluate_alignscore.py` follows the released AlignScore
`nli_sp` inference path while avoiding the original package's old
PyTorch/Transformers pins. Both the AlignScore backbone directory and the
trained `.ckpt` must already exist locally. Transformers offline mode and
`local_files_only=True` prevent accidental downloads.

EviSeq prediction files contain `id`, `prediction` and `reference`, but do not
include the source text. Pass the original test JSONL with `--source-file`;
the evaluator joins its `text` field to predictions by `id` before scoring.

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_alignscore.py \
  runs/model/last_test_predictions.jsonl \
  --model-path /models/roberta-large \
  --checkpoint-path /models/AlignScore-large.ckpt \
  --source-file datasets/pubmed/test.jsonl \
  --batch-size 32 \
  --device cuda:0 \
  --details
```

Use `--source-file-field` when the test JSONL stores the source under another
field such as `source` or `document`; use `--source-id-field` when its ID field
is not `id`. Prediction files that already contain `source` can still be
evaluated without `--source-file`.

The JSON output contains both `alignscore_consistency` and the transformed
`hallucination_score`, together with the checkpoint path, chunk size and
sentence splitter for reproducibility. Cite:

> Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. *AlignScore:
> Evaluating Factual Consistency with a Unified Alignment Function*. ACL.

Paper: <https://aclanthology.org/2023.acl-long.634/>. Reference implementation:
<https://github.com/yuh-zha/AlignScore>.
