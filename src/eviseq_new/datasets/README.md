# Local dataset snapshots

Production datasets stay outside the repository and are referenced by an AFMR
task YAML. The files in this directory are small development fixtures only.

The preparation command writes canonical `train.jsonl`, `validation.jsonl`, and
`test.jsonl` files containing `id`, `text`, and `summary`:

```bash
PYTHON=/absolute/path/to/bienkieu_env/bin/python \
  bash ../scripts/prepare_benchmark_datasets.sh \
  --input-root /data/summarization
```

The input root should contain `arxiv`, `booksum`, and `govreport`. The GovReport
adapter recognizes both flat `train`/`validation`/`test` files and the public
`gao_*` plus `crs_*` JSONL files. For BookSum, `chapter`/`chapter_text` and
`summary_text` are recognized automatically. Raw files are copied under
`data/raw/<dataset>` and a `preparation_report.json` records the source and
output paths and skipped rows. The existing `prepare_govreport_booksum.py`
export is accepted as-is: its `source`/`target` columns are detected and
rewritten to AFMR's `text`/`summary` columns.

Prepare a canonical split with the public AFMR command:

```bash
PYTHONPATH=src/eviseq_new python -m eviseq_afmr.cli prepare \
  /absolute/path/to/input.jsonl datasets/task/train.jsonl
```
