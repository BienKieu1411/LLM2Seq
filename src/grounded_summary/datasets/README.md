# Local dataset snapshots

Production datasets stay outside the repository and are referenced by an AFMR
task YAML. The files in this directory are small development fixtures only.

Prepare a canonical split with the public AFMR command:

```bash
PYTHONPATH=src/grounded_summary python -m grounded_summary.cli prepare \
  /absolute/path/to/input.jsonl datasets/task/train.jsonl
```
