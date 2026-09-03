# PubMed data

The built-in runner reads three JSONL files:

- `datasets/pubmed/train.jsonl`
- `datasets/pubmed/validation.jsonl`
- `datasets/pubmed/test.jsonl`

Each row uses the fixed fields `id`, `source`, `target` and, for the PCEB
objective, a zero-based `label` list containing the selected source-sentence
indices.

Prepare the files from a local PubMed source directory with:

```bash
bash ../scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
```

Then verify the splits with:

```bash
bash ../scripts/run.sh validate-data
```
