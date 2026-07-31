# Data

Dataset files are JSONL: one JSON object per line. Field names are configured
in YAML, so users do not need to rewrite their data to `source` and `target`.

Supported mapping options:

- `source_field`, `target_field`, `id_field`;
- `source_template`, `target_template` for combining several top-level fields;
- `list_separator` for list-valued fields.

Run `eviseq validate-data --config CONFIG.yaml` before training to verify the
configured splits and print their sizes. Train and validation are required;
test is optional.

The CNN/DailyMail and PubMed conversion helpers remain available:

```bash
bash eviseq/scripts/run.sh prepare-cnndm /absolute/path/to/cnndm
bash eviseq/scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
```
