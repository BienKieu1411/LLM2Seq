# Data

Dataset files are JSONL: one JSON object per line. Field names are configured
in YAML, so users do not need to rewrite their data to `source` and `target`.

Supported mapping options:

- `source_field`, `target_field`, `id_field`;
- `source_template`, `target_template` for combining several top-level fields;
- `list_separator` for list-valued fields.

The KD package reads the configured splits directly; train and validation are
required, while test is optional. For a custom dataset, point
`data.train_file`, `data.validation_file`, and (optionally) `data.test_file` at
JSONL files and keep the field mapping in the KD YAML config.

Dataset conversion is intentionally outside the KD runtime. Prepare any
source dataset separately, then provide its JSONL paths in the KD config.
