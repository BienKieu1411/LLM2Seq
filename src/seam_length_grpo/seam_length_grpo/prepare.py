"""Streaming annotation only: preserve records, add the matching gold word count."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from eviseq_afmr.data.normalization import detokenize
from eviseq_afmr.data.schema import _as_text

from .reward import count_words


def target_text(row, field, separator):
    value = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            if field == "summary" and "target" in row:
                return _as_text(row["target"], separator).strip()
            raise ValueError(f"Missing target field {field!r}; set --target-field")
        value = value[part]
    return _as_text(value, separator).strip()


def annotate(
    input_path,
    output_path,
    *,
    target_field="summary",
    words_field="target_words",
    normalize=False,
    separator="\n",
    overwrite=False,
    replace_length=False,
):
    source, output = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().resolve()
    if source == output:
        raise ValueError("Use a separate output path; input is never modified")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite for an intentional rerun")
    if words_field == target_field.split(".")[0]:
        raise ValueError("The word-count field cannot replace the target text")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count, total, minimum, maximum = 0, 0, None, 0
    try:
        with (
            source.open(encoding="utf-8") as handle,
            tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
            ) as destination,
        ):
            temporary = Path(destination.name)
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL record must be an object")
                    text = target_text(row, target_field, separator)
                    words = count_words(detokenize(text) if normalize else text)
                    if words < 1:
                        raise ValueError("Target has no countable words")
                    if (
                        words_field in row
                        and (type(row[words_field]) is not int or row[words_field] != words)
                        and not replace_length
                    ):
                        raise ValueError(
                            f"Existing {words_field} differs from gold count {words}; use --replace-length"
                        )
                    row[words_field] = words
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    count, total = count + 1, total + words
                    minimum, maximum = words if minimum is None else min(minimum, words), max(maximum, words)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid sample at {source}:{line_number}: {exc}; no samples were silently dropped"
                    ) from exc
            if not count:
                raise ValueError("Input has no JSONL samples")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Publish atomically without replacing a concurrently-created output.
            os.link(temporary, output)
        return {
            "input": str(source),
            "output": str(output),
            "samples": count,
            "dropped_samples": 0,
            "target_field": target_field,
            "words_field": words_field,
            "detokenize_for_count": normalize,
            "min_words": minimum,
            "max_words": maximum,
            "mean_words": total / count,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="Optional length-GRPO YAML; reuse its target field and detokenize setting")
    parser.add_argument("--target-field", help="Default: summary (or the configured target_field)")
    parser.add_argument("--target-words-field", help="Default: target_words (or the configured field)")
    parser.add_argument(
        "--detokenize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Count normalized text exactly as the training loader; stored text stays unchanged",
    )
    parser.add_argument("--replace-length", action="store_true", help="Recompute an existing, different word count")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data, length = {}, {}
    if args.config:
        from eviseq_afmr.config import load_config

        from .config import load

        settings = load(args.config)
        data = load_config(settings["base_config"], train_only=True)["data"]
        length = settings["length_control"]
    result = annotate(
        args.input,
        args.output,
        target_field=args.target_field or data.get("target_field", "summary"),
        words_field=args.target_words_field or length.get("target_words_field", "target_words"),
        normalize=args.detokenize if args.detokenize is not None else bool(data.get("detokenize", False)),
        separator=str(data.get("list_separator", "\n")),
        overwrite=args.overwrite,
        replace_length=args.replace_length,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
