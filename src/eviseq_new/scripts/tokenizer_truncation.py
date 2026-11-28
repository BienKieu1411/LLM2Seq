from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT.parent
sys.path.append(str(PACKAGE_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from export_samples import source_documents

SHARED = "/workspace/storage-shared/nlp"
TOKENIZER_FOLDERS = {
    "seam": [f"{SHARED}/users/dungdx4/BERT/pplx-embed-v1-0.6b", f"{SHARED}/dungdx4/BERT/pplx-embed-v1-0.6b"],
    "t5gemma": [f"{SHARED}/users/dungdx4/BERT/t5gemma-2-1b-1b", f"{SHARED}/dungdx4/BERT/t5gemma-2-1b-1b"],
    "qwen4b": [f"{SHARED}/users/dungdx4/BERT/Qwen3-4B", f"{SHARED}/dungdx4/BERT/Qwen3-4B"],
}
LABELS = {"seam": "SEAM (PPLX encoder)", "t5gemma": "T5Gemma 2 1B", "qwen4b": "Qwen3-4B decoder-only"}


@dataclass
class System:
    name: str
    tokenizer: Any
    budget: int
    prefix: str
    prefix_in_budget: bool
    output_budget: int


def _encode(tokenizer: Any, text: str, **options: Any) -> dict[str, Any]:
    return tokenizer(text, return_offsets_mapping=True, **options)


def source_coverage(source: str, tokenizer: Any, budget: int, prefix: str, prefix_in_budget: bool) -> dict[str, Any]:
    alone = _encode(tokenizer, source, add_special_tokens=False)
    tokens = len(alone["input_ids"])
    result = {"source_tokens": tokens, "truncated": False, "read_fraction": 1.0}
    if prefix_in_budget:
        text = prefix + source
        if len(_encode(tokenizer, text, add_special_tokens=True)["input_ids"]) <= budget:
            return result
        kept = _encode(tokenizer, text, add_special_tokens=True, truncation=True, max_length=budget)
        end = max((end for _, end in kept["offset_mapping"]), default=0)
        read = max(0, end - len(prefix))
    else:
        if tokens <= budget:
            return result
        read = alone["offset_mapping"][budget - 1][1]
    result.update(truncated=True, read_fraction=read / len(source))
    return result


def summarise(documents: Sequence[tuple[str, str, str]], systems: Sequence[System]) -> dict[str, Any]:
    full = {identifier for identifier, _, _ in documents}
    per_system = {}
    for system in systems:
        coverage = {i: source_coverage(source, system.tokenizer, system.budget, system.prefix, system.prefix_in_budget)
                    for i, source, _ in documents}
        full &= {identifier for identifier, row in coverage.items() if not row["truncated"]}
        long_references = sum(
            len(system.tokenizer(reference, add_special_tokens=False)["input_ids"]) > system.output_budget - 1
            for _, _, reference in documents
        )
        count = len(documents)
        per_system[system.name] = {
            "input_budget": system.budget,
            "output_budget": system.output_budget,
            "truncated_pct": 100 * sum(row["truncated"] for row in coverage.values()) / count,
            "read_pct": 100 * sum(row["read_fraction"] for row in coverage.values()) / count,
            "mean_source_tokens": sum(row["source_tokens"] for row in coverage.values()) / count,
            "references_over_output_budget_pct": 100 * long_references / count,
        }
    return {
        "documents": len(documents),
        "systems": per_system,
        "read_in_full_by_every_system": [identifier for identifier, _, _ in documents if identifier in full],
    }


def markdown(dataset: str, result: dict[str, Any]) -> str:
    lines = [
        f"### {dataset} ({result['documents']} test documents)",
        "",
        "| system | input budget | output budget | truncated docs (%) | source characters read (%) "
        "| mean source tokens | references over output budget (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in result["systems"].items():
        lines.append(
            f"| {name} | {row['input_budget']} | {row['output_budget']} | {row['truncated_pct']:.1f} "
            f"| {row['read_pct']:.1f} | {row['mean_source_tokens']:.0f} "
            f"| {row['references_over_output_budget_pct']:.1f} |"
        )
    shared = len(result["read_in_full_by_every_system"])
    lines += ["", f"Read in full by every system: {shared} of {result['documents']} documents "
              f"({100 * shared / result['documents']:.1f}%)."]
    return "\n".join(lines)


def configured_systems(dataset: str, tokenizers: dict[str, Any]) -> list[System]:
    from eviseq_afmr.config import load_config

    seam = load_config(PACKAGE_ROOT / "configs" / f"afmr_{dataset}.yaml")["data"]
    t5_files = sorted((SOURCE_ROOT / "T5Gemma/configs").glob(f"{dataset}_full_1b_1b_*.yaml"))
    if not t5_files:
        raise SystemExit(f"No T5Gemma config for {dataset}")
    t5 = yaml.safe_load(t5_files[0].read_text(encoding="utf-8"))["data"]
    suite = yaml.safe_load((SOURCE_ROOT / "decoder_baselines/configs/suite.yaml").read_text(encoding="utf-8"))
    qwen = suite["datasets"][dataset]
    return [
        System(LABELS["seam"], tokenizers["seam"], int(seam["max_source_length"]),
               str(seam.get("encoder_prefix", "")), True, int(seam["max_target_length"])),
        System(LABELS["t5gemma"], tokenizers["t5gemma"], int(t5["max_source_length"]),
               str(t5.get("source_prefix", "")), True, int(t5["max_target_length"])),
        System(LABELS["qwen4b"], tokenizers["qwen4b"], int(qwen["max_source_length"]),
               str(qwen.get("source_prefix", "")), False, int(qwen["max_target_length"])),
    ]


def load_tokenizers(overrides: Sequence[str]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    paths = {name: next((folder for folder in folders if Path(folder).is_dir()), None)
             for name, folders in TOKENIZER_FOLDERS.items()}
    for value in overrides:
        name, separator, path = value.partition("=")
        if not separator or name not in paths:
            raise SystemExit(f"--tokenizer expects NAME=PATH with NAME in {sorted(paths)}, got {value!r}")
        paths[name] = path
    missing = [name for name, path in paths.items() if not path]
    if missing:
        raise SystemExit(f"Tokenizer folder not found for {', '.join(missing)}; pass --tokenizer NAME=PATH")
    for name, path in paths.items():
        print(f"{name} tokenizer: {path}")
    return {name: AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=True)
            for name, path in paths.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="How much of each test document every system's input budget reads.")
    parser.add_argument("--datasets", nargs="+", default=["pubmed", "arxiv", "govreport", "booksum"])
    parser.add_argument("--data-root", type=Path, default=PACKAGE_ROOT / "datasets")
    parser.add_argument("--tokenizer", action="append", default=[], metavar="seam|t5gemma|qwen4b=PATH")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/truncation.md")
    args = parser.parse_args(argv)
    tokenizers = load_tokenizers(args.tokenizer)
    sections, details = [], {}
    for dataset in args.datasets:
        path = args.data_root / dataset / "test.jsonl"
        if not path.is_file():
            print(f"{dataset}: skipped, {path} not found")
            continue
        documents = [(row["id"], row["source"], row["reference"]) for row in source_documents(path)]
        details[dataset] = summarise(documents, configured_systems(dataset, tokenizers))
        sections.append(markdown(dataset, details[dataset]))
        print(sections[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
