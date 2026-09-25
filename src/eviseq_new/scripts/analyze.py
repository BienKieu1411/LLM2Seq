#!/usr/bin/env python3
"""Plot paired PubMed ablation diagnostics from three predictions and test sources.

This is a *diagnostic figure*, not a replacement for the paper's ROUGE or
factuality evaluation. It measures exact reference word sequences/numbers
that also occur in the encoder-visible source. No outcome is called factual
merely because its surface form matches the source.

Pass --pred-dir with full.txt, wo_bridge.txt, and wo_copy.txt, plus the local
--test-jsonl used for evaluation. A pinned public PubMed parquet remains an
optional fallback. IDs and available references are checked before analysis.
Only an incomplete final prediction line is tolerated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eviseq_new"))
from eviseq_afmr.config import load_config  # noqa: E402
from eviseq_afmr.data.normalization import detokenize  # noqa: E402

SOURCE_REPO = "mehdielg/pubmed-orig-test"
SOURCE_REVISION = "29953aff9e233083d3d540a5f21a66fdbf6272c2"
SOURCE_FILE = "data/test-00000-of-00001.parquet"
ENCODER_TOKENIZER = "perplexity-ai/pplx-embed-v1-0.6b"
DECODER_TOKENIZER = "Qwen/Qwen3-0.6B"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?:\s?%)?(?![A-Za-z0-9])")
S_TAG_RE = re.compile(r"</?S>")
SYSTEMS = ("full", "no_bridge", "no_copy")
POSITION_LABELS = ("0-25%", "25-50%", "50-75%", "75-100%")
NUMBER_LABELS = ("integer", "decimal", "percent")


def load_predictions(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    contents = path.read_text(encoding="utf-8")
    lines = contents.splitlines()
    records: dict[str, dict[str, str]] = {}
    incomplete_line = None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"Blank line in {path}:{number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if number != len(lines) or contents.endswith("\n"):
                raise ValueError(f"Malformed complete JSONL line in {path}:{number}") from exc
            incomplete_line = number
            break
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("id", "prediction", "reference")):
            raise ValueError(f"Expected string id/prediction/reference in {path}:{number}")
        identifier = row["id"]
        if not identifier or identifier in records:
            raise ValueError(f"Empty or duplicate ID {identifier!r} in {path}:{number}")
        records[identifier] = {"prediction": row["prediction"], "reference": row["reference"]}
    return records, {"valid_rows": len(records), "incomplete_final_line": incomplete_line}


def load_test_jsonl(path: Path, wanted_ids: set[str], data_config: dict) -> tuple[dict[str, dict], int]:
    """Read model inputs by ID, preserving the dataset loader's text handling."""
    id_field = str(data_config.get("id_field", "id"))
    source_field = str(data_config.get("source_field", "text"))
    target_field = str(data_config.get("target_field", "summary"))
    separator = str(data_config.get("list_separator", "\n"))

    def as_text(value: object, label: str) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return separator.join(item.strip() for item in value if item.strip()).strip()
        raise ValueError(f"{label} must be a string or list of strings")

    rows: dict[str, dict] = {}
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid test JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Test row must be an object at {path}:{line_number}")
            raw_id = row.get(id_field, row.get("article_id"))
            identifier = str(raw_id).strip() if raw_id is not None else ""
            if not identifier or identifier in seen:
                raise ValueError(f"Missing or duplicate test ID at {path}:{line_number}: {identifier!r}")
            seen.add(identifier)
            if identifier not in wanted_ids:
                continue
            source_key = next((key for key in (source_field, "source", "article_text") if key in row), None)
            if source_key is None:
                raise ValueError(f"Missing source field for {identifier}; expected {source_field!r}")
            source = as_text(row[source_key], f"source for {identifier}")
            reference_key = next((key for key in (target_field, "target", "abstract_text") if key in row), None)
            reference = as_text(row[reference_key], f"reference for {identifier}") if reference_key else None
            if reference_key == "abstract_text":
                reference = S_TAG_RE.sub("", reference or "").strip()
            if data_config.get("detokenize", False):
                source = detokenize(source)
                reference = detokenize(reference) if reference is not None else None
            if not source:
                raise ValueError(f"Empty source for {identifier}")
            rows[identifier] = {"source": source, "reference": reference}
    if not seen:
        raise ValueError(f"Empty test JSONL: {path}")
    return rows, len(seen)


def number_type(value: str) -> int:
    if value.endswith("%"):
        return 2
    return 1 if "." in value else 0


def numbers(text: str) -> set[str]:
    return {match.group().replace(" ", "") for match in NUMBER_RE.finditer(text)}


def words_with_offsets(text: str) -> tuple[list[str], list[int]]:
    matches = list(WORD_RE.finditer(text))
    return [m.group().casefold() for m in matches], [m.start() for m in matches]


def phrase_counts(source: str, reference: str, predictions: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Count unique source-occurring reference trigrams by source quartile.

    A phrase is eligible only when it occurs exactly once in the visible source.
    Punctuation is ignored by the word tokenizer; matching is case-insensitive.
    """
    denom = np.zeros(4, dtype=np.int64)
    hits = np.zeros((3, 4), dtype=np.int64)
    source_words, offsets = words_with_offsets(source)
    ref_words, _ = words_with_offsets(reference)
    gold = {tuple(ref_words[i : i + 3]) for i in range(max(0, len(ref_words) - 2))}
    if not gold or not source_words:
        return denom, hits
    locations: dict[tuple[str, str, str], list[int]] = {}
    for i in range(len(source_words) - 2):
        phrase = tuple(source_words[i : i + 3])
        if phrase in gold:
            locations.setdefault(phrase, []).append(offsets[i])
    prediction_sets = {}
    for name in SYSTEMS:
        tokens, _ = words_with_offsets(predictions[name])
        prediction_sets[name] = {tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))}
    for phrase, positions in locations.items():
        if len(positions) != 1:
            continue
        quarter = min(3, 4 * positions[0] // len(source))
        denom[quarter] += 1
        for system_index, name in enumerate(SYSTEMS):
            hits[system_index, quarter] += phrase in prediction_sets[name]
    return denom, hits


def number_counts(source: str, reference: str, predictions: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Count exact numeric forms in both reference and encoder-visible source."""
    denom = np.zeros(3, dtype=np.int64)
    hits = np.zeros((3, 3), dtype=np.int64)
    eligible = numbers(reference) & numbers(source)
    prediction_sets = {name: numbers(predictions[name]) for name in SYSTEMS}
    for value in eligible:
        kind = number_type(value)
        denom[kind] += 1
        for system_index, name in enumerate(SYSTEMS):
            hits[system_index, kind] += value in prediction_sets[name]
    return denom, hits


def paired_bootstrap(
    denominators: np.ndarray,
    hits: np.ndarray,
    ablation_index: int,
    *,
    draws: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Micro-recall gaps with 95% document-cluster bootstrap intervals."""
    rng = np.random.default_rng(seed)
    total = denominators.sum(axis=0)
    full = hits[:, 0, :].sum(axis=0)
    ablated = hits[:, ablation_index, :].sum(axis=0)
    bootstrap = np.empty((draws, denominators.shape[1]), dtype=float)
    for b in range(draws):
        take = rng.integers(0, len(denominators), len(denominators))
        sampled_total = denominators[take].sum(axis=0)
        delta = hits[take, 0, :].sum(axis=0) - hits[take, ablation_index, :].sum(axis=0)
        bootstrap[b] = np.divide(
            100 * delta, sampled_total, out=np.full_like(delta, np.nan, dtype=float), where=sampled_total > 0
        )
    output = []
    for i, n in enumerate(total):
        if n == 0:
            raise ValueError(f"No eligible units in category {i}")
        output.append(
            {
                "eligible_units": int(n),
                "documents_with_units": int(np.count_nonzero(denominators[:, i])),
                "full_recall_percent": round(100 * full[i] / n, 4),
                "ablation_recall_percent": round(100 * ablated[i] / n, 4),
                "gap_percentage_points": round(100 * (full[i] - ablated[i]) / n, 4),
                "ci_low": round(float(np.nanquantile(bootstrap[:, i], 0.025)), 4),
                "ci_high": round(float(np.nanquantile(bootstrap[:, i], 0.975)), 4),
            }
        )
    return output


def late_early_contrast(denominators: np.ndarray, hits: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    """Check whether the bridge's Q4 gain actually exceeds its Q1 gain."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(draws, dtype=float)
    for b in range(draws):
        take = rng.integers(0, len(denominators), len(denominators))
        denominator = denominators[take].sum(axis=0)
        gain = (hits[take, 0, :].sum(axis=0) - hits[take, 1, :].sum(axis=0)) / denominator * 100
        diffs[b] = gain[3] - gain[0]
    denominator = denominators.sum(axis=0)
    gain = (hits[:, 0, :].sum(axis=0) - hits[:, 1, :].sum(axis=0)) / denominator * 100
    return {
        "q4_minus_q1_percentage_points": round(float(gain[3] - gain[0]), 4),
        "ci_low": round(float(np.quantile(diffs, 0.025)), 4),
        "ci_high": round(float(np.quantile(diffs, 0.975)), 4),
    }


def plot_results(position: list[dict], numeric: list[dict], output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    panels = (
        (axes[0], position, POSITION_LABELS, "A  Source-position trigrams", "Full - no bridge", "#177e89"),
        (axes[1], numeric, ("all", *NUMBER_LABELS), "B  Source-visible numbers", "Full - no copy", "#b9553e"),
    )
    for ax, rows, labels, title, annotation, color in panels:
        x = np.arange(len(rows))
        y = np.array([row["gap_percentage_points"] for row in rows])
        low = np.array([row["ci_low"] for row in rows])
        high = np.array([row["ci_high"] for row in rows])
        ax.axhline(0, color="#6c737a", linewidth=0.8, linestyle="--", zorder=0)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - low, high - y)),
            fmt="o",
            markersize=5.5,
            color=color,
            ecolor=color,
            capsize=3,
            elinewidth=1.25,
            zorder=2,
        )
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.45, len(x) - 0.55)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5, pad=10)
        ax.text(0.98, 0.97, annotation, transform=ax.transAxes, ha="right", va="top", color=color, fontsize=8)
        ax.grid(axis="y", color="#e6e9ec", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=4)
    axes[0].set_ylabel("Paired recall gain (percentage points)")
    axes[1].set_ylabel("Paired recall gain (percentage points)")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"pubmed_ablation_diagnostics.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, help="Folder containing full.txt, wo_bridge.txt, wo_copy.txt")
    parser.add_argument("--test-jsonl", type=Path, help="Local test JSONL with id/text and optional summary")
    parser.add_argument("--full", type=Path, help="Full-system predictions (old interface)")
    parser.add_argument("--no-bridge", type=Path, help="No-bridge predictions (old interface)")
    parser.add_argument("--no-copy", type=Path, help="No-copy predictions (old interface)")
    parser.add_argument("--source-parquet", type=Path, help="Public PubMed test parquet (old interface)")
    parser.add_argument("--config", type=Path, default=ROOT / "src/eviseq_new/configs/afmr_pubmed.yaml")
    parser.add_argument("--encoder-tokenizer", default=ENCODER_TOKENIZER, help="Encoder tokenizer path or HF ID")
    parser.add_argument("--decoder-tokenizer", default=DECODER_TOKENIZER, help="Decoder tokenizer path or HF ID")
    parser.add_argument("--max-source-tokens", type=int, help="Override the config's encoder limit")
    parser.add_argument("--output-dir", type=Path, help="Where to save the PNG, PDF, and report")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100 or (args.max_source_tokens is not None and args.max_source_tokens < 1):
        parser.error("--bootstrap must be >= 100 and --max-source-tokens must be positive")
    if args.pred_dir is not None and args.test_jsonl is None:
        parser.error("--pred-dir requires --test-jsonl")
    if args.output_dir is None:
        suffix = "pubmed_ablation_from_txt" if args.pred_dir is not None else "pubmed_ablation_diagnostics"
        args.output_dir = ROOT / "Paper/analysis" / suffix

    if args.pred_dir is not None:
        if not args.pred_dir.is_dir() or any((args.full, args.no_bridge, args.no_copy)):
            parser.error("--pred-dir must be a directory and cannot be combined with individual prediction paths")
        paths = {
            "full": args.pred_dir / "full.txt",
            "no_bridge": args.pred_dir / "wo_bridge.txt",
            "no_copy": args.pred_dir / "wo_copy.txt",
        }
    else:
        if not all((args.full, args.no_bridge, args.no_copy)):
            parser.error("Provide --pred-dir, or all of --full, --no-bridge, and --no-copy")
        paths = {"full": args.full, "no_bridge": args.no_bridge, "no_copy": args.no_copy}
    if args.test_jsonl is not None and args.source_parquet is not None:
        parser.error("Use either --test-jsonl or --source-parquet, not both")
    if args.test_jsonl is not None and not args.test_jsonl.is_file():
        parser.error(f"Test JSONL not found: {args.test_jsonl}")
    for path in paths.values():
        if not path.is_file():
            parser.error(f"Prediction file not found: {path}")
    records = {}
    audits = {}
    for name, path in paths.items():
        records[name], audits[name] = load_predictions(path)
    shared = [identifier for identifier in records["full"] if all(identifier in records[name] for name in SYSTEMS)]
    if not shared:
        raise ValueError("No IDs shared by all three prediction files")
    for identifier in shared:
        refs = [records[name][identifier]["reference"] for name in SYSTEMS]
        if len(set(refs)) != 1:
            raise ValueError(f"Different references for shared ID {identifier}")

    config = load_config(args.config)
    max_length = args.max_source_tokens or int(config["data"]["max_source_length"])
    prefix = str(config["data"]["encoder_prefix"])
    shared_set = set(shared)
    if args.test_jsonl is not None:
        source_rows, test_rows = load_test_jsonl(args.test_jsonl, shared_set, config["data"])
        source_info = {"test_jsonl": str(args.test_jsonl), "test_rows": test_rows}
    else:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        source_path = args.source_parquet or Path(
            hf_hub_download(repo_id=SOURCE_REPO, filename=SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION)
        )
        table = pq.read_table(source_path, columns=["article_id", "article_text", "abstract_text"])
        source_rows = {}
        for row in table.to_pylist():
            identifier = row["article_id"]
            if identifier in shared_set:
                if identifier in source_rows:
                    raise ValueError(f"Duplicate public PubMed source ID {identifier}")
                source_rows[identifier] = {
                    "source": detokenize("\n".join(row["article_text"])),
                    "reference": detokenize(
                        "\n".join(S_TAG_RE.sub("", value).strip() for value in row["abstract_text"])
                    ),
                }
        test_rows = table.num_rows
        source_info = {"repo": SOURCE_REPO, "revision": SOURCE_REVISION, "file": SOURCE_FILE, "test_rows": test_rows}
    if shared_set != set(source_rows):
        raise ValueError(f"Test sources lack {len(shared_set - set(source_rows))} shared prediction IDs")
    verified_references = 0
    for identifier in shared:
        reference = source_rows[identifier]["reference"]
        if reference is not None:
            if reference != records["full"][identifier]["reference"]:
                raise ValueError(f"Test reference mismatch for {identifier}; cannot join sources safely")
            verified_references += 1
    source_info["matched_references_exact"] = verified_references

    encoder = AutoTokenizer.from_pretrained(args.encoder_tokenizer, use_fast=True, trust_remote_code=False)
    decoder = AutoTokenizer.from_pretrained(args.decoder_tokenizer, use_fast=True, trust_remote_code=False)
    if not encoder.is_fast or not decoder.is_fast:
        raise ValueError("Both tokenizers must provide fast offset mappings")
    encoder_backend = json.loads(encoder.backend_tokenizer.to_str())
    decoder_backend = json.loads(decoder.backend_tokenizer.to_str())
    same_core_tokenization = all(
        encoder_backend[key] == decoder_backend[key] for key in ("normalizer", "pre_tokenizer", "model")
    )
    position_denoms, position_hits, numeric_denoms, numeric_hits = [], [], [], []
    truncated = 0
    visible_fractions = []
    for index, identifier in enumerate(shared, 1):
        source = source_rows[identifier]["source"]
        encoded = encoder(
            prefix + source,
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        visible_end = max(
            (end - len(prefix) for start, end in encoded["offset_mapping"] if end > max(start, len(prefix))),
            default=0,
        )
        if visible_end <= 0:
            raise ValueError(f"No encoder-visible source content for {identifier}")
        visible = source[:visible_end]
        truncated += visible_end < len(source.rstrip())
        visible_fractions.append(visible_end / len(source))
        prediction_texts = {name: records[name][identifier]["prediction"] for name in SYSTEMS}
        reference = records["full"][identifier]["reference"]
        p_denom, p_hits = phrase_counts(visible, reference, prediction_texts)
        n_denom, n_hits = number_counts(visible, reference, prediction_texts)
        position_denoms.append(p_denom)
        position_hits.append(p_hits)
        numeric_denoms.append(n_denom)
        numeric_hits.append(n_hits)
        if index % 500 == 0:
            print(f"Analyzed {index}/{len(shared)} matched documents", flush=True)

    p_den, p_hit = np.stack(position_denoms), np.stack(position_hits)
    n_den, n_hit = np.stack(numeric_denoms), np.stack(numeric_hits)
    position = paired_bootstrap(p_den, p_hit, 1, draws=args.bootstrap, seed=args.seed)
    numeric = paired_bootstrap(n_den, n_hit, 2, draws=args.bootstrap, seed=args.seed + 1)
    all_numbers = paired_bootstrap(
        n_den.sum(axis=1, keepdims=True),
        n_hit.sum(axis=2, keepdims=True),
        2,
        draws=args.bootstrap,
        seed=args.seed + 1,
    )[0]
    all_phrases = paired_bootstrap(
        p_den.sum(axis=1, keepdims=True),
        p_hit.sum(axis=2, keepdims=True),
        1,
        draws=args.bootstrap,
        seed=args.seed,
    )[0]
    late_vs_early = late_early_contrast(p_den, p_hit, draws=args.bootstrap, seed=args.seed + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(position, [all_numbers, *numeric], args.output_dir)
    warnings = []
    if len(shared) < test_rows:
        warnings.append(f"Only {len(shared)}/{test_rows} test IDs occur in all three prediction files")
    if any(audits[name]["incomplete_final_line"] is not None for name in SYSTEMS):
        warnings.append("At least one prediction file ends with an incomplete JSON line")
    if verified_references < len(shared):
        warnings.append(f"The test JSONL has no reference for {len(shared) - verified_references} matched IDs")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    report = {
        "prediction_files": {name: {"path": str(paths[name]), **audits[name]} for name in SYSTEMS},
        "matched_documents": len(shared),
        "source_dataset": source_info,
        "encoder_input": {
            "config": str(args.config),
            "max_source_tokens": max_length,
            "prefix": prefix,
            "documents_truncated": truncated,
            "median_visible_character_fraction": round(float(np.median(visible_fractions)), 4),
        },
        "tokenizers": {
            "encoder": args.encoder_tokenizer,
            "decoder": args.decoder_tokenizer,
            "same_core_tokenization": same_core_tokenization,
        },
        "warnings": warnings,
        "bootstrap": {
            "method": "paired document-cluster percentile",
            "draws": args.bootstrap,
            "seed": args.seed,
            "interval": "95%",
        },
        "position_trigram_recall": {label: value for label, value in zip(POSITION_LABELS, position)},
        "position_trigram_recall_all": all_phrases,
        "late_minus_early_bridge_gain": late_vs_early,
        "source_visible_exact_number_recall": {label: value for label, value in zip(NUMBER_LABELS, numeric)},
        "source_visible_exact_number_recall_all": all_numbers,
        "interpretation_limits": [
            "Exact lexical overlap is not a factuality or causal-attribution metric.",
            "Predictions cover only the IDs common to all three files; check test-set coverage before paper use.",
            "Source visibility assumes the supplied predictions used the specified config and tokenizer.",
            "This contrast does not isolate tokenizer-mismatch robustness, even when encoder and decoder tokenization differs.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "matched_documents": len(shared),
                "position": report["position_trigram_recall"],
                "numeric": report["source_visible_exact_number_recall"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
