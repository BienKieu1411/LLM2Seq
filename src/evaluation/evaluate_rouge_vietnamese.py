#!/usr/bin/env python3
"""Generate summaries from local source files and evaluate with historical ROUGE.

This script intentionally preserves the old preprocessing and ROUGE command:

* lowercase and remap legacy bracket tokens;
* remove the old English summary preamble from predictions;
* treat blank-separated blocks in each reference file as multiple references;
* run ROUGE-1.5.5 with ``-a -m -n 2 -d``.

Sources, references, and generated predictions use this tree contract::

    data_raw/<dataset>/<file>.txt
    data_ref/<dataset>/<file>.txt
    data_sum/<model>/<dataset>/<file>.txt

Pass a local Hugging Face checkpoint with ``--model-path``. Model files are
loaded offline; this script never downloads a checkpoint. For the repository's
custom SEAM/AFMR model, pass ``--backend afmr`` with its resolved config and
checkpoint; this runs the AFMR PyTorch inference path directly, without vLLM.
ROUGE preprocessing and flags match the historical evaluator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# CONFIG -- server defaults can also be overridden with CLI options/env vars
# =============================================================================

PLATFORM_ROOT_CANDIDATES = (Path("/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval"),)


def default_platform_root() -> Path:
    configured = os.environ.get("TEXTSUM_PLATFORM_EVAL_ROOT")
    if configured:
        return Path(configured).expanduser()
    return next((path for path in PLATFORM_ROOT_CANDIDATES if path.is_dir()), PLATFORM_ROOT_CANDIDATES[0])


PLATFORM_ROOT = default_platform_root()
DATA_RAW_ROOT = Path(
    os.environ.get("TEXTSUM_SOURCE_ROOT", os.environ.get("TEXTSUM_PREDICTION_ROOT", PLATFORM_ROOT / "data_raw"))
)
DATA_REF_ROOT = Path(os.environ.get("TEXTSUM_REFERENCE_ROOT", PLATFORM_ROOT / "data_ref"))
DATA_SUM_ROOT = Path(os.environ.get("TEXTSUM_SUMMARY_ROOT", PLATFORM_ROOT / "data_sum"))
ROUGE_PATH = Path(
    os.environ.get(
        "ROUGE155_PATH",
        PLATFORM_ROOT / "pyrouge-master" / "tools" / "ROUGE-1.5.5",
    )
)
DATASETS_TO_EVALUATE = ("ViM", "book", "legal", "vietnews", "vlsp22", "wikilingua", "xlsum")
DEFAULT_PROMPT = (
    "Hãy viết một bản tóm tắt bằng tiếng Việt cho văn bản sau. Chỉ trả về bản tóm tắt, không thêm lời dẫn.\n\n{source}"
)

PREAMBLE = "Here is a summary of the text in Vietnamese"
REFERENCE_LABELS = ("A", "B", "C", "D", "E", "F", "G")
REMAP = {
    "-lrb-": "(",
    "-rrb-": ")",
    "-lcb-": "{",
    "-rcb-": "}",
    "-lsb-": "[",
    "-rsb-": "]",
    "``": '"',
    "''": '"',
}
REMAP_RE = re.compile(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''")
METRIC_KEYS = (
    "rouge1_precision",
    "rouge1_recall",
    "rouge1_fmeasure",
    "rouge2_precision",
    "rouge2_recall",
    "rouge2_fmeasure",
    "rougeL_precision",
    "rougeL_recall",
    "rougeL_fmeasure",
)


@dataclass(frozen=True)
class Pair:
    model: str
    dataset: str
    relative_file: str
    source: Path
    prediction: Path
    reference: Path


def clean(text: str) -> str:
    """Match the old script's lowercase and token remapping."""
    return REMAP_RE.sub(lambda match: REMAP[match.group()], text.lower())


def read_prediction(path: Path) -> str:
    """Match the old line joining and English-preamble removal."""
    kept = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if PREAMBLE in line:
            continue
        if line:
            kept.append(line)
    return " ".join(kept)


def read_references(path: Path) -> list[str]:
    """Split a file into references exactly as the old blank-line loop did."""
    references: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            current.append(line)
        elif current:
            references.append(" ".join(current))
            current = []
    if current:
        references.append(" ".join(current))
    return references


def is_empty_text_file(path: Path) -> bool:
    """Return whether a UTF-8 text file contains only whitespace."""
    with path.open("r", encoding="utf-8-sig") as handle:
        return not any(line.strip() for line in handle)


def discover_source_pairs(
    source_root: Path,
    reference_root: Path,
    prediction_dir: Path,
    model_name: str,
) -> tuple[list[Pair], list[str], list[str]]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Reference root not found: {reference_root}")
    missing_dataset_dirs = [
        str(reference_root / dataset) for dataset in DATASETS_TO_EVALUATE if not (reference_root / dataset).is_dir()
    ]
    if missing_dataset_dirs:
        raise FileNotFoundError("Missing required reference dataset directories: " + ", ".join(missing_dataset_dirs))

    missing_source_dirs = [
        str(source_root / dataset) for dataset in DATASETS_TO_EVALUATE if not (source_root / dataset).is_dir()
    ]
    if missing_source_dirs:
        raise FileNotFoundError("Missing required source dataset directories: " + ", ".join(missing_source_dirs))

    pairs: list[Pair] = []
    missing_references: list[str] = []
    empty_sources: list[str] = []
    for dataset in DATASETS_TO_EVALUATE:
        dataset_source_dir = source_root / dataset
        for source in sorted(dataset_source_dir.rglob("*.txt")):
            relative = source.relative_to(dataset_source_dir)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if is_empty_text_file(source):
                empty_sources.append(str(source))
                continue
            reference = reference_root / dataset / relative
            if not reference.is_file():
                missing_references.append(str(reference))
                continue
            prediction = prediction_dir / dataset / relative
            pairs.append(
                Pair(
                    model=model_name,
                    dataset=dataset,
                    relative_file=relative.as_posix(),
                    source=source,
                    prediction=prediction,
                    reference=reference,
                )
            )
    return pairs, missing_references, empty_sources


def load_local_generator(
    model_path: Path,
    *,
    device_name: str,
    dtype_name: str,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any, str]:
    """Load a local causal or encoder-decoder checkpoint without hub access."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Prediction requires torch and transformers in the server environment") from exc

    resolved_model = model_path.expanduser().resolve()
    if not resolved_model.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {resolved_model}")
    common = {"local_files_only": True, "trust_remote_code": trust_remote_code}
    config = AutoConfig.from_pretrained(resolved_model, **common)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, **common)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM

    dtypes = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in dtypes:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose auto, bf16, fp16, or fp32")
    model = model_class.from_pretrained(resolved_model, config=config, torch_dtype=dtypes[dtype_name], **common)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("Tokenizer must define a pad, eos, or unk token for batched generation")
    if not config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name}, but CUDA is unavailable")
    model.to(device_name)
    model.eval()
    return torch, tokenizer, model, device_name


def generate_predictions(
    pairs: list[Pair],
    model_path: Path,
    *,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    prompt_template: str,
    use_chat_template: bool,
    device_name: str,
    dtype_name: str,
    trust_remote_code: bool,
    overwrite: bool,
) -> None:
    pending = [
        pair for pair in pairs if overwrite or not pair.prediction.is_file() or pair.prediction.stat().st_size == 0
    ]
    if not pending:
        print("All predictions already exist; skipping generation.")
        return
    if batch_size < 1 or max_input_tokens < 1 or max_new_tokens < 1:
        raise ValueError("batch size and token limits must be positive")

    torch, tokenizer, model, device = load_local_generator(
        model_path,
        device_name=device_name,
        dtype_name=dtype_name,
        trust_remote_code=trust_remote_code,
    )
    print(
        f"Generating {len(pending)} summaries from local checkpoint {model_path} "
        f"(device={device}, batch={batch_size}, max_input={max_input_tokens}, max_new={max_new_tokens})"
    )

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        sources = [pair.source.read_text(encoding="utf-8-sig").strip() for pair in batch]
        if any(not source for source in sources):
            empty = [str(pair.source) for pair, source in zip(batch, sources, strict=True) if not source]
            raise ValueError("Empty source file(s): " + ", ".join(empty))

        prompts = [prompt_template.format(source=source) for source in sources]
        if use_chat_template and tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        is_causal = not model.config.is_encoder_decoder
        input_width = encoded["input_ids"].shape[1]
        for pair, token_ids in zip(batch, generated, strict=True):
            if is_causal:
                token_ids = token_ids[input_width:]
            summary = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            pair.prediction.parent.mkdir(parents=True, exist_ok=True)
            temporary = pair.prediction.with_name(pair.prediction.name + ".tmp")
            temporary.write_text(summary + "\n", encoding="utf-8")
            temporary.replace(pair.prediction)

        print(f"[predict] {min(start + len(batch), len(pending))}/{len(pending)} summaries written")


def _afmr_example_id(pair: Pair) -> str:
    return f"{pair.dataset}/{pair.relative_file}"


def _afmr_work_key(config_path: Path, checkpoint_path: Path, pairs: list[Pair]) -> str:
    """Build a stable cache key so AFMR evaluation can safely resume."""
    digest = hashlib.sha256()
    for path in (config_path, checkpoint_path):
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    for pair in pairs:
        digest.update(_afmr_example_id(pair).encode("utf-8"))
        for path in (pair.source, pair.reference):
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return digest.hexdigest()[:20]


def generate_afmr_predictions(
    pairs: list[Pair],
    model_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    device: str,
    max_input_tokens: int | None,
    max_new_tokens: int | None,
    overwrite: bool,
) -> None:
    """Generate file predictions through the local AFMR model, without vLLM."""
    pending = [
        pair for pair in pairs if overwrite or not pair.prediction.is_file() or pair.prediction.stat().st_size == 0
    ]
    if not pending:
        print("All predictions already exist; skipping AFMR generation.")
        return
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    afmr_root = Path(__file__).resolve().parents[1] / "eviseq_new"
    if not afmr_root.is_dir():
        raise FileNotFoundError(f"AFMR source package not found: {afmr_root}")
    if str(afmr_root) not in sys.path:
        sys.path.insert(0, str(afmr_root))

    import yaml
    from eviseq_afmr.config import load_config, resolve_path
    from eviseq_afmr.runtime import evaluate as evaluate_afmr

    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"AFMR resolved config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"AFMR checkpoint not found: {checkpoint_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    work_dir = model_dir / ".afmr_work" / _afmr_work_key(config_path, checkpoint_path, pending)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = work_dir / "input.jsonl"
    output_jsonl = work_dir / "predictions.jsonl"
    temporary_config_path = work_dir / "resolved_config.yaml"

    rows: list[dict[str, str]] = []
    pair_by_id: dict[str, Pair] = {}
    for pair in pending:
        example_id = _afmr_example_id(pair)
        if example_id in pair_by_id:
            raise ValueError(f"Duplicate source ID for AFMR prediction: {example_id}")
        source = pair.source.read_text(encoding="utf-8-sig").strip()
        references = read_references(pair.reference)
        if not source:
            raise ValueError(f"Empty source file: {pair.source}")
        if not references:
            raise ValueError(f"Reference file is empty: {pair.reference}")
        pair_by_id[example_id] = pair
        rows.append({"id": example_id, "text": source, "summary": references[0]})

    input_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    if overwrite and output_jsonl.is_file():
        output_jsonl.unlink()

    resolved = load_config(config_path)
    temporary_config = copy.deepcopy(resolved)
    temporary_config.pop("_meta", None)
    model_config = temporary_config["model"]
    for key in ("encoder_name", "decoder_name"):
        model_config[key] = str(resolve_path(resolved["model"][key], resolved))
    data_config = temporary_config["data"]
    for key in ("train_file", "validation_file", "test_file"):
        value = resolved["data"].get(key)
        if value:
            data_config[key] = str(resolve_path(value, resolved))
    data_config.update(
        {
            "test_file": str(input_jsonl.resolve()),
            "source_field": "text",
            "target_field": "summary",
            "id_field": "id",
        }
    )
    if max_input_tokens is not None:
        data_config["max_source_length"] = int(max_input_tokens)
    if max_new_tokens is not None:
        temporary_config["generation"]["max_new_tokens"] = int(max_new_tokens)
    temporary_config_path.write_text(
        yaml.safe_dump(temporary_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(
        f"Generating {len(pending)} summaries with AFMR checkpoint {checkpoint_path} "
        f"(device={device}, batch={batch_size}); vLLM is not used."
    )
    evaluate_afmr(
        temporary_config_path,
        checkpoint_path,
        output_jsonl,
        split="test",
        batch_size=batch_size,
        device=None if device == "auto" else device,
        do_sample=False,
    )

    generated: dict[str, str] = {}
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            example_id = str(row.get("id", ""))
            prediction = row.get("prediction")
            if example_id not in pair_by_id or not isinstance(prediction, str):
                raise ValueError(f"Unexpected AFMR prediction at {output_jsonl}:{line_number}")
            generated[example_id] = prediction
    missing = sorted(set(pair_by_id) - set(generated))
    if missing:
        raise RuntimeError(f"AFMR did not produce {len(missing)} prediction(s); first missing ID: {missing[0]}")

    for example_id, pair in pair_by_id.items():
        pair.prediction.parent.mkdir(parents=True, exist_ok=True)
        temporary = pair.prediction.with_name(pair.prediction.name + ".tmp")
        temporary.write_text(generated[example_id].strip() + "\n", encoding="utf-8")
        temporary.replace(pair.prediction)
    print(f"[predict] wrote {len(generated)} AFMR summaries under {model_dir}")


def validate_rouge_install(rouge_path: Path) -> None:
    script = rouge_path / "ROUGE-1.5.5.pl"
    data = rouge_path / "data"
    if not script.is_file() or not data.is_dir():
        raise FileNotFoundError(f"Invalid ROUGE-1.5.5 directory: {rouge_path} (expected ROUGE-1.5.5.pl and data/)")


def _convert_scores(output: dict[str, Any]) -> dict[str, float | int]:
    return {
        "rouge1_precision": float(output["rouge_1_precision"]),
        "rouge1_recall": float(output["rouge_1_recall"]),
        "rouge1_fmeasure": float(output["rouge_1_f_score"]),
        "rouge2_precision": float(output["rouge_2_precision"]),
        "rouge2_recall": float(output["rouge_2_recall"]),
        "rouge2_fmeasure": float(output["rouge_2_f_score"]),
        "rougeL_precision": float(output["rouge_l_precision"]),
        "rougeL_recall": float(output["rouge_l_recall"]),
        "rougeL_fmeasure": float(output["rouge_l_f_score"]),
    }


def run_pyrouge(pairs: list[Pair], rouge_path: Path, keep_temp: Path | None) -> dict[str, Any]:
    """Run one corpus-level ROUGE job using the exact historical arguments."""
    from pyrouge import Rouge155

    if not pairs:
        raise ValueError("Cannot run ROUGE on an empty pair list")

    if keep_temp is None:
        work_root = Path(tempfile.mkdtemp(prefix="pyrouge-old-"))
        remove_after = True
    else:
        keep_temp.mkdir(parents=True, exist_ok=True)
        work_root = Path(tempfile.mkdtemp(prefix="pyrouge-old-", dir=keep_temp))
        remove_after = False

    system_path = work_root / "result"
    model_path = work_root / "gold"
    system_path.mkdir()
    model_path.mkdir()

    try:
        for index, pair in enumerate(pairs):
            hypothesis = clean(read_prediction(pair.prediction))
            (system_path / f"Model.{index}.txt").write_bytes(hypothesis.encode("utf-8"))

            references = read_references(pair.reference)
            if not references:
                raise ValueError(f"Reference file is empty: {pair.reference}")
            if len(references) > len(REFERENCE_LABELS):
                raise ValueError(
                    f"{pair.reference} has {len(references)} blank-separated references; "
                    f"the historical script supports at most {len(REFERENCE_LABELS)}"
                )
            for label, reference in zip(REFERENCE_LABELS, references, strict=False):
                filename = model_path / f"Reference.{label}.{index}.txt"
                filename.write_bytes(clean(reference).encode("utf-8"))

        rouge = Rouge155(str(rouge_path))
        rouge.system_dir = str(system_path)
        rouge.model_dir = str(model_path)
        rouge.system_filename_pattern = r"Model.(\d+).txt"
        rouge.model_filename_pattern = "Reference.[A-Z].#ID#.txt"
        args = f"-e {rouge_path / 'data'} -a -m -n 2 -d"
        raw_output = rouge.convert_and_evaluate(rouge_args=args)
        scores = _convert_scores(rouge.output_to_dict(raw_output))
        return {"count": len(pairs), **scores}
    finally:
        if remove_after:
            shutil.rmtree(work_root, ignore_errors=True)


def compare_section(old: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"count_old": old.get("count"), "count_current": current.get("count")}
    for key in METRIC_KEYS:
        if key not in old or key not in current:
            continue
        old_score = float(old[key])
        current_score = float(current[key])
        comparison[key] = {
            "old": old_score,
            "current": current_score,
            "delta_old_minus_current": old_score - current_score,
            "delta_percentage_points": 100.0 * (old_score - current_score),
        }
    return comparison


def write_comparison(model_dir: Path, old_report: dict[str, Any]) -> Path | None:
    current_path = model_dir / "rouge_summary.json"
    if not current_path.is_file():
        return None
    current = json.loads(current_path.read_text(encoding="utf-8"))
    comparison = {
        "model": old_report["model"],
        "positive_delta_means": "historical PyROUGE score is higher",
        "overall": compare_section(old_report["overall"], current.get("overall", {})),
        "datasets": {},
    }
    current_datasets = current.get("datasets", {})
    for dataset, old_scores in old_report["datasets"].items():
        if dataset in current_datasets:
            comparison["datasets"][dataset] = compare_section(old_scores, current_datasets[dataset])
    output_path = model_dir / "rouge_protocol_comparison.json"
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def evaluate_model(
    model: str,
    pairs: list[Pair],
    model_dir: Path,
    rouge_path: Path,
    keep_temp: Path | None,
    generation_config: dict[str, Any],
    compare: bool = True,
) -> None:
    by_dataset: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        by_dataset[pair.dataset].append(pair)

    print(f"\n{model}: historical PyROUGE on {len(pairs)} files")
    overall = run_pyrouge(pairs, rouge_path, keep_temp)
    datasets: dict[str, Any] = {}
    for dataset, dataset_pairs in sorted(by_dataset.items()):
        print(f"  scoring {dataset} ({len(dataset_pairs)} files)")
        datasets[dataset] = run_pyrouge(dataset_pairs, rouge_path, keep_temp)

    report = {
        "model": model,
        "implementation": "PyROUGE with ROUGE-1.5.5.pl",
        "rouge_path": str(rouge_path),
        "rouge_args": "-a -m -n 2 -d",
        "score_scale": "0_to_1",
        "display_scale": "percentage",
        "prediction_preprocessing": (f"lowercase, legacy bracket remapping, remove lines containing {PREAMBLE!r}"),
        "reference_preprocessing": (
            "blank-separated blocks are multiple references; lowercase and legacy bracket remapping"
        ),
        "aggregation": "ROUGE-1.5.5 corpus/system aggregation",
        "generation": generation_config,
        "overall": overall,
        "datasets": datasets,
    }
    output_path = model_dir / "rouge_old_summary.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "  ROUGE-1/2/L F1 (%): "
        f"{100.0 * overall['rouge1_fmeasure']:.3f} / "
        f"{100.0 * overall['rouge2_fmeasure']:.3f} / "
        f"{100.0 * overall['rougeL_fmeasure']:.3f}"
    )
    print(f"  Historical summary: {output_path}")
    if compare:
        comparison_path = write_comparison(model_dir, report)
        if comparison_path is not None:
            print(f"  Protocol comparison: {comparison_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summaries from local sources and score them with historical ROUGE-1.5.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-raw-root",
        "--source-root",
        dest="source_root",
        type=Path,
        default=DATA_RAW_ROOT,
        help="Root containing <dataset>/*.txt source documents",
    )
    parser.add_argument("--data-ref-root", type=Path, default=DATA_REF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DATA_SUM_ROOT)
    parser.add_argument(
        "--backend",
        choices=("transformers", "afmr"),
        default="transformers",
        help="Use a local HF checkpoint or the AFMR runtime directly, without vLLM.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Hugging Face checkpoint for --backend transformers; no network download is performed",
    )
    parser.add_argument("--afmr-config", type=Path, default=None, help="AFMR resolved_config.yaml")
    parser.add_argument("--afmr-checkpoint", type=Path, default=None, help="AFMR .pt checkpoint")
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        default=None,
        help="Output folder name under data_sum (defaults to the local checkpoint directory name)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help="Override source limit; otherwise Transformers uses 8192 and AFMR uses its config value",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override output limit; otherwise Transformers uses 512 and AFMR uses its config value",
    )
    parser.add_argument("--device", default="auto", help="Device such as cuda:0, cpu, or auto")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT, help="Use {source} where source text goes")
    parser.add_argument("--no-chat-template", action="store_true", help="Use the plain prompt, even if available")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--rouge-path", type=Path, default=ROUGE_PATH)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip source files without a same-named reference instead of failing.",
    )
    parser.add_argument(
        "--keep-temp",
        type=Path,
        default=None,
        help="Keep generated PyROUGE input directories below this directory.",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Run only historical PyROUGE; do not read or write comparison results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backend == "transformers" and (args.afmr_config is not None or args.afmr_checkpoint is not None):
        print("ERROR: --afmr-config/--afmr-checkpoint require --backend afmr", file=sys.stderr)
        return 2
    if args.backend == "afmr" and args.model_path is not None:
        print("ERROR: use --afmr-config and --afmr-checkpoint with --backend afmr, not --model-path", file=sys.stderr)
        return 2
    if args.backend == "afmr" and ((args.afmr_config is None) != (args.afmr_checkpoint is None)):
        print("ERROR: --backend afmr requires both --afmr-config and --afmr-checkpoint", file=sys.stderr)
        return 2

    afmr_generation = args.backend == "afmr" and args.afmr_config is not None
    model_name = args.model_name
    if model_name is None and args.model_path is not None:
        model_name = args.model_path.name
    if model_name is None and args.afmr_checkpoint is not None:
        model_name = args.afmr_checkpoint.parent.name
    if not model_name:
        print(
            "ERROR: pass a model/checkpoint path for generation or --model-name to score existing predictions",
            file=sys.stderr,
        )
        return 2
    model_dir = args.output_root.expanduser().resolve() / model_name
    if not args.model_path and not afmr_generation and not model_dir.is_dir():
        print(
            f"ERROR: prediction directory not found: {model_dir}; pass model checkpoint arguments to generate them",
            file=sys.stderr,
        )
        return 2

    rouge_path = args.rouge_path.expanduser().resolve()
    try:
        pairs, missing, empty_sources = discover_source_pairs(
            args.source_root.expanduser().resolve(),
            args.data_ref_root,
            model_dir,
            model_name,
        )
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if empty_sources:
        print(f"Skipped {len(empty_sources)} empty/whitespace-only source files.", file=sys.stderr)
        for path in empty_sources[:20]:
            print(f"  empty: {path}", file=sys.stderr)
        if len(empty_sources) > 20:
            print(f"  ... and {len(empty_sources) - 20} more", file=sys.stderr)

    if missing:
        print(f"Found {len(missing)} source files without matching references.", file=sys.stderr)
        for path in missing[:20]:
            print(f"  missing: {path}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        if not args.allow_missing:
            print("Use --allow-missing to evaluate matched pairs only.", file=sys.stderr)
            return 2
    if not pairs:
        print("ERROR: no matching source/reference .txt files found", file=sys.stderr)
        return 2

    try:
        validate_rouge_install(rouge_path)
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        import pyrouge  # noqa: F401
    except ImportError:
        print("ERROR: pyrouge is not installed in this environment", file=sys.stderr)
        return 2

    if afmr_generation:
        try:
            generate_afmr_predictions(
                pairs,
                model_dir,
                args.afmr_config,
                args.afmr_checkpoint,
                batch_size=args.batch_size,
                device=args.device,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                overwrite=args.overwrite_predictions,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR during AFMR prediction: {exc}", file=sys.stderr)
            return 2
    elif args.model_path:
        try:
            generate_predictions(
                pairs,
                args.model_path,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens if args.max_input_tokens is not None else 8192,
                max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 512,
                prompt_template=args.prompt_template,
                use_chat_template=not args.no_chat_template,
                device_name=args.device,
                dtype_name=args.dtype,
                trust_remote_code=args.trust_remote_code,
                overwrite=args.overwrite_predictions,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR during prediction: {exc}", file=sys.stderr)
            return 2

    missing_predictions = [str(pair.prediction) for pair in pairs if not pair.prediction.is_file()]
    if missing_predictions:
        print(
            f"ERROR: {len(missing_predictions)} prediction(s) are missing; pass --model-path to generate them",
            file=sys.stderr,
        )
        for path in missing_predictions[:20]:
            print(f"  missing: {path}", file=sys.stderr)
        return 2

    started = time.monotonic()
    try:
        evaluate_model(
            model_name,
            pairs,
            model_dir,
            rouge_path,
            args.keep_temp,
            generation_config={
                "backend": args.backend if args.model_path or afmr_generation else None,
                "checkpoint": (
                    str(args.model_path.expanduser().resolve())
                    if args.model_path
                    else str(args.afmr_checkpoint.expanduser().resolve())
                    if afmr_generation
                    else None
                ),
                "afmr_config": str(args.afmr_config.expanduser().resolve()) if afmr_generation else None,
                "prompt_template": args.prompt_template if args.model_path else None,
                "uses_tokenizer_chat_template": not args.no_chat_template if args.model_path else None,
                "batch_size": args.batch_size if args.model_path or afmr_generation else None,
                "max_input_tokens": (args.max_input_tokens if args.max_input_tokens is not None else 8192)
                if args.model_path
                else args.max_input_tokens,
                "max_new_tokens": (args.max_new_tokens if args.max_new_tokens is not None else 512)
                if args.model_path
                else args.max_new_tokens,
                "device": args.device if args.model_path or afmr_generation else None,
                "dtype": args.dtype if args.model_path else None,
                "existing_predictions_reused": bool(
                    (args.model_path or afmr_generation) and not args.overwrite_predictions
                ),
            },
            compare=not args.no_compare,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"\nCompleted historical ROUGE evaluation in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
