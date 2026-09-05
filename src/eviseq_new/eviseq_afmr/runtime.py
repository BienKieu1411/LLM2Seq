"""Runtime assembly for real Transformers checkpoints.

Importing this module does not instantiate or download a model. Checkpoint
loading happens only when the train/evaluate commands are invoked.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .config import load_config, resolve_path
from .data.collate import SummarizationCollator
from .data.dataset import JsonlSummarizationDataset
from .data.sampling import LengthBucketBatchSampler
from .modeling.model import EviSeqAFMR
from .training.checkpoint import load_checkpoint
from .training.engine import AFMRTrainer, seed_everything

LOGGER = logging.getLogger("eviseq_afmr.runtime")


def _configure_precision(config: dict[str, Any]) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", False))


class _TinyTokenizer:
    """Deterministic tokenizer used only by the offline integration smoke."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
    ) -> dict:
        matches = list(re.finditer(r"\S+", str(text)))
        tokens = [match.group() for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        ids = []
        for token in tokens:
            value = sum((position + 1) * ord(character) for position, character in enumerate(token))
            ids.append(4 + value % 124)
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
            offsets = [(0, 0), *offsets, (0, 0)]
        if truncation and max_length is not None:
            ids, offsets = ids[:max_length], offsets[:max_length]
        return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = True) -> list[str]:
        rows = sequences.tolist() if isinstance(sequences, torch.Tensor) else sequences
        output = []
        for row in rows:
            tokens = []
            for value in row:
                value = int(value)
                if skip_special_tokens and value in {self.pad_token_id, self.bos_token_id, self.eos_token_id}:
                    continue
                tokens.append(f"tok{value}")
            output.append(" ".join(tokens))
        return output


def _tokenizers(config: dict[str, Any]):
    model = config["model"]
    kwargs = {
        "use_fast": bool(model.get("tokenizer_use_fast", True)),
        "trust_remote_code": bool(model.get("trust_remote_code", True)),
    }
    from transformers import AutoTokenizer

    encoder = (
        _TinyTokenizer()
        if str(model["encoder_name"]) == "__tiny__"
        else AutoTokenizer.from_pretrained(str(model["encoder_name"]), **kwargs)
    )
    decoder = (
        _TinyTokenizer()
        if str(model["decoder_name"]) == "__tiny__"
        else AutoTokenizer.from_pretrained(str(model["decoder_name"]), **kwargs)
    )
    if decoder.pad_token_id is None:
        decoder.pad_token = decoder.eos_token or decoder.unk_token
    if encoder.pad_token_id is None:
        encoder.pad_token = encoder.eos_token or encoder.unk_token
    return encoder, decoder


def build_loaders(
    config: dict[str, Any],
    split: str | None = None,
    batch_size_override: int | None = None,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
):
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    data = config["data"]
    paths = {"train": data["train_file"], "validation": data["validation_file"], "test": data["test_file"]}
    selected = (split,) if split else ("train", "validation")
    loaders = {}
    for name in selected:
        split_data = copy.deepcopy(data)
        limit = max_train_examples if name == "train" else max_validation_examples if name == "validation" else 0
        dataset = JsonlSummarizationDataset(resolve_path(paths[name], config), split_data, max_examples=limit)
        collator = SummarizationCollator(encoder_tokenizer, decoder_tokenizer, split_data)
        batch_size = (
            int(config["training"].get("validation_batch_size", 4))
            if name != "train"
            else int(config["training"].get("batch_size", 4))
        )
        if batch_size_override is not None:
            batch_size = int(batch_size_override)
        workers = (
            int(config["training"].get("num_workers", 0))
            if name == "train"
            else int(config["training"].get("validation_num_workers", 0))
        )
        sampling = {"batch_size": batch_size, "shuffle": name == "train"}
        if name == "train" and config["training"].get("length_bucketing", False):
            sampling = {
                "batch_sampler": LengthBucketBatchSampler(
                    dataset.length_estimates,
                    batch_size,
                    int(config["training"].get("seed", 42)),
                    int(config["training"].get("length_bucket_multiplier", 50)),
                )
            }
        loaders[name] = DataLoader(
            dataset,
            **sampling,
            num_workers=workers,
            persistent_workers=workers > 0 and bool(config["training"].get("persistent_workers", True)),
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        LOGGER.info(
            "[data] split=%s | examples=%d | batch=%d | workers=%d | length_bucketing=%s",
            name,
            len(dataset),
            batch_size,
            workers,
            "batch_sampler" in sampling,
        )
    return loaders


def _write_resolved_config(config: dict[str, Any], output_dir: Path) -> None:
    resolved = copy.deepcopy(config)
    resolved.pop("_meta", None)
    for key in ("train_file", "validation_file", "test_file"):
        resolved["data"][key] = str(resolve_path(resolved["data"][key], config))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)


def _clear_run_artifacts(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for pattern in ("*.pt", "*.jsonl", "*.json", "resolved_config.yaml"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def train(
    config_path: str | Path,
    *,
    device: str | None = None,
    resume_checkpoint: str | None = None,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
    overwrite_output_dir: bool = False,
    output_dir_override: str | None = None,
) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This AFMR runner is single-process; do not launch it with torchrun/DDP")
    config = load_config(config_path)
    _configure_precision(config)
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override
    checkpoint = resume_checkpoint or str(config["training"].get("resume_checkpoint", "")).strip()
    if overwrite_output_dir and checkpoint:
        raise ValueError("--overwrite-output-dir cannot be combined with --resume-checkpoint")
    output_dir = resolve_path(config["experiment"]["output_dir"], config)
    config["experiment"]["output_dir"] = str(output_dir)
    if overwrite_output_dir:
        _clear_run_artifacts(output_dir)
    elif not checkpoint and output_dir.exists() and any(output_dir.glob("*.pt")):
        raise FileExistsError(f"Existing checkpoints in {output_dir}; resume or explicitly use --overwrite-output-dir")
    seed_everything(int(config["training"].get("seed", 42)))
    _write_resolved_config(config, output_dir)
    loaders = build_loaders(
        config, max_train_examples=max_train_examples, max_validation_examples=max_validation_examples
    )
    model = EviSeqAFMR(config)
    counts = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in (("encoder", model.encoder), ("bridge", model.bridge), ("decoder", model.decoder))
    }
    LOGGER.info("model parameters=%s total=%d", counts, sum(p.numel() for p in model.parameters()))
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = AFMRTrainer(model, config, selected_device)
    if checkpoint:
        LOGGER.info("resumed AFMR checkpoint: %s", checkpoint)
    trainer.fit(loaders["train"], loaders.get("validation"), resume_checkpoint=checkpoint or None)


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "test",
    batch_size: int | None = None,
    device: str | None = None,
    max_examples: int = 0,
) -> dict[str, Any]:
    config = load_config(config_path)
    _configure_precision(config)
    selected_batch_size = int(batch_size if batch_size is not None else config["generation"]["batch_size"])
    if selected_batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    loaders = build_loaders(config, split=split, batch_size_override=selected_batch_size)
    from .evaluation.generate import append_jsonl, generate_greedy

    loader = loaders[split]
    loader.collate_fn.include_targets = False
    seen = set()
    predictions: list[str] = []
    references: list[str] = []
    started = time.monotonic()
    output_file = Path(output_path)
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                index = len(predictions)
                if index >= len(loader.dataset):
                    raise ValueError("Resume file contains more rows than the active split")
                expected = loader.dataset[index]
                example_id = str(row.get("id", ""))
                if example_id in seen or example_id != expected.example_id or row.get("reference") != expected.target:
                    raise ValueError(f"Evaluation resume must match the exact ID/reference prefix; mismatch at {index}")
                if not isinstance(row.get("prediction"), str):
                    raise ValueError(f"Missing prediction at resume row {index}")
                seen.add(example_id)
                predictions.append(row["prediction"])
                references.append(row["reference"])
    resumed_count = len(predictions)
    total = min(len(loader.dataset), max_examples) if max_examples > 0 else len(loader.dataset)
    if resumed_count > total:
        raise ValueError("Resume file exceeds requested max_examples")
    LOGGER.info(
        "resuming evaluation: %d/%d predictions already present; batch_size=%d",
        resumed_count,
        total,
        selected_batch_size,
    )
    if resumed_count == total:
        from .evaluation.metrics import summarization_metrics

        result = summarization_metrics(predictions, references)
        Path(str(output_path) + ".metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "[evaluation] complete | split=%s | examples=%d | ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f",
            split,
            total,
            result["rouge1"],
            result["rouge2"],
            result["rougeL"],
        )
        return result
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EviSeqAFMR(config).to(device_obj)
    load_checkpoint(checkpoint_path, model, config=config, restore_rng=False)
    model.eval()
    loader = DataLoader(
        Subset(loader.dataset, range(resumed_count, total)),
        batch_size=selected_batch_size,
        collate_fn=loader.collate_fn,
        num_workers=loader.num_workers,
        pin_memory=device_obj.type == "cuda",
    )
    started = time.monotonic()
    generated_this_run = 0
    pending_batches = []
    for batch in loader:
        pending_batches.append(batch)
        while pending_batches:
            batch = pending_batches.pop(0)
            if len(set(batch["ids"])) != len(batch["ids"]) or any(str(value) in seen for value in batch["ids"]):
                raise ValueError("Evaluation requires unique example IDs; repeated ID in active split")
            device_obj = next(model.parameters()).device
            narrowed = {
                key: value.to(device_obj, non_blocking=True) if isinstance(value, torch.Tensor) else list(value)
                for key, value in batch.items()
                if key
                in {
                    "input_ids",
                    "attention_mask",
                    "source_content_mask",
                    "decoder_prompt_ids",
                    "decoder_prompt_mask",
                    "ids",
                    "references",
                }
            }
            try:
                texts, _ = generate_greedy(
                    model,
                    narrowed,
                    loaders[split].collate_fn.decoder_tokenizer,
                    int(config["generation"]["max_new_tokens"]),
                    int(config["generation"].get("min_new_tokens", 0)),
                    float(config["generation"].get("repetition_penalty", 1.0)),
                    int(config["generation"].get("no_repeat_ngram_size", 0)),
                    bool(config["generation"].get("compact_finished", True)),
                )
            except torch.cuda.OutOfMemoryError:
                size = len(narrowed["ids"])
                if device_obj.type != "cuda" or size <= 1:
                    raise
                half = size // 2
                LOGGER.warning(
                    "[evaluation] CUDA OOM | retrying batch=%d as %d+%d",
                    size,
                    half,
                    size - half,
                )
                torch.cuda.empty_cache()
                left = {
                    key: value[:half].detach().cpu() if isinstance(value, torch.Tensor) else value[:half]
                    for key, value in narrowed.items()
                }
                right = {
                    key: value[half:].detach().cpu() if isinstance(value, torch.Tensor) else value[half:]
                    for key, value in narrowed.items()
                }
                del narrowed, batch
                torch.cuda.empty_cache()
                pending_batches[0:0] = [left, right]
                continue
            append_jsonl(
                output_path,
                (
                    {"id": example_id, "prediction": text, "reference": reference}
                    for example_id, text, reference in zip(narrowed["ids"], texts, narrowed["references"])
                ),
            )
            seen.update(str(value) for value in narrowed["ids"])
            predictions.extend(texts)
            references.extend(narrowed["references"])
            generated_this_run += len(texts)
            processed = resumed_count + generated_this_run
            elapsed = time.monotonic() - started
            eta = elapsed * max(0, total - processed) / max(1, generated_this_run)
            peak = torch.cuda.max_memory_allocated(device_obj) / 2**30 if device_obj.type == "cuda" else 0
            message = (
                "[evaluation] split=%s | processed=%d/%d | %.2f ex/s | elapsed=%.1fs | ETA=%.1fs | peak=%.2f GiB"
                % (
                    split,
                    processed,
                    total,
                    generated_this_run / max(elapsed, 1e-6),
                    elapsed,
                    eta,
                    peak,
                )
            )
            LOGGER.info(message)
            print(message, flush=True)
    from .evaluation.metrics import summarization_metrics

    result = summarization_metrics(predictions, references)
    Path(str(output_path) + ".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "[evaluation] complete | split=%s | examples=%d | ROUGE-1=%.3f | ROUGE-2=%.3f | ROUGE-L=%.3f",
        split,
        total,
        result["rouge1"],
        result["rouge2"],
        result["rougeL"],
    )
    return result
