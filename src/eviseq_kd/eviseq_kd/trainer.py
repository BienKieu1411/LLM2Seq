"""Standalone training entry point for EviSeq-KD.

It reuses the stable EviSeq engine at runtime through local monkey-patching,
while every KD-specific model/dataset/config lives under ``eviseq_kd``.  The
legacy source files are never edited or imported with KD symbols.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch

from .cache import TeacherCache, load_cache
from .build_cache import build_cache as build_teacher_cache
from .dataset import KDCollator, KDText2TextDataset
from .model import EviSeqKD
from .student.configuration import load_config
from .student.data.dataset import Text2TextDataset, decoder_seed_ids
from .student.training import engine as stable
from .student.training import trainer as legacy_trainer


_ORIGINAL_COLLATOR = stable._collator
LOGGER = logging.getLogger("eviseq_kd.trainer")


def _resolve_path(value: str, config: Dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = config.get("_meta", {}).get("config_path")
    candidates = []
    if config_path:
        candidates.append(Path(str(config_path)).resolve().parent / path)
    package_root = Path(__file__).resolve().parents[1]
    if path.parts[:2] == ("src", "eviseq_kd"):
        candidates.append(package_root.joinpath(*path.parts[2:]))
    elif path.parts[:1] == ("eviseq_kd",):
        candidates.append(package_root.joinpath(*path.parts[1:]))
    candidates.extend(
        [
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_teacher_cache(config: Dict[str, Any]) -> TeacherCache:
    distillation = config.get("training", {}).get("distillation", {})
    if not bool(distillation.get("enabled", False)):
        raise ValueError("eviseq_kd requires training.distillation.enabled=true")
    cache_path = str(distillation.get("cache_path", "")).strip()
    if not cache_path:
        raise ValueError("KD is enabled but training.distillation.cache_path is empty")
    cache = load_cache(_resolve_path(cache_path, config))
    expected_split = str(distillation.get("cache_split", "train"))
    actual_split = str(cache.metadata.get("split", ""))
    if actual_split != expected_split:
        raise ValueError(
            f"Teacher cache split mismatch: expected {expected_split!r}, found {actual_split or '<missing>'!r}. "
            "Rebuild the cache with --force-rebuild-cache."
        )
    expected_teacher = str(distillation.get("teacher_model", "")).strip()
    actual_teacher = str(cache.metadata.get("teacher_model", "")).strip()
    if expected_teacher and actual_teacher != expected_teacher:
        raise ValueError(
            f"Teacher cache model mismatch: expected {expected_teacher!r}, found {actual_teacher or '<missing>'!r}. "
            "Rebuild the cache with --force-rebuild-cache."
        )
    return cache


def _ensure_teacher_cache(config_path: str, *, auto_build: bool, force_rebuild: bool = False) -> Path:
    """Materialize the offline teacher cache on demand for one-command training."""

    config = load_config(config_path)
    distillation = config.get("training", {}).get("distillation", {})
    if not bool(distillation.get("enabled", False)):
        return Path()
    cache_path = str(distillation.get("cache_path", "")).strip()
    if not cache_path:
        raise ValueError("KD is enabled but training.distillation.cache_path is empty")
    resolved_cache = _resolve_path(cache_path, config)
    if resolved_cache.is_file() and not force_rebuild:
        LOGGER.info("reusing teacher cache: %s", resolved_cache)
        return resolved_cache
    if not auto_build:
        raise FileNotFoundError(
            f"Teacher cache not found: {resolved_cache}. Either build it with eviseq_kd.build_cache "
            "or allow trainer auto-build."
        )

    limits = config.get("limits", {})
    generation = config.get("generation", {})
    teacher_model = str(distillation.get("teacher_model", "Qwen/Qwen3-4B")).strip()
    LOGGER.info("teacher cache missing; generating it now with %s", teacher_model)
    built_cache = build_teacher_cache(
        config_path,
        str(resolved_cache),
        teacher_model_name=teacher_model,
        split=str(distillation.get("cache_split", "train")),
        max_examples=int(distillation.get("cache_max_examples", limits.get("max_train_examples", 0))),
        device_name=str(distillation.get("teacher_device", "auto")),
        batch_size=int(distillation.get("teacher_batch_size", 1)),
        max_input_length=int(distillation.get("teacher_max_input_length", config["data"]["max_source_length"])),
        max_new_tokens=int(distillation.get("teacher_max_new_tokens", generation.get("max_new_tokens", 384))),
        num_beams=int(distillation.get("teacher_num_beams", generation.get("num_beams", 4))),
    )
    if not built_cache.is_file():
        raise RuntimeError(f"Teacher cache generation finished without creating: {built_cache}")
    LOGGER.info("teacher cache ready: %s", built_cache)
    return resolved_cache


def build_experiment(config: Dict[str, Any], *, include_train: bool = True):
    """Build the KD model and datasets without changing EviSeq's builder."""

    encoder_tokenizer, decoder_tokenizer = stable._tokenizers(config)
    model = EviSeqKD(config)
    data = config["data"]
    limits = config.get("limits", {})
    teacher_cache = _load_teacher_cache(config) if include_train else None
    train_dataset = None
    if include_train:
        assert teacher_cache is not None
        train_dataset = KDText2TextDataset(
            _resolve_path(str(data["train_file"]), config),
            encoder_tokenizer,
            decoder_tokenizer,
            data,
            max_examples=int(limits.get("max_train_examples", 0)),
            precompute_evidence=bool(data.get("precompute_evidence", True)),
            teacher_cache=teacher_cache,
            require_teacher_cache=True,
        )
    validation_dataset = Text2TextDataset(
        _resolve_path(str(data["validation_file"]), config),
        encoder_tokenizer,
        decoder_tokenizer,
        data,
        max_examples=int(limits.get("max_validation_examples", 0)),
        precompute_evidence=bool(data.get("precompute_validation_evidence", False)),
    )
    return model, encoder_tokenizer, decoder_tokenizer, train_dataset, validation_dataset


def _collator(config: Dict[str, Any], encoder_tokenizer: Any, decoder_tokenizer: Any) -> KDCollator:
    base = _ORIGINAL_COLLATOR(config, encoder_tokenizer, decoder_tokenizer)
    return KDCollator(
        base,
        decoder_pad_id=int(decoder_tokenizer.pad_token_id),
        max_decoder_length=int(config["data"]["max_target_length"])
        + len(decoder_seed_ids(decoder_tokenizer, config["data"]))
        - 1,
    )


def _parameter_component(name: str) -> str:
    name = name[5:] if name.startswith("base.") else name
    if (
        name.startswith("adapter.")
        or name.startswith("alignment_head.")
        or name.startswith("evidence_contrastive_head.")
    ):
        return "adapter"
    if name.startswith("encoder.") and any(
        marker in name for marker in ("evidence_norm", "evidence_head", "evidence_view_gate", "generic_token_gate")
    ):
        return "adapter"
    if ".cross_attn" in name or name.endswith(".cross_gate") or name.endswith(".memory_router_logits"):
        return "cross_attention"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("decoder."):
        return "decoder"
    raise ValueError(f"Unclassified trainable parameter: {name}")


def _initialize_from_checkpoint_compat(
    model: torch.nn.Module, path: str | Path, *, strict: bool = True
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a model state dictionary")
    expected = set(model.state_dict())
    if not any(key in expected for key in state) and any(not key.startswith("base.") for key in state):
        state = {key if key.startswith("base.") else f"base.{key}": value for key, value in state.items()}
    if strict:
        model.load_state_dict(state, strict=True)
        loaded = len(state)
        skipped: list[str] = []
    else:
        compatible = {
            name: value
            for name, value in state.items()
            if name in expected and tuple(value.shape) == tuple(expected[name].shape)
        }
        if not compatible:
            raise RuntimeError("No compatible parameters were found in the initialization checkpoint")
        model.load_state_dict(compatible, strict=False)
        loaded = len(compatible)
        skipped = sorted(set(state) - set(compatible))
    return {
        "epoch": int(payload.get("epoch", 0)) if isinstance(payload, dict) else 0,
        "global_step": int(payload.get("global_step", 0)) if isinstance(payload, dict) else 0,
        "loaded_tensors": loaded,
        "skipped_tensors": skipped,
    }


def train(
    config_path: str,
    overwrite_output_dir: bool = False,
    init_checkpoint: str = "",
    output_dir: str = "",
    allow_partial_init: bool = False,
    auto_build_cache: bool = True,
    force_rebuild_cache: bool = False,
) -> Path:
    _ensure_teacher_cache(config_path, auto_build=auto_build_cache, force_rebuild=force_rebuild_cache)
    originals = {
        "legacy_EviSeq": legacy_trainer.EviSeq,
        "legacy_build_experiment": legacy_trainer.build_experiment,
        "legacy_parameter_component": legacy_trainer._parameter_component,
        "stable_collator": stable._collator,
        "stable_initialize_from_checkpoint": stable.initialize_from_checkpoint,
    }
    # The bundled student trainer owns the evidence-contrastive warmup and
    # GradCache implementation. We invoke it with a temporary local model and
    # dataset builder; no external EviSeq package is imported.
    legacy_trainer.EviSeq = EviSeqKD
    legacy_trainer.build_experiment = build_experiment
    legacy_trainer._parameter_component = _parameter_component
    stable._collator = _collator
    stable.initialize_from_checkpoint = _initialize_from_checkpoint_compat
    try:
        return legacy_trainer.train(config_path, overwrite_output_dir, init_checkpoint, output_dir, allow_partial_init)
    finally:
        legacy_trainer.EviSeq = originals["legacy_EviSeq"]
        legacy_trainer.build_experiment = originals["legacy_build_experiment"]
        legacy_trainer._parameter_component = originals["legacy_parameter_component"]
        stable._collator = originals["stable_collator"]
        stable.initialize_from_checkpoint = originals["stable_initialize_from_checkpoint"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train the independent EviSeq-KD pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-partial-init", action="store_true")
    parser.add_argument(
        "--no-auto-build-cache",
        action="store_true",
        help="Fail if the configured teacher cache is missing instead of generating it",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Regenerate the configured teacher cache before training",
    )
    args = parser.parse_args()
    train(
        args.config,
        args.overwrite_output_dir,
        args.init_checkpoint,
        args.output_dir,
        args.allow_partial_init,
        auto_build_cache=not args.no_auto_build_cache,
        force_rebuild_cache=args.force_rebuild_cache,
    )


if __name__ == "__main__":
    main()
