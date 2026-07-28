"""Split-aware EviSeq evaluation; final paper test is an explicit operation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from . import runtime_evaluate as stable
from .config import load_config
from .data import dataset_fingerprint
from .model import EviSeq
from .parameter_manifest import build_parameter_manifest


def _evaluation_config(path: str, split: str) -> Dict[str, Any]:
    config = load_config(path)
    if split == "validation":
        config["data"]["test_file"] = config["data"]["validation_file"]
        config.setdefault("limits", {})["max_test_examples"] = int(
            config.get("limits", {}).get("max_validation_examples", 0)
        )
    return config


def _verify_checkpoint_data(
    checkpoint: Path,
    config: Dict[str, Any],
    split: str,
    max_samples: int,
) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    resolved_config_path = checkpoint.parent / "resolved_config.yaml"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config sidecar: {resolved_config_path}")
    checkpoint_config = load_config(resolved_config_path)
    if checkpoint_config["_meta"]["architecture_sha256"] != config["_meta"]["architecture_sha256"]:
        raise RuntimeError("Evaluation architecture does not match the resolved config saved beside last.pt")
    if checkpoint_config["_meta"]["evaluation_contract_sha256"] != config["_meta"]["evaluation_contract_sha256"]:
        raise RuntimeError("Prompt/generation evaluation contract differs from the config saved beside last.pt")
    if checkpoint_config.get("_runtime", {}).get("parameter_manifest") != config.get("_runtime", {}).get(
        "parameter_manifest"
    ):
        raise RuntimeError("Evaluation parameter manifest does not match the checkpoint sidecar")
    manifest_path = checkpoint.parent / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}; EviSeq will not load a multi-GB checkpoint merely to discover data provenance"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get(split)
    if expected is None:
        raise RuntimeError(f"Checkpoint has no {split} data fingerprint")
    configured_limit = int(config.get("limits", {}).get(f"max_{split}_examples", 0))
    effective = int(max_samples) if int(max_samples) > 0 else configured_limit
    actual = dataset_fingerprint(config["data"][f"{split}_file"], effective)
    matches = bool(actual["sha256"] == expected["sha256"] and actual["num_examples"] == expected["num_examples"])
    if split == "test" and not matches:
        raise RuntimeError(f"Test fingerprint mismatch: checkpoint={expected}, current={actual}")
    return actual, expected, matches


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_identity(value: Any) -> tuple[int, str]:
    if not isinstance(value, dict):
        raise RuntimeError("Checkpoint data fingerprint is missing")
    return int(value.get("num_examples", -1)), str(value.get("sha256", ""))


def _reserve_paper_test(
    checkpoint: Path,
    output: Path,
    config: Dict[str, Any],
    test_fingerprint: Dict[str, Any],
) -> tuple[Path, Dict[str, Any]]:
    """Create a checkpoint-local, one-shot paper-test reservation.

    The reservation deliberately survives a failed or interrupted decode.  A
    retry then requires a conscious audit and manual removal instead of
    silently turning the test set into a generation-hyperparameter dev set.
    """

    output = output.expanduser().resolve()
    derived = (
        output,
        output.with_suffix(".metrics.json"),
        output.with_suffix(".rouge155.json"),
        output.with_suffix(".rouge155.raw.txt"),
        output.with_suffix(".rouge_data"),
    )
    existing = [str(path) for path in derived if path.exists()]
    if existing:
        raise FileExistsError("Paper-test artifacts already exist; refusing silent overwrite: " + ", ".join(existing))
    marker = checkpoint.parent.resolve() / "paper_test_manifest.json"
    payload = {
        "status": "reserved",
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "output": str(output),
        "architecture_sha256": config["_meta"]["architecture_sha256"],
        "inference_protocol_sha256": config["_meta"]["inference_protocol_sha256"],
        "evaluation_contract_sha256": config["_meta"]["evaluation_contract_sha256"],
        "test_data_fingerprint": test_fingerprint,
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"Paper test was already reserved or completed for this checkpoint: {marker}") from exc
    return marker, payload


def _complete_paper_test(
    marker: Path,
    reservation: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    completed = {
        **reservation,
        "status": "complete",
        "predictions_sha256": metrics["predictions_sha256"],
        "num_examples": int(metrics["num_examples"]),
        "metrics_file": str(Path(metrics["predictions_file"]).with_suffix(".metrics.json")),
    }
    temporary = marker.with_name(marker.name + ".tmp")
    temporary.write_text(
        json.dumps(completed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def _load_verified_checkpoint(
    model: EviSeq,
    path: Path,
    *,
    config: Dict[str, Any],
    checkpoint: Path,
    original_loader: Any,
    actual_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    runtime_manifest = build_parameter_manifest(model, config)
    expected_manifest = config.get("_runtime", {}).get("parameter_manifest")
    if runtime_manifest != expected_manifest:
        raise RuntimeError("Instantiated model parameter count does not match resolved_config.yaml")
    manifest_path = checkpoint.parent / "parameter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint parameter manifest: {manifest_path}")
    standalone_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if standalone_manifest != runtime_manifest:
        raise RuntimeError("Standalone parameter manifest does not match the instantiated model")

    payload = original_loader(model, path)
    payload_config = payload.get("config")
    if not isinstance(payload_config, dict):
        raise RuntimeError("Checkpoint payload has no embedded resolved config")
    if payload_config.get("_meta", {}).get("architecture_sha256") != config["_meta"]["architecture_sha256"]:
        raise RuntimeError("Checkpoint payload architecture does not match the evaluation config")
    if (
        payload_config.get("_meta", {}).get("evaluation_contract_sha256")
        != config["_meta"]["evaluation_contract_sha256"]
    ):
        raise RuntimeError("Checkpoint-embedded prompt/generation contract does not match the evaluation config")
    if payload_config.get("_runtime", {}).get("parameter_manifest") != runtime_manifest:
        raise RuntimeError("Checkpoint-embedded parameter manifest does not match the model")

    sidecar_data = json.loads((checkpoint.parent / "data_manifest.json").read_text(encoding="utf-8"))
    payload_data = payload.get("data_manifest")
    if not isinstance(payload_data, dict):
        raise RuntimeError("Checkpoint payload has no embedded data manifest")
    for name in ("train", "validation", "test"):
        if _fingerprint_identity(payload_data.get(name)) != _fingerprint_identity(sidecar_data.get(name)):
            raise RuntimeError(f"Checkpoint-embedded {name} fingerprint differs from sidecar")
    actual_manifest.update(runtime_manifest)
    return payload


def evaluate(
    config_path: str,
    checkpoint_path: str,
    output_path: str,
    *,
    split: str = "validation",
    max_samples: int = 0,
    paper_test: bool = False,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if split == "test" and not paper_test:
        raise ValueError("Full test is locked; pass --paper-test after freezing the configuration")
    if paper_test and (split != "test" or int(max_samples) != 0):
        raise ValueError("Paper test must evaluate the complete test split with max_samples=0")
    checkpoint = Path(checkpoint_path)
    config = load_config(config_path)
    if paper_test and int(config.get("limits", {}).get("max_test_examples", 0)) != 0:
        raise ValueError("Paper test requires limits.max_test_examples=0")
    current_fingerprint, checkpoint_fingerprint, fingerprint_matches = _verify_checkpoint_data(
        checkpoint, config, split, max_samples
    )
    paper_reservation: tuple[Path, Dict[str, Any]] | None = None
    if paper_test:
        paper_reservation = _reserve_paper_test(
            checkpoint,
            Path(output_path),
            config,
            current_fingerprint,
        )

    actual_manifest: Dict[str, Any] = {}
    original_class = stable.LLM2SeqV2
    original_loader = stable.load_config
    original_checkpoint_loader = stable.load_last_checkpoint

    def load_verified_checkpoint(model: EviSeq, path: Path) -> Dict[str, Any]:
        return _load_verified_checkpoint(
            model,
            path,
            config=config,
            checkpoint=checkpoint,
            original_loader=original_checkpoint_loader,
            actual_manifest=actual_manifest,
        )

    stable.LLM2SeqV2 = EviSeq
    stable.load_config = lambda _: _evaluation_config(config_path, split)
    stable.load_last_checkpoint = load_verified_checkpoint
    try:
        metrics = stable.evaluate(config_path, checkpoint_path, output_path, max_samples)
    finally:
        stable.LLM2SeqV2 = original_class
        stable.load_config = original_loader
        stable.load_last_checkpoint = original_checkpoint_loader
    if not actual_manifest:
        raise RuntimeError("Evaluation did not verify the instantiated model parameter manifest")

    metrics["evaluation_split"] = split
    metrics["paper_test"] = bool(paper_test)
    metrics["architecture_sha256"] = config["_meta"]["architecture_sha256"]
    metrics["inference_protocol_sha256"] = config["_meta"]["inference_protocol_sha256"]
    metrics["evaluation_contract_sha256"] = config["_meta"]["evaluation_contract_sha256"]
    metrics["native_attention_variant"] = config["native_attention"]["variant"]
    metrics["final_graph"] = "one_encoder_one_decoder"
    output = Path(output_path).resolve()
    if not output.is_file():
        raise FileNotFoundError(f"Evaluation did not write predictions: {output}")
    generation = config.get("generation", {})
    metrics.update(
        {
            "evaluation_data_fingerprint": current_fingerprint,
            "checkpoint_data_fingerprint": checkpoint_fingerprint,
            "checkpoint_data_matches_current": fingerprint_matches,
            "predictions_file": str(output),
            "predictions_sha256": _file_sha256(output),
            "generation": {
                "max_new_tokens": int(generation.get("max_new_tokens", 256)),
                "min_new_tokens": int(generation.get("min_new_tokens", 16)),
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": float(generation.get("repetition_penalty", 1.05)),
                "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 3)),
            },
            "source_prefix": str(config["data"].get("source_prefix", "")),
            "max_source_length": int(config["data"]["max_source_length"]),
            "max_target_length": int(config["data"]["max_target_length"]),
            "checkpoint_parameters_match_model": True,
        }
    )
    if split == "test":
        metrics["test_data_fingerprint"] = current_fingerprint
        metrics["checkpoint_test_data_fingerprint"] = checkpoint_fingerprint
        metrics["checkpoint_test_matches_current"] = fingerprint_matches
    metrics["parameter_manifest"] = actual_manifest
    metrics["training_parameters"] = int(actual_manifest["resident_training_total_unique"])
    metrics["deployable_parameters"] = int(actual_manifest["deployable_resident_without_train_aux"])
    metrics["inference_active_parameters"] = int(actual_manifest["inference_active_unique"])
    target = config.get("benchmark", {}).get("diagnostic", {})
    if split == "test" and all(name in target for name in ("rouge1", "rouge2", "rougeL")):
        metrics["diagnostic_gap_to_t5gemma"] = {
            name: round(float(metrics[name]) - float(target[name]), 4) for name in ("rouge1", "rouge2", "rougeL")
        }
    Path(output_path).with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if paper_reservation is not None:
        _complete_paper_test(*paper_reservation, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--paper-test", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.config,
        args.checkpoint,
        args.output,
        split=args.split,
        max_samples=args.max_samples,
        paper_test=args.paper_test,
    )


if __name__ == "__main__":
    main()
