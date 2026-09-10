"""Bind resumable predictions to the checkpoint and effective evaluation inputs.

Digests are computed once at evaluation startup, never in the training loop.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from ..config import config_fingerprint
from ..training.checkpoint import architecture_spec


def _file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_digest(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        state = backend.to_str()
    elif hasattr(tokenizer, "get_vocab"):
        state = tokenizer.get_vocab()
    else:
        # The offline smoke tokenizer has no vocab file or Rust backend.
        state = inspect.getsource(type(tokenizer))
    settings = {
        key: getattr(tokenizer, key, None)
        for key in (
            "special_tokens_map",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "unk_token_id",
            "chat_template",
            "padding_side",
            "truncation_side",
            "clean_up_tokenization_spaces",
        )
    }
    return _object_digest({"class": type(tokenizer).__qualname__, "state": state, "settings": settings})


def evaluation_identity(config, checkpoint_path, dataset, collator, split, device_type) -> dict[str, Any]:
    """Exclude batch size and prefix limit so interrupted evaluation can continue."""
    effective = {key: config.get(key, {}) for key in ("model", "encoder", "architecture", "decoder")}
    effective["data"] = {
        key: value for key, value in config["data"].items() if key not in {"train_file", "validation_file", "test_file"}
    }
    effective["generation"] = {key: value for key, value in config["generation"].items() if key != "batch_size"}
    effective["tf32"] = bool(config["training"].get("tf32", False))
    # Include implementation changes even if graph/weight shapes remain equal.
    package = Path(__file__).resolve().parents[1]
    implementation = {str(path.relative_to(package)): _file_digest(path) for path in sorted(package.rglob("*.py"))}
    libraries = {}
    for name in ("torch", "transformers", "tokenizers"):
        try:
            libraries[name] = version(name)
        except PackageNotFoundError:
            libraries[name] = None
    return {
        "schema_version": 2,
        "checkpoint_sha256": _file_digest(checkpoint_path),
        "dataset_sha256": _file_digest(dataset.path),
        "split": split,
        "evaluation_config_sha256": _object_digest(effective),
        "encoder_tokenizer_sha256": _tokenizer_digest(collator.encoder_tokenizer),
        "decoder_tokenizer_sha256": _tokenizer_digest(collator.decoder_tokenizer),
        "prompt_ids_sha256": _object_digest(collator._prompt_ids),
        "implementation_sha256": _object_digest(implementation),
        "libraries": libraries,
        "device_type": device_type,
        "config_fingerprint": config_fingerprint(config),
        "architecture_spec": architecture_spec(config),
    }


def verify_checkpoint_metadata(checkpoint_path: str | Path, config: dict[str, Any]) -> None:
    """Validate structural metadata even when all prediction rows are cached.

    This lightweight read avoids constructing a decoder solely to decide
    whether a complete prediction file may be used.
    """

    state = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if state.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Checkpoint config_fingerprint does not match the active evaluation config")
    if state.get("architecture_spec") != architecture_spec(config):
        raise ValueError("Checkpoint architecture_spec does not match the active evaluation graph")


def ensure_evaluation_manifest(output_path: str | Path, identity: dict[str, Any]) -> None:
    """Reject unverified complete files and partial files from a different run."""
    output = Path(output_path)
    manifest = Path(str(output) + ".manifest.json")
    if output.exists() and output.stat().st_size:
        if not manifest.is_file():
            raise ValueError(
                f"Existing predictions have no evaluation manifest: {output}. "
                "Choose a new output path to evaluate this checkpoint."
            )
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(f"Invalid evaluation manifest: {manifest}")
        if saved != identity:
            mismatches = sorted(key for key in saved.keys() | identity.keys() if saved.get(key) != identity.get(key))
            raise ValueError(
                f"Evaluation provenance mismatch ({', '.join(mismatches)}): {output}. "
                "Choose a new output path; predictions from different runs cannot be mixed."
            )
        return
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=manifest.name + ".", dir=manifest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
