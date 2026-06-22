#!/usr/bin/env python3
"""Upload H200 run artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def load_env_file() -> None:
    h200_root = Path(__file__).resolve().parents[1]
    project_root = h200_root.parent
    env_file = Path(os.environ.get("ENV_FILE", h200_root / "env.txt"))
    if not env_file.is_absolute():
        env_file = project_root / env_file
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def validate_checkpoint_for_upload(path: Path) -> None:
    """Reject checkpoints that accidentally contain base encoder weights.

    Base encoder weights (McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp) are loaded
    at runtime from HuggingFace; they must never be stored in our checkpoint
    files or uploaded to the project repo.
    """
    import torch

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 500:
        print(f"  WARNING: {path.name} is {size_mb:.1f} MB - unusually large for a trainable-only checkpoint")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        print(f"  {path.name}: skipped (no model_state_dict)")
        return
    state_dict = obj["model_state_dict"]
    bad_keys = [
        key for key in state_dict
        if key.startswith("encoder.") and "lora_" not in key
    ]
    if bad_keys:
        raise RuntimeError(
            f"BLOCKED: {path} contains {len(bad_keys)} base encoder weight(s) and "
            f"must not be uploaded. Examples: {', '.join(bad_keys[:20])}"
        )
    if obj.get("stores_base_encoder_weights") is True:
        raise RuntimeError(
            f"BLOCKED: {path} is marked stores_base_encoder_weights=True; refusing upload."
        )
    # Count encoder LoRA keys (allowed) vs total keys for transparency.
    encoder_lora_keys = [k for k in state_dict if k.startswith("encoder.") and "lora_" in k]
    print(
        f"  {path.name}: OK ({len(state_dict)} tensors, "
        f"{len(encoder_lora_keys)} encoder LoRA, "
        f"stores_base_encoder={obj.get('stores_base_encoder_weights', 'N/A')}, "
        f"{size_mb:.1f} MB)"
    )


def validate_folder_checkpoints(folder: Path, include_resume_checkpoints: bool) -> None:
    """Validate all .pt files in folder before uploading."""
    pt_files = sorted(folder.rglob("*.pt"))
    if not include_resume_checkpoints:
        pt_files = [p for p in pt_files if not p.name.startswith("checkpoint_")]
    if pt_files:
        print(f"Validating {len(pt_files)} checkpoint(s) in {folder}...")
        for path in pt_files:
            validate_checkpoint_for_upload(path)
        print("All checkpoints passed base-encoder guard.\n")
    else:
        print(f"No .pt files to validate in {folder}.\n")


def main() -> None:
    load_env_file()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        default=os.environ.get("HF_REPO_ID"),
        help="Example: username/llm2seq-h200-wikilingua. Defaults to HF_REPO_ID from env.",
    )
    parser.add_argument("--folder", action="append", required=True, help="Folder to upload. Can be repeated.")
    parser.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--commit_message", default="Upload LLM2Seq H200 artifacts")
    parser.add_argument("--path_in_repo_prefix", default="")
    parser.add_argument(
        "--include_resume_checkpoints",
        action="store_true",
        help="Also upload checkpoint_*.pt optimizer/resume checkpoints. Disabled by default.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        raise EnvironmentError("HF_REPO_ID is not set. Put it in llm2seq_h200/env.txt or pass --repo_id.")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is not set. Put it in llm2seq_h200/env.txt or export it on the server.")

    api = HfApi(token=token)
    create_repo(args.repo_id, token=token, repo_type=args.repo_type, exist_ok=True)

    for folder_str in args.folder:
        folder = Path(folder_str)
        if not folder.exists():
            raise FileNotFoundError(folder)
        path_in_repo = args.path_in_repo_prefix.strip("/")
        if path_in_repo:
            path_in_repo = f"{path_in_repo}/{folder.name}"
        else:
            path_in_repo = folder.name
        print(f"Uploading {folder} -> {args.repo_id}/{path_in_repo}")
        validate_folder_checkpoints(folder, include_resume_checkpoints=args.include_resume_checkpoints)
        ignore_patterns = [
            "__pycache__/*",
            "**/__pycache__/*",
            "*.pyc",
            "**/*.pyc",
            ".env",
            "**/.env",
            "env.txt",
            "**/env.txt",
        ]
        if not args.include_resume_checkpoints:
            # best.pt/final.pt/epoch_*.pt are trainable-only model weights.
            # checkpoint_*.pt is a local optimizer-resume file and can be huge.
            ignore_patterns.append("checkpoint_*.pt")
            ignore_patterns.append("**/checkpoint_*.pt")
        upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            folder_path=str(folder),
            path_in_repo=path_in_repo,
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
