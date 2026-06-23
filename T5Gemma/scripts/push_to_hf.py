#!/usr/bin/env python3
"""Upload a folder to Hugging Face Hub using env/config defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from huggingface_hub import HfApi


def load_env_file(root: Path) -> None:
    env_file = Path(os.environ.get("ENV_FILE", root / "env.txt"))
    if not env_file.is_absolute():
        env_file = root.parent / env_file
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_env_file(root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "configs/wikilingua_lora_3072.yaml"))
    parser.add_argument("--folder", required=True)
    parser.add_argument("--path_in_repo", default=None)
    parser.add_argument("--commit_message", default="Upload T5Gemma baseline artifact")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    hf_cfg = cfg.get("huggingface", {})
    repo_id = os.environ.get("HF_REPO_ID") or hf_cfg.get("repo_id")
    token = os.environ.get("HF_TOKEN")
    repo_type = hf_cfg.get("repo_type", "model")
    if not repo_id:
        raise SystemExit("HF_REPO_ID is not set.")
    if not token:
        raise SystemExit("HF_TOKEN is not set.")

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=bool(hf_cfg.get("private", False)),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(folder),
        path_in_repo=(args.path_in_repo or hf_cfg.get("path_in_repo") or "").strip("/"),
        commit_message=args.commit_message,
    )
    print(f"Uploaded {folder} -> {repo_id}/{args.path_in_repo or hf_cfg.get('path_in_repo')}")


if __name__ == "__main__":
    main()

