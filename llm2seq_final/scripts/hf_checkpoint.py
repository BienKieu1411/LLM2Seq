#!/usr/bin/env python3
"""Resolve local checkpoints, with optional Hugging Face fallback."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - install_deps.sh installs pyyaml on server.
    yaml = None


H200_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = H200_ROOT.parent

PHASE_REMOTE_DIRS = {
    "phase1": "checkpoints/h200_phase1_warmup",
    "h200_phase1_warmup": "checkpoints/h200_phase1_warmup",
    "phase2": "checkpoints/h200_phase2_lora_encoder",
    "h200_phase2_lora_encoder": "checkpoints/h200_phase2_lora_encoder",
    "phase3": "checkpoints/h200_phase3_mtp_self_distill",
    "h200_phase3_mtp_self_distill": "checkpoints/h200_phase3_mtp_self_distill",
}


def log(message: str) -> None:
    print(message, file=sys.stderr)


def load_env_file() -> None:
    env_file = Path(os.environ.get("ENV_FILE", H200_ROOT / "env.txt"))
    if not env_file.is_absolute():
        env_file = PROJECT_ROOT / env_file
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


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    if yaml is None:
        raise RuntimeError("pyyaml is required to resolve a checkpoint from config.")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_phase_from_path(path: str) -> Optional[str]:
    lowered = path.lower()
    if "phase1" in lowered or "warmup" in lowered:
        return "phase1"
    if "phase2" in lowered or "lora_encoder" in lowered:
        return "phase2"
    if "phase3" in lowered or "mtp_self_distill" in lowered:
        return "phase3"
    return None


def remote_dir_from_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    if args.path_in_repo:
        return args.path_in_repo.strip("/")
    if args.phase:
        phase = args.phase.strip()
        if phase in PHASE_REMOTE_DIRS:
            return PHASE_REMOTE_DIRS[phase]
        raise ValueError(f"Unknown phase '{phase}'. Use one of: {', '.join(sorted(PHASE_REMOTE_DIRS))}")
    hf_cfg = cfg.get("huggingface", {})
    if hf_cfg.get("path_in_repo"):
        return str(hf_cfg["path_in_repo"]).strip("/")
    if args.local:
        inferred = infer_phase_from_path(args.local)
        if inferred:
            return PHASE_REMOTE_DIRS[inferred]
    raise ValueError("Cannot infer HF checkpoint path. Pass --phase, --config, or --path_in_repo.")


def checkpoint_names(local_path: str, requested_name: Optional[str]) -> list[str]:
    if requested_name:
        names = [requested_name]
    else:
        basename = Path(local_path).name if local_path else "best.pt"
        names = [basename]
    for fallback in ("best.pt", "final.pt"):
        if fallback not in names:
            names.append(fallback)
    return names


def download_checkpoint(
    repo_id: str,
    repo_type: str,
    token: Optional[str],
    remote_dir: str,
    names: list[str],
    cache_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    last_error: Optional[BaseException] = None
    for name in names:
        remote_file = f"{remote_dir.rstrip('/')}/{name}"
        try:
            log(f"Downloading HF checkpoint: {repo_id}/{remote_file}")
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                    filename=remote_file,
                    local_dir=str(cache_dir),
                )
            )
        except Exception as exc:  # Try best.pt then final.pt.
            last_error = exc
            log(f"HF checkpoint not available: {remote_file} ({exc})")
    assert last_error is not None
    raise last_error


def resolve(args: argparse.Namespace) -> Path:
    load_env_file()

    local_path = Path(args.local) if args.local else None
    if local_path and local_path.exists():
        return local_path

    if os.environ.get("HF_AUTO_DOWNLOAD_CHECKPOINTS", "true").lower() not in {"1", "true", "yes", "on"}:
        if local_path:
            raise FileNotFoundError(local_path)
        raise FileNotFoundError("No local checkpoint and HF_AUTO_DOWNLOAD_CHECKPOINTS=false")

    cfg = load_config(args.config)
    hf_cfg = cfg.get("huggingface", {})
    repo_id = args.repo_id or hf_cfg.get("repo_id") or os.environ.get("HF_REPO_ID")
    repo_type = args.repo_type or hf_cfg.get("repo_type", "model")
    token = args.token or os.environ.get("HF_TOKEN")
    if not repo_id:
        if local_path:
            raise FileNotFoundError(f"{local_path}; HF_REPO_ID is not set for fallback download.")
        raise EnvironmentError("HF_REPO_ID is not set for fallback download.")

    remote_dir = remote_dir_from_args(args, cfg)
    cache_dir = Path(os.environ.get("HF_CHECKPOINT_CACHE", "runs/hf_checkpoints"))
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    names = checkpoint_names(str(local_path) if local_path else "", args.filename)
    return download_checkpoint(repo_id, repo_type, token, remote_dir, names, cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--local", default=None, help="Preferred local checkpoint path.")
    resolve_parser.add_argument("--config", default=None, help="Config YAML whose huggingface.path_in_repo should be used.")
    resolve_parser.add_argument("--phase", default=None, help="phase1, phase2, phase3, or full training stage name.")
    resolve_parser.add_argument("--path_in_repo", default=None, help="Remote HF folder containing best.pt/final.pt.")
    resolve_parser.add_argument("--filename", default=None, help="Remote checkpoint filename. Defaults to local basename.")
    resolve_parser.add_argument("--repo_id", default=None)
    resolve_parser.add_argument("--repo_type", default=None)
    resolve_parser.add_argument("--token", default=None)

    args = parser.parse_args()
    if args.command == "resolve":
        path = resolve(args)
        print(path)


if __name__ == "__main__":
    main()
