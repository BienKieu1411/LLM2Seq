"""Build and launch a native vLLM HTTP server for a local SEAM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from eviseq_afmr.config import load_config, resolve_path

from ._bootstrap import ensure_source_plugin
from ._compat import validate_decoder_config


def build_command(config_path: Path, checkpoint_path: Path, args) -> list[str]:
    from transformers import AutoConfig, AutoTokenizer

    config_path = config_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SEAM checkpoint not found: {checkpoint_path}")
    config = load_config(config_path)
    model = config["model"]
    decoder = resolve_path(model["decoder_name"], config)
    encoder = resolve_path(model["encoder_name"], config)
    for name, path in (("encoder", encoder), ("decoder", decoder)):
        if not path.is_dir():
            raise FileNotFoundError(f"SEAM {name} must be available locally: {path}")
    trust = bool(model.get("trust_remote_code", True))
    validate_decoder_config(AutoConfig.from_pretrained(str(decoder), trust_remote_code=trust, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(str(decoder), trust_remote_code=trust, local_files_only=True)
    if config["decoder"].get("grounded_copy", {}).get("enabled", False) and not tokenizer.is_fast:
        raise ValueError("SEAM grounded copy requires the trained decoder's fast tokenizer")
    overrides = {
        "architectures": ["SEAMForQwen3"],
        "is_encoder_decoder": True,
        "seam_config_path": str(config_path),
        "seam_checkpoint_path": str(checkpoint_path),
        "max_source_positions": int(config["data"]["max_source_length"]),
        "eos_token_id": tokenizer.eos_token_id,
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(decoder),
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        str(model.get("compute_dtype", model.get("dtype", "float32"))),
        "--hf-overrides",
        json.dumps(overrides),
        "--middleware",
        "seam_vllm.http.SEAMMiddleware",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--kv-cache-dtype",
        "auto",
        "--generation-config",
        "vllm",
        "--runner",
        "generate",
        # Input state is attached to each source, so disable processor cache
        # paths that can replace its multimodal payload with a cache reference.
        "--mm-processor-cache-gb",
        "0",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if trust:
        command.append("--trust-remote-code")
    for name in ("max_model_len", "max_num_batched_tokens", "api_key"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Local resolved_config.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local SEAM .pt checkpoint")
    parser.add_argument("--served-model-name", default="seam")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None, help="Decoder context limit")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--api-key", default=None, help="Native vLLM authentication key (or set VLLM_API_KEY)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print native launch command without loading model weights"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_num_seqs < 1 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("max-num-seqs must be positive and gpu-memory-utilization must be in (0, 1]")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
        raise ValueError("SEAM requires VLLM_USE_V2_MODEL_RUNNER=1")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    command = build_command(args.config, args.checkpoint, args)
    if args.dry_run:
        # Keep API keys out of terminal logs and copied dry-run commands.
        printable = list(command)
        if "--api-key" in printable:
            printable[printable.index("--api-key") + 1] = "<redacted>"
        print(shlex.join(printable))
        return
    ensure_source_plugin()
    import vllm

    if vllm.__version__.split("+")[0] != "0.25.1":
        raise RuntimeError(f"SEAM HTTP requires vLLM 0.25.1; received {vllm.__version__}")
    from vllm.entrypoints.cli.main import main as vllm_main

    # Keep source-only entry-point metadata alive until the native host and its
    # workers stop. execv would discard its cleanup handler and Python state.
    sys.argv = ["vllm", *command[3:]]
    vllm_main()


if __name__ == "__main__":
    main()
