"""Run a model x dataset matrix sequentially on one visible GPU."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_MODEL_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
    return result


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _selected(value: str | None, env_name: str, default: list[str], available: dict[str, Any]) -> list[str]:
    raw = value or os.environ.get(env_name, "")
    names = [part.strip() for part in raw.split(",") if part.strip()] if raw else list(default)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"Unknown {env_name.lower()} entry(s): {unknown}; available={sorted(available)}")
    if not names:
        raise ValueError(f"No entries selected for {env_name.lower()}")
    return names


def _resolve_model(model_name: str, spec: dict[str, Any]) -> str:
    env_name = str(spec.get("path_env", "")).strip()
    if env_name:
        configured = os.environ.get(env_name, "").strip()
        if configured:
            return str(Path(configured).expanduser().resolve())
    model_root = os.environ.get("MODEL_ROOT", "").strip()
    local_dir = str(spec.get("local_dir", "")).strip()
    if model_root and local_dir:
        return str((Path(model_root).expanduser() / local_dir).resolve())
    if local_dir:
        # Keep the fallback local as well.  The run preflight below will give
        # a clear error if this directory is not present; never pass a Hub
        # repository ID to Transformers.
        return str(Path(local_dir).expanduser().resolve())
    raise ValueError(
        f"No local path configured for {model_name}; set {env_name or 'MODEL_ROOT'} or provide a model.local_dir"
    )


def build_run_config(
    suite: dict[str, Any],
    suite_path: Path,
    model_name: str,
    dataset_name: str,
    *,
    max_train_examples: int = 0,
    max_validation_examples: int = 0,
    max_test_examples: int = 0,
) -> tuple[dict[str, Any], Path]:
    models = suite["models"]
    datasets = suite["datasets"]
    model_spec = models[model_name]
    dataset_spec = datasets[dataset_name]
    defaults = suite.get("defaults", {})
    output_root = _resolve(suite.get("output_root", "../../../runs/decoder_baselines"), suite_path.parent)
    data_root = _resolve(suite.get("data_root", "../../eviseq_new/datasets"), suite_path.parent)
    data_dir = _resolve(dataset_spec.get("data_dir", dataset_name), data_root)
    run_name = f"{model_name}__{dataset_name}"
    model_config = _merge(defaults.get("model", {}), model_spec.get("model", {}))
    model_config.update(
        {
            "model_id": str(model_spec.get("model_id", model_config.get("model_id", model_name))),
            "name_or_path": _resolve_model(model_name, model_spec),
            "family": str(model_spec.get("family", model_config.get("family", "causal_lm"))),
            "local_files_only": True,
        }
    )
    if "diffusion_paradigm" in model_spec:
        model_config["diffusion_paradigm"] = model_spec["diffusion_paradigm"]
    data_config = _merge(defaults.get("data", {}), dataset_spec)
    for key in ("data_dir", "training", "generation"):
        data_config.pop(key, None)
    data_config["dataset"] = dataset_name
    data_config.update(
        {
            "train_file": str(data_dir / str(dataset_spec.get("train_file", "train.jsonl"))),
            "validation_file": str(data_dir / str(dataset_spec.get("validation_file", "validation.jsonl"))),
            "test_file": str(data_dir / str(dataset_spec.get("test_file", "test.jsonl"))),
        }
    )
    training = _merge(defaults.get("training", {}), model_spec.get("training", {}), dataset_spec.get("training", {}))
    limits = {
        "max_train_examples": int(max_train_examples),
        "max_validation_examples": int(max_validation_examples),
        "max_test_examples": int(max_test_examples),
    }
    config: dict[str, Any] = {
        "run": {"name": run_name, "output_dir": str(output_root / run_name)},
        "model": model_config,
        "data": data_config,
        "training": training,
        # Evaluation memory is independent from training memory.  A model may
        # therefore override generation.batch_size while dataset recipes still
        # provide task-specific length limits.
        "generation": _merge(
            defaults.get("generation", {}),
            model_spec.get("generation", {}),
            dataset_spec.get("generation", {}),
        ),
        "limits": limits,
    }
    safe_name = _MODEL_NAME.sub("_", run_name)
    config_path = output_root / ".configs" / f"{safe_name}.yaml"
    return config, config_path


def _write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _data_preflight(config: dict[str, Any]) -> None:
    missing = [
        config["data"][key]
        for key in ("train_file", "validation_file", "test_file")
        if not Path(config["data"][key]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing prepared dataset files: " + ", ".join(missing))


def _model_preflight(config: dict[str, Any]) -> None:
    model_path = Path(config["model"]["name_or_path"]).expanduser()
    if not model_path.is_dir():
        model_id = config["model"].get("model_id", "unknown")
        raise FileNotFoundError(
            "Local model directory does not exist: "
            f"{model_path}. Set the corresponding *_PATH variable (model_id={model_id}); "
            "Hugging Face loading/downloads are disabled."
        )


def run_suite(args: argparse.Namespace) -> int:
    suite_path = Path(args.config).expanduser().resolve()
    with suite_path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    if (
        not isinstance(suite, dict)
        or not isinstance(suite.get("models"), dict)
        or not isinstance(suite.get("datasets"), dict)
    ):
        raise ValueError("Suite config requires mapping sections: models and datasets")
    model_names = _selected(
        args.models, "DECODER_MODELS", suite.get("model_order", list(suite["models"])), suite["models"]
    )
    dataset_names = _selected(
        args.datasets, "DECODER_DATASETS", suite.get("dataset_order", list(suite["datasets"])), suite["datasets"]
    )
    gpu = os.environ.get("GPU_ID", str(suite.get("gpu", "0"))).strip()
    if not gpu or "," in gpu or " " in gpu:
        raise ValueError("GPU_ID must identify exactly one GPU, for example GPU_ID=0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"
    output_root = _resolve(suite.get("output_root", "../../../runs/decoder_baselines"), suite_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "suite_status.jsonl"
    entries = [(model_name, dataset_name) for model_name in model_names for dataset_name in dataset_names]
    print(f"Single-GPU sequential suite: GPU_ID={gpu}; runs={len(entries)}")
    for index, (model_name, dataset_name) in enumerate(entries, start=1):
        config, config_path = build_run_config(
            suite,
            suite_path,
            model_name,
            dataset_name,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            max_test_examples=args.max_test_examples,
        )
        _write_config(config, config_path)
        if not args.dry_run:
            _model_preflight(config)
            _data_preflight(config)
        run_dir = Path(config["run"]["output_dir"])
        print(f"[{index}/{len(entries)}] {model_name} on {dataset_name} -> {run_dir}")
        started = time.time()
        status = "planned"
        error = ""
        if not args.dry_run:
            command_env = os.environ.copy()
            train_command = [sys.executable, "-m", "decoder_baselines.train", "--config", str(config_path)]
            if args.overwrite_output_dir:
                train_command.append("--overwrite-output-dir")
            try:
                subprocess.run(train_command, cwd=str(suite_path.parents[2]), env=command_env, check=True)
                if not args.skip_eval:
                    prediction_path = run_dir / f"{args.split}_predictions.jsonl"
                    evaluate_command = [
                        sys.executable,
                        "-m",
                        "decoder_baselines.evaluate",
                        "--config",
                        str(config_path),
                        "--checkpoint",
                        str(run_dir / "final_model"),
                        "--output",
                        str(prediction_path),
                        "--split",
                        args.split,
                    ]
                    if args.max_eval_examples > 0:
                        evaluate_command.extend(["--max-examples", str(args.max_eval_examples)])
                    subprocess.run(evaluate_command, cwd=str(suite_path.parents[2]), env=command_env, check=True)
                status = "complete"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"exit_status={exc.returncode}"
                if not args.continue_on_error:
                    with status_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps({"model": model_name, "dataset": dataset_name, "status": status, "error": error})
                            + "\n"
                        )
                    raise
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "run_dir": str(run_dir),
                        "config": str(config_path),
                        "status": status,
                        "error": error,
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Suite finished; status log: {status_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential single-GPU decoder-baseline matrix runner")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "suite.yaml"))
    parser.add_argument(
        "--models", default=None, help="Comma-separated model keys; defaults to DECODER_MODELS or suite order"
    )
    parser.add_argument(
        "--datasets", default=None, help="Comma-separated dataset keys; defaults to DECODER_DATASETS or suite order"
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--max-test-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_suite(args))


if __name__ == "__main__":
    main()
