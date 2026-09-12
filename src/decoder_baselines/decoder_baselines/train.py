"""Full fine-tuning entry point for one decoder-only baseline run."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import load_config
from .data import CausalCollator, CausalSummarizationDataset

LOGGER = logging.getLogger("decoder_baselines.train")


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"float32", "fp32"}:
        return torch.float32
    if value in {"float16", "fp16"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _context_length(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "max_seq_len", "n_positions", "max_sequence_length"):
        value = getattr(config, name, None)
        if value is not None and int(value) > 0:
            return int(value)
    raise RuntimeError("Cannot verify decoder context length")


def _load_tokenizer_and_model(config: dict[str, Any], *, evaluation: bool = False) -> tuple[Any, torch.nn.Module]:
    # A baseline run must never turn a typo in a local path into a network
    # download.  The shell runner also exports these flags, but keeping them
    # here protects direct ``python -m`` usage.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    name = str(model_config.get("name_or_path", model_config.get("model_id", "")))
    common = {
        "local_files_only": True,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }
    tokenizer = AutoTokenizer.from_pretrained(name, **common)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if evaluation else "right"
    dtype_name = model_config.get("eval_torch_dtype" if evaluation else "torch_dtype", "bfloat16")
    model_kwargs = {
        **common,
        "dtype": _dtype(str(dtype_name)),
        "attn_implementation": str(model_config.get("attn_implementation", "sdpa")),
        "low_cpu_mem_usage": True,
    }
    if str(model_config.get("family", "causal_lm")) == "nemotron_diffusion":
        # Nemotron-Labs-Diffusion exposes its tri-mode model through
        # AutoModel, not AutoModelForCausalLM.  The checkpoint is kept intact,
        # while the same weights are instantiated in AR mode so its CE loss
        # and generation semantics match the other decoder-only controls.
        raw_config = AutoConfig.from_pretrained(name, **common)
        raw_config.dlm_paradigm = str(model_config.get("diffusion_paradigm", "autoregressive"))
        model = AutoModel.from_pretrained(name, config=raw_config, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = bool(model_config.get("use_cache", evaluation))
    return tokenizer, model


def _enable_training_features(model: torch.nn.Module, config: dict[str, Any]) -> None:
    model_config = config["model"]
    training = config["training"]
    if bool(model_config.get("gradient_checkpointing", training.get("gradient_checkpointing", True))):
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if enable is None:
            raise RuntimeError("Configured gradient checkpointing is unsupported by this model")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
        input_grads = getattr(model, "enable_input_require_grads", None)
        if input_grads is not None:
            input_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def _training_arguments(config: dict[str, Any], output_dir: Path) -> Any:
    from transformers import TrainingArguments

    training = config["training"]
    values: dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": int(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "adam_beta1": float(training.get("adam_beta1", 0.9)),
        "adam_beta2": float(training.get("adam_beta2", 0.95)),
        "adam_epsilon": float(training.get("adam_epsilon", 1e-8)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": "cosine",
        "optim": str(training.get("optim", "adamw_torch_fused")),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "tf32": bool(training.get("tf32", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "logging_strategy": "steps",
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": True,
        "seed": int(training.get("seed", 42)),
        "ddp_find_unused_parameters": False,
    }
    parameters = __import__("inspect").signature(TrainingArguments.__init__).parameters
    values = {key: value for key, value in values.items() if key in parameters}
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        values["evaluation_strategy"] = "no"
    return TrainingArguments(**values)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train(config_path: str | Path, *, overwrite_output_dir: bool = False) -> Path:
    config = load_config(config_path)
    output_dir = Path(config["run"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite_output_dir:
            raise FileExistsError(
                f"Refusing to mix a decoder-baseline run with existing artifacts: {output_dir}. "
                "Use --overwrite-output-dir for an intentional rerun."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "RUNNING"
    marker.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
    started = time.time()
    try:
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "train.log", encoding="utf-8")],
            force=True,
        )
        from transformers import set_seed

        training = config["training"]
        set_seed(int(training.get("seed", 42)))
        target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if target.type != "cuda":
            LOGGER.warning("No CUDA device is visible; this run is configured for a single GPU")
        if target.type == "cuda" and bool(training.get("tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        tokenizer, model = _load_tokenizer_and_model(config)
        context_length = _context_length(model)
        configured_context = int(config["data"]["max_sequence_length"])
        if context_length < configured_context:
            LOGGER.info("Capping sequence length from %d to model context %d", configured_context, context_length)
        _enable_training_features(model, config)
        model.to(target)
        train_cfg = config.get("limits", {})
        dataset = CausalSummarizationDataset(
            config["data"]["train_file"],
            tokenizer,
            config["data"],
            max_examples=int(train_cfg.get("max_train_examples", 0)),
            model_context_length=context_length,
        )
        collator = CausalCollator(tokenizer.pad_token_id)
        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "args": _training_arguments(config, output_dir),
            "train_dataset": dataset,
            "data_collator": collator,
        }
        from transformers import Trainer

        trainer_parameters = __import__("inspect").signature(Trainer.__init__).parameters
        if "processing_class" in trainer_parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_kwargs["tokenizer"] = tokenizer
        trainer = Trainer(**trainer_kwargs)
        total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        LOGGER.info(
            "run=%s model=%s device=%s examples=%d epochs=%d batch=%d accumulation=%d parameters=%d",
            config["run"]["name"],
            config["model"].get("model_id", config["model"].get("name_or_path")),
            target,
            len(dataset),
            int(training["num_train_epochs"]),
            int(training["per_device_train_batch_size"]),
            int(training["gradient_accumulation_steps"]),
            total_parameters,
        )
        result = trainer.train()
        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.config.use_cache = True
        model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
        _write_json(
            output_dir / "run_manifest.json",
            {
                "run": config["run"]["name"],
                "model_id": config["model"].get("model_id", config["model"].get("name_or_path")),
                "model_path": config["model"].get("name_or_path"),
                "dataset": config["data"].get("dataset", ""),
                "num_train_examples": len(dataset),
                "num_epochs": int(training["num_train_epochs"]),
                "trainable_parameter_elements": total_parameters,
                "train_metrics": result.metrics,
                "elapsed_seconds": round(time.time() - started, 3),
                "prompt_protocol": "t5gemma_source_prefix_plus_causal_target_masking",
            },
        )
        marker.unlink(missing_ok=True)
        (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
        LOGGER.info("completed run=%s elapsed_seconds=%.1f", config["run"]["name"], time.time() - started)
        return final_dir
    except Exception:
        LOGGER.exception("decoder-baseline run failed: %s", config["run"]["name"])
        raise
    finally:
        marker.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full fine-tune one decoder-only summarization baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(args.config, overwrite_output_dir=args.overwrite_output_dir)


if __name__ == "__main__":
    main()
