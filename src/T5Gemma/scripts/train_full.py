#!/usr/bin/env python3
"""Full fine-tune T5Gemma for abstractive summarization."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

T5GEMMA_ROOT = Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    env_file = Path(os.environ.get("ENV_FILE", T5GEMMA_ROOT / "env.txt"))
    if not env_file.is_absolute():
        env_file = T5GEMMA_ROOT.parent / env_file
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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def jsonl_fingerprint(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            digest.update(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "source": row.get("source"),
                        "target": row.get("target"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
            count += 1
    if count == 0:
        raise ValueError(f"Dataset is empty: {path}")
    return {
        "path": str(path.resolve()),
        "num_examples": count,
        "sha256": digest.hexdigest(),
    }


class SummarizationDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        source_prefix: str,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.examples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.source_prefix = source_prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.examples[index]
        model_inputs = self.tokenizer(
            self.source_prefix + row["source"],
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=row["target"],
            max_length=self.max_target_length - 1,  # Leave room for EOS
            truncation=True,
        )
        label_ids = labels["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            if not label_ids or label_ids[-1] != self.tokenizer.eos_token_id:
                label_ids.append(self.tokenizer.eos_token_id)

        model_inputs["labels"] = label_ids
        return model_inputs


def torch_dtype_from_config(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if lowered in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def make_training_arguments(cfg: Dict[str, Any], output_dir: Path) -> Seq2SeqTrainingArguments:
    train_cfg = cfg["training"]
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": int(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg["learning_rate"]),
        "adam_beta1": float(train_cfg.get("adam_beta1", 0.9)),
        "adam_beta2": float(train_cfg.get("adam_beta2", 0.95)),
        "adam_epsilon": float(train_cfg.get("adam_epsilon", 1e-8)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "cosine")),
        "optim": str(train_cfg.get("optim", "adamw_torch")),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", False)),
        "tf32": bool(train_cfg.get("tf32", True)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "logging_strategy": "steps",
        "save_strategy": "no",
        "save_safetensors": True,
        "report_to": [],
        "predict_with_generate": False,
        "group_by_length": bool(train_cfg.get("group_by_length", False)),
        "remove_unused_columns": True,
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": True,
        "seed": int(train_cfg.get("seed", 42)),
    }
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    valid_kwargs = {k: v for k, v in kwargs.items() if k in params}
    # Full checkpoints are large. Evaluation is performed once from final_model
    # by the pipeline, not implicitly at every training epoch.
    eval_strat = str(train_cfg.get("eval_strategy", "no"))
    if "eval_strategy" in params:
        valid_kwargs["eval_strategy"] = eval_strat
    else:
        valid_kwargs["evaluation_strategy"] = eval_strat
    return Seq2SeqTrainingArguments(**valid_kwargs)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_copy_config(config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "training_config.yaml")


def get_hf_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hf_cfg = cfg.get("huggingface", {})
    return {
        "enabled": bool(hf_cfg.get("enabled", False)),
        "repo_id": os.environ.get("HF_REPO_ID") or hf_cfg.get("repo_id"),
        "repo_type": hf_cfg.get("repo_type", "model"),
        "token": os.environ.get("HF_TOKEN"),
        "path_in_repo": str(hf_cfg.get("path_in_repo", "checkpoints/t5gemma2_1b_1b_full_wikilingua")).strip("/"),
        "push_final": bool(hf_cfg.get("push_final", True)),
        "fail_on_error": bool(hf_cfg.get("fail_on_error", True)),
        "private": bool(hf_cfg.get("private", False)),
    }


def upload_folder(folder: Path, hf: Dict[str, Any], path_in_repo: str, message: str) -> None:
    if not hf["enabled"]:
        return
    if not hf["repo_id"] or not hf["token"]:
        logging.warning("Skipping HF upload because HF_REPO_ID or HF_TOKEN is not set.")
        return
    try:
        api = HfApi(token=hf["token"])
        api.create_repo(
            repo_id=hf["repo_id"],
            repo_type=hf["repo_type"],
            private=hf["private"],
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=hf["repo_id"],
            repo_type=hf["repo_type"],
            folder_path=str(folder),
            path_in_repo=path_in_repo.strip("/"),
            commit_message=message,
        )
        logging.info("Uploaded %s -> %s/%s", folder, hf["repo_id"], path_in_repo)
    except Exception as exc:
        if hf["fail_on_error"]:
            raise
        logging.warning("HF upload failed: %s", exc)


def log_model_summary(model: torch.nn.Module, cfg: Dict[str, Any], train_size: int, eval_size: int) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    ratio = 100.0 * trainable / max(1, total)
    logging.info("T5Gemma Full Fine-tuning Summary")
    logging.info("=" * 50)
    logging.info("Model:            %s", cfg["model"]["model_name_or_path"])
    logging.info("Source/Target:    %s / %s tokens", cfg["data"]["max_source_length"], cfg["data"]["max_target_length"])
    logging.info("Train examples:   %s", train_size)
    logging.info("Eval examples:    %s", eval_size)
    logging.info("Epochs:           %s", cfg["training"]["num_train_epochs"])
    logging.info("Batch size:       %s", cfg["training"]["per_device_train_batch_size"])
    logging.info("Grad accum:       %s", cfg["training"]["gradient_accumulation_steps"])
    logging.info(
        "Effective batch:  %s",
        int(cfg["training"]["per_device_train_batch_size"]) * int(cfg["training"]["gradient_accumulation_steps"]),
    )
    logging.info("Learning rate:    %s", cfg["training"]["learning_rate"])
    logging.info("Trainable params: %s", f"{trainable:,}")
    logging.info("Total params:     %s", f"{total:,}")
    logging.info("Trainable ratio:  %.4f%%", ratio)
    logging.info("=" * 50)
    if trainable != total:
        frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
        raise RuntimeError(
            "Full fine-tuning requires 100% trainable parameters, but found "
            f"{len(frozen)} frozen tensors: {frozen[:20]}"
        )


def main() -> None:
    load_env_file()
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if "lora" in cfg:
        raise ValueError("This is a full fine-tuning pipeline; remove the obsolete 'lora' config block.")
    mode = str(cfg.get("training", {}).get("mode", "full_finetune"))
    if mode != "full_finetune":
        raise ValueError(f"training.mode must be 'full_finetune', got {mode!r}")

    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    if bool(cfg["training"].get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(cfg["project"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite_output_dir:
            raise FileExistsError(
                f"Refusing to mix a new T5Gemma run with existing artifacts in {output_dir}. "
                "Use a fresh project.output_dir or pass --overwrite-output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    running_marker = output_dir / "RUNNING"
    running_marker.write_text(
        json.dumps(
            {"config": str(config_path.resolve()), "status": "running"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    safe_copy_config(config_path, output_dir)

    token = os.environ.get("HF_TOKEN")
    model_name = cfg["model"]["model_name_or_path"]
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    dtype = torch_dtype_from_config(cfg["model"].get("torch_dtype", "bfloat16"))

    train_file = Path(cfg["data"]["train_file"])
    eval_file_str = cfg["data"].get("eval_file")
    eval_file = Path(eval_file_str) if eval_file_str else None
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if eval_file and not eval_file.exists():
        raise FileNotFoundError(eval_file)
    data_manifest = {"train": jsonl_fingerprint(train_file)}
    if eval_file:
        data_manifest["validation"] = jsonl_fingerprint(eval_file)
    test_file_value = cfg.get("data", {}).get("test_file")
    if test_file_value and Path(test_file_value).exists():
        data_manifest["test"] = jsonl_fingerprint(Path(test_file_value))
    logging.info("Data manifest: %s", json.dumps(data_manifest, ensure_ascii=False))

    logging.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    logging.info("Loading base model: %s", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if bool(cfg["training"].get("require_fp32_master_weights", True)):
        low_precision = [
            f"{name}:{parameter.dtype}"
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if low_precision:
            raise RuntimeError(
                "Full fine-tuning requires FP32 master parameters; set "
                "model.torch_dtype=float32. Low-precision tensors: "
                + ", ".join(low_precision[:20])
            )

    train_dataset = SummarizationDataset(
        train_file,
        tokenizer,
        cfg["data"].get("source_prefix", ""),
        int(cfg["data"]["max_source_length"]),
        int(cfg["data"]["max_target_length"]),
    )
    eval_dataset = None
    if eval_file:
        eval_dataset = SummarizationDataset(
            eval_file,
            tokenizer,
            cfg["data"].get("source_prefix", ""),
            int(cfg["data"]["max_source_length"]),
            int(cfg["data"]["max_target_length"]),
        )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    log_model_summary(model, cfg, len(train_dataset), len(eval_dataset) if eval_dataset else 0)
    training_args = make_training_arguments(cfg, output_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logging.info("Starting full fine-tuning of all T5Gemma parameters...")
    train_result = trainer.train()
    logging.info("Training complete: %s", train_result.metrics)

    final_folder = output_dir / "final_model"
    final_folder.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_folder))
    tokenizer.save_pretrained(final_folder)
    safe_copy_config(config_path, final_folder)
    write_json(
        final_folder / "checkpoint_manifest.json",
        {
            "tag": "final",
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch or 0.0),
            "base_model": model_name,
            "stores_base_model_weights": True,
            "checkpoint_type": "full_finetuned_seq2seq_model",
            "trainable_ratio_percent": 100.0,
            "metrics": train_result.metrics,
            "data": cfg.get("data", {}),
            "data_manifest": data_manifest,
            "generation": cfg.get("generation", {}),
        },
    )
    write_json(output_dir / "train_metrics.json", train_result.metrics)
    running_marker.unlink(missing_ok=True)

    epochs_dir = output_dir / "epochs"
    if epochs_dir.exists():
        shutil.rmtree(epochs_dir, ignore_errors=True)
        logging.info("Deleted all epoch checkpoints after phase completion to save disk space.")

    hf = get_hf_settings(cfg)
    if hf["push_final"]:
        upload_folder(final_folder, hf, f"{hf['path_in_repo']}/final_model", "T5Gemma full fine-tuned model")


if __name__ == "__main__":
    main()
