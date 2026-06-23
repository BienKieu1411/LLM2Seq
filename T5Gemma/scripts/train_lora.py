#!/usr/bin/env python3
"""Fine-tune T5Gemma on WikiLingua with LoRA adapter checkpoints only."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
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
            max_length=self.max_target_length - 1, # Leave room for EOS
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


def module_suffix_present(model: torch.nn.Module, suffix: str) -> bool:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and (name == suffix or name.endswith(f".{suffix}")):
            return True
    return False


def resolve_lora_targets(model: torch.nn.Module, requested: List[str]) -> List[str]:
    requested = [item for item in requested if item and item != "auto"]
    selected = [name for name in requested if module_suffix_present(model, name)]
    if selected:
        skipped = [name for name in requested if name not in selected]
        if skipped:
            logging.info("Skipping absent LoRA target modules: %s", ", ".join(skipped))
        return selected

    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "q",
        "k",
        "v",
        "o",
        "wi",
        "wi_0",
        "wi_1",
        "wo",
    ]
    selected = [name for name in candidates if module_suffix_present(model, name)]
    if not selected:
        linear_suffixes = sorted({name.rsplit(".", 1)[-1] for name, mod in model.named_modules() if isinstance(mod, torch.nn.Linear)})
        raise RuntimeError(f"Could not infer LoRA targets. Linear module suffixes: {linear_suffixes[:80]}")
    logging.info("Auto-selected LoRA target modules: %s", ", ".join(selected))
    return selected


def make_training_arguments(cfg: Dict[str, Any], output_dir: Path) -> Seq2SeqTrainingArguments:
    train_cfg = cfg["training"]
    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "num_train_epochs": int(train_cfg["num_train_epochs"]),
        "per_device_train_batch_size": int(train_cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_cfg["learning_rate"]),
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
    if "eval_strategy" in params:
        valid_kwargs["eval_strategy"] = "epoch"
    else:
        valid_kwargs["evaluation_strategy"] = "epoch"
    return Seq2SeqTrainingArguments(**valid_kwargs)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


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
        "path_in_repo": str(hf_cfg.get("path_in_repo", "checkpoints/t5gemma2_1b_1b_lora_wikilingua")).strip("/"),
        "push_each_epoch": bool(hf_cfg.get("push_each_epoch", True)),
        "push_final_best": bool(hf_cfg.get("push_final_best", True)),
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


@dataclass
class AdapterCheckpointCallback(TrainerCallback):
    output_dir: Path
    tokenizer: Any
    cfg: Dict[str, Any]
    config_path: Path
    hf: Dict[str, Any]
    best_eval_loss: float = float("inf")

    def _save_adapter(
        self,
        model: torch.nn.Module,
        folder: Path,
        state: TrainerState,
        metrics: Optional[Dict[str, float]],
        tag: str,
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder, safe_serialization=True)
        self.tokenizer.save_pretrained(folder)
        safe_copy_config(self.config_path, folder)
        manifest = {
            "tag": tag,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "base_model": self.cfg["model"]["model_name_or_path"],
            "stores_base_model_weights": False,
            "checkpoint_type": "peft_lora_adapter_only",
            "metrics": metrics or {},
            "lora": self.cfg.get("lora", {}),
            "data": self.cfg.get("data", {}),
            "generation": self.cfg.get("generation", {}),
        }
        write_json(folder / "adapter_manifest.json", manifest)

    def on_evaluate(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        model = kwargs["model"]
        metrics = metrics or {}
        epoch_num = max(1, int(round(float(state.epoch or 0.0))))
        epoch_folder = self.output_dir / "epochs" / f"epoch_{epoch_num:03d}_adapter"
        self._save_adapter(model, epoch_folder, state, metrics, tag=f"epoch_{epoch_num:03d}")
        
        keep_limit = int(self.cfg.get("huggingface", {}).get("keep_local_epoch_checkpoints", 1))
        if keep_limit > 0:
            epochs_dir = self.output_dir / "epochs"
            if epochs_dir.exists():
                all_epochs = sorted([d for d in epochs_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")])
                for d in all_epochs[:-keep_limit]:
                    import shutil
                    shutil.rmtree(d, ignore_errors=True)
                    logging.info("Deleted old epoch checkpoint to save space: %s", d)
        if self.hf["push_each_epoch"]:
            upload_folder(
                epoch_folder,
                self.hf,
                f"{self.hf['path_in_repo']}/epochs/epoch_{epoch_num:03d}_adapter",
                f"T5Gemma LoRA epoch {epoch_num}",
            )

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and float(eval_loss) < self.best_eval_loss:
            self.best_eval_loss = float(eval_loss)
            best_folder = self.output_dir / "best_adapter"
            self._save_adapter(model, best_folder, state, metrics, tag="best")
            write_json(
                self.output_dir / "best_metrics.json",
                {"best_eval_loss": self.best_eval_loss, "epoch": state.epoch, "global_step": state.global_step},
            )
            logging.info("New best eval_loss=%.6f at epoch %.3f", self.best_eval_loss, float(state.epoch or 0.0))
            if self.hf["push_each_epoch"]:
                upload_folder(
                    best_folder,
                    self.hf,
                    f"{self.hf['path_in_repo']}/best_adapter",
                    f"T5Gemma LoRA best adapter step {state.global_step}",
                )
        return control


def log_model_summary(model: torch.nn.Module, cfg: Dict[str, Any], train_size: int, eval_size: int) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logging.info("T5Gemma LoRA Baseline Summary")
    logging.info("=" * 50)
    logging.info("Model:            %s", cfg["model"]["model_name_or_path"])
    logging.info("Source/Target:    %s / %s tokens", cfg["data"]["max_source_length"], cfg["data"]["max_target_length"])
    logging.info("Train examples:   %s", train_size)
    logging.info("Eval examples:    %s", eval_size)
    logging.info("Epochs:           %s", cfg["training"]["num_train_epochs"])
    logging.info("Batch size:       %s", cfg["training"]["per_device_train_batch_size"])
    logging.info("Grad accum:       %s", cfg["training"]["gradient_accumulation_steps"])
    logging.info("Effective batch:  %s", int(cfg["training"]["per_device_train_batch_size"]) * int(cfg["training"]["gradient_accumulation_steps"]))
    logging.info("Learning rate:    %s", cfg["training"]["learning_rate"])
    logging.info("LoRA r/alpha:     %s / %s", cfg["lora"]["r"], cfg["lora"]["alpha"])
    logging.info("Trainable params: %s", f"{trainable:,}")
    logging.info("Total params:     %s", f"{total:,}")
    logging.info("Trainable ratio:  %.4f%%", 100.0 * trainable / max(1, total))
    logging.info("=" * 50)


def main() -> None:
    load_env_file()
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    if bool(cfg["training"].get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_copy_config(config_path, output_dir)

    token = os.environ.get("HF_TOKEN")
    model_name = cfg["model"]["model_name_or_path"]
    trust_remote_code = bool(cfg["model"].get("trust_remote_code", True))
    dtype = torch_dtype_from_config(cfg["model"].get("torch_dtype", "bfloat16"))

    train_file = Path(cfg["data"]["train_file"])
    eval_file = Path(cfg["data"]["eval_file"])
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if not eval_file.exists():
        raise FileNotFoundError(eval_file)

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

    lora_targets = resolve_lora_targets(model, list(cfg["lora"].get("target_modules", ["auto"])))
    cfg["lora"]["resolved_target_modules"] = lora_targets
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["alpha"]),
        lora_dropout=float(cfg["lora"].get("dropout", 0.05)),
        target_modules=lora_targets,
        bias=str(cfg["lora"].get("bias", "none")),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_dataset = SummarizationDataset(
        train_file,
        tokenizer,
        cfg["data"].get("source_prefix", ""),
        int(cfg["data"]["max_source_length"]),
        int(cfg["data"]["max_target_length"]),
    )
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

    hf = get_hf_settings(cfg)
    callback = AdapterCheckpointCallback(
        output_dir=output_dir,
        tokenizer=tokenizer,
        cfg=cfg,
        config_path=config_path,
        hf=hf,
    )

    log_model_summary(model, cfg, len(train_dataset), len(eval_dataset))
    training_args = make_training_arguments(cfg, output_dir)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [callback],
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    logging.info("Starting LoRA training...")
    train_result = trainer.train()
    logging.info("Training complete: %s", train_result.metrics)

    final_folder = output_dir / "final_adapter"
    callback._save_adapter(model, final_folder, trainer.state, train_result.metrics, tag="final")
    write_json(output_dir / "train_metrics.json", train_result.metrics)

    epochs_dir = output_dir / "epochs"
    if epochs_dir.exists():
        import shutil
        shutil.rmtree(epochs_dir, ignore_errors=True)
        logging.info("Deleted all epoch checkpoints after phase completion to save disk space.")

    if hf["push_final_best"]:
        upload_folder(final_folder, hf, f"{hf['path_in_repo']}/final_adapter", "T5Gemma LoRA final adapter")
        best_folder = output_dir / "best_adapter"
        if best_folder.exists():
            upload_folder(best_folder, hf, f"{hf['path_in_repo']}/best_adapter", "T5Gemma LoRA best adapter")
        upload_folder(output_dir, hf, hf["path_in_repo"], "T5Gemma LoRA training artifacts")


if __name__ == "__main__":
    main()
