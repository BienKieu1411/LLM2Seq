"""A short, online gold-prefix KD phase for an already fine-tuned EviSeq run.

The deployable graph remains exactly one encoder -> EvidenceBridge -> decoder.
The Qwen teacher lives outside the student module, is frozen, and is discarded
as soon as this training command exits.  Unlike candidate KD, this phase never
generates pseudo summaries: it transfers a Qwen teacher distribution only at
the gold reference positions.  This keeps the phase suitable as a conservative
post-fine-tuning refinement rather than a replacement for EviSeq training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..configuration import load_config, resolve_data_path
from ..data.dataset import LengthBucketBatchSampler, Seq2SeqCollator, Text2TextDataset, clean_text, decoder_seed_ids
from . import engine as stable
from .checkpoint import (
    assert_evaluation_config_matches_checkpoint,
    load_last_checkpoint,
    save_configured_epoch_checkpoints,
    save_last_checkpoint,
)
from .trainer import _parameter_component, validation_loss

LOGGER = logging.getLogger("eviseq.online_kd")


def _teacher_prompt(source: str, data: Dict[str, Any], tokenizer: Any) -> str:
    """Serialize source exactly once for the teacher; no target is included."""

    instruction = str(data.get("decoder_instruction", "")).strip()
    source_prefix = str(data.get("source_prefix", ""))
    decoder_prefix = str(data.get("decoder_prefix", ""))
    content = f"{instruction}\n\n{source_prefix}{source}\n{decoder_prefix}".strip()
    if bool(data.get("use_decoder_chat_template", True)) and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(data.get("enable_thinking", False)),
                )
            )
        except TypeError:
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return content


class OnlineKDDataset(Text2TextDataset):
    """Attach a teacher prompt to train examples without storing teacher text."""

    def __init__(self, *args: Any, teacher_tokenizer: Any, teacher_max_input_length: int, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_input_length = int(teacher_max_input_length)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        source = clean_text(self.examples[index]["source"], self.clean_metadata)
        prompt = _teacher_prompt(source, self.config, self.teacher_tokenizer)
        encoded = self.teacher_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.teacher_max_input_length,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        if not ids:
            raise ValueError(f"Teacher prompt tokenization is empty at dataset index {index}")
        item["teacher_input_ids"] = torch.tensor(ids, dtype=torch.long)
        item["teacher_attention_mask"] = torch.ones(len(ids), dtype=torch.long)
        return item


class OnlineKDCollator:
    def __init__(self, base: Seq2SeqCollator, teacher_pad_id: int):
        self.base = base
        self.teacher_pad_id = int(teacher_pad_id)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = self.base(features)
        width = max(item["teacher_input_ids"].numel() for item in features)
        rows, masks = [], []
        for item in features:
            ids = item["teacher_input_ids"]
            mask = item["teacher_attention_mask"]
            padding = width - ids.numel()
            # Right padding leaves each row's teacher prompt at position zero;
            # the teacher combines rows individually before a left-padded pass.
            rows.append(torch.cat([ids, torch.full((padding,), self.teacher_pad_id, dtype=torch.long)]))
            masks.append(torch.cat([mask, torch.zeros((padding,), dtype=torch.long)]))
        result["teacher_input_ids"] = torch.stack(rows)
        result["teacher_attention_mask"] = torch.stack(masks)
        return result


def _token_vocab(tokenizer: Any) -> Dict[str, int]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Online KD requires tokenizers exposing a non-empty vocabulary")
    return {str(token): int(index) for token, index in vocabulary.items()}


class GoldPrefixTeacher:
    """Frozen Qwen teacher that returns top-k soft targets on gold positions."""

    def __init__(
        self,
        model_name: str,
        *,
        device: torch.device,
        tokenizer: Any,
        student_tokenizer: Any,
        topk: int,
        temperature: float,
        batch_size: int,
    ):
        if not Path(model_name).is_dir():
            raise FileNotFoundError(
                f"Online KD requires a pre-downloaded local teacher directory; not a Hub ID: {model_name}"
            )
        if _token_vocab(tokenizer) != _token_vocab(student_tokenizer):
            raise ValueError("Online KD requires exactly identical Qwen teacher/student vocabularies and token IDs")
        if tokenizer.eos_token_id != student_tokenizer.eos_token_id:
            raise ValueError("Online KD teacher/student EOS token IDs differ")
        if topk <= 0:
            raise ValueError("online_kd.topk must be positive")
        if temperature <= 0.0:
            raise ValueError("online_kd.temperature must be positive")
        if batch_size <= 0:
            raise ValueError("online_kd.teacher_batch_size must be positive")
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.pad_id = int(tokenizer.pad_token_id)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.batch_size = int(batch_size)

    @staticmethod
    def _gold_rows(labels: torch.Tensor) -> List[List[int]]:
        return [[int(token) for token in row[row.ne(-100)].tolist()] for row in labels]

    @torch.inference_mode()
    def soft_targets(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        labels: torch.Tensor,
        *,
        output_device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Return teacher top-k and log normalizer for each student target position."""

        gold_rows = self._gold_rows(labels)
        prompt_rows: List[List[int]] = []
        for row, mask, gold in zip(prompt_ids, prompt_mask, gold_rows):
            prompt = [int(value) for value in row[mask.bool()].tolist()]
            if not prompt or not gold:
                raise ValueError("Online KD requires non-empty prompt and gold target for every example")
            prompt_rows.append(prompt)
        width_k = min(self.topk, int(self.model.config.vocab_size))
        ids = torch.zeros((len(gold_rows), labels.shape[1], width_k), dtype=torch.long, device=output_device)
        values = torch.zeros((len(gold_rows), labels.shape[1], width_k), dtype=torch.float32, device=output_device)
        normalizers = torch.zeros((len(gold_rows), labels.shape[1]), dtype=torch.float32, device=output_device)
        valid = torch.zeros((len(gold_rows), labels.shape[1]), dtype=torch.bool, device=output_device)
        prefix_widths = labels.eq(-100).sum(dim=1)
        if prefix_widths.numel() and not bool(prefix_widths.eq(prefix_widths[0]).all()):
            raise ValueError("Student labels must use one fixed decoder prompt width for online KD")
        prefix_width = int(prefix_widths[0].item()) if prefix_widths.numel() else 0
        # Full teacher logits are by far the largest allocation in this
        # phase.  Process a physical student batch as teacher microbatches;
        # teacher and student batch sizes intentionally do not have to match.
        for batch_start in range(0, len(gold_rows), self.batch_size):
            batch_end = min(len(gold_rows), batch_start + self.batch_size)
            combined_rows = [prompt_rows[index] + gold_rows[index] for index in range(batch_start, batch_end)]
            width = max(len(row) for row in combined_rows)
            inputs = torch.full((len(combined_rows), width), self.pad_id, dtype=torch.long, device=self.device)
            attention = torch.zeros((len(combined_rows), width), dtype=torch.long, device=self.device)
            left_offsets: List[int] = []
            for local_index, row in enumerate(combined_rows):
                offset = width - len(row)
                left_offsets.append(offset)
                inputs[local_index, offset:] = torch.tensor(row, dtype=torch.long, device=self.device)
                attention[local_index, offset:] = 1
            logits = self.model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
            for local_index, index in enumerate(range(batch_start, batch_end)):
                gold = gold_rows[index]
                start = left_offsets[local_index] + len(prompt_rows[index]) - 1
                # Keep logits BF16 except for this exact target slice.  A
                # full FP32 teacher-vocab tensor is needlessly huge.
                target_logits = logits[local_index, start : start + len(gold)].float()
                if target_logits.shape[0] != len(gold):
                    raise RuntimeError("Teacher forward did not yield a logit row for every gold target token")
                top_values, top_ids = torch.topk(target_logits, k=width_k, dim=-1)
                destination = slice(prefix_width, prefix_width + len(gold))
                ids[index, destination] = top_ids.to(output_device)
                values[index, destination] = top_values.to(output_device)
                normalizers[index, destination] = torch.logsumexp(target_logits / self.temperature, dim=-1).to(
                    output_device
                )
                valid[index, destination] = True
            del logits
        return {
            "teacher_topk_ids": ids,
            "teacher_topk_logits": values,
            "teacher_log_normalizers": normalizers,
            "teacher_mask": valid,
        }


def topk_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_log_normalizers: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Exact teacher top-k + OTHER bucket KL, with no gradient to teacher."""

    if student_logits.ndim != 3 or teacher_topk_ids.shape[:2] != student_logits.shape[:2]:
        raise ValueError("Online KD student and teacher targets must align in [B, T]")
    if teacher_topk_ids.shape != teacher_topk_logits.shape:
        raise ValueError("Online KD teacher top-k IDs/logits have different shapes")
    if teacher_log_normalizers.shape != student_logits.shape[:2] or mask.shape != student_logits.shape[:2]:
        raise ValueError("Online KD masks must align with student [B, T]")
    ids = teacher_topk_ids.to(device=student_logits.device, dtype=torch.long)
    if ids.numel() and (bool(ids.lt(0).any()) or bool(ids.ge(student_logits.shape[-1]).any())):
        raise ValueError("Online KD teacher token ID is outside student vocabulary")
    # Keep no full FP32 [B, target, vocab] log-softmax; it can dominate B200
    # memory even though the KD target is only top-k + OTHER.
    student_normalizer = torch.logsumexp(student_logits / temperature, dim=-1).float()
    student_top = student_logits.gather(-1, ids).float() / temperature - student_normalizer.unsqueeze(-1)
    teacher_top_log = teacher_topk_logits.detach().to(student_logits.device, torch.float32) / temperature
    teacher_top_log = teacher_top_log - teacher_log_normalizers.detach().to(student_logits.device).unsqueeze(-1)
    teacher_top_prob = teacher_top_log.exp()
    # FP16/BF16 top-k values can have a microscopic mass overshoot.
    teacher_top_prob = teacher_top_prob / teacher_top_prob.sum(dim=-1, keepdim=True).clamp_min(1.0).detach()
    teacher_top_log = teacher_top_prob.clamp_min(1.0e-30).log()
    teacher_tail = (1.0 - teacher_top_prob.sum(dim=-1)).clamp(0.0, 1.0)
    student_tail = (1.0 - student_top.exp().sum(dim=-1)).clamp_min(1.0e-12)
    top = (teacher_top_prob * (teacher_top_log - student_top)).sum(dim=-1)
    tail = torch.where(
        teacher_tail > 0.0,
        teacher_tail * (teacher_tail.clamp_min(1.0e-12).log() - student_tail.log()),
        torch.zeros_like(teacher_tail),
    )
    valid = mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return student_logits.float().sum() * 0.0
    return (top + tail)[valid].mean() * (temperature**2)


def _move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to mix KD artifacts with an existing run: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _optimizer(model: torch.nn.Module, training: Dict[str, Any], steps: int, kd: Dict[str, Any]):
    learning_rates = {
        "encoder": float(kd.get("encoder_lr", 5.0e-6)),
        "adapter": float(kd.get("adapter_lr", 1.0e-5)),
        "decoder": float(kd.get("decoder_lr", 5.0e-6)),
        "cross_attention": float(kd.get("cross_attention_lr", 1.0e-5)),
    }
    groups: Dict[tuple[str, bool], List[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = _parameter_component(name)
        no_decay = parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or name.endswith("_gate")
        groups.setdefault((component, no_decay), []).append(parameter)
    optimizer = AdamW(
        [
            {
                "params": parameters,
                "lr": learning_rates[component],
                "initial_lr": learning_rates[component],
                "weight_decay": 0.0 if no_decay else float(training.get("weight_decay", 0.01)),
                "component": component,
            }
            for (component, no_decay), parameters in sorted(groups.items())
        ],
        betas=(float(training.get("adam_beta1", 0.9)), float(training.get("adam_beta2", 0.95))),
        eps=float(training.get("adam_epsilon", 1e-8)),
        fused=bool(training.get("fused_optimizer", True)) and torch.cuda.is_available(),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler


def train(
    config_path: str,
    init_checkpoint: str,
    output_dir: str,
    *,
    teacher_model: str = "",
    teacher_device: str = "",
    epochs: int = 0,
    overwrite_output_dir: bool = False,
) -> Path:
    """Run phase 3 from a strict compatible phase-2 checkpoint."""

    if not str(init_checkpoint).strip():
        raise ValueError("Online KD phase requires --init-checkpoint phase2/last.pt")
    checkpoint_path = Path(init_checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Phase-2 checkpoint does not exist: {checkpoint_path}")
    config = load_config(config_path)
    kd = dict(config.get("online_kd", {}))
    if not bool(kd.get("enabled", False)):
        raise ValueError("Set online_kd.enabled=true in the phase-3 config")
    model_path = str(teacher_model).strip() or str(kd.get("teacher_model", "")).strip()
    if not model_path:
        raise ValueError("Pass --teacher-model /absolute/path/to/local/Qwen3-4B")
    phase_epochs = int(epochs or kd.get("epochs", 1))
    if phase_epochs <= 0:
        raise ValueError("Online KD epochs must be positive")
    weight = float(kd.get("weight", 0.1))
    if weight <= 0.0:
        raise ValueError("online_kd.weight must be positive")
    output = Path(output_dir)
    _prepare_output(output, overwrite_output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "train.log", encoding="utf-8")],
        force=True,
    )
    training = config["training"]
    stable._set_seed(int(training.get("seed", 42)))
    device = stable._device()
    if device.type != "cuda":
        raise RuntimeError("Online Qwen3 KD is supported only on CUDA")
    torch.backends.cuda.matmul.allow_tf32 = bool(training.get("tf32", True))
    torch.set_float32_matmul_precision("high")
    teacher_device_value = str(teacher_device).strip() or str(kd.get("teacher_device", "cuda:1"))
    resolved_teacher_device = torch.device(teacher_device_value)
    if resolved_teacher_device.type != "cuda":
        raise ValueError("online_kd.teacher_device must be a CUDA device")
    teacher_index = 0 if resolved_teacher_device.index is None else int(resolved_teacher_device.index)
    if teacher_index >= torch.cuda.device_count():
        raise ValueError(
            f"Teacher device {resolved_teacher_device} is unavailable; visible CUDA devices="
            f"{torch.cuda.device_count()}. Launch with CUDA_VISIBLE_DEVICES=0,1 for two GPUs."
        )
    if resolved_teacher_device == device:
        LOGGER.warning("Teacher and student share %s; use a second GPU when available", device)

    encoder_tokenizer, decoder_tokenizer = stable._tokenizers(config)
    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    data = config["data"]
    train_dataset = OnlineKDDataset(
        resolve_data_path(data["train_file"], config),
        encoder_tokenizer,
        decoder_tokenizer,
        data,
        max_examples=int(config.get("limits", {}).get("max_train_examples", 0)),
        precompute_evidence=bool(data.get("precompute_evidence", True)),
        teacher_tokenizer=teacher_tokenizer,
        teacher_max_input_length=int(kd.get("teacher_max_input_length", data["max_source_length"])),
    )
    validation_dataset = Text2TextDataset(
        resolve_data_path(data["validation_file"], config),
        encoder_tokenizer,
        decoder_tokenizer,
        data,
        max_examples=int(config.get("limits", {}).get("max_validation_examples", 0)),
        precompute_evidence=False,
    )
    from ..modeling.architecture import EviSeq

    model = EviSeq(config)
    # KD is a continuation, not a second pretraining run.  Require the final
    # checkpoint from the exact phase-2 graph and task/generation contract.
    initialization = load_last_checkpoint(model, checkpoint_path)
    assert_evaluation_config_matches_checkpoint(initialization, config)
    model.to(device)
    model.set_training_stage("full_finetune")
    model.set_evidence_contrastive_scale(1.0)
    if model.use_contrastive:
        raise ValueError("Online KD phase currently requires objectives.use_contrastive=false")
    teacher = GoldPrefixTeacher(
        model_path,
        device=resolved_teacher_device,
        tokenizer=teacher_tokenizer,
        student_tokenizer=decoder_tokenizer,
        topk=int(kd.get("topk", 32)),
        temperature=float(kd.get("temperature", 2.0)),
        batch_size=int(kd.get("teacher_batch_size", 8)),
    )
    base_collator = Seq2SeqCollator(
        encoder_pad_id=int(encoder_tokenizer.pad_token_id),
        decoder_pad_id=int(decoder_tokenizer.pad_token_id),
        max_source_length=int(data["max_source_length"]),
        max_decoder_length=int(data["max_target_length"]) + len(decoder_seed_ids(decoder_tokenizer, data)) - 1,
    )
    collator = OnlineKDCollator(base_collator, int(teacher_tokenizer.pad_token_id))
    batch_size = int(training.get("batch_size", 32))
    loader_kwargs = {
        "collate_fn": collator,
        "num_workers": int(training.get("num_workers", 4)),
        "pin_memory": True,
        "persistent_workers": int(training.get("num_workers", 4)) > 0,
    }
    if bool(training.get("length_bucketing", False)):
        loader = DataLoader(
            train_dataset,
            batch_sampler=LengthBucketBatchSampler(
                train_dataset.source_length_estimates(),
                batch_size,
                seed=int(training.get("seed", 42)),
                bucket_size_multiplier=int(training.get("length_bucket_multiplier", 50)),
            ),
            **loader_kwargs,
        )
    else:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training.get("validation_batch_size", 32)),
        shuffle=False,
        collate_fn=base_collator,
        num_workers=int(training.get("validation_num_workers", 2)),
        pin_memory=True,
        persistent_workers=int(training.get("validation_num_workers", 2)) > 0,
    )
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    total_steps = phase_epochs * math.ceil(len(loader) / accumulation)
    optimizer, scheduler = _optimizer(model, training, total_steps, kd)
    use_fp16_scaler = bool(training.get("fp16", False)) and not bool(training.get("bf16", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    config["experiment"] = dict(config["experiment"])
    config["experiment"]["output_dir"] = str(output)
    config["training"] = dict(training)
    config["training"]["init_checkpoint"] = str(checkpoint_path)
    config["online_kd"] = dict(
        kd, teacher_model=model_path, teacher_device=str(resolved_teacher_device), epochs=phase_epochs
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    LOGGER.info(
        "Online KD phase: init=%s phase2_epoch=%s kd_epochs=%s teacher=%s teacher_device=%s "
        "teacher_batch=%s student_device=%s weight=%.3f",
        checkpoint_path,
        initialization["epoch"],
        phase_epochs,
        model_path,
        resolved_teacher_device,
        teacher.batch_size,
        device,
        weight,
    )
    LOGGER.info(
        "KD data train=%d validation=%d batch=%d accumulation=%d optimizer_steps_per_epoch=%d",
        len(train_dataset),
        len(validation_dataset),
        batch_size,
        accumulation,
        math.ceil(len(loader) / accumulation),
    )
    global_step = 0
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_every = int(training.get("log_every_steps", 10))
    temperature = float(kd.get("temperature", 2.0))
    for epoch in range(1, phase_epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "evi": 0.0}
        count = 0
        iterator = iter(loader)
        optimizer.zero_grad(set_to_none=True)
        for window_start in range(1, len(loader) + 1, accumulation):
            window_size = min(accumulation, len(loader) - window_start + 1)
            for _ in range(window_size):
                batch = next(iterator)
                teacher_inputs = batch.pop("teacher_input_ids")
                teacher_mask = batch.pop("teacher_attention_mask")
                batch = _move(batch, device)
                targets = teacher.soft_targets(teacher_inputs, teacher_mask, batch["labels"], output_device=device)
                with stable._autocast(device, training):
                    outputs = model(**batch, return_full_logits=True)
                    kd_loss = topk_kl_loss(
                        outputs.pop("logits"),
                        targets["teacher_topk_ids"],
                        targets["teacher_topk_logits"],
                        targets["teacher_log_normalizers"],
                        targets["teacher_mask"] & batch["labels"].ne(-100),
                        temperature=temperature,
                    )
                    loss = outputs["loss"] + weight * kd_loss
                scaler.scale(loss / window_size).backward()
                count += 1
                running["loss"] += float(loss.detach().float())
                running["ce"] += float(outputs["loss_ce"].detach().float())
                running["kd"] += float(kd_loss.detach().float())
                running["evi"] += float(outputs["loss_evidence_contrastive"].detach().float())
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % log_every == 0 or window_start + window_size - 1 == len(loader):
                divisor = max(1, count)
                LOGGER.info(
                    "online_kd %s",
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": round(running["loss"] / divisor, 5),
                            "ce": round(running["ce"] / divisor, 5),
                            "kd": round(running["kd"] / divisor, 5),
                            "kd_weight": weight,
                            "evi_cl": round(running["evi"] / divisor, 5),
                            "grad": round(float(grad), 5),
                        },
                        separators=(",", ":"),
                    ),
                )
                running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "evi": 0.0}
                count = 0
        metrics = validation_loss(model, validation_loader, device, training)
        LOGGER.info("online_kd validation %s", json.dumps({"epoch": epoch, **metrics}, separators=(",", ":")))
        save_configured_epoch_checkpoints(model, output, config, epoch, global_step, metrics)
    last = output / "last.pt"
    save_last_checkpoint(model, last, config, phase_epochs, global_step)
    LOGGER.info("Online KD complete: last=%s", last)
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Short online gold-prefix KD phase for a completed EviSeq run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--teacher-device", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    args = parser.parse_args()
    train(
        args.config,
        args.init_checkpoint,
        args.output_dir,
        teacher_model=args.teacher_model,
        teacher_device=args.teacher_device,
        epochs=args.epochs,
        overwrite_output_dir=args.overwrite_output_dir,
    )


if __name__ == "__main__":
    main()
