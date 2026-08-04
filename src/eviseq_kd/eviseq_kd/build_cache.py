"""Generate an offline teacher cache for full-objective EviSeq KD.

Only the teacher tokenizer/model are loaded here.  The resulting JSONL file
contains all information needed by a later offline training process, so cache
loading never needs to construct a student model or contact a model hub.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from .cache import (
    TOPK_ALIGNMENT,
    TeacherRecord,
    source_hash,
    tokenizer_fingerprint,
    tokenizer_vocab_size,
    write_cache,
)
from .paths import resolve_artifact_path, resolve_input_path
from .student.configuration import load_config
from .student.data.dataset import clean_text, read_jsonl

LOGGER = logging.getLogger("eviseq_kd.cache")


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _teacher_prompt(row: Dict[str, Any], data: Dict[str, Any], tokenizer: Any) -> str:
    source = clean_text(row["source"], bool(data.get("clean_wikihow_metadata", False)))
    instruction = str(data.get("decoder_instruction", "")).strip()
    source_prefix = str(data.get("source_prefix", ""))
    decoder_prefix = str(data.get("decoder_prefix", ""))
    content = f"{instruction}\n\n{source_prefix}{source}\n{decoder_prefix}".strip()
    if bool(data.get("use_teacher_chat_template", True)) and getattr(tokenizer, "chat_template", None):
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


def _configured_top_k(config: Mapping[str, Any], explicit: int | None) -> int:
    """Resolve top-k without requiring a config-file schema change."""

    if explicit is not None:
        value = explicit
    else:
        distillation = config.get("training", {}).get("distillation", {})
        if not bool(distillation.get("logit_enabled", False)):
            return 0
        value = None
        for key in ("teacher_top_k", "top_k", "topk", "logit_top_k"):
            if key in distillation:
                value = distillation[key]
                break
        if value is None:
            value = 5
    value = int(value)
    if value < 0:
        raise ValueError("top_k must be non-negative; use zero to disable cached top-k rows")
    return value


def _output_logits(outputs: Any) -> torch.Tensor:
    """Read logits from either a Transformers output object or a test double."""

    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Teacher forward output does not contain a tensor named logits")
    if logits.ndim != 3:
        raise ValueError(f"Teacher logits must have shape [batch, sequence, vocab], got {tuple(logits.shape)}")
    return logits


def _prediction_logits(
    teacher: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_width: int,
    target_width: int,
) -> torch.Tensor:
    """Return only logits that predict the target suffix when supported.

    Qwen3 accepts ``logits_to_keep``. Keeping ``target_width + 1`` gives the
    prompt's final position plus target positions; the last row predicts one
    token beyond the cached target and is dropped. A full-logit fallback keeps
    the builder compatible with simpler test doubles and other causal LMs.
    """

    try:
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=target_width + 1,
        )
    except TypeError:
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    sequence_logits = _output_logits(outputs)
    if sequence_logits.shape[1] == target_width + 1:
        return sequence_logits[:, :-1]
    prediction_logits = sequence_logits[:, prompt_width - 1 : prompt_width - 1 + target_width]
    if prediction_logits.shape[1] != target_width:
        raise ValueError("Teacher forward did not return logits for every target token")
    return prediction_logits


def _generated_suffix(generated: torch.Tensor, prompt_width: int) -> torch.Tensor:
    """Return the generated suffix from the standard decoder-only generate output."""

    if generated.ndim != 2:
        raise ValueError(f"Teacher generate output must have shape [batch, sequence], got {tuple(generated.shape)}")
    if int(generated.shape[1]) < prompt_width:
        raise ValueError(
            "Teacher generate returned fewer tokens than the encoded prompt; "
            "decoder-only generation must return prompt plus generated tokens"
        )
    return generated[:, prompt_width:]


def _normalize_generated_row(
    token_ids: list[int],
    *,
    pad_id: int | None,
    eos_id: int | None,
) -> tuple[list[int], bool]:
    """Keep one EOS even when a tokenizer uses EOS as its padding token."""

    effective: list[int] = []
    saw_eos = False
    for value in token_ids:
        token_id = int(value)
        if eos_id is not None and token_id == eos_id:
            effective.append(token_id)
            saw_eos = True
            break
        if pad_id is not None and token_id == pad_id:
            break
        effective.append(token_id)
    if eos_id is not None and not saw_eos:
        effective.append(int(eos_id))
    return effective, saw_eos


def _pad_token_rows(
    rows: list[list[int]],
    *,
    pad_id: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length decoder targets and return IDs plus attention mask."""

    width = max((len(row) for row in rows), default=0)
    values = torch.full((len(rows), width), int(pad_id), dtype=dtype, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        if row:
            values[row_index, : len(row)] = torch.tensor(row, dtype=dtype, device=device)
            mask[row_index, : len(row)] = 1
    return values, mask


@torch.no_grad()
def build_cache(
    config_path: str,
    output_path: str,
    *,
    teacher_model_name: str = "",
    split: str = "train",
    max_examples: int = 0,
    device_name: str = "auto",
    batch_size: int | None = None,
    max_input_length: int = 4096,
    max_new_tokens: int = 384,
    num_beams: int = 4,
    top_k: int | None = None,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    data = config["data"]
    distillation = config.get("training", {}).get("distillation", {})
    if batch_size is None:
        batch_size = int(distillation.get("teacher_batch_size", 1))
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    teacher_name = teacher_model_name.strip() or str(
        config.get("training", {}).get("distillation", {}).get("teacher_model", "Qwen/Qwen3-4B")
    )
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    top_k = _configured_top_k(config, top_k)
    cache_logits_dtype_name = (
        str(config.get("training", {}).get("distillation", {}).get("cache_logits_dtype", "float16")).strip().lower()
    )
    if cache_logits_dtype_name not in {"float16", "float32"}:
        raise ValueError("cache_logits_dtype must be float16 or float32")
    cache_logits_dtype = torch.float16 if cache_logits_dtype_name == "float16" else torch.float32
    kd_temperature = float(distillation.get("temperature", 2.0))
    if kd_temperature <= 0.0:
        raise ValueError("training.distillation.temperature must be positive")
    data_key = {"train": "train_file", "validation": "validation_file", "test": "test_file"}[split]
    rows = read_jsonl(
        resolve_input_path(str(data[data_key]), config),
        max_examples=max_examples,
        data_config=data,
    )
    device = _device(device_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"
    teacher_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        trust_remote_code=True,
        torch_dtype=teacher_dtype,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    records: List[TeacherRecord] = []
    special_ids = set(int(value) for value in getattr(teacher_tokenizer, "all_special_ids", []))
    eos_id = teacher_tokenizer.eos_token_id
    pad_id = teacher_tokenizer.pad_token_id
    tokenizer_size = tokenizer_vocab_size(teacher_tokenizer)
    tokenizer_digest = tokenizer_fingerprint(teacher_tokenizer)
    model_vocab_size: int | None = None
    alignment_metadata = dict(TOPK_ALIGNMENT)
    for start in range(0, len(rows), max(1, int(batch_size))):
        chunk = rows[start : start + max(1, int(batch_size))]
        prompts = [_teacher_prompt(row, data, teacher_tokenizer) for row in chunk]
        encoded = teacher_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_input_length),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        generated = teacher.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            num_beams=max(1, int(num_beams)),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )
        generated_suffix = _generated_suffix(generated, prompt_width)
        # Normalize each generated row to one explicit target sequence.  EOS
        # is part of the distillation event space; when generation reaches
        # max_new_tokens without emitting EOS, append EOS so the student and
        # cached teacher rows still have identical target lengths.
        effective_suffixes: list[list[int]] = []
        generated_eos_observed: list[bool] = []
        for row_values in generated_suffix.tolist():
            effective, saw_eos = _normalize_generated_row(
                row_values,
                pad_id=pad_id,
                eos_id=eos_id,
            )
            effective_suffixes.append(effective)
            generated_eos_observed.append(saw_eos)
        alignment_width = max((len(row_values) for row_values in effective_suffixes), default=0)
        aligned_suffix = torch.full(
            (len(effective_suffixes), alignment_width),
            int(pad_id if pad_id is not None else 0),
            dtype=generated.dtype,
            device=device,
        )
        suffix_attention = torch.zeros(
            (len(effective_suffixes), alignment_width), dtype=encoded["attention_mask"].dtype, device=device
        )
        for row_index, row_values in enumerate(effective_suffixes):
            if row_values:
                aligned_suffix[row_index, : len(row_values)] = torch.tensor(
                    row_values, dtype=generated.dtype, device=device
                )
                suffix_attention[row_index, : len(row_values)] = 1
        if top_k > 0:
            # Re-run the teacher on the complete prompt+generation sequence.
            # Generation scores can include beam/logits processors; this
            # forward gives the model's exact raw conditional logits for every
            # generated token in the sequence that was ultimately selected.
            full_attention_mask = torch.cat(
                [
                    encoded["attention_mask"],
                    suffix_attention,
                ],
                dim=1,
            )
            prediction_logits = _prediction_logits(
                teacher,
                input_ids=torch.cat([encoded["input_ids"], aligned_suffix], dim=1),
                attention_mask=full_attention_mask,
                prompt_width=prompt_width,
                target_width=alignment_width,
            )
            model_vocab_size = int(prediction_logits.shape[-1])
            if top_k > model_vocab_size:
                raise ValueError(
                    f"Requested top_k={top_k}, but teacher forward exposes only {model_vocab_size} vocabulary logits"
                )
            topk_logits_tensor, topk_ids_tensor = torch.topk(prediction_logits.float(), k=top_k, dim=-1)
            topk_log_normalizers_tensor = torch.logsumexp(
                prediction_logits.float() / kd_temperature,
                dim=-1,
            )
            topk_logits_tensor = topk_logits_tensor.to(dtype=cache_logits_dtype)

            # Also cache the teacher distribution on the gold trajectory.  It
            # stabilizes token-level KD and lets training mix gold-prefix and
            # pseudo-prefix soft targets instead of relying on only one path.
            gold_token_rows: list[list[int]] = []
            gold_limit = max(1, int(data.get("max_target_length", max_new_tokens)))
            for row in chunk:
                target_text = clean_text(row.get("target", ""), bool(data.get("clean_wikihow_metadata", False)))
                encoded_gold = teacher_tokenizer(
                    target_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, gold_limit - int(eos_id is not None)),
                )["input_ids"]
                if isinstance(encoded_gold, torch.Tensor) and encoded_gold.ndim == 2:
                    encoded_gold = encoded_gold[0].tolist()
                gold_ids = [int(value) for value in encoded_gold]
                if eos_id is not None and (not gold_ids or gold_ids[-1] != int(eos_id)):
                    gold_ids.append(int(eos_id))
                gold_token_rows.append(gold_ids)
            gold_suffix, gold_attention = _pad_token_rows(
                gold_token_rows,
                pad_id=int(pad_id if pad_id is not None else 0),
                dtype=generated.dtype,
                device=device,
            )
            gold_prediction_logits = _prediction_logits(
                teacher,
                input_ids=torch.cat([encoded["input_ids"], gold_suffix], dim=1),
                attention_mask=torch.cat([encoded["attention_mask"], gold_attention], dim=1),
                prompt_width=prompt_width,
                target_width=int(gold_suffix.shape[1]),
            )
            if int(gold_prediction_logits.shape[-1]) != model_vocab_size:
                raise ValueError("Teacher vocabulary width changed between pseudo and gold KD forwards")
            gold_topk_logits_tensor, gold_topk_ids_tensor = torch.topk(gold_prediction_logits.float(), k=top_k, dim=-1)
            gold_topk_log_normalizers_tensor = torch.logsumexp(
                gold_prediction_logits.float() / kd_temperature,
                dim=-1,
            )
            gold_topk_logits_tensor = gold_topk_logits_tensor.to(dtype=cache_logits_dtype)
        else:
            topk_logits_tensor = None
            topk_ids_tensor = None
            topk_log_normalizers_tensor = None
            gold_token_rows = [[] for _ in chunk]
            gold_topk_logits_tensor = None
            gold_topk_ids_tensor = None
            gold_topk_log_normalizers_tensor = None
        for row_index, row in enumerate(chunk):
            effective_ids = effective_suffixes[row_index]
            token_ids: List[int] = []
            topk_ids: List[List[int]] = []
            topk_logits: List[List[float]] = []
            topk_positions: List[int] = []
            topk_log_normalizers: List[float] = []
            for generated_position, token_id in enumerate(effective_ids):
                if token_id == eos_id:
                    token_ids.append(token_id)
                    if topk_ids_tensor is not None and topk_logits_tensor is not None:
                        topk_positions.append(generated_position)
                        topk_ids.append(
                            [int(value) for value in topk_ids_tensor[row_index, generated_position].tolist()]
                        )
                        topk_logits.append(
                            [float(value) for value in topk_logits_tensor[row_index, generated_position].tolist()]
                        )
                        assert topk_log_normalizers_tensor is not None
                        topk_log_normalizers.append(float(topk_log_normalizers_tensor[row_index, generated_position]))
                    break
                if token_id == pad_id:
                    break
                if token_id in special_ids:
                    continue
                token_ids.append(token_id)
                if topk_ids_tensor is not None and topk_logits_tensor is not None:
                    topk_positions.append(generated_position)
                    topk_ids.append([int(value) for value in topk_ids_tensor[row_index, generated_position].tolist()])
                    topk_logits.append(
                        [float(value) for value in topk_logits_tensor[row_index, generated_position].tolist()]
                    )
                    assert topk_log_normalizers_tensor is not None
                    topk_log_normalizers.append(float(topk_log_normalizers_tensor[row_index, generated_position]))
            decoded_ids = [token_id for token_id in token_ids if token_id not in special_ids]
            pseudo_target = teacher_tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()
            gold_ids = gold_token_rows[row_index]
            gold_topk_ids = (
                [
                    [int(value) for value in gold_topk_ids_tensor[row_index, position].tolist()]
                    for position in range(len(gold_ids))
                ]
                if gold_topk_ids_tensor is not None
                else []
            )
            gold_topk_logits = (
                [
                    [float(value) for value in gold_topk_logits_tensor[row_index, position].tolist()]
                    for position in range(len(gold_ids))
                ]
                if gold_topk_logits_tensor is not None
                else []
            )
            gold_topk_log_normalizers = (
                [float(gold_topk_log_normalizers_tensor[row_index, position]) for position in range(len(gold_ids))]
                if gold_topk_log_normalizers_tensor is not None
                else []
            )
            source = clean_text(row["source"], bool(data.get("clean_wikihow_metadata", False)))
            raw_id = row.get("id")
            records.append(
                TeacherRecord(
                    example_id=None if raw_id is None or not str(raw_id).strip() else str(raw_id).strip(),
                    source_hash=source_hash(source),
                    pseudo_target=pseudo_target,
                    pseudo_token_ids=token_ids,
                    teacher_topk_ids=topk_ids,
                    teacher_topk_logits=topk_logits,
                    teacher_generated_token_ids=effective_ids,
                    teacher_topk_positions=topk_positions,
                    prompt_token_count=int(encoded["attention_mask"][row_index].sum().item()),
                    prompt_sequence_width=prompt_width,
                    generated_token_count=len(token_ids),
                    generated_eos_observed=generated_eos_observed[row_index],
                    gold_token_ids=gold_ids,
                    gold_topk_ids=gold_topk_ids,
                    gold_topk_logits=gold_topk_logits,
                    teacher_topk_log_normalizers=topk_log_normalizers,
                    gold_topk_log_normalizers=gold_topk_log_normalizers,
                )
            )
        LOGGER.info("cached %d/%d examples", len(records), len(rows))

    metadata = {
        "cache_schema": "eviseq_kd_teacher_cache",
        "teacher_model": teacher_name,
        "teacher_tokenizer_fingerprint": tokenizer_digest,
        "tokenizer_fingerprint": tokenizer_digest,
        "teacher_tokenizer_vocab_size": tokenizer_size,
        "tokenizer_vocab_size": tokenizer_size,
        "teacher_vocab_size": model_vocab_size or tokenizer_size,
        "teacher_model_vocab_size": model_vocab_size or tokenizer_size,
        "split": split,
        "record_count": len(records),
        "max_new_tokens": int(max_new_tokens),
        "num_beams": int(num_beams),
        "has_topk": top_k > 0,
        "has_gold_topk": top_k > 0,
        "top_k": top_k,
        "topk_alignment": alignment_metadata,
        # The KD consumer needs an explicit declaration before it can map
        # teacher token IDs into the student logit space.  This builder does
        # not load the student; the training-side fingerprint check remains
        # responsible for verifying that the declared identity is valid.
        "vocab_alignment": "identity",
        "topk_logit_dtype": cache_logits_dtype_name,
        "kd_temperature": kd_temperature,
        "topk_includes_eos": True,
    }
    output = resolve_artifact_path(output_path)
    write_cache(output, metadata, records)
    LOGGER.info("wrote teacher cache to %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline Qwen sequence and soft-target cache for EviSeq-KD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Teacher generation batch size; defaults to training.distillation.teacher_batch_size",
    )
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build_cache(
        args.config,
        args.output,
        teacher_model_name=args.teacher_model,
        split=args.split,
        max_examples=args.max_examples,
        device_name=args.device,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
