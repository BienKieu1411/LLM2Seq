"""Generate an offline teacher cache for sequence-level EviSeq KD."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

from .cache import TeacherRecord, source_hash, write_cache
from .student.configuration import load_config
from .student.data.dataset import clean_text, read_jsonl

LOGGER = logging.getLogger("eviseq_kd.cache")


def _resolve_path(value: str, config: Dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = config.get("_meta", {}).get("config_path")
    candidates = []
    if config_path:
        candidates.append(Path(str(config_path)).resolve().parent / path)
    package_root = Path(__file__).resolve().parents[1]
    if path.parts[:2] == ("src", "eviseq_kd"):
        candidates.append(package_root.joinpath(*path.parts[2:]))
    elif path.parts[:1] == ("eviseq_kd",):
        candidates.append(package_root.joinpath(*path.parts[1:]))
    candidates.extend(
        [
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


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


@torch.no_grad()
def build_cache(
    config_path: str,
    output_path: str,
    *,
    teacher_model_name: str = "",
    split: str = "train",
    max_examples: int = 0,
    device_name: str = "auto",
    batch_size: int = 1,
    max_input_length: int = 4096,
    max_new_tokens: int = 384,
    num_beams: int = 4,
) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = load_config(config_path)
    data = config["data"]
    teacher_name = teacher_model_name.strip() or str(
        config.get("training", {}).get("distillation", {}).get("teacher_model", "Qwen/Qwen3-4B")
    )
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    data_key = {"train": "train_file", "validation": "validation_file", "test": "test_file"}[split]
    rows = read_jsonl(_resolve_path(str(data[data_key]), config), max_examples=max_examples, data_config=data)
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
        for row_index, row in enumerate(chunk):
            raw_ids = generated[row_index, prompt_width:].tolist()
            token_ids: List[int] = []
            for token_id in raw_ids:
                token_id = int(token_id)
                if token_id == eos_id or token_id == pad_id:
                    break
                if token_id in special_ids:
                    continue
                token_ids.append(token_id)
            pseudo_target = teacher_tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            source = clean_text(row["source"], bool(data.get("clean_wikihow_metadata", False)))
            raw_id = row.get("id")
            records.append(
                TeacherRecord(
                    example_id=None if raw_id is None or not str(raw_id).strip() else str(raw_id).strip(),
                    source_hash=source_hash(source),
                    pseudo_target=pseudo_target,
                    pseudo_token_ids=token_ids,
                )
            )
        LOGGER.info("cached %d/%d examples", len(records), len(rows))

    metadata = {
        "teacher_model": teacher_name,
        "split": split,
        "record_count": len(records),
        "max_new_tokens": int(max_new_tokens),
        "num_beams": int(num_beams),
    }
    output = _resolve_path(output_path, config)
    write_cache(output, metadata, records)
    LOGGER.info("wrote teacher cache to %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline Qwen teacher cache for EviSeq-KD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-model", default="")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=4)
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
    )


if __name__ == "__main__":
    main()
