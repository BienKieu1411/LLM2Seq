import json
import re
from pathlib import Path

import torch
import yaml
from eviseq_afmr.config import load_config
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _TinyTokenizer
from eviseq_afmr.training.checkpoint import save_checkpoint
from seam_length_grpo.config import LENGTH_DEFAULTS, SOURCE_DEFAULTS, load
from seam_length_grpo.data import LengthCollator


class NumeralTokenizer(_TinyTokenizer):
    """Different W values deliberately have different decoder prompt widths."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, truncation=False, max_length=None):
        matches = list(re.finditer(r"\d|[^\s\d]+", text))
        ids = [4 + sum((i + 1) * ord(c) for i, c in enumerate(match.group())) % 124 for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        if add_special_tokens:
            ids, offsets = [1, *ids, 2], [(0, 0), *offsets, (0, 0)]
        if truncation and max_length is not None:
            ids, offsets = ids[:max_length], offsets[:max_length]
        return {"input_ids": ids, **({"offset_mapping": offsets} if return_offsets_mapping else {})}


def fixture():
    torch.manual_seed(73)
    torch.set_num_threads(1)
    cfg = load_config(Path(__file__).resolve().parents[2] / "eviseq_new/configs/afmr_smoke.yaml")
    cfg["model"]["gradient_checkpointing"] = False
    cfg["decoder"]["grounded_copy"]["enabled"] = True
    cfg["generation"].update(max_new_tokens=4, min_new_tokens=0, repetition_penalty=1.0, no_repeat_ngram_size=0)
    cfg["data"].update(
        max_source_length=16,
        max_target_length=24,
        encoder_prefix="",
        decoder_prompt="summarize",
        decoder_prefix="",
        decoder_chat_template=False,
    )
    length = {
        **LENGTH_DEFAULTS,
        "source_content_tokens": 4,
        "encoder_prompt_reserve": 32,
        "decoder_prompt_reserve": 32,
        "encoder_instruction": "Read for {target_words} words. Source:\n",
        "decoder_instruction": "Summarize in {target_words} words.",
    }
    rows = [
        {
            "id": "a",
            "prompt": "alpha beta gamma delta epsilon zeta",
            "reference": "alpha beta",
            "target_words": 2,
            "error_budget": 3.0,
            "dataset": "default",
            "word_request_origin": "reference_length",
        },
        {
            "id": "b",
            "prompt": "jupiter saturn moon mars venus uranus",
            "reference": " ".join(["moon"] * 12),
            "target_words": 12,
            "error_budget": 3.0,
            "dataset": "default",
            "word_request_origin": "reference_length",
        },
    ]
    tokenizer = NumeralTokenizer()
    collator = LengthCollator((tokenizer, tokenizer), cfg["data"], length, grounded_copy=True)
    return cfg, tokenizer, length, rows, collator


def settings_file(tmp_path, *, source_weight=0.05, backend="native", max_steps=2):
    cfg, tokenizer, length, rows, _ = fixture()
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": row["id"], "text": row["prompt"], "summary": row["reference"]}) + "\n" for row in rows
            )
        )
        cfg["data"][f"{split}_file"] = str(path)
    cfg.pop("_meta", None)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "sft.pt"
    save_checkpoint(checkpoint, EviSeqAFMR(cfg), None, cfg, epoch=4, step=100)
    path = tmp_path / "length.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base),
                "sft_checkpoint": str(checkpoint),
                "output_dir": str(tmp_path / "out"),
                "length_control": length,
                "source_reward": {**SOURCE_DEFAULTS, "weight": source_weight},
                "training": {
                    "epochs": 2,
                    "max_steps": max_steps,
                    "group_size": 2,
                    "max_new_tokens": 3,
                    "gradient_accumulation_steps": 2,
                    "validation_batch_size": 2,
                    "rollout_backend": backend,
                },
                "reward": {"length_weight": 0.1},
            }
        )
    )
    return load(path), tokenizer, path
