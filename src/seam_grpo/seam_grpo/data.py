"""Source collation and validation shared by the TRL architecture adapter."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from eviseq_afmr.config import load_config, resolve_path
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.dataset import JsonlSummarizationDataset
from eviseq_afmr.evaluation.generate import generate_greedy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def prepare(settings: dict) -> dict:
    base = load_config(settings["base_config"], train_only=True)
    if not Path(settings["sft_checkpoint"]).is_file():
        raise FileNotFoundError(settings["sft_checkpoint"])
    for name in ("encoder_name", "decoder_name"):
        value = str(base["model"][name])
        if value != "__tiny__":
            cwd_path = Path(value).expanduser().resolve()
            config_path = resolve_path(Path(value).expanduser(), base).resolve()
            if cwd_path.is_dir() and config_path.is_dir() and cwd_path != config_path:
                raise ValueError(f"Ambiguous model.{name}: {value}; use an absolute path")
            local_path = config_path if config_path.is_dir() else cwd_path
            if not local_path.is_dir():
                raise ValueError(f"model.{name} must point to an existing local model directory: {value}")
            # Prefer the SFT config's directory. Retain unambiguous legacy CWD
            # paths, and persist absolute names for the relocated rollout config.
            base["model"][name] = str(local_path.resolve())
    for split in ("train", "validation", "test"):
        key = f"{split}_file"
        value = settings.get(key) or base["data"].get(key)
        if value:
            base["data"][key] = str(resolve_path(value, base))
    if base["data"].get("validation_file") == base["data"]["train_file"]:
        raise ValueError("Train and validation paths must differ")
    if base["data"].get("test_file") == base["data"]["train_file"]:
        raise ValueError("The configured test file cannot be used as GRPO training data")
    if base["data"].get("validation_file") and base["data"].get("validation_file") == base["data"].get("test_file"):
        raise ValueError("Validation and test must differ; test cannot select a GRPO checkpoint")
    for split in ("train", "validation"):
        value = base["data"].get(f"{split}_file")
        if value and not Path(value).is_file():
            raise FileNotFoundError(value)
    cfg = settings["training"]
    if cfg["save_best"] and not base["data"].get("validation_file"):
        raise ValueError("save_best requires a validation split; otherwise set save_best=false")
    cfg["max_new_tokens"] = cfg["max_new_tokens"] or int(base["generation"]["max_new_tokens"])
    if int(base["generation"].get("min_new_tokens", 0)) >= cfg["max_new_tokens"]:
        raise ValueError("GRPO max_new_tokens must exceed inherited validation min_new_tokens")
    base["generation"]["max_new_tokens"] = cfg["max_new_tokens"]
    base["experiment"]["output_dir"] = settings["output_dir"]
    for component in ("encoder", "decoder", "bridge", "cross_attention"):
        base["training"][f"full_{component}_lr"] = cfg[f"{component}_lr"]
    base["training"]["weight_decay"] = cfg["weight_decay"]
    return base


def make_loader(base, cfg, tokenizers, split, rank=0, world=1):
    dataset = JsonlSummarizationDataset(
        base["data"][f"{split}_file"],
        base["data"],
        max_examples=cfg["max_train_examples"] if split == "train" else cfg["validation_max_examples"],
    )
    collator = SummarizationCollator(
        *tokenizers,
        base["data"],
        grounded_copy=bool(base["decoder"].get("grounded_copy", {}).get("enabled", False)),
    )
    sampler = DistributedSampler(dataset, world, rank, seed=cfg["seed"], drop_last=False) if world > 1 else None
    # DistributedSampler pads at most world-1 rows; all ranks retain complete
    # local candidate groups and execute the same number of gradient forwards.
    return DataLoader(
        dataset,
        batch_size=cfg["prompt_batch_size"] if split == "train" else cfg["validation_batch_size"],
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg["num_workers"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )


def trajectory(batch, actions, mask, budget):
    result = dict(batch)
    result["action_ids"], result["action_mask"] = actions, mask
    result["decoder_input_ids"] = torch.cat((batch["decoder_prompt_ids"], actions), dim=1)
    result["decoder_attention_mask"] = torch.cat((batch["decoder_prompt_mask"], mask), dim=1)
    result["output_budget"] = torch.full((actions.shape[0],), budget, dtype=torch.float32, device=actions.device)
    return result


@torch.no_grad()
def validate(model, loader, tokenizer, cfg, base, reward, device, autocast, output):
    totals = {key: 0.0 for key in ("reward", "rouge1", "rouge2", "rougeL")}
    count = 0
    words = 0
    gen = base["generation"]
    with Path(output).open("w", encoding="utf-8") as handle:
        for raw in loader:
            with autocast():
                texts, _ = generate_greedy(
                    model,
                    move(raw, device),
                    tokenizer,
                    cfg["max_new_tokens"],
                    min_new_tokens=int(gen.get("min_new_tokens", 0)),
                    repetition_penalty=float(gen.get("repetition_penalty", 1)),
                    no_repeat_ngram_size=int(gen.get("no_repeat_ngram_size", 0)),
                )
            for example_id, text, reference in zip(raw["ids"], texts, raw["references"]):
                for key, value in reward(text, reference).items():
                    totals[key] += value
                handle.write(
                    json.dumps({"id": example_id, "prediction": text, "reference": reference}, ensure_ascii=False)
                    + "\n"
                )
                handle.flush()
                words += len(text.split())
                count += 1
    return {
        **{key: value / max(1, count) for key, value in totals.items()},
        "mean_words": words / max(1, count),
        "count": count,
    }
