"""Native validation with explicit requested words and per-dataset constraints."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from eviseq_afmr.evaluation.generate import generate_greedy
from seam_grpo.context import validate_context
from seam_grpo.data import move


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    keys = ("quality", "reward", "rouge1", "rouge2", "rougeL", "abs_len_loss", "length_cost", "length_deviation")

    def mean(values):
        return {key: sum(row[key] for row in values) / len(values) for key in keys}

    datasets = {name: {"count": len(values), **mean(values)} for name, values in groups.items()}
    return {
        "count": len(rows),
        **(mean(rows) if rows else {}),
        "datasets": datasets,
        "all_length_constraints_met": bool(rows) and all(row["length_cost"] <= 1 for row in datasets.values()),
        "macro_quality": sum(row["quality"] for row in datasets.values()) / max(1, len(datasets)),
        "max_length_violation": max([max(0, row["length_cost"] - 1) for row in datasets.values()] or [0]),
    }


@torch.no_grad()
def predict(model, loader, tokenizer, cfg, base, reward, device, autocast, output, *, overwrite=False):
    rows = []
    generation = base["generation"]
    with Path(output).open("w" if overwrite else "x", encoding="utf-8") as handle:
        for raw in loader:
            batch = move(raw, device)
            validate_context(model.decoder, batch["decoder_prompt_ids"].shape[1], cfg["max_new_tokens"])
            with autocast():
                texts, _ = generate_greedy(
                    model,
                    batch,
                    tokenizer,
                    cfg["max_new_tokens"],
                    min_new_tokens=int(generation.get("min_new_tokens", 0)),
                    repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
                    no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
                    do_sample=bool(generation.get("do_sample", False)),
                    temperature=float(generation.get("temperature", 1.0)),
                    top_k=int(generation.get("top_k", 0)),
                    top_p=float(generation.get("top_p", 1.0)),
                )
            for i, text in enumerate(texts):
                scores = reward(text, raw["references"][i], raw["target_words"][i], raw["error_budgets"][i])
                row = {
                    "id": raw["ids"][i],
                    "prediction": text,
                    "reference": raw["references"][i],
                    "dataset": raw["datasets"][i],
                    "word_request_origin": raw["word_request_origins"][i],
                    "error_budget": raw["error_budgets"][i],
                    **scores,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    return summarize(rows)
