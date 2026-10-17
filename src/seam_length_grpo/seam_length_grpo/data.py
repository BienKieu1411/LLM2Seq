"""Per-record dual prompts, word metadata, and encoder-only source budgets."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import unicodedata
from collections import OrderedDict

import torch
import torch.nn.functional as F
from eviseq_afmr.data.collate import SummarizationCollator, _ids
from eviseq_afmr.data.dataset import JsonlSummarizationDataset
from eviseq_afmr.data.schema import CanonicalRecord
from torch.utils.data import DataLoader

from .config import positive
from .reward import count_words


def metadata(row, record, length, *, derive_words):
    value = row.get(length["target_words_field"])
    origin = "provided"
    if value is None and derive_words:
        value, origin = count_words(record.target), "reference_length"
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= length["max_requested_words"]:
        raise ValueError(f"{record.example_id}: target_words must be an integer in [1,{length['max_requested_words']}]")
    domain = str(row.get(length["dataset_field"], length["default_dataset"]))
    budget = row.get(
        length["error_budget_field"], length["dataset_error_budgets"].get(domain, length["default_error_budget_words"])
    )
    positive(budget, "length error budget")
    return {
        "id": record.example_id,
        "prompt": record.source,
        "reference": record.target,
        "target_words": value,
        "word_request_origin": origin,
        "dataset": domain,
        "error_budget": float(budget),
        "language": str(row.get("language") or "unspecified"),
    }


class LengthDataset(JsonlSummarizationDataset):
    def __init__(
        self,
        path,
        data_config,
        length,
        *,
        derive_words=True,
        max_examples=0,
        requested_words=None,
        force_reference_length=False,
    ):
        if force_reference_length and requested_words is not None:
            raise ValueError("Choose external requested_words or reference-length oracle")
        self.force_reference_length = force_reference_length
        self.length, self.derive_words, self.requested_words = (
            length,
            derive_words or force_reference_length,
            requested_words,
        )
        super().__init__(path, data_config, max_examples=max_examples)

    def __getitem__(self, index):
        record = super().__getitem__(index)
        self._handle.seek(self.offsets[index])
        raw = json.loads(self._handle.readline())
        if self.force_reference_length:
            raw.pop(self.length["target_words_field"], None)
        if self.requested_words is not None:
            raw[self.length["target_words_field"]] = self.requested_words
        result = metadata(raw, record, self.length, derive_words=self.derive_words)
        if self.requested_words is not None:
            result["word_request_origin"] = "external_cli"
        return result


def source_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


class Negatives:
    """Deterministic, bounded TRAIN-only pool; exact/near duplicates are excluded."""

    def __init__(self, dataset, config, seed, visible=lambda text: text):
        self.config, self.seed = config, seed
        self.visible = visible
        pools, retained = {}, {}
        for i in range(len(dataset)):
            row = dataset[i]
            text = visible(row["prompt"])
            key = source_key(text)
            domain = (row["dataset"], row.get("language", "unspecified"))
            pool, seen = pools.setdefault(domain, []), retained.setdefault(domain, set())
            rank = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest(), "big")
            if key in seen or (len(pool) >= config["reservoir_size"] and rank >= -pool[0][0]):
                continue
            if len(pool) >= config["reservoir_size"]:
                seen.remove(heapq.heappop(pool)[1])
            heapq.heappush(pool, (-rank, key, row, text))
            seen.add(key)
        self.rows = [
            (row, key, count_words(text), set(text.casefold().split()))
            for pool in pools.values()
            for _, key, row, text in pool
        ]
        if len(self.rows) < 2:
            raise ValueError("Source reward needs at least two distinct training sources")

    def choose(self, row):
        text = self.visible(row["prompt"])
        key, size = source_key(text), max(1, count_words(text))
        words = set(text.casefold().split())
        matches = []
        for other, other_key, other_size, other_words in self.rows:
            if (
                key == other_key
                or row["dataset"] != other["dataset"]
                or row.get("language", "unspecified") != other.get("language", "unspecified")
            ):
                continue
            ratio = max(size, other_size) / max(1, min(size, other_size))
            similarity = len(words & other_words) / max(1, len(words | other_words))
            if ratio <= self.config["max_length_ratio"] and similarity <= self.config["max_jaccard"]:
                rank = hashlib.sha256(f"{self.seed}:{key}:{other_key}".encode()).digest()
                matches.append((rank, other))
        if not matches:
            raise ValueError(f"{row['id']}: no matched unrelated TRAIN source; enlarge source_reward.reservoir_size")
        return min(matches, key=lambda item: item[0])[1]


def source_records(path, data_config, length, source_config, seed, max_examples, update_sources, source_tokenizer=None):
    dataset = LengthDataset(path, data_config, length, max_examples=max_examples)
    if source_config["weight"] and source_tokenizer is None:
        raise ValueError("Source reward requires the encoder tokenizer for visible-source matching")
    limit = length["source_content_tokens"] or data_config["max_source_length"]
    pool = (
        Negatives(dataset, source_config, seed, lambda text: visible_source(source_tokenizer, text, limit))
        if source_config["weight"]
        else None
    )
    size = math.ceil(len(dataset) / update_sources) * update_sources
    for index in range(size):
        row = dict(dataset[index % len(dataset)])
        if pool is not None:
            negative = pool.choose(row)
            row.update(negative_source=negative["prompt"], negative_id=negative["id"])
        yield row


def visible_source(tokenizer, source, limit):
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    if len(ids) <= limit:
        return source
    end = max(end for start, end in offsets[:limit])
    if end <= 0:
        raise ValueError("Encoder tokenizer returned no usable source offsets")
    return source[:end]


class LengthCollator:
    """Reuse the canonical copy aligner with the actual prefix of EACH record."""

    def __init__(self, tokenizers, data, length, *, grounded_copy=False, include_targets=True):
        self.encoder_tokenizer, self.decoder_tokenizer = tokenizers
        self.data, self.length = dict(data), dict(length)
        self.grounded_copy, self.include_targets = grounded_copy, include_targets
        self.source_limit = length["source_content_tokens"] or int(data["max_source_length"])
        self.encoder_limit = self.source_limit + length["encoder_prompt_reserve"]
        self.cache = OrderedDict()

    def rendered(self, words):
        encoder = str(self.data.get("encoder_prefix") or "")
        if self.length["condition_encoder"]:
            encoder += ("\n" if encoder and not encoder.endswith("\n") else "") + self.length[
                "encoder_instruction"
            ].format(target_words=words)
        decoder = (
            self.length["decoder_instruction"].format(target_words=words)
            if self.length["condition_decoder"]
            else str(self.data.get("decoder_prompt") or "Tóm tắt văn bản đã cung cấp.")
        )
        config = {
            **self.data,
            "encoder_prefix": encoder,
            "decoder_prompt": decoder,
            "max_source_length": self.encoder_limit,
        }
        return config

    def _collator(self, words):
        if words not in self.cache:
            collator = SummarizationCollator(
                self.encoder_tokenizer, self.decoder_tokenizer, self.rendered(words), grounded_copy=self.grounded_copy
            )
            prefix = collator.encoder_prefix
            if (
                len(self.encoder_tokenizer(prefix, add_special_tokens=True)["input_ids"])
                > self.length["encoder_prompt_reserve"]
            ):
                raise ValueError("Encoder instruction exceeds encoder_prompt_reserve; increase its explicit budget")
            if len(collator._prompt_ids_for()) > self.length["decoder_prompt_reserve"]:
                raise ValueError("Decoder instruction exceeds decoder_prompt_reserve")
            self.cache[words] = collator
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
        self.cache.move_to_end(words)
        return self.cache[words]

    def visible_source(self, source):
        return visible_source(self.encoder_tokenizer, source, self.source_limit)

    def __call__(self, rows):
        batches, prefixes, instructions, visible = [], [], [], []
        for row in rows:
            collator = self._collator(row["target_words"])
            source = self.visible_source(row["prompt"])
            if (
                len(self.encoder_tokenizer(collator.encoder_prefix + source, add_special_tokens=True)["input_ids"])
                > self.encoder_limit
            ):
                raise ValueError("Encoder input exceeds reserved budget; increase encoder_prompt_reserve")
            record = CanonicalRecord(str(row["id"]), source, row["reference"])
            if self.include_targets:
                if count_words(record.target) != row["target_words"]:
                    raise ValueError(f"{record.example_id}: gold CE requires target_words equal the gold word count")
                if len(_ids(self.decoder_tokenizer, record.target)) + 1 > collator.max_target_length:
                    raise ValueError(f"{record.example_id}: gold would be truncated; increase max_target_length")
            collator.include_targets = self.include_targets
            batches.append(collator([record]))
            prefixes.append(collator.encoder_prefix)
            instructions.append(collator.decoder_prompt)
            visible.append(source)
        width = max(batch["decoder_prompt_ids"].shape[1] for batch in batches)
        result = {}
        for key in batches[0]:
            values = [batch[key] for batch in batches]
            if not isinstance(values[0], torch.Tensor):
                result[key] = [value[0] for value in values]
                continue
            fill = (
                -100
                if key == "labels"
                else int(getattr(self.encoder_tokenizer, "pad_token_id", 0) or 0)
                if key == "input_ids"
                else int(getattr(self.decoder_tokenizer, "pad_token_id", 0) or 0)
                if key in {"decoder_prompt_ids", "decoder_input_ids"}
                else 0
            )
            if key in {
                "decoder_prompt_ids",
                "decoder_prompt_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
            }:
                values = [
                    F.pad(value, (width - batch["decoder_prompt_ids"].shape[1], 0), value=fill)
                    for value, batch in zip(values, batches, strict=True)
                ]
            maximum = max(value.shape[1] for value in values)
            result[key] = torch.cat([F.pad(value, (0, maximum - value.shape[1]), value=fill) for value in values])
        result.update(
            target_words=[row["target_words"] for row in rows],
            error_budgets=[row["error_budget"] for row in rows],
            datasets=[row["dataset"] for row in rows],
            word_request_origins=[row["word_request_origin"] for row in rows],
            encoder_prompts=prefixes,
            decoder_instructions=instructions,
            raw_sources=[row["prompt"] for row in rows],
            sources=visible,
        )
        return result


def make_loader(base, cfg, tokenizers, length, split, *, derive_words=True):
    dataset = LengthDataset(
        base["data"][f"{split}_file"],
        base["data"],
        length,
        derive_words=derive_words,
        max_examples=cfg["validation_max_examples"],
    )
    collator = LengthCollator(
        tokenizers,
        base["data"],
        length,
        grounded_copy=bool(base["decoder"].get("grounded_copy", {}).get("enabled", False)),
        include_targets=False,
    )
    return DataLoader(
        dataset, batch_size=cfg["validation_batch_size"], collate_fn=collator, num_workers=cfg["num_workers"]
    )


def prepare_eval(settings):
    """Inference needs its chosen split, not the historical SFT train files."""
    from pathlib import Path

    from eviseq_afmr.config import load_config, resolve_path

    base = load_config(settings["base_config"], train_only=True)
    for key in ("encoder_name", "decoder_name"):
        name = base["model"][key]
        if name != "__tiny__":
            local = resolve_path(name, base)
            if not local.is_dir():
                local = Path(name).expanduser().resolve()
            if not local.is_dir():
                raise ValueError(f"model.{key} must be an existing local directory")
            base["model"][key] = str(local.resolve())
    for split in ("train", "validation", "test"):
        key = f"{split}_file"
        value = settings.get(key) or base["data"].get(key)
        if value:
            base["data"][key] = str(resolve_path(value, base))
    cfg = settings["training"]
    cfg["max_new_tokens"] = cfg["max_new_tokens"] or int(base["generation"]["max_new_tokens"])
    base["generation"]["max_new_tokens"] = cfg["max_new_tokens"]
    return base
