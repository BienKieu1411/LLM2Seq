"""Tokenizer-aware batch construction with separate content and validity masks."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .schema import CanonicalRecord


def _ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [int(value) for value in values]


class SummarizationCollator:
    def __init__(
        self,
        encoder_tokenizer: Any,
        decoder_tokenizer: Any,
        data_config: dict[str, Any],
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.data = data_config
        self.max_source_length = int(data_config.get("max_source_length", 4096))
        self.max_target_length = int(data_config.get("max_target_length", 512))
        self.encoder_prefix = str(data_config.get("encoder_prefix", ""))
        self.decoder_prompt = str(data_config.get("decoder_prompt", ""))
        self.pad_encoder = int(getattr(encoder_tokenizer, "pad_token_id", 0) or 0)
        self.pad_decoder = int(getattr(decoder_tokenizer, "pad_token_id", 0) or 0)

    def _encode_source(self, record: CanonicalRecord) -> tuple[list[int], list[bool]]:
        try:
            encoded = self.encoder_tokenizer(
                self.encoder_prefix + record.source,
                add_special_tokens=True,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_source_length,
            )
            offsets = encoded["offset_mapping"]
        except (TypeError, KeyError, NotImplementedError) as exc:
            raise ValueError("AFMR requires a fast encoder tokenizer with offset mapping") from exc
        source = list(encoded["input_ids"])
        content = []
        for start, end in offsets:
            article = end > max(start, len(self.encoder_prefix))
            content.append(article)
        if not any(content):
            raise ValueError(f"No source content remains after truncation for {record.example_id}")
        return source, content

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], value: int) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(map(len, rows), default=1)
        ids = torch.full((len(rows), width), value, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for row, values in enumerate(rows):
            size = min(len(values), width)
            ids[row, :size] = torch.tensor(values[:size], dtype=torch.long)
            mask[row, :size] = True
        return ids, mask

    def __call__(self, records: Sequence[CanonicalRecord]) -> dict[str, Any]:
        encoder_rows: list[list[int]] = []
        content_rows: list[list[bool]] = []
        prompt_rows: list[list[int]] = []
        decoder_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for record in records:
            source, content = self._encode_source(record)
            encoder_rows.append(source)
            content_rows.append(content)

            prompt = _ids(self.decoder_tokenizer, self.decoder_prompt)
            if not prompt:
                start_token = getattr(self.decoder_tokenizer, "bos_token_id", None)
                if start_token is None:
                    start_token = getattr(self.decoder_tokenizer, "eos_token_id", None)
                if start_token is None:
                    raise ValueError("An empty decoder prompt requires a BOS or EOS start token")
                prompt = [int(start_token)]
            target = _ids(self.decoder_tokenizer, record.target)[: max(1, self.max_target_length - 1)]
            eos_target = getattr(self.decoder_tokenizer, "eos_token_id", None)
            if eos_target is not None:
                target = target + [int(eos_target)]
            prompt_rows.append(prompt)
            decoder_rows.append(prompt + target if target else prompt)
            labels = [-100] * len(prompt) + target
            label_rows.append(labels[: len(prompt) + self.max_target_length])

        input_ids, attention_mask = self._pad(encoder_rows, self.pad_encoder)
        source_content_mask, _ = self._pad([[int(x) for x in row] for row in content_rows], 0)
        prompt_ids, prompt_mask = self._pad(prompt_rows, self.pad_decoder)
        decoder_input_ids, decoder_attention_mask = self._pad(decoder_rows, self.pad_decoder)
        labels, _ = self._pad(label_rows, -100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "source_content_mask": source_content_mask.bool(),
            "decoder_prompt_ids": prompt_ids,
            "decoder_prompt_mask": prompt_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "ids": [record.example_id for record in records],
            "sources": [record.source for record in records],
            "references": [record.target for record in records],
        }
