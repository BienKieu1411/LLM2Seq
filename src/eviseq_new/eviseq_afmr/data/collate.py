"""Tokenizer-aware batch construction with separate content and validity masks."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any, Sequence

import torch

from .copy_alignment import align_copy_tokens, pad_copy_alignments
from .schema import CanonicalRecord


def _token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError("Tokenizer output must contain input_ids")
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    if not isinstance(encoded, (list, tuple)):
        raise ValueError("Tokenizer output must be one sequence of integer token IDs")
    try:
        return [index(value) for value in encoded]
    except TypeError as exc:
        raise ValueError(
            "Tokenizer output must be one sequence of integer token IDs, not text or multiple sequences"
        ) from exc


def _ids(tokenizer: Any, text: str) -> list[int]:
    return _token_ids(tokenizer(text, add_special_tokens=False))


class SummarizationCollator:
    def __init__(
        self,
        encoder_tokenizer: Any,
        decoder_tokenizer: Any,
        data_config: dict[str, Any],
        *,
        grounded_copy: bool = False,
        system_prompt_override: str | None = None,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.data = data_config
        self.include_targets = True
        self.grounded_copy = grounded_copy
        decoder_prompt = data_config.get("decoder_prompt", "")
        system_prompt = data_config.get("system_prompt", "")
        self.decoder_prompt = "" if decoder_prompt is None else str(decoder_prompt)
        self.system_prompt = "" if system_prompt is None else str(system_prompt)
        self.system_prompt_override = system_prompt_override
        self.decoder_chat_template = bool(data_config.get("decoder_chat_template", False))
        decoder_prefix = data_config.get("decoder_prefix", "")
        self.decoder_prefix = "" if decoder_prefix is None else str(decoder_prefix)
        self._prompt_cache: dict[str, list[int]] = {}
        # Validate and cache the YAML/default prompt in the main process. This
        # keeps malformed chat-template outputs from surfacing later inside a
        # DataLoader worker, while row-specific prompts are still built lazily.
        self._prompt_ids = (
            self._prompt_ids_for("")
            if self.decoder_prompt.strip() or self.system_prompt.strip() or system_prompt_override is not None
            else []
        )
        self.max_source_length = int(data_config.get("max_source_length", 4096))
        self.max_target_length = int(data_config.get("max_target_length", 512))
        self.encoder_prefix = str(data_config.get("encoder_prefix", ""))
        self.pad_encoder = int(getattr(encoder_tokenizer, "pad_token_id", 0) or 0)
        self.pad_decoder = int(getattr(decoder_tokenizer, "pad_token_id", 0) or 0)

    def _prompt_ids_for(self, record_system_prompt: str) -> list[int]:
        """Build the decoder prefix, including an optional system role.

        The system prompt is deliberately kept separate from the source text:
        EviSeq supplies the document through the encoder/bridge, so putting
        ``{text}`` into this decoder-side prompt would duplicate the input.
        Prompt IDs are cached because most datasets use one shared instruction.
        """

        if self.system_prompt_override is not None:
            system_prompt = str(self.system_prompt_override).strip()
        else:
            system_prompt = str(record_system_prompt).strip() or self.system_prompt.strip()
        cache_key = f"{system_prompt}\x00{self.decoder_prompt}\x00{self.decoder_prefix}\x00{self.decoder_chat_template}"
        cached = self._prompt_cache.get(cache_key)
        if cached is not None:
            return cached

        instruction = self.decoder_prompt.strip()
        if self.decoder_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if instruction:
                messages.append({"role": "user", "content": instruction})
            if not messages:
                raise ValueError("decoder_chat_template requires a non-empty system_prompt or decoder_prompt")
            prompt = _token_ids(
                self.decoder_tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        else:
            parts = [part for part in (system_prompt, instruction) if part]
            prompt = _ids(self.decoder_tokenizer, "\n\n".join(parts))
        prompt += _ids(self.decoder_tokenizer, self.decoder_prefix)
        if not prompt:
            start_token = getattr(self.decoder_tokenizer, "bos_token_id", None)
            if start_token is None:
                start_token = getattr(self.decoder_tokenizer, "eos_token_id", None)
            if start_token is None:
                raise ValueError("An empty decoder prompt requires a BOS or EOS start token")
            prompt = [int(start_token)]
        self._prompt_cache[cache_key] = prompt
        return prompt

    def _encode_source(self, record: CanonicalRecord, return_offsets: bool = False):
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
        return (source, content, offsets) if return_offsets else (source, content)

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
        copy_rows = []
        for record in records:
            source, content, offsets = self._encode_source(record, return_offsets=True)
            if self.grounded_copy:
                copy_rows.append(
                    align_copy_tokens(record.source, len(self.encoder_prefix), offsets, self.decoder_tokenizer)
                )
            encoder_rows.append(source)
            content_rows.append(content)

            prompt = self._prompt_ids_for(record.system_prompt)
            target = (
                _ids(self.decoder_tokenizer, record.target)[: max(1, self.max_target_length - 1)]
                if self.include_targets
                else []
            )
            eos_target = getattr(self.decoder_tokenizer, "eos_token_id", None)
            if eos_target is not None and self.include_targets:
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
        result = {
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
        if self.grounded_copy:
            result.update(pad_copy_alignments(copy_rows))
        return result
