"""Tokenizer-aware batch construction with separate content and validity masks."""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any, Sequence

import torch

from .copy_alignment import align_copy_tokens, pad_copy_alignments
from .evidence import EVIDENCE_TENSOR_KEYS, empty_evidence_tensors, ids_sha256
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
        evidence_mode: str | None = None,
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.data = data_config
        self.include_targets = True
        self.grounded_copy = grounded_copy
        if evidence_mode not in {None, "copy", "semantic", "both"}:
            raise ValueError("evidence_mode must be copy, semantic, both, or None")
        if evidence_mode is not None and not grounded_copy:
            raise ValueError("Evidence contrastive training requires grounded copy")
        self.evidence_mode = evidence_mode
        instruction = str(data_config.get("decoder_prompt", ""))
        if data_config.get("decoder_chat_template", False):
            if not instruction.strip():
                raise ValueError("decoder_chat_template requires a non-empty decoder_prompt")
            self._prompt_ids = _token_ids(
                decoder_tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        else:
            self._prompt_ids = _ids(decoder_tokenizer, instruction)
        self._prompt_ids += _ids(decoder_tokenizer, str(data_config.get("decoder_prefix", "")))
        self.max_source_length = int(data_config.get("max_source_length", 4096))
        self.max_target_length = int(data_config.get("max_target_length", 512))
        self.encoder_prefix = str(data_config.get("encoder_prefix", ""))
        self.decoder_prompt = str(data_config.get("decoder_prompt", ""))
        self.pad_encoder = int(getattr(encoder_tokenizer, "pad_token_id", 0) or 0)
        self.pad_decoder = int(getattr(decoder_tokenizer, "pad_token_id", 0) or 0)

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
        target_rows: list[list[int]] = []
        for record in records:
            source, content, offsets = self._encode_source(record, return_offsets=True)
            if self.grounded_copy:
                copy_rows.append(
                    align_copy_tokens(record.source, len(self.encoder_prefix), offsets, self.decoder_tokenizer)
                )
            encoder_rows.append(source)
            content_rows.append(content)

            prompt = self._prompt_ids
            if not prompt:
                start_token = getattr(self.decoder_tokenizer, "bos_token_id", None)
                if start_token is None:
                    start_token = getattr(self.decoder_tokenizer, "eos_token_id", None)
                if start_token is None:
                    raise ValueError("An empty decoder prompt requires a BOS or EOS start token")
                prompt = [int(start_token)]
            target = (
                _ids(self.decoder_tokenizer, record.target)[: max(1, self.max_target_length - 1)]
                if self.include_targets
                else []
            )
            target_rows.append(target)
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
        if self.evidence_mode is not None and self.include_targets:
            result.update(
                self._evidence_tensors(
                    records,
                    target_rows,
                    prompt_rows,
                    labels,
                    input_ids,
                    attention_mask,
                    source_content_mask.bool(),
                    copy_rows,
                )
            )
        return result

    @staticmethod
    def _span_positions(spans, width: int) -> set[int]:
        positions: set[int] = set()
        for start, end in spans:
            if not 0 <= start < end <= width:
                raise ValueError("Evidence semantic span lies outside the encoded source")
            positions.update(range(start, end))
        return positions

    def _evidence_tensors(
        self,
        records: Sequence[CanonicalRecord],
        target_rows: Sequence[Sequence[int]],
        prompt_rows: Sequence[Sequence[int]],
        labels: torch.Tensor,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        copy_rows: Sequence[dict[str, list]],
    ) -> dict[str, torch.Tensor]:
        result = empty_evidence_tensors()
        if not any(record.evidence is not None for record in records):
            return result
        if not all(record.evidence is not None for record in records):
            raise ValueError("Evidence recipe requires cache annotations for every training record")
        unit_batches: list[int] = []
        confidences: list[float] = []
        target_units: list[int] = []
        target_hidden: list[int] = []
        target_weights: list[float] = []
        copy_owner: list[int] = []
        copy_position: list[int] = []
        copy_positive: list[bool] = []
        semantic_owner: list[int] = []
        semantic_position: list[int] = []
        semantic_positive: list[bool] = []
        for batch_index, (record, target_ids, prompt, copy) in enumerate(
            zip(records, target_rows, prompt_rows, copy_rows)
        ):
            annotation = record.evidence
            assert annotation is not None
            source_ids = source_input_ids[batch_index, source_attention_mask[batch_index].bool()].tolist()
            if annotation.source_input_ids_sha256 != ids_sha256(source_ids):
                raise ValueError("Evidence source tokenizer hash mismatch; rebuild the cache")
            if annotation.target_input_ids_sha256 != ids_sha256(target_ids):
                raise ValueError("Evidence target tokenizer hash mismatch; rebuild the cache")
            copy_ids = copy["copy_token_ids"]
            for unit in annotation.units:
                semantic_pos = self._span_positions(unit.semantic_positive_spans, source_input_ids.shape[1])
                semantic_neg = self._span_positions(unit.semantic_negative_spans, source_input_ids.shape[1])
                semantic_pos = {
                    position for position in semantic_pos if bool(source_content_mask[batch_index, position])
                }
                semantic_neg = {
                    position for position in semantic_neg if bool(source_content_mask[batch_index, position])
                }
                if not semantic_pos or not semantic_neg or semantic_pos & semantic_neg:
                    continue
                positive_map, negative_map = unit.copy_positive, unit.copy_negative
                eligible: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
                for target_position in unit.target_positions:
                    if not 0 <= target_position < len(target_ids):
                        continue
                    positives = tuple(
                        position for position in positive_map.get(target_position, ()) if position < len(copy_ids)
                    )
                    negatives = tuple(
                        position for position in negative_map.get(target_position, ()) if position < len(copy_ids)
                    )
                    copy_ok = bool(positives and negatives and not set(positives) & set(negatives))
                    if copy_ok and any(
                        copy_ids[position] != target_ids[target_position] for position in (*positives, *negatives)
                    ):
                        raise ValueError("Evidence copy candidate does not match its gold target token")
                    if self.evidence_mode == "copy" and not copy_ok:
                        continue
                    if self.evidence_mode == "both" and not copy_ok:
                        continue
                    label_position = len(prompt) + target_position
                    hidden_position = label_position - 1
                    if (
                        hidden_position < 0
                        or label_position >= labels.shape[1]
                        or int(labels[batch_index, label_position]) != target_ids[target_position]
                    ):
                        raise ValueError("Evidence target position no longer matches shifted decoder labels")
                    eligible.append((target_position, positives, negatives))
                if not eligible:
                    continue
                unit_index = len(unit_batches)
                unit_batches.append(batch_index)
                confidences.append(unit.confidence)
                for target_position, positives, negatives in eligible:
                    owner = len(target_units)
                    target_units.append(unit_index)
                    target_hidden.append(len(prompt) + target_position - 1)
                    target_weights.append(1.0 / len(eligible))
                    if self.evidence_mode in {"copy", "both"}:
                        for position in positives:
                            copy_owner.append(owner)
                            copy_position.append(position)
                            copy_positive.append(True)
                        for position in negatives:
                            copy_owner.append(owner)
                            copy_position.append(position)
                            copy_positive.append(False)
                    if self.evidence_mode in {"semantic", "both"}:
                        for position in sorted(semantic_pos):
                            semantic_owner.append(owner)
                            semantic_position.append(position)
                            semantic_positive.append(True)
                        for position in sorted(semantic_neg):
                            semantic_owner.append(owner)
                            semantic_position.append(position)
                            semantic_positive.append(False)
        if not unit_batches:
            return result
        result.update(
            evidence_unit_batch_index=torch.tensor(unit_batches, dtype=torch.long),
            evidence_unit_confidence=torch.tensor(confidences, dtype=torch.float32),
            evidence_unit_valid=torch.ones(len(unit_batches), dtype=torch.bool),
            evidence_target_unit=torch.tensor(target_units, dtype=torch.long),
            evidence_target_hidden_pos=torch.tensor(target_hidden, dtype=torch.long),
            evidence_target_weight=torch.tensor(target_weights, dtype=torch.float32),
            evidence_copy_owner=torch.tensor(copy_owner, dtype=torch.long),
            evidence_copy_source_position=torch.tensor(copy_position, dtype=torch.long),
            evidence_copy_positive=torch.tensor(copy_positive, dtype=torch.bool),
            evidence_semantic_owner=torch.tensor(semantic_owner, dtype=torch.long),
            evidence_semantic_source_position=torch.tensor(semantic_position, dtype=torch.long),
            evidence_semantic_positive=torch.tensor(semantic_positive, dtype=torch.bool),
        )
        return result
