"""KD dataset/collator layered over the bundled student data implementation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import torch

from .cache import TeacherCache, TeacherRecord, source_hash
from .student.data.dataset import Text2TextDataset, clean_text, decoder_seed_ids


class KDText2TextDataset(Text2TextDataset):
    """Add teacher pseudo-targets without modifying the bundled student dataset."""

    def __init__(
        self,
        *args: Any,
        teacher_cache: TeacherCache,
        require_teacher_cache: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_cache = teacher_cache
        self.require_teacher_cache = bool(require_teacher_cache)

    def _target_features(self, target: str, token_ids: List[int] | None = None) -> Dict[str, torch.Tensor]:
        eos_id = self.decoder_tokenizer.eos_token_id
        target_budget = self.max_target_length - int(eos_id is not None)
        if token_ids is None:
            target_ids = list(
                self.decoder_tokenizer(
                    target,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, target_budget),
                )["input_ids"]
            )
        else:
            target_ids = [int(value) for value in token_ids[: max(1, target_budget)]]
        if eos_id is not None:
            target_ids.append(int(eos_id))
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        seed = torch.tensor(decoder_seed_ids(self.decoder_tokenizer, self.config), dtype=torch.long)
        decoder_input = torch.cat([seed, target_tensor[:-1]])
        labels = torch.cat(
            [
                torch.full((seed.numel() - 1,), -100, dtype=torch.long),
                target_tensor,
            ]
        )
        if decoder_input.numel() != labels.numel():
            raise RuntimeError("Shifted KD decoder inputs and labels must have equal length")
        return {
            "decoder_input_ids": decoder_input,
            "decoder_attention_mask": torch.ones_like(decoder_input),
            "labels": labels,
        }

    def _topk_features(self, record: TeacherRecord, target_count: int) -> Dict[str, torch.Tensor]:
        # The simple runner deliberately uses sequence KD only. Teacher text
        # is always retokenized with the student tokenizer; cached top-k rows,
        # when present in an older cache, are ignored.
        return {}

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        row = self.examples[index]
        source = clean_text(row["source"], self.clean_metadata)
        example_id = row.get("id")
        has_example_id = example_id is not None and str(example_id).strip() != ""
        try:
            record = self.teacher_cache.get(
                example_id,
                index,
                source_hash=source_hash(source),
                require_source_match=True,
                allow_index_fallback=not has_example_id,
            )
        except (IndexError, KeyError, ValueError):
            if self.require_teacher_cache:
                raise
            return item
        pseudo_target = clean_text(record.pseudo_target, self.clean_metadata)
        if not pseudo_target and self.require_teacher_cache:
            raise ValueError(f"Teacher cache has an empty pseudo target for sample index {index}")
        if not pseudo_target:
            return item
        pseudo = self._target_features(pseudo_target)
        item.update({f"pseudo_{key}": value for key, value in pseudo.items()})
        item.update(self._topk_features(record, max(0, int(pseudo["labels"].ne(-100).sum().item())) - 1))
        return item


class KDCollator:
    """Pad gold and pseudo branches independently; top-k rows follow the pseudo branch."""

    def __init__(self, base_collator: Any, decoder_pad_id: int, max_decoder_length: int):
        self.base_collator = base_collator
        self.decoder_pad_id = int(decoder_pad_id)
        self.max_decoder_length = int(max_decoder_length)

    @staticmethod
    def _pad(values: Iterable[torch.Tensor], length: int, fill: float) -> torch.Tensor:
        rows = []
        for value in values:
            value = value[:length]
            padding = torch.full((length - value.numel(),), fill, dtype=value.dtype)
            rows.append(torch.cat([value, padding]))
        return torch.stack(rows)

    @staticmethod
    def _pad_matrix(values: Iterable[torch.Tensor], length: int, width: int, fill: float) -> torch.Tensor:
        rows = []
        for value in values:
            value = value[:length, :width]
            if value.shape[1] != width:
                raise ValueError("Teacher top-k matrices must have a constant width")
            padding = torch.full((length - value.shape[0], width), fill, dtype=value.dtype)
            rows.append(torch.cat([value, padding], dim=0))
        return torch.stack(rows)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        base_features = [
            {
                key: value
                for key, value in item.items()
                if not key.startswith("pseudo_") and not key.startswith("teacher_")
            }
            for item in features
        ]
        batch = self.base_collator(base_features)
        if any("pseudo_decoder_input_ids" in item for item in features):
            if not all("pseudo_decoder_input_ids" in item for item in features):
                raise ValueError("Every item in a KD batch must have a pseudo target")
            pseudo_length = min(
                self.max_decoder_length,
                max(item["pseudo_decoder_input_ids"].numel() for item in features),
            )
            batch.update(
                {
                    "pseudo_decoder_input_ids": self._pad(
                        (item["pseudo_decoder_input_ids"] for item in features), pseudo_length, self.decoder_pad_id
                    ),
                    "pseudo_decoder_attention_mask": self._pad(
                        (item["pseudo_decoder_attention_mask"] for item in features), pseudo_length, 0
                    ),
                    "pseudo_labels": self._pad((item["pseudo_labels"] for item in features), pseudo_length, -100),
                }
            )
        if any("teacher_topk_ids" in item for item in features):
            if not all("teacher_topk_ids" in item for item in features):
                raise ValueError("Every item in a top-k KD batch must have teacher top-k tensors")
            pseudo_length = int(batch["pseudo_decoder_input_ids"].shape[1])
            width = int(features[0]["teacher_topk_ids"].shape[1])
            if any(int(item["teacher_topk_ids"].shape[1]) != width for item in features):
                raise ValueError("Teacher top-k width must be constant in a batch")
            batch.update(
                {
                    "teacher_topk_ids": self._pad_matrix(
                        (item["teacher_topk_ids"] for item in features), pseudo_length, width, 0
                    ),
                    "teacher_topk_logits": self._pad_matrix(
                        (item["teacher_topk_logits"] for item in features), pseudo_length, width, 0.0
                    ),
                    "teacher_kd_mask": self._pad(
                        (item["teacher_kd_mask"] for item in features), pseudo_length, 0
                    ).bool(),
                }
            )
        return batch
