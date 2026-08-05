"""KD dataset/collator layered over the bundled student data implementation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

import torch

from .build_cache import _teacher_prompt
from .cache import TeacherCache, TeacherRecord, source_hash, tokenizer_fingerprint
from .student.data.dataset import Text2TextDataset, clean_text, decoder_seed_ids


class OnlineKDText2TextDataset(Text2TextDataset):
    """Attach a teacher prompt, but never a pre-generated teacher target."""

    def __init__(self, *args: Any, teacher_tokenizer: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.teacher_tokenizer = teacher_tokenizer
        self.teacher_max_input_length = int(
            self.config.get("teacher_max_input_length", self.config.get("max_source_length", 3072))
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        row = self.examples[index]
        prompt = _teacher_prompt(row, self.config, self.teacher_tokenizer)
        encoded = self.teacher_tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.teacher_max_input_length,
        )
        teacher_ids = [int(token_id) for token_id in encoded["input_ids"]]
        if not teacher_ids:
            raise ValueError(f"Teacher prompt tokenization is empty at training index {index}")
        item["teacher_input_ids"] = torch.tensor(teacher_ids, dtype=torch.long)
        item["teacher_attention_mask"] = torch.ones(len(teacher_ids), dtype=torch.long)
        return item


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
        self.student_vocab_size = self._tokenizer_vocab_size(self.decoder_tokenizer)
        self._teacher_vocab_mapping: torch.Tensor | None = None

    @staticmethod
    def _tokenizer_vocab_size(tokenizer: Any) -> int:
        try:
            size = int(len(tokenizer))
        except (TypeError, AttributeError):
            size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if size <= 0:
            raise ValueError("Student decoder tokenizer must expose a positive vocabulary size")
        return size

    def _build_teacher_vocab_mapping(self) -> torch.Tensor:
        """Resolve an explicit cache-declared teacher→student ID mapping.

        Top-k logits contain token IDs, not token strings.  Equal-looking
        vocab sizes are not sufficient evidence that those IDs mean the same
        thing, so an identity mapping must be declared by cache metadata.
        """

        metadata = self.teacher_cache.metadata
        expected_fingerprint = str(
            metadata.get("teacher_tokenizer_fingerprint", metadata.get("tokenizer_fingerprint", ""))
        ).strip()
        if expected_fingerprint:
            actual_fingerprint = tokenizer_fingerprint(self.decoder_tokenizer)
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    "Teacher/student decoder tokenizer fingerprint mismatch; "
                    "rebuild the teacher cache with the exact decoder tokenizer"
                )
        raw_teacher_size = next(
            (
                metadata.get(key)
                for key in (
                    "teacher_model_vocab_size",
                    "teacher_vocab_size",
                    "teacher_tokenizer_vocab_size",
                    "teacher_vocab",
                )
                if metadata.get(key) is not None
            ),
            None,
        )
        if raw_teacher_size is None:
            raise ValueError(
                "Teacher top-k KD requires cache metadata.teacher_vocab_size; rebuild the cache with vocabulary metadata"
            )
        try:
            teacher_size = int(raw_teacher_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("Teacher cache metadata.teacher_vocab_size must be a positive integer") from exc
        if teacher_size <= 0:
            raise ValueError("Teacher cache metadata.teacher_vocab_size must be positive")

        raw_student_size = next(
            (
                metadata.get(key)
                for key in ("student_vocab_size", "student_tokenizer_vocab_size", "student_vocab")
                if metadata.get(key) is not None
            ),
            None,
        )
        if raw_student_size is not None and int(raw_student_size) != self.student_vocab_size:
            raise ValueError(
                "Teacher cache/student tokenizer vocabulary size mismatch: "
                f"cache={int(raw_student_size)}, student={self.student_vocab_size}"
            )

        alignment = metadata.get("vocab_alignment")
        mapping = metadata.get("teacher_to_student_vocab")
        if mapping is None:
            mapping = metadata.get("teacher_to_student_ids")
        if mapping is None:
            mapping = metadata.get("vocab_mapping")

        if isinstance(alignment, Mapping):
            mapping = alignment.get("teacher_to_student") or alignment.get("mapping") or mapping
            alignment_type = str(alignment.get("type", "")).strip().lower()
        else:
            alignment_type = str(alignment or "").strip().lower()

        identity_types = {"identity", "shared", "shared_vocab", "same", "exact"}
        if mapping is None and alignment_type not in identity_types:
            raise ValueError(
                "Teacher top-k KD requires cache metadata.vocab_alignment='identity'/'shared' "
                "or metadata.teacher_to_student_vocab"
            )
        if mapping is None:
            if teacher_size != self.student_vocab_size:
                raise ValueError(
                    "Identity teacher/student vocabulary alignment was declared, but vocabulary sizes differ: "
                    f"teacher={teacher_size}, student={self.student_vocab_size}"
                )
            return torch.arange(teacher_size, dtype=torch.long)

        if isinstance(mapping, Mapping):
            values = []
            for teacher_id in range(teacher_size):
                if teacher_id not in mapping and str(teacher_id) not in mapping:
                    raise ValueError(f"Teacher top-k vocabulary mapping is missing teacher token ID {teacher_id}")
                values.append(mapping.get(teacher_id, mapping.get(str(teacher_id))))
        elif isinstance(mapping, (list, tuple)):
            if len(mapping) != teacher_size:
                raise ValueError("Teacher top-k vocabulary mapping length must equal metadata.teacher_vocab_size")
            values = list(mapping)
        else:
            raise ValueError("Teacher top-k vocabulary mapping must be a list or object")
        try:
            result = torch.tensor([int(value) for value in values], dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise ValueError("Teacher top-k vocabulary mapping must contain integer student IDs") from exc
        if bool(result.lt(0).any()) or bool(result.ge(self.student_vocab_size).any()):
            raise ValueError("Teacher top-k vocabulary mapping contains an ID outside the student vocabulary")
        return result

    def _map_teacher_ids(self, teacher_ids: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_teacher_vocab_mapping", None) is None:
            self._teacher_vocab_mapping = self._build_teacher_vocab_mapping()
        mapping = self._teacher_vocab_mapping
        if bool(teacher_ids.lt(0).any()) or bool(teacher_ids.ge(mapping.numel()).any()):
            raise ValueError("Teacher top-k IDs contain an ID outside the declared teacher vocabulary")
        return mapping[teacher_ids]

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
        if eos_id is not None and (not target_ids or target_ids[-1] != int(eos_id)):
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

    def _topk_features(
        self,
        record: TeacherRecord,
        target_count: int,
        prompt_prefix_count: int,
        student_target_ids: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        has_ids = bool(record.teacher_topk_ids)
        has_logits = bool(record.teacher_topk_logits)
        if not has_ids and not has_logits:
            return {}
        if not has_ids or not has_logits:
            raise ValueError(
                "Teacher cache record must contain both teacher_topk_ids and teacher_topk_logits for top-k KD"
            )
        if target_count <= 0:
            raise ValueError("Teacher top-k KD cannot be aligned to an empty pseudo target")
        cached_count = len(record.teacher_topk_ids)
        includes_eos = self.teacher_cache.metadata.get("topk_includes_eos")
        if includes_eos is None:
            if cached_count == target_count:
                includes_eos = True
            elif cached_count == target_count - 1 and record.pseudo_token_ids:
                includes_eos = False
            else:
                raise ValueError(
                    "Teacher top-k EOS alignment metadata is missing or inconsistent with the student target: "
                    f"rows={cached_count}, target_tokens={target_count}"
                )
        includes_eos = bool(includes_eos)
        eos_id = self.decoder_tokenizer.eos_token_id
        expected_content_count = target_count - int(eos_id is not None)
        cached_content_count = len(record.pseudo_token_ids)
        if eos_id is not None and record.pseudo_token_ids and record.pseudo_token_ids[-1] == int(eos_id):
            cached_content_count -= 1
        target_was_truncated = cached_content_count > expected_content_count
        expected_count = (
            expected_content_count
            if target_was_truncated
            else (target_count if includes_eos else expected_content_count)
        )
        if cached_count < expected_count:
            raise ValueError(
                "Teacher top-k rows are shorter than the student-retokenized pseudo target: "
                f"rows={cached_count}, required_rows={expected_count}, target_tokens={target_count}"
            )
        if len(record.teacher_topk_logits) < expected_count:
            raise ValueError(
                "Teacher top-k logits are shorter than the student-retokenized pseudo target: "
                f"rows={len(record.teacher_topk_logits)}, required_rows={expected_count}, target_tokens={target_count}"
            )

        if student_target_ids is not None and record.pseudo_token_ids:
            student_content = student_target_ids.to(dtype=torch.long)
            if self.decoder_tokenizer.eos_token_id is not None and student_content.numel():
                if int(student_content[-1]) == int(self.decoder_tokenizer.eos_token_id):
                    student_content = student_content[:-1]
            if len(record.pseudo_token_ids) < student_content.numel():
                raise ValueError(
                    "Teacher pseudo_token_ids are shorter than the student-retokenized pseudo target; "
                    "teacher/student tokenization is not aligned"
                )
            mapped_pseudo_ids = self._map_teacher_ids(
                torch.tensor(record.pseudo_token_ids[: student_content.numel()], dtype=torch.long)
            )
            if not torch.equal(mapped_pseudo_ids, student_content.cpu()):
                raise ValueError(
                    "Teacher pseudo_token_ids do not match student pseudo-target tokenization; "
                    "token-level KD cannot be aligned safely"
                )

        ids_rows = record.teacher_topk_ids[:expected_count]
        logit_rows = record.teacher_topk_logits[:expected_count]
        normalizer_rows = record.teacher_topk_log_normalizers[:expected_count]
        if len(normalizer_rows) != expected_count:
            raise ValueError(
                "Teacher top-k log normalizers are missing or shorter than the aligned pseudo target; "
                "rebuild the teacher cache"
            )
        width = len(ids_rows[0]) if ids_rows else 0
        if width <= 0:
            raise ValueError("Teacher top-k rows must have positive width")
        if any(len(row) != width for row in ids_rows) or any(len(row) != width for row in logit_rows):
            raise ValueError("Teacher top-k IDs/logits must have a constant [T,K] shape")
        if any(len(ids_row) != len(logit_row) for ids_row, logit_row in zip(ids_rows, logit_rows)):
            raise ValueError("Teacher top-k ID/logit rows must have identical widths")

        teacher_ids = torch.tensor(ids_rows, dtype=torch.long)
        teacher_logits = torch.tensor(logit_rows, dtype=torch.float32)
        teacher_log_normalizers = torch.tensor(normalizer_rows, dtype=torch.float32)
        if not bool(torch.isfinite(teacher_logits).all()):
            raise ValueError("Teacher top-k logits must be finite")
        if not bool(torch.isfinite(teacher_log_normalizers).all()):
            raise ValueError("Teacher top-k log normalizers must be finite")
        student_ids = self._map_teacher_ids(teacher_ids)
        mask_eos = eos_id is not None and (
            target_was_truncated or not includes_eos or record.generated_eos_observed is False
        )
        if mask_eos:
            # The cache builder omits EOS because it is not part of the
            # generated pseudo trajectory, or the student target truncated a
            # longer trajectory. Preserve shape with an invalid EOS row so
            # soft KD never aligns an ordinary continuation logit to EOS.
            student_ids = torch.cat([student_ids, torch.zeros((1, width), dtype=torch.long)], dim=0)
            teacher_logits = torch.cat([teacher_logits, torch.zeros((1, width), dtype=torch.float32)], dim=0)
            teacher_log_normalizers = torch.cat(
                [teacher_log_normalizers, torch.zeros(1, dtype=torch.float32)],
                dim=0,
            )
            target_mask = torch.cat(
                [torch.ones(expected_content_count, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)], dim=0
            )
        else:
            target_mask = torch.ones(target_count, dtype=torch.bool)
        prefix_ids = torch.zeros((max(0, int(prompt_prefix_count)), width), dtype=torch.long)
        prefix_logits = torch.zeros((max(0, int(prompt_prefix_count)), width), dtype=torch.float32)
        prefix_normalizers = torch.zeros(max(0, int(prompt_prefix_count)), dtype=torch.float32)
        prefix_mask = torch.zeros(max(0, int(prompt_prefix_count)), dtype=torch.bool)
        return {
            "teacher_topk_ids": torch.cat([prefix_ids, student_ids], dim=0),
            "teacher_topk_logits": torch.cat([prefix_logits, teacher_logits], dim=0),
            "teacher_topk_log_normalizers": torch.cat(
                [prefix_normalizers, teacher_log_normalizers],
                dim=0,
            ),
            "teacher_kd_mask": torch.cat(
                [prefix_mask, target_mask],
                dim=0,
            ),
        }

    def _gold_topk_features(
        self,
        record: TeacherRecord,
        gold_token_ids: torch.Tensor,
        prompt_prefix_count: int,
    ) -> Dict[str, torch.Tensor]:
        """Align cached teacher soft targets with the original gold branch."""

        if not record.gold_topk_ids and not record.gold_topk_logits:
            return {}
        if not record.gold_topk_ids or not record.gold_topk_logits:
            raise ValueError("Gold teacher top-k IDs and logits must be supplied together")
        cached_gold_ids = torch.tensor(record.gold_token_ids, dtype=torch.long)
        expected_ids = gold_token_ids.to(dtype=torch.long).cpu()
        if cached_gold_ids.numel() and not torch.equal(cached_gold_ids, expected_ids):
            raise ValueError(
                "Gold target tokenization does not match the teacher cache; rebuild with the exact decoder tokenizer"
            )
        if len(record.gold_topk_ids) != expected_ids.numel() or len(record.gold_topk_logits) != expected_ids.numel():
            raise ValueError("Gold teacher top-k rows must align one-to-one with gold target tokens")
        width = len(record.gold_topk_ids[0]) if record.gold_topk_ids else 0
        if width <= 0 or any(len(row) != width for row in record.gold_topk_ids + record.gold_topk_logits):
            raise ValueError("Gold teacher top-k rows must have a constant positive width")
        teacher_ids = torch.tensor(record.gold_topk_ids, dtype=torch.long)
        teacher_logits = torch.tensor(record.gold_topk_logits, dtype=torch.float32)
        teacher_log_normalizers = torch.tensor(record.gold_topk_log_normalizers, dtype=torch.float32)
        if not bool(torch.isfinite(teacher_logits).all()):
            raise ValueError("Gold teacher top-k logits must be finite")
        if teacher_log_normalizers.numel() != expected_ids.numel():
            raise ValueError("Gold teacher top-k log normalizers must align with gold target tokens")
        if not bool(torch.isfinite(teacher_log_normalizers).all()):
            raise ValueError("Gold teacher top-k log normalizers must be finite")
        student_ids = self._map_teacher_ids(teacher_ids)
        prefix_ids = torch.zeros((max(0, int(prompt_prefix_count)), width), dtype=torch.long)
        prefix_logits = torch.zeros((max(0, int(prompt_prefix_count)), width), dtype=torch.float32)
        prefix_normalizers = torch.zeros(max(0, int(prompt_prefix_count)), dtype=torch.float32)
        prefix_mask = torch.zeros(max(0, int(prompt_prefix_count)), dtype=torch.bool)
        return {
            "teacher_gold_topk_ids": torch.cat([prefix_ids, student_ids], dim=0),
            "teacher_gold_topk_logits": torch.cat([prefix_logits, teacher_logits], dim=0),
            "teacher_gold_topk_log_normalizers": torch.cat(
                [prefix_normalizers, teacher_log_normalizers],
                dim=0,
            ),
            "teacher_gold_kd_mask": torch.cat([prefix_mask, torch.ones(expected_ids.numel(), dtype=torch.bool)], dim=0),
        }

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
        # A top-k cache carries the teacher's exact token trajectory.  Use it
        # directly rather than decode->clean->retokenize, because whitespace
        # normalization can change BPE boundaries and invalidate logit rows.
        pseudo_token_ids = record.pseudo_token_ids or None
        pseudo = self._target_features(pseudo_target, token_ids=pseudo_token_ids)
        item.update({f"pseudo_{key}": value for key, value in pseudo.items()})
        target_count = int(pseudo["labels"].ne(-100).sum().item())
        prompt_prefix_count = int(pseudo["labels"].eq(-100).sum().item())
        student_target_ids = pseudo["labels"][prompt_prefix_count:]
        item.update(
            self._topk_features(
                record,
                target_count,
                prompt_prefix_count,
                student_target_ids=student_target_ids,
            )
        )
        gold_prefix_count = int(item["labels"].eq(-100).sum().item())
        gold_target_ids = item["labels"][gold_prefix_count:]
        item.update(self._gold_topk_features(record, gold_target_ids, gold_prefix_count))
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
            if value.ndim != 2:
                raise ValueError("Teacher matrices must be [T,K] per example")
            if value.shape[1] != width:
                raise ValueError("Teacher matrices must have a constant width")
            value = value[:length, :width]
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
        pseudo_keys = {
            "pseudo_decoder_input_ids",
            "pseudo_decoder_attention_mask",
            "pseudo_labels",
        }
        if any(key in item for item in features for key in pseudo_keys):
            if not all(pseudo_keys.issubset(item) for item in features):
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
        teacher_keys_present = {key for item in features for key in item if key.startswith("teacher_")}
        if teacher_keys_present and "pseudo_decoder_input_ids" not in batch:
            raise ValueError("Teacher KD tensors require a pseudo decoder branch")

        def collate_topk(prefix: str, length: int, labels_key: str) -> None:
            ids_key = f"teacher_{prefix}topk_ids"
            logits_key = f"teacher_{prefix}topk_logits"
            normalizers_key = f"teacher_{prefix}topk_log_normalizers"
            mask_key = f"teacher_{prefix}kd_mask"
            present = ids_key in teacher_keys_present or logits_key in teacher_keys_present
            if not present:
                return
            if not all(ids_key in item and logits_key in item and normalizers_key in item for item in features):
                raise ValueError(f"Every item in a {prefix or 'pseudo'} top-k KD batch must have teacher tensors")
            width = int(features[0][ids_key].shape[1])
            if width <= 0 or any(int(item[ids_key].shape[1]) != width for item in features):
                raise ValueError("Teacher top-k width must be constant and positive in a batch")
            if any(item[ids_key].shape != item[logits_key].shape for item in features):
                raise ValueError("Teacher top-k IDs/logits must have identical shapes per item")
            if any(item[normalizers_key].shape != item[ids_key].shape[:1] for item in features):
                raise ValueError("Teacher top-k log normalizers must have one value per token row")
            masks = []
            for item in features:
                mask = item.get(mask_key)
                if mask is None:
                    mask = item[labels_key].ne(-100)
                if mask.ndim != 1 or mask.shape[0] != item[ids_key].shape[0]:
                    raise ValueError(f"{mask_key} must align with each teacher top-k row")
                masks.append(mask)
            batch.update(
                {
                    ids_key: self._pad_matrix((item[ids_key] for item in features), length, width, 0),
                    logits_key: self._pad_matrix((item[logits_key] for item in features), length, width, 0.0),
                    normalizers_key: self._pad(
                        (item[normalizers_key] for item in features),
                        length,
                        0.0,
                    ),
                    mask_key: self._pad(masks, length, 0).bool(),
                }
            )

        pseudo_length = int(batch["pseudo_decoder_input_ids"].shape[1])
        collate_topk("", pseudo_length, "pseudo_labels")
        if any(key in teacher_keys_present for key in ("teacher_gold_topk_ids", "teacher_gold_topk_logits")):
            if "labels" not in batch:
                raise ValueError("Gold teacher KD tensors require gold labels in the base collator output")
            collate_topk("gold_", int(batch["labels"].shape[1]), "labels")

        allowed_teacher_keys = {
            "teacher_topk_ids",
            "teacher_topk_logits",
            "teacher_topk_log_normalizers",
            "teacher_kd_mask",
            "teacher_gold_topk_ids",
            "teacher_gold_topk_logits",
            "teacher_gold_topk_log_normalizers",
            "teacher_gold_kd_mask",
        }
        unknown = teacher_keys_present - allowed_teacher_keys
        if unknown:
            raise ValueError(f"Unsupported teacher KD fields: {sorted(unknown)}")
        return batch


class OnlineKDCollator:
    """Left-pad teacher prompts and delegate student tensors to EviSeq."""

    def __init__(self, base_collator: Any, teacher_pad_id: int, teacher_max_input_length: int):
        self.base_collator = base_collator
        self.teacher_pad_id = int(teacher_pad_id)
        self.teacher_max_input_length = int(teacher_max_input_length)

    @staticmethod
    def _left_pad(values: Iterable[torch.Tensor], length: int, fill: int) -> torch.Tensor:
        rows = []
        for value in values:
            value = value[-length:]
            padding = torch.full((length - value.numel(),), int(fill), dtype=value.dtype)
            rows.append(torch.cat([padding, value]))
        return torch.stack(rows)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        required = {"teacher_input_ids", "teacher_attention_mask"}
        if not any(required.intersection(item) for item in features):
            return self.base_collator(features)
        if not all(required.issubset(item) for item in features):
            raise ValueError("Every online-KD training item must contain a teacher prompt")
        student_features = [{key: value for key, value in item.items() if key not in required} for item in features]
        batch = self.base_collator(student_features)
        length = min(
            self.teacher_max_input_length,
            max(int(item["teacher_input_ids"].numel()) for item in features),
        )
        batch["teacher_input_ids"] = self._left_pad(
            (item["teacher_input_ids"] for item in features),
            length,
            self.teacher_pad_id,
        )
        batch["teacher_attention_mask"] = self._left_pad(
            (item["teacher_attention_mask"] for item in features),
            length,
            0,
        )
        return batch
