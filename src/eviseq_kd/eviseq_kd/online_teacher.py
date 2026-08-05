"""Frozen online teacher used only while training EviSeq-KD.

The teacher is deliberately kept outside the student's ``nn.Module`` tree:
its parameters must never enter the optimizer, checkpoints, or deployable
student graph.  It is loaded lazily on the first supervised training batch.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from .build_cache import _normalize_generated_row, _prediction_logits
from .student.data.dataset import decoder_seed_ids


def _tokenizer_vocab(tokenizer: Any) -> dict[str, int]:
    try:
        vocabulary = tokenizer.get_vocab()
    except AttributeError as exc:
        raise ValueError("Online KD requires tokenizers that expose get_vocab()") from exc
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ValueError("Online KD requires a non-empty tokenizer vocabulary")
    return {str(token): int(token_id) for token, token_id in vocabulary.items()}


def assert_shared_output_vocabulary(teacher_tokenizer: Any, student_tokenizer: Any) -> None:
    """Require exact token→ID equality before teacher logits supervise student IDs."""

    teacher_vocab = _tokenizer_vocab(teacher_tokenizer)
    student_vocab = _tokenizer_vocab(student_tokenizer)
    if teacher_vocab != student_vocab:
        raise ValueError(
            "Online logit/sequence KD requires identical teacher and student decoder token IDs. "
            "Use a Qwen3 teacher with the Qwen3 student decoder, or add an explicit vocabulary mapper."
        )
    teacher_eos = getattr(teacher_tokenizer, "eos_token_id", None)
    student_eos = getattr(student_tokenizer, "eos_token_id", None)
    if teacher_eos != student_eos:
        raise ValueError(
            f"Teacher/student EOS IDs differ ({teacher_eos} != {student_eos}); online KD cannot align trajectories"
        )


def _pad_rows(
    rows: Iterable[list[int]],
    *,
    fill: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = list(rows)
    width = max((len(row) for row in rows), default=0)
    values = torch.full((len(rows), width), int(fill), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        if row:
            values[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
            mask[row_index, : len(row)] = 1
    return values, mask


class OnlineTeacher:
    """Generate trajectories and soft targets online under ``inference_mode``."""

    def __init__(self, config: Dict[str, Any]):
        distillation = config["training"]["distillation"]
        self.model_name = str(distillation.get("teacher_model", "")).strip()
        if not self.model_name:
            raise ValueError("Online KD requires training.distillation.teacher_model")
        self.device_name = str(distillation.get("teacher_device", "auto")).strip().lower()
        self.batch_size = int(distillation.get("teacher_batch_size", 1))
        self.max_new_tokens = int(
            distillation.get("teacher_max_new_tokens", config.get("generation", {}).get("max_new_tokens", 384))
        )
        self.num_beams = int(distillation.get("teacher_num_beams", 1))
        self.top_k = int(distillation.get("topk", 32))
        self.temperature = float(distillation.get("temperature", 2.0))
        self.logit_enabled = bool(distillation.get("logit_enabled", False))
        self.logit_path_mix = float(distillation.get("logit_path_mix", 0.5))
        self.data_config = config["data"]
        self.max_target_length = int(self.data_config.get("max_target_length", 384))
        if self.batch_size <= 0:
            raise ValueError("training.distillation.teacher_batch_size must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("training.distillation.teacher_max_new_tokens must be positive")
        if self.max_target_length <= 1:
            raise ValueError("data.max_target_length must be greater than one for online KD")
        if self.num_beams <= 0:
            raise ValueError("training.distillation.teacher_num_beams must be positive")
        if self.logit_enabled and self.top_k <= 0:
            raise ValueError("Online logit KD requires training.distillation.topk > 0")
        if self.temperature <= 0.0:
            raise ValueError("training.distillation.temperature must be positive")

        self._tokenizer: Any = None
        self._model: Any = None
        self._model_device: torch.device | None = None
        self._decoder_seed: list[int] | None = None

    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("Online teacher tokenizer has neither PAD nor EOS")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            self._tokenizer = tokenizer
        return self._tokenizer

    def configure_student_tokenizer(self, student_tokenizer: Any) -> None:
        teacher_tokenizer = self.tokenizer()
        assert_shared_output_vocabulary(teacher_tokenizer, student_tokenizer)
        self._decoder_seed = decoder_seed_ids(student_tokenizer, self.data_config)

    def _resolve_device(self, student_device: torch.device) -> torch.device:
        if self.device_name in {"", "auto", "student"}:
            return student_device
        return torch.device(self.device_name)

    def _ensure_model(self, student_device: torch.device) -> tuple[Any, torch.device]:
        target_device = self._resolve_device(student_device)
        if self._model is None:
            from transformers import AutoModelForCausalLM

            dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(target_device)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model
            self._model_device = target_device
        elif self._model_device != target_device:
            raise RuntimeError(
                f"Online teacher was loaded on {self._model_device}, but this batch is assigned to {target_device}"
            )
        return self._model, target_device

    @staticmethod
    def _gold_rows(labels: torch.Tensor) -> list[list[int]]:
        return [[int(token) for token in row[row.ne(-100)].tolist()] for row in labels]

    def _student_pseudo_features(
        self,
        rows: list[list[int]],
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if self._decoder_seed is None:
            raise RuntimeError("configure_student_tokenizer() must be called before online KD")
        seed = self._decoder_seed
        decoder_rows = [seed + row[:-1] for row in rows]
        label_rows = [[-100] * max(0, len(seed) - 1) + row for row in rows]
        pad_id = int(self.tokenizer().pad_token_id)
        decoder_input_ids, decoder_attention_mask = _pad_rows(decoder_rows, fill=pad_id, device=device)
        pseudo_labels, _ = _pad_rows(label_rows, fill=-100, device=device)
        return {
            "pseudo_decoder_input_ids": decoder_input_ids,
            "pseudo_decoder_attention_mask": decoder_attention_mask,
            "pseudo_labels": pseudo_labels,
        }

    def _fit_generated_row(self, row: list[int], observed_eos: bool) -> tuple[list[int], bool]:
        """Apply the student's target budget without aligning a wrong EOS row."""

        eos_id = self.tokenizer().eos_token_id
        budget = self.max_target_length - int(eos_id is not None)
        content = row[:-1] if eos_id is not None and row and row[-1] == int(eos_id) else row
        content = content[: max(1, budget)]
        if eos_id is None:
            return content, observed_eos
        reached_real_eos = observed_eos and len(row) <= budget + 1 and row[-1] == int(eos_id)
        return content + [int(eos_id)], reached_real_eos

    def _soft_targets(
        self,
        teacher: Any,
        prompts: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_rows: list[list[int]],
        *,
        output_device: torch.device,
        prefix_width: int,
        mask_synthetic_eos: list[bool] | None = None,
    ) -> Dict[str, torch.Tensor]:
        teacher_device = prompts.device
        pad_id = int(self.tokenizer().pad_token_id)
        targets, target_mask = _pad_rows(target_rows, fill=pad_id, device=teacher_device)
        prompt_width = int(prompts.shape[1])
        target_width = int(targets.shape[1])
        combined_ids = torch.cat([prompts, targets], dim=1)
        combined_mask = torch.cat([prompt_mask, target_mask], dim=1)
        logits = _prediction_logits(
            teacher,
            input_ids=combined_ids,
            attention_mask=combined_mask,
            prompt_width=prompt_width,
            target_width=target_width,
        )
        logits = logits.float()
        width = min(self.top_k, int(logits.shape[-1]))
        topk_logits, topk_ids = torch.topk(logits, k=width, dim=-1)
        normalizers = torch.logsumexp(logits / self.temperature, dim=-1)
        kd_mask = target_mask.bool()
        if mask_synthetic_eos is not None:
            for row_index, synthetic in enumerate(mask_synthetic_eos):
                if synthetic and target_rows[row_index]:
                    kd_mask[row_index, len(target_rows[row_index]) - 1] = False

        batch = len(target_rows)
        prefix_ids = torch.zeros((batch, prefix_width, width), dtype=torch.long, device=teacher_device)
        prefix_logits = torch.zeros((batch, prefix_width, width), dtype=torch.float32, device=teacher_device)
        prefix_normalizers = torch.zeros((batch, prefix_width), dtype=torch.float32, device=teacher_device)
        prefix_mask = torch.zeros((batch, prefix_width), dtype=torch.bool, device=teacher_device)
        return {
            "topk_ids": torch.cat([prefix_ids, topk_ids], dim=1).to(output_device),
            "topk_logits": torch.cat([prefix_logits, topk_logits], dim=1).to(output_device),
            "topk_log_normalizers": torch.cat([prefix_normalizers, normalizers], dim=1).to(output_device),
            "kd_mask": torch.cat([prefix_mask, kd_mask], dim=1).to(output_device),
        }

    @torch.inference_mode()
    def distill_batch(
        self,
        *,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        gold_labels: torch.Tensor,
        output_device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        teacher, teacher_device = self._ensure_model(output_device)
        tokenizer = self.tokenizer()
        pad_id = int(tokenizer.pad_token_id)
        eos_id = tokenizer.eos_token_id
        all_pseudo_rows: list[list[int]] = []
        all_eos_observed: list[bool] = []
        pseudo_soft_chunks: list[Dict[str, torch.Tensor]] = []
        gold_soft_chunks: list[Dict[str, torch.Tensor]] = []
        gold_rows = self._gold_rows(gold_labels)
        gold_prefix_counts = gold_labels.eq(-100).sum(dim=1)
        if gold_prefix_counts.numel() and not bool(gold_prefix_counts.eq(gold_prefix_counts[0]).all()):
            raise ValueError("Student gold labels must use one constant decoder-prompt width per batch")
        gold_prefix_width = int(gold_prefix_counts[0].item()) if gold_prefix_counts.numel() else 0

        for start in range(0, int(teacher_input_ids.shape[0]), self.batch_size):
            stop = min(start + self.batch_size, int(teacher_input_ids.shape[0]))
            prompts = teacher_input_ids[start:stop].to(teacher_device)
            prompt_mask = teacher_attention_mask[start:stop].to(teacher_device)
            generated = teacher.generate(
                input_ids=prompts,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                use_cache=True,
            )
            generated_suffix = generated[:, prompts.shape[1] :]
            pseudo_rows: list[list[int]] = []
            eos_observed: list[bool] = []
            for raw_row in generated_suffix.tolist():
                row, observed = _normalize_generated_row(raw_row, pad_id=pad_id, eos_id=eos_id)
                row, observed = self._fit_generated_row(row, observed)
                pseudo_rows.append(row)
                eos_observed.append(observed)
            all_pseudo_rows.extend(pseudo_rows)
            all_eos_observed.extend(eos_observed)

            if self.logit_enabled:
                pseudo_soft_chunks.append(
                    self._soft_targets(
                        teacher,
                        prompts,
                        prompt_mask,
                        pseudo_rows,
                        output_device=output_device,
                        prefix_width=max(0, len(self._decoder_seed or []) - 1),
                        mask_synthetic_eos=[not observed for observed in eos_observed],
                    )
                )
                if self.logit_path_mix < 1.0:
                    gold_soft_chunks.append(
                        self._soft_targets(
                            teacher,
                            prompts,
                            prompt_mask,
                            gold_rows[start:stop],
                            output_device=output_device,
                            prefix_width=gold_prefix_width,
                        )
                    )

        result = self._student_pseudo_features(all_pseudo_rows, device=output_device)

        def merge(chunks: list[Dict[str, torch.Tensor]], output_prefix: str) -> None:
            if not chunks:
                return
            max_length = max(int(chunk["topk_ids"].shape[1]) for chunk in chunks)

            def pad(tensor: torch.Tensor, fill: float) -> torch.Tensor:
                missing = max_length - int(tensor.shape[1])
                if missing <= 0:
                    return tensor
                shape = (tensor.shape[0], missing, *tensor.shape[2:])
                return torch.cat([tensor, torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)], dim=1)

            result[f"teacher_{output_prefix}topk_ids"] = torch.cat(
                [pad(chunk["topk_ids"], 0) for chunk in chunks], dim=0
            )
            result[f"teacher_{output_prefix}topk_logits"] = torch.cat(
                [pad(chunk["topk_logits"], 0.0) for chunk in chunks], dim=0
            )
            result[f"teacher_{output_prefix}topk_log_normalizers"] = torch.cat(
                [pad(chunk["topk_log_normalizers"], 0.0) for chunk in chunks], dim=0
            )
            result[f"teacher_{output_prefix}kd_mask"] = torch.cat(
                [pad(chunk["kd_mask"], 0).bool() for chunk in chunks], dim=0
            )

        merge(pseudo_soft_chunks, "")
        merge(gold_soft_chunks, "gold_")
        return result
