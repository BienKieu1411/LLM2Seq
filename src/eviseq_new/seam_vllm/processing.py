"""vLLM 0.25.1 text-to-encoder prompt processing for SEAM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cached_property
from typing import Any

import torch
from eviseq_afmr.config import load_config, resolve_path
from eviseq_afmr.data.copy_alignment import align_copy_tokens
from eviseq_afmr.data.normalization import detokenize
from transformers import AutoTokenizer, Qwen3Config
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder

from ._text import normalize_text_payload


def _to_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer must return a single token-id sequence")
    return [int(item) for item in value]


class _TextItems(ProcessorBatchItems[str]):
    def __init__(self, data):
        if isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class _TextParser(MultiModalDataParser):
    def _parse_text(self, data: ModalityData[str]) -> ModalityDataItems[Any, Any] | None:
        payload = normalize_text_payload(data)
        return None if payload is None else _TextItems(payload)

    def _get_subparsers(self) -> Mapping[str, Any]:
        return {"text": self._parse_text}


class SEAMProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> Qwen3Config:
        return self.ctx.get_hf_config(Qwen3Config)

    @cached_property
    def seam_config(self) -> dict[str, Any]:
        path = getattr(self.get_hf_config(), "seam_config_path", None)
        if not path:
            raise ValueError("SEAM requires hf_overrides.seam_config_path pointing to a local resolved config")
        return load_config(str(path))

    @cached_property
    def encoder_tokenizer(self):
        model = self.seam_config["model"]
        if not model.get("tokenizer_use_fast", True):
            raise ValueError("SEAM requires the encoder's fast tokenizer for source offsets")
        return AutoTokenizer.from_pretrained(
            str(resolve_path(model["encoder_name"], self.seam_config)),
            use_fast=True,
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            local_files_only=True,
        )

    @cached_property
    def decoder_prompt_ids(self) -> list[int]:
        from ._prompt import configured_decoder_prompt_ids

        return configured_decoder_prompt_ids(self.get_tokenizer(), self.seam_config["data"])

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"text": 1}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int] | None:
        return {"text": int(self.seam_config["data"].get("max_source_length", seq_len))}

    def get_data_parser(self) -> MultiModalDataParser:
        return _TextParser()

    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)


class SEAMDummyInputsBuilder(BaseDummyInputsBuilder[SEAMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Summary:\n"

    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options):
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        # Decoding/re-encoding a chat prompt is not guaranteed to round-trip.
        inputs.prompt = list(self.info.decoder_prompt_ids)
        return inputs

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        if mm_counts.get("text", 0) == 0:
            return {}
        return {"text": " ".join(["source"] * max(1, seq_len))}


class SEAMMultimodalProcessor(EncDecMultiModalProcessor[SEAMProcessingInfo]):
    # The caller already supplies the exact trained decoder prompt, including
    # any BOS/chat-template tokens. Do not prepend a second start token.
    skip_decoder_start_token = True

    def create_encoder_prompt(self, prompt: str | list[int], mm_items: MultiModalDataItems) -> str | list[int]:
        return [0]

    def create_decoder_prompt(self, prompt: str | list[int], mm_items: MultiModalDataItems) -> str | list[int]:
        tokenizer = self.info.get_tokenizer()
        actual_prompt_ids = _to_ids(
            tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
        )
        if not actual_prompt_ids:
            raise ValueError("SEAM requires a non-empty decoder prompt")
        return actual_prompt_ids

    def apply(self, inputs, timing_ctx):
        from ._prompt import validate_decoder_prompt_ids

        actual = _to_ids(
            self.info.get_tokenizer().encode(inputs.prompt, add_special_tokens=False)
            if isinstance(inputs.prompt, str)
            else inputs.prompt
        )
        expected = inputs.hf_processor_mm_kwargs.get("seam_decoder_prompt_ids", self.info.decoder_prompt_ids)
        validate_decoder_prompt_ids(actual, list(expected))
        return super().apply(inputs, timing_ctx)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # The custom HF processor only returns the one-token encoder sentinel
        # [0]. vLLM must apply _get_prompt_updates to expand it to one placeholder
        # per encoded source token before the encoder outputs are gathered.
        return False

    def _encoder_prefix(self, encoder_prompt: str | None = None) -> str:
        data = self.info.seam_config["data"]
        if encoder_prompt is not None and not isinstance(encoder_prompt, str):
            raise ValueError("seam_encoder_prompt must be text")
        prefix = str(data.get("encoder_prefix", "")) if encoder_prompt is None else encoder_prompt
        marker = str(data.get("encoder_task_marker", ""))
        return prefix if not marker or prefix.startswith(marker) else marker + prefix

    def _encode_source(
        self, source: str, max_source_length: int | None = None, encoder_prompt: str | None = None
    ) -> dict[str, torch.Tensor]:
        config = self.info.seam_config
        data = config["data"]
        source = source.strip()
        if data.get("detokenize", False):
            source = detokenize(source)
        prefix = self._encoder_prefix(encoder_prompt)
        configured_limit = int(data.get("max_source_length", 4096))
        limit = configured_limit if max_source_length is None else int(max_source_length)
        if not 0 < limit <= configured_limit:
            raise ValueError(f"seam_max_source_length must be in [1, {configured_limit}]")
        encoded = self.info.encoder_tokenizer(
            prefix + source,
            add_special_tokens=True,
            truncation=True,
            max_length=limit,
            return_offsets_mapping=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = encoded["offset_mapping"]
        content = [end > max(start, len(prefix)) for start, end in offsets]
        if not any(content):
            raise ValueError("No source content remains after SEAM source truncation")
        result = {
            "encoder_input_ids": torch.tensor([ids], dtype=torch.long),
            "encoder_attention_mask": torch.ones((1, len(ids)), dtype=torch.bool),
            "source_content_mask": torch.tensor([content], dtype=torch.bool),
        }

        copy_config = config["decoder"].get("grounded_copy", {})
        if copy_config.get("enabled", False):
            decoder_tokenizer = self.info.get_tokenizer()
            if not getattr(decoder_tokenizer, "is_fast", False):
                raise ValueError("SEAM grounded copy requires vLLM's trained decoder fast tokenizer")
            copy = align_copy_tokens(source, len(prefix), offsets, decoder_tokenizer)
            token_ids = copy["copy_token_ids"]
            encoder_indices = copy["copy_encoder_indices"]
            token_indices = copy["copy_token_indices"]
            weights = copy["copy_alignment_weights"]
            token_width = max(1, len(token_ids))
            pair_width = max(1, len(encoder_indices))
            result.update(
                {
                    "copy_token_ids": torch.tensor(
                        [token_ids + [0] * (token_width - len(token_ids))], dtype=torch.long
                    ),
                    "copy_token_mask": torch.tensor(
                        [[True] * len(token_ids) + [False] * (token_width - len(token_ids))], dtype=torch.bool
                    ),
                    "copy_encoder_indices": torch.tensor(
                        [encoder_indices + [0] * (pair_width - len(encoder_indices))], dtype=torch.long
                    ),
                    "copy_token_indices": torch.tensor(
                        [token_indices + [0] * (pair_width - len(token_indices))], dtype=torch.long
                    ),
                    "copy_alignment_weights": torch.tensor(
                        [weights + [0.0] * (pair_width - len(weights))], dtype=torch.float32
                    ),
                }
            )
        return result

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        texts = None if mm_data is None else mm_data.get("texts")
        fields = (
            self._encode_source(texts[0], mm_kwargs.get("seam_max_source_length"), mm_kwargs.get("seam_encoder_prompt"))
            if texts
            else {}
        )
        if texts:
            budget = int(mm_kwargs.get("seam_output_budget", self.info.seam_config["generation"]["max_new_tokens"]))
            if budget < 1:
                raise ValueError("seam_output_budget must be positive")
            fields["requested_output_tokens"] = torch.tensor([[budget]], dtype=torch.float32)
            prompt_ids = mm_kwargs.get("seam_decoder_prompt_ids", self.info.decoder_prompt_ids)
            if not prompt_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in prompt_ids):
                raise ValueError("seam_decoder_prompt_ids must be non-empty token IDs")
            fields["bridge_prompt_ids"] = torch.tensor([prompt_ids], dtype=torch.long)
            fields["bridge_prompt_mask"] = torch.ones((1, len(prompt_ids)), dtype=torch.bool)
        if isinstance(prompt, (list, tuple)) and (not prompt or isinstance(prompt[0], int)):
            fields["input_ids"] = torch.tensor([prompt], dtype=torch.long)
        else:
            fields["input_ids"] = torch.as_tensor(
                self.info.get_tokenizer()(prompt or "", **tok_kwargs)["input_ids"], dtype=torch.long
            )
        return BatchFeature(fields)

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            key: MultiModalFieldConfig.batched("text")
            for key in (
                "encoder_input_ids",
                "encoder_attention_mask",
                "source_content_mask",
                "copy_token_ids",
                "copy_token_mask",
                "copy_encoder_indices",
                "copy_token_indices",
                "copy_alignment_weights",
                "requested_output_tokens",
                "bridge_prompt_ids",
                "bridge_prompt_mask",
            )
            if key in hf_inputs
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        texts = mm_items.get_items("text", _TextItems)
        if texts is None or len(texts) == 0:
            return []
        from vllm.multimodal.processing import PromptReplacement

        # The processor already tokenized and aligned the source. Reuse its
        # result instead of repeating tokenization and copy alignment here.
        source_token_count = int(out_mm_kwargs["text"][0]["encoder_input_ids"].data.numel())
        return [PromptReplacement(modality="text", target=[0], replacement=[0] * source_token_count)]


def register_processor() -> None:
    from .model import SEAMForQwen3

    MULTIMODAL_REGISTRY.register_processor(
        SEAMMultimodalProcessor,
        info=SEAMProcessingInfo,
        dummy_inputs=SEAMDummyInputsBuilder,
    )(SEAMForQwen3)
