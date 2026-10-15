"""Request-side copy state for vLLM's encoder-decoder runner."""

from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.encoder_decoder import EncoderDecoderModelState
from vllm.v1.worker.gpu.states import RequestState


class SEAMModelState(EncoderDecoderModelState):
    """Keep each source's exact copy route attached to its request ID."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model,
        encoder_cache: EncoderCache | None,
        device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.copy_states: dict[str, Any] = {}

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> None:
        ordered_inputs: dict[str, list[int]] = {}
        for req_id in input_batch.req_ids:
            indices = scheduled_encoder_inputs.get(req_id, [])
            if indices:
                ordered_inputs[req_id] = indices

        for req_id, indices in ordered_inputs.items():
            for index in indices:
                feature = self.encoder_cache.mm_features[req_id][index]
                if feature.data is None:
                    raise RuntimeError(
                        "SEAM cannot encode a scheduled source whose multimodal payload was replaced by a cache "
                        "reference. Encoder outputs and grounded-copy alignment must be restored together; this "
                        "vLLM cache path is not supported."
                    )

        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(ordered_inputs)
        if not mm_kwargs:
            self.encoder_outputs = []
            return

        self.encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
        encoded_copy_states = self.model.drain_copy_states()
        if len(encoded_copy_states) != len(mm_hashes):
            raise RuntimeError(
                "SEAM encoder/copy output count differs from vLLM's source-item count; refusing to lose copy state"
            )
        copy_by_hash = dict(zip(mm_hashes, encoded_copy_states, strict=True))

        for req_id, indices in ordered_inputs.items():
            if len(indices) != 1:
                raise ValueError("SEAM expects exactly one source-text item per request")
            feature = self.encoder_cache.mm_features[req_id][indices[0]]
            state = copy_by_hash.get(feature.identifier)
            if state is None and self.model.copy_head is not None:
                raise RuntimeError(f"SEAM copy state is missing for request {req_id}")
            self.copy_states[req_id] = state

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.model.copy_head is not None:
            missing = [req_id for req_id in input_batch.req_ids if req_id not in self.copy_states]
            if missing:
                raise RuntimeError(f"SEAM copy state missing for active requests: {missing[:3]}")
            model_inputs["copy_states"] = [self.copy_states[req_id] for req_id in input_batch.req_ids]
        else:
            model_inputs["copy_states"] = None
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"encoder_outputs": [], "copy_states": None}

    def remove_request(self, req_id: str) -> None:
        self.copy_states.pop(req_id, None)
        super().remove_request(req_id)
