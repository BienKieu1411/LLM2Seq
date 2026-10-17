# Local Code Audit: SEAM vLLM State Management for n=G and Multimodal Caching

- Source: `src/eviseq_new/seam_vllm/state.py:49-121`, `src/eviseq_new/seam_vllm/model.py:466-585`, `src/eviseq_new/seam_vllm/api.py:84-160`, `src/seam_grpo/seam_grpo/vllm_rollout.py:17-43, 89-145`.
- Accessed: 2026-10-17.
- Type: primary local codebase inspection.
- Credibility: direct code inspection. Recency: current workspace snapshot. Bias: none.

Verified code excerpt (`src/eviseq_new/seam_vllm/state.py:83-108`):
```python
copy_by_hash = dict(zip(mm_hashes, encoded_copy_states, strict=True))
for req_id, indices in ordered_inputs.items():
    feature = self.encoder_cache.mm_features[req_id][indices[0]]
    state = copy_by_hash.get(feature.identifier)
    if state is None:
        raise RuntimeError(f"Missing SEAM copy state for encoded source {feature.identifier}")
    self.copy_states[req_id] = state
...
missing = [req_id for req_id in input_batch.req_ids if req_id not in self.copy_states]
if missing:
    raise RuntimeError(f"SEAM copy state missing for active requests: {missing[:3]}")
```

Key findings:
1. **Why `SamplingParams(n=G)` or split-step groups currently fail with grounded copy:** vLLM's `EncoderCacheManager` deduplicates encoder scheduling by `feature.identifier` (`mm_hash`). When a second request (or child request from `n=G`) shares an already-cached `mm_hash` from an earlier scheduler step (or when `scheduled_encoder_inputs` only schedules the first child request of `n=G`), `ordered_inputs` in `get_mm_embeddings` omits the subsequent `req_id`, leaving `self.copy_states[req_id]` unpopulated and raising `RuntimeError: SEAM copy state missing for active requests` in `prepare_inputs`.
2. **Fix required to unlock `n=G` and multi-step group scheduling in `seam_vllm`:** Maintain a hash-indexed copy-state map `self.copy_states_by_hash[feature.identifier]` alongside `self.copy_states[req_id]` in `SEAMModelState` (and clear it in `reset_encoder_cache` / `begin_actor_update`). In `prepare_inputs`, if `req_id not in self.copy_states`, look up `feature = self.encoder_cache.mm_features[req_id][0]` and resolve `self.copy_states[req_id] = self.copy_states_by_hash[feature.identifier]`.
3. **Why `mm_processor_cache_gb=0` must stay enforced:** When `mm_processor_cache_gb > 0`, vLLM replaces cached multimodal payloads with `feature.data = None`, which `state.py:64-70` explicitly rejects.
