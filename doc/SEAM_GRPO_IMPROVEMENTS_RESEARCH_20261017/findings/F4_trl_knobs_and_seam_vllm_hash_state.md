# Finding F4 — High-Leverage `trl==1.16.0` Algorithmic Knobs, Zero-Advantage Short-Circuiting, and `seam_vllm` Hash-Keyed `CopyState`

- **Thesis:** Exposing `trl==1.16.0`'s native `epsilon_high` (DAPO Clip-Higher), `importance_sampling_level` (GSPO), and `off_policy_mask_threshold` in `seam_grpo/config.py`, fixing the detached-entropy trap in `Policy.forward`, short-circuiting candidate backward passes on zero-advantage rows (`advantages == 0`), and keying `SEAMModelState.copy_states` by `feature.identifier` (`mm_hash`) fallback in `seam_vllm` simultaneously resolves a latent vLLM crash and unlocks $G\times$ faster `SamplingParams(n=G)` rollout.
- **Status:** Confirmed (Triangulated across Sources [01](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/01_dapo.md), [03](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/03_gspo.md), [05](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/05_tis_offpolicy.md), [12](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/12_vllm_encoder_cache.md), [15](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/15_local_seam_vllm_state_audit.md), [17](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/17_local_trl_1_16_0_audit.md)).

## Evidence & Technical Details

1. **Zero-Advantage Candidate Short-Circuit (`src/seam_grpo/seam_grpo/actor.py:59-83` & `trainer.py:217-229`):**
   - Whenever a candidate row has `advantages[i] == 0` (e.g., all $G=4$ completions in a group tie on reward or produce identical summaries) and `beta == 0.0`, its policy gradient contribution in `GRPOTrainer._compute_loss` is identically zero: $\nabla_\theta \mathcal{L}_{\text{RL}, i} = 0$.
   - Currently, `SEAMActor._score_candidates` still runs forward + checkpointed backward on all $B G$ rows even when `advantages == 0`.
   - Filtering `_score_candidates` to only active rows with `advantages != 0` (when `beta == 0` and `entropy_coef == 0`, substituting detached `old_per_token_logps` for zero-advantage rows so TRL logging stays valid) skips 100% of RL forward/backward compute on tied groups while preserving `_score_gold` CE anchor updates!

2. **Detached Entropy Trap in `Policy.forward` (`src/seam_grpo/seam_grpo/policy.py:210-216, 230`):**
   - `Policy.forward` computes `entropy` inside `with torch.no_grad():` and returns `entropy=entropy.detach()`.
   - If a user sets `entropy_coef > 0` in TRL, `loss = loss - self.entropy_coef * entropies` (`grpo_trainer.py:1946`) silently receives zero gradient!
   - Making `compute_entropy_with_grad: bool = False` configurable on `Policy.forward` (enabled automatically when `trainer.entropy_coef != 0.0`) fixes this silent bug without adding memory overhead when `entropy_coef == 0.0`.

3. **Latent `seam_vllm` `CopyState` Missing Crash & `n=G` Rollout Unlock (`src/eviseq_new/seam_vllm/state.py:83-108`):**
   - vLLM V1's `EncoderCacheManager` deduplicates multimodal encoder runs by `feature.identifier` (`mm_hash`).
   - If $G=4$ requests for the same document are split across two vLLM scheduler steps (e.g., under memory pressure) or generated via `SamplingParams(n=G)` on 1 prompt, `embed_multimodal` only executes on the first request. Subsequent requests hit the encoder cache for `prefix_embeds` (`bridge`) but miss `SEAMModelState.copy_states[req_id]`, raising `RuntimeError: SEAM copy state missing for active requests`.
   - Storing a hash-indexed cache `self.copy_states_by_hash[mm_hash]` in `SEAMModelState` (cleared on `reset_encoder_cache()` / `sync_weights()`) and falling back from `req_id` to `req_to_mm_hash[req_id]` inside `gather_copy_batch()` both **fixes the latent split-step crash** and **enables `SamplingParams(n=G)` on $B$ unique prompts**, cutting `seam_vllm` collator, encoder, bridge, and copy-prep overhead by $G\times$ ($4\times$).
