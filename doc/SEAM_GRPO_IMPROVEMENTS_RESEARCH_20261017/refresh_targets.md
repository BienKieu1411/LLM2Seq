# Refresh Targets — SEAM GRPO & Length-GRPO Research (`2026-10-17`)

When refreshing or extending this research folder after implementing Phase 1–3 changes, re-verify the following concrete targets:

## 1. Codebase & Package Targets
- **Installed `trl` version:** Currently `trl==1.16.0` in `/Users/kieugiangbien/bienkieu_env/lib/python3.13/site-packages/trl`. If upgraded to `trl>=1.17`, check `GRPOTrainer._prepare_inputs` (`shuffle_sequence_dict`), `_calculate_rewards` (`multi_objective_aggregation`), and `_compute_loss` (`loss_type`, `epsilon_high`, `importance_sampling_level`, `off_policy_mask_threshold`).
- **Installed `vllm` version:** Check `vllm.v1.core.encoder_cache_manager.EncoderCacheManager` behavior when `SamplingParams(n=G)` is used with `mm_processor_cache_gb=0.0` and `reset_encoder_cache()` after `sync_weights()`.
- **Local files to re-audit after edits:**
  - `src/seam_grpo/seam_grpo/actor.py` (unique-source encoding & zero-advantage short-circuit)
  - `src/seam_grpo/seam_grpo/policy.py` (pre-encoded `(BridgeState, CopyState)` support & optional differentiable entropy)
  - `src/seam_grpo/seam_grpo/config.py` (TRL 1.16.0 config passthrough)
  - `src/seam_length_grpo/seam_length_grpo/reward.py` (Huber-clipped length cost & GDPO / GR3 advantage modes)
  - `src/seam_length_grpo/seam_length_grpo/teacher.py` (group-unique source encoding, copy-logprob cap, `ref_model` logprob reuse)
  - `src/seam_length_grpo/seam_length_grpo/train.py` (loading `reference` after `run_ce_warmup`)
  - `src/eviseq_new/seam_vllm/state.py` (`copy_states_by_hash` fallback and cleanup on `sync_weights`)

## 2. Empirical Metrics to Watch (Lifecycle Gate)
- **Throughput / Step Time:** Wall-clock seconds per optimizer step (`A=8, B=1, G=4`) and peak GPU VRAM (`torch.cuda.max_memory_allocated()`) before vs. after Finding F1 (`encode_source` deduplication) and Finding F4 (`n=G` vLLM rollout).
- **Length Compliance vs. ROUGE Frontier (`Lifecycle` report):**
  - `band_compliance` and `mean_band_cost` across all 4 datasets (`vietnews`, `vlsp22`, `wikilingua`, `xlsum`) and 3 buckets (`short`, `medium`, `long`).
  - `rouge1`, `rouge2`, `rougeL` F1 deltas relative to `length_ce_warmup.pt` (ensuring `vlsp22` and `vietnews` ROUGE-2 do not degrade when enforcing `short` / `medium` length bands).
  - Fraction of rollout groups with zero advantage or verbose outlier hijacking ($C_i > 2.0$).
