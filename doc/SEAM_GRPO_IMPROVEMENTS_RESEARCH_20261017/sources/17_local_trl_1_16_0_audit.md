# Source 17 — Local Package Audit: Installed `trl==1.16.0` `GRPOConfig` and `GRPOTrainer` Mechanics

- **Title:** Direct Inspection of `trl==1.16.0` (`grpo_config.py` and `grpo_trainer.py`) and Interaction with `SEAMGRPOTrainer`
- **Source Type:** Primary Local Package Audit
- **Files Audited:**
  - `/Users/kieugiangbien/bienkieu_env/lib/python3.13/site-packages/trl/trainer/grpo_config.py`
  - `/Users/kieugiangbien/bienkieu_env/lib/python3.13/site-packages/trl/trainer/grpo_trainer.py`
  - `src/seam_grpo/seam_grpo/trainer.py`
  - `src/seam_grpo/seam_grpo/config.py`
  - `src/seam_grpo/seam_grpo/policy.py`
- **Credibility:** High | **Recency:** Current (`trl==1.16.0`) | **Bias:** None

## Verbatim Quotes (<25 words each)

> `trl/trainer/grpo_trainer.py:1388`:
> `"generation_batch = shuffle_sequence_dict(generation_batch)"`

> `trl/trainer/grpo_trainer.py:1945-1946`:
> `"if self.entropy_coef != 0.0: loss = loss - self.entropy_coef * entropies"`

## Key Takeaways

1. **Supported Knobs in Installed `trl==1.16.0` (`grpo_config.py:391-670`):**
   - `loss_type`: `"grpo"`, `"bnpo"`, `"dr_grpo"`, `"dapo"` (default in TRL 1.16.0), `"cispo"`, `"sapo"`, `"luspo"`, `"vespo"`.
   - `epsilon` (default `0.2`) and `epsilon_high` (default `None` $\to$ falls back to `epsilon`; enables DAPO Clip-Higher when set to `0.28`).
   - `delta` (default `None`; two-sided upper ratio clamp).
   - `importance_sampling_level`: `"token"` (default) or `"sequence"` (enables GSPO sequence-level importance ratios).
   - `scale_rewards`: `"group"`, `"batch"`, or `"none"`.
   - `multi_objective_aggregation`: `"sum_then_normalize"` (default) or `"normalize_then_sum"` (native GDPO implementation in `grpo_trainer.py:1661-1689` when multiple reward functions or multi-column rewards are returned).
   - `mask_truncated_completions`: `bool` (default `False` in TRL, set `True` in SEAM configs).
   - `top_entropy_quantile`: `float` (default `1.0`; masks policy gradient updates to only the top-$q$ highest-entropy tokens).
   - `entropy_coef`: `float` (default `0.0`).
   - `off_policy_mask_threshold`: `float | None` (default `None`; sequence-level off-policy KL/ratio masking).
   - `use_bias_correction_kl`: `bool` (default `False`; importance-weighted k3 KL estimator).

2. **Three Critical Interactions Between `trl==1.16.0` and `SEAMGRPOTrainer`:**
   - **Interaction A (`shuffle_sequence_dict` at `grpo_trainer.py:1388`):**
     After `_generate_and_score_completions` finishes computing group-normalized advantages on the $A B G$ rollout batch, `GRPOTrainer._prepare_inputs` permutes the $A B G$ rows via `shuffle_sequence_dict(generation_batch)` before splitting into $A$ microbatches of size $B G$.
     - *Implication for H1:* If `SEAMActor` assumes every contiguous block of $G$ rows in `inputs` shares the same source, `shuffle_sequence_dict` violates that assumption whenever $A > 1$! Instead, deduplicating rows dynamically by `source_input_ids` within `SEAMActor` (or overriding the batch split in `SEAMGRPOTrainer._prepare_inputs` to permute at the **group** level rather than the individual row level) both preserves group locality ($U = B$ unique sources per microbatch) and remains 100% compatible with TRL.
   - **Interaction B (`Policy.forward` detaches `entropy` at `src/seam_grpo/seam_grpo/policy.py:210-216, 230`):**
     `Policy.forward` computes `entropy` inside `with torch.no_grad():` and returns `entropy=entropy.detach()`.
     - *Implication:* Setting `top_entropy_quantile < 1.0` works immediately (as quantile thresholding only needs detached entropy values), **but** setting `entropy_coef != 0.0` in `GRPOConfig` silently contributes zero gradient unless `Policy.forward` computes differentiable entropy when `compute_entropy_with_grad=True`.
   - **Interaction C (`config.py` whitelist in `src/seam_grpo/seam_grpo/config.py:131-168`):**
     `Training.to_trl()` currently passes only a fixed subset of fields to `GRPOConfig(...)`, omitting `epsilon_high`, `delta`, `importance_sampling_level`, `top_entropy_quantile`, `off_policy_mask_threshold`, and `use_bias_correction_kl`. Adding these fields to `Training` and `Training.to_trl()` immediately unlocks TRL 1.16.0's native implementations in `SEAMGRPOTrainer._compute_loss`.
