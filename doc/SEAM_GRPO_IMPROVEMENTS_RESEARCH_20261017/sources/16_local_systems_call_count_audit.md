# Source 16 — Local Codebase Audit: Per-Step Call Counts & Autograd Equivalence of Group-Unique Source Encoding

- **Title:** Static & Dynamic Call-Count Audit of `LengthCollator`, `align_copy_tokens`, `EviSeqAFMR.encode_source`, and `SEAMActor` with Autograd Equivalence Proof
- **Source Type:** Primary Local Codebase Audit
- **Files Audited:**
  - `src/seam_grpo/seam_grpo/actor.py`
  - `src/seam_grpo/seam_grpo/policy.py`
  - `src/seam_grpo/seam_grpo/trainer.py`
  - `src/seam_length_grpo/seam_length_grpo/collator.py`
  - `src/seam_length_grpo/seam_length_grpo/teacher.py`
  - `src/eviseq_new/eviseq_afmr/modeling/model.py`
  - `src/eviseq_new/eviseq_afmr/modeling/grounded_copy.py`
  - `src/eviseq_new/eviseq_afmr/modeling/outputs.py`
- **Credibility:** High | **Recency:** Current (2026-10-17) | **Bias:** None (exact structural trace)

## Verbatim Quotes (<25 words each)

> `src/seam_grpo/seam_grpo/actor.py:73-74`:
> `"scores = checkpoint_Utilities.checkpoint(self._score_slice, self.policy, source_slice, action_slice, attention_mask, use_reentrant=False)"`

> `src/eviseq_new/eviseq_afmr/modeling/model.py:165-169`:
> `"encoder_hidden, encoder_mask = self.encoder(input_ids=source_input_ids, attention_mask=source_attention_mask)"`

## Key Takeaways & Call-Count Accounting

For `prompt_batch_size` $= B$, `group_size` $= G$, `gradient_accumulation_steps` $= A$, `score_micro_batch_size` $= S$ ($A \cdot B$ unique prompts per optimizer step; default $B=1, G=4, A=8, S=1 \implies A B = 8$ unique prompts, $A B G = 32$ candidate rows):

1. **`LengthCollator.__call__` & `align_copy_tokens` (`src/seam_length_grpo/seam_length_grpo/collator.py:42-83`):**
   - Called inside `SEAMGRPOTrainer._prepare_inputs` on all $A B G = 32$ repeated rows (32 calls to `align_copy_tokens` for only $A B = 8$ unique sources).
   - When `vllm_enable: true`, `SEAMProcessor._prepare_batch` calls `SummarizationCollator` + `align_copy_tokens` another $A B G = 32$ times.
   - When `source_weight > 0`, `SourceTeacher.__call__` calls `self.collator(negatives)` on $A B G = 32$ rows (another 32 calls to `align_copy_tokens` for only $A B = 8$ unique negative sources).
   - **Total `align_copy_tokens` calls per optimizer step:** Up to **$96$ calls** for only **$16$ unique texts** ($8$ positive + $8$ negative).

2. **`EviSeqAFMR.encode_source` (`encoder` + FP32 `bridge` + `GroundedCopyHead.prepare`) Row Executions per Optimizer Step:**
   - **Stage 1 (Rollout):** $A B G = 32$ source rows (both in native `Rollout` and in `LengthVLLMRollout` with `n=1` over $G$-repeated inputs).
   - **Stage 2 (`SourceTeacher.__call__` when `source_weight > 0`):** $A B G = 32$ `true_source` rows + $A B G = 32$ `wrong_source` rows = **$64$ rows** (`no_grad`).
   - **Stage 3 (`ref_model` when `beta > 0`):** $A B G = 32$ rows (`no_grad`) + $A B = 8$ gold rows (`no_grad`).
   - **Stage 4 (`SEAMActor._score_candidates` with `gradient_checkpointing=True`):**
     - Forward pass: $A B G = 32$ rows (`no_grad` inside `torch.utils.checkpoint`).
     - Backward recomputation: $A B G = 32$ rows (`enable_grad`).
     - Subtotal: **$64$ rows**.
   - **Stage 5 (`SEAMActor._score_gold` when `ce_weight > 0`):**
     - Forward pass: $A B = 8$ rows (`no_grad` inside `torch.utils.checkpoint`).
     - Backward recomputation: $A B = 8$ rows (`enable_grad`).
     - Subtotal: **$16$ rows**.
   - **Grand Total (with `source_weight > 0, ce_weight > 0, beta == 0`):** **$176$ `encode_source` row executions** per optimizer step when only **$16$ unique documents** ($8$ positive, $8$ negative) exist in the step!

3. **Exact Autograd Equivalence of Unique-Source Encoding + `index_select` Expansion:**
   - In `Policy.train()` (`src/seam_grpo/seam_grpo/policy.py:134-142`), `self.model.encoder.eval()`, `self.model.bridge.eval()`, and `self.model.copy_head.eval()` are enforced, disabling all dropout inside `encode_source`.
   - Every operation in `encode_source` (`encoder`, `AFMRBridge`, `GroundedCopyHead.prepare`) is strictly independent across batch rows and completely independent of candidate `action_ids` or `labels`.
   - Let $U \le B G$ be the unique source rows in a microbatch and $\pi \in \{0, \dots, U-1\}^{B G}$ be the inverse mapping such that row $i$ uses unique source $\pi(i)$.
   - Computing `(bridge_u, copy_u) = model.encode_source(unique_source)` on $U$ rows and indexing `bridge_u.prefix_embeds.index_select(0, pi)` (and similarly for `copy_u` tensors) before running the decoder produces **bit-identical forward activations** and **mathematically identical parameter gradients**:
     $$\frac{\partial \mathcal{L}}{\partial \theta_{\text{enc/bridge/copy}}} = \sum_{u=0}^{U-1} \left(\sum_{i: \pi(i)=u} \frac{\partial \mathcal{L}}{\partial Z_i}\right) \frac{\partial Z_u}{\partial \theta_{\text{enc/bridge/copy}}}$$
   - Furthermore, `_score_candidates` and `_score_gold` share the exact same `source_input_ids` for the same prompt! Sharing `(bridge_u, copy_u)` across both the $G$ candidate trajectories and the $1$ gold CE trajectory eliminates `_score_gold`'s separate `encode_source` calls altogether.
