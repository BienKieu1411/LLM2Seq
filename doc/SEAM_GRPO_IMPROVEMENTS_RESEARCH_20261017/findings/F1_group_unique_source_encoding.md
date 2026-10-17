# Finding F1 — Group-Unique Source Encoding Eliminates ~75%–80% of Redundant Encoder/Bridge/Copy Compute with Bit-Identical Gradients

- **Thesis:** Because `EviSeqAFMR.encode_source` (`encoder` + FP32 `AFMRBridge` + `GroundedCopyHead.prepare`) has zero dependence on candidate `action_ids` or `labels` and executes in `eval()` mode (`dropout=0`), computing `(BridgeState, CopyState)` once per unique source row in a microbatch and indexing across the $G$ candidate trajectories and $1$ gold CE trajectory via `index_select(0, inverse_idx)` produces **bit-identical forward activations and mathematically identical parameter gradients** while eliminating $(G-1)/G = 75\%$ (at $G=4$) to $(G+1-1)/(G+1) = 80\%$ (including gold CE) of `encode_source` executions.
- **Status:** Confirmed (Triangulated across Sources [14](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/14_local_source_teacher_audit.md), [15](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/15_local_seam_vllm_state_audit.md), [16](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/16_local_systems_call_count_audit.md), [17](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/17_local_trl_1_16_0_audit.md)).

## Evidence & Call-Count Proof

For $B=1, G=4, A=8, S=1$ ($A \cdot B = 8$ unique prompts per optimizer step):

| Pipeline Stage | Current `encode_source` Rows / Step | Optimized `encode_source` Rows / Step | Savings |
| :--- | :---: | :---: | :---: |
| Rollout (`Rollout` or `LengthVLLMRollout`) | $A B G = 32$ | $A B = 8$ | **$-75.0\%$** |
| `SourceTeacher.__call__` (`true_source` + `wrong_source`) | $2 A B G = 64$ | $2 A B = 16$ | **$-75.0\%$** |
| `SEAMActor._score_candidates` (forward + ckpt backward) | $2 A B G = 64$ | $2 A B = 16$ | **$-75.0\%$** |
| `SEAMActor._score_gold` (forward + ckpt backward) | $2 A B = 16$ | $0$ (reuses candidate `encode_source`!) | **$-100.0\%$** |
| **Total per Optimizer Step** | **$176$ rows** | **$40$ rows** | **$-77.3\%$ ($4.4\times$ reduction)** |

Similarly, `align_copy_tokens` inside `LengthCollator` and `SourceTeacher` drops from **$96$ calls/step** to **$16$ calls/step** ($8$ positive + $8$ negative sources, **$-83.3\%$**).

## Important Implementation Guardrail (`shuffle_sequence_dict`)

In `trl==1.16.0` (`grpo_trainer.py:1388`), `GRPOTrainer._prepare_inputs` runs `generation_batch = shuffle_sequence_dict(generation_batch)` on all $A B G$ rows before splitting into $A$ microbatches of size $B G$.
- Therefore, `SEAMActor` must **not** assume contiguous blocks of $G$ identical rows unless group-level shuffling is enforced.
- Deduplicating rows dynamically by `torch.unique(source_input_ids, dim=0, return_inverse=True)` inside `SEAMActor` works regardless of row order, and overriding `_prepare_inputs` to shuffle at the **group level** ($(A B, G)$ blocks) guarantees $U = B$ unique sources per microbatch instead of up to $\min(A B, B G)$.
