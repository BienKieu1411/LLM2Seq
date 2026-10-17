# Local implementation evidence

- Accessed: 2026-10-17.
- Type: user-provided scripts and current workspace implementation.
- Credibility: direct evidence of this code snapshot; not evidence of model quality. Recency: current. Bias: no independent experimental replication.
- Fingerprints: `../checks/result.json` for the six supplied scripts; `../checks/workspace_fingerprints.json` for the inspected integration points.

User-supplied files: `lafrt.py`, `lead_eigen.py`, `lead_eigen_threshold.py`, `lead_len_control.py`, `semantic_lafrt_len_balanced.py`, `schedule_running.py` in `/Users/kieugiangbien/Downloads`.

Verified local excerpt from `lead_len_control.py`: `if semantic_reward <= 0.0: final_reward = semantic_reward` (line break normalized).

Current repository paths:

- `src/evaluation/evaluate_length_vietnamese.py`: NFKC Unicode word counter, MAE, signed/absolute relative deviation, asymmetric length score; counts the entire reference TXT.
- `src/eviseq_new/eviseq_afmr/modeling/controller.py`: source, prompt and token-budget control.
- `src/eviseq_new/eviseq_afmr/modeling/model.py`: encoder/bridge path into decoder and copy.
- `src/eviseq_new/eviseq_afmr/data/schema.py`: no requested word field in CanonicalRecord.
- `src/eviseq_new/eviseq_afmr/data/collate.py`: fixed configured decoder prompt, per-record source/target construction.
- `src/seam_grpo/seam_grpo/{trainer,data,rollout,vllm_rollout}.py`: group identity, score conditioning, fixed-width native prompts and rollout metadata.
- `src/eviseq_new/seam_vllm/{_request,api,processing}.py`: source/prompt/budget metadata, no explicit requested-word interface.

The supplied-code audit and the design distinguish observations from proposed changes. No claim here establishes better ROUGE, length adherence or factual consistency after training.
