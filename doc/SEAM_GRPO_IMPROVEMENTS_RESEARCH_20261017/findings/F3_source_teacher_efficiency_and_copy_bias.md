# Finding F3 — `SourceTeacher` Copy-Gate Bias, `tanh` Saturation, and Reference Checkpoint Alignment

- **Thesis:** In `seam_length_grpo`, `SourceTeacher` (`teacher.py:42-96`) exhibits three concrete flaws when enabled (`source_weight > 0`): (1) `GroundedCopyHead` assigns $-3.4 \times 10^{38}$ copy log-prob on low-Jaccard negative sources $x^-$, turning `source_score` into an extractive copy-rate reward rather than an abstractive faithfulness signal; (2) $\tanh((\ell^+ - \ell^-)/\tau)$ with $\tau=1.0$ saturates ($\ge 0.987$) for typical $\Delta \ell \in [2.5, 4.0]$ nats/token; and (3) `train.py:102` initializes `reference` from `sft_checkpoint` *before* running `ce_warmup`, whereas `Lifecycle` evaluates against `length_ce_warmup.pt`.
- **Status:** Confirmed (Triangulated across Sources [13](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/13_local_reward_variance_audit.md), [14](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/14_local_source_teacher_audit.md), [16](file:///Users/kieugiangbien/Downloads/Project/LLM2Seq/doc/SEAM_GRPO_IMPROVEMENTS_RESEARCH_20261017/sources/16_local_systems_call_count_audit.md)).

## Evidence

1. **Extractive Copy-Gate Bias (`src/eviseq_new/eviseq_afmr/modeling/grounded_copy.py:211-232` & `teacher.py:67-89`):**
   - `Negatives` (`data.py:94-110`) selects negative documents $x^-$ from the lowest 25% Jaccard similarity pool (`jaccard <= 0.08`).
   - Any source entity or content word copied from $x^+$ via `GroundedCopyHead` is almost certainly absent from $x^-$, so `copy_log_probs` on $x^-$ equals `torch.finfo(dtype).min` (`-3.4e38`) and $P_{\text{final}}(y_t \mid x^-) = (1 - g_t) P_{\text{vocab}}(y_t \mid x^-)$.
   - Meanwhile on $x^+$, $P_{\text{final}}(y_t \mid x^+) = (1 - g_t) P_{\text{vocab}}(y_t \mid x^+) + g_t P_{\text{copy}}(y_t \mid x^+)$.
   - Thus $\ell(y \mid x^+) - \ell(y \mid x^-)$ grows monotonically with the fraction of verbatim copied tokens, penalizing legitimate abstractive compression required to hit tight length buckets (`short`).
   - **Fix:** Either evaluate `SourceTeacher` with vocab-only log-probs (or detach/cap the per-token copy contribution in $\ell^+ - \ell^-$ at $\le c_{\text{cap}}$ nats) and increase `source_temperature` $\tau$ from `1.0` to `3.0–4.0` (or group-normalize $\Delta \ell_i$ directly via GDPO instead of saturating with $\tanh$).

2. **Reference / Teacher Checkpoint Ordering (`src/seam_length_grpo/seam_length_grpo/train.py:102-119`):**
   - `train.py:102` loads `reference = load_actor(...)` from `cfg.model.sft_checkpoint` **before** `warmup_checkpoint = run_ce_warmup(...)` (`train.py:109`).
   - If `ce_warmup` runs during `train.py`, the actor begins GRPO from `length_ce_warmup.pt`, while `reference` (used for both KL `beta > 0` and `SourceTeacher`) remains at the pre-warmup `sft_checkpoint` (whose prefix MLP has not adapted to length tags!).
   - **Fix:** Load `reference` **after** `run_ce_warmup` from `warmup_checkpoint` so both `ref_model` and `SourceTeacher` condition properly on the warmed-up length prefix embeddings.

3. **Zero-Cost Reuse When `beta > 0`:**
   - Both `SEAMGRPOTrainer` (for `ref_per_token_logps` when `beta > 0`) and `SourceTeacher` use the exact same frozen `reference` model at `temperature=1.0`. Caching `true_source` log-probs from `ref_model` eliminates 50% of `SourceTeacher`'s forward passes.
