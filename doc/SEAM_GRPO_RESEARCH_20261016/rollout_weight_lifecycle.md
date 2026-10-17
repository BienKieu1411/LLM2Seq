# SEAM rollout weight lifecycle: TRL and separated vLLM

Historical audit before the runtime-sync implementation. The blocker identified
here motivated the new initialization-only loader and full-actor transaction.
Current code/status: [vLLM training audit](2026-10-17_vllm_training_audit.md).

Research snapshot: 2026-10-17. Scope: local EviSeq/SEAM model and vLLM adapter, plus official TRL, vLLM 0.25.1, VERL, and GRPO sources. This is an integration audit, not a proposal for a new GRPO algorithm.

## Finding

The implemented in-process path uses TRL 1.16.0's `GRPOTrainer` loop through `SEAMGRPOTrainer`, while native rollout and differentiable action scoring use the same full actor tree at `SEAMActor.policy.seam` (`src/seam_grpo/seam_grpo/{actor,policy,rollout,trainer,train}.py`). There is no trainer-to-rollout weight transfer: rollout receives that same actor model in-process, and TRL's optimizer updates it after the configured generation/accumulation window. The vLLM serving adapter is a separate model instance and has no compatible update surface. A future vLLM transfer path must move every changed trainable tensor through an explicit name/shape mapping, then invalidate all request-derived SEAM state before new generations.

There is a concrete blocker in the adapter as currently written. `SEAMForQwen3.load_weights()` first applies received tensors and then calls `_restore_afmr_checkpoint(self.seam_checkpoint_path)` (`src/eviseq_new/seam_vllm/model.py:630-635`). Official vLLM 0.25.1 NCCL transfer routes each received chunk through this model `load_weights` method (`[NCCL API](https://docs.vllm.ai/en/v0.25.1/api/vllm/distributed/weight_transfer/nccl_engine/)`). The restore reads the static checkpoint's `model` state and recopies its encoder, bridge, copy, decoder, and cross-attention tensors (`model.py:335-450`). Therefore a normal runtime update would be overwritten by the configured checkpoint; chunked updates could repeatedly restore it. This fixed-checkpoint restore is appropriate for initialization, but it is incompatible with online actor sync unless the runtime load path is changed to retain incoming actor tensors and not restore the seed checkpoint.

Separately, the transfer keys need conversion. The native model uses names such as `decoder.backbone.layers.N.base.*`, `decoder.backbone.layers.N.cross.*`, `decoder.lm_head.*`, and `decoder.grounded_copy.*`; vLLM uses `model.layers.N.*`, `lm_head.*`, and root `copy_head.*`. Startup restore explicitly translates/loads these groups (`model.py:380-450`), but runtime `load_weights()` only invokes `AutoWeightsLoader` before the static restore. The current source is nested under `SEAMActor.policy.seam`, so a raw actor state dict adds `policy.seam.` (or `module.policy.seam.` under DDP). A standalone `Policy` source would have `seam.` (or `module.seam.`), but that is not the active trainer wrapper. None of these prefixes is the vLLM namespace or a substitute for the startup conversion; a dedicated source/name converter and complete-parameter audit are required.

## Exact action probability

Let (h_t) be the decoder state at the position predicting sampled token (a_t), and (z_t=W_{lm}h_t) the generator logits. With grounded copy, `mix_logits` returns the log-mixture plus the generator log-normalizer offset:

\[
m_t(v)=\log\!\left[\exp(\log\operatorname{softmax}(z_t)_v+\log g_t)
 + \exp(\log c_t(v)+\log q_t)\right]+\log\sum_u e^{z_t(u)}.
\]

Here (g_t) and (q_t) are generate/copy gate probabilities, and (c_t(v)) is copy attention mass aggregated over source positions aligned to vocabulary token (v). The final additive offset is constant across vocabulary entries. `Policy` and `SEAMActor` use the action distribution

\[
\pi_{\theta,\tau}(a_t\mid x,p,a_{<t})
=\operatorname{softmax}\!\left(m_t/\tau\right)[a_t],\qquad
\log\pi=\operatorname{log\_softmax}(m_t/\tau)[a_t].
\]

For no-copy models, (m_t=z_t). Temperature is applied to the *already mixed* logits; `log_softmax` follows the mixture (`src/seam_grpo/seam_grpo/policy.py`, `rollout.py`). Do not separately temperature-scale the generator and copy distributions, or score with plain LM logits: either changes the behavior policy. Store the generated action's behavior log-probability under this exact distribution and compute the GRPO ratio (r_t=\exp(\log\pi_{\theta}(a_t)-\log\pi_{old}(a_t))). The native sampler uses full-support temperature-only sampling. If another sampler applies top-p/top-k or other logits processors, those transformations must also be reflected in behavior probabilities or corrected as an explicit train/inference mismatch.

## Dependency map: parameters to derived state

| Trainable parameters / fixed inputs | Derived value | Cached inference state that becomes stale after a parameter update |
|---|---|---|
| `encoder.*` parameters; source IDs, attention/content masks | Final encoder states (H_0), intermediate depth taps (H_1...H_k) | vLLM encoder output cache stores the final packed bridge outputs rather than taps, but any cached encoder result was produced by the old encoder. Re-run encoding or invalidate it. `encoder.py:129-194`; `model.py:522-580`. |
| `bridge.*` (controller, projection, depth/feature residuals, focus prior); (H_0)/taps; decoder prompt embeddings; output budget | Controller (c), memory (M), source bias (b), optional value memory (V) | Packed encoder output contains (M), optional (V), and (b) (`_tensors.py:10-29`). These are not reusable after bridge or input changes. The controller explicitly consumes source pool, prompt embedding pool, and output-budget scalars (`controller.py:26-41`). |
| Decoder token embedding (also an LM head when tied); configured prompt token IDs | Prompt embeddings used by the bridge/controller; lexical embeddings used by copy | Prompt IDs are a non-persistent buffer (`model.py:308`), but their embeddings come from trainable `model.embed_tokens`; unchanged IDs therefore do not make bridge output invariant. Copy lexical keys also depend on this embedding (`model.py:524-541`, `grounded_copy.py:74-78`). |
| Decoder base self-attention, MLP, norms, embedding, positional/model config | Decoder hidden states; autoregressive causal KV for prior tokens | Causal KV is history-dependent and invalid after relevant decoder parameters change. New requests must start with current weights; in-flight sequences must not silently continue as one old-policy trajectory. |
| Cross-attention `cross_norm`, `cross.{q,k,v,o}_proj`, `q_norm`, `k_norm`, `memory_norm`, `cross_gate`; (M,V,b) | Cross-attention results; cross-attention K/V derived from memory/value-memory and source bias | vLLM cross-attention pages contain request-specific encoder K/V. A parameter update can invalidate them even when decoder self KV is cleared. Keys also encode `source_bias` in the padded key coordinates (`model.py:61-135`, `168-175`). |
| Copy head `query`, `context_key`, `lexical_key`, `gate`; (M) or (V), (b), copy token IDs/mask/alignment | Copy key bank, aggregated copy bias, per-step copy attention/gate, final mixed logits | `CopyState` lives per request in `SEAMModelState.copy_states`; the one-entry copy batch and destination-scatter caches are additional derived state (`state.py:25, 72-120`; `_copy_batch.py:36-55`; `_copy_logits.py:22-68`). All request copy state must be rebuilt after relevant changes. |
| LM head; tied-embedding setting | Generator logits and generator component of the mixture | Tied head aliases `model.embed_tokens`; sync the single canonical shared tensor and preserve aliasing. With untied head, `lm_head.weight` is separate. Startup mapping skips root `lm_head` when tied (`model.py:250-258, 396-397`; `_weights.py:8-47`). |

Non-parameter state: `decoder_prompt_ids` is a non-persistent buffer; RoPE/inverse-frequency buffers are configuration-derived; masks, source text/alignment, tokenizer IDs, and output budget are request data. They are not actor weights, but any data/config/tokenizer change can still change policy inputs or output semantics. vLLM's weight checker only hashes parameters and does not verify buffers, copy states, or KV caches ([weight checker docs](https://docs.vllm.ai/en/latest/training/weight_checker/)).

## Update semantics and safe boundary

**TRL-backed shared-model route.** `train.py` constructs `EviSeqAFMR` from the resolved SFT config, loads SFT weights with `restore_rng=False` and no optimizer, then wraps the model in `SEAMActor`. `SEAMGRPOTrainer` subclasses the pinned `trl.GRPOTrainer` and uses its `Trainer` loop, optimizer/scheduler lifecycle, group rewards/advantages, and GRPO objective. Its `_native_rollout` passes `actor.policy.seam` to the native sampler; its generation hook carries source/copy context alongside TRL's completion rows; its likelihood hook calls `SEAMActor.forward` for full-mixture action scores and entropy; and its `compute_loss` adds the globally token-normalized gold CE anchor. `SEAMActor.train()` preserves the Trainer-visible root mode but keeps the nested SEAM model in eval mode so rollout and scoring share dropout-free behavior. Candidate rows are chunked inside one actor forward with non-reentrant checkpointing. On supported DDP runs, accelerator wrapping synchronizes the full actor; rollout uses that same local actor, so there is no separate inference model to update. When KL is enabled, `train.py` builds another complete actor from the SFT checkpoint, freezes it, and keeps it as the reference; actor updates must never reach that reference.

Each rollout group records the trainer's current global step as its policy version. The callback checks the version before and after sampling, and the loss hook rejects buffered rows from a different trainer step. This preserves a complete group under one actor snapshot and prevents using a sampled trajectory after an optimizer update.

**Separate vLLM route.** The local `SEAMLLM` facade constructs a persistent vLLM 0.25.1 engine from a resolved config and fixed checkpoint, sets TP=PP=1, disables prefix caching, and exposes generation but no `sync_weights`, update, or version API (`src/eviseq_new/seam_vllm/api.py:29-108, 135-181`). Synchronous TRL's vLLM path synchronizes weights after a training-step change; colocated mode still uses a separate inference model and thus does not imply shared parameters. Server mode has an explicit training-to-server transfer. The generic vLLM contract is init → start → one or more update calls → finish (`[vLLM 0.25.1 weight transfer](https://docs.vllm.ai/en/v0.25.1/training/weight_transfer/)`). The update must contain every changed actor parameter, with all SEAM custom modules represented. Do not infer that syncing only the causal decoder is sufficient.

For a strict synchronous rollout cycle: (1) generate a complete candidate group under actor version (k) and save token IDs, exact old log-probs, temperature, output budget, and version; (2) score/update the trainer actor; (3) stop new requests and wait for old requests to finish or abort/discard them; (4) transfer the complete new parameter set as one paused update transaction; (5) verify parameter coverage/version and clear encoder outputs, self KV, cross-attention KV, request copy states, batch/mixer caches, and any prefix cache; (6) resume with version (k+1). A transfer may be chunked for memory, but generation must remain paused until the entire actor is installed. Do not sample from partially replaced weights.

An asynchronous GRPO design may deliberately train on *older but internally consistent* rollouts if it stores their actual behavior log-probabilities and constrains staleness/importance ratios. It must still keep every sequence on one model version. vLLM's documented `mode="keep"` can resume an in-flight request with newly installed weights; even with decoder KV clearing the completion is split across policies, so it is unsuitable as one ordinary GRPO trajectory. With cache clearing disabled, the decoder also consumes stale self KV. SEAM has an extra risk: keeping the request can preserve old encoder memory and `CopyState`, neither cleared by decoder KV invalidation. Use wait/abort for this architecture, or hold a versioned immutable engine/model state per in-flight request and recompute all derived states for that version. Current SEAM adapter exposes no such multi-version model-state facility.

If a frozen reference is needed (β > 0), keep a distinct SFT reference actor or engine at the SFT version. The current TRL adapter builds a second `SEAMActor` from the SFT checkpoint and freezes it; `SEAMGRPOTrainer` supplies it to the base KL scorer. In-process DDP synchronization targets only the trainable actor. vLLM's ability to preserve selected frozen parameter names is not an automatic filter: a future sender must omit reference weights and runtime reload must preserve their values and storage ([sleep mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode/)).

## TRL and VERL fit

TRL's documented stock model path targets causal language models ([TRL GRPO docs](https://huggingface.co/docs/trl/grpo_trainer#grpotrainer)). `EviSeqAFMR` remains a custom `nn.Module`; the implemented adapter supplies a generic `PretrainedConfig` and `PreTrainedModel` wrapper (`SEAMActor`) whose forward scores through the full SEAM policy. `SEAMGRPOTrainer` replaces the stock rollout and likelihood hooks while delegating the optimizer loop and GRPO objective to TRL. The native `rollout_func` alone would not make stock scoring pass through SEAM's encoder/bridge/copy graph; the model-forward adapter is essential. This is architecture integration around an established objective, not a new GRPO algorithm.

TRL's asynchronous path documents `weight_sync_steps`, `max_staleness`, and old-policy samples; that can guide a custom engine, but the current API requirements (vLLM, Transformers >= 5.2, FSDP2 for distributed training) and standard causal model assumption do not establish compatibility with this adapter. VERL gives a useful integration pattern: synchronous colocated rollout updates directly from the model engine; disaggregated async rollout sends through a checkpoint engine (`[VERL ActorRolloutRefWorker](https://github.com/verl-project/verl/blob/main/verl/workers/engine_workers.py)`). It is a weight-transport pattern, not evidence the current SEAM vLLM module can load such updates.

**Novelty boundary.** Replacing a causal-LM distribution with the exact SEAM encoder-conditioned copy-mixture distribution is an architecture integration requirement. Applying established GRPO to sampled SEAM actions does not by itself create a new GRPO algorithm. Any claim of algorithmic novelty would need to specify a changed objective, estimator, sampling law, or update rule, separate from full-actor weight synchronization and state invalidation.

## Checks and unresolved items

- Local trace covers model constructors, runtime encoder/bridge/copy preparation, model loading, per-request state, the TRL subclass hooks, and actor/rollout call order. No pretrained model was downloaded. Offline checks use the repository's tiny local configuration. All 21 checks passed, including actor log-probability/gradient parity, actual TRL integration, and two-process CPU/Gloo with accumulation and CE enabled/disabled. The lifecycle callback uses a Gloo-compatible CPU barrier after a Mac-specific MPS dispatch failure. CUDA/NCCL remains unverified.
- Official vLLM 0.25.1 docs confirm incoming NCCL tensors route through model `load_weights`; this version matches the local adapter's explicit vLLM pin. Latest vLLM docs add APIs and semantics beyond 0.25.1, so use the pinned API, not the latest endpoint names, when implementing.
- Before any server-side GRPO implementation, prove: (a) every actor parameter is mapped exactly once, (b) tied embedding aliasing is preserved, (c) after sync the vLLM parameter checksums match the intended actor tensors, (d) fresh-source/token action log-probabilities match native scoring within tolerance, and (e) no old request/cache state survives a version switch. The vLLM parameter checksum endpoint alone cannot prove (d) or (e).
- The current vLLM adapter's static-checkpoint reload and lack of a public update API are enough to reject direct online sync today; they do not imply online weight updates are impossible after an adapter change.

## Sources and supporting notes

- [Local architecture / caches](sources/lifecycle_local_architecture.md)
- [vLLM 0.25.1 weight transfer](sources/lifecycle_vllm_weight_transfer.md)
- [vLLM async requests and caches](sources/lifecycle_vllm_async_cache.md)
- [TRL GRPO compatibility](sources/lifecycle_trl_grpo.md)
- [TRL asynchronous GRPO](sources/lifecycle_trl_async.md)
- [VERL actor/rollout update patterns](sources/lifecycle_verl.md)
- [GRPO objective source](sources/lifecycle_grpo_paper.md)
- [Weight-update verification limits](sources/lifecycle_vllm_weight_checker.md)
