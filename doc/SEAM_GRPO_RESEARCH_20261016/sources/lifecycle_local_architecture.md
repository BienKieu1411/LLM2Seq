# Local source note: SEAM model and runtime state

Snapshot: 2026-10-17. This is a source inventory, not a claim about all possible EviSeq configurations.

## Files checked

- `src/eviseq_new/eviseq_afmr/modeling/model.py:16-112`: `EviSeqAFMR` owns encoder, decoder, bridge; encodes source, prompt embeddings, bridge memory, bias, optional value/copy state; forwards to custom decoder.
- `modeling/encoder.py:129-194`: depth hooks capture intermediate encoder layers; output includes final state/taps and source masks.
- `modeling/controller.py:14-41`, `modeling/afmr.py:223-272`: controller depends on final source, prompt embeddings, output budget; bridge computes memory, source bias, optional value memory.
- `modeling/grounded_copy.py:12-103`: copy state contains keys/token IDs/mask/bias; keys and bias depend on bridge outputs, copy head weights, embedding and alignment.
- `modeling/decoder.py:88-175, 302-523`: custom decoder cross-attention, cache methods, logits/copy mixture and teacher-forced CE.
- `src/eviseq_new/seam_vllm/model.py:61-175, 234-313, 335-450, 468-636`: vLLM decoder/cross attention and parameter names, load mapping, model-side runtime caches, source encode/pack/copy-state construction and `load_weights` behavior.
- `seam_vllm/state.py:14-120`, `_copy_batch.py:36-55`, `_copy_logits.py:9-68`: request-lifetime copy states and identity/order-dependent derived caches.
- `seam_vllm/api.py:29-108, 135-181`: static config/checkpoint, exact vLLM pin, serving setup, generate-only facade.
- `src/seam_grpo/seam_grpo/train.py`: SFT actor/reference initialization, TRL config, tokenizer EOS alignment, and adapter construction.
- `src/seam_grpo/seam_grpo/actor.py`: generic HF config/model wrapper, `actor.policy.seam` nesting, candidate/CE chunking, detached entropy, and eval-mode SEAM tree.
- `src/seam_grpo/seam_grpo/trainer.py`: pinned TRL subclass, source-context propagation, native rollout callback, action likelihood override, CE normalization, and policy-version checks.
- `src/seam_grpo/seam_grpo/policy.py`, `rollout.py`: full-mixture differentiable scoring and native full-support rollout.
- `eviseq_afmr/training/checkpoint.py:132-172`: loading a model checkpoint does not load optimizer state unless an optimizer is supplied.

## Parameter-to-state route

`encoder.*` produces final (H_0) and depth taps. `bridge.*` plus (H_0)/taps, decoder prompt embeddings, masks, and output budget produce controller (c), memory (M), source bias (b), and optionally value memory (V). The vLLM adapter packs (M), optional (V), and (b) together into encoder outputs; it separately builds per-request `CopyState`. Decoder causal KV depends on decoder token embedding/self-attention and token history. Cross-attention K/V depends on bridge memory/value/bias and custom cross-attention projection/norm parameters. Copy state keys/bias depend on memory/value, source bias, token embedding, alignment and copy projections. `CopyBatchCache`/`CopyLogitsMixer` cache batch/scatter structures derived from the ordered request copy states.

The locally observed vLLM/native parameter namespace conversion is not a one-to-one name identity: native `decoder.backbone.*` (including `.base.` decoder layers and custom `.cross*`) must become vLLM `model.*`; native `decoder.lm_head.*` becomes `lm_head.*`; native `decoder.grounded_copy.*` becomes adapter `copy_head.*`. Tied embeddings alias vLLM `lm_head` to `model.embed_tokens`; the untied case has a separate output parameter. The startup routine explicitly handles these translations, but the runtime loader does not reproduce the whole startup routine.

## Direct local update blocker

`SEAMForQwen3.load_weights` calls the generic `AutoWeightsLoader`, then restores `self.seam_checkpoint_path` (the static seed checkpoint). vLLM 0.25.1's NCCL receiver calls model `load_weights` on incoming weights, including separate calls/chunks. Therefore the seed restore overwrites online actor updates. `SEAMLLM` has no public weight-update/sync or policy-version method. These observations are based on the code snapshot only; they do not mean a custom runtime loader cannot be written.

## DDP and reference notes

`train.py` loads the actor's SFT weights with `load_checkpoint(sft_path, seam, config=..., restore_rng=False)` and does not pass an optimizer; TRL then constructs the optimizer for `actor.policy.seam`. When `beta` is nonzero, it separately loads another full `SEAMActor` from the SFT checkpoint and freezes it as the reference. The accelerator wraps the actor, while rollout receives the unwrapped inner `actor.policy.seam`. A hypothetical inference weight source must unwrap `DDP -> SEAMActor -> Policy -> seam`; its state-dict prefix is `module.policy.seam.` under DDP and `policy.seam.` without DDP. Keep the SFT reference separate and never send actor updates to it.
