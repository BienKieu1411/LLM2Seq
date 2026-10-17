# Source note: TRL GRPO trainer fit and policy mismatch

**Source:** [TRL GRPO Trainer documentation](https://huggingface.co/docs/trl/grpo_trainer). Accessed 2026-10-17. Primary official framework docs; the page is mutable, so deployment must record the installed TRL version.

**Short quote:** “Only causal language models are supported.”

**Relevant evidence:** Current docs accept a model identifier, `PreTrainedModel`, or `PeftModel` and restrict model-object support to causal language models. The trainer offers `rollout_func`, but its loss still recomputes probabilities through the trainer model. The docs also explain vLLM/train numerical mismatch and default truncated importance-sampling correction, and expose sampler-vs-trainer log-probability difference metrics.

**Application to SEAM:** Local `EviSeqAFMR` is a custom `nn.Module` with a separate source encoder, AFMR bridge, custom causal decoder cross-attention, optional grounded-copy head, and bespoke generation. Supplying an external SEAM rollout function to a stock trainer does not replace the training policy forward, so it would score completions under an unrelated base causal LM unless a custom trainer/model/loss path is supplied. The repository now supplies that adapter in `SEAMActor` and `SEAMGRPOTrainer`, pinning the hooks to TRL 1.16.0 and rescoring through `actor.policy.seam`. TRL's importance correction can address a measured sampling-vs-training probability discrepancy; it cannot compensate for absent encoder/bridge/copy parameters, stale SEAM states, or the vLLM runtime checkpoint overwrite.

**Limit:** This is a statement about the current documented stock API, not a mathematical restriction against GRPO over encoder-decoder policies or an assertion that a custom subclass cannot integrate SEAM.
