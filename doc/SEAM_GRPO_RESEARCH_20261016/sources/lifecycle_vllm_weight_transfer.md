# Source note: vLLM 0.25.1 weight transfer

**Source:** [vLLM 0.25.1 weight-transfer docs](https://docs.vllm.ai/en/v0.25.1/training/weight_transfer/) and [NCCL engine API/source excerpt](https://docs.vllm.ai/en/v0.25.1/api/vllm/distributed/weight_transfer/nccl_engine/). Accessed 2026-10-17. Primary official project docs; exact version is aligned to `SEAMLLM`'s local `vllm.__version__ == "0.25.1"` check (`src/eviseq_new/seam_vllm/api.py:67-68`).

**Short quote:** “Received weights are loaded via the model's `load_weights`” (NCCL engine docs).

**Relevant evidence:** The documented lifecycle is initialization, start, one or more update calls (including chunks), and finish. The 0.25.1 NCCL receiver's packed path uses `post_unpack_func=self.model.load_weights`; the unpacked path calls `self.model.load_weights([(name, weight)])` per parameter. Thus the custom model's loader is the runtime sync hook. The trainer-side API takes an iterator of `(name, tensor)` parameters.

**Application to SEAM:** This makes the static checkpoint restore in `SEAMForQwen3.load_weights` an online-sync blocker: the call writes incoming values, then the override reads and loads the fixed checkpoint again. It also means chunked transfer is not automatically transactional with respect to an active sampler. The runner must remain paused throughout all update chunks, and the current custom loader must be revised before relying on this path.

**Limits:** The exact 0.25.1 docs establish how the NCCL engine uses `load_weights`; APIs on current/latest docs may differ. The local SEAM adapter pins 0.25.1. This note does not assume that using the same parameter names as a plain causal LM is enough for custom encoder/bridge/copy modules.
