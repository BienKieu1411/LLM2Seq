# Serve a local T5Gemma 2 checkpoint with vLLM

T5Gemma 2 is served as its own encoder-decoder model. It is not loaded through
AFMR's causal-decoder adapter and does not use AFMR's bridge or copy head.
vLLM 0.25.1 does not include a built-in `T5Gemma2ForConditionalGeneration`
implementation, so this path installs a separate community vLLM plugin pinned
to a source revision. The plugin's README reports development against a vLLM
nightly build; compatibility with the server's exact 0.25.1 wheel must be
confirmed by the smoke test below.

The model checkpoint remains local. The install script installs only the
plugin source with `--no-deps`, so it does not replace the existing CUDA,
PyTorch, Transformers, or vLLM packages. Run it inside the already-active vLLM
environment:

```bash
bash scripts/install_t5gemma2_vllm_plugin.sh
```

Start the server by passing the local model directory. The script checks its
`config.json`, enables offline model loading, and does not download a checkpoint:

```bash
bash scripts/serve_t5gemma2_vllm.sh \
  /workspace/path/to/t5gemma-2-1b-1b \
  --max-num-seqs 4
```

Set `PORT`, `HOST`, and `SERVED_MODEL_NAME` to change the OpenAI-compatible
endpoint. Additional arguments are passed directly to `vllm serve`. Use a
`--max-model-len` only when needed, and set it consistently with the model
config and available GPU memory.

Check that the model loads before sending a dataset:

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"t5gemma2","prompt":"Summarize the following article:\n\n...\n\nSummary:","max_tokens":128,"temperature":0}'
```

If plugin import or model startup fails under vLLM 0.25.1, keep the error log
and test against the plugin's documented nightly-compatible vLLM build. Do not
upgrade the production environment in place just to try the alternate version;
use a separate environment/container. This serving path is not evidence of
quality or throughput parity until outputs and latency are compared with the
existing T5Gemma 2 evaluator on the same local checkpoint and prompts.

The pinned plugin source is
[`d0rj/t5gemma2-vllm-plugin`](https://github.com/d0rj/t5gemma2-vllm-plugin).
