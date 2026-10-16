# SEAM HTTP through native vLLM 0.25.1

The HTTP path runs the registered SEAM model inside the native `vllm serve`
process. It does not host the synchronous `SEAMLLM.generate()` facade behind a
separate HTTP queue. The model still runs encoder, bridge, decoder
cross-attention, and checkpoint-configured grounded copy before sampling.

This implementation targets vLLM **0.25.1**, a **Qwen3 causal decoder**, and
**one GPU**. The encoder uses the existing AFMR encoder adapter. Other decoder
families are not implemented by this plugin.

## Start from source, without installing the plugin

From the repository root, in the environment already containing vLLM 0.25.1:

```bash
CUDA_VISIBLE_DEVICES=0 bash src/eviseq_new/scripts/serve_seam_vllm.sh \
  --config /absolute/path/resolved_config.yaml \
  --checkpoint /absolute/path/best.pt \
  --served-model-name seam \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90
```

The config must point to the local encoder and decoder directories used for the
checkpoint. The launcher does not change the config or download weights.
The source folders `seam_vllm` and `eviseq_afmr` must be present alongside each
other. No `pip install` is required: the launcher exposes a temporary plugin
entry point and propagates its source/metadata paths to vLLM worker processes.
It does not modify vLLM or site-packages. Existing vLLM/PyTorch/Transformers
dependencies must already be available in the active environment.

Installing with `scripts/install_seam_vllm_plugin.sh` remains optional if the
`seam-serve` console command or bare native vLLM commands are wanted.

Use `--api-key YOUR_KEY`, or the native `VLLM_API_KEY` environment variable, for
vLLM authentication. The launcher also accepts `--max-model-len` for decoder
context and `--max-num-batched-tokens` for the native scheduling budget. The
encoder source cap comes from `data.max_source_length` in the resolved config.
`max-num-seqs` controls concurrent sequences, rather than a fixed batch size.
Start conservatively and measure GPU memory with the actual document lengths.

`--dry-run` prints the native engine arguments without loading weights
or starting a server (local config/tokenizer files are read). API keys are
redacted in this printout. In source-only mode, keep using the SEAM launcher
to prepare worker plugin discovery rather than copying that raw command.
Internally the launcher executes the same CLI entry
point as `vllm serve`, with the SEAM model overrides and:

```bash
--middleware seam_vllm.http.SEAMMiddleware
```

This middleware is opt-in and only adapts this server instance's native serving
handlers; it does not edit the installed vLLM library. Native HTTP routes handle
model validation, authentication, request cancellation, AsyncLLM scheduling,
sampling, SSE streaming, usage serialization, and normal responses.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
response = client.completions.create(
    model="seam",
    prompt="Văn bản nguồn cần tóm tắt...",
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    extra_body={
        "top_k": 0,
        "max_input_tokens": 3072,
        "encoder_prompt": "Văn bản cần tóm tắt:\n",
        "decoder_prompt": "Tóm tắt nội dung chính bằng tiếng Việt.",
    },
)
print(response.choices[0].text)
```

For a batch with the same settings, `prompt` can be a list of source strings.
For independent per-document settings, send independent concurrent requests;
the native engine batches them. `AsyncOpenAI` works with the same API contract.

Chat completions expose the same summarization operation. Supply one user
message containing the source, optionally preceded by one system message. A
system message replaces the decoder instruction and is also sent to the bridge:

```python
response = client.chat.completions.create(
    model="seam",
    messages=[
        {"role": "system", "content": "Tóm tắt ngắn gọn bằng tiếng Việt."},
        {"role": "user", "content": "Văn bản nguồn..."},
    ],
    max_completion_tokens=512,
    temperature=0.0,
    extra_body={"top_k": 0, "top_p": 1.0, "max_input_tokens": 3072},
)
print(response.choices[0].message.content)
```

A user message alone uses the decoder instruction in the resolved config.
Text content blocks are accepted. Multi-turn conversations, tools, and image or
audio messages are rejected: this model summarizes a source document. Do not
pass both a system message and `decoder_prompt`.

## Custom inference settings

The following custom fields can be passed through `extra_body` on either route:

- `encoder_prompt`: replace the encoder prefix text before the source. The
  configured encoder task marker is still applied, without duplication.
- `decoder_prompt`: replace the decoder instruction, using the configured
  decoder chat-template setting by default.
- `decoder_prefix`: replace the decoder continuation prefix after the instruction.
- `decoder_chat_template`: boolean override for instruction tokenization; defaults
  to the checkpoint config's setting.
- `max_input_tokens`: encoder token cap, including prefix, task marker, and special
  tokens. It must be positive and no larger than the server config's source cap.

The native `max_tokens` (or chat `max_completion_tokens`) is the output budget.
It excludes the decoder prompt. Actual output can stop earlier at EOS or a stop
condition. The budget is capped by the decoder context available after its
prompt; that same effective budget is passed to the bridge. Encoder source
length does not consume decoder context. Increasing the server's source cap
requires restarting with the corresponding config and sufficient memory.

`temperature`, `top_p`, `top_k`, `min_p`, `seed`, repetition/presence/frequency
penalties, stop strings/IDs, output logprobs, and other native request fields
remain handled by vLLM. No custom sampler emulates unavailable constraints.
Prompt logprobs, echo, beam search, decoder prompt truncation, and arbitrary
prompt embeddings are rejected. Unsupported custom fields are rejected instead
of silently ignored. Use `max_input_tokens` rather than `truncate_prompt_tokens`.

Absent request settings use the run's temperature/top-p/top-k and min/max output
limits, with the SEAM vLLM defaults: no no-repeat-ngram constraint and neutral
repetition penalty `1.0`. An explicit native repetition penalty includes decoder
prompt tokens. A positive explicitly supplied temperature enables sampling.

All prompt fields default to the resolved config when omitted. Custom decoder
prompt IDs are sent to both decoder and bridge, with padding masks when requests
have different prompt lengths. Source/prompt/budget combinations participate in
multimodal input hashing. Architecture dimensions, gates, weights, and whether a
copy module exists remain fixed by the checkpoint. Changing inference prompts
can change quality; preserving the trained defaults is the initial parity case.

The Python facade accepts these same prompt fields:

```python
summary = llm.summarize(
    "Source ...",
    encoder_prompt="Document: ",
    decoder_prompt="Write a brief summary.",
    max_input_tokens=3072,
    max_tokens=512,
    temperature=0.0,
)
```

## curl and streaming

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models

curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "seam",
    "prompt": "Văn bản nguồn cần tóm tắt...",
    "max_tokens": 512,
    "max_input_tokens": 3072,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "stream": true
  }'
```

For an authenticated server add `-H 'Authorization: Bearer YOUR_KEY'` and set the
same key in the SDK. Set `stream=True` for either SDK route and iterate the
returned events. Native usage counts follow the decoder token stream; they are
not a combined encoder-plus-decoder compute-cost measure.

## Boundaries and verification

The implemented public generation routes are `/v1/completions` and
`/v1/chat/completions`. `/v1/responses`, Anthropic messages, render-only servers,
tool execution, and arbitrary multimodal inputs are not part of this adapter.
Native `/health`, `/v1/models`, and metrics remain available.

The adapter uses paged attention and native continuous batching. Source
processing uses the native renderer's async thread pool; default decoder prompt
IDs are cached in the HTTP handler. CUDA graphs, prefix caching, processor
caching, quantization, speculative decoding, and TP/PP are disabled by this
launcher to preserve the currently supported request-state contract. The HF
encoder, eager cross route, and per-step copy-state packing remain costs that
need GPU measurement.

Local verification checks parameter forwarding, separate budgets, dynamic bridge
prompt masks, concurrent request isolation, native command construction, errors,
and unbuffered ASGI event forwarding. The Mac has no vLLM/CUDA runtime; these
CPU tests do not prove GPU logits, exact greedy outputs, actual SSE generation,
or throughput. The optional `test_seam_vllm_processor.py` module exercises real
vLLM protocol/builders when run in the server environment. Validate model load,
non-streaming/streaming outputs, concurrent requests, and quality against the
normal AFMR evaluator before production use.

The extension points were checked against the pinned upstream source:
[vLLM middleware](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/api_server.py),
[native completion serving](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/completion/serving.py),
[native chat serving](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/entrypoints/openai/chat_completion/serving.py),
[encoder-decoder inputs](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/inputs/engine.py).

## Vietnamese ROUGE evaluation through HTTP

Update the plugin files and restart the host before using the HTTP evaluator.
The current host provides `/v1/seam/info` and an `x-seam-fingerprint` header.
This fingerprint covers the resolved config, checkpoint path/size/mtime,
decoder context, and adapter code. It does not hash the full checkpoint weights.
The metadata route uses native `/v1` authentication.

To start a temporary host, predict and score in one command from the repository
root (in the environment with vLLM 0.25.1; plugin installation is not needed):

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate_rouge_vietnamese_vllm.py \
  --start-server \
  --afmr-config /absolute/path/resolved_config.yaml \
  --afmr-checkpoint /absolute/path/epoch_003.pt \
  --model-name seam_vi_vllm_003 \
  --concurrency 16 \
  --gpu-memory-utilization 0.90 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --allow-missing
```

`--start-server` (alias `--start-vllm-service`) launches the native SEAM host
on `127.0.0.1:8000`, waits for its metadata/model routes, verifies the checkpoint
identity, then sends concurrent requests. The host uses the evaluator's Python
environment and inherits `CUDA_VISIBLE_DEVICES`; no model downloads are allowed.
It is stopped after prediction, before ROUGE, including prediction errors and
Ctrl+C. Only the subprocess group created by this evaluator is stopped.

The host log is `<model output>/seam_vllm_server.log`. Use `--server-log` to
choose another path, `--server-port` to change the port, or `--startup-timeout`
to change the default 1200-second loading timeout. A busy port is rejected;
an existing host is never stopped or silently reused. `--max-num-seqs` defaults
to `--concurrency` for this host. Input/output limits and prompts default to
the supplied config; request overrides follow the same HTTP rules below.

With the server already running, omit `--start-server` and local model paths:

```bash
python3 src/evaluation/evaluate_rouge_vietnamese_vllm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model seam \
  --model-name seam_vi_http \
  --concurrency 16 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0
```

`--model` must match the host's `--served-model-name`. `--model-name` names the
local prediction/report folder; it can differ from the served alias. Set
`--api-key` (or `VLLM_API_KEY`/`OPENAI_API_KEY`) when required by the server.
When connecting to an existing host, the evaluator does not load weights or
require a local GPU or local checkpoint/config. It uses the server's prompt and
input/output token defaults.

Optional request overrides are `--encoder-prompt`, `--decoder-prompt`,
`--decoder-prefix`, `--decoder-chat-template` / `--no-decoder-chat-template`,
`--max-input-tokens`, `--max-new-tokens` (alias `--max-tokens`), and the supported
sampling flags listed by `--help`. The encoder cap cannot exceed the host cap.
Configure `max-num-seqs` and GPU memory on the host; `--concurrency` (alias
`--batch-size`) is the number of independent HTTP requests in flight.

Source/reference/output roots retain the historical defaults under
`/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval`. Override them with
`--data-raw-root`, `--data-ref-root`, and `--output-root`. Predictions are written
to `data_sum/<model-name>/<dataset>/<file>.txt` as soon as each response arrives.
The report is `data_sum/<model-name>/rouge_vllm_summary.json`. All seven datasets,
empty/hidden-file handling, multiple-reference parsing, and ROUGE-1.5.5 flags
remain those of `evaluate_rouge_vietnamese.py`.

Rerunning the same command resumes missing predictions. Fingerprint and request
parameters must match the saved manifest. Each completion's fingerprint header
is checked before writing, so a host restart onto another checkpoint cannot
silently mix predictions. For a different model/prompt/checkpoint use a new
`--model-name` or explicitly pass `--overwrite-predictions`. Successful in-flight
responses are saved even if another request fails. Transient HTTP errors have
bounded retries (`--retries`, default 2); `--timeout` defaults to 600 seconds.

The previous in-process evaluator remains available with `--backend local`,
`--afmr-config`, and `--afmr-checkpoint`. Supplying those local paths without a
base URL also selects local mode, preserving earlier commands. The original
non-vLLM evaluator is unchanged.
