"""SEAM input adaptation inside the native vLLM 0.25.1 HTTP server.

The native serving classes retain authentication, async scheduling, cancellation,
sampling, SSE streaming and response serialization. Only request rendering and
decoder context accounting differ from the decoder-only serving path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eviseq_afmr.config import load_config

from ._prompt import configured_decoder_prompt_ids
from ._request import request_prompt_ids, source_prompt
from ._sampling import sampling_kwargs, validate_sampling_params


def _text_content(content: Any) -> str:
    if (
        isinstance(content, list)
        and content
        and all(
            isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            for part in content
        )
    ):
        content = "\n".join(part["text"] for part in content)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("SEAM expects non-empty text, without image/audio/tool content")
    return content


def _chat_source(messages) -> tuple[str, str | None]:
    system = None
    if messages and messages[0].get("role") == "system":
        system = _text_content(messages[0].get("content"))
        messages = messages[1:]
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("SEAM summarizes one user source, optionally preceded by one system instruction")
    return _text_content(messages[0].get("content")), system


def server_info(hf_config, config, *, max_model_len: int | None = None) -> dict[str, Any]:
    """Freeze inexpensive run provenance once, without re-reading model weights."""
    checkpoint = Path(hf_config.seam_checkpoint_path).expanduser().resolve()
    stat = checkpoint.stat()
    identity = {"path": str(checkpoint), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    adapter = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        adapter.update(path.name.encode())
        adapter.update(path.read_bytes())
    payload = {
        "config": config,
        "checkpoint": identity,
        "adapter": adapter.hexdigest(),
        "vllm": "0.25.1",
        "decoder_context": max_model_len,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    data = config["data"]
    return {
        "implementation": "SEAMForQwen3",
        "vllm_version": "0.25.1",
        "fingerprint": fingerprint,
        "checkpoint": identity,
        "generation_defaults": sampling_kwargs(config["generation"], {}),
        "max_input_tokens": int(data["max_source_length"]),
        "decoder_context": max_model_len,
        "prompt_defaults": {
            "encoder_prompt": data.get("encoder_prefix", ""),
            "encoder_task_marker": data.get("encoder_task_marker", ""),
            "decoder_prompt": data.get("decoder_prompt", ""),
            "decoder_prefix": data.get("decoder_prefix", ""),
            "decoder_chat_template": bool(data.get("decoder_chat_template", False)),
        },
    }


class _SEAMOnlineRenderer:
    def __init__(self, handler, original, config):
        self.handler = handler
        self.original = original
        self.config = config
        self.default_prompt_ids = configured_decoder_prompt_ids(original.renderer.get_tokenizer(), config["data"])

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _prepare(self, request, system_prompt=None):
        custom = dict(getattr(request, "model_extra", None) or {})
        allowed = {
            "encoder_prompt",
            "decoder_prompt",
            "decoder_prefix",
            "decoder_chat_template",
            "max_input_tokens",
            "mm_processor_kwargs",
        }
        if unknown := custom.keys() - allowed:
            raise ValueError(f"Unsupported SEAM request option(s): {', '.join(sorted(unknown))}")
        if system_prompt is not None:
            if "decoder_prompt" in custom:
                raise ValueError("Use either a system message or decoder_prompt, not both")
            custom["decoder_prompt"] = system_prompt
        for name in ("use_beam_search", "echo", "continue_final_message"):
            if getattr(request, name, False):
                raise ValueError(f"SEAM does not support {name}")
        for name in (
            "prompt_logprobs",
            "prompt_embeds",
            "suffix",
            "tools",
            "documents",
            "chat_template",
            "media_io_kwargs",
        ):
            if getattr(request, name, None) is not None:
                raise ValueError(f"SEAM does not support {name}")
        if getattr(request, "chat_template_kwargs", None):
            raise ValueError("Use decoder_prompt/decoder_chat_template instead of chat_template_kwargs")
        if getattr(request, "tool_choice", None) not in (None, "none"):
            raise ValueError("SEAM does not support tool_choice")
        if getattr(request, "truncate_prompt_tokens", None) is not None:
            raise ValueError("Use max_input_tokens for encoder truncation; decoder prompts are kept intact")
        if getattr(request, "add_generation_prompt", True) is False:
            raise ValueError("SEAM requires add_generation_prompt=true")

        defaults = sampling_kwargs(self.config["generation"], {})
        fields = request.model_fields_set
        for name in ("temperature", "top_p", "top_k", "repetition_penalty", "min_tokens"):
            if name not in fields or getattr(request, name, None) is None:
                setattr(request, name, defaults[name])
        if getattr(request, "max_completion_tokens", None) is None and (
            "max_tokens" not in fields or request.max_tokens is None
        ):
            request.max_tokens = defaults["max_tokens"]

        data = self.config["data"]
        prompt_overrides = {
            key: custom[key] for key in ("decoder_prompt", "decoder_prefix", "decoder_chat_template") if key in custom
        }
        prompt_ids = (
            request_prompt_ids(self.original.renderer.get_tokenizer(), data, **prompt_overrides)
            if prompt_overrides
            else self.default_prompt_ids
        )
        configured_limit = int(data["max_source_length"])
        mm_kwargs = getattr(request, "mm_processor_kwargs", None) or {}
        if set(mm_kwargs) - {"seam_max_source_length"}:
            raise ValueError("Only seam_max_source_length is accepted in mm_processor_kwargs")
        if "max_input_tokens" in custom and "seam_max_source_length" in mm_kwargs:
            raise ValueError("Pass max_input_tokens or seam_max_source_length, not both")
        source_limit = custom.get("max_input_tokens", mm_kwargs.get("seam_max_source_length", configured_limit))
        if (
            isinstance(source_limit, bool)
            or not isinstance(source_limit, int)
            or not 0 < source_limit <= configured_limit
        ):
            raise ValueError(f"max_input_tokens must be an integer in [1, {configured_limit}]")

        from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

        requested = getattr(request, "max_completion_tokens", None)
        requested = request.max_tokens if requested is None else requested
        budget = get_max_tokens(
            self.handler.model_config.max_model_len,
            requested,
            len(prompt_ids),
            self.handler.default_sampling_params,
            self.handler.override_max_tokens,
        )
        # Validate using the real native request conversion, before encoding a source.
        validate_sampling_params(request.to_sampling_params(budget, self.handler.default_sampling_params))
        return prompt_ids, source_limit, budget, custom.get("encoder_prompt")

    async def _render(self, request, sources, system_prompt=None, *, skip_mm_cache=False):
        from vllm.inputs import build_enc_dec_input
        from vllm.renderers import TokenizeParams

        prompt_ids, source_limit, budget, encoder_prompt = self._prepare(request, system_prompt)
        prompts = [
            source_prompt(
                source, prompt_ids, source_limit=source_limit, output_budget=budget, encoder_prompt=encoder_prompt
            )
            for source in sources
        ]
        for prompt in prompts:
            if getattr(request, "cache_salt", None):
                prompt["cache_salt"] = request.cache_salt
        # Singleton prompts intentionally enter the registered EncDec processor
        # directly. It produces the encoder source plus the exact decoder IDs.
        rendered = await self.original.renderer.render_cmpl_async(
            prompts,
            TokenizeParams(
                max_total_tokens=self.handler.model_config.max_model_len,
                max_output_tokens=budget,
                add_special_tokens=False,
            ),
            skip_mm_cache=skip_mm_cache,
        )
        results = []
        for raw in rendered:
            if raw.get("type") != "multimodal" or "encoder_prompt_token_ids" not in raw:
                raise RuntimeError("SEAM HTTP rendering did not produce encoder-decoder multimodal inputs")
            result = build_enc_dec_input(
                encoder_input=raw,
                decoder_input=None,
                decoder_start_token_id=prompt_ids[0],
                skip_decoder_start_token=True,
            )
            if result["decoder_prompt"]["prompt_token_ids"] != prompt_ids:
                raise RuntimeError("SEAM HTTP rendering changed the request's decoder prompt IDs")
            if "arrival_time" in raw:
                result["arrival_time"] = raw["arrival_time"]
            results.append(result)
        return results

    async def render_completion(self, request, *, skip_mm_cache=False):
        try:
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("SEAM prompt must be source text or a non-empty list of source texts")
            sources = [_text_content(value) for value in prompts]
            return await self._render(request, sources, skip_mm_cache=skip_mm_cache)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return self.original.create_error_response(str(exc))

    async def render_chat(self, request, *, skip_mm_cache=False):
        try:
            source, instruction = _chat_source(request.messages)
            inputs = await self._render(request, [source], instruction, skip_mm_cache=skip_mm_cache)
            return [{"role": "user", "content": source}], inputs
        except (ValueError, TypeError, NotImplementedError) as exc:
            return self.original.create_error_response(str(exc))


class _SEAMServingMixin:
    def __init__(self, original, config):
        # Reuse the native handler's initialized engine, model registry, parser,
        # logging and response options without starting another engine.
        self.__dict__.update(vars(original))
        self.online_renderer = _SEAMOnlineRenderer(self, original.online_renderer, config)

    def _extract_prompt_components(self, prompt):
        from vllm.renderers.inputs.preprocess import PromptComponents

        target = prompt["decoder_prompt"] if prompt.get("type") == "enc_dec" else prompt
        return PromptComponents(text=target.get("prompt"), token_ids=target.get("prompt_token_ids"))

    def _extract_prompt_len(self, prompt):
        return len(self._extract_prompt_components(prompt).token_ids or [])


def install_http_handlers(state) -> None:
    import vllm
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    if vllm.__version__.split("+")[0] != "0.25.1":
        raise RuntimeError("SEAM HTTP requires vLLM 0.25.1")
    hf_config = state.vllm_config.model_config.hf_config
    if "SEAMForQwen3" not in (getattr(hf_config, "architectures", None) or []):
        raise ValueError("SEAM HTTP middleware requires the SEAMForQwen3 model override")
    config = load_config(hf_config.seam_config_path)
    if not hasattr(state, "seam_http_info"):
        state.seam_http_info = server_info(
            hf_config, config, max_model_len=getattr(state.vllm_config.model_config, "max_model_len", None)
        )

    class SEAMCompletion(_SEAMServingMixin, OpenAIServingCompletion):
        """Native completion responses with SEAM rendering and context accounting."""

    class SEAMChat(_SEAMServingMixin, OpenAIServingChat):
        """Native chat responses with SEAM rendering and context accounting."""

    for name, cls in (("openai_serving_completion", SEAMCompletion), ("openai_serving_chat", SEAMChat)):
        original = getattr(state, name, None)
        if original is None:
            raise RuntimeError(f"Native vLLM handler {name} is unavailable; use a generate-mode API server")
        if not isinstance(original, _SEAMServingMixin):
            setattr(state, name, cls(original, config))


class SEAMMiddleware:
    """Opt-in native ``vllm serve --middleware seam_vllm.http.SEAMMiddleware``."""

    def __init__(self, app):
        self.app = app
        self.installed = False

    def _install(self, scope):
        if not self.installed:
            app = scope["app"]
            install_http_handlers(app.state)
            if hasattr(app.state, "seam_http_info") and not getattr(app.state, "seam_info_route_registered", False):

                async def info():
                    return app.state.seam_http_info

                # Add a regular /v1 route, inside native authentication. Do not
                # return this data directly from the outer ASGI middleware.
                app.add_api_route("/v1/seam/info", info, methods=["GET"], include_in_schema=False)
                app.state.seam_info_route_registered = True
            self.installed = True

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":

            async def lifespan_send(message):
                if message["type"] == "lifespan.startup.complete":
                    self._install(scope)
                await send(message)

            await self.app(scope, receive, lifespan_send)
            return
        if scope["type"] == "http":
            self._install(scope)
            info = getattr(scope["app"].state, "seam_http_info", None)
            if info is not None:

                async def response_send(message):
                    if message["type"] == "http.response.start":
                        message = dict(message)
                        message["headers"] = [
                            *message.get("headers", []),
                            (b"x-seam-fingerprint", info["fingerprint"].encode("ascii")),
                        ]
                    await send(message)

                await self.app(scope, receive, response_send)
                return
        await self.app(scope, receive, send)
