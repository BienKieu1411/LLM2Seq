from __future__ import annotations

import json
from typing import Any

from decoder_baselines.evaluate import _resolve_backend, _vllm_model_impl
from decoder_baselines.vllm_service import VLLMClient, build_vllm_command, normalize_base_url


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def test_normalize_vllm_base_url() -> None:
    assert normalize_base_url("http://localhost:8000") == "http://localhost:8000/v1"
    assert normalize_base_url("http://localhost:8000/v1/") == "http://localhost:8000/v1"


def test_vllm_completion_uses_token_batches_and_preserves_order(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "choices": [
                    {"index": 1, "text": "second"},
                    {"index": 0, "text": "first"},
                ]
            }
        )

    monkeypatch.setattr("decoder_baselines.vllm_service.urlopen", fake_urlopen)
    client = VLLMClient("http://localhost:8000/v1", timeout=12)
    result = client.complete(
        [[1, 2], [3, 4]],
        {
            "max_new_tokens": 32,
            "min_new_tokens": 4,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 3,
        },
        model="local-model",
    )
    assert result == ["first", "second"]
    assert captured["url"] == "http://localhost:8000/v1/completions"
    assert captured["payload"]["prompt"] == [[1, 2], [3, 4]]
    assert captured["payload"]["min_tokens"] == 4
    assert captured["payload"]["repetition_penalty"] == 1.05
    assert "no_repeat_ngram_size" not in captured["payload"]


def test_auto_backend_uses_vllm_for_custom_nemotron(monkeypatch: Any) -> None:
    assert _resolve_backend({"model": {"family": "causal_lm"}}, "auto") == "vllm"
    assert _resolve_backend({"model": {"family": "nemotron_diffusion"}}, "auto") == "vllm"
    assert _vllm_model_impl({"model": {"family": "nemotron_diffusion"}}) == "transformers"
    assert _vllm_model_impl({"model": {"family": "causal_lm"}}) == "auto"
    monkeypatch.setenv("VLLM_MODEL_IMPL", "auto")
    assert _vllm_model_impl({"model": {"family": "nemotron_diffusion"}}) == "auto"


def test_vllm_model_name_must_match_served_model(monkeypatch: Any) -> None:
    client = VLLMClient("http://localhost:8000/v1")
    monkeypatch.setattr(client, "models", lambda: [{"id": "local/nemotron"}])
    assert client.model_name("local/nemotron") == "local/nemotron"
    try:
        client.model_name("wrong-model")
    except RuntimeError as exc:
        assert "wrong-model" in str(exc)
        assert "local/nemotron" in str(exc)
    else:  # pragma: no cover - defensive assertion for the test itself
        raise AssertionError("model_name accepted an unserved model")


def test_build_vllm_command_for_nemotron(tmp_path: Any) -> None:
    checkpoint = tmp_path / "Nemotron-Labs-Diffusion-3B"
    checkpoint.mkdir()
    command = build_vllm_command(
        checkpoint,
        host="127.0.0.1",
        port=8123,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        model_impl="transformers",
        served_model_name="nvidia/Nemotron-Labs-Diffusion-3B",
        enforce_eager=True,
    )
    assert command[:3] == ["vllm", "serve", str(checkpoint.resolve())]
    assert "--model-impl" in command
    assert command[command.index("--model-impl") + 1] == "transformers"
    assert "--served-model-name" in command
    assert command[command.index("--served-model-name") + 1] == "nvidia/Nemotron-Labs-Diffusion-3B"
    assert "--trust-remote-code" in command
    assert "--enforce-eager" in command
