from __future__ import annotations

import json
from typing import Any

from decoder_baselines.evaluate import _resolve_backend, _write_prediction_batch
from decoder_baselines.vllm_service import VLLMClient, normalize_base_url


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


def test_auto_backend_keeps_custom_nemotron_local() -> None:
    assert _resolve_backend({"model": {"family": "causal_lm"}}, "auto") == "vllm"
    assert _resolve_backend({"model": {"family": "nemotron_diffusion"}}, "auto") == "local"
    # A stale command-line vLLM flag must not bring back the incompatible
    # custom-model server path.
    assert _resolve_backend({"model": {"family": "nemotron_diffusion"}}, "vllm") == "local"


def test_prediction_writer_flushes_each_completed_sample() -> None:
    class Handle:
        def __init__(self) -> None:
            self.lines: list[str] = []
            self.flushes = 0

        def write(self, value: str) -> None:
            self.lines.append(value)

        def flush(self) -> None:
            self.flushes += 1

    handle = Handle()
    _write_prediction_batch(
        handle,
        [{"id": "a", "source": "s", "reference": "r"}, {"id": "b", "source": "s", "reference": "r"}],
        ["first", "second"],
    )
    assert len(handle.lines) == 2
    assert handle.flushes == 2
    assert '"prediction": "first"' in handle.lines[0]
    assert '"prediction": "second"' in handle.lines[1]
