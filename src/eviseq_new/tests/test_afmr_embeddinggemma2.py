from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from eviseq_afmr.modeling.encoder import PretrainedEncoderAdapter


class _FakeEmbeddingGemmaTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))
        self.norm = nn.LayerNorm(4)
        self.embedding_projection = nn.Linear(4, 6, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = torch.nn.functional.one_hot(input_ids.remainder(4), num_classes=4).float()
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=self.embedding_projection(self.norm(hidden)))


class _FakeEmbeddingGemma2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = _FakeEmbeddingGemmaTextModel()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.language_model(input_ids, attention_mask=attention_mask)


def _fake_config():
    return SimpleNamespace(
        model_type="embedding_gemma2",
        text_config=SimpleNamespace(hidden_size=4, embedding_dim=6, num_hidden_layers=4),
        vision_config=object(),
        audio_config=object(),
        output_hidden_states=False,
        use_cache=False,
    )


def _patch_auto_classes(monkeypatch, config, on_load=None):
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: config),
    )

    def from_pretrained(*args, config, **kwargs):
        if on_load is not None:
            on_load(config)
        return _FakeEmbeddingGemma2Model(config)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(from_pretrained))


def test_embeddinggemma2_uses_projected_token_width_and_text_only_taps(monkeypatch):
    raw_config = _fake_config()
    _patch_auto_classes(
        monkeypatch,
        raw_config,
        on_load=lambda config: (
            pytest.fail("unused modality towers were not disabled")
            if config.vision_config is not None or config.audio_config is not None
            else None
        ),
    )

    encoder = PretrainedEncoderAdapter(
        "local/embeddinggemma-2",
        depth_taps=3,
        dtype=torch.float32,
        gradient_checkpointing=False,
    )
    ids = torch.tensor([[1, 2, 3, 0]])
    mask = torch.tensor([[True, True, True, False]])
    output = encoder(ids, mask, mask)

    assert encoder.hidden_size == 6  # embedding_dim, not the text tower's width of 4
    assert output.final.shape == (1, 4, 6)
    assert len(output.taps) == 3
    assert all(tap.shape == output.final.shape for tap in output.taps)
    assert torch.equal(output.taps[-1], output.final)

    sum(tap.sum() for tap in output.taps).backward()
    assert encoder.model.language_model.embedding_projection.weight.grad is not None
    assert encoder.model.language_model.layers[1].weight.grad is not None


def test_embeddinggemma2_rejects_float16(monkeypatch):
    _patch_auto_classes(monkeypatch, _fake_config())

    with pytest.raises(ValueError, match="bfloat16 or float32"):
        PretrainedEncoderAdapter(
            "local/embeddinggemma-2",
            depth_taps=1,
            dtype=torch.float16,
            gradient_checkpointing=False,
        )


def test_embeddinggemma2_reports_minimum_transformers_version(monkeypatch):
    import transformers

    def unsupported_config(*args, **kwargs):
        raise ValueError("model type embedding_gemma2 is not recognized")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(unsupported_config))

    with pytest.raises(RuntimeError, match=r"transformers>=5\.19\.0"):
        PretrainedEncoderAdapter(
            "local/embeddinggemma-2",
            depth_taps=1,
            dtype=torch.float32,
            gradient_checkpointing=False,
        )
