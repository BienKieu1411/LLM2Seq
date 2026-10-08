from __future__ import annotations

import torch
from torch import nn

from eviseq_afmr.parameter_audit import parameter_breakdown


class CrossLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False)
        self.cross = nn.Linear(4, 4, bias=False)
        self.cross_norm = nn.LayerNorm(4)
        self.cross_gate = nn.Parameter(torch.zeros(()))


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([CrossLayer()])
        self.backbone.token_embedding = nn.Embedding(5, 4)
        self.lm_head = nn.Linear(4, 5, bias=False)
        self.grounded_copy = nn.Linear(4, 2, bias=False)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4, bias=False)
        self.bridge = nn.Linear(3, 4, bias=False)
        self.decoder = FakeDecoder()


def test_parameter_breakdown_reconciles_bridge_and_decoder_additions():
    counts = parameter_breakdown(FakeModel())
    assert counts["encoder_params"] == 12
    assert counts["seam_params"] == 12
    assert counts["decoder_cross_attention_params"] == 16 + 8 + 1
    assert counts["grounded_copy_params"] == 8
    assert counts["decoder_base_params"] == 16 + 20 + 20
    assert counts["decoder_params"] == (
        counts["decoder_base_params"] + counts["decoder_cross_attention_params"] + counts["grounded_copy_params"]
    )
    assert counts["interface_total_params"] == (
        counts["seam_params"] + counts["decoder_cross_attention_params"] + counts["grounded_copy_params"]
    )
    assert counts["total_params"] == counts["encoder_params"] + counts["seam_params"] + counts["decoder_params"]


def test_parameter_breakdown_counts_tied_decoder_weight_once():
    model = FakeModel()
    model.decoder.lm_head.weight = model.decoder.backbone.token_embedding.weight
    counts = parameter_breakdown(model)
    assert counts["total_params"] == counts["encoder_params"] + counts["seam_params"] + counts["decoder_params"]
    assert counts["decoder_params"] == sum(parameter.numel() for parameter in model.decoder.parameters())
