from __future__ import annotations

from typing import Iterable

from torch import nn


def _unique_parameters(parameters: Iterable[nn.Parameter]) -> dict[int, nn.Parameter]:
    return {id(parameter): parameter for parameter in parameters}


def _count(parameters: dict[int, nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters.values())


def _decoder_cross_attention_parameters(decoder: nn.Module) -> dict[int, nn.Parameter]:
    parameters: dict[int, nn.Parameter] = {}
    for layer in decoder.backbone.layers:
        for module_name in ("cross", "cross_norm"):
            module = getattr(layer, module_name, None)
            if isinstance(module, nn.Module):
                parameters.update(_unique_parameters(module.parameters()))
        gate = getattr(layer, "cross_gate", None)
        if isinstance(gate, nn.Parameter):
            parameters[id(gate)] = gate
    return parameters


def parameter_breakdown(model: nn.Module) -> dict[str, int]:
    required = ("encoder", "bridge", "decoder")
    missing = [name for name in required if not isinstance(getattr(model, name, None), nn.Module)]
    if missing:
        raise TypeError(f"Model is missing parameter groups: {', '.join(missing)}")

    encoder_parameters = _unique_parameters(model.encoder.parameters())
    bridge_parameters = _unique_parameters(model.bridge.parameters())
    decoder_parameters = _unique_parameters(model.decoder.parameters())
    model_parameters = _unique_parameters(model.parameters())

    component_ids = [set(encoder_parameters), set(bridge_parameters), set(decoder_parameters)]
    if any(component_ids[left] & component_ids[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Encoder, bridge and decoder share parameter objects; group attribution is ambiguous")
    if set().union(*component_ids) != set(model_parameters):
        raise ValueError("Encoder, bridge and decoder groups do not cover every model parameter")

    cross_parameters = _decoder_cross_attention_parameters(model.decoder)
    copy_module = getattr(model.decoder, "grounded_copy", None)
    copy_parameters = _unique_parameters(copy_module.parameters()) if isinstance(copy_module, nn.Module) else {}
    if set(cross_parameters) & set(copy_parameters):
        raise ValueError("Cross-attention and grounded-copy groups share parameter objects")
    decoder_ids = set(decoder_parameters)
    addition_ids = set(cross_parameters) | set(copy_parameters)
    if not addition_ids <= decoder_ids:
        raise ValueError("Decoder interface parameters are not registered under the decoder")

    encoder_count = _count(encoder_parameters)
    bridge_count = _count(bridge_parameters)
    decoder_count = _count(decoder_parameters)
    total_count = _count(model_parameters)
    cross_count = _count(cross_parameters)
    copy_count = _count(copy_parameters)
    decoder_base_count = decoder_count - cross_count - copy_count
    interface_count = bridge_count + cross_count + copy_count

    if decoder_base_count < 0 or encoder_count + bridge_count + decoder_count != total_count:
        raise ValueError("Parameter groups do not reconcile with the unique total parameter count")

    return {
        "encoder_params": encoder_count,
        "seam_params": bridge_count,
        "decoder_params": decoder_count,
        "decoder_base_params": decoder_base_count,
        "decoder_cross_attention_params": cross_count,
        "grounded_copy_params": copy_count,
        "interface_total_params": interface_count,
        "total_params": total_count,
    }
