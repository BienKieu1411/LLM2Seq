"""Complete native-actor to unquantized TP=1 Qwen3 weight mapping.

No vLLM import: the same mapping and transaction checks can run on tiny CPU
modules. Transactions block inference until every destination slice is loaded.
"""

from __future__ import annotations

from collections import defaultdict

import torch


class ActorWeightUpdate:
    def __init__(self, model, manifest, version: int):
        self.model = model
        self.version = version
        self.received = set()
        self.failed = False
        self.slots = {}
        parameters = dict(model.named_parameters())
        devices = {value.device for value in parameters.values()}
        if len(devices) != 1:
            raise ValueError("Full actor sync requires one runtime device; CPU offload is not supported")
        # Resolve aliases (e.g. tied lm_head) to the canonical destination name.
        canonical = {id(value): name for name, value in parameters.items()}
        aliases = dict(model.named_parameters(remove_duplicate=False))
        coverage = defaultdict(list)
        config = model.config
        for source, shape in manifest:
            if source in self.slots:
                raise ValueError(f"Duplicate actor tensor: {source}")
            target = source
            if source.startswith("decoder.grounded_copy."):
                target = "copy_head." + source.removeprefix("decoder.grounded_copy.")
            elif source.startswith("decoder.backbone."):
                target = "model." + source.removeprefix("decoder.backbone.").replace(".base.", ".")
            elif source.startswith("decoder.lm_head."):
                target = source.removeprefix("decoder.")
            elif not source.startswith(("encoder.", "bridge.")):
                raise ValueError(f"Unsupported actor tensor: {source}")

            start = 0
            rows = None
            for projection, index in (("q", 0), ("k", 1), ("v", 2)):
                marker = f".self_attn.{projection}_proj.weight"
                if target.startswith("model.layers.") and target.endswith(marker):
                    target = target.removesuffix(marker) + ".self_attn.qkv_proj.weight"
                    head_dim = int(config.head_dim)
                    q = int(config.num_attention_heads) * head_dim
                    kv = int(config.num_key_value_heads) * head_dim
                    start, rows = ((0, q), (q, kv), (q + kv, kv))[index]
                    break
            for projection, index in (("gate", 0), ("up", 1)):
                marker = f".mlp.{projection}_proj.weight"
                if target.startswith("model.layers.") and target.endswith(marker):
                    target = target.removesuffix(marker) + ".mlp.gate_up_proj.weight"
                    rows = int(config.intermediate_size)
                    start = index * rows
                    break
            if target not in aliases:
                raise ValueError(f"Actor tensor {source} has no runtime destination: {target}")
            parameter = aliases[target]
            name = canonical[id(parameter)]
            shape = tuple(shape)
            expected = tuple(parameter.shape)
            if rows is not None:
                expected = (rows, *expected[1:])
            elif target in {"model.embed_tokens.weight", "lm_head.weight"}:
                # vLLM pads vocab rows for its kernels; native rows stay exact.
                if shape and shape[0] == int(config.vocab_size) and parameter.shape[0] >= shape[0]:
                    rows = shape[0]
                    expected = (shape[0], *parameter.shape[1:])
            if shape != expected:
                raise ValueError(f"Actor shape mismatch for {source}: {shape} != {expected}")
            stop = start + rows if rows is not None else (parameter.shape[0] if parameter.ndim else 1)
            coverage[name].append((start, stop))
            self.slots[source] = (parameter, start, rows, shape)

        missing = set(parameters) - set(coverage)
        if missing:
            raise ValueError(f"Actor update does not cover runtime parameters: {sorted(missing)[:8]}")
        for name, intervals in coverage.items():
            previous = 0
            for start, stop in sorted(intervals):
                if start != previous:
                    raise ValueError(f"Actor mapping overlaps or omits rows of {name}")
                previous = stop
            parameter = parameters[name]
            required = parameter.shape[0] if parameter.ndim else 1
            if name in {"model.embed_tokens.weight", "lm_head.weight"}:
                required = int(config.vocab_size)
            if previous != required:
                raise ValueError(f"Actor mapping does not cover all rows of {name}")

    @torch.no_grad()
    def load(self, weights):
        if self.failed:
            raise RuntimeError("Actor update previously failed; rebuild the engine")
        try:
            self._load(weights)
        except Exception:
            self.failed = True
            raise

    def _load(self, weights):
        finite = []
        for name, value in weights:
            if name not in self.slots or name in self.received:
                raise ValueError(f"Unexpected or repeated actor tensor: {name}")
            parameter, start, rows, shape = self.slots[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"Invalid actor tensor: {name}")
            destination = parameter if rows is None else parameter[start : start + rows]
            destination.copy_(value.detach())
            finite.append(torch.isfinite(destination).all())
            if (
                rows is not None
                and parameter.shape[0] > rows
                and name in {"decoder.backbone.embed_tokens.weight", "decoder.lm_head.weight"}
            ):
                parameter[rows:].zero_()
            self.received.add(name)
        if finite and not torch.stack(finite).all():
            raise ValueError("Nonfinite actor weights; update remains blocked")

    def finish(self):
        if self.failed:
            raise RuntimeError("Actor update previously failed; rebuild the engine")
        missing = set(self.slots) - self.received
        if missing:
            raise RuntimeError(f"Incomplete actor update: {sorted(missing)[:8]}")
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"policy_version": self.version, "tensors": len(self.received)}
