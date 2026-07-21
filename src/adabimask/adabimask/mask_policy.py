"""Layer grouping, fixed ablations, and learnable bidirectional gates."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MaskMode = Literal["causal", "full", "fixed", "learnable"]
FixedStrategy = Literal["bottom", "middle", "top", "random", "custom"]
Route = Literal["causal", "bidirectional", "mix"]


@dataclass(frozen=True)
class MaskPolicyConfig:
    mode: MaskMode = "causal"
    num_groups: int = 7
    budget_groups: int = 2
    fixed_strategy: FixedStrategy = "middle"
    selected_groups: Tuple[int, ...] = ()
    random_seed: int = 42
    init_probability: float = 0.25
    temperature: float = 1.0
    curriculum_ratio: float = 0.2
    hard_eval: bool = True
    budget_weight: float = 0.05
    binary_weight: float = 0.01

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, object]]) -> "MaskPolicyConfig":
        raw = raw or {}
        selected = tuple(int(index) for index in raw.get("selected_groups", []) or [])
        return cls(
            mode=str(raw.get("mode", "causal")),
            num_groups=int(raw.get("num_groups", 7)),
            budget_groups=int(raw.get("budget_groups", 2)),
            fixed_strategy=str(raw.get("fixed_strategy", "middle")),
            selected_groups=selected,
            random_seed=int(raw.get("random_seed", 42)),
            init_probability=float(raw.get("init_probability", 0.25)),
            temperature=float(raw.get("temperature", 1.0)),
            curriculum_ratio=float(raw.get("curriculum_ratio", 0.2)),
            hard_eval=bool(raw.get("hard_eval", True)),
            budget_weight=float(raw.get("budget_weight", 0.05)),
            binary_weight=float(raw.get("binary_weight", 0.01)),
        )


def balanced_layer_groups(num_layers: int, num_groups: int) -> List[Tuple[int, ...]]:
    """Partition layers into contiguous groups whose sizes differ by at most one."""

    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 1 <= num_groups <= num_layers:
        raise ValueError(f"num_groups must be in [1, {num_layers}], got {num_groups}")

    base_size, remainder = divmod(num_layers, num_groups)
    groups: List[Tuple[int, ...]] = []
    start = 0
    for group_index in range(num_groups):
        size = base_size + int(group_index < remainder)
        groups.append(tuple(range(start, start + size)))
        start += size
    return groups


def choose_fixed_groups(config: MaskPolicyConfig) -> Tuple[int, ...]:
    total = config.num_groups
    budget = config.budget_groups
    if not 0 <= budget <= total:
        raise ValueError(f"budget_groups must be in [0, {total}], got {budget}")

    if config.fixed_strategy == "custom":
        selected = tuple(sorted(set(config.selected_groups)))
        if any(index < 0 or index >= total for index in selected):
            raise ValueError(f"selected_groups must be within [0, {total - 1}]")
        if len(selected) != budget:
            raise ValueError(f"custom strategy requires exactly budget_groups={budget} unique groups, got {selected}")
        return selected
    if budget == 0:
        return ()
    if config.fixed_strategy == "bottom":
        return tuple(range(budget))
    if config.fixed_strategy == "top":
        return tuple(range(total - budget, total))
    if config.fixed_strategy == "middle":
        start = (total - budget) // 2
        return tuple(range(start, start + budget))
    if config.fixed_strategy == "random":
        generator = random.Random(config.random_seed)
        return tuple(sorted(generator.sample(range(total), budget)))
    raise ValueError(f"Unsupported fixed_strategy: {config.fixed_strategy}")


class LayerMaskPolicy(nn.Module):
    """Maps transformer layers to causal, bidirectional, or soft mixed attention.

    During learnable search, one scalar gate is used per contiguous layer group.
    During evaluation, ``hard_eval=True`` selects the top-K groups and avoids the
    dual-attention compute. The exported policy is therefore exactly the model
    that will be retrained and deployed.
    """

    def __init__(self, num_layers: int, config: MaskPolicyConfig):
        super().__init__()
        self.num_layers = int(num_layers)
        self.policy_config = config
        self.register_buffer("_device_anchor", torch.zeros(()), persistent=False)
        # Progress is controlled by the trainer.  A value of zero keeps the
        # encoder exactly causal during interface warm-up; one enables the
        # complete learned bidirectional budget.
        self.register_buffer("_training_progress", torch.ones(()), persistent=False)
        self._force_causal = False
        self.groups = balanced_layer_groups(self.num_layers, config.num_groups)
        self._layer_to_group = {
            layer_index: group_index for group_index, layers in enumerate(self.groups) for layer_index in layers
        }

        self.register_parameter("gate_logits", None)
        if config.mode == "learnable":
            probability = min(max(config.init_probability, 1e-4), 1.0 - 1e-4)
            initial_logit = math.log(probability / (1.0 - probability))
            self.gate_logits = nn.Parameter(torch.full((config.num_groups,), initial_logit))

        self._fixed_groups: Tuple[int, ...] = ()
        if config.mode == "full":
            self._fixed_groups = tuple(range(config.num_groups))
        elif config.mode == "fixed":
            self._fixed_groups = choose_fixed_groups(config)

    @property
    def mode(self) -> MaskMode:
        return self.policy_config.mode

    def group_for_layer(self, layer_index: int) -> int:
        if layer_index not in self._layer_to_group:
            raise IndexError(f"Layer {layer_index} is outside [0, {self.num_layers - 1}]")
        return self._layer_to_group[layer_index]

    def probabilities(self) -> torch.Tensor:
        if self.gate_logits is None:
            values = torch.zeros(self.policy_config.num_groups, device=self._device_anchor.device)
            if self.mode == "full":
                values.fill_(1.0)
            elif self.mode == "fixed" and self._fixed_groups:
                values[list(self._fixed_groups)] = 1.0
            return values
        temperature = max(self.policy_config.temperature, 1e-4)
        return torch.sigmoid(self.gate_logits / temperature)

    def set_progress(self, progress: float) -> None:
        self._training_progress.fill_(min(max(float(progress), 0.0), 1.0))

    def set_force_causal(self, enabled: bool) -> None:
        self._force_causal = bool(enabled)

    def curriculum_scale(self) -> torch.Tensor:
        ratio = max(float(self.policy_config.curriculum_ratio), 1e-8)
        return torch.clamp(self._training_progress / ratio, min=0.0, max=1.0)

    def effective_probabilities(self) -> torch.Tensor:
        if self.mode != "learnable" or not self.training:
            return self.probabilities()
        return self.probabilities() * self.curriculum_scale()

    def topk_groups(self) -> Tuple[int, ...]:
        budget = self.policy_config.budget_groups
        if budget == 0:
            return ()
        if self.mode == "learnable":
            probabilities = self.probabilities().detach()
            indices = torch.topk(probabilities, k=budget, largest=True, sorted=False).indices.tolist()
            return tuple(sorted(int(index) for index in indices))
        return self._fixed_groups

    def route(self, layer_index: int) -> Route:
        group_index = self.group_for_layer(layer_index)
        if self._force_causal:
            return "causal"
        if self.mode == "causal":
            return "causal"
        if self.mode in {"full", "fixed"}:
            return "bidirectional" if group_index in self._fixed_groups else "causal"
        if self.training and float(self.curriculum_scale().item()) == 0.0:
            return "causal"
        if not self.training and self.policy_config.hard_eval:
            return "bidirectional" if group_index in self.topk_groups() else "causal"
        return "mix"

    def gate_for_layer(self, layer_index: int) -> torch.Tensor:
        if self.gate_logits is None:
            raise RuntimeError("Soft gates only exist when mask.mode=learnable")
        return self.effective_probabilities()[self.group_for_layer(layer_index)]

    def regularization(self) -> Dict[str, torch.Tensor]:
        """Return normalized budget and binary penalties plus their weighted sum."""

        if self.gate_logits is None:
            zero = torch.zeros((), device=self._device_anchor.device)
            return {"loss_gate": zero, "loss_budget": zero, "loss_binary": zero}

        probabilities = self.effective_probabilities()
        num_groups = float(self.policy_config.num_groups)
        target_budget = float(self.policy_config.budget_groups) * self.curriculum_scale()
        budget_error = (probabilities.sum() - target_budget) / num_groups
        loss_budget = budget_error.square()
        loss_binary = (probabilities * (1.0 - probabilities)).mean()
        loss_gate = self.policy_config.budget_weight * loss_budget + self.policy_config.binary_weight * loss_binary
        return {
            "loss_gate": loss_gate,
            "loss_budget": loss_budget,
            "loss_binary": loss_binary,
        }

    def selected_layers(self, hard: bool = True) -> Tuple[int, ...]:
        if self.mode == "causal":
            return ()
        selected_groups = self.topk_groups() if hard else self._fixed_groups
        layers = [layer for group in selected_groups for layer in self.groups[group]]
        return tuple(sorted(layers))

    def describe(self) -> Dict[str, object]:
        probabilities = [round(float(value), 6) for value in self.probabilities().detach().cpu().tolist()]
        effective = [
            round(float(value), 6) for value in self.effective_probabilities().detach().cpu().tolist()
        ]
        return {
            "mode": self.mode,
            "num_layers": self.num_layers,
            "groups": [list(group) for group in self.groups],
            "budget_groups": self.policy_config.budget_groups,
            "probabilities": probabilities,
            "effective_probabilities": effective,
            "curriculum_progress": round(float(self._training_progress.item()), 6),
            "force_causal": self._force_causal,
            "selected_groups": list(self.topk_groups()),
            "selected_layers": list(self.selected_layers()),
        }
