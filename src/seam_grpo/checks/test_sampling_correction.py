"""Compare the adapter's loss AND gradients against pinned TRL's native TIS."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from seam_grpo.sampling import correct_vllm_policy_inputs
from trl import GRPOTrainer


def loss_trainer(logps, loss_type, beta, importance_level, *, upstream_tis):
    trainer = object.__new__(GRPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer.accelerator = SimpleNamespace(
        reduce=lambda x, **kw: x, gather=lambda x: x, num_processes=1, sync_gradients=True
    )
    trainer.args = SimpleNamespace(
        delta=None,
        use_bias_correction_kl=True,
        steps_per_generation=2,
        sapo_temperature_pos=1.0,
        sapo_temperature_neg=1.5,
        vespo_k_pos=2.0,
        vespo_lambda_pos=3.0,
        vespo_k_neg=3.0,
        vespo_lambda_neg=2.0,
    )
    trainer.loss_type = loss_type
    trainer.top_entropy_quantile = 1.0
    trainer.off_policy_mask_threshold = None
    trainer.aux_loss_enabled = False
    trainer.beta = beta
    trainer.importance_sampling_level = importance_level
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.max_completion_length = 3
    trainer.current_gradient_accumulation_steps = 2
    trainer.use_vllm = upstream_tis
    trainer.vllm_importance_sampling_correction = True
    trainer._entropy_bonus_enabled = True
    trainer.entropy_coef = 0.05
    trainer.use_adaptive_entropy = False
    trainer._metrics = {"train": defaultdict(list)}
    trainer._get_per_token_logps_and_entropies = lambda *a, **kw: (logps, logps.square() + 1, None)
    return trainer


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "cispo", "sapo", "luspo", "vespo"])
@pytest.mark.parametrize("beta", [0.0, 0.03])
@pytest.mark.parametrize("importance_level", ["token", "sequence"])
def test_full_upstream_tis_loss_and_gradient_equivalence(loss_type, beta, importance_level):
    actual = torch.tensor([[-2.0, -3.0, -1.0], [-1.0, -0.5, -4.0]], requires_grad=True)
    expected = actual.detach().clone().requires_grad_()
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    ratios = torch.tensor([[1.3, 0.8, 1.0], [2.9, 1.2, 0.9]])
    behavior = actual.detach() - ratios.log()
    behavior[0, 2] = float("nan")  # padded probability is not an action
    inputs = {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "prompt_mask": torch.ones(2, 1, dtype=torch.bool),
        "completion_ids": torch.ones(2, 3, dtype=torch.long),
        "completion_mask": mask,
        "advantages": torch.tensor([0.7, -0.7]),
        "old_per_token_logps": expected.detach(),
        "ref_per_token_logps": actual.detach() - 0.1,
        "importance_sampling_ratio": ratios.clamp(max=2),
        "num_items_in_batch": mask.sum(),
    }
    adapted = dict(inputs)
    correction = correct_vllm_policy_inputs(adapted, actual, behavior, mask, cap=2, loss_type=loss_type)
    assert not correction.requires_grad and not adapted["old_per_token_logps"].requires_grad
    torch.testing.assert_close(correction, ratios.clamp(max=2))
    assert inputs["advantages"].ndim == 1  # original buffered values are unchanged
    actual_loss = loss_trainer(actual, loss_type, beta, importance_level, upstream_tis=False)._compute_loss(
        None, adapted
    )
    expected_loss = loss_trainer(expected, loss_type, beta, importance_level, upstream_tis=True)._compute_loss(
        None, inputs
    )
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual.grad, expected.grad)
    assert actual.grad[0, 2] == 0


def test_extreme_ratio_is_capped_before_exponentiation():
    current = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    inputs = {"advantages": torch.tensor([-1.0])}
    weights = correct_vllm_policy_inputs(
        inputs, current, torch.tensor([[-1000.0, -1.0]]), torch.ones(1, 2, dtype=torch.bool), cap=2, loss_type="dr_grpo"
    )
    assert weights.tolist() == [[2.0, 1.0]]
    assert torch.isfinite(weights).all()
