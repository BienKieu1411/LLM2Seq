import torch

from afmr_core.evaluation.generate import _apply_repetition_penalty
from afmr_core.modeling.dual_readout import DualReadout, mixture_nll_from_state
from afmr_core.modeling.grounded_copy import CopyRead, CopyState, GroundedCopyHead


def _copy_inputs(active=True):
    token_ids = torch.tensor([[2, 3, 2, 5]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, True]]) if active else torch.zeros_like(token_ids, dtype=torch.bool)
    state = CopyState(
        keys=torch.randn(1, 4, 3),
        token_ids=token_ids,
        mask=mask,
        bias=torch.zeros(1, 4),
    )
    attention = torch.log(torch.tensor([[[0.2, 0.3, 0.1, 0.4]]]))
    if not active:
        attention = torch.full_like(attention, torch.finfo(torch.float32).min)
    raw = torch.tensor([[[torch.logit(torch.tensor(0.2)).item()]]], requires_grad=True)
    g = torch.sigmoid(raw)
    read = CopyRead(
        query=torch.randn(1, 1, 3),
        log_attention=attention,
        raw_gate=raw,
        g=g if active else torch.zeros_like(g),
        active=torch.tensor([[[active]]]),
    )
    return state, read


def test_duplicate_copy_ids_are_marginalized():
    state, read = _copy_inputs()
    log_prob = GroundedCopyHead.copy_log_prob(read, state, 8)
    probabilities = log_prob.exp()
    assert torch.allclose(probabilities[0, 0, 2], torch.tensor(0.3), atol=1e-6)
    assert torch.allclose(probabilities[0, 0, 3], torch.tensor(0.3), atol=1e-6)
    assert torch.allclose(probabilities[0, 0, 5], torch.tensor(0.4), atol=1e-6)
    assert torch.allclose(probabilities.sum(-1), torch.ones(1, 1), atol=1e-6)


def test_zero_semantic_endpoint_matches_legacy_copy_mixture():
    torch.manual_seed(7)
    state, read = _copy_inputs()
    hidden = torch.randn(1, 1, 6)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3, alpha_max=0.2, alpha_init=0.05)
    diagnostics = {
        "q": torch.zeros(1, 1, 3),
        "u": torch.zeros(1, 1, 3),
        "evidence": torch.zeros(1, 1, 1),
    }
    result = readout(hidden, hidden, diagnostics, read, state, lm_head)
    z = lm_head(hidden).float()
    log_p0 = z - torch.logsumexp(z, dim=-1, keepdim=True)
    log_pcopy = GroundedCopyHead.copy_log_prob(read, state, z.shape[-1])
    expected = torch.logaddexp(
        log_p0 + torch.nn.functional.logsigmoid(-read.raw_gate),
        log_pcopy + torch.nn.functional.logsigmoid(read.raw_gate),
    ) + torch.logsumexp(z, dim=-1, keepdim=True)
    assert torch.allclose(result.output_logits, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(result.probability.sum(-1), torch.ones(1, 1), atol=1e-6)


def test_legacy_mode_uses_semantic_branch_for_non_copy_mass():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 1, 6)
    semantic_hidden = hidden + 0.25 * torch.randn_like(hidden)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3, mode="legacy_copy_mixture", alpha_max=0.2, alpha_init=0.05)
    result = readout(
        hidden,
        semantic_hidden,
        {"q": torch.zeros(1, 1, 3), "u": torch.zeros(1, 1, 3), "evidence": torch.ones(1, 1, 1)},
        read,
        state,
        lm_head,
    )
    z0 = lm_head(hidden).float()
    zs = lm_head(semantic_hidden).float()
    log_p0 = z0 - torch.logsumexp(z0, dim=-1, keepdim=True)
    log_ps = zs - torch.logsumexp(zs, dim=-1, keepdim=True)
    log_pcopy = GroundedCopyHead.copy_log_prob(read, state, z0.shape[-1])
    expected = torch.logaddexp(
        log_ps + torch.nn.functional.logsigmoid(-read.raw_gate),
        log_pcopy + torch.nn.functional.logsigmoid(read.raw_gate),
    )
    # legacy uses the legacy gate as copy mass and all remaining mass for its
    # semantic vocabulary branch when source evidence is present.
    assert torch.allclose(result.log_p, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(result.log_ps, log_p0)


def test_routes_use_simplex_and_empty_source_falls_back_to_base():
    state, read = _copy_inputs(active=False)
    read = CopyRead(
        read.query.expand(1, 2, -1),
        read.log_attention.expand(1, 2, -1),
        read.raw_gate.expand(1, 2, -1),
        read.g.expand(1, 2, -1),
        read.active,
    )
    hidden = torch.randn(1, 2, 6)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3, alpha_max=0.2, alpha_init=0.05)
    diagnostics = {"q": torch.zeros(1, 2, 3), "u": torch.zeros(1, 2, 3), "evidence": torch.zeros(1, 2, 1)}
    result = readout(hidden, hidden, diagnostics, read, state, lm_head)
    assert torch.allclose(result.pi_copy, torch.zeros_like(result.pi_copy))
    assert torch.allclose(result.pi_sem, torch.zeros_like(result.pi_sem))
    assert torch.allclose(result.pi_base, torch.ones_like(result.pi_base))
    assert torch.allclose(result.output_logits, lm_head(hidden), rtol=1e-5, atol=1e-5)


def test_copy_and_semantic_mass_sum_to_one_with_cap():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 2, 6, requires_grad=True)
    hs = hidden + 0.1 * torch.randn_like(hidden)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3, alpha_max=0.2, generate_reserve=0.05, alpha_init=0.05)
    evidence = torch.ones(1, 2, 1)
    result = readout(
        hidden,
        hs,
        {"q": torch.randn(1, 2, 3), "u": torch.randn(1, 2, 3), "evidence": evidence},
        CopyRead(
            read.query.expand(1, 2, -1),
            read.log_attention.expand(1, 2, -1),
            read.raw_gate.expand(1, 2, -1),
            read.g.expand(1, 2, -1),
            read.active,
        ),
        state,
        lm_head,
    )
    assert torch.allclose((result.pi_base + result.pi_sem + result.pi_copy), torch.ones_like(result.pi_base), atol=1e-6)
    assert bool((result.pi_sem <= 1.0 - result.g.detach() - 0.05 + 1e-6).all())
    labels = torch.tensor([[2, 3]])
    loss_sum, count = mixture_nll_from_state(result, labels)
    loss_sum.backward()
    assert count.item() == 2
    assert torch.isfinite(loss_sum)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_gauge_is_scalar_and_softmax_recovers_final_mixture():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 1, 6)
    result = DualReadout(6, rank=3)(
        hidden,
        hidden + 0.01 * torch.randn_like(hidden),
        {"q": torch.zeros(1, 1, 3), "u": torch.zeros(1, 1, 3), "evidence": torch.ones(1, 1, 1)},
        read,
        state,
        torch.nn.Linear(6, 8, bias=False),
    )
    assert torch.allclose(result.output_logits - result.log_p, result.gauge, atol=1e-6)
    assert torch.allclose(result.output_logits.softmax(-1), result.probability, rtol=1e-5, atol=1e-6)
    scores = result.output_logits[:, 0].clone()
    _apply_repetition_penalty(scores, torch.tensor([[0, 1]]), 1.05)
    assert torch.isfinite(scores).all()


def test_live_copy_gate_jacobian_is_copy_minus_base():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 1, 6)
    live_gate = torch.full((1, 1, 1), 0.20, requires_grad=True)
    read = CopyRead(read.query, read.log_attention, read.raw_gate.detach(), live_gate, read.active)
    result = DualReadout(6, rank=3)(
        hidden,
        hidden + 0.01 * torch.randn_like(hidden),
        {"q": torch.zeros(1, 1, 3), "u": torch.zeros(1, 1, 3), "evidence": torch.ones(1, 1, 1)},
        read,
        state,
        torch.nn.Linear(6, 8, bias=False),
    )
    derivative = torch.autograd.grad(result.probability[..., 3].sum(), live_gate)[0]
    expected = result.copy_log_prob.exp()[..., 3:4] - result.log_p0.exp()[..., 3:4]
    assert torch.allclose(derivative, expected, rtol=1e-5, atol=1e-6)


def test_independent_control_preserves_copy_prior_and_base_floor():
    state, read = _copy_inputs()
    read.g = torch.full_like(read.g, 0.7)
    hidden = torch.randn(1, 1, 6)
    readout = DualReadout(
        6,
        rank=3,
        mode="independent_capped_simplex",
        alpha_max=0.2,
        alpha_init=0.05,
        base_floor=0.05,
    )
    result = readout(
        hidden,
        hidden,
        {"q": torch.zeros(1, 1, 3), "u": torch.zeros(1, 1, 3), "evidence": torch.ones(1, 1, 1)},
        read,
        state,
        torch.nn.Linear(6, 8, bias=False),
    )
    assert torch.allclose(result.pi_base + result.pi_sem + result.pi_copy, torch.ones_like(result.pi_base), atol=1e-6)
    assert result.pi_copy.item() > result.pi_sem.item()
    assert result.pi_base.item() >= 0.05


def test_hidden_interpolation_control_is_observable():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 1, 6)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3, mode="hidden_interpolation", hidden_lambda=1.0)
    zero = readout.hidden_interpolation_state(hidden, torch.zeros_like(hidden), read, state, lm_head)
    shifted = readout.hidden_interpolation_state(hidden, torch.ones_like(hidden), read, state, lm_head)
    assert not torch.allclose(zero.output_logits, shifted.output_logits)


def test_dense_and_shared_loss_kernel_agree():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 3, 6)
    result = DualReadout(6, rank=3)(
        hidden,
        hidden,
        {"q": torch.zeros(1, 3, 3), "u": torch.zeros(1, 3, 3), "evidence": torch.ones(1, 3, 1)},
        CopyRead(
            read.query,
            read.log_attention.expand(1, 3, -1),
            read.raw_gate.expand(1, 3, -1),
            read.g.expand(1, 3, -1),
            read.active,
        ),
        state,
        torch.nn.Linear(6, 8, bias=False),
    )
    labels = torch.tensor([[2, -100, 5]])
    loss_sum, count = mixture_nll_from_state(result, labels)
    expected = -result.log_p[0, [0, 2], :].gather(1, labels[0, [0, 2], None]).sum()
    assert count.item() == 2
    assert torch.allclose(loss_sum, expected)


def test_chunked_vocabulary_loss_matches_dense_oracle():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 3, 6, requires_grad=True)
    hs = hidden + 0.01 * torch.randn_like(hidden)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3)
    diagnostics = {"q": torch.randn(1, 3, 3), "u": torch.randn(1, 3, 3), "evidence": torch.ones(1, 3, 1)}
    expanded = CopyRead(
        read.query.detach(),
        read.log_attention.detach().expand(1, 3, -1),
        read.raw_gate.detach().expand(1, 3, -1),
        read.g.detach().expand(1, 3, -1),
        read.active,
    )
    dense = readout(hidden, hs, diagnostics, expanded, state, lm_head)
    labels = torch.tensor([[2, 6, 5]])
    dense_sum, dense_count = mixture_nll_from_state(dense, labels)
    chunk_sum, chunk_count = readout.loss_sum_from_hidden(
        hidden, hs, diagnostics, expanded, state, lm_head, labels, chunk_size=3
    )
    assert dense_count.item() == chunk_count.item() == 3
    assert torch.allclose(dense_sum, chunk_sum, rtol=1e-5, atol=1e-5)


def test_chunked_vocabulary_loss_matches_dense_gradients():
    state, read = _copy_inputs()
    hidden = torch.randn(1, 3, 6, requires_grad=True)
    hs = hidden + 0.01 * torch.randn_like(hidden)
    lm_head = torch.nn.Linear(6, 8, bias=False)
    readout = DualReadout(6, rank=3)
    diagnostics = {"q": torch.randn(1, 3, 3), "u": torch.randn(1, 3, 3), "evidence": torch.ones(1, 3, 1)}
    expanded = CopyRead(
        read.query.detach(),
        read.log_attention.detach().expand(1, 3, -1),
        read.raw_gate.detach().expand(1, 3, -1),
        read.g.detach().expand(1, 3, -1),
        read.active,
    )
    labels = torch.tensor([[2, 6, 5]])
    dense = readout(hidden, hs, diagnostics, expanded, state, lm_head)
    dense_sum, _ = mixture_nll_from_state(dense, labels)
    dense_sum.backward()
    dense_hidden_grad = hidden.grad.detach().clone()
    dense_head_grad = lm_head.weight.grad.detach().clone()
    hidden.grad.zero_()
    lm_head.weight.grad.zero_()
    readout.zero_grad(set_to_none=True)
    chunk_sum, _ = readout.loss_sum_from_hidden(hidden, hs, diagnostics, expanded, state, lm_head, labels, chunk_size=3)
    chunk_sum.backward()
    assert torch.allclose(hidden.grad, dense_hidden_grad, rtol=1e-5, atol=1e-6)
    assert torch.allclose(lm_head.weight.grad, dense_head_grad, rtol=1e-5, atol=1e-6)
