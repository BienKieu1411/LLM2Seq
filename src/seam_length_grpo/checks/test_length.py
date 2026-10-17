import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from eviseq_afmr.modeling.model import EviSeqAFMR
from fixtures import fixture
from seam_grpo.data import trajectory
from seam_grpo.policy import Policy
from seam_grpo.rollout import rollout
from seam_length_grpo.data import LengthDataset, Negatives
from seam_length_grpo.policy import LengthPolicy, unpad_row
from seam_length_grpo.reward import LengthReward, count_words
from seam_length_grpo.teacher import SourceTeacher


def test_word_counter_matches_actual_evaluator():
    path = Path(__file__).resolve().parents[2] / "evaluation/evaluate_length_vietnamese.py"
    spec = importlib.util.spec_from_file_location("length_eval", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for text in ("công nghệ thông tin", "tiếng Việt", "doesn't 1.2% 🙂", ""):
        assert count_words(text) == module.count_words(text)


def test_absolute_length_cost_active_even_at_zero_quality():
    reward = LengthReward({"length_weight": 0.1})
    short = reward("unrelated", "gold", 3, 2)
    exact = reward("three unrelated words", "gold", 3, 2)
    assert exact["quality"] == short["quality"] == 0
    assert exact["reward"] > short["reward"]
    assert short["length_cost"] == 1


def test_dynamic_prefix_copy_and_source_span_preserved():
    cfg, tok, length, rows, collator = fixture()
    batch = collator(rows)
    assert batch["decoder_prompt_mask"].sum(-1).tolist() == [4, 5]
    assert batch["labels"][0, :5].eq(-100).all()
    assert batch["sources"][0] == "alpha beta gamma delta"
    same = {**rows[0], "target_words": 12}
    collator.include_targets = False
    other = collator([same])
    assert other["sources"][0] == batch["sources"][0]
    assert other["encoder_prompts"][0] != batch["encoder_prompts"][0]
    for row in range(2):
        indices = batch["copy_encoder_indices"][row][batch["copy_alignment_weights"][row].gt(0)]
        assert batch["source_content_mask"][row][indices].all()
    collator.include_targets = True
    with pytest.raises(ValueError, match="gold CE"):
        collator([same])
    collator.data["max_target_length"] = 2
    collator.cache.clear()
    with pytest.raises(ValueError, match="truncated"):
        collator(rows)


def test_ragged_logps_and_full_actor_gradients_match_unpadded():
    cfg, tok, length, rows, collator = fixture()
    batch = collator(rows)
    seam = EviSeqAFMR(cfg).eval()
    expected = copy.deepcopy(seam)
    actions = torch.tensor([[4, 5, 2], [6, 7, 2]])
    packed = trajectory(batch, actions, torch.ones_like(actions, dtype=torch.bool), 4)
    actual = LengthPolicy(seam, chunk_size=1)(packed, temperature=0.8)
    target = torch.cat([Policy(expected, chunk_size=1)(unpad_row(packed, i), temperature=0.8) for i in range(2)])
    torch.testing.assert_close(actual, target, rtol=1e-6, atol=1e-6)
    weights = torch.tensor([[1.0, 0.5, 1.0], [-1.0, -0.5, -1.0]])
    (actual * weights).sum().backward()
    (target * weights).sum().backward()
    for (name, p), (other, q) in zip(seam.named_parameters(), expected.named_parameters(), strict=True):
        assert name == other
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=3e-5, msg=name)
    for prefix in ("encoder.", "bridge.", "decoder.grounded_copy.", "decoder.backbone."):
        grads = [p.grad for n, p in seam.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0, prefix


def test_variable_prompt_rollout_scoring_parity():
    cfg, tok, length, rows, collator = fixture()
    batch = collator(rows)
    seam = EviSeqAFMR(cfg).eval()
    policy = LengthPolicy(seam, chunk_size=1)
    for i in range(2):
        row = unpad_row(batch, i)
        result = rollout(seam, row, tok, 4, 0.8)
        packed = trajectory(row, result["action_ids"], result["action_mask"], 4)
        scored = policy(packed, temperature=0.8)
        torch.testing.assert_close(scored, result["old_logps"], rtol=2e-5, atol=2e-5)


def test_batched_equal_width_candidates_keep_exact_behavior_scores():
    from seam_length_grpo.policy import prompt_groups, unpad_rows

    cfg, tok, length, rows, collator = fixture()
    batch = collator([rows[0], rows[0], rows[1], rows[1]])
    seam = EviSeqAFMR(cfg).eval()
    policy = LengthPolicy(seam, chunk_size=1)
    assert list(prompt_groups(batch, 4)) == [[0, 1], [2, 3]]
    for indices in prompt_groups(batch, 4):
        source = unpad_rows(batch, indices)
        result = rollout(seam, source, tok, 4, 0.8)
        packed = trajectory(source, result["action_ids"], result["action_mask"], 4)
        torch.testing.assert_close(policy(packed, temperature=0.8), result["old_logps"], rtol=2e-5, atol=2e-5)


def test_frozen_teacher_margin_reverses_when_sources_swap():
    from seam_grpo.actor import SEAMActor
    from seam_length_grpo.data import LengthCollator

    cfg, tok, length, rows, collator = fixture()
    actor = SEAMActor(EviSeqAFMR(cfg))
    actor.policy = LengthPolicy(actor.policy.seam)
    actor.requires_grad_(False).eval()
    frozen = {n: p.clone() for n, p in actor.named_parameters()}
    negative = rows[1]["prompt"]
    first = {**rows[0], "negative_source": negative, "negative_id": "b"}
    no_gold = LengthCollator((tok, tok), cfg["data"], length, grounded_copy=True, include_targets=False)
    teacher = SourceTeacher(actor, no_gold, {"tau": 1.0}, max_new_tokens=4, micro_batch_size=1)
    ids = [tok("alpha beta")["input_ids"] + [2]]
    score, margin = teacher(no_gold([first]), [first], ids)
    reverse = {**first, "prompt": negative, "negative_source": rows[0]["prompt"], "negative_id": "a"}
    reverse_score, reverse_margin = teacher(no_gold([reverse]), [reverse], ids)
    assert abs(margin[0]) > 1e-5
    assert margin[0] == pytest.approx(-reverse_margin[0], abs=2e-6)
    assert score[0] == pytest.approx(-reverse_score[0], abs=2e-6)
    for name, p in actor.named_parameters():
        assert p.grad is None and not p.requires_grad
        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)


def test_external_requests_do_not_derive_test_length(tmp_path):
    cfg, tok, length, rows, collator = fixture()
    path = tmp_path / "test.jsonl"
    path.write_text(
        json.dumps({"id": "a", "text": "source text", "summary": "gold text", "target_words": False}) + "\n"
    )
    dataset = LengthDataset(path, cfg["data"], length, derive_words=False, requested_words=100)
    assert dataset[0]["target_words"] == 100 and dataset[0]["word_request_origin"] == "external_cli"
    with pytest.raises(ValueError, match="target_words"):
        LengthDataset(path, cfg["data"], length, derive_words=False)


def test_negatives_match_visible_source_and_cover_ordered_domains():
    from seam_length_grpo.config import SOURCE_DEFAULTS

    cfg, tok, length, rows, collator = fixture()
    examples = []
    for domain in ("first", "second"):
        for i, text in enumerate(
            ("same first four words tail_a", "same first four words tail_b", "other distant short text tail")
        ):
            examples.append({**rows[0], "id": f"{domain}{i}", "prompt": text, "dataset": domain, "language": "vi"})
    pool = Negatives(examples, {**SOURCE_DEFAULTS, "reservoir_size": 2}, 42, collator.visible_source)
    for row in examples:
        other = pool.choose(row)
        assert other["dataset"] == row["dataset"] and other["language"] == row["language"]
        assert collator.visible_source(other["prompt"]) != collator.visible_source(row["prompt"])


def test_budget_checks_before_generation():
    from seam_length_grpo.policy import validate_budgets

    cfg, tok, length, rows, collator = fixture()
    model = EviSeqAFMR(cfg)
    validate_budgets(model, collator, 4)
    collator.encoder_limit = 300
    with pytest.raises(ValueError, match="Encoder body"):
        validate_budgets(model, collator, 4)
