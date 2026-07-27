from pathlib import Path

import pytest
from llm2seq_v5.config import load_config
from llm2seq_v5.evaluate import evaluate

ROOT = Path(__file__).parents[1]


def test_v2_core_control_is_protocol_matched_and_removes_v5_mechanisms():
    candidate = load_config(ROOT / "configs/pilot_phrase_continuation_2000.yaml")
    control = load_config(ROOT / "configs/pilot_ablations/v2_core_single_bank_2000.yaml")
    assert control["training"] == candidate["training"]
    assert control["data"] == candidate["data"]
    assert control["generation"] == candidate["generation"]
    assert control["limits"] == candidate["limits"]
    assert control["adapter"]["use_summary_planner"] is False
    assert control["adapter"]["depth_routed_memory"] is False
    assert control["decoder"]["memory_bank_count"] == 1
    assert control["phrase_pointer"]["enabled"] is False
    assert control["objectives"]["use_contrastive"] is False
    assert control["objectives"]["response_alignment_weight"] == 0.0
    assert control["objectives"]["phrase_mixture_weight"] == 0.0
    assert control["objectives"]["plan_only_probability_start"] == 0.0
    assert control["objectives"]["oracle_evidence_mix_start"] == 0.0


def test_pilot_config_cannot_evaluate_test_before_loading_models(tmp_path: Path):
    with pytest.raises(ValueError, match="disabled for smoke/pilot"):
        evaluate(
            str(ROOT / "configs/pilot_phrase_continuation_2000.yaml"),
            str(tmp_path / "last.pt"),
            str(tmp_path / "predictions.jsonl"),
            split="test",
        )


def test_partial_locked_test_evaluation_is_rejected_before_loading_models(tmp_path: Path):
    with pytest.raises(ValueError, match="must be evaluated in full"):
        evaluate(
            str(ROOT / "configs/qwen3_embedding_0_6b_phrase_continuation.yaml"),
            str(tmp_path / "last.pt"),
            str(tmp_path / "predictions.jsonl"),
            max_samples=20,
            split="test",
        )
