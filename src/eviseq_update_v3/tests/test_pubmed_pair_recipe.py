"""Exercise the real shell config generator with CUDA/train/eval calls stubbed."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from eviseq_update_v3.config import load_config


@pytest.mark.parametrize(
    "variant,enabled,attention,cap",
    [
        ("independent_bounded", True, "independent_source", 0.1),
        ("shared_v1", True, "shared_copy", None),
        ("shared_bounded", True, "shared_copy", 0.1),
        ("independent_unbounded", True, "independent_source", None),
        ("independent_bounded", False, "independent_source", 0.1),
        ("hierarchical_coverage", True, "hierarchical_coverage", 0.1),
    ],
)
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("query_gate", [False, True])
def test_pair_generates_fair_protocol_and_selected_graph(
    tmp_path, variant, enabled, attention, cap, workers, query_gate
):
    # Only GPU work is stubbed. The script's Python YAML generator really runs.
    wrapper = tmp_path / "python-stub"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$#" -eq 1 && "$1" == - ]]; then cat >/dev/null; exit 0; fi\n'
        'if [[ "$1" != - ]]; then exit 0; fi\n'
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    wrapper.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation", "test"):
        (data / f"{split}.jsonl").write_text('{"id":"1","text":"alpha","summary":"beta"}\n')
    env = dict(os.environ)
    for key in (
        "GRADIENT_ACCUMULATION_STEPS",
        "BATCH_SIZE",
        "MAX_GRAD_NORM",
        "ROUGE155_SCRIPT",
        "PARTITION_HEADS",
        "SEMANTIC_HEAD_GATE_POSITION",
    ):
        env.pop(key, None)
    env.update(
        PYTHON=str(wrapper),
        NPROC_PER_NODE=str(workers),
        RUN_ROOT=str(tmp_path / "runs"),
        LOG_DIR=str(tmp_path / "logs"),
        PPLX_ENCODER=str(tmp_path),
        DECODER_MODEL=str(tmp_path),
        QWEN_ENCODER=str(tmp_path / "unused-encoder"),
        PROCESSED_DATA_DIR=str(data),
        RUN_ENCODERS="pplx",
        AFMR_ARCHITECTURE="afmr_value_anchor",
        AFMR_GROUNDED_COPY="true",
        AFMR_SEMANTIC_READ=str(enabled).lower(),
        AFMR_SEMANTIC_VARIANT=variant,
        CROSS_QUERY_GATE=str(query_gate).lower(),
        SEMANTIC_HEADS="1" if attention == "shared_copy" else "4",
    )
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/run_pubmed_pair.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = load_config(tmp_path / "runs/configs/pplx.yaml")
    training = config["training"]
    assert training["batch_size"] == 48
    assert training["batch_size"] * workers * training["gradient_accumulation_steps"] == 96
    assert training["max_grad_norm"] == 1.0
    assert training["interface_warmup_epochs"] + training["full_finetune_epochs"] == 4
    assert training.get("lr_scheduler", "linear") == "linear"
    assert not training.get("evidence_contrastive", {}).get("enabled", False)
    assert config["decoder"]["query_cross_gate"] == query_gate
    semantic = config["decoder"]["grounded_copy"]["semantic_read"]
    assert semantic["enabled"] == enabled and semantic["attention"] == attention
    assert semantic["max_relative_rms"] == cap
    assert semantic["num_heads"] == (1 if attention == "shared_copy" else 4)
    assert semantic["rank"] == 512
    assert semantic["head_gate_position"] == "post_norm"
    assert semantic["planner"]["partition_heads"] is False
    assert semantic["fusion"] == ("norm_preserving" if attention == "hierarchical_coverage" else "residual")
    assert "Effective batch: 96" in result.stdout
    assert not (tmp_path / "runs/configs/qwen_embedding.yaml").exists()


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"CROSS_QUERY_GATE": "yes"}, "CROSS_QUERY_GATE must be true or false"),
        ({"SEMANTIC_HEADS": "2"}, "SEMANTIC_HEADS must be 1 or 4"),
        ({"SEMANTIC_HEAD_GATE_POSITION": "middle"}, "SEMANTIC_HEAD_GATE_POSITION must be pre_norm or post_norm"),
        ({"AFMR_SEMANTIC_VARIANT": "shared_bounded", "SEMANTIC_HEADS": "4"}, "require SEMANTIC_HEADS=1"),
    ],
)
def test_pair_rejects_invalid_v3_ablation_before_launch(overrides, error):
    env = dict(os.environ, CROSS_QUERY_GATE="true", SEMANTIC_HEADS="4", AFMR_SEMANTIC_VARIANT="independent_bounded")
    env.update(overrides)
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/run_pubmed_pair.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert error in result.stderr
