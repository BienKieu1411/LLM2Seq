"""Exercise the real shell config generator with CUDA/train/eval calls stubbed."""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from eviseq_update.config import load_config


@pytest.mark.parametrize(
    "variant,enabled,attention,cap",
    [
        ("independent_bounded", True, "independent_source", 0.1),
        ("shared_v1", True, "shared_copy", None),
        ("shared_bounded", True, "shared_copy", 0.1),
        ("independent_unbounded", True, "independent_source", None),
        ("independent_bounded", False, "independent_source", 0.1),
    ],
)
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("recipe", ["ce", "cosine", "dropout"])
def test_pair_generates_fair_protocol_and_selected_graph(tmp_path, variant, enabled, attention, cap, workers, recipe):
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
    for key in ("GRADIENT_ACCUMULATION_STEPS", "BATCH_SIZE", "MAX_GRAD_NORM", "ROUGE155_SCRIPT"):
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
        AFMR_TRAINING_RECIPE=recipe,
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
    assert "neftune_noise_alpha" not in training
    assert training["lr_scheduler"] == ("linear" if recipe == "ce" else "cosine")
    assert config["decoder"]["attention_dropout"] == (0.1 if recipe == "dropout" else 0.0)
    assert config["generation"]["do_sample"] is False
    semantic = config["decoder"]["grounded_copy"]["semantic_read"]
    assert semantic["enabled"] == enabled and semantic["attention"] == attention
    assert semantic["max_relative_rms"] == cap
    assert "Effective batch: 96" in result.stdout
    assert not (tmp_path / "runs/configs/qwen_embedding.yaml").exists()


def test_pair_evidence_recipe_selects_encoder_specific_sidecar(tmp_path):
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
    env = dict(
        os.environ,
        PYTHON=str(wrapper),
        NPROC_PER_NODE="2",
        RUN_ROOT=str(tmp_path / "runs"),
        LOG_DIR=str(tmp_path / "logs"),
        PPLX_ENCODER=str(tmp_path),
        DECODER_MODEL=str(tmp_path),
        PROCESSED_DATA_DIR=str(data),
        RUN_ENCODERS="pplx",
        AFMR_ARCHITECTURE="afmr_value_anchor",
        AFMR_GROUNDED_COPY="true",
        AFMR_SEMANTIC_READ="true",
        AFMR_SEMANTIC_VARIANT="independent_bounded",
        AFMR_TRAINING_RECIPE="evidence",
    )
    for key in ("GRADIENT_ACCUMULATION_STEPS", "BATCH_SIZE", "MAX_GRAD_NORM", "ROUGE155_SCRIPT"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "scripts/run_pubmed_pair.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = load_config(tmp_path / "runs/configs/pplx.yaml")
    evidence = config["training"]["evidence_contrastive"]
    assert evidence["enabled"] is True and evidence["mode"] == "both"
    assert evidence["cache_path"] == str(tmp_path / "runs/evidence/pplx/evidence.jsonl")
