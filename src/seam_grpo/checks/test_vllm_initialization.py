"""Execute the real initialization/control path with no CUDA/model allocation."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from seam_grpo.config import DEFAULTS
from seam_grpo.data import prepare
from seam_grpo.vllm_rollout import VLLMRollout, rollout_token_budget
from test_policy import tiny


@pytest.mark.parametrize("world,rank,has_port", [(1, 0, True), (1, 0, False), (2, 0, True), (2, 1, True)])
def test_real_initializer_keeps_rank_local_device_and_rollout_contract(tmp_path, monkeypatch, world, rank, has_port):
    import seam_vllm

    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(key, "unused")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    if has_port:
        monkeypatch.setenv("MASTER_PORT", "29513")
    else:
        monkeypatch.delenv("MASTER_PORT", raising=False)
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(seam_vllm, "SEAMLLM", lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace())
    cfg, _, _ = tiny()
    cfg["data"]["detokenize"] = True
    cfg["data"]["max_source_length"] = 8192
    actor = SimpleNamespace(
        policy=SimpleNamespace(
            seam=SimpleNamespace(decoder=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=256)))
        )
    )
    barriers = []
    accelerator = SimpleNamespace(
        device=torch.device("cuda", rank),
        process_index=rank,
        local_process_index=rank,
        num_processes=world,
        is_main_process=rank == 0,
        unwrap_model=lambda x: x,
        wait_for_everyone=lambda: barriers.append(True),
    )
    path = tmp_path / "rollout_config.yaml"
    if rank:
        path.write_text("data:\n  detokenize: false\n")
    trainer = SimpleNamespace(
        accelerator=accelerator,
        model=actor,
        base_config=cfg,
        settings={**DEFAULTS, "max_new_tokens": 4},
        collator=SimpleNamespace(_prompt_ids=[1, 42]),
        args=SimpleNamespace(output_dir=str(tmp_path)),
        sft_checkpoint="local_sft.pt",
    )
    VLLMRollout(trainer).initialize()
    args, options = calls[0]
    assert args == (path, "local_sft.pt")
    assert options["distributed_executor_backend"] == "external_launcher"
    assert options["logprobs_mode"] == "processed_logprobs"
    assert options["mm_processor_cache_gb"] == 0
    assert options["max_model_len"] == 6
    assert options["max_num_batched_tokens"] == 8192
    assert trainer.settings["vllm_max_num_batched_tokens"] == DEFAULTS["vllm_max_num_batched_tokens"]
    assert options["seed"] == DEFAULTS["seed"] + rank
    assert len(barriers) == 1
    import os

    assert os.environ["RANK"] == os.environ["LOCAL_RANK"] == str(rank)
    assert os.environ["WORLD_SIZE"] == str(world)
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert int(os.environ["MASTER_PORT"]) > 0
    assert cfg["data"]["detokenize"] is True
    assert yaml.safe_load(path.read_text())["data"]["detokenize"] is False


@pytest.mark.parametrize(
    "source,requested,decoder,sequences,expected",
    [
        (8192, 2048, 514, 64, 8192),
        (8192, 16384, 514, 64, 16384),
        (8320, 8192, 704, 64, 8320),
        (512, 2048, 4096, 64, 4096),
        (16, 4, 6, 64, 64),
    ],
)
def test_scheduler_budget_fits_unchunked_inputs(source, requested, decoder, sequences, expected, caplog):
    trainer = SimpleNamespace(
        base_config={"data": {"max_source_length": source}},
        settings={"vllm_max_num_batched_tokens": requested, "vllm_max_num_seqs": sequences},
    )
    assert rollout_token_budget(trainer, decoder) == expected
    assert trainer.settings["vllm_max_num_batched_tokens"] == requested
    assert ("Raising vllm_max_num_batched_tokens" in caplog.text) == (expected != requested)


def test_multirank_initializer_rejects_missing_shared_port(tmp_path, monkeypatch):
    monkeypatch.delenv("MASTER_PORT", raising=False)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(key, "unused")
    accelerator = SimpleNamespace(
        device=torch.device("cuda", 1), process_index=1, local_process_index=1, num_processes=2
    )
    with pytest.raises(ValueError, match="shared MASTER_PORT"):
        VLLMRollout(SimpleNamespace(accelerator=accelerator)).initialize()


def test_recreated_engine_must_sync_even_at_same_training_step(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    syncs, shutdowns = [], []
    trainer = SimpleNamespace(
        state=SimpleNamespace(global_step=7),
        settings={**DEFAULTS, "max_new_tokens": 4},
        accelerator=SimpleNamespace(unwrap_model=lambda model: model),
        model=SimpleNamespace(policy=SimpleNamespace(seam=object())),
        decoder_tokenizer=SimpleNamespace(eos_token_id=2),
    )

    def initialize(self):
        completion = SimpleNamespace(token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-0.5)}])
        self.engine = SimpleNamespace(
            sync_weights=lambda actor, policy_version: syncs.append(policy_version),
            generate=lambda *a: [SimpleNamespace(prompt_token_ids=[1], outputs=[completion])],
            engine=SimpleNamespace(
                llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: shutdowns.append(True)))
            ),
        )

    monkeypatch.setattr(VLLMRollout, "initialize", initialize)
    engine = VLLMRollout(trainer)
    source = {"decoder_prompt_ids": torch.tensor([[1]])}
    engine(["source"], source)
    engine(["source"], source)
    assert syncs == [7]
    engine.close()
    engine(["source"], source)
    assert syncs == [7, 7] and shutdowns == [True]


def settings_for_paths(tmp_path):
    cfg, _, _ = tiny()
    for name in ("encoder", "decoder"):
        cfg["model"][f"{name}_name"] = f"backbones/{name}"
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"text": "source", "summary": "gold"}) + "\n")
        cfg["data"][f"{split}_file"] = str(path)
    cfg.pop("_meta", None)
    config = tmp_path / "sft.yaml"
    config.write_text(yaml.safe_dump(cfg))
    checkpoint = tmp_path / "sft.pt"
    checkpoint.touch()
    return {
        "base_config": str(config),
        "sft_checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "run"),
        "training": {**DEFAULTS, "max_new_tokens": 4},
    }


@pytest.mark.parametrize("instruction", ["", "Summarize:"])
def test_bos_less_decoder_gets_start_metadata_in_python_and_http_launchers(tmp_path, monkeypatch, instruction):
    import sys

    import transformers
    from eviseq_afmr.runtime import _TinyTokenizer
    from seam_vllm import api, serve
    from seam_vllm._prompt import configured_decoder_prompt_ids

    settings = settings_for_paths(tmp_path)
    for name in ("encoder", "decoder"):
        (tmp_path / "backbones" / name).mkdir(parents=True)
    config_path = Path(settings["base_config"])
    config = yaml.safe_load(config_path.read_text())
    config["data"].update(decoder_prompt=instruction, decoder_prefix="", decoder_chat_template=False)
    config_path.write_text(yaml.safe_dump(config))
    before = config_path.read_bytes()
    tokenizer = _TinyTokenizer()
    tokenizer.bos_token_id = None
    tokenizer.is_fast = True
    expected = configured_decoder_prompt_ids(tokenizer, config["data"])
    captured = []

    def create_engine(**options):
        start = options["hf_overrides"].get("decoder_start_token_id")
        if start is None and tokenizer.bos_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")
        captured.append(options)
        return SimpleNamespace()

    monkeypatch.setattr(api, "ensure_source_plugin", lambda: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="qwen3")
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.25.1", LLM=create_engine))
    model = api.SEAMLLM(config_path, settings["sft_checkpoint"])
    assert captured[0]["hf_overrides"]["decoder_start_token_id"] == expected[0]
    assert model.prompt_token_ids == expected
    args = SimpleNamespace(
        served_model_name="seam",
        host="127.0.0.1",
        port=8000,
        max_num_seqs=16,
        gpu_memory_utilization=0.2,
        max_model_len=32,
        max_num_batched_tokens=8192,
        api_key=None,
    )
    command = serve.build_command(config_path, Path(settings["sft_checkpoint"]), args)
    assert json.loads(command[command.index("--hf-overrides") + 1])["decoder_start_token_id"] == expected[0]
    assert tokenizer.bos_token_id is None
    assert config_path.read_bytes() == before


def test_model_paths_resolve_from_sft_config_and_reject_ambiguous_legacy_cwd(tmp_path, monkeypatch):
    settings = settings_for_paths(tmp_path)
    for name in ("encoder", "decoder"):
        (tmp_path / "backbones" / name).mkdir(parents=True)
    cwd = tmp_path / "caller"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    base = prepare(settings)
    for name in ("encoder", "decoder"):
        assert base["model"][f"{name}_name"] == str(tmp_path / "backbones" / name)
    (cwd / "backbones" / "encoder").mkdir(parents=True)
    with pytest.raises(ValueError, match="Ambiguous"):
        prepare(settings)


def test_cuda_requirement_is_checked_before_loading_any_backbones(monkeypatch):
    from seam_grpo import train as runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("A backbone was loaded before validating CUDA")

    monkeypatch.setattr(runtime, "EviSeqAFMR", forbidden)
    with pytest.raises(ValueError, match="requires CUDA"):
        runtime.train({"training": {"rollout_backend": "seam_vllm"}})
