import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from fixtures import fixture
from seam_grpo.config import DEFAULTS
from seam_length_grpo.vllm_rollout import LengthVLLMRollout


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
def test_initializer_reserves_variable_decoder_prompts(tmp_path, monkeypatch, world, rank):
    import seam_vllm

    cfg, _, length, _, _ = fixture()
    calls = []
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("MASTER_PORT", "29517")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(seam_vllm, "SEAMLLM", lambda *a, **kw: calls.append((a, kw)))
    trainer = SimpleNamespace(
        accelerator=SimpleNamespace(
            device=torch.device("cuda", rank),
            process_index=rank,
            local_process_index=rank,
            num_processes=world,
            is_main_process=rank == 0,
            unwrap_model=lambda x: x,
            wait_for_everyone=lambda: None,
        ),
        model=SimpleNamespace(
            policy=SimpleNamespace(
                seam=SimpleNamespace(decoder=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=256)))
            )
        ),
        collator=SimpleNamespace(length=length),
        base_config=cfg,
        settings={**DEFAULTS, "max_new_tokens": 4},
        args=SimpleNamespace(output_dir=str(tmp_path)),
        sft_checkpoint="local.pt",
    )
    LengthVLLMRollout(trainer).initialize()
    assert calls[0][1]["max_model_len"] == length["decoder_prompt_reserve"] + 4
    assert calls[0][1]["mm_processor_cache_gb"] == 0
    assert calls[0][1]["distributed_executor_backend"] == "external_launcher"
    if rank == 0:
        assert yaml.safe_load((tmp_path / "rollout_config.yaml").read_text())["data"]["detokenize"] is False


def test_recreate_resynchronizes_entire_actor(monkeypatch):
    from eviseq_afmr.modeling.model import EviSeqAFMR
    from fake_vllm import NativeEngine
    from seam_grpo.actor import SEAMActor

    cfg, tokenizer, _, rows, collator = fixture()
    source = collator(rows)
    trainer = SimpleNamespace(
        model=SEAMActor(EviSeqAFMR(cfg)),
        state=SimpleNamespace(global_step=7),
        settings={**DEFAULTS, "max_new_tokens": 3},
        collator=collator,
        accelerator=SimpleNamespace(unwrap_model=lambda x: x),
        decoder_tokenizer=tokenizer,
        _likelihood_source=source,
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    engines = []

    def initialize(self):
        self.engine = NativeEngine(self.trainer)
        engines.append(self.engine)

    monkeypatch.setattr(LengthVLLMRollout, "initialize", initialize)
    engine = LengthVLLMRollout(trainer)
    result = engine([row["prompt"] for row in rows], source)
    assert len(result["completion_ids"]) == 2
    engine.close()
    engine([row["prompt"] for row in rows], source)
    assert [item.version for item in engines] == [7, 7]
