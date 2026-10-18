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


def test_seam_vllm_state_hash_fallback():
    pytest.importorskip("vllm")
    from seam_vllm.state import SEAMModelState

    state = SEAMModelState.__new__(SEAMModelState)
    state.copy_states = {}
    state.copy_states_by_hash = {}
    state._warmup_requests = set()
    state._dummy_inputs_pending = False
    state.model = SimpleNamespace(copy_head=object(), clear_copy_batch_cache=lambda: None)

    # Step 1: Request 1 encodes and populates copy_states and copy_states_by_hash
    state.copy_states["req_1"] = "copy_state_doc_1"
    state.copy_states_by_hash["mm_hash_doc_1"] = "copy_state_doc_1"

    # Step 2: Request 2 shares mm_hash_doc_1; vLLM encoder cache skips embed_multimodal
    assert "req_2" not in state.copy_states
    state.encoder_cache = SimpleNamespace(mm_features={"req_2": [SimpleNamespace(identifier="mm_hash_doc_1")]})

    input_batch = SimpleNamespace(req_ids=["req_2"], num_reqs=1, num_tokens_after_padding=8)

    # Calling prepare_inputs resolves req_2 from copy_states_by_hash fallback
    model_inputs = SEAMModelState.prepare_inputs(state, input_batch, None)
    assert model_inputs["copy_states"] == ["copy_state_doc_1"]
    assert state.copy_states["req_2"] == "copy_state_doc_1"

    # Step 3: reset_encoder_cache clears both caches
    state.reset_encoder_cache()
    assert len(state.copy_states) == 0
    assert len(state.copy_states_by_hash) == 0
