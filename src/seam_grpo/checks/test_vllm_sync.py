"""Exercise production mapping/control hooks without CUDA or vLLM kernels."""

import copy
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from eviseq_afmr.modeling.model import EviSeqAFMR
from seam_grpo.vllm_rollout import completion_rows
from seam_vllm._actor_rpc import begin_actor_update, finish_actor_update, load_actor_chunk
from seam_vllm._actor_weights import ActorWeightUpdate
from seam_vllm.api import SEAMLLM
from test_policy import tiny
from torch import nn


def packed_runtime(native, *, padded=0):
    """A real torch parameter tree with vLLM's Qwen3 packed layout."""
    runtime = nn.Module()
    runtime.config = native.decoder.config
    runtime.seam_config = native.config
    runtime.encoder = copy.deepcopy(native.encoder)
    runtime.bridge = copy.deepcopy(native.bridge)
    runtime.copy_head = copy.deepcopy(native.decoder.grounded_copy)
    runtime.model = nn.Module()
    backbone = native.decoder.backbone
    runtime.model.embed_tokens = nn.Embedding(
        backbone.embed_tokens.num_embeddings + padded, backbone.embed_tokens.embedding_dim
    )
    runtime.model.norm = copy.deepcopy(backbone.norm)
    runtime.model.layers = nn.ModuleList()
    for layer in backbone.layers:
        block = nn.Module()
        block.input_layernorm = copy.deepcopy(layer.base.input_layernorm)
        block.post_attention_layernorm = copy.deepcopy(layer.base.post_attention_layernorm)
        block.self_attn = nn.Module()
        attn = layer.base.self_attn
        qkv = torch.cat([getattr(attn, f"{p}_proj").weight for p in ("q", "k", "v")])
        block.self_attn.qkv_proj = nn.Linear(qkv.shape[1], qkv.shape[0], bias=False)
        block.self_attn.q_norm = copy.deepcopy(attn.q_norm)
        block.self_attn.k_norm = copy.deepcopy(attn.k_norm)
        block.self_attn.o_proj = copy.deepcopy(attn.o_proj)
        block.mlp = nn.Module()
        mlp = layer.base.mlp
        gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight])
        block.mlp.gate_up_proj = nn.Linear(gu.shape[1], gu.shape[0], bias=False)
        block.mlp.down_proj = copy.deepcopy(mlp.down_proj)
        block.cross = copy.deepcopy(layer.cross)
        block.cross_norm = copy.deepcopy(layer.cross_norm)
        block.cross_gate = nn.Parameter(layer.cross_gate.detach().clone())
        runtime.model.layers.append(block)
    if runtime.config.tie_word_embeddings:
        runtime.lm_head = runtime.model.embed_tokens
    else:
        runtime.lm_head = nn.Linear(
            backbone.embed_tokens.embedding_dim, backbone.embed_tokens.num_embeddings + padded, bias=False
        )
    runtime._actor_update = None
    runtime._actor_version = -1
    runtime._completed_copy_states = [object()]
    runtime._pending_copy_state = object()
    runtime._has_copy_requests = True
    runtime.clears = 0

    def clear():
        runtime.clears += 1

    runtime.clear_copy_batch_cache = clear
    return runtime


def actor_and_runtime(*, tied=False, copy_enabled=True):
    cfg, _, _ = tiny()
    cfg["decoder"]["grounded_copy"]["enabled"] = copy_enabled
    actor = EviSeqAFMR(cfg)
    if tied:
        actor.decoder.lm_head.weight = actor.decoder.embed_tokens.weight
        actor.decoder.config.tie_word_embeddings = True
    runtime = packed_runtime(actor, padded=5)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.123)
    return actor, runtime


def worker_for(runtime):
    reqs = {"old-finished-request": 0}
    state = SimpleNamespace(
        copy_states={"old-finished-request": object()},
        encoder_outputs=[object()],
        _warmup_requests={"warmup"},
        _dummy_inputs_pending=True,
    )

    def remove(key):
        reqs.pop(key, None)
        state.copy_states.pop(key, None)

    runner = SimpleNamespace(
        get_model=lambda: runtime,
        execute_model_state=None,
        req_states=SimpleNamespace(req_id_to_index=reqs),
        model_state=state,
        _remove_request=remove,
        reset_encoder_cache=lambda: None,
    )
    return SimpleNamespace(model_runner=runner)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("copy_enabled", [False, True])
def test_full_actor_update_restores_all_routes_packed_shards_and_padded_vocab(tied, copy_enabled):
    actor, runtime = actor_and_runtime(tied=tied, copy_enabled=copy_enabled)
    weights = list(actor.named_parameters())
    update = ActorWeightUpdate(runtime, [(n, list(p.shape)) for n, p in weights], 3)
    for offset in range(0, len(weights), 3):
        update.load(weights[offset : offset + 3])
    assert update.finish() == {"policy_version": 3, "tensors": len(weights)}
    for name in ("encoder", "bridge"):
        for key, value in getattr(actor, name).state_dict().items():
            torch.testing.assert_close(value, getattr(runtime, name).state_dict()[key], rtol=0, atol=0)
    for native, packed in zip(actor.decoder.backbone.layers, runtime.model.layers, strict=True):
        torch.testing.assert_close(
            packed.self_attn.qkv_proj.weight,
            torch.cat([getattr(native.base.self_attn, f"{p}_proj").weight for p in ("q", "k", "v")]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            packed.mlp.gate_up_proj.weight,
            torch.cat([native.base.mlp.gate_proj.weight, native.base.mlp.up_proj.weight]),
            rtol=0,
            atol=0,
        )
        for key, value in native.cross.state_dict().items():
            torch.testing.assert_close(value, packed.cross.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(native.cross_gate, packed.cross_gate, rtol=0, atol=0)
    if copy_enabled:
        for key, value in actor.decoder.grounded_copy.state_dict().items():
            torch.testing.assert_close(value, runtime.copy_head.state_dict()[key], rtol=0, atol=0)
    vocab = actor.decoder.config.vocab_size
    torch.testing.assert_close(
        actor.decoder.embed_tokens.weight, runtime.model.embed_tokens.weight[:vocab], rtol=0, atol=0
    )
    assert runtime.model.embed_tokens.weight[vocab:].count_nonzero() == 0
    if runtime.config.tie_word_embeddings:
        assert runtime.lm_head is runtime.model.embed_tokens


def test_incomplete_shape_duplicate_and_nonfinite_updates_fail_closed():
    actor, runtime = actor_and_runtime()
    weights = list(actor.named_parameters())
    manifest = [(n, list(p.shape)) for n, p in weights]
    with pytest.raises(ValueError, match="cover"):
        ActorWeightUpdate(runtime, manifest[:-1], 0)
    bad = copy.deepcopy(manifest)
    bad[0] = (bad[0][0], [1])
    with pytest.raises(ValueError, match="shape"):
        ActorWeightUpdate(runtime, bad, 0)
    plan = ActorWeightUpdate(runtime, manifest, 0)
    plan.load(weights[:1])
    with pytest.raises(RuntimeError, match="Incomplete"):
        plan.finish()
    with pytest.raises(ValueError, match="repeated"):
        plan.load(weights[:1])
    with pytest.raises(RuntimeError, match="previously failed"):
        plan.finish()

    plan = ActorWeightUpdate(runtime, manifest, 0)
    with pytest.raises(ValueError, match="Nonfinite"):
        plan.load([(weights[0][0], torch.full_like(weights[0][1], float("nan")))])
    with pytest.raises(RuntimeError, match="previously failed"):
        plan.finish()


def test_offloaded_runtime_rejected_before_any_weight_copy():
    actor, runtime = actor_and_runtime()
    runtime.bridge.to("meta")
    with pytest.raises(ValueError, match="one runtime device"):
        ActorWeightUpdate(runtime, [(n, list(p.shape)) for n, p in actor.named_parameters()], 0)


def test_facade_transaction_clears_finished_request_state_and_commits_versions():
    actor, runtime = actor_and_runtime()
    worker = worker_for(runtime)
    facade = object.__new__(SEAMLLM)
    facade.config = actor.config
    facade._actor_lock = RLock()
    facade._actor_sync_supported = True
    facade._actor_sync_failed = False
    facade._actor_version = -1
    calls = []

    def rpc(method, args=()):
        calls.append(method.__name__)
        return [method(worker, *args)]

    facade.engine = SimpleNamespace(
        collective_rpc=rpc,
        reset_prefix_cache=lambda: True,
        reset_mm_cache=lambda: None,
        llm_engine=SimpleNamespace(has_unfinished_requests=lambda: False),
    )
    report = facade.sync_weights(actor, policy_version=0, chunk_bytes=512)
    assert report["policy_version"] == runtime._actor_version == 0
    assert not worker.model_runner.req_states.req_id_to_index
    assert not worker.model_runner.model_state.copy_states
    assert runtime._pending_copy_state is None and runtime._completed_copy_states == []
    assert calls[0] == "begin_actor_update" and calls[-1] == "finish_actor_update"
    assert calls.count("load_actor_chunk") > 1
    with pytest.raises(RuntimeError, match="stale"):
        facade.sync_weights(actor, policy_version=0)
    facade.sync_weights(actor, policy_version=1)
    assert runtime._actor_version == 1


def test_completion_conversion_keeps_eos_and_processed_sampled_logprobs():
    completion = SimpleNamespace(
        token_ids=[7, 2], logprobs=[{7: SimpleNamespace(logprob=-0.3)}, {2: SimpleNamespace(logprob=-0.5)}]
    )
    output = SimpleNamespace(prompt_token_ids=[1], outputs=[completion])
    rows = completion_rows([output], [[1]], eos_token_id=2, budget=4)
    assert rows == {"prompt_ids": [[1]], "completion_ids": [[7, 2]], "logprobs": [[-0.3, -0.5]]}
    with pytest.raises(RuntimeError, match="prompt"):
        completion_rows([output], [[3]], eos_token_id=2, budget=4)
    completion.token_ids = [7]
    completion.logprobs = completion.logprobs[:1]
    with pytest.raises(RuntimeError, match="stopping"):
        completion_rows([output], [[1]], eos_token_id=2, budget=4)


def test_grouped_completion_conversion_expands_n_candidates_per_unique_prompt():
    def completion(token, logp):
        return SimpleNamespace(
            token_ids=[token, 2], logprobs=[{token: SimpleNamespace(logprob=logp)}, {2: SimpleNamespace(logprob=-0.5)}]
        )

    outputs = [
        SimpleNamespace(prompt_token_ids=[1], outputs=[completion(7, -0.1), completion(8, -0.2)]),
        SimpleNamespace(prompt_token_ids=[3], outputs=[completion(9, -0.3), completion(6, -0.4)]),
    ]
    rows = completion_rows(outputs, [[1], [1], [3], [3]], eos_token_id=2, budget=4, group_size=2)
    assert rows["prompt_ids"] == [[1], [1], [3], [3]]
    assert [ids[0] for ids in rows["completion_ids"]] == [7, 8, 9, 6]
    with pytest.raises(RuntimeError, match="count"):
        completion_rows(outputs[:1], [[1], [1], [3], [3]], eos_token_id=2, budget=4, group_size=2)
    with pytest.raises(RuntimeError, match="share"):
        completion_rows(outputs, [[1], [3], [3], [3]], eos_token_id=2, budget=4, group_size=2)
