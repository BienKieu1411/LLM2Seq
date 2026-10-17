"""Control calls for a paused, in-process vLLM 0.25.1 actor update."""

from __future__ import annotations

from eviseq_afmr.training.checkpoint import architecture_spec

from ._actor_weights import ActorWeightUpdate


def begin_actor_update(worker, manifest, spec, version):
    runner = worker.model_runner
    model = runner.get_model()
    if architecture_spec(model.seam_config) != spec:
        raise ValueError("Actor architecture differs from the SEAM engine")
    if version <= getattr(model, "_actor_version", -1):
        raise ValueError("Actor policy version must increase monotonically")
    if getattr(model, "_actor_update", None) is not None:
        raise RuntimeError("Previous actor update is incomplete; rebuild the engine")
    state = runner.model_state
    if runner.execute_model_state is not None:
        raise RuntimeError("Actor sync requires a drained engine with no pending forward")
    plan = ActorWeightUpdate(model, manifest, version)
    model._actor_update = plan
    # LLM.generate has drained the scheduler. The V2 worker can still retain
    # the final request until the next scheduler step carries finished_req_ids.
    # Explicitly retire these now, after the facade's idle-engine assertion.
    for req_id in list(runner.req_states.req_id_to_index):
        runner._remove_request(req_id)
    model.clear_copy_batch_cache()
    model._completed_copy_states.clear()
    model._pending_copy_state = None
    model._has_copy_requests = False
    state.encoder_outputs = []
    state._warmup_requests.clear()
    state._dummy_inputs_pending = False
    runner.reset_encoder_cache()
    return {"tensors": len(plan.slots)}


def load_actor_chunk(worker, weights):
    plan = worker.model_runner.get_model()._actor_update
    if plan is None:
        raise RuntimeError("No actor update transaction")
    plan.load(weights)


def finish_actor_update(worker):
    model = worker.model_runner.get_model()
    plan = model._actor_update
    if plan is None:
        raise RuntimeError("No actor update transaction")
    result = plan.finish()
    model._actor_version = plan.version
    model._actor_update = None
    return result
