from types import SimpleNamespace

from seam_grpo.progress import TrainingProgress, duration


def test_eta_counts_optimizer_updates_and_excludes_startup_after_first_step(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("seam_grpo.progress.time.perf_counter", lambda: clock["now"])
    progress = TrainingProgress()
    state = SimpleNamespace(global_step=0, max_steps=10)
    assert progress.metrics(state) == {}
    progress.on_train_begin(None, state, None)
    progress.on_step_begin(None, state, None)
    progress.rollout_seconds = 90
    clock["now"] = 120
    state.global_step = 1
    progress.on_step_end(None, state, None)
    metrics = progress.metrics(state)
    assert metrics["eta"] == "00:18:00"
    assert metrics["rollout_reward_seconds"] == 90
    assert metrics["train_seconds"] == 30
    progress.on_step_begin(None, state, None)
    assert progress.rollout_seconds == 0
    clock["now"] = 140
    state.global_step = 2
    progress.on_step_end(None, state, None)
    metrics = progress.metrics(state)
    assert metrics["progress"] == "2/10" and metrics["eta"] == "00:02:40"
    assert metrics["elapsed"] == "00:02:20"
    state.global_step = 10
    assert progress.metrics(state)["eta"] == "00:00:00"
    assert duration(90061) == "25:01:01"
