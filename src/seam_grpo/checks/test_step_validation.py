from types import SimpleNamespace

import pytest
from seam_grpo.config import DEFAULTS
from seam_grpo.lifecycle import Lifecycle


def _lifecycle(tmp_path, every):
    decoder = SimpleNamespace(clear_cross_cache=lambda: None)
    actor = SimpleNamespace(policy=SimpleNamespace(seam=SimpleNamespace(decoder=decoder)))
    accelerator = SimpleNamespace(is_main_process=False, unwrap_model=lambda model: actor)
    trainer = SimpleNamespace(
        accelerator=accelerator,
        model=None,
        rollout_history=[{"policy_version": 0}],
        settings={"validate_every_steps": every},
        _buffered_inputs=object(),
    )
    life = Lifecycle(trainer, None, None, None)
    phases = []
    life.measure = lambda phase, epoch: phases.append((phase, epoch))
    return life, phases, SimpleNamespace(output_dir=str(tmp_path))


@pytest.mark.parametrize("every, expected", [(0, []), (2, ["step_0000002", "step_0000004"])])
def test_validates_on_schedule_and_not_on_final_step(tmp_path, every, expected):
    life, phases, args = _lifecycle(tmp_path, every)
    for step in range(1, 7):
        state = SimpleNamespace(global_step=step, max_steps=6, epoch=step / 6)
        life.on_step_end(args, state, None)
    assert [phase for phase, _ in phases] == expected


def test_default_disables_step_validation():
    assert DEFAULTS["validate_every_steps"] == 0
