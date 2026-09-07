"""Real two-process checks of token-weighted DDP, stage changes and resume."""

import copy
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from eviseq_update.config import load_config
from eviseq_update.data.sampling import DistributedBatchSampler
from eviseq_update.distributed import rank, run_on_main, training_process_group
from eviseq_update.modeling.model import EviSeqAFMR
from eviseq_update.runtime import _write_resolved_config, build_loaders, evaluate, train
from eviseq_update.training.checkpoint import load_checkpoint, save_checkpoint
from eviseq_update.training.engine import AFMRTrainer, seed_everything
from eviseq_update.training.optimizer import _component, set_stage_trainability


@pytest.mark.parametrize("size", [1, 3, 5, 11, 29])
@pytest.mark.parametrize("world,batch", [(2, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("shuffle,bucket", [(False, False), (True, False), (True, True)])
def test_distributed_batches_cover_each_example_once(size, world, batch, shuffle, bucket):
    lengths = [(i * 17) % 23 for i in range(size)]
    samplers = [DistributedBatchSampler(lengths, batch, r, world, shuffle=shuffle, bucket=bucket) for r in range(world)]
    for epoch in (0, 1, 10):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        batches = [list(sampler) for sampler in samplers]
        assert len({len(rows) for rows in batches}) == 1
        assert all(len(rows) == len(samplers[0]) for rows in batches)
        assert all(1 <= len(row) <= batch for rows in batches for row in rows)
        real = [i for rows in batches for row in rows for i in row if i >= 0]
        assert sorted(real) == list(range(size))
        assert batches == [list(sampler) for sampler in samplers]


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, model, gradients):
        super().__init__((p for p in model.parameters() if p.requires_grad), lr=0.03)
        self.named = list(model.named_parameters())
        self.gradients = gradients

    def step(self, closure=None):
        self.gradients.append(
            {name: None if p.grad is None else p.grad.detach().cpu().clone() for name, p in self.named}
        )
        return super().step(closure)


def _sgd_run(config, distributed):
    seed_everything(91)
    model = EviSeqAFMR(config)
    # Exercise semantic gradients beyond its zero-initialized output layer.
    head = model.decoder.grounded_copy
    if head is not None and head.semantic_output is not None:
        with torch.no_grad():
            head.semantic_output.weight.normal_(std=0.01)
    loaders = build_loaders(config, distributed=distributed)
    trainer = AFMRTrainer(model, config, "cpu")
    metrics = []
    gradients = []
    for epoch, stage in enumerate(("interface_warmup", "full_finetune"), 1):
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        set_stage_trainability(model, stage)
        trainer._configure_distributed()
        loaders["train"].batch_sampler.set_epoch(epoch)
        optimizer = _RecordingSGD(model, gradients)
        metrics.append(trainer._run_epoch(loaders["train"], optimizer, stage, True))
        metrics.append(trainer._run_epoch(loaders["validation"], optimizer, stage, False))
        changed = {_component(name) for name, p in model.named_parameters() if not torch.equal(p, before[name])}
        assert changed == (
            {"bridge", "cross_attention"} if epoch == 1 else {"encoder", "decoder", "bridge", "cross_attention"}
        )
        if head is not None and head.semantic_output is not None:
            for branch in ("semantic_value", "semantic_output", "semantic_gate"):
                assert any(
                    branch in name and grad is not None and grad.abs().sum() > 0 for name, grad in gradients[-1].items()
                )
    return model, metrics, gradients


def _worker(config_path):
    torch.set_num_threads(1)
    config = load_config(config_path)
    root = Path(config_path).parent
    with training_process_group("cpu"):
        for mode in ("plain", "copy", "semantic"):
            case = copy.deepcopy(config)
            case["experiment"]["output_dir"] = str(root / mode)
            case["decoder"]["grounded_copy"]["enabled"] = mode != "plain"
            case["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = mode == "semantic"
            model, metrics, gradients = _sgd_run(case, True)
            torch.save(
                {"model": model.state_dict(), "metrics": metrics, "gradients": gradients},
                root / f"{mode}_rank{rank()}.pt",
            )

        # Exercise the public runtime, stage-specific optimizers, shared output
        # guards, atomic rank-zero checkpoints and resume from warmup.
        train(config_path, device="cpu")
        train(
            config_path,
            device="cpu",
            resume_checkpoint=str(root / "fit/epoch_001.pt"),
            output_dir_override=str(root / "resumed"),
        )

        random.seed(100 + rank())
        torch.manual_seed(200 + rank())
        save_checkpoint(root / "rng.pt", model, None, config, epoch=2, step=4)
        expected = (random.random(), torch.rand(3))
        load_checkpoint(root / "rng.pt", model, config=config)
        assert random.random() == expected[0]
        torch.testing.assert_close(torch.rand(3), expected[1], rtol=0, atol=0)

        def fail():
            raise OSError("simulated disk error")

        with pytest.raises(RuntimeError, match="simulated disk error"):
            run_on_main(fail)


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Requires Gloo")
def test_two_process_training_matches_serial_and_resumes(tmp_path):
    torch.set_num_threads(1)
    config = load_config(Path(__file__).parents[1] / "configs/afmr_smoke.yaml")
    config["experiment"]["output_dir"] = str(tmp_path / "fit")
    config["training"].update(
        batch_size=1,
        validation_batch_size=1,
        gradient_accumulation_steps=2,
        log_every_steps=1,
        max_grad_norm=None,
        save_best=True,
    )
    config["decoder"]["grounded_copy"]["enabled"] = True
    config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = True
    # Deliberately unequal target lengths, odd train/validation sizes and an
    # accumulation remainder: rank 1 must backpropagate a zero-label batch.
    rows = [
        {"id": str(i), "source": "alpha beta gamma delta " * (i + 1), "target": "alpha " * (i + 1)} for i in range(5)
    ]
    for split, count in (("train", 5), ("validation", 3), ("test", 2)):
        data_path = tmp_path / f"{split}.jsonl"
        data_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:count]))
        config["data"][f"{split}_file"] = str(data_path)
    _write_resolved_config(config, tmp_path)
    config_path = tmp_path / "resolved_config.yaml"
    environment = dict(os.environ, OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc_per_node=2",
            str(Path(__file__).resolve()),
            str(config_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    for mode in ("plain", "copy", "semantic"):
        case = copy.deepcopy(config)
        case["training"].update(batch_size=2, validation_batch_size=2)
        case["experiment"]["output_dir"] = str(tmp_path / f"serial_{mode}")
        case["decoder"]["grounded_copy"]["enabled"] = mode != "plain"
        case["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = mode == "semantic"
        expected_model, expected_metrics, expected_gradients = _sgd_run(case, False)
        states = [torch.load(tmp_path / f"{mode}_rank{r}.pt", weights_only=False) for r in (0, 1)]
        for step, expected in enumerate(expected_gradients):
            for name, gradient in expected.items():
                actual = states[0]["gradients"][step][name]
                peer = states[1]["gradients"][step][name]
                if gradient is None:
                    assert actual is peer is None, name
                else:
                    torch.testing.assert_close(actual, peer, rtol=0, atol=0, msg=name)
                    torch.testing.assert_close(actual, gradient, rtol=2e-5, atol=2e-6, msg=name)
        for name, parameter in expected_model.state_dict().items():
            torch.testing.assert_close(states[0]["model"][name], states[1]["model"][name], rtol=0, atol=0)
            torch.testing.assert_close(states[0]["model"][name], parameter, rtol=1e-5, atol=1e-6, msg=name)
        for actual, expected in zip(states[0]["metrics"], expected_metrics):
            assert actual["ce"] == pytest.approx(expected["ce"], abs=2e-6)
        logs = [json.loads(line) for line in (tmp_path / mode / "training_metrics.jsonl").read_text().splitlines()]
        assert len(logs) == 4  # Once per optimizer step, never once per rank.
        assert sum(row["examples"] for row in logs) == 10
        assert sum(row["tokens"] for row in logs) == 40  # 15 words + 5 EOS, twice.
        assert all(row["world_size"] == 2 and row["max_grad_norm"] is None for row in logs)

    original = torch.load(tmp_path / "fit/last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert original["world_size"] == 2 and len(original["rng_states_by_rank"]) == 2
    assert original["step"] == resumed["step"] == 4
    assert not any(name.startswith("module.") for name in original["model"])
    for name, parameter in original["model"].items():
        torch.testing.assert_close(resumed["model"][name], parameter, atol=0, rtol=0, msg=name)
    config["generation"]["max_new_tokens"] = 2
    _write_resolved_config(config, tmp_path)
    evaluate(config_path, tmp_path / "fit/last.pt", tmp_path / "predictions.jsonl", device="cpu")
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == 2

    # Check the documented shell launcher and spawned DataLoader workers too.
    config["training"].update(gradient_accumulation_steps=7, num_workers=1, validation_num_workers=1)
    _write_resolved_config(config, tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).parents[1] / "scripts/run_afmr.sh"),
            "train",
            str(config_path),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "launcher"),
        ],
        env=dict(environment, NPROC_PER_NODE="2", GRADIENT_ACCUMULATION_STEPS="2", PYTHON=sys.executable),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    resolved = load_config(tmp_path / "launcher/resolved_config.yaml")
    assert resolved["training"]["gradient_accumulation_steps"] == 2
    launched = torch.load(tmp_path / "launcher/last.pt", weights_only=False)
    assert launched["step"] == 4 and launched["world_size"] == 2


if __name__ == "__main__":
    _worker(sys.argv[1])
