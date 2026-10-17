"""Version boundaries, native validation and standard SEAM exports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch.distributed as dist
from transformers import TrainerCallback

from .evaluation import predict


class Lifecycle(TrainerCallback):
    def __init__(self, trainer, validation, decoder_tokenizer, reward):
        self.trainer = trainer
        self.validation = validation
        self.tokenizer = decoder_tokenizer
        self.reward = reward
        self.best = None
        self.best_key = None

    def wait(self):
        accelerator = self.trainer.accelerator
        if accelerator.device.type == "cpu" and dist.is_initialized():
            # Gloo must not receive CUDA-style device_ids: on a Mac with MPS,
            # recent torch otherwise selects an unsupported MPS barrier.
            dist.barrier()
        else:
            accelerator.wait_for_everyone()

    def measure(self, phase, epoch):
        trainer = self.trainer
        accelerator = trainer.accelerator
        self.wait()
        if accelerator.is_main_process and self.validation is not None:
            actor = accelerator.unwrap_model(trainer.model)
            output = Path(trainer.args.output_dir)
            metrics = predict(
                actor.policy.seam,
                self.validation,
                self.tokenizer,
                trainer.settings,
                trainer.base_config,
                self.reward,
                accelerator.device,
                accelerator.autocast,
                output / f"validation_{phase}.jsonl",
            )
            with (output / "validation_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps({"epoch": epoch, "step": trainer.state.global_step, "phase": phase, **metrics}) + "\n"
                )
            feasible = metrics["all_length_constraints_met"]
            key = (
                feasible,
                metrics["macro_quality"] if feasible else -metrics["max_length_violation"],
                metrics["macro_quality"],
            )
            if self.best_key is None or key > self.best_key:
                self.best_key, self.best = key, metrics["macro_quality"]
                if trainer.settings["save_best"]:
                    trainer.export_checkpoint("best.pt", epoch=epoch, best_metric=self.best)
                    with (output / "best_selection.json").open("w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "epoch": epoch,
                                "step": trainer.state.global_step,
                                "feasible": feasible,
                                "rule": "all dataset length constraints, then macro ROUGE quality; otherwise minimize worst violation",
                                "metrics": metrics,
                            },
                            handle,
                            ensure_ascii=False,
                            indent=2,
                        )
        self.wait()

    def on_train_begin(self, args, state, control, **kwargs):
        if self.trainer.settings["evaluate_before_training"]:
            phase = (
                "length_ce_baseline" if (Path(args.output_dir) / "length_ce_warmup.pt").is_file() else "sft_baseline"
            )
            self.measure(phase, 0)

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        # With num_iterations=1 and one generation window per optimizer update,
        # every buffered trajectory is consumed. Drop it before version k+1.
        trainer._buffered_inputs = None
        actor = trainer.accelerator.unwrap_model(trainer.model)
        actor.policy.seam.decoder.clear_cross_cache()
        if trainer.accelerator.is_main_process:
            with (Path(args.output_dir) / "rollout_versions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "updated_policy_version": state.global_step,
                            "last_rollout_version": trainer.rollout_history[-1]["policy_version"],
                        }
                    )
                    + "\n"
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = math.ceil(state.epoch or 1)
        self.measure(f"epoch_{epoch:03d}", epoch)
        trainer = self.trainer
        trainer.export_checkpoint("last.pt", epoch=epoch, best_metric=self.best)
        if trainer.settings["save_each_epoch"]:
            trainer.export_checkpoint(f"epoch_{epoch:03d}.pt", epoch=epoch, best_metric=self.best)
        self.wait()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and self.trainer.accelerator.is_main_process:
            with (Path(args.output_dir) / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": state.global_step, **logs}) + "\n")
