"""Pinned TRL GRPOTrainer with explicit SEAM source/mixture adapters."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import torch
from eviseq_afmr.data.schema import CanonicalRecord
from eviseq_afmr.training.checkpoint import save_checkpoint
from trl import GRPOTrainer

from .data import move, trajectory
from .rollout import rollout, select_rows

TRL_VERSION = "1.16.0"


class SEAMGRPOTrainer(GRPOTrainer):
    """Keep TRL's sampler, buffers, optimizer loop and GRPO objective.

    Internal hooks are pinned and covered by integration tests. The stock
    fused causal-LM likelihood path is never used for this conditional mixture.
    """

    def __init__(
        self,
        *,
        model,
        args,
        base_config,
        settings,
        collator,
        decoder_tokenizer,
        reward,
        train_dataset,
        processing_class,
        reference=None,
        callbacks=None,
        optimizers=(None, None),
        sft_checkpoint=None,
    ):
        if version("trl") != TRL_VERSION:
            raise RuntimeError(
                f"This architecture adapter requires trl=={TRL_VERSION}; re-audit hooks before upgrading"
            )
        self.base_config = base_config
        self.settings = settings
        self.collator = collator
        self.decoder_tokenizer = decoder_tokenizer
        self.rouge_reward = reward
        self._likelihood_source = None
        self._ce_selector = None
        self._ce_sum = None
        self._active_behavior = None
        self.rollout_history = []
        self.sft_checkpoint = sft_checkpoint
        self.rollout_engine = None

        def rouge_reward(completions, reference, **kwargs):
            return [reward(prediction, gold)["reward"] for prediction, gold in zip(completions, reference, strict=True)]

        # TRL's default reference reconstruction uses AutoModelForCausalLM.
        # Construct the complete SFT reference explicitly, outside that path.
        requested_beta = float(settings["beta"])
        if args.beta != 0:
            raise ValueError("Construct GRPOConfig(beta=0); pass the full reference explicitly")
        super().__init__(
            model=model,
            args=args,
            reward_funcs=rouge_reward,
            train_dataset=train_dataset,
            processing_class=processing_class,
            rollout_func=self._native_rollout,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.beta = requested_beta
        self.args.beta = requested_beta
        if requested_beta:
            if reference is None:
                raise ValueError("beta > 0 requires the full frozen SFT actor")
            reference.requires_grad_(False).eval()
            self.ref_model = self.accelerator.prepare_model(reference, evaluation_mode=True)
        # SEAM vLLM uses our own full-actor callback. Keep the stock causal-only
        # vLLM path disabled so it cannot construct a decoder-only rollout model.
        if self.use_vllm or self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError("SEAM GRPO currently supports native rollout with single GPU / DDP only")

    def create_scheduler(self, num_training_steps, optimizer=None):
        from .config import cosine_multiplier

        if self.lr_scheduler is None:
            cfg = self.settings
            warmup = round(num_training_steps * cfg["warmup_ratio"])
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer or self.optimizer,
                lambda step: cosine_multiplier(step, num_training_steps, warmup, cfg["min_lr_ratio"]),
            )
        return self.lr_scheduler

    def _native_rollout(self, prompts, trainer):
        source = self._likelihood_source
        if source is None or len(prompts) != source["input_ids"].shape[0]:
            raise RuntimeError("TRL rollout rows do not match the sampler-expanded SEAM source batch")
        if list(prompts) != source["sources"]:
            raise RuntimeError("Rollout source order changed")
        actor = self.accelerator.unwrap_model(self.model)
        version_before = self.state.global_step
        if self.settings["rollout_backend"] == "seam_vllm":
            from .vllm_rollout import VLLMRollout

            if self.rollout_engine is None:
                self.rollout_engine = VLLMRollout(self)
            output = self.rollout_engine(prompts, source)
            self.rollout_history.append({"policy_version": version_before, "candidates": len(prompts)})
            return output
        output = {"prompt_ids": [], "completion_ids": [], "logprobs": []}
        cfg = self.settings
        with torch.no_grad(), self.accelerator.autocast():
            for start in range(0, len(prompts), cfg["rollout_micro_batch_size"]):
                indices = torch.arange(start, min(len(prompts), start + cfg["rollout_micro_batch_size"]))
                batch = select_rows(source, indices)
                generated = rollout(
                    actor.policy.seam, batch, self.decoder_tokenizer, cfg["max_new_tokens"], cfg["temperature"]
                )
                for row in range(len(indices)):
                    active = generated["action_mask"][row]
                    output["prompt_ids"].append(batch["decoder_prompt_ids"][row].tolist())
                    output["completion_ids"].append(generated["action_ids"][row][active].tolist())
                    output["logprobs"].append(generated["old_logps"][row][active].tolist())
        if self.state.global_step != version_before:
            raise RuntimeError("Actor weights changed while a rollout group was being collected")
        self.rollout_history.append({"policy_version": version_before, "candidates": len(prompts)})
        return output

    def _generate_and_score_completions(self, inputs):
        cfg = self.settings
        group = self.num_generations if self.model.training else self.num_generations_eval
        if len(inputs) % group:
            raise RuntimeError("Each rank must receive complete GRPO prompt groups")
        for offset in range(0, len(inputs), group):
            expected = (inputs[offset]["id"], inputs[offset]["prompt"], inputs[offset]["reference"])
            if any((row["id"], row["prompt"], row["reference"]) != expected for row in inputs[offset : offset + group]):
                raise RuntimeError("Pinned TRL sampler no longer supplies contiguous identical-source groups")
        from .rollout import repeat_batch

        records = [
            CanonicalRecord.from_mapping(row, source_field="prompt", target_field="reference", id_field="id")
            for row in inputs[::group]
        ]
        source = move(repeat_batch(self.collator(records), group), self.accelerator.device)
        self._likelihood_source = source
        try:
            # TRL performs generation assembly, global reward grouping/advantages,
            # EOS/padding masks, reference scoring, and reward logging.
            result = super()._generate_and_score_completions(inputs)
        finally:
            self._likelihood_source = None
        old = result.get("sampling_per_token_logps")
        if old is None:
            raise RuntimeError("Native rollout must provide the actual behavior log-probabilities")
        result["old_per_token_logps"] = old.detach().clone()
        selector = torch.zeros(len(inputs), device=self.accelerator.device, dtype=torch.bool)
        selector[::group] = True
        gold_tokens = (source["labels"][:, 1:].ne(-100).sum(-1) * selector).sum()
        global_gold_tokens = self.accelerator.reduce(gold_tokens, reduction="sum")
        result["seam_gold_selector"] = selector
        result["seam_global_gold_tokens"] = global_gold_tokens
        result["seam_policy_version"] = torch.full(
            (len(inputs),), self.state.global_step, device=self.accelerator.device, dtype=torch.long
        )
        # Flat fields are split TOGETHER by _prepare_inputs.
        for key, value in source.items():
            result[f"seam_source_{key}"] = value
        return result

    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                from trl.trainer.grpo_trainer import (
                    split_pixel_values_by_grid,
                    split_tensor_dict,
                    unsplit_pixel_values_by_grid,
                )

                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            return self._buffered_inputs[self._step % self.args.steps_per_generation]
        return self._generate_and_score_completions(generation_batch)

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        **kwargs,
    ):
        if compute_aux_loss or any(value is not None for value in kwargs.values()):
            raise ValueError("SEAM likelihood adapter does not support MoE auxiliary or visual inputs")
        source = self._likelihood_source
        if source is None:
            raise RuntimeError("Missing SEAM source conditioning for TRL likelihood scoring")
        if input_ids.shape[0] != source["input_ids"].shape[0]:
            raise RuntimeError("Source/action row mismatch")
        prompt_width = input_ids.shape[1] - logits_to_keep
        if not torch.equal(input_ids[:, :prompt_width], source["decoder_prompt_ids"]):
            raise RuntimeError("TRL decoder prompt changed independently of bridge conditioning")
        packed = trajectory(
            source,
            input_ids[:, -logits_to_keep:],
            attention_mask[:, -logits_to_keep:].bool(),
            self.settings["max_new_tokens"],
        )
        packed["gold_decoder_input_ids"] = source["decoder_input_ids"]
        packed["gold_decoder_attention_mask"] = source["decoder_attention_mask"]
        with self.accelerator.autocast():
            logps, entropies, ce_sum = model(
                packed,
                temperature=self.temperature,
                gold_selector=self._ce_selector,
                compute_entropy_with_grad=(getattr(self, "entropy_coef", 0.0) != 0.0),
            )
        if torch.is_grad_enabled():
            self._ce_sum = ce_sum
            if self._active_behavior is not None:
                active = packed["action_mask"]
                totals = torch.stack(
                    ((logps.detach() - self._active_behavior).abs().masked_fill(~active, 0).sum(), active.sum().float())
                )
                totals = self.accelerator.reduce(totals, reduction="sum")
                mean_error = float(totals[0] / totals[1].clamp_min(1))
                self._metrics["train"]["behavior_logp_mae"].append(mean_error)
                if self.settings["rollout_backend"] == "seam_vllm":
                    maximum = (logps.detach() - self._active_behavior).abs().masked_fill(~active, 0).max()
                    # All ranks must agree on a failure before any rank starts
                    # backward, otherwise a parity failure can strand DDP peers.
                    maxima = self.accelerator.gather(maximum.reshape(1))
                    max_error = float(maxima.max())
                    self._metrics["train"]["behavior_logp_max_error"].append(max_error)
                    if (
                        not torch.isfinite(totals).all()
                        or not torch.isfinite(maxima).all()
                        or mean_error > self.settings["vllm_logp_mean_tolerance"]
                        or max_error > self.settings["vllm_logp_max_tolerance"]
                    ):
                        raise RuntimeError(
                            f"SEAM vLLM/native likelihood mismatch: mean={mean_error:.5f}, max={max_error:.5f}; "
                            "check weights, dtype, tokenizer, prompts, source budget and sampling before training"
                        )
        return logps, entropies if compute_entropy else None, None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not torch.all(inputs["seam_policy_version"] == self.state.global_step):
            raise RuntimeError("Refusing stale SEAM rollouts after an optimizer update")
        self._likelihood_source = {
            key.removeprefix("seam_source_"): value for key, value in inputs.items() if key.startswith("seam_source_")
        }
        self._ce_selector = inputs["seam_gold_selector"] if self.settings["ce_weight"] else None
        self._ce_sum = None
        self._active_behavior = inputs["old_per_token_logps"]
        try:
            # Clipping, importance ratios, reference KL, Dr. GRPO normalization,
            # gradient-accumulation scaling, and policy metrics remain TRL's.
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            if self.settings["ce_weight"]:
                if self._ce_sum is None:
                    raise RuntimeError("Gold CE was not returned by the same DDP actor forward")
                # The denominator spans the complete rollout/accumulation window,
                # so do NOT divide this term by gradient_accumulation again.
                denominator = inputs["seam_global_gold_tokens"].clamp_min(1)
                loss = loss + self.settings["ce_weight"] * self._ce_sum * self.accelerator.num_processes / denominator
                global_ce_sum = self.accelerator.reduce(self._ce_sum.detach(), reduction="sum")
                self._metrics["train"]["ce"].append(
                    float(global_ce_sum / denominator * self.current_gradient_accumulation_steps)
                )
            return loss
        finally:
            self._likelihood_source = None
            self._ce_selector = None
            self._ce_sum = None
            self._active_behavior = None

    def export_checkpoint(self, name, *, epoch, best_metric=None):
        if not self.accelerator.is_main_process:
            return
        actor = self.accelerator.unwrap_model(self.model)
        save_checkpoint(
            Path(self.args.output_dir) / name,
            actor.policy.seam,
            self.optimizer,
            self.base_config,
            epoch=epoch,
            step=self.state.global_step,
            stage="grpo",
            best_metric=best_metric,
            scheduler=self.lr_scheduler,
        )

    def close_rollout(self):
        if self.rollout_engine is not None:
            self.rollout_engine.close()
            self.rollout_engine = None
