"""Small pinned-TRL adapter retaining per-sample length conditions everywhere."""

from __future__ import annotations

import torch
from seam_grpo.data import move
from seam_grpo.rollout import rollout
from seam_grpo.trainer import SEAMGRPOTrainer
from trl import GRPOTrainer

from .policy import prompt_groups, unpad_rows


class LengthTrainer(SEAMGRPOTrainer):
    def __init__(self, *, length_reward, source_teacher, length_settings, **kwargs):
        super().__init__(reward=length_reward.rouge, **kwargs)
        self.length_reward, self.length_settings = length_reward, length_settings
        self.source_teacher = source_teacher
        self.reward_func_names = ["length_content_reward"]

    def _native_rollout(self, prompts, trainer):
        source = self._likelihood_source
        if source is None or list(prompts) != source["raw_sources"]:
            raise RuntimeError("Rollout order does not match conditioned source rows")
        actor = self.accelerator.unwrap_model(self.model)
        version = self.state.global_step
        if self.settings["rollout_backend"] == "seam_vllm":
            from .vllm_rollout import LengthVLLMRollout

            if self.rollout_engine is None:
                self.rollout_engine = LengthVLLMRollout(self)
            result = self.rollout_engine(prompts, source)
        else:
            result = {"prompt_ids": [], "completion_ids": [], "logprobs": []}
            # Unpadding is exact for every tokenizer family, without guessed numeral widths.
            with torch.no_grad(), self.accelerator.autocast():
                for indices in prompt_groups(source, self.settings["rollout_micro_batch_size"]):
                    batch = unpad_rows(source, indices)
                    generated = rollout(
                        actor.policy.seam,
                        batch,
                        self.decoder_tokenizer,
                        self.settings["max_new_tokens"],
                        self.settings["temperature"],
                    )
                    for row in range(len(indices)):
                        active = generated["action_mask"][row]
                        result["prompt_ids"].append(batch["decoder_prompt_ids"][row].tolist())
                        result["completion_ids"].append(generated["action_ids"][row][active].tolist())
                        result["logprobs"].append(generated["old_logps"][row][active].tolist())
        if self.state.global_step != version:
            raise RuntimeError("Actor changed during a candidate group")
        self.rollout_history.append({"policy_version": version, "candidates": len(prompts)})
        return result

    def _generate_and_score_completions(self, inputs):
        group = self.num_generations if self.model.training else self.num_generations_eval
        keys = ("id", "prompt", "reference", "target_words", "error_budget", "dataset", "language", "negative_source")
        if len(inputs) % group:
            raise RuntimeError("Each rank must retain complete source/word-request groups")
        for start in range(0, len(inputs), group):
            expected = tuple(inputs[start].get(key) for key in keys)
            if any(tuple(row.get(key) for key in keys) != expected for row in inputs[start : start + group]):
                raise RuntimeError("Candidate group mixed sources, word requests or reward conditions")
        source = move(self.collator(inputs), self.accelerator.device)
        self._likelihood_source = source
        try:
            # The original adapter's CanonicalRecord conversion drops word metadata.
            # Keep TRL's group normalization/objective, replace only that collation boundary.
            result = GRPOTrainer._generate_and_score_completions(self, inputs)
        finally:
            self._likelihood_source = None
        old = result.get("sampling_per_token_logps")
        if old is None:
            raise RuntimeError("Missing actual behavior log-probabilities")
        result["old_per_token_logps"] = old.detach().clone()
        selector = torch.zeros(len(inputs), device=self.accelerator.device, dtype=torch.bool)
        selector[::group] = True
        result["seam_gold_selector"] = selector
        result["seam_global_gold_tokens"] = self.accelerator.reduce(
            (source["labels"][:, 1:].ne(-100).sum(-1) * selector).sum(), reduction="sum"
        )
        result["seam_policy_version"] = torch.full(
            (len(inputs),), self.state.global_step, device=self.accelerator.device, dtype=torch.long
        )
        result.update({f"seam_source_{key}": value for key, value in source.items()})
        return result

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids):
        source = self._likelihood_source
        if source is None:
            raise RuntimeError("Missing source conditions while calculating rewards")
        source_scores, margins = [0.0] * len(inputs), [0.0] * len(inputs)
        if self.source_teacher is not None:
            with self.accelerator.autocast():
                source_scores, margins = self.source_teacher(source, inputs, completion_ids)
        texts = self.decoder_tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        values = [
            self.length_reward(
                text,
                row["reference"],
                row["target_words"],
                row["error_budget"],
                score,
                self.length_settings["source_weight"],
            )
            for text, row, score in zip(texts, inputs, source_scores, strict=True)
        ]
        device = self.accelerator.device
        rewards = torch.tensor([row["reward"] for row in values], device=device).reshape(-1, 1)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError("Non-finite length/content reward")
        mode = "train" if self.model.training else "eval"
        if self.source_teacher is not None:
            tensor = torch.tensor(margins, device=device, dtype=torch.float32)
            self._metrics[mode]["source_teacher/mean_logp_margin"].append(float(self.accelerator.gather(tensor).mean()))
        for key in ("quality", "abs_len_loss", "length_cost", "source_score"):
            tensor = torch.tensor([row[key] for row in values], device=device, dtype=torch.float32)
            self._metrics[mode][f"length/{key}"].append(float(self.accelerator.gather(tensor).mean()))
        group = self.num_generations if self.model.training else self.num_generations_eval
        costs = torch.tensor([row["length_cost"] for row in values], device=device).reshape(-1, group)
        flat = (costs.max(-1).values - costs.min(-1).values).lt(1e-6).float()
        self._metrics[mode]["length/flat_cost_groups"].append(float(self.accelerator.gather(flat).mean()))
        return self.accelerator.gather(rewards)
