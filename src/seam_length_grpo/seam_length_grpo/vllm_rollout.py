"""Synchronous per-rank SEAM vLLM rollout with full GPU actor synchronization."""

from __future__ import annotations

import os
import socket
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from seam_grpo.context import validate_context
from seam_grpo.vllm_rollout import completion_rows, rollout_token_budget


class LengthVLLMRollout:
    def __init__(self, trainer):
        self.trainer = trainer
        self.engine = None
        self.version = -1

    def initialize(self):
        from seam_vllm import SEAMLLM

        trainer = self.trainer
        accelerator = trainer.accelerator
        if accelerator.device.type != "cuda":
            raise ValueError("SEAM vLLM rollout requires CUDA; use rollout_backend=native for CPU")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" not in os.environ:
            if accelerator.num_processes > 1:
                raise ValueError("Multi-GPU SEAM vLLM requires the launcher's shared MASTER_PORT")
            # Multi-rank launchers already supply this; only single rank picks.
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(listener.getsockname()[1])
        cfg = trainer.settings
        seam = accelerator.unwrap_model(trainer.model).policy.seam
        prompt_width = trainer.collator.length["decoder_prompt_reserve"]
        # Actual prompts are asserted again below; the fallback prompt is BOS
        # on configs with an empty instruction.
        prompt_width = max(prompt_width, 1)
        validate_context(seam.decoder, prompt_width, cfg["max_new_tokens"])
        # Dataset.__getitem__ already detokenized these exact source strings.
        # The serving processor normally handles raw text; disable a second
        # detokenization here to preserve the training tokenizer input exactly.
        rollout_config = Path(trainer.args.output_dir) / "rollout_config.yaml"
        if accelerator.is_main_process:
            config = deepcopy(trainer.base_config)
            config.pop("_meta", None)
            config["data"]["detokenize"] = False
            with rollout_config.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        engine_class = SEAMLLM
        extra = {}
        if cfg.get("vllm_mode", "colocate") == "dedicated":
            from seam_grpo.remote_rollout import RemoteSEAMLLM

            engine_class = RemoteSEAMLLM
            extra["trainer"] = trainer
        self.engine = engine_class(
            rollout_config,
            trainer.sft_checkpoint,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
            max_model_len=prompt_width + cfg["max_new_tokens"],
            max_num_seqs=cfg["vllm_max_num_seqs"],
            max_num_batched_tokens=rollout_token_budget(trainer, prompt_width + cfg["max_new_tokens"]),
            encoder_max_batch_tokens=cfg["vllm_encoder_max_batch_tokens"],
            mm_processor_cache_gb=0,
            logprobs_mode="processed_logprobs",
            seed=cfg["seed"] + accelerator.process_index,
            **extra,
        )

    def __call__(self, prompts, source):
        if self.engine is None:
            self.initialize()
            # A recreated engine starts from SFT, regardless of the training
            # step at which the previous engine was closed.
            self.version = -1
        trainer = self.trainer
        cfg = trainer.settings
        actor = trainer.accelerator.unwrap_model(trainer.model).policy.seam
        version = trainer.state.global_step
        if self.version != version:
            self.engine.sync_weights(actor, policy_version=version)
            self.version = version
        from vllm import SamplingParams

        params = SamplingParams(
            n=1,
            temperature=cfg["temperature"],
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            min_tokens=0,
            max_tokens=cfg["max_new_tokens"],
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logprobs=0,
            detokenize=False,
            stop=[],
            stop_token_ids=[],
        )
        if params.temperature != cfg["temperature"]:
            raise ValueError("vLLM altered the requested sampling temperature")
        from seam_vllm._request import source_prompt

        prompt_ids = [
            ids[mask.bool()].tolist()
            for ids, mask in zip(source["decoder_prompt_ids"], source["decoder_prompt_mask"], strict=True)
        ]
        requests = [
            source_prompt(
                text,
                ids,
                source_limit=trainer.collator.encoder_limit,
                output_budget=cfg["max_new_tokens"],
                encoder_prompt=prefix,
            )
            for text, ids, prefix in zip(source["sources"], prompt_ids, source["encoder_prompts"], strict=True)
        ]
        group = 1
        if cfg.get("vllm_group_sampling"):
            group = trainer.num_generations if trainer.model.training else trainer.num_generations_eval
            texts = source["sources"]
            if len(texts) % group or any(texts[row] != texts[row - row % group] for row in range(len(texts))):
                raise RuntimeError("Grouped vLLM sampling needs contiguous identical-source groups")
            params.n = group
        requests = requests[::group]
        # The version-pinned vLLM batch accepts a different multimodal prompt per row.
        # SEAMLLM's convenience string API applies overrides batch-wide, so build
        # requests here using the SAME renderer/IDs as native likelihood scoring.
        outputs = self.engine.engine.generate(requests, params, use_tqdm=False)
        if trainer.state.global_step != version:
            raise RuntimeError("Actor changed during vLLM rollout")
        return completion_rows(
            outputs,
            prompt_ids,
            eos_token_id=trainer.decoder_tokenizer.eos_token_id,
            budget=cfg["max_new_tokens"],
            group_size=group,
        )

    def close(self):
        if self.engine is not None:
            # Version-pinned EngineCoreClient lifecycle, before Trainer tears
            # down the distributed process group (LLMEngine has no shutdown()).
            try:
                if self.trainer.settings.get("vllm_mode", "colocate") == "dedicated":
                    self.engine.close()
                else:
                    self.engine.engine.llm_engine.engine_core.shutdown()
            finally:
                self.engine = None
                self.version = -1
