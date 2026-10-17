"""Post-SFT GRPO using TRL's training loop and objective for the full SEAM actor."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _tokenizers, _write_resolved_config
from eviseq_afmr.training.checkpoint import load_checkpoint
from eviseq_afmr.training.engine import seed_everything
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability
from seam_grpo.data import prepare
from seam_grpo.train import check_rollout_device, processing_tokenizer

from .config import load
from .data import LengthCollator, LengthDataset, make_loader, source_records
from .policy import LengthPolicy, validate_budgets
from .reward import LengthReward
from .teacher import SourceTeacher
from .warmup import adapt


def train(settings: dict, device_name: str | None = None) -> dict:
    # Fail before constructing full backbones or running baseline validation.
    check_rollout_device(settings, device_name)
    from seam_grpo.actor import SEAMActor
    from trl import GRPOConfig

    from .lifecycle import Lifecycle
    from .trainer import LengthTrainer

    base = prepare(settings)
    cfg = settings["training"]
    if settings["source_reward"]["weight"] and not Path(settings["source_reward"]["teacher_checkpoint"]).is_file():
        raise FileNotFoundError(settings["source_reward"]["teacher_checkpoint"])
    output = Path(settings["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output_dir to avoid mixing GRPO artifacts: {output}")
    if device_name and device_name != "cpu" and not device_name.startswith("cuda"):
        raise ValueError("Use --device cpu for offline checks or CUDA_VISIBLE_DEVICES for GPU training")
    seed_everything(cfg["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(base["training"].get("tf32", False))
    tokenizers = _tokenizers(base)
    processor = processing_tokenizer(tokenizers[1])
    collator = LengthCollator(
        tokenizers,
        base["data"],
        settings["length_control"],
        grounded_copy=bool(base["decoder"].get("grounded_copy", {}).get("enabled", False)),
        include_targets=bool(cfg["ce_weight"]),
    )
    # Serving needs capacity for the content budget PLUS its explicit prompt reserve.
    base["data"]["max_source_length"] = collator.encoder_limit

    def build_actor(checkpoint=None):
        seam = EviSeqAFMR(base).float()
        if cfg["rollout_backend"] == "seam_vllm":
            from seam_vllm._compat import validate_decoder_config

            validate_decoder_config(seam.decoder.config)
            if int(base["decoder"].get("cross_attention_every", 1)) != 1:
                raise ValueError("SEAM vLLM rollout requires cross_attention_every=1")
        load_checkpoint(checkpoint or settings["sft_checkpoint"], seam, config=base, restore_rng=False)
        set_stage_trainability(seam, "full_finetune")
        actor = SEAMActor(seam, chunk_size=cfg["logprob_chunk_size"], score_batch_size=cfg["score_micro_batch_size"])
        actor.policy = LengthPolicy(seam, chunk_size=cfg["logprob_chunk_size"])
        # The collator's EOS convention is part of the SFT trajectory contract.
        actor.generation_config.eos_token_id = tokenizers[1].eos_token_id
        return actor

    actor = build_actor()
    validate_budgets(actor.policy.seam, collator, cfg["max_new_tokens"])
    use_cpu = device_name == "cpu" or not torch.cuda.is_available()
    compute_dtype = base["model"].get("compute_dtype", base["model"].get("dtype", "float32"))
    originals = LengthDataset(
        base["data"]["train_file"], collator.data, settings["length_control"], max_examples=cfg["max_train_examples"]
    )
    warmup = adapt(
        actor,
        originals,
        collator,
        settings,
        base,
        use_cpu=use_cpu,
        dtype=compute_dtype,
        processor=processor,
        world=int(os.environ.get("WORLD_SIZE", "1")),
    )
    # Reference/teacher are loaded independently from SFT, never from the adapted actor.
    reference = build_actor().requires_grad_(False).eval() if cfg["beta"] else None
    teacher_actor = None
    if settings["source_reward"]["weight"]:
        teacher_path = settings["source_reward"]["teacher_checkpoint"]
        teacher_actor = (
            reference
            if reference is not None and teacher_path == settings["sft_checkpoint"]
            else build_actor(teacher_path).requires_grad_(False).eval()
        )
    optimizer = build_optimizer(actor.policy.seam, base, "full_finetune")
    args = GRPOConfig(
        output_dir=str(output),
        per_device_train_batch_size=cfg["prompt_batch_size"] * cfg["group_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        steps_per_generation=cfg["gradient_accumulation_steps"],
        num_generations=cfg["group_size"],
        num_iterations=1,
        max_completion_length=cfg["max_new_tokens"],
        pad_to_multiple_of=1,
        temperature=cfg["temperature"],
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        loss_type="dr_grpo",
        scale_rewards=cfg["scale_rewards"],
        epsilon=cfg["epsilon"],
        beta=0.0,
        max_steps=cfg["max_steps"] or -1,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["decoder_lr"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        seed=cfg["seed"],
        bf16=not use_cpu and compute_dtype == "bfloat16",
        fp16=not use_cpu and compute_dtype == "float16",
        use_cpu=use_cpu,
        gradient_checkpointing=False,
        disable_dropout=True,
        use_vllm=False,
        remove_unused_columns=False,
        report_to=[],
        push_to_hub=False,
        save_strategy="no",
        eval_strategy="no",
        logging_steps=cfg["log_every_steps"],
        disable_tqdm=True,
        dataloader_num_workers=cfg["num_workers"],
        ddp_find_unused_parameters=True,
        ddp_timeout=86400,
        optim="adamw_torch",
        use_bias_correction_kl=True,
    )
    update_sources = cfg["prompt_batch_size"] * cfg["gradient_accumulation_steps"] * args.world_size
    dataset = Dataset.from_generator(
        source_records,
        gen_kwargs={
            "path": base["data"]["train_file"],
            "data_config": collator.data,
            "length": settings["length_control"],
            "source_config": settings["source_reward"],
            "seed": cfg["seed"],
            "max_examples": cfg["max_train_examples"],
            "update_sources": update_sources,
            "source_tokenizer": tokenizers[0] if settings["source_reward"]["weight"] else None,
        },
        cache_dir=str(output / "dataset_cache"),
    )
    trainer = LengthTrainer(
        model=actor,
        args=args,
        base_config=base,
        settings=cfg,
        collator=collator,
        decoder_tokenizer=tokenizers[1],
        length_reward=LengthReward(settings["reward"]),
        source_teacher=None,
        length_settings={"source_weight": settings["source_reward"]["weight"]},
        train_dataset=dataset,
        processing_class=processor,
        reference=reference,
        optimizers=(optimizer, None),
        sft_checkpoint=settings["sft_checkpoint"],
    )
    if teacher_actor is not None:
        teacher_actor = (
            trainer.ref_model
            if teacher_actor is reference
            else trainer.accelerator.prepare_model(teacher_actor, evaluation_mode=True)
        )
        teacher_collator = LengthCollator(
            tokenizers,
            collator.data,
            settings["length_control"],
            grounded_copy=collator.grounded_copy,
            include_targets=False,
        )
        trainer.source_teacher = SourceTeacher(
            teacher_actor,
            teacher_collator,
            settings["source_reward"],
            max_new_tokens=cfg["max_new_tokens"],
            micro_batch_size=cfg["score_micro_batch_size"],
        )
    validation = (
        make_loader({**base, "data": collator.data}, cfg, tokenizers, settings["length_control"], "validation")
        if trainer.accelerator.is_main_process and base["data"].get("validation_file")
        else None
    )
    callback = Lifecycle(trainer, validation, tokenizers[1], trainer.length_reward)
    trainer.add_callback(callback)
    if trainer.accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        _write_resolved_config(base, output)
        # Inference uses this run's exported architecture/data config. Persist
        # the resolved BODY budget so the prompt reserve is not added twice.
        exported_settings = {
            **settings,
            "base_config": str(output / "resolved_config.yaml"),
            "length_control": {**settings["length_control"], "source_content_tokens": collator.source_limit},
        }
        with (output / "length_grpo_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(exported_settings, stream, sort_keys=False, allow_unicode=True)
        with (output / "implementation.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "trainer": "trl.GRPOTrainer subclass",
                    "trl": "1.16.0",
                    "rollout": cfg["rollout_backend"],
                    "sft_checkpoint": settings["sft_checkpoint"],
                    "padded_training_sources": len(dataset),
                    "original_training_sources": len(originals),
                    "method": "dual-prompt length GRPO",
                    "ce_warmup_steps": warmup["steps"],
                    "source_reward_weight": settings["source_reward"]["weight"],
                    "teacher_checkpoint": settings["source_reward"]["teacher_checkpoint"]
                    if teacher_actor is not None
                    else None,
                },
                stream,
                indent=2,
            )
    try:
        logging.getLogger(__name__).info(
            "[grpo] starting at step=0 after ce_warmup_steps=%d; fresh optimizer/cosine", warmup["steps"]
        )
        trainer.train()
        return {
            "steps": trainer.state.global_step,
            "ce_warmup_steps": warmup["steps"],
            "best_validation_proxy": callback.best,
            "output_dir": str(output),
            "trainer": "trl.GRPOTrainer",
            "rollouts": trainer.rollout_history,
        }
    finally:
        try:
            trainer.close_rollout()
        finally:
            if int(os.environ.get("WORLD_SIZE", "1")) > 1 and dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/settings without loading model weights")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    settings = load(args.config)
    if args.dry_run:
        base = prepare(settings)
        LengthReward(settings["reward"])
        print(
            yaml.safe_dump(
                {"grpo": settings, "model": base["model"], "data": base["data"]}, sort_keys=False, allow_unicode=True
            )
        )
    else:
        result = train(settings, args.device)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
