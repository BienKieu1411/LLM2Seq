"""Post-SFT GRPO using TRL's training loop and objective for the full SEAM actor."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset
from eviseq_afmr.data.collate import SummarizationCollator
from eviseq_afmr.data.dataset import JsonlSummarizationDataset
from eviseq_afmr.modeling.model import EviSeqAFMR
from eviseq_afmr.runtime import _tokenizers, _write_resolved_config
from eviseq_afmr.training.checkpoint import load_checkpoint
from eviseq_afmr.training.engine import seed_everything
from eviseq_afmr.training.optimizer import build_optimizer, set_stage_trainability
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .config import load
from .data import make_loader, prepare, trajectory
from .output import reset_output
from .reward import RougeReward


def source_records(path, data_config, max_examples, update_sources):
    """Memory-mapped Arrow input, padded to complete updates without dropping a tail."""
    dataset = JsonlSummarizationDataset(path, data_config, max_examples=max_examples)
    size = math.ceil(len(dataset) / update_sources) * update_sources
    for index in range(size):
        record = dataset[index % len(dataset)]
        yield {"id": record.example_id, "prompt": record.source, "reference": record.target}


def processing_tokenizer(tokenizer):
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        return tokenizer
    # Offline tiny tests need a real HF processing class; actual source/action
    # tokenization still uses the exact collator tokenizers from the SFT model.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, **{f"tok{i}": i for i in range(4, 128)}}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def check_rollout_device(settings, device_name):
    if settings["training"]["rollout_backend"] == "seam_vllm" and (
        device_name == "cpu" or not torch.cuda.is_available()
    ):
        raise ValueError("SEAM vLLM rollout requires CUDA; use rollout_backend=native for CPU")
    if (
        settings["training"]["rollout_backend"] == "seam_vllm"
        and settings["training"].get("vllm_mode", "colocate") == "dedicated"
        and (int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1)
    ):
        raise ValueError("Dedicated rollout requires one visible training GPU; use scripts/run_grpo_two_gpu.sh")


def train(settings: dict, device_name: str | None = None, *, overwrite_output_dir: bool = False) -> dict:
    # Fail before constructing full backbones or running baseline validation.
    check_rollout_device(settings, device_name)
    from trl import GRPOConfig

    from .actor import SEAMActor
    from .lifecycle import Lifecycle
    from .trainer import SEAMGRPOTrainer

    base = prepare(settings)
    cfg = settings["training"]
    output = Path(settings["output_dir"])
    if device_name and device_name != "cpu" and not device_name.startswith("cuda"):
        raise ValueError("Use --device cpu for offline checks or CUDA_VISIBLE_DEVICES for GPU training")
    if overwrite_output_dir:
        reset_output(settings, base, use_cpu=device_name == "cpu" or not torch.cuda.is_available())
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh output_dir or --overwrite-output-dir to restart GRPO: {output}")
    seed_everything(cfg["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(base["training"].get("tf32", False))
    tokenizers = _tokenizers(base)
    processor = processing_tokenizer(tokenizers[1])

    def build_actor():
        seam = EviSeqAFMR(base).float()
        if cfg["rollout_backend"] == "seam_vllm":
            from seam_vllm._compat import validate_decoder_config

            validate_decoder_config(seam.decoder.config)
            if int(base["decoder"].get("cross_attention_every", 1)) != 1:
                raise ValueError("SEAM vLLM rollout requires cross_attention_every=1")
        load_checkpoint(settings["sft_checkpoint"], seam, config=base, restore_rng=False)
        set_stage_trainability(seam, "full_finetune")
        actor = SEAMActor(seam, chunk_size=cfg["logprob_chunk_size"], score_batch_size=cfg["score_micro_batch_size"])
        # The collator's EOS convention is part of the SFT trajectory contract.
        actor.generation_config.eos_token_id = tokenizers[1].eos_token_id
        return actor

    actor = build_actor()
    reference = build_actor().requires_grad_(False).eval() if cfg["beta"] else None
    optimizer = build_optimizer(actor.policy.seam, base, "full_finetune")
    use_cpu = device_name == "cpu" or not torch.cuda.is_available()
    compute_dtype = base["model"].get("compute_dtype", base["model"].get("dtype", "float32"))
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
        loss_type=cfg.get("loss_type", "dr_grpo"),
        scale_rewards=cfg["scale_rewards"],
        epsilon=cfg["epsilon"],
        epsilon_high=cfg.get("epsilon_high", None),
        delta=cfg.get("delta", None),
        importance_sampling_level=cfg.get("importance_sampling_level", "token"),
        top_entropy_quantile=cfg.get("top_entropy_quantile", 1.0),
        entropy_coef=cfg.get("entropy_coef", 0.0),
        off_policy_mask_threshold=cfg.get("off_policy_mask_threshold", None),
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
    originals = JsonlSummarizationDataset(
        base["data"]["train_file"], base["data"], max_examples=cfg["max_train_examples"]
    )
    update_sources = cfg["prompt_batch_size"] * cfg["gradient_accumulation_steps"] * args.world_size
    dataset = Dataset.from_generator(
        source_records,
        gen_kwargs={
            "path": base["data"]["train_file"],
            "data_config": base["data"],
            "max_examples": cfg["max_train_examples"],
            "update_sources": update_sources,
        },
        cache_dir=str(output / "dataset_cache"),
    )
    collator = SummarizationCollator(
        *tokenizers, base["data"], grounded_copy=bool(base["decoder"].get("grounded_copy", {}).get("enabled", False))
    )
    trainer = SEAMGRPOTrainer(
        model=actor,
        args=args,
        base_config=base,
        settings=cfg,
        collator=collator,
        decoder_tokenizer=tokenizers[1],
        reward=RougeReward(settings["reward"]),
        train_dataset=dataset,
        processing_class=processor,
        reference=reference,
        optimizers=(optimizer, None),
        sft_checkpoint=settings["sft_checkpoint"],
    )
    validation = (
        make_loader(base, cfg, tokenizers, "validation")
        if trainer.accelerator.is_main_process and base["data"].get("validation_file")
        else None
    )
    callback = Lifecycle(trainer, validation, tokenizers[1], trainer.rouge_reward)
    trainer.add_callback(callback)
    if trainer.accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        _write_resolved_config(base, output)
        with (output / "grpo_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False, allow_unicode=True)
        with (output / "implementation.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "trainer": "trl.GRPOTrainer subclass",
                    "trl": "1.16.0",
                    "rollout": cfg["rollout_backend"],
                    "sft_checkpoint": settings["sft_checkpoint"],
                    "padded_training_sources": len(dataset),
                    "original_training_sources": len(originals),
                },
                stream,
                indent=2,
            )
    try:
        trainer.train()
        return {
            "steps": trainer.state.global_step,
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
    parser.add_argument("--vllm-mode", choices=("colocate", "dedicated"))
    parser.add_argument("--rollout-gpu", help="Physical GPU index/UUID visible only to the dedicated rollout host")
    parser.add_argument(
        "--overwrite-output-dir", action="store_true", help="Delete previous output artifacts and restart from SFT"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    settings = load(args.config)
    if args.vllm_mode is not None:
        settings["training"]["vllm_mode"] = args.vllm_mode
        settings["training"]["rollout_backend"] = "seam_vllm"
    if args.rollout_gpu is not None:
        if not args.rollout_gpu.strip() or "," in args.rollout_gpu:
            parser.error("--rollout-gpu must identify exactly one GPU")
        settings["training"]["vllm_rollout_gpu"] = args.rollout_gpu
    if args.dry_run:
        base = prepare(settings)
        RougeReward(settings["reward"])
        print(
            yaml.safe_dump(
                {"grpo": settings, "model": base["model"], "data": base["data"]}, sort_keys=False, allow_unicode=True
            )
        )
    else:
        output = Path(settings["output_dir"]).resolve()
        if args.overwrite_output_dir and Path(args.config).resolve().is_relative_to(output):
            parser.error("GRPO config must be outside output_dir when overwriting")
        result = train(settings, args.device, overwrite_output_dir=args.overwrite_output_dir)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
