"""Materialize the two configs used by the summary-to-WikiLingua run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from eviseq_afmr.config import load_config, validate_config

ENCODER_PREFIX = (
    "Hãy đọc văn bản nguồn và viết một bản tóm tắt ngắn gọn, chính xác, mạch lạc. "
    "Chỉ giữ các thông tin quan trọng; không thêm thông tin không có trong văn bản nguồn.\n\n"
    "Văn bản nguồn:\n"
)
DECODER_PROMPT = (
    "Viết bản tóm tắt ngắn gọn và chính xác từ văn bản nguồn. "
    "Chỉ sử dụng thông tin trong nguồn, không thêm thông tin mới. Chỉ trả về bản tóm tắt."
)
DECODER_PREFIX = "Tóm tắt:\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--summary-train", required=True)
    parser.add_argument("--wikilingua-dir", required=True)
    parser.add_argument("--stage1-output", required=True)
    parser.add_argument("--stage2-output", required=True)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--decoder-model", required=True)
    parser.add_argument("--source-field", default="text")
    parser.add_argument("--target-field", default="summary")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-num-workers", type=int, default=1)
    parser.add_argument("--summary-epochs", type=int, default=4)
    parser.add_argument("--wikilingua-epochs", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=3072)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=16)
    return parser.parse_args()


def _shared_data(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_field": args.source_field,
        "target_field": args.target_field,
        "id_field": args.id_field,
        "encoder_prefix": ENCODER_PREFIX,
        "decoder_prompt": DECODER_PROMPT,
        "decoder_chat_template": True,
        "decoder_prefix": DECODER_PREFIX,
        "detokenize": True,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
    }


def _base_config(args: argparse.Namespace) -> dict:
    config = load_config(args.template, train_only=True)
    config.pop("_meta", None)
    config["model"]["encoder_name"] = str(Path(args.encoder_model).expanduser().resolve())
    config["model"]["decoder_name"] = str(Path(args.decoder_model).expanduser().resolve())
    config["training"].update(
        {
            "interface_warmup_epochs": 0,
            "batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "validation_batch_size": args.validation_batch_size,
            "num_workers": args.num_workers,
            "validation_num_workers": args.validation_num_workers,
            "save_each_epoch": True,
            "resume_checkpoint": "",
            "resume_scheduler": False,
        }
    )
    config["generation"].update(
        {
            "batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "num_beams": 1,
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
        }
    )
    return config


def main() -> None:
    args = _parse_args()
    if args.summary_epochs <= 0 or args.wikilingua_epochs <= 0:
        raise SystemExit("summary and WikiLingua epochs must be positive")
    for name in (
        "train_batch_size",
        "gradient_accumulation",
        "validation_batch_size",
        "num_workers",
        "validation_num_workers",
        "max_source_length",
        "max_target_length",
        "eval_batch_size",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be positive")
    wiki = Path(args.wikilingua_dir).expanduser().resolve()

    stage1 = _base_config(args)
    stage1["experiment"].update(
        {"name": "afmr_summary_pretrain", "output_dir": str(Path(args.stage1_output).resolve())}
    )
    stage1["data"].update(_shared_data(args))
    stage1["data"].update(
        {
            "train_file": str(Path(args.summary_train).expanduser().resolve()),
            "validation_file": "",
            "test_file": "",
        }
    )
    stage1["training"].update({"full_finetune_epochs": args.summary_epochs, "save_best": False})
    validate_config(stage1, train_only=True)

    stage2 = _base_config(args)
    stage2["experiment"].update(
        {"name": "afmr_wikilingua_continuation", "output_dir": str(Path(args.stage2_output).resolve())}
    )
    stage2["data"].update(
        {
            **_shared_data(args),
            "train_file": str(wiki / "train.jsonl"),
            "validation_file": str(wiki / "validation.jsonl"),
            "test_file": str(wiki / "test.jsonl"),
            "source_field": "text",
            "target_field": "summary",
            "id_field": "id",
        }
    )
    # A resumed checkpoint carries stage_epoch=summary_epochs. The trainer
    # therefore executes the next wikilingua_epochs epochs when this total is
    # summary_epochs + wikilingua_epochs.
    stage2["training"].update({"full_finetune_epochs": args.summary_epochs + args.wikilingua_epochs, "save_best": True})
    validate_config(stage2)

    stage1_path = Path(args.stage1_config)
    stage2_path = Path(args.stage2_config)
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_path.write_text(yaml.safe_dump(stage1, sort_keys=False, allow_unicode=True), encoding="utf-8")
    stage2_path.write_text(yaml.safe_dump(stage2, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"stage1_config={stage1_path}")
    print(f"stage2_config={stage2_path}")
    print("shared_prompt=plain_summary")
    print(f"max_source_length={args.max_source_length}")
    print(f"max_target_length={args.max_target_length}")


if __name__ == "__main__":
    main()
