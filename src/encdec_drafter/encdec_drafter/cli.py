from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .benchmark import benchmark_runtime, profile_t5gemma2
from .config import load_config, resolve_path
from .data import collect_teacher_traces, load_source_path, read_jsonl, source_text, write_jsonl
from .diagnostics import diagnose_t5gemma2_verifier
from .drafter import load_drafter
from .generation import GreedySpeculativeGenerator
from .model import load_adapter
from .training import train_drafter


def _resolve_cli_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="encdec_drafter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("profile", "collect-traces", "train", "generate", "benchmark", "diagnose-verifier"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        if command in ("profile", "collect-traces", "generate", "benchmark", "diagnose-verifier"):
            child.add_argument("--input-file", default=None)
        if command == "profile":
            child.add_argument("--output", default=None)
        elif command == "collect-traces":
            child.add_argument("--output", default=None)
            child.add_argument("--limit", type=int, default=-1)
            child.add_argument("--overwrite", action="store_true")
        elif command == "train":
            child.add_argument("--traces", default=None)
            child.add_argument("--output", default=None)
            child.add_argument("--steps", type=int, default=None)
            child.add_argument("--resume", default=None)
        elif command == "generate":
            child.add_argument("--output", required=True)
            child.add_argument("--mode", choices=("native", "speculative"), default="native")
            child.add_argument("--draft-checkpoint", default=None)
            child.add_argument("--limit", type=int, default=-1)
            child.add_argument("--overwrite", action="store_true")
        elif command == "benchmark":
            child.add_argument("--draft-checkpoint", required=True)
            child.add_argument("--output-dir", default=None)
            child.add_argument("--limit", type=int, default=32)
            child.add_argument("--repeats", type=int, default=3)
            child.add_argument("--warmup", type=int, default=1)
            child.add_argument("--require-exact", action="store_true")
        elif command == "diagnose-verifier":
            child.add_argument("--output", default=None)
            child.add_argument("--limit", type=int, default=8)
            child.add_argument("--prefix-tokens", type=int, default=32)
            child.add_argument("--block-sizes", type=int, nargs="+", default=[2, 4, 8])
    return parser


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    adapter = load_adapter(config)
    return config, adapter


def _command_profile(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    input_path = load_source_path(config, args.input_file, default_key="validation_file")
    row = next(read_jsonl(input_path, limit=1), None)
    if row is None:
        raise ValueError(f"Input file contains no JSONL rows: {input_path}")
    result = profile_t5gemma2(adapter, config, row)
    result["target"] = adapter.target_manifest()
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output is None:
        output = Path(config.get("benchmark", {}).get("output_dir", "./runs/drafter/benchmark"))
        if not output.is_absolute():
            output = (Path(config["_config_dir"]) / output).resolve()
        output = output / "profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"profile_file": str(output), **result}


def _command_collect(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    input_path = load_source_path(config, args.input_file, default_key="training_file")
    output_path = _resolve_cli_path(args.output) if args.output else config.get("training", {}).get("trace_file")
    if not output_path:
        raise ValueError("Pass --output or set training.trace_file")
    if not isinstance(output_path, Path):
        output_path = resolve_path(output_path, config)
    result = collect_teacher_traces(
        adapter,
        config,
        input_path,
        output_path,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    return result


def _command_train(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    trace_path = _resolve_cli_path(args.traces) if args.traces else config.get("training", {}).get("trace_file")
    if not trace_path:
        raise ValueError("Pass --traces or set training.trace_file")
    if not isinstance(trace_path, Path):
        trace_path = resolve_path(trace_path, config)
    output = _resolve_cli_path(args.output)
    resume = _resolve_cli_path(args.resume)
    return train_drafter(
        adapter,
        config,
        trace_path,
        output_path=output,
        max_steps=args.steps,
        resume_path=resume,
    )


def _command_generate(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    input_path = load_source_path(config, args.input_file, default_key="test_file")
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite predictions: {output_path}")
    drafter = None
    generator: Optional[GreedySpeculativeGenerator] = None
    if args.mode == "speculative":
        checkpoint = args.draft_checkpoint or config["draft"].get("checkpoint")
        if not checkpoint:
            raise ValueError("Speculative mode requires --draft-checkpoint or draft.checkpoint")
        checkpoint = (
            _resolve_cli_path(args.draft_checkpoint) if args.draft_checkpoint else resolve_path(checkpoint, config)
        )
        drafter = load_drafter(checkpoint, adapter, config)
        generator = GreedySpeculativeGenerator(adapter, config, drafter)

    data_config = config["data"]
    id_field = str(data_config.get("id_field", "id"))
    target_field = str(data_config.get("target_field", "summary"))
    produced = 0

    def prediction_rows():
        nonlocal produced
        for index, row in enumerate(read_jsonl(input_path, limit=args.limit)):
            text = source_text(row, config)
            input_ids, attention_mask = adapter.tokenize_source(text)
            if args.mode == "native":
                token_ids = [int(value) for value in adapter.native_generate(input_ids, attention_mask).cpu().tolist()]
                runtime_stats = None
            else:
                source = adapter.encode_ids(input_ids, attention_mask)
                result = generator.generate_source(source, str(row.get(id_field, index)))
                token_ids = result.decoder_ids
                runtime_stats = result.stats
            if token_ids and token_ids[0] == adapter.start_token_id:
                decoded_ids = token_ids[1:]
            else:
                decoded_ids = token_ids
            record = {
                "id": str(row.get(id_field, index)),
                "source": text,
                "prediction": adapter.tokenizer.decode(decoded_ids, skip_special_tokens=True).strip(),
                "prediction_ids": token_ids,
            }
            if target_field in row:
                record["reference"] = row[target_field]
            if runtime_stats is not None:
                record["runtime"] = runtime_stats
            produced += 1
            if produced % 10 == 0:
                print(f"Generated {produced} examples", flush=True)
            yield record

    count = write_jsonl(output_path, prediction_rows(), overwrite=args.overwrite)
    return {
        "mode": args.mode,
        "examples": count,
        "output_file": str(output_path),
        "target": adapter.target_manifest(),
        "draft_checkpoint": str(args.draft_checkpoint) if drafter is not None else None,
    }


def _command_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    input_path = load_source_path(config, args.input_file, default_key="test_file")
    draft_checkpoint = _resolve_cli_path(args.draft_checkpoint)
    output_dir = _resolve_cli_path(args.output_dir)
    return benchmark_runtime(
        adapter,
        config,
        input_path,
        limit=args.limit,
        repeats=args.repeats,
        warmup=args.warmup,
        draft_checkpoint=draft_checkpoint,
        output_dir=output_dir,
    )


def _command_diagnose_verifier(args: argparse.Namespace) -> Dict[str, Any]:
    config, adapter = _load(args)
    input_path = load_source_path(config, args.input_file, default_key="validation_file")
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    return diagnose_t5gemma2_verifier(
        adapter,
        config,
        input_path,
        limit=args.limit,
        prefix_tokens=args.prefix_tokens,
        block_sizes=args.block_sizes,
        output_path=output_path,
    )


def main() -> None:
    args = _parser().parse_args()
    handlers = {
        "profile": _command_profile,
        "collect-traces": _command_collect,
        "train": _command_train,
        "generate": _command_generate,
        "benchmark": _command_benchmark,
        "diagnose-verifier": _command_diagnose_verifier,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if args.command == "diagnose-verifier" and result["status"] != "passed":
        raise SystemExit(2)
    if args.command == "benchmark" and args.require_exact and result["exact_token_match_rate"] != 1.0:
        raise SystemExit(2)
