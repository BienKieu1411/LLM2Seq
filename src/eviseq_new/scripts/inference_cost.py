from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 3
SEAM_SYSTEMS = {"seam": "full", "vanilla": "vanilla"}
DECODER_ONLY_MODELS = {
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    "qwen3_1_7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
}
SEQ2SEQ_MODELS = {"t5gemma2_1b": "google/t5gemma-2-1b-1b"}
BASELINE_MODELS = {**DECODER_ONLY_MODELS, **SEQ2SEQ_MODELS}
ALL_SYSTEMS = (*SEAM_SYSTEMS, *BASELINE_MODELS)
SEAM_BATCH_KEYS = (
    "input_ids",
    "attention_mask",
    "source_content_mask",
    "decoder_prompt_ids",
    "decoder_prompt_mask",
)
_WORDS = (
    "the study patients results method analysis clinical data model treatment effect group risk "
    "increased reduced significant associated measured observed compared baseline outcome cohort "
    "response protein expression cells therapy disease sample rate level change after before with "
    "without between during overall primary secondary conclusion objective background we found that"
).split()


def synthetic_document(n_words: int, seed: int = 0) -> str:
    generator = random.Random(seed)
    words = []
    for index in range(n_words):
        word = str(generator.randint(1, 9999)) if index % 11 == 10 else generator.choice(_WORDS)
        words.append(word + "." if index % 17 == 16 else word)
    return " ".join(words)


def arm_config(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ("full", "vanilla"):
        raise ValueError(f"Unknown arm {arm!r}; choose full or vanilla")
    result = copy.deepcopy(config)
    full = arm == "full"
    result["architecture"]["bridge_mode"] = "afmr" if full else "direct_projection"
    result["decoder"]["grounded_copy"]["enabled"] = full
    return result


def baseline_models(overrides: Sequence[str]) -> dict[str, str]:
    models = dict(BASELINE_MODELS)
    for value in overrides:
        name, separator, path = value.partition("=")
        if not separator or name not in models or not path:
            raise ValueError(f"--model expects NAME=PATH with NAME in {sorted(models)}, got {value!r}")
        models[name] = path
    return models


def _count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _weights_gib(model: torch.nn.Module) -> float:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 2**30


def _require_budget(length: int, source_tokens: int) -> None:
    if length < source_tokens:
        raise ValueError(f"Document too short: {length} tokens for a source budget of {source_tokens}")


class SeamSystem:
    def __init__(self, config: dict[str, Any], device: torch.device):
        from eviseq_afmr.data.collate import SummarizationCollator
        from eviseq_afmr.modeling.model import EviSeqAFMR
        from eviseq_afmr.runtime import _tokenizers

        self.device = device
        encoder_tokenizer, self.tokenizer = _tokenizers(config)
        self.collator = SummarizationCollator(
            encoder_tokenizer,
            self.tokenizer,
            config["data"],
            grounded_copy=bool(config["decoder"]["grounded_copy"]["enabled"]),
        )
        self.collator.include_targets = False
        self.model = EviSeqAFMR(config).to(device).eval()

    def prepare(self, documents: Sequence[str], source_tokens: int) -> dict[str, torch.Tensor]:
        from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
        from eviseq_afmr.data.schema import CanonicalRecord

        self.collator.max_source_length = int(source_tokens)
        records = [
            CanonicalRecord(example_id=str(index), source=text, target="") for index, text in enumerate(documents)
        ]
        batch = self.collator(records)
        _require_budget(int(batch["attention_mask"].sum(-1).min()), source_tokens)
        return {
            key: value.to(self.device)
            for key, value in batch.items()
            if key in (*SEAM_BATCH_KEYS, *COPY_INPUT_KEYS)
        }

    def source_length(self, batch: dict[str, torch.Tensor]) -> int:
        return int(batch["attention_mask"].sum(-1).min())

    def generate(self, batch: dict[str, torch.Tensor], new_tokens: int) -> int:
        from eviseq_afmr.evaluation.generate import generate_greedy

        _, token_ids = generate_greedy(
            self.model,
            batch,
            self.tokenizer,
            new_tokens,
            new_tokens,
            REPETITION_PENALTY,
            NO_REPEAT_NGRAM_SIZE,
        )
        return int(token_ids.shape[1] - batch["decoder_prompt_ids"].shape[1])

    def parameter_counts(self) -> dict[str, int]:
        decoder = self.model.decoder
        cross = []
        for layer in decoder.decoder_layers:
            cross += [layer.cross_gate, *layer.cross_norm.parameters(), *layer.cross.parameters()]
        counts = {
            "encoder": _count(self.model.encoder.parameters()),
            "cross_attention": _count(cross),
            "bridge": _count(self.model.bridge.parameters()),
            "source_copy": _count(decoder.grounded_copy.parameters()) if decoder.grounded_copy is not None else 0,
        }
        total = _count(self.model.parameters())
        return {
            **counts,
            "decoder_backbone": total - sum(counts.values()),
            "interface": counts["cross_attention"] + counts["bridge"] + counts["source_copy"],
            "total": total,
        }

    def weights_gib(self) -> float:
        return _weights_gib(self.model)


class _RepositoryPenalties:
    def __init__(self, start: int):
        self.start = start

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        from eviseq_afmr.evaluation import generate

        history = input_ids[:, self.start :]
        generate._apply_repetition_penalty(scores, history, REPETITION_PENALTY)
        generate._apply_no_repeat_ngram(scores, history, NO_REPEAT_NGRAM_SIZE)
        return scores


class DecoderOnlySystem:
    PREFIX ="Summarize the following document concisely.\nDocument:\n"
    SUFFIX = "\nSummary:\n"

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def prepare(self, documents: Sequence[str], source_tokens: int) -> dict[str, torch.Tensor]:
        suffix = list(self.tokenizer.encode(self.SUFFIX, add_special_tokens=False))
        keep = source_tokens - len(suffix)
        rows = []
        for text in documents:
            ids = list(self.tokenizer.encode(self.PREFIX + text, add_special_tokens=False))
            _require_budget(len(ids) + len(suffix), source_tokens)
            rows.append(ids[:keep] + suffix)
        input_ids = torch.tensor(rows, device=self.device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def source_length(self, batch: dict[str, torch.Tensor]) -> int:
        return int(batch["input_ids"].shape[1])

    @torch.inference_mode()
    def generate(self, batch: dict[str, torch.Tensor], new_tokens: int) -> int:
        from transformers import LogitsProcessorList

        start = self._history_start(batch)
        output = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            num_beams=1,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            logits_processor=LogitsProcessorList([_RepositoryPenalties(start)]),
            pad_token_id=int(self.tokenizer.pad_token_id),
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        count = int(output.shape[1] - start)
        self.tokenizer.batch_decode(output[:, start:], skip_special_tokens=True)
        return count

    def _history_start(self, batch: dict[str, torch.Tensor]) -> int:
        return int(batch["input_ids"].shape[1])

    def parameter_counts(self) -> dict[str, int]:
        return {"total": _count(self.model.parameters())}

    def weights_gib(self) -> float:
        return _weights_gib(self.model)


class Seq2SeqSystem(DecoderOnlySystem):
    def prepare(self, documents: Sequence[str], source_tokens: int) -> dict[str, torch.Tensor]:
        rows = []
        for text in documents:
            ids = self.tokenizer(
                self.PREFIX + text, add_special_tokens=True, truncation=True, max_length=source_tokens
            )["input_ids"]
            _require_budget(len(ids), source_tokens)
            rows.append(list(ids))
        input_ids = torch.tensor(rows, device=self.device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def _history_start(self, batch: dict[str, torch.Tensor]) -> int:
        return 1

    def parameter_counts(self) -> dict[str, int]:
        vision = {
            id(parameter): parameter.numel()
            for name, module in self.model.named_modules()
            if name.rsplit(".", 1)[-1] in ("vision_tower", "multi_modal_projector")
            for parameter in module.parameters()
        }
        total = _count(self.model.parameters())
        return {"text": total - sum(vision.values()), "vision": sum(vision.values()), "total": total}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _timed(action: Callable[[], Any], device: torch.device, repeats: int) -> tuple[float, Any]:
    seconds, result = [], None
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        result = action()
        _synchronize(device)
        seconds.append(time.perf_counter() - started)
    return statistics.median(seconds), result


def measure(
    system: Any, *, source_tokens: int, batch_size: int, new_tokens: int, repeats: int
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_tokens": source_tokens,
        "batch_size": batch_size,
        "new_tokens": new_tokens,
        "repeats": repeats,
    }
    device = system.device
    documents = [synthetic_document(source_tokens + 64, seed=index) for index in range(batch_size)]
    try:
        preprocess, batch = _timed(lambda: system.prepare(documents, source_tokens), device, repeats)
        row["source_tokens"] = system.source_length(batch)
        system.generate(batch, min(4, new_tokens))
        first_token, _ = _timed(lambda: system.generate(batch, 1), device, repeats)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        total, generated = _timed(lambda: system.generate(batch, new_tokens), device, repeats)
        if generated != new_tokens:
            raise RuntimeError(f"Expected {new_tokens} new tokens, the system produced {generated}")
        on_gpu = device.type == "cuda"
        row.update(
            status="ok",
            preprocess_s_per_example=preprocess / batch_size,
            first_token_s=first_token,
            total_s=total,
            latency_s_per_example=total / batch_size,
            output_tokens_per_s=batch_size * new_tokens / total,
            decode_tokens_per_s=batch_size * (new_tokens - 1) / (total - first_token) if total > first_token else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if on_gpu else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if on_gpu else None,
        )
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
    batch = None
    _release_memory()
    return row


def read_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _workload_key(row: dict[str, Any]) -> tuple:
    names = ("system", "source_tokens", "batch_size", "new_tokens", "gpu", "dtype", "torch", "transformers")
    return tuple(row.get(name) for name in names)


def run(
    systems: Sequence[str],
    *,
    source_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    new_tokens: int,
    repeats: int,
    output: Path,
    builder: Callable[[str], Any],
    environment: dict[str, Any] | None = None,
) -> dict[str, str]:
    output = Path(output)
    environment = dict(environment or {})
    finished = {_workload_key(row) for row in read_rows(output)}
    failures: dict[str, str] = {}
    for name in systems:
        base = {"system": name, "new_tokens": new_tokens, **environment}
        pending = [
            (length, size)
            for length in source_lengths
            for size in batch_sizes
            if _workload_key({**base, "source_tokens": length, "batch_size": size}) not in finished
        ]
        if not pending:
            continue
        try:
            system = builder(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[cost] could not load {name}: {failures[name]}", flush=True)
            _release_memory()
            continue
        static = {**base, "parameters": system.parameter_counts(), "weights_gib": system.weights_gib()}
        for length, size in pending:
            row = {
                **static,
                **measure(system, source_tokens=length, batch_size=size, new_tokens=new_tokens, repeats=repeats),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(f"[cost] {name} source={length} batch={size} -> {row['status']}", flush=True)
        system = None
        _release_memory()
    return failures


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    columns = (
        "system",
        "params (B)",
        "weights (GiB)",
        "source tokens",
        "batch",
        "first token (s)",
        "total (s)",
        "output tok/s",
        "peak (GiB)",
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        if row["status"] == "ok":
            peak = row.get("peak_allocated_gib")
            measured = [
                f"{row['first_token_s']:.2f}",
                f"{row['total_s']:.2f}",
                f"{row['output_tokens_per_s']:.1f}",
                "-" if peak is None else f"{peak:.2f}",
            ]
        else:
            measured = ["OOM"] * 4
        cells = [
            row["system"],
            f"{row['parameters']['total'] / 1e9:.3f}",
            f"{row['weights_gib']:.2f}",
            str(row["source_tokens"]),
            str(row["batch_size"]),
            *measured,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _builder(
    config_path: Path, models: dict[str, str], device: torch.device, dtype_name: str
) -> Callable[[str], Any]:
    def build(name: str) -> Any:
        if name in SEAM_SYSTEMS:
            from eviseq_afmr.config import load_config
            from eviseq_afmr.runtime import _configure_precision

            config = arm_config(load_config(config_path), SEAM_SYSTEMS[name])
            config["model"]["dtype"] = dtype_name
            _configure_precision(config)
            return SeamSystem(config, device)
        import transformers

        seq2seq = name in SEQ2SEQ_MODELS
        loader = transformers.AutoModelForSeq2SeqLM if seq2seq else transformers.AutoModelForCausalLM
        tokenizer = transformers.AutoTokenizer.from_pretrained(models[name], trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = loader.from_pretrained(
            models[name], dtype=getattr(torch, dtype_name), attn_implementation="sdpa", trust_remote_code=True
        )
        return (Seq2SeqSystem if seq2seq else DecoderOnlySystem)(model, tokenizer, device)

    return build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inference latency, throughput and peak memory: full model, vanilla arm and baselines."
    )
    parser.add_argument("--config", type=Path, required=True, help="resolved_config.yaml of a trained run")
    parser.add_argument("--output", type=Path, required=True, help="JSONL file; finished workloads are skipped")
    parser.add_argument("--systems", nargs="+", choices=ALL_SYSTEMS, default=list(ALL_SYSTEMS))
    parser.add_argument("--model", action="append", default=[], help="NAME=PATH for a local baseline; repeatable")
    parser.add_argument("--source-lengths", nargs="+", type=int, default=[4096, 8192, 12288])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    models = baseline_models(args.model)
    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU is visible; CPU timings would not answer the deployment-cost question")
    import transformers

    device = torch.device("cuda")
    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    print(f"[cost] {environment}", flush=True)
    failures = run(
        args.systems,
        source_lengths=args.source_lengths,
        batch_sizes=args.batch_sizes,
        new_tokens=args.new_tokens,
        repeats=args.repeats,
        output=args.output,
        builder=_builder(args.config, models, device, args.dtype),
        environment=environment,
    )
    rows = read_rows(args.output)
    print(markdown_table(rows))
    for name in (*SEAM_SYSTEMS, *SEQ2SEQ_MODELS):
        counts = next((row["parameters"] for row in rows if row["system"] == name), None)
        if counts:
            print(f"[cost] {name} parameters: {counts}")
    for name, reason in failures.items():
        print(f"[cost] NOT MEASURED {name}: {reason}")


if __name__ == "__main__":
    main()
