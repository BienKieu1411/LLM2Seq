#!/usr/bin/env python3
"""Generate Vietnamese summaries with SEAM vLLM and score with legacy ROUGE.

This is a separate vLLM runner for ``evaluate_rouge_vietnamese.py``. It keeps
the same source/reference directory layout, prediction preprocessing, and
ROUGE-1.5.5 command. HTTP mode calls an existing SEAM server without loading
weights; the previous in-process vLLM backend remains available as local mode.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

EVALUATION_DIR = Path(__file__).resolve().parent
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

import evaluate_rouge_vietnamese as rouge_eval


def _write_prediction(pair: rouge_eval.Pair, summary: str) -> None:
    pair.prediction.parent.mkdir(parents=True, exist_ok=True)
    temporary = pair.prediction.with_name(pair.prediction.name + ".tmp")
    temporary.write_text(summary.strip() + "\n", encoding="utf-8")
    temporary.replace(pair.prediction)


class SEAMHTTPClient:
    """Small stdlib client; no local model, torch or OpenAI SDK is required."""

    def __init__(self, base_url: str, api_key: str | None, *, timeout: float, retries: int):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base-url must be an HTTP(S) URL without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base-url must not contain a query or fragment")
        if timeout <= 0 or retries < 0:
            raise ValueError("HTTP timeout must be positive and retries must be nonnegative")
        self.base_url = base_url.rstrip("/") + ("/v1" if not parsed.path.strip("/") else "")
        self.api_key, self.timeout, self.retries = api_key, timeout, retries

    def request(self, path: str, body=None, *, fingerprint: str | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + path, data=data, headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    if fingerprint is not None and response.headers.get("x-seam-fingerprint") != fingerprint:
                        raise RuntimeError("SEAM server fingerprint changed or is missing; refusing to mix predictions")
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise RuntimeError("SEAM server returned a non-object JSON response")
                return result
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f"SEAM HTTP {exc.code} at {path}: {detail}") from None
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"SEAM HTTP request failed at {path}: {exc}") from None
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(f"SEAM returned invalid JSON at {path}") from None
            time.sleep(min(2.0, 0.5 * (2**attempt)))
        raise RuntimeError("SEAM HTTP retries exhausted")


def generate_http_predictions(
    pairs: list[rouge_eval.Pair],
    model_dir: Path,
    *,
    base_url: str,
    server_model: str,
    api_key: str | None,
    batch_size: int,
    timeout: float,
    retries: int,
    max_input_tokens: int | None,
    max_new_tokens: int | None,
    overwrite: bool,
    generation_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch-size/concurrency must be positive")
    client = SEAMHTTPClient(base_url, api_key, timeout=timeout, retries=retries)
    try:
        info = client.request("/seam/info")
    except RuntimeError as exc:
        raise RuntimeError(
            "Cannot read SEAM server metadata. Check the URL/API key and restart the host with the updated plugin. "
            + str(exc)
        ) from exc
    if (
        info.get("implementation") != "SEAMForQwen3"
        or not isinstance(info.get("fingerprint"), str)
        or not info["fingerprint"]
    ):
        raise ValueError("Endpoint is not the updated SEAM HTTP server; restart it with the current plugin")
    models = client.request("/models")
    if server_model not in [model.get("id") for model in models.get("data", [])]:
        raise ValueError(f"Server does not expose model {server_model!r}; check --served-model-name on the host")
    options = dict(info["generation_defaults"])
    overrides = dict(generation_options or {})
    if "min_new_tokens" in overrides:
        overrides["min_tokens"] = overrides.pop("min_new_tokens")
    do_sample = overrides.pop("do_sample", None)
    options.update(overrides)
    if max_new_tokens is not None:
        options["max_tokens"] = max_new_tokens
    if do_sample is False:
        options["temperature"] = 0.0
    elif do_sample is True and float(options.get("temperature", 0)) <= 0:
        raise ValueError("--do-sample requires a positive --temperature")
    source_limit = info["max_input_tokens"] if max_input_tokens is None else max_input_tokens
    if not 0 < source_limit <= info["max_input_tokens"] or options["max_tokens"] < 1:
        raise ValueError("Input/output budgets must be positive and the input cap must not exceed the server limit")
    options["max_input_tokens"] = source_limit
    body_defaults = {**options, "model": server_model, "n": 1, "stream": False}
    provenance = {
        "backend": "seam_vllm_http",
        "base_url": client.base_url,
        "server_model": server_model,
        "server": info,
        "generation_options": options,
    }
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode())
    for pair in pairs:
        stat = pair.source.stat()
        signature.update(f"{pair.source}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / ".seam_vllm_generation.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    existing = [pair for pair in pairs if pair.prediction.is_file() and pair.prediction.stat().st_size > 0]
    if existing and not overwrite and previous.get("signature") != signature.hexdigest():
        raise ValueError(
            "Existing predictions have a different or unknown server/checkpoint/prompt signature; use a new --model-name or --overwrite-predictions"
        )
    if overwrite:
        for pair in pairs:
            pair.prediction.unlink(missing_ok=True)
    manifest_path.write_text(
        json.dumps({"signature": signature.hexdigest(), **provenance}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    pending = [pair for pair in pairs if not pair.prediction.is_file() or pair.prediction.stat().st_size == 0]
    generated, started = 0, time.monotonic()

    def predict(pair):
        source = pair.source.read_text(encoding="utf-8-sig").strip()
        if not source:
            raise ValueError(f"Empty source file: {pair.source}")
        result = client.request("/completions", {**body_defaults, "prompt": source}, fingerprint=info["fingerprint"])
        choices = result.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("text"), str)
        ):
            raise RuntimeError(f"SEAM returned an invalid completion for {pair.source}")
        return pair, choices[0]["text"]

    print(
        f"[predict] backend=seam_vllm_http | model={server_model} | pending={len(pending)} | concurrency={batch_size}"
    )
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(predict, pair)
            for pair in (next(iterator, None) for _ in range(batch_size))
            if pair is not None
        }
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            errors = []
            for future in done:
                try:
                    pair, summary = future.result()
                    _write_prediction(pair, summary)
                    generated += 1
                except Exception as exc:
                    errors.append(exc)
            if errors:
                # Persist successful in-flight results as well; no new requests
                # are submitted after an error. Resume only reruns missing files.
                for future in futures:
                    if future.cancel():
                        continue
                    try:
                        pair, summary = future.result()
                        _write_prediction(pair, summary)
                    except Exception:
                        continue
                raise RuntimeError(f"Prediction failed after saving successful results: {errors[0]}") from errors[0]
            if generated == 1 or generated == len(pending) or generated % 10 == 0:
                print(
                    f"[predict] {generated}/{len(pending)} written | elapsed={(time.monotonic() - started) / 60:.1f}m"
                )
            for _ in done:
                pair = next(iterator, None)
                if pair is not None:
                    futures.add(executor.submit(predict, pair))
    return {
        **provenance,
        "generated": generated,
        "reused": len(pairs) - len(pending),
        "batch_size": batch_size,
        "max_input_tokens": source_limit,
        "max_new_tokens": options["max_tokens"],
    }


def _load_afmr_config(
    config_path: Path,
    *,
    max_input_tokens: int | None,
    max_new_tokens: int | None,
    temporary_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Write a temporary, absolute-path config without editing the run config."""
    afmr_root = Path(__file__).resolve().parents[1] / "eviseq_new"
    if not afmr_root.is_dir():
        raise FileNotFoundError(f"SEAM/AFMR source package not found: {afmr_root}")
    if str(afmr_root) not in sys.path:
        sys.path.insert(0, str(afmr_root))

    try:
        import yaml
        from eviseq_afmr.config import load_config, resolve_path
    except ImportError as exc:
        raise RuntimeError("SEAM vLLM evaluation requires PyYAML and the local eviseq_afmr package") from exc

    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SEAM resolved config not found: {config_path}")

    original = load_config(config_path)
    effective = copy.deepcopy(original)
    effective.pop("_meta", None)

    for key in ("encoder_name", "decoder_name"):
        effective["model"][key] = str(resolve_path(original["model"][key], original))
    for key in ("train_file", "validation_file", "test_file"):
        value = original["data"].get(key)
        if value:
            effective["data"][key] = str(resolve_path(value, original))

    if max_input_tokens is not None:
        effective["data"]["max_source_length"] = max_input_tokens
    if max_new_tokens is not None:
        effective["generation"]["max_new_tokens"] = max_new_tokens

    temporary_config = temporary_dir / "resolved_config.yaml"
    temporary_config.write_text(
        yaml.safe_dump(effective, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    lengths = {
        "max_input_tokens": int(effective["data"]["max_source_length"]),
        "max_new_tokens": int(effective["generation"]["max_new_tokens"]),
    }
    return temporary_config, lengths


def generate_vllm_predictions(
    pairs: list[rouge_eval.Pair],
    model_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_input_tokens: int | None,
    max_new_tokens: int | None,
    overwrite: bool,
    generation_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate through the local SEAM vLLM facade and save after each batch."""
    generation_options = dict(generation_options or {})
    if batch_size < 1 or max_num_seqs < 1:
        raise ValueError("batch size and max-num-seqs must be positive")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SEAM checkpoint not found: {checkpoint_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    generated_count = 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="seam-vllm-eval-") as temporary:
        temporary_config, lengths = _load_afmr_config(
            config_path,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            temporary_dir=Path(temporary),
        )
        signature = hashlib.sha256(temporary_config.read_bytes())
        stat = checkpoint_path.stat()
        signature.update(f"{checkpoint_path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        signature.update(json.dumps(generation_options, sort_keys=True).encode())
        for pair in pairs:
            stat = pair.source.stat()
            signature.update(f"{pair.source}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        manifest_path = model_dir / ".seam_vllm_generation.json"
        manifest = {"signature": signature.hexdigest(), "generation_options": generation_options, **lengths}
        existing = [pair for pair in pairs if pair.prediction.is_file() and pair.prediction.stat().st_size > 0]
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        if existing and not overwrite and previous.get("signature") != manifest["signature"]:
            raise ValueError(
                "Existing predictions have a different or unknown checkpoint/config/generation signature. "
                "Choose a new --model-name or pass --overwrite-predictions."
            )
        if overwrite:
            # Prevent an interrupted overwrite from later reusing old summaries
            # under the new model/parameter signature.
            for pair in pairs:
                pair.prediction.unlink(missing_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        pending = [pair for pair in pairs if not pair.prediction.is_file() or pair.prediction.stat().st_size == 0]
        if not pending:
            print("All predictions already exist; skipping vLLM generation.")
            return {
                "generated": 0,
                "reused": len(pairs),
                "generation_options": generation_options,
                **lengths,
            }

        try:
            from seam_vllm import SEAMLLM
        except ImportError as exc:
            raise RuntimeError(
                "Could not import seam_vllm. Install the SEAM plugin into the active vLLM 0.25.1 environment."
            ) from exc

        print(
            f"[predict] backend=seam_vllm | pending={len(pending)} | batch_size={batch_size} "
            f"| max_num_seqs={max_num_seqs} | max_input={lengths['max_input_tokens']} "
            f"| max_new={lengths['max_new_tokens']} | gpu_memory_utilization={gpu_memory_utilization:.2f}"
        )
        model = SEAMLLM(
            config_path=temporary_config,
            checkpoint_path=checkpoint_path,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            sources = [pair.source.read_text(encoding="utf-8-sig").strip() for pair in batch]
            empty = [str(pair.source) for pair, source in zip(batch, sources, strict=True) if not source]
            if empty:
                raise ValueError("Empty source file(s): " + ", ".join(empty))

            summaries = model.summarize(sources, **generation_options)
            if isinstance(summaries, str):
                summaries = [summaries]
            if len(summaries) != len(batch):
                raise RuntimeError(f"SEAM vLLM returned {len(summaries)} summaries for a batch of {len(batch)} sources")

            for pair, summary in zip(batch, summaries, strict=True):
                _write_prediction(pair, summary)

            generated_count += len(batch)
            elapsed = time.monotonic() - started
            print(
                f"[predict] {min(start + len(batch), len(pending))}/{len(pending)} written "
                f"| elapsed={elapsed / 60:.1f}m"
            )

    return {
        "generated": generated_count,
        "reused": len(pairs) - len(pending),
        "generation_options": generation_options,
        **lengths,
    }


def score_and_report(
    model_name: str,
    pairs: list[rouge_eval.Pair],
    model_dir: Path,
    rouge_path: Path,
    keep_temp: Path | None,
    generation: dict[str, Any],
) -> Path:
    """Use the same ROUGE-1.5.5 preprocessing and flags as the existing evaluator."""
    by_dataset: dict[str, list[rouge_eval.Pair]] = defaultdict(list)
    for pair in pairs:
        by_dataset[pair.dataset].append(pair)

    print(f"\nROUGE-1.5.5 on {len(pairs)} matched files")
    overall = rouge_eval.run_pyrouge(pairs, rouge_path, keep_temp)
    datasets: dict[str, Any] = {}
    for dataset, dataset_pairs in sorted(by_dataset.items()):
        print(f"  scoring {dataset} ({len(dataset_pairs)} files)")
        datasets[dataset] = rouge_eval.run_pyrouge(dataset_pairs, rouge_path, keep_temp)

    report = {
        "model": model_name,
        "implementation": "PyROUGE with ROUGE-1.5.5.pl",
        "rouge_path": str(rouge_path),
        "rouge_args": "-a -m -n 2 -d",
        "score_scale": "0_to_1",
        "display_scale": "percentage",
        "prediction_preprocessing": (
            f"lowercase, legacy bracket remapping, remove lines containing {rouge_eval.PREAMBLE!r}"
        ),
        "reference_preprocessing": (
            "blank-separated blocks are multiple references; lowercase and legacy bracket remapping"
        ),
        "aggregation": "ROUGE-1.5.5 corpus/system aggregation",
        "generation": generation,
        "overall": overall,
        "datasets": datasets,
    }
    output_path = model_dir / "rouge_vllm_summary.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "  ROUGE-1/2/L F1 (%): "
        f"{100.0 * overall['rouge1_fmeasure']:.3f} / "
        f"{100.0 * overall['rouge2_fmeasure']:.3f} / "
        f"{100.0 * overall['rougeL_fmeasure']:.3f}"
    )
    print(f"  Summary: {output_path}")
    return output_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese summaries with SEAM vLLM and evaluate using ROUGE-1.5.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-raw-root",
        "--source-root",
        dest="source_root",
        type=Path,
        default=rouge_eval.DATA_RAW_ROOT,
        help="Root containing <dataset>/*.txt source documents",
    )
    parser.add_argument("--data-ref-root", type=Path, default=rouge_eval.DATA_REF_ROOT)
    parser.add_argument("--output-root", type=Path, default=rouge_eval.DATA_SUM_ROOT)
    parser.add_argument(
        "--backend",
        choices=("http", "local"),
        default=None,
        help="Auto: HTTP unless local config/checkpoint are supplied",
    )
    parser.add_argument("--base-url", default=None, help="SEAM HTTP API base; defaults to http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model",
        "--served-model-name",
        dest="server_model",
        default="seam",
        help="Model alias exposed by the HTTP server",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=600.0, help="HTTP request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transient HTTP errors")
    parser.add_argument("--afmr-config", type=Path, default=None, help="Local backend: run's resolved_config.yaml")
    parser.add_argument("--afmr-checkpoint", type=Path, default=None, help="Local backend: SEAM/AFMR .pt checkpoint")
    parser.add_argument(
        "--model-name", default=None, help="Prediction subfolder; defaults to served-model_http or checkpoint_vllm"
    )
    parser.add_argument(
        "--batch-size",
        "--concurrency",
        type=int,
        default=4,
        help="HTTP requests in flight, or local inference batch size",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM concurrent sequence limit; defaults to --batch-size",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=None, help="Local backend only; defaults to 0.90"
    )
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Encoder token cap including its prefix")
    parser.add_argument("--max-new-tokens", "--max-tokens", type=int, default=None, help="Decoder output token budget")
    parser.add_argument("--encoder-prompt", default=None, help="Override encoder prefix; default from model config")
    parser.add_argument("--decoder-prompt", default=None, help="Override decoder instruction and bridge conditioning")
    parser.add_argument("--decoder-prefix", default=None)
    parser.add_argument("--decoder-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--repetition-penalty", type=float, default=None, help="Native vLLM penalty (includes prompt); default 1.0"
    )
    parser.add_argument("--seed", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--do-sample", dest="do_sample", action="store_true")
    mode.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=None)
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--rouge-path", type=Path, default=rouge_eval.ROUGE_PATH)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--keep-temp", type=Path, default=None)
    args = parser.parse_args(argv)
    args.backend = args.backend or (
        "http" if args.base_url or not (args.afmr_config or args.afmr_checkpoint) else "local"
    )
    if args.backend == "local":
        if not args.afmr_config or not args.afmr_checkpoint:
            parser.error("Local evaluation requires --afmr-config and --afmr-checkpoint")
        if args.base_url:
            parser.error("--base-url is an HTTP backend option")
    else:
        if args.afmr_config or args.afmr_checkpoint:
            parser.error("HTTP evaluation uses the hosted model; omit local --afmr-config/--afmr-checkpoint")
        if args.max_num_seqs is not None or args.gpu_memory_utilization is not None:
            parser.error("Set max-num-seqs/GPU memory on the host; use --concurrency for HTTP evaluation")
    if args.batch_size < 1:
        parser.error("--batch-size/--concurrency must be positive")
    for key in ("max_input_tokens", "max_new_tokens"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    return args


def main() -> int:
    args = parse_args()
    checkpoint_path = config_path = None
    if args.backend == "local":
        checkpoint_path = args.afmr_checkpoint.expanduser().resolve()
        config_path = args.afmr_config.expanduser().resolve()
        for label, path in (("checkpoint", checkpoint_path), ("config", config_path)):
            if not path.is_file():
                print(f"ERROR: SEAM {label} not found: {path}", file=sys.stderr)
                return 2
    default_name = (
        f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_vllm"
        if checkpoint_path
        else args.server_model.replace("/", "_") + "_http"
    )
    model_name = args.model_name or default_name
    model_dir = args.output_root.expanduser().resolve() / model_name
    try:
        pairs, missing, empty_sources = rouge_eval.discover_source_pairs(
            args.source_root.expanduser().resolve(),
            args.data_ref_root.expanduser().resolve(),
            model_dir,
            model_name,
        )
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if empty_sources:
        print(f"Skipped {len(empty_sources)} empty/whitespace-only source files.", file=sys.stderr)
        for path in empty_sources[:20]:
            print(f"  empty: {path}", file=sys.stderr)
    if missing:
        print(f"Found {len(missing)} source files without matching references.", file=sys.stderr)
        for path in missing[:20]:
            print(f"  missing: {path}", file=sys.stderr)
        if not args.allow_missing:
            print("Use --allow-missing to evaluate matched pairs only.", file=sys.stderr)
            return 2
    if not pairs:
        print("ERROR: no matching source/reference .txt files found", file=sys.stderr)
        return 2

    rouge_path = args.rouge_path.expanduser().resolve()
    try:
        rouge_eval.validate_rouge_install(rouge_path)
        import pyrouge  # noqa: F401
    except (ImportError, OSError) as exc:
        print(f"ERROR: ROUGE-1.5.5 dependency unavailable: {exc}", file=sys.stderr)
        return 2

    max_num_seqs = args.max_num_seqs if args.max_num_seqs is not None else args.batch_size
    local_memory = 0.90 if args.gpu_memory_utilization is None else args.gpu_memory_utilization
    started = time.monotonic()
    try:
        common = dict(
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite_predictions,
            generation_options={
                key: getattr(args, key)
                for key in (
                    "min_new_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "seed",
                    "do_sample",
                    "encoder_prompt",
                    "decoder_prompt",
                    "decoder_prefix",
                    "decoder_chat_template",
                )
                if getattr(args, key) is not None
            },
        )
        if args.backend == "http":
            generation = generate_http_predictions(
                pairs,
                model_dir,
                base_url=args.base_url or "http://127.0.0.1:8000/v1",
                server_model=args.server_model,
                api_key=args.api_key or os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                timeout=args.timeout,
                retries=args.retries,
                **common,
            )
        else:
            generation = generate_vllm_predictions(
                pairs,
                model_dir,
                config_path,
                checkpoint_path,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=local_memory,
                **common,
            )
            generation.update(
                {
                    "backend": "seam_vllm",
                    "checkpoint": str(checkpoint_path),
                    "afmr_config": str(config_path),
                    "batch_size": args.batch_size,
                    "max_num_seqs": max_num_seqs,
                    "gpu_memory_utilization": local_memory,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
            )
        generation["existing_predictions_reused"] = generation["reused"] > 0
        score_and_report(
            model_name,
            pairs,
            model_dir,
            rouge_path,
            args.keep_temp,
            generation,
        )
    except (ImportError, OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"ERROR during SEAM vLLM evaluation: {exc}", file=sys.stderr)
        return 2

    print(f"\nCompleted SEAM vLLM + ROUGE evaluation in {(time.monotonic() - started) / 60:.2f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
