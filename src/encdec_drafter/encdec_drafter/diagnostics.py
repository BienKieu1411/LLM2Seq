from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import torch

from .data import read_jsonl, source_text
from .generation import GreedySpeculativeGenerator
from .policy import GreedyPolicy


class _ProbeDrafter:
    """Deliberately wrong fixed-token proposals for verifier parity checks."""

    def __init__(self, vocab_size: int, max_block_size: int, target_feature_window: int, eos_token_ids):
        self.vocab_size = int(vocab_size)
        self.max_block_size = int(max_block_size)
        self.shape = SimpleNamespace(target_feature_window=int(target_feature_window))
        self.probe_token_id = self.vocab_size - 1
        while self.probe_token_id in eos_token_ids and self.probe_token_id > 0:
            self.probe_token_id -= 1

    def prepare_source(self, hidden_states, attention_mask, fingerprint, mode, budget):
        return SimpleNamespace(retained_tokens=int(attention_mask.sum().item()))

    def __call__(self, pending_ids, accepted_features, source, input_embedding, output_head, block_size):
        logits = torch.zeros(
            (1, int(block_size), self.vocab_size),
            dtype=torch.float32,
            device=pending_ids.device,
        )
        logits[:, :, self.probe_token_id] = 10.0
        return logits


class _OracleDrafter(_ProbeDrafter):
    """Uses native greedy tokens to exercise the all-proposals-accepted path."""

    def __init__(self, vocab_size: int, max_block_size: int, target_feature_window: int, eos_token_ids, target_ids):
        super().__init__(vocab_size, max_block_size, target_feature_window, eos_token_ids)
        self.target_ids = [int(token_id) for token_id in target_ids]

    def __call__(self, pending_ids, accepted_features, source, input_embedding, output_head, block_size):
        start = int(accepted_features.shape[1]) + 1
        proposals = self.target_ids[start : start + int(block_size)]
        if len(proposals) != int(block_size):
            raise RuntimeError("Oracle drafter ran past the native reference continuation")
        logits = torch.full(
            (1, int(block_size), self.vocab_size),
            -100.0,
            dtype=torch.float32,
            device=pending_ids.device,
        )
        for position, token_id in enumerate(proposals):
            logits[0, position, token_id] = 10.0
        return logits


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    """Compare same-prefix logits without hiding top-token changes behind a tolerance."""
    if reference.shape != candidate.shape:
        return {
            "same_shape": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "top1_exact": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "top1_mismatch_positions": [],
        }

    reference = reference.detach().float()
    candidate = candidate.detach().float()
    differences = (reference - candidate).abs()
    mismatch_positions = (
        (reference.argmax(dim=-1) != candidate.argmax(dim=-1))
        .reshape(-1)
        .nonzero(as_tuple=False)
        .flatten()
        .cpu()
        .tolist()
    )
    return {
        "same_shape": True,
        "top1_exact": not mismatch_positions,
        "top1_mismatch_positions": [int(position) for position in mismatch_positions],
        "max_abs_diff": float(differences.max().item()) if differences.numel() else 0.0,
        "mean_abs_diff": float(differences.mean().item()) if differences.numel() else 0.0,
    }


def _fresh_generation_config(config: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    result = dict(config)
    generation = dict(config["generation"])
    generation["max_new_tokens"] = int(max_new_tokens)
    generation["min_new_tokens"] = min(int(generation.get("min_new_tokens", 0)), max_new_tokens)
    result["generation"] = generation
    return result


def compare_processed_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    decoder_ids: Sequence[int],
    policy: GreedyPolicy,
    ambiguity_margin: float,
) -> Dict[str, Any]:
    """Compare actual greedy decisions after generation processors are applied."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[0] != 1:
        return {
            "same_shape": False,
            "top1_exact": False,
            "fallback_safe": False,
            "top1_mismatch_positions": [],
            "mismatch_details": [],
        }

    ref = reference[0].float()
    cand = candidate[0].float()
    mismatch_positions: List[int] = []
    details: List[Dict[str, Any]] = []
    candidate_margins: List[float] = []
    for position in range(ref.shape[0]):
        prefix = decoder_ids[: position + 1]
        ref_scores = policy.process_block(ref[position : position + 1], prefix, None, position)[0]
        cand_scores = policy.process_block(cand[position : position + 1], prefix, None, position)[0]
        ref_top = torch.topk(ref_scores, k=2)
        cand_top = torch.topk(cand_scores, k=2)
        ref_id = int(ref_top.indices[0].item())
        cand_id = int(cand_top.indices[0].item())
        cand_margin = float((cand_top.values[0] - cand_top.values[1]).item())
        candidate_margins.append(cand_margin)
        if ref_id != cand_id:
            mismatch_positions.append(position)
            details.append(
                {
                    "position": position,
                    "reference_top1": ref_id,
                    "candidate_top1": cand_id,
                    "reference_margin": float((ref_top.values[0] - ref_top.values[1]).item()),
                    "candidate_margin": cand_margin,
                    "candidate_within_fallback_margin": cand_margin <= ambiguity_margin,
                }
            )
    return {
        "same_shape": True,
        "top1_exact": not mismatch_positions,
        "top1_mismatch_positions": mismatch_positions,
        "mismatch_details": details,
        "candidate_min_margin": min(candidate_margins, default=None),
        "ambiguity_fallback_margin": ambiguity_margin,
        "fallback_safe": all(item["candidate_within_fallback_margin"] for item in details),
    }


@torch.inference_mode()
def diagnose_t5gemma2_verifier(
    adapter,
    config: Dict[str, Any],
    input_path: str | Path,
    *,
    limit: int = 8,
    prefix_tokens: int = 32,
    block_sizes: Sequence[int] = (2, 4, 8),
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Check cached token-wise and cached block-wise logits against the same prefixes.

    This is a pre-training correctness gate. It deliberately does not load a drafter:
    target verification must be correct regardless of draft quality.
    """
    if limit < 1 or prefix_tokens < 2:
        raise ValueError("limit must be positive and prefix_tokens must be at least 2")
    sizes = tuple(sorted({int(size) for size in block_sizes}))
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("block_sizes must contain positive integers")

    rows = list(read_jsonl(input_path, limit=limit))
    if not rows:
        raise ValueError(f"No diagnostic examples found in {input_path}")

    cached_generator = GreedySpeculativeGenerator(
        adapter,
        _fresh_generation_config(config, prefix_tokens),
        drafter=None,
    )
    probe_config = _fresh_generation_config(config, prefix_tokens)
    probe_block_size = min(4, int(config["draft"].get("max_block_size", 4)))
    probe_config["speculation"] = {
        **dict(config["speculation"]),
        "block_size": probe_block_size,
    }
    probe_drafter = _ProbeDrafter(
        adapter.vocab_size,
        max_block_size=probe_block_size,
        target_feature_window=int(config["draft"].get("target_feature_window", 64)),
        eos_token_ids=adapter.eos_token_ids,
    )
    probe_generator = GreedySpeculativeGenerator(adapter, probe_config, drafter=probe_drafter)
    generation = config["generation"]
    policy = GreedyPolicy(
        vocab_size=adapter.vocab_size,
        eos_token_ids=adapter.eos_token_ids,
        min_new_tokens=int(generation.get("min_new_tokens", 0)),
        repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
        pad_token_id=int(adapter.pad_token_id),
    )
    ambiguity_margin = float(config["speculation"].get("ambiguity_fallback_margin", 0.5))
    request_results: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        text = source_text(row, config)
        input_ids, attention_mask = adapter.tokenize_source(text)
        source = adapter.encode_ids(input_ids, attention_mask)
        native_ids = adapter.native_generate(input_ids, attention_mask, max_new_tokens=prefix_tokens)
        cached_ids = cached_generator.generate_source(source).decoder_ids
        probe_result = probe_generator.generate_source(source)
        oracle_drafter = _OracleDrafter(
            adapter.vocab_size,
            max_block_size=probe_block_size,
            target_feature_window=int(config["draft"].get("target_feature_window", 64)),
            eos_token_ids=adapter.eos_token_ids,
            target_ids=[int(value) for value in native_ids.tolist()],
        )
        oracle_generator = GreedySpeculativeGenerator(adapter, probe_config, drafter=oracle_drafter)
        oracle_result = oracle_generator.generate_source(source)
        decoder_inputs = native_ids[: min(int(native_ids.numel()), prefix_tokens + 1)]
        if decoder_inputs.numel() < 2:
            raise RuntimeError(f"Native generation returned no decoder continuation for row {row_index}")

        full_cache = adapter.new_cache()
        full_logits, _ = adapter.forward_decoder(source, decoder_inputs, full_cache)

        serial_cache = adapter.new_cache()
        serial_logits: List[torch.Tensor] = []
        for token_id in decoder_inputs:
            logits, _ = adapter.forward_decoder(source, token_id.reshape(1), serial_cache)
            serial_logits.append(logits)
        serial_logits_tensor = torch.cat(serial_logits, dim=1)
        full_vs_serial = compare_logits(full_logits, serial_logits_tensor)
        full_policy_vs_serial = compare_processed_logits(
            full_logits, serial_logits_tensor, decoder_inputs.tolist(), policy, ambiguity_margin
        )

        by_block_size: Dict[str, Any] = {}
        processed_by_block_size: Dict[str, Any] = {}
        for block_size in sizes:
            block_logits: List[torch.Tensor] = []
            decoder_length = int(decoder_inputs.numel())
            for start in range(0, decoder_length, block_size):
                cache = adapter.new_cache()
                if start:
                    adapter.forward_decoder(source, decoder_inputs[:start], cache)
                end = min(decoder_length, start + block_size)
                logits, _ = adapter.forward_decoder(source, decoder_inputs[start:end], cache)
                block_logits.append(logits)
            block_logits_tensor = torch.cat(block_logits, dim=1)
            by_block_size[str(block_size)] = compare_logits(serial_logits_tensor, block_logits_tensor)
            processed_by_block_size[str(block_size)] = compare_processed_logits(
                serial_logits_tensor,
                block_logits_tensor,
                decoder_inputs.tolist(),
                policy,
                ambiguity_margin,
            )

        identifier = row.get(config["data"].get("id_field", "id"), row_index)
        request_results.append(
            {
                "id": str(identifier),
                "source_tokens": int(attention_mask[0].sum().item()),
                "decoder_prefix_tokens": int(decoder_inputs.numel()),
                "native_vs_cached_target_only_exact": [int(value) for value in native_ids.tolist()] == cached_ids,
                "native_vs_probe_speculative_exact": [int(value) for value in native_ids.tolist()]
                == probe_result.decoder_ids,
                "probe_ambiguity_fallbacks": int(probe_result.stats["ambiguity_fallbacks"]),
                "native_vs_oracle_speculative_exact": [int(value) for value in native_ids.tolist()]
                == oracle_result.decoder_ids,
                "oracle_accepted_tokens": int(oracle_result.stats["accepted_tokens"]),
                "oracle_ambiguity_fallbacks": int(oracle_result.stats["ambiguity_fallbacks"]),
                "native_decoder_ids": [int(value) for value in native_ids.tolist()],
                "cached_target_only_decoder_ids": cached_ids,
                "full_prefix_vs_serial_cache": full_vs_serial,
                "full_prefix_vs_serial_policy": full_policy_vs_serial,
                "block_vs_serial_cache_by_block_size": by_block_size,
                "block_vs_serial_policy_by_block_size": processed_by_block_size,
            }
        )

    target_only_exact = all(item["native_vs_cached_target_only_exact"] for item in request_results)
    full_serial_exact = all(item["full_prefix_vs_serial_cache"]["top1_exact"] for item in request_results)
    block_exact = all(
        result["top1_exact"]
        for item in request_results
        for result in item["block_vs_serial_cache_by_block_size"].values()
    )
    block_fallback_safe = all(
        result["fallback_safe"]
        for item in request_results
        for result in item["block_vs_serial_policy_by_block_size"].values()
    )
    probe_exact = all(item["native_vs_probe_speculative_exact"] for item in request_results)
    oracle_exact = all(item["native_vs_oracle_speculative_exact"] for item in request_results)
    summary: Dict[str, Any] = {
        "status": "passed" if target_only_exact and block_fallback_safe and probe_exact and oracle_exact else "failed",
        "gate": "native cached greedy parity + block/serial margin safety + wrong-draft and oracle-draft speculative parity",
        "target_only_exact_all_examples": target_only_exact,
        "full_prefix_vs_serial_top1_exact_all_examples": full_serial_exact,
        "block_vs_serial_top1_exact_all_examples": block_exact,
        "block_vs_serial_margin_fallback_safe_all_examples": block_fallback_safe,
        "native_vs_probe_speculative_exact_all_examples": probe_exact,
        "probe_ambiguity_fallback_count": sum(item["probe_ambiguity_fallbacks"] for item in request_results),
        "native_vs_oracle_speculative_exact_all_examples": oracle_exact,
        "oracle_accepted_tokens": sum(item["oracle_accepted_tokens"] for item in request_results),
        "oracle_ambiguity_fallback_count": sum(item["oracle_ambiguity_fallbacks"] for item in request_results),
        "ambiguity_fallback_margin": ambiguity_margin,
        "examples": len(request_results),
        "prefix_tokens": prefix_tokens,
        "block_sizes": list(sizes),
        "input_file": str(Path(input_path).expanduser().resolve()),
        "target": adapter.target_manifest(),
        "requests": request_results,
        "decision": (
            "Proceed: cached greedy, wrong-draft correction, and oracle-draft speculative generation match native output; block disagreements are covered by serial fallback."
            if target_only_exact and block_fallback_safe and probe_exact and oracle_exact
            else "Stop: greedy parity failed, a block disagreement exceeds the fallback margin, or either speculative probe differs from native generation."
        ),
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        summary["output_file"] = str(destination)
    return summary
