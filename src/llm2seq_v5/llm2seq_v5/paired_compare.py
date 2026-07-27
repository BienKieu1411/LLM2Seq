"""Fail-closed paired comparison of two held-out prediction artifacts.

This module deliberately has no built-in T5Gemma score.  A T5Gemma claim is
possible only when an on-disk baseline prediction artifact, its metrics, and
its resolved configuration are mutually consistent and identify T5Gemma.
The scores computed here are the diagnostic ``rouge==1.0.0`` scores, not the
Perl ROUGE-1.5.5 paper scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from .metrics import rouge_per_example

_ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
_GENERATION_KEYS = (
    "max_new_tokens",
    "min_new_tokens",
    "num_beams",
    "do_sample",
    "repetition_penalty",
    "no_repeat_ngram_size",
)
_SOURCE_PROTOCOL_KEYS = ("source_prefix", "max_source_length", "max_target_length")
_FINGERPRINT_KEYS = (
    "test_subset_fingerprint",
    "evaluated_subset_fingerprint",
    "subset_fingerprint",
    "test_data_fingerprint",
)
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 1729
_METRIC_TOLERANCE = 1e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_rouge_version() -> Optional[str]:
    try:
        return version("rouge")
    except PackageNotFoundError:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _read_predictions(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        missing = [key for key in ("id", "prediction", "reference") if key not in raw]
        if missing:
            raise ValueError(f"Missing {missing} at {path}:{line_number}")
        if raw["id"] is None or not isinstance(raw["prediction"], str) or not isinstance(raw["reference"], str):
            raise ValueError(f"Invalid id/prediction/reference types at {path}:{line_number}")
        rows.append(
            {
                # Preserve the JSON type in its canonical representation.  In
                # particular, integer 1 and string "1" are not silently paired.
                "id": json.dumps(raw["id"], ensure_ascii=False, sort_keys=True),
                "prediction": raw["prediction"],
                "reference": raw["reference"],
            }
        )
    if not rows:
        raise ValueError(f"Prediction artifact is empty: {path}")
    return rows


def _canonical_reference(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_fingerprint(value: Any) -> Optional[Tuple[int, str]]:
    if not isinstance(value, Mapping):
        return None
    try:
        count = int(value["num_examples"])
        sha256 = str(value["sha256"]).lower()
    except (KeyError, TypeError, ValueError):
        return None
    if count <= 0 or len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        return None
    return count, sha256


def _metrics_fingerprint(metrics: Mapping[str, Any]) -> Tuple[Optional[Tuple[int, str]], List[str]]:
    found: List[Tuple[str, Optional[Tuple[int, str]]]] = [
        (key, _canonical_fingerprint(metrics[key])) for key in _FINGERPRINT_KEYS if key in metrics
    ]
    if not found:
        return None, []
    errors = [f"invalid {key}" for key, value in found if value is None]
    valid = [value for _, value in found if value is not None]
    if len(set(valid)) > 1:
        errors.append("conflicting fingerprint fields in one metrics artifact")
    return (valid[0] if valid else None), errors


def _locked_config_fingerprint(config: Mapping[str, Any]) -> Optional[Tuple[int, str]]:
    benchmark = config.get("benchmark", {})
    if not isinstance(benchmark, Mapping):
        return None
    data = benchmark.get("data", {})
    if not isinstance(data, Mapping):
        return None
    return _canonical_fingerprint(data.get("test"))


def _generation_from_config(config: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    raw = config.get("generation")
    if not isinstance(raw, Mapping):
        return None, ["resolved config has no generation mapping"]
    defaults = {"num_beams": 1, "do_sample": False}
    protocol: Dict[str, Any] = {}
    errors: List[str] = []
    for key in _GENERATION_KEYS:
        if key not in raw and key not in defaults:
            errors.append(f"resolved config is missing generation.{key}")
            continue
        value = raw.get(key, defaults.get(key))
        try:
            if key == "do_sample":
                if not isinstance(value, bool):
                    raise ValueError
                protocol[key] = value
            elif key in {"repetition_penalty"}:
                protocol[key] = float(value)
            else:
                protocol[key] = int(value)
        except (TypeError, ValueError):
            errors.append(f"invalid resolved generation.{key}={value!r}")
    return (protocol if not errors else None), errors


def _generation_from_metrics(metrics: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    raw = metrics.get("generation")
    if not isinstance(raw, Mapping):
        return None, ["metrics has no generation mapping"]
    protocol: Dict[str, Any] = {}
    errors: List[str] = []
    for key in _GENERATION_KEYS:
        if key not in raw:
            errors.append(f"metrics is missing generation.{key}")
            continue
        value = raw[key]
        try:
            if key == "do_sample":
                if not isinstance(value, bool):
                    raise ValueError
                protocol[key] = value
            elif key == "repetition_penalty":
                protocol[key] = float(value)
            else:
                protocol[key] = int(value)
        except (TypeError, ValueError):
            errors.append(f"invalid metrics generation.{key}={value!r}")
    return (protocol if not errors else None), errors


def _source_protocol_from_config(config: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    raw = config.get("data")
    if not isinstance(raw, Mapping):
        return None, ["resolved config has no data mapping"]
    errors = [f"resolved config is missing data.{key}" for key in _SOURCE_PROTOCOL_KEYS if key not in raw]
    if errors:
        return None, errors
    try:
        return {
            "source_prefix": str(raw["source_prefix"]),
            "max_source_length": int(raw["max_source_length"]),
            "max_target_length": int(raw["max_target_length"]),
        }, []
    except (TypeError, ValueError):
        return None, ["resolved config has an invalid source/length protocol"]


def _source_protocol_from_metrics(metrics: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors = [f"metrics is missing {key}" for key in _SOURCE_PROTOCOL_KEYS if key not in metrics]
    if errors:
        return None, errors
    try:
        return {
            "source_prefix": str(metrics["source_prefix"]),
            "max_source_length": int(metrics["max_source_length"]),
            "max_target_length": int(metrics["max_target_length"]),
        }, []
    except (TypeError, ValueError):
        return None, ["metrics has an invalid source/length protocol"]


def _protocol_equal(left: Optional[Mapping[str, Any]], right: Optional[Mapping[str, Any]]) -> bool:
    if left is None or right is None or set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, float) or isinstance(right_value, float):
            try:
                if not math.isclose(float(left_value), float(right_value), rel_tol=0.0, abs_tol=1e-12):
                    return False
            except (TypeError, ValueError):
                return False
        elif left_value != right_value:
            return False
    return True


def _backend_family(metrics: Mapping[str, Any]) -> Optional[str]:
    value = str(metrics.get("rouge_backend", "")).lower().replace(" ", "")
    return "rouge==1.0.0" if "rouge==1.0.0" in value else None


def _mean_scores(per_example: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {key: float(sum(float(row[key]) for row in per_example) / len(per_example)) for key in _ROUGE_KEYS}


def _metrics_match_recomputed(metrics: Mapping[str, Any], recomputed: Mapping[str, float]) -> bool:
    try:
        return all(abs(float(metrics[key]) - float(recomputed[key])) <= _METRIC_TOLERANCE for key in _ROUGE_KEYS)
    except (KeyError, TypeError, ValueError):
        return False


def _repeated_ngram_rate(text: str, n: int = 3) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _diagnostics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    prediction_lengths = [len(value.split()) for value in predictions]
    reference_lengths = [len(value.split()) for value in references]
    ratios = [prediction / max(1, reference) for prediction, reference in zip(prediction_lengths, reference_lengths)]
    normalized = [" ".join(value.lower().split()) for value in predictions if value.strip()]
    prefixes = Counter(" ".join(value.split()[:5]) for value in normalized)
    count = len(predictions)
    return {
        "num_examples": count,
        "prediction_words_mean": round(sum(prediction_lengths) / count, 6),
        "reference_words_mean": round(sum(reference_lengths) / count, 6),
        "length_ratio_mean": round(sum(ratios) / count, 6),
        "empty_prediction_rate": round(100.0 * sum(not value.strip() for value in predictions) / count, 6),
        "too_short_rate": round(100.0 * sum(value < 0.5 for value in ratios) / count, 6),
        "too_long_rate": round(100.0 * sum(value > 1.5 for value in ratios) / count, 6),
        "repeated_trigram_rate_mean": round(
            100.0 * sum(_repeated_ngram_rate(value) for value in predictions) / count,
            6,
        ),
        "unique_prediction_rate": round(100.0 * len(set(normalized)) / max(1, len(normalized)), 6),
        "dominant_prefix_5gram_rate": round(
            100.0 * max(prefixes.values(), default=0) / max(1, len(normalized)),
            6,
        ),
    }


def _paired_bootstrap(delta: np.ndarray) -> Dict[str, Dict[str, float]]:
    if delta.ndim != 2 or delta.shape[1] != len(_ROUGE_KEYS) or delta.shape[0] == 0:
        raise ValueError("Paired delta matrix must have shape [examples, 3]")
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = np.empty((_BOOTSTRAP_SAMPLES, delta.shape[1]), dtype=np.float64)
    offset = 0
    while offset < _BOOTSTRAP_SAMPLES:
        size = min(256, _BOOTSTRAP_SAMPLES - offset)
        indices = generator.integers(0, delta.shape[0], size=(size, delta.shape[0]))
        samples[offset : offset + size] = delta[indices].mean(axis=1)
        offset += size
    lower, upper = np.quantile(samples, [0.025, 0.975], axis=0)
    point = delta.mean(axis=0)
    return {
        key: {
            "mean": round(float(point[index]), 6),
            "ci95_low": round(float(lower[index]), 6),
            "ci95_high": round(float(upper[index]), 6),
        }
        for index, key in enumerate(_ROUGE_KEYS)
    }


def _identity_values(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    model = config.get("model", {})
    config_values: List[str] = []
    if isinstance(model, Mapping):
        for key in ("model_name_or_path", "base_model", "decoder_name"):
            if model.get(key):
                config_values.append(str(model[key]))
    metric_values = [
        str(metrics[key])
        for key in ("base_model", "checkpoint_base_model", "model_id", "decoder_name")
        if metrics.get(key)
    ]
    return config_values, metric_values


def _identifies_t5gemma(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> bool:
    config_values, metric_values = _identity_values(config, metrics)
    return bool(
        config_values
        and metric_values
        and any("t5gemma" in value.lower().replace("-", "") for value in config_values)
        and any("t5gemma" in value.lower().replace("-", "") for value in metric_values)
    )


def compare_paired_artifacts(
    candidate_predictions_path: str | Path,
    candidate_metrics_path: str | Path,
    candidate_config_path: str | Path,
    baseline_predictions_path: str | Path,
    baseline_metrics_path: str | Path,
    baseline_config_path: str | Path,
) -> Dict[str, Any]:
    """Compare actual prediction artifacts and return a fail-closed report."""

    paths = {
        "candidate_predictions": Path(candidate_predictions_path),
        "candidate_metrics": Path(candidate_metrics_path),
        "candidate_config": Path(candidate_config_path),
        "baseline_predictions": Path(baseline_predictions_path),
        "baseline_metrics": Path(baseline_metrics_path),
        "baseline_config": Path(baseline_config_path),
    }
    candidate_rows = _read_predictions(paths["candidate_predictions"])
    baseline_rows = _read_predictions(paths["baseline_predictions"])
    candidate_metrics = _read_json(paths["candidate_metrics"])
    baseline_metrics = _read_json(paths["baseline_metrics"])
    candidate_config = _read_config(paths["candidate_config"])
    baseline_config = _read_config(paths["baseline_config"])

    reasons: List[str] = []
    gates: Dict[str, bool] = {}

    def gate(name: str, passed: bool, reason: str) -> None:
        gates[name] = bool(passed)
        if not passed:
            reasons.append(reason)

    gate(
        "resolved_configs_are_flat",
        "_base_" not in candidate_config and "_base_" not in baseline_config,
        "Both configuration artifacts must be resolved; an _base_ inheritance key remains",
    )
    gate(
        "same_row_count",
        len(candidate_rows) == len(baseline_rows),
        f"Prediction row counts differ: candidate={len(candidate_rows)}, baseline={len(baseline_rows)}",
    )
    candidate_ids = [row["id"] for row in candidate_rows]
    baseline_ids = [row["id"] for row in baseline_rows]
    gate(
        "candidate_ids_are_unique",
        len(set(candidate_ids)) == len(candidate_ids),
        "Candidate prediction IDs are not unique",
    )
    gate(
        "baseline_ids_are_unique",
        len(set(baseline_ids)) == len(baseline_ids),
        "Baseline prediction IDs are not unique",
    )
    gate(
        "aligned_ids_and_order",
        candidate_ids == baseline_ids,
        "Candidate and baseline IDs/order are not exactly aligned",
    )
    candidate_references = [_canonical_reference(row["reference"]) for row in candidate_rows]
    baseline_references = [_canonical_reference(row["reference"]) for row in baseline_rows]
    gate(
        "aligned_references",
        candidate_references == baseline_references,
        "Candidate and baseline references differ after NFC normalization",
    )
    try:
        candidate_metric_count = int(candidate_metrics.get("num_examples", -1))
        baseline_metric_count = int(baseline_metrics.get("num_examples", -1))
    except (TypeError, ValueError):
        candidate_metric_count = baseline_metric_count = -1
    gate(
        "metrics_counts_match_artifacts",
        candidate_metric_count == len(candidate_rows) and baseline_metric_count == len(baseline_rows),
        "metrics.num_examples does not match its prediction artifact",
    )

    candidate_fingerprint, candidate_fingerprint_errors = _metrics_fingerprint(candidate_metrics)
    baseline_fingerprint, baseline_fingerprint_errors = _metrics_fingerprint(baseline_metrics)
    gate(
        "valid_fingerprint_metadata",
        not candidate_fingerprint_errors and not baseline_fingerprint_errors,
        "; ".join(candidate_fingerprint_errors + baseline_fingerprint_errors) or "Invalid fingerprint metadata",
    )
    if candidate_fingerprint is None and baseline_fingerprint is None:
        fingerprints_compatible = True
        fingerprint_status = "absent_in_both_metrics; exact aligned IDs/references still verified"
    elif candidate_fingerprint is None or baseline_fingerprint is None:
        fingerprints_compatible = False
        fingerprint_status = "fingerprint metadata is present for only one artifact"
    else:
        fingerprints_compatible = (
            candidate_fingerprint == baseline_fingerprint
            and candidate_fingerprint[0] == len(candidate_rows)
            and baseline_fingerprint[0] == len(baseline_rows)
        )
        fingerprint_status = "matched" if fingerprints_compatible else "mismatched"
    gate("same_dataset_or_subset_fingerprint", fingerprints_compatible, fingerprint_status)

    candidate_locked = _locked_config_fingerprint(candidate_config)
    baseline_locked = _locked_config_fingerprint(baseline_config)
    locked_applicable = [
        value for value in (candidate_locked, baseline_locked) if value is not None and value[0] == len(candidate_rows)
    ]
    gate(
        "compatible_applicable_locked_fingerprints",
        len(set(locked_applicable)) <= 1
        and all(candidate_fingerprint is None or value == candidate_fingerprint for value in locked_applicable),
        "A full-split config fingerprint conflicts with the evaluated artifact",
    )

    candidate_config_generation, errors = _generation_from_config(candidate_config)
    reasons.extend(f"candidate: {error}" for error in errors)
    baseline_config_generation, errors = _generation_from_config(baseline_config)
    reasons.extend(f"baseline: {error}" for error in errors)
    candidate_metrics_generation, errors = _generation_from_metrics(candidate_metrics)
    reasons.extend(f"candidate: {error}" for error in errors)
    baseline_metrics_generation, errors = _generation_from_metrics(baseline_metrics)
    reasons.extend(f"baseline: {error}" for error in errors)
    gate(
        "generation_metadata_complete",
        all(
            value is not None
            for value in (
                candidate_config_generation,
                baseline_config_generation,
                candidate_metrics_generation,
                baseline_metrics_generation,
            )
        ),
        "Generation protocol metadata is incomplete",
    )
    gate(
        "generation_matches_each_resolved_config",
        _protocol_equal(candidate_config_generation, candidate_metrics_generation)
        and _protocol_equal(baseline_config_generation, baseline_metrics_generation),
        "At least one metrics artifact does not match its resolved generation config",
    )
    gate(
        "same_generation_protocol",
        _protocol_equal(candidate_metrics_generation, baseline_metrics_generation),
        "Candidate and baseline generation protocols differ",
    )

    candidate_config_source, errors = _source_protocol_from_config(candidate_config)
    reasons.extend(f"candidate: {error}" for error in errors)
    baseline_config_source, errors = _source_protocol_from_config(baseline_config)
    reasons.extend(f"baseline: {error}" for error in errors)
    candidate_metrics_source, errors = _source_protocol_from_metrics(candidate_metrics)
    reasons.extend(f"candidate: {error}" for error in errors)
    baseline_metrics_source, errors = _source_protocol_from_metrics(baseline_metrics)
    reasons.extend(f"baseline: {error}" for error in errors)
    gate(
        "source_metadata_complete",
        all(
            value is not None
            for value in (
                candidate_config_source,
                baseline_config_source,
                candidate_metrics_source,
                baseline_metrics_source,
            )
        ),
        "Source/length protocol metadata is incomplete",
    )
    gate(
        "source_protocol_matches_each_resolved_config",
        _protocol_equal(candidate_config_source, candidate_metrics_source)
        and _protocol_equal(baseline_config_source, baseline_metrics_source),
        "At least one metrics artifact does not match its resolved source/length config",
    )
    gate(
        "same_source_protocol",
        _protocol_equal(candidate_metrics_source, baseline_metrics_source),
        "Candidate and baseline source/length protocols differ",
    )

    candidate_backend = _backend_family(candidate_metrics)
    baseline_backend = _backend_family(baseline_metrics)
    runtime_rouge_version = _runtime_rouge_version()
    gate(
        "runtime_rouge_is_exact_1_0_0",
        runtime_rouge_version == "1.0.0",
        f"Runtime rouge distribution must be exactly 1.0.0, found {runtime_rouge_version!r}",
    )
    gate(
        "same_supported_evaluation_backend",
        candidate_backend == baseline_backend == "rouge==1.0.0",
        "Both metrics artifacts must use the rouge==1.0.0 diagnostic backend",
    )
    candidate_preprocessing = str(candidate_metrics.get("rouge_preprocessing", ""))
    baseline_preprocessing = str(baseline_metrics.get("rouge_preprocessing", ""))
    gate(
        "same_declared_rouge_preprocessing",
        bool(candidate_preprocessing) and candidate_preprocessing == baseline_preprocessing,
        "ROUGE preprocessing metadata is missing or differs",
    )

    candidate_predictions = [row["prediction"] for row in candidate_rows]
    baseline_predictions = [row["prediction"] for row in baseline_rows]
    candidate_per_example = rouge_per_example(candidate_predictions, candidate_references)
    baseline_per_example = rouge_per_example(baseline_predictions, baseline_references)
    candidate_recomputed = _mean_scores(candidate_per_example)
    baseline_recomputed = _mean_scores(baseline_per_example)
    candidate_metrics_bound = _metrics_match_recomputed(candidate_metrics, candidate_recomputed)
    baseline_metrics_bound = _metrics_match_recomputed(baseline_metrics, baseline_recomputed)
    gate(
        "metrics_recomputed_from_prediction_artifacts",
        candidate_metrics_bound and baseline_metrics_bound,
        "Reported aggregate ROUGE does not match one or both prediction artifacts",
    )

    comparable = all(gates.values()) and not reasons
    bootstrap: Optional[Dict[str, Dict[str, float]]] = None
    paired_examples: Optional[List[Dict[str, Any]]] = None
    if comparable:
        candidate_array = np.asarray(
            [[row[key] for key in _ROUGE_KEYS] for row in candidate_per_example],
            dtype=np.float64,
        )
        baseline_array = np.asarray(
            [[row[key] for key in _ROUGE_KEYS] for row in baseline_per_example],
            dtype=np.float64,
        )
        bootstrap = _paired_bootstrap(candidate_array - baseline_array)
        paired_examples = []
        for row_index, (candidate_score, baseline_score) in enumerate(zip(candidate_per_example, baseline_per_example)):
            paired_examples.append(
                {
                    "id": json.loads(candidate_rows[row_index]["id"]),
                    "candidate": {key: round(float(candidate_score[key]), 6) for key in _ROUGE_KEYS},
                    "baseline": {key: round(float(baseline_score[key]), 6) for key in _ROUGE_KEYS},
                    "delta": {
                        key: round(float(candidate_score[key]) - float(baseline_score[key]), 6) for key in _ROUGE_KEYS
                    },
                }
            )

    candidate_diagnostics = _diagnostics(candidate_predictions, candidate_references)
    baseline_diagnostics = _diagnostics(baseline_predictions, baseline_references)
    diagnostic_delta = {
        key: round(float(candidate_diagnostics[key]) - float(baseline_diagnostics[key]), 6)
        for key in (
            "prediction_words_mean",
            "length_ratio_mean",
            "empty_prediction_rate",
            "too_short_rate",
            "too_long_rate",
            "repeated_trigram_rate_mean",
            "unique_prediction_rate",
            "dominant_prefix_5gram_rate",
        )
    }

    baseline_is_t5gemma = _identifies_t5gemma(baseline_config, baseline_metrics)
    actual_baseline_artifact_bound = bool(
        baseline_metrics_bound
        and gates["metrics_counts_match_artifacts"]
        and gates["baseline_ids_are_unique"]
        and paths["baseline_predictions"].is_file()
        and paths["baseline_metrics"].is_file()
        and paths["baseline_config"].is_file()
    )
    point_superiority = bool(
        comparable and bootstrap is not None and all(bootstrap[key]["mean"] > 0.0 for key in _ROUGE_KEYS)
    )
    supported_superiority = bool(
        comparable and bootstrap is not None and all(bootstrap[key]["ci95_low"] > 0.0 for key in _ROUGE_KEYS)
    )
    t5_eligible = bool(comparable and actual_baseline_artifact_bound and baseline_is_t5gemma)

    return {
        "comparison_scope": "paired held-out rouge==1.0.0 diagnostic; not Perl ROUGE-1.5.5 paper scoring",
        "comparable": comparable,
        "comparability_gates": gates,
        "comparability_reasons": reasons,
        "num_aligned_examples": (
            len(candidate_rows)
            if gates["same_row_count"] and gates["aligned_ids_and_order"] and gates["aligned_references"]
            else None
        ),
        "fingerprint_status": fingerprint_status,
        "artifact_sha256": {name: _sha256(path) for name, path in paths.items()},
        "protocol": {
            "generation": candidate_metrics_generation if comparable else None,
            "source": candidate_metrics_source if comparable else None,
            "rouge_backend": candidate_backend if comparable else None,
            "rouge_distribution_version": runtime_rouge_version,
            "rouge_preprocessing": candidate_preprocessing if comparable else None,
        },
        "candidate": {
            "recomputed_rouge": {key: round(value, 6) for key, value in candidate_recomputed.items()},
            "diagnostics": candidate_diagnostics,
        },
        "baseline": {
            "recomputed_rouge": {key: round(value, 6) for key, value in baseline_recomputed.items()},
            "diagnostics": baseline_diagnostics,
            "actual_prediction_artifact_bound_to_metrics": actual_baseline_artifact_bound,
            "identified_as_t5gemma_by_config_and_metrics": baseline_is_t5gemma,
        },
        "candidate_minus_baseline_diagnostics": diagnostic_delta,
        "paired_per_example_rouge": paired_examples,
        "paired_bootstrap": {
            "samples": _BOOTSTRAP_SAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "confidence": 0.95,
            "delta_definition": "candidate_minus_baseline",
            "rouge": bootstrap,
        },
        "candidate_strictly_higher_all_rouge_point_estimates": point_superiority,
        "candidate_superior_all_rouge_with_95pct_paired_support": supported_superiority,
        "t5gemma_claim": {
            "eligible": t5_eligible,
            "point_estimate_beaten": bool(t5_eligible and point_superiority),
            # The unqualified field intentionally uses the stronger claim.
            "beaten": bool(t5_eligible and supported_superiority),
            "reason_if_ineligible": (
                None
                if t5_eligible
                else "Requires comparable on-disk predictions, bound metrics/config, and T5Gemma identity in both config and metrics"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed paired comparison of actual prediction artifacts")
    parser.add_argument("--candidate-predictions", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--baseline-predictions", required=True, type=Path)
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_paired_artifacts(
            args.candidate_predictions,
            args.candidate_metrics,
            args.candidate_config,
            args.baseline_predictions,
            args.baseline_metrics,
            args.baseline_config,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        result = {
            "comparable": False,
            "comparability_reasons": [f"{type(exc).__name__}: {exc}"],
            "t5gemma_claim": {"eligible": False, "point_estimate_beaten": False, "beaten": False},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("comparable", False) else 2)


if __name__ == "__main__":
    main()
