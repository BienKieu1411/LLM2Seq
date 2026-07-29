"""Fail-closed artifact comparison against the full T5Gemma baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Dict

from .config import load_config

_ROUGE_NAMES = ("rouge1", "rouge2", "rougeL")
_ROUGE_PROTOCOL = "-c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a -m"
_GENERATION_FIELDS = (
    "max_new_tokens",
    "min_new_tokens",
    "num_beams",
    "do_sample",
    "repetition_penalty",
    "no_repeat_ngram_size",
)


def _load_json(path: str | Path, label: str) -> tuple[Path, Dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return resolved, value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scores(artifact: Dict[str, Any], label: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name in _ROUGE_NAMES:
        try:
            score = float(artifact[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} has no numeric {name}") from exc
        _require(math.isfinite(score), f"{label}.{name} must be finite")
        _require(0.0 <= score <= 100.0, f"{label}.{name} must be in [0, 100]")
        scores[name] = score
    return scores


def _validate_rouge_protocol(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    for label, artifact in (("candidate ROUGE", candidate), ("baseline ROUGE", baseline)):
        _finite_scores(artifact, label)
        _require(
            "Perl ROUGE-1.5.5" in str(artifact.get("backend", "")),
            f"{label} is not official Perl ROUGE-1.5.5",
        )
        _require(artifact.get("stemming") is True, f"{label} must use stemming")
        _require(
            artifact.get("pyrouge_default_args") == _ROUGE_PROTOCOL,
            f"{label} uses a different pyrouge protocol",
        )
        _require(
            artifact.get("prediction_field") == "prediction" and artifact.get("reference_field") == "reference",
            f"{label} uses different prediction/reference fields",
        )
    _require(
        candidate.get("backend") == baseline.get("backend"),
        "Candidate and baseline must be scored by the exact same pyrouge installation",
    )


def _prediction_path(
    rouge_path: Path,
    rouge: Dict[str, Any],
    override: str | Path | None,
    label: str,
) -> Path:
    raw = override if override is not None else rouge.get("predictions_file")
    _require(bool(raw), f"{label} ROUGE artifact does not identify its predictions file")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = rouge_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label} predictions: {path}; pass --{label}-predictions after copying artifacts"
        )
    return path


def _normalize_reference(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_reference(item) for item in value]
    return unicodedata.normalize("NFC", str(value))


def _rows_contract(path: Path, label: str) -> Dict[str, Any]:
    rows: list[tuple[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid {label} JSONL at line {line_number}") from exc
            _require("id" in row and row["id"] is not None, f"{label} row {line_number} has no ID")
            _require("reference" in row, f"{label} row {line_number} has no reference")
            row_id = json.dumps(row["id"], ensure_ascii=False, sort_keys=True)
            _require(row_id not in seen, f"{label} contains duplicate ID {row['id']!r}")
            seen.add(row_id)
            rows.append((row_id, _normalize_reference(row["reference"])))
    _require(bool(rows), f"{label} predictions are empty")
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "num_examples": len(rows),
        "id_reference_sha256": hashlib.sha256(encoded).hexdigest(),
        "rows": rows,
    }


def _fingerprint(metrics: Dict[str, Any], label: str) -> Dict[str, Any]:
    value = metrics.get("test_data_fingerprint")
    _require(isinstance(value, dict), f"{label} metrics have no test_data_fingerprint")
    try:
        count = int(value["num_examples"])
        sha256 = str(value["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} test fingerprint is malformed") from exc
    _require(count > 0 and len(sha256) == 64, f"{label} test fingerprint is malformed")
    return {"num_examples": count, "sha256": sha256}


def _validate_generation(config: Dict[str, Any], candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    candidate_generation = candidate.get("generation")
    baseline_generation = baseline.get("generation")
    _require(isinstance(candidate_generation, dict), "Candidate metrics have no generation contract")
    _require(isinstance(baseline_generation, dict), "Baseline metrics have no generation contract")
    for name in _GENERATION_FIELDS:
        _require(name in candidate_generation, f"Candidate generation contract is missing {name}")
        _require(name in baseline_generation, f"Baseline generation contract is missing {name}")
        _require(
            candidate_generation[name] == baseline_generation[name],
            f"Generation mismatch for {name}: candidate={candidate_generation[name]!r}, "
            f"baseline={baseline_generation[name]!r}",
        )
    expected = {
        "max_new_tokens": int(config.get("generation", {}).get("max_new_tokens", 256)),
        "min_new_tokens": int(config.get("generation", {}).get("min_new_tokens", 16)),
        "num_beams": 1,
        "do_sample": False,
        "repetition_penalty": float(config.get("generation", {}).get("repetition_penalty", 1.05)),
        "no_repeat_ngram_size": int(config.get("generation", {}).get("no_repeat_ngram_size", 3)),
    }
    for name, value in expected.items():
        _require(
            candidate_generation[name] == value,
            f"Candidate generation {name} does not match the frozen EviSeq config",
        )
    for name in ("source_prefix", "max_source_length", "max_target_length"):
        _require(name in candidate and name in baseline, f"Missing {name} from evaluation metrics")
        _require(candidate[name] == baseline[name], f"Candidate/baseline {name} mismatch")
    _require(candidate["source_prefix"] == str(config["data"].get("source_prefix", "")), "Source prefix drift")
    _require(candidate["max_source_length"] == int(config["data"]["max_source_length"]), "Source length drift")
    _require(candidate["max_target_length"] == int(config["data"]["max_target_length"]), "Target length drift")
    return {name: candidate_generation[name] for name in _GENERATION_FIELDS}


def _baseline_identity(config: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    expected = str(config.get("benchmark", {}).get("paper", {}).get("model_id", "")).strip()
    _require(bool(expected), "benchmark.paper.model_id must lock the baseline identity")
    expected_tail = expected.rstrip("/").split("/")[-1].lower()
    identities = {
        name: str(metrics.get(name) or "").strip()
        for name in ("base_model", "checkpoint_base_model")
        if str(metrics.get(name) or "").strip()
    }
    _require(bool(identities), "Baseline metrics do not identify the base model")
    for name, actual in identities.items():
        _require(
            expected.lower() in actual.lower() or expected_tail in actual.lower(),
            f"Wrong baseline {name}: expected {expected!r}, got {actual!r}",
        )
    return identities.get("checkpoint_base_model") or identities["base_model"]


def compare(
    config_path: str,
    candidate_rouge_path: str,
    candidate_metrics_path: str,
    baseline_rouge_path: str,
    baseline_metrics_path: str,
    output_path: str | None = None,
    *,
    candidate_predictions_path: str | None = None,
    baseline_predictions_path: str | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    candidate_rouge_file, candidate_rouge = _load_json(candidate_rouge_path, "candidate ROUGE")
    baseline_rouge_file, baseline_rouge = _load_json(baseline_rouge_path, "baseline ROUGE")
    candidate_metrics_file, candidate_metrics = _load_json(candidate_metrics_path, "candidate metrics")
    baseline_metrics_file, baseline_metrics = _load_json(baseline_metrics_path, "baseline metrics")
    _validate_rouge_protocol(candidate_rouge, baseline_rouge)

    _require(candidate_metrics.get("evaluation_split") == "test", "Candidate is not a test evaluation")
    _require(candidate_metrics.get("paper_test") is True, "Candidate was not produced by the paper-test gate")
    _require(baseline_metrics.get("evaluation_split") == "test", "Baseline is not a test evaluation")
    _require(
        candidate_metrics.get("checkpoint_test_matches_current") is True,
        "Candidate checkpoint/test fingerprint is not verified",
    )
    _require(
        baseline_metrics.get("checkpoint_test_matches_current") is True,
        "Baseline checkpoint/test fingerprint is not verified",
    )
    _require(
        candidate_metrics.get("architecture_sha256") == config["_meta"]["architecture_sha256"],
        "Candidate architecture hash does not match the frozen EviSeq config",
    )
    _require(
        candidate_metrics.get("inference_protocol_sha256") == config["_meta"]["inference_protocol_sha256"],
        "Candidate prompt/generation protocol does not match the frozen EviSeq config",
    )
    _require(
        candidate_metrics.get("evaluation_contract_sha256") == config["_meta"]["evaluation_contract_sha256"],
        "Candidate evaluation contract does not match the frozen EviSeq config",
    )
    parameter_manifest = candidate_metrics.get("parameter_manifest")
    _require(isinstance(parameter_manifest, dict), "Candidate metrics have no parameter manifest")
    _require(
        parameter_manifest.get("architecture_sha256") == config["_meta"]["architecture_sha256"],
        "Candidate parameter manifest is bound to another architecture",
    )

    candidate_fingerprint = _fingerprint(candidate_metrics, "Candidate")
    baseline_fingerprint = _fingerprint(baseline_metrics, "Baseline")
    _require(candidate_fingerprint == baseline_fingerprint, "Candidate and baseline use different test rows")
    paper_target = config.get("benchmark", {}).get("paper", {})
    locked_test_sha = paper_target.get("test_sha256")
    if locked_test_sha is not None:
        _require(
            candidate_fingerprint["sha256"] == str(locked_test_sha),
            "Artifacts do not use the locked official paper test split",
        )

    candidate_predictions = _prediction_path(
        candidate_rouge_file, candidate_rouge, candidate_predictions_path, "candidate"
    )
    baseline_predictions = _prediction_path(baseline_rouge_file, baseline_rouge, baseline_predictions_path, "baseline")
    candidate_hash = _file_sha256(candidate_predictions)
    baseline_hash = _file_sha256(baseline_predictions)
    for label, actual_hash, rouge, metrics in (
        ("Candidate", candidate_hash, candidate_rouge, candidate_metrics),
        ("Baseline", baseline_hash, baseline_rouge, baseline_metrics),
    ):
        _require(actual_hash == rouge.get("predictions_sha256"), f"{label} ROUGE is not bound to its predictions")
        _require(
            actual_hash == metrics.get("predictions_sha256"), f"{label} metrics are not bound to their predictions"
        )

    candidate_rows = _rows_contract(candidate_predictions, "candidate")
    baseline_rows = _rows_contract(baseline_predictions, "baseline")
    _require(
        candidate_rows["rows"] == baseline_rows["rows"],
        "Prediction files do not contain the same ordered IDs and references",
    )
    count = int(candidate_rows["num_examples"])
    for label, artifact in (
        ("candidate ROUGE", candidate_rouge),
        ("baseline ROUGE", baseline_rouge),
        ("candidate metrics", candidate_metrics),
        ("baseline metrics", baseline_metrics),
    ):
        _require(int(artifact.get("num_examples", -1)) == count, f"{label} example count mismatch")
    _require(candidate_fingerprint["num_examples"] == count, "Test fingerprint count mismatch")
    locked_count = config.get("benchmark", {}).get("paper", {}).get("num_examples")
    if locked_count is not None:
        _require(count == int(locked_count), "Artifact count does not match the locked paper test count")

    generation_contract = _validate_generation(config, candidate_metrics, baseline_metrics)
    baseline_identity = _baseline_identity(config, baseline_metrics)
    candidate_scores = _finite_scores(candidate_rouge, "candidate ROUGE")
    baseline_scores = _finite_scores(baseline_rouge, "baseline ROUGE")
    tolerance = float(paper_target.get("score_tolerance", 0.02))
    if all(name in paper_target for name in _ROUGE_NAMES):
        for name in _ROUGE_NAMES:
            _require(
                abs(baseline_scores[name] - float(paper_target[name])) <= tolerance,
                f"Baseline {name} does not reproduce the locked full-T5Gemma artifact",
            )

    _require(candidate_metrics.get("checkpoint_parameters_match_model") is True, "Candidate parameter check failed")
    _require(baseline_metrics.get("checkpoint_parameters_match_model") is True, "Baseline parameter check failed")
    candidate_parameters = int(candidate_metrics.get("training_parameters", 0))
    candidate_deployable = int(candidate_metrics.get("deployable_parameters", 0))
    baseline_parameters = int(baseline_metrics.get("total_parameters", 0))
    _require(candidate_parameters > 0, "Candidate metrics have no exact unique parameter count")
    _require(candidate_deployable > 0, "Candidate metrics have no deployable parameter count")
    _require(baseline_parameters > 0, "Baseline metrics have no exact unique parameter count")
    _require(
        candidate_parameters == int(parameter_manifest.get("resident_training_total_unique", -1)),
        "Candidate parameter count disagrees with its manifest",
    )
    _require(
        candidate_deployable == int(parameter_manifest.get("deployable_resident_without_train_aux", -1)),
        "Candidate deployable count disagrees with its manifest",
    )
    reporting = config.get("reporting", {})
    target_footprint = int(reporting.get("target_total_footprint_approx", 0))
    eligible = bool(reporting.get("parameter_efficiency_claim_eligible", False))
    if eligible:
        _require(candidate_parameters < target_footprint, "Candidate exceeds its configured parameter budget")
        _require(
            candidate_parameters < baseline_parameters, "Candidate is not smaller than the actual T5Gemma baseline"
        )

    gaps = {name: round(candidate_scores[name] - baseline_scores[name], 4) for name in _ROUGE_NAMES}
    result: Dict[str, Any] = {
        "comparison_valid": True,
        "candidate": candidate_scores,
        "baseline_name": paper_target.get("name", baseline_identity),
        "baseline_model": baseline_identity,
        "baseline": baseline_scores,
        "candidate_minus_baseline": gaps,
        "strictly_exceeds_all": all(value > 0 for value in gaps.values()),
        "matches_or_exceeds_all": all(value >= 0 for value in gaps.values()),
        "rouge2_target_reached": gaps["rouge2"] >= 0,
        "candidate_parameters_resident": candidate_parameters,
        "candidate_parameters_deployable": candidate_deployable,
        "baseline_parameters": baseline_parameters,
        "candidate_is_smaller": candidate_parameters < baseline_parameters,
        "paper_target_pass": (
            all(value > 0 for value in gaps.values()) and (not eligible or candidate_parameters < baseline_parameters)
        ),
        "num_examples": count,
        "test_data_fingerprint": candidate_fingerprint,
        "id_reference_sha256": candidate_rows["id_reference_sha256"],
        "generation": generation_contract,
        "rouge_backend": candidate_rouge["backend"],
        "rouge_protocol": _ROUGE_PROTOCOL,
        "architecture_sha256": config["_meta"]["architecture_sha256"],
        "inference_protocol_sha256": config["_meta"]["inference_protocol_sha256"],
        "evaluation_contract_sha256": config["_meta"]["evaluation_contract_sha256"],
        "artifacts": {
            "candidate_predictions": str(candidate_predictions),
            "candidate_metrics": str(candidate_metrics_file),
            "candidate_rouge": str(candidate_rouge_file),
            "baseline_predictions": str(baseline_predictions),
            "baseline_metrics": str(baseline_metrics_file),
            "baseline_rouge": str(baseline_rouge_file),
        },
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if output_path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate-rouge", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--baseline-rouge", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-predictions")
    parser.add_argument("--baseline-predictions")
    parser.add_argument("--output")
    args = parser.parse_args()
    compare(
        args.config,
        args.candidate_rouge,
        args.candidate_metrics,
        args.baseline_rouge,
        args.baseline_metrics,
        args.output,
        candidate_predictions_path=args.candidate_predictions,
        baseline_predictions_path=args.baseline_predictions,
    )


if __name__ == "__main__":
    main()
