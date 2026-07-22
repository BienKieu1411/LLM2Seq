"""Paired statistical comparison for GenBridge and T5Gemma predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .metrics import heter_sum_graph_rouge_per_example

METRICS = ("rouge1", "rouge2", "rougeL")


def _read_predictions(path: str | Path) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            missing = [key for key in ("id", "prediction", "reference") if key not in row]
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {missing}")
            example_id = str(row["id"])
            if row["id"] is None or not example_id:
                raise ValueError(f"{path}:{line_number} has no stable example id")
            if example_id in rows:
                raise ValueError(f"Duplicate id {example_id!r} in {path}")
            rows[example_id] = row
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _aligned_rows(
    candidate_path: str | Path,
    baseline_path: str | Path,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    candidate = _read_predictions(candidate_path)
    baseline = _read_predictions(baseline_path)
    if set(candidate) != set(baseline):
        only_candidate = sorted(set(candidate) - set(baseline))
        only_baseline = sorted(set(baseline) - set(candidate))
        raise ValueError(
            "Prediction files contain different test IDs; "
            f"candidate-only={only_candidate[:10]}, baseline-only={only_baseline[:10]}"
        )
    ids = list(candidate)
    candidate_predictions: List[str] = []
    baseline_predictions: List[str] = []
    references: List[str] = []
    for example_id in ids:
        candidate_reference = str(candidate[example_id]["reference"])
        baseline_reference = str(baseline[example_id]["reference"])
        if candidate_reference != baseline_reference:
            raise ValueError(
                f"Reference mismatch for id {example_id!r}; refusing an unpaired comparison"
            )
        candidate_predictions.append(str(candidate[example_id]["prediction"]))
        baseline_predictions.append(str(baseline[example_id]["prediction"]))
        references.append(candidate_reference)
    return ids, candidate_predictions, baseline_predictions, references


def _reference_fingerprint(ids: Iterable[str], references: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for example_id, reference in zip(ids, references):
        digest.update(
            json.dumps(
                [example_id, reference],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _bootstrap_means(
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    example_count = candidate.shape[0]
    candidate_boot = np.empty((samples, candidate.shape[1]), dtype=np.float64)
    baseline_boot = np.empty_like(candidate_boot)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, example_count, size=(stop - start, example_count))
        candidate_boot[start:stop] = candidate[indices].mean(axis=1)
        baseline_boot[start:stop] = baseline[indices].mean(axis=1)
    return candidate_boot, baseline_boot, candidate_boot - baseline_boot


def _paired_randomization_pvalues(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 256,
) -> np.ndarray:
    if samples <= 0:
        raise ValueError("randomization_samples must be positive")
    observed = np.abs(differences.mean(axis=0))
    extreme = np.zeros(differences.shape[1], dtype=np.int64)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        signs = rng.integers(
            0,
            2,
            size=(stop - start, differences.shape[0], 1),
            dtype=np.int8,
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        randomized = (signs * differences[None, :, :]).mean(axis=1)
        extreme += (np.abs(randomized) >= observed[None, :] - 1e-12).sum(axis=0)
    return (extreme + 1.0) / (samples + 1.0)


def compare_predictions(
    candidate_path: str | Path,
    baseline_path: str | Path,
    *,
    candidate_name: str = "GenBridge",
    baseline_name: str = "T5Gemma",
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    ids, candidate_predictions, baseline_predictions, references = _aligned_rows(
        candidate_path,
        baseline_path,
    )
    candidate_scores = heter_sum_graph_rouge_per_example(candidate_predictions, references)
    baseline_scores = heter_sum_graph_rouge_per_example(baseline_predictions, references)
    candidate = np.asarray(
        [[score[metric] for metric in METRICS] for score in candidate_scores],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [[score[metric] for metric in METRICS] for score in baseline_scores],
        dtype=np.float64,
    )
    bootstrap_rng = np.random.default_rng(seed)
    candidate_boot, baseline_boot, delta_boot = _bootstrap_means(
        candidate,
        baseline,
        bootstrap_samples,
        bootstrap_rng,
    )
    randomization_rng = np.random.default_rng(seed + 1)
    pvalues = _paired_randomization_pvalues(
        candidate - baseline,
        randomization_samples,
        randomization_rng,
    )
    result: Dict[str, Any] = {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "candidate_file": str(Path(candidate_path).resolve()),
        "baseline_file": str(Path(baseline_path).resolve()),
        "num_examples": len(ids),
        "reference_sha256": _reference_fingerprint(ids, references),
        "rouge_backend": "rouge==1.0.0 (HeterSumGraph)",
        "bootstrap_samples": int(bootstrap_samples),
        "randomization_samples": int(randomization_samples),
        "seed": int(seed),
        "metrics": {},
    }
    for index, metric in enumerate(METRICS):
        candidate_interval = np.percentile(candidate_boot[:, index], [2.5, 97.5])
        baseline_interval = np.percentile(baseline_boot[:, index], [2.5, 97.5])
        delta_interval = np.percentile(delta_boot[:, index], [2.5, 97.5])
        result["metrics"][metric] = {
            "candidate": round(float(candidate[:, index].mean()), 4),
            "candidate_ci95": [round(float(value), 4) for value in candidate_interval],
            "baseline": round(float(baseline[:, index].mean()), 4),
            "baseline_ci95": [round(float(value), 4) for value in baseline_interval],
            "delta": round(float((candidate[:, index] - baseline[:, index]).mean()), 4),
            "delta_ci95": [round(float(value), 4) for value in delta_interval],
            "bootstrap_probability_candidate_better": round(
                float((delta_boot[:, index] > 0).mean()),
                6,
            ),
            "paired_randomization_p": round(float(pvalues[index]), 6),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-name", default="GenBridge")
    parser.add_argument("--baseline-name", default="T5Gemma")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--randomization-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = compare_predictions(
        args.candidate,
        args.baseline,
        candidate_name=args.candidate_name,
        baseline_name=args.baseline_name,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
