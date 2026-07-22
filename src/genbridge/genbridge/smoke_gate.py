"""Fail fast when a 100-example overfit run is still degenerate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def smoke_failures(metrics: Dict[str, Any]) -> List[str]:
    """Return plumbing failures, not paper-quality significance tests."""

    failures = []
    checks = (
        (float(metrics.get("rouge1", 0.0)) >= 20.0, "ROUGE-1 < 20"),
        (float(metrics.get("rouge2", 0.0)) >= 5.0, "ROUGE-2 < 5"),
        (
            float(metrics.get("empty_prediction_rate", 100.0)) <= 1.0,
            "empty predictions > 1%",
        ),
        (
            float(metrics.get("unique_prediction_rate", 0.0)) >= 80.0,
            "unique predictions < 80%",
        ),
        (
            float(metrics.get("dominant_prefix_5gram_rate", 100.0)) <= 25.0,
            "one 5-word prefix dominates > 25%",
        ),
        (
            float(metrics.get("repeated_trigram_rate_mean", 100.0)) <= 10.0,
            "mean repeated-trigram rate > 10%",
        ),
        (
            0.3 <= float(metrics.get("length_ratio_mean", 0.0)) <= 2.0,
            "mean output/reference length ratio outside [0.3, 2.0]",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    if "cross_residual_ratio" in metrics and float(metrics["cross_residual_ratio"]) <= 1e-3:
        failures.append("decoder is effectively ignoring encoder memory")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    path = Path(args.metrics)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    failures = smoke_failures(metrics)
    print(
        json.dumps(
            {
                "status": "FAIL" if failures else "PASS",
                "metrics": str(path),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
