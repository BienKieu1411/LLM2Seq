"""Fail-closed WikiLingua DEV ablation table from Perl ROUGE artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from .data import dataset_fingerprint
from .paper_compare import (
    _file_sha256,
    _finite_scores,
    _load_json,
    _prediction_path,
    _rows_contract,
    _validate_rouge_protocol,
)

_SPECS: Dict[str, tuple[str, bool, str]] = {
    "c0": ("causal", True, "Causal encoder control"),
    "c2": ("dec2enc", True, "Generic dual-mask conversion"),
    "c3-no-cl": ("evidence", False, "Evidence conversion w/o hard evidence InfoNCE"),
    "c3": ("evidence", True, "EviSeq"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fingerprint(value: Any, label: str) -> Dict[str, Any]:
    _require(isinstance(value, dict), f"{label} has no validation fingerprint")
    try:
        result = {
            "num_examples": int(value["num_examples"]),
            "sha256": str(value["sha256"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} validation fingerprint is malformed") from exc
    _require(
        result["num_examples"] > 0 and len(result["sha256"]) == 64,
        f"{label} validation fingerprint is malformed",
    )
    return result


def _matched_config_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    """Everything that must match the main run, excluding tested factors."""

    objectives = dict(config["objectives"])
    objectives.pop("use_evidence_contrastive", None)
    objectives.pop("evidence_contrastive_weight", None)
    attention = dict(config["native_attention"])
    attention.pop("variant", None)
    return {
        "model": config["model"],
        "native_attention": attention,
        "bridge": config["bridge"],
        "decoder": config["decoder"],
        "objectives": objectives,
        "training": config["training"],
        "data": config["data"],
        "generation": config["generation"],
        "checkpoint": config["checkpoint"],
        "benchmark": config["benchmark"],
        "reporting": config["reporting"],
        "limits": config["limits"],
    }


def _validate_config(name: str, config: Dict[str, Any], main: Dict[str, Any]) -> None:
    expected_variant, expected_contrastive, _ = _SPECS[name]
    _require(
        config["native_attention"]["variant"] == expected_variant,
        f"{name} has the wrong attention variant",
    )
    _require(
        bool(config["objectives"]["use_evidence_contrastive"]) is expected_contrastive,
        f"{name} has the wrong contrastive switch",
    )
    expected_weight = float(main["objectives"]["evidence_contrastive_weight"])
    actual_weight = float(config["objectives"]["evidence_contrastive_weight"])
    _require(
        actual_weight == (expected_weight if expected_contrastive else 0.0),
        f"{name} has the wrong contrastive weight",
    )
    _require(
        _matched_config_contract(config) == _matched_config_contract(main),
        f"{name} changes fields outside the declared ablation factor",
    )
    _require(
        int(config.get("limits", {}).get("max_validation_examples", 0)) == 0,
        f"{name} does not evaluate the complete validation split",
    )


def _entry(
    name: str,
    config_path: str | Path,
    rouge_path: str | Path,
    metrics_path: str | Path,
    main_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = load_config(config_path)
    _validate_config(name, config, main_config)
    rouge_file, rouge = _load_json(rouge_path, f"{name} ROUGE")
    metrics_file, metrics = _load_json(metrics_path, f"{name} metrics")

    _require(metrics.get("evaluation_split") == "validation", f"{name} is not DEV evaluation")
    _require(metrics.get("paper_test") is False, f"{name} incorrectly uses the paper-test gate")
    _require(
        metrics.get("checkpoint_data_matches_current") is True,
        f"{name} does not cover the complete checkpoint validation split",
    )
    for field in (
        "architecture_sha256",
        "inference_protocol_sha256",
        "evaluation_contract_sha256",
    ):
        _require(
            metrics.get(field) == config["_meta"][field],
            f"{name} metrics are not bound to the resolved {field}",
        )
    _require(
        metrics.get("native_attention_variant") == config["native_attention"]["variant"],
        f"{name} metrics report the wrong attention variant",
    )

    current = _fingerprint(metrics.get("evaluation_data_fingerprint"), name)
    checkpoint = _fingerprint(metrics.get("checkpoint_data_fingerprint"), name)
    _require(current == checkpoint, f"{name} validation data differs from its checkpoint")
    expected = dataset_fingerprint(config["data"]["validation_file"], 0)
    expected = {"num_examples": int(expected["num_examples"]), "sha256": str(expected["sha256"])}
    _require(current == expected, f"{name} does not use its complete configured validation file")

    predictions = _prediction_path(rouge_file, rouge, None, name)
    predictions_hash = _file_sha256(predictions)
    _require(
        predictions_hash == rouge.get("predictions_sha256"),
        f"{name} Perl ROUGE is not bound to its predictions",
    )
    _require(
        predictions_hash == metrics.get("predictions_sha256"),
        f"{name} metrics are not bound to its predictions",
    )
    rows = _rows_contract(predictions, name)
    count = int(rows["num_examples"])
    _require(count == current["num_examples"], f"{name} prediction count mismatch")
    _require(int(rouge.get("num_examples", -1)) == count, f"{name} ROUGE count mismatch")
    _require(int(metrics.get("num_examples", -1)) == count, f"{name} metrics count mismatch")
    parameters = int(metrics.get("training_parameters", 0))
    deployable = int(metrics.get("deployable_parameters", 0))
    _require(parameters > 0 and deployable > 0, f"{name} has no verified parameter count")
    return {
        "name": name,
        "display_name": _SPECS[name][2],
        "attention_variant": config["native_attention"]["variant"],
        "evidence_conditioned_encoder_view": config["native_attention"]["variant"] == "evidence",
        "contrastive": bool(config["objectives"]["use_evidence_contrastive"]),
        "scores": _finite_scores(rouge, f"{name} ROUGE"),
        "resident_parameters": parameters,
        "deployable_parameters": deployable,
        "training_only_parameters": parameters - deployable,
        "validation_fingerprint": current,
        "id_reference_sha256": rows["id_reference_sha256"],
        "ordered_rows": rows["rows"],
        "generation": metrics.get("generation"),
        "source_prefix": metrics.get("source_prefix"),
        "max_source_length": metrics.get("max_source_length"),
        "max_target_length": metrics.get("max_target_length"),
        "config": str(Path(config_path).expanduser().resolve()),
        "metrics": str(metrics_file),
        "rouge": str(rouge_file),
        "predictions": str(predictions),
        "predictions_sha256": predictions_hash,
    }


def _markdown(rows: list[Dict[str, Any]]) -> str:
    main_r2 = rows[-1]["scores"]["rouge2"]
    lines = [
        "# WikiLingua DEV ablation — Perl ROUGE-1.5.5",
        "",
        "Primary selection metric: ROUGE-2 F1. This table is validation-only.",
        "",
        "| Method | Encoder view | Evidence-conditioned encoder view | InfoNCE | Deployable params | Train-only params | R-1 | R-2 | R-L | ΔR-2 vs c3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        score = row["scores"]
        lines.append(
            "| {name} | {variant} | {evidence} | {contrastive} | {deployable:,} | {training_only:,} | "
            "{r1:.3f} | {r2:.3f} | {rl:.3f} | {delta:+.3f} |".format(
                name=row["display_name"],
                variant=row["attention_variant"],
                evidence="yes" if row["evidence_conditioned_encoder_view"] else "no",
                contrastive="yes" if row["contrastive"] else "no",
                deployable=row["deployable_parameters"],
                training_only=row["training_only_parameters"],
                r1=score["rouge1"],
                r2=score["rouge2"],
                rl=score["rougeL"],
                delta=score["rouge2"] - main_r2,
            )
        )
    return "\n".join(lines) + "\n"


def build_table(
    artifacts: Dict[str, Dict[str, str | Path]],
    output_path: str | Path,
) -> Dict[str, Any]:
    _require(set(artifacts) == set(_SPECS), "DEV table requires exactly c0, c2, c3-no-cl and c3")
    main_config = load_config(artifacts["c3"]["config"])
    rows = [
        _entry(
            name,
            artifacts[name]["config"],
            artifacts[name]["rouge"],
            artifacts[name]["metrics"],
            main_config,
        )
        for name in _SPECS
    ]
    first = rows[0]
    for row in rows[1:]:
        _require(
            row["validation_fingerprint"] == first["validation_fingerprint"],
            "Ablations use different validation rows",
        )
        _require(
            row["ordered_rows"] == first["ordered_rows"],
            "Ablations do not contain identical ordered IDs/references",
        )
        for field in ("generation", "source_prefix", "max_source_length", "max_target_length"):
            _require(row[field] == first[field], f"Ablations differ in {field}")

    rouge_artifacts = [_load_json(artifacts[name]["rouge"], f"{name} ROUGE")[1] for name in _SPECS]
    for rouge in rouge_artifacts[1:]:
        _validate_rouge_protocol(rouge_artifacts[0], rouge)

    public_rows = []
    for row in rows:
        public = dict(row)
        public.pop("ordered_rows")
        public_rows.append(public)
    main_r2 = public_rows[-1]["scores"]["rouge2"]
    for row in public_rows:
        row["delta_rouge2_vs_c3"] = round(row["scores"]["rouge2"] - main_r2, 4)
    result = {
        "table_role": "DEV_ONLY_MODEL_SELECTION",
        "dataset": "WikiLingua-Vietnamese",
        "primary_metric": "rouge2",
        "num_core_ablations": 3,
        "num_systems": 4,
        "validation_fingerprint": first["validation_fingerprint"],
        "id_reference_sha256": first["id_reference_sha256"],
        "rouge_backend": rouge_artifacts[0]["backend"],
        "rouge_protocol": rouge_artifacts[0]["pyrouge_default_args"],
        "rows": public_rows,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(public_rows), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in _SPECS:
        slug = name.replace("-", "_")
        parser.add_argument(f"--{name}-config", dest=f"{slug}_config", required=True)
        parser.add_argument(f"--{name}-rouge", dest=f"{slug}_rouge", required=True)
        parser.add_argument(f"--{name}-metrics", dest=f"{slug}_metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifacts = {}
    for name in _SPECS:
        slug = name.replace("-", "_")
        artifacts[name] = {field: getattr(args, f"{slug}_{field}") for field in ("config", "rouge", "metrics")}
    build_table(artifacts, args.output)


if __name__ == "__main__":
    main()
