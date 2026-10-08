#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_config_digest(model_path: str) -> str | None:
    path = Path(model_path).expanduser()
    return _sha256(path / "config.json") if path.is_dir() else None


def _parse_encoder_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Encoder must use LABEL=LOCAL_MODEL_PATH format")
    label, model_path = spec.split("=", 1)
    label = label.strip()
    model_path = model_path.strip()
    if not label or not model_path:
        raise argparse.ArgumentTypeError("Encoder label and model path must both be non-empty")
    return label, model_path


def _format_millions(count: int) -> str:
    return f"{count / 1_000_000:.2f}M"


def _paper_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Encoder | Enc. Params | SEAM Params | Total Params |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['encoder_label']} | {_format_millions(row['encoder_params'])} | "
            f"{_format_millions(row['seam_params'])} | {_format_millions(row['total_params'])} |"
        )
    lines.extend(
        [
            "",
            "Note. SEAM Params counts the encoder-conditioned bridge. Decoder Params includes the pretrained decoder, added cross-attention, and grounded-copy modules; Total Params counts all unique model parameters.",
        ]
    )
    return "\n".join(lines)


def _json_text(config_path: Path, rows: list[dict[str, Any]]) -> str:
    payload = {
        "parameter_definition": {
            "encoder_params": "All unique parameters in the pretrained source encoder.",
            "seam_params": "All unique parameters in model.bridge; the encoder-conditioned SEAM bridge only.",
            "decoder_params": "All unique parameters in model.decoder, including pretrained decoder weights, added cross-attention and grounded-copy modules.",
            "interface_total_params": "SEAM bridge plus added cross-attention and grounded-copy parameters.",
            "total_params": "All unique parameters in the complete EviSeq-new model; verified to equal encoder + bridge + decoder.",
        },
        "base_config": str(config_path),
        "base_config_sha256": _sha256(config_path),
        "rows": rows,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    columns = (
        "encoder_label",
        "encoder_model",
        "resolved_config_sha256",
        "encoder_params",
        "seam_params",
        "decoder_params",
        "decoder_base_params",
        "decoder_cross_attention_params",
        "grounded_copy_params",
        "interface_total_params",
        "total_params",
    )
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows({key: row[key] for key in columns} for row in rows)
    return output.getvalue()


def _load_model_config(config_path: Path, encoder_path: str, decoder_path: str | None, dtype: str | None):
    from eviseq_afmr.config import load_config

    config = load_config(config_path)
    config["model"]["encoder_name"] = str(Path(encoder_path).expanduser().resolve())
    if decoder_path:
        config["model"]["decoder_name"] = str(Path(decoder_path).expanduser().resolve())
    if dtype:
        config["model"]["dtype"] = dtype
    return config


def audit(config_path: Path, encoders: list[tuple[str, str]], decoder_path: str | None, dtype: str | None):
    from eviseq_afmr.modeling.model import EviSeqAFMR
    from eviseq_afmr.parameter_audit import parameter_breakdown

    rows = []
    for label, encoder_path in encoders:
        encoder_path = str(Path(encoder_path).expanduser().resolve())
        config = _load_model_config(config_path, encoder_path, decoder_path, dtype)
        resolved_decoder = Path(config["model"]["decoder_name"]).expanduser()
        if not resolved_decoder.is_dir():
            raise FileNotFoundError(f"Local decoder directory not found: {resolved_decoder}")
        print(f"Loading {label}: encoder={encoder_path}; decoder={config['model']['decoder_name']}", flush=True)
        model = EviSeqAFMR(config)
        counts = parameter_breakdown(model)
        rows.append(
            {
                "encoder_label": label,
                "encoder_model": encoder_path,
                "encoder_config_sha256": _model_config_digest(encoder_path),
                "decoder_model": config["model"]["decoder_name"],
                "decoder_config_sha256": _model_config_digest(config["model"]["decoder_name"]),
                "resolved_config_sha256": _config_sha256(config),
                "parameter_count_load_dtype": config["model"].get("dtype", "float32"),
                **counts,
            }
        )
        print(
            f"{label}: encoder={counts['encoder_params']:,} | "
            f"SEAM bridge={counts['seam_params']:,} | decoder={counts['decoder_params']:,} | "
            f"total={counts['total_params']:,}",
            flush=True,
        )
        del model, config
        gc.collect()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count exact unique EviSeq-new parameters by encoder, SEAM bridge, decoder and total."
    )
    parser.add_argument("config", type=Path, help="AFMR task config or run's resolved_config.yaml")
    parser.add_argument(
        "--encoder",
        action="append",
        type=_parse_encoder_spec,
        required=True,
        metavar="LABEL=LOCAL_PATH",
        help="Encoder label and local checkpoint directory; repeat for each encoder",
    )
    parser.add_argument("--decoder", help="Optional local decoder checkpoint override")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), help="Load dtype; does not change counts")
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown")
    parser.add_argument("--output", type=Path, help="Save the selected output format to this file")
    parser.add_argument("--details-json", type=Path, help="Also save exact counts and model config hashes as JSON")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")
    for label, model_path in args.encoder:
        if not Path(model_path).expanduser().is_dir():
            parser.error(f"Local encoder directory for {label} not found: {model_path}")
    if args.decoder and not Path(args.decoder).expanduser().is_dir():
        parser.error(f"Local decoder directory not found: {args.decoder}")

    rows = audit(config_path, args.encoder, args.decoder, args.dtype)
    if args.format == "markdown":
        output = _paper_markdown(rows) + "\n"
    elif args.format == "csv":
        output = _csv_text(rows)
    else:
        output = _json_text(config_path, rows)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Saved {args.format} parameter report: {args.output}")
    else:
        print(output, end="")
    if args.details_json:
        details_path = args.details_json.expanduser()
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_path.write_text(_json_text(config_path, rows), encoding="utf-8")
        print(f"Saved exact parameter counts: {details_path}")


if __name__ == "__main__":
    main()
