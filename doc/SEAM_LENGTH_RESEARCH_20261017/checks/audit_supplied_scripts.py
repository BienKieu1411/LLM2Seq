"""Offline, model-free numerical audit of the supplied length scripts.

Only selected pure function definitions and counting constants are extracted
through AST. Training entrypoints, imports and model loading are never run.
The ROUGE function in the gating experiment is an explicit controlled stub;
this checks reward algebra, not measured summary quality.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

SCRIPT_NAMES = (
    "lafrt.py",
    "lead_eigen_threshold.py",
    "lead_eigen.py",
    "lead_len_control.py",
    "schedule_running.py",
    "semantic_lafrt_len_balanced.py",
)


def definitions(path: Path, functions: set[str], constants: set[str], namespace: dict) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            selected.append(node)
            found.update(target.id for target in node.targets if isinstance(target, ast.Name))
    missing = (functions | constants) - found
    if missing:
        raise ValueError(f"Missing definitions in {path}: {sorted(missing)}")
    # Restore imports explicitly rather than executing source-module imports.
    result = {"math": math, "re": re, "unicodedata": unicodedata, **namespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), result)
    return result


def run(scripts: Path, repo: Path) -> dict:
    files = {name: scripts / name for name in SCRIPT_NAMES}
    fingerprint = {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in files.items()
    }
    evaluator = repo / "src/evaluation/evaluate_length_vietnamese.py"
    official = definitions(evaluator, {"count_words"}, {"WORD_RE"}, {})
    texts = ["Việt Nam đang phát triển", "công nghệ thông tin", "🙂", "tiếng Việt", "tiếng Việt"]
    word_counts = []
    for name in ("lafrt.py", "semantic_lafrt_len_balanced.py"):
        namespace = definitions(
            files[name], {"count_words"}, {"LARFT_WORD_PATTERN", "UNICODE_WORD_PATTERN", "WORD_COUNT_MODE"}, {}
        )
        for text in texts:
            namespace["WORD_COUNT_MODE"] = "larft"
            larft = namespace["count_words"](text)
            namespace["WORD_COUNT_MODE"] = "unicode"
            unicode_count = namespace["count_words"](text)
            word_counts.append(
                {
                    "script": name,
                    "text": text,
                    "legacy": larft,
                    "unicode": unicode_count,
                    "evaluator": official["count_words"](text),
                }
            )

    controlled_quality = 0.4

    def controlled_rouge(reference, response):
        return dict.fromkeys(("rouge", "rouge1", "rouge2", "rougeL"), controlled_quality)

    lead = definitions(
        files["lead_len_control.py"],
        {"word_count", "gaussian_length_factor", "compute_reward"},
        {"ROUGE_WEIGHT", "EIGEN_WEIGHT", "LENGTH_SIGMA_RATIO", "LENGTH_SIGMA_MIN"},
        {"normalize_text": str, "clean_completion": str, "compute_rouge": controlled_rouge},
    )
    gate_rows = []
    for quality in (0.4, 0.5, 0.6):
        controlled_quality = quality
        for count in (50, 100, 150):
            row = lead["compute_reward"]("reference", " ".join(["word"] * count), 100, "unused", "unused")
            gate_rows.append(
                {
                    key: row[key]
                    for key in ("content_score", "response_words", "target_length", "length_penalty", "final_reward")
                }
            )
    low = [row["final_reward"] for row in gate_rows if row["content_score"] == 0.4]
    high = [row["final_reward"] for row in gate_rows if row["content_score"] == 0.6]
    assert max(low) == min(low), "Expected the supplied low-quality gating dead zone"
    assert high[1] > high[0] and high[1] > high[2]
    unicode_rows = [row for row in word_counts if row["script"] == "semantic_lafrt_len_balanced.py"]
    assert all(row["unicode"] == row["evaluator"] for row in unicode_rows)
    assert any(row["legacy"] != row["evaluator"] for row in word_counts)
    return {
        "scope": "pure functions with controlled ROUGE stub; no training or model inference",
        "fingerprints": fingerprint,
        "word_counts": word_counts,
        "lead_len_control_gate": gate_rows,
        "verified": {
            "length_inactive_at_quality_0_4": True,
            "length_active_at_quality_0_6": True,
            "balanced_unicode_mode_matches_evaluator_on_examples": True,
            "default_larft_counter_disagrees_on_vietnamese": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripts-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("result.json"))
    args = parser.parse_args()
    result = run(args.scripts_dir, args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "verified": result["verified"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
