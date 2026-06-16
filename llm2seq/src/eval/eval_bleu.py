"""
BLEU evaluation for LLM2Seq.

Supports sacrebleu BLEU and chrF metrics.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import torch


def evaluate_bleu(
    predictions: List[str],
    references: List[str],
    lowercase: bool = False,
) -> Dict[str, float]:
    """
    Compute BLEU and chrF scores.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        lowercase: Whether to lowercase before scoring.

    Returns:
        Dict with "bleu", "chrf" scores.
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu is required. Install with: pip install sacrebleu")

    if lowercase:
        predictions = [p.lower() for p in predictions]
        references = [r.lower() for r in references]

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    # chrF
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    return {
        "bleu": bleu.score,
        "bleu_bp": bleu.bp,
        "bleu_precisions": bleu.precisions,
        "chrf": chrf.score,
    }


def evaluate_bleu_from_file(
    predictions_file: str,
    references_file: str,
    pred_field: str = "prediction",
    ref_field: str = "reference",
) -> Dict[str, float]:
    """
    Evaluate BLEU from JSONL files.

    Args:
        predictions_file: JSONL file with predictions.
        references_file: JSONL file with references.
        pred_field: Field name for prediction text.
        ref_field: Field name for reference text.

    Returns:
        BLEU and chrF scores.
    """
    predictions = []
    references = []

    with open(predictions_file, "r") as pf, open(references_file, "r") as rf:
        for p_line, r_line in zip(pf, rf):
            pred = json.loads(p_line.strip())
            ref = json.loads(r_line.strip())
            predictions.append(pred.get(pred_field, ""))
            references.append(ref.get(ref_field, ""))

    return evaluate_bleu(predictions, references)
