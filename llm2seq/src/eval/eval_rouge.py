"""
ROUGE evaluation for LLM2Seq.

Supports ROUGE-1, ROUGE-2, ROUGE-L.
"""

from __future__ import annotations

from typing import Dict, List


def evaluate_rouge(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = False,
) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        predictions: List of predicted strings.
        references: List of reference strings.
        use_stemmer: Whether to use stemmer (typically False for non-English).

    Returns:
        Dict with "rouge1", "rouge2", "rougeL", "rougeLsum".
    """
    try:
        import evaluate
    except ImportError:
        raise ImportError("evaluate is required. Install with: pip install evaluate")

    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer,
    )

    return {k: round(v, 4) for k, v in scores.items()}
