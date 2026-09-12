"""Paper-grounded factuality evaluation with a local AlignScore checkpoint.

This module implements the ``nli_sp`` inference path of AlignScore
(Zha et al., ACL 2023) without importing the original package.  The original
package pins older PyTorch/Transformers releases and can therefore conflict
with the project environment.  The implementation keeps the method's
important evaluation steps:

* split the source into roughly 350-word chunks and the claim into sentences;
* score every source-chunk/claim-sentence pair with AlignScore's three-way
  alignment head; and
* take the best source support for each claim sentence, then average.

The reported ``alignscore_consistency`` is high-is-better.  The accompanying
``hallucination_score`` is ``1 - alignscore_consistency`` and is low-is-better.
Both the AlignScore model directory and its trained ``.ckpt`` must already be
available locally.  No model or tokenizer download is attempted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

try:
    from .metric_io import EvaluationRow, load_jsonl, metric_output_path
except ImportError:  # Direct ``python src/rouge155/evaluate_alignscore.py`` execution.
    from metric_io import EvaluationRow, load_jsonl, metric_output_path


ALIGN_SCORE_PAPER = "Zha et al., ACL 2023"
DEFAULT_CHUNK_WORDS = 350
DEFAULT_MAX_LENGTH = 512
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-ZÀ-ÖØ-Þ0-9])")


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences, using Punkt when its local data is present.

    AlignScore requires sentence-level claim aggregation.  NLTK's Punkt model
    is used when available; the regex fallback keeps server evaluation
    deterministic when the optional Punkt data was not installed.  The
    selected splitter is recorded in the result metadata.
    """

    value = text.strip()
    if not value:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(value)
        if sentences:
            return [sentence.strip() for sentence in sentences if sentence.strip()]
    except (ImportError, LookupError):
        pass
    return [part.strip() for part in _SENTENCE_BREAK.split(value) if part.strip()]


def sentence_splitter_name() -> str:
    """Return the splitter that will be used by :func:`_sentence_split`."""

    try:
        from nltk.data import find

        try:
            find("tokenizers/punkt_tab")
        except LookupError:
            find("tokenizers/punkt")
        return "nltk.sent_tokenize"
    except (ImportError, LookupError):
        return "regex_fallback"


def _source_chunks(source: str, chunk_words: int) -> list[str]:
    """Reproduce AlignScore's approximately ``chunk_words`` source chunks."""

    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    sentences = _sentence_split(source)
    if not sentences:
        return [""]

    # This mirrors the reference implementation: choose a sentence-group
    # size from the source word count, then concatenate complete sentences.
    n_groups = len(source.strip().split()) // chunk_words + 1
    sentences_per_group = max(len(sentences) // n_groups, 1)
    return [
        " ".join(sentences[start : start + sentences_per_group])
        for start in range(0, len(sentences), sentences_per_group)
    ]


def _load_checkpoint(path: Path) -> dict[str, Tensor]:
    """Load a Lightning or plain PyTorch checkpoint state dict."""

    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only``.
        payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"AlignScore checkpoint does not contain a state dict: {path}")
    state = {str(key): value for key, value in payload.items() if isinstance(value, Tensor)}
    if not state:
        raise ValueError(f"AlignScore checkpoint state dict is empty: {path}")
    return state


def _state_value(state: dict[str, Tensor], suffix: str) -> Tensor | None:
    """Find a checkpoint tensor despite Lightning's optional key prefixes."""

    for key, value in state.items():
        if key == suffix or key.endswith("." + suffix):
            return value
    return None


def _base_model_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Extract ``base_model`` weights from an AlignScore checkpoint."""

    extracted: dict[str, Tensor] = {}
    marker = "base_model."
    for key, value in state.items():
        if marker in key:
            extracted[key.split(marker, 1)[1]] = value
    return extracted


class LocalAlignScore:
    """AlignScore ``nli_sp`` scorer backed entirely by local files."""

    def __init__(
        self,
        model_path: Path,
        checkpoint_path: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        chunk_words: int = DEFAULT_CHUNK_WORDS,
        verbose: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if chunk_words <= 0:
            raise ValueError("chunk_words must be positive")
        if not model_path.is_dir():
            raise FileNotFoundError(f"AlignScore backbone path must be a local directory: {model_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AlignScore checkpoint must be a local file: {checkpoint_path}")

        # The evaluator must never turn a missing local artifact into a Hub
        # download.  ``local_files_only`` is also passed to both constructors.
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("AlignScore evaluation requires the `transformers` package") from exc

        self.model_path = model_path.expanduser().resolve()
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.batch_size = batch_size
        self.device = device or _default_device()
        self.max_length = max_length
        self.chunk_words = chunk_words
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.encoder = AutoModel.from_pretrained(str(self.model_path), local_files_only=True)
        checkpoint = _load_checkpoint(self.checkpoint_path)

        # AlignScore checkpoints contain the fine-tuned base encoder.  Loading
        # it is essential: using only the vanilla local RoBERTa would no longer
        # be the paper's trained metric.
        encoder_state = _base_model_state(checkpoint)
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)

        tri_weight = _state_value(checkpoint, "tri_layer.weight")
        tri_bias = _state_value(checkpoint, "tri_layer.bias")
        if tri_weight is None or tri_bias is None or tri_weight.ndim != 2 or tri_weight.shape[0] != 3:
            raise ValueError(
                "The checkpoint is missing AlignScore's tri_layer; pass an AlignScore .ckpt trained for nli_sp"
            )
        hidden_size = int(tri_weight.shape[1])
        self.tri_layer = nn.Linear(hidden_size, 3)
        self.tri_layer.load_state_dict({"weight": tri_weight, "bias": tri_bias})

        self.encoder.to(self.device).eval()
        self.tri_layer.to(self.device).eval()
        self._splitter = sentence_splitter_name()

    def _encode(self, contexts: Sequence[str], claims: Sequence[str]) -> dict[str, Tensor]:
        try:
            encoded = self.tokenizer(
                list(contexts),
                list(claims),
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            # Some tokenizer versions do not expose ``only_first``.  Pair
            # truncation is still source-first in the common local backbones.
            encoded = self.tokenizer(
                list(contexts),
                list(claims),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return {
            key: value for key, value in encoded.items() if key in {"input_ids", "attention_mask", "token_type_ids"}
        }

    def _forward(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:
        encoded = self._encode(contexts, claims)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        try:
            with torch.inference_mode():
                output = self.encoder(**encoded)
        except TypeError:
            # RoBERTa-style backbones reject token_type_ids even when a generic
            # tokenizer returns them.
            encoded.pop("token_type_ids", None)
            with torch.inference_mode():
                output = self.encoder(**encoded)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        with torch.inference_mode():
            logits = self.tri_layer(pooled.float())
            probabilities = torch.softmax(logits, dim=-1)[:, 0]
        return [float(value) for value in probabilities.detach().cpu().tolist()]

    def score_pairs(self, contexts: Sequence[str], claims: Sequence[str]) -> list[float]:
        """Return AlignScore entailment probabilities for aligned text pairs."""

        if len(contexts) != len(claims):
            raise ValueError("contexts and claims must have the same length")
        scores: list[float] = []
        for start in range(0, len(contexts), self.batch_size):
            scores.extend(
                self._forward(contexts[start : start + self.batch_size], claims[start : start + self.batch_size])
            )
        return scores

    def score_document(self, source: str, prediction: str) -> float:
        """Compute the paper's chunk-to-sentence AlignScore for one example."""

        claim_sentences = _sentence_split(prediction)
        if not claim_sentences:
            # Empty output is not a factual success.  Returning zero
            # consistency prevents a degenerate summary from receiving a free
            # low hallucination score after the 1-score transformation.
            return 0.0
        context_chunks = _source_chunks(source, self.chunk_words)
        contexts = [chunk for chunk in context_chunks for _ in claim_sentences]
        claims = claim_sentences * len(context_chunks)
        pair_scores = self.score_pairs(contexts, claims)
        support_scores = [
            max(pair_scores[offset + sentence_index] for offset in range(0, len(pair_scores), len(claim_sentences)))
            for sentence_index in range(len(claim_sentences))
        ]
        return sum(support_scores) / len(support_scores)


def _score_rows(rows: Sequence[EvaluationRow], scorer: Any) -> list[float]:
    return [float(scorer.score_document(row.source or "", row.prediction)) for row in rows]


def evaluate(
    predictions_file: Path,
    model_path: Path,
    checkpoint_path: Path,
    output_file: Path | None = None,
    *,
    prediction_field: str = "prediction",
    reference_field: str = "reference",
    source_field: str = "source",
    batch_size: int = 32,
    device: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    details: bool = False,
    scorer: Any | None = None,
) -> dict[str, Any]:
    predictions_file = predictions_file.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    rows = load_jsonl(
        predictions_file,
        prediction_field=prediction_field,
        reference_field=reference_field,
        source_field=source_field,
    )
    evaluator = scorer or LocalAlignScore(
        model_path,
        checkpoint_path,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        chunk_words=chunk_words,
    )
    consistency = [min(1.0, max(0.0, score)) for score in _score_rows(rows, evaluator)]
    hallucination = [1.0 - score for score in consistency]
    result: dict[str, Any] = {
        "schema_version": "eviseq.alignscore_hallucination.v1",
        "metric": "AlignScore-nli_sp",
        "paper": ALIGN_SCORE_PAPER,
        "definition": (
            "AlignScore factual-consistency score from source chunks to prediction sentences; "
            "hallucination_score is 1 - consistency"
        ),
        "score_direction": "hallucination_score_lower_is_better",
        "alignscore_consistency": sum(consistency) / len(consistency),
        "hallucination_score": sum(hallucination) / len(hallucination),
        "score_scale": "0-1",
        "num_examples": len(rows),
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path),
        "device": device or _default_device(),
        "batch_size": batch_size,
        "max_length": max_length,
        "chunk_words": chunk_words,
        "sentence_splitter": getattr(evaluator, "_splitter", sentence_splitter_name()),
        "prediction_field": prediction_field,
        "reference_field": reference_field,
        "source_field": source_field,
        "predictions_file": str(predictions_file),
    }
    if details:
        result["rows"] = [
            {
                "row_index": row.row_index,
                "id": row.identifier,
                "alignscore_consistency": consistency[row.row_index],
                "hallucination_score": hallucination[row.row_index],
            }
            for row in rows
        ]

    output_path = metric_output_path(predictions_file, output_file, ".alignscore.json")
    result["output_file"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate source-grounded factuality with AlignScore (Zha et al., ACL 2023). "
            "Both --model-path and --checkpoint-path are local; network access is disabled."
        )
    )
    parser.add_argument("predictions", type=Path, help="Prediction JSONL containing source, prediction and reference")
    parser.add_argument("--model-path", type=Path, required=True, help="Local AlignScore backbone directory")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Local AlignScore .ckpt file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Torch device, for example cpu or cuda:0")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--details", action="store_true", help="Include per-example scores in the JSON output")
    args = parser.parse_args()
    output = evaluate(
        args.predictions,
        args.model_path,
        args.checkpoint_path,
        args.output,
        prediction_field=args.prediction_field,
        reference_field=args.reference_field,
        source_field=args.source_field,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        chunk_words=args.chunk_words,
        details=args.details,
    )
    print(
        "AlignScore="
        f"{output['alignscore_consistency']:.4f} "
        "HallucinationScore="
        f"{output['hallucination_score']:.4f} (lower is better)"
    )
    print(f"Saved metrics: {Path(output['output_file']).resolve()}")


if __name__ == "__main__":
    main()
