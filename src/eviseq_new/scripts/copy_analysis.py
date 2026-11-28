from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from export_samples import DEFAULT_SOURCE, matched_documents, source_documents

WORD = re.compile(r"[A-Za-z0-9]+(?:[.,'-][A-Za-z0-9]+)*")
TERM = re.compile(r"[A-Z][A-Za-z0-9-]*[A-Z0-9][A-Za-z0-9-]*")
RARE_SHARE, COMMON_SHARE = 0.005, 0.2
CATEGORIES = ("number", "term", "rare word", "other word", "common word")
CHECKED = ("number", "term", "rare word")


def key(word: str) -> str:
    return word.replace(",", "").lower()


def category(word: str, frequency: dict[str, float]) -> str:
    if TERM.fullmatch(word):
        return "term"
    if any(character.isdigit() for character in word):
        return "number"
    share = frequency.get(word.lower(), 0.0)
    if share <= RARE_SHARE:
        return "rare word"
    return "common word" if share >= COMMON_SHARE else "other word"


def document_frequency(sources: Sequence[str]) -> dict[str, float]:
    counts: Counter = Counter()
    for source in sources:
        counts.update({word.lower() for word in WORD.findall(source)})
    return {word: count / len(sources) for word, count in counts.items()}


def _keys(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for word in WORD.findall(text):
        found.setdefault(key(word), word)
    return found


def unsupported_rates(
    predictions: dict[str, str], documents: dict[str, dict[str, str]], frequency: dict[str, float]
) -> dict[str, tuple[int, int]]:
    unsupported, total = Counter(), Counter()
    for identifier, prediction in predictions.items():
        source = _keys(documents[identifier]["source"])
        for word in WORD.findall(prediction):
            kind = category(word, frequency)
            if kind in CHECKED:
                total[kind] += 1
                unsupported[kind] += key(word) not in source
    return {kind: (unsupported[kind], total[kind]) for kind in CHECKED}


def _sentence(text: str, word: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if key(word) in _keys(sentence):
            return sentence.strip()
    return ""


def candidate_examples(
    candidate: dict[str, str],
    baseline: dict[str, str],
    documents: dict[str, dict[str, str]],
    frequency: dict[str, float],
    *,
    minimum_similarity: float = 0.6,
) -> list[dict[str, Any]]:
    found = []
    for identifier, written in baseline.items():
        if identifier not in candidate:
            continue
        document = documents[identifier]
        source, kept = _keys(document["source"]), _keys(candidate[identifier])
        reference = _keys(document["reference"])
        for word_key, word in _keys(written).items():
            kind = category(word, frequency)
            if kind not in CHECKED or word_key in source:
                continue
            options = [
                (SequenceMatcher(None, word_key, other).ratio(), other)
                for other in kept
                if other in source and other != word_key and category(kept[other], frequency) != "common word"
                and any(c.isdigit() for c in other) == any(c.isdigit() for c in word_key)
            ]
            if not options:
                continue
            similarity, other = max(options)
            needed = 0.5 if any(c.isdigit() for c in word_key) else minimum_similarity
            if similarity < needed:
                continue
            found.append({
                "id": identifier,
                "category": kind,
                "baseline_wrote": word,
                "source_has": kept[other],
                "similarity": round(similarity, 2),
                "in_reference": other in reference,
                "baseline_sentence": _sentence(written, word),
                "candidate_sentence": _sentence(candidate[identifier], kept[other]),
                "reference_sentence": _sentence(document["reference"], kept[other]),
                "source_sentence": _sentence(document["source"], kept[other]),
            })
    return sorted(found, key=lambda row: (not row["in_reference"], -row["similarity"], row["id"]))


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.batch_decode([list(ids)], skip_special_tokens=True)[0]


def token_spans(tokenizer: Any, ids: Sequence[int]) -> list[tuple[int, int]]:
    spans, previous = [], 0
    for end in range(1, len(ids) + 1):
        length = len(_decode(tokenizer, ids[:end]))
        spans.append((previous, max(previous, length)))
        previous = max(previous, length)
    return spans


def words_for_tokens(text: str, spans: Sequence[tuple[int, int]]) -> list[str | None]:
    words = [(match.start(), match.end(), match.group()) for match in WORD.finditer(text)]
    return [next((word for start, end, word in words if start < stop and end > begin), None) for begin, stop in spans]


@torch.inference_mode()
def gate_records(
    model: Any,
    batch: dict[str, Any],
    token_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> list[dict[str, Any]]:
    from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
    from eviseq_afmr.evaluation import generate as generator

    if token_ids.shape[0] != 1:
        raise ValueError("Analyse one document at a time")
    head = model.decoder.grounded_copy
    if head is None:
        raise ValueError("This run has no copy head; use a full SEAM run")
    device = token_ids.device
    bridge = model.encode_source(
        batch["input_ids"], batch["attention_mask"], batch["source_content_mask"], batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"], torch.full((1,), max_new_tokens, device=device, dtype=torch.float32),
        **{name: batch[name] for name in COPY_INPUT_KEYS if name in batch},
    )
    width = batch["decoder_prompt_ids"].shape[1]
    inputs = token_ids[:, :-1]
    mask = torch.cat(
        (batch["decoder_prompt_mask"].bool(),
         torch.ones((1, inputs.shape[1] - width), dtype=torch.bool, device=device)), dim=1)
    seen: dict[str, torch.Tensor] = {}
    original = head.distribution

    def recorded(hidden, state):
        attention, copy_gate, generate_gate = original(hidden, state)
        seen["attention"], seen["gate"] = attention, copy_gate
        return attention, copy_gate, generate_gate

    head.distribution = recorded
    try:
        logits, _, _ = model.decoder(
            inputs, bridge.memory, bridge.memory_mask, bridge.source_bias, mask,
            value_memory=bridge.value_memory, copy_state=bridge.copy_state,
        )
    finally:
        del head.distribution
    state = bridge.copy_state
    chat_prompt = bool(model.config.get("data", {}).get("decoder_chat_template", False))
    records = []
    for position in range(width, token_ids.shape[1]):
        token = int(token_ids[0, position])
        if eos_id is not None and token == eos_id:
            break
        step = position - 1
        scores = logits[0, step].float()
        gate = float(seen["gate"][0, step, 0].float().exp())
        matches = (state.token_ids[0] == token) & state.mask[0]
        from_source = gate * float(seen["attention"][0, step].float().exp()[matches].sum())
        probability = float(torch.softmax(scores, dim=-1)[token])
        chosen = scores.clone()[None]
        history = token_ids[:, width:position] if chat_prompt else token_ids[:, :position]
        generator._apply_repetition_penalty(chosen, history, repetition_penalty)
        generator._apply_no_repeat_ngram(chosen, history, no_repeat_ngram_size)
        if eos_id is not None and position - width < min_new_tokens:
            chosen[:, eos_id] = float("-inf")
        records.append({
            "token_id": token,
            "gate": min(1.0, max(0.0, gate)),
            "copy_share": min(1.0, from_source / probability) if probability > 0 else 0.0,
            "is_argmax": int(chosen.argmax()) == token,
        })
    return records


def gate_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "| word kind | tokens | mean copy gate | median copy gate | tokens with gate >= 0.5 (%) "
        "| mean share of probability from copy | words found in source (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in CATEGORIES:
        chosen = [row for row in rows if row["category"] == kind]
        if not chosen:
            continue
        gates = [row["gate"] for row in chosen]
        lines.append(
            f"| {kind} | {len(chosen)} | {statistics.mean(gates):.3f} | {statistics.median(gates):.3f} "
            f"| {100 * sum(g >= 0.5 for g in gates) / len(gates):.1f} "
            f"| {statistics.mean(row['copy_share'] for row in chosen):.3f} "
            f"| {100 * sum(row['in_source'] for row in chosen) / len(chosen):.1f} |"
        )
    return "\n".join(lines)


def run_gate(args: argparse.Namespace) -> None:
    from eviseq_afmr.config import load_config
    from eviseq_afmr.data.collate import SummarizationCollator
    from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
    from eviseq_afmr.data.schema import CanonicalRecord
    from eviseq_afmr.evaluation.generate import generate_greedy
    from eviseq_afmr.modeling.model import EviSeqAFMR
    from eviseq_afmr.runtime import _configure_precision, _tokenizers
    from eviseq_afmr.training.checkpoint import load_checkpoint

    config = load_config(args.run_dir / "resolved_config.yaml")
    _configure_precision(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = copy.deepcopy(config)
    if device.type == "cuda":
        inference["model"]["dtype"] = config["model"].get("compute_dtype", config["model"].get("dtype", "float32"))
    model = EviSeqAFMR(inference).to(device)
    load_checkpoint(args.checkpoint or args.run_dir / "last.pt", model, config=config, restore_rng=False)
    model.eval()
    encoder_tokenizer, tokenizer = _tokenizers(config)
    collator = SummarizationCollator(encoder_tokenizer, tokenizer, config["data"], grounded_copy=True)
    collator.include_targets = False
    documents = source_documents(args.source)
    frequency = document_frequency([document["source"] for document in documents])
    generation = config["generation"]
    maximum = int(generation["max_new_tokens"])
    eos = tokenizer.eos_token_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, checked, matching = [], 0, 0
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for number, document in enumerate(documents[: args.count], start=1):
            batch = collator([CanonicalRecord(document["id"], document["source"], "")])
            batch = {name: value.to(device) for name, value in batch.items()
                     if isinstance(value, torch.Tensor)
                     and (name in COPY_INPUT_KEYS or name in
                          ("input_ids", "attention_mask", "source_content_mask", "decoder_prompt_ids",
                           "decoder_prompt_mask"))}
            texts, token_ids = generate_greedy(
                model, batch, tokenizer, maximum, int(generation.get("min_new_tokens", 0)),
                float(generation.get("repetition_penalty", 1.0)), int(generation.get("no_repeat_ngram_size", 0)),
            )
            records = gate_records(
                model, batch, token_ids, max_new_tokens=maximum, eos_id=eos,
                min_new_tokens=int(generation.get("min_new_tokens", 0)),
                repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
                no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
            )
            ids = [record["token_id"] for record in records]
            spans = token_spans(tokenizer, ids)
            text = _decode(tokenizer, ids)
            source = _keys(document["source"])
            for record, word in zip(records, words_for_tokens(text, spans)):
                checked += 1
                matching += record["is_argmax"]
                if word is not None:
                    rows.append({"id": document["id"], "word": word, "category": category(word, frequency),
                                 "in_source": key(word) in source, **record})
            predictions.write(json.dumps({"id": document["id"], "reference": document["reference"],
                                          "prediction": texts[0]}, ensure_ascii=False) + "\n")
            predictions.flush()
            print(f"{number}/{args.count} documents", flush=True)
    (args.output_dir / "gate_tokens.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = (f"{gate_table(rows)}\n\nDocuments: {min(args.count, len(documents))}. Word tokens: {len(rows)}. "
               f"The recomputed pass picks the generated token for {100 * matching / max(1, checked):.1f}% of tokens.\n")
    (args.output_dir / "gate_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def run_examples(args: argparse.Namespace) -> None:
    label, separator, other = args.baseline.partition("=")
    if not separator or not label or not Path(other).is_file():
        raise SystemExit(f"--baseline expects LABEL=PATH of an existing file, got {args.baseline!r}")
    matched = matched_documents(args.source, {"candidate": args.candidate, "baseline": Path(other)})
    if not matched:
        raise SystemExit("No test document has a prediction from both systems")
    documents = {document["id"]: document for document, _ in matched}
    candidate = {document["id"]: summaries["candidate"] for document, summaries in matched}
    baseline = {document["id"]: summaries["baseline"] for document, summaries in matched}
    frequency = document_frequency([document["source"] for document in source_documents(args.source)])
    lines = [f"Documents: {len(matched)}", "", f"| word kind | SEAM not in source | {label} not in source |",
             "|---|---|---|"]
    ours = unsupported_rates(candidate, documents, frequency)
    theirs = unsupported_rates(baseline, documents, frequency)
    for kind in CHECKED:
        cells = [f"{100 * bad / total:.2f}% ({bad}/{total})" if total else "-" for bad, total in (ours[kind], theirs[kind])]
        lines.append(f"| {kind} | {cells[0]} | {cells[1]} |")
    reverse = candidate_examples(baseline, candidate, documents, frequency)
    examples = candidate_examples(candidate, baseline, documents, frequency)
    lines += ["", f"{label} miswrites a source word that SEAM keeps: {len(examples)} cases. "
              f"The other way round: {len(reverse)} cases.", ""]
    for row in examples[: args.top]:
        lines += [f"## {row['id']} | {row['category']} | {label}: {row['baseline_wrote']} | source: {row['source_has']}"
                  f" | in reference: {row['in_reference']}",
                  f"- source: {row['source_sentence']}", f"- reference: {row['reference_sentence']}",
                  f"- SEAM: {row['candidate_sentence']}", f"- {label}: {row['baseline_sentence']}", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print(f"top {min(args.top, len(examples))} cases -> {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy-gate statistics and grounded error examples.")
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("gate", help="copy-gate statistics of a full SEAM run (needs its checkpoint)")
    gate.add_argument("--run-dir", type=Path, required=True)
    gate.add_argument("--checkpoint", type=Path)
    gate.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    gate.add_argument("--count", type=int, default=100)
    gate.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/copy_gate")
    examples = commands.add_parser("examples", help="unsupported words and side-by-side cases from prediction files")
    examples.add_argument("--candidate", type=Path, required=True, help="SEAM predictions")
    examples.add_argument("--baseline", required=True, metavar="LABEL=PREDICTIONS.jsonl",
                          help="the system to compare SEAM against")
    examples.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    examples.add_argument("--top", type=int, default=30)
    examples.add_argument("--output", type=Path, default=PACKAGE_ROOT / "runs/rebuttal/copy_examples.md")
    args = parser.parse_args(argv)
    run_gate(args) if args.command == "gate" else run_examples(args)


if __name__ == "__main__":
    main()
