"""Audit and filter a balanced Vietnamese summarization JSONL without changing its input."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+(?:[.,]\d+)*", re.UNICODE)
VI_MARK = re.compile(r"[ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
FOREIGN_SCRIPT = re.compile(
    r"[\u0370-\u03ff\u0400-\u052f\u0590-\u08ff\u0900-\u0dff\u0e00-\u0eff\u1100-\u11ff"
    r"\u3040-\u30ff\u3400-\u9fff\uac00-\ud7ff]"
)
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d▲▶►◆★☆●■□◉☑✓✔]")
PLACEHOLDER = re.compile(r"(?:_{4,}|-{6,}|\.{6,}|…{3,})")
TABLE_ROW = re.compile(r"^\s*\|(?:[^\n|]*\|){2,}\s*$", re.M)
LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|begin\{|end\{|alpha|beta|theta|cdot)\b|\$[^$\n]{2,100}\$", re.I)
EQUATION = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:\([^\n)]{1,30}\))?\s*(?:=|≈|≤|≥)\s*[-+\dA-Za-z(]", re.I)
OCR_MARKER = re.compile(r"\[(?:image|page|scan)\b[^\]]{0,80}\]|download\s+e-?book|tr[oò]n\s+b[oộ]\s+sgk", re.I)
URL = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+|\b(?:[A-Za-z0-9-]+\.)+(?:com|vn|org|net|io|edu|gov)(?:/[^\s<>\"']*)?",
    re.I,
)
MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\(((?:https?://|www\.)[^\s)]+)\)", re.I)
DEFAULT_JUNK_DOMAINS = frozenset({"downloadsachmienphi.com", "bookgiaokhoa.com", "example.com", "test.com"})
EN_WORDS = frozenset(
    "the and of to in is are was were that this with from for by as on at an a be have has had it its "
    "which can will their these those not or but into between such through about than then when where while "
    "chapter section table figure introduction conclusion exercise answer question choose write read following".split()
)
VI_WORDS = frozenset(
    "và của là có được trong cho một những các này đó với từ theo khi sẽ đã không về trên dưới "
    "để người năm tháng ngày bài văn bản nội dung nghiên cứu kết quả phần thông tin trường hợp quy định".split()
)


def english_passages(text: str) -> int:
    count = 0
    for passage in re.split(r"\n+|(?<=[.!?])\s+", text):
        words = [word.lower() for word in WORD.findall(passage) if word.isalpha()]
        if len(words) < 12:
            continue
        english = sum(word in EN_WORDS for word in words)
        vietnamese = sum(word in VI_WORDS for word in words)
        if english / len(words) >= 0.25 and english >= 3 * max(vietnamese, 1) and not VI_MARK.search(passage):
            count += 1
    return count


def text_field(row: dict[str, Any], field: str) -> str:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing field {field!r}")
        value = value[part]
    if not isinstance(value, str):
        raise ValueError(f"field {field!r} must be a string")
    return value


def set_text_field(row: dict[str, Any], field: str, text: str) -> None:
    value = row
    parts = field.split(".")
    for part in parts[:-1]:
        value = value[part]
    value[parts[-1]] = text


def junk_url(url: str, junk_domains: set[str]) -> bool:
    normalized = url.rstrip(r".,;:!?)\]")
    if not re.match(r"^[a-z]+://", normalized, re.I):
        normalized = "https://" + normalized
    host = (urlsplit(normalized).hostname or "").lower().removeprefix("www.")
    return any(host == domain or host.endswith("." + domain) for domain in junk_domains)


def clean_symbols(text: str, junk_domains: set[str]) -> tuple[str, int, int]:
    text = unicodedata.normalize("NFC", text)
    removed_markdown = 0

    def clean_markdown(match: re.Match[str]) -> str:
        nonlocal removed_markdown
        if junk_url(match.group(2), junk_domains):
            removed_markdown += 1
            return ""
        return match.group(0)

    text = MARKDOWN_LINK.sub(clean_markdown, text)
    removed_urls = 0

    def clean_url(match: re.Match[str]) -> str:
        nonlocal removed_urls
        if junk_url(match.group(0), junk_domains):
            removed_urls += 1
            return ""
        return match.group(0)

    text = URL.sub(clean_url, text)
    cleaned, count = EMOJI.subn("", text)
    cleaned = "".join(char for char in cleaned if unicodedata.category(char) != "Cf")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip(), count, removed_markdown + removed_urls


def inspect_text(text: str, junk_domains: set[str]) -> dict[str, float | int]:
    words = WORD.findall(text)
    lower = [word.lower() for word in words if word.isalpha()]
    letters = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    alnum = letters + digits
    numeric_tokens = sum(any(char.isdigit() for char in word) for word in words)
    foreign_letters = len(FOREIGN_SCRIPT.findall(text))
    placeholder_runs = PLACEHOLDER.findall(text)
    table_rows = len(TABLE_ROW.findall(text))
    lines = [line for line in text.splitlines() if line.strip()]
    tabular_lines = sum(line.count("\t") >= 2 for line in lines)
    formula_count = len(LATEX.findall(text)) + len(EQUATION.findall(text))
    ocr_markers = len(OCR_MARKER.findall(text))
    urls = URL.findall(text)
    junk_urls = sum(junk_url(url, junk_domains) for url in urls)
    foreign_passages = english_passages(text)
    vi_marks = len(VI_MARK.findall(text))
    en_words = sum(word in EN_WORDS for word in lower)
    vi_words = sum(word in VI_WORDS for word in lower)
    return {
        "words": len(words),
        "digit_ratio": round(digits / max(1, alnum), 4),
        "numeric_token_ratio": round(numeric_tokens / max(1, len(words)), 4),
        "foreign_script_ratio": round(foreign_letters / max(1, letters), 4),
        "foreign_script_chars": foreign_letters,
        "vi_mark_ratio": round(vi_marks / max(1, letters), 4),
        "en_stopword_ratio": round(en_words / max(1, len(lower)), 4),
        "vi_stopword_ratio": round(vi_words / max(1, len(lower)), 4),
        "placeholder_runs": len(placeholder_runs),
        "placeholder_chars": sum(len(run) for run in placeholder_runs),
        "table_rows": table_rows,
        "tabular_lines": tabular_lines,
        "formula_count": formula_count,
        "ocr_markers": ocr_markers,
        "url_count": len(urls),
        "junk_url_count": junk_urls,
        "english_passages": foreign_passages,
        "url_char_ratio": round(sum(len(url) for url in urls) / max(1, len(text)), 4),
    }


def classify(source: dict[str, float | int], target: dict[str, float | int]) -> tuple[list[str], list[str]]:
    drop: list[str] = []
    review: list[str] = []
    for name, stats in (("source", source), ("target", target)):
        words = int(stats["words"])
        digit_ratio = float(stats["digit_ratio"])
        numeric_ratio = float(stats["numeric_token_ratio"])
        foreign_chars = int(stats["foreign_script_chars"])
        foreign_ratio = float(stats["foreign_script_ratio"])
        en_ratio = float(stats["en_stopword_ratio"])
        vi_ratio = float(stats["vi_stopword_ratio"])
        vi_mark_ratio = float(stats["vi_mark_ratio"])
        placeholders = int(stats["placeholder_runs"])
        placeholder_chars = int(stats["placeholder_chars"])
        tables = int(stats["table_rows"]) + int(stats["tabular_lines"])
        formulas = int(stats["formula_count"])
        ocr_markers = int(stats["ocr_markers"])
        url_count = int(stats["url_count"])
        junk_urls = int(stats["junk_url_count"])
        foreign_passages = int(stats["english_passages"])
        url_char_ratio = float(stats["url_char_ratio"])
        if name == "source" and words < 12:
            review.append(f"{name}:very_short")
        if foreign_chars >= 20 and foreign_ratio >= 0.04:
            drop.append(f"{name}:foreign_script")
        elif foreign_chars >= 5:
            review.append(f"{name}:mixed_script")
        english_threshold = 0.24 if name == "target" else 0.12
        minimum_words = 12 if name == "target" else 25
        if words >= minimum_words and en_ratio >= english_threshold and en_ratio >= 2.5 * max(vi_ratio, 0.01):
            drop.append(f"{name}:english_dominant")
        elif words >= 40 and en_ratio >= 0.07 and vi_mark_ratio < 0.01 and vi_ratio < 0.04:
            review.append(f"{name}:language_uncertain")
        elif words >= 40 and vi_mark_ratio < 0.005 and vi_ratio < 0.02:
            review.append(f"{name}:latin_language_uncertain")
        if foreign_passages >= 2:
            drop.append(f"{name}:english_passages")
        elif foreign_passages == 1:
            review.append(f"{name}:english_passage")
        if words >= 40 and (digit_ratio >= 0.30 or numeric_ratio >= 0.38):
            drop.append(f"{name}:numeric_dense")
        elif words >= 40 and (digit_ratio >= 0.18 or numeric_ratio >= 0.25):
            review.append(f"{name}:numeric_moderate")
        if tables >= 3:
            drop.append(f"{name}:table")
        elif tables >= 1:
            review.append(f"{name}:possible_table")
        if formulas >= 3:
            drop.append(f"{name}:math")
        elif formulas >= 1:
            review.append(f"{name}:possible_math")
        if placeholders >= 3 or placeholder_chars >= 30:
            drop.append(f"{name}:form_placeholders")
        elif placeholders >= 1:
            review.append(f"{name}:possible_form")
        if ocr_markers >= 3:
            drop.append(f"{name}:ocr_watermark")
        elif ocr_markers >= 1:
            review.append(f"{name}:possible_ocr")
        if junk_urls >= 3:
            drop.append(f"{name}:junk_url_repeated")
        elif junk_urls >= 2 or (name == "target" and junk_urls >= 1):
            review.append(f"{name}:junk_urls_removed")
        if url_count >= 5 or (url_count >= 3 and url_char_ratio >= 0.2):
            review.append(f"{name}:link_heavy")
    return drop, review


def atomic_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return os.fdopen(descriptor, "w", encoding="utf-8"), Path(name)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else None
    decisions = args.decisions.expanduser().resolve() if args.decisions else None
    report_path = args.report.expanduser().resolve() if args.report else None
    destinations = [path for path in (output, decisions, report_path) if path]
    if len(set(destinations)) != len(destinations) or input_path in destinations:
        raise ValueError("input, output, decisions, and report must be different paths")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    junk_domains = set() if args.no_default_junk_domains else set(DEFAULT_JUNK_DOMAINS)
    junk_domains.update(domain.lower().removeprefix("www.") for domain in args.junk_domain)
    handles = {}
    temporary = {}
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    try:
        for name, path in (("output", output), ("decisions", decisions)):
            if path:
                handles[name], temporary[name] = atomic_writer(path)
        with input_path.open("r", encoding="utf-8-sig") as source_file:
            for line_number, raw in enumerate(source_file, 1):
                if not raw.strip():
                    continue
                counts["rows"] += 1
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError("row is not an object")
                    source = text_field(row, args.source_field)
                    target = text_field(row, args.target_field)
                    if not source.strip() or not target.strip():
                        raise ValueError("source or target is empty")
                except (json.JSONDecodeError, ValueError) as exc:
                    status, drop_reasons, review_reasons = "drop", ["invalid_row"], []
                    identifier, cleaned_source, cleaned_target = None, "", ""
                    source_stats, target_stats = {}, {}
                    counts["invalid_rows"] += 1
                    detail = str(exc)
                else:
                    identifier = row.get(args.id_field)
                    source_stats = inspect_text(source, junk_domains)
                    target_stats = inspect_text(target, junk_domains)
                    drop_reasons, review_reasons = classify(source_stats, target_stats)
                    cleaned_source, source_icons, source_urls = clean_symbols(source, junk_domains)
                    cleaned_target, target_icons, target_urls = clean_symbols(target, junk_domains)
                    counts["removed_icons"] += source_icons + target_icons
                    counts["removed_urls"] += source_urls + target_urls
                    if not cleaned_source.strip() or not cleaned_target.strip():
                        drop_reasons.append("empty_after_cleaning")
                    status = "drop" if drop_reasons else "review" if review_reasons else "keep"
                    detail = ""
                    if status == "keep" or (status == "review" and args.keep_review):
                        set_text_field(row, args.source_field, cleaned_source)
                        set_text_field(row, args.target_field, cleaned_target)
                        if "output" in handles:
                            handles["output"].write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts["written"] += 1
                counts[status] += 1
                for reason in drop_reasons + review_reasons:
                    reasons[reason] += 1
                if "decisions" in handles:
                    handles["decisions"].write(
                        json.dumps(
                            {
                                "line": line_number,
                                "id": identifier,
                                "status": status,
                                "drop_reasons": drop_reasons,
                                "review_reasons": review_reasons,
                                "source_metrics": source_stats,
                                "target_metrics": target_stats,
                                "source_preview": cleaned_source[: args.preview_chars],
                                "target_preview": cleaned_target[: args.preview_chars],
                                "error": detail,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        for handle in handles.values():
            handle.close()
        for name, path in (("output", output), ("decisions", decisions)):
            if path:
                temporary[name].replace(path)
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    report = {
        "input": str(input_path),
        "output": str(output) if output else None,
        "decisions": str(decisions) if decisions else None,
        "keep_review": args.keep_review,
        "junk_domains": sorted(junk_domains),
        "counts": dict(counts),
        "reasons": dict(reasons),
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Balanced input JSONL; never modified")
    parser.add_argument("--output", type=Path, help="Clean output JSONL; omitted means audit only")
    parser.add_argument("--decisions", type=Path, help="One decision and preview per input row")
    parser.add_argument("--report", type=Path, help="Summary counts by decision and reason")
    parser.add_argument("--source-field", default="input")
    parser.add_argument("--target-field", default="output")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--keep-review", action="store_true", help="Include review rows in output")
    parser.add_argument("--junk-domain", action="append", default=[], help="Additional junk domain to remove")
    parser.add_argument("--no-default-junk-domains", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=160)
    args = parser.parse_args()
    if args.preview_chars < 0:
        parser.error("--preview-chars must be nonnegative")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
