from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import torch

from .config import resolve_path

_DETOKENIZE_SLOW_RE = re.compile(
    r"(?:``|''|[^\S \n]|\s{2,}|^\s|\s$|\n\s|\s\n|"
    r"\s+[,.;:!?%]|\s+n['’]t\b|\s+['’](?:s|re|ve|ll|d|m)\b|"
    r"[\(\[\{]\s+|\s+[\)\]\}])",
    flags=re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:!?%])")
_OPEN_BRACKET_SPACE_RE = re.compile(r"([\(\[\{])\s+")
_CLOSE_BRACKET_SPACE_RE = re.compile(r"\s+([\)\]\}])")
_CONTRACTION_NT_RE = re.compile(r"\s+n['’]t\b", flags=re.IGNORECASE)
_CONTRACTION_SUFFIX_RE = re.compile(r"\s+(['’](?:s|re|ve|ll|d|m))\b", flags=re.IGNORECASE)


def detokenize(text: str) -> str:
    if not text or (
        not _DETOKENIZE_SLOW_RE.search(text) and (text.isascii() or unicodedata.is_normalized("NFKC", text))
    ):
        return text
    text = unicodedata.normalize("NFKC", text).replace("``", '"').replace("''", '"')
    lines = []
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        line = _PUNCT_SPACE_RE.sub(r"\1", line)
        line = _OPEN_BRACKET_SPACE_RE.sub(r"\1", line)
        line = _CLOSE_BRACKET_SPACE_RE.sub(r"\1", line)
        line = _CONTRACTION_NT_RE.sub("n't", line)
        line = _CONTRACTION_SUFFIX_RE.sub(r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def read_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit == 0:
        return
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield row
            count += 1
            if limit is not None and limit >= 0 and count >= limit:
                break


def source_text(row: Dict[str, Any], config: Dict[str, Any]) -> str:
    field = str(config["data"].get("source_field", "text"))
    value = row.get(field)
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"Input field {field!r} must contain only strings")
        separator = str(config["data"].get("list_separator", "\n"))
        value = separator.join(item.strip() for item in value if item.strip())
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Input row has no nonempty string in source field {field!r}")
    text = value.strip()
    if bool(config["data"].get("detokenize", False)):
        text = detokenize(text)
    return text.strip()


def load_source_path(
    config: Dict[str, Any],
    explicit_path: Optional[str] = None,
    default_key: str = "input_file",
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    path = config["data"].get(default_key)
    if not path:
        raise ValueError(f"Set data.{default_key} or pass --input-file")
    return resolve_path(path, config)


def collect_teacher_traces(
    adapter,
    config: Dict[str, Any],
    input_path: str | Path,
    output_path: str | Path,
    limit: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite teacher traces: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    data_config = config["data"]
    id_field = str(data_config.get("id_field", "id"))
    rows_written = 0
    generated_tokens = 0
    with temporary_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(read_jsonl(input_path, limit=limit)):
            text = source_text(row, config)
            input_ids, attention_mask = adapter.tokenize_source(text)
            target_ids = adapter.native_generate(input_ids, attention_mask).detach().cpu().tolist()
            if not target_ids or target_ids[0] != adapter.start_token_id:
                target_ids = [adapter.start_token_id, *target_ids]
            record = {
                "id": str(row.get(id_field, index)),
                "source_ids": input_ids[0].detach().cpu().tolist(),
                "decoder_ids": [int(value) for value in target_ids],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1
            generated_tokens += max(0, len(target_ids) - 1)
            if rows_written % 10 == 0:
                handle.flush()
    os.replace(temporary_path, output_path)
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "examples": rows_written,
        "teacher_generated_tokens": generated_tokens,
        "target": adapter.target_manifest(),
    }


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]], overwrite: bool = False) -> int:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    count = 0
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(temporary_path, path)
    return count


def all_ones_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


class RandomAccessJsonl:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._offsets = []
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self._offsets[index])
            raw = handle.readline()
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL at {self.path}:{self._offsets[index]}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {self.path}:{self._offsets[index]}")
        return row
