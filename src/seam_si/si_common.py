"""Streaming data and provenance helpers for the SEAM SI preparation scripts."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import islice
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "eviseq_new"))
# Some server checkouts retain the earlier evaluation directory name.
EVALUATION_ROOT = SOURCE_ROOT / "evaluation"
if not EVALUATION_ROOT.is_dir():
    EVALUATION_ROOT = SOURCE_ROOT / "rouge155"
sys.path.insert(0, str(EVALUATION_ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_line(value):
    return json.dumps(value, ensure_ascii=False) + "\n"


def read_rows(path, *, recover_tail=False):
    """Only prediction resume may repair an interrupted final JSONL write."""
    if not path.exists():
        return
    with path.open("r+b" if recover_tail else "rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                if recover_tail and not raw.endswith(b"\n") and not stream.read(1):
                    stream.truncate(offset)
                    print(f"[resume] Repaired interrupted final write in {path}", flush=True)
                    break
                raise ValueError(f"Invalid JSONL in {path} at byte {offset}") from exc
            if recover_tail and not raw.endswith(b"\n"):
                stream.write(b"\n")
            yield row


def records(path, data):
    from eviseq_afmr.data.normalization import detokenize
    from eviseq_afmr.data.schema import CanonicalRecord

    index = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                record = CanonicalRecord.from_mapping(
                    json.loads(raw),
                    source_field=data.get("source_field", "text"),
                    target_field=data.get("target_field", "summary"),
                    id_field=data.get("id_field", "id"),
                    separator=data.get("list_separator", "\n"),
                )
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid train sample {path}:{line_number}: {exc}") from exc
            row = {
                "row_index": index,
                "line_number": line_number,
                "id": record.example_id or str(index + 1),
                # The server applies source detokenization from its resolved config.
                "source": record.source,
                "reference": detokenize(record.target) if data.get("detokenize") else record.target,
            }
            row["record_sha256"] = digest(row)
            yield row
            index += 1


def check_candidates(row):
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 8:
        raise ValueError(f"Expected 8 candidates at row {row.get('row_index')}")
    if [item.get("index") for item in candidates] != list(range(8)):
        raise ValueError(f"Invalid candidate indices at row {row.get('row_index')}")
    if not all(isinstance(item.get("prediction"), str) for item in candidates):
        raise ValueError("Candidate prediction must be text")


def batches(rows, size):
    iterator = iter(rows)
    while batch := list(islice(iterator, size)):
        yield batch


def shard_bounds(total, rank, count):
    if count < 1 or not 0 <= rank < count:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_rank < num_shards")
    return total * rank // count, total * (rank + 1) // count
