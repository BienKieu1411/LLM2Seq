"""
Data preprocessing utilities for LLM2Seq.

Converts raw datasets into JSONL format suitable for training.
Supports loading from HuggingFace datasets or local files.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from datasets import load_dataset


def preprocess_and_save(
    dataset_name: str,
    output_dir: str,
    source_field: str = "source",
    target_field: str = "target",
    task_name: str = "seq2seq",
    train_split: str = "train",
    eval_split: str = "validation",
    test_split: str = "test",
    max_train_samples: int = -1,
    max_eval_samples: int = -1,
    dataset_config: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load a HuggingFace dataset and save as JSONL files.

    Args:
        dataset_name: HuggingFace dataset name.
        output_dir: Directory to save JSONL files.
        source_field: Field name for source text in the dataset.
        target_field: Field name for target text in the dataset.
        task_name: Task identifier to include in each example.
        train_split: Training split name.
        eval_split: Eval split name.
        test_split: Test split name.
        max_train_samples: Max samples for training (-1 for all).
        max_eval_samples: Max samples for eval (-1 for all).
        dataset_config: Optional dataset configuration name.

    Returns:
        Dict mapping split names to output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config)
    else:
        ds = load_dataset(dataset_name)

    output_paths = {}

    for split_name, max_samples in [
        (train_split, max_train_samples),
        (eval_split, max_eval_samples),
        (test_split, max_eval_samples),
    ]:
        if split_name not in ds:
            continue

        split_data = ds[split_name]
        if max_samples > 0:
            split_data = split_data.select(range(min(max_samples, len(split_data))))

        output_file = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, example in enumerate(split_data):
                record = {
                    "id": f"{split_name}_{idx:06d}",
                    "source": str(example.get(source_field, "")),
                    "target": str(example.get(target_field, "")),
                    "task": task_name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        output_paths[split_name] = output_file
        print(f"Saved {len(split_data)} examples to {output_file}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset into JSONL")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--source_field", default="source", help="Source text field name")
    parser.add_argument("--target_field", default="target", help="Target text field name")
    parser.add_argument("--task", default="seq2seq", help="Task name")
    parser.add_argument("--dataset_config", default=None, help="Dataset config name")
    parser.add_argument("--max_train", type=int, default=-1, help="Max training samples")
    parser.add_argument("--max_eval", type=int, default=-1, help="Max eval samples")
    args = parser.parse_args()

    preprocess_and_save(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        source_field=args.source_field,
        target_field=args.target_field,
        task_name=args.task,
        max_train_samples=args.max_train,
        max_eval_samples=args.max_eval,
        dataset_config=args.dataset_config,
    )


if __name__ == "__main__":
    main()
