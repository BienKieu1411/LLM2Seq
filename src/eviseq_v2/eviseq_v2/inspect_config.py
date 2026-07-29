"""Offline config/parameter-budget preflight that never loads a model."""

from __future__ import annotations

import argparse
import json

from .config import architecture_contract, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "experiment": config["experiment"],
                "architecture_sha256": config["_meta"]["architecture_sha256"],
                "architecture": architecture_contract(config),
                "data": config["data"],
                "training": config["training"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
