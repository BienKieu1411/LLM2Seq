"""Run the actual eval CLI with an offline tokenizer fixture."""

import importlib.util
from pathlib import Path

import torch
from fixtures import NumeralTokenizer

torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("length_eval_entry", Path(__file__).resolve().parents[1] / "evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tokenizer = NumeralTokenizer()
module._tokenizers = lambda config: (tokenizer, tokenizer)
module.main()
