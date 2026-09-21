# Cross-Tokenizer Ordered Value Bridge

This folder is an independent experiment built from the `eviseq_new` training and decoding pipeline. It leaves the direct encoder projection unchanged for cross-attention keys and grounded copy, then adds character-span-aligned decoder-token features only to cross-attention values.

The default PubMed run uses PPLX, grounded copy, FP32 parameters and optimizer states, and BF16 autocast. It evaluates the validation split after training. Test evaluation is opt-in so architecture selection does not repeatedly inspect the test set.

```bash
cd src/xov_bridge
CUDA_VISIBLE_DEVICES=0 \
PPLX_ENCODER=/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/path/to/Qwen3-0.6B \
ROUGE155_SCRIPT=/path/to/evaluate_rouge.py \
PYROUGE_HOME_DIR=/path/to/ROUGE-1.5.5 \
bash scripts/run_pubmed_pair.sh
```

Run the direct-projection control with the same training configuration:

```bash
XOV_BRIDGE_MODE=direct_projection bash scripts/run_pubmed_pair.sh
```

After locking the architecture on validation, add `XOV_EVAL_TEST=true` for the final test evaluation. Use `XOV_ENCODERS=qwen_embedding` or `XOV_ENCODERS=both` only when the corresponding encoder control is required.
An existing checkpoint stops a repeat run by default; set `OVERWRITE_OUTPUT_DIR=true` only to intentionally replace that run.

Offline verification does not download checkpoints:

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q
/Users/kieugiangbien/bienkieu_env/bin/python -m ruff check .
/Users/kieugiangbien/bienkieu_env/bin/python -m ruff format --check .
```
