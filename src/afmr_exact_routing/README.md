# AFMR Exact Routing

This independent experiment keeps the projected final encoder states, copied
decoder cross-attention and grounded-copy head. It replaces a static source
prior with a decoder-query-dependent route over **original source tokens**.
Each decoder query first distributes probability over non-overlapping source
blocks, then over the valid tokens in each block. The joint distribution reads
the original projected token values. It does not prune source tokens or change
the grounded-copy value anchor.

The route enters each decoder layer through a zero-initialized bounded gate.
At initialization, the full graph matches `direct_projection` exactly; the
gate receives the first CE gradient, then its router learns. No auxiliary
salience loss, candidate generation or contrastive training is used.

Default PubMed routing uses rank 128, block size 128 and query chunks of 32.
Because the route evaluates query–token scores in every decoder layer, it is
more expensive than the other variants. The PubMed config uses batch 8 and
accumulation 12 to retain the baseline effective batch of 96 on one GPU.
Real GPU memory and speed remain to be measured.

## Run

```bash
cd src/afmr_exact_routing
CUDA_VISIBLE_DEVICES=0 AFMR_ENCODERS=pplx \
PPLX_ENCODER=/local/path/to/pplx-embed-v1-0.6b \
DECODER_MODEL=/local/path/to/Qwen3-0.6B \
bash scripts/run_pubmed_pair.sh
```

For the direct-projection + grounded-copy ablation, also set
`AFMR_BRIDGE_MODE=direct_projection` and use a fresh `AFMR_OUTPUT_DIR`.
For two GPUs, set `CUDA_VISIBLE_DEVICES=0,1`; the runner uses DDP. Evaluation
uses `last.pt` and the same greedy temperature/top-k/top-p configuration as
the baseline.

The folder carries its own package, configs, runners and tests. To verify
without downloading a model:

```bash
cd src/afmr_exact_routing
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q tests/test_exact_routing.py
```

The tests cover query-dependent token reads, masking, route gradients,
same-seed initialization parity with the ablation, warm-up optimizer coverage
and checkpoint graph separation. A ROUGE gain is an experimental hypothesis,
not a property established by these tests.
