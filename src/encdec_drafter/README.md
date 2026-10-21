# Encoder-decoder drafter prototype

This directory starts the final-source-memory drafter prototype from the implementation plan. The Colab notebook trains a small draft module against the frozen T5Gemma 2 1B-1B teacher and checks greedy speculative decoding on a small CNN/DailyMail slice.

## Colab

Open `colab/t5gemma2_drafter.ipynb` in Google Colab with a GPU runtime. Before running it, accept the Gemma license for `google/t5gemma-2-1b-1b` on Hugging Face and authenticate with a Hugging Face token that can read the gated model. The notebook does not contain or save a token.

It downloads 128 CNN/DailyMail training articles and 32 validation articles, generates teacher continuations, trains only the small drafter, and saves its checkpoint under `/content/t5gemma2_drafter/`. The teacher remains frozen. CNN/DailyMail references are not used as teacher targets: the drafter learns to predict the frozen teacher's own greedy outputs.

This is the plan's final-memory prototype. It does not yet capture a live intermediate encoder state or overlap drafting with the encoder. The notebook reports token equality and draft acceptance diagnostics, not a production speedup. Exact token equality with ordinary greedy generation is a prerequisite before implementing and benchmarking a KV-cache-aware verifier.
