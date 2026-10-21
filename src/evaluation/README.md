# Text quality diagnostics

The scripts in this directory keep the existing Perl ROUGE-1.5.5 evaluator
separate from semantic and factuality diagnostics. The BERTScore, AlignScore,
and SCALE scripts consume JSONL rows; the Vietnamese ROUGE runner below reads
the server's source/reference TXT tree and can generate summaries first.

## Vietnamese ROUGE-1.5.5 with local generation

`evaluate_rouge_vietnamese.py` reads source files from
`data_raw/<dataset>/**/*.txt`, generates predictions with a local Transformers
checkpoint or the repository's AFMR runtime, writes them under
`data_sum/<model>/<dataset>/`, and scores them against matching files in
`data_ref/<dataset>/`. It includes `ViM`, `book`,
`legal`, `vietnews`, `vlsp22`, `wikilingua`, and `xlsum`; `ViM_sum_tmp` is
excluded. Causal and encoder-decoder checkpoints are supported, and the model
is loaded with local-only paths so no checkpoint is downloaded.

```bash
python3 src/evaluation/evaluate_rouge_vietnamese.py \
  --model-path /path/to/local/checkpoint \
  --batch-size 1 \
  --device cuda:0 \
  --dtype bf16 \
  --max-input-tokens 8192 \
  --max-new-tokens 512
```

The default prompt is Vietnamese and the tokenizer's chat template is used
when available. Change the prompt with `--prompt-template '... {source} ...'`;
use `--no-chat-template` to force a plain prompt. Existing non-empty prediction
files are reused. Pass `--overwrite-predictions` to regenerate them. ROUGE
uses the historical preprocessing and flags (`-a -m -n 2 -d`), and results are
written to `rouge_old_summary.json` under the model output directory.

### Generate with a local AFMR/SEAM checkpoint (no vLLM)

For the repository's custom encoder-decoder checkpoint, use the AFMR runtime
backend. It reads the encoder/decoder prompt, source limit, generation limit,
and precision from the supplied resolved config, writes per-file predictions
to the same `data_sum/<model>/<dataset>/` layout, then runs ROUGE-1.5.5.
`--batch-size` and `--device` control inference. vLLM is not used.

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate_rouge_vietnamese.py \
  --backend afmr \
  --afmr-config /path/to/resolved_config.yaml \
  --afmr-checkpoint /path/to/last.pt \
  --model-name seam \
  --batch-size 4 \
  --device cuda:0
```

The AFMR source package must be present at `src/eviseq_new/eviseq_afmr` next
to this evaluator. Resumable intermediate JSONL files are kept under the
model's hidden `.afmr_work/` directory. Use `--overwrite-predictions` to
regenerate summaries instead of reusing existing per-file predictions.

### Generate with SEAM's local vLLM HTTP host

The Vietnamese vLLM evaluator can start its own SEAM HTTP host. If FlashInfer
fails during JIT kernel compilation on startup, select another attention
backend and disable its separate sampling kernel:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate_rouge_vietnamese_vllm.py \
  --backend http --start-server \
  --afmr-config /path/to/resolved_config.yaml \
  --afmr-checkpoint /path/to/last.pt \
  --attention-backend FLASH_ATTN \
  --disable-flashinfer-sampler \
  --model-name seam_vllm_test --batch-size 16
```

These options affect only a host started by this command. For an already
running external host, set the backend and `VLLM_USE_FLASHINFER_SAMPLER=0`
when launching `seam_vllm.serve` instead.

## BERTScore with a local model

`evaluate_bertscore.py` requires an existing Hugging Face model directory and
sets Transformers/Hugging Face offline mode before loading it. It never
downloads a model implicitly.

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_bertscore.py \
  src/T5Gemma/results/lrsum/predictions.jsonl \
  --model-path /path/to/local/bert-score-encoder \
  --num-layers 12 \
  --batch-size 16 \
  --device cuda:0 \
  --details
```

If `--num-layers` is omitted, the script reads the layer count from the local
`config.json`. When a row has multiple references, BERTScore uses the best
reference for that example, as implemented by the `bert-score` package. The
headline precision, recall and F1 values are reported on a 0--100 scale.

The evaluator patches the very-large `model_max_length` sentinel emitted by
some Transformers tokenizer configs. This avoids
`OverflowError: int too big to convert` with bert-score 0.3.x; pass
`--max-length` only when an explicit truncation limit is required.

## Paper-grounded hallucination score: AlignScore

For the paper's primary factuality result, use **AlignScore-nli_sp** from
Zha et al. (ACL 2023). AlignScore evaluates whether each prediction sentence
is supported by a source chunk, takes the best source support for each sentence,
and averages those values. Its native score is factual consistency (higher is
better); this repository also reports
`hallucination_score = 1 - alignscore_consistency`, so lower is better.

The evaluator in `evaluate_alignscore.py` follows the released AlignScore
`nli_sp` inference path while avoiding the original package's old
PyTorch/Transformers pins. Both the AlignScore backbone directory and the
trained `.ckpt` must already exist locally. Transformers offline mode and
`local_files_only=True` prevent accidental downloads.

EviSeq prediction files contain `id`, `prediction` and `reference`, but do not
include the source text. Pass the original test JSONL with `--source-file`;
the evaluator joins its `text` field to predictions by `id` before scoring.

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_alignscore.py \
  runs/model/last_test_predictions.jsonl \
  --model-path /models/roberta-large \
  --checkpoint-path /models/AlignScore-large.ckpt \
  --source-file datasets/pubmed/test.jsonl \
  --batch-size 32 \
  --device cuda:0 \
  --details
```

Use `--source-file-field` when the test JSONL stores the source under another
field such as `source` or `document`; use `--source-id-field` when its ID field
is not `id`. Prediction files that already contain `source` can still be
evaluated without `--source-file`.

The JSON output contains both `alignscore_consistency` and the transformed
`hallucination_score`, together with the checkpoint path, chunk size and
sentence splitter for reproducibility. Cite:

> Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu. 2023. *AlignScore:
> Evaluating Factual Consistency with a Unified Alignment Function*. ACL.

Paper: <https://aclanthology.org/2023.acl-long.634/>. Reference implementation:
<https://github.com/yuh-zha/AlignScore>.

## SCALE source-support score

`evaluate_scale.py` implements the released SCALE inference path from Lattimer
et al. (EMNLP 2023). It splits each prediction into sentences, scores each
sentence against overlapping source windows with the Flan-T5 `Yes`/`No` NLI
prompt, keeps the best supporting window, and averages over sentences. The
reported `scale_consistency` is high-is-better. The accompanying
`unsupported_content_proxy = 1 - scale_consistency` is only a monotonic
diagnostic, not a calibrated hallucination probability.

The script loads a local Flan-T5 Hugging Face directory and performs batched
one-token generation; it does not download a model. The `scale-score` package
is not required, but can be installed for comparison with the released API:

```bash
python3 -m pip install scale-score
```

```bash
PYTHONPATH=src \
  /Users/kieugiangbien/bienkieu_env/bin/python \
  src/rouge155/evaluate_scale.py \
  runs/model/last_test_predictions.jsonl \
  --model-path /models/flan-t5-large \
  --size large \
  --source-file datasets/pubmed/test.jsonl \
  --batch-size 8 \
  --chunk-size 1000 \
  --window-size 0.25 \
  --device cuda:0 \
  --progress-every 100 \
  --details
```

Prediction rows may contain `source` directly. Otherwise, pass the original
test JSONL through `--source-file`; sources are joined by `id`, just as in the
AlignScore evaluator. Keep the same local checkpoint, tokenizer, chunk size,
overlap, sentence splitter, and aggregation rule for every model comparison.

Paper: <https://aclanthology.org/2023.emnlp-main.105/>. Reference
implementation: <https://github.com/asappresearch/scale-score>.

## SEAM candidate sampling diagnostic

Run from the repository root on a CUDA server with the existing SEAM vLLM
0.25.1 environment and Perl ROUGE installation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate_seam_candidates.py \
  --config src/eviseq_new/runs/afmr/train/resolved_config.yaml \
  --checkpoint src/eviseq_new/runs/afmr/train/epoch_004.pt \
  --train-file src/eviseq_new/datasets/grpo_rouge/train.jsonl \
  --output-dir runs/candidates_sft_train100 \
  --num-samples 100 --seed 42 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 1024 --batch-size 8
```

The script randomly samples 100 train rows without replacement, saves their
IDs and physical line numbers, and generates eight candidates per source via
`SEAMLLM.generate()` and `SamplingParams(n=8)`. Comparisons use the same nested
first 1, 4, and 8 candidates. Input limits and encoder/decoder prompts come
from the resolved config; `--max-input-tokens` can lower the input budget.
Only local model files are loaded. Each completed source is immediately
appended to `candidates.jsonl`; rerunning the same command resumes completed
rows. Use a new output directory for another checkpoint, keeping seed and
train file unchanged for a comparison on the same examples.

`comparison.json` reports the macro mean of per-candidate Perl ROUGE-1.5.5 F1
with the Vietnamese evaluator's preprocessing and `-a -m -n 2 -d` flags:

- `random_candidate_expectation`: mean over all candidates in each group,
  estimating selection uniformly at random.
- `oracle_weighted_rouge`: choose each source's candidate using gold and
  `(R1 + 2*R2 + RL)/4`, then report that candidate's R1/R2/RL.
- `oracle_rouge2`: choose using gold ROUGE-2, then report its R1/R2/RL.

The oracle rows measure available candidate quality, not an inference method
that can be used without a reference. N=1 is a sampled candidate at temperature
1, not a greedy baseline. These scores describe 100 train samples, not held-out
quality. ROUGE is in [0, 1] in JSON and displayed as percentages in the console.
`scored_candidates.jsonl` preserves all candidates and individual scores;
`rouge_raw.txt` preserves the Perl output. Add `--score-only` to the same
command to rescore saved candidates without loading model weights.

If engine startup fails while importing an incompatible FlashInfer cache,
pass `--attention-backend FLASH_ATTN` to select FlashAttention explicitly
through vLLM's `attention_config`. Set `VLLM_USE_FLASHINFER_SAMPLER=0` for the
same command so sampling also avoids importing FlashInfer. This does not bypass FlashInfer's version
checks. The choice is recorded in the experiment manifest; keep it fixed for
checkpoint comparisons. A failed startup with no saved candidates can reuse
the same output directory when changing this option.
