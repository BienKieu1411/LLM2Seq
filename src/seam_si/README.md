# SEAM Self-Improve data preparation

This pipeline creates a new SFT training set. For **every row of the input train
JSONL**, the SEAM HTTP host samples eight summaries at temperature 1. After all
predictions finish, Perl ROUGE-1.5.5 scores every candidate against that row's
original gold. The candidate with the highest **ROUGE-2 F1** becomes the new
`summary`. ROUGE-1/L and the weighted score do not affect selection. Ties select
the lower candidate index. Empty candidates cannot become training targets;
if all eight are empty, selection stops rather than dropping the source.

The source remains unchanged. The output has the canonical AFMR fields
`id`, `text`, `summary`. Duplicate IDs are preserved: physical row indices,
not IDs, identify requests and resume records. Output order follows HTTP
completion order; every input row appears once. No validation/test file is
read, and no training/weight update is started by these scripts.

## Dependencies

One requirements file lists the direct dependencies for SI generation with
vLLM, CPU Perl ROUGE selection, AFMR fine-tuning and regression checks.
Install it in the compatible Python/CUDA environment on the GPU server:

```bash
python3 -m pip install -r src/seam_si/requirements.txt
```

vLLM, Python ROUGE and PyROUGE have exact version pins. Other constraints
follow the AFMR package requirements; this is not a full environment lockfile.
pip must also satisfy vLLM's transitive Torch/CUDA requirements. This full
requirements file is for the CUDA server, not the Mac verification environment.

If that environment already trains/hosts SEAM successfully, no reinstall is
required. SI uses no TRL dependency. `--score-workers` uses the Python standard
library and independent Perl processes; no extra parallel-computing package
is needed.

The ROUGE Python wrapper does not install the Perl ROUGE-1.5.5 script or its
data directory. Keep the existing local ROUGE installation and pass its path
with `--rouge-path`. On Debian/Ubuntu, install the Perl modules if missing:

```bash
apt-get update
apt-get install -y perl libxml-parser-perl libxml-dom-perl libdb-file-perl
```

## One command, including an automatically started host

Run from the repository root on the CUDA server, using its vLLM 0.25.1,
PyROUGE, and Perl ROUGE installation. All backbone/checkpoint paths are local.

```bash
CUDA_VISIBLE_DEVICES=0 bash src/seam_si/scripts/run_si.sh \
  --config src/eviseq_new/runs/afmr/train/resolved_config.yaml \
  --checkpoint src/eviseq_new/runs/afmr/train/epoch_004.pt \
  --train-file src/eviseq_new/datasets/grpo_rouge/train.jsonl \
  --output-dir src/eviseq_new/datasets/si_rouge2 \
  --start-server --server-port 8012 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 1024 \
  --batch-size 8 --max-num-seqs 64 \
  --attention-backend FLASH_ATTN
```

`--batch-size` is the number of concurrent **sources**, each producing eight
candidates. `--max-num-seqs` controls the host scheduler. Start conservatively
and increase concurrency on the server as memory permits. The auto host uses
the resolved config's encoder/decoder prompts and source limit. The input
budget can be lowered with `--max-input-tokens`; output budget defaults to the
host config unless `--max-new-tokens` is supplied. The auto host is stopped
before CPU/Perl scoring begins. FlashAttention is the default for this SI
launcher; it also disables the FlashInfer sampler in the spawned host to
avoid the incompatible FlashInfer cache encountered on the server.

The host's config/checkpoint remain unchanged throughout this single SI round.
This pipeline does not sync weights or run GRPO during generation.

## Two GPUs: one host per GPU, half the train rows each

```bash
GPU_IDS=0,1 bash src/seam_si/scripts/run_si_two_gpu.sh \
  --config src/eviseq_new/runs/afmr/train/resolved_config.yaml \
  --checkpoint src/eviseq_new/runs/afmr/train/epoch_004.pt \
  --train-file src/eviseq_new/datasets/grpo_rouge/train.jsonl \
  --output-dir src/eviseq_new/datasets/si_rouge2_two_gpu \
  --server-port 8012 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 1024 \
  --batch-size 8 --max-num-seqs 64 \
  --attention-backend FLASH_ATTN
```

The script starts two independent hosts concurrently: GPU 0 on port 8012,
GPU 1 on 8013. Each host loads the same complete teacher checkpoint with
tensor parallel size 1. Sources are divided into contiguous halves by
nonblank input-row index. With N rows, shard 0 gets `[0, floor(N/2))`, shard 1
gets `[floor(N/2), N)`. The original train file is read directly; no intermediate
train copies are needed. Every source still produces eight candidates. The
source seed is `seed + original_row_index`, independent of GPU/shard assignment.
Small floating-point differences across serving batches can still affect samples.

`--batch-size 8` and `--max-num-seqs 64` apply **per GPU**: up to eight sources /
64 candidate sequences per host. `GPU_IDS` can also contain the two GPU UUIDs;
if omitted, the wrapper uses `CUDA_VISIBLE_DEVICES`, falling back to `0,1`.

Each GPU saves/resumes its own `shard0/` or `shard1/` directory. The two host
logs are `<output>/shard0/seam_vllm_server.log` and
`<output>/shard1/seam_vllm_server.log`. After both workers and hosts exit,
the parent verifies the teacher/config/sampling identities and merges exactly
N unique row indices into `<output>/candidates.jsonl`. Perl selection then
creates the same final `train_si.jsonl` layout as the single-GPU pipeline.
If either GPU fails, owned hosts are stopped, completed predictions are kept,
and merging/scoring do not run. Rerun the same command to resume both shards.

If prediction finished but merging was interrupted, merge on CPU without hosts:

```bash
bash src/seam_si/scripts/run_si.sh --stage merge \
  --output-dir src/eviseq_new/datasets/si_rouge2_two_gpu
bash src/seam_si/scripts/run_si.sh --stage select \
  --output-dir src/eviseq_new/datasets/si_rouge2_two_gpu
```

Use `--stage predict` with the two-GPU wrapper to generate and merge without
scoring yet. Worker-level `--shard-rank` / `--num-shards` flags are only for
manual prediction dispatch; a shard cannot be selected as a full training set.
Update the shared HTTP-host helper along with this folder so its subprocess
environment pins each host to its assigned GPU.

## Use an existing host

Omit `--start-server`. Point the client at the host URL and model alias:

```bash
bash src/seam_si/scripts/run_si.sh \
  --config src/eviseq_new/runs/afmr/train/resolved_config.yaml \
  --train-file src/eviseq_new/datasets/grpo_rouge/train.jsonl \
  --base-url http://127.0.0.1:8012/v1 --model seam \
  --output-dir src/eviseq_new/datasets/si_rouge2 \
  --temperature 1.0 --top-p 1.0 --top-k 0 \
  --max-new-tokens 1024 --batch-size 8
```

An optional `--checkpoint` verifies that the host is serving that checkpoint.
Use `VLLM_API_KEY` for authentication if needed. The original gold is never
included in the HTTP prompt. Source normalization happens in the host;
reference detokenization follows `data.detokenize` in `--config`.

## Resume and score without a host

Rerun the same prediction command to resume; each completed source is flushed
to `candidates.jsonl` immediately. Successful in-flight requests are saved
before a request error is raised. Resume verifies the train content, resolved
config, host fingerprint, generation settings, and per-row source/reference.
An interrupted final JSONL write is repaired; completed rows are retained.
Use a new output directory for changed data/checkpoints/sampling settings.

Use `--stage predict` for prediction only. Once that stage is complete, score
and build the SI set without loading weights or contacting a host:

```bash
bash src/seam_si/scripts/run_si.sh \
  --stage select \
  --output-dir src/eviseq_new/datasets/si_rouge2 \
  --rouge-path /workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5 \
  --score-batch-size 128 \
  --score-workers 8
```

Perl runs on bounded batches (default 128 sources / 1024 candidates). Completed
scoring batches are cached for resume. Selection refuses incomplete/duplicate
predictions; final train and audit files are published only after every source
has a selected target. There is no score threshold and no source filtering.

`--score-workers` runs independent Perl jobs concurrently on CPU (default 1).
Start with 4 or 8 according to available CPU cores and temporary-disk capacity.
Each worker has a separate temporary directory; scoring flags, R2 selection,
tie handling and final row order remain unchanged. At most one batch per worker
is queued. Changing worker count reuses existing scoring caches; keep
`--score-batch-size` unchanged when resuming. Do not run two selection commands
against the same output directory simultaneously. `TMPDIR` can point at a fast
local scratch disk; concurrent jobs increase temporary storage and RAM use.

## Outputs inside `--output-dir`

- `train_si.jsonl`: new canonical SFT train set; `summary` is the best generated
  candidate by Perl ROUGE-2 F1, **not** the original gold.
- `candidates.jsonl`: all eight candidates plus original source/reference,
  row index, physical input line, and source ID.
- `selected.jsonl`: selected index/text, original gold, all eight score triples.
- `summary.json`: number of rows, selected mean R1/R2/RL, index counts, provenance.
- `rouge155/`: resumable selected batches and raw Perl output.
- `manifest.json`: train/server/generation identity for safe resume.
- `prediction_complete.json`: confirms all input rows completed under that manifest.

The Perl preprocessing and `-a -m -n 2 -d` flags are reused from the existing
Vietnamese candidate evaluator. Each JSONL target is one reference even if it
contains blank lines. Scores are F1 in [0, 1], with higher ROUGE-2 winning.
When fine-tuning this output, use `data.source_field: text` and
`data.target_field: summary` if the original config used other field names.
