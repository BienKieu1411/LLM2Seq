# AFMR implementation evidence

## Scope

Implementation is now committed on local `main`. No remote push, pull request
or model download was used. The
authoritative specification is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
The only end-to-end model test uses Transformers' local `__tiny__` constructors.

## Commands and results

| Check | Command/result |
|---|---|
| Python compilation | `PYTHONPATH=src/afmr_core bienkieu_env/bin/python -m compileall -q src/afmr_core/afmr_core` — passed |
| Ruff lint | `bienkieu_env/bin/ruff check src/afmr_core` — passed |
| Ruff formatter | `bienkieu_env/bin/ruff format --check src/afmr_core` — passed |
| AFMR acceptance tests | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src/afmr_core bienkieu_env/bin/pytest -q src/afmr_core/tests` — **32 passed**, 2 dependency deprecation warnings; no model download path was exercised |
| Offline smoke | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON=bienkieu_env/bin/python src/afmr_core/run_afmr.sh smoke` — passed (`checkpoint_step=4`, `examples=2`, train CE `4.4775705839 -> 4.4749635646`) |
| PubMed runner dry run | `DRY_RUN=true .../scripts/run_pubmed_pair.sh` — passed; printed `workers=2`, `batch_per_gpu=84`, `accumulation=1`, `effective_examples_per_update=168`; default evaluation checkpoint is validation-selected `best.pt` |
| Cross-platform lint gate | Root and package Ruff check/format, `compileall`, shell syntax and all 12 YAML configs — passed |
| Tiny train/evaluate | two local tiny epochs plus test generation — passed; prediction resume returned identical metrics |

The v3 regression suite was not used as a v5 acceptance gate because its
existing recipe tests encode the older v3 schema. Its baseline was recorded
before implementation (`294 passed, 24 failed`) and remains a separate legacy
diagnostic.

## C1–C19 gap analysis

| ID | Evidence | Status |
|---|---|---|
| C1 | Readout zero-semantic endpoint and tiny model forward/backward tests; no `new` checkpoint is present for bitwise comparison | Partial: formula verified, historical checkpoint comparison pending |
| C2 | Explicit `legacy_copy_mixture` readout/cap/inner-gate config plus flat reader ordering/source-mask tests; no v2 checkpoint is present | Partial: graph contract verified, checkpoint parity pending |
| C3 | Duplicate-ID marginalization, probability mass and empty-target tests | Pass |
| C4 | Zero semantic route compared directly with the legacy copy mixture formula | Pass |
| C5 | Tiny candidate backward gives finite gradients through copy, semantic and router parameters; trainer calibrates `Wo` on a prompt-only batch before updates | Pass on tiny; zero-endpoint gradient table pending |
| C6 | Empty semantic/copy source fallback tests | Pass |
| C7 | Causal decoder graph and no gold/reference inputs in router; future-token perturbation test remains to be run | Partial |
| C8 | Dense and streamed vocabulary loss and hidden/lm-head gradients agree at `rtol=1e-5, atol=1e-5` through one shared mixture-NLL kernel | Pass |
| C9 | Token scaling code is implemented; no CUDA/NCCL hardware is available in this local run | Unrun hardware gate |
| C10 | FP32 finite checks and a BF16 reader forward/backward probe pass; BF16 CUDA rounding gate is unrun | Partial |
| C11 | Gauge identity, softmax recovery and a finite repetition-penalty 1.05 probe are covered by the readout test; full headline decoder probe remains pending | Partial |
| C12 | Tiny generation exercises source cache, prefix decode and prediction resume | Pass on CPU tiny |
| C13 | Atomic save/load, RNG/state metadata, strict config mismatch and resume world-size/protocol checks; scheduler state is restored | Pass |
| C14 | Sampled API validates temperature/top-k/top-p and applies them after final scores | Pass |
| C15 | Runner dry-run prints resolved config, global batch, dtype, seed and paths | Pass |
| C16 | Evaluation identity plus checkpoint metadata are verified before complete-cache metrics are returned; stale-manifest rejection is tested | Pass on tiny |
| C17 | Router features/cap use detached `g`, mixture uses live `g`, and the explicit Jacobian test verifies `dP/dg=Pcopy-P0` | Pass on oracle; shared-trunk drift logging remains a training gate |
| C18 | Fixed two-example tiny set trains for two epochs and asserts CE decrease (`4.4775705839 -> 4.4749635646`) | Pass on tiny smoke; longer registered run remains a research gate |
| C19 | Canonical manifest round-trip and rank sharding tests pass; CUDA replay is unrun | Pass logic / unrun hardware |

## Karpathy bounded review gate

The four stdlib checks were run after implementation. The strict complexity scan
warns on several inherited/copy-forward files, while the relaxed scan reports
9 low-severity findings across 37 files (mainly nesting in the existing
data/generation paths).
The diff-surgeon reports a 13% noise ratio because the package is a new folder
and therefore every docstring/blank line appears as an addition; it does not
identify a drive-by edit outside `src/afmr_core`. The plan/assumption
lint still reports a small number of generic research-plan words and one
documentation verification reminder; these are documentation quality warnings,
not unverified runtime behavior. The bounded reviewer report is recorded in
[`KARPATHY_REVIEW.md`](KARPATHY_REVIEW.md) and is the final gate for actionable
code findings; no model or dataset download is allowed during that review.

## Remaining research gates

The implementation is ready for a registered PubMed pilot only after local model
and processed split paths are supplied. The headline ROUGE targets are not
claimed here: no PubMed training, Perl ROUGE-1.5.5 run, three-seed comparison,
T5Gemma rerun or hallucination evaluation was performed in this CPU-only
implementation turn. Those results must use the protocol in
[`EVALUATION_PLAN.md`](EVALUATION_PLAN.md).
