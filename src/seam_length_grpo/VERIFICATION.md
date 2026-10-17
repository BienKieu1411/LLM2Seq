# Local verification - 2026-10-17

Scope: the new `src/seam_length_grpo` experiment. This implementation reused
the existing `seam_grpo` training core and `eviseq_new` architecture without
changing their production logic. The later audit adds regression checks/docs.
No model checkpoint was downloaded. No commit or push was made.

Runtime: `/Users/kieugiangbien/bienkieu_env/bin/python`, tiny randomly initialized
Qwen3 fixtures, CPU/Gloo, real TRL 1.16.0 and its actual training/objective path.
This is a correctness check, not evidence of downstream ROUGE/length gains.

Before CE warm-up integration: **22 passed** in 44.54s; Ruff lint/format and shell syntax passed.
Warnings were dependency deprecations and TRL's experimental `rollout_func`
notice; no test was skipped. GPU kernel tests were not part of this suite.

The suite now also exercises a CE-only adaptation stage before GRPO: distinct
per-sample word prompts, global gold-token normalization under accumulation,
enabled/disabled schedules, separate warm-up export, and the two-stage Gloo
transition. See the final update below for the current test result.

The data annotation CLI is also checked for distinct per-record lengths,
preservation of every other field/sample, target lists/nested fields,
normalization parity with the training loader, atomic failure without dropping
bad records, explicit output/length replacement, and execution without any
model directories.

Final update after CE warm-up + annotation integration: **35 passed** in 56.17s,
Ruff lint/format, shell syntax and `git diff --check` passed. Checks explicitly
confirm that the adapted actor matches `length_ce_warmup.pt` at GRPO entry,
GRPO optimizer moments start empty, and frozen teacher weights still equal the
original SFT checkpoint. No CUDA/NCCL kernel test or downstream score run was
performed.

After enabling the source term by default (eta=0.05), the targeted reward,
per-sample length and real TRL/teacher checks passed: **14 passed** in 10.92s.
This includes default-config activation and the complete three-term formula.

Architecture-specific audit on 2026-10-17: the baseline suites passed 41
(`seam_grpo`) and 37 (`seam_length_grpo`) checks. Three added base checks passed:
full-reference encoder independence with/without copy and direct RL gradients
through H0/M/value memory/source bias/copy keys/copy bias. The length runtime
check additionally passed with beta=0.01, a shared frozen SFT teacher/reference,
per-sample ragged word prompts, CE adaptation and fresh GRPO optimizer state.
No new production correctness defect was reproduced in these routes.

Final full-suite rerun after adding these regressions: **44 passed** in 45.53s
for `seam_grpo`; **38 passed** in 54.87s for `seam_length_grpo` (82 total, no
skips). Ruff lint/format and whitespace checks passed. Only checks and
documentation changed during this audit; production settings and logic did not.

## Checks

```bash
/Users/kieugiangbien/bienkieu_env/bin/python -m pytest -q src/seam_length_grpo/checks
/Users/kieugiangbien/bienkieu_env/bin/python -m ruff check src/seam_length_grpo
/Users/kieugiangbien/bienkieu_env/bin/python -m ruff format --check src/seam_length_grpo
bash -n src/seam_length_grpo/scripts/run_grpo.sh src/seam_length_grpo/scripts/evaluate.sh
```

The check suite covers:

- Word counting agrees with the existing Vietnamese length evaluator, including
  decomposed Unicode and punctuation. Length penalty remains active when ROUGE
  is zero.
- Different W values produce different prompt widths. Left padding, ignored
  prompt labels and copy offsets remain correct. Source visibility stays fixed
  when W changes. Contradictory CE requests and truncated gold are rejected.
- Unpadded policy scores and parameter gradients match the reference per-row
  computation. Nonzero finite gradients reach encoder, bridge, decoder and copy.
- Native sampling log-probabilities match full differentiable policy scoring,
  including temperature, EOS, copy mixture and batched equal-width candidates.
- Teacher scores reverse sign when true/negative sources are swapped, and all
  frozen teacher weights/gradients remain unchanged through actual updates.
- Negative pools cover multiple ordered domains and reject documents that only
  differ beyond the visible source prefix. Selection uses the encoder tokenizer
  and the same content budget as actor/teacher inputs.
- Real single-process TRL executes two optimizer updates with teacher weight 0
  and 0.05, changes encoder weights and uses fresh policy versions 0 then 1.
- Real two-process TRL/Gloo checks exact equality of gradients and post-update
  weights across ranks, with CE disabled/enabled and a frozen source teacher.
- A transport double exercises the new vLLM request path in distributed TRL:
  per-row encoder prefix, decoder prompt IDs, source/token cap, version sync,
  engine recreation and likelihood parity checks. Initializer checks cover
  single/two-rank setup and variable prompt reserves.
- Evaluation reloads standard checkpoints, requires explicit test word requests,
  distinguishes external requests from a reference-length oracle, rejects
  accidental overwrite, and does not require historical SFT weights to remain.
- Two-process evaluation merges unequal shard sizes in original input order and
  matches the single-process output exactly on the tiny fixture.
- Exported inference settings preserve the source BODY budget separately from
  prompt reserve, including when the input config leaves body size implicit.
- Checkpoint ranking uses every dataset's length constraint, rather than letting
  one large dataset hide a smaller dataset's violation.

A bounded read-only review identified the visible-prefix negative collision;
it was fixed and added to regression checks. Documentation, CLI `--help`, Ruff
and shell syntax are checked separately from the training math.

## Limits that need server verification

No CUDA/NCCL or real vLLM kernels run on this Mac. The transport double validates
control flow, not numerical equivalence of GPU kernels or throughput. Run a short
server training job first with native rollout, then the SEAM vLLM backend and
inspect `behavior_logp_mae` / `behavior_logp_max_error` and memory usage.
Keep the runtime mismatch guard enabled.

The teacher term is experimental and now enabled by default with weight 0.05
to implement the agreed three-term reward; weight 0 disables it. It costs a frozen full
model plus two candidate-scoring passes; it has not been shown to improve ROUGE,
source support or length. Native scoring recomputes encoder/bridge states with
gradients; the implementation does not claim encoder-once-per-group training.

ROUGE reward and validation selection use the fast Unicode proxy, not the Perl
metric. Final comparisons must use the existing ROUGE-1.5.5 and length evaluators
under the same word-request protocol. Test references never supply W implicitly.

Exports are standard SEAM weight checkpoints with optimizer metadata, but this
entrypoint does not implement exact TRL optimizer/RNG/rollout resume. Evaluation
does not yet resume partially written predictions.
