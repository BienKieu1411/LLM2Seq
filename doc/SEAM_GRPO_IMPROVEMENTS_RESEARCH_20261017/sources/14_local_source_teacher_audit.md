# Local Code Audit: SourceTeacher Redundancy, Copy-Gate Bias, and Tanh Saturation

- Source: `src/seam_length_grpo/seam_length_grpo/teacher.py:18-58`, `src/seam_length_grpo/seam_length_grpo/data.py:88-155, 220-240`, `src/seam_length_grpo/seam_length_grpo/train.py:90-110, 185-204`, `src/eviseq_new/eviseq_afmr/modeling/grounded_copy.py:80-103`.
- Accessed: 2026-10-17.
- Type: primary local codebase inspection.
- Credibility: direct code inspection. Recency: current workspace snapshot. Bias: none.

Verified code excerpt (`src/seam_length_grpo/seam_length_grpo/teacher.py:44-57`):
```python
for source in (true_source, wrong_source):
    values = []
    for start in range(0, len(rows), self.micro_batch_size):
        end = min(len(rows), start + self.micro_batch_size)
        batch = trajectory(
            select_rows(source, range(start, end)), actions[start:end], mask[start:end], self.budget
        )
        logps = self.actor.policy(batch, temperature=1.0)
        values.append(logps.masked_fill(~mask[start:end], 0).sum(-1) / mask[start:end].sum(-1))
    means.append(torch.cat(values))
delta = means[0] - means[1]
return torch.tanh(delta / self.config["tau"]).tolist(), delta.tolist()
```

Key findings:
1. **$2G$ full encoder+bridge+copy passes per prompt group:** With `score_micro_batch_size=1` and $G=4$, `SourceTeacher` invokes `self.actor.policy(batch, temperature=1.0)` $8$ times per source group, re-encoding the 1 unique `true_source` $4$ times and the 1 unique `wrong_source` $4$ times, and re-collating/copy-aligning the negative source $G=4$ times on CPU.
2. **Duplicate forward pass when `beta > 0`:** When `beta > 0` and `teacher_checkpoint` defaults to `sft_checkpoint`, `SourceTeacher.actor` is `trainer.ref_model`. TRL already computes `ref_per_token_logps` on `true_source` inside `super()._generate_and_score_completions(inputs)`, and `SourceTeacher` immediately recomputes the identical `true_source` log-probabilities a second time.
3. **Copy-gate asymmetry on `wrong_source`:** When `grounded_copy` is enabled, tokens copied from $x^+$ have $P_{\text{copy}}(y_t \mid x^+) > 0$ on `true_source` but $P_{\text{copy}}(y_t \mid x^-) = 0$ (`log_source = -3.4e38`) on the low-Jaccard `wrong_source` $x^-$, strongly rewarding verbatim copying of rare source tokens over abstractive synthesis.
4. **Saturation at $\tau = 1.0$:** Typical per-token NLL margins $\Delta = \ell_{\text{true}} - \ell_{\text{neg}} \in [1.5, 4.0]$ nats/token yield $\tanh(\Delta / 1.0) \in [0.905, 0.9993]$, saturating the teacher score and compressing within-group variance among valid candidates.
