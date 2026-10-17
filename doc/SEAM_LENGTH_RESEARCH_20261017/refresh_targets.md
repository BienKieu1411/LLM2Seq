# Refresh and implementation gates

Research snapshot: 2026-10-17.

1. Before code: recheck fingerprints of controller/collator/GRPO/vLLM; preserve existing uncommitted work. Dynamic `target_words` is not supported end-to-end in this snapshot.
2. Before training: freeze the word-count definition; measure local decoder token/word distribution on train; inspect gold truncation and cap feasibility for long requests.
3. Inspect reference TXT blocks: report how many are multi-reference. Do not silently equate their combined length with a single-summary request.
4. Revisit H1 after prompt+content/length GRPO validation, H2 after numeric-bridge ablation, H4 only if length-bin diagnostics justify sampling changes. Record failed hypotheses too.
5. If adding a module: version checkpoint architecture, update complete actor sync, cache invalidation, request identity and policy likelihood tests before any server run.
6. Track current pinned TRL 1.16.0 and vLLM 0.25.1 APIs from their installed/source versions when implementing; research notes are not a promise that a future API matches.
7. Recent method follow-up: inspect full LARFT and GR3 algorithms if implementing their objectives; current extraction of their method is abstract-level. Check HVO publication status and full method before adopting it. Do not label a Gaussian gate as GR3.
8. Acceptance data: ROUGE-1.5.5 and legacy MAE thresholds per dataset, requested-vs-reference length errors, cap/EOS rates, native/vLLM parity, 1-/2-GPU behavior. No test-based tuning.

Do not refresh by rerunning all model experiments. First identify changed code, source metadata or assumptions; then update the affected finding and preserve the old result.

Dual-prompt update: use `2026-10-17_dual_prompt_update.md` as the current design. Before implementation, close per-record encoder-prefix offsets, common visible source spans, dynamic decoder prompt widths and group-level length-cost cancellation diagnostics. Numeric word conditioning is not a prerequisite for the first dual-prompt ablation.

Reward update: `2026-10-17_reward_update.md` adds MAE/B and a gated teacher diagnostic proposal. Before enabling teacher reward: verify candidate/source-swap rankings, correct copy-state rebuilding, frozen temperature-1 scoring, negative-source mapping, memory cost, and teacher/actor separation. Do not interpret margin as fact-level support or automatic proof of length-budgeted coverage.
