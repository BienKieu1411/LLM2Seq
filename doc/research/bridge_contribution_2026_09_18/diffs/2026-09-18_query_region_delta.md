# Delta: decoder-conditioned regional bridge candidate

This is an update to `plan.md` and `third_pass_audit.md`, not a replacement for the earlier evidence ledger. Decision: test a new bridge **in `src/afmr_query_regions`**, leaving `src/eviseq_new` as the static AFMR control. No new ROUGE or full-checkpoint measurement is available locally.

## Falsifiable hypotheses

1. **H5:** The static `[B,S]` prior is too coarse to distinguish objective, methods, results and conclusion positions in a PubMed abstract. A decoder-conditioned regional read will change semantic source allocation at different timesteps. Falsify if trained region attention/gate is flat or disabling it leaves CE and paired predictions unchanged.
2. **H6:** A bounded residual to the semantic cross-attention query can influence content selection while retaining full-source token attention, `H0` values and grounded-copy keys. Falsify if ROUGE-1/L regress or copy/source precision worsens versus the matched static AFMR run.
3. **H7:** Region SDPA over ~63 windows at source length 4096 will be materially cheaper than dense `[B,T,S]` dynamic bias or a second full token cross-attention. Falsify with measured B200 throughput/VRAM; asymptotic cost alone is insufficient.

## Evidence and opposition

- [TopDownFormer S01](../sources/s01_topdownformer.md) provides a PubMed ablation for coarse-to-token information transfer, but its intervention is encoder-side and its checkpoint selection differs.
- [SEASON S02](../sources/s02_season.md) motivates preserving original values when adding selection guidance to keys, but uses static sentence salience and gold guidance in generator training.
- [DYLE S10](../sources/s10_dyle.md) conditions source weights on decoder history. Its arXiv cross-system result shows this design family has no universal ROUGE gain.
- [Fast Evidence S14](../sources/s14_fast_evidence.md) supports query/source fusion in evidence extraction, not a summarization score.
- [Latent Queries S15](../sources/s15_latent_queries.md) shows focused and original document views already have precedent, limiting novelty claims.
- [PROM S05](../sources/s05_prom.md) and [CODI S06](../sources/s06_codi_spancopy.md) warn against assuming a stronger source/copy signal raises every ROUGE metric.

These academic sources are independent but largely one source type and none test AFMR/PPLX/Qwen. The thesis “new bridge beats static AFMR and plain projection by at least 0.30 on R-1/R-2/R-L” remains **insufficient evidence** until matched training and evaluation.

## Decision and trade-off

Use masked overlapping 128-token windows with stride 64, keys pooled from AFMR `M`, values pooled from `H0`. Each decoder layer uses a separate 128-rank region SDPA read from its current hidden state; a zero-initialized, bounded output residual adjusts the query before the existing token cross-attention. The original source tokens remain accessible. Keep the old static prior and grounded-copy path initially, so `C−B` isolates the regional read and no new objective is confounded with it. Do not apply the new signal directly to copy logits until the semantic route is measured.

Implementation audit: the scalar gate by itself did **not** bound a growing output projection, so the residual now uses a smooth cap on its actual per-token RMS relative to the decoder query. A BF16 inference test also exposed that newly created region modules initially remained FP32 while the copied decoder was BF16; construction now matches the copied Q projection's dtype/device. These are code corrections, not ROUGE evidence. Local tiny-model checks cover the first/second CE gradient update, FP32 master gradients under BF16 autocast, exact zero-init parity, mask safety and generation-cache parity. The GPU/DDP path and long-document cost still need server verification.

Rejected alternatives for this first experiment: (i) a source-only vector residual to `M`, because existing AFMR already has source-only depth/feature residuals; (ii) dense per-timestep `[B,T,S]` attention bias, which can be large at batch 48–84 and may change SDPA kernel selection; (iii) a two-pass full-source mixture, which roughly doubles token cross-attention. The regional read still adds one attention operation per decoder layer and may be redundant with QK; the zero output delays gradients to its internal projections by one update. A tiny initialization with nonzero RMS is a follow-up only if measured route dormancy persists.

## Clean comparison

- **A:** plain projection, CE only, grounded copy on.
- **B:** static AFMR bridge, CE only, grounded copy on, region query off.
- **C:** identical B plus region query on, CE only.

Use the same PPLX/Qwen local checkpoints, split and preprocessing, prompt, max lengths, batch×accumulation, optimizer, update count, gradient clip, seed, `last.pt`, greedy decoding and ROUGE-1.5.5. Inspect the resolved config of each run. Validation is used to choose; test is locked until then. `C−B` tests the new route; `B−A` tests existing AFMR; `C−A` is the complete bridge effect. Tiny-model gradient/cache parity tests establish implementation correctness, **not** ROUGE gains. Track gate, region-attention entropy, copy mass, time/VRAM, and paired output changes.
