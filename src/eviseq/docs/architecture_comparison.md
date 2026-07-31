# EviSeq versus LLM2Seq-v2 and LLM2Seq-v3

This comparison is based on the implementations and canonical YAML files.
Matched validation and test runs determine whether it is a stronger model.

| Component | LLM2Seq-v2 | LLM2Seq-v3 main | EviSeq |
|---|---|---|---|
| Bidirectional conversion | Fuses selected encoder depths, then applies 4 newly initialized full-attention Transformer blocks | Keeps the post-encoder conversion and expands the main HiRoute profile to 8 blocks | Reuses native Q/K/V, norms, RoPE, GQA, MLP, and output projection; only the top 6 Qwen blocks add an evidence-aware noncausal view |
| Noncausal view | Unrestricted full attention in the adapter | Unrestricted full attention in the summary adapter; other banks receive projected/context-broadcast states | Evidence-conditioned noncausal attention over source keys, mixed with the causal view through per-layer, per-head gates |
| Content selection | Salience predicts an auxiliary decoder cross-attention bias | Salience plus three routed memory banks and source-utilization objectives | One predicted evidence distribution controls both encoder noncausal key bias and decoder cross-attention bias |
| Decoder memory | One full token memory in the base model | Three full-length banks in the canonical HiRoute model | One full token memory |
| Cross-attention | Copied from decoder self-attention in every decoder layer | Same, plus per-layer/query bank routing | Same copied cross-attention, directly biased by the shared predicted evidence |
| Contrastive coupling | None | Target-free prompt/source InfoNCE over each physical microbatch, plus source-swap ranking | Multi-positive evidence/summary InfoNCE with four hard negatives mined inside the same document; no virtual-batch GradCache |
| Main loss | CE + 0.10 salience | CE + 0.10 salience + 0.05 InfoNCE + 0.10 source-swap + label smoothing; HiRoute also uses routing balance | CE + 0.10 salience + 0.10 InfoNCE |
| Initialization | New adapter blocks and cross-attention must warm up | More new adapter/routing capacity must warm up | Selective native attention gates start at 0.01 and cross-attention is copied from decoder self-attention before interface warm-up |
| Deployed graph | One encoder, adapter, decoder | One encoder, multi-bank adapter/router, decoder | Exactly one encoder, a small evidence interface, and one decoder; contrastive heads are training-only |
| Core ablations | Five registered adapter ablations | Many interacting HiRoute/objective ablations | Three decisive controls: causal+CL, generic dual-mask+CL, evidence dual-mask without CL |

## What EviSeq genuinely improves

1. **It changes the encoder itself instead of repairing only its final hidden
   states.** V2/V3 add newly initialized bidirectional blocks after the
   pretrained encoder. EviSeq computes causal and evidence-conditioned views
   in six selected top layers and reuses the pretrained projections and MLP.
   This is the central architectural novelty.

2. **Evidence is causally connected to generation through two paths.** The
   source-only evidence head changes which future source keys the encoder can
   use and also changes decoder cross-attention scores. In V2, salience mainly
   biases the decoder-side memory path; in V3, source specialization is split
   across routing banks and several objectives.

3. **It removes HiRoute's shortcut and collapse surface.** EviSeq has no
   lexical/semantic/summary router and no threefold full-length memory tensor.
   The decoder cannot bypass the evidence-conditioned encoder by selecting a
   raw causal bank. This also makes the ablation story much easier to defend.

4. **Contrastive learning is summarization-specific.** The old document
   retrieval task used other documents as easy negatives and could reach
   perfect accuracy without improving evidence selection. EviSeq instead
   contrasts the teacher-forced summary representation with all oracle evidence
   sentences and the hardest non-evidence sentences from the same document.
   Hard mining is detached, while the selected similarities remain fully
   differentiable through the encoder and decoder.

5. **The paper claim is narrower and testable.** The main experiment has one
   architecture and three controlled ablations, uses one memory bank across
   WikiLingua, CNN/DailyMail, and PubMed, and saves only `last.pt`. This separates encoder conversion, evidence guidance,
   and contrastive coupling without a large grid of coupled mechanisms.

## What is not yet proven

- EviSeq has not yet demonstrated higher ROUGE than V2, V3, or T5Gemma. Its
  improvement is currently architectural and experimental, not empirical.
- Native dual-view attention still evaluates two attention paths in six Qwen
  layers. It is cheaper than converting every layer, but not computationally
  free.
- Against the recorded V2 Perl ROUGE result (58.392/27.645/54.292), matching
  the T5Gemma target (62.013/32.654/58.143) still requires approximately
  +3.621 ROUGE-1, +5.009 ROUGE-2, and +3.851 ROUGE-L.
- The decisive evidence is the matched `c0`, `c2`, `c3-no-cl`, and `c3`
  validation table followed by full-test evaluation. If `c3` does
  not beat the controls, the native evidence mechanism is not supported even
  if one isolated test score is high.
