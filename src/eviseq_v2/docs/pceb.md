# Prompt-Conditioned Evidence Bridge (PCEB)

## Question and hypothesis

The legacy EviSeq evidence objective pools decoder states that were produced
under teacher forcing.  For a source document `x` and reference summary `y`,
its query is effectively:

\[
q_{\mathrm{TF}} = f_\theta(x, y^*_{<t}).
\]

Greedy decoding, however, conditions on a fixed instruction followed by its
own tokens.  A source/evidence representation that is easy to separate under
`q_TF` is therefore not necessarily useful at the first generated token.

PCEB tests one falsifiable hypothesis: **evidence supervision is more useful
when its query is available before generation starts and directly updates the
salience logits that form the bridge's cross-attention prior.**

## Method

Let `h_p` be the Qwen decoder state at the final fixed prompt token, i.e. the
state that predicts the first summary token.  It is target-free but task-aware.
To avoid a document-agnostic query from a shared instruction, PCEB also pools
the valid source-token memory after the bridge, `m_x`:

\[
m_x = \operatorname{MeanPool}(M_x), \qquad
q_x = \operatorname{norm}\!\left(W_p h_p + \sigma(g) W_m m_x\right).
\]

Each visible source sentence has a bridge-memory key:

\[
k_i = \operatorname{norm}\!\left(W_k\operatorname{MeanPool}(M_{x,i})\right).
\]

For up to three greedily selected global evidence sentences `P(x)` and four
same-document hard negatives `N_H(x)`, the final evidence score is

\[
z_i = q_x^\top k_i/\tau + \beta a_i,
\]

where `a_i` is the predicted salience logit.  The loss is one-vs-negative-set
multi-positive InfoNCE:

\[
\mathcal L_{\mathrm{PCE}}
= \frac{1}{|P(x)|}\sum_{p\in P(x)}
  \operatorname{softplus}
  \left(\log\sum_{n\in N_H(x)}e^{z_n}-z_p\right).
\]

The full objective is

\[
\mathcal L = \mathcal L_{\mathrm{CE}}
+0.10\mathcal L_{\mathrm{salience}}
+0.10\mathcal L_{\mathrm{PCE}}.
\]

Hard-negative *selection* is detached, but the final `q·k` score and the
salience-logit term are differentiable.  Thus gradients reach the prompt-side
decoder state, PPLX encoder, bridge projection, evidence heads, and the exact
unit logits converted to the decoder's attention bias.

The inference graph is unchanged:

```text
PPLX encoder -> one EvidenceBridge -> Qwen3 decoder -> greedy generation
```

The query/head is training-only.  It receives no reference-summary state and
adds neither a reranker, a second model, candidates, nor an extra decoding
pass.

## Locked main-run settings

These settings are fixed **before** inspecting a test result.  They are the
main PCEB recipe in `configs/models/pplx_pubmed_pceb.yaml`, rather than a
test-set search.

| Setting | Main value | Reason |
|---|---:|---|
| Global evidence positives | at most 3 | Keeps the bridge selective.  A larger set increasingly turns pseudo-extractive supervision into a broad ``attend everywhere'' target. |
| Same-document hard negatives | 4 in warm-up, 8 in full tuning | The copied cross-attention and bridge first align against four negatives; full tuning then uses the prior PubMed recipe's 4 → 8 curriculum.  Eight is still far below large candidate queues that would amplify pseudo-label false negatives. |
| Contrastive temperature `τ` | 0.07 | A widely used contrastive starting point; it is retained unless validation shows systematic saturation or collapse. |
| Evidence-loss weight | 0.10 | Keeps the observed auxiliary-to-CE scale near one tenth instead of allowing pseudo labels to dominate generation CE. |
| Salience-loss weight | 0.10 | Preserves direct learning of the bridge's unit logits. |
| Direct salience score coefficient `β` | 0.15 | Matches the earlier successful PubMed evidence recipe.  PCEB has one global evidence set, so this direct gradient has no cross-summary-sentence sign conflict. |
| Prompt/source mixture gate initial value | 0.50 | Starts neither prompt-only nor source-only; the learned value is logged. |
| Bridge salience gate | sigmoid, initial value 0.10 | Lets positive salience strengthen the source prior without ever reversing it; unit-invariant normalization removes a hidden long-sentence preference. |
| Decoder cross gate | 0.10 | The stable starting branch strength.  A larger gate is a separately pre-registered ablation, not bundled into the main result. |
| Training budget | 1 interface warm-up + 4 full epochs | Warm-up aligns the copied cross-attention/bridge; all LLM weights update only in the four full epochs. |
| Effective batch size | 32 | Matches the PubMed T5Gemma optimization batch.  Increasing physical batch does not create additional PCE hard negatives, which are deliberately within-document. |

QFCL uses `τ=0.07`, but its 64 hard negatives and 4096-item queue are
generated medical-question candidates rather than imperfect source-sentence
pseudo labels.  Copying that negative count here would raise false-negative
pressure without making the bridge more informative.  Temperature is also a
known sensitive InfoNCE parameter, so any sensitivity run is restricted to
`τ ∈ {0.05, 0.07, 0.10}` on validation only.

The only first-order sensitivity runs worth spending compute on are one at a
time: global positives `{3, 4}`, hard negatives `{4, 8}`, evidence weight
`{0.05, 0.10, 0.15}`, or salience coefficient `β ∈ {0, 0.10, 0.15, 0.20}`.  Do not
combine them, and do not choose among them from test ROUGE.

## What is established and what is new

PCEB is **not** claimed to invent query-evidence contrastive learning or
prefix representations.

- Caciularu et al. train an inference-available question representation to
  discriminate evidence sentences from negatives in long-context QA:
  [Long Context QA via Supervised Contrastive Learning](https://aclanthology.org/2022.naacl-main.207/).
- RankGen maps a generation prefix close to its gold continuation and away
  from negative continuations using contrastive learning:
  [RankGen](https://aclanthology.org/2022.emnlp-main.15/).
- GECSum identifies teacher-forcing versus autoregressive generation as a
  central mismatch for contrastive summarization objectives:
  [GECSum](https://aclanthology.org/2024.lrec-main.670/).

The EviSeq contribution under test is their combination at the
decoder-only-to-encoder--decoder interface: an inference-compatible,
prompt-and-source-conditioned evidence query that supervises the *single
bridge* used by greedy summarization.

## Paper-valid experiment

Use the exact same source serialization, 4096/512 limits, full-fine-tuning
epoch count, optimizer hyperparameters, checkpoint rule, greedy decoding
(`do_sample=false`, one beam), and Perl ROUGE-1.5.5 for every run.  The PCEB
recipe deliberately has one additional *frozen-backbone interface warm-up*
epoch; report that extra alignment cost rather than describing total training
passes as identical to T5Gemma.

The minimum ablation table is:

| Run | Query | Positive set | Direct salience coupling |
|---|---|---|---|
| CE + salience | none | same global top-3 | no evidence CL |
| Teacher-forced EviSeq | pooled gold-summary decoder state | same global top-3 | no |
| PCEB w/o coupling | prompt + bridge-memory state | same global top-3 | `β=0` |
| PCEB | prompt + bridge-memory state | same global top-3 | `β=0.15` |

Choose the checkpoint using validation CE (and inspect validation greedy
ROUGE-2); run the test set once for the locked recipe.  Report ROUGE-1,
ROUGE-2, ROUGE-L, generated/reference length ratio, repetition rate,
evidence top-1 accuracy, evidence similarity gap, and the bridge attention
prior gap.  Compare final test predictions by paired bootstrap under the
same Perl scorer.

## Failure conditions

Do not claim PCEB works merely because contrastive loss falls.  Reject or
revise it if any of these occur:

1. `prompt_context_gate` collapses near zero and PCEB behaves as prompt-only;
2. evidence top-1/gap improve but validation greedy ROUGE-2 does not;
3. salience predicted-positive rate collapses or generation becomes more
   extractive/repetitive;
4. the length ratio changes enough to explain the ROUGE change.
