# EviSeq: Evidence-guided LLM-to-Sequence

EviSeq tests a deliberately compact paper claim: independently pretrained LLMs
can be assembled into a conventional encoder-decoder summarizer, then fully
fine-tuned on task data, without T5Gemma's trillion-token conversion stage.
The target is to match or beat T5Gemma2-1B-1B while keeping the assembled graph
below its official approximately 1.7B total footprint. The Qwen-Embedding 0.6B
to Qwen 0.6B graph is expected to be roughly 1.37B before the runtime manifest
records the exact unique parameter count.

The code-level differences from LLM2Seq-v2 and LLM2Seq-v3 are audited in
[`VERSION_COMPARISON.md`](VERSION_COMPARISON.md). Architectural advantages in
that document are hypotheses until matched B200 runs establish validation and
locked-test ROUGE.

## Architecture

The deployed graph exposes exactly one encoder and one decoder. Internally, it
contains:

1. `Qwen3-Embedding-0.6B` reused as the source encoder;
2. a small source-only evidence head inside the encoder output interface and a
   dimension projection only when encoder and decoder widths differ;
3. `Qwen3-0.6B` reused as the causal decoder, with self-initialized
   cross-attention after every self-attention layer.

Unlike V2, bidirectionality is not delegated to scratch Transformer blocks
after a causal encoder. EviSeq reuses each native Qwen layer's Q/K/V, Q/K norms,
RoPE, GQA, MLP, and output projection. The layer computes causal and full
attention views from the same Q/K/V and applies a conservatively initialized
correction. The canonical run starts with a 1% evidence-view contribution;
zero remains an available exact-causal ablation. This avoids starving the
evidence path of generation gradients at the beginning of training while
remaining close to the pretrained causal computation.
The second view is noncausal but key-selective: it adds each predicted unit
log-odds directly to the scores of that unit's source keys, while the separate
causal view keeps the original prefix, past, and EOS path:

```text
A_evidence(q,k) = A(q,k) + evidence_logit(unit(k))
                              - log(tokens_in_unit(k))
O = O_causal + tanh(head_gate) * (O_evidence - O_causal)
```

Thus moving evidence from one source unit to another changes the attended
keys even when total evidence mass is identical; the mechanism does not open
unrestricted full attention with one scalar query gate. The length correction
also prevents a long sentence from receiving more prior softmax mass merely
because it contains more subword tokens. The decoder salience bias uses the
same allocation rule. Encoder prompt and EOS tokens form one neutral
pseudo-unit, so they cannot gain extra prior mass just because their unit id is
zero.

Gold summary evidence never enters the encoder; it supervises only source-unit
logits. The same predicted source logits control the encoder correction and
softly bias decoder cross-attention. Salience supervision combines balanced
pointwise logistic loss with a small within-document pairwise ranking term,
because the logits are consumed as relative attention scores rather than only
as independent binary decisions.

The main training objective restores V3's target-free source--prompt InfoNCE:

```text
L = L_CE + 0.10 L_salience + 0.10 L_prompt-source
```

The decoder representation is taken at the final fixed prompt token, before
teacher forcing reveals a summary token. Therefore a decoder that ignores
cross-attention cannot solve the matching task. The two 256-dimensional
projection heads and the loss are training-only; generation remains exactly
one encoder and one autoregressive decoder. Source-swap, phrase,
response-alignment, and routing-balance losses are not used.

InfoNCE uses exact two-pass GradCache over each optimizer window, so its
negative set has `physical_batch * gradient_accumulation` examples (128 on the
WikiLingua main run and 32 on the CNN/DM/PubMed runs). The first pass caches
only normalized representations; the second replays each microbatch under the
same dropout RNG and propagates the cached representation gradient. This gives
the same contrastive gradient as a physical virtual batch without retaining
all encoder/decoder activations. CE and salience still use ordinary gradient
accumulation. Adam moments for the warmed-up interface are retained when the
pretrained encoder and decoder are unfrozen.

Training writes one compact JSON line every `training.log_every_steps` and at
each epoch boundary. It keeps only total/CE/salience loss, micro-averaged
salience F1, within-document salience ranking accuracy, InfoNCE loss and
retrieval accuracy, `cl_n` (the retrieval candidate count), cross-attention
residual use, the bidirectional gate, gradient norm and component learning
rates. `sal_rank` is the fraction of positive--negative source-unit pairs
ordered correctly (ties count as one half), so ranking improvements remain
visible before logits cross the fixed 0.5 threshold used by `sal_f1`.
`cl_scale` appears only while it is ramped during interface warm-up. Phrase,
source-swap, routing-bank, and response-alignment payloads are not emitted, and
a log window is never carried across epoch boundaries.

## Main method and three decisive ablations

- `c0`: native causal encoder + the same contrastive objective;
- `c2`: generic learned dual-mask conversion + contrastive;
- `c3-no-cl`: evidence-conditioned conversion without contrastive;
- `c3`: evidence-conditioned conversion + contrastive (the canonical `wiki`
  main run, not another ablation checkpoint).

This is one main run plus exactly three controlled ablations. It isolates native
conversion, evidence-conditioned routing, and contrastive coupling without a
redundant retraining of the main model. `c1` (hard full bidirectional) remains
an optional appendix diagnostic, stored outside the core ablation directory.
The attention comparisons keep the loss fixed; the `c3`/`c3-no-cl` pair
changes the training objective and removes its training-only projection head
and extra cache pass, while keeping the deployed encoder-decoder graph fixed.
Importantly, c0 and c2 still use the same supervised evidence bias in decoder
cross-attention: the ablated factor is the **encoder view**, not the whole
evidence mechanism. V2 is reported as a legacy baseline, not disguised as a
matched ablation. Ablations generate validation predictions only; the full
test split is locked behind an explicit paper-test command.

## One architecture across datasets

WikiLingua, CNN/DM, and PubMed load an identical model/bridge/decoder/objective
contract; the offline test requires the same architecture hash for all three.
Only dataset protocol and optimization budget change: prompt, sequence/output
lengths, epoch count, batch/accumulation, and the maximum number of oracle
evidence units. These values must be disclosed (`InfoNCE N=128/32/32`, total
epochs `14/6/8`, oracle cap `12/8/12`) rather than presented as architectural
changes. Every T5Gemma comparison must use the identical split, source prompt,
source/target limits, and greedy generation contract. PubMed remains a pending
target until its matched T5Gemma artifact has been produced and verified.

The T5Gemma comparison matches raw rows, split IDs, source instruction,
source/target token limits, and greedy decoding settings. It is not described
as an identical full prompt: EviSeq necessarily has a Qwen-native decoder seed,
whereas T5Gemma initializes its decoder natively. Equal numeric token limits
also do not imply identical retained raw-text spans under different
tokenizers. Training compute is disclosed rather than called matched: on
WikiLingua EviSeq uses 2 interface-only + 12 full epochs (effective batch 128),
while the locked T5Gemma result uses 6 full epochs (effective batch 32).

## Which embedding encoder is most promising?

`perplexity-ai/pplx-embed-v1-0.6b` is the strongest score-oriented alternative:
it is Qwen3-derived, width-compatible with the decoder, and already native
bidirectional. It is therefore a clean same-scale encoder replacement, but it
tests modular composition rather than the causal-to-encoder mechanism.

`nvidia/Nemotron-3-Embed-1B-BF16` may improve semantic recall, especially on
PubMed and Vietnamese retrieval, but it is 1.14B and 2048-wide. It needs a
learned 2048->1024 projection and the complete graph is expected to exceed the
approximately 1.7B T5Gemma2 footprint. Use it as an accuracy-oriented upper
control, never as evidence for the smaller-parameter claim.

## Commands

Activate `bienkieu_env`, change to `src`, then run:

```bash
bash eviseq/run.sh test
bash eviseq/run.sh smoke --overwrite-output-dir
bash eviseq/run.sh wiki --overwrite-output-dir
bash eviseq/run.sh ablation-all --overwrite-output-dir
bash eviseq/run.sh dev-table-wiki
# Optional appendix diagnostic only:
bash eviseq/run.sh c1 --overwrite-output-dir
bash eviseq/run.sh pplx --overwrite-output-dir
```

If weights already exist on the B200 server, replace `model.encoder_name` and
`model.decoder_name` in the selected YAML with their absolute local paths. The
runner contains no download, upload, or Hub-push step of its own; Transformers
resolves whatever identifiers/paths the config supplies.

Only after model/config selection on validation:

```bash
bash eviseq/run.sh paper-test-wiki
bash eviseq/run.sh rouge155 \
  runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.jsonl --details
```

`dev-table-wiki` runs the same Perl ROUGE wrapper on the four complete
validation artifacts, verifies identical ordered IDs/references, split and
prediction hashes, decoding protocol and the three declared ablation factors,
then writes both JSON and Markdown tables under `runs/eviseq/`. It refuses
test artifacts and sampled validation files.

Paper-test is one-shot per checkpoint directory. Before decoding it writes a
`paper_test_manifest.json` reservation and refuses any existing prediction,
metrics, Perl ROUGE or prior reservation artifact. The checkpoint-embedded
evaluation contract binds source preprocessing, encoder/decoder prompts,
length limits and every generation setting; editing `resolved_config.yaml`
after training therefore cannot be used to tune on test. An interrupted paper
test remains reserved and must be audited before the marker is removed by
hand.

Score the full T5Gemma prediction file with the same `rouge155` command, then
run the fail-closed artifact comparison:

```bash
bash eviseq/run.sh compare-paper \
  runs/eviseq/wikilingua_qwen3_evidence/resolved_config.yaml \
  runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.rouge155.json \
  runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.metrics.json \
  T5Gemma/eval_outputs/full_test/predictions.rouge155.json \
  T5Gemma/eval_outputs/full_test/metrics.json \
  runs/eviseq/wikilingua_qwen3_evidence/paper_comparison.json
```

This gate compares the two actual artifacts rather than subtracting a YAML
number. It requires identical test fingerprints, ordered IDs/references,
greedy decoding settings, source/target limits, Perl ROUGE protocol and
prediction hashes. It also checks exact unique resident parameters against the
actual full T5Gemma count. The older tracked LoRA T5Gemma output is therefore
deliberately rejected.

After both systems have been scored with `--details`, compute a deterministic
10,000-resample paired bootstrap interval over the official Perl per-example
scores:

```bash
bash eviseq/run.sh bootstrap \
  runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.rouge155.json \
  T5Gemma/eval_outputs/full_test/predictions.rouge155.json \
  runs/eviseq/wikilingua_qwen3_evidence/paired_bootstrap_vs_t5gemma.json
```

This command refuses mismatched or reordered IDs/references, altered
prediction/detail/raw-output hashes, different ROUGE protocols, missing rows,
and an existing output path. The aggregate Perl score remains the headline;
the paired 95% interval is the significance evidence.

For the final paper, compute Perl ROUGE on the four validation prediction files
(`c0`, `c2`, `c3-no-cl`, and main `c3`) and label that table **DEV**. Run the
full test only for frozen main systems. Direct Qwen3-0.6B full fine-tuning is an
external matched-segment control, not a fourth ablation or a native-chat Qwen
ceiling. If budget remains after the core table, run hard-full `c1` first; it
tests the destructive-conversion concern and belongs in the appendix.

Never substitute the tracked legacy LoRA T5Gemma prediction artifact for the
locked full-fine-tuned target `62.013/32.654/58.143`. The fail-closed comparison
rejects its mismatched prompt, generation settings, parameters, and score.

There is no Hub upload or push command. Checkpoints are `last.pt` only.
