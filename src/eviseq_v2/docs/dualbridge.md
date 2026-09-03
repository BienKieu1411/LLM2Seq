# Dynamic evidence bridge

The dynamic evidence bridge adds a target-free prompt/source route to the
static source salience route. It still uses one source memory and one decoder.

Let (a_i) be the static salience logit for source unit (i), and let (M)
be the projected token memory. A short decoder probe produces (h_0) from the
fixed instruction. The prompt query combines (h_0) with unit-balanced pooled
memory:

\[
q=\operatorname{norm}\left(W_p h_0+\sigma(c)W_m
\frac{1}{|U|}\sum_{i\in U}M_i\right),
\qquad
k_i=\operatorname{norm}(W_k M_i).
\]

The bounded prompt score is fused with the static evidence score:

\[
d_i=\operatorname{clip}(\alpha q^\top k_i,-d_{max},d_{max}),
\qquad
\widetilde a_i=a_i+\sigma(r)d_i.
\]

The existing bridge converts the fused unit logits to a token-level additive
cross-attention bias:

\[
b_t=\sigma(g_s)s\operatorname{clip}(\widetilde a_i,-5,5)-\log n_i,
\]

where token (t) belongs to unit (i), and (n_i) is that unit's visible
token count. The final term keeps the total prior mass of a source unit
independent of its length.

## Execution path

Training performs the following sequence:

```text
source -> encoder -> static bridge
fixed instruction -> decoder probe -> prompt query
prompt query + pooled memory -> fused unit logits
fused bridge memory -> teacher-forced decoder -> training objectives
```

Evaluation performs the same probe without gradients and then uses ordinary
greedy decoding. It does not read reference summaries or sentence labels and
does not generate candidates.

The probe reuses the projected source memory. Only the configured final probe
layers perform prompt/source interaction; it does not create a second memory
bank or a second decoder.

## Configuration

```yaml
objectives:
  prompt_conditioned_inference_bridge: true
  evidence_prompt_bridge_fusion_init: 0.50
  prompt_bridge_dynamic_logit_scale: 8.0
  prompt_bridge_dynamic_logit_clip: 2.0
  prompt_bridge_dynamic_salience_mix: 0.50
  prompt_bridge_source_probe_layers: 2
```

The static route remains active in every fused logit. If the dynamic route is
disabled, the model reduces to the static evidence bridge.

## Diagnostics

Useful runtime values are the evidence top-1 accuracy, evidence similarity
gap, positive attention-prior gap, effective dynamic-bias RMS and the decoder
cross-residual ratio. The decisive task metric is the selected evaluation
metric configured in the task YAML.
