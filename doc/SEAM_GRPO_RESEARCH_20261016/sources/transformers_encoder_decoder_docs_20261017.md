# Source note: Transformers encoder-decoder forward contract

- **Checked:** 2026-10-17.
- **Source type:** Official Hugging Face Transformers model documentation.
- **URL:** [Encoder-decoder model documentation](https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder).
- **Credibility:** High for the documented model-input contract.
- **Recency:** Current rendered documentation at check date; it is versioned separately from the TRL pin and may change.
- **Bias:** Maintainer documentation; the local AFMR forward signature was inspected independently.

## Evidence used

The documented encoder-decoder forward separates encoder-side `input_ids` and `attention_mask` from `decoder_input_ids` and `decoder_attention_mask`; it also accepts `encoder_outputs` and target labels. Generation consumes encoder input and returns decoder output. This supports the architectural distinction: an encoder-decoder policy cannot be scored by concatenating decoder prompt and completion into a single causal `input_ids` stream.

## Direct excerpt

> “`decoder_input_ids` are automatically created by the model by shifting the `labels` to the right”
