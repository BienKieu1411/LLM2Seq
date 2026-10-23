# S02 — T5Gemma 2 Transformers documentation

- Publisher: Hugging Face Transformers.
- Link: https://huggingface.co/docs/transformers/model_doc/t5gemma2
- Source role: official model/library API documentation.

The documented conditional-generation forward path includes `decoder_input_ids`, `decoder_attention_mask`, `encoder_outputs`, `past_key_values` typed as an `EncoderDecoderCache`, and `use_cache`. This confirms the intended cache interface. It does not prove that the local adapter supplies equivalent masks/positions or that cached block logits match serial cached logits; the local target must pass the new diagnostic on the exact runtime.
