from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
)


@dataclass
class AdaptationConfig:
    encoder_model: str
    decoder_model: Optional[str] = None
    output_dir: str = ""
    max_length: int = 1024
    decoder_start_token_id: Optional[int] = None
    tied_embeddings: bool = False
    cross_attn_init: str = "from_self_attention" # "from_self_attention" or "random"
    warmup_steps: int = 0
    
    def __post_init__(self):
        if self.decoder_model is None:
            self.decoder_model = self.encoder_model


def initialize_cross_attention_from_self_attention(model: EncoderDecoderModel) -> int:
    """
    Initialize decoder cross-attention weights from decoder self-attention weights
    when module shapes are compatible. Returns number of decoder blocks initialized.
    """
    initialized = 0
    decoder_layers = getattr(model.decoder, "model", None)
    if decoder_layers is None:
        return initialized

    layers = getattr(decoder_layers, "layers", None)
    if layers is None:
        return initialized

    with torch.no_grad():
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            cross_attn = getattr(layer, "encoder_attn", None)
            if self_attn is None or cross_attn is None:
                continue
            try:
                cross_attn.load_state_dict(self_attn.state_dict(), strict=False)
                initialized += 1
            except Exception:
                continue
    return initialized


def build_encoder_decoder_from_causal_lm(cfg: AdaptationConfig) -> EncoderDecoderModel:
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc_causal_lm = AutoModelForCausalLM.from_pretrained(cfg.encoder_model)
    if cfg.encoder_model == cfg.decoder_model:
        dec_causal_lm = enc_causal_lm
    else:
        dec_causal_lm = AutoModelForCausalLM.from_pretrained(cfg.decoder_model)

    enc_dec_config = EncoderDecoderConfig.from_encoder_decoder_configs(
        enc_causal_lm.config,
        dec_causal_lm.config,
    )
    enc_dec_config.encoder.is_decoder = False
    enc_dec_config.encoder.add_cross_attention = False
    enc_dec_config.decoder.is_decoder = True
    enc_dec_config.decoder.add_cross_attention = True
    enc_dec_config.tie_encoder_decoder = cfg.tied_embeddings
    
    # Enable tie_word_embeddings for the overall model
    enc_dec_config.tie_word_embeddings = cfg.tied_embeddings

    enc_dec_config.decoder_start_token_id = (
        cfg.decoder_start_token_id
        if cfg.decoder_start_token_id is not None
        else tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )
    enc_dec_config.pad_token_id = tokenizer.pad_token_id
    enc_dec_config.eos_token_id = tokenizer.eos_token_id
    enc_dec_config.max_length = cfg.max_length

    model = EncoderDecoderModel(config=enc_dec_config)

    with torch.no_grad():
        model.encoder.load_state_dict(enc_causal_lm.model.state_dict(), strict=False)
        model.decoder.load_state_dict(dec_causal_lm.model.state_dict(), strict=False)
        
        # Load lm_head from decoder model
        if hasattr(dec_causal_lm, "lm_head") and hasattr(model, "lm_head"):
            model.lm_head.load_state_dict(dec_causal_lm.lm_head.state_dict(), strict=False)
            
        if cfg.cross_attn_init == "from_self_attention":
            n = initialize_cross_attention_from_self_attention(model)
            print(f"Initialized cross-attention from self-attention in {n} decoder layers.")
        else:
            print("Cross-attention initialized randomly (default behavior).")

    # Tie embeddings if requested
    if cfg.tied_embeddings:
        print("Tying encoder and decoder embeddings.")
        model.encoder.set_input_embeddings(model.decoder.get_input_embeddings())
        model.tie_weights()

    model.config.vocab_size = model.config.decoder.vocab_size
    return model, tokenizer


def save_adapted_model(cfg: AdaptationConfig) -> None:
    model, tokenizer = build_encoder_decoder_from_causal_lm(cfg)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
