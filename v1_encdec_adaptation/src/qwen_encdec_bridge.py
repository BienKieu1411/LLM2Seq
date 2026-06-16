from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput


@dataclass
class QwenEncDecBridgeConfig:
    encoder_model: str
    decoder_model: Optional[str] = None
    output_dir: str = ""
    max_source_length: int = 1024
    max_target_length: int = 256
    tied_embeddings: bool = True
    bridge_layers: int = 1
    share_backbone_weights: bool = True

    def __post_init__(self):
        if self.decoder_model is None:
            self.decoder_model = self.encoder_model


class CrossAttentionBridge(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, query_states: torch.Tensor, encoder_states: torch.Tensor, encoder_mask: Optional[torch.Tensor] = None):
        key_padding_mask = None
        if encoder_mask is not None:
            key_padding_mask = encoder_mask == 0

        x = query_states
        for attn, norm in zip(self.layers, self.norms):
            attn_out, _ = attn(x, encoder_states, encoder_states, key_padding_mask=key_padding_mask, need_weights=False)
            x = norm(x + attn_out)
        return x


class QwenEncDecBridgeModel(nn.Module):
    def __init__(self, encoder_lm: nn.Module, decoder_lm: nn.Module, pad_token_id: int, eos_token_id: int, bridge_layers: int = 1):
        super().__init__()
        self.encoder_lm = encoder_lm
        self.decoder_lm = decoder_lm

        self.encoder = getattr(encoder_lm, "model", encoder_lm)
        self.decoder = getattr(decoder_lm, "model", decoder_lm)

        hidden_size = self.decoder.config.hidden_size
        num_heads = self.decoder.config.num_attention_heads
        self.bridge = CrossAttentionBridge(hidden_size=hidden_size, num_heads=num_heads, num_layers=bridge_layers)

        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.config = self.decoder_lm.config
        self.config.pad_token_id = pad_token_id
        self.config.eos_token_id = eos_token_id
        self._gc_enabled = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._gc_enabled = True
        if hasattr(self.encoder_lm, "gradient_checkpointing_enable"):
            try:
                self.encoder_lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.encoder_lm.gradient_checkpointing_enable()
        # Avoid double-call if encoder and decoder share the same object.
        if self.decoder_lm is not self.encoder_lm and hasattr(self.decoder_lm, "gradient_checkpointing_enable"):
            try:
                self.decoder_lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.decoder_lm.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self._gc_enabled = False
        if hasattr(self.encoder_lm, "gradient_checkpointing_disable"):
            self.encoder_lm.gradient_checkpointing_disable()
        if self.decoder_lm is not self.encoder_lm and hasattr(self.decoder_lm, "gradient_checkpointing_disable"):
            self.decoder_lm.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return self._gc_enabled

    def freeze_backbones(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        if hasattr(self.decoder_lm, "lm_head"):
            for p in self.decoder_lm.lm_head.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def shift_right(self, labels: torch.Tensor):
        shifted = torch.full_like(labels, fill_value=self.pad_token_id)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0] = self.eos_token_id
        shifted = shifted.masked_fill(shifted == -100, self.pad_token_id)
        return shifted

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        encoder_hidden = enc_out.last_hidden_state

        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Need either decoder_input_ids or labels")
            decoder_input_ids = self.shift_right(labels)

        if decoder_attention_mask is None:
            decoder_attention_mask = (decoder_input_ids != self.pad_token_id).long()

        dec_embed = self.decoder_lm.get_input_embeddings()(decoder_input_ids)
        bridge_first_param = next(self.bridge.parameters(), None)
        bridge_dtype = bridge_first_param.dtype if bridge_first_param is not None else dec_embed.dtype
        dec_embed_bridge = dec_embed.to(bridge_dtype)
        encoder_hidden_bridge = encoder_hidden.to(bridge_dtype)
        bridged_embed = self.bridge(dec_embed_bridge, encoder_hidden_bridge, attention_mask)
        dec_dtype = self.decoder_lm.get_input_embeddings().weight.dtype
        bridged_embed = bridged_embed.to(dec_dtype)

        dec_out = self.decoder_lm(
            inputs_embeds=bridged_embed,
            attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        return Seq2SeqLMOutput(
            loss=dec_out.loss,
            logits=dec_out.logits,
            encoder_last_hidden_state=encoder_hidden,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ):
        if num_beams != 1:
            raise ValueError("This lightweight bridge currently supports greedy decoding only (num_beams=1).")

        bsz = input_ids.size(0)
        device = input_ids.device
        generated = torch.full((bsz, 1), self.eos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=generated,
            )
            next_token_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == self.eos_token_id):
                break

        return generated[:, 1:]


def build_qwen_encdec_bridge(cfg: QwenEncDecBridgeConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder_lm = AutoModelForCausalLM.from_pretrained(cfg.encoder_model)
    if cfg.decoder_model == cfg.encoder_model and cfg.share_backbone_weights:
        decoder_lm = encoder_lm
    else:
        decoder_lm = AutoModelForCausalLM.from_pretrained(cfg.decoder_model)

    encoder_lm = encoder_lm.float()
    if decoder_lm is not encoder_lm:
        decoder_lm = decoder_lm.float()

    model = QwenEncDecBridgeModel(
        encoder_lm=encoder_lm,
        decoder_lm=decoder_lm,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bridge_layers=cfg.bridge_layers,
    )

    if cfg.tied_embeddings:
        model.encoder.set_input_embeddings(model.decoder_lm.get_input_embeddings())

    return model, tokenizer
