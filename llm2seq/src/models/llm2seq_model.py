"""
LLM2Seq: Main Model Class.

Combines all components:
    Encoder (LLM2Vec/HF LLM) → Adaptor → Lightweight Decoder → LM Head → Optional MTP

LLM2Seq training uses:
    1. Phase 1/2 main summarizer training without MTP/KD.
    2. Phase 3 MTP-D self-distillation for speculative decoding heads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .encoder_wrapper import EncoderWrapper
from .adaptor import Adaptor
from .decoder import LightweightDecoder
from .mtp_heads import ParallelMTPHeads
from .mtp_cascaded import CascadedMTP


class LLM2SeqConfig:
    """
    Configuration container for LLM2Seq model.

    Constructed from a parsed YAML config dict.
    """

    def __init__(self, cfg: Dict[str, Any]):
        # Model
        model_cfg = cfg.get("model", {})
        self.encoder_name: str = model_cfg.get("encoder_name", "gpt2")
        self.encoder_trainable: bool = model_cfg.get("encoder_trainable", False)
        self.use_lora_for_encoder: bool = model_cfg.get("use_lora_for_encoder", False)
        self.d_enc: int = model_cfg.get("d_enc", 4096)
        self.d_dec: int = model_cfg.get("d_dec", 768)
        self.encoder_torch_dtype: str = model_cfg.get("encoder_torch_dtype", "auto")
        self.lora_r: int = model_cfg.get("lora_r", 16)
        self.lora_alpha: int = model_cfg.get("lora_alpha", 32)
        self.lora_dropout: float = model_cfg.get("lora_dropout", 0.05)
        self.lora_target_modules: List[str] = model_cfg.get(
            "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        # Adaptor
        adaptor_cfg = cfg.get("adaptor", {})
        self.adaptor_type: str = adaptor_cfg.get("type", "mlp")
        self.use_layer_fusion: bool = adaptor_cfg.get("use_layer_fusion", True)
        self.fusion_type: str = adaptor_cfg.get("fusion_type", "scalar")
        self.fuse_layers: List[int] = adaptor_cfg.get("fuse_layers", [-1, -4, -8, -12])
        self.use_encstack: bool = adaptor_cfg.get("use_encstack", False)
        self.encstack_layers: int = adaptor_cfg.get("encstack_layers", 2)
        self.use_global_memory_tokens: bool = adaptor_cfg.get("use_global_memory_tokens", False)
        self.num_global_memory_tokens: int = adaptor_cfg.get("num_global_memory_tokens", 0)
        self.use_salience_gate: bool = adaptor_cfg.get("use_salience_gate", False)

        # Small decoder
        dec_cfg = cfg.get("small_decoder", {})
        self.dec_num_layers: int = dec_cfg.get("num_layers", 6)
        self.dec_hidden_size: int = dec_cfg.get("hidden_size", 768)
        self.dec_num_heads: int = dec_cfg.get("num_heads", 12)
        self.dec_ffn_size: int = dec_cfg.get("ffn_size", 3072)
        self.dec_dropout: float = dec_cfg.get("dropout", 0.1)
        self.dec_tie_embeddings: bool = dec_cfg.get("tie_embeddings", True)
        self.dec_max_seq_len: int = dec_cfg.get("max_seq_len", 512)

        # Features
        features_cfg = cfg.get("features", {})
        self.use_mtp: bool = features_cfg.get("use_mtp", False)
        self.use_distillation: bool = features_cfg.get("use_distillation", False)

        # MTP
        mtp_cfg = cfg.get("mtp", {})
        self.mtp_type: str = mtp_cfg.get("type", "parallel")
        self.mtp_num_heads: int = mtp_cfg.get("num_heads", 4)
        self.mtp_loss_weight: float = mtp_cfg.get("loss_weight", 0.3)
        self.mtp_head_weights: List[float] = mtp_cfg.get("head_weights", [1.0, 0.8, 0.6, 0.4])
        self.mtp_train_only: bool = mtp_cfg.get("train_only", False)
        self.mtp_self_distillation: bool = mtp_cfg.get("self_distillation", False)
        self.mtp_self_distill_top_k: int = mtp_cfg.get("self_distill_top_k", 10000)
        self.mtp_self_distill_temperature: float = mtp_cfg.get("self_distill_temperature", 1.0)
        self.mtp_self_distill_loss_weight: float = mtp_cfg.get("self_distill_loss_weight", 0.5)
        self.mtp_self_distill_start_ratio: float = mtp_cfg.get("self_distill_start_ratio", 0.0)
        self.mtp_self_distill_warmup_ratio: float = mtp_cfg.get("self_distill_warmup_ratio", 0.0)
        self.mtp_self_distill_head_weights: Optional[List[float]] = mtp_cfg.get(
            "self_distill_head_weights", None
        )
        self.mtp_use_at_inference: bool = mtp_cfg.get("use_mtp_at_inference", False)
        self.mtp_inference_strategy: str = mtp_cfg.get("inference_strategy", "confidence_adaptive")
        self.mtp_confidence_threshold: float = mtp_cfg.get("confidence_threshold", 0.9)

        # Knowledge Distillation
        kd_cfg = cfg.get("knowledge_distillation", {})
        self.kd_teacher_name: Optional[str] = kd_cfg.get("teacher_name", None)
        self.kd_type: str = kd_cfg.get("type", "topk_kl")
        self.kd_temperature: float = kd_cfg.get("temperature", 2.0)
        self.kd_loss_weight: float = kd_cfg.get("loss_weight", 0.5)
        self.kd_top_k: int = kd_cfg.get("top_k", 10000)


class LLM2Seq(nn.Module):
    """
    LLM2Seq: LLM2Vec Encoder + Adaptor + Lightweight Decoder.

    End-to-end encoder-decoder model for sequence-to-sequence tasks.

    Args:
        cfg: LLM2SeqConfig — model configuration.
        vocab_size: Vocabulary size (from tokenizer).
    """

    def __init__(self, cfg: LLM2SeqConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        # 1. Encoder
        self.encoder = EncoderWrapper(
            model_name=cfg.encoder_name,
            trainable=cfg.encoder_trainable,
            use_lora=cfg.use_lora_for_encoder,
            torch_dtype=cfg.encoder_torch_dtype,
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=cfg.lora_target_modules,
        )

        # 2. Adaptor
        self.adaptor = Adaptor(
            d_enc=cfg.d_enc,
            d_dec=cfg.d_dec,
            use_layer_fusion=cfg.use_layer_fusion,
            fusion_type=cfg.fusion_type,
            fuse_layers=cfg.fuse_layers,
            projection_type=cfg.adaptor_type,
            use_encstack=cfg.use_encstack,
            encstack_layers=cfg.encstack_layers,
            encstack_heads=cfg.dec_num_heads,
            encstack_ffn=cfg.dec_ffn_size,
            dropout=cfg.dec_dropout,
            use_global_memory_tokens=cfg.use_global_memory_tokens,
            num_global_memory_tokens=cfg.num_global_memory_tokens,
            use_salience_gate=cfg.use_salience_gate,
        )

        # 3. Lightweight Decoder
        self.decoder = LightweightDecoder(
            vocab_size=vocab_size,
            hidden_size=cfg.dec_hidden_size,
            num_layers=cfg.dec_num_layers,
            num_heads=cfg.dec_num_heads,
            ffn_size=cfg.dec_ffn_size,
            max_seq_len=cfg.dec_max_seq_len,
            dropout=cfg.dec_dropout,
            tie_embeddings=cfg.dec_tie_embeddings,
        )

        # 4. LM Head
        self.lm_head = nn.Linear(cfg.dec_hidden_size, vocab_size, bias=False)

        # Tie embeddings: share decoder embedding ↔ LM head weight
        if cfg.dec_tie_embeddings:
            self.lm_head.weight = self.decoder.embed_tokens.weight

        # 5. MTP Module (optional)
        self.mtp_module: Optional[nn.Module] = None
        if cfg.use_mtp:
            if cfg.mtp_type == "parallel":
                self.mtp_module = ParallelMTPHeads(
                    hidden_size=cfg.dec_hidden_size,
                    vocab_size=vocab_size,
                    num_heads=cfg.mtp_num_heads,
                )
            elif cfg.mtp_type == "cascaded":
                self.mtp_module = CascadedMTP(
                    hidden_size=cfg.dec_hidden_size,
                    vocab_size=vocab_size,
                    num_heads=cfg.mtp_num_heads,
                    attn_heads=cfg.dec_num_heads,
                    ffn_size=cfg.dec_ffn_size,
                    dropout=cfg.dec_dropout,
                    embedding_layer=self.decoder.embed_tokens,
                    lm_head=self.lm_head,
                )
            else:
                raise ValueError(f"Unknown MTP type: {cfg.mtp_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_topk_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass.

        Args:
            input_ids: [B, S_src] — source token IDs.
            attention_mask: [B, S_src] — source attention mask.
            decoder_input_ids: [B, S_tgt] — decoder input token IDs (shifted targets).
            decoder_attention_mask: [B, S_tgt] — decoder attention mask.
            labels: [B, S_tgt] — target token IDs for loss computation.
            teacher_logits: [B, S_tgt, K_or_V] — teacher logits for KD.
            teacher_topk_indices: [B, S_tgt, K] — teacher top-k indices for KD.

        Returns:
            Dict with:
                - "logits": [B, S_tgt, vocab_size]
                - "loss": total loss (if labels provided)
                - "loss_ce": CE loss component
                - "loss_kd": KD loss component (if distillation enabled)
                - "loss_mtp": MTP loss component (if MTP enabled)
                - "mtp_logits": list of MTP logits (if MTP enabled)
        """
        # 1. Encode source
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.cfg.use_layer_fusion,
        )

        # 2. Adapt encoder output to decoder memory
        adaptor_output = self.adaptor(encoder_output, attention_mask=attention_mask)
        if isinstance(adaptor_output, tuple):
            h_mem, memory_attention_mask = adaptor_output
        else:
            h_mem = adaptor_output
            memory_attention_mask = attention_mask

        # 3. Decode
        decoder_states, _ = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=h_mem,
            encoder_attention_mask=memory_attention_mask,
            attention_mask=decoder_attention_mask,
        )

        # 4. LM Head
        main_logits = self.lm_head(decoder_states)  # [B, S_tgt, vocab_size]

        # 5. Optional MTP
        mtp_logits = None
        if self.cfg.use_mtp and self.mtp_module is not None:
            mtp_logits = self.mtp_module(
                decoder_states=decoder_states,
                decoder_input_ids=decoder_input_ids,
            )

        # 6. Compute losses
        result: Dict[str, Any] = {"logits": main_logits}

        if labels is not None:
            from ..training.losses import compute_total_loss

            loss_dict = compute_total_loss(
                main_logits=main_logits,
                labels=labels,
                mtp_logits=mtp_logits,
                teacher_logits=teacher_logits,
                teacher_topk_indices=teacher_topk_indices,
                cfg=self.cfg,
            )
            result.update(loss_dict)

        if mtp_logits is not None:
            result["mtp_logits"] = mtp_logits

        return result

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention_mask: bool = False,
    ) -> Any:
        """
        Encode source and produce decoder memory (for inference).

        Returns:
            h_mem: [B, S_src, d_dec]
        """
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.cfg.use_layer_fusion,
        )
        adaptor_output = self.adaptor(encoder_output, attention_mask=attention_mask)
        if isinstance(adaptor_output, tuple):
            h_mem, memory_attention_mask = adaptor_output
        else:
            h_mem = adaptor_output
            memory_attention_mask = attention_mask
        if return_attention_mask:
            return h_mem, memory_attention_mask
        return h_mem

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.embed_tokens.weight.numel()
        return n_params

    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a human-readable model summary."""
        total = self.get_num_params()
        trainable = self.get_trainable_params()
        frozen = total - trainable
        lines = [
            f"LLM2Seq Model Summary",
            f"{'='*50}",
            f"Encoder:          {self.cfg.encoder_name}",
            f"  Trainable:      {self.cfg.encoder_trainable}",
            f"Adaptor:          {self.cfg.adaptor_type}",
            f"  Layer Fusion:   {self.cfg.use_layer_fusion}",
            f"  Fusion Type:    {self.cfg.fusion_type}",
            f"  Salience Gate:  {self.cfg.use_salience_gate}",
            f"  EncStack:       {self.cfg.use_encstack}",
            f"  Global Tokens:  {self.cfg.num_global_memory_tokens if self.cfg.use_global_memory_tokens else 0}",
            f"Decoder:",
            f"  Layers:         {self.cfg.dec_num_layers}",
            f"  Hidden Size:    {self.cfg.dec_hidden_size}",
            f"  Heads:          {self.cfg.dec_num_heads}",
            f"  FFN Size:       {self.cfg.dec_ffn_size}",
        ]
        feature_lines = []
        if self.cfg.use_mtp:
            feature_lines.extend([
                f"  MTP:            True ({self.cfg.mtp_type})",
                f"  MTP Train Only: {self.cfg.mtp_train_only}",
            ])
            if self.cfg.mtp_self_distillation:
                feature_lines.append("  MTP Self-Dist:  True")
        if self.cfg.use_distillation:
            feature_lines.append(f"  Distillation:   True ({self.cfg.kd_type})")
        if feature_lines:
            lines.append("Features:")
            lines.extend(feature_lines)
        lines.extend([
            f"{'='*50}",
            f"Total params:     {total:,}",
            f"Trainable params: {trainable:,}",
            f"Frozen params:    {frozen:,}",
        ])
        if self.cfg.use_lora_for_encoder:
            lines.insert(4, "  LoRA:           True")
        return "\n".join(lines)
