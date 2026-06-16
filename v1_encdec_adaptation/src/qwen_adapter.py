import torch
import torch.nn as nn
import math
import inspect
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM, Qwen2RMSNorm
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
try:
    from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
except ImportError:
    CausalLMOutputWithCrossAttentions = None

class Qwen2CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor = None,
        past_key_value = None, 
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None and len(past_key_value) == 2:
            key_states, value_states = past_key_value
        else:
            kv_seq_len = encoder_hidden_states.shape[1]
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if self.num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
                value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if encoder_attention_mask is not None:
            # Mask format expected: [bsz, 1, 1, seq_len] with 0 for attend, large negative for ignore
            attn_weights = attn_weights + encoder_attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

def patch_qwen_for_cross_attention():
    """
    Monkey-patches Qwen2 to support Cross-Attention and encoder_hidden_states,
    allowing it to be used inside EncoderDecoderModel.
    """
    # 1. Patch Qwen2DecoderLayer
    orig_layer_init = Qwen2DecoderLayer.__init__
    def new_layer_init(self, config, layer_idx):
        orig_layer_init(self, config, layer_idx)
        if getattr(config, "is_decoder", False) and getattr(config, "add_cross_attention", False):
            self.cross_attn = Qwen2CrossAttention(config)
            self.post_cross_attn_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    Qwen2DecoderLayer.__init__ = new_layer_init

    def new_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self_attn_outputs[0]

        # Cross-Attention
        cross_attn_present = None
        if hasattr(self, "cross_attn") and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_cross_attn_layernorm(hidden_states)
            cross_attn_outputs = self.cross_attn(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=kwargs.get("cross_attn_past_key_value", None),
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + cross_attn_outputs[0]
            cross_attn_present = cross_attn_outputs[1]

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (self_attn_outputs[1],)
        if output_attentions:
            outputs += (self_attn_outputs[2],)
        if cross_attn_present is not None:
            outputs += (cross_attn_present,)

        return outputs

    Qwen2DecoderLayer.forward = new_layer_forward

    # 2. Patch Qwen2Model to pass encoder_hidden_states
    orig_model_forward = Qwen2Model.forward
    def new_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            pass # standard checks
            
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Standard processing
        hidden_states = inputs_embeds
        bsz, seq_len = hidden_states.shape[:2]

        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).expand(bsz, -1)
            else:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_attention_mask
        
        causal_mask = None
        if attention_mask is not None:
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (hidden_states.shape[0], hidden_states.shape[1]),
                inputs_embeds,
                past_key_values.get_seq_length() if past_key_values is not None else 0,
            )

        encoder_extended_attention_mask = None
        if encoder_attention_mask is not None and encoder_hidden_states is not None:
            encoder_extended_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, hidden_states.dtype, tgt_len=hidden_states.shape[1]
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    Qwen2Model.forward = new_model_forward

    # 3. Patch Qwen2ForCausalLM to accept encoder_hidden_states
    orig_clm_forward = Qwen2ForCausalLM.forward
    def new_clm_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if CausalLMOutputWithCrossAttentions is not None:
            return CausalLMOutputWithCrossAttentions(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                cross_attentions=getattr(outputs, "cross_attentions", None),
            )

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # Compatibility shim for EncoderDecoderModel expecting `cross_attentions` field.
        out.cross_attentions = getattr(outputs, "cross_attentions", None)
        return out

    Qwen2ForCausalLM.forward = new_clm_forward
    # Add dummy prepare_inputs_for_generation change if needed, but EncoderDecoderModel manages this for the decoder.

    # Also register the signature properly so inspect.signature doesn't fail
    import inspect
    old_sig = inspect.signature(Qwen2ForCausalLM.forward)
    new_params = list(old_sig.parameters.values())
    if "encoder_hidden_states" not in [p.name for p in new_params]:
        new_params.append(inspect.Parameter("encoder_hidden_states", inspect.Parameter.KEYWORD_ONLY, default=None))
        new_params.append(inspect.Parameter("encoder_attention_mask", inspect.Parameter.KEYWORD_ONLY, default=None))
        Qwen2ForCausalLM.forward.__signature__ = old_sig.replace(parameters=new_params)
