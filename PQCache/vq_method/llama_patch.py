import os
import math
import torch
import types
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaDecoderLayer,
    LlamaModel,
    rotate_half,
    repeat_kv,
)
from .baseline_compressor import *
from .flash_attn_with_score import flash_attn_with_score
from .retrieval_based.pq_search import *
from .retrieval_based.sparq import *
from flash_attn import flash_attn_func
import seaborn as sns
from loguru import logger

__all__ = ["VQLlamaForCausalLM", "VQLlamaAttention"]


def layer2device(idx, layer_cnt):
    gpu_in_use = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    step = math.ceil(layer_cnt / gpu_in_use)
    return torch.device(f"cuda:{idx // step}")


def get_device(layer: nn.Module):
    for param in layer.parameters():
        return param.device


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    
    cos = cos.squeeze(1).squeeze(0)  
    sin = sin.squeeze(1).squeeze(0)  
    cos = cos[position_ids].unsqueeze(1)  
    sin = sin[position_ids].unsqueeze(1)  
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def LlamaAttentionPatch(attn: LlamaAttention, config, idx):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1, "Do not support bsz > 1 yet."

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        first_time = past_key_value is None
        position_length = key_states.shape[-2]
        if not first_time:
            assert position_ids.nelement() == 1
            if position_length < position_ids.item() + 1:
                position_length = position_ids.item() + 1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)
        
        if first_time:
            self.seq_cnt += 1
            self.fwd_cnt = 0
        
        if self.compressor == "original":
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            if self.fwd_cnt <= 0 and self.idx <= 0:
                print(f"Using naive flash-attn, NO COMPRESSION IS CONVEYED NOW, {os.getpid()}")
            attn_output = flash_attn_func(
                    query_states.transpose(1,2),
                    key_states.transpose(1,2),
                    value_states.transpose(1,2),
                    causal = True
                ).transpose(1,2)
        elif self.compressor in ["pq_search", "sparq_f"]: 
            if first_time:
                past_key_value = (key_states, value_states) if use_cache else None
                attn_output, _ = self.kvcache_quantizer.prefill_attn(query_states, past_key_value)
                
                past_key_value = (key_states.new_zeros([1]), value_states.new_zeros([1])) 
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups) 
                value_states = repeat_kv(value_states, self.num_key_value_groups)
                attn_output = self.kvcache_quantizer.decoding_attn(
                                                    self.num_key_value_groups, 
                                                    query_states,
                                                    key_states, value_states).to(query_states.dtype)
                attn_output = attn_output.to(query_states.dtype)
        else: 
            if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            if self.use_flash_attn and first_time:

                attn_output, score = flash_attn_with_score(query_states, key_states, value_states, \
                                                            phase="prefill", gumbel_adjustment=False, \
                                                            score_func=self.score_func)
                score = score.reshape([bsz, self.num_key_value_heads, self.num_key_value_groups, q_len])

                if self.score_func == "sum":
                    score = score.sum(dim=2)
                elif self.score_func == "max":
                    score = score.max(dim=2).values
                else:
                    raise Exception(f"Given score func {self.score_func} do not support yet.")
                compressed_k, compressed_v, _ = self.kvcache_quantizer.apply(past_key_value, 
                                                                                        attention_score=score, 
                                                                                        query_states=query_states)

                past_key_value = (compressed_k, compressed_v)
            else:
                attention_mask = None 
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = self.kvcache_quantizer.restore(
                    attn_weights, self.num_key_value_groups).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        self.fwd_cnt += 1

        return attn_output, attn_weights, past_key_value

    attn.forward = types.MethodType(forward, attn)
    attn.use_flash_attn = True
    attn.fwd_cnt = 0
    attn.idx = idx
    attn.score_func = config.score_func
    attn.compressor = config.compressor
    attn.seq_cnt = -1
    if config.compressor == "h2o":
        attn.kvcache_quantizer = KVCacheH2OOfficial(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            config.sink_size,
        )
    elif config.compressor == "no_drop_lb":
        assert attn.score_func == "sum", "full KV limited based compressor only accept sum function"
        attn.kvcache_quantizer = fullKVLimitBasedCompressor(
            config.compress_ratio,
            config.important_ratio,
            config.recent_ratio,
            config.gqa,
            config.sink_size,
        )
    elif config.compressor == "pq_search":
        attn.kvcache_quantizer = PqBasedSearchCompressor(
            config.compress_ratio,
            config.recent_ratio,
            config.n_subvec_per_head,
            config.n_subbits,
            config.gqa,
            config.sink_size,
            layer_idx = attn.idx,
            cur_device=layer2device(attn.idx, config.num_hidden_layers),
            max_iter = config.max_iter,
            kv_head = config.num_key_value_heads,
            dim = config.hidden_size // config.num_attention_heads,
            num_layer_cnt = config.num_hidden_layers
        )
    elif config.compressor == "sparq_f":
        if os.environ.get("MODE","off") == "profile":
            raise NotImplementedError("profile mode for Sparq is not done yet.")
        else:
            if attn.idx <= 2:
                print(f"Using Sparq Compressor, gpu version.")
            attn.kvcache_quantizer = SparQCompressorGPU(
                config.compress_ratio,
                config.recent_ratio,
                config.sink_size,
                config.gqa,
                r = config.topr,
                idx = attn.idx,
                model_config = config,
                layer_idx = attn.idx,
                cur_device=layer2device(attn.idx, config.num_hidden_layers),
                kv_head = config.num_key_value_heads,
                dim = config.hidden_size // config.num_key_value_heads
            )
    elif config.compressor == "original":
        pass
    else:
        raise Exception("Invalid compression strategy name")

def LlamaDecoderLayerPatch(layer: LlamaDecoderLayer, config, layer_idx):
    layer.device = layer2device(layer_idx, config.num_hidden_layers)
    LlamaAttentionPatch(layer.self_attn, config, layer_idx)
    return layer.half()

def PPLlamaModelPatch(model:LlamaModel, config):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if position_ids is None:
            raise Exception("We assume that position id is not None")
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        padding_mask = None

        attention_mask = None
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if past_key_value is not None:
                assert past_key_values[idx][0].device == get_device(decoder_layer)

            if hidden_states.device != decoder_layer.device:
                hidden_states = hidden_states.to(decoder_layer.device)

            if position_ids.device != decoder_layer.device:
                position_ids = position_ids.to(decoder_layer.device)

            if attention_mask is not None and attention_mask.device != decoder_layer.device:
                attention_mask = attention_mask.to(decoder_layer.device)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if hidden_states.device != get_device(self.norm):
            hidden_states = hidden_states.to(get_device(self.norm))

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    model.vocab_size = config.vocab_size
    model.forward = types.MethodType(forward, model)
    model.embed_tokens = model.embed_tokens.to(torch.device("cuda:0"))

    for i in range(config.num_hidden_layers):
        model.layers[i] = LlamaDecoderLayerPatch(model.layers[i].to(layer2device(i, config.num_hidden_layers)), config, i)
    model.norm = model.norm.to(torch.device(f"cuda:{config.pp_size - 1}"))

    model.gradient_checkpointing = False
    
    model.post_init()
    return model.half()
    

class VQLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        a = time.perf_counter()
        super().__init__(config)
        b = time.perf_counter()

        self.model = PPLlamaModelPatch(self.model, config)
        self.lm_head = self.lm_head.to(torch.device(f"cuda:{config.pp_size - 1}")).half()
        self.model.embed_tokens = self.model.embed_tokens.to(torch.device("cuda:0")).half()
        self.layer_num = config.num_hidden_layers
        self.kv_head_cnt = config.num_key_value_heads

        self._device = torch.device("cuda:0")
        self.fwd_cnt = 0
        self.gen_seq_cnt = 0
        self.prefill_len = 0

        c = time.perf_counter()
        print(f"Init model from llama patch, Time elapsed:{c - b}, {b - a}")
        self.gradient_checkpointing = False
        self.post_init()
        torch.cuda.empty_cache()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        
        loss = loss.to(torch.device(f"cuda:0")) if loss is not None else None
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_state.to(torch.device(f"cuda:0"))
        logits = logits.to(torch.device(f"cuda:0"))
        if outputs.attentions is not None:
            outputs.attentions = outputs.attentions.to(torch.device(f"cuda:0"))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )