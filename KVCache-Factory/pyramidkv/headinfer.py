"""HeadInfer: lossless head-wise KV cache offloading (arXiv:2502.12574).

Adapted from the reference implementation at https://github.com/wdlctc/headinfer.

Unlike the compression methods in this repo, HeadInfer does not drop or
approximate any KV entries. It reduces GPU memory by storing the cache in one
slot per (layer, kv head) pair and keeping only a rolling window of slots on
the GPU: while attention runs on slot ``s``, slot ``s+1`` is prefetched from
CPU on a side stream and slot ``s-1`` is evicted back to CPU. The attention
forward is rewritten to iterate over KV heads so each slot is touched exactly
once per model forward, which shrinks the resident KV footprint from
``num_layers * num_kv_heads`` head-caches to a small constant number of
head-caches and lets host<->device transfers overlap with attention compute.

Only the FlashAttention-2 path is patched (matching the reference
implementation); ``run_longbench.py`` rejects other ``--attn_implementation``
values for this method.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from pyramidkv.llama_model import _flash_attention_forward


def headinfer_slot(layer_idx, kv_head_idx, num_key_value_heads):
    """Map a (layer, kv head) pair to its cache slot.

    Slots are laid out layer-major so a forward pass visits them strictly in
    increasing order, which is what makes the prefetch-next/evict-previous
    schedule of :class:`HeadwiseOffloadedCache` line up with compute.
    """
    return layer_idx * num_key_value_heads + kv_head_idx


def get_headwise_projections(module):
    """Split an attention module's q/k/v projections into per-KV-head slices.

    Returns three lists (query, key, value) of ``(weight, bias)`` pairs, one
    entry per KV head. Query rows are grouped so that group ``i`` holds the
    ``num_heads // num_key_value_heads`` query heads attending to KV head
    ``i``, matching the ``repeat_kv`` layout. The slices are views of the
    original parameters (no copies), and are cached on the module.
    """
    projections = getattr(module, "_headinfer_projections", None)
    if projections is None:
        num_kv_heads = module.num_key_value_heads
        queries_per_group = module.num_heads // num_kv_heads

        def split_rows(proj, rows_per_group):
            weights = proj.weight.split(rows_per_group, dim=0)
            if proj.bias is not None:
                biases = proj.bias.split(rows_per_group, dim=0)
            else:
                biases = [None] * num_kv_heads
            return list(zip(weights, biases))

        projections = (
            split_rows(module.q_proj, module.head_dim * queries_per_group),
            split_rows(module.k_proj, module.head_dim),
            split_rows(module.v_proj, module.head_dim),
        )
        module._headinfer_projections = projections
    return projections


class HeadwiseOffloadedCache(DynamicCache):
    """A DynamicCache whose entries live on CPU except for a rolling GPU window.

    Each entry ("slot") holds the cache of a single (layer, kv head) pair; the
    HeadInfer attention forward updates slots one at a time in increasing
    order. Reading slot ``s`` schedules an async prefetch of slot ``s+1`` on a
    side stream and an async eviction of slot ``s-1`` back to CPU, so at any
    moment only a handful of head-caches occupy GPU memory. Contents are
    identical to a plain DynamicCache; only their placement differs.
    """

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("HeadwiseOffloadedCache can only be used with a GPU")
        super().__init__()
        self.original_device = []
        self.prefetch_stream = torch.cuda.Stream()
        self.evict_stream = torch.cuda.Stream()
        self.beam_idx = None  # used to delay beam search operations

    def prefetch_slot(self, slot_idx: int):
        "Starts prefetching the next slot's cache back to its GPU device."
        if slot_idx < len(self):
            with torch.cuda.stream(self.prefetch_stream):
                device = self.original_device[slot_idx]
                self.key_cache[slot_idx] = self.key_cache[slot_idx].to(device, non_blocking=True)
                self.value_cache[slot_idx] = self.value_cache[slot_idx].to(device, non_blocking=True)

    def evict_previous_slot(self, slot_idx: int):
        "Moves the previous slot's cache to the CPU."
        if len(self) > 2:
            # The current stream was synchronized by __getitem__, so all
            # computations on the previous slot are finished before the copy.
            prev_slot_idx = (slot_idx - 1) % len(self)
            with torch.cuda.stream(self.evict_stream):
                self.key_cache[prev_slot_idx] = self.key_cache[prev_slot_idx].to("cpu", non_blocking=True)
                self.value_cache[prev_slot_idx] = self.value_cache[prev_slot_idx].to("cpu", non_blocking=True)

    def __getitem__(self, slot_idx: int) -> List[Tuple[torch.Tensor]]:
        "Gets the cache for this slot, prefetching the next and evicting the previous slot."
        if slot_idx < len(self):
            torch.cuda.current_stream().synchronize()
            self.evict_previous_slot(slot_idx)
            original_device = self.original_device[slot_idx]
            self.prefetch_stream.synchronize()
            key_tensor = self.key_cache[slot_idx]
            value_tensor = self.value_cache[slot_idx]
            # Now deal with beam search ops which were delayed
            if self.beam_idx is not None:
                self.beam_idx = self.beam_idx.to(original_device)
                key_tensor = key_tensor.index_select(0, self.beam_idx)
                value_tensor = value_tensor.index_select(0, self.beam_idx)
            self.prefetch_slot((slot_idx + 1) % len(self))
            return (key_tensor, value_tensor)
        else:
            raise KeyError(f"Cache only has {len(self)} slots, attempted to access slot with index {slot_idx}")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Saves the beam indices and reorders the cache when the tensor is back to its device."""
        # We delay this operation until the tensors are back to their original
        # device because performing torch.index_select on the CPU is very slow
        del self.beam_idx
        self.beam_idx = beam_idx.clone()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        slot_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the cache with new `key_states`/`value_states` for slot `slot_idx`."""
        # slot 0 belongs to (layer 0, kv head 0), so this runs once per model forward
        if slot_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) < slot_idx:
            raise ValueError(
                "HeadwiseOffloadedCache does not support model usage where slots are skipped. Use DynamicCache."
            )
        elif len(self.key_cache) == slot_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.original_device.append(key_states.device)
            self.evict_previous_slot(slot_idx)
        else:
            key_tensor, value_tensor = self[slot_idx]
            self.key_cache[slot_idx] = torch.cat([key_tensor, key_states], dim=-2)
            self.value_cache[slot_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[slot_idx], self.value_cache[slot_idx]

    # According to https://docs.python.org/3/library/exceptions.html#NotImplementedError
    # if a method is not supposed to be supported in a subclass we should set it to None
    from_legacy_cache = None

    to_legacy_cache = None


def headinfer_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """FlashAttention-2 forward that computes attention one KV head at a time.

    Numerically equivalent to the stock GQA forward (each query-head group is
    an independent attention problem), but each (layer, kv head) cache slot is
    read and written exactly once per forward, enabling head-wise offloading.
    """
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()
    num_kv_heads = self.num_key_value_heads
    queries_per_group = self.num_heads // num_kv_heads
    q_projs, k_projs, v_projs = get_headwise_projections(self)

    cos = sin = None
    attn_outputs = []
    for i in range(num_kv_heads):
        q_weight, q_bias = q_projs[i]
        k_weight, k_bias = k_projs[i]
        v_weight, v_bias = v_projs[i]

        query_states = F.linear(hidden_states, q_weight, q_bias)
        key_states = F.linear(hidden_states, k_weight, k_bias)
        value_states = F.linear(hidden_states, v_weight, v_bias)

        query_states = query_states.view(bsz, q_len, queries_per_group, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)

        if cos is None:
            # rotary embeddings depend on positions only, not on head values
            cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            slot_idx = headinfer_slot(self.layer_idx, i, num_kv_heads)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, slot_idx, cache_kwargs)

        # flash_attn handles GQA natively (num query heads divisible by num kv heads)
        attn_output = _flash_attention_forward(
            self,
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            dropout=0.0,
        )
        attn_outputs.append(attn_output.reshape(bsz, q_len, queries_per_group * self.head_dim))

    attn_output = torch.cat(attn_outputs, dim=-1).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


# Both Llama and Mistral flash-attention modules expose the same attributes
# used above (q/k/v/o_proj, rotary_emb, num_heads, num_key_value_heads,
# head_dim), so they share one implementation.
llama_flash_attn2_forward_HeadInfer = headinfer_flash_attn2_forward
mistral_flash_attn2_forward_HeadInfer = headinfer_flash_attn2_forward
