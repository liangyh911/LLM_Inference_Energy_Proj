import unittest

import torch
import torch.nn.functional as F

from pyramidkv.headinfer import (
    HeadwiseOffloadedCache,
    get_headwise_projections,
    headinfer_slot,
)


class _FakeAttention(torch.nn.Module):
    """Minimal stand-in exposing the attributes get_headwise_projections uses."""

    def __init__(self, hidden_size=16, num_heads=4, num_key_value_heads=2, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = torch.nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = torch.nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=bias)


class HeadInferSlotTest(unittest.TestCase):
    def test_slots_enumerate_in_forward_traversal_order(self):
        num_kv_heads = 4
        slots = [
            headinfer_slot(layer, head, num_kv_heads)
            for layer in range(3)
            for head in range(num_kv_heads)
        ]
        # Strictly sequential slots are what make prefetch-next/evict-previous
        # line up with the order the attention forwards touch the cache.
        self.assertEqual(slots, list(range(3 * num_kv_heads)))


class HeadwiseProjectionTest(unittest.TestCase):
    def test_split_projections_match_full_projection(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                torch.manual_seed(0)
                module = _FakeAttention(bias=bias)
                hidden = torch.randn(2, 5, 16)

                q_projs, k_projs, v_projs = get_headwise_projections(module)

                q_cat = torch.cat([F.linear(hidden, w, b) for w, b in q_projs], dim=-1)
                k_cat = torch.cat([F.linear(hidden, w, b) for w, b in k_projs], dim=-1)
                v_cat = torch.cat([F.linear(hidden, w, b) for w, b in v_projs], dim=-1)

                torch.testing.assert_close(q_cat, module.q_proj(hidden))
                torch.testing.assert_close(k_cat, module.k_proj(hidden))
                torch.testing.assert_close(v_cat, module.v_proj(hidden))

    def test_query_groups_align_with_repeat_kv_layout(self):
        torch.manual_seed(0)
        module = _FakeAttention()
        hidden = torch.randn(1, 3, 16)
        head_dim = module.head_dim
        queries_per_group = module.num_heads // module.num_key_value_heads

        q_projs, _, _ = get_headwise_projections(module)
        q_full = module.q_proj(hidden)

        for i, (weight, bias) in enumerate(q_projs):
            start = i * queries_per_group * head_dim
            end = (i + 1) * queries_per_group * head_dim
            torch.testing.assert_close(F.linear(hidden, weight, bias), q_full[..., start:end])

    def test_slices_are_views_and_cached(self):
        module = _FakeAttention()
        first = get_headwise_projections(module)
        # slices share storage with the original parameters (no weight copies)
        self.assertEqual(first[0][0][0].data_ptr(), module.q_proj.weight.data_ptr())
        # repeated calls reuse the cached split
        self.assertIs(first, get_headwise_projections(module))


class HeadwiseAttentionEquivalenceTest(unittest.TestCase):
    def test_per_kv_head_attention_matches_full_gqa_attention(self):
        """HeadInfer's per-KV-head loop must be exact, not an approximation."""
        torch.manual_seed(0)
        bsz, q_len, num_heads, num_kv_heads, head_dim = 1, 6, 4, 2, 8
        queries_per_group = num_heads // num_kv_heads

        query = torch.randn(bsz, num_heads, q_len, head_dim)
        key = torch.randn(bsz, num_kv_heads, q_len, head_dim)
        value = torch.randn(bsz, num_kv_heads, q_len, head_dim)

        # reference: repeat_kv layout (kv head i serves query heads
        # [i * queries_per_group, (i + 1) * queries_per_group))
        full = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(queries_per_group, dim=1),
            value.repeat_interleave(queries_per_group, dim=1),
            is_causal=True,
        )

        grouped = torch.cat(
            [
                F.scaled_dot_product_attention(
                    query[:, i * queries_per_group : (i + 1) * queries_per_group],
                    key[:, i : i + 1].expand(-1, queries_per_group, -1, -1),
                    value[:, i : i + 1].expand(-1, queries_per_group, -1, -1),
                    is_causal=True,
                )
                for i in range(num_kv_heads)
            ],
            dim=1,
        )

        torch.testing.assert_close(grouped, full)


class HeadwiseOffloadedCacheTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "covers the CPU-only error path")
    def test_cache_requires_cuda(self):
        with self.assertRaises(RuntimeError):
            HeadwiseOffloadedCache()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cache_contents_match_dynamic_cache_and_slots_offload(self):
        from transformers.cache_utils import DynamicCache

        torch.manual_seed(0)
        num_slots, head_dim = 6, 4

        cache = HeadwiseOffloadedCache()
        reference = DynamicCache()

        # prefill pass followed by one decode step, slots visited in order
        for step_len in (5, 1):
            keys = [torch.randn(1, 1, step_len, head_dim, device="cuda") for _ in range(num_slots)]
            values = [torch.randn(1, 1, step_len, head_dim, device="cuda") for _ in range(num_slots)]
            for slot in range(num_slots):
                cache.update(keys[slot], values[slot], slot)
                reference.update(keys[slot], values[slot], slot)

        torch.cuda.synchronize()

        self.assertEqual(cache.get_seq_length(), 6)
        # earlier slots were evicted off the GPU
        self.assertTrue(any(t.device.type == "cpu" for t in cache.key_cache))
        for slot in range(num_slots):
            torch.testing.assert_close(cache.key_cache[slot].to("cuda"), reference.key_cache[slot])
            torch.testing.assert_close(cache.value_cache[slot].to("cuda"), reference.value_cache[slot])


if __name__ == "__main__":
    unittest.main()
