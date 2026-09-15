"""HeadInfer long-context demo: head-wise KV cache offloading with chunked prefill.

LongBench-scale contexts work through ``run_longbench.py --method headinfer``
directly. This example shows the regime HeadInfer was built for: contexts far
beyond what the full KV cache (or even prefill activations) would allow on a
single GPU. The KV cache lives in one slot per (layer, kv head) pair and is
streamed between CPU and GPU, and prefill is fed in chunks so activation
memory stays bounded.

Example (Llama-3-8B on a 24GB GPU):

    python examples/headinfer_example.py \
        --model_path /path/to/Llama-3-8B-Instruct \
        --context_tokens 409600 --chunk_tokens 10240 --max_new_tokens 16
"""

import argparse
import inspect
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyramidkv.headinfer import HeadwiseOffloadedCache
from pyramidkv.monkeypatch import replace_llama, replace_mistral


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--context_tokens", type=int, default=102400)
    parser.add_argument("--chunk_tokens", type=int, default=10240,
                        help="prefill chunk size; smaller chunks bound activation memory")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    return parser.parse_args()


def main():
    args = parse_args()

    replace_llama("headinfer")
    replace_mistral("headinfer")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()

    # Synthetic context: the benchmark only measures memory/latency, so random
    # token ids are enough and avoid tokenizing a huge string.
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.context_tokens), device="cuda")

    # num_logits_to_keep skips materializing the full [chunk, vocab] logits;
    # fall back gracefully on transformers versions without it.
    extra_forward_kwargs = {}
    if "num_logits_to_keep" in inspect.signature(model.forward).parameters:
        extra_forward_kwargs["num_logits_to_keep"] = 1

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        past_key_values = HeadwiseOffloadedCache()

        # chunked prefill
        prefill_start = time.perf_counter()
        for start in range(0, args.context_tokens, args.chunk_tokens):
            chunk = input_ids[:, start : start + args.chunk_tokens]
            position_ids = torch.arange(start, start + chunk.shape[1], device="cuda").unsqueeze(0)
            outputs = model(
                input_ids=chunk,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **extra_forward_kwargs,
            )
            torch.cuda.synchronize()
            print(f"prefilled {min(start + args.chunk_tokens, args.context_tokens)}/{args.context_tokens} tokens")
        prefill_sec = time.perf_counter() - prefill_start

        # greedy decode
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = [next_token]
        decode_start = time.perf_counter()
        for step in range(args.max_new_tokens - 1):
            position_ids = torch.tensor([[args.context_tokens + step]], device="cuda")
            outputs = model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                **extra_forward_kwargs,
            )
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)
        torch.cuda.synchronize()
        decode_sec = time.perf_counter() - decode_start

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"context tokens : {args.context_tokens}")
    print(f"prefill time   : {prefill_sec:.1f}s")
    print(f"decode         : {args.max_new_tokens} tokens in {decode_sec:.1f}s")
    print(f"peak GPU memory: {peak_gib:.2f} GiB")
    print(f"sample output  : {tokenizer.decode(torch.cat(generated, dim=-1)[0])!r}")


if __name__ == "__main__":
    main()
