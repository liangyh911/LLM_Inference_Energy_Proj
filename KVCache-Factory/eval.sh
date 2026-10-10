python benchmark_latency_memory.py \
  --model_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --method fullkv \
  --attn_implementation flash_attention_2 \
  --max_capacity_prompt 512 \
  --max_new_tokens 1024 \
  --prompt_repeats 500\
  --repeat 3