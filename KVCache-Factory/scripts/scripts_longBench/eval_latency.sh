export CUDA_VISIBLE_DEVICES=0

python ./benchmark_latency_memory.py \
  --model_path "meta-llama/Meta-Llama-3-8B-Instruct" \
  --method fullkv \
  --attn_implementation flash_attention_2 \
  --max_capacity_prompt 128 \
  --max_new_tokens 256 \
  --repeat 3 \
  --kv_cache_granularity kv_head