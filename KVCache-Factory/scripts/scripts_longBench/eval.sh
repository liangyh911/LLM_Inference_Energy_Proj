export CUDA_VISIBLE_DEVICES=0

method="PyramidKV" # Support PyramidKV, SnapKV, H2O, StreamingLLM, CAM, L2Norm, ThinK, HeadKV, default FullKV
max_capacity_prompts=4096 # 128,2048 in paper
attn_implementation="flash_attention_2" # Support "flash_attention_2", "sdpa", "eager".
source_path="./Results/"
model_path="meta-llama/Meta-Llama-3-8B-Instruct"
merge_method="pivot" # Support "pivot"(LOOK-M_PivotMerge).
quant_method="kvquant" # Support kivi and kvquant, default None.
nbits="8" # Quantization bit-width support 8,4,2. Need to set quant_method first.
save_dir=${source_path}"results_long_bench" # path to result save_dir

extra_args=()
if [ -n "${merge_method}" ] && [ "${merge_method}" != "none" ] && [ "${merge_method}" != "None" ]; then
    extra_args+=(--merge "${merge_method}")
fi
if [ -n "${quant_method}" ] && [ "${quant_method}" != "none" ] && [ "${quant_method}" != "None" ]; then
    extra_args+=(--quant_method "${quant_method}")
    if [ -n "${nbits}" ]; then
        extra_args+=(--nbits "${nbits}")
    fi
fi

python3 run_longbench_energy.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --eval_batch_size 8 \
    --dtype bfloat16 \
    --enable_power_monitor "True" \
    --kv_cache_granularity kv_head
    # "${extra_args[@]}"
