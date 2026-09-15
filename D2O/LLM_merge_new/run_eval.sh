CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench_sample.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --cache_dir /projects/cdux/yuhangl/Cache_Dir \
    --use_full False \
    --use_d2o True \
    --model_type llama3 \
    --hh_ratio 0.1 \
    --recent_ratio 0.1 \
    --action_name d2o \
    --e False 
