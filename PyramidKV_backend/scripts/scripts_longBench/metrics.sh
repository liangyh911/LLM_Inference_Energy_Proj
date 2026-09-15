# bash ./scripts/scripts_longBench/metrics.sh ./Results/results_long_bench/meta-llama-3-8b-instruct_1024

results_dir=$1


python3 eval.py \
    --results_dir ${results_dir}
