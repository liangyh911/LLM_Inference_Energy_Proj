// energy_vs_kv_size.cu
//
// 目的：用 decoding_loop.cu 里真实的 update_kv_cache_kernel / decoding_attention_kernel，
//      测量 KV cache 长度 L 增长对二者单次能耗(mJ)、运行时间、平均功率的影响。
//
// 方法：
//   对一组预先选定的 L 值（log 间隔采样，随机顺序），每个 L 分别对两个 kernel 做：
//     - warm-up（不计时，让时钟/温度进入稳态）
//     - 校准（跑 20 次，估计单次运行时间，从而反推需要跑多少次才能让总窗口
//       >= kMinWindowMs，也就是 >= 3 倍能耗计数器刷新周期，H100 上实测约 100ms）
//     - 正式批量测量：前后各读一次 nvmlDeviceGetTotalEnergyConsumption
//   为控制热漂移/时钟状态带来的系统性偏差：
//     - 所有 (L, trial) 任务随机打乱顺序
//     - 每个 L 重复 kTrialsPerL 轮，事后取中位数
//     - 记录测量前后温度、以及 nvmlDeviceGetCurrentClocksThrottleReasons，
//       非 0 说明该样本测量期间发生过降频，分析时应剔除或标注
//
// 注意：seq_lens 在整个某个 (L, trial) 的测量过程中保持固定为 L（不像原始
//       decoding_loop.cu 那样每个 step 后递增），这样才能对同一个 L 反复采样、
//       拿到统计意义上可靠的 (L, energy) 数据点。
//
// 编译：
//   nvcc -O2 energy_vs_kv_size.cu -o energy_vs_kv_size -lnvidia-ml
// 运行：
//   ./energy_vs_kv_size 0

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <nvml.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_NVML(call) do { \
    nvmlReturn_t err = call; \
    if (err != NVML_SUCCESS) { \
        std::cerr << "NVML Error: " << nvmlErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

using Clock = std::chrono::steady_clock;

// =========================================================
// Kernel 1: 更新 KV Cache  (与 decoding_loop.cu 完全一致)
// =========================================================
template <typename T>
__global__ void update_kv_cache_kernel(
    T* k_cache, T* v_cache,
    const T* new_k, const T* new_v,
    const int* seq_lens,
    int max_seq_len, int num_heads, int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;

    if (d < head_dim) {
        int current_len = seq_lens[b];
        if (current_len >= max_seq_len) return;

        int new_idx = b * (num_heads * head_dim) + h * head_dim + d;
        int cache_idx = b * (num_heads * max_seq_len * head_dim) +
                        h * (max_seq_len * head_dim) +
                        current_len * head_dim + d;

        k_cache[cache_idx] = new_k[new_idx];
        v_cache[cache_idx] = new_v[new_idx];
    }
}

// =========================================================
// Kernel 2: Single-Query Attention 计算  (与 decoding_loop.cu 完全一致)
// =========================================================
template <typename T>
__global__ void decoding_attention_kernel(
    const T* q, const T* k_cache, const T* v_cache, T* out,
    const int* seq_lens, int max_seq_len, int head_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int current_len = seq_lens[b];

    const T* my_q = q + (b * gridDim.y + h) * head_dim;
    const T* my_k_cache = k_cache + (b * gridDim.y + h) * max_seq_len * head_dim;
    const T* my_v_cache = v_cache + (b * gridDim.y + h) * max_seq_len * head_dim;
    T* my_out = out + (b * gridDim.y + h) * head_dim;

    extern __shared__ float s_mem[];
    float* s_reduce = s_mem;
    float* s_scores = &s_mem[head_dim];

    float q_val = my_q[tid];
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int l = 0; l < current_len; ++l) {
        float k_val = my_k_cache[l * head_dim + tid];
        s_reduce[tid] = q_val * k_val * scale;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) s_scores[l] = s_reduce[0];
        __syncthreads();
    }

    __shared__ float s_max, s_sum;
    if (tid == 0) {
        float max_val = -1e20f;
        for (int l = 0; l < current_len; ++l) {
            if (s_scores[l] > max_val) max_val = s_scores[l];
        }
        s_max = max_val;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int l = 0; l < current_len; ++l) {
            s_scores[l] = expf(s_scores[l] - s_max);
            sum += s_scores[l];
        }
        s_sum = sum;
    }
    __syncthreads();

    if (tid == 0) {
        for (int l = 0; l < current_len; ++l) s_scores[l] /= s_sum;
    }
    __syncthreads();

    float out_val = 0.0f;
    for (int l = 0; l < current_len; ++l) {
        out_val += s_scores[l] * my_v_cache[l * head_dim + tid];
    }
    my_out[tid] = out_val;
}

// ---- 实验参数，按需调整 ----
// 注意：max_seq_len 必须 > 这里最大的 L，否则 update kernel 会因为
// current_len >= max_seq_len 直接跳过、attention 的 shared mem 也不够用。
// 显存占用 ~ 2 * batch_size * num_heads * max_seq_len * head_dim * 4 bytes
// (k_cache + v_cache)，L 上到 16384 时在 batch=16/heads=64/head_dim=128 下
// 大约 17GB，如果你的卡显存吃紧，请调小 batch_size/num_heads 或缩小 L 范围。
static const std::vector<int> kSeqLens = {
    128, 256, 512, 1024, 2048, 4096, 8192, 16384
};
static const int kTrialsPerL = 5;
static const int kWarmupIters = 20;
static const double kMinWindowMs = 300.0;   // >= 3x 能耗计数器刷新周期
static const int kCalibIters = 20;

struct TrialResult {
    std::string kernel_name;
    int L;
    int trial_idx;
    int n_iters;
    double total_time_ms;
    unsigned long long total_energy_mJ;
    double avg_runtime_us;
    double avg_energy_mJ;
    double avg_power_W;
    unsigned int temp_before_C;
    unsigned int temp_after_C;
    unsigned long long throttle_reasons;
};

// 通用批量测量：warm-up -> 校准迭代次数 -> 正式测量能耗
template <typename LaunchFn>
TrialResult measure_batch(nvmlDevice_t device, LaunchFn&& launch_fn,
                           const std::string& kernel_name, int L, int trial_idx) {
    for (int i = 0; i < kWarmupIters; ++i) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto calib_start = Clock::now();
    for (int i = 0; i < kCalibIters; ++i) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());
    double calib_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - calib_start).count();
    double per_iter_ms = std::max(calib_ms / kCalibIters, 1e-6);
    int n_iters = std::max(1, static_cast<int>(kMinWindowMs / per_iter_ms) + 1);

    unsigned int temp_before = 0, temp_after = 0;
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_before));

    unsigned long long e_start, e_end;
    CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &e_start));
    auto t_start = Clock::now();

    for (int i = 0; i < n_iters; ++i) launch_fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    auto t_end = Clock::now();
    CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &e_end));
    CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_after));

    unsigned long long throttle_reasons = 0;
    CHECK_NVML(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle_reasons));

    double total_time_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    unsigned long long total_energy_mJ = e_end - e_start;

    TrialResult r;
    r.kernel_name = kernel_name;
    r.L = L;
    r.trial_idx = trial_idx;
    r.n_iters = n_iters;
    r.total_time_ms = total_time_ms;
    r.total_energy_mJ = total_energy_mJ;
    r.avg_runtime_us = (total_time_ms * 1000.0) / n_iters;
    r.avg_energy_mJ = static_cast<double>(total_energy_mJ) / n_iters;
    r.avg_power_W = (total_energy_mJ / 1000.0) / (total_time_ms / 1000.0);
    r.temp_before_C = temp_before;
    r.temp_after_C = temp_after;
    r.throttle_reasons = throttle_reasons;
    return r;
}

int main(int argc, char** argv) {
    int device_index = (argc > 1) ? std::atoi(argv[1]) : 0;
    CHECK_CUDA(cudaSetDevice(device_index));

    CHECK_NVML(nvmlInit());
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_index, &device));

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    CHECK_NVML(nvmlDeviceGetName(device, name, sizeof(name)));
    std::cout << "Device: " << name << " (index " << device_index << ")\n";

    // --- 参数与内存分配（与 decoding_loop.cu 保持一致的 batch/heads/head_dim）---
    int batch_size = 16;
    int num_heads = 64;
    int head_dim = 128;
    int max_seq_len = *std::max_element(kSeqLens.begin(), kSeqLens.end()) + 1;

    size_t cache_size = (size_t)batch_size * num_heads * max_seq_len * head_dim * sizeof(float);
    size_t vector_size = (size_t)batch_size * num_heads * head_dim * sizeof(float);
    size_t seq_lens_size = batch_size * sizeof(int);

    double cache_gb = (2.0 * cache_size) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "max_seq_len=" << max_seq_len
              << "  预计 k_cache+v_cache 显存占用: " << cache_gb << " GB\n";

    float *d_k_cache, *d_v_cache, *d_q, *d_k, *d_v, *d_out;
    int *d_seq_lens;
    CHECK_CUDA(cudaMalloc(&d_k_cache, cache_size));
    CHECK_CUDA(cudaMalloc(&d_v_cache, cache_size));
    CHECK_CUDA(cudaMalloc(&d_q, vector_size));
    CHECK_CUDA(cudaMalloc(&d_k, vector_size));
    CHECK_CUDA(cudaMalloc(&d_v, vector_size));
    CHECK_CUDA(cudaMalloc(&d_out, vector_size));
    CHECK_CUDA(cudaMalloc(&d_seq_lens, seq_lens_size));

    std::vector<float> h_init_cache(cache_size / sizeof(float), 0.1f);
    std::vector<float> h_init_vec(vector_size / sizeof(float), 1.0f);
    CHECK_CUDA(cudaMemcpy(d_k_cache, h_init_cache.data(), cache_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_cache, h_init_cache.data(), cache_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));

    dim3 grid(batch_size, num_heads);
    dim3 block(head_dim);
    size_t shared_mem_size = (size_t)(head_dim + max_seq_len) * sizeof(float);

    // L 较大时动态共享内存会超过默认 48KB 上限，需要显式放开。
    CHECK_CUDA(cudaFuncSetAttribute(
        decoding_attention_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shared_mem_size));
    std::cout << "shared_mem_size=" << shared_mem_size << " bytes\n\n";

    // --- 任务列表：(L, trial) 全部打乱顺序，减少热漂移带来的系统性偏差 ---
    struct Task { int L; int trial; };
    std::vector<Task> tasks;
    for (int L : kSeqLens)
        for (int t = 0; t < kTrialsPerL; ++t)
            tasks.push_back({L, t});
    std::mt19937 rng(12345);
    std::shuffle(tasks.begin(), tasks.end(), rng);

    std::vector<int> h_seq_lens(batch_size);
    std::vector<TrialResult> results;

    for (const auto& task : tasks) {
        int L = task.L;
        std::fill(h_seq_lens.begin(), h_seq_lens.end(), L);
        CHECK_CUDA(cudaMemcpy(d_seq_lens, h_seq_lens.data(), seq_lens_size, cudaMemcpyHostToDevice));

        auto launch_update = [&]() {
            update_kv_cache_kernel<float><<<grid, block>>>(
                d_k_cache, d_v_cache, d_k, d_v, d_seq_lens, max_seq_len, num_heads, head_dim);
        };
        auto launch_attention = [&]() {
            decoding_attention_kernel<float><<<grid, block, shared_mem_size>>>(
                d_q, d_k_cache, d_v_cache, d_out, d_seq_lens, max_seq_len, head_dim);
        };

        TrialResult r_update = measure_batch(device, launch_update, "kv_update", L, task.trial);
        TrialResult r_attn = measure_batch(device, launch_attention, "attention", L, task.trial);

        for (const auto& r : {r_update, r_attn}) {
            results.push_back(r);
            printf("[%-9s L=%6d trial=%d] iters=%7d  window=%7.1fms  "
                   "avg_runtime=%9.3fus  avg_energy=%9.4fmJ  avg_power=%6.1fW  "
                   "temp=%u->%uC  throttle=0x%llx%s\n",
                   r.kernel_name.c_str(), r.L, r.trial_idx, r.n_iters, r.total_time_ms,
                   r.avg_runtime_us, r.avg_energy_mJ, r.avg_power_W,
                   r.temp_before_C, r.temp_after_C, r.throttle_reasons,
                   r.throttle_reasons != 0 ? "  <-- 降频样本，建议剔除" : "");
        }
    }

    // --- 输出 CSV，供后续用 python 画图 / 回归 ---
    FILE* fp = fopen("energy_vs_kv_size.csv", "w");
    fprintf(fp, "kernel,L,trial,n_iters,total_time_ms,total_energy_mJ,avg_runtime_us,"
                "avg_energy_mJ,avg_power_W,temp_before_C,temp_after_C,throttle_reasons\n");
    for (const auto& r : results) {
        fprintf(fp, "%s,%d,%d,%d,%.4f,%llu,%.4f,%.4f,%.2f,%u,%u,0x%llx\n",
                r.kernel_name.c_str(), r.L, r.trial_idx, r.n_iters, r.total_time_ms,
                r.total_energy_mJ, r.avg_runtime_us, r.avg_energy_mJ, r.avg_power_W,
                r.temp_before_C, r.temp_after_C, r.throttle_reasons);
    }
    fclose(fp);
    std::cout << "\n结果已写入 energy_vs_kv_size.csv\n"
                 "按 kernel 分组，再按 L 分组取 avg_energy_mJ 的中位数，\n"
                 "对 (L, median_energy) 做线性回归即可得到 energy ~= a*L + b。\n"
                 "预期：kv_update 的 energy 应大致与 L 无关（接近水平线，因为写入是 O(1)）；\n"
                 "      attention 的 energy 应随 L 大致线性增长（点积/加权求和是 O(L)）。\n";

    CHECK_NVML(nvmlShutdown());
    cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_seq_lens);
    return 0;
}
