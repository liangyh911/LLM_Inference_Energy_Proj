#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <nvml.h>

// nvcc -O3 decoding_loop.cu -o decoding_loop -lnvidia-ml

// CUDA 错误检查
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// NVML 错误检查
#define CHECK_NVML(call) do { \
    nvmlReturn_t err = call; \
    if (err != NVML_SUCCESS) { \
        std::cerr << "NVML Error: " << nvmlErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// =========================================================
// Kernel 1: 更新 KV Cache
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
// Kernel 2: Single-Query Attention 计算
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

    // 1. Q * K^T
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

    // 2. Softmax
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

    // 3. Output = Prob * V
    float out_val = 0.0f;
    for (int l = 0; l < current_len; ++l) {
        out_val += s_scores[l] * my_v_cache[l * head_dim + tid];
    }
    my_out[tid] = out_val;
}


// =========================================================
// 主函数: 包含测量逻辑的 Decoding Loop
// =========================================================
int main() {
    // --- 1. 初始化 NVML ---
    CHECK_NVML(nvmlInit());
    nvmlDevice_t device;
    // 获取 0 号 GPU，如果使用多卡请更改索引
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));

    // --- 2. 初始化 CUDA Events 用于测速 ---
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));

    // --- 3. 参数与内存分配 ---
    int batch_size = 16;
    int max_seq_len = 4096; 
    int num_heads = 64;     
    int head_dim = 128;    

    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim * sizeof(float);
    size_t vector_size = batch_size * num_heads * head_dim * sizeof(float);
    size_t seq_lens_size = batch_size * sizeof(int);

    float *d_k_cache, *d_v_cache, *d_q, *d_k, *d_v, *d_out;
    int *d_seq_lens;

    CHECK_CUDA(cudaMalloc(&d_k_cache, cache_size));
    CHECK_CUDA(cudaMalloc(&d_v_cache, cache_size));
    CHECK_CUDA(cudaMalloc(&d_q, vector_size));
    CHECK_CUDA(cudaMalloc(&d_k, vector_size));
    CHECK_CUDA(cudaMalloc(&d_v, vector_size));
    CHECK_CUDA(cudaMalloc(&d_out, vector_size));
    CHECK_CUDA(cudaMalloc(&d_seq_lens, seq_lens_size));

    int initial_len = 128;

    std::vector<int> h_seq_lens(batch_size, initial_len);
    CHECK_CUDA(cudaMemcpy(d_seq_lens, h_seq_lens.data(), seq_lens_size, cudaMemcpyHostToDevice));

    std::vector<float> h_init_cache(cache_size / sizeof(float), 0.1f);
    std::vector<float> h_init_vec(vector_size / sizeof(float), 1.0f);
    CHECK_CUDA(cudaMemcpy(d_k_cache, h_init_cache.data(), cache_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_cache, h_init_cache.data(), cache_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_init_vec.data(), vector_size, cudaMemcpyHostToDevice));

    dim3 grid(batch_size, num_heads);
    dim3 block(head_dim);
    size_t shared_mem_size = (head_dim + max_seq_len) * sizeof(float);

    std::cout << "Starting Profiling Loop..." << std::endl;
    std::cout << "========================================================\n";

    // --- 4. 模拟 Decoding Loop ---
    int num_decode_steps = 128;
    int attn_loop = 5000;
    int update_loop = 1000000;
    for (int step = 0; step < num_decode_steps; ++step) {
        
        unsigned long long energy_start, energy_end;
        float ms_kv, ms_attn;

        // ==========================================
        // 测量 KV Cache Update Kernel
        // ==========================================
        CHECK_CUDA(cudaDeviceSynchronize()); // 确保之前任务跑完再测
        CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &energy_start));
        CHECK_CUDA(cudaEventRecord(start_event));
        
        for(int i = 0; i < update_loop; i++){
            update_kv_cache_kernel<float><<<grid, block>>>(
                d_k_cache, d_v_cache, d_k, d_v, d_seq_lens, max_seq_len, num_heads, head_dim
            );
        }
        
        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event)); // 阻塞直至 kernel 跑完
        CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &energy_end));
        
        CHECK_CUDA(cudaEventElapsedTime(&ms_kv, start_event, stop_event));
        float energy_kv = (float)(energy_end - energy_start) / update_loop;

        // --- 更新 seq_lens (CPU 发起，不算入 Kernel runtime 核心区) ---
        for (int b = 0; b < batch_size; ++b) h_seq_lens[b]++;
        CHECK_CUDA(cudaMemcpy(d_seq_lens, h_seq_lens.data(), seq_lens_size, cudaMemcpyHostToDevice));

        // ==========================================
        // 测量 Attention Kernel
        // ==========================================
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &energy_start));
        CHECK_CUDA(cudaEventRecord(start_event));

        for(int i = 0; i < attn_loop; i++){
            decoding_attention_kernel<float><<<grid, block, shared_mem_size>>>(
                d_q, d_k_cache, d_v_cache, d_out, d_seq_lens, max_seq_len, head_dim
            );
        }

        CHECK_CUDA(cudaEventRecord(stop_event));
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        CHECK_NVML(nvmlDeviceGetTotalEnergyConsumption(device, &energy_end));

        CHECK_CUDA(cudaEventElapsedTime(&ms_attn, start_event, stop_event));
        float energy_attn = (float)(energy_end - energy_start) / attn_loop;

        // 映射输出到下一步的输入
        CHECK_CUDA(cudaMemcpy(d_q, d_out, vector_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_k, d_out, vector_size, cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_v, d_out, vector_size, cudaMemcpyDeviceToDevice));

        // ==========================================
        // 打印本轮统计数据
        // ==========================================
        std::cout << "Iteration " << step + 1 << " (Seq Len: " << h_seq_lens[0] << ")\n";
        std::cout << "  [KV Update] Runtime: " << (ms_kv / update_loop) << " ms \t Energy: " << energy_kv << " mJ\n";
        std::cout << "  [Attention] Runtime: " << (ms_attn / attn_loop) << " ms \t Energy: " << energy_attn << " mJ\n";
        std::cout << "--------------------------------------------------------\n";
    }

    // --- 5. 清理资源 ---
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_NVML(nvmlShutdown());
    cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaFree(d_seq_lens);

    return 0;
}