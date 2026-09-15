#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// 检查 CUDA 错误的宏
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * @brief 更新 KV Cache 的 CUDA Kernel
 * 
 * 内存布局约定:
 * - k_cache / v_cache : [batch_size, num_heads, max_seq_len, head_dim]
 * - new_k / new_v     : [batch_size, num_heads, head_dim]
 * - seq_lens          : [batch_size] 记录每个 batch 当前句子的长度
 * 
 * 线程模型规划:
 * - Grid 维度:  (batch_size, num_heads, 1)
 * - Block 维度: (head_dim, 1, 1)  // 假设 head_dim <= 1024
 */
template <typename T>
__global__ void update_kv_cache_kernel(
    T* k_cache, T* v_cache,
    const T* new_k, const T* new_v,
    const int* seq_lens,
    int max_seq_len,
    int num_heads,
    int head_dim
) {
    // 确定当前线程处理的 Batch, Head 和 Dimension 索引
    int b = blockIdx.x;      // Batch 索引
    int h = blockIdx.y;      // Head 索引
    int d = threadIdx.x;     // Head Dimension 索引

    if (d < head_dim) {
        // 获取当前 sequence 应该写入的位置 (相当于当前已经生成的 token 数量)
        int current_len = seq_lens[b];

        // 越界保护（防止生成的句子超过预分配的 max_seq_len）
        if (current_len >= max_seq_len) return;

        // 计算 new_k/new_v 的一维扁平化索引
        // 形状: [batch_size, num_heads, head_dim]
        int new_idx = b * (num_heads * head_dim) + 
                      h * head_dim + 
                      d;

        // 计算 KV Cache 的一维扁平化索引
        // 形状: [batch_size, num_heads, max_seq_len, head_dim]
        int cache_idx = b * (num_heads * max_seq_len * head_dim) + 
                        h * (max_seq_len * head_dim) + 
                        current_len * head_dim + 
                        d;

        // 将新计算出的 K 和 V 写入缓存
        k_cache[cache_idx] = new_k[new_idx];
        v_cache[cache_idx] = new_v[new_idx];
    }
}

// Host 端调用 Kernel 的包装函数
void update_kv_cache(
    float* d_k_cache, float* d_v_cache,
    const float* d_new_k, const float* d_new_v,
    const int* d_seq_lens,
    int batch_size, int max_seq_len, int num_heads, int head_dim
) {
    // 设置线程网格和块的大小
    // Grid: 每个 batch_size 和 num_heads 对应一个 Block
    dim3 grid(batch_size, num_heads);
    // Block: 每个线程负责拷贝 head_dim 上的一个标量元素
    dim3 block(head_dim);

    update_kv_cache_kernel<float><<<grid, block>>>(
        d_k_cache, d_v_cache,
        d_new_k, d_new_v,
        d_seq_lens,
        max_seq_len, num_heads, head_dim
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

int main() {
    // --- 模拟参数设定 ---
    int batch_size = 4;
    int max_seq_len = 2048; // 分配的最大 Context Length
    int num_heads = 32;     // Attention 头的数量
    int head_dim = 128;     // 每个头的维度 (例如 4096 hidden_size / 32 heads = 128)

    // 计算大小 (单位：字节)
    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim * sizeof(float);
    size_t new_kv_size = batch_size * num_heads * 1 * head_dim * sizeof(float);
    size_t seq_lens_size = batch_size * sizeof(int);

    // --- Device 内存分配 ---
    float *d_k_cache, *d_v_cache, *d_new_k, *d_new_v;
    int *d_seq_lens;

    CHECK_CUDA_ERROR(cudaMalloc(&d_k_cache, cache_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v_cache, cache_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_new_k, new_kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_new_v, new_kv_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seq_lens, seq_lens_size));

    // --- Host 端初始化模拟数据 ---
    // 假设 batch 中的 4 个序列当前分别已经生成了 10, 5, 20, 100 个 token
    int h_seq_lens[4] = {10, 5, 20, 100};
    CHECK_CUDA_ERROR(cudaMemcpy(d_seq_lens, h_seq_lens, seq_lens_size, cudaMemcpyHostToDevice));

    // 初始化 new_k 和 new_v (全部设为 1.0 方便测试，实际中应由前向传播输出)
    std::vector<float> h_new_k(batch_size * num_heads * head_dim, 1.0f);
    CHECK_CUDA_ERROR(cudaMemcpy(d_new_k, h_new_k.data(), new_kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_new_v, h_new_k.data(), new_kv_size, cudaMemcpyHostToDevice));
    
    // 初始化 Cache 为 0
    CHECK_CUDA_ERROR(cudaMemset(d_k_cache, 0, cache_size));
    CHECK_CUDA_ERROR(cudaMemset(d_v_cache, 0, cache_size));

    std::cout << "Starting KV Cache Update Simulation..." << std::endl;

    int iterations = 10;
    for(int i = 0; i < iterations; i++){
        // --- 执行 KV Cache 更新 ---
        update_kv_cache(
            d_k_cache, d_v_cache, 
            d_new_k, d_new_v, 
            d_seq_lens, 
            batch_size, max_seq_len, num_heads, head_dim
        );

        // 更新序列长度 (seq_lens = seq_lens + 1)
        for (int b = 0; b < batch_size; ++b) h_seq_lens[b]++;
        CHECK_CUDA_ERROR(cudaMemcpy(d_seq_lens, h_seq_lens, seq_lens_size, cudaMemcpyHostToDevice));
    }

    std::cout << "KV Cache successfully updated!" << std::endl;

    // --- (可选) 验证写入是否正确 ---
    // 验证 batch 0, head 0, seq_idx 10, dim 0 处的值是否被成功更新为 1.0
    float test_val;
    int test_b = 0;
    int test_h = 0;
    int test_seq = h_seq_lens[test_b]; // 写入的 sequence index 为 10
    int test_d = 0;
    
    int test_cache_idx = test_b * (num_heads * max_seq_len * head_dim) + 
                         test_h * (max_seq_len * head_dim) + 
                         test_seq * head_dim + 
                         test_d;

    CHECK_CUDA_ERROR(cudaMemcpy(&test_val, d_k_cache + test_cache_idx, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Value at (b=0, h=0, seq=10, d=0) in K Cache: " << test_val << " (Expected: 1.0)" << std::endl;

    // --- 清理内存 ---
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_new_k);
    cudaFree(d_new_v);
    cudaFree(d_seq_lens);

    return 0;
}