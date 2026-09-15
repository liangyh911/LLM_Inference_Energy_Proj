// nvml_energy_period.cu
//
// 目的：实测 nvmlDeviceGetTotalEnergyConsumption 底层计数器的真实刷新周期。
//
// 原理：
//   计数器不是连续更新的，而是每隔一个固定周期才跳变一次。
//   我们用高分辨率主机时钟，以尽可能高的频率轮询这个计数器，
//   每当发现数值发生变化，就记录下“上一次变化”到“这一次变化”之间的时间差。
//   多次采样后，这些时间差的分布（尤其是众数/中位数）就是计数器的真实刷新周期。
//
// 同时用一个持续占用 GPU 的 busy kernel，模拟真实推理负载下的场景，
// 因为满载和空闲时，某些 GPU 的计数器刷新行为可能不同。
//
// 编译：
//   nvcc -O2 nvml_energy_period.cu -o nvml_energy_period -lnvidia-ml
//   如果找不到 nvml.h / libnvidia-ml.so，通常在:
//     /usr/local/cuda/targets/x86_64-linux/include (头文件, 一般已在默认路径)
//     /usr/lib/x86_64-linux-gnu/libnvidia-ml.so (驱动自带, 需要 -L 指定或建软链接)
//   例如: nvcc -O2 nvml_energy_period.cu -o nvml_energy_period \
//           -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -lnvidia-ml
//
// 运行：
//   ./nvml_energy_period [device_index] [duration_seconds]
//   例如: ./nvml_energy_period 0 5

#include <cuda_runtime.h>
#include <nvml.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

#define NVML_CHECK(call)                                                       \
    do {                                                                       \
        nvmlReturn_t _st = (call);                                             \
        if (_st != NVML_SUCCESS) {                                             \
            fprintf(stderr, "NVML error %s:%d: %s\n", __FILE__, __LINE__,      \
                    nvmlErrorString(_st));                                     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _st = (call);                                              \
        if (_st != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_st));                                  \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                      \
    } while (0)

using Clock = std::chrono::steady_clock;

// 用 clock64() 做一个可控时长的忙等 kernel，让 GPU 在轮询期间保持高占用率，
// 更贴近你实际 decoding workload 时的功耗状态。
__global__ void busy_kernel(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
        // spin
    }
}

int main(int argc, char** argv) {
    int device_index = (argc > 1) ? std::atoi(argv[1]) : 0;
    double duration_sec = (argc > 2) ? std::atof(argv[2]) : 5.0;

    CUDA_CHECK(cudaSetDevice(device_index));

    NVML_CHECK(nvmlInit_v2());
    nvmlDevice_t device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex_v2(device_index, &device));

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    NVML_CHECK(nvmlDeviceGetName(device, name, sizeof(name)));
    printf("Device: %s (index %d)\n", name, device_index);
    printf("Polling duration: %.1f s\n\n", duration_sec);

    // 估算 GPU 核心时钟频率(MHz)，用来把“忙等时长”换算成 clock64() 的 cycle 数。
    unsigned int sm_clock_mhz = 0;
    NVML_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock_mhz));
    if (sm_clock_mhz == 0) sm_clock_mhz = 1000; // fallback
    long long cycles_per_launch =
        static_cast<long long>(sm_clock_mhz) * 1000LL * 2; // 约 2ms 忙等/次

    // 用一个和 GPU 上 SM 数量匹配的 grid，制造实际算力占用，而不是单线程空转。
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int blocks = prop.multiProcessorCount * 4;
    int threads = 256;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ---- 预热，让 GPU 从低功耗/低频状态进入稳定运行状态 ----
    for (int i = 0; i < 20; ++i) {
        busy_kernel<<<blocks, threads, 0, stream>>>(cycles_per_launch);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- 正式轮询 ----
    unsigned long long last_energy = 0;
    NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &last_energy));

    auto t_start = Clock::now();
    auto t_last_change = t_start;

    std::vector<double> intervals_ms;      // 相邻两次跳变之间的时间差
    std::vector<unsigned long long> deltas_mJ; // 每次跳变对应的能耗增量

    unsigned long long poll_count = 0;
    unsigned long long change_count = 0;

    while (true) {
        // 持续往 stream 里塞 kernel，保证轮询期间 GPU 一直在跑东西，
        // 不会因为 queue 空了而掉回空闲态。
        busy_kernel<<<blocks, threads, 0, stream>>>(cycles_per_launch);

        unsigned long long cur_energy;
        NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &cur_energy));
        auto t_now = Clock::now();
        poll_count++;

        if (cur_energy != last_energy) {
            double dt_ms =
                std::chrono::duration<double, std::milli>(t_now - t_last_change).count();
            // 丢弃第一次跳变（没有真正的“上一次”基准），从第二次开始记录
            if (change_count > 0) {
                intervals_ms.push_back(dt_ms);
                deltas_mJ.push_back(cur_energy - last_energy);
            }
            t_last_change = t_now;
            last_energy = cur_energy;
            change_count++;
        }

        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        if (elapsed >= duration_sec) break;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // ---- 统计结果 ----
    printf("Total polls:            %llu\n", poll_count);
    printf("Total counter changes:  %llu\n", change_count);

    if (intervals_ms.empty()) {
        printf("\n没有采集到足够的跳变样本，试着把 duration 调大一些（比如 10s）。\n");
    } else {
        std::vector<double> sorted_intervals = intervals_ms;
        std::sort(sorted_intervals.begin(), sorted_intervals.end());

        double sum = std::accumulate(intervals_ms.begin(), intervals_ms.end(), 0.0);
        double mean = sum / intervals_ms.size();
        double median = sorted_intervals[sorted_intervals.size() / 2];
        double p10 = sorted_intervals[static_cast<size_t>(sorted_intervals.size() * 0.1)];
        double p90 = sorted_intervals[static_cast<size_t>(sorted_intervals.size() * 0.9)];
        double min_v = sorted_intervals.front();
        double max_v = sorted_intervals.back();

        double sq_sum = 0.0;
        for (double v : intervals_ms) sq_sum += (v - mean) * (v - mean);
        double stddev = std::sqrt(sq_sum / intervals_ms.size());

        unsigned long long total_energy_delta =
            std::accumulate(deltas_mJ.begin(), deltas_mJ.end(), 0ULL);

        printf("\n=== 计数器跳变间隔统计 (即刷新周期估计, 单位 ms) ===\n");
        printf("  样本数:   %zu\n", intervals_ms.size());
        printf("  最小值:   %.4f ms\n", min_v);
        printf("  P10:      %.4f ms\n", p10);
        printf("  中位数:   %.4f ms   <-- 一般以这个作为真实刷新周期的估计\n", median);
        printf("  平均值:   %.4f ms\n", mean);
        printf("  P90:      %.4f ms\n", p90);
        printf("  最大值:   %.4f ms\n", max_v);
        printf("  标准差:   %.4f ms\n", stddev);

        // sanity check: 用总能耗/总时间 反推平均功率，量级应在合理的 GPU 功耗范围内
        double total_time_s =
            std::chrono::duration<double>(t_last_change - t_start).count();
        double avg_power_w = (total_energy_delta / 1000.0) / total_time_s;
        printf("\n=== Sanity check ===\n");
        printf("  区间总能耗: %llu mJ\n", total_energy_delta);
        printf("  区间总时长: %.4f s\n", total_time_s);
        printf("  推算平均功率: %.2f W  (应与该 GPU 满载功耗量级吻合，否则说明测量有问题)\n",
               avg_power_w);

        printf("\n建议：在你的能耗统计代码里，把单次测量窗口拉长到至少 %.2f ms 以上，\n"
               "      或者按批量跑 N 次相同 kernel 再统一读取能耗差值，\n"
               "      这样才能避免大部分样本读到 0 或采样锯齿的问题。\n",
               median * 3.0); // 建议窗口至少覆盖 3 个周期，避免边界效应
    }

    NVML_CHECK(nvmlShutdown());
    return 0;
}
