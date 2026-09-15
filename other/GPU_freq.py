import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get the handle for the first GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Retrieve clock speeds
sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

print(f"SM Clock: {sm_clock} MHz")
print(f"Memory Clock: {mem_clock} MHz")

# Always shut down NVML when done
pynvml.nvmlShutdown()