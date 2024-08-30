# Benchmarks
**Based on preliminary benchmarking data on an Nvida RTX 4060 ti and AMD Ryzen 5 2600 as of 8/29/2024,
 1025*512 grid and 500 iterations**

Runtimes:

CUDA Kernel: 54ms
CUDA call (including setup and copying GPU memory): 490ms
Sequential: 390ms

Based on this data, it is likely that the overhead of calling a CUDA kernel is
high enough that sequential escape time is faster under normal circumstances.
Additional data and profiling is needed to learn more.
