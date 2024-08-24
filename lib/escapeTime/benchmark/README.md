# Benchmarks
**Based on preliminary benchmarking data on an Nvida RTX 4060 ti and AMD Ryzen 5 2600 as of 8/24/2024**
BAsed on a a rudimentary benchmark of the CUDA and sequential escape time algorithms, CUDA does not seem to provide an advantage over sequential. However, the CUDA escape time algorithm scales far better.
This may be due to overheads in launching a CUDA kernel, utilizing benchmark data that is not representative of the improvement CUDA provides, or inefficiencies in how CUDA is utilized.
