#ifndef ESCAPE_TIME_CUDA
#define ESCAPE_TIME_CUDA

#include "escapeTime/escapeTimeCuda.h"

__global__ void escapeTime(int* escapeTimes, int max_iters, int sizeX, int sizeY, double scale, double panX, double panY);

#endif 

#ifdef BENCHMARKING
#include <chrono>

#define RUN_ESCAPE_TIME_KERNEL auto start = std::chrono::steady_clock::now(); \
				escapeTime <<<blocks, threads >>> (cudaEscapeTimes, maxIters, sizeX, sizeY, scale, panX, panY); \
				cudaDeviceSynchronize(); \
				auto end = std::chrono::steady_clock::now(); \
				auto cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
				std::cout << "Kernel elapsed time: " << cudaTime.count() << "\n";
#else
#define RUN_ESCAPE_TIME_KERNEL escapeTime<<<blocks, threads>>> (cudaEscapeTimes, maxIters, sizeX, sizeY, scale, panX, panY); \
				cudaDeviceSynchronize();
#endif