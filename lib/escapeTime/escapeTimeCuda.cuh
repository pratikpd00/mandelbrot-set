#ifndef ESCAPE_TIME_CUDA
#define ESCAPE_TIME_CUDA

#include "escapeTime/coloring.h"
#include "escapeTimeCudaWrapper.h"
#include "util.h"

__global__ void escapeTime(RGBColor* escapedColors, uint maxIters, uint sizeX, uint sizeY, double scale, double panX, double panY, ColoringFunctionType func);



#ifdef BENCHMARKING
#include <chrono>

#define RUN_ESCAPE_TIME_KERNEL auto start = std::chrono::steady_clock::now(); \
				escapeTime <<<blocks, threads >>> (cudaEscapeTimes, maxIters, sizeX, sizeY, scale, panX, panY, ColoringFunctionType::DEFAULT); \
				cudaDeviceSynchronize(); \
				auto end = std::chrono::steady_clock::now(); \
				auto cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
				std::cout << "Kernel elapsed time: " << cudaTime.count() << "ms\n";
#else
#define RUN_ESCAPE_TIME_KERNEL escapeTime<<<blocks, threads>>> (cudaEscapeTimes, maxIters, sizeX, sizeY, scale, panX, panY, ColoringFunctionType::DEFAULT); \
				cudaDeviceSynchronize();
#endif


#endif 
