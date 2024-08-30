#include <cmath>
#include <chrono>
#include <iostream>
#include "escapeTimeCuda.cuh"

__global__ void escapeTime(int* escapeTimes, int maxIters, int sizeX, int sizeY,  double scale, double panX, double panY) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= sizeX || y >= sizeY) {
		return;
	}
	
	//Mandelbrot recurrence for reference: z_n+1 = z_n^2 + c
	const double cReal = x * scale + panX;
	const double cImaginary = y * scale + panY;

	double zReal = 0;
	double zImaginary = 0;

	double zRealSqr = 0;
	double zImaginarySqr = 0;

	//if the absolute value of z ever exceeds 2, it is guaranteed to escape the mandelbrot set. Therefore, this loop can be terminated
	int i;
	for (i = 0; i < maxIters && zRealSqr + zImaginarySqr <= 4; i++) {
		//(ai + b) ^2 = -a^2 + 2abi + b^2
		zImaginary = 2*zReal*zImaginary + cImaginary;
		zReal = zRealSqr - zImaginarySqr + cReal;
		
		zRealSqr = zReal*zReal;
		zImaginarySqr = zImaginary*zImaginary;
	}

	escapeTimes[x * sizeY + y] = i; 
}

void escapeTimeCUDA(int* escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY) {
	dim3 threads(16, 16);
	int blockXNum = ceil(sizeX/(float)threads.x);
	int blockYNum = ceil(sizeY/(float)threads.y);
	dim3 blocks(blockXNum, blockYNum);
	int* cudaEscapeTimes;
	cudaMalloc(&cudaEscapeTimes, sizeof(*cudaEscapeTimes) * sizeX * sizeY);
	
	auto start = std::chrono::steady_clock::now();
	escapeTime<<<blocks, threads>>>(cudaEscapeTimes, maxIters, sizeX, sizeY, scale, panX, panY);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	auto cudaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "Kernel elapsed time: " << cudaTime.count() << "\n";

	cudaMemcpy(escapeTimes, cudaEscapeTimes, sizeof(*cudaEscapeTimes) * sizeX * sizeY, cudaMemcpyDeviceToHost);
	cudaFree(cudaEscapeTimes);
}