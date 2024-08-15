#include <math.h>
#include "escapeTimeCuda.cuh"

__global__ void escapeTime(int* escapeTimes, int sizeX, int sizeY, int max_iters, double scale, double panX, double panY) {
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
	for (i = 0; i < max_iters && zRealSqr + zImaginarySqr <= 4; i++) {
		//(ai + b) ^2 = -a^2 + 2abi + b^2
		zImaginary = 2*zReal*zImaginary + cImaginary;
		zReal = zRealSqr - zImaginarySqr + cReal;
		
		zRealSqr = zReal*zReal;
		zImaginarySqr = zImaginary*zImaginary;
	}

	escapeTimes[x * sizeX + y] = i; 
}

void CudaEscapeTime(int* escapeTimes, int max_iters, int sizeX, int sizeY, double scale, double panX, double panY) {
	dim3 threads(16, 16);
	int blockXNum = ceil(sizeX/threads.x);
	int blockYNum = ceil(sizeY/threads.y);
	dim3 blocks(blockXNum, blockYNum);
	int* cudaEscapeTimes;
	cudaMalloc(&cudaEscapeTimes, sizeof(*cudaEscapeTimes) * sizeX * sizeY);
	escapeTime<<<blocks, threads>>>(escapeTimes, max_iters, sizeX, sizeY, scale, panX, panY);
}