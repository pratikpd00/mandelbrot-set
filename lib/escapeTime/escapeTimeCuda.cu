#include <cmath>
#include <iostream>

#include "escapeTimeCuda.cuh"
#include "coloringFunctions.cuh"

__global__ void escapeTime(RGBColor* escapedColors, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY, ColoringFunction::Function func) {
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

	escapedColors[x * sizeY + y] = ColoringFunction::color(i, maxIters, func);
}


//This function is for testing. The actual UI should use imageTransformGrid's children instead
void escapeTimeCUDA(RGBColor* escapeTimes, int maxIters, int sizeX, int sizeY, double scale, double panX, double panY) {
	dim3 threads(16, 16);
	int blockXNum = ceil(sizeX/(float)threads.x);
	int blockYNum = ceil(sizeY/(float)threads.y);
	dim3 blocks(blockXNum, blockYNum);
	RGBColor* cudaEscapeTimes;
	cudaMalloc(&cudaEscapeTimes, sizeof(RGBColor) * sizeX * sizeY);
	
	RUN_ESCAPE_TIME_KERNEL;
	
	cudaMemcpy(escapeTimes, cudaEscapeTimes, sizeof(*cudaEscapeTimes) * sizeX * sizeY, cudaMemcpyDeviceToHost);
	cudaFree(cudaEscapeTimes);
}