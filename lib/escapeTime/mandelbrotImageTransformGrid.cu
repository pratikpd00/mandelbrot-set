#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeCuda.cuh"

#include <iostream>

#define THREAD_SIZE 16

void CudaMandelbrotImageTransformGrid::updateGrid() {
    dim3 threads(THREAD_SIZE, THREAD_SIZE);
	int blockXNum = ceil(sizeX/(float)threads.x);
	int blockYNum = ceil(sizeY/(float)threads.y);
	dim3 blocks(blockXNum, blockYNum);

    escapeTime<<<blocks, threads>>>(colorGridCUDA, maxIters, sizeX, sizeY, scale, startX, startY, coloringFunction);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error running kernel " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    err = cudaMemcpy(colorGrid.data(), colorGridCUDA, sizeof(*colorGridCUDA) * sizeX * sizeY, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
		std::cerr << "CUDA error running memcpy " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
	}
}

CudaMandelbrotImageTransformGrid::CudaMandelbrotImageTransformGrid(uint sizeX, uint sizeY, uint maxIters, double scale, double startX, double startY) {
    auto err = cudaMalloc(&colorGridCUDA, sizeof(RGBColor) * sizeX * sizeY);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error while allocating memory " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
    }
    colorGrid = std::vector<RGBColor>(sizeX * sizeY);
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->maxIters = maxIters; 
	this->scale = scale;
	this->startX = startX;
	this->startY = startY;
    coloringFunction = ColoringFunctionType::DEFAULT;
    updateGrid();
}

RGBColor CudaMandelbrotImageTransformGrid::get(int x, int y) {
    if (x < 0 || y < 0 || x >= sizeX || y >= sizeY) {
        return ALPHA_OPAQUE;
    }

    return colorGrid[x * sizeY + y];
}

void CudaMandelbrotImageTransformGrid::zoom(double scale, int centerX, int centerY) {
    auto scaleChange = this->scale - (this->scale*scale);
    auto xOffset = centerX * scaleChange;
    auto yOffset = centerY * scaleChange;
    this->scale *= scale;
    startY += yOffset;
    startX += xOffset;
    updateGrid();
}

void CudaMandelbrotImageTransformGrid::resizeGrid(uint sizeX, uint sizeY) {
    if (sizeX == this->sizeX && sizeY == this->sizeY) {
		return;
	}

    //To prevent unnecessary calls to cudaMalloc and cudaFree by resizing the grid, we only reallocate if the new size is larger
    if (sizeX * sizeY > colorGrid.size()) {
		cudaFree(colorGridCUDA);
		cudaMalloc(&colorGridCUDA, sizeof(RGBColor) * sizeX * sizeY);
		colorGrid.resize(sizeX * sizeY);
	}

    this->sizeX = sizeX;
    this->sizeY = sizeY;

    updateGrid();

}

void CudaMandelbrotImageTransformGrid::translate(double offsetX, double offsetY) {
    startY -= offsetY * scale;
    startX -= offsetX * scale;
    updateGrid();
}

void CudaMandelbrotImageTransformGrid::setColoring(ColoringFunctionType func) {
    coloringFunction = func;
    updateGrid();
}

ColoringFunctionType CudaMandelbrotImageTransformGrid::getColoring() {
    return coloringFunction;
}

CudaMandelbrotImageTransformGrid::~CudaMandelbrotImageTransformGrid() {
    cudaFree(colorGridCUDA);
}

//add the size getters declared in the header
int CudaMandelbrotImageTransformGrid::getSizeX() { return sizeX; }
int CudaMandelbrotImageTransformGrid::getSizeY() { return sizeY; }
