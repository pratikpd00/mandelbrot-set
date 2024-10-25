#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeCuda.cuh"

#define THREAD_SIZE 16

void MandelbrotImageTransformGrid::updateGrid() {
    dim3 threads(THREAD_SIZE, THREAD_SIZE);
	int blockXNum = ceil(sizeX/(float)threads.x);
	int blockYNum = ceil(sizeY/(float)threads.y);
	dim3 blocks(blockXNum, blockYNum);

    escapeTime<<<blocks, threads>>> ((int*)colorGridCUDA, maxIters, sizeX, sizeY, scale, startX, startY);
}

MandelbrotImageTransformGrid::MandelbrotImageTransformGrid(int sizeX, int sizeY, int maxIters, double scale, double startX, double startY){
    cudaMalloc(&colorGridCUDA, sizeX * sizeY);
    colorGrid = std::vector<RGBColor>(sizeX * sizeY);
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->maxIters = maxIters; 
	this->scale = scale;
	this->startX = startX;
	this->startY = startY;
    updateGrid();
}

RGBColor MandelbrotImageTransformGrid::get(int x, int y) {
    return colorGrid[y * sizeX + x];
}

MandelbrotImageTransformGrid::~MandelbrotImageTransformGrid() {
    cudaFree(colorGridCUDA);
}
