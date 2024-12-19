#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeCuda.cuh"

#define THREAD_SIZE 16

void MandelbrotImageTransformGrid::updateGrid() {
    dim3 threads(THREAD_SIZE, THREAD_SIZE);
	int blockXNum = ceil(sizeX/(float)threads.x);
	int blockYNum = ceil(sizeY/(float)threads.y);
	dim3 blocks(blockXNum, blockYNum);

    escapeTime<<<blocks, threads>>>(colorGridCUDA, maxIters, sizeX, sizeY, scale, startX, startY, coloringFunction);
    cudaDeviceSynchronize();
    cudaMemcpy(colorGrid.data(), colorGridCUDA, sizeof(RGBColor) * sizeX * sizeY, cudaMemcpyDeviceToHost);
}

MandelbrotImageTransformGrid::MandelbrotImageTransformGrid(int sizeX, int sizeY, int maxIters, double scale, double startX, double startY) {
    cudaMalloc(&colorGridCUDA, sizeof(RGBColor) * sizeX * sizeY);
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

RGBColor MandelbrotImageTransformGrid::get(int x, int y) {
    if (x < 0 || y < 0 || x >= sizeX || y >= sizeY) {
        return -1;
    }

    return colorGrid[y * sizeX + x];
}

void MandelbrotImageTransformGrid::zoom(double scale, int centerX, int centerY) {
    auto scaleChange = this->scale - scale;
    auto xOffset = centerX * scaleChange;
    auto yOffset = centerY * scaleChange;
    this->scale = scale;
    startY += yOffset;
    startX += xOffset;
    updateGrid();
}

void MandelbrotImageTransformGrid::resizeGrid(int sizeX, int sizeY) {
    auto negativeSize = false;
    if (sizeX < 0) {
        this->sizeX = 0;
        negativeSize = true;
    }

    if (sizeY < 0) {
        this->sizeY = 0;
        negativeSize = true;
    }

    if (negativeSize) {
        return;
    }

    this->sizeX = sizeX;
    this->sizeY = sizeY;
    updateGrid();

}

void MandelbrotImageTransformGrid::translate(double offsetX, double offsetY) {
    startY += offsetX;
    startX += offsetY;
    updateGrid();
}

void MandelbrotImageTransformGrid::setColoring(ColoringFunctionType func) {
    coloringFunction = func;
    updateGrid();
}

ColoringFunctionType MandelbrotImageTransformGrid::getColoring() {
    return ColoringFunctionType();
}

MandelbrotImageTransformGrid::~MandelbrotImageTransformGrid() {
    cudaFree(colorGridCUDA);
}
