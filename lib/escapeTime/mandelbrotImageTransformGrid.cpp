#include "escapeTime/mandelbrotImageTransformGrid.h"
#include "escapeTimeSequential.h"



void CpuMandelbrotImageTransformGrid::updateGrid() {
	//call the sequential escape time function
	escapeTimeSequential(colorGrid, maxIters, sizeX, sizeY, scale, startX, startY, ColoringFunctionType::DEFAULT);
}

CpuMandelbrotImageTransformGrid::CpuMandelbrotImageTransformGrid(uint sizeX, uint sizeY, uint maxIters, double scale, double startX, double startY) {
	this->sizeX = sizeX;
	this->sizeY = sizeY;
	this->maxIters = maxIters;
	this->scale = scale;
	this->startX = startX;
	this->startY = startY;
	this->coloringFunction = ColoringFunctionType::DEFAULT;
	colorGrid = std::vector<RGBColor>(sizeX * sizeY);

	updateGrid();
}

RGBColor CpuMandelbrotImageTransformGrid::get(int x, int y)  {
	//for the edge case where x or y is out of bounds, return black
	if (x < 0 || x >= sizeX || y < 0 || y >= sizeY) return ALPHA_OPAQUE;
	return colorGrid[x * sizeY + y];
}

void CpuMandelbrotImageTransformGrid::zoom(double scale, int centerX, int centerY)  {
	auto scaleChange = this->scale - scale;
	auto xOffset = centerX * scaleChange;
	auto yOffset = centerY * scaleChange;
	this->scale = scale;
	startY += yOffset;
	startX += xOffset;
	updateGrid();
}

void CpuMandelbrotImageTransformGrid::resizeGrid(uint sizeX, uint sizeY) {
	if (sizeX * sizeY > colorGrid.size()) {
		colorGrid.resize(sizeX * sizeY);
	}

	this->sizeX = sizeX;
	this->sizeY = sizeY;

	updateGrid();
}

void CpuMandelbrotImageTransformGrid::translate(double offsetX, double offsetY) {
	startY -= offsetY * scale;
	startX -= offsetX * scale;
	updateGrid();
}

void CpuMandelbrotImageTransformGrid::setColoring(ColoringFunctionType func) {
	coloringFunction = func;
	updateGrid();
}

ColoringFunctionType CpuMandelbrotImageTransformGrid::getColoring() {
	return coloringFunction;
}

int CpuMandelbrotImageTransformGrid::getSizeX() { return sizeX; }
int CpuMandelbrotImageTransformGrid::getSizeY() { return sizeY; }
 