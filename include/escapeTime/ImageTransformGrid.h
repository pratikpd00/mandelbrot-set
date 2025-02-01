#ifndef IMAGETRANSFORMGRID_H
#define IMAGETRANSFORMGRID_H

#include "coloring.h"
#include "util.h"


//Represents a grid of pixels making up an image that can be transformed
class ImageTransformGrid {
public:
	virtual ~ImageTransformGrid() {};
	virtual void zoom(double scale, int centerX, int centerY) = 0;
	virtual void resizeGrid(uint sizeX, uint sizeY) = 0;
	virtual void translate(double offsetX, double offsetY) = 0;
	virtual RGBColor get(int escapeTime, int maxEscapeTime) = 0;
	virtual void setColoring(ColoringFunctionType func) = 0;
	virtual ColoringFunctionType getColoring() = 0;
	virtual int getSizeX() = 0;
	virtual int getSizeY() = 0;
};

#endif