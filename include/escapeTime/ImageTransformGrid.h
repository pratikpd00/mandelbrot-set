#ifndef IMAGETRANSFORMGRID_H
#define IMAGETRANSFORMGRID_H

#include "coloring.h"

//Represents a grid of pixels making up an image that can be transformed
class ImageTransformGrid {
public:
	virtual ~ImageTransformGrid() {};
	virtual void zoom(double scale, int centerX, int centerY) = 0;
	virtual void resizeGrid(int sizeX, int sizeY) = 0;
	virtual void translate(double offsetX, double offsetY) = 0;
	virtual RGBColor get(int escapeTime, int maxEscapeTime) = 0;
	virtual void setColoring(ColoringFunctionType func) = 0;
	virtual ColoringFunctionType getColoring() = 0;
};

#endif