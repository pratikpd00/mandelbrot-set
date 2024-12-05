#ifndef IMAGETRANSFORMGRID_H
#define IMAGETRANSFORMGRID_H

#include "coloring.h"

//Represents a grid of pixels making up an image that can be transformed
class ImageTransformGrid {
public:
	virtual ~ImageTransformGrid() {};
	virtual void zoom(double scale) = 0;
	virtual void resize(int sizeX, int sizeY, int centerX, int centerY) = 0;
	virtual void translate(double panX, double panY) = 0;
	virtual RGBColor get(int escapeTime, int maxEscapeTime) = 0;
};

#endif