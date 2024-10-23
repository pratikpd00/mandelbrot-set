#ifndef MANDELBROTIMAGETRANSFORMGRID_H
#define MANDELBROTIMAGETRANSFORMGRID_H

#include "imageTransformGrid.h"
#include <vector>


class MandelbrotImageTransformGrid : public ImageTransformGrid {
int sizeX;
int sizeY;
double scale;
double startX;
double startY;
RGBColor* colorGridCUDA;
std::vector<RGBColor> colorGrid;

public:
	MandelbrotImageTransformGrid(int sizeX, int sizeY, double scale, double startX, double startY);
	~MandelbrotImageTransformGrid();

	virtual void zoom(double scale) override;
	virtual void resize(int sizeX, int sizeY, int centerX, int centerY) override;
	virtual void translate(double panX, double panY) override;
	virtual RGBColor get(int x, int, y) override;
};

#endif