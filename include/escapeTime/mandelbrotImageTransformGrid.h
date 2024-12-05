#ifndef MANDELBROTIMAGETRANSFORMGRID_H
#define MANDELBROTIMAGETRANSFORMGRID_H

#include "imageTransformGrid.h"
#include <vector>


class MandelbrotImageTransformGrid : public ImageTransformGrid {
private:
	int sizeX;
	int sizeY;
	int maxIters;
	double scale;
	double startX;
	double startY;
	color* colorGridCUDA;
	std::vector<color> colorGrid;

	void updateGrid();
public:
	MandelbrotImageTransformGrid(int sizeX, int sizeY, int maxIters, double scale, double startX, double startY);
	
	virtual RGBcolor get(int x, int y) override;
	virtual void zoom(double scale) override;
	virtual void resize(int sizeX, int sizeY, int centerX, int centerY) override;
	virtual void translate(double panX, double panY) override;

	~MandelbrotImageTransformGrid();
	
};

#endif