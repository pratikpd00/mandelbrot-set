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
	RGBColor* colorGridCUDA;
	std::vector<RGBColor> colorGrid;
	ColoringFunctionType coloringFunction;

	void updateGrid();
public:
	MandelbrotImageTransformGrid(int sizeX, int sizeY, int maxIters, double scale, double startX, double startY);
	
	virtual RGBColor get(int x, int y) override;
	virtual void zoom(double scale, int centerX, int centerY) override;
	virtual void resizeGrid(int sizeX, int sizeY) override;
	virtual void translate(double offsetX, double offsetY) override;
	//add the remaining functions from the base class ImageTransformGrid
	virtual void setColoring(ColoringFunctionType func) override;
	virtual ColoringFunctionType getColoring() override;
	


	~MandelbrotImageTransformGrid();
	
};

#endif