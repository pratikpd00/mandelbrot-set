#ifndef MANDELBROTIMAGETRANSFORMGRID_H
#define MANDELBROTIMAGETRANSFORMGRID_H

#include "imageTransformGrid.h"
#include <vector>

/* This class wraps the cuda kernel that calculates the escape time for points and provides an interface to access the image 
 * colored using those values.
 * Since this class represents an image, parameters are in terms of pixels, not the complex numbers used to calculate the mandelbrot
 * set.
 */ 
class CudaMandelbrotImageTransformGrid : public ImageTransformGrid {
private:
	int sizeX;
	int sizeY;
	int maxIters;
	double scale;
	double startX;
	double startY;
	RGBColor* colorGridCUDA;

	//For efficient use of CUDA, memory should be contiguous. To ensure that this is the case, a flattened 2d
	//vector is used
	std::vector<RGBColor> colorGrid;
	ColoringFunctionType coloringFunction;

	void updateGrid();
public:
	CudaMandelbrotImageTransformGrid(uint sizeX, uint sizeY, uint maxIters, double scale, double startX, double startY);
	
	virtual RGBColor get(int x, int y) override;
	virtual void zoom(double scale, int centerX, int centerY) override;
	virtual void resizeGrid(uint sizeX, uint sizeY) override;
	virtual void translate(double offsetX, double offsetY) override;
	virtual void setColoring(ColoringFunctionType func) override;
	virtual ColoringFunctionType getColoring() override;
	virtual int getSizeX();
	virtual int getSizeY();


	~CudaMandelbrotImageTransformGrid();
	
};

class CpuMandelbrotImageTransformGrid : public ImageTransformGrid {
private:
	int sizeX;
	int sizeY;
	int maxIters;
	double scale;
	double startX;
	double startY;

	//For efficient use of CUDA, memory should be contiguous. To ensure that this is the case, a flattened 2d
	//vector is used
	std::vector<RGBColor> colorGrid;
	ColoringFunctionType coloringFunction;

	void updateGrid();
public:
	CpuMandelbrotImageTransformGrid(uint sizeX, uint sizeY, uint maxIters, double scale, double startX, double startY);

	virtual RGBColor get(int x, int y) override;
	virtual void zoom(double scale, int centerX, int centerY) override;
	virtual void resizeGrid(uint sizeX, uint sizeY) override;
	virtual void translate(double offsetX, double offsetY) override;
	virtual void setColoring(ColoringFunctionType func) override;
	virtual ColoringFunctionType getColoring() override;
	virtual int getSizeX();
	virtual int getSizeY();


};

#endif