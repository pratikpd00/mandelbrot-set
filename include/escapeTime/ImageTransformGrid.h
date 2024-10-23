#ifndef IMAGETRANSFORMGRID_H
#define IMAGETRANSFORMGRID_H

class ImageTransformGrid {
public:
	ImageTransformGrid();
	virtual ~ImageTransformGrid();
	virtual void zoom(double scale) = 0;
	virtual void resize(int sizeX, int sizeY) = 0;
	virtual void translate(double panX, double panY) = 0;
};

#endif